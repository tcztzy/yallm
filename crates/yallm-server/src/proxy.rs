//! Provider routing and upstream calling — the heart of yallm.
//!
//! Every downstream request funnels through this module. The canonical flow:
//!
//! 1. [`choose_provider`] resolves a model string (`gpt-5.2`, `acp:codex`,
//!    `anthropic/...`, or a LiteLLM alias) into a [`ProviderTarget`].
//!    Route precedence: LiteLLM alias (via `YALLM_LITELLM_CONFIG`) wins,
//!    then a `provider:` / `provider/` prefix, then the default provider.
//! 2. [`should_proxy`] decides between real upstream call and deterministic
//!    mock reply, based on `YALLM_MODE` (auto / proxy / mock). Auto = proxy
//!    when the provider is configured, mock otherwise.
//! 3. [`complete_ir`] (non-streaming) or [`call_provider_stream`] (streaming)
//!    dispatch to the provider: OpenAI / Anthropic / Ollama go over the
//!    transport as HTTP; ACP spawns a subprocess (`YALLM_ACP_COMMAND`) that
//!    speaks the Agent Client Protocol over stdio.
//!
//! Downstream request bodies are converted into the shared IR via
//! `openai_downstream_to_ir` / `anthropic_downstream_to_ir`; upstream
//! responses come back as IR or as a raw byte stream that
//! [`map_provider_stream_to_downstream`](crate::proxy::stream::map_provider_stream_to_downstream)
//! converts to the downstream wire format.
//!
//! Gotchas:
//! - Header redaction here (`redact_headers`) must stay consistent with the
//!   HTTP logging middleware in `crate::logging`.
//! - ACP is the one provider with no HTTP fallback: when `YALLM_ACP_COMMAND`
//!   is unset, `call_acp` errors with status 400 ("ACP upstream is not
//!   configured") and `should_proxy` treats ACP as unconfigured in auto mode.
//! - `Mode::Proxy` forces upstream calls even when nothing is configured —
//!   expect errors for unset keys; auto mode is the safe default.

use std::{convert::Infallible, path::PathBuf, pin::Pin, time::Instant};

use axum::http::HeaderMap;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{Value, json};
use yallm_ir::{ChatRequest, ChatResponse, Choice, Content, Message, Role, Source, Usage};

use crate::{
    logging::redact_json_secrets,
    state::{
        AppState, DEFAULT_ANTHROPIC_BASE_URL, DEFAULT_OPENAI_BASE_URL, Mode, ModelRoute, Provider,
        TransportByteStream, TransportError, TransportRequest,
    },
};

mod stream;

#[derive(Debug)]
/// Upstream/proxy failure mapped to a client-facing HTTP response.
pub struct ProxyError {
    /// HTTP status to return to the client
    pub status: u16,
    /// Error message (logged; also exposed to the client)
    pub message: String,
    /// Upstream error body, when one was returned
    pub upstream_body: Option<Value>,
}

fn provider_from_model(model: &str) -> (Option<Provider>, String) {
    // Accept `openai:<m>` / `openai/<m>` (and same for anthropic/ollama/acp).
    for (prefix, p) in [
        ("openai:", Provider::OpenAI),
        ("openai/", Provider::OpenAI),
        ("anthropic:", Provider::Anthropic),
        ("anthropic/", Provider::Anthropic),
        ("ollama:", Provider::Ollama),
        ("ollama/", Provider::Ollama),
        ("acp:", Provider::Acp),
        ("acp/", Provider::Acp),
    ] {
        if let Some(rest) = model.strip_prefix(prefix) {
            return (Some(p), rest.to_string());
        }
    }
    (None, model.to_string())
}

#[derive(Debug, Clone)]
/// Resolved upstream target for a model string.
pub struct ProviderTarget {
    /// Upstream provider
    pub provider: Provider,
    /// Model name as the client wrote it
    pub downstream_model: String,
    /// Model name sent to the upstream
    pub upstream_model: String,
    /// LiteLLM route this target came from, when any
    pub route: Option<ModelRoute>,
}

/// Resolve a model string to a [`ProviderTarget`]: LiteLLM alias first, then `provider:`/`provider/` prefix, then the default provider.
pub fn choose_provider(state: &AppState, model: &str) -> ProviderTarget {
    if let Some(route) = state.model_routes.get(model) {
        return ProviderTarget {
            provider: route.provider,
            downstream_model: model.to_string(),
            upstream_model: route.upstream_model.clone(),
            route: Some(route.clone()),
        };
    }

    let (from_prefix, stripped) = provider_from_model(model);
    let provider = from_prefix.unwrap_or(state.default_provider);
    ProviderTarget {
        provider,
        downstream_model: model.to_string(),
        upstream_model: stripped,
        route: None,
    }
}

/// Decide between real upstream call and mock reply for a target.
pub fn should_proxy(
    state: &AppState,
    target: &ProviderTarget,
    incoming_headers: &HeaderMap,
) -> Result<bool, ProxyError> {
    match state.mode {
        Mode::Mock => Ok(false),
        Mode::Auto => Ok(provider_is_configured(state, target, incoming_headers)),
        Mode::Proxy => Ok(true),
    }
}

/// Run an IR [`ChatRequest`] end-to-end: choose provider, proxy or mock, and return the IR [`ChatResponse`]. Public non-streaming entry point used by routes and the ACP backend.
pub async fn complete_ir(
    state: &AppState,
    request_id: u64,
    mut ir: ChatRequest,
    incoming_headers: &HeaderMap,
) -> Result<ChatResponse, ProxyError> {
    let target = choose_provider(state, &ir.model);
    ir.model = target.upstream_model.clone();

    match should_proxy(state, &target, incoming_headers)? {
        true => call_provider(state, request_id, &target, &ir, incoming_headers).await,
        false => {
            let reply = mock_reply(extract_last_user_text(&ir));
            Ok(mock_ir_response(&ir.model, reply))
        }
    }
}

fn provider_is_configured(
    state: &AppState,
    target: &ProviderTarget,
    incoming_headers: &HeaderMap,
) -> bool {
    if !forwarded_request_headers(state, target, incoming_headers).is_empty() {
        return true;
    }

    if let Some(route) = &target.route
        && (route.api_base.is_some()
            || route.api_key.is_some()
            || route.api_version.is_some()
            || !route.headers.is_empty())
    {
        return true;
    }

    match target.provider {
        Provider::OpenAI => {
            state.provider.openai_api_key.is_some()
                || !state.provider.openai_headers.is_empty()
                || state.provider.openai_base_url != DEFAULT_OPENAI_BASE_URL
        }
        Provider::Anthropic => {
            state.provider.anthropic_api_key.is_some()
                || state.provider.anthropic_auth_token.is_some()
                || !state.provider.anthropic_headers.is_empty()
                || state.provider.anthropic_base_url != DEFAULT_ANTHROPIC_BASE_URL
        }
        Provider::Ollama => true, // base_url is always present; Ollama typically does not need a key
        Provider::Acp => state.provider.acp_command.is_some(),
    }
}

/// Call the upstream for a resolved target (non-streaming).
pub async fn call_provider(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
    incoming_headers: &HeaderMap,
) -> Result<ChatResponse, ProxyError> {
    match target.provider {
        Provider::OpenAI => call_openai(state, request_id, target, ir, incoming_headers).await,
        Provider::Anthropic => {
            call_anthropic(state, request_id, target, ir, incoming_headers).await
        }
        Provider::Ollama => call_ollama(state, request_id, target, ir, incoming_headers).await,
        Provider::Acp => call_acp(state, request_id, target, ir).await,
    }
}

/// A streaming upstream call: provider kind + raw upstream byte stream.
pub struct ProviderStream {
    /// Which provider produced the stream
    pub provider: Provider,
    /// Raw upstream stream (provider wire format)
    pub body: TransportByteStream,
}

/// Byte stream in the downstream protocol's wire format.
pub type DownstreamByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, Infallible>> + Send + 'static>>;

#[derive(Debug, Clone, Copy)]
/// The downstream API surface a stream is rendered for.
pub enum DownstreamProtocol {
    /// OpenAI SSE (`data:` frames, `[DONE]`)
    OpenAI,
    /// Anthropic SSE (`event:`/`data:` frames)
    Anthropic,
    /// Ollama newline-delimited JSON
    Ollama,
}

/// Start a streaming upstream call for a resolved target.
pub async fn call_provider_stream(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
    incoming_headers: &HeaderMap,
) -> Result<ProviderStream, ProxyError> {
    match target.provider {
        Provider::OpenAI => {
            call_openai_stream(state, request_id, target, ir, incoming_headers).await
        }
        Provider::Anthropic => {
            call_anthropic_stream(state, request_id, target, ir, incoming_headers).await
        }
        Provider::Ollama => {
            call_ollama_stream(state, request_id, target, ir, incoming_headers).await
        }
        Provider::Acp => call_acp_stream(state, request_id, target, ir).await,
    }
}

/// API key for an OpenAI target (route override or global config).
pub fn openai_api_key(state: &AppState, target: &ProviderTarget) -> Option<String> {
    target
        .route
        .as_ref()
        .and_then(|route| route.api_key.clone())
        .or_else(|| state.provider.openai_api_key.clone())
}

/// Base URL for an OpenAI target (route override or global config).
pub fn openai_base_url(state: &AppState, target: &ProviderTarget) -> String {
    target
        .route
        .as_ref()
        .and_then(|route| route.api_base.clone())
        .unwrap_or_else(|| state.provider.openai_base_url.clone())
}

/// Build an OpenAI URL, trimming a trailing `/v1` so `path` decides.
pub fn normalized_openai_url(state: &AppState, target: &ProviderTarget, path: &str) -> String {
    format!(
        "{}{}",
        trim_version_suffix(&openai_base_url(state, target), "v1"),
        path
    )
}

fn anthropic_base_url(state: &AppState, target: &ProviderTarget) -> String {
    target
        .route
        .as_ref()
        .and_then(|route| route.api_base.clone())
        .unwrap_or_else(|| state.provider.anthropic_base_url.clone())
}

fn ollama_base_url(state: &AppState, target: &ProviderTarget) -> String {
    target
        .route
        .as_ref()
        .and_then(|route| route.api_base.clone())
        .unwrap_or_else(|| state.provider.ollama_base_url.clone())
}

fn trim_version_suffix(base_url: &str, suffix: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    let version_suffix = format!("/{suffix}");
    trimmed
        .strip_suffix(&version_suffix)
        .unwrap_or(trimmed)
        .to_string()
}

fn redact_headers(headers: &[(String, String)]) -> Value {
    // Keep this consistent with the HTTP middleware redaction.
    let mut obj = serde_json::Map::new();
    for (k, v) in headers.iter() {
        let name = k.to_ascii_lowercase();
        let value = if matches!(
            name.as_str(),
            "authorization" | "proxy-authorization" | "x-api-key" | "api-key"
        ) {
            "[REDACTED]".to_string()
        } else {
            v.clone()
        };
        obj.insert(name, Value::String(value));
    }
    Value::Object(obj)
}

pub(crate) fn upstream_headers(
    state: &AppState,
    target: &ProviderTarget,
    incoming_headers: &HeaderMap,
) -> Vec<(String, String)> {
    let mut headers = Vec::new();
    merge_header_source(&mut headers, provider_default_headers(state, target));
    merge_header_source(&mut headers, route_headers(target));
    merge_header_source(
        &mut headers,
        forwarded_request_headers(state, target, incoming_headers),
    );
    headers
}

fn provider_default_headers(state: &AppState, target: &ProviderTarget) -> Vec<(String, String)> {
    match target.provider {
        Provider::OpenAI => {
            let mut headers = Vec::new();
            if let Some(key) = state.provider.openai_api_key.clone() {
                headers.push(("authorization".to_string(), format!("Bearer {key}")));
            }
            headers.extend(state.provider.openai_headers.clone());
            headers
        }
        Provider::Anthropic => {
            let mut headers = vec![(
                "anthropic-version".to_string(),
                state.provider.anthropic_version.clone(),
            )];
            if let Some(key) = state.provider.anthropic_api_key.clone() {
                headers.push(("x-api-key".to_string(), key));
            } else if let Some(token) = state.provider.anthropic_auth_token.clone() {
                headers.push(("authorization".to_string(), format!("Bearer {token}")));
            }
            headers.extend(state.provider.anthropic_headers.clone());
            headers
        }
        Provider::Ollama => state.provider.ollama_headers.clone(),
        Provider::Acp => Vec::new(),
    }
}

fn route_headers(target: &ProviderTarget) -> Vec<(String, String)> {
    let Some(route) = &target.route else {
        return Vec::new();
    };

    let mut headers = Vec::new();
    match target.provider {
        Provider::OpenAI => {
            if let Some(key) = route.api_key.clone() {
                headers.push(("authorization".to_string(), format!("Bearer {key}")));
            }
        }
        Provider::Anthropic => {
            if let Some(version) = route.api_version.clone() {
                headers.push(("anthropic-version".to_string(), version));
            }
            if let Some(key) = route.api_key.clone() {
                headers.push(("x-api-key".to_string(), key));
            }
        }
        Provider::Ollama | Provider::Acp => {}
    }
    headers.extend(route.headers.clone());
    headers
}

fn forwarded_request_headers(
    state: &AppState,
    target: &ProviderTarget,
    incoming_headers: &HeaderMap,
) -> Vec<(String, String)> {
    let allowlist = forward_header_allowlist(state, target);
    let mut headers = Vec::new();
    for (name, value) in incoming_headers {
        let name = name.as_str().to_ascii_lowercase();
        if !allowlist.contains(&name) || !is_forwardable_header(&name) {
            continue;
        }
        let Ok(value) = value.to_str() else {
            continue;
        };
        headers.push((name, value.to_string()));
    }
    headers
}

fn forward_header_allowlist(state: &AppState, target: &ProviderTarget) -> Vec<String> {
    let mut allowlist = state.provider.forward_headers.clone();
    if let Some(route) = &target.route {
        for header in &route.forward_headers {
            if !allowlist
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(header))
            {
                allowlist.push(header.to_ascii_lowercase());
            }
        }
    }
    allowlist
}

fn merge_header_source(headers: &mut Vec<(String, String)>, source: Vec<(String, String)>) {
    if source.iter().any(|(name, _)| is_auth_header(name)) {
        headers.retain(|(name, _)| !is_auth_header(name));
    }
    for (name, value) in source {
        insert_or_replace_header(headers, name, value);
    }
}

fn insert_or_replace_header(headers: &mut Vec<(String, String)>, name: String, value: String) {
    if let Some((existing_name, existing_value)) = headers
        .iter_mut()
        .find(|(existing_name, _)| existing_name.eq_ignore_ascii_case(&name))
    {
        *existing_name = name;
        *existing_value = value;
    } else {
        headers.push((name, value));
    }
}

fn is_auth_header(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "authorization" | "x-api-key" | "api-key"
    )
}

fn is_forwardable_header(name: &str) -> bool {
    !matches!(
        name,
        "connection"
            | "content-length"
            | "host"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

async fn call_openai(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
    incoming_headers: &HeaderMap,
) -> Result<ChatResponse, ProxyError> {
    let url = normalized_openai_url(state, target, "/v1/chat/completions");
    let body = ir_to_openai_request_json(ir);
    let started = Instant::now();

    let headers = upstream_headers(state, target, incoming_headers);
    state.record_monitor_upstream_url(request_id, &url);

    tracing::info!(
        event = "provider.out",
        request_id,
        provider = "openai",
        method = "POST",
        url = %url,
        body = %redact_json_secrets(&body),
    );

    let resp = state
        .transport
        .send(TransportRequest {
            method: "POST",
            url: url.clone(),
            headers,
            body: body.clone(),
        })
        .await
        .map_err(|e| ProxyError {
            status: 502,
            message: format!("Failed to call OpenAI upstream: {}", e.message),
            upstream_body: None,
        })?;

    let status = resp.status;
    let parsed: Value = serde_json::from_slice(&resp.body).unwrap_or_else(|_| json!({}));

    tracing::info!(
        event = "provider.in",
        request_id,
        provider = "openai",
        status,
        headers = %redact_headers(&resp.headers),
        body = %redact_json_secrets(&parsed),
        latency_ms = started.elapsed().as_millis(),
    );

    if !(200..300).contains(&status) {
        return Err(ProxyError {
            status,
            message: "OpenAI upstream returned error".to_string(),
            upstream_body: Some(parsed),
        });
    }

    openai_response_json_to_ir(parsed).ok_or_else(|| ProxyError {
        status: 502,
        message: "Failed to parse OpenAI response".to_string(),
        upstream_body: None,
    })
}

async fn call_anthropic(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
    incoming_headers: &HeaderMap,
) -> Result<ChatResponse, ProxyError> {
    let url = format!(
        "{}/v1/messages",
        trim_version_suffix(&anthropic_base_url(state, target), "v1")
    );
    let body = ir_to_anthropic_request_json(ir);
    let started = Instant::now();

    let headers = upstream_headers(state, target, incoming_headers);
    state.record_monitor_upstream_url(request_id, &url);

    tracing::info!(
        event = "provider.out",
        request_id,
        provider = "anthropic",
        method = "POST",
        url = %url,
        body = %redact_json_secrets(&body),
    );

    let resp = state
        .transport
        .send(TransportRequest {
            method: "POST",
            url: url.clone(),
            headers,
            body: body.clone(),
        })
        .await
        .map_err(|e| ProxyError {
            status: 502,
            message: format!("Failed to call Anthropic upstream: {}", e.message),
            upstream_body: None,
        })?;

    let status = resp.status;
    let parsed: Value = serde_json::from_slice(&resp.body).unwrap_or_else(|_| json!({}));

    tracing::info!(
        event = "provider.in",
        request_id,
        provider = "anthropic",
        status,
        headers = %redact_headers(&resp.headers),
        body = %redact_json_secrets(&parsed),
        latency_ms = started.elapsed().as_millis(),
    );

    if !(200..300).contains(&status) {
        return Err(ProxyError {
            status,
            message: "Anthropic upstream returned error".to_string(),
            upstream_body: Some(parsed),
        });
    }

    anthropic_response_json_to_ir(parsed).ok_or_else(|| ProxyError {
        status: 502,
        message: "Failed to parse Anthropic response".to_string(),
        upstream_body: None,
    })
}

async fn call_ollama(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
    incoming_headers: &HeaderMap,
) -> Result<ChatResponse, ProxyError> {
    let url = format!(
        "{}/api/chat",
        ollama_base_url(state, target).trim_end_matches('/')
    );
    let body = ir_to_ollama_request_json(ir);
    let started = Instant::now();
    state.record_monitor_upstream_url(request_id, &url);

    tracing::info!(
        event = "provider.out",
        request_id,
        provider = "ollama",
        method = "POST",
        url = %url,
        body = %redact_json_secrets(&body),
    );

    let resp = state
        .transport
        .send(TransportRequest {
            method: "POST",
            url: url.clone(),
            headers: upstream_headers(state, target, incoming_headers),
            body: body.clone(),
        })
        .await
        .map_err(|e| ProxyError {
            status: 502,
            message: format!("Failed to call Ollama upstream: {}", e.message),
            upstream_body: None,
        })?;

    let status = resp.status;
    let parsed: Value = serde_json::from_slice(&resp.body).unwrap_or_else(|_| json!({}));

    tracing::info!(
        event = "provider.in",
        request_id,
        provider = "ollama",
        status,
        headers = %redact_headers(&resp.headers),
        body = %redact_json_secrets(&parsed),
        latency_ms = started.elapsed().as_millis(),
    );

    if !(200..300).contains(&status) {
        return Err(ProxyError {
            status,
            message: "Ollama upstream returned error".to_string(),
            upstream_body: Some(parsed),
        });
    }

    ollama_response_json_to_ir(parsed).ok_or_else(|| ProxyError {
        status: 502,
        message: "Failed to parse Ollama response".to_string(),
        upstream_body: None,
    })
}

async fn call_acp(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
) -> Result<ChatResponse, ProxyError> {
    let Some(command) = state.provider.acp_command.clone() else {
        return Err(ProxyError {
            status: 400,
            message: "ACP upstream is not configured; set YALLM_ACP_COMMAND".to_string(),
            upstream_body: None,
        });
    };

    let cwd = state
        .provider
        .acp_cwd
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")));
    let started = Instant::now();
    state.record_monitor_upstream_url(request_id, &format!("acp://{}", target.upstream_model));

    tracing::info!(
        event = "provider.out",
        request_id,
        provider = "acp",
        command = %command,
        cwd = %cwd.display(),
        model = %target.upstream_model,
    );

    let response = yallm_acp::complete_with_command(&command, cwd, ir.clone())
        .await
        .map_err(|e| ProxyError {
            status: 502,
            message: format!("Failed to call ACP upstream: {e}"),
            upstream_body: None,
        })?;

    tracing::info!(
        event = "provider.in",
        request_id,
        provider = "acp",
        latency_ms = started.elapsed().as_millis(),
    );

    Ok(response)
}

async fn call_acp_stream(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
) -> Result<ProviderStream, ProxyError> {
    let Some(command) = state.provider.acp_command.clone() else {
        return Err(ProxyError {
            status: 400,
            message: "ACP upstream is not configured; set YALLM_ACP_COMMAND".to_string(),
            upstream_body: None,
        });
    };

    let cwd = state
        .provider
        .acp_cwd
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")));
    let started = Instant::now();
    state.record_monitor_upstream_url(request_id, &format!("acp://{}", target.upstream_model));

    tracing::info!(
        event = "provider.out.stream",
        request_id,
        provider = "acp",
        command = %command,
        cwd = %cwd.display(),
        model = %target.upstream_model,
    );

    let events =
        yallm_acp::stream_with_command(&command, cwd, ir.clone()).map_err(|e| ProxyError {
            status: 502,
            message: format!("Failed to start ACP upstream stream: {e}"),
            upstream_body: None,
        })?;

    tracing::info!(
        event = "provider.in.stream",
        request_id,
        provider = "acp",
        latency_ms = started.elapsed().as_millis(),
    );

    let body = events.map(|event| {
        event
            .map(acp_stream_event_to_line)
            .map(Bytes::from)
            .map_err(|message| TransportError { message })
    });

    Ok(ProviderStream {
        provider: Provider::Acp,
        body: Box::pin(body),
    })
}

fn acp_stream_event_to_line(event: yallm_acp::AcpStreamEvent) -> String {
    let payload = match event {
        yallm_acp::AcpStreamEvent::TextDelta(text) => {
            json!({ "type": "text_delta", "text": text })
        }
        yallm_acp::AcpStreamEvent::Stop { finish_reason } => {
            json!({ "type": "stop", "finish_reason": finish_reason })
        }
    };
    format!("{}\n", json_to_compact_string(&payload))
}

fn extract_last_user_text(ir: &ChatRequest) -> Option<String> {
    ir.messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| {
            m.content
                .iter()
                .find_map(|c| match c {
                    Content::Text(t) => Some(t.text.clone()),
                    _ => None,
                })
                .unwrap_or_default()
        })
        .filter(|s| !s.is_empty())
}

fn mock_reply(last_user_text: Option<String>) -> String {
    match last_user_text {
        Some(text) => format!("yallm (mock): {text}"),
        None => "yallm (mock): hello".to_string(),
    }
}

fn mock_ir_response(model: &str, reply: String) -> ChatResponse {
    ChatResponse {
        id: format!("mock_{}", unix_seconds()),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message::text(Role::Assistant, reply),
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(Usage::default()),
    }
}

async fn call_openai_stream(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
    incoming_headers: &HeaderMap,
) -> Result<ProviderStream, ProxyError> {
    let url = normalized_openai_url(state, target, "/v1/chat/completions");
    let body = with_stream_flag(ir_to_openai_request_json(ir), true);
    let started = Instant::now();
    let headers = upstream_headers(state, target, incoming_headers);
    state.record_monitor_upstream_url(request_id, &url);

    tracing::info!(
        event = "provider.out",
        request_id,
        provider = "openai",
        method = "POST",
        url = %url,
        body = %redact_json_secrets(&body),
    );

    let resp = state
        .transport
        .send_stream(TransportRequest {
            method: "POST",
            url: url.clone(),
            headers,
            body: body.clone(),
        })
        .await
        .map_err(|e| ProxyError {
            status: 502,
            message: format!("Failed to call OpenAI upstream: {}", e.message),
            upstream_body: None,
        })?;

    tracing::info!(
        event = "provider.in.stream",
        request_id,
        provider = "openai",
        status = resp.status,
        headers = %redact_headers(&resp.headers),
        latency_ms = started.elapsed().as_millis(),
    );

    if !(200..300).contains(&resp.status) {
        let parsed = collect_stream_json(resp.body).await;
        return Err(ProxyError {
            status: resp.status,
            message: "OpenAI upstream returned error".to_string(),
            upstream_body: Some(parsed),
        });
    }

    Ok(ProviderStream {
        provider: Provider::OpenAI,
        body: resp.body,
    })
}

async fn call_anthropic_stream(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
    incoming_headers: &HeaderMap,
) -> Result<ProviderStream, ProxyError> {
    let url = format!(
        "{}/v1/messages",
        trim_version_suffix(&anthropic_base_url(state, target), "v1")
    );
    let body = with_stream_flag(ir_to_anthropic_request_json(ir), true);
    let started = Instant::now();

    let headers = upstream_headers(state, target, incoming_headers);
    state.record_monitor_upstream_url(request_id, &url);

    tracing::info!(
        event = "provider.out",
        request_id,
        provider = "anthropic",
        method = "POST",
        url = %url,
        body = %redact_json_secrets(&body),
    );

    let resp = state
        .transport
        .send_stream(TransportRequest {
            method: "POST",
            url: url.clone(),
            headers,
            body: body.clone(),
        })
        .await
        .map_err(|e| ProxyError {
            status: 502,
            message: format!("Failed to call Anthropic upstream: {}", e.message),
            upstream_body: None,
        })?;

    tracing::info!(
        event = "provider.in.stream",
        request_id,
        provider = "anthropic",
        status = resp.status,
        headers = %redact_headers(&resp.headers),
        latency_ms = started.elapsed().as_millis(),
    );

    if !(200..300).contains(&resp.status) {
        let parsed = collect_stream_json(resp.body).await;
        return Err(ProxyError {
            status: resp.status,
            message: "Anthropic upstream returned error".to_string(),
            upstream_body: Some(parsed),
        });
    }

    Ok(ProviderStream {
        provider: Provider::Anthropic,
        body: resp.body,
    })
}

async fn call_ollama_stream(
    state: &AppState,
    request_id: u64,
    target: &ProviderTarget,
    ir: &ChatRequest,
    incoming_headers: &HeaderMap,
) -> Result<ProviderStream, ProxyError> {
    let url = format!(
        "{}/api/chat",
        ollama_base_url(state, target).trim_end_matches('/')
    );
    let body = with_stream_flag(ir_to_ollama_request_json(ir), true);
    let started = Instant::now();
    state.record_monitor_upstream_url(request_id, &url);

    tracing::info!(
        event = "provider.out",
        request_id,
        provider = "ollama",
        method = "POST",
        url = %url,
        body = %redact_json_secrets(&body),
    );

    let resp = state
        .transport
        .send_stream(TransportRequest {
            method: "POST",
            url: url.clone(),
            headers: upstream_headers(state, target, incoming_headers),
            body: body.clone(),
        })
        .await
        .map_err(|e| ProxyError {
            status: 502,
            message: format!("Failed to call Ollama upstream: {}", e.message),
            upstream_body: None,
        })?;

    tracing::info!(
        event = "provider.in.stream",
        request_id,
        provider = "ollama",
        status = resp.status,
        headers = %redact_headers(&resp.headers),
        latency_ms = started.elapsed().as_millis(),
    );

    if !(200..300).contains(&resp.status) {
        let parsed = collect_stream_json(resp.body).await;
        return Err(ProxyError {
            status: resp.status,
            message: "Ollama upstream returned error".to_string(),
            upstream_body: Some(parsed),
        });
    }

    Ok(ProviderStream {
        provider: Provider::Ollama,
        body: resp.body,
    })
}

async fn collect_stream_json(mut body: TransportByteStream) -> Value {
    let mut out = Vec::new();
    while let Some(chunk) = body.next().await {
        let Ok(bytes) = chunk else { break };
        out.extend_from_slice(&bytes);
    }
    serde_json::from_slice(&out).unwrap_or_else(|_| json!({}))
}

fn with_stream_flag(mut body: Value, stream: bool) -> Value {
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".to_string(), json!(stream));
    }
    body
}

// ============================================================================
// Downstream (any) -> IR
// ============================================================================

/// Parse an OpenAI `/v1/chat/completions` request body into IR.
pub fn openai_downstream_to_ir(req: &Value) -> Option<ChatRequest> {
    let model = req.get("model")?.as_str()?.to_string();
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let temperature = req
        .get("temperature")
        .and_then(|v| v.as_f64())
        .map(|n| n as f32);
    let top_p = req.get("top_p").and_then(|v| v.as_f64()).map(|n| n as f32);

    let mut messages_out = Vec::new();
    let messages = req.get("messages")?.as_array()?;
    for m in messages {
        let role_s = m.get("role")?.as_str().unwrap_or("user");
        let role = match role_s {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        };

        let mut content = Vec::new();
        if role != Role::Tool
            && let Some(c) = m.get("content")
        {
            let text = openai_content_to_text(c);
            if !text.is_empty() {
                content.push(Content::text(text));
            }
        }

        // tool_calls (assistant)
        if let Some(tcs) = m.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let args = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if !name.is_empty() {
                    content.push(Content::ToolCall(yallm_ir::ToolCallContent {
                        id,
                        name,
                        arguments: args,
                    }));
                }
            }
        }

        // tool result (tool role)
        if role == Role::Tool {
            let tool_call_id = m
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let text = m
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if !text.is_empty() {
                content.push(Content::ToolResult(yallm_ir::ToolResultContent {
                    tool_call_id,
                    content: text,
                }));
            }
        }

        messages_out.push(
            Message::new(role, content)
                .with_source(Source::OpenAI)
                .with_raw(m.clone()),
        );
    }

    Some(ChatRequest {
        model,
        messages: messages_out,
        max_tokens,
        temperature,
        top_p,
        stream,
    })
}

/// Parse an Anthropic `/v1/messages` request body into IR.
pub fn anthropic_downstream_to_ir(req: &Value) -> Option<ChatRequest> {
    let model = req.get("model")?.as_str()?.to_string();
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as u32);
    let temperature = req
        .get("temperature")
        .and_then(|v| v.as_f64())
        .map(|n| n as f32);
    let top_p = req.get("top_p").and_then(|v| v.as_f64()).map(|n| n as f32);

    let mut messages_out = Vec::new();

    if let Some(system) = req.get("system") {
        let sys_text = match system {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !sys_text.is_empty() {
            messages_out.push(Message::text(Role::System, sys_text).with_source(Source::Anthropic));
        }
    }

    let messages = req.get("messages")?.as_array()?;
    for m in messages {
        let role_s = m.get("role")?.as_str().unwrap_or("user");
        let role = match role_s {
            "assistant" => Role::Assistant,
            _ => Role::User,
        };
        let mut content = Vec::new();
        if let Some(c) = m.get("content") {
            let text = anthropic_content_to_text(c);
            if !text.is_empty() {
                content.push(Content::text(text));
            }
        }
        messages_out.push(
            Message::new(role, content)
                .with_source(Source::Anthropic)
                .with_raw(m.clone()),
        );
    }

    Some(ChatRequest {
        model,
        messages: messages_out,
        max_tokens,
        temperature,
        top_p,
        stream,
    })
}

/// Parse an Ollama `/api/chat` request body into IR.
pub fn ollama_downstream_to_ir(req: &Value) -> Option<ChatRequest> {
    let model = req.get("model")?.as_str()?.to_string();
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let (temperature, top_p) = req
        .get("options")
        .and_then(|o| o.as_object())
        .map(|o| {
            let t = o
                .get("temperature")
                .and_then(|v| v.as_f64())
                .map(|n| n as f32);
            let p = o.get("top_p").and_then(|v| v.as_f64()).map(|n| n as f32);
            (t, p)
        })
        .unwrap_or((None, None));

    let mut messages_out = Vec::new();
    let messages = req.get("messages")?.as_array()?;
    for m in messages {
        let role_s = m.get("role")?.as_str().unwrap_or("user");
        let role = match role_s {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        };
        let text = m
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let content = if text.is_empty() {
            Vec::new()
        } else {
            vec![Content::text(text)]
        };
        messages_out.push(
            Message::new(role, content)
                .with_source(Source::Ollama)
                .with_raw(m.clone()),
        );
    }

    Some(ChatRequest {
        model,
        messages: messages_out,
        max_tokens: None,
        temperature,
        top_p,
        stream,
    })
}

fn openai_content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                let obj = p.as_object()?;
                match obj.get("type")?.as_str()? {
                    "text" => obj.get("text")?.as_str().map(|s| s.to_string()),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn anthropic_content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                let obj = b.as_object()?;
                match obj.get("type")?.as_str()? {
                    "text" => obj.get("text")?.as_str().map(|s| s.to_string()),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

// ============================================================================
// IR -> provider request JSON
// ============================================================================

fn ir_to_openai_request_json(ir: &ChatRequest) -> Value {
    let messages: Vec<Value> = ir.messages.iter().map(ir_message_to_openai).collect();

    let mut obj = serde_json::Map::new();
    obj.insert("model".to_string(), Value::String(ir.model.clone()));
    obj.insert("messages".to_string(), Value::Array(messages));
    if let Some(t) = ir.temperature {
        obj.insert("temperature".to_string(), json!(t));
    }
    if let Some(p) = ir.top_p {
        obj.insert("top_p".to_string(), json!(p));
    }
    if let Some(mt) = ir.max_tokens {
        obj.insert("max_tokens".to_string(), json!(mt));
    }
    obj.insert("stream".to_string(), json!(false));
    Value::Object(obj)
}

fn ir_message_to_openai(m: &Message) -> Value {
    let role = m.role.as_str();
    let text = join_text(&m.content, "\n");

    let mut obj = serde_json::Map::new();
    obj.insert("role".to_string(), Value::String(role.to_string()));
    obj.insert("content".to_string(), Value::String(text));

    if m.role == Role::Tool {
        let tool_call_id = m
            .content
            .iter()
            .find_map(|c| match c {
                Content::ToolResult(tr) => Some(tr.tool_call_id.clone()),
                _ => None,
            })
            .unwrap_or_default();
        obj.insert("tool_call_id".to_string(), Value::String(tool_call_id));
    }

    if m.role == Role::Assistant {
        let tool_calls: Vec<Value> = m
            .content
            .iter()
            .filter_map(|c| match c {
                Content::ToolCall(tc) => Some(json!({
                    "id": tc.id,
                    "type": "function",
                    "function": { "name": tc.name, "arguments": tc.arguments }
                })),
                _ => None,
            })
            .collect();
        if !tool_calls.is_empty() {
            obj.insert("tool_calls".to_string(), Value::Array(tool_calls));
        }
    }

    Value::Object(obj)
}

fn ir_to_anthropic_request_json(ir: &ChatRequest) -> Value {
    let system = ir
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| join_text(&m.content, "\n"))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    let messages: Vec<Value> = ir
        .messages
        .iter()
        .filter(|m| matches!(m.role, Role::User | Role::Assistant))
        .map(|m| {
            json!({
                "role": m.role.as_str(),
                "content": join_text(&m.content, "\n"),
            })
        })
        .collect();

    let mut obj = serde_json::Map::new();
    obj.insert("model".to_string(), Value::String(ir.model.clone()));
    obj.insert(
        "max_tokens".to_string(),
        json!(ir.max_tokens.unwrap_or(1024)),
    );
    obj.insert("messages".to_string(), Value::Array(messages));
    obj.insert("stream".to_string(), json!(false));
    if !system.is_empty() {
        obj.insert("system".to_string(), Value::String(system));
    }
    if let Some(t) = ir.temperature {
        obj.insert("temperature".to_string(), json!(t));
    }
    if let Some(p) = ir.top_p {
        obj.insert("top_p".to_string(), json!(p));
    }
    Value::Object(obj)
}

fn ir_to_ollama_request_json(ir: &ChatRequest) -> Value {
    let messages: Vec<Value> = ir
        .messages
        .iter()
        .map(|m| {
            json!({
                "role": m.role.as_str(),
                "content": join_text(&m.content, "\n"),
            })
        })
        .collect();

    let options = if ir.temperature.is_some() || ir.top_p.is_some() {
        json!({
            "temperature": ir.temperature,
            "top_p": ir.top_p,
        })
    } else {
        Value::Null
    };

    let mut obj = serde_json::Map::new();
    obj.insert("model".to_string(), Value::String(ir.model.clone()));
    obj.insert("messages".to_string(), Value::Array(messages));
    obj.insert("stream".to_string(), json!(false));
    if !options.is_null() {
        obj.insert("options".to_string(), options);
    }
    Value::Object(obj)
}

fn join_text(content: &[Content], sep: &str) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.text.as_str()),
            Content::ToolResult(tr) => Some(tr.content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(sep)
}

// ============================================================================
// Provider response JSON -> IR
// ============================================================================

fn openai_response_json_to_ir(v: Value) -> Option<ChatResponse> {
    let id = v.get("id")?.as_str()?.to_string();
    let model = v.get("model")?.as_str()?.to_string();
    let choices_v = v.get("choices")?.as_array()?.first()?.clone();
    let msg_v = choices_v.get("message")?.clone();
    let content_text = msg_v
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let finish_reason = choices_v
        .get("finish_reason")
        .and_then(|r| r.as_str())
        .map(|s| s.to_string());

    let mut content = Vec::new();
    if !content_text.is_empty() {
        content.push(Content::text(content_text));
    }

    // tool_calls
    if let Some(tcs) = msg_v.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tcs {
            let id = tc
                .get("id")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            if !name.is_empty() {
                content.push(Content::ToolCall(yallm_ir::ToolCallContent {
                    id,
                    name,
                    arguments: args,
                }));
            }
        }
    }

    let usage = v.get("usage").and_then(|u| {
        Some(Usage {
            prompt_tokens: u.get("prompt_tokens")?.as_u64()? as u32,
            completion_tokens: u.get("completion_tokens")?.as_u64()? as u32,
            total_tokens: u.get("total_tokens")?.as_u64()? as u32,
        })
    });

    Some(ChatResponse {
        id,
        model: model.clone(),
        choices: vec![Choice {
            index: 0,
            message: Message::new(Role::Assistant, content)
                .with_source(Source::OpenAI)
                .with_raw(msg_v),
            finish_reason,
        }],
        usage,
    })
}

fn anthropic_response_json_to_ir(v: Value) -> Option<ChatResponse> {
    let id = v.get("id")?.as_str()?.to_string();
    let model = v.get("model")?.as_str()?.to_string();

    let content_blocks = v.get("content").and_then(|c| c.as_array());
    let mut content = Vec::new();

    // DeepSeek may put reasoning in a top-level field instead of content blocks.
    if let Some(reasoning) = v
        .get("reasoning_content")
        .and_then(|r| r.as_str())
        .filter(|s| !s.is_empty())
    {
        content.push(Content::text(reasoning.to_string()));
    }

    let Some(content_blocks) = content_blocks else {
        // No content blocks at all — maybe a bare error or unsupported format.
        if content.is_empty() {
            return None;
        }
        return Some(ChatResponse {
            id,
            model: model.clone(),
            choices: vec![Choice {
                index: 0,
                message: Message::new(Role::Assistant, content).with_source(Source::Anthropic),
                finish_reason: v
                    .get("stop_reason")
                    .and_then(|r| r.as_str())
                    .map(|s| s.to_string()),
            }],
            usage: v.get("usage").and_then(|u| {
                Some(Usage {
                    prompt_tokens: u.get("input_tokens")?.as_u64()? as u32,
                    completion_tokens: u.get("output_tokens")?.as_u64()? as u32,
                    total_tokens: (u.get("input_tokens")?.as_u64()?
                        + u.get("output_tokens")?.as_u64()?)
                        as u32,
                })
            }),
        });
    };
    for b in content_blocks {
        let ty = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match ty {
            "text" => {
                let text = b.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if !text.is_empty() {
                    content.push(Content::text(text.to_string()));
                }
            }
            "tool_use" | "server_tool_use" => {
                let id = b
                    .get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = b
                    .get("name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = b.get("input").cloned().unwrap_or(Value::Null);
                if !name.is_empty() {
                    content.push(Content::ToolCall(yallm_ir::ToolCallContent {
                        id,
                        name,
                        arguments: serde_json::to_string(&input).unwrap_or_default(),
                    }));
                }
            }
            "thinking" | "reasoning" => {
                let text = b
                    .get("thinking")
                    .or(b.get("reasoning"))
                    .or(b.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                if !text.is_empty() {
                    content.push(Content::text(text.to_string()));
                }
            }
            "redacted_thinking" => {
                let text = b.get("data").and_then(|t| t.as_str()).unwrap_or("");
                if !text.is_empty() {
                    content.push(Content::text(format!("[redacted thinking: {text}]")));
                }
            }
            _ => {}
        }
    }

    let finish_reason = v
        .get("stop_reason")
        .and_then(|r| r.as_str())
        .map(|s| s.to_string());

    let usage = v.get("usage").and_then(|u| {
        Some(Usage {
            prompt_tokens: u.get("input_tokens")?.as_u64()? as u32,
            completion_tokens: u.get("output_tokens")?.as_u64()? as u32,
            total_tokens: (u.get("input_tokens")?.as_u64()? + u.get("output_tokens")?.as_u64()?)
                as u32,
        })
    });

    Some(ChatResponse {
        id,
        model: model.clone(),
        choices: vec![Choice {
            index: 0,
            message: Message::new(Role::Assistant, content).with_source(Source::Anthropic),
            finish_reason,
        }],
        usage,
    })
}

fn ollama_response_json_to_ir(v: Value) -> Option<ChatResponse> {
    let model = v.get("model")?.as_str()?.to_string();
    let done = v.get("done").and_then(|d| d.as_bool()).unwrap_or(true);
    let msg = v.get("message")?.as_object()?;
    let role = msg
        .get("role")
        .and_then(|r| r.as_str())
        .unwrap_or("assistant");
    let content_text = msg
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let finish_reason = if done { Some("stop".to_string()) } else { None };

    let usage = Some(Usage {
        prompt_tokens: v
            .get("prompt_eval_count")
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as u32,
        completion_tokens: v.get("eval_count").and_then(|n| n.as_u64()).unwrap_or(0) as u32,
        total_tokens: (v
            .get("prompt_eval_count")
            .and_then(|n| n.as_u64())
            .unwrap_or(0)
            + v.get("eval_count").and_then(|n| n.as_u64()).unwrap_or(0))
            as u32,
    });

    let mut content = Vec::new();
    if !content_text.is_empty() {
        content.push(Content::text(content_text));
    }

    Some(ChatResponse {
        id: model.clone(),
        model: model.clone(),
        choices: vec![Choice {
            index: 0,
            message: Message::new(
                match role {
                    "assistant" => Role::Assistant,
                    "user" => Role::User,
                    "system" => Role::System,
                    "tool" => Role::Tool,
                    _ => Role::Assistant,
                },
                content,
            )
            .with_source(Source::Ollama),
            finish_reason,
        }],
        usage,
    })
}

pub use stream::map_provider_stream_to_downstream;

// ============================================================================
// IR -> downstream response JSON
// ============================================================================

/// Render an IR response as an OpenAI `/v1/chat/completions` body.
pub fn ir_to_openai_downstream_response(ir_resp: &ChatResponse, model: &str) -> Value {
    let created = unix_seconds();
    let (content_text, reasoning_text, tool_calls) = ir_assistant_to_openai_message(ir_resp);

    let mut msg = serde_json::Map::new();
    msg.insert("role".to_string(), Value::String("assistant".to_string()));
    msg.insert("content".to_string(), Value::String(content_text));
    if !reasoning_text.is_empty() {
        msg.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_text),
        );
    }
    if let Some(tcs) = tool_calls {
        msg.insert("tool_calls".to_string(), Value::Array(tcs));
    }

    let mut choice = serde_json::Map::new();
    choice.insert("index".to_string(), json!(0));
    choice.insert("message".to_string(), Value::Object(msg));
    choice.insert(
        "finish_reason".to_string(),
        Value::String(
            ir_resp
                .choices
                .first()
                .and_then(|c| c.finish_reason.clone())
                .unwrap_or_else(|| "stop".to_string()),
        ),
    );

    let mut obj = serde_json::Map::new();
    obj.insert("id".to_string(), Value::String(ir_resp.id.clone()));
    obj.insert(
        "object".to_string(),
        Value::String("chat.completion".to_string()),
    );
    obj.insert("created".to_string(), json!(created));
    obj.insert("model".to_string(), Value::String(model.to_string()));
    obj.insert(
        "choices".to_string(),
        Value::Array(vec![Value::Object(choice)]),
    );
    if let Some(u) = &ir_resp.usage {
        obj.insert(
            "usage".to_string(),
            json!({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }),
        );
    }
    Value::Object(obj)
}

/// Render an IR response as an OpenAI SSE stream body.
pub fn ir_to_openai_downstream_stream(ir_resp: &ChatResponse, model: &str) -> String {
    let created = unix_seconds();
    let (content_text, reasoning_text, tool_calls) = ir_assistant_to_openai_message(ir_resp);

    let mut delta = serde_json::Map::new();
    delta.insert("role".to_string(), Value::String("assistant".to_string()));
    if !reasoning_text.is_empty() {
        delta.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_text),
        );
    }
    if !content_text.is_empty() {
        delta.insert("content".to_string(), Value::String(content_text));
    }
    if let Some(tcs) = tool_calls {
        delta.insert("tool_calls".to_string(), Value::Array(tcs));
    }

    let start_chunk = json!({
        "id": ir_resp.id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": Value::Null,
        }]
    });

    let stop_chunk = json!({
        "id": ir_resp.id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": ir_resp
                .choices
                .first()
                .and_then(|c| c.finish_reason.clone())
                .unwrap_or_else(|| "stop".to_string()),
        }]
    });

    let mut out = String::new();
    out.push_str(&sse_data_frame(start_chunk));
    out.push_str(&sse_data_frame(stop_chunk));
    out.push_str("data: [DONE]\n\n");
    out
}

fn ir_assistant_to_openai_message(ir_resp: &ChatResponse) -> (String, String, Option<Vec<Value>>) {
    let msg = ir_resp.choices.first().map(|c| &c.message);
    let Some(msg) = msg else {
        return (String::new(), String::new(), None);
    };

    let content_text = join_text(&msg.content, "\n");
    let reasoning_text = msg
        .raw
        .as_ref()
        .map(openai_reasoning_text)
        .unwrap_or_default();
    let tool_calls: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|c| match c {
            Content::ToolCall(tc) => Some(json!({
                "id": tc.id,
                "type": "function",
                "function": { "name": tc.name, "arguments": tc.arguments }
            })),
            _ => None,
        })
        .collect();

    (
        content_text,
        reasoning_text,
        if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
    )
}

/// Render an IR response as an Anthropic `/v1/messages` body.
pub fn ir_to_anthropic_downstream_response(ir_resp: &ChatResponse, model: &str) -> Value {
    let (text, reasoning_text) = ir_assistant_text_and_reasoning(ir_resp);

    let usage = ir_resp.usage.as_ref().map(|u| {
        json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
        })
    });

    let mut obj = serde_json::Map::new();
    obj.insert("id".to_string(), Value::String(ir_resp.id.clone()));
    obj.insert("type".to_string(), Value::String("message".to_string()));
    obj.insert("role".to_string(), Value::String("assistant".to_string()));
    obj.insert(
        "content".to_string(),
        anthropic_content_blocks(&ir_resp.id, reasoning_text, text),
    );
    obj.insert("model".to_string(), Value::String(model.to_string()));
    obj.insert(
        "stop_reason".to_string(),
        Value::String(map_to_anthropic_stop_reason(
            ir_resp
                .choices
                .first()
                .and_then(|c| c.finish_reason.as_deref()),
        )),
    );
    obj.insert("stop_sequence".to_string(), Value::Null);
    obj.insert(
        "usage".to_string(),
        usage.unwrap_or_else(|| json!({"input_tokens": 0, "output_tokens": 0})),
    );
    Value::Object(obj)
}

/// Render an IR response as an Anthropic SSE stream body.
pub fn ir_to_anthropic_downstream_stream(ir_resp: &ChatResponse, model: &str) -> String {
    let (text, reasoning_text) = ir_assistant_text_and_reasoning(ir_resp);
    let prompt_tokens = ir_resp.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0);
    let completion_tokens = ir_resp
        .usage
        .as_ref()
        .map(|u| u.completion_tokens)
        .unwrap_or(0);
    let stop_reason = map_to_anthropic_stop_reason(
        ir_resp
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_deref()),
    );

    let mut out = String::new();
    out.push_str(&sse_event_frame(
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": ir_resp.id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": 0,
                }
            }
        }),
    ));

    let mut next_index = 0;
    if !reasoning_text.is_empty() {
        let index = next_index;
        next_index += 1;
        out.push_str(&sse_event_frame(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                }
            }),
        ));
        out.push_str(&sse_event_frame(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": reasoning_text,
                }
            }),
        ));
        out.push_str(&sse_event_frame(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "signature_delta",
                    "signature": ir_resp.id,
                }
            }),
        ));
        out.push_str(&sse_event_frame(
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": index,
            }),
        ));
    }

    if !text.is_empty() {
        let index = next_index;
        out.push_str(&sse_event_frame(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "text",
                    "text": "",
                }
            }),
        ));
        out.push_str(&sse_event_frame(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "text_delta",
                    "text": text,
                }
            }),
        ));
        out.push_str(&sse_event_frame(
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": index,
            }),
        ));
    }

    out.push_str(&sse_event_frame(
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": Value::Null,
            },
            "usage": {
                "output_tokens": completion_tokens,
            }
        }),
    ));
    out.push_str(&sse_event_frame(
        "message_stop",
        json!({ "type": "message_stop" }),
    ));
    out
}

fn ir_assistant_text_and_reasoning(ir_resp: &ChatResponse) -> (String, String) {
    ir_resp
        .choices
        .first()
        .map(|c| {
            (
                join_text(&c.message.content, "\n"),
                c.message
                    .raw
                    .as_ref()
                    .map(openai_reasoning_text)
                    .unwrap_or_default(),
            )
        })
        .unwrap_or_default()
}

fn anthropic_content_blocks(signature: &str, reasoning_text: String, text: String) -> Value {
    let mut blocks = Vec::new();
    if !reasoning_text.is_empty() {
        blocks.push(json!({
            "type": "thinking",
            "thinking": reasoning_text,
            "signature": signature,
        }));
    }
    if !text.is_empty() || blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    Value::Array(blocks)
}

fn openai_reasoning_text(v: &Value) -> String {
    ["reasoning_content", "reasoning"]
        .into_iter()
        .find_map(|key| {
            v.get(key)
                .and_then(|value| value.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

fn map_to_anthropic_stop_reason(reason: Option<&str>) -> String {
    match reason {
        Some("length") | Some("max_tokens") => "max_tokens".to_string(),
        Some("tool_calls") | Some("tool_use") => "tool_use".to_string(),
        Some("stop_sequence") => "stop_sequence".to_string(),
        Some("stop") | Some("end_turn") => "end_turn".to_string(),
        Some(other) => other.to_string(),
        None => "end_turn".to_string(),
    }
}

/// Render an IR response as an Ollama `/api/chat` body.
pub fn ir_to_ollama_downstream_response(ir_resp: &ChatResponse, model: &str) -> Value {
    let text = ir_resp
        .choices
        .first()
        .map(|c| join_text(&c.message.content, "\n"))
        .unwrap_or_default();

    let mut obj = serde_json::Map::new();
    obj.insert("model".to_string(), Value::String(model.to_string()));
    obj.insert(
        "message".to_string(),
        json!({
            "role": "assistant",
            "content": text,
        }),
    );
    obj.insert("done".to_string(), json!(true));
    if let Some(u) = &ir_resp.usage {
        obj.insert("prompt_eval_count".to_string(), json!(u.prompt_tokens));
        obj.insert("eval_count".to_string(), json!(u.completion_tokens));
    }
    Value::Object(obj)
}

/// Render an IR response as an Ollama stream body.
pub fn ir_to_ollama_downstream_stream(ir_resp: &ChatResponse, model: &str) -> String {
    let text = ir_resp
        .choices
        .first()
        .map(|c| join_text(&c.message.content, "\n"))
        .unwrap_or_default();

    let chunk = json!({
        "model": model,
        "message": {
            "role": "assistant",
            "content": text,
        },
        "done": false,
    });

    let mut done = serde_json::Map::new();
    done.insert("model".to_string(), Value::String(model.to_string()));
    done.insert(
        "message".to_string(),
        json!({
            "role": "assistant",
            "content": "",
        }),
    );
    done.insert("done".to_string(), json!(true));
    if let Some(u) = &ir_resp.usage {
        done.insert("prompt_eval_count".to_string(), json!(u.prompt_tokens));
        done.insert("eval_count".to_string(), json!(u.completion_tokens));
    }

    format!(
        "{}\n{}\n",
        json_to_compact_string(&chunk),
        json_to_compact_string(&Value::Object(done))
    )
}

fn sse_data_frame(payload: Value) -> String {
    format!("data: {}\n\n", json_to_compact_string(&payload))
}

fn sse_event_frame(event: &str, payload: Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        json_to_compact_string(&payload)
    )
}

fn json_to_compact_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string())
}

fn unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
