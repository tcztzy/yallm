use std::{convert::Infallible, pin::Pin, time::Instant};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{Value, json};
use yallm_ir::{ChatRequest, ChatResponse, Choice, Content, Message, Role, Source, Usage};

use crate::{
    logging::redact_json_secrets,
    state::{AppState, Mode, Provider, TransportByteStream, TransportRequest},
};

mod stream;

#[derive(Debug)]
pub struct ProxyError {
    pub status: u16,
    pub message: String,
    pub upstream_body: Option<Value>,
}

fn provider_from_model(model: &str) -> (Option<Provider>, String) {
    // Accept `openai:<m>` / `openai/<m>` (and same for anthropic/ollama).
    for (prefix, p) in [
        ("openai:", Provider::OpenAI),
        ("openai/", Provider::OpenAI),
        ("anthropic:", Provider::Anthropic),
        ("anthropic/", Provider::Anthropic),
        ("ollama:", Provider::Ollama),
        ("ollama/", Provider::Ollama),
    ] {
        if let Some(rest) = model.strip_prefix(prefix) {
            return (Some(p), rest.to_string());
        }
    }
    (None, model.to_string())
}

pub fn choose_provider(state: &AppState, model: &str) -> (Provider, String, String) {
    let (from_prefix, stripped) = provider_from_model(model);
    let provider = from_prefix.unwrap_or(state.default_provider);
    (provider, model.to_string(), stripped)
}

pub fn should_proxy(state: &AppState, provider: Provider) -> Result<bool, ProxyError> {
    match state.mode {
        Mode::Mock => Ok(false),
        Mode::Auto => Ok(provider_is_configured(state, provider)),
        Mode::Proxy => {
            if provider_is_configured(state, provider) {
                Ok(true)
            } else {
                Err(ProxyError {
                    status: 500,
                    message: format!(
                        "Missing configuration for provider {} (set required env vars)",
                        provider.as_str()
                    ),
                    upstream_body: None,
                })
            }
        }
    }
}

fn provider_is_configured(state: &AppState, provider: Provider) -> bool {
    match provider {
        Provider::OpenAI => state.provider.openai_api_key.is_some(),
        Provider::Anthropic => state.provider.anthropic_api_key.is_some(),
        Provider::Ollama => true, // base_url is always present; Ollama typically does not need a key
    }
}

pub async fn call_provider(
    state: &AppState,
    request_id: u64,
    provider: Provider,
    ir: &ChatRequest,
) -> Result<ChatResponse, ProxyError> {
    match provider {
        Provider::OpenAI => call_openai(state, request_id, ir).await,
        Provider::Anthropic => call_anthropic(state, request_id, ir).await,
        Provider::Ollama => call_ollama(state, request_id, ir).await,
    }
}

pub struct ProviderStream {
    pub provider: Provider,
    pub body: TransportByteStream,
}

pub type DownstreamByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, Infallible>> + Send + 'static>>;

#[derive(Debug, Clone, Copy)]
pub enum DownstreamProtocol {
    OpenAI,
    Anthropic,
    Ollama,
}

pub async fn call_provider_stream(
    state: &AppState,
    request_id: u64,
    provider: Provider,
    ir: &ChatRequest,
) -> Result<ProviderStream, ProxyError> {
    match provider {
        Provider::OpenAI => call_openai_stream(state, request_id, ir).await,
        Provider::Anthropic => call_anthropic_stream(state, request_id, ir).await,
        Provider::Ollama => call_ollama_stream(state, request_id, ir).await,
    }
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

async fn call_openai(
    state: &AppState,
    request_id: u64,
    ir: &ChatRequest,
) -> Result<ChatResponse, ProxyError> {
    let Some(key) = state.provider.openai_api_key.clone() else {
        return Err(ProxyError {
            status: 500,
            message: "OPENAI_API_KEY is not set".to_string(),
            upstream_body: None,
        });
    };

    let url = format!(
        "{}/v1/chat/completions",
        state.provider.openai_base_url.trim_end_matches('/')
    );
    let body = ir_to_openai_request_json(ir);
    let started = Instant::now();

    let headers = vec![("authorization".to_string(), format!("Bearer {key}"))];

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
    ir: &ChatRequest,
) -> Result<ChatResponse, ProxyError> {
    let Some(key) = state.provider.anthropic_api_key.clone() else {
        return Err(ProxyError {
            status: 500,
            message: "ANTHROPIC_API_KEY is not set".to_string(),
            upstream_body: None,
        });
    };

    let url = format!(
        "{}/v1/messages",
        state.provider.anthropic_base_url.trim_end_matches('/')
    );
    let body = ir_to_anthropic_request_json(ir);
    let started = Instant::now();

    let headers = vec![
        ("x-api-key".to_string(), key),
        (
            "anthropic-version".to_string(),
            state.provider.anthropic_version.clone(),
        ),
    ];

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
    ir: &ChatRequest,
) -> Result<ChatResponse, ProxyError> {
    let url = format!(
        "{}/api/chat",
        state.provider.ollama_base_url.trim_end_matches('/')
    );
    let body = ir_to_ollama_request_json(ir);
    let started = Instant::now();

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
            headers: Vec::new(),
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

async fn call_openai_stream(
    state: &AppState,
    request_id: u64,
    ir: &ChatRequest,
) -> Result<ProviderStream, ProxyError> {
    let Some(key) = state.provider.openai_api_key.clone() else {
        return Err(ProxyError {
            status: 500,
            message: "OPENAI_API_KEY is not set".to_string(),
            upstream_body: None,
        });
    };

    let url = format!(
        "{}/v1/chat/completions",
        state.provider.openai_base_url.trim_end_matches('/')
    );
    let body = with_stream_flag(ir_to_openai_request_json(ir), true);
    let started = Instant::now();
    let headers = vec![("authorization".to_string(), format!("Bearer {key}"))];

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
    ir: &ChatRequest,
) -> Result<ProviderStream, ProxyError> {
    let Some(key) = state.provider.anthropic_api_key.clone() else {
        return Err(ProxyError {
            status: 500,
            message: "ANTHROPIC_API_KEY is not set".to_string(),
            upstream_body: None,
        });
    };

    let url = format!(
        "{}/v1/messages",
        state.provider.anthropic_base_url.trim_end_matches('/')
    );
    let body = with_stream_flag(ir_to_anthropic_request_json(ir), true);
    let started = Instant::now();
    let headers = vec![
        ("x-api-key".to_string(), key),
        (
            "anthropic-version".to_string(),
            state.provider.anthropic_version.clone(),
        ),
    ];

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
    ir: &ChatRequest,
) -> Result<ProviderStream, ProxyError> {
    let url = format!(
        "{}/api/chat",
        state.provider.ollama_base_url.trim_end_matches('/')
    );
    let body = with_stream_flag(ir_to_ollama_request_json(ir), true);
    let started = Instant::now();

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
            headers: Vec::new(),
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
            message: Message::new(Role::Assistant, content).with_source(Source::OpenAI),
            finish_reason,
        }],
        usage,
    })
}

fn anthropic_response_json_to_ir(v: Value) -> Option<ChatResponse> {
    let id = v.get("id")?.as_str()?.to_string();
    let model = v.get("model")?.as_str()?.to_string();

    let content_blocks = v.get("content")?.as_array()?;
    let mut content = Vec::new();
    for b in content_blocks {
        let ty = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match ty {
            "text" => {
                let text = b.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if !text.is_empty() {
                    content.push(Content::text(text.to_string()));
                }
            }
            "tool_use" => {
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

pub fn ir_to_openai_downstream_response(ir_resp: &ChatResponse, model: &str) -> Value {
    let created = unix_seconds();
    let (content_text, tool_calls) = ir_assistant_to_openai_message(ir_resp);

    let mut msg = serde_json::Map::new();
    msg.insert("role".to_string(), Value::String("assistant".to_string()));
    msg.insert("content".to_string(), Value::String(content_text));
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

pub fn ir_to_openai_downstream_stream(ir_resp: &ChatResponse, model: &str) -> String {
    let created = unix_seconds();
    let (content_text, tool_calls) = ir_assistant_to_openai_message(ir_resp);

    let mut delta = serde_json::Map::new();
    delta.insert("role".to_string(), Value::String("assistant".to_string()));
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

fn ir_assistant_to_openai_message(ir_resp: &ChatResponse) -> (String, Option<Vec<Value>>) {
    let msg = ir_resp.choices.first().map(|c| &c.message);
    let Some(msg) = msg else {
        return (String::new(), None);
    };

    let content_text = join_text(&msg.content, "\n");
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
        if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
    )
}

pub fn ir_to_anthropic_downstream_response(ir_resp: &ChatResponse, model: &str) -> Value {
    let text = ir_resp
        .choices
        .first()
        .map(|c| join_text(&c.message.content, "\n"))
        .unwrap_or_default();

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
        Value::Array(vec![json!({"type": "text", "text": text})]),
    );
    obj.insert("model".to_string(), Value::String(model.to_string()));
    obj.insert(
        "stop_reason".to_string(),
        Value::String(
            ir_resp
                .choices
                .first()
                .and_then(|c| c.finish_reason.clone())
                .unwrap_or_else(|| "end_turn".to_string()),
        ),
    );
    obj.insert("stop_sequence".to_string(), Value::Null);
    obj.insert(
        "usage".to_string(),
        usage.unwrap_or_else(|| json!({"input_tokens": 0, "output_tokens": 0})),
    );
    Value::Object(obj)
}

pub fn ir_to_anthropic_downstream_stream(ir_resp: &ChatResponse, model: &str) -> String {
    let text = ir_resp
        .choices
        .first()
        .map(|c| join_text(&c.message.content, "\n"))
        .unwrap_or_default();
    let prompt_tokens = ir_resp.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0);
    let completion_tokens = ir_resp
        .usage
        .as_ref()
        .map(|u| u.completion_tokens)
        .unwrap_or(0);
    let stop_reason = ir_resp
        .choices
        .first()
        .and_then(|c| c.finish_reason.clone())
        .unwrap_or_else(|| "end_turn".to_string());

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

    if !text.is_empty() {
        out.push_str(&sse_event_frame(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
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
                "index": 0,
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
                "index": 0,
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
