//! Route handlers for the LLM API hub.
//!
//! Endpoints:
//! - OpenAI-compatible: `POST /v1/chat/completions`
//! - Anthropic-compatible: `POST /v1/messages`
//! - Ollama-compatible: `POST /api/chat`

use axum::{
    Json,
    body::Body,
    extract::{Query, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    logging::RequestId,
    proxy::{
        DownstreamByteStream, DownstreamProtocol, ProxyError, anthropic_downstream_to_ir,
        call_provider, call_provider_stream, choose_provider, ir_to_anthropic_downstream_response,
        ir_to_anthropic_downstream_stream, ir_to_ollama_downstream_response,
        ir_to_ollama_downstream_stream, ir_to_openai_downstream_response,
        ir_to_openai_downstream_stream, map_provider_stream_to_downstream, ollama_downstream_to_ir,
        openai_downstream_to_ir, should_proxy,
    },
    state::{AppState, Provider},
};

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

/// Health check endpoint
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// Root endpoint — gateway liveness check (used by Claude for Office, etc.)
pub async fn root() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "message": "yallm gateway ready"
        })),
    )
}

/// Query parameters for `GET /v1/models`
#[derive(Debug, Deserialize)]
pub struct ModelsQuery {
    pub interface: Option<String>,
}

/// List models for the given API interface type.
/// Defaults to `default_provider` when `interface` query param is absent.
/// Model list configured via `YALLM_OPENAI_MODELS` / `YALLM_ANTHROPIC_MODELS` env vars.
pub async fn models_list(
    State(state): State<AppState>,
    Query(query): Query<ModelsQuery>,
) -> impl IntoResponse {
    let interface = query
        .interface
        .as_deref()
        .unwrap_or_else(|| state.default_provider.as_str());
    let litellm_models = !state.model_routes.is_empty();

    match interface {
        "anthropic" => {
            let data: Vec<Value> = state
                .anthropic_models
                .iter()
                .map(|id| {
                    json!({
                        "type": "model",
                        "id": id,
                    })
                })
                .collect();
            Json(json!({ "data": data })).into_response()
        }
        _ => {
            let data: Vec<Value> = state
                .openai_models
                .iter()
                .map(|id| {
                    json!({
                        "id": id,
                        "object": "model",
                        "created": 0,
                        "owned_by": if litellm_models { "litellm" } else { "openai" },
                    })
                })
                .collect();
            Json(json!({ "object": "list", "data": data })).into_response()
        }
    }
}

pub async fn fallback() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": "Not found",
                "type": "not_found"
            }
        })),
    )
}

fn sse_response(payload: String) -> Response {
    let mut resp = Response::new(Body::from(payload));
    let headers = resp.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    resp
}

fn sse_stream_response(stream: DownstreamByteStream) -> Response {
    let mut resp = Response::new(Body::from_stream(stream));
    let headers = resp.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    resp
}

fn ndjson_response(payload: String) -> Response {
    let mut resp = Response::new(Body::from(payload));
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    resp
}

fn ndjson_stream_response(stream: DownstreamByteStream) -> Response {
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    resp
}

pub async fn chat_completions(
    State(state): State<AppState>,
    axum::extract::Extension(RequestId(request_id)): axum::extract::Extension<RequestId>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let model = match req.get("model").and_then(|v| v.as_str()) {
        Some(m) => m,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": "Missing 'model'", "type": "invalid_request"}})),
            )
                .into_response();
        }
    };

    let target = choose_provider(&state, model);
    let provider = target.provider;
    let upstream_model = target.upstream_model.clone();
    tracing::info!(
        event = "route",
        request_id,
        downstream = "openai",
        provider = provider.as_str(),
        downstream_model = %target.downstream_model,
        upstream_model = %upstream_model
    );

    let mut ir = match openai_downstream_to_ir(&req) {
        Some(ir) => ir,
        None => {
            tracing::info!(
                event = "convert.error",
                request_id,
                stage = "downstream_to_ir",
                downstream = "openai"
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": "Invalid OpenAI request body", "type": "invalid_request"}})),
            )
                .into_response();
        }
    };
    ir.model = upstream_model.clone();

    tracing::info!(
        event = "convert.downstream_to_ir",
        request_id,
        downstream = "openai",
        message_count = ir.messages.len(),
        model = %ir.model
    );

    match should_proxy(&state, &target) {
        Ok(true) => {
            if stream {
                match call_provider_stream(&state, request_id, &target, &ir).await {
                    Ok(stream_resp) => {
                        tracing::info!(
                            event = "convert.provider_to_ir_stream",
                            request_id,
                            provider = provider.as_str()
                        );
                        let out = map_provider_stream_to_downstream(
                            stream_resp.provider,
                            DownstreamProtocol::OpenAI,
                            stream_resp.body,
                            upstream_model.clone(),
                        );
                        tracing::info!(
                            event = "convert.ir_to_downstream_stream",
                            request_id,
                            downstream = "openai"
                        );
                        sse_stream_response(out)
                    }
                    Err(e) => proxy_error_to_response(provider, e).into_response(),
                }
            } else {
                match call_provider(&state, request_id, &target, &ir).await {
                    Ok(ir_resp) => {
                        tracing::info!(
                            event = "convert.provider_to_ir",
                            request_id,
                            provider = provider.as_str()
                        );
                        let out = ir_to_openai_downstream_response(&ir_resp, &upstream_model);
                        tracing::info!(
                            event = "convert.ir_to_downstream",
                            request_id,
                            downstream = "openai"
                        );
                        (StatusCode::OK, Json(out)).into_response()
                    }
                    Err(e) => proxy_error_to_response(provider, e).into_response(),
                }
            }
        }
        Ok(false) => {
            // Mock fallback: deterministic response based on last user message.
            let reply = mock_reply(extract_last_user_text(&ir));
            let mock = mock_ir_response(&ir.model, reply);
            tracing::info!(event = "convert.mock", request_id, downstream = "openai");
            if stream {
                let out = ir_to_openai_downstream_stream(&mock, &upstream_model);
                sse_response(out)
            } else {
                let out = ir_to_openai_downstream_response(&mock, &upstream_model);
                (StatusCode::OK, Json(out)).into_response()
            }
        }
        Err(e) => proxy_error_to_response(provider, e).into_response(),
    }
}

pub async fn anthropic_messages(
    State(state): State<AppState>,
    axum::extract::Extension(RequestId(request_id)): axum::extract::Extension<RequestId>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let model = match req.get("model").and_then(|v| v.as_str()) {
        Some(m) => m,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"type":"error","error":{"type":"invalid_request","message":"Missing 'model'"}})),
            )
                .into_response();
        }
    };

    let target = choose_provider(&state, model);
    let provider = target.provider;
    let upstream_model = target.upstream_model.clone();
    tracing::info!(
        event = "route",
        request_id,
        downstream = "anthropic",
        provider = provider.as_str(),
        downstream_model = %target.downstream_model,
        upstream_model = %upstream_model
    );

    let mut ir = match anthropic_downstream_to_ir(&req) {
        Some(ir) => ir,
        None => {
            tracing::info!(
                event = "convert.error",
                request_id,
                stage = "downstream_to_ir",
                downstream = "anthropic"
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"type":"error","error":{"type":"invalid_request","message":"Invalid Anthropic request body"}})),
            )
                .into_response();
        }
    };
    ir.model = upstream_model.clone();

    tracing::info!(
        event = "convert.downstream_to_ir",
        request_id,
        downstream = "anthropic",
        message_count = ir.messages.len(),
        model = %ir.model
    );

    match should_proxy(&state, &target) {
        Ok(true) => {
            if stream {
                match call_provider_stream(&state, request_id, &target, &ir).await {
                    Ok(stream_resp) => {
                        tracing::info!(
                            event = "convert.provider_to_ir_stream",
                            request_id,
                            provider = provider.as_str()
                        );
                        let out = map_provider_stream_to_downstream(
                            stream_resp.provider,
                            DownstreamProtocol::Anthropic,
                            stream_resp.body,
                            upstream_model.clone(),
                        );
                        tracing::info!(
                            event = "convert.ir_to_downstream_stream",
                            request_id,
                            downstream = "anthropic"
                        );
                        sse_stream_response(out)
                    }
                    Err(e) => proxy_error_to_response(provider, e).into_response(),
                }
            } else {
                match call_provider(&state, request_id, &target, &ir).await {
                    Ok(ir_resp) => {
                        tracing::info!(
                            event = "convert.provider_to_ir",
                            request_id,
                            provider = provider.as_str()
                        );
                        let out = ir_to_anthropic_downstream_response(&ir_resp, &upstream_model);
                        tracing::info!(
                            event = "convert.ir_to_downstream",
                            request_id,
                            downstream = "anthropic"
                        );
                        (StatusCode::OK, Json(out)).into_response()
                    }
                    Err(e) => proxy_error_to_response(provider, e).into_response(),
                }
            }
        }
        Ok(false) => {
            let reply = mock_reply(extract_last_user_text(&ir));
            let mock = mock_ir_response(&ir.model, reply);
            tracing::info!(event = "convert.mock", request_id, downstream = "anthropic");
            if stream {
                let out = ir_to_anthropic_downstream_stream(&mock, &upstream_model);
                sse_response(out)
            } else {
                let out = ir_to_anthropic_downstream_response(&mock, &upstream_model);
                (StatusCode::OK, Json(out)).into_response()
            }
        }
        Err(e) => proxy_error_to_response(provider, e).into_response(),
    }
}

pub async fn ollama_chat(
    State(state): State<AppState>,
    axum::extract::Extension(RequestId(request_id)): axum::extract::Extension<RequestId>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let model = match req.get("model").and_then(|v| v.as_str()) {
        Some(m) => m,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Missing 'model'"})),
            )
                .into_response();
        }
    };

    let target = choose_provider(&state, model);
    let provider = target.provider;
    let upstream_model = target.upstream_model.clone();
    tracing::info!(
        event = "route",
        request_id,
        downstream = "ollama",
        provider = provider.as_str(),
        downstream_model = %target.downstream_model,
        upstream_model = %upstream_model
    );

    let mut ir = match ollama_downstream_to_ir(&req) {
        Some(ir) => ir,
        None => {
            tracing::info!(
                event = "convert.error",
                request_id,
                stage = "downstream_to_ir",
                downstream = "ollama"
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "Invalid Ollama request body"})),
            )
                .into_response();
        }
    };
    ir.model = upstream_model.clone();

    tracing::info!(
        event = "convert.downstream_to_ir",
        request_id,
        downstream = "ollama",
        message_count = ir.messages.len(),
        model = %ir.model
    );

    match should_proxy(&state, &target) {
        Ok(true) => {
            if stream {
                match call_provider_stream(&state, request_id, &target, &ir).await {
                    Ok(stream_resp) => {
                        tracing::info!(
                            event = "convert.provider_to_ir_stream",
                            request_id,
                            provider = provider.as_str()
                        );
                        let out = map_provider_stream_to_downstream(
                            stream_resp.provider,
                            DownstreamProtocol::Ollama,
                            stream_resp.body,
                            upstream_model.clone(),
                        );
                        tracing::info!(
                            event = "convert.ir_to_downstream_stream",
                            request_id,
                            downstream = "ollama"
                        );
                        ndjson_stream_response(out)
                    }
                    Err(e) => proxy_error_to_response(provider, e).into_response(),
                }
            } else {
                match call_provider(&state, request_id, &target, &ir).await {
                    Ok(ir_resp) => {
                        tracing::info!(
                            event = "convert.provider_to_ir",
                            request_id,
                            provider = provider.as_str()
                        );
                        let out = ir_to_ollama_downstream_response(&ir_resp, &upstream_model);
                        tracing::info!(
                            event = "convert.ir_to_downstream",
                            request_id,
                            downstream = "ollama"
                        );
                        (StatusCode::OK, Json(out)).into_response()
                    }
                    Err(e) => proxy_error_to_response(provider, e).into_response(),
                }
            }
        }
        Ok(false) => {
            let reply = mock_reply(extract_last_user_text(&ir));
            let mock = mock_ir_response(&ir.model, reply);
            tracing::info!(event = "convert.mock", request_id, downstream = "ollama");
            if stream {
                let out = ir_to_ollama_downstream_stream(&mock, &upstream_model);
                ndjson_response(out)
            } else {
                let out = ir_to_ollama_downstream_response(&mock, &upstream_model);
                (StatusCode::OK, Json(out)).into_response()
            }
        }
        Err(e) => proxy_error_to_response(provider, e).into_response(),
    }
}

fn proxy_error_to_response(provider: Provider, err: ProxyError) -> (StatusCode, Json<Value>) {
    tracing::info!(
        event = "proxy.error",
        provider = provider.as_str(),
        status = err.status,
        message = %err.message,
        upstream_body = %serde_json::to_string(&err.upstream_body).unwrap_or_default()
    );

    let status = StatusCode::from_u16(err.status).unwrap_or(StatusCode::BAD_GATEWAY);
    (
        status,
        Json(json!({
            "error": {
                "message": err.message,
                "type": "upstream_error",
                "provider": provider.as_str(),
                "upstream": err.upstream_body
            }
        })),
    )
}

fn extract_last_user_text(ir: &yallm_ir::ChatRequest) -> Option<String> {
    ir.messages
        .iter()
        .rev()
        .find(|m| m.role == yallm_ir::Role::User)
        .map(|m| {
            m.content
                .iter()
                .find_map(|c| match c {
                    yallm_ir::Content::Text(t) => Some(t.text.clone()),
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

fn mock_ir_response(model: &str, reply: String) -> yallm_ir::ChatResponse {
    yallm_ir::ChatResponse {
        id: format!("mock_{}", unix_seconds()),
        model: model.to_string(),
        choices: vec![yallm_ir::Choice {
            index: 0,
            message: yallm_ir::Message::text(yallm_ir::Role::Assistant, reply),
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(yallm_ir::Usage::default()),
    }
}

fn unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
