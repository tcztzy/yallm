use axum::{
    Json,
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::{Value, json};
use yallm_storage::{ListOrder, SaveResponseRequest, StoreError};

use crate::{
    logging::RequestId,
    proxy::{ProxyError, call_provider, choose_provider, should_proxy},
    state::{AppState, Provider, TransportRequest},
};

#[derive(Debug, Deserialize, Default)]
pub struct ListQuery {
    pub limit: Option<usize>,
    pub order: Option<String>,
    pub after: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct GetResponseQuery {
    pub stream: Option<bool>,
}

pub async fn responses_create(
    State(state): State<AppState>,
    axum::extract::Extension(RequestId(request_id)): axum::extract::Extension<RequestId>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let conversation_id = yallm_responses::extract_conversation_id(&req);
    let previous_response_id = yallm_responses::extract_previous_response_id(&req);

    if conversation_id.is_some() && previous_response_id.is_some() {
        return bad_request("conversation and previous_response_id cannot be used together");
    }

    let model = match req.get("model").and_then(Value::as_str) {
        Some(model) => model,
        None => return bad_request("Missing 'model'"),
    };

    let context = match state
        .store
        .resolve_context(conversation_id.as_deref(), previous_response_id.as_deref())
        .await
    {
        Ok(ctx) => ctx,
        Err(err) => return store_error_to_response(err),
    };

    let (provider, _downstream_model, upstream_model) = choose_provider(&state, model);
    let should_openai_passthrough = provider == Provider::OpenAI
        && conversation_id.is_none()
        && previous_response_id.is_none()
        && context.items.is_empty();

    if should_openai_passthrough && matches!(should_proxy(&state, provider), Ok(true)) {
        let upstream = match call_openai_post_json(&state, request_id, "/v1/responses", &req).await
        {
            Ok(v) => v,
            Err(resp) => return resp,
        };

        let stream_events = if stream {
            yallm_responses::response_to_stream_events(&upstream)
        } else {
            Vec::new()
        };
        let input_items = yallm_responses::parse_input_items(&req);
        let saved = match state
            .store
            .save_response(SaveResponseRequest {
                response: upstream,
                request_input_items: input_items,
                conversation_id: None,
                previous_response_id: None,
                provider: "openai".to_string(),
                stream_events: stream_events.clone(),
            })
            .await
        {
            Ok(saved) => saved,
            Err(err) => return store_error_to_response(err),
        };

        if stream {
            return sse_response(stream_events.concat());
        }
        return (StatusCode::OK, Json(saved.response)).into_response();
    }

    let mut ir = match yallm_responses::create_response_to_ir(&req, &context.items) {
        Some(ir) => ir,
        None => return bad_request("Invalid Responses request body"),
    };
    ir.model = upstream_model.clone();

    let ir_resp = match should_proxy(&state, provider) {
        Ok(true) => match call_provider(&state, request_id, provider, &ir).await {
            Ok(resp) => resp,
            Err(err) => return proxy_error_to_response(provider, err),
        },
        Ok(false) => {
            let reply = mock_reply(extract_last_user_text(&ir));
            mock_ir_response(&ir.model, reply)
        }
        Err(err) => return proxy_error_to_response(provider, err),
    };

    let created_at = unix_seconds();
    let response_id = format!("resp_{request_id}_{created_at}");
    let response = yallm_responses::ir_to_response(
        &ir_resp,
        &req,
        &response_id,
        created_at,
        Some(created_at),
        context.conversation_id.as_deref(),
        previous_response_id.as_deref(),
    );
    let stream_events = if stream {
        yallm_responses::response_to_stream_events(&response)
    } else {
        Vec::new()
    };

    let input_items = yallm_responses::parse_input_items(&req);
    let saved = match state
        .store
        .save_response(SaveResponseRequest {
            response,
            request_input_items: input_items,
            conversation_id: context.conversation_id,
            previous_response_id,
            provider: provider.as_str().to_string(),
            stream_events: stream_events.clone(),
        })
        .await
    {
        Ok(saved) => saved,
        Err(err) => return store_error_to_response(err),
    };

    if stream {
        return sse_response(stream_events.concat());
    }

    (StatusCode::OK, Json(saved.response)).into_response()
}

pub async fn responses_get(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
    Query(query): Query<GetResponseQuery>,
) -> impl IntoResponse {
    let response = match state.store.get_response(&response_id).await {
        Ok(Some(response)) => response,
        Ok(None) => return not_found(format!("Response '{response_id}' not found")),
        Err(err) => return store_error_to_response(err),
    };

    if query.stream.unwrap_or(false) {
        let events = match state.store.get_response_stream_events(&response_id).await {
            Ok(Some(events)) => events,
            Ok(None) => yallm_responses::response_to_stream_events(&response),
            Err(err) => return store_error_to_response(err),
        };
        return sse_response(events.concat());
    }

    (StatusCode::OK, Json(response)).into_response()
}

pub async fn responses_delete(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    match state.store.delete_response(&response_id).await {
        Ok(Some(resp)) => (StatusCode::OK, Json(resp)).into_response(),
        Ok(None) => not_found(format!("Response '{response_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn responses_cancel(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    match state.store.cancel_response(&response_id).await {
        Ok(Some(resp)) => (StatusCode::OK, Json(resp)).into_response(),
        Ok(None) => not_found(format!("Response '{response_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn responses_input_items(
    State(state): State<AppState>,
    Path(response_id): Path<String>,
    Query(query): Query<ListQuery>,
) -> impl IntoResponse {
    let limit = normalize_limit(query.limit);
    let order = normalize_order(query.order.as_deref());
    let after = query.after.as_deref();

    match state
        .store
        .list_response_input_items(&response_id, limit, order, after)
        .await
    {
        Ok(Some(page)) => (StatusCode::OK, Json(list_page_json(page))).into_response(),
        Ok(None) => not_found(format!("Response '{response_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn responses_input_tokens(
    State(state): State<AppState>,
    axum::extract::Extension(RequestId(request_id)): axum::extract::Extension<RequestId>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    if let Some(model) = req.get("model").and_then(Value::as_str) {
        let (provider, _, _) = choose_provider(&state, model);
        let convo = yallm_responses::extract_conversation_id(&req);
        let prev = yallm_responses::extract_previous_response_id(&req);
        if provider == Provider::OpenAI
            && convo.is_none()
            && prev.is_none()
            && matches!(should_proxy(&state, provider), Ok(true))
        {
            match call_openai_post_json(&state, request_id, "/v1/responses/input_tokens", &req)
                .await
            {
                Ok(v) => return (StatusCode::OK, Json(v)).into_response(),
                Err(resp) => return resp,
            }
        }
    }

    (
        StatusCode::OK,
        Json(yallm_responses::estimate_input_tokens(&req)),
    )
        .into_response()
}

pub async fn responses_compact(
    State(state): State<AppState>,
    axum::extract::Extension(RequestId(request_id)): axum::extract::Extension<RequestId>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    if let Some(model) = req.get("model").and_then(Value::as_str) {
        let (provider, _, _) = choose_provider(&state, model);
        let convo = yallm_responses::extract_conversation_id(&req);
        let prev = yallm_responses::extract_previous_response_id(&req);
        if provider == Provider::OpenAI
            && convo.is_none()
            && prev.is_none()
            && matches!(should_proxy(&state, provider), Ok(true))
        {
            match call_openai_post_json(&state, request_id, "/v1/responses/compact", &req).await {
                Ok(v) => return (StatusCode::OK, Json(v)).into_response(),
                Err(resp) => return resp,
            }
        }
    }

    (
        StatusCode::OK,
        Json(yallm_responses::fallback_compact(&req)),
    )
        .into_response()
}

pub async fn conversations_create(
    State(state): State<AppState>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let metadata = req.get("metadata").cloned();
    let items = req
        .get("items")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    let normalized_items = yallm_responses::parse_input_items(&json!({ "input": items }));

    match state
        .store
        .create_conversation(metadata, normalized_items)
        .await
    {
        Ok(conv) => (StatusCode::OK, Json(conv)).into_response(),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn conversations_get(
    State(state): State<AppState>,
    Path(conversation_id): Path<String>,
) -> impl IntoResponse {
    match state.store.get_conversation(&conversation_id).await {
        Ok(Some(conv)) => (StatusCode::OK, Json(conv)).into_response(),
        Ok(None) => not_found(format!("Conversation '{conversation_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn conversations_update(
    State(state): State<AppState>,
    Path(conversation_id): Path<String>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let metadata = req.get("metadata").cloned().unwrap_or_else(|| json!({}));
    match state
        .store
        .update_conversation(&conversation_id, metadata)
        .await
    {
        Ok(Some(conv)) => (StatusCode::OK, Json(conv)).into_response(),
        Ok(None) => not_found(format!("Conversation '{conversation_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn conversations_delete(
    State(state): State<AppState>,
    Path(conversation_id): Path<String>,
) -> impl IntoResponse {
    match state.store.delete_conversation(&conversation_id).await {
        Ok(Some(resp)) => (StatusCode::OK, Json(resp)).into_response(),
        Ok(None) => not_found(format!("Conversation '{conversation_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn conversation_items_create(
    State(state): State<AppState>,
    Path(conversation_id): Path<String>,
    Json(req): Json<Value>,
) -> impl IntoResponse {
    let items = req
        .get("items")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    let normalized_items = yallm_responses::parse_input_items(&json!({ "input": items }));
    match state
        .store
        .add_conversation_items(&conversation_id, normalized_items)
        .await
    {
        Ok(Some(page)) => (StatusCode::OK, Json(list_page_json(page))).into_response(),
        Ok(None) => not_found(format!("Conversation '{conversation_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn conversation_items_list(
    State(state): State<AppState>,
    Path(conversation_id): Path<String>,
    Query(query): Query<ListQuery>,
) -> impl IntoResponse {
    let limit = normalize_limit(query.limit);
    let order = normalize_order(query.order.as_deref());
    let after = query.after.as_deref();
    match state
        .store
        .list_conversation_items(&conversation_id, limit, order, after)
        .await
    {
        Ok(Some(page)) => (StatusCode::OK, Json(list_page_json(page))).into_response(),
        Ok(None) => not_found(format!("Conversation '{conversation_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn conversation_item_get(
    State(state): State<AppState>,
    Path((conversation_id, item_id)): Path<(String, String)>,
) -> impl IntoResponse {
    match state
        .store
        .get_conversation_item(&conversation_id, &item_id)
        .await
    {
        Ok(Some(item)) => (StatusCode::OK, Json(item)).into_response(),
        Ok(None) => not_found(format!("Conversation item '{item_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
}

pub async fn conversation_item_delete(
    State(state): State<AppState>,
    Path((conversation_id, item_id)): Path<(String, String)>,
) -> impl IntoResponse {
    match state
        .store
        .delete_conversation_item(&conversation_id, &item_id)
        .await
    {
        Ok(Some(conv)) => (StatusCode::OK, Json(conv)).into_response(),
        Ok(None) => not_found(format!("Conversation item '{item_id}' not found")),
        Err(err) => store_error_to_response(err),
    }
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

async fn call_openai_post_json(
    state: &AppState,
    request_id: u64,
    path: &str,
    body: &Value,
) -> Result<Value, Response> {
    let Some(key) = state.provider.openai_api_key.clone() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "OPENAI_API_KEY is not set",
                    "type": "server_error"
                }
            })),
        )
            .into_response());
    };

    let url = format!(
        "{}{path}",
        state.provider.openai_base_url.trim_end_matches('/')
    );
    tracing::info!(
        event = "provider.out",
        request_id,
        provider = "openai",
        method = "POST",
        url = %url
    );

    let mut upstream_body = body.clone();
    if path == "/v1/responses"
        && body.get("stream").and_then(Value::as_bool) == Some(true)
        && let Some(obj) = upstream_body.as_object_mut()
    {
        obj.insert("stream".to_string(), Value::Bool(false));
    }

    let resp = state
        .transport
        .send(TransportRequest {
            method: "POST",
            url: url.clone(),
            headers: vec![("authorization".to_string(), format!("Bearer {key}"))],
            body: upstream_body,
        })
        .await
        .map_err(|err| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({
                    "error": {
                        "message": format!("Failed to call OpenAI upstream: {}", err.message),
                        "type": "upstream_error"
                    }
                })),
            )
                .into_response()
        })?;

    let status = StatusCode::from_u16(resp.status).unwrap_or(StatusCode::BAD_GATEWAY);
    let parsed: Value = serde_json::from_slice(&resp.body).unwrap_or_else(|_| json!({}));
    tracing::info!(
        event = "provider.in",
        request_id,
        provider = "openai",
        status = resp.status
    );

    if !status.is_success() {
        return Err((
            status,
            Json(json!({
                "error": {
                    "message": "OpenAI upstream returned error",
                    "type": "upstream_error",
                    "provider": "openai",
                    "upstream": parsed
                }
            })),
        )
            .into_response());
    }

    Ok(parsed)
}

fn list_page_json(page: yallm_storage::ListPage<Value>) -> Value {
    json!({
        "object": "list",
        "data": page.data,
        "first_id": page.first_id,
        "last_id": page.last_id,
        "has_more": page.has_more
    })
}

fn normalize_limit(limit: Option<usize>) -> usize {
    limit.unwrap_or(20).clamp(1, 100)
}

fn normalize_order(order: Option<&str>) -> ListOrder {
    match order.unwrap_or("desc") {
        "asc" => ListOrder::Asc,
        _ => ListOrder::Desc,
    }
}

fn proxy_error_to_response(provider: Provider, err: ProxyError) -> Response {
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
        .into_response()
}

fn store_error_to_response(err: StoreError) -> Response {
    match err {
        StoreError::NotFound(msg) => not_found(msg),
        StoreError::Conflict(msg) | StoreError::Invalid(msg) => bad_request(&msg),
        StoreError::Io(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": format!("Internal storage error: {err}"),
                    "type": "server_error"
                }
            })),
        )
            .into_response(),
    }
}

fn bad_request(message: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request"
            }
        })),
    )
        .into_response()
}

fn not_found(message: String) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": message,
                "type": "not_found"
            }
        })),
    )
        .into_response()
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
