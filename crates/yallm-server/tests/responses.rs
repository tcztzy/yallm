use std::{
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Debug, Default)]
struct Capture {
    requests: tokio::sync::Mutex<Vec<yallm_server::TransportRequest>>,
}

#[derive(Clone)]
struct MockTransport {
    capture: Arc<Capture>,
}

impl yallm_server::Transport for MockTransport {
    fn send<'a>(
        &'a self,
        req: yallm_server::TransportRequest,
    ) -> yallm_server::TransportFuture<'a> {
        Box::pin(async move {
            self.capture.requests.lock().await.push(req.clone());

            if req.url.ends_with("/v1/responses/input_tokens") {
                return Ok(yallm_server::TransportResponse {
                    status: 200,
                    headers: vec![],
                    body: serde_json::to_vec(&json!({
                        "object": "response.input_tokens",
                        "input_tokens": 42
                    }))
                    .unwrap(),
                });
            }

            if req.url.ends_with("/v1/responses/compact") {
                return Ok(yallm_server::TransportResponse {
                    status: 200,
                    headers: vec![],
                    body: serde_json::to_vec(&json!({
                        "id": "cmp_upstream",
                        "object": "response.compaction",
                        "created_at": 1,
                        "output": [],
                        "usage": {"input_tokens":1,"output_tokens":1,"total_tokens":2}
                    }))
                    .unwrap(),
                });
            }

            if req.url.ends_with("/v1/responses") {
                let model = req
                    .body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let body = json!({
                    "id": "resp_upstream",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "completed_at": 1,
                    "error": null,
                    "incomplete_details": null,
                    "instructions": null,
                    "max_output_tokens": null,
                    "model": model,
                    "output": [{
                        "id": "msg_upstream",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "upstream response text",
                            "annotations": []
                        }]
                    }],
                    "parallel_tool_calls": true,
                    "previous_response_id": null,
                    "reasoning": {"effort": null, "summary": null},
                    "store": true,
                    "temperature": 1.0,
                    "text": {"format": {"type":"text"}},
                    "tool_choice": "auto",
                    "tools": [],
                    "top_p": 1.0,
                    "truncation": "disabled",
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 15
                    },
                    "user": null,
                    "metadata": {}
                });
                return Ok(yallm_server::TransportResponse {
                    status: 200,
                    headers: vec![],
                    body: serde_json::to_vec(&body).unwrap(),
                });
            }

            if req.url.ends_with("/v1/chat/completions") {
                let model = req
                    .body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let last_user = req
                    .body
                    .get("messages")
                    .and_then(Value::as_array)
                    .and_then(|messages| {
                        messages
                            .iter()
                            .rev()
                            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                    })
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_str)
                    .unwrap_or("hello");
                let body = json!({
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": format!("hello from chat: {last_user}")},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
                });
                return Ok(yallm_server::TransportResponse {
                    status: 200,
                    headers: vec![],
                    body: serde_json::to_vec(&body).unwrap(),
                });
            }

            if req.url.ends_with("/v1/messages") {
                let model = req
                    .body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let last_user = req
                    .body
                    .get("messages")
                    .and_then(Value::as_array)
                    .and_then(|messages| {
                        messages
                            .iter()
                            .rev()
                            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                    })
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_str)
                    .unwrap_or("hello");
                let body = json!({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type":"text","text": format!("hello from anthropic: {last_user}")}],
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": {"input_tokens": 3, "output_tokens": 4}
                });
                return Ok(yallm_server::TransportResponse {
                    status: 200,
                    headers: vec![],
                    body: serde_json::to_vec(&body).unwrap(),
                });
            }

            if req.url.ends_with("/api/chat") {
                let model = req
                    .body
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let body = json!({
                    "model": model,
                    "message": {"role": "assistant", "content": "hello from ollama"},
                    "done": true,
                    "prompt_eval_count": 1,
                    "eval_count": 2
                });
                return Ok(yallm_server::TransportResponse {
                    status: 200,
                    headers: vec![],
                    body: serde_json::to_vec(&body).unwrap(),
                });
            }

            Err(yallm_server::TransportError {
                message: "unknown url".to_string(),
            })
        })
    }

    fn send_stream<'a>(
        &'a self,
        _req: yallm_server::TransportRequest,
    ) -> yallm_server::TransportStreamFuture<'a> {
        Box::pin(async {
            Err(yallm_server::TransportError {
                message: "stream not implemented in test transport".to_string(),
            })
        })
    }
}

fn temp_store_path(name: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("yallm-responses-test-{name}-{ts}.json"))
}

fn temp_db_url(name: &str) -> String {
    format!(
        "sqlite://{}",
        temp_store_path(name).with_extension("sqlite3").display()
    )
}

fn app(name: &str, capture: Arc<Capture>) -> Router {
    let mut state = yallm_server::AppState {
        transport: Arc::new(MockTransport { capture }),
        mode: yallm_server::Mode::Proxy,
        ..state_with_temp_store(name)
    };
    state.provider.openai_api_key = Some("test_openai".to_string());
    state.provider.anthropic_api_key = Some("test_anthropic".to_string());
    state.provider.openai_base_url = "http://openai.test".to_string();
    state.provider.anthropic_base_url = "http://anthropic.test".to_string();
    state.provider.ollama_base_url = "http://ollama.test".to_string();
    yallm_server::app_with_state(state)
}

fn state_with_temp_store(name: &str) -> yallm_server::AppState {
    let mut env = std::collections::HashMap::new();
    env.insert(yallm_storage::DB_URL_ENV.to_string(), temp_db_url(name));
    yallm_server::AppState::from_loaded_config(yallm_config::LoadedConfig {
        env,
        litellm_models: Vec::new(),
        warnings: Vec::new(),
    })
}

async fn post_json(app: &Router, uri: &str, payload: Value) -> (StatusCode, Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let value: Value = serde_json::from_slice(&body).unwrap();
    (status, value)
}

#[tokio::test]
async fn create_response_persists_conversation_and_input_items() {
    let capture = Arc::new(Capture::default());
    let app = app("persist", capture);

    let (status, response) = post_json(
        &app,
        "/v1/responses",
        json!({"model":"openai:gpt-4o-mini","input":"hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let response_id = response["id"].as_str().unwrap().to_string();
    let conversation_id = response["conversation"]["id"].as_str().unwrap().to_string();

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/responses/{response_id}/input_items"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let items: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(items["object"], "list");
    assert!(!items["data"].as_array().unwrap().is_empty());

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/conversations/{conversation_id}/items"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn followup_with_conversation_continues_and_switches_provider() {
    let capture = Arc::new(Capture::default());
    let app = app("switch-provider", capture.clone());

    let (_, first) = post_json(
        &app,
        "/v1/responses",
        json!({"model":"openai:gpt-4o-mini","input":"hello"}),
    )
    .await;
    let conversation_id = first["conversation"]["id"].as_str().unwrap().to_string();

    let (status, second) = post_json(
        &app,
        "/v1/responses",
        json!({
            "model":"anthropic:claude-3-haiku-20240307",
            "conversation":{"id": conversation_id},
            "input":"follow up"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = second["output"][0]["content"][0]["text"]
        .as_str()
        .unwrap_or("");
    assert!(text.contains("anthropic"));

    let requests = capture.requests.lock().await;
    assert!(requests.iter().any(|r| r.url.ends_with("/v1/messages")));
}

#[tokio::test]
async fn followup_with_previous_response_id_continues_history() {
    let capture = Arc::new(Capture::default());
    let app = app("prev-id", capture);

    let (_, first) = post_json(
        &app,
        "/v1/responses",
        json!({"model":"openai:gpt-4o-mini","input":"hello"}),
    )
    .await;
    let first_id = first["id"].as_str().unwrap().to_string();

    let (status, second) = post_json(
        &app,
        "/v1/responses",
        json!({
            "model":"anthropic:claude-3-haiku-20240307",
            "previous_response_id": first_id,
            "input":"continue"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(second["previous_response_id"].is_string());
    assert!(second["conversation"]["id"].is_string());
}

#[tokio::test]
async fn multi_turn_responses_use_chat_completions_upstream() {
    let capture = Arc::new(Capture::default());
    let app = app("chat-completions-upstream", capture.clone());

    let (status, conv) = post_json(
        &app,
        "/v1/conversations",
        json!({"metadata":{"topic":"chat-completions-multiturn"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let conv_id = conv["id"].as_str().unwrap().to_string();

    let (status, first) = post_json(
        &app,
        "/v1/responses",
        json!({
            "model":"openai:gpt-4o-mini",
            "conversation":{"id": conv_id},
            "input":"turn-1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        first["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap_or(""),
        "hello from chat: turn-1"
    );

    let (status, second) = post_json(
        &app,
        "/v1/responses",
        json!({
            "model":"openai:gpt-4o-mini",
            "conversation":{"id": conv_id},
            "input":"turn-2"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        second["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap_or(""),
        "hello from chat: turn-2"
    );

    let requests = capture.requests.lock().await;
    assert!(!requests.iter().any(|r| r.url.ends_with("/v1/responses")));

    let chat_requests: Vec<_> = requests
        .iter()
        .filter(|r| r.url.ends_with("/v1/chat/completions"))
        .collect();
    assert_eq!(chat_requests.len(), 2);

    let first_messages = chat_requests[0]
        .body
        .get("messages")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    let second_messages = chat_requests[1]
        .body
        .get("messages")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(second_messages > first_messages);
}

#[tokio::test]
async fn multi_turn_responses_use_anthropic_upstream() {
    let capture = Arc::new(Capture::default());
    let app = app("anthropic-upstream", capture.clone());

    let (status, conv) = post_json(
        &app,
        "/v1/conversations",
        json!({"metadata":{"topic":"anthropic-multiturn"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let conv_id = conv["id"].as_str().unwrap().to_string();

    let (status, first) = post_json(
        &app,
        "/v1/responses",
        json!({
            "model":"anthropic:claude-3-haiku-20240307",
            "conversation":{"id": conv_id},
            "input":"turn-a1"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        first["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap_or(""),
        "hello from anthropic: turn-a1"
    );

    let (status, second) = post_json(
        &app,
        "/v1/responses",
        json!({
            "model":"anthropic:claude-3-haiku-20240307",
            "conversation":{"id": conv_id},
            "input":"turn-a2"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        second["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap_or(""),
        "hello from anthropic: turn-a2"
    );

    let requests = capture.requests.lock().await;
    assert!(!requests.iter().any(|r| r.url.ends_with("/v1/responses")));

    let anthropic_requests: Vec<_> = requests
        .iter()
        .filter(|r| r.url.ends_with("/v1/messages"))
        .collect();
    assert_eq!(anthropic_requests.len(), 2);

    let first_messages = anthropic_requests[0]
        .body
        .get("messages")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    let second_messages = anthropic_requests[1]
        .body
        .get("messages")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(second_messages > first_messages);
}

#[tokio::test]
async fn conversation_and_previous_response_id_together_returns_400() {
    let capture = Arc::new(Capture::default());
    let app = app("conflict", capture);

    let (status, _) = post_json(
        &app,
        "/v1/responses",
        json!({
            "model":"openai:gpt-4o-mini",
            "conversation":{"id":"conv_123"},
            "previous_response_id":"resp_123",
            "input":"nope"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn passthrough_only_for_brand_new_then_replay_uses_chat_path() {
    let capture = Arc::new(Capture::default());
    let app = app("passthrough", capture.clone());

    let (_, first) = post_json(
        &app,
        "/v1/responses",
        json!({"model":"openai:gpt-4o-mini","input":"hello"}),
    )
    .await;
    let conversation_id = first["conversation"]["id"].as_str().unwrap();

    let (status, _) = post_json(
        &app,
        "/v1/responses",
        json!({
            "model":"openai:gpt-4o-mini",
            "conversation":{"id":conversation_id},
            "input":"second turn"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let requests = capture.requests.lock().await;
    assert!(requests.iter().any(|r| r.url.ends_with("/v1/responses")));
    assert!(
        requests
            .iter()
            .any(|r| r.url.ends_with("/v1/chat/completions"))
    );
}

#[tokio::test]
async fn stream_response_returns_sse_and_is_retrievable() {
    let capture = Arc::new(Capture::default());
    let app = app("stream", capture);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model":"anthropic:claude-3-haiku-20240307",
                        "stream": true,
                        "input":"stream this"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(content_type.starts_with("text/event-stream"));
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("response.created"));
    assert!(text.contains("data: [DONE]"));
}

#[tokio::test]
async fn response_cancel_delete_and_get_lifecycle() {
    let capture = Arc::new(Capture::default());
    let app = app("lifecycle", capture);

    let (_, created) = post_json(
        &app,
        "/v1/responses",
        json!({"model":"openai:gpt-4o-mini","input":"hello"}),
    )
    .await;
    let response_id = created["id"].as_str().unwrap();

    let (status, cancelled) = post_json(
        &app,
        &format!("/v1/responses/{response_id}/cancel"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cancelled["status"], "cancelled");

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/responses/{response_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/v1/responses/{response_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn conversations_crud_and_item_pagination_work() {
    let capture = Arc::new(Capture::default());
    let app = app("conversation-crud", capture);

    let (status, conv) = post_json(
        &app,
        "/v1/conversations",
        json!({"metadata":{"topic":"demo"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let conv_id = conv["id"].as_str().unwrap().to_string();

    let (status, _) = post_json(
        &app,
        &format!("/v1/conversations/{conv_id}/items"),
        json!({"items":[
            {"type":"message","role":"user","content":[{"type":"input_text","text":"a"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"b"}]},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"c"}]}
        ]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!(
                    "/v1/conversations/{conv_id}/items?limit=2&order=asc"
                ))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let list: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(list["data"].as_array().unwrap().len(), 2);
    assert_eq!(list["has_more"], true);

    let (status, updated) = post_json(
        &app,
        &format!("/v1/conversations/{conv_id}"),
        json!({"metadata":{"topic":"project-x"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated["metadata"]["topic"], "project-x");
}

#[tokio::test]
async fn input_tokens_and_compact_have_fallback_and_passthrough() {
    let capture = Arc::new(Capture::default());
    let app = app("advanced", capture.clone());

    let (status, token_resp) = post_json(
        &app,
        "/v1/responses/input_tokens",
        json!({"model":"openai:gpt-4o-mini","input":"hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(token_resp["object"], "response.input_tokens");
    assert_eq!(token_resp["input_tokens"], 42);

    let (status, compact_resp) = post_json(
        &app,
        "/v1/responses/compact",
        json!({"model":"anthropic:claude-3-haiku-20240307","input":"hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(compact_resp["object"], "response.compaction");

    let requests = capture.requests.lock().await;
    assert!(
        requests
            .iter()
            .any(|r| r.url.ends_with("/v1/responses/input_tokens"))
    );
}
