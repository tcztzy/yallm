use std::sync::Arc;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Debug, Default)]
struct TransportCapture {
    requests: tokio::sync::Mutex<Vec<yallm_server::TransportRequest>>,
}

#[derive(Clone)]
struct MockTransport {
    cap: Arc<TransportCapture>,
}

impl yallm_server::Transport for MockTransport {
    fn send<'a>(
        &'a self,
        req: yallm_server::TransportRequest,
    ) -> yallm_server::TransportFuture<'a> {
        Box::pin(async move {
            self.cap.requests.lock().await.push(req.clone());

            if req.url.ends_with("/v1/chat/completions") {
                let model = req.body.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let body = json!({
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                });

                return Ok(yallm_server::TransportResponse {
                    status: 200,
                    headers: vec![("content-type".to_string(), "application/json".to_string())],
                    body: serde_json::to_vec(&body).unwrap(),
                });
            }

            if req.url.ends_with("/v1/messages") {
                let model = req.body.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let body = json!({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [{"type":"text","text":"hello from anthropic"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": {"input_tokens": 1, "output_tokens": 2}
                });

                return Ok(yallm_server::TransportResponse {
                    status: 200,
                    headers: vec![("content-type".to_string(), "application/json".to_string())],
                    body: serde_json::to_vec(&body).unwrap(),
                });
            }

            if req.url.ends_with("/api/chat") {
                let model = req.body.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let body = json!({
                    "model": model,
                    "message": {"role": "assistant", "content": "hello from ollama"},
                    "done": true,
                    "prompt_eval_count": 1,
                    "eval_count": 2
                });

                return Ok(yallm_server::TransportResponse {
                    status: 200,
                    headers: vec![("content-type".to_string(), "application/json".to_string())],
                    body: serde_json::to_vec(&body).unwrap(),
                });
            }

            Err(yallm_server::TransportError {
                message: "unknown url".to_string(),
            })
        })
    }
}

#[tokio::test]
async fn proxies_openai_to_openai_upstream_without_network() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState::default();
    state.transport = Arc::new(transport);
    state.mode = yallm_server::Mode::Proxy;
    state.provider.openai_api_key = Some("test_openai_key".to_string());
    state.provider.openai_base_url = "http://openai.test".to_string();

    let app = yallm_server::app_with_state(state);

    let payload = json!({
        "model": "openai:gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["model"], "gpt-4o-mini");
    assert_eq!(v["choices"][0]["message"]["content"], "hi");

    let reqs = cap.requests.lock().await;
    assert_eq!(reqs.len(), 1);
    assert!(reqs[0].url.ends_with("/v1/chat/completions"));
    assert_eq!(reqs[0].body["model"], "gpt-4o-mini");
    assert!(
        reqs[0].headers.iter().any(
            |(k, v)| k.to_ascii_lowercase() == "authorization" && v == "Bearer test_openai_key"
        )
    );
}

#[tokio::test]
async fn proxies_openai_to_anthropic_upstream_without_network() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState::default();
    state.transport = Arc::new(transport);
    state.mode = yallm_server::Mode::Proxy;
    state.provider.anthropic_api_key = Some("test_anthropic_key".to_string());
    state.provider.anthropic_base_url = "http://anthropic.test".to_string();
    state.provider.anthropic_version = "2023-06-01".to_string();

    let app = yallm_server::app_with_state(state);

    let payload = json!({
        "model": "anthropic:claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["model"], "claude-3-haiku-20240307");
    assert_eq!(
        v["choices"][0]["message"]["content"],
        "hello from anthropic"
    );

    let reqs = cap.requests.lock().await;
    assert_eq!(reqs.len(), 1);
    assert!(reqs[0].url.ends_with("/v1/messages"));
    assert_eq!(reqs[0].body["model"], "claude-3-haiku-20240307");
    assert!(
        reqs[0]
            .headers
            .iter()
            .any(|(k, v)| k.to_ascii_lowercase() == "x-api-key" && v == "test_anthropic_key")
    );
    assert!(
        reqs[0]
            .headers
            .iter()
            .any(|(k, v)| k.to_ascii_lowercase() == "anthropic-version" && v == "2023-06-01")
    );
}
