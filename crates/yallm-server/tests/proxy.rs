use std::sync::Arc;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use bytes::Bytes;
use futures::stream;
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Debug, Default)]
struct TransportCapture {
    requests: tokio::sync::Mutex<Vec<yallm_server::TransportRequest>>,
    stream_requests: tokio::sync::Mutex<Vec<yallm_server::TransportRequest>>,
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

    fn send_stream<'a>(
        &'a self,
        req: yallm_server::TransportRequest,
    ) -> yallm_server::TransportStreamFuture<'a> {
        Box::pin(async move {
            self.cap.stream_requests.lock().await.push(req.clone());

            if req.url.ends_with("/v1/chat/completions") {
                let model = req.body.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let is_tool_case = req
                    .body
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .and_then(|m| m.first())
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .map(|c| c.contains("tool"))
                    .unwrap_or(false);
                let body = if is_tool_case {
                    let chunks = vec![
                        json!({
                            "id": "chatcmpl_test",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": Value::Null
                            }]
                        }),
                        json!({
                            "id": "chatcmpl_test",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": ""
                                        }
                                    }]
                                },
                                "finish_reason": Value::Null
                            }]
                        }),
                        json!({
                            "id": "chatcmpl_test",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": 0,
                                        "function": {
                                            "arguments": "{\"city\":\"Beijing\"}"
                                        }
                                    }]
                                },
                                "finish_reason": Value::Null
                            }]
                        }),
                        json!({
                            "id": "chatcmpl_test",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls"
                            }]
                        }),
                    ];

                    let mut body = String::new();
                    for chunk in chunks {
                        body.push_str("data: ");
                        body.push_str(&chunk.to_string());
                        body.push_str("\n\n");
                    }
                    body.push_str("data: [DONE]\n\n");
                    body
                } else {
                    let chunks = vec![
                        json!({
                            "id": "chatcmpl_test",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": Value::Null
                            }]
                        }),
                        json!({
                            "id": "chatcmpl_test",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": "hi"},
                                "finish_reason": Value::Null
                            }]
                        }),
                        json!({
                            "id": "chatcmpl_test",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }),
                    ];

                    let mut body = String::new();
                    for chunk in chunks {
                        body.push_str("data: ");
                        body.push_str(&chunk.to_string());
                        body.push_str("\n\n");
                    }
                    body.push_str("data: [DONE]\n\n");
                    body
                };

                return Ok(yallm_server::TransportStreamResponse {
                    status: 200,
                    headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
                    body: Box::pin(stream::iter(vec![Ok(Bytes::from(body))])),
                });
            }

            if req.url.ends_with("/v1/messages") {
                let model = req.body.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let is_tool_case = req
                    .body
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .and_then(|m| m.first())
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .map(|c| c.contains("tool"))
                    .unwrap_or(false);
                let body = if is_tool_case {
                    format!(
                        "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_test\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"{model}\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{{\"input_tokens\":1,\"output_tokens\":0}}}}}}\n\n\
event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{{}}}}}}\n\n\
event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":\"{{\\\"city\\\":\\\"Bei\"}}}}\n\n\
event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":\"jing\\\"}}\"}}}}\n\n\
event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"tool_use\",\"stop_sequence\":null}},\"usage\":{{\"output_tokens\":2}}}}\n\n\
event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
                    )
                } else {
                    format!(
                        "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_test\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"{model}\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{{\"input_tokens\":1,\"output_tokens\":0}}}}}}\n\n\
event: content_block_start\ndata: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{{\"type\":\"text\",\"text\":\"\"}}}}\n\n\
event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"hello from anthropic\"}}}}\n\n\
event: content_block_stop\ndata: {{\"type\":\"content_block_stop\",\"index\":0}}\n\n\
event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}},\"usage\":{{\"output_tokens\":2}}}}\n\n\
event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
                    )
                };

                return Ok(yallm_server::TransportStreamResponse {
                    status: 200,
                    headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
                    body: Box::pin(stream::iter(vec![Ok(Bytes::from(body))])),
                });
            }

            if req.url.ends_with("/api/chat") {
                let model = req.body.get("model").and_then(|v| v.as_str()).unwrap_or("");
                let body = format!(
                    "{{\"model\":\"{model}\",\"message\":{{\"role\":\"assistant\",\"content\":\"hello from ollama\"}},\"done\":false}}\n\
{{\"model\":\"{model}\",\"message\":{{\"role\":\"assistant\",\"content\":\"\"}},\"done\":true,\"prompt_eval_count\":1,\"eval_count\":2}}\n"
                );

                return Ok(yallm_server::TransportStreamResponse {
                    status: 200,
                    headers: vec![(
                        "content-type".to_string(),
                        "application/x-ndjson".to_string(),
                    )],
                    body: Box::pin(stream::iter(vec![Ok(Bytes::from(body))])),
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

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..Default::default()
    };
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
        reqs[0]
            .headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("authorization") && v == "Bearer test_openai_key")
    );
}

#[tokio::test]
async fn proxies_openai_to_anthropic_upstream_without_network() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..Default::default()
    };
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
            .any(|(k, v)| k.eq_ignore_ascii_case("x-api-key") && v == "test_anthropic_key")
    );
    assert!(
        reqs[0]
            .headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("anthropic-version") && v == "2023-06-01")
    );
}

#[tokio::test]
async fn proxies_streaming_openai_to_anthropic_upstream_without_network() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..Default::default()
    };
    state.provider.anthropic_api_key = Some("test_anthropic_key".to_string());
    state.provider.anthropic_base_url = "http://anthropic.test".to_string();
    state.provider.anthropic_version = "2023-06-01".to_string();

    let app = yallm_server::app_with_state(state);
    let payload = json!({
        "model": "anthropic:claude-3-haiku-20240307",
        "stream": true,
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
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(content_type.starts_with("text/event-stream"));

    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(body.contains("\"chat.completion.chunk\""));
    assert!(body.contains("hello from anthropic"));
    assert!(body.contains("data: [DONE]"));

    let stream_reqs = cap.stream_requests.lock().await;
    assert_eq!(stream_reqs.len(), 1);
    assert!(stream_reqs[0].url.ends_with("/v1/messages"));
    assert_eq!(stream_reqs[0].body["stream"], true);
}

#[tokio::test]
async fn maps_streaming_tool_calls_anthropic_to_openai() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..Default::default()
    };
    state.provider.anthropic_api_key = Some("test_anthropic_key".to_string());
    state.provider.anthropic_base_url = "http://anthropic.test".to_string();
    state.provider.anthropic_version = "2023-06-01".to_string();

    let app = yallm_server::app_with_state(state);
    let payload = json!({
        "model": "anthropic:claude-3-haiku-20240307",
        "stream": true,
        "messages": [{"role": "user", "content": "please call tool"}]
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
    let body = String::from_utf8(bytes.to_vec()).unwrap();

    assert!(body.contains("\"tool_calls\""));
    assert!(body.contains("\"get_weather\""));
    assert!(body.contains("\"finish_reason\":\"tool_calls\""));
}

#[tokio::test]
async fn maps_streaming_tool_calls_openai_to_anthropic() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..Default::default()
    };
    state.provider.openai_api_key = Some("test_openai_key".to_string());
    state.provider.openai_base_url = "http://openai.test".to_string();

    let app = yallm_server::app_with_state(state);
    let payload = json!({
        "model": "openai:gpt-4o-mini",
        "stream": true,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "please call tool"}]
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();

    assert!(body.contains("event: content_block_start"));
    assert!(body.contains("\"type\":\"tool_use\""));
    assert!(body.contains("\"type\":\"input_json_delta\""));
    assert!(body.contains("\"stop_reason\":\"tool_use\""));
}
