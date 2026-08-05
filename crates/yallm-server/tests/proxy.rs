use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

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

fn state_from_litellm_config(
    config: &str,
    mut env: HashMap<String, String>,
    transport: MockTransport,
) -> yallm_server::AppState {
    env.entry(yallm_storage::DB_URL_ENV.to_string())
        .or_insert_with(|| temp_db_url("proxy-litellm"));
    let mut warnings = Vec::new();
    let litellm_models = yallm_config::parse_litellm_config_str(config, &env, &mut warnings);
    let mut state = yallm_server::AppState::from_loaded_config(yallm_config::LoadedConfig {
        env,
        litellm_models,
        warnings,
    });
    state.transport = Arc::new(transport);
    state.mode = yallm_server::Mode::Proxy;
    state
}

fn state_with_temp_store(name: &str) -> yallm_server::AppState {
    let mut env = HashMap::new();
    env.insert(yallm_storage::DB_URL_ENV.to_string(), temp_db_url(name));
    yallm_server::AppState::from_loaded_config(yallm_config::LoadedConfig {
        env,
        litellm_models: Vec::new(),
        warnings: Vec::new(),
    })
}

#[test]
fn acp_provider_prefix_and_env_config_are_supported() {
    let mut env = HashMap::new();
    env.insert(
        yallm_storage::DB_URL_ENV.to_string(),
        temp_db_url("acp-config"),
    );
    env.insert("YALLM_DEFAULT_PROVIDER".to_string(), "acp".to_string());
    env.insert(
        "YALLM_ACP_COMMAND".to_string(),
        "npx -y @agentclientprotocol/codex-acp".to_string(),
    );
    let state = yallm_server::AppState::from_loaded_config(yallm_config::LoadedConfig {
        env,
        litellm_models: Vec::new(),
        warnings: Vec::new(),
    });

    assert_eq!(state.default_provider, yallm_server::Provider::Acp);
    assert_eq!(
        state.provider.acp_command.as_deref(),
        Some("npx -y @agentclientprotocol/codex-acp")
    );

    let target = yallm_server::choose_provider(&state, "acp:codex");
    assert_eq!(target.provider, yallm_server::Provider::Acp);
    assert_eq!(target.upstream_model, "codex");
}

fn temp_store_path(name: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("yallm-proxy-test-{name}-{ts}.json"))
}

fn temp_db_url(name: &str) -> String {
    format!(
        "sqlite://{}",
        temp_store_path(name).with_extension("sqlite3").display()
    )
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
                let is_reasoning_case = req
                    .body
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .and_then(|m| m.first())
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .map(|c| c.contains("reason"))
                    .unwrap_or(false);
                let body = if is_reasoning_case {
                    json!({
                        "id": "chatcmpl_test",
                        "object": "chat.completion",
                        "created": 0,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "I considered the request."
                            },
                            "finish_reason": "length"
                        }],
                        "usage": {"prompt_tokens": 84, "completion_tokens": 64, "total_tokens": 148}
                    })
                } else {
                    json!({
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
                    })
                };

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
                let is_reasoning_case = req
                    .body
                    .get("messages")
                    .and_then(|m| m.as_array())
                    .and_then(|m| m.first())
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .map(|c| c.contains("reason"))
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
                } else if is_reasoning_case {
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
                                "delta": {"reasoning_content": "I considered "},
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
                                "delta": {"reasoning_content": "the request."},
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
                                "finish_reason": "length"
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
async fn litellm_openai_alias_routes_to_configured_upstream() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };
    let mut env = HashMap::new();
    env.insert("OPENAI_KEY".to_string(), "alias_openai_key".to_string());
    let state = state_from_litellm_config(
        r#"
model_list:
  - model_name: gpt-alias
    litellm_params:
      model: openai/real-gpt
      api_base: http://openai.test/v1
      api_key: os.environ/OPENAI_KEY
"#,
        env,
        transport,
    );
    let app = yallm_server::app_with_state(state);

    let payload = json!({
        "model": "gpt-alias",
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
    let reqs = cap.requests.lock().await;
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].url, "http://openai.test/v1/chat/completions");
    assert_eq!(reqs[0].body["model"], "real-gpt");
    assert!(reqs[0].headers.iter().any(|(k, v)| {
        k.eq_ignore_ascii_case("authorization") && v == "Bearer alias_openai_key"
    }));
}

#[tokio::test]
async fn request_headers_override_route_headers_and_use_route_allowlist() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };
    let mut env = HashMap::new();
    env.insert("ROUTE_TOKEN".to_string(), "route_token".to_string());
    let state = state_from_litellm_config(
        r#"
model_list:
  - model_name: gpt-headered
    yallm_params:
      provider: openai
      model: real-gpt
      api_base: http://openai.test/v1
      headers:
        Authorization: Bearer ${ROUTE_TOKEN}
        x-route-header: route-value
      forward_headers:
        - authorization
        - x-request-id
"#,
        env,
        transport,
    );
    let app = yallm_server::app_with_state(state);

    let payload = json!({
        "model": "gpt-headered",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer request_token")
                .header("x-request-id", "req-123")
                .header("x-not-forwarded", "drop-me")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let reqs = cap.requests.lock().await;
    assert_eq!(reqs.len(), 1);
    assert!(
        reqs[0].headers.iter().any(|(k, v)| {
            k.eq_ignore_ascii_case("authorization") && v == "Bearer request_token"
        })
    );
    assert!(
        reqs[0]
            .headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("x-route-header") && v == "route-value")
    );
    assert!(
        reqs[0]
            .headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("x-request-id") && v == "req-123")
    );
    assert!(
        !reqs[0]
            .headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("x-not-forwarded"))
    );
}

#[tokio::test]
async fn litellm_anthropic_alias_routes_with_env_key() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };
    let mut env = HashMap::new();
    env.insert(
        "ANTHROPIC_KEY".to_string(),
        "alias_anthropic_key".to_string(),
    );
    let state = state_from_litellm_config(
        r#"
model_list:
  - model_name: claude-alias
    litellm_params:
      model: anthropic/claude-real
      api_base: http://anthropic.test/v1
      api_key: os.environ/ANTHROPIC_KEY
      api_version: "2024-01-01"
"#,
        env,
        transport,
    );
    let app = yallm_server::app_with_state(state);

    let payload = json!({
        "model": "claude-alias",
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
    let reqs = cap.requests.lock().await;
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].url, "http://anthropic.test/v1/messages");
    assert_eq!(reqs[0].body["model"], "claude-real");
    assert!(
        reqs[0]
            .headers
            .iter()
            .any(|(k, v)| { k.eq_ignore_ascii_case("x-api-key") && v == "alias_anthropic_key" })
    );
    assert!(
        reqs[0]
            .headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("anthropic-version") && v == "2024-01-01")
    );
}

#[tokio::test]
async fn anthropic_auth_token_routes_as_authorization_header() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..state_with_temp_store("proxy")
    };
    state.provider.anthropic_auth_token = Some("test_auth_token".to_string());
    state.provider.anthropic_base_url = "http://anthropic.test".to_string();

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
    let reqs = cap.requests.lock().await;
    assert_eq!(reqs.len(), 1);
    assert!(reqs[0].headers.iter().any(|(k, v)| {
        k.eq_ignore_ascii_case("authorization") && v == "Bearer test_auth_token"
    }));
    assert!(
        !reqs[0]
            .headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("x-api-key"))
    );
}

#[tokio::test]
async fn request_auth_header_replaces_configured_anthropic_api_key() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..state_with_temp_store("proxy")
    };
    state.provider.anthropic_api_key = Some("configured_key".to_string());
    state.provider.anthropic_base_url = "http://anthropic.test".to_string();

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
                .header("authorization", "Bearer request_token")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let reqs = cap.requests.lock().await;
    assert_eq!(reqs.len(), 1);
    assert!(
        reqs[0].headers.iter().any(|(k, v)| {
            k.eq_ignore_ascii_case("authorization") && v == "Bearer request_token"
        })
    );
    assert!(
        !reqs[0]
            .headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("x-api-key"))
    );
}

#[tokio::test]
async fn litellm_ollama_alias_routes_without_auth() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };
    let state = state_from_litellm_config(
        r#"
model_list:
  - model_name: local-llama
    litellm_params:
      model: ollama/llama3
      api_base: http://ollama.test
"#,
        HashMap::new(),
        transport,
    );
    let app = yallm_server::app_with_state(state);

    let payload = json!({
        "model": "local-llama",
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
    let reqs = cap.requests.lock().await;
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].url, "http://ollama.test/api/chat");
    assert_eq!(reqs[0].body["model"], "llama3");
    assert!(reqs[0].headers.is_empty());
}

#[tokio::test]
async fn litellm_models_list_exposes_supported_aliases_only() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };
    let state = state_from_litellm_config(
        r#"
model_list:
  - model_name: gpt-alias
    litellm_params:
      model: gpt-4o
      api_key: test-key
  - model_name: azure-alias
    litellm_params:
      model: azure/gpt-4o
"#,
        HashMap::new(),
        transport,
    );
    let app = yallm_server::app_with_state(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert_eq!(ids, vec!["gpt-alias"]);
    assert_eq!(v["data"][0]["owned_by"], "litellm");
}

#[tokio::test]
async fn litellm_models_list_exposes_all_aliases_under_all_interfaces() {
    // Every alias is reachable via every protocol (yallm converts on the fly),
    // so /v1/models lists the same alias set under every interface filter.
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };
    let state = state_from_litellm_config(
        r#"
model_list:
  - model_name: gpt-alias
    litellm_params:
      model: openai/gpt-4o
      api_key: test-key
  - model_name: claude-alias
    litellm_params:
      model: anthropic/claude-3-haiku-20240307
      api_key: test-key
  - model_name: llama-alias
    litellm_params:
      model: ollama/llama3
"#,
        HashMap::new(),
        transport,
    );
    let app = yallm_server::app_with_state(state);

    for (uri, expected) in [
        (
            "/v1/models?interface=openai",
            vec!["claude-alias", "gpt-alias", "llama-alias"],
        ),
        (
            "/v1/models?interface=anthropic",
            vec!["claude-alias", "gpt-alias", "llama-alias"],
        ),
    ] {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let mut ids: Vec<&str> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["id"].as_str())
            .collect();
        ids.sort();
        assert_eq!(ids, expected, "interface={uri}");
    }
}

#[tokio::test]
async fn run_rejects_partial_tls_config() {
    use std::net::SocketAddr;
    let cfg = yallm_server::ServerConfig {
        addr: SocketAddr::from(([127, 0, 0, 1], 0)),
        tls_cert: Some("/nonexistent/cert.pem".to_string()),
        tls_key: None,
        litellm_config: None,
    };
    let err = yallm_server::run(cfg).await.expect_err("should reject");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

    let cfg = yallm_server::ServerConfig {
        addr: SocketAddr::from(([127, 0, 0, 1], 0)),
        tls_cert: None,
        tls_key: Some("/nonexistent/key.pem".to_string()),
        litellm_config: None,
    };
    let err = yallm_server::run(cfg).await.expect_err("should reject");
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[tokio::test]
async fn proxies_openai_to_openai_upstream_without_network() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..state_with_temp_store("proxy")
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
async fn maps_openai_reasoning_to_anthropic_thinking_without_network() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..state_with_temp_store("proxy")
    };
    state.provider.openai_api_key = Some("test_openai_key".to_string());
    state.provider.openai_base_url = "http://openai.test".to_string();

    let app = yallm_server::app_with_state(state);
    let payload = json!({
        "model": "openai:deepseek-v4-flash",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "please reason"}]
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
    let v: Value = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(v["model"], "deepseek-v4-flash");
    assert_eq!(v["stop_reason"], "max_tokens");
    assert_eq!(v["content"][0]["type"], "thinking");
    assert_eq!(v["content"][0]["thinking"], "I considered the request.");
    assert_eq!(v["content"][0]["signature"], "chatcmpl_test");
}

#[tokio::test]
async fn proxies_openai_to_anthropic_upstream_without_network() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..state_with_temp_store("proxy")
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
        ..state_with_temp_store("proxy")
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
        ..state_with_temp_store("proxy")
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
        ..state_with_temp_store("proxy")
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

#[tokio::test]
async fn maps_streaming_openai_reasoning_to_anthropic_thinking() {
    let cap = Arc::new(TransportCapture::default());
    let transport = MockTransport { cap: cap.clone() };

    let mut state = yallm_server::AppState {
        transport: Arc::new(transport),
        mode: yallm_server::Mode::Proxy,
        ..state_with_temp_store("proxy")
    };
    state.provider.openai_api_key = Some("test_openai_key".to_string());
    state.provider.openai_base_url = "http://openai.test".to_string();

    let app = yallm_server::app_with_state(state);
    let payload = json!({
        "model": "openai:deepseek-v4-flash",
        "stream": true,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "please reason"}]
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

    assert!(body.contains("\"type\":\"thinking\""));
    assert!(body.contains("\"type\":\"thinking_delta\""));
    assert!(body.contains("I considered "));
    assert!(body.contains("the request."));
    assert!(body.contains("\"type\":\"signature_delta\""));
    assert!(body.contains("\"signature\":\"chatcmpl_test\""));
    assert!(body.contains("\"stop_reason\":\"max_tokens\""));
}
