use std::{
    collections::HashMap,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tower::ServiceExt;

fn app() -> Router {
    yallm_server::app_with_state(state_with_temp_store("compat"))
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

fn temp_store_path(name: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("yallm-compat-test-{name}-{ts}.json"))
}

fn temp_db_url(name: &str) -> String {
    format!(
        "sqlite://{}",
        temp_store_path(name).with_extension("sqlite3").display()
    )
}

#[tokio::test]
async fn health_ok() {
    let resp = app()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], "ok");
}

#[tokio::test]
async fn openai_chat_completions_ok() {
    let payload = serde_json::json!({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    });

    let resp = app()
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
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(v["object"], "chat.completion");
    assert!(v.get("id").is_some());
    assert_eq!(v["model"], "gpt-4o-mini");
    assert!(v["choices"].is_array());
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
}

#[tokio::test]
async fn openai_chat_completions_stream_ok() {
    let payload = serde_json::json!({
        "model": "gpt-4o-mini",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}],
    });

    let resp = app()
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

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("\"chat.completion.chunk\""));
    assert!(text.contains("data: [DONE]"));
}

#[tokio::test]
async fn anthropic_messages_ok() {
    let payload = serde_json::json!({
        "model": "claude-3-haiku-20240307",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}],
    });

    let resp = app()
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
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["model"], "claude-3-haiku-20240307");
    assert!(v["content"].is_array());
    assert_eq!(v["content"][0]["type"], "text");
    assert!(v["usage"]["input_tokens"].is_number());
    assert!(v["usage"]["output_tokens"].is_number());
}

#[tokio::test]
async fn anthropic_messages_stream_ok() {
    let payload = serde_json::json!({
        "model": "claude-3-haiku-20240307",
        "stream": true,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}],
    });

    let resp = app()
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
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(content_type.starts_with("text/event-stream"));

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("event: message_start"));
    assert!(text.contains("event: message_delta"));
    assert!(text.contains("event: message_stop"));
}

#[tokio::test]
async fn models_list_default() {
    let resp = app()
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
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Default provider is OpenAI, so expect "object": "list"
    assert_eq!(v["object"], "list");
    assert!(v["data"].is_array());
    assert!(!v["data"].as_array().unwrap().is_empty());
    assert_eq!(v["data"][0]["object"], "model");
}

#[tokio::test]
async fn models_list_openai() {
    let resp = app()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models?interface=openai")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(v["object"], "list");
    assert!(v["data"].is_array());
    assert!(!v["data"].as_array().unwrap().is_empty());
    assert_eq!(v["data"][0]["object"], "model");
    assert_eq!(v["data"][0]["owned_by"], "openai");
}

#[tokio::test]
async fn models_list_anthropic() {
    let resp = app()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models?interface=anthropic")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Anthropic format: top-level "data" array, each has "type": "model"
    assert!(v["data"].is_array());
    assert!(!v["data"].as_array().unwrap().is_empty());
    assert_eq!(v["data"][0]["type"], "model");
    assert!(v["data"][0]["id"].as_str().unwrap().contains("claude"));
}

#[tokio::test]
async fn models_list_invalid_interface_falls_back() {
    // Unknown interface falls back to default provider (OpenAI)
    let resp = app()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models?interface=unknown")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(v["object"], "list");
}

#[tokio::test]
async fn ollama_chat_ok() {
    let payload = serde_json::json!({
        "model": "llama3",
        "messages": [{"role": "user", "content": "hello"}],
    });

    let resp = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(payload.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(v["model"], "llama3");
    assert_eq!(v["message"]["role"], "assistant");
    assert_eq!(v["done"], true);
}

#[tokio::test]
async fn ollama_chat_stream_ok() {
    let payload = serde_json::json!({
        "model": "llama3",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}],
    });

    let resp = app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
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
    assert_eq!(content_type, "application/x-ndjson");

    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();

    assert!(lines.len() >= 2);
    let first: serde_json::Value = serde_json::from_str(lines.first().unwrap()).unwrap();
    let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
    assert_eq!(first["done"], false);
    assert_eq!(last["done"], true);
}
