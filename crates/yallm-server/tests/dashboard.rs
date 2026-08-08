use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use serde_json::{Value, json};
use tower::ServiceExt;

struct DashboardTransport;

impl yallm_server::Transport for DashboardTransport {
    fn send<'a>(
        &'a self,
        req: yallm_server::TransportRequest,
    ) -> yallm_server::TransportFuture<'a> {
        Box::pin(async move {
            let model = req.body.get("model").and_then(Value::as_str).unwrap_or("");
            let body = json!({
                "id": "chatcmpl_dashboard_test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            });
            Ok(yallm_server::TransportResponse {
                status: 200,
                headers: vec![("content-type".to_string(), "application/json".to_string())],
                body: serde_json::to_vec(&body).unwrap(),
            })
        })
    }

    fn send_stream<'a>(
        &'a self,
        _req: yallm_server::TransportRequest,
    ) -> yallm_server::TransportStreamFuture<'a> {
        Box::pin(async {
            Err(yallm_server::TransportError {
                message: "streaming is not used in dashboard tests".to_string(),
            })
        })
    }
}

// pid + seq instead of clock: SystemTime has µs granularity on macOS, so
// parallel tests can compute the same timestamp and collide on one sqlite
// file (one open wins the schema lock, the other gets SQLITE_BUSY).
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);
fn temp_store_path(name: &str) -> PathBuf {
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "yallm-dashboard-test-{name}-{}-{seq}.json",
        std::process::id()
    ))
}

fn app(name: &str) -> Router {
    let mut env = HashMap::new();
    env.insert("YALLM_MODE".to_string(), "mock".to_string());
    env.insert(yallm_storage::DB_URL_ENV.to_string(), temp_db_url(name));
    let state = yallm_server::AppState::from_loaded_config(yallm_config::LoadedConfig {
        env,
        litellm_models: Vec::new(),
        warnings: Vec::new(),
    });
    yallm_server::app_with_state(state)
}

fn proxy_app(name: &str) -> Router {
    let mut env = HashMap::new();
    env.insert("YALLM_MODE".to_string(), "proxy".to_string());
    env.insert(yallm_storage::DB_URL_ENV.to_string(), temp_db_url(name));
    let mut state = yallm_server::AppState::from_loaded_config(yallm_config::LoadedConfig {
        env,
        litellm_models: Vec::new(),
        warnings: Vec::new(),
    });
    state.transport = Arc::new(DashboardTransport);
    state.provider.openai_api_key = Some("test-key".to_string());
    state.provider.openai_base_url = "http://user:secret@openai.test/v1".to_string();
    yallm_server::app_with_state(state)
}

fn temp_db_url(name: &str) -> String {
    format!(
        "sqlite://{}",
        temp_store_path(name).with_extension("sqlite3").display()
    )
}

async fn request(app: &Router, req: Request<Body>) -> (StatusCode, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

async fn get_json(app: &Router, uri: &str) -> (StatusCode, Value) {
    let (status, body) = request(
        app,
        Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    (status, serde_json::from_str(&body).unwrap())
}

#[tokio::test]
async fn dashboard_serves_page_and_assets() {
    let app = app("assets");

    let (status, body) = request(
        &app,
        Request::builder()
            .method("GET")
            .uri("/dashboard")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("/dashboard/assets/main.js"));
    assert!(body.contains("/dashboard/assets/styles.css"));

    let (status, body) = request(
        &app,
        Request::builder()
            .method("GET")
            .uri("/dashboard/assets/main.js")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("/dashboard/api/events"));

    let (status, body) = request(
        &app,
        Request::builder()
            .method("GET")
            .uri("/dashboard/assets/styles.css")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains(".filters-surface"));
}

#[tokio::test]
async fn dashboard_events_record_api_requests_and_exclude_dashboard() {
    let app = app("events");

    let (status, _) = request(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "openai:gpt-test",
                    "stream": true,
                    "messages": [{"role": "user", "content": "hello"}]
                })
                .to_string(),
            ))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, events) = get_json(&app, "/dashboard/api/events?limit=10").await;
    assert_eq!(status, StatusCode::OK);
    let data = events["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["endpoint"], "/v1/chat/completions");
    assert_eq!(data[0]["provider"], "openai");
    assert_eq!(data[0]["model"], "openai:gpt-test");
    assert_eq!(data[0]["upstream_model"], "gpt-test");
    assert!(data[0]["upstream_url"].is_null());
    assert_eq!(data[0]["stream"], true);

    let (status, _) = request(
        &app,
        Request::builder()
            .method("GET")
            .uri("/dashboard")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (_, events) = get_json(&app, "/dashboard/api/events?limit=10").await;
    assert_eq!(events["data"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn dashboard_events_record_sanitized_upstream_url_for_proxy_requests() {
    let app = proxy_app("upstream-url");

    let (status, _) = request(
        &app,
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "openai:gpt-test",
                    "messages": [{"role": "user", "content": "hello"}]
                })
                .to_string(),
            ))
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, events) = get_json(&app, "/dashboard/api/events?limit=10").await;
    assert_eq!(status, StatusCode::OK);
    let data = events["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(
        data[0]["upstream_url"],
        "http://[redacted]@openai.test/v1/chat/completions"
    );
}

#[tokio::test]
async fn dashboard_events_can_be_cleared() {
    let app = app("clear");

    let (status, _) = request(
        &app,
        Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, body) = request(
        &app,
        Request::builder()
            .method("DELETE")
            .uri("/dashboard/api/events")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let deleted: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(deleted["deleted"], 1);

    let (_, events) = get_json(&app, "/dashboard/api/events?limit=10").await;
    assert!(events["data"].as_array().unwrap().is_empty());
}
