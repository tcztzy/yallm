use std::sync::Arc;

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use bytes::Bytes;
use futures::stream;
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Clone, Copy)]
enum EdgeMode {
    NonStream429,
    Stream500,
    StreamMalformedChunk,
    StreamTransportError,
}

#[derive(Clone)]
struct EdgeTransport {
    mode: EdgeMode,
}

impl yallm_server::Transport for EdgeTransport {
    fn send<'a>(
        &'a self,
        _req: yallm_server::TransportRequest,
    ) -> yallm_server::TransportFuture<'a> {
        Box::pin(async move {
            match self.mode {
                EdgeMode::NonStream429 => Ok(yallm_server::TransportResponse {
                    status: 429,
                    headers: vec![("content-type".to_string(), "application/json".to_string())],
                    body: serde_json::to_vec(&json!({
                        "error": {"message": "rate_limited", "code": "too_many_requests"}
                    }))
                    .unwrap(),
                }),
                _ => Err(yallm_server::TransportError {
                    message: "unexpected non-stream call".to_string(),
                }),
            }
        })
    }

    fn send_stream<'a>(
        &'a self,
        _req: yallm_server::TransportRequest,
    ) -> yallm_server::TransportStreamFuture<'a> {
        Box::pin(async move {
            match self.mode {
                EdgeMode::Stream500 => Ok(yallm_server::TransportStreamResponse {
                    status: 500,
                    headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
                    body: Box::pin(stream::iter(vec![Ok(Bytes::from(
                        "data: {\"error\":\"upstream_failed\"}\n\n",
                    ))])),
                }),
                EdgeMode::StreamMalformedChunk => Ok(yallm_server::TransportStreamResponse {
                    status: 200,
                    headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
                    body: Box::pin(stream::iter(vec![Ok(Bytes::from(
                        "data: {this is not json}\n\n",
                    ))])),
                }),
                EdgeMode::StreamTransportError => Err(yallm_server::TransportError {
                    message: "timeout".to_string(),
                }),
                EdgeMode::NonStream429 => Err(yallm_server::TransportError {
                    message: "unexpected stream call".to_string(),
                }),
            }
        })
    }
}

fn app_with_mode(mode: EdgeMode) -> axum::Router {
    let mut state = yallm_server::AppState {
        transport: Arc::new(EdgeTransport { mode }),
        mode: yallm_server::Mode::Proxy,
        ..Default::default()
    };
    state.provider.openai_api_key = Some("test_openai_key".to_string());
    state.provider.openai_base_url = "http://openai.test".to_string();
    yallm_server::app_with_state(state)
}

#[tokio::test]
async fn passes_through_openai_429_status_and_body() {
    let payload = json!({
        "model": "openai:gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let resp = app_with_mode(EdgeMode::NonStream429)
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

    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["provider"], "openai");
    assert_eq!(v["error"]["upstream"]["error"]["code"], "too_many_requests");
}

#[tokio::test]
async fn passes_through_stream_openai_500_status() {
    let payload = json!({
        "model": "openai:gpt-4o-mini",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}]
    });

    let resp = app_with_mode(EdgeMode::Stream500)
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

    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn keeps_stream_open_when_upstream_chunk_is_malformed() {
    let payload = json!({
        "model": "openai:gpt-4o-mini",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}]
    });

    let resp = app_with_mode(EdgeMode::StreamMalformedChunk)
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
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn maps_stream_transport_timeout_to_bad_gateway() {
    let payload = json!({
        "model": "openai:gpt-4o-mini",
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}]
    });

    let resp = app_with_mode(EdgeMode::StreamTransportError)
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

    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["provider"], "openai");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("Failed")
    );
}
