//! Route handlers for the LLM API proxy

use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

/// Health check endpoint
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// Chat completions request (OpenAI-compatible)
#[derive(Debug, Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub stream: bool,
}

/// Chat completions endpoint placeholder
pub async fn chat_completions(Json(request): Json<ChatCompletionsRequest>) -> impl IntoResponse {
    // TODO: Implement actual conversion and proxying logic
    tracing::debug!(
        "Received chat completions request for model: {}",
        request.model
    );

    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "Chat completions endpoint not yet implemented",
                "type": "not_implemented"
            }
        })),
    )
}
