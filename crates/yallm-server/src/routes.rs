//! Route handlers for the LLM API hub.
//!
//! Today this crate exposes a few compatibility endpoints so downstream SDKs can
//! talk to `yallm`:
//! - OpenAI-compatible: `POST /v1/chat/completions`
//! - Anthropic-compatible: `POST /v1/messages`
//! - Ollama-compatible: `POST /api/chat`
//!
//! The implementations below currently use a lightweight mock backend so that
//! SDK integration can be validated before wiring real provider proxying.

use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

/// Health check endpoint
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

// ============================================================================
// OpenAI-compatible: /v1/chat/completions
// ============================================================================

/// Chat completions request (OpenAI-compatible, simplified)
#[derive(Debug, Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessageIn>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiMessageIn {
    pub role: String,
    #[serde(default)]
    pub content: Value,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChatCompletionsResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiChoice {
    pub index: u32,
    pub message: OpenAiMessageOut,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct OpenAiMessageOut {
    pub role: &'static str,
    pub content: String,
}

pub async fn chat_completions(Json(request): Json<ChatCompletionsRequest>) -> impl IntoResponse {
    if request.stream {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": {
                    "message": "Streaming is not implemented yet",
                    "type": "not_implemented"
                }
            })),
        )
            .into_response();
    }

    tracing::debug!(
        "Received chat completions request for model: {}",
        request.model
    );

    let reply = mock_reply(extract_last_user_text_openai(&request.messages));

    let resp = OpenAiChatCompletionsResponse {
        id: format!("chatcmpl_mock_{}", unix_seconds()),
        object: "chat.completion",
        created: unix_seconds(),
        model: request.model,
        choices: vec![OpenAiChoice {
            index: 0,
            message: OpenAiMessageOut {
                role: "assistant",
                content: reply,
            },
            finish_reason: "stop",
        }],
    };

    (StatusCode::OK, Json(resp)).into_response()
}

fn extract_last_user_text_openai(messages: &[OpenAiMessageIn]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| openai_content_to_text(&m.content))
        .filter(|s| !s.is_empty())
}

fn openai_content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                let obj = p.as_object()?;
                match obj.get("type")?.as_str()? {
                    "text" => obj.get("text")?.as_str().map(|s| s.to_string()),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

// ============================================================================
// Anthropic-compatible: /v1/messages
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<AnthropicMessageIn>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageIn {
    pub role: String,
    #[serde(default)]
    pub content: Value,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: &'static str,
    pub role: &'static str,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: &'static str,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text { text: String },
}

pub async fn anthropic_messages(
    Json(request): Json<AnthropicMessagesRequest>,
) -> impl IntoResponse {
    if request.stream {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "not_implemented",
                    "message": "Streaming is not implemented yet"
                }
            })),
        )
            .into_response();
    }

    tracing::debug!(
        "Received anthropic messages request for model: {}",
        request.model
    );

    let reply = mock_reply(extract_last_user_text_anthropic(&request.messages));

    let resp = AnthropicMessageResponse {
        id: format!("msg_mock_{}", unix_seconds()),
        type_: "message",
        role: "assistant",
        content: vec![AnthropicContentBlock::Text { text: reply }],
        model: request.model,
        stop_reason: "end_turn",
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: 0,
            output_tokens: 0,
        },
    };

    (StatusCode::OK, Json(resp)).into_response()
}

fn extract_last_user_text_anthropic(messages: &[AnthropicMessageIn]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| anthropic_content_to_text(&m.content))
        .filter(|s| !s.is_empty())
}

fn anthropic_content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                let obj = b.as_object()?;
                match obj.get("type")?.as_str()? {
                    "text" => obj.get("text")?.as_str().map(|s| s.to_string()),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

// ============================================================================
// Ollama-compatible: /api/chat
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<OllamaMessageIn>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct OllamaMessageIn {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaChatResponse {
    pub model: Option<String>,
    pub message: OllamaMessageOut,
    pub done: bool,
}

#[derive(Debug, Serialize)]
pub struct OllamaMessageOut {
    pub role: &'static str,
    pub content: String,
}

pub async fn ollama_chat(Json(request): Json<OllamaChatRequest>) -> impl IntoResponse {
    if request.stream {
        return (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": "Streaming is not implemented yet"
            })),
        )
            .into_response();
    }

    tracing::debug!(
        "Received ollama chat request for model: {:?}",
        request.model
    );

    let reply = mock_reply(extract_last_user_text_ollama(&request.messages));

    let resp = OllamaChatResponse {
        model: request.model,
        message: OllamaMessageOut {
            role: "assistant",
            content: reply,
        },
        done: true,
    };

    (StatusCode::OK, Json(resp)).into_response()
}

fn extract_last_user_text_ollama(messages: &[OllamaMessageIn]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .filter(|s| !s.is_empty())
}

fn mock_reply(last_user_text: Option<String>) -> String {
    match last_user_text {
        Some(text) => format!("yallm (mock): {text}"),
        None => "yallm (mock): hello".to_string(),
    }
}

fn unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
