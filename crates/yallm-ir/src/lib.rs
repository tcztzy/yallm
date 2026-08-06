//! yallm-ir: the Intermediate Representation shared by every crate.
//!
//! All protocol conversions collapse through this model so yallm avoids an
//! N×M matrix of per-protocol translators: `openai ↔ ir ↔ anthropic ↔ ir`,
//! plus Ollama and ACP on the same IR. Every downstream request body is
//! parsed into a [`ChatRequest`] (messages + sampling knobs), every upstream
//! reply is a [`ChatResponse`] (choices with [`Message`]s + finish reason).
//!
//! Content model: a [`Message`] holds `Vec<Content>` — text, tool calls,
//! tool results, or images. Conversions are lossy by design: providers that
//! cannot represent a content kind drop it silently (e.g. `yallm-acp`'s
//! `message_text` skips images and tool calls when building a prompt), never
//! error.
//!
//! [`Source`] is stamped on messages by the provider conversions (provenance
//! for storage/round-tripping); `Message.raw` may carry the original payload.
//! Nothing here does I/O — this crate is pure types + conversions.

#![warn(missing_docs)]

use serde::{Deserialize, Serialize};

/// Source API type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// Message came from the OpenAI API
    OpenAI,
    /// Message came from the Anthropic API
    Anthropic,
    /// Message came from the Ollama API
    Ollama,
}

/// Message role in conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt / instructions
    System,
    /// User turn
    User,
    /// Assistant turn
    Assistant,
    /// Tool result / tool-call carrier
    Tool,
}

impl Role {
    /// Serialized role name (lowercase)
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// Text content block
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextContent {
    /// The text payload
    pub text: String,
}

impl TextContent {
    /// Create a text block
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Image source type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSourceType {
    /// Image data is raw base64
    Base64,
    /// Image data is a URL
    Url,
}

/// Image content block
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageContent {
    /// How `data` is encoded
    pub source_type: ImageSourceType,
    /// MIME type of the image
    pub media_type: String,
    /// Base64 payload or URL, per `source_type`
    pub data: String,
}

/// Tool call content block
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallContent {
    /// Tool call id (matches `ToolResultContent.tool_call_id`)
    pub id: String,
    /// Name of the tool to invoke
    pub name: String,
    /// JSON-encoded tool arguments
    pub arguments: String,
}

/// Tool result content block
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultContent {
    /// Id of the tool call this result answers
    pub tool_call_id: String,
    /// Tool output (string; structured results are JSON-encoded by the caller)
    pub content: String,
}

/// Content block: exactly one kind per `Message`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Plain text
    Text(TextContent),
    /// Image (not representable by every provider)
    Image(ImageContent),
    /// Tool invocation request
    ToolCall(ToolCallContent),
    /// Tool invocation result
    ToolResult(ToolResultContent),
}

impl Content {
    /// Convenience: a text content block
    pub fn text(text: impl Into<String>) -> Self {
        Content::Text(TextContent::new(text))
    }
}

/// Unified message representation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who sent the message
    pub role: Role,
    /// Content blocks (may be empty)
    pub content: Vec<Content>,
    /// Provider that produced this message (provenance), when known
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    /// Original provider payload, when kept for round-tripping
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

impl Message {
    /// Create a message from role + content blocks
    pub fn new(role: Role, content: Vec<Content>) -> Self {
        Self {
            role,
            content,
            source: None,
            raw: None,
        }
    }

    /// Create a single-text message
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self::new(role, vec![Content::text(text)])
    }

    /// Stamp the producing provider on this message
    pub fn with_source(mut self, source: Source) -> Self {
        self.source = Some(source);
        self
    }

    /// Attach the original provider payload to this message
    pub fn with_raw(mut self, raw: serde_json::Value) -> Self {
        self.raw = Some(raw);
        self
    }
}

/// Unified chat completion request
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Target model (may carry a `provider:`/`provider/` prefix or a LiteLLM alias)
    pub model: String,
    /// Conversation so far, oldest first
    pub messages: Vec<Message>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Whether the caller wants a streaming reply
    #[serde(default)]
    pub stream: bool,
}

/// Token usage information
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens
    pub prompt_tokens: u32,
    /// Output tokens
    pub completion_tokens: u32,
    /// prompt + completion
    pub total_tokens: u32,
}

/// A single completion choice
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// Position of this choice (0-based)
    pub index: u32,
    /// The assistant message
    pub message: Message,
    /// Why generation stopped (`stop`, `length`, …)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Unified chat completion response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Response id
    pub id: String,
    /// Model that produced the reply (upstream model)
    pub model: String,
    /// Generated choices
    pub choices: Vec<Choice>,
    /// Token usage, when reported
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// ============================================================================
// Stream IR Types
// ============================================================================

/// Text content delta in streaming response
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDelta {
    /// Incremental text chunk
    pub text: String,
}

/// Tool call delta in streaming response
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Tool index this delta belongs to
    pub index: u32,
    /// Tool call id, set on the first delta of a call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool name, set on the first delta of a call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Incremental arguments chunk (JSON fragments)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Delta content types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DeltaContent {
    /// Text fragment
    TextDelta(TextDelta),
    /// Tool call fragment
    ToolCallDelta(ToolCallDelta),
}

/// A single choice delta in streaming response
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceDelta {
    /// Position of this choice (0-based)
    pub index: u32,
    /// Content fragments so far
    pub content: Vec<DeltaContent>,
    /// Set on the final delta of a choice
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Unified streaming response chunk
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Response id
    pub id: String,
    /// Model that produced the reply
    pub model: String,
    /// Choice deltas (usually one)
    pub choices: Vec<ChoiceDelta>,
    /// Token usage, when reported
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Provider that produced this stream (provenance), when known
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    /// Original provider payload, when kept for round-tripping
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}
