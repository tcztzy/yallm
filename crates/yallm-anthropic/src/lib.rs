//! Anthropic API types and conversions

use serde::{Deserialize, Serialize};
use yallm_ir::{
    ChatRequest, ChatResponse, Choice, ChoiceDelta, Content, DeltaContent, ImageContent,
    ImageSourceType, Message, Role, Source, StreamChunk, TextDelta, ToolCallContent,
    ToolCallDelta, ToolResultContent, Usage,
};

// ============================================================================
// Anthropic Request Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text { text: String },
    Image { source: AnthropicImageSource },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

// ============================================================================
// Anthropic Response Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<AnthropicContentBlock>,
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ============================================================================
// Anthropic Stream Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    MessageStart { message: AnthropicMessageStart },
    ContentBlockStart { index: u32, content_block: AnthropicContentBlockStart },
    ContentBlockDelta { index: u32, delta: AnthropicDelta },
    ContentBlockStop { index: u32 },
    MessageDelta { delta: AnthropicMessageDelta, usage: Option<AnthropicStreamUsage> },
    MessageStop,
    Ping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessageStart {
    pub id: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockStart {
    Text { text: String },
    ToolUse { id: String, name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicStreamUsage {
    pub output_tokens: u32,
}

// ============================================================================
// Conversions: Anthropic -> IR
// ============================================================================

impl From<AnthropicRequest> for ChatRequest {
    fn from(req: AnthropicRequest) -> Self {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(system) = req.system {
            messages.push(Message::text(Role::System, system));
        }

        // Convert messages
        for msg in req.messages {
            messages.extend(convert_anthropic_message(msg));
        }

        ChatRequest {
            model: req.model,
            messages,
            max_tokens: Some(req.max_tokens),
            temperature: req.temperature,
            top_p: req.top_p,
            stream: req.stream.unwrap_or(false),
        }
    }
}

fn convert_anthropic_message(msg: AnthropicMessage) -> Vec<Message> {
    let role = match msg.role.as_str() {
        "assistant" => Role::Assistant,
        _ => Role::User,
    };

    match msg.content {
        AnthropicContent::Text(text) => {
            vec![Message::text(role, text).with_source(Source::Anthropic)]
        }
        AnthropicContent::Blocks(blocks) => {
            let mut messages = Vec::new();
            let mut content = Vec::new();

            for block in blocks {
                match block {
                    AnthropicContentBlock::Text { text } => {
                        content.push(Content::text(text));
                    }
                    AnthropicContentBlock::Image { source } => {
                        let source_type = if source.source_type == "url" {
                            ImageSourceType::Url
                        } else {
                            ImageSourceType::Base64
                        };
                        content.push(Content::Image(ImageContent {
                            source_type,
                            media_type: source.media_type,
                            data: source.data,
                        }));
                    }
                    AnthropicContentBlock::ToolUse { id, name, input } => {
                        content.push(Content::ToolCall(ToolCallContent {
                            id,
                            name,
                            arguments: serde_json::to_string(&input).unwrap_or_default(),
                        }));
                    }
                    AnthropicContentBlock::ToolResult { tool_use_id, content: c } => {
                        // Tool results become separate TOOL messages
                        if !content.is_empty() {
                            messages.push(
                                Message::new(role, std::mem::take(&mut content))
                                    .with_source(Source::Anthropic),
                            );
                        }
                        messages.push(
                            Message::new(
                                Role::Tool,
                                vec![Content::ToolResult(ToolResultContent {
                                    tool_call_id: tool_use_id,
                                    content: c,
                                })],
                            )
                            .with_source(Source::Anthropic),
                        );
                    }
                }
            }

            if !content.is_empty() {
                messages.push(Message::new(role, content).with_source(Source::Anthropic));
            }

            messages
        }
    }
}

// ============================================================================
// Conversions: IR -> Anthropic
// ============================================================================

impl From<&ChatRequest> for AnthropicRequest {
    fn from(req: &ChatRequest) -> Self {
        let mut system = None;
        let mut messages = Vec::new();
        let mut pending_tool_results: Vec<AnthropicContentBlock> = Vec::new();

        for msg in &req.messages {
            if msg.role == Role::System {
                system = Some(extract_text(&msg.content));
                continue;
            }

            if msg.role == Role::Tool {
                // Collect tool results
                for c in &msg.content {
                    if let Content::ToolResult(tr) = c {
                        pending_tool_results.push(AnthropicContentBlock::ToolResult {
                            tool_use_id: tr.tool_call_id.clone(),
                            content: tr.content.clone(),
                        });
                    }
                }
                continue;
            }

            // Flush pending tool results
            if !pending_tool_results.is_empty() {
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicContent::Blocks(std::mem::take(&mut pending_tool_results)),
                });
            }

            messages.push(AnthropicMessage {
                role: msg.role.as_str().to_string(),
                content: content_to_anthropic(&msg.content),
            });
        }

        // Flush remaining tool results
        if !pending_tool_results.is_empty() {
            messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicContent::Blocks(pending_tool_results),
            });
        }

        AnthropicRequest {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens.unwrap_or(4096),
            system,
            temperature: req.temperature,
            top_p: req.top_p,
            stream: if req.stream { Some(true) } else { None },
        }
    }
}

fn extract_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| {
            if let Content::Text(t) = c {
                Some(t.text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn content_to_anthropic(content: &[Content]) -> AnthropicContent {
    let has_complex = content.iter().any(|c| {
        matches!(c, Content::Image(_) | Content::ToolCall(_))
    });

    if !has_complex && content.len() == 1 {
        if let Content::Text(t) = &content[0] {
            return AnthropicContent::Text(t.text.clone());
        }
    }

    let blocks: Vec<_> = content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(AnthropicContentBlock::Text {
                text: t.text.clone(),
            }),
            Content::Image(i) => Some(AnthropicContentBlock::Image {
                source: AnthropicImageSource {
                    source_type: match i.source_type {
                        ImageSourceType::Url => "url",
                        ImageSourceType::Base64 => "base64",
                    }
                    .to_string(),
                    media_type: i.media_type.clone(),
                    data: i.data.clone(),
                },
            }),
            Content::ToolCall(tc) => {
                let input: serde_json::Value =
                    serde_json::from_str(&tc.arguments).unwrap_or_default();
                Some(AnthropicContentBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    input,
                })
            }
            Content::ToolResult(_) => None,
        })
        .collect();

    AnthropicContent::Blocks(blocks)
}

// ============================================================================
// Response Conversions
// ============================================================================

impl From<AnthropicResponse> for ChatResponse {
    fn from(resp: AnthropicResponse) -> Self {
        let content: Vec<Content> = resp
            .content
            .into_iter()
            .filter_map(|b| match b {
                AnthropicContentBlock::Text { text } => Some(Content::text(text)),
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    Some(Content::ToolCall(ToolCallContent {
                        id,
                        name,
                        arguments: serde_json::to_string(&input).unwrap_or_default(),
                    }))
                }
                _ => None,
            })
            .collect();

        let usage = resp.usage.map(|u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });

        ChatResponse {
            id: resp.id,
            model: resp.model,
            choices: vec![Choice {
                index: 0,
                message: Message::new(Role::Assistant, content)
                    .with_source(Source::Anthropic),
                finish_reason: resp.stop_reason,
            }],
            usage,
        }
    }
}

impl From<&ChatResponse> for AnthropicResponse {
    fn from(resp: &ChatResponse) -> Self {
        let msg = resp.choices.first().map(|c| &c.message);

        let content: Vec<AnthropicContentBlock> = msg
            .map(|m| {
                m.content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text(t) => {
                            Some(AnthropicContentBlock::Text { text: t.text.clone() })
                        }
                        Content::ToolCall(tc) => {
                            let input = serde_json::from_str(&tc.arguments).unwrap_or_default();
                            Some(AnthropicContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                input,
                            })
                        }
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();

        AnthropicResponse {
            id: resp.id.clone(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            model: resp.model.clone(),
            content,
            stop_reason: resp.choices.first().and_then(|c| c.finish_reason.clone()),
            usage: resp.usage.as_ref().map(|u| AnthropicUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
            }),
        }
    }
}

// ============================================================================
// Stream Conversions
// ============================================================================

impl From<AnthropicStreamEvent> for StreamChunk {
    fn from(event: AnthropicStreamEvent) -> Self {
        let mut chunk = StreamChunk {
            id: String::new(),
            model: String::new(),
            choices: Vec::new(),
            usage: None,
            source: Some(Source::Anthropic),
            raw: None,
        };

        match event {
            AnthropicStreamEvent::MessageStart { message } => {
                chunk.id = message.id;
                chunk.model = message.model;
                if let Some(u) = message.usage {
                    chunk.usage = Some(Usage {
                        prompt_tokens: u.input_tokens,
                        completion_tokens: u.output_tokens,
                        total_tokens: u.input_tokens + u.output_tokens,
                    });
                }
            }
            AnthropicStreamEvent::ContentBlockStart { index, content_block } => {
                let content = match content_block {
                    AnthropicContentBlockStart::Text { text } => {
                        vec![DeltaContent::TextDelta(TextDelta { text })]
                    }
                    AnthropicContentBlockStart::ToolUse { id, name } => {
                        vec![DeltaContent::ToolCallDelta(ToolCallDelta {
                            index,
                            id: Some(id),
                            name: Some(name),
                            arguments: None,
                        })]
                    }
                };
                chunk.choices.push(ChoiceDelta {
                    index,
                    content,
                    finish_reason: None,
                });
            }
            AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                let content = match delta {
                    AnthropicDelta::TextDelta { text } => {
                        vec![DeltaContent::TextDelta(TextDelta { text })]
                    }
                    AnthropicDelta::InputJsonDelta { partial_json } => {
                        vec![DeltaContent::ToolCallDelta(ToolCallDelta {
                            index,
                            id: None,
                            name: None,
                            arguments: Some(partial_json),
                        })]
                    }
                };
                chunk.choices.push(ChoiceDelta {
                    index,
                    content,
                    finish_reason: None,
                });
            }
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                chunk.choices.push(ChoiceDelta {
                    index: 0,
                    content: Vec::new(),
                    finish_reason: delta.stop_reason,
                });
                if let Some(u) = usage {
                    chunk.usage = Some(Usage {
                        prompt_tokens: 0,
                        completion_tokens: u.output_tokens,
                        total_tokens: u.output_tokens,
                    });
                }
            }
            _ => {}
        }

        chunk
    }
}

impl From<&StreamChunk> for AnthropicStreamEvent {
    fn from(chunk: &StreamChunk) -> Self {
        // Check for finish_reason first
        if let Some(choice) = chunk.choices.first() {
            if choice.finish_reason.is_some() {
                return AnthropicStreamEvent::MessageDelta {
                    delta: AnthropicMessageDelta {
                        stop_reason: choice.finish_reason.clone(),
                    },
                    usage: chunk.usage.as_ref().map(|u| AnthropicStreamUsage {
                        output_tokens: u.completion_tokens,
                    }),
                };
            }

            // Check for content
            for c in &choice.content {
                match c {
                    DeltaContent::TextDelta(t) => {
                        return AnthropicStreamEvent::ContentBlockDelta {
                            index: choice.index,
                            delta: AnthropicDelta::TextDelta { text: t.text.clone() },
                        };
                    }
                    DeltaContent::ToolCallDelta(tc) => {
                        if tc.id.is_some() {
                            return AnthropicStreamEvent::ContentBlockStart {
                                index: tc.index,
                                content_block: AnthropicContentBlockStart::ToolUse {
                                    id: tc.id.clone().unwrap_or_default(),
                                    name: tc.name.clone().unwrap_or_default(),
                                },
                            };
                        } else if tc.arguments.is_some() {
                            return AnthropicStreamEvent::ContentBlockDelta {
                                index: tc.index,
                                delta: AnthropicDelta::InputJsonDelta {
                                    partial_json: tc.arguments.clone().unwrap_or_default(),
                                },
                            };
                        }
                    }
                }
            }
        }

        // Default: message_start for metadata
        if !chunk.id.is_empty() && !chunk.model.is_empty() {
            return AnthropicStreamEvent::MessageStart {
                message: AnthropicMessageStart {
                    id: chunk.id.clone(),
                    model: chunk.model.clone(),
                    usage: chunk.usage.as_ref().map(|u| AnthropicUsage {
                        input_tokens: u.prompt_tokens,
                        output_tokens: u.completion_tokens,
                    }),
                },
            };
        }

        AnthropicStreamEvent::Ping
    }
}
