//! Ollama API types and conversions

use serde::{Deserialize, Serialize};
use yallm_ir::{
    ChatRequest, ChatResponse, Choice, Content, ImageContent, ImageSourceType, Message, Role,
    Source, ToolCallContent, ToolResultContent, Usage,
};

// ============================================================================
// Ollama Request Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    pub function: OllamaFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

// ============================================================================
// Ollama Response Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaResponse {
    pub model: String,
    pub message: OllamaMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
}

// ============================================================================
// Conversions: Ollama -> IR
// ============================================================================

impl From<OllamaRequest> for ChatRequest {
    fn from(req: OllamaRequest) -> Self {
        let messages = req.messages.into_iter().map(Into::into).collect();

        let (temperature, top_p) = req
            .options
            .map(|o| (o.temperature, o.top_p))
            .unwrap_or((None, None));

        ChatRequest {
            model: req.model,
            messages,
            max_tokens: None,
            temperature,
            top_p,
            stream: req.stream.unwrap_or(false),
        }
    }
}

impl From<OllamaMessage> for Message {
    fn from(msg: OllamaMessage) -> Self {
        let role = match msg.role.as_str() {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::User,
        };

        let mut content = Vec::new();

        if role == Role::Tool {
            content.push(Content::ToolResult(ToolResultContent {
                tool_call_id: String::new(),
                content: msg.content,
            }));
        } else {
            if !msg.content.is_empty() {
                content.push(Content::text(msg.content));
            }

            if let Some(images) = msg.images {
                for img in images {
                    content.push(Content::Image(ImageContent {
                        source_type: ImageSourceType::Base64,
                        media_type: String::new(),
                        data: img,
                    }));
                }
            }

            if let Some(tcs) = msg.tool_calls {
                for tc in tcs {
                    content.push(Content::ToolCall(ToolCallContent {
                        id: String::new(),
                        name: tc.function.name,
                        arguments: serde_json::to_string(&tc.function.arguments)
                            .unwrap_or_default(),
                    }));
                }
            }
        }

        Message::new(role, content).with_source(Source::Ollama)
    }
}

// ============================================================================
// Conversions: IR -> Ollama
// ============================================================================

impl From<&ChatRequest> for OllamaRequest {
    fn from(req: &ChatRequest) -> Self {
        let messages: Vec<OllamaMessage> = req.messages.iter().map(Into::into).collect();

        let options = if req.temperature.is_some() || req.top_p.is_some() {
            Some(OllamaOptions {
                temperature: req.temperature,
                top_p: req.top_p,
            })
        } else {
            None
        };

        OllamaRequest {
            model: req.model.clone(),
            messages,
            options,
            stream: if req.stream { Some(true) } else { None },
        }
    }
}

impl From<&Message> for OllamaMessage {
    fn from(msg: &Message) -> Self {
        if msg.role == Role::Tool {
            let content = msg
                .content
                .iter()
                .find_map(|c| {
                    if let Content::ToolResult(tr) = c {
                        Some(tr.content.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            return OllamaMessage {
                role: "tool".to_string(),
                content,
                images: None,
                tool_calls: None,
            };
        }

        let text: String = msg
            .content
            .iter()
            .filter_map(|c| {
                if let Content::Text(t) = c {
                    Some(t.text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let images: Vec<String> = msg
            .content
            .iter()
            .filter_map(|c| {
                if let Content::Image(i) = c {
                    Some(i.data.clone())
                } else {
                    None
                }
            })
            .collect();

        let tool_calls: Vec<OllamaToolCall> = msg
            .content
            .iter()
            .filter_map(|c| {
                if let Content::ToolCall(tc) = c {
                    let args: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or_default();
                    Some(OllamaToolCall {
                        function: OllamaFunction {
                            name: tc.name.clone(),
                            arguments: args,
                        },
                    })
                } else {
                    None
                }
            })
            .collect();

        OllamaMessage {
            role: msg.role.as_str().to_string(),
            content: text,
            images: if images.is_empty() { None } else { Some(images) },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
        }
    }
}

// ============================================================================
// Response Conversions
// ============================================================================

impl From<OllamaResponse> for ChatResponse {
    fn from(resp: OllamaResponse) -> Self {
        let msg: Message = resp.message.into();

        let usage = Usage {
            prompt_tokens: resp.prompt_eval_count.unwrap_or(0),
            completion_tokens: resp.eval_count.unwrap_or(0),
            total_tokens: resp.prompt_eval_count.unwrap_or(0) + resp.eval_count.unwrap_or(0),
        };

        ChatResponse {
            id: resp.model.clone(),
            model: resp.model,
            choices: vec![Choice {
                index: 0,
                message: msg,
                finish_reason: if resp.done { Some("stop".to_string()) } else { None },
            }],
            usage: Some(usage),
        }
    }
}

impl From<&ChatResponse> for OllamaResponse {
    fn from(resp: &ChatResponse) -> Self {
        let msg = resp
            .choices
            .first()
            .map(|c| (&c.message).into())
            .unwrap_or(OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                images: None,
                tool_calls: None,
            });

        OllamaResponse {
            model: resp.model.clone(),
            message: msg,
            done: true,
            prompt_eval_count: resp.usage.as_ref().map(|u| u.prompt_tokens),
            eval_count: resp.usage.as_ref().map(|u| u.completion_tokens),
        }
    }
}
