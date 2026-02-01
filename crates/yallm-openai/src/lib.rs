//! OpenAI API types and conversions

use yallm_ir::{
    ChatResponse, Choice, ChoiceDelta, Content, DeltaContent, ImageContent, ImageSourceType,
    Message, Role, Source, StreamChunk, TextDelta, ToolCallContent, ToolCallDelta,
    ToolResultContent, Usage,
};

yallm_macros::include_openapi! {
    url = "https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml",
    local_file = "openapi.documented.yml",
    root_types = [
        "CreateChatCompletionRequest",
        "CreateChatCompletionResponse",
        "CreateChatCompletionStreamResponse",
    ],
    extra_definitions = {
       "ChatModel": {"type": "string"}
    },
}

// ============================================================================
// Conversions: OpenAI -> IR
// ============================================================================

impl From<ChatCompletionRequestMessage> for Message {
    fn from(msg: ChatCompletionRequestMessage) -> Self {
        match msg {
            ChatCompletionRequestMessage::SystemMessage(m) => {
                let content = parse_system_content(m.content);
                Message::new(Role::System, content).with_source(Source::OpenAI)
            }
            ChatCompletionRequestMessage::DeveloperMessage(m) => {
                let content = parse_developer_content(m.content);
                Message::new(Role::System, content).with_source(Source::OpenAI)
            }
            ChatCompletionRequestMessage::UserMessage(m) => {
                let content = parse_user_content(m.content);
                Message::new(Role::User, content).with_source(Source::OpenAI)
            }
            ChatCompletionRequestMessage::AssistantMessage(m) => {
                let mut content = Vec::new();

                // Parse content
                if let Some(c) = m.content {
                    content.extend(parse_assistant_content(c));
                }

                // Parse tool_calls
                if let Some(tcs) = m.tool_calls {
                    for tc in tcs.0 {
                        if let ChatCompletionMessageToolCallsItem::ToolCall(tc) = tc {
                            content.push(Content::ToolCall(ToolCallContent {
                                id: tc.id,
                                name: tc.function.name,
                                arguments: tc.function.arguments,
                            }));
                        }
                    }
                }

                Message::new(Role::Assistant, content).with_source(Source::OpenAI)
            }
            ChatCompletionRequestMessage::ToolMessage(m) => {
                let tool_content = match m.content {
                    ChatCompletionRequestToolMessageContent::TextContent(t) => t,
                    ChatCompletionRequestToolMessageContent::ArrayOfContentParts(parts) => parts
                        .into_iter()
                        .map(|p| p.0.text)
                        .collect::<Vec<_>>()
                        .join(""),
                };
                let content = vec![Content::ToolResult(ToolResultContent {
                    tool_call_id: m.tool_call_id,
                    content: tool_content,
                })];
                Message::new(Role::Tool, content).with_source(Source::OpenAI)
            }
            ChatCompletionRequestMessage::FunctionMessage(m) => {
                let content = m
                    .content
                    .map(|text| vec![Content::text(text)])
                    .unwrap_or_default();
                Message::new(Role::User, content).with_source(Source::OpenAI)
            }
        }
    }
}

fn parse_system_content(content: ChatCompletionRequestSystemMessageContent) -> Vec<Content> {
    match content {
        ChatCompletionRequestSystemMessageContent::TextContent(t) => vec![Content::text(t)],
        ChatCompletionRequestSystemMessageContent::ArrayOfContentParts(parts) => {
            parts.into_iter().map(|p| Content::text(p.0.text)).collect()
        }
    }
}

fn parse_developer_content(content: ChatCompletionRequestDeveloperMessageContent) -> Vec<Content> {
    match content {
        ChatCompletionRequestDeveloperMessageContent::TextContent(t) => vec![Content::text(t)],
        ChatCompletionRequestDeveloperMessageContent::ArrayOfContentParts(parts) => {
            parts.into_iter().map(|p| Content::text(p.text)).collect()
        }
    }
}

fn parse_user_content(content: ChatCompletionRequestUserMessageContent) -> Vec<Content> {
    match content {
        ChatCompletionRequestUserMessageContent::TextContent(t) => vec![Content::text(t)],
        ChatCompletionRequestUserMessageContent::ArrayOfContentParts(parts) => parts
            .into_iter()
            .filter_map(|p| match p {
                ChatCompletionRequestUserMessageContentPart::Text(t) => Some(Content::text(t.text)),
                ChatCompletionRequestUserMessageContentPart::Image(img) => {
                    let url = img.image_url.url;
                    if url.starts_with("data:") {
                        let parts: Vec<&str> = url.splitn(2, ";base64,").collect();
                        let media_type = parts[0].trim_start_matches("data:");
                        let data = parts.get(1).unwrap_or(&"");
                        Some(Content::Image(ImageContent {
                            source_type: ImageSourceType::Base64,
                            media_type: media_type.to_string(),
                            data: data.to_string(),
                        }))
                    } else {
                        Some(Content::Image(ImageContent {
                            source_type: ImageSourceType::Url,
                            media_type: String::new(),
                            data: url,
                        }))
                    }
                }
                _ => None,
            })
            .collect(),
    }
}

fn parse_assistant_content(content: ChatCompletionRequestAssistantMessageContent) -> Vec<Content> {
    match content {
        ChatCompletionRequestAssistantMessageContent::TextContent(t) => vec![Content::text(t)],
        ChatCompletionRequestAssistantMessageContent::ArrayOfContentParts(parts) => parts
            .into_iter()
            .filter_map(|p| match p {
                ChatCompletionRequestAssistantMessageContentPart::Text(t) => {
                    Some(Content::text(t.text))
                }
                _ => None,
            })
            .collect(),
    }
}

// ============================================================================
// Conversions: IR -> OpenAI
// ============================================================================

impl From<&Message> for ChatCompletionRequestMessage {
    fn from(msg: &Message) -> Self {
        match msg.role {
            Role::System => {
                ChatCompletionRequestMessage::SystemMessage(ChatCompletionRequestSystemMessage {
                    content: content_to_system(msg),
                    name: None,
                    role: ChatCompletionRequestSystemMessageRole::System,
                })
            }
            Role::User => {
                ChatCompletionRequestMessage::UserMessage(ChatCompletionRequestUserMessage {
                    content: content_to_user(msg),
                    name: None,
                    role: ChatCompletionRequestUserMessageRole::User,
                })
            }
            Role::Assistant => ChatCompletionRequestMessage::AssistantMessage(
                ChatCompletionRequestAssistantMessage {
                    content: content_to_assistant(msg),
                    name: None,
                    role: ChatCompletionRequestAssistantMessageRole::Assistant,
                    tool_calls: tool_calls_from_message(msg),
                    audio: None,
                    function_call: None,
                    refusal: None,
                },
            ),
            Role::Tool => {
                let tr = msg.content.iter().find_map(|c| {
                    if let Content::ToolResult(t) = c {
                        Some(t)
                    } else {
                        None
                    }
                });
                ChatCompletionRequestMessage::ToolMessage(ChatCompletionRequestToolMessage {
                    content: ChatCompletionRequestToolMessageContent::TextContent(
                        tr.map(|t| t.content.clone()).unwrap_or_default(),
                    ),
                    role: ChatCompletionRequestToolMessageRole::Tool,
                    tool_call_id: tr.map(|t| t.tool_call_id.clone()).unwrap_or_default(),
                })
            }
        }
    }
}

fn content_to_system(msg: &Message) -> ChatCompletionRequestSystemMessageContent {
    let texts: Vec<_> = msg
        .content
        .iter()
        .filter_map(|c| {
            if let Content::Text(t) = c {
                Some(t.text.clone())
            } else {
                None
            }
        })
        .collect();

    if texts.len() == 1 {
        ChatCompletionRequestSystemMessageContent::TextContent(texts.into_iter().next().unwrap())
    } else {
        ChatCompletionRequestSystemMessageContent::TextContent(texts.join(""))
    }
}

fn content_to_user(msg: &Message) -> ChatCompletionRequestUserMessageContent {
    let has_images = msg.content.iter().any(|c| matches!(c, Content::Image(_)));

    if !has_images {
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
            .collect();
        return ChatCompletionRequestUserMessageContent::TextContent(text);
    }

    let parts: Vec<_> = msg
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(ChatCompletionRequestUserMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText {
                    text: t.text.clone(),
                    type_: ChatCompletionRequestMessageContentPartTextType::Text,
                },
            )),
            Content::Image(ic) => {
                let url = if ic.source_type == ImageSourceType::Url {
                    ic.data.clone()
                } else {
                    format!("data:{};base64,{}", ic.media_type, ic.data)
                };
                Some(ChatCompletionRequestUserMessageContentPart::Image(
                    ChatCompletionRequestMessageContentPartImage {
                        image_url: ChatCompletionRequestMessageContentPartImageImageUrl {
                            url,
                            detail:
                                ChatCompletionRequestMessageContentPartImageImageUrlDetail::Auto,
                        },
                        type_: ChatCompletionRequestMessageContentPartImageType::ImageUrl,
                    },
                ))
            }
            _ => None,
        })
        .collect();

    ChatCompletionRequestUserMessageContent::ArrayOfContentParts(parts)
}

fn content_to_assistant(msg: &Message) -> Option<ChatCompletionRequestAssistantMessageContent> {
    let texts: Vec<_> = msg
        .content
        .iter()
        .filter_map(|c| {
            if let Content::Text(t) = c {
                Some(t.text.clone())
            } else {
                None
            }
        })
        .collect();

    if texts.is_empty() {
        None
    } else if texts.len() == 1 {
        Some(ChatCompletionRequestAssistantMessageContent::TextContent(
            texts.into_iter().next().unwrap(),
        ))
    } else {
        Some(ChatCompletionRequestAssistantMessageContent::TextContent(
            texts.join(""),
        ))
    }
}

fn tool_calls_from_message(msg: &Message) -> Option<ChatCompletionMessageToolCalls> {
    let tool_calls: Vec<_> = msg
        .content
        .iter()
        .filter_map(|c| {
            if let Content::ToolCall(tc) = c {
                Some(ChatCompletionMessageToolCallsItem::ToolCall(
                    ChatCompletionMessageToolCall {
                        id: tc.id.clone(),
                        type_: ChatCompletionMessageToolCallType::Function,
                        function: ChatCompletionMessageToolCallFunction {
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        },
                    },
                ))
            } else {
                None
            }
        })
        .collect();

    if tool_calls.is_empty() {
        None
    } else {
        Some(ChatCompletionMessageToolCalls(tool_calls))
    }
}

// ============================================================================
// Response Conversions
// ============================================================================

impl From<CreateChatCompletionResponse> for ChatResponse {
    fn from(resp: CreateChatCompletionResponse) -> Self {
        let choices = resp
            .choices
            .into_iter()
            .map(|c| Choice {
                index: c.index as u32,
                message: response_message_to_ir(c.message),
                finish_reason: Some(c.finish_reason.to_string()),
            })
            .collect();

        ChatResponse {
            id: resp.id,
            model: resp.model,
            choices,
            usage: resp.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens as u32,
                completion_tokens: u.completion_tokens as u32,
                total_tokens: u.total_tokens as u32,
            }),
        }
    }
}

fn response_message_to_ir(msg: ChatCompletionResponseMessage) -> Message {
    let mut content = Vec::new();

    if let Some(text) = msg.content
        && !text.is_empty() {
            content.push(Content::text(text));
        }

    if let Some(tcs) = msg.tool_calls {
        for tc in tcs.0 {
            if let ChatCompletionMessageToolCallsItem::ToolCall(tc) = tc {
                content.push(Content::ToolCall(ToolCallContent {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                }));
            }
        }
    }

    Message::new(Role::Assistant, content).with_source(Source::OpenAI)
}

// ============================================================================
// Stream Conversions
// ============================================================================

impl From<CreateChatCompletionStreamResponse> for StreamChunk {
    fn from(chunk: CreateChatCompletionStreamResponse) -> Self {
        let choices = chunk
            .choices
            .into_iter()
            .map(|c| {
                let mut content = Vec::new();

                if let Some(text) = c.delta.content {
                    content.push(DeltaContent::TextDelta(TextDelta { text }));
                }

                for tc in c.delta.tool_calls {
                    content.push(DeltaContent::ToolCallDelta(ToolCallDelta {
                        index: tc.index as u32,
                        id: tc.id,
                        name: tc.function.as_ref().and_then(|f| f.name.clone()),
                        arguments: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                    }));
                }

                ChoiceDelta {
                    index: c.index as u32,
                    content,
                    finish_reason: c.finish_reason.map(|r| r.to_string()),
                }
            })
            .collect();

        StreamChunk {
            id: chunk.id,
            model: chunk.model,
            choices,
            usage: chunk.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens as u32,
                completion_tokens: u.completion_tokens as u32,
                total_tokens: u.total_tokens as u32,
            }),
            source: Some(Source::OpenAI),
            raw: None,
        }
    }
}
