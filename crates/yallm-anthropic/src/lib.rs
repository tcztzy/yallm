//! Anthropic API types and conversions
//!
//! Types are generated from the official Anthropic OpenAPI specification.
//! The spec is automatically fetched from the Anthropic TypeScript SDK repository.

// Generate types from Anthropic OpenAPI spec
yallm_macros::include_openapi! {
    local_file = "openapi.yml",
    root_types = [
        "CreateMessageParams",
        "Message",
        "MessageStreamEvent",
    ],
    extra_definitions = {
        "Model": {"type": "string"}
    },
}

// Re-export key types with Anthropic prefix for clarity
pub use CreateMessageParams as AnthropicRequest;
pub use Message as AnthropicResponse;
pub use MessageStreamEvent as AnthropicStreamEvent;

// ============================================================================
// IR Type Imports (renamed to avoid conflicts with generated types)
// ============================================================================

use yallm_ir::{
    ChatRequest, ChatResponse, Choice, ChoiceDelta, Content as IrContent, DeltaContent,
    Message as IrMessage, Role as IrRole, Source as IrSource, StreamChunk, TextDelta,
    ToolCallContent, ToolCallDelta, Usage as IrUsage,
};

// ============================================================================
// Conversions: Anthropic -> IR
// ============================================================================

impl From<CreateMessageParams> for ChatRequest {
    fn from(req: CreateMessageParams) -> Self {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(ref system) = req.system {
            let system_text = match system {
                System::String(s) => s.clone(),
                System::Array(blocks) => blocks
                    .iter()
                    .map(|b| b.text.to_string())
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            if !system_text.is_empty() {
                messages.push(IrMessage::text(IrRole::System, system_text));
            }
        }

        // Convert messages - simplified, just extract text content
        for msg in req.messages {
            let role = match msg.role {
                Role::User => IrRole::User,
                Role::Assistant => IrRole::Assistant,
            };

            let text = match msg.content {
                Content::String(s) => s,
                Content::ContentBlockSourceContent(_) => {
                    // TODO: Handle complex content blocks
                    String::new()
                }
            };

            if !text.is_empty() {
                messages.push(IrMessage::text(role, text).with_source(IrSource::Anthropic));
            }
        }

        ChatRequest {
            model: req.model.to_string(),
            messages,
            max_tokens: Some(req.max_tokens.get() as u32),
            temperature: req.temperature.map(|t| t as f32),
            top_p: req.top_p.map(|t| t as f32),
            stream: req.stream.unwrap_or(false),
        }
    }
}

// ============================================================================
// Response Conversions
// ============================================================================

impl From<Message> for ChatResponse {
    fn from(resp: Message) -> Self {
        let content: Vec<IrContent> = resp
            .content
            .into_iter()
            .filter_map(|b| match b {
                ContentBlock::TextBlock(t) => Some(IrContent::text(t.text.to_string())),
                ContentBlock::ToolUseBlock(tu) => Some(IrContent::ToolCall(ToolCallContent {
                    id: tu.id.to_string(),
                    name: tu.name.to_string(),
                    arguments: serde_json::to_string(&tu.input).unwrap_or_default(),
                })),
                _ => None,
            })
            .collect();

        let usage = IrUsage {
            prompt_tokens: resp.usage.input_tokens as u32,
            completion_tokens: resp.usage.output_tokens as u32,
            total_tokens: (resp.usage.input_tokens + resp.usage.output_tokens) as u32,
        };

        ChatResponse {
            id: resp.id.to_string(),
            model: resp.model.to_string(),
            choices: vec![Choice {
                index: 0,
                message: IrMessage::new(IrRole::Assistant, content)
                    .with_source(IrSource::Anthropic),
                finish_reason: resp.stop_reason.map(|r| r.to_string()),
            }],
            usage: Some(usage),
        }
    }
}

// ============================================================================
// Stream Conversions
// ============================================================================

impl From<MessageStreamEvent> for StreamChunk {
    fn from(event: MessageStreamEvent) -> Self {
        let mut chunk = StreamChunk {
            id: String::new(),
            model: String::new(),
            choices: Vec::new(),
            usage: None,
            source: Some(IrSource::Anthropic),
            raw: None,
        };

        match event {
            MessageStreamEvent::MessageStartEvent(e) => {
                chunk.id = e.message.id.to_string();
                chunk.model = e.message.model.to_string();
                chunk.usage = Some(IrUsage {
                    prompt_tokens: e.message.usage.input_tokens as u32,
                    completion_tokens: e.message.usage.output_tokens as u32,
                    total_tokens: (e.message.usage.input_tokens + e.message.usage.output_tokens)
                        as u32,
                });
            }
            MessageStreamEvent::ContentBlockStartEvent(e) => {
                let content = match e.content_block {
                    ContentBlock::TextBlock(t) => {
                        vec![DeltaContent::TextDelta(TextDelta {
                            text: t.text.to_string(),
                        })]
                    }
                    ContentBlock::ToolUseBlock(tu) => {
                        vec![DeltaContent::ToolCallDelta(ToolCallDelta {
                            index: e.index as u32,
                            id: Some(tu.id.to_string()),
                            name: Some(tu.name.to_string()),
                            arguments: None,
                        })]
                    }
                    _ => Vec::new(),
                };
                if !content.is_empty() {
                    chunk.choices.push(ChoiceDelta {
                        index: e.index as u32,
                        content,
                        finish_reason: None,
                    });
                }
            }
            MessageStreamEvent::ContentBlockDeltaEvent(e) => {
                let content = match e.delta {
                    Delta::TextContentBlockDelta(d) => {
                        vec![DeltaContent::TextDelta(TextDelta {
                            text: d.text.to_string(),
                        })]
                    }
                    Delta::InputJsonContentBlockDelta(d) => {
                        vec![DeltaContent::ToolCallDelta(ToolCallDelta {
                            index: e.index as u32,
                            id: None,
                            name: None,
                            arguments: Some(d.partial_json.to_string()),
                        })]
                    }
                    _ => Vec::new(),
                };
                if !content.is_empty() {
                    chunk.choices.push(ChoiceDelta {
                        index: e.index as u32,
                        content,
                        finish_reason: None,
                    });
                }
            }
            MessageStreamEvent::MessageDeltaEvent(e) => {
                chunk.choices.push(ChoiceDelta {
                    index: 0,
                    content: Vec::new(),
                    finish_reason: e.delta.stop_reason.map(|r| r.to_string()),
                });
                chunk.usage = Some(IrUsage {
                    prompt_tokens: 0,
                    completion_tokens: e.usage.output_tokens as u32,
                    total_tokens: e.usage.output_tokens as u32,
                });
            }
            MessageStreamEvent::ContentBlockStopEvent(_)
            | MessageStreamEvent::MessageStopEvent(_) => {}
        }

        chunk
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Serialization/Deserialization Tests
    // ========================================================================

    #[test]
    fn test_deserialize_create_message_params() {
        let json = r#"{
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, Claude!"}
            ]
        }"#;

        let params: CreateMessageParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.model.to_string(), "claude-sonnet-4-20250514");
        assert_eq!(params.max_tokens.get(), 1024);
        assert_eq!(params.messages.len(), 1);
    }

    #[test]
    fn test_deserialize_create_message_params_with_system() {
        let json = r#"{
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }"#;

        let params: CreateMessageParams = serde_json::from_str(json).unwrap();
        assert!(params.system.is_some());
        match params.system.unwrap() {
            System::String(s) => assert_eq!(s, "You are a helpful assistant."),
            _ => panic!("Expected System::String"),
        }
    }

    #[test]
    fn test_deserialize_message_response() {
        let json = r#"{
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello! How can I help you today?"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 25
            }
        }"#;

        let message: Message = serde_json::from_str(json).unwrap();
        assert_eq!(message.id.to_string(), "msg_01XFDUDYJgAACzvnptvVoYEL");
        assert_eq!(message.model.to_string(), "claude-sonnet-4-20250514");
        assert_eq!(message.content.len(), 1);
        assert_eq!(message.usage.input_tokens, 10);
        assert_eq!(message.usage.output_tokens, 25);
    }

    #[test]
    fn test_deserialize_message_with_tool_use() {
        let json = r#"{
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lgs0",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"}
                }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 50,
                "output_tokens": 100
            }
        }"#;

        let message: Message = serde_json::from_str(json).unwrap();
        assert_eq!(message.content.len(), 1);
        match &message.content[0] {
            ContentBlock::ToolUseBlock(tu) => {
                assert_eq!(tu.name.to_string(), "get_weather");
            }
            _ => panic!("Expected ToolUseBlock"),
        }
    }

    // ========================================================================
    // Conversion Tests: Anthropic -> IR
    // ========================================================================

    #[test]
    fn test_convert_create_message_params_to_chat_request() {
        let json = r#"{
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, Claude!"}
            ]
        }"#;

        let params: CreateMessageParams = serde_json::from_str(json).unwrap();
        let chat_request: ChatRequest = params.into();

        assert_eq!(chat_request.model, "claude-sonnet-4-20250514");
        assert_eq!(chat_request.max_tokens, Some(1024));
        assert_eq!(chat_request.messages.len(), 1);
        assert_eq!(chat_request.messages[0].role, IrRole::User);
    }

    #[test]
    fn test_convert_with_system_message() {
        let json = r#"{
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }"#;

        let params: CreateMessageParams = serde_json::from_str(json).unwrap();
        let chat_request: ChatRequest = params.into();

        assert_eq!(chat_request.messages.len(), 2);
        assert_eq!(chat_request.messages[0].role, IrRole::System);
        assert_eq!(chat_request.messages[1].role, IrRole::User);
    }

    #[test]
    fn test_convert_message_to_chat_response() {
        let json = r#"{
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello! How can I help you?"}
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 25
            }
        }"#;

        let message: Message = serde_json::from_str(json).unwrap();
        let chat_response: ChatResponse = message.into();

        assert_eq!(chat_response.id, "msg_01XFDUDYJgAACzvnptvVoYEL");
        assert_eq!(chat_response.model, "claude-sonnet-4-20250514");
        assert_eq!(chat_response.choices.len(), 1);
        assert_eq!(chat_response.choices[0].message.role, IrRole::Assistant);
        assert_eq!(
            chat_response.choices[0].finish_reason,
            Some("end_turn".to_string())
        );

        let usage = chat_response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 25);
        assert_eq!(usage.total_tokens, 35);
    }

    #[test]
    fn test_convert_tool_use_response() {
        let json = r#"{
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lgs0",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"}
                }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 50,
                "output_tokens": 100
            }
        }"#;

        let message: Message = serde_json::from_str(json).unwrap();
        let chat_response: ChatResponse = message.into();

        assert_eq!(chat_response.choices[0].message.content.len(), 1);
        match &chat_response.choices[0].message.content[0] {
            IrContent::ToolCall(tc) => {
                assert_eq!(tc.name, "get_weather");
                assert!(tc.arguments.contains("San Francisco"));
            }
            _ => panic!("Expected ToolCall content"),
        }
    }

    // ========================================================================
    // Streaming Event Tests
    // ========================================================================

    #[test]
    fn test_deserialize_message_start_event() {
        let json = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-20250514",
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 0
                }
            }
        }"#;

        let event: MessageStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            MessageStreamEvent::MessageStartEvent(e) => {
                assert_eq!(e.message.id.to_string(), "msg_01XFDUDYJgAACzvnptvVoYEL");
            }
            _ => panic!("Expected MessageStartEvent"),
        }
    }

    #[test]
    fn test_deserialize_content_block_delta_event() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello"
            }
        }"#;

        let event: MessageStreamEvent = serde_json::from_str(json).unwrap();
        match event {
            MessageStreamEvent::ContentBlockDeltaEvent(e) => {
                assert_eq!(e.index, 0);
                match e.delta {
                    Delta::TextContentBlockDelta(d) => {
                        assert_eq!(d.text.to_string(), "Hello");
                    }
                    _ => panic!("Expected TextContentBlockDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDeltaEvent"),
        }
    }

    #[test]
    fn test_convert_stream_event_to_chunk() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello"
            }
        }"#;

        let event: MessageStreamEvent = serde_json::from_str(json).unwrap();
        let chunk: StreamChunk = event.into();

        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].index, 0);
        match &chunk.choices[0].content[0] {
            DeltaContent::TextDelta(td) => {
                assert_eq!(td.text, "Hello");
            }
            _ => panic!("Expected TextDelta"),
        }
    }

    // ========================================================================
    // Round-trip Serialization Tests
    // ========================================================================

    #[test]
    fn test_serialize_create_message_params() {
        let json = r#"{
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }"#;

        let params: CreateMessageParams = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&params).unwrap();
        let deserialized: CreateMessageParams = serde_json::from_str(&serialized).unwrap();

        assert_eq!(params.model.to_string(), deserialized.model.to_string());
        assert_eq!(params.max_tokens, deserialized.max_tokens);
    }
}
