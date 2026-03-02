//! OpenAI Responses/Conversations API support and IR conversion helpers.

use serde_json::{Value, json};
use yallm_ir::{
    ChatRequest as IrChatRequest, ChatResponse as IrChatResponse, Content as IrContent,
    Message as IrMessage, Role as IrRole,
};

yallm_macros::include_openapi! {
    local_file = "../yallm-openai/openapi.documented.yml",
    root_types = [
        "CreateResponse",
        "Response",
        "ResponseStreamEvent",
        "ResponseItemList",
        "TokenCountsBody",
        "TokenCountsResource",
        "CompactResponseMethodPublicBody",
        "CompactResource",
        "CreateConversationBody",
        "UpdateConversationBody",
        "ConversationResource",
        "ConversationItem",
        "ConversationItemList",
        "DeletedConversationResource",
    ],
}

pub fn extract_previous_response_id(req: &Value) -> Option<String> {
    req.get("previous_response_id")
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

pub fn extract_conversation_id(req: &Value) -> Option<String> {
    let conv = req.get("conversation")?;
    match conv {
        Value::String(s) if !s.is_empty() && s != "auto" && s != "none" => Some(s.clone()),
        Value::Object(obj) => obj
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        _ => None,
    }
}

pub fn parse_input_items(req: &Value) -> Vec<Value> {
    let Some(input) = req.get("input") else {
        return Vec::new();
    };
    match input {
        Value::String(text) => vec![json!({
            "type": "message",
            "role": "user",
            "content": [{"type":"input_text","text": text}]
        })],
        Value::Array(items) => items
            .iter()
            .cloned()
            .map(normalize_input_item)
            .collect::<Vec<_>>(),
        Value::Null => Vec::new(),
        other => vec![normalize_input_item(other.clone())],
    }
}

pub fn create_response_to_ir(req: &Value, history_items: &[Value]) -> Option<IrChatRequest> {
    let model = req.get("model")?.as_str()?.to_string();
    let mut messages = Vec::new();

    for item in history_items {
        if let Some(msg) = item_to_ir_message(item) {
            messages.push(msg);
        }
    }
    for item in parse_input_items(req) {
        if let Some(msg) = item_to_ir_message(&item) {
            messages.push(msg);
        }
    }

    if let Some(instructions) = req.get("instructions")
        && let Some(text) = extract_text_content(instructions)
        && !text.is_empty()
    {
        messages.insert(0, IrMessage::text(IrRole::System, text));
    }

    let max_tokens = req
        .get("max_output_tokens")
        .and_then(Value::as_u64)
        .map(|n| n as u32);
    let temperature = req
        .get("temperature")
        .and_then(Value::as_f64)
        .map(|n| n as f32);
    let top_p = req.get("top_p").and_then(Value::as_f64).map(|n| n as f32);
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);

    Some(IrChatRequest {
        model,
        messages,
        max_tokens,
        temperature,
        top_p,
        stream,
    })
}

pub fn ir_to_response(
    ir_resp: &IrChatResponse,
    req: &Value,
    response_id: &str,
    created_at: u64,
    completed_at: Option<u64>,
    conversation_id: Option<&str>,
    previous_response_id: Option<&str>,
) -> Value {
    let output_item_id = format!("msg_{response_id}");
    let text = ir_resp
        .choices
        .first()
        .map(|c| join_text(&c.message.content))
        .unwrap_or_default();

    json!({
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "completed_at": completed_at.unwrap_or(created_at),
        "error": Value::Null,
        "incomplete_details": Value::Null,
        "instructions": req.get("instructions").cloned().unwrap_or(Value::Null),
        "max_output_tokens": req.get("max_output_tokens").cloned().unwrap_or(Value::Null),
        "model": ir_resp.model,
        "output": [{
            "id": output_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": []
            }]
        }],
        "parallel_tool_calls": req.get("parallel_tool_calls").cloned().unwrap_or(Value::Bool(true)),
        "previous_response_id": previous_response_id.map(|s| Value::String(s.to_string())).unwrap_or(Value::Null),
        "reasoning": req.get("reasoning").cloned().unwrap_or_else(|| json!({"effort": Value::Null, "summary": Value::Null})),
        "store": req.get("store").cloned().unwrap_or(Value::Bool(true)),
        "temperature": req.get("temperature").cloned().unwrap_or_else(|| json!(1.0)),
        "text": req.get("text").cloned().unwrap_or_else(|| json!({"format":{"type":"text"}})),
        "tool_choice": req.get("tool_choice").cloned().unwrap_or_else(|| Value::String("auto".to_string())),
        "tools": req.get("tools").cloned().unwrap_or_else(|| Value::Array(Vec::new())),
        "top_p": req.get("top_p").cloned().unwrap_or_else(|| json!(1.0)),
        "truncation": req.get("truncation").cloned().unwrap_or_else(|| Value::String("disabled".to_string())),
        "usage": {
            "input_tokens": ir_resp.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
            "input_tokens_details": { "cached_tokens": 0 },
            "output_tokens": ir_resp.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": ir_resp.usage.as_ref().map(|u| u.total_tokens).unwrap_or(0),
        },
        "user": req.get("user").cloned().unwrap_or(Value::Null),
        "metadata": req.get("metadata").cloned().unwrap_or_else(|| json!({})),
        "conversation": conversation_id.map(|id| json!({"id": id})).unwrap_or(Value::Null),
    })
}

pub fn response_to_stream_events(response: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let created = json!({"type":"response.created","response": response});
    out.push(format!("event: response.created\ndata: {created}\n\n"));

    if let Some(text) = response
        .get("output")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("content"))
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(|content| content.get("text"))
        .and_then(Value::as_str)
    {
        let item_id = response
            .get("output")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("id"))
            .and_then(Value::as_str)
            .unwrap_or("msg_0");
        let delta = json!({
            "type":"response.output_text.delta",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text
        });
        out.push(format!(
            "event: response.output_text.delta\ndata: {delta}\n\n"
        ));

        let done = json!({
            "type":"response.output_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": text
        });
        out.push(format!(
            "event: response.output_text.done\ndata: {done}\n\n"
        ));
    }

    let completed = json!({"type":"response.completed","response": response});
    out.push(format!("event: response.completed\ndata: {completed}\n\n"));
    out.push("data: [DONE]\n\n".to_string());
    out
}

pub fn estimate_input_tokens(req: &Value) -> Value {
    let text = req
        .get("input")
        .and_then(extract_text_content)
        .unwrap_or_default();
    let chars = text.chars().count() as u64;
    let est = (chars / 4).max(1);
    json!({
        "object": "response.input_tokens",
        "input_tokens": est
    })
}

pub fn fallback_compact(req: &Value) -> Value {
    let created_at = unix_seconds();
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown-model");
    json!({
        "id": format!("cmp_{created_at}"),
        "object": "response.compaction",
        "created_at": created_at,
        "output": [{
            "type": "compaction",
            "id": format!("cmpitem_{created_at}"),
            "encrypted_content": format!("local-compaction:{model}")
        }],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    })
}

fn normalize_input_item(item: Value) -> Value {
    match item {
        Value::Object(mut obj) => {
            if !obj.contains_key("type") {
                obj.insert("type".to_string(), Value::String("message".to_string()));
            }
            if !obj.contains_key("role") {
                obj.insert("role".to_string(), Value::String("user".to_string()));
            }
            if !obj.contains_key("content") {
                obj.insert(
                    "content".to_string(),
                    Value::Array(vec![json!({"type":"input_text","text":""})]),
                );
            }
            Value::Object(obj)
        }
        Value::String(text) => json!({
            "type":"message",
            "role":"user",
            "content":[{"type":"input_text","text": text}]
        }),
        other => json!({
            "type":"message",
            "role":"user",
            "content":[{"type":"input_text","text": scalar_to_text(&other)}]
        }),
    }
}

fn item_to_ir_message(item: &Value) -> Option<IrMessage> {
    if item
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message")
        != "message"
    {
        return None;
    }
    let role = match item.get("role").and_then(Value::as_str).unwrap_or("user") {
        "system" | "developer" => IrRole::System,
        "assistant" => IrRole::Assistant,
        "tool" => IrRole::Tool,
        _ => IrRole::User,
    };
    let text = item
        .get("content")
        .and_then(extract_text_content)
        .unwrap_or_default();

    if text.is_empty() {
        Some(IrMessage::new(role, Vec::new()))
    } else {
        Some(IrMessage::text(role, text))
    }
}

fn extract_text_content(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|p| match p {
                    Value::String(s) => Some(s.clone()),
                    Value::Object(obj) => {
                        if let Some(text) = obj.get("text").and_then(Value::as_str) {
                            Some(text.to_string())
                        } else {
                            obj.get("content").and_then(extract_text_content)
                        }
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(text)
        }
        Value::Object(obj) => {
            if let Some(text) = obj.get("text").and_then(Value::as_str) {
                return Some(text.to_string());
            }
            obj.get("content").and_then(extract_text_content)
        }
        _ => None,
    }
}

fn join_text(content: &[IrContent]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            IrContent::Text(t) => Some(t.text.clone()),
            IrContent::ToolResult(t) => Some(t.content.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn scalar_to_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => v.clone(),
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn unix_seconds() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_string_input_to_item() {
        let req = json!({"input":"hello"});
        let items = parse_input_items(&req);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "message");
    }

    #[test]
    fn creates_ir_with_history() {
        let req = json!({"model":"gpt-4o-mini","input":"next"});
        let history = vec![json!({
            "type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]
        })];
        let ir = create_response_to_ir(&req, &history).expect("ir");
        assert_eq!(ir.messages.len(), 2);
    }

    #[test]
    fn builds_stream_events() {
        let resp = json!({
            "id":"resp_1",
            "output":[{
                "id":"msg_1",
                "type":"message",
                "content":[{"type":"output_text","text":"hello"}]
            }]
        });
        let events = response_to_stream_events(&resp);
        assert!(events.iter().any(|e| e.contains("response.created")));
        assert!(events.last().unwrap().contains("[DONE]"));
    }
}
