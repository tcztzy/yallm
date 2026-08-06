//! Cross-provider stream translation.
//!
//! [`map_provider_stream_to_downstream`] is the single entry point: it takes
//! a raw upstream byte stream (from [`super::call_provider_stream`]), parses
//! it into provider-neutral [`ProviderStreamEvent`]s, and renders those into
//! the downstream protocol's wire format (OpenAI SSE, Anthropic SSE, or
//! Ollama line JSON).
//!
//! Pipeline: `upstream bytes` → `ProviderStreamParser` (protocol-specific:
//! SSE decoder for OpenAI/Anthropic, newline JSON for Ollama/ACP) →
//! `ProviderStreamEvent` (start / text / reasoning / tool deltas / stop) →
//! `DownstreamRenderer` → framed bytes → `DownstreamByteStream`.
//!
//! Invariants:
//! - Every provider parser must emit a `Stop` event at end of stream —
//!   renderers close with `[DONE]` only after it. The `finish()` path
//!   synthesizes a stop if the upstream ended without one.
//! - Tool calls are rebuilt from `ToolStart` + `ToolArgsDelta` pairs keyed by
//!   tool index; a stray `ToolArgsDelta` still creates a tool entry (with
//!   empty id/name), it is not dropped.
//! - Event types unsupported by the downstream protocol (e.g. reasoning on
//!   Ollama) are silently filtered, not errored — translation is lossy by
//!   design to keep streams alive.
//!
//! Gotcha: parsing and rendering run in a spawned task; upstream errors are
//! logged and the stream is cut (`[DONE]` or EOF) rather than propagated —
//! callers must not rely on error delivery mid-stream.

use std::collections::BTreeMap;

use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::{
    DownstreamByteStream, DownstreamProtocol, Provider, TransportByteStream,
    json_to_compact_string, sse_data_frame, sse_event_frame, unix_seconds,
};

/// Translate an upstream stream to the downstream wire protocol (see module
/// docs for the pipeline and its invariants).
pub fn map_provider_stream_to_downstream(
    provider: Provider,
    downstream: DownstreamProtocol,
    upstream: TransportByteStream,
    model: String,
) -> DownstreamByteStream {
    let (tx, rx) = mpsc::channel::<Bytes>(64);

    tokio::spawn(async move {
        let mut parser = ProviderStreamParser::new(provider);
        let mut renderer = DownstreamRenderer::new(downstream, model);
        let mut upstream = upstream;

        while let Some(chunk) = upstream.next().await {
            let chunk = match chunk {
                Ok(bytes) => bytes,
                Err(e) => {
                    tracing::warn!(
                        event = "provider.stream.error",
                        provider = provider.as_str(),
                        message = %e.message
                    );
                    break;
                }
            };

            for event in parser.push(&chunk) {
                renderer.push_event(event, &tx).await;
            }
        }

        for event in parser.finish() {
            renderer.push_event(event, &tx).await;
        }
        renderer.finish(&tx).await;
    });

    Box::pin(futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|bytes| (Ok(bytes), rx))
    }))
}

#[derive(Debug)]
enum ProviderStreamEvent {
    Start {
        id: Option<String>,
        model: Option<String>,
        prompt_tokens: Option<u32>,
    },
    ReasoningDelta(String),
    TextDelta(String),
    ToolStart {
        index: u32,
        id: Option<String>,
        name: Option<String>,
    },
    ToolArgsDelta {
        index: u32,
        arguments: String,
    },
    Stop {
        finish_reason: Option<String>,
        prompt_tokens: Option<u32>,
        completion_tokens: Option<u32>,
    },
}

enum ProviderStreamParser {
    OpenAI(SseDecoder),
    Anthropic(SseDecoder),
    Ollama(LineDecoder),
    Acp(LineDecoder),
}

impl ProviderStreamParser {
    fn new(provider: Provider) -> Self {
        match provider {
            Provider::OpenAI => Self::OpenAI(SseDecoder::default()),
            Provider::Anthropic => Self::Anthropic(SseDecoder::default()),
            Provider::Ollama => Self::Ollama(LineDecoder::default()),
            Provider::Acp => Self::Acp(LineDecoder::default()),
        }
    }

    fn push(&mut self, bytes: &[u8]) -> Vec<ProviderStreamEvent> {
        match self {
            ProviderStreamParser::OpenAI(sse) => sse
                .push(bytes)
                .into_iter()
                .flat_map(|(_, data)| parse_openai_sse_record(&data))
                .collect(),
            ProviderStreamParser::Anthropic(sse) => sse
                .push(bytes)
                .into_iter()
                .flat_map(|(event, data)| parse_anthropic_sse_record(event.as_deref(), &data))
                .collect(),
            ProviderStreamParser::Ollama(lines) => lines
                .push(bytes)
                .into_iter()
                .flat_map(|line| parse_ollama_line(&line))
                .collect(),
            ProviderStreamParser::Acp(lines) => lines
                .push(bytes)
                .into_iter()
                .flat_map(|line| parse_acp_line(&line))
                .collect(),
        }
    }

    fn finish(&mut self) -> Vec<ProviderStreamEvent> {
        match self {
            ProviderStreamParser::OpenAI(sse) => sse
                .finish()
                .into_iter()
                .flat_map(|(_, data)| parse_openai_sse_record(&data))
                .collect(),
            ProviderStreamParser::Anthropic(sse) => sse
                .finish()
                .into_iter()
                .flat_map(|(event, data)| parse_anthropic_sse_record(event.as_deref(), &data))
                .collect(),
            ProviderStreamParser::Ollama(lines) => lines
                .finish()
                .into_iter()
                .flat_map(|line| parse_ollama_line(&line))
                .collect(),
            ProviderStreamParser::Acp(lines) => lines
                .finish()
                .into_iter()
                .flat_map(|line| parse_acp_line(&line))
                .collect(),
        }
    }
}

#[derive(Default)]
struct LineDecoder {
    buf: Vec<u8>,
}

impl LineDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();

        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let mut line = self.buf.drain(..=pos).collect::<Vec<_>>();
            if line.last().is_some_and(|b| *b == b'\n') {
                line.pop();
            }
            if line.last().is_some_and(|b| *b == b'\r') {
                line.pop();
            }
            out.push(String::from_utf8_lossy(&line).to_string());
        }
        out
    }

    fn finish(&mut self) -> Vec<String> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        let line = String::from_utf8_lossy(&self.buf).to_string();
        self.buf.clear();
        vec![line]
    }
}

#[derive(Default)]
struct SseDecoder {
    lines: LineDecoder,
    event: Option<String>,
    data_lines: Vec<String>,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<(Option<String>, String)> {
        let mut out = Vec::new();
        for line in self.lines.push(bytes) {
            self.push_line(line, &mut out);
        }
        out
    }

    fn finish(&mut self) -> Vec<(Option<String>, String)> {
        let mut out = Vec::new();
        for line in self.lines.finish() {
            self.push_line(line, &mut out);
        }
        if !self.data_lines.is_empty() {
            out.push((self.event.take(), self.data_lines.join("\n")));
            self.data_lines.clear();
        }
        out
    }

    fn push_line(&mut self, line: String, out: &mut Vec<(Option<String>, String)>) {
        if line.is_empty() {
            if !self.data_lines.is_empty() {
                out.push((self.event.take(), self.data_lines.join("\n")));
                self.data_lines.clear();
            }
            return;
        }
        if let Some(value) = line.strip_prefix("event:") {
            self.event = Some(value.trim_start().to_string());
            return;
        }
        if let Some(value) = line.strip_prefix("data:") {
            self.data_lines.push(value.trim_start().to_string());
        }
    }
}

fn parse_openai_sse_record(data: &str) -> Vec<ProviderStreamEvent> {
    if data.trim() == "[DONE]" {
        return vec![ProviderStreamEvent::Stop {
            finish_reason: None,
            prompt_tokens: None,
            completion_tokens: None,
        }];
    }

    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let id = v.get("id").and_then(|x| x.as_str()).map(str::to_string);
    let model = v.get("model").and_then(|x| x.as_str()).map(str::to_string);
    if id.is_some() || model.is_some() {
        out.push(ProviderStreamEvent::Start {
            id,
            model,
            prompt_tokens: None,
        });
    }

    let choice = v
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first());

    if let Some(text) = choice
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(|t| t.as_str())
        .filter(|s| !s.is_empty())
    {
        out.push(ProviderStreamEvent::TextDelta(text.to_string()));
    }

    if let Some(reasoning) = choice
        .and_then(|c| c.get("delta"))
        .map(openai_delta_reasoning_text)
        .filter(|s| !s.is_empty())
    {
        out.push(ProviderStreamEvent::ReasoningDelta(reasoning));
    }

    if let Some(tool_calls) = choice
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("tool_calls"))
        .and_then(|tc| tc.as_array())
    {
        for tc in tool_calls {
            let index = tc.get("index").and_then(as_u32).unwrap_or(0);
            let id = tc.get("id").and_then(|x| x.as_str()).map(str::to_string);
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|x| x.as_str())
                .map(str::to_string);
            if id.is_some() || name.is_some() {
                out.push(ProviderStreamEvent::ToolStart { index, id, name });
            }

            if let Some(arguments) = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|x| x.as_str())
                .filter(|s| !s.is_empty())
            {
                out.push(ProviderStreamEvent::ToolArgsDelta {
                    index,
                    arguments: arguments.to_string(),
                });
            }
        }
    }

    if let Some(reason) = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|r| r.as_str())
    {
        out.push(ProviderStreamEvent::Stop {
            finish_reason: Some(reason.to_string()),
            prompt_tokens: None,
            completion_tokens: v
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(as_u32),
        });
    }
    out
}

fn openai_delta_reasoning_text(delta: &Value) -> String {
    ["reasoning_content", "reasoning"]
        .into_iter()
        .find_map(|key| {
            delta
                .get(key)
                .and_then(|value| value.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

fn parse_anthropic_sse_record(event: Option<&str>, data: &str) -> Vec<ProviderStreamEvent> {
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        return Vec::new();
    };
    let event = event
        .or_else(|| v.get("type").and_then(|t| t.as_str()))
        .unwrap_or_default();

    match event {
        "message_start" => vec![ProviderStreamEvent::Start {
            id: v
                .get("message")
                .and_then(|m| m.get("id"))
                .and_then(|x| x.as_str())
                .map(str::to_string),
            model: v
                .get("message")
                .and_then(|m| m.get("model"))
                .and_then(|x| x.as_str())
                .map(str::to_string),
            prompt_tokens: v
                .get("message")
                .and_then(|m| m.get("usage"))
                .and_then(|u| u.get("input_tokens"))
                .and_then(as_u32),
        }],
        "content_block_start" => {
            let index = v.get("index").and_then(as_u32).unwrap_or(0);
            match v
                .get("content_block")
                .and_then(|cb| cb.get("type"))
                .and_then(|t| t.as_str())
            {
                Some("tool_use") => {
                    let id = v
                        .get("content_block")
                        .and_then(|cb| cb.get("id"))
                        .and_then(|x| x.as_str())
                        .map(str::to_string);
                    let name = v
                        .get("content_block")
                        .and_then(|cb| cb.get("name"))
                        .and_then(|x| x.as_str())
                        .map(str::to_string);
                    let mut out = vec![ProviderStreamEvent::ToolStart { index, id, name }];
                    let input = v
                        .get("content_block")
                        .and_then(|cb| cb.get("input"))
                        .cloned()
                        .unwrap_or(Value::Null);
                    if input != Value::Null {
                        let encoded = json_to_compact_string(&input);
                        if encoded != "{}" {
                            out.push(ProviderStreamEvent::ToolArgsDelta {
                                index,
                                arguments: encoded,
                            });
                        }
                    }
                    out
                }
                Some("text") => v
                    .get("content_block")
                    .and_then(|cb| cb.get("text"))
                    .and_then(|t| t.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| vec![ProviderStreamEvent::TextDelta(s.to_string())])
                    .unwrap_or_default(),
                _ => Vec::new(),
            }
        }
        "content_block_delta" => v
            .get("delta")
            .and_then(|d| d.get("type"))
            .and_then(|t| t.as_str())
            .map(|delta_type| match delta_type {
                "text_delta" => v
                    .get("delta")
                    .and_then(|d| d.get("text"))
                    .and_then(|t| t.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| vec![ProviderStreamEvent::TextDelta(s.to_string())])
                    .unwrap_or_default(),
                "input_json_delta" => {
                    let index = v.get("index").and_then(as_u32).unwrap_or(0);
                    v.get("delta")
                        .and_then(|d| d.get("partial_json"))
                        .and_then(|t| t.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| {
                            vec![ProviderStreamEvent::ToolArgsDelta {
                                index,
                                arguments: s.to_string(),
                            }]
                        })
                        .unwrap_or_default()
                }
                "thinking_delta" => v
                    .get("delta")
                    .and_then(|d| d.get("thinking"))
                    .and_then(|t| t.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| vec![ProviderStreamEvent::ReasoningDelta(s.to_string())])
                    .unwrap_or_default(),
                _ => Vec::new(),
            })
            .unwrap_or_default(),
        "message_delta" => vec![ProviderStreamEvent::Stop {
            finish_reason: v
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(|r| r.as_str())
                .map(str::to_string),
            prompt_tokens: None,
            completion_tokens: v
                .get("usage")
                .and_then(|u| u.get("output_tokens"))
                .and_then(as_u32),
        }],
        "message_stop" => vec![ProviderStreamEvent::Stop {
            finish_reason: None,
            prompt_tokens: None,
            completion_tokens: None,
        }],
        _ => Vec::new(),
    }
}

fn parse_ollama_line(line: &str) -> Vec<ProviderStreamEvent> {
    if line.trim().is_empty() {
        return Vec::new();
    }
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let model = v.get("model").and_then(|m| m.as_str()).map(str::to_string);
    let prompt_tokens = v.get("prompt_eval_count").and_then(as_u32);
    if model.is_some() || prompt_tokens.is_some() {
        out.push(ProviderStreamEvent::Start {
            id: None,
            model,
            prompt_tokens,
        });
    }

    if let Some(text) = v
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .filter(|s| !s.is_empty())
    {
        out.push(ProviderStreamEvent::TextDelta(text.to_string()));
    }

    if let Some(tool_calls) = v
        .get("message")
        .and_then(|m| m.get("tool_calls"))
        .and_then(|tc| tc.as_array())
    {
        for (i, tc) in tool_calls.iter().enumerate() {
            let index = i as u32;
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .map(str::to_string);

            out.push(ProviderStreamEvent::ToolStart {
                index,
                id: None,
                name,
            });

            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .cloned()
                .unwrap_or(Value::Null);
            if args != Value::Null {
                out.push(ProviderStreamEvent::ToolArgsDelta {
                    index,
                    arguments: json_to_compact_string(&args),
                });
            }
        }
    }

    if v.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
        out.push(ProviderStreamEvent::Stop {
            finish_reason: Some("stop".to_string()),
            prompt_tokens,
            completion_tokens: v.get("eval_count").and_then(as_u32),
        });
    }

    out
}

fn parse_acp_line(line: &str) -> Vec<ProviderStreamEvent> {
    if line.trim().is_empty() {
        return Vec::new();
    }
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };

    match v.get("type").and_then(|t| t.as_str()) {
        Some("text_delta") => v
            .get("text")
            .and_then(|t| t.as_str())
            .filter(|s| !s.is_empty())
            .map(|text| vec![ProviderStreamEvent::TextDelta(text.to_string())])
            .unwrap_or_default(),
        Some("stop") => vec![ProviderStreamEvent::Stop {
            finish_reason: v
                .get("finish_reason")
                .and_then(|r| r.as_str())
                .map(str::to_string),
            prompt_tokens: None,
            completion_tokens: None,
        }],
        _ => Vec::new(),
    }
}

#[derive(Default, Clone)]
struct ToolState {
    id: String,
    name: String,
    arguments: String,
    sent_start_openai: bool,
    sent_start_anthropic: bool,
    anthropic_index: Option<u32>,
}

struct DownstreamRenderer {
    protocol: DownstreamProtocol,
    id: String,
    model: String,
    created: u64,
    prompt_tokens: u32,
    completion_tokens: u32,
    openai_role_sent: bool,
    anthropic_message_started: bool,
    anthropic_thinking_started: bool,
    anthropic_thinking_closed: bool,
    anthropic_thinking_index: Option<u32>,
    anthropic_text_started: bool,
    anthropic_text_closed: bool,
    anthropic_text_index: Option<u32>,
    anthropic_next_index: u32,
    tools: BTreeMap<u32, ToolState>,
    stopped: bool,
}

impl DownstreamRenderer {
    fn new(protocol: DownstreamProtocol, model: String) -> Self {
        Self {
            protocol,
            id: format!("stream_{}", unix_seconds()),
            model,
            created: unix_seconds(),
            prompt_tokens: 0,
            completion_tokens: 0,
            openai_role_sent: false,
            anthropic_message_started: false,
            anthropic_thinking_started: false,
            anthropic_thinking_closed: false,
            anthropic_thinking_index: None,
            anthropic_text_started: false,
            anthropic_text_closed: false,
            anthropic_text_index: None,
            anthropic_next_index: 0,
            tools: BTreeMap::new(),
            stopped: false,
        }
    }

    async fn push_event(&mut self, event: ProviderStreamEvent, tx: &mpsc::Sender<Bytes>) {
        match event {
            ProviderStreamEvent::Start {
                id,
                model,
                prompt_tokens,
            } => {
                if let Some(id) = id {
                    self.id = id;
                }
                if let Some(model) = model {
                    self.model = model;
                }
                if let Some(prompt_tokens) = prompt_tokens {
                    self.prompt_tokens = prompt_tokens;
                }

                if matches!(self.protocol, DownstreamProtocol::OpenAI) && !self.openai_role_sent {
                    self.emit_openai_role_chunk(tx).await;
                }
                if matches!(self.protocol, DownstreamProtocol::Anthropic)
                    && !self.anthropic_message_started
                {
                    self.emit_anthropic_message_start(tx).await;
                }
            }
            ProviderStreamEvent::ReasoningDelta(reasoning) => {
                if reasoning.is_empty() {
                    return;
                }
                match self.protocol {
                    DownstreamProtocol::OpenAI => {
                        if !self.openai_role_sent {
                            self.emit_openai_role_chunk(tx).await;
                        }
                        self.emit_openai_reasoning_chunk(reasoning, tx).await;
                    }
                    DownstreamProtocol::Anthropic => {
                        if !self.anthropic_message_started {
                            self.emit_anthropic_message_start(tx).await;
                        }
                        self.emit_anthropic_thinking_start(tx).await;
                        self.emit_anthropic_thinking_delta(reasoning, tx).await;
                    }
                    DownstreamProtocol::Ollama => {}
                }
            }
            ProviderStreamEvent::TextDelta(text) => {
                if text.is_empty() {
                    return;
                }
                match self.protocol {
                    DownstreamProtocol::OpenAI => {
                        if !self.openai_role_sent {
                            self.emit_openai_role_chunk(tx).await;
                        }
                        self.emit_openai_content_chunk(text, tx).await;
                    }
                    DownstreamProtocol::Anthropic => {
                        if !self.anthropic_message_started {
                            self.emit_anthropic_message_start(tx).await;
                        }
                        self.close_anthropic_thinking(tx).await;
                        self.emit_anthropic_text_start(tx).await;
                        self.emit_anthropic_content_delta(text, tx).await;
                    }
                    DownstreamProtocol::Ollama => {
                        self.emit_ollama_delta(text, tx).await;
                    }
                }
            }
            ProviderStreamEvent::ToolStart { index, id, name } => {
                let generated_id = format!("call_{}", index + 1);
                let tool = self.tools.entry(index).or_insert_with(|| ToolState {
                    id: generated_id.clone(),
                    ..ToolState::default()
                });
                if tool.id.is_empty() {
                    tool.id = generated_id;
                }
                if let Some(id) = id
                    && !id.is_empty()
                {
                    tool.id = id;
                }
                if let Some(name) = name
                    && !name.is_empty()
                {
                    tool.name = name;
                }

                match self.protocol {
                    DownstreamProtocol::OpenAI => {
                        if !self.openai_role_sent {
                            self.emit_openai_role_chunk(tx).await;
                        }
                        self.emit_openai_tool_start(index, tx).await;
                    }
                    DownstreamProtocol::Anthropic => {
                        if !self.anthropic_message_started {
                            self.emit_anthropic_message_start(tx).await;
                        }
                        self.close_anthropic_thinking(tx).await;
                        self.emit_anthropic_tool_start(index, tx).await;
                    }
                    DownstreamProtocol::Ollama => {}
                }
            }
            ProviderStreamEvent::ToolArgsDelta { index, arguments } => {
                if arguments.is_empty() {
                    return;
                }
                let generated_id = format!("call_{}", index + 1);
                let tool = self.tools.entry(index).or_insert_with(|| ToolState {
                    id: generated_id,
                    ..ToolState::default()
                });
                tool.arguments.push_str(&arguments);

                match self.protocol {
                    DownstreamProtocol::OpenAI => {
                        if !self.openai_role_sent {
                            self.emit_openai_role_chunk(tx).await;
                        }
                        self.emit_openai_tool_args_delta(index, arguments, tx).await;
                    }
                    DownstreamProtocol::Anthropic => {
                        if !self.anthropic_message_started {
                            self.emit_anthropic_message_start(tx).await;
                        }
                        self.close_anthropic_thinking(tx).await;
                        self.emit_anthropic_tool_start(index, tx).await;
                        self.emit_anthropic_tool_args_delta(index, arguments, tx)
                            .await;
                    }
                    DownstreamProtocol::Ollama => {}
                }
            }
            ProviderStreamEvent::Stop {
                finish_reason,
                prompt_tokens,
                completion_tokens,
            } => {
                if let Some(prompt_tokens) = prompt_tokens {
                    self.prompt_tokens = prompt_tokens;
                }
                if let Some(completion_tokens) = completion_tokens {
                    self.completion_tokens = completion_tokens;
                }
                self.emit_stop(finish_reason, tx).await;
            }
        }
    }

    async fn finish(&mut self, tx: &mpsc::Sender<Bytes>) {
        self.emit_stop(None, tx).await;
    }

    async fn emit_stop(&mut self, finish_reason: Option<String>, tx: &mpsc::Sender<Bytes>) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        match self.protocol {
            DownstreamProtocol::OpenAI => {
                if !self.openai_role_sent {
                    self.emit_openai_role_chunk(tx).await;
                }
                let reason = map_to_openai_finish_reason(finish_reason.as_deref());
                let payload = json!({
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": reason,
                    }]
                });
                send_text(tx, sse_data_frame(payload)).await;
                send_text(tx, "data: [DONE]\n\n".to_string()).await;
            }
            DownstreamProtocol::Anthropic => {
                if !self.anthropic_message_started {
                    self.emit_anthropic_message_start(tx).await;
                }
                self.close_anthropic_thinking(tx).await;
                if self.anthropic_text_started
                    && !self.anthropic_text_closed
                    && let Some(index) = self.anthropic_text_index
                {
                    send_text(
                        tx,
                        sse_event_frame(
                            "content_block_stop",
                            json!({"type": "content_block_stop", "index": index}),
                        ),
                    )
                    .await;
                    self.anthropic_text_closed = true;
                }
                for tool in self.tools.values() {
                    if tool.sent_start_anthropic
                        && let Some(index) = tool.anthropic_index
                    {
                        send_text(
                            tx,
                            sse_event_frame(
                                "content_block_stop",
                                json!({"type": "content_block_stop", "index": index}),
                            ),
                        )
                        .await;
                    }
                }
                let reason = map_to_anthropic_stop_reason(finish_reason.as_deref());
                send_text(
                    tx,
                    sse_event_frame(
                        "message_delta",
                        json!({
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": reason,
                                "stop_sequence": Value::Null,
                            },
                            "usage": {
                                "output_tokens": self.completion_tokens,
                            }
                        }),
                    ),
                )
                .await;
                send_text(
                    tx,
                    sse_event_frame("message_stop", json!({"type": "message_stop"})),
                )
                .await;
            }
            DownstreamProtocol::Ollama => {
                self.emit_ollama_tool_chunk(tx).await;
                self.emit_ollama_done(tx).await;
            }
        }
    }

    async fn emit_openai_role_chunk(&mut self, tx: &mpsc::Sender<Bytes>) {
        if self.openai_role_sent {
            return;
        }
        self.openai_role_sent = true;
        let payload = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": Value::Null,
            }]
        });
        send_text(tx, sse_data_frame(payload)).await;
    }

    async fn emit_openai_content_chunk(&self, text: String, tx: &mpsc::Sender<Bytes>) {
        let payload = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": Value::Null,
            }]
        });
        send_text(tx, sse_data_frame(payload)).await;
    }

    async fn emit_openai_reasoning_chunk(&self, reasoning: String, tx: &mpsc::Sender<Bytes>) {
        let payload = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": reasoning},
                "finish_reason": Value::Null,
            }]
        });
        send_text(tx, sse_data_frame(payload)).await;
    }

    async fn emit_openai_tool_start(&mut self, index: u32, tx: &mpsc::Sender<Bytes>) {
        let Some(tool) = self.tools.get_mut(&index) else {
            return;
        };
        if tool.sent_start_openai {
            return;
        }
        tool.sent_start_openai = true;

        let mut call = serde_json::Map::new();
        call.insert("index".to_string(), json!(index));
        call.insert("id".to_string(), Value::String(tool.id.clone()));
        call.insert("type".to_string(), Value::String("function".to_string()));

        let mut function = serde_json::Map::new();
        if !tool.name.is_empty() {
            function.insert("name".to_string(), Value::String(tool.name.clone()));
        }
        function.insert("arguments".to_string(), Value::String(String::new()));
        call.insert("function".to_string(), Value::Object(function));

        let payload = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [Value::Object(call)]},
                "finish_reason": Value::Null,
            }]
        });
        send_text(tx, sse_data_frame(payload)).await;
    }

    async fn emit_openai_tool_args_delta(
        &self,
        index: u32,
        arguments: String,
        tx: &mpsc::Sender<Bytes>,
    ) {
        let payload = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": index,
                        "function": {
                            "arguments": arguments,
                        }
                    }]
                },
                "finish_reason": Value::Null,
            }]
        });
        send_text(tx, sse_data_frame(payload)).await;
    }

    async fn emit_anthropic_message_start(&mut self, tx: &mpsc::Sender<Bytes>) {
        if self.anthropic_message_started {
            return;
        }
        self.anthropic_message_started = true;
        send_text(
            tx,
            sse_event_frame(
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": self.id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {
                            "input_tokens": self.prompt_tokens,
                            "output_tokens": 0,
                        }
                    }
                }),
            ),
        )
        .await;
    }

    async fn emit_anthropic_text_start(&mut self, tx: &mpsc::Sender<Bytes>) {
        if self.anthropic_text_started {
            return;
        }
        self.anthropic_text_started = true;
        self.anthropic_text_closed = false;
        let index = self.anthropic_next_index;
        self.anthropic_next_index += 1;
        self.anthropic_text_index = Some(index);
        send_text(
            tx,
            sse_event_frame(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "text",
                        "text": "",
                    }
                }),
            ),
        )
        .await;
    }

    async fn emit_anthropic_thinking_start(&mut self, tx: &mpsc::Sender<Bytes>) {
        if self.anthropic_thinking_started {
            return;
        }
        self.anthropic_thinking_started = true;
        self.anthropic_thinking_closed = false;
        let index = self.anthropic_next_index;
        self.anthropic_next_index += 1;
        self.anthropic_thinking_index = Some(index);
        send_text(
            tx,
            sse_event_frame(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                    }
                }),
            ),
        )
        .await;
    }

    async fn emit_anthropic_thinking_delta(&self, thinking: String, tx: &mpsc::Sender<Bytes>) {
        let index = self.anthropic_thinking_index.unwrap_or(0);
        send_text(
            tx,
            sse_event_frame(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": thinking,
                    }
                }),
            ),
        )
        .await;
    }

    async fn close_anthropic_thinking(&mut self, tx: &mpsc::Sender<Bytes>) {
        if !self.anthropic_thinking_started || self.anthropic_thinking_closed {
            return;
        }
        let Some(index) = self.anthropic_thinking_index else {
            return;
        };
        send_text(
            tx,
            sse_event_frame(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": self.id,
                    }
                }),
            ),
        )
        .await;
        send_text(
            tx,
            sse_event_frame(
                "content_block_stop",
                json!({
                    "type": "content_block_stop",
                    "index": index,
                }),
            ),
        )
        .await;
        self.anthropic_thinking_closed = true;
    }

    async fn emit_anthropic_content_delta(&self, text: String, tx: &mpsc::Sender<Bytes>) {
        let index = self.anthropic_text_index.unwrap_or(0);
        send_text(
            tx,
            sse_event_frame(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "text_delta",
                        "text": text,
                    }
                }),
            ),
        )
        .await;
    }

    async fn emit_anthropic_tool_start(&mut self, index: u32, tx: &mpsc::Sender<Bytes>) {
        let Some(tool) = self.tools.get_mut(&index) else {
            return;
        };
        if tool.sent_start_anthropic {
            return;
        }
        tool.sent_start_anthropic = true;
        let anthropic_index = if let Some(existing) = tool.anthropic_index {
            existing
        } else {
            let next = self.anthropic_next_index;
            self.anthropic_next_index += 1;
            tool.anthropic_index = Some(next);
            next
        };

        send_text(
            tx,
            sse_event_frame(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": anthropic_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool.id.clone(),
                        "name": tool.name.clone(),
                        "input": {},
                    }
                }),
            ),
        )
        .await;
    }

    async fn emit_anthropic_tool_args_delta(
        &self,
        index: u32,
        arguments: String,
        tx: &mpsc::Sender<Bytes>,
    ) {
        let Some(tool) = self.tools.get(&index) else {
            return;
        };
        let Some(anthropic_index) = tool.anthropic_index else {
            return;
        };
        send_text(
            tx,
            sse_event_frame(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": anthropic_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": arguments,
                    }
                }),
            ),
        )
        .await;
    }

    async fn emit_ollama_delta(&self, text: String, tx: &mpsc::Sender<Bytes>) {
        let payload = json!({
            "model": self.model,
            "message": {
                "role": "assistant",
                "content": text,
            },
            "done": false,
        });
        send_text(tx, format!("{}\n", json_to_compact_string(&payload))).await;
    }

    async fn emit_ollama_tool_chunk(&self, tx: &mpsc::Sender<Bytes>) {
        let tool_calls: Vec<Value> = self
            .tools
            .values()
            .filter(|tool| !tool.name.is_empty())
            .map(|tool| {
                let arguments = serde_json::from_str::<Value>(&tool.arguments)
                    .unwrap_or_else(|_| json!({"_raw": tool.arguments.clone()}));
                json!({
                    "function": {
                        "name": tool.name.clone(),
                        "arguments": arguments,
                    }
                })
            })
            .collect();

        if tool_calls.is_empty() {
            return;
        }

        let payload = json!({
            "model": self.model,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
            "done": false,
        });
        send_text(tx, format!("{}\n", json_to_compact_string(&payload))).await;
    }

    async fn emit_ollama_done(&self, tx: &mpsc::Sender<Bytes>) {
        let mut done = serde_json::Map::new();
        done.insert("model".to_string(), Value::String(self.model.clone()));
        done.insert(
            "message".to_string(),
            json!({
                "role": "assistant",
                "content": "",
            }),
        );
        done.insert("done".to_string(), json!(true));
        done.insert("prompt_eval_count".to_string(), json!(self.prompt_tokens));
        done.insert("eval_count".to_string(), json!(self.completion_tokens));
        send_text(
            tx,
            format!("{}\n", json_to_compact_string(&Value::Object(done))),
        )
        .await;
    }
}

async fn send_text(tx: &mpsc::Sender<Bytes>, text: String) {
    let _ = tx.send(Bytes::from(text)).await;
}

fn as_u32(v: &Value) -> Option<u32> {
    v.as_u64().map(|n| n as u32)
}

fn map_to_openai_finish_reason(reason: Option<&str>) -> String {
    match reason {
        Some("max_tokens") | Some("length") => "length".to_string(),
        Some("tool_use") | Some("tool_calls") => "tool_calls".to_string(),
        Some("end_turn") | Some("stop_sequence") | Some("stop") => "stop".to_string(),
        Some(other) => other.to_string(),
        None => "stop".to_string(),
    }
}

fn map_to_anthropic_stop_reason(reason: Option<&str>) -> String {
    match reason {
        Some("length") | Some("max_tokens") => "max_tokens".to_string(),
        Some("tool_calls") | Some("tool_use") => "tool_use".to_string(),
        Some("stop_sequence") => "stop_sequence".to_string(),
        Some("stop") | Some("end_turn") => "end_turn".to_string(),
        Some(other) => other.to_string(),
        None => "end_turn".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures::StreamExt;

    use super::*;

    #[tokio::test]
    async fn maps_acp_text_events_to_openai_stream() {
        let upstream = Box::pin(futures::stream::iter(vec![Ok(Bytes::from_static(
            br#"{"type":"text_delta","text":"hello"}
{"type":"stop","finish_reason":"stop"}
"#,
        ))]));

        let mut downstream = map_provider_stream_to_downstream(
            Provider::Acp,
            DownstreamProtocol::OpenAI,
            upstream,
            "codex".to_string(),
        );
        let mut body = Vec::new();
        while let Some(chunk) = downstream.next().await {
            body.extend_from_slice(&chunk.expect("stream chunk"));
        }
        let body = String::from_utf8(body).expect("utf8 stream");

        assert!(body.contains(r#""content":"hello""#), "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
    }
}
