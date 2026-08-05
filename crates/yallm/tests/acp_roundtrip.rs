//! End-to-end round trip through the ACP upstream provider: the proxy layer
//! (`complete_ir` / `call_provider_stream`) spawns a real `yallm acp` child
//! process — this crate's own binary — as the ACP agent over stdio, and the
//! child's provider backend falls back to the deterministic mock.
//!
//! This covers the chain that in-process unit tests cannot:
//! `call_acp` → `complete_with_command` → `AcpAgent` stdio spawn →
//! `serve_stdio` agent → (child) provider routing.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use futures::StreamExt;
use serde_json::Value;
use yallm_ir::{ChatRequest, Content, Message, Role};

/// Absolute path to this crate's `yallm` binary, available in integration tests.
const BIN: &str = env!("CARGO_BIN_EXE_yallm");

/// Unique per call — unlike wall-clock timestamps, this cannot collide between
/// tests running concurrently (nanosecond precision is not enough when two
/// tests call `temp_db_url` at the same instant, and SQLite WAL mode is
/// unforgiving about two processes opening the same file).
static DB_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_db_url(name: &str) -> String {
    let unique = DB_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "sqlite://{}/{}-{unique}-{name}.sqlite3",
        std::env::temp_dir().display(),
        std::process::id(),
    )
}

fn acp_state() -> yallm_server::AppState {
    let mut env = HashMap::new();
    env.insert(
        yallm_storage::DB_URL_ENV.to_string(),
        temp_db_url("acp-roundtrip"),
    );
    // Spawn ourselves as the ACP agent; `openai:mock` routes to the mock backend.
    // The JSON stdio spec form is used so the child gets its own DB URL — the
    // default store lives in a shared cache dir and would lock between
    // concurrently running test children. This also exercises the JSON
    // command form that real configs (e.g. npx agents) use.
    env.insert(
        "YALLM_ACP_COMMAND".to_string(),
        serde_json::json!({
            "type": "stdio",
            "name": "yallm-test-agent",
            "command": BIN,
            "args": ["acp", "--model", "openai:mock"],
            "env": [
                {
                    "name": yallm_storage::DB_URL_ENV,
                    "value": temp_db_url("acp-child"),
                }
            ],
        })
        .to_string(),
    );
    yallm_server::AppState::from_loaded_config(yallm_config::LoadedConfig {
        env,
        litellm_models: Vec::new(),
        warnings: Vec::new(),
    })
}

fn chat_request(text: &str) -> ChatRequest {
    ChatRequest {
        model: "acp:codex".to_string(),
        messages: vec![Message::text(Role::User, text)],
        max_tokens: None,
        temperature: None,
        top_p: None,
        stream: false,
    }
}

fn assistant_text(response: &yallm_ir::ChatResponse) -> String {
    response
        .choices
        .iter()
        .flat_map(|choice| {
            choice
                .message
                .content
                .iter()
                .filter_map(|content| match content {
                    Content::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
        })
        .collect()
}

#[tokio::test]
async fn acp_upstream_child_process_completes_full_round_trip() {
    let state = acp_state();

    let response = yallm_server::complete_ir(
        &state,
        1,
        chat_request("Round-trip works"),
        &Default::default(),
    )
    .await
    .expect("acp completion");

    // Upstream model is rewritten from `acp:codex` to the target's model.
    assert_eq!(response.model, "codex");
    // The child agent's mock backend echoes the prompt text.
    assert_eq!(assistant_text(&response), "yallm (mock): Round-trip works");
    assert_eq!(
        response.choices[0].finish_reason.as_deref(),
        Some("stop"),
        "EndTurn stop reason survives the round trip"
    );
}

#[tokio::test]
async fn acp_upstream_child_process_streams_text_and_stop() {
    let state = acp_state();
    let target = yallm_server::choose_provider(&state, "acp:codex");

    let stream = yallm_server::call_provider_stream(
        &state,
        2,
        &target,
        &chat_request("Stream it"),
        &Default::default(),
    )
    .await
    .expect("acp stream start");

    let mut body = Vec::new();
    let mut chunks = stream.body;
    while let Some(chunk) = chunks.next().await {
        body.extend_from_slice(&chunk.expect("stream chunk"));
    }

    let mut saw_text = false;
    let mut saw_stop = false;
    for line in String::from_utf8(body).expect("utf8 stream body").lines() {
        let event: Value = serde_json::from_str(line).expect("acp event line");
        match event.get("type").and_then(|t| t.as_str()) {
            Some("text_delta") => {
                assert_eq!(event["text"], "yallm (mock): Stream it");
                saw_text = true;
            }
            Some("stop") => {
                assert_eq!(event["finish_reason"], "stop");
                saw_stop = true;
            }
            other => panic!("unexpected acp event line: {line} (type {other:?})"),
        }
    }
    assert!(saw_text, "expected a text_delta event");
    assert!(saw_stop, "expected a stop event");
}
