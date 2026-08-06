//! ACP bridge: connect yallm to Agent Client Protocol agents, both ways.
//!
//! Two directions, mirror images:
//!
//! **yallm as ACP client** (upstream — `acp:` provider): `complete_with_agent`
//! / `complete_with_command` run one prompt turn against an ACP agent
//! (stdio subprocess per the command string, e.g.
//! `"npx -y @agentclientprotocol/codex-acp"`) and return the assistant text as
//! a yallm IR [`ChatResponse`]; `stream_with_agent` / `stream_with_command`
//! yield [`AcpStreamEvent`]s (`TextDelta`… `Stop`) instead. Used by
//! `yallm-server` `call_acp` / `call_acp_stream`.
//!
//! **yallm as ACP agent** (downstream — `yallm acp` subcommand):
//! [`agent_for_backend`] wraps any IR backend closure in an ACP agent
//! (handles `initialize` / `new_session` / `prompt`, streams the assistant
//! text as `AgentMessageChunk` notifications), and [`serve_stdio`] attaches
//! that agent to stdin/stdout. Wire format is newline-delimited JSON-RPC;
//! anything the agent writes to stdout must be protocol, so the binary
//! routes logs to stderr in ACP mode.
//!
//! Conversion helpers: [`ir_to_prompt`] (IR → ACP content blocks, a single
//! user message stays bare, multi-turn chats become a `Role: text` transcript)
//! and [`session_update_text`] (extract assistant text from a session update).
//!
//! Errors are `agent_client_protocol::Error`; use [`internal_error`] to build
//! them for backend failures.

#![warn(missing_docs)]

use std::{
    future::Future,
    path::PathBuf,
    pin::Pin,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use agent_client_protocol::{
    AcpAgent, Agent, Client, ConnectTo, ConnectionTo, Stdio,
    schema::{
        AgentCapabilities, ContentBlock, ContentChunk, InitializeRequest, InitializeResponse,
        NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, ProtocolVersion,
        SessionNotification, SessionUpdate, StopReason, TextContent,
    },
};
use futures::Stream;
use tokio::sync::{Mutex, mpsc};
use yallm_ir::{ChatRequest, ChatResponse, Choice, Content, Message, Role};

/// One step of an ACP streaming turn, normalized for yallm's stream pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcpStreamEvent {
    /// Assistant text chunk
    TextDelta(String),
    /// Turn finished; `finish_reason` matches yallm's IR finish reasons
    Stop {
        /// Why the turn ended (`stop`, `length`, ...)
        finish_reason: String,
    },
}

/// Streaming turn: `TextDelta` events, then exactly one `Stop`, then EOF.
pub type AcpEventStream =
    Pin<Box<dyn Stream<Item = Result<AcpStreamEvent, String>> + Send + 'static>>;

/// Convert an IR chat request into ACP prompt content blocks. A single user
/// message stays bare text; multi-message chats become a `Role: text`
/// transcript joined with blank lines.
///
/// ```
/// use yallm_ir::{ChatRequest, Message, Role};
///
/// let request = ChatRequest {
///     model: "acp:codex".to_string(),
///     messages: vec![Message::text(Role::User, "Hello")],
///     max_tokens: None,
///     temperature: None,
///     top_p: None,
///     stream: false,
/// };
///
/// let blocks = yallm_acp::ir_to_prompt(&request);
/// assert_eq!(blocks.len(), 1);
/// ```
pub fn ir_to_prompt(request: &ChatRequest) -> Vec<ContentBlock> {
    let text = if request.messages.len() == 1 && request.messages[0].role == Role::User {
        message_text(&request.messages[0])
    } else {
        request
            .messages
            .iter()
            .map(render_transcript_message)
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    vec![ContentBlock::Text(TextContent::new(text))]
}

/// Extract assistant text from a session update; returns `None` for any
/// other update kind (thoughts, tool calls, ...).
pub fn session_update_text(update: &SessionUpdate) -> Option<String> {
    match update {
        SessionUpdate::AgentMessageChunk(ContentChunk {
            content: ContentBlock::Text(text),
            ..
        }) => Some(text.text.clone()),
        _ => None,
    }
}

/// Run one prompt turn against an in-process ACP agent connection and return
/// the assistant reply as an IR [`ChatResponse`] (accumulated from
/// `AgentMessageChunk` notifications).
pub async fn complete_with_agent<A>(
    agent: A,
    cwd: impl Into<PathBuf>,
    request: ChatRequest,
) -> Result<ChatResponse, agent_client_protocol::Error>
where
    A: ConnectTo<Client> + 'static,
{
    let cwd = cwd.into();
    let model = request.model.clone();
    let prompt = ir_to_prompt(&request);
    let output = Arc::new(Mutex::new(String::new()));
    let output_for_notifications = output.clone();

    let stop_reason = Client
        .builder()
        .name("yallm-acp-client")
        .on_receive_notification(
            async move |notification: SessionNotification, _connection| {
                if let Some(text) = session_update_text(&notification.update) {
                    output_for_notifications.lock().await.push_str(&text);
                }
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .connect_with(agent, async move |connection| {
            connection
                .send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;

            let session = connection
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;

            let prompt_response = connection
                .send_request(PromptRequest::new(session.session_id, prompt))
                .block_task()
                .await?;

            Ok(prompt_response.stop_reason)
        })
        .await?;

    let text = output.lock().await.clone();
    Ok(chat_response_from_text(
        &model,
        text,
        finish_reason(stop_reason),
    ))
}

/// [`complete_with_agent`] for a subprocess: spawn `command` (shell string or
/// JSON stdio spec) and speak ACP to it over stdio. This is what
/// `yallm-server`'s `acp:` provider calls.
pub async fn complete_with_command(
    command: &str,
    cwd: impl Into<PathBuf>,
    request: ChatRequest,
) -> Result<ChatResponse, agent_client_protocol::Error> {
    let agent = AcpAgent::from_str(command)?;
    complete_with_agent(agent, cwd, request).await
}

/// [`complete_with_agent`] as a stream: yields [`AcpStreamEvent`]s as the
/// agent pushes updates, ending with `Stop`. The turn runs in a spawned task;
/// the returned stream is cancel-safe.
pub fn stream_with_agent<A>(
    agent: A,
    cwd: impl Into<PathBuf>,
    request: ChatRequest,
) -> AcpEventStream
where
    A: ConnectTo<Client> + Send + 'static,
{
    let cwd = cwd.into();
    let prompt = ir_to_prompt(&request);
    let (tx, rx) = mpsc::channel::<Result<AcpStreamEvent, String>>(64);

    tokio::spawn(async move {
        let tx_for_notifications = tx.clone();
        let result = Client
            .builder()
            .name("yallm-acp-stream-client")
            .on_receive_notification(
                async move |notification: SessionNotification, _connection| {
                    if let Some(text) = session_update_text(&notification.update) {
                        let _ = tx_for_notifications
                            .send(Ok(AcpStreamEvent::TextDelta(text)))
                            .await;
                    }
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(agent, async move |connection| {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let session = connection
                    .send_request(NewSessionRequest::new(cwd))
                    .block_task()
                    .await?;

                let prompt_response = connection
                    .send_request(PromptRequest::new(session.session_id, prompt))
                    .block_task()
                    .await?;

                Ok(prompt_response.stop_reason)
            })
            .await;

        match result {
            Ok(stop_reason) => {
                let _ = tx
                    .send(Ok(AcpStreamEvent::Stop {
                        finish_reason: finish_reason(stop_reason).to_string(),
                    }))
                    .await;
            }
            Err(err) => {
                let _ = tx.send(Err(format!("{err}"))).await;
            }
        }
    });

    Box::pin(futures::stream::unfold(rx, |mut rx| async {
        rx.recv().await.map(|event| (event, rx))
    }))
}

/// [`stream_with_agent`] for a subprocess (see [`complete_with_command`]).
pub fn stream_with_command(
    command: &str,
    cwd: impl Into<PathBuf>,
    request: ChatRequest,
) -> Result<AcpEventStream, agent_client_protocol::Error> {
    let agent = AcpAgent::from_str(command)?;
    Ok(stream_with_agent(agent, cwd, request))
}

/// Wrap an IR backend closure in an ACP agent: handles `initialize` /
/// `new_session` / `prompt`, converts prompts via `prompt_to_ir` (a single
/// user message with `default_model`), streams the assistant text back as
/// `AgentMessageChunk` notifications, and maps the backend's finish reason
/// to an ACP stop reason.
pub fn agent_for_backend<B, Fut>(
    default_model: impl Into<String>,
    backend: B,
) -> impl ConnectTo<Client>
where
    B: Fn(ChatRequest) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<ChatResponse, agent_client_protocol::Error>> + Send + 'static,
{
    let default_model = Arc::new(default_model.into());
    let next_session_id = Arc::new(AtomicU64::new(1));

    Agent
        .builder()
        .name("yallm-acp-agent")
        .on_receive_request(
            async |initialize: InitializeRequest, responder, _connection| {
                responder.respond(
                    InitializeResponse::new(initialize.protocol_version)
                        .agent_capabilities(AgentCapabilities::new()),
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let next_session_id = next_session_id.clone();
                async move |_session: NewSessionRequest, responder, _connection| {
                    let id = next_session_id.fetch_add(1, Ordering::Relaxed);
                    responder.respond(NewSessionResponse::new(format!("yallm-session-{id}")))
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_request(
            {
                let default_model = default_model.clone();
                let backend = backend.clone();
                async move |prompt: PromptRequest, responder, connection: ConnectionTo<Client>| {
                    let session_id = prompt.session_id.clone();
                    let request = prompt_to_ir(&default_model, &prompt);
                    let response = backend(request).await?;

                    let text = first_assistant_text(&response);
                    if !text.is_empty() {
                        connection.send_notification(SessionNotification::new(
                            session_id,
                            SessionUpdate::AgentMessageChunk(ContentChunk::new(
                                ContentBlock::Text(TextContent::new(text)),
                            )),
                        ))?;
                    }

                    responder.respond(PromptResponse::new(stop_reason(&response)))
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
}

/// Serve the [`agent_for_backend`] agent on stdin/stdout — the `yallm acp`
/// subcommand's body. Only ACP protocol bytes may go to stdout.
pub async fn serve_stdio<B, Fut>(
    default_model: impl Into<String>,
    backend: B,
) -> Result<(), agent_client_protocol::Error>
where
    B: Fn(ChatRequest) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<ChatResponse, agent_client_protocol::Error>> + Send + 'static,
{
    agent_for_backend(default_model, backend)
        .connect_to(Stdio::new())
        .await
}

/// Build a protocol-level `internal_error` (surfaced to the ACP peer as a
/// JSON-RPC error) for backend failures.
pub fn internal_error(message: impl Into<String>) -> agent_client_protocol::Error {
    agent_client_protocol::util::internal_error(message.into())
}

fn render_transcript_message(message: &Message) -> String {
    format!("{}: {}", role_label(message.role), message_text(message))
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::System => "System",
        Role::User => "User",
        Role::Assistant => "Assistant",
        Role::Tool => "Tool",
    }
}

fn message_text(message: &Message) -> String {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            Content::Text(text) => Some(text.text.as_str()),
            Content::ToolResult(result) => Some(result.content.as_str()),
            Content::Image(_) | Content::ToolCall(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn chat_response_from_text(model: &str, text: String, finish_reason: &'static str) -> ChatResponse {
    ChatResponse {
        id: "acp_response".to_string(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message::text(Role::Assistant, text),
            finish_reason: Some(finish_reason.to_string()),
        }],
        usage: None,
    }
}

fn finish_reason(stop_reason: StopReason) -> &'static str {
    match stop_reason {
        StopReason::EndTurn => "stop",
        StopReason::MaxTokens => "length",
        StopReason::MaxTurnRequests => "max_turn_requests",
        StopReason::Refusal => "refusal",
        StopReason::Cancelled => "cancelled",
        _ => "stop",
    }
}

fn prompt_to_ir(default_model: &str, prompt: &PromptRequest) -> ChatRequest {
    ChatRequest {
        model: default_model.to_string(),
        messages: vec![Message::text(Role::User, prompt_text(&prompt.prompt))],
        max_tokens: None,
        temperature: None,
        top_p: None,
        stream: false,
    }
}

fn prompt_text(prompt: &[ContentBlock]) -> String {
    prompt
        .iter()
        .filter_map(|content| match content {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn first_assistant_text(response: &ChatResponse) -> String {
    response
        .choices
        .first()
        .map(|choice| message_text(&choice.message))
        .unwrap_or_default()
}

fn stop_reason(response: &ChatResponse) -> StopReason {
    match response
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.as_deref())
    {
        Some("length") => StopReason::MaxTokens,
        Some("cancelled") => StopReason::Cancelled,
        Some("refusal") => StopReason::Refusal,
        _ => StopReason::EndTurn,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use agent_client_protocol::schema::{
        AgentCapabilities, ContentBlock, ContentChunk, InitializeRequest, InitializeResponse,
        NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, ProtocolVersion,
        SessionNotification, SessionUpdate, StopReason, TextContent,
    };
    use agent_client_protocol::{Agent, Client, ConnectionTo};
    use futures::StreamExt;
    use tokio::sync::Mutex;
    use yallm_ir::{ChatRequest, ChatResponse, Choice, Content, Message, Role};

    use super::*;

    #[test]
    fn ir_to_prompt_preserves_single_user_text_without_transcript_wrapping() {
        let request = ChatRequest {
            model: "acp:codex".to_string(),
            messages: vec![Message::text(Role::User, "Explain this repository")],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
        };

        let prompt = ir_to_prompt(&request);

        assert_eq!(
            prompt,
            vec![ContentBlock::Text(TextContent::new(
                "Explain this repository"
            ))]
        );
    }

    #[test]
    fn ir_to_prompt_renders_multi_message_chat_as_marked_transcript() {
        let request = ChatRequest {
            model: "acp:claude".to_string(),
            messages: vec![
                Message::text(Role::System, "Answer tersely."),
                Message::text(Role::User, "What changed?"),
                Message::new(
                    Role::Assistant,
                    vec![Content::text("I need to inspect the diff.")],
                ),
                Message::text(Role::User, "Focus on risks."),
            ],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: false,
        };

        let prompt = ir_to_prompt(&request);

        assert_eq!(
            prompt,
            vec![ContentBlock::Text(TextContent::new(
                "System: Answer tersely.\n\nUser: What changed?\n\nAssistant: I need to inspect the diff.\n\nUser: Focus on risks."
            ))]
        );
    }

    #[test]
    fn session_update_text_extracts_agent_text_chunks_only() {
        let update = SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
            TextContent::new("partial answer"),
        )));

        assert_eq!(
            session_update_text(&update).as_deref(),
            Some("partial answer")
        );

        let thought = SessionUpdate::AgentThoughtChunk(ContentChunk::new(ContentBlock::Text(
            TextContent::new("hidden reasoning"),
        )));

        assert_eq!(session_update_text(&thought), None);
    }

    #[tokio::test]
    async fn complete_with_agent_returns_ir_response_from_acp_session_updates() {
        let agent = Agent
            .builder()
            .on_receive_request(
                async |initialize: InitializeRequest, responder, _connection| {
                    assert_eq!(initialize.protocol_version, ProtocolVersion::V1);
                    responder.respond(
                        InitializeResponse::new(initialize.protocol_version)
                            .agent_capabilities(AgentCapabilities::new()),
                    )
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async |session: NewSessionRequest, responder, _connection| {
                    assert!(session.cwd.is_absolute());
                    responder.respond(NewSessionResponse::new("test-session"))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async |prompt: PromptRequest, responder, connection: ConnectionTo<Client>| {
                    assert_eq!(prompt.session_id.to_string(), "test-session");
                    assert_eq!(
                        prompt.prompt,
                        vec![ContentBlock::Text(TextContent::new("Summarize"))]
                    );

                    connection.send_notification(SessionNotification::new(
                        prompt.session_id.clone(),
                        SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                            TextContent::new("ACP answer"),
                        ))),
                    ))?;

                    responder.respond(PromptResponse::new(StopReason::EndTurn))
                },
                agent_client_protocol::on_receive_request!(),
            );

        let response = complete_with_agent(
            agent,
            std::env::current_dir().expect("current dir"),
            ChatRequest {
                model: "acp:fake".to_string(),
                messages: vec![Message::text(Role::User, "Summarize")],
                max_tokens: None,
                temperature: None,
                top_p: None,
                stream: false,
            },
        )
        .await
        .expect("agent completion");

        assert_eq!(response.model, "acp:fake");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            response.choices[0].message,
            Message::text(Role::Assistant, "ACP answer")
        );
    }

    #[tokio::test]
    async fn stream_with_agent_yields_text_chunks_before_stop() {
        let agent = Agent
            .builder()
            .on_receive_request(
                async |initialize: InitializeRequest, responder, _connection| {
                    responder.respond(
                        InitializeResponse::new(initialize.protocol_version)
                            .agent_capabilities(AgentCapabilities::new()),
                    )
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async |_session: NewSessionRequest, responder, _connection| {
                    responder.respond(NewSessionResponse::new("stream-session"))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async |prompt: PromptRequest, responder, connection: ConnectionTo<Client>| {
                    connection.send_notification(SessionNotification::new(
                        prompt.session_id.clone(),
                        SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                            TextContent::new("hel"),
                        ))),
                    ))?;
                    connection.send_notification(SessionNotification::new(
                        prompt.session_id.clone(),
                        SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                            TextContent::new("lo"),
                        ))),
                    ))?;

                    responder.respond(PromptResponse::new(StopReason::EndTurn))
                },
                agent_client_protocol::on_receive_request!(),
            );

        let mut stream = stream_with_agent(
            agent,
            std::env::current_dir().expect("current dir"),
            ChatRequest {
                model: "acp:fake".to_string(),
                messages: vec![Message::text(Role::User, "Stream")],
                max_tokens: None,
                temperature: None,
                top_p: None,
                stream: true,
            },
        );

        assert_eq!(
            stream.next().await.transpose().expect("first event"),
            Some(AcpStreamEvent::TextDelta("hel".to_string()))
        );
        assert_eq!(
            stream.next().await.transpose().expect("second event"),
            Some(AcpStreamEvent::TextDelta("lo".to_string()))
        );
        assert_eq!(
            stream.next().await.transpose().expect("stop event"),
            Some(AcpStreamEvent::Stop {
                finish_reason: "stop".to_string()
            })
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn agent_for_backend_exposes_yallm_as_acp_downstream_agent() {
        let seen_requests = Arc::new(Mutex::new(Vec::new()));
        let seen_requests_for_backend = seen_requests.clone();
        let agent = agent_for_backend("test-model", move |request: ChatRequest| {
            let seen_requests = seen_requests_for_backend.clone();
            async move {
                seen_requests.lock().await.push(request.clone());
                Ok(ChatResponse {
                    id: "backend_response".to_string(),
                    model: request.model,
                    choices: vec![Choice {
                        index: 0,
                        message: Message::text(Role::Assistant, "backend answer"),
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: None,
                })
            }
        });

        let streamed_text = Arc::new(Mutex::new(String::new()));
        let streamed_text_for_handler = streamed_text.clone();

        Client
            .builder()
            .on_receive_notification(
                async move |notification: SessionNotification, _connection| {
                    if let Some(text) = session_update_text(&notification.update) {
                        streamed_text_for_handler.lock().await.push_str(&text);
                    }
                    Ok(())
                },
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(agent, async |connection| {
                connection
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                let session = connection
                    .send_request(NewSessionRequest::new(
                        std::env::current_dir().expect("current dir"),
                    ))
                    .block_task()
                    .await?;

                let response = connection
                    .send_request(PromptRequest::new(
                        session.session_id,
                        vec![ContentBlock::Text(TextContent::new("Hello ACP"))],
                    ))
                    .block_task()
                    .await?;

                assert_eq!(response.stop_reason, StopReason::EndTurn);
                Ok(())
            })
            .await
            .expect("downstream acp turn");

        let requests = seen_requests.lock().await;
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "test-model");
        assert_eq!(
            requests[0].messages,
            vec![Message::text(Role::User, "Hello ACP")]
        );
        assert_eq!(streamed_text.lock().await.as_str(), "backend answer");
    }
}
