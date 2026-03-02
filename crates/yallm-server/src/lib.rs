//! yallm-server: HTTP server for LLM API conversion proxy
//!
//! This crate provides an axum-based HTTP server that proxies and converts
//! between different LLM API formats (OpenAI, Anthropic, Ollama).

use axum::{
    Router,
    routing::{get, post},
};
use std::net::SocketAddr;
use tokio::net::TcpListener;

mod logging;
mod proxy;
mod responses_routes;
mod routes;
mod state;

pub use logging::*;
pub use proxy::*;
pub use responses_routes::*;
pub use routes::*;
pub use state::*;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub addr: SocketAddr,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: SocketAddr::from(([127, 0, 0, 1], 8080)),
        }
    }
}

/// Create the application router
pub fn app() -> Router {
    app_with_state(AppState::default())
}

/// Create the application router with a provided state (useful for tests).
pub fn app_with_state(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/messages", post(routes::anthropic_messages))
        .route("/api/chat", post(routes::ollama_chat))
        .route("/v1/responses", post(responses_routes::responses_create))
        .route(
            "/v1/responses/input_tokens",
            post(responses_routes::responses_input_tokens),
        )
        .route(
            "/v1/responses/compact",
            post(responses_routes::responses_compact),
        )
        .route(
            "/v1/responses/{response_id}",
            get(responses_routes::responses_get).delete(responses_routes::responses_delete),
        )
        .route(
            "/v1/responses/{response_id}/cancel",
            post(responses_routes::responses_cancel),
        )
        .route(
            "/v1/responses/{response_id}/input_items",
            get(responses_routes::responses_input_items),
        )
        .route(
            "/v1/conversations",
            post(responses_routes::conversations_create),
        )
        .route(
            "/v1/conversations/{conversation_id}",
            get(responses_routes::conversations_get)
                .post(responses_routes::conversations_update)
                .delete(responses_routes::conversations_delete),
        )
        .route(
            "/v1/conversations/{conversation_id}/items",
            post(responses_routes::conversation_items_create)
                .get(responses_routes::conversation_items_list),
        )
        .route(
            "/v1/conversations/{conversation_id}/items/{item_id}",
            get(responses_routes::conversation_item_get)
                .delete(responses_routes::conversation_item_delete),
        )
        .fallback(routes::fallback)
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            logging::log_http,
        ))
        .with_state(state)
}

/// Run the server with the given configuration
pub async fn run(config: ServerConfig) -> std::io::Result<()> {
    let app = app();
    let listener = TcpListener::bind(config.addr).await?;
    tracing::info!("Server listening on {}", config.addr);
    axum::serve(listener, app).await
}
