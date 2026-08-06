//! yallm-server: the HTTP server that proxies and converts LLM APIs.
//!
//! axum-based. Entry points:
//! - [`app()`] / [`app_with_state(state)`](app_with_state) — the router.
//!   Route table: OpenAI `/v1/chat/completions`, `/v1/models`; Anthropic
//!   `/v1/messages`; Ollama `/api/chat`; Responses/Conversations
//!   `/v1/responses*`, `/v1/conversations*`; monitoring `/dashboard*`;
//!   `/health`, `/` (info). All routes sit behind the `logging::log_http`
//!   middleware (request ids, JSON logs, monitor events) and a permissive
//!   CORS layer.
//! - [`run(config)`](run) — bind + serve with optional TLS (rustls; partial
//!   TLS config is rejected), constructing `AppState` from the environment.
//! - [`ServerConfig`] — addr, TLS paths, LiteLLM config path.
//!
//! Modules: `proxy` (routing + upstream calls), `proxy::stream`
//! (stream translation), `state` (config/transport), `responses_routes`
//! (Responses/Conversations API), `logging` (middleware), `dashboard`
//! (monitoring UI), `tls` (acceptor), `routes` (chat/messages/models).
//! Everything public is re-exported at the crate root.

#![warn(missing_docs)]

use axum::{
    Router,
    routing::{get, post},
};
use std::{net::SocketAddr, path::PathBuf};
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};

mod dashboard;
mod logging;
mod proxy;
mod responses_routes;
mod routes;
mod state;
mod tls;

pub use dashboard::*;
pub use logging::*;
pub use proxy::*;
pub use responses_routes::*;
pub use routes::*;
pub use state::*;
pub use tls::*;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bind address (host:port)
    pub addr: SocketAddr,
    /// PEM certificate path; must be set together with `tls_key`
    pub tls_cert: Option<String>,
    /// PEM private key path; must be set together with `tls_cert`
    pub tls_key: Option<String>,
    /// Optional LiteLLM config.yaml path (loaded by `yallm-config`)
    pub litellm_config: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: SocketAddr::from(([127, 0, 0, 1], 4000)),
            tls_cert: None,
            tls_key: None,
            litellm_config: None,
        }
    }
}

impl ServerConfig {
    /// True when both cert and key are present (partial config is rejected
    /// by [`run`]).
    pub fn tls_enabled(&self) -> bool {
        self.tls_cert.is_some() && self.tls_key.is_some()
    }
}

/// Create the application router
pub fn app() -> Router {
    app_with_state(AppState::default())
}

/// Create the application router with a provided state (useful for tests).
pub fn app_with_state(state: AppState) -> Router {
    Router::new()
        .route("/", get(routes::root))
        .route("/health", get(routes::health))
        .route("/dashboard", get(dashboard::dashboard_page))
        .route(
            "/dashboard/api/events",
            get(dashboard::dashboard_events).delete(dashboard::dashboard_clear),
        )
        .route("/dashboard/assets/main.js", get(dashboard::dashboard_js))
        .route(
            "/dashboard/assets/styles.css",
            get(dashboard::dashboard_css),
        )
        .route("/v1/models", get(routes::models_list))
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
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            logging::log_http,
        ))
        .with_state(state)
}

/// Run the server with the given configuration
pub async fn run(config: ServerConfig) -> std::io::Result<()> {
    // Reject partial TLS config; silent HTTP fallback is a footgun in prod.
    match (config.tls_cert.as_deref(), config.tls_key.as_deref()) {
        (Some(_), None) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--tls-cert was set but --tls-key was not; both are required for HTTPS",
            ));
        }
        (None, Some(_)) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--tls-key was set but --tls-cert was not; both are required for HTTPS",
            ));
        }
        _ => {}
    }

    rustls::crypto::ring::default_provider()
        .install_default()
        .ok();
    let loaded_config = yallm_config::load_with_options(yallm_config::LoadOptions {
        litellm_config: config.litellm_config.clone(),
    });
    let app = app_with_state(AppState::from_loaded_config(loaded_config));
    let addr = config.addr;

    if config.tls_enabled() {
        let cert = config.tls_cert.as_ref().unwrap();
        let key = config.tls_key.as_ref().unwrap();
        let acceptor = tls_acceptor(cert, key)?;
        let listener = TlsListener::bind(addr, acceptor).await?;
        tracing::info!("Server listening on https://{addr}");
        axum::serve(listener, app).await
    } else {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!("Server listening on http://{addr}");
        axum::serve(listener, app).await
    }
}
