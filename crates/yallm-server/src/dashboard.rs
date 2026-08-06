//! Monitoring dashboard: static UI + events API.
//!
//! The SPA lives in `crates/yallm-server/dashboard/` (Vite build) and is
//! embedded at compile time via `include_str!` — the served HTML/JS/CSS
//! assets are the build artifacts under `dashboard/dist/`, so the dashboard
//! must be rebuilt before changes to it appear in the server.
//!
//! Data sources: [`dashboard_events`] reads monitor events from the store
//! (written by the `logging::log_http` middleware and the proxy layer),
//! `dashboard_clear` wipes them. No event data is kept in memory here.

use axum::{
    Json,
    extract::{Query, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;

use crate::state::AppState;

/// Query parameters for `GET /dashboard/api/events`.
#[derive(Debug, Deserialize, Default)]
pub struct DashboardEventsQuery {
    /// Maximum number of events to return (default 500)
    pub limit: Option<usize>,
}

/// Serve the SPA shell (embedded `dashboard/dist/index.html`).
pub async fn dashboard_page() -> Html<&'static str> {
    Html(include_str!("../dashboard/dist/index.html"))
}

/// Serve the embedded JS bundle.
pub async fn dashboard_js() -> Response {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("../dashboard/dist/assets/main.js"),
    )
        .into_response()
}

/// Serve the embedded CSS bundle.
pub async fn dashboard_css() -> Response {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../dashboard/dist/assets/styles.css"),
    )
        .into_response()
}

/// List monitor events (newest first) from the store.
pub async fn dashboard_events(
    State(state): State<AppState>,
    Query(query): Query<DashboardEventsQuery>,
) -> Response {
    match state
        .store
        .list_monitor_events(query.limit.unwrap_or(500))
        .await
    {
        Ok(events) => (
            StatusCode::OK,
            Json(json!({
                "object": "list",
                "data": events,
            })),
        )
            .into_response(),
        Err(err) => store_error_to_response(err),
    }
}

/// Delete all monitor events; returns the count deleted.
pub async fn dashboard_clear(State(state): State<AppState>) -> Response {
    match state.store.clear_monitor_events().await {
        Ok(deleted) => (
            StatusCode::OK,
            Json(json!({
                "object": "monitor.deleted",
                "deleted": deleted,
            })),
        )
            .into_response(),
        Err(err) => store_error_to_response(err),
    }
}

fn store_error_to_response(err: yallm_storage::StoreError) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "message": err.to_string(),
                "type": "storage_error",
            }
        })),
    )
        .into_response()
}
