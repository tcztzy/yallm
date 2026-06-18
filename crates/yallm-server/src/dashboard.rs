use axum::{
    Json,
    extract::{Query, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;

use crate::state::AppState;

#[derive(Debug, Deserialize, Default)]
pub struct DashboardEventsQuery {
    pub limit: Option<usize>,
}

pub async fn dashboard_page() -> Html<&'static str> {
    Html(include_str!("../dashboard/dist/index.html"))
}

pub async fn dashboard_js() -> Response {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("../dashboard/dist/assets/main.js"),
    )
        .into_response()
}

pub async fn dashboard_css() -> Response {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../dashboard/dist/assets/styles.css"),
    )
        .into_response()
}

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
