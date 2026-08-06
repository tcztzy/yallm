//! Local persistence for yallm: conversations, responses, monitor events.
//!
//! [`LocalStore`] is the single storage handle used by `yallm-server`. Two
//! backends behind one interface:
//! - **SQLite** (default): `YALLM_DB_URL=sqlite:///path/to/yallm.sqlite3`,
//!   WAL mode, opened synchronously ([`LocalStore::open_database_url_sync`])
//!   because it is constructed during server startup.
//! - **Legacy JSON file**: `YALLM_STORAGE_PATH` (deprecated, warns).
//!
//! What is stored:
//! - Responses & conversations (OpenAI Responses API objects as JSON, plus
//!   the SSE event list for replay on `stream=true` GETs).
//! - Context items (`resolve_context`) — the conversation history assembled
//!   for a new response, keyed by conversation id or previous response id.
//! - Monitor events — one row per proxied HTTP request, read by the
//!   `/dashboard/api/events` endpoint; capped with `list_monitor_events`.
//!
//! Concurrency: SQLite connections are wrapped in a `Mutex` — safe to share
//! across tasks, but writes serialize. WAL keeps readers non-blocking.
//!
//! Gotcha: opening the DB eagerly at state construction means tests must
//! pass a unique `YALLM_DB_URL` (see `crates/yallm/tests/acp_roundtrip.rs`
//! for the pid+counter temp-file pattern).

#![warn(missing_docs)]

use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::RwLock;

/// Deprecated legacy JSON-store env var; use [`DB_URL_ENV`].
pub const STORAGE_PATH_ENV: &str = "YALLM_STORAGE_PATH";
/// SQLite URL env var (`sqlite:///path/to/yallm.sqlite3`).
pub const DB_URL_ENV: &str = "YALLM_DB_URL";
const MAX_MONITOR_EVENTS: usize = 2_000;

#[derive(Debug)]
/// Storage failure kinds.
pub enum StoreError {
    /// Underlying I/O or SQLite error
    Io(std::io::Error),
    /// Invalid argument or state
    Invalid(String),
    /// Requested entity does not exist
    NotFound(String),
    /// Constraint conflict (e.g. duplicate id)
    Conflict(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Io(e) => write!(f, "{e}"),
            StoreError::Invalid(msg) => write!(f, "{msg}"),
            StoreError::NotFound(msg) => write!(f, "{msg}"),
            StoreError::Conflict(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<std::io::Error> for StoreError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(value: serde_json::Error) -> Self {
        Self::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, value))
    }
}

/// Convenience alias for store operations.
pub type StoreResult<T> = Result<T, StoreError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// List ordering.
pub enum ListOrder {
    /// Oldest first
    Asc,
    /// Newest first
    Desc,
}

#[derive(Debug, Clone)]
/// Cursor-paginated list result.
pub struct ListPage<T> {
    /// Items in this page
    pub data: Vec<T>,
    /// Id of the first item (cursor)
    pub first_id: Option<String>,
    /// Id of the last item (cursor)
    pub last_id: Option<String>,
    /// Whether more items exist after this page
    pub has_more: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// One recorded HTTP request (dashboard data).
pub struct MonitorEvent {
    /// Server-assigned request id
    pub request_id: u64,
    /// Unix milliseconds at completion
    pub timestamp_ms: u64,
    /// HTTP method
    pub method: String,
    /// Request path (with query)
    pub uri: String,
    /// Normalized route pattern
    pub endpoint: String,
    /// Response status
    pub status: u16,
    /// Total handling time
    pub latency_ms: u64,
    /// Request body size
    pub request_bytes: u64,
    /// Response body size
    pub response_bytes: u64,
    /// Upstream provider, when proxied
    pub provider: Option<String>,
    /// Model as requested by the client
    pub model: Option<String>,
    /// Model sent to the upstream
    pub upstream_model: Option<String>,
    /// Upstream URL, when proxied
    pub upstream_url: Option<String>,
    /// Whether the request was a stream
    pub stream: bool,
}

#[derive(Clone)]
struct SqliteStore {
    url: String,
    inner: Arc<Mutex<Connection>>,
}

impl fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SqliteStore")
            .field("url", &self.url)
            .finish_non_exhaustive()
    }
}

impl SqliteStore {
    fn open_sync(url: Option<&str>) -> StoreResult<Self> {
        let url = url
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(default_db_url);
        let conn = open_db_connection(&url)?;
        init_db_schema(&conn)?;
        Ok(Self {
            url,
            inner: Arc::new(Mutex::new(conn)),
        })
    }

    fn url(&self) -> &str {
        &self.url
    }
}

#[derive(Debug, Clone)]
/// Resolved conversation context for a new response.
pub struct ContextItems {
    /// Conversation the context came from, when any
    pub conversation_id: Option<String>,
    /// Context items (history) to feed the model
    pub items: Vec<Value>,
}

#[derive(Debug, Clone)]
/// Payload for [`LocalStore::save_response`].
pub struct SaveResponseRequest {
    /// The Responses API object to persist
    pub response: Value,
    /// Input items of the request (for `/input_items`)
    pub request_input_items: Vec<Value>,
    /// Conversation this response belongs to
    pub conversation_id: Option<String>,
    /// Previous response id, when chained
    pub previous_response_id: Option<String>,
    /// Provider that produced it
    pub provider: String,
    /// SSE events for stream replay
    pub stream_events: Vec<String>,
}

#[derive(Debug, Clone)]
/// Result of saving a response.
pub struct SavedResponse {
    /// The stored response object
    pub response: Value,
    /// Conversation id it was attached to
    pub conversation_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConversationRecord {
    id: String,
    created_at: u64,
    metadata: Value,
    deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ItemRecord {
    id: String,
    conversation_id: Option<String>,
    item: Value,
    deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponseRecord {
    id: String,
    response: Value,
    conversation_id: Option<String>,
    previous_response_id: Option<String>,
    provider: String,
    input_item_ids: Vec<String>,
    output_item_ids: Vec<String>,
    context_item_ids: Vec<String>,
    stream_events: Vec<String>,
    deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoreFile {
    version: u32,
    next_seq: u64,
    conversations: BTreeMap<String, ConversationRecord>,
    items: BTreeMap<String, ItemRecord>,
    conversation_item_order: BTreeMap<String, Vec<String>>,
    responses: BTreeMap<String, ResponseRecord>,
}

impl Default for StoreFile {
    fn default() -> Self {
        Self {
            version: 1,
            next_seq: 0,
            conversations: BTreeMap::new(),
            items: BTreeMap::new(),
            conversation_item_order: BTreeMap::new(),
            responses: BTreeMap::new(),
        }
    }
}

#[derive(Clone)]
/// Storage handle: conversations, responses, monitor events.
pub struct LocalStore {
    backend: StoreBackend,
}

#[derive(Clone)]
enum StoreBackend {
    Json(JsonStore),
    Sqlite(SqliteStore),
}

#[derive(Clone)]
struct JsonStore {
    path: PathBuf,
    inner: Arc<RwLock<StoreFile>>,
}

impl fmt::Debug for LocalStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.backend {
            StoreBackend::Json(store) => f
                .debug_struct("LocalStore")
                .field("path", &store.path)
                .finish_non_exhaustive(),
            StoreBackend::Sqlite(store) => f
                .debug_struct("LocalStore")
                .field("url", &store.url())
                .finish_non_exhaustive(),
        }
    }
}

impl LocalStore {
    /// Open the SQLite store synchronously (startup path). `None` = default
    /// path (see [`default_db_url`]); `sqlite::memory:` for a throwaway store.
    ///
    /// ```
    /// let store = yallm_storage::LocalStore::open_database_url_sync(
    ///     Some("sqlite::memory:"),
    /// )
    /// .expect("in-memory store");
    /// assert_eq!(store.location(), "sqlite::memory:");
    /// ```
    pub fn open_database_url_sync(url: Option<&str>) -> StoreResult<Self> {
        Ok(Self {
            backend: StoreBackend::Sqlite(SqliteStore::open_sync(url)?),
        })
    }

    /// Open the legacy JSON-file store synchronously (deprecated path).
    pub fn open_sync(path: Option<PathBuf>) -> StoreResult<Self> {
        let path = path.unwrap_or_else(default_storage_path);
        let data = load_store_file_sync(&path)?;
        Ok(Self {
            backend: StoreBackend::Json(JsonStore {
                path,
                inner: Arc::new(RwLock::new(data)),
            }),
        })
    }

    /// Open the legacy JSON-file store asynchronously (deprecated path).
    pub async fn open(path: Option<PathBuf>) -> StoreResult<Self> {
        let path = path.unwrap_or_else(default_storage_path);
        let data = load_store_file(&path).await?;
        Ok(Self {
            backend: StoreBackend::Json(JsonStore {
                path,
                inner: Arc::new(RwLock::new(data)),
            }),
        })
    }

    /// Backend location string (db path or JSON path).
    pub fn location(&self) -> &str {
        match &self.backend {
            StoreBackend::Json(store) => store.path.to_str().unwrap_or("<non-utf8-path>"),
            StoreBackend::Sqlite(store) => store.url(),
        }
    }

    /// Create a conversation; returns the stored conversation object.
    pub async fn create_conversation(
        &self,
        metadata: Option<Value>,
        items: Vec<Value>,
    ) -> StoreResult<Value> {
        self.apply_mutation(|store| {
            let conv_id = next_id(store, "conv");
            let conv = ConversationRecord {
                id: conv_id.clone(),
                created_at: unix_seconds(),
                metadata: normalize_metadata(metadata),
                deleted: false,
            };
            store.conversations.insert(conv_id.clone(), conv);
            store
                .conversation_item_order
                .insert(conv_id.clone(), Vec::new());
            let page = append_items(store, Some(&conv_id), items, "user");
            let _ = page;
            Ok(conversation_json(
                store
                    .conversations
                    .get(&conv_id)
                    .expect("conversation exists"),
            ))
        })
        .await
    }

    /// Fetch a conversation by id.
    pub async fn get_conversation(&self, conversation_id: &str) -> StoreResult<Option<Value>> {
        let store = self.read_snapshot().await?;
        let Some(conv) = store.conversations.get(conversation_id) else {
            return Ok(None);
        };
        if conv.deleted {
            return Ok(None);
        }
        Ok(Some(conversation_json(conv)))
    }

    /// Update conversation metadata; `NotFound` when absent.
    pub async fn update_conversation(
        &self,
        conversation_id: &str,
        metadata: Value,
    ) -> StoreResult<Option<Value>> {
        self.apply_mutation(|store| {
            let Some(conv) = store.conversations.get_mut(conversation_id) else {
                return Ok(None);
            };
            if conv.deleted {
                return Ok(None);
            }
            conv.metadata = normalize_metadata(Some(metadata));
            Ok(Some(conversation_json(conv)))
        })
        .await
    }

    /// Delete a conversation; returns the deleted object when present.
    pub async fn delete_conversation(&self, conversation_id: &str) -> StoreResult<Option<Value>> {
        self.apply_mutation(|store| {
            let Some(conv) = store.conversations.get_mut(conversation_id) else {
                return Ok(None);
            };
            conv.deleted = true;
            Ok(Some(json!({
                "id": conversation_id,
                "object": "conversation.deleted",
                "deleted": true
            })))
        })
        .await
    }

    /// Append items to a conversation.
    pub async fn add_conversation_items(
        &self,
        conversation_id: &str,
        items: Vec<Value>,
    ) -> StoreResult<Option<ListPage<Value>>> {
        self.apply_mutation(|store| {
            let Some(conv) = store.conversations.get(conversation_id) else {
                return Ok(None);
            };
            if conv.deleted {
                return Ok(None);
            }
            let page = append_items(store, Some(conversation_id), items, "user");
            Ok(Some(page))
        })
        .await
    }

    /// List a conversation's items (paginated).
    pub async fn list_conversation_items(
        &self,
        conversation_id: &str,
        limit: usize,
        order: ListOrder,
        after: Option<&str>,
    ) -> StoreResult<Option<ListPage<Value>>> {
        let store = self.read_snapshot().await?;
        let Some(conv) = store.conversations.get(conversation_id) else {
            return Ok(None);
        };
        if conv.deleted {
            return Ok(None);
        }
        let Some(order_ids) = store.conversation_item_order.get(conversation_id) else {
            return Ok(Some(ListPage {
                data: Vec::new(),
                first_id: None,
                last_id: None,
                has_more: false,
            }));
        };
        let ids = visible_ids(&store.items, order_ids);
        let page = paginate_values(&store.items, ids, limit, order, after);
        Ok(Some(page))
    }

    /// Fetch one item by id.
    pub async fn get_conversation_item(
        &self,
        conversation_id: &str,
        item_id: &str,
    ) -> StoreResult<Option<Value>> {
        let store = self.read_snapshot().await?;
        let Some(conv) = store.conversations.get(conversation_id) else {
            return Ok(None);
        };
        if conv.deleted {
            return Ok(None);
        }
        let Some(item) = store.items.get(item_id) else {
            return Ok(None);
        };
        if item.deleted {
            return Ok(None);
        }
        if item.conversation_id.as_deref() != Some(conversation_id) {
            return Ok(None);
        }
        Ok(Some(item.item.clone()))
    }

    /// Delete one item by id.
    pub async fn delete_conversation_item(
        &self,
        conversation_id: &str,
        item_id: &str,
    ) -> StoreResult<Option<Value>> {
        self.apply_mutation(|store| {
            let Some(conv) = store.conversations.get(conversation_id) else {
                return Ok(None);
            };
            if conv.deleted {
                return Ok(None);
            }
            let Some(item) = store.items.get_mut(item_id) else {
                return Ok(None);
            };
            if item.conversation_id.as_deref() != Some(conversation_id) {
                return Ok(None);
            }
            item.deleted = true;
            Ok(Some(conversation_json(conv)))
        })
        .await
    }

    /// Assemble the context for a new response: full conversation items, or the chain starting at `previous_response_id`.
    pub async fn resolve_context(
        &self,
        conversation_id: Option<&str>,
        previous_response_id: Option<&str>,
    ) -> StoreResult<ContextItems> {
        if conversation_id.is_some() && previous_response_id.is_some() {
            return Err(StoreError::Conflict(
                "conversation and previous_response_id cannot be used together".to_string(),
            ));
        }

        let store = self.read_snapshot().await?;

        if let Some(prev_id) = previous_response_id {
            let Some(resp) = store.responses.get(prev_id) else {
                return Err(StoreError::NotFound(format!(
                    "Response '{prev_id}' not found"
                )));
            };
            if resp.deleted {
                return Err(StoreError::NotFound(format!(
                    "Response '{prev_id}' not found"
                )));
            }
            let items = resp
                .context_item_ids
                .iter()
                .filter_map(|id| store.items.get(id))
                .filter(|it| !it.deleted)
                .map(|it| it.item.clone())
                .collect();
            return Ok(ContextItems {
                conversation_id: resp.conversation_id.clone(),
                items,
            });
        }

        if let Some(conv_id) = conversation_id {
            let Some(conv) = store.conversations.get(conv_id) else {
                return Err(StoreError::NotFound(format!(
                    "Conversation '{conv_id}' not found"
                )));
            };
            if conv.deleted {
                return Err(StoreError::NotFound(format!(
                    "Conversation '{conv_id}' not found"
                )));
            }
            let ids = store
                .conversation_item_order
                .get(conv_id)
                .cloned()
                .unwrap_or_default();
            let items = ids
                .into_iter()
                .filter_map(|id| store.items.get(&id))
                .filter(|it| !it.deleted)
                .map(|it| it.item.clone())
                .collect();
            return Ok(ContextItems {
                conversation_id: Some(conv_id.to_string()),
                items,
            });
        }

        Ok(ContextItems {
            conversation_id: None,
            items: Vec::new(),
        })
    }

    /// Persist a response (and its SSE events); returns the stored object.
    pub async fn save_response(&self, req: SaveResponseRequest) -> StoreResult<SavedResponse> {
        self.apply_mutation(|store| {
            let mut response = req.response;
            let response_id = response_id_or_insert(store, &mut response);
            let previous_response_id = req.previous_response_id.clone();

            let conversation_id = if let Some(id) = req.conversation_id.clone() {
                ensure_conversation_exists(store, &id)?;
                Some(id)
            } else if let Some(prev_id) = previous_response_id.as_deref() {
                let Some(prev) = store.responses.get(prev_id) else {
                    return Err(StoreError::NotFound(format!(
                        "Response '{prev_id}' not found"
                    )));
                };
                prev.conversation_id.clone()
            } else {
                let id = next_id(store, "conv");
                let conv = ConversationRecord {
                    id: id.clone(),
                    created_at: unix_seconds(),
                    metadata: json!({}),
                    deleted: false,
                };
                store.conversations.insert(id.clone(), conv);
                store.conversation_item_order.insert(id.clone(), Vec::new());
                Some(id)
            };

            let base_context_ids = if let Some(prev_id) = previous_response_id.as_deref() {
                store
                    .responses
                    .get(prev_id)
                    .map(|r| r.context_item_ids.clone())
                    .unwrap_or_default()
            } else if let Some(conv_id) = conversation_id.as_deref() {
                visible_ids(
                    &store.items,
                    store
                        .conversation_item_order
                        .get(conv_id)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]),
                )
            } else {
                Vec::new()
            };

            let input_page = append_items(
                store,
                conversation_id.as_deref(),
                req.request_input_items,
                "user",
            );
            let input_item_ids: Vec<String> = input_page
                .data
                .iter()
                .filter_map(item_id)
                .collect::<Vec<_>>();

            let output_items = response
                .get("output")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let output_page =
                append_items(store, conversation_id.as_deref(), output_items, "assistant");
            let output_items_normalized = output_page.data.clone();
            let output_item_ids = output_items_normalized
                .iter()
                .filter_map(item_id)
                .collect::<Vec<_>>();

            if let Some(obj) = response.as_object_mut() {
                obj.insert("output".to_string(), Value::Array(output_items_normalized));
                obj.insert(
                    "previous_response_id".to_string(),
                    previous_response_id
                        .as_ref()
                        .map(|s| Value::String(s.clone()))
                        .unwrap_or(Value::Null),
                );
                obj.insert(
                    "conversation".to_string(),
                    conversation_id
                        .as_ref()
                        .map(|id| json!({"id": id}))
                        .unwrap_or(Value::Null),
                );
            }

            let mut context_item_ids = base_context_ids;
            context_item_ids.extend(input_item_ids.iter().cloned());
            context_item_ids.extend(output_item_ids.iter().cloned());

            let record = ResponseRecord {
                id: response_id.clone(),
                response: response.clone(),
                conversation_id: conversation_id.clone(),
                previous_response_id,
                provider: req.provider,
                input_item_ids,
                output_item_ids,
                context_item_ids,
                stream_events: req.stream_events,
                deleted: false,
            };
            store.responses.insert(response_id, record);

            Ok(SavedResponse {
                response,
                conversation_id,
            })
        })
        .await
    }

    /// Fetch a stored response by id.
    pub async fn get_response(&self, response_id: &str) -> StoreResult<Option<Value>> {
        let store = self.read_snapshot().await?;
        let Some(resp) = store.responses.get(response_id) else {
            return Ok(None);
        };
        if resp.deleted {
            return Ok(None);
        }
        Ok(Some(resp.response.clone()))
    }

    /// Fetch the stored SSE events for a response (stream replay).
    pub async fn get_response_stream_events(
        &self,
        response_id: &str,
    ) -> StoreResult<Option<Vec<String>>> {
        let store = self.read_snapshot().await?;
        let Some(resp) = store.responses.get(response_id) else {
            return Ok(None);
        };
        if resp.deleted {
            return Ok(None);
        }
        if resp.stream_events.is_empty() {
            return Ok(None);
        }
        Ok(Some(resp.stream_events.clone()))
    }

    /// Delete a response; returns the deleted object when present.
    pub async fn delete_response(&self, response_id: &str) -> StoreResult<Option<Value>> {
        self.apply_mutation(|store| {
            let Some(resp) = store.responses.get_mut(response_id) else {
                return Ok(None);
            };
            resp.deleted = true;
            Ok(Some(json!({
                "id": response_id,
                "object": "response",
                "deleted": true
            })))
        })
        .await
    }

    /// Mark a response cancelled (status update).
    pub async fn cancel_response(&self, response_id: &str) -> StoreResult<Option<Value>> {
        self.apply_mutation(|store| {
            let Some(resp) = store.responses.get_mut(response_id) else {
                return Ok(None);
            };
            if let Some(obj) = resp.response.as_object_mut() {
                obj.insert("status".to_string(), Value::String("cancelled".to_string()));
            }
            Ok(Some(resp.response.clone()))
        })
        .await
    }

    /// List the stored input items of a response.
    pub async fn list_response_input_items(
        &self,
        response_id: &str,
        limit: usize,
        order: ListOrder,
        after: Option<&str>,
    ) -> StoreResult<Option<ListPage<Value>>> {
        let store = self.read_snapshot().await?;
        let Some(resp) = store.responses.get(response_id) else {
            return Ok(None);
        };
        if resp.deleted {
            return Ok(None);
        }
        let page = paginate_values(
            &store.items,
            resp.input_item_ids.clone(),
            limit,
            order,
            after,
        );
        Ok(Some(page))
    }

    /// Persist one monitor event.
    pub async fn record_monitor_event(&self, event: MonitorEvent) -> StoreResult<()> {
        match &self.backend {
            StoreBackend::Json(_) => Ok(()),
            StoreBackend::Sqlite(store) => {
                let inner = store.inner.clone();
                spawn_db_task(move || {
                    let conn = lock_db_connection(&inner)?;
                    insert_monitor_event(&conn, event)?;
                    trim_monitor_events(&conn)?;
                    Ok(())
                })
                .await
            }
        }
    }

    /// List monitor events, newest first, capped at `limit`.
    pub async fn list_monitor_events(&self, limit: usize) -> StoreResult<Vec<MonitorEvent>> {
        match &self.backend {
            StoreBackend::Json(_) => Ok(Vec::new()),
            StoreBackend::Sqlite(store) => {
                let inner = store.inner.clone();
                spawn_db_task(move || {
                    let conn = lock_db_connection(&inner)?;
                    list_monitor_events(&conn, limit)
                })
                .await
            }
        }
    }

    /// Delete all monitor events; returns how many were removed.
    pub async fn clear_monitor_events(&self) -> StoreResult<usize> {
        match &self.backend {
            StoreBackend::Json(_) => Ok(0),
            StoreBackend::Sqlite(store) => {
                let inner = store.inner.clone();
                spawn_db_task(move || {
                    let conn = lock_db_connection(&inner)?;
                    clear_monitor_events(&conn)
                })
                .await
            }
        }
    }

    async fn read_snapshot(&self) -> StoreResult<StoreFile> {
        match &self.backend {
            StoreBackend::Json(store) => Ok(store.inner.read().await.clone()),
            StoreBackend::Sqlite(store) => {
                let conn = lock_db_connection(&store.inner)?;
                load_store_file_from_db(&conn)
            }
        }
    }

    async fn apply_mutation<R, F>(&self, f: F) -> StoreResult<R>
    where
        F: FnOnce(&mut StoreFile) -> StoreResult<R>,
    {
        match &self.backend {
            StoreBackend::Json(store) => {
                let (result, snapshot) = {
                    let mut store_file = store.inner.write().await;
                    let result = f(&mut store_file)?;
                    (result, store_file.clone())
                };
                persist_store_file(&store.path, &snapshot).await?;
                Ok(result)
            }
            StoreBackend::Sqlite(store) => {
                let mut conn = lock_db_connection(&store.inner)?;
                let tx = conn
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(sqlite_error)?;
                let mut store_file = load_store_file_from_db(&tx)?;
                let result = f(&mut store_file)?;
                persist_store_file_to_db(&tx, &store_file)?;
                tx.commit().map_err(sqlite_error)?;
                Ok(result)
            }
        }
    }
}

fn normalize_metadata(metadata: Option<Value>) -> Value {
    match metadata {
        Some(Value::Object(obj)) => Value::Object(obj),
        _ => json!({}),
    }
}

fn ensure_conversation_exists(store: &StoreFile, conversation_id: &str) -> StoreResult<()> {
    let Some(conv) = store.conversations.get(conversation_id) else {
        return Err(StoreError::NotFound(format!(
            "Conversation '{conversation_id}' not found"
        )));
    };
    if conv.deleted {
        return Err(StoreError::NotFound(format!(
            "Conversation '{conversation_id}' not found"
        )));
    }
    Ok(())
}

fn conversation_json(conv: &ConversationRecord) -> Value {
    json!({
        "id": conv.id,
        "object": "conversation",
        "created_at": conv.created_at,
        "metadata": conv.metadata
    })
}

fn append_items(
    store: &mut StoreFile,
    conversation_id: Option<&str>,
    items: Vec<Value>,
    default_role: &str,
) -> ListPage<Value> {
    let mut inserted = Vec::with_capacity(items.len());
    for raw_item in items {
        let item_id = item_id(&raw_item).unwrap_or_else(|| next_id(store, "msg"));
        if let Some(existing) = store.items.get(&item_id)
            && !existing.deleted
        {
            inserted.push(existing.item.clone());
            continue;
        }

        let normalized = normalize_item(raw_item, &item_id, default_role);
        let record = ItemRecord {
            id: item_id.clone(),
            conversation_id: conversation_id.map(ToString::to_string),
            item: normalized.clone(),
            deleted: false,
        };
        store.items.insert(item_id.clone(), record);
        if let Some(conv_id) = conversation_id {
            store
                .conversation_item_order
                .entry(conv_id.to_string())
                .or_default()
                .push(item_id);
        }
        inserted.push(normalized);
    }

    ListPage {
        first_id: inserted.first().and_then(item_id),
        last_id: inserted.last().and_then(item_id),
        has_more: false,
        data: inserted,
    }
}

fn normalize_item(item: Value, item_id: &str, default_role: &str) -> Value {
    let mut item = if item.is_object() {
        item
    } else {
        json!({
            "type": "message",
            "role": default_role,
            "content": [{"type": "input_text", "text": scalar_to_text(&item)}]
        })
    };

    let obj = item
        .as_object_mut()
        .expect("item object should be object after normalization");
    obj.entry("id".to_string())
        .or_insert_with(|| Value::String(item_id.to_string()));
    obj.entry("type".to_string())
        .or_insert_with(|| Value::String("message".to_string()));
    obj.entry("status".to_string())
        .or_insert_with(|| Value::String("completed".to_string()));
    if !obj.contains_key("role") {
        obj.insert("role".to_string(), Value::String(default_role.to_string()));
    }
    item
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

fn item_id(item: &Value) -> Option<String> {
    item.get("id")
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn response_id_or_insert(store: &mut StoreFile, response: &mut Value) -> String {
    if let Some(id) = response
        .get("id")
        .and_then(Value::as_str)
        .map(ToString::to_string)
    {
        return id;
    }
    let id = next_id(store, "resp");
    if let Some(obj) = response.as_object_mut() {
        obj.insert("id".to_string(), Value::String(id.clone()));
    }
    id
}

fn visible_ids(items: &BTreeMap<String, ItemRecord>, ids: &[String]) -> Vec<String> {
    ids.iter()
        .filter_map(|id| items.get(id).map(|it| (id, it)))
        .filter(|(_, it)| !it.deleted)
        .map(|(id, _)| id.clone())
        .collect()
}

fn paginate_values(
    item_map: &BTreeMap<String, ItemRecord>,
    mut ids: Vec<String>,
    limit: usize,
    order: ListOrder,
    after: Option<&str>,
) -> ListPage<Value> {
    if matches!(order, ListOrder::Desc) {
        ids.reverse();
    }

    let start_index = after
        .and_then(|after_id| ids.iter().position(|id| id == after_id))
        .map(|idx| idx + 1)
        .unwrap_or(0);

    let limit = limit.min(100);
    let slice = if start_index >= ids.len() {
        Vec::new()
    } else {
        ids[start_index..].to_vec()
    };
    let has_more = slice.len() > limit;
    let page_ids = slice.into_iter().take(limit).collect::<Vec<_>>();

    let data = page_ids
        .iter()
        .filter_map(|id| item_map.get(id))
        .filter(|item| !item.deleted)
        .map(|item| item.item.clone())
        .collect::<Vec<_>>();

    ListPage {
        first_id: page_ids.first().cloned(),
        last_id: page_ids.last().cloned(),
        has_more,
        data,
    }
}

fn next_id(store: &mut StoreFile, prefix: &str) -> String {
    store.next_seq = store.next_seq.saturating_add(1);
    format!("{prefix}_{:x}{:x}", unix_seconds(), store.next_seq)
}

fn open_db_connection(url: &str) -> StoreResult<Connection> {
    let conn = match sqlite_path_from_url(url)? {
        None => Connection::open_in_memory().map_err(sqlite_error)?,
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Connection::open(path).map_err(sqlite_error)?
        }
    };
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(sqlite_error)?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(sqlite_error)?;
    conn.pragma_update(None, "synchronous", "NORMAL")
        .map_err(sqlite_error)?;
    Ok(conn)
}

fn sqlite_path_from_url(url: &str) -> StoreResult<Option<PathBuf>> {
    let url = url.trim();
    if url == "sqlite::memory:" || url == "sqlite://:memory:" || url == ":memory:" {
        return Ok(None);
    }

    if let Some(raw) = url.strip_prefix("sqlite://") {
        let path = raw.split_once('?').map(|(path, _)| path).unwrap_or(raw);
        if path.is_empty() {
            return Err(StoreError::Invalid(
                "sqlite database URL must include a database path".to_string(),
            ));
        }
        if let Some(abs) = path.strip_prefix('/') {
            return Ok(Some(PathBuf::from(format!("/{abs}"))));
        }
        return Ok(Some(PathBuf::from(path)));
    }

    if let Some(path) = url.strip_prefix("file://") {
        let path = if path.starts_with('/') {
            PathBuf::from(path)
        } else {
            PathBuf::from(format!("/{path}"))
        };
        return Ok(Some(path));
    }

    if url.contains("://") {
        let scheme = url
            .split_once("://")
            .map(|(scheme, _)| scheme)
            .unwrap_or(url);
        return Err(StoreError::Invalid(format!(
            "unsupported DB URL scheme '{scheme}'; currently supported: sqlite://, sqlite::memory:, file://"
        )));
    }

    Ok(Some(PathBuf::from(url)))
}

fn init_db_schema(conn: &Connection) -> StoreResult<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS store_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT OR IGNORE INTO store_meta (key, value) VALUES ('version', '1');
        INSERT OR IGNORE INTO store_meta (key, value) VALUES ('next_seq', '0');

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL,
            metadata TEXT NOT NULL,
            deleted INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            item TEXT NOT NULL,
            deleted INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_items_conversation_id ON items(conversation_id);

        CREATE TABLE IF NOT EXISTS conversation_item_order (
            conversation_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            item_id TEXT NOT NULL,
            PRIMARY KEY (conversation_id, position)
        );
        CREATE INDEX IF NOT EXISTS idx_conversation_item_order_item_id
            ON conversation_item_order(item_id);

        CREATE TABLE IF NOT EXISTS responses (
            id TEXT PRIMARY KEY,
            response TEXT NOT NULL,
            conversation_id TEXT,
            previous_response_id TEXT,
            provider TEXT NOT NULL,
            input_item_ids TEXT NOT NULL,
            output_item_ids TEXT NOT NULL,
            context_item_ids TEXT NOT NULL,
            stream_events TEXT NOT NULL,
            deleted INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_responses_conversation_id ON responses(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_responses_previous_response_id ON responses(previous_response_id);

        CREATE TABLE IF NOT EXISTS monitor_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            method TEXT NOT NULL,
            uri TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            status INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            request_bytes INTEGER NOT NULL,
            response_bytes INTEGER NOT NULL,
            provider TEXT,
            model TEXT,
            upstream_model TEXT,
            upstream_url TEXT,
            stream INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_monitor_events_id ON monitor_events(id);
        CREATE INDEX IF NOT EXISTS idx_monitor_events_timestamp_ms ON monitor_events(timestamp_ms);
        "#,
    )
    .map_err(sqlite_error)?;
    ensure_sqlite_column(conn, "monitor_events", "upstream_url", "upstream_url TEXT")?;
    Ok(())
}

fn ensure_sqlite_column(
    conn: &Connection,
    table: &str,
    column: &str,
    column_def: &str,
) -> StoreResult<()> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(sqlite_error)?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(sqlite_error)?;
    for existing in columns {
        if existing.map_err(sqlite_error)? == column {
            return Ok(());
        }
    }
    conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {column_def}"), [])
        .map(|_| ())
        .map_err(sqlite_error)
}

fn load_store_file_from_db(conn: &Connection) -> StoreResult<StoreFile> {
    Ok(StoreFile {
        version: db_meta_u32(conn, "version", 1)?,
        next_seq: db_meta_u64(conn, "next_seq", 0)?,
        conversations: load_conversations(conn)?,
        items: load_items(conn)?,
        conversation_item_order: load_conversation_item_order(conn)?,
        responses: load_responses(conn)?,
    })
}

fn persist_store_file_to_db(tx: &Transaction<'_>, store: &StoreFile) -> StoreResult<()> {
    tx.execute(
        "INSERT INTO store_meta (key, value) VALUES ('version', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![store.version.to_string()],
    )
    .map_err(sqlite_error)?;
    tx.execute(
        "INSERT INTO store_meta (key, value) VALUES ('next_seq', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![store.next_seq.to_string()],
    )
    .map_err(sqlite_error)?;

    tx.execute("DELETE FROM conversation_item_order", [])
        .map_err(sqlite_error)?;
    tx.execute("DELETE FROM responses", [])
        .map_err(sqlite_error)?;
    tx.execute("DELETE FROM items", []).map_err(sqlite_error)?;
    tx.execute("DELETE FROM conversations", [])
        .map_err(sqlite_error)?;

    for conv in store.conversations.values() {
        tx.execute(
            "INSERT INTO conversations (id, created_at, metadata, deleted)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                &conv.id,
                i64::try_from(conv.created_at).unwrap_or(i64::MAX),
                serde_json::to_string(&conv.metadata)?,
                bool_to_i64(conv.deleted),
            ],
        )
        .map_err(sqlite_error)?;
    }

    for item in store.items.values() {
        tx.execute(
            "INSERT INTO items (id, conversation_id, item, deleted)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                &item.id,
                item.conversation_id.as_deref(),
                serde_json::to_string(&item.item)?,
                bool_to_i64(item.deleted),
            ],
        )
        .map_err(sqlite_error)?;
    }

    for (conversation_id, item_ids) in &store.conversation_item_order {
        for (position, item_id) in item_ids.iter().enumerate() {
            tx.execute(
                "INSERT INTO conversation_item_order (conversation_id, position, item_id)
                 VALUES (?1, ?2, ?3)",
                params![
                    conversation_id,
                    i64::try_from(position).unwrap_or(i64::MAX),
                    item_id,
                ],
            )
            .map_err(sqlite_error)?;
        }
    }

    for resp in store.responses.values() {
        tx.execute(
            "INSERT INTO responses (
                id, response, conversation_id, previous_response_id, provider,
                input_item_ids, output_item_ids, context_item_ids, stream_events, deleted
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                &resp.id,
                serde_json::to_string(&resp.response)?,
                resp.conversation_id.as_deref(),
                resp.previous_response_id.as_deref(),
                &resp.provider,
                serde_json::to_string(&resp.input_item_ids)?,
                serde_json::to_string(&resp.output_item_ids)?,
                serde_json::to_string(&resp.context_item_ids)?,
                serde_json::to_string(&resp.stream_events)?,
                bool_to_i64(resp.deleted),
            ],
        )
        .map_err(sqlite_error)?;
    }

    Ok(())
}

fn db_meta_u32(conn: &Connection, key: &str, default: u32) -> StoreResult<u32> {
    let raw = db_meta_string(conn, key)?;
    Ok(raw
        .as_deref()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default))
}

fn db_meta_u64(conn: &Connection, key: &str, default: u64) -> StoreResult<u64> {
    let raw = db_meta_string(conn, key)?;
    Ok(raw
        .as_deref()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default))
}

fn db_meta_string(conn: &Connection, key: &str) -> StoreResult<Option<String>> {
    conn.query_row(
        "SELECT value FROM store_meta WHERE key = ?1",
        params![key],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(sqlite_error)
}

fn load_conversations(conn: &Connection) -> StoreResult<BTreeMap<String, ConversationRecord>> {
    let mut stmt = conn
        .prepare("SELECT id, created_at, metadata, deleted FROM conversations ORDER BY id")
        .map_err(sqlite_error)?;
    let rows = stmt
        .query_map([], |row| {
            let id: String = row.get(0)?;
            let metadata: String = row.get(2)?;
            Ok((
                id.clone(),
                id,
                row.get::<_, i64>(1)?,
                metadata,
                row.get::<_, i64>(3)?,
            ))
        })
        .map_err(sqlite_error)?;

    let mut out = BTreeMap::new();
    for row in rows {
        let (key, id, created_at, metadata, deleted) = row.map_err(sqlite_error)?;
        out.insert(
            key,
            ConversationRecord {
                id,
                created_at: i64_to_u64(created_at),
                metadata: serde_json::from_str(&metadata)?,
                deleted: deleted != 0,
            },
        );
    }
    Ok(out)
}

fn load_items(conn: &Connection) -> StoreResult<BTreeMap<String, ItemRecord>> {
    let mut stmt = conn
        .prepare("SELECT id, conversation_id, item, deleted FROM items ORDER BY id")
        .map_err(sqlite_error)?;
    let rows = stmt
        .query_map([], |row| {
            let id: String = row.get(0)?;
            let item: String = row.get(2)?;
            Ok((
                id.clone(),
                id,
                row.get::<_, Option<String>>(1)?,
                item,
                row.get::<_, i64>(3)?,
            ))
        })
        .map_err(sqlite_error)?;

    let mut out = BTreeMap::new();
    for row in rows {
        let (key, id, conversation_id, item, deleted) = row.map_err(sqlite_error)?;
        out.insert(
            key,
            ItemRecord {
                id,
                conversation_id,
                item: serde_json::from_str(&item)?,
                deleted: deleted != 0,
            },
        );
    }
    Ok(out)
}

fn load_conversation_item_order(conn: &Connection) -> StoreResult<BTreeMap<String, Vec<String>>> {
    let mut stmt = conn
        .prepare(
            "SELECT conversation_id, item_id
             FROM conversation_item_order
             ORDER BY conversation_id, position",
        )
        .map_err(sqlite_error)?;
    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(sqlite_error)?;

    let mut out = BTreeMap::<String, Vec<String>>::new();
    for row in rows {
        let (conversation_id, item_id) = row.map_err(sqlite_error)?;
        out.entry(conversation_id).or_default().push(item_id);
    }
    Ok(out)
}

fn load_responses(conn: &Connection) -> StoreResult<BTreeMap<String, ResponseRecord>> {
    let mut stmt = conn
        .prepare(
            r#"
            SELECT id, response, conversation_id, previous_response_id, provider,
                   input_item_ids, output_item_ids, context_item_ids, stream_events, deleted
            FROM responses
            ORDER BY id
            "#,
        )
        .map_err(sqlite_error)?;
    let rows = stmt
        .query_map([], |row| {
            let id: String = row.get(0)?;
            Ok((
                id.clone(),
                id,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, i64>(9)?,
            ))
        })
        .map_err(sqlite_error)?;

    let mut out = BTreeMap::new();
    for row in rows {
        let (
            key,
            id,
            response,
            conversation_id,
            previous_response_id,
            provider,
            input_item_ids,
            output_item_ids,
            context_item_ids,
            stream_events,
            deleted,
        ) = row.map_err(sqlite_error)?;
        out.insert(
            key,
            ResponseRecord {
                id,
                response: serde_json::from_str(&response)?,
                conversation_id,
                previous_response_id,
                provider,
                input_item_ids: serde_json::from_str(&input_item_ids)?,
                output_item_ids: serde_json::from_str(&output_item_ids)?,
                context_item_ids: serde_json::from_str(&context_item_ids)?,
                stream_events: serde_json::from_str(&stream_events)?,
                deleted: deleted != 0,
            },
        );
    }
    Ok(out)
}

fn insert_monitor_event(conn: &Connection, event: MonitorEvent) -> StoreResult<()> {
    conn.execute(
        r#"
        INSERT INTO monitor_events (
            request_id, timestamp_ms, method, uri, endpoint, status,
            latency_ms, request_bytes, response_bytes, provider, model,
            upstream_model, upstream_url, stream
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
        "#,
        params![
            i64::try_from(event.request_id).unwrap_or(i64::MAX),
            i64::try_from(event.timestamp_ms).unwrap_or(i64::MAX),
            event.method,
            event.uri,
            event.endpoint,
            i64::from(event.status),
            i64::try_from(event.latency_ms).unwrap_or(i64::MAX),
            i64::try_from(event.request_bytes).unwrap_or(i64::MAX),
            i64::try_from(event.response_bytes).unwrap_or(i64::MAX),
            event.provider,
            event.model,
            event.upstream_model,
            event.upstream_url,
            bool_to_i64(event.stream),
        ],
    )
    .map(|_| ())
    .map_err(sqlite_error)
}

fn list_monitor_events(conn: &Connection, limit: usize) -> StoreResult<Vec<MonitorEvent>> {
    let limit = limit.clamp(1, MAX_MONITOR_EVENTS);
    let mut stmt = conn
        .prepare(
            r#"
            SELECT request_id, timestamp_ms, method, uri, endpoint, status,
                   latency_ms, request_bytes, response_bytes, provider, model,
                   upstream_model, upstream_url, stream
            FROM monitor_events
            ORDER BY id DESC
            LIMIT ?1
            "#,
        )
        .map_err(sqlite_error)?;
    let rows = stmt
        .query_map(params![i64::try_from(limit).unwrap_or(i64::MAX)], |row| {
            Ok(MonitorEvent {
                request_id: i64_to_u64(row.get(0)?),
                timestamp_ms: i64_to_u64(row.get(1)?),
                method: row.get(2)?,
                uri: row.get(3)?,
                endpoint: row.get(4)?,
                status: i64_to_u16(row.get(5)?),
                latency_ms: i64_to_u64(row.get(6)?),
                request_bytes: i64_to_u64(row.get(7)?),
                response_bytes: i64_to_u64(row.get(8)?),
                provider: row.get(9)?,
                model: row.get(10)?,
                upstream_model: row.get(11)?,
                upstream_url: row.get(12)?,
                stream: row.get::<_, i64>(13)? != 0,
            })
        })
        .map_err(sqlite_error)?;

    let mut events = Vec::new();
    for row in rows {
        events.push(row.map_err(sqlite_error)?);
    }
    Ok(events)
}

fn clear_monitor_events(conn: &Connection) -> StoreResult<usize> {
    let count = conn
        .query_row("SELECT COUNT(*) FROM monitor_events", [], |row| {
            row.get::<_, i64>(0)
        })
        .optional()
        .map_err(sqlite_error)?
        .unwrap_or(0);
    conn.execute("DELETE FROM monitor_events", [])
        .map_err(sqlite_error)?;
    Ok(usize::try_from(count).unwrap_or(usize::MAX))
}

fn trim_monitor_events(conn: &Connection) -> StoreResult<()> {
    conn.execute(
        r#"
        DELETE FROM monitor_events
        WHERE id NOT IN (
            SELECT id FROM monitor_events ORDER BY id DESC LIMIT ?1
        )
        "#,
        params![i64::try_from(MAX_MONITOR_EVENTS).unwrap_or(i64::MAX)],
    )
    .map(|_| ())
    .map_err(sqlite_error)
}

fn lock_db_connection(
    inner: &Arc<Mutex<Connection>>,
) -> StoreResult<std::sync::MutexGuard<'_, Connection>> {
    inner
        .lock()
        .map_err(|_| StoreError::Invalid("database connection lock poisoned".to_string()))
}

async fn spawn_db_task<T, F>(f: F) -> StoreResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> StoreResult<T> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|err| StoreError::Invalid(format!("database task failed: {err}")))?
}

fn bool_to_i64(value: bool) -> i64 {
    if value { 1 } else { 0 }
}

fn sqlite_error(err: rusqlite::Error) -> StoreError {
    StoreError::Io(std::io::Error::other(err))
}

fn i64_to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or(0)
}

fn i64_to_u16(value: i64) -> u16 {
    u16::try_from(value).unwrap_or(0)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn load_store_file(path: &Path) -> StoreResult<StoreFile> {
    match tokio::fs::read(path).await {
        Ok(bytes) => {
            let parsed = serde_json::from_slice::<StoreFile>(&bytes)?;
            Ok(parsed)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(StoreFile::default()),
        Err(e) => Err(StoreError::Io(e)),
    }
}

fn load_store_file_sync(path: &Path) -> StoreResult<StoreFile> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let parsed = serde_json::from_slice::<StoreFile>(&bytes)?;
            Ok(parsed)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(StoreFile::default()),
        Err(e) => Err(StoreError::Io(e)),
    }
}

async fn persist_store_file(path: &Path, store: &StoreFile) -> StoreResult<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let encoded = serde_json::to_vec_pretty(store)?;
    let tmp_path = path.with_extension("tmp");
    tokio::fs::write(&tmp_path, encoded).await?;
    tokio::fs::rename(&tmp_path, path).await?;
    Ok(())
}

/// Default legacy JSON store path (cache dir or temp dir).
pub fn default_storage_path() -> PathBuf {
    if let Ok(path) = std::env::var(STORAGE_PATH_ENV)
        && !path.is_empty()
    {
        return PathBuf::from(path);
    }

    if let Some(cache_dir) = dirs::cache_dir() {
        return cache_dir.join("yallm").join("storage.json");
    }

    std::env::temp_dir().join("yallm").join("storage.json")
}

/// Default SQLite URL: `<cache-dir>/yallm/storage.sqlite3` (temp dir fallback).
pub fn default_db_url() -> String {
    let path = if let Some(cache_dir) = dirs::cache_dir() {
        cache_dir.join("yallm").join("storage.sqlite3")
    } else {
        std::env::temp_dir().join("yallm").join("storage.sqlite3")
    };
    format!("sqlite://{}", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("yallm-storage-{name}-{ts}.json"))
    }

    #[tokio::test]
    async fn creates_and_paginates_conversation_items() {
        let store = LocalStore::open(Some(temp_path("pagination")))
            .await
            .expect("store open");
        let conv = store
            .create_conversation(None, vec![])
            .await
            .expect("create conversation");
        let conv_id = conv["id"].as_str().unwrap();

        store
            .add_conversation_items(
                conv_id,
                vec![
                    json!({"type":"message","role":"user","content":[{"type":"input_text","text":"a"}]}),
                    json!({"type":"message","role":"user","content":[{"type":"input_text","text":"b"}]}),
                    json!({"type":"message","role":"user","content":[{"type":"input_text","text":"c"}]}),
                ],
            )
            .await
            .expect("add items");

        let page = store
            .list_conversation_items(conv_id, 2, ListOrder::Asc, None)
            .await
            .expect("list ok")
            .expect("conversation exists");
        assert_eq!(page.data.len(), 2);
        assert!(page.has_more);
    }

    #[tokio::test]
    async fn stores_response_and_resolves_previous_context() {
        let store = LocalStore::open(Some(temp_path("response-chain")))
            .await
            .expect("store open");
        let saved = store
            .save_response(SaveResponseRequest {
                response: json!({
                    "object":"response",
                    "status":"completed",
                    "model":"gpt-4o-mini",
                    "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]
                }),
                request_input_items: vec![json!({
                    "type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]
                })],
                conversation_id: None,
                previous_response_id: None,
                provider: "openai".to_string(),
                stream_events: vec!["data: [DONE]\n\n".to_string()],
            })
            .await
            .expect("save response");

        let response_id = saved.response["id"].as_str().unwrap().to_string();
        let context = store
            .resolve_context(None, Some(&response_id))
            .await
            .expect("resolve context");
        assert!(context.items.len() >= 2);
        assert_eq!(context.conversation_id, saved.conversation_id);
    }

    #[tokio::test]
    async fn sqlite_store_persists_responses_and_previous_context() {
        let db_url = format!(
            "sqlite://{}",
            temp_path("sqlite-response-chain")
                .with_extension("sqlite3")
                .display()
        );
        let store = LocalStore::open_database_url_sync(Some(&db_url)).expect("sqlite store open");
        let saved = store
            .save_response(SaveResponseRequest {
                response: json!({
                    "object":"response",
                    "status":"completed",
                    "model":"gpt-4o-mini",
                    "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]
                }),
                request_input_items: vec![json!({
                    "type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]
                })],
                conversation_id: None,
                previous_response_id: None,
                provider: "openai".to_string(),
                stream_events: vec!["data: [DONE]\n\n".to_string()],
            })
            .await
            .expect("save response");

        let reopened =
            LocalStore::open_database_url_sync(Some(&db_url)).expect("reopen sqlite store");
        let response_id = saved.response["id"].as_str().unwrap().to_string();
        let context = reopened
            .resolve_context(None, Some(&response_id))
            .await
            .expect("resolve context");
        assert!(context.items.len() >= 2);
        assert_eq!(context.conversation_id, saved.conversation_id);
    }

    #[tokio::test]
    async fn sqlite_store_records_lists_and_clears_monitor_events() {
        let db_path = temp_path("monitor").with_extension("sqlite3");
        let db_url = format!("sqlite://{}", db_path.display());
        let store = LocalStore::open_database_url_sync(Some(&db_url)).expect("sqlite store open");

        store
            .record_monitor_event(MonitorEvent {
                request_id: 7,
                timestamp_ms: 123,
                method: "POST".to_string(),
                uri: "/v1/chat/completions".to_string(),
                endpoint: "/v1/chat/completions".to_string(),
                status: 200,
                latency_ms: 42,
                request_bytes: 11,
                response_bytes: 22,
                provider: Some("openai".to_string()),
                model: Some("openai:gpt-test".to_string()),
                upstream_model: Some("gpt-test".to_string()),
                upstream_url: Some("https://api.openai.com/v1/chat/completions".to_string()),
                stream: true,
            })
            .await
            .expect("record event");

        let events = store.list_monitor_events(10).await.expect("list events");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].request_id, 7);
        assert_eq!(events[0].provider.as_deref(), Some("openai"));
        assert_eq!(
            events[0].upstream_url.as_deref(),
            Some("https://api.openai.com/v1/chat/completions")
        );
        assert!(events[0].stream);

        let deleted = store.clear_monitor_events().await.expect("clear events");
        assert_eq!(deleted, 1);
        assert!(
            store
                .list_monitor_events(10)
                .await
                .expect("list events")
                .is_empty()
        );
    }

    #[test]
    fn db_url_parses_sqlite_file_and_rejects_unknown_scheme() {
        assert_eq!(
            sqlite_path_from_url("sqlite:///tmp/yallm.sqlite3")
                .expect("sqlite path")
                .unwrap(),
            PathBuf::from("/tmp/yallm.sqlite3")
        );
        assert!(sqlite_path_from_url("sqlite::memory:").unwrap().is_none());
        assert!(sqlite_path_from_url("odbc://dsn/yallm").is_err());
    }
}
