use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::RwLock;

pub const STORAGE_PATH_ENV: &str = "YALLM_STORAGE_PATH";

#[derive(Debug)]
pub enum StoreError {
    Io(std::io::Error),
    Invalid(String),
    NotFound(String),
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

pub type StoreResult<T> = Result<T, StoreError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListOrder {
    Asc,
    Desc,
}

#[derive(Debug, Clone)]
pub struct ListPage<T> {
    pub data: Vec<T>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
}

#[derive(Debug, Clone)]
pub struct ContextItems {
    pub conversation_id: Option<String>,
    pub items: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct SaveResponseRequest {
    pub response: Value,
    pub request_input_items: Vec<Value>,
    pub conversation_id: Option<String>,
    pub previous_response_id: Option<String>,
    pub provider: String,
    pub stream_events: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SavedResponse {
    pub response: Value,
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
pub struct LocalStore {
    path: PathBuf,
    inner: Arc<RwLock<StoreFile>>,
}

impl fmt::Debug for LocalStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalStore")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl LocalStore {
    pub fn open_sync(path: Option<PathBuf>) -> StoreResult<Self> {
        let path = path.unwrap_or_else(default_storage_path);
        let data = load_store_file_sync(&path)?;
        Ok(Self {
            path,
            inner: Arc::new(RwLock::new(data)),
        })
    }

    pub async fn open(path: Option<PathBuf>) -> StoreResult<Self> {
        let path = path.unwrap_or_else(default_storage_path);
        let data = load_store_file(&path).await?;
        Ok(Self {
            path,
            inner: Arc::new(RwLock::new(data)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

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

    pub async fn get_conversation(&self, conversation_id: &str) -> StoreResult<Option<Value>> {
        let store = self.inner.read().await;
        let Some(conv) = store.conversations.get(conversation_id) else {
            return Ok(None);
        };
        if conv.deleted {
            return Ok(None);
        }
        Ok(Some(conversation_json(conv)))
    }

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

    pub async fn list_conversation_items(
        &self,
        conversation_id: &str,
        limit: usize,
        order: ListOrder,
        after: Option<&str>,
    ) -> StoreResult<Option<ListPage<Value>>> {
        let store = self.inner.read().await;
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

    pub async fn get_conversation_item(
        &self,
        conversation_id: &str,
        item_id: &str,
    ) -> StoreResult<Option<Value>> {
        let store = self.inner.read().await;
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

        let store = self.inner.read().await;

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

    pub async fn get_response(&self, response_id: &str) -> StoreResult<Option<Value>> {
        let store = self.inner.read().await;
        let Some(resp) = store.responses.get(response_id) else {
            return Ok(None);
        };
        if resp.deleted {
            return Ok(None);
        }
        Ok(Some(resp.response.clone()))
    }

    pub async fn get_response_stream_events(
        &self,
        response_id: &str,
    ) -> StoreResult<Option<Vec<String>>> {
        let store = self.inner.read().await;
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

    pub async fn list_response_input_items(
        &self,
        response_id: &str,
        limit: usize,
        order: ListOrder,
        after: Option<&str>,
    ) -> StoreResult<Option<ListPage<Value>>> {
        let store = self.inner.read().await;
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

    async fn apply_mutation<R, F>(&self, f: F) -> StoreResult<R>
    where
        F: FnOnce(&mut StoreFile) -> StoreResult<R>,
    {
        let (result, snapshot) = {
            let mut store = self.inner.write().await;
            let result = f(&mut store)?;
            (result, store.clone())
        };
        persist_store_file(&self.path, &snapshot).await?;
        Ok(result)
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
}
