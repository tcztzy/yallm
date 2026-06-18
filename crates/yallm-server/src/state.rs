use std::{
    collections::HashMap,
    env,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use bytes::Bytes;
use futures::{Stream, StreamExt};

pub(crate) const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com";
pub(crate) const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
pub(crate) const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_FORWARD_HEADERS: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
    "openai-organization",
    "openai-project",
    "anthropic-version",
    "anthropic-beta",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Ollama,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Ollama => "ollama",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Auto,
    Proxy,
    Mock,
}

#[derive(Debug, Clone)]
pub struct LoggingConfig {
    pub redact_secrets: bool,
    pub body_max_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
    pub openai_headers: Vec<(String, String)>,

    pub anthropic_api_key: Option<String>,
    pub anthropic_auth_token: Option<String>,
    pub anthropic_base_url: String,
    pub anthropic_version: String,
    pub anthropic_headers: Vec<(String, String)>,

    pub ollama_base_url: String,
    pub ollama_headers: Vec<(String, String)>,

    pub forward_headers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRoute {
    pub provider: Provider,
    pub upstream_model: String,
    pub api_base: Option<String>,
    pub api_key: Option<String>,
    pub api_version: Option<String>,
    pub headers: Vec<(String, String)>,
    pub forward_headers: Vec<String>,
}

#[derive(Clone)]
pub struct AppState {
    pub transport: Arc<dyn Transport>,
    pub store: Arc<yallm_storage::LocalStore>,
    pub provider: ProviderConfig,
    pub mode: Mode,
    pub default_provider: Provider,
    pub logging: LoggingConfig,
    pub request_id: Arc<AtomicU64>,
    pub monitor_upstream_urls: Arc<Mutex<HashMap<u64, String>>>,
    pub openai_models: Vec<String>,
    pub anthropic_models: Vec<String>,
    pub model_routes: HashMap<String, ModelRoute>,
}

impl AppState {
    pub fn next_request_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn record_monitor_upstream_url(&self, request_id: u64, url: &str) {
        if let Ok(mut urls) = self.monitor_upstream_urls.lock() {
            urls.insert(request_id, url.to_string());
        }
    }

    pub fn take_monitor_upstream_url(&self, request_id: u64) -> Option<String> {
        self.monitor_upstream_urls
            .lock()
            .ok()
            .and_then(|mut urls| urls.remove(&request_id))
    }

    pub fn from_loaded_config(config: yallm_config::LoadedConfig) -> Self {
        for warning in &config.warnings {
            tracing::warn!("{warning}");
        }
        Self::from_env_map(config.env, config.litellm_models)
    }

    fn from_env_map(
        env_map: HashMap<String, String>,
        litellm_models: Vec<yallm_config::LiteLlmModel>,
    ) -> Self {
        let mode = env_map
            .get("YALLM_MODE")
            .and_then(|s| parse_mode(s))
            .unwrap_or(Mode::Auto);

        let default_provider = env_map
            .get("YALLM_DEFAULT_PROVIDER")
            .and_then(|s| parse_provider(s))
            .unwrap_or(Provider::OpenAI);

        let provider = ProviderConfig {
            openai_api_key: env_opt(&env_map, "OPENAI_API_KEY"),
            openai_base_url: env_string(&env_map, "OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            openai_headers: env_headers(&env_map, "YALLM_OPENAI_HEADERS"),

            anthropic_api_key: env_opt(&env_map, "ANTHROPIC_API_KEY"),
            anthropic_auth_token: env_opt(&env_map, "ANTHROPIC_AUTH_TOKEN"),
            anthropic_base_url: env_string(
                &env_map,
                "ANTHROPIC_BASE_URL",
                DEFAULT_ANTHROPIC_BASE_URL,
            ),
            anthropic_version: env_string(&env_map, "ANTHROPIC_VERSION", "2023-06-01"),
            anthropic_headers: env_headers(&env_map, "YALLM_ANTHROPIC_HEADERS"),

            ollama_base_url: env_string(&env_map, "OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            ollama_headers: env_headers(&env_map, "YALLM_OLLAMA_HEADERS"),

            forward_headers: env_header_names(&env_map, "YALLM_FORWARD_HEADERS"),
        };

        let logging = LoggingConfig {
            redact_secrets: env_bool(&env_map, "YALLM_LOG_REDACT_SECRETS", true),
            body_max_bytes: env_usize(&env_map, "YALLM_LOG_BODY_MAX_BYTES", 0),
        };

        let store = if let Some(db_url) = env_opt(&env_map, yallm_storage::DB_URL_ENV) {
            yallm_storage::LocalStore::open_database_url_sync(Some(&db_url))
                .expect("failed to open yallm database")
        } else if let Some(storage_path) = env_opt(&env_map, yallm_storage::STORAGE_PATH_ENV) {
            tracing::warn!(
                "{} is deprecated; use {}=sqlite:///path/to/yallm.sqlite3",
                yallm_storage::STORAGE_PATH_ENV,
                yallm_storage::DB_URL_ENV
            );
            yallm_storage::LocalStore::open_sync(Some(storage_path.into()))
                .expect("failed to open legacy local yallm storage")
        } else {
            yallm_storage::LocalStore::open_database_url_sync(None)
                .expect("failed to open default yallm database")
        };

        // Avoid consulting OS proxy configuration (which can require platform APIs that
        // aren't always available in sandboxed/test environments).
        let http = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("reqwest client build");

        let (model_routes, all_aliases) = model_routes_from_litellm(litellm_models);
        let any_litellm = !model_routes.is_empty();

        if any_litellm {
            for env_name in ["YALLM_OPENAI_MODELS", "YALLM_ANTHROPIC_MODELS"] {
                if env_map.get(env_name).is_some_and(|v| !v.is_empty()) {
                    tracing::warn!("{env_name} is ignored because LiteLLM aliases are configured");
                }
            }
        }

        // Every alias is reachable via every protocol — yallm converts on the fly.
        // List the same alias set under every interface filter.
        let openai_models = if any_litellm {
            all_aliases.clone()
        } else {
            env_models(
                &env_map,
                "YALLM_OPENAI_MODELS",
                vec!["gpt-5.2", "gpt-5.3-codex", "gpt-5.4", "gpt-5.5"],
            )
        };

        let anthropic_models = if any_litellm {
            all_aliases
        } else {
            env_models(
                &env_map,
                "YALLM_ANTHROPIC_MODELS",
                vec![
                    "claude-opus-4-1-20250805",
                    "claude-sonnet-4-5-20250929",
                    "claude-haiku-4-5-20251001",
                    "claude-opus-4-5-20251101",
                    "claude-sonnet-4-6",
                    "claude-opus-4-6",
                    "claude-opus-4-7",
                ],
            )
        };

        AppState {
            transport: Arc::new(ReqwestTransport { http }),
            store: Arc::new(store),
            provider,
            mode,
            default_provider,
            logging,
            request_id: Arc::new(AtomicU64::new(1)),
            monitor_upstream_urls: Arc::new(Mutex::new(HashMap::new())),
            openai_models,
            anthropic_models,
            model_routes,
        }
    }
}

fn env_opt(env_map: &HashMap<String, String>, name: &str) -> Option<String> {
    env_map.get(name).filter(|s| !s.is_empty()).cloned()
}

fn env_bool(env_map: &HashMap<String, String>, name: &str, default: bool) -> bool {
    match env_map.get(name).map(String::as_str) {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO") => false,
        _ => default,
    }
}

fn env_usize(env_map: &HashMap<String, String>, name: &str, default: usize) -> usize {
    env_map
        .get(name)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_string(env_map: &HashMap<String, String>, name: &str, default: &str) -> String {
    env_map
        .get(name)
        .filter(|s| !s.is_empty())
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn env_headers(env_map: &HashMap<String, String>, env_name: &str) -> Vec<(String, String)> {
    let Some(raw) = env_opt(env_map, env_name) else {
        return Vec::new();
    };
    let raw = raw.trim();
    if raw.eq_ignore_ascii_case("none") {
        return Vec::new();
    }

    let parsed = match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(value) => value,
        Err(err) => {
            tracing::warn!("{env_name} must be a JSON object of header names to values: {err}");
            return Vec::new();
        }
    };
    let serde_json::Value::Object(obj) = parsed else {
        tracing::warn!("{env_name} must be a JSON object of header names to values");
        return Vec::new();
    };

    let mut entries = obj.into_iter().collect::<Vec<_>>();
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    entries
        .into_iter()
        .filter_map(|(header_name, value)| {
            let header_name = header_name.trim();
            if header_name.is_empty() {
                return None;
            }
            json_scalar_to_string(value)
                .and_then(|value| resolve_env_header_value(&value, env_map, env_name, header_name))
                .map(|value| (header_name.to_string(), value))
        })
        .collect()
}

fn json_scalar_to_string(value: serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(v) => Some(v.to_string()),
        serde_json::Value::Number(v) => Some(v.to_string()),
        serde_json::Value::String(v) => Some(v),
        _ => None,
    }
}

fn resolve_env_header_value(
    value: &str,
    env_map: &HashMap<String, String>,
    env_name: &str,
    header_name: &str,
) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("none") {
        return None;
    }

    if let Some(name) = value.strip_prefix("os.environ/") {
        return env_header_lookup(name, env_map, env_name, header_name);
    }

    interpolate_env_header_refs(value, env_map, env_name, header_name)
}

fn interpolate_env_header_refs(
    value: &str,
    env_map: &HashMap<String, String>,
    env_name: &str,
    header_name: &str,
) -> Option<String> {
    let mut rest = value;
    let mut out = String::new();

    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after_start = &rest[start + 2..];
        let Some(end) = after_start.find('}') else {
            out.push_str(&rest[start..]);
            return Some(out);
        };
        let name = after_start[..end].trim();
        let value = env_header_lookup(name, env_map, env_name, header_name)?;
        out.push_str(&value);
        rest = &after_start[end + 1..];
    }

    out.push_str(rest);
    Some(out)
}

fn env_header_lookup(
    name: &str,
    env_map: &HashMap<String, String>,
    env_name: &str,
    header_name: &str,
) -> Option<String> {
    let name = name.trim();
    match env_map.get(name).map(String::as_str).map(str::trim) {
        Some(value) if !value.is_empty() => Some(value.to_string()),
        _ => {
            tracing::warn!("{env_name} header '{header_name}' references missing env var {name}");
            None
        }
    }
}

fn env_header_names(env_map: &HashMap<String, String>, name: &str) -> Vec<String> {
    match env_map.get(name).map(String::as_str).map(str::trim) {
        Some(raw) if raw.is_empty() || raw.eq_ignore_ascii_case("none") => Vec::new(),
        Some(raw) => parse_header_names(raw),
        None => DEFAULT_FORWARD_HEADERS
            .iter()
            .map(|h| h.to_string())
            .collect(),
    }
}

fn parse_header_names(raw: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for header in raw.split(',') {
        let header = header.trim().to_ascii_lowercase();
        if !header.is_empty() && seen.insert(header.clone()) {
            out.push(header);
        }
    }
    out
}

/// Parse comma-separated model list from env var. Returns `default` when empty/absent.
fn env_models(env_map: &HashMap<String, String>, name: &str, default: Vec<&str>) -> Vec<String> {
    let raw = env_map.get(name).filter(|s| !s.is_empty());
    match raw {
        Some(s) => s
            .split(',')
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .collect(),
        None => default.into_iter().map(String::from).collect(),
    }
}

fn parse_provider(s: &str) -> Option<Provider> {
    match s.trim().to_ascii_lowercase().as_str() {
        "openai" => Some(Provider::OpenAI),
        "anthropic" => Some(Provider::Anthropic),
        "ollama" => Some(Provider::Ollama),
        _ => None,
    }
}

fn parse_mode(s: &str) -> Option<Mode> {
    match s.trim().to_ascii_lowercase().as_str() {
        "auto" => Some(Mode::Auto),
        "proxy" => Some(Mode::Proxy),
        "mock" => Some(Mode::Mock),
        _ => None,
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::from_env_map(env::vars().collect(), Vec::new())
    }
}

fn model_routes_from_litellm(
    models: Vec<yallm_config::LiteLlmModel>,
) -> (HashMap<String, ModelRoute>, Vec<String>) {
    let mut routes = HashMap::new();
    let mut aliases = Vec::new();

    for model in models {
        let provider = match model.provider {
            yallm_config::LiteLlmProvider::OpenAI => Provider::OpenAI,
            yallm_config::LiteLlmProvider::Anthropic => Provider::Anthropic,
            yallm_config::LiteLlmProvider::Ollama => Provider::Ollama,
        };
        let alias = model.model_name;
        let route = ModelRoute {
            provider,
            upstream_model: model.upstream_model,
            api_base: model.api_base,
            api_key: model.api_key,
            api_version: model.api_version,
            headers: model.headers,
            forward_headers: model.forward_headers,
        };
        if routes.insert(alias.clone(), route).is_none() {
            aliases.push(alias);
        }
    }

    (routes, aliases)
}

#[derive(Debug, Clone)]
pub struct TransportRequest {
    pub method: &'static str,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct TransportResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

pub type TransportByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, TransportError>> + Send + 'static>>;

pub struct TransportStreamResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: TransportByteStream,
}

#[derive(Debug, Clone)]
pub struct TransportError {
    pub message: String,
}

pub type TransportFuture<'a> =
    Pin<Box<dyn Future<Output = Result<TransportResponse, TransportError>> + Send + 'a>>;

pub type TransportStreamFuture<'a> =
    Pin<Box<dyn Future<Output = Result<TransportStreamResponse, TransportError>> + Send + 'a>>;

pub trait Transport: Send + Sync {
    fn send<'a>(&'a self, req: TransportRequest) -> TransportFuture<'a>;

    fn send_stream<'a>(&'a self, req: TransportRequest) -> TransportStreamFuture<'a>;
}

#[derive(Debug, Clone)]
pub struct ReqwestTransport {
    http: reqwest::Client,
}

impl Transport for ReqwestTransport {
    fn send<'a>(&'a self, req: TransportRequest) -> TransportFuture<'a> {
        Box::pin(async move {
            let mut rb = match req.method {
                "POST" => self.http.post(req.url),
                "GET" => self.http.get(req.url),
                "PUT" => self.http.put(req.url),
                "DELETE" => self.http.delete(req.url),
                _ => {
                    return Err(TransportError {
                        message: format!("unsupported method {}", req.method),
                    });
                }
            };

            for (k, v) in req.headers {
                rb = rb.header(k, v);
            }

            let resp = rb
                .json(&req.body)
                .send()
                .await
                .map_err(|e| TransportError {
                    message: format!("{e}"),
                })?;

            let status = resp.status().as_u16();
            let headers = resp
                .headers()
                .iter()
                .map(|(k, v)| {
                    (
                        k.as_str().to_string(),
                        v.to_str().unwrap_or("<non-utf8>").to_string(),
                    )
                })
                .collect::<Vec<_>>();
            let body = resp.bytes().await.map_err(|e| TransportError {
                message: format!("{e}"),
            })?;

            Ok(TransportResponse {
                status,
                headers,
                body: body.to_vec(),
            })
        })
    }

    fn send_stream<'a>(&'a self, req: TransportRequest) -> TransportStreamFuture<'a> {
        Box::pin(async move {
            let mut rb = match req.method {
                "POST" => self.http.post(req.url),
                "GET" => self.http.get(req.url),
                "PUT" => self.http.put(req.url),
                "DELETE" => self.http.delete(req.url),
                _ => {
                    return Err(TransportError {
                        message: format!("unsupported method {}", req.method),
                    });
                }
            };

            for (k, v) in req.headers {
                rb = rb.header(k, v);
            }

            let resp = rb
                .json(&req.body)
                .send()
                .await
                .map_err(|e| TransportError {
                    message: format!("{e}"),
                })?;

            let status = resp.status().as_u16();
            let headers = resp
                .headers()
                .iter()
                .map(|(k, v)| {
                    (
                        k.as_str().to_string(),
                        v.to_str().unwrap_or("<non-utf8>").to_string(),
                    )
                })
                .collect::<Vec<_>>();

            let body = resp.bytes_stream().map(|chunk| {
                chunk.map_err(|e| TransportError {
                    message: format!("{e}"),
                })
            });

            Ok(TransportStreamResponse {
                status,
                headers,
                body: Box::pin(body),
            })
        })
    }
}
