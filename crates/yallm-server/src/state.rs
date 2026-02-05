use std::{
    env,
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

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

    pub anthropic_api_key: Option<String>,
    pub anthropic_base_url: String,
    pub anthropic_version: String,

    pub ollama_base_url: String,
}

#[derive(Clone)]
pub struct AppState {
    pub transport: Arc<dyn Transport>,
    pub provider: ProviderConfig,
    pub mode: Mode,
    pub default_provider: Provider,
    pub logging: LoggingConfig,
    pub request_id: Arc<AtomicU64>,
}

impl AppState {
    pub fn next_request_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::Relaxed)
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO") => false,
        _ => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
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
        let mode = env::var("YALLM_MODE")
            .ok()
            .and_then(|s| parse_mode(&s))
            .unwrap_or(Mode::Auto);

        let default_provider = env::var("YALLM_DEFAULT_PROVIDER")
            .ok()
            .and_then(|s| parse_provider(&s))
            .unwrap_or(Provider::OpenAI);

        let provider = ProviderConfig {
            openai_api_key: env::var("OPENAI_API_KEY").ok().filter(|s| !s.is_empty()),
            openai_base_url: env_string("OPENAI_BASE_URL", "https://api.openai.com"),

            anthropic_api_key: env::var("ANTHROPIC_API_KEY").ok().filter(|s| !s.is_empty()),
            anthropic_base_url: env_string("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            anthropic_version: env_string("ANTHROPIC_VERSION", "2023-06-01"),

            ollama_base_url: env_string("OLLAMA_BASE_URL", "http://localhost:11434"),
        };

        let logging = LoggingConfig {
            redact_secrets: env_bool("YALLM_LOG_REDACT_SECRETS", true),
            body_max_bytes: env_usize("YALLM_LOG_BODY_MAX_BYTES", 0),
        };

        // Avoid consulting OS proxy configuration (which can require platform APIs that
        // aren't always available in sandboxed/test environments).
        let http = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("reqwest client build");

        AppState {
            transport: Arc::new(ReqwestTransport { http }),
            provider,
            mode,
            default_provider,
            logging,
            request_id: Arc::new(AtomicU64::new(1)),
        }
    }
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

#[derive(Debug, Clone)]
pub struct TransportError {
    pub message: String,
}

pub type TransportFuture<'a> =
    Pin<Box<dyn Future<Output = Result<TransportResponse, TransportError>> + Send + 'a>>;

pub trait Transport: Send + Sync {
    fn send<'a>(&'a self, req: TransportRequest) -> TransportFuture<'a>;
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
}
