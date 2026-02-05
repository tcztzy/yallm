use std::time::Instant;

use axum::{
    body::{Body, Bytes, to_bytes},
    extract::{Request, State},
    http::HeaderMap,
    middleware::Next,
    response::Response,
};
use serde_json::json;

use crate::state::AppState;

#[derive(Debug, Clone, Copy)]
pub struct RequestId(pub u64);

#[derive(Debug, Clone)]
struct BodyLog {
    pub len: usize,
    pub truncated: bool,
    pub is_utf8: bool,
    pub text: Option<String>,
    pub b64: Option<String>,
}

fn base64_encode(data: &[u8]) -> String {
    // Minimal base64 encoder to avoid an extra dependency.
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);

    let mut i = 0;
    while i + 3 <= data.len() {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8) | (data[i + 2] as u32);
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(TABLE[((n >> 6) & 63) as usize] as char);
        out.push(TABLE[(n & 63) as usize] as char);
        i += 3;
    }

    let rem = data.len() - i;
    if rem == 1 {
        let n = (data[i] as u32) << 16;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8);
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(TABLE[((n >> 6) & 63) as usize] as char);
        out.push('=');
    }

    out
}

fn headers_to_json(headers: &HeaderMap, redact: bool) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    for (k, v) in headers.iter() {
        let name = k.as_str().to_ascii_lowercase();
        let value = if redact && is_secret_header(&name) {
            "[REDACTED]".to_string()
        } else {
            v.to_str().unwrap_or("<non-utf8>").to_string()
        };
        obj.insert(name, serde_json::Value::String(value));
    }
    serde_json::Value::Object(obj)
}

fn is_secret_header(name_lower: &str) -> bool {
    matches!(
        name_lower,
        "authorization"
            | "proxy-authorization"
            | "x-api-key"
            | "api-key"
            | "x-openai-api-key"
            | "openai-api-key"
    )
}

fn body_to_log(bytes: &Bytes, max_bytes: usize) -> BodyLog {
    let len = bytes.len();
    let (slice, truncated) = if max_bytes > 0 && len > max_bytes {
        (bytes.slice(0..max_bytes), true)
    } else {
        (bytes.clone(), false)
    };

    match std::str::from_utf8(&slice) {
        Ok(s) => BodyLog {
            len,
            truncated,
            is_utf8: true,
            text: Some(s.to_string()),
            b64: None,
        },
        Err(_) => BodyLog {
            len,
            truncated,
            is_utf8: false,
            text: None,
            b64: Some(base64_encode(&slice)),
        },
    }
}

pub async fn log_http(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let request_id = state.next_request_id();
    let started = Instant::now();

    let (mut parts, body) = req.into_parts();
    let body_bytes = to_bytes(body, usize::MAX)
        .await
        .unwrap_or_else(|_| Bytes::new());

    let in_body = body_to_log(&body_bytes, state.logging.body_max_bytes);
    let in_headers = headers_to_json(&parts.headers, state.logging.redact_secrets);

    tracing::info!(
        event = "http.in",
        request_id,
        method = %parts.method,
        uri = %parts.uri,
        headers = %in_headers,
        body_len = in_body.len,
        body_truncated = in_body.truncated,
        body_is_utf8 = in_body.is_utf8,
        body_text = %in_body.text.clone().unwrap_or_default(),
        body_b64 = %in_body.b64.clone().unwrap_or_default(),
    );

    parts.extensions.insert(RequestId(request_id));
    let req2 = Request::from_parts(parts, Body::from(body_bytes));

    let resp = next.run(req2).await;
    let (parts, body) = resp.into_parts();
    let out_body_bytes = to_bytes(body, usize::MAX)
        .await
        .unwrap_or_else(|_| Bytes::new());

    let out_body = body_to_log(&out_body_bytes, state.logging.body_max_bytes);
    let out_headers = headers_to_json(&parts.headers, state.logging.redact_secrets);

    tracing::info!(
        event = "http.out",
        request_id,
        status = parts.status.as_u16(),
        headers = %out_headers,
        body_len = out_body.len,
        body_truncated = out_body.truncated,
        body_is_utf8 = out_body.is_utf8,
        body_text = %out_body.text.clone().unwrap_or_default(),
        body_b64 = %out_body.b64.clone().unwrap_or_default(),
        latency_ms = started.elapsed().as_millis(),
    );

    // Rebuild response with buffered body.
    let mut resp2 = Response::from_parts(parts, Body::from(out_body_bytes));
    // Best-effort: attach request id header for correlation.
    resp2.headers_mut().insert(
        "x-yallm-request-id",
        request_id.to_string().parse().unwrap(),
    );
    resp2
}

pub fn redact_json_secrets(value: &serde_json::Value) -> serde_json::Value {
    fn go(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(map) => {
                let mut out = serde_json::Map::new();
                for (k, vv) in map {
                    let kl = k.to_ascii_lowercase();
                    if matches!(
                        kl.as_str(),
                        "authorization"
                            | "x-api-key"
                            | "api_key"
                            | "apikey"
                            | "token"
                            | "access_token"
                    ) {
                        out.insert(k.clone(), json!("[REDACTED]"));
                    } else {
                        out.insert(k.clone(), go(vv));
                    }
                }
                serde_json::Value::Object(out)
            }
            serde_json::Value::Array(arr) => serde_json::Value::Array(arr.iter().map(go).collect()),
            _ => v.clone(),
        }
    }

    go(value)
}
