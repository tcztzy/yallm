//! Layered configuration loading for yallm.
//!
//! Priority (high to low), each layer overwrites lower layers:
//! 1. `.yallm/secrets.toml` - project secrets (gitignored)
//! 2. `.yallm/*.local.toml` - per-machine overrides (gitignored)
//! 3. `.yallm/config.toml` - project config (committed)
//! 4. `~/.yallm/config.toml` - user defaults
//! 5. OS env vars
//! 6. `.env` file (only fills keys absent from OS env)
//!
//! Config files can have an `[env]` section. Values are returned in a merged
//! map instead of being written back into the process environment.

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Deserializer};

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    pub litellm_config: Option<PathBuf>,
}

#[derive(Debug, Clone, Default)]
pub struct LoadedConfig {
    pub env: HashMap<String, String>,
    pub litellm_models: Vec<LiteLlmModel>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteLlmModel {
    pub model_name: String,
    pub provider: LiteLlmProvider,
    pub upstream_model: String,
    pub api_base: Option<String>,
    pub api_key: Option<String>,
    pub api_version: Option<String>,
    pub headers: Vec<(String, String)>,
    pub forward_headers: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteLlmProvider {
    OpenAI,
    Anthropic,
    Ollama,
    Acp,
}

/// Top-level yallm TOML structure.
#[derive(Debug, Deserialize, Default)]
pub struct ConfigFile {
    /// Env vars to inject into the merged config map.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

/// Load config from all layers using default options.
pub fn load() -> LoadedConfig {
    load_with_options(LoadOptions::default())
}

/// Load config from all layers.
pub fn load_with_options(options: LoadOptions) -> LoadedConfig {
    let current_dir = env::current_dir().unwrap_or_default();
    let mut merged_env: HashMap<String, String> = env::vars().collect();
    let mut warnings = Vec::new();

    // .env only fills keys absent from OS env (matches dotenvy/python-dotenv default).
    apply_dotenv(&current_dir.join(".env"), &mut merged_env, &mut warnings);

    if let Some(path) = user_config_path(&merged_env) {
        apply_toml_file(&path, &mut merged_env, "user config", &mut warnings);
    }
    let project_dir = current_dir.join(".yallm");
    apply_toml_file(
        &project_dir.join("config.toml"),
        &mut merged_env,
        "project config",
        &mut warnings,
    );
    apply_toml_glob(
        &project_dir,
        ".local.toml",
        &mut merged_env,
        "local override",
        &mut warnings,
    );
    apply_toml_file(
        &project_dir.join("secrets.toml"),
        &mut merged_env,
        "project secrets",
        &mut warnings,
    );

    let litellm_path = options.litellm_config.or_else(|| {
        merged_env
            .get("YALLM_LITELLM_CONFIG")
            .map(String::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
    });

    let litellm_models = litellm_path
        .as_deref()
        .map(|path| parse_litellm_config_file(path, &merged_env, &mut warnings))
        .unwrap_or_default();

    LoadedConfig {
        env: merged_env,
        litellm_models,
        warnings,
    }
}

fn apply_toml_file(
    path: &Path,
    env_map: &mut HashMap<String, String>,
    label: &str,
    warnings: &mut Vec<String>,
) {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return,
    };
    let cfg: ConfigFile = match toml::from_str(&raw) {
        Ok(c) => c,
        Err(e) => {
            warnings.push(format!("yallm-config: {label} parse error: {e}"));
            return;
        }
    };
    for (key, val) in cfg.env {
        env_map.insert(key, val);
    }
}

fn apply_dotenv(path: &Path, env_map: &mut HashMap<String, String>, warnings: &mut Vec<String>) {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return,
    };
    for (key, val) in parse_dotenv(&raw, warnings) {
        // Only fill keys absent from OS env (or earlier layers); never override.
        env_map.entry(key).or_insert(val);
    }
}

/// Apply every TOML file in `dir` whose name ends with `suffix`.
/// Files are sorted by name for deterministic ordering; later files overwrite earlier ones.
fn apply_toml_glob(
    dir: &Path,
    suffix: &str,
    env_map: &mut HashMap<String, String>,
    label: &str,
    warnings: &mut Vec<String>,
) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with(suffix))
        })
        .collect();
    files.sort();
    for path in files {
        apply_toml_file(&path, env_map, label, warnings);
    }
}

fn parse_dotenv(raw: &str, warnings: &mut Vec<String>) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for (idx, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, val)) = line.split_once('=') else {
            warnings.push(format!(
                "yallm-config: ignored malformed .env line {}",
                idx + 1
            ));
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            warnings.push(format!(
                "yallm-config: ignored empty .env key on line {}",
                idx + 1
            ));
            continue;
        }
        out.insert(key.to_string(), unquote_env_value(val.trim()));
    }
    out
}

fn unquote_env_value(value: &str) -> String {
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

fn user_config_path(env_map: &HashMap<String, String>) -> Option<PathBuf> {
    let home = env_map
        .get("HOME")
        .or_else(|| env_map.get("USERPROFILE"))
        .filter(|s| !s.is_empty())?;
    Some(PathBuf::from(home).join(".yallm").join("config.toml"))
}

fn parse_litellm_config_file(
    path: &Path,
    env_map: &HashMap<String, String>,
    warnings: &mut Vec<String>,
) -> Vec<LiteLlmModel> {
    let raw = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            warnings.push(format!(
                "yallm-config: failed to read LiteLLM config {}: {e}",
                path.display()
            ));
            return Vec::new();
        }
    };
    parse_litellm_config_str(&raw, env_map, warnings)
}

pub fn parse_litellm_config_str(
    raw: &str,
    env_map: &HashMap<String, String>,
    warnings: &mut Vec<String>,
) -> Vec<LiteLlmModel> {
    let cfg: LiteLlmConfigFile = match serde_yaml_ng::from_str(raw) {
        Ok(cfg) => cfg,
        Err(e) => {
            warnings.push(format!("yallm-config: LiteLLM config parse error: {e}"));
            return Vec::new();
        }
    };

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for entry in cfg.model_list {
        let model_name = entry.model_name.unwrap_or_default().trim().to_string();
        if model_name.is_empty() {
            warnings.push("yallm-config: skipped LiteLLM model with empty model_name".to_string());
            continue;
        }
        if model_name == "*" {
            warnings.push("yallm-config: skipped LiteLLM wildcard model_name '*'".to_string());
            continue;
        }

        // yallm_params wins over litellm_params when both are present.
        let resolved = match entry.yallm_params {
            Some(yp) => resolve_from_yallm_params(&model_name, yp, env_map, warnings),
            None => {
                resolve_from_litellm_params(&model_name, entry.litellm_params, env_map, warnings)
            }
        };
        let Some(model) = resolved else {
            continue;
        };

        if !seen.insert(model.model_name.clone()) {
            warnings.push(format!(
                "yallm-config: skipped duplicate model_name '{}'",
                model.model_name
            ));
            continue;
        }

        out.push(model);
    }
    out
}

fn resolve_from_yallm_params(
    model_name: &str,
    params: YallmParams,
    env_map: &HashMap<String, String>,
    warnings: &mut Vec<String>,
) -> Option<LiteLlmModel> {
    let upstream_model = params.model.unwrap_or_default().trim().to_string();
    if upstream_model.is_empty() || upstream_model == "*" {
        warnings.push(format!(
            "yallm-config: skipped model '{model_name}' with empty or wildcard yallm_params.model"
        ));
        return None;
    }
    // Provider must be explicit in yallm_params — no prefix-encoded fallback.
    let Some(provider_str) = params
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        warnings.push(format!(
            "yallm-config: skipped model '{model_name}' with missing yallm_params.provider"
        ));
        return None;
    };
    let Some(provider) = parse_yallm_provider(provider_str) else {
        warnings.push(format!(
            "yallm-config: skipped model '{model_name}' with unsupported yallm_params.provider '{provider_str}'"
        ));
        return None;
    };

    let api_base = params
        .api_base
        .as_deref()
        .and_then(|v| resolve_config_value(v, env_map, model_name, "api_base", false, warnings));
    let api_key = params
        .api_key
        .as_deref()
        .and_then(|v| resolve_config_value(v, env_map, model_name, "api_key", true, warnings));
    let api_version = params
        .api_version
        .as_deref()
        .and_then(|v| resolve_config_value(v, env_map, model_name, "api_version", false, warnings));
    let headers = resolve_headers(params.headers, env_map, model_name, warnings);
    let forward_headers = normalize_header_names(params.forward_headers);

    Some(LiteLlmModel {
        model_name: model_name.to_string(),
        provider,
        upstream_model,
        api_base,
        api_key,
        api_version,
        headers,
        forward_headers,
    })
}

fn resolve_from_litellm_params(
    model_name: &str,
    params: LiteLlmParams,
    env_map: &HashMap<String, String>,
    warnings: &mut Vec<String>,
) -> Option<LiteLlmModel> {
    let litellm_model = params.model.unwrap_or_default();
    if litellm_model.trim().is_empty() || litellm_model.trim() == "*" {
        warnings.push(format!(
            "yallm-config: skipped LiteLLM model '{model_name}' with empty or wildcard upstream model"
        ));
        return None;
    }

    let Some((provider, upstream_model)) =
        infer_provider_and_model(&litellm_model, params.custom_llm_provider.as_deref())
    else {
        warnings.push(format!(
            "yallm-config: skipped unsupported LiteLLM model '{model_name}' ({litellm_model})"
        ));
        return None;
    };

    let api_base = params
        .api_base
        .as_deref()
        .and_then(|v| resolve_config_value(v, env_map, model_name, "api_base", false, warnings));
    let api_key = params
        .api_key
        .as_deref()
        .and_then(|v| resolve_config_value(v, env_map, model_name, "api_key", true, warnings));
    let api_version = params
        .api_version
        .as_deref()
        .and_then(|v| resolve_config_value(v, env_map, model_name, "api_version", false, warnings));
    let headers = resolve_headers(params.headers, env_map, model_name, warnings);
    let forward_headers = normalize_header_names(params.forward_headers);

    Some(LiteLlmModel {
        model_name: model_name.to_string(),
        provider,
        upstream_model,
        api_base,
        api_key,
        api_version,
        headers,
        forward_headers,
    })
}

fn parse_yallm_provider(s: &str) -> Option<LiteLlmProvider> {
    match s.trim().to_ascii_lowercase().as_str() {
        "openai" => Some(LiteLlmProvider::OpenAI),
        "anthropic" => Some(LiteLlmProvider::Anthropic),
        "ollama" => Some(LiteLlmProvider::Ollama),
        "acp" => Some(LiteLlmProvider::Acp),
        _ => None,
    }
}

fn infer_provider_and_model(
    model: &str,
    custom_provider: Option<&str>,
) -> Option<(LiteLlmProvider, String)> {
    let model = model.trim();
    let model_prefix = model.split_once('/').map(|(prefix, _)| prefix);
    let provider = custom_provider
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or(model_prefix)
        .unwrap_or("openai")
        .to_ascii_lowercase();

    match provider.as_str() {
        "openai" | "openai_compatible" | "openai-compatible" | "openai_like" | "openai-like" => {
            Some((LiteLlmProvider::OpenAI, strip_model_prefix(model, "openai")))
        }
        "anthropic" => Some((
            LiteLlmProvider::Anthropic,
            strip_model_prefix(model, "anthropic"),
        )),
        "ollama" => Some((LiteLlmProvider::Ollama, strip_model_prefix(model, "ollama"))),
        "acp" => Some((LiteLlmProvider::Acp, strip_model_prefix(model, "acp"))),
        _ => None,
    }
}

fn strip_model_prefix(model: &str, provider: &str) -> String {
    model
        .strip_prefix(&format!("{provider}/"))
        .unwrap_or(model)
        .to_string()
}

fn resolve_config_value(
    value: &str,
    env_map: &HashMap<String, String>,
    model_name: &str,
    field: &str,
    warn_literal_secret: bool,
    warnings: &mut Vec<String>,
) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("none") {
        return None;
    }

    if let Some(name) = value.strip_prefix("os.environ/") {
        return env_lookup(name, env_map, model_name, field, warnings);
    }

    if let Some(name) = value.strip_prefix("${").and_then(|v| v.strip_suffix('}')) {
        return env_lookup(name, env_map, model_name, field, warnings);
    }

    if warn_literal_secret {
        warnings.push(format!(
            "yallm-config: LiteLLM model '{model_name}' uses a literal {field}; prefer os.environ/VAR"
        ));
    }
    Some(value.to_string())
}

fn env_lookup(
    name: &str,
    env_map: &HashMap<String, String>,
    model_name: &str,
    field: &str,
    warnings: &mut Vec<String>,
) -> Option<String> {
    let name = name.trim();
    match env_map.get(name).map(String::as_str).map(str::trim) {
        Some(value) if !value.is_empty() => Some(value.to_string()),
        _ => {
            warnings.push(format!(
                "yallm-config: LiteLLM model '{model_name}' references missing env var {name} for {field}"
            ));
            None
        }
    }
}

fn resolve_headers(
    raw: HashMap<String, serde_yaml_ng::Value>,
    env_map: &HashMap<String, String>,
    model_name: &str,
    warnings: &mut Vec<String>,
) -> Vec<(String, String)> {
    let mut entries = raw.into_iter().collect::<Vec<_>>();
    entries.sort_by(|(a, _), (b, _)| a.cmp(b));

    let mut out = Vec::new();
    for (name, value) in entries {
        let name = name.trim().to_string();
        if name.is_empty() {
            warnings.push(format!(
                "yallm-config: skipped LiteLLM model '{model_name}' header with empty name"
            ));
            continue;
        }

        let Some(value) = yaml_value_to_string(value) else {
            warnings.push(format!(
                "yallm-config: skipped LiteLLM model '{model_name}' header '{name}' with non-scalar value"
            ));
            continue;
        };

        let field = format!("headers.{name}");
        let Some(value) =
            resolve_header_value(&name, &value, env_map, model_name, &field, warnings)
        else {
            continue;
        };
        out.push((name, value));
    }
    out
}

fn resolve_header_value(
    header_name: &str,
    value: &str,
    env_map: &HashMap<String, String>,
    model_name: &str,
    field: &str,
    warnings: &mut Vec<String>,
) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("none") {
        return None;
    }

    if let Some(name) = value.strip_prefix("os.environ/") {
        return env_lookup(name, env_map, model_name, field, warnings);
    }

    if let Some(name) = value.strip_prefix("${").and_then(|v| v.strip_suffix('}')) {
        return env_lookup(name, env_map, model_name, field, warnings);
    }

    let resolved = interpolate_env_refs(value, env_map, model_name, field, warnings)?;
    if is_secret_header(header_name) && resolved == value {
        warnings.push(format!(
            "yallm-config: LiteLLM model '{model_name}' uses a literal secret header '{header_name}'; prefer ${{VAR}}"
        ));
    }
    Some(resolved)
}

fn interpolate_env_refs(
    value: &str,
    env_map: &HashMap<String, String>,
    model_name: &str,
    field: &str,
    warnings: &mut Vec<String>,
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
        match env_map.get(name).map(String::as_str).map(str::trim) {
            Some(value) if !value.is_empty() => out.push_str(value),
            _ => {
                warnings.push(format!(
                    "yallm-config: LiteLLM model '{model_name}' references missing env var {name} for {field}"
                ));
                return None;
            }
        }
        rest = &after_start[end + 1..];
    }

    out.push_str(rest);
    Some(out)
}

fn normalize_header_names(headers: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for header in headers {
        let name = header.trim().to_ascii_lowercase();
        if !name.is_empty() && seen.insert(name.clone()) {
            out.push(name);
        }
    }
    out
}

fn is_secret_header(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "authorization" | "proxy-authorization" | "x-api-key" | "api-key"
    )
}

#[derive(Debug, Deserialize, Default)]
struct LiteLlmConfigFile {
    #[serde(default)]
    model_list: Vec<LiteLlmEntry>,
}

#[derive(Debug, Deserialize, Default)]
struct LiteLlmEntry {
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    model_name: Option<String>,
    /// yallm-native parameters. Takes priority over `litellm_params` when both
    /// are present on the same entry. Provider must be specified explicitly via
    /// the `provider` field instead of as a `provider/model` prefix.
    #[serde(default)]
    yallm_params: Option<YallmParams>,
    /// LiteLLM-compatible parameters. Used as a fallback when `yallm_params` is
    /// absent. Provider may be encoded as a `provider/model` prefix or via
    /// `custom_llm_provider`.
    #[serde(default)]
    litellm_params: LiteLlmParams,
}

#[derive(Debug, Deserialize, Default)]
struct YallmParams {
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    provider: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    model: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    api_base: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    api_key: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    api_version: Option<String>,
    #[serde(default)]
    headers: HashMap<String, serde_yaml_ng::Value>,
    #[serde(default, deserialize_with = "deserialize_string_list")]
    forward_headers: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
struct LiteLlmParams {
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    model: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    api_base: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    api_key: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    api_version: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string")]
    custom_llm_provider: Option<String>,
    #[serde(default)]
    headers: HashMap<String, serde_yaml_ng::Value>,
    #[serde(default, deserialize_with = "deserialize_string_list")]
    forward_headers: Vec<String>,
}

fn deserialize_optional_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_yaml_ng::Value>::deserialize(deserializer)?;
    Ok(value.and_then(yaml_value_to_string))
}

fn deserialize_string_list<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_yaml_ng::Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(Vec::new());
    };

    let values = match value {
        serde_yaml_ng::Value::Null => Vec::new(),
        serde_yaml_ng::Value::Sequence(seq) => seq
            .into_iter()
            .filter_map(yaml_value_to_string)
            .flat_map(|s| split_header_names(&s))
            .collect(),
        value => yaml_value_to_string(value)
            .map(|s| split_header_names(&s))
            .unwrap_or_default(),
    };
    Ok(values)
}

fn split_header_names(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn yaml_value_to_string(value: serde_yaml_ng::Value) -> Option<String> {
    match value {
        serde_yaml_ng::Value::Null => None,
        serde_yaml_ng::Value::Bool(v) => Some(v.to_string()),
        serde_yaml_ng::Value::Number(v) => Some(v.to_string()),
        serde_yaml_ng::Value::String(v) => Some(v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_toml() {
        let c: ConfigFile = toml::from_str("").unwrap();
        assert!(c.env.is_empty());
    }

    #[test]
    fn parse_env_section() {
        let c: ConfigFile = toml::from_str(
            r#"
[env]
ANTHROPIC_API_KEY = "sk-ant-test"
YALLM_MODE = "proxy"
"#,
        )
        .unwrap();
        assert_eq!(c.env.get("ANTHROPIC_API_KEY").unwrap(), "sk-ant-test");
        assert_eq!(c.env.get("YALLM_MODE").unwrap(), "proxy");
    }

    #[test]
    fn dotenv_parses_simple_values() {
        let mut warnings = Vec::new();
        let env = parse_dotenv(
            r#"TEST_DOTENV_KEY=dotenv_value
ANOTHER_KEY="quoted val"
"#,
            &mut warnings,
        );

        assert_eq!(env.get("TEST_DOTENV_KEY").unwrap(), "dotenv_value");
        assert_eq!(env.get("ANOTHER_KEY").unwrap(), "quoted val");
        assert!(warnings.is_empty());
    }

    #[test]
    fn litellm_parses_supported_models_and_env_key() {
        let mut env = HashMap::new();
        env.insert("OPENAI_KEY".to_string(), "sk-env".to_string());
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: gpt-alias
    litellm_params:
      model: openai/gpt-4o
      api_base: https://openai-compatible.test/v1
      api_key: os.environ/OPENAI_KEY
      ignored_field: true
  - model_name: claude-alias
    litellm_params:
      model: anthropic/claude-3-haiku-20240307
      api_key: none
  - model_name: llama-alias
    litellm_params:
      model: ollama/llama3
  - model_name: agent-alias
    litellm_params:
      model: acp/codex
"#,
            &env,
            &mut warnings,
        );

        assert_eq!(models.len(), 4);
        assert_eq!(models[0].model_name, "gpt-alias");
        assert_eq!(models[0].provider, LiteLlmProvider::OpenAI);
        assert_eq!(models[0].upstream_model, "gpt-4o");
        assert_eq!(models[0].api_key.as_deref(), Some("sk-env"));
        assert_eq!(models[1].provider, LiteLlmProvider::Anthropic);
        assert_eq!(models[1].api_key, None);
        assert_eq!(models[2].provider, LiteLlmProvider::Ollama);
        assert_eq!(models[3].provider, LiteLlmProvider::Acp);
        assert_eq!(models[3].upstream_model, "codex");
        assert!(warnings.is_empty());
    }

    #[test]
    fn litellm_warns_for_literal_api_key() {
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: literal
    litellm_params:
      model: gpt-4o
      api_key: sk-literal
"#,
            &HashMap::new(),
            &mut warnings,
        );

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].api_key.as_deref(), Some("sk-literal"));
        assert!(warnings.iter().any(|w| w.contains("literal api_key")));
    }

    #[test]
    fn litellm_missing_env_reference_leaves_key_unset() {
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: missing
    litellm_params:
      model: gpt-4o
      api_key: ${MISSING_OPENAI_KEY}
"#,
            &HashMap::new(),
            &mut warnings,
        );

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].api_key, None);
        assert!(warnings.iter().any(|w| w.contains("MISSING_OPENAI_KEY")));
    }

    #[test]
    fn litellm_skips_unsupported_and_duplicate_models() {
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: gpt
    litellm_params:
      model: gpt-4o
  - model_name: gpt
    litellm_params:
      model: openai/gpt-4o-mini
  - model_name: azure-gpt
    litellm_params:
      model: azure/gpt-4o
      api_key: os.environ/AZURE_API_KEY
      extra: ignored
  - model_name: mixed
    litellm_params:
      model: azure/gpt-4o
  - model_name: mixed
    litellm_params:
      model: gpt-4o
"#,
            &HashMap::new(),
            &mut warnings,
        );

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].model_name, "gpt");
        assert_eq!(models[1].model_name, "mixed");
        assert!(warnings.iter().any(|w| w.contains("duplicate")));
        assert!(warnings.iter().any(|w| w.contains("unsupported")));
    }

    #[test]
    fn yallm_params_takes_priority_over_litellm_params() {
        // Same entry has both blocks; yallm_params wins for every field.
        let mut env = HashMap::new();
        env.insert("YALLM_KEY".to_string(), "yallm-secret".to_string());
        env.insert("LITELLM_KEY".to_string(), "litellm-secret".to_string());
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: dual
    yallm_params:
      provider: anthropic
      model: claude-yallm
      api_base: https://yallm.test
      api_key: os.environ/YALLM_KEY
      api_version: "2025-01-01"
    litellm_params:
      model: openai/gpt-litellm
      api_base: https://litellm.test
      api_key: os.environ/LITELLM_KEY
      api_version: "2020-01-01"
"#,
            &env,
            &mut warnings,
        );

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider, LiteLlmProvider::Anthropic);
        assert_eq!(models[0].upstream_model, "claude-yallm");
        assert_eq!(models[0].api_base.as_deref(), Some("https://yallm.test"));
        assert_eq!(models[0].api_key.as_deref(), Some("yallm-secret"));
        assert_eq!(models[0].api_version.as_deref(), Some("2025-01-01"));
        assert!(warnings.is_empty());
    }

    #[test]
    fn yallm_params_requires_explicit_provider() {
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: bad
    yallm_params:
      model: gpt-4o
"#,
            &HashMap::new(),
            &mut warnings,
        );

        assert_eq!(models.len(), 0);
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("missing yallm_params.provider"))
        );
    }

    #[test]
    fn yallm_params_rejects_unknown_provider() {
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: nope
    yallm_params:
      provider: bedrock
      model: anthropic.claude-3
"#,
            &HashMap::new(),
            &mut warnings,
        );

        assert_eq!(models.len(), 0);
        assert!(warnings.iter().any(|w| w.contains("'bedrock'")));
    }

    #[test]
    fn yallm_params_accepts_acp_provider() {
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: codex-agent
    yallm_params:
      provider: acp
      model: codex
"#,
            &HashMap::new(),
            &mut warnings,
        );

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider, LiteLlmProvider::Acp);
        assert_eq!(models[0].upstream_model, "codex");
        assert!(warnings.is_empty());
    }

    #[test]
    fn yallm_params_parses_headers_and_forward_header_allowlist() {
        let mut env = HashMap::new();
        env.insert("ROUTE_TOKEN".to_string(), "route-token".to_string());
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: headered
    yallm_params:
      provider: openai
      model: gpt-4o
      headers:
        Authorization: Bearer ${ROUTE_TOKEN}
        x-tenant: tenant-a
      forward_headers: authorization, x-request-id, Authorization
"#,
            &env,
            &mut warnings,
        );

        assert_eq!(models.len(), 1);
        assert!(
            models[0]
                .headers
                .iter()
                .any(|(k, v)| { k == "Authorization" && v == "Bearer route-token" })
        );
        assert!(
            models[0]
                .headers
                .iter()
                .any(|(k, v)| k == "x-tenant" && v == "tenant-a")
        );
        assert_eq!(
            models[0].forward_headers,
            vec!["authorization", "x-request-id"]
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn entry_without_yallm_params_falls_back_to_litellm_params() {
        let mut warnings = Vec::new();
        let models = parse_litellm_config_str(
            r#"
model_list:
  - model_name: legacy
    litellm_params:
      model: anthropic/claude-x
"#,
            &HashMap::new(),
            &mut warnings,
        );

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider, LiteLlmProvider::Anthropic);
        assert_eq!(models[0].upstream_model, "claude-x");
    }

    #[test]
    fn dotenv_does_not_override_existing_env() {
        let dir = tmpdir("yallm_dotenv_no_override");
        let env_path = dir.join(".env");
        fs::write(&env_path, "FOO=from_dotenv\nBAR=from_dotenv\n").unwrap();

        let mut env_map = HashMap::new();
        env_map.insert("FOO".to_string(), "from_os".to_string());
        let mut warnings = Vec::new();
        apply_dotenv(&env_path, &mut env_map, &mut warnings);

        assert_eq!(env_map.get("FOO").unwrap(), "from_os");
        assert_eq!(env_map.get("BAR").unwrap(), "from_dotenv");
        assert!(warnings.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn toml_glob_applies_local_overrides_in_order() {
        let dir = tmpdir("yallm_toml_glob");
        fs::write(
            dir.join("01.local.toml"),
            "[env]\nFOO = \"layer1\"\nBAR = \"layer1\"\n",
        )
        .unwrap();
        fs::write(dir.join("02.local.toml"), "[env]\nFOO = \"layer2\"\n").unwrap();
        // Non-matching file ignored.
        fs::write(dir.join("ignore.toml"), "[env]\nFOO = \"ignored\"\n").unwrap();

        let mut env_map = HashMap::new();
        let mut warnings = Vec::new();
        apply_toml_glob(&dir, ".local.toml", &mut env_map, "local", &mut warnings);

        assert_eq!(env_map.get("FOO").unwrap(), "layer2");
        assert_eq!(env_map.get("BAR").unwrap(), "layer1");
        assert!(warnings.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn secrets_toml_overrides_config_toml() {
        let dir = tmpdir("yallm_secrets_layer");
        fs::write(
            dir.join("config.toml"),
            "[env]\nANTHROPIC_API_KEY = \"public\"\n",
        )
        .unwrap();
        fs::write(
            dir.join("secrets.toml"),
            "[env]\nANTHROPIC_API_KEY = \"sk-real\"\n",
        )
        .unwrap();

        let mut env_map = HashMap::new();
        let mut warnings = Vec::new();
        apply_toml_file(
            &dir.join("config.toml"),
            &mut env_map,
            "project",
            &mut warnings,
        );
        apply_toml_file(
            &dir.join("secrets.toml"),
            &mut env_map,
            "secrets",
            &mut warnings,
        );

        assert_eq!(env_map.get("ANTHROPIC_API_KEY").unwrap(), "sk-real");
        let _ = fs::remove_dir_all(&dir);
    }

    fn tmpdir(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = env::temp_dir().join(format!("{name}_{nonce}_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
