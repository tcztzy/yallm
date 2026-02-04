//! Procedural macros for yallm, including OpenAPI type generation.
//!
//! # Example
//!
//! ```ignore
//! yallm_macros::include_openapi! {
//!     url = "https://example.com/openapi.yml",
//!     root_types = ["CreateChatCompletionRequest"],
//! }
//! ```

use std::collections::HashSet;

use proc_macro::TokenStream;
use quote::quote;
use regex::Regex;
use serde_json::Value;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, LitStr, Token, braced, bracketed};

struct OpenApiConfig {
    url: Option<String>,
    local_file: Option<String>,
    root_types: Vec<String>,
    extra_definitions: Option<String>,
    debug_schema_path: Option<String>,
}

impl Parse for OpenApiConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut url = None;
        let mut local_file = None;
        let mut root_types = Vec::new();
        let mut extra_definitions = None;
        let mut debug_schema_path = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "url" => {
                    let lit: LitStr = input.parse()?;
                    url = Some(lit.value());
                }
                "local_file" => {
                    let lit: LitStr = input.parse()?;
                    local_file = Some(lit.value());
                }
                "root_types" => {
                    let content;
                    bracketed!(content in input);
                    while !content.is_empty() {
                        let lit: LitStr = content.parse()?;
                        root_types.push(lit.value());
                        if content.peek(Token![,]) {
                            content.parse::<Token![,]>()?;
                        }
                    }
                }
                "extra_definitions" => {
                    if input.peek(LitStr) {
                        // 兼容旧的字符串模式
                        let lit: LitStr = input.parse()?;
                        extra_definitions = Some(lit.value());
                    } else if input.peek(syn::token::Brace) {
                        // 新的直接 JSON 模式
                        let content;
                        braced!(content in input);
                        let tokens: proc_macro2::TokenStream = content.parse()?;
                        let json_str = format!("{{{}}}", tokens);
                        // 验证是有效的 JSON
                        let _: serde_json::Value =
                            serde_json::from_str(&json_str).map_err(|e| {
                                syn::Error::new(key.span(), format!("invalid JSON: {}", e))
                            })?;
                        extra_definitions = Some(json_str);
                    } else {
                        return Err(syn::Error::new(
                            input.span(),
                            "expected string literal or JSON object",
                        ));
                    }
                }
                "debug_schema_path" => {
                    let lit: LitStr = input.parse()?;
                    debug_schema_path = Some(lit.value());
                }
                _ => {
                    return Err(syn::Error::new(key.span(), format!("unknown key: {}", key)));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        // URL is required unless local_file is provided
        if url.is_none() && local_file.is_none() {
            return Err(syn::Error::new(
                input.span(),
                "missing `url` or `local_file`",
            ));
        }

        Ok(OpenApiConfig {
            url,
            local_file,
            root_types,
            extra_definitions,
            debug_schema_path,
        })
    }
}

/// Include OpenAPI-generated types inline.
///
/// This macro fetches the OpenAPI spec at compile time (with HTTP caching),
/// generates Rust types, and includes them directly in the source.
///
/// # Example
///
/// ```ignore
/// include_openapi! {
///     url = "https://example.com/openapi.yml",
///     root_types = ["ChatCompletionRequestMessage"],
///     extra_definitions = {
///         "MissingType": {
///             "type": "object",
///             "properties": { "name": { "type": "string" } }
///         }
///     },
/// }
/// ```
#[proc_macro]
pub fn include_openapi(input: TokenStream) -> TokenStream {
    let config = syn::parse_macro_input!(input as OpenApiConfig);

    let code = match generate_types(&config) {
        Ok(code) => code,
        Err(e) => {
            return syn::Error::new(proc_macro2::Span::call_site(), e.to_string())
                .to_compile_error()
                .into();
        }
    };

    let tokens: proc_macro2::TokenStream = match code.parse() {
        Ok(t) => t,
        Err(e) => {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("Failed to parse generated code: {}", e),
            )
            .to_compile_error()
            .into();
        }
    };

    quote! { #tokens }.into()
}

fn generate_types(config: &OpenApiConfig) -> Result<String, Box<dyn std::error::Error>> {
    // Try local file first, then URL with caching
    let spec_yaml = if let Some(ref local) = config.local_file {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
        let local_path = std::path::Path::new(&manifest_dir).join(local);
        if local_path.exists() {
            std::fs::read_to_string(&local_path)?
        } else if let Some(ref url) = config.url {
            fetch_with_cache(url)?
        } else {
            return Err(format!("Local file not found: {}", local_path.display()).into());
        }
    } else if let Some(ref url) = config.url {
        fetch_with_cache(url)?
    } else {
        return Err("No URL or local file specified".into());
    };

    let spec_yaml = preprocess_yaml(&spec_yaml);
    let spec: Value = serde_yaml_ng::from_str(&spec_yaml)?;

    let mut schemas = spec
        .get("components")
        .and_then(|c| c.get("schemas"))
        .ok_or("No components/schemas in OpenAPI spec")?
        .clone();

    convert_openapi_to_json_schema(&mut schemas);

    // Extract inline type enums to avoid typify merging them
    extract_inline_type_enums(&mut schemas);

    // Add extra definitions if provided
    if let Some(ref extra) = config.extra_definitions {
        let extra_defs: serde_json::Map<String, Value> = serde_json::from_str(extra)?;
        if let Value::Object(ref mut map) = schemas {
            for (k, v) in extra_defs {
                map.insert(k, v);
            }
        }
    }

    let root_refs: Vec<&str> = config.root_types.iter().map(|s| s.as_str()).collect();
    let schemas = filter_schemas(schemas, &root_refs);

    let mut json_schema = serde_json::json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": schemas,
    });
    convert_openapi_to_json_schema(&mut json_schema);

    // Write debug schema if requested
    if let Some(ref debug_path) = config.debug_schema_path {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
        let debug_file = std::path::Path::new(&manifest_dir).join(debug_path);
        let formatted = serde_json::to_string_pretty(&json_schema)?;
        std::fs::write(&debug_file, formatted)?;
    }

    let mut type_space = typify::TypeSpace::new(
        typify::TypeSpaceSettings::default().with_derive("PartialEq".to_string()),
    );

    let root_schema: schemars::schema::RootSchema = serde_json::from_value(json_schema.clone())
        .map_err(|e| format!("Failed to parse JSON schema: {}", e,))?;
    type_space
        .add_root_schema(root_schema)
        .map_err(|e| format!("Failed to add root schema to type space: {}", e))?;

    Ok(type_space.to_stream().to_string())
}

fn fetch_with_cache(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    use http_cache_reqwest::{CACacheManager, Cache, CacheMode, HttpCache, HttpCacheOptions};
    use reqwest_middleware::ClientBuilder;

    let cache_dir = resolve_cache_dir()?;

    let rt = tokio::runtime::Runtime::new()?;

    rt.block_on(async {
        let client = ClientBuilder::new(reqwest::Client::new())
            .with(Cache(HttpCache {
                mode: CacheMode::Default,
                manager: CACacheManager { path: cache_dir },
                options: HttpCacheOptions::default(),
            }))
            .build();

        let response = client.get(url).send().await?;
        let text = response.text().await?;
        Ok(text)
    })
}

fn resolve_cache_dir() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    if let Ok(dir) = std::env::var("YALLM_CACHE_DIR") {
        candidates.push(std::path::PathBuf::from(dir));
    }

    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        candidates.push(std::path::PathBuf::from(dir).join("yallm-cache"));
    }

    if let Some(dir) = dirs::cache_dir() {
        candidates.push(dir.join("yallm"));
    }

    candidates.push(std::env::temp_dir().join("yallm-cache"));

    for candidate in candidates {
        if ensure_writable_dir(&candidate).is_ok() {
            return Ok(candidate);
        }
    }

    Err("Failed to create cache directory for OpenAPI spec".into())
}

fn ensure_writable_dir(path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;

    std::fs::create_dir_all(path)?;
    let unique = format!(
        ".yallm_cache_write_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let test_path = path.join(unique);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&test_path)?;
    file.write_all(b"ok")?;
    std::fs::remove_file(test_path)?;
    Ok(())
}

// ============================================================================
// OpenAPI to JSON Schema conversion utilities
// ============================================================================

/// Preprocess YAML to fix problematic values.
fn preprocess_yaml(yaml: &str) -> String {
    let re = Regex::new(r"minimum:\s*-\d{15,}").unwrap();
    let yaml = re.replace_all(yaml, "minimum: -2147483648").to_string();

    let re = Regex::new(r"maximum:\s*\d{15,}").unwrap();
    re.replace_all(&yaml, "maximum: 2147483647").to_string()
}

/// Convert OpenAPI schema to JSON Schema format.
fn convert_openapi_to_json_schema(value: &mut Value) {
    match value {
        Value::Object(map) => {
            // Remove x- extension fields
            let keys_to_remove: Vec<String> = map
                .keys()
                .filter(|k| k.starts_with("x-"))
                .cloned()
                .collect();
            for key in keys_to_remove {
                map.remove(&key);
            }

            // Convert $ref paths
            if let Some(Value::String(ref_path)) = map.get_mut("$ref")
                && ref_path.starts_with("#/components/schemas/")
            {
                *ref_path = ref_path.replace("#/components/schemas/", "#/definitions/");
            }

            // For object types with properties, identify nullable properties BEFORE simplification
            // and remove them from required
            let nullable_props: HashSet<String> = if map.get("type")
                == Some(&Value::String("object".to_string()))
            {
                if let Some(Value::Object(props)) = map.get("properties") {
                    props
                        .iter()
                        .filter_map(|(name, prop_schema)| {
                            if let Value::Object(prop_obj) = prop_schema {
                                // Check if property has anyOf with null type
                                if let Some(Value::Array(any_of)) = prop_obj.get("anyOf") {
                                    let has_null = any_of.iter().any(|v| {
                                        matches!(v, Value::Object(m) if m.get("type") == Some(&Value::String("null".to_string())))
                                    });
                                    if has_null {
                                        return Some(name.clone());
                                    }
                                }
                                // Check if property has default: null (indicates nullable)
                                if prop_obj.get("default") == Some(&Value::Null) {
                                    return Some(name.clone());
                                }
                            }
                            None
                        })
                        .collect()
                } else {
                    HashSet::new()
                }
            } else {
                HashSet::new()
            };

            // Remove nullable properties from required
            if !nullable_props.is_empty()
                && let Some(Value::Array(required)) = map.get_mut("required")
            {
                required.retain(|v| {
                    if let Value::String(s) = v {
                        !nullable_props.contains(s)
                    } else {
                        true
                    }
                });
            }

            // Handle anyOf with null type - simplify to single type
            let replacement = if let Some(Value::Array(any_of)) = map.get("anyOf") {
                let non_null: Vec<&Value> = any_of
                    .iter()
                    .filter(|v| {
                        !matches!(v, Value::Object(m) if m.get("type") == Some(&Value::String("null".to_string())))
                    })
                    .collect();

                let has_null = any_of.len() != non_null.len();

                if has_null && non_null.len() == 1 {
                    if let Value::Object(inner) = non_null[0] {
                        Some(inner.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(inner) = replacement {
                map.remove("anyOf");
                for (k, v) in inner {
                    map.insert(k, v);
                }
            }

            // Handle OpenAPI 3.0 exclusiveMinimum/exclusiveMaximum boolean format
            if let Some(Value::Bool(true)) = map.get("exclusiveMinimum") {
                if let Some(min_val) = map.remove("minimum") {
                    map.insert("exclusiveMinimum".to_string(), min_val);
                } else {
                    map.remove("exclusiveMinimum");
                }
            } else if let Some(Value::Bool(false)) = map.get("exclusiveMinimum") {
                map.remove("exclusiveMinimum");
            }

            if let Some(Value::Bool(true)) = map.get("exclusiveMaximum") {
                if let Some(max_val) = map.remove("maximum") {
                    map.insert("exclusiveMaximum".to_string(), max_val);
                } else {
                    map.remove("exclusiveMaximum");
                }
            } else if let Some(Value::Bool(false)) = map.get("exclusiveMaximum") {
                map.remove("exclusiveMaximum");
            }

            // Handle nullable: true
            if let Some(Value::Bool(true)) = map.remove("nullable") {
                if let Some(type_val) = map.get("type").cloned() {
                    // If type exists, convert to array with null
                    // e.g., "string" -> ["string", "null"]
                    match type_val {
                        Value::String(t) => {
                            map.insert(
                                "type".to_string(),
                                Value::Array(vec![
                                    Value::String(t),
                                    Value::String("null".to_string()),
                                ]),
                            );
                        }
                        Value::Array(mut arr) => {
                            // Already an array, just add "null" if not present
                            if !arr.contains(&Value::String("null".to_string())) {
                                arr.push(Value::String("null".to_string()));
                            }
                            map.insert("type".to_string(), Value::Array(arr));
                        }
                        _ => {}
                    }
                } else if let Some(Value::String(_)) = map.get("$ref") {
                    // Has $ref but no type, use anyOf
                    let ref_val = map.remove("$ref").unwrap();
                    let ref_schema = serde_json::json!({"$ref": ref_val});
                    let null_schema = serde_json::json!({"type": "null"});
                    map.insert(
                        "anyOf".to_string(),
                        Value::Array(vec![ref_schema, null_schema]),
                    );
                } else {
                    // No type and no $ref, just set type to null
                    map.insert("type".to_string(), Value::String("null".to_string()));
                }
            }

            // Remove OpenAPI-specific fields
            map.remove("discriminator");
            map.remove("example");
            map.remove("examples");
            map.remove("externalDocs");
            map.remove("xml");
            map.remove("nullable");

            // Convert const to enum with single value (more compatible)
            if let Some(const_val) = map.remove("const") {
                map.insert("enum".to_string(), Value::Array(vec![const_val]));
            }

            // Remove default: null when type is not nullable (causes validation errors)
            if let Some(Value::Null) = map.get("default") {
                let type_val = map.get("type");
                let is_nullable = match type_val {
                    Some(Value::Array(arr)) => arr.contains(&Value::String("null".to_string())),
                    Some(Value::String(s)) => s == "null",
                    _ => false,
                };
                if !is_nullable {
                    map.remove("default");
                }
            }

            // Relax overly generic string titles that collide across schemas.
            if let Some(Value::String(title)) = map.get("title") {
                let is_string = match map.get("type") {
                    Some(Value::String(t)) => t == "string",
                    Some(Value::Array(arr)) => arr
                        .iter()
                        .any(|v| matches!(v, Value::String(s) if s == "string")),
                    _ => false,
                };
                if is_string {
                    if title == "Id" {
                        map.remove("pattern");
                    } else if title == "Name" {
                        map.remove("enum");
                        map.remove("minLength");
                        map.remove("maxLength");
                    }
                }
            }

            // Recurse
            for (_, v) in map.iter_mut() {
                convert_openapi_to_json_schema(v);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                convert_openapi_to_json_schema(item);
            }
        }
        _ => {}
    }
}

/// Collect all $ref references from a JSON value.
fn collect_refs(value: &Value, refs: &mut HashSet<String>) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(ref_path)) = map.get("$ref") {
                // Support both OpenAPI format and JSON Schema format
                if let Some(name) = ref_path
                    .strip_prefix("#/definitions/")
                    .or_else(|| ref_path.strip_prefix("#/components/schemas/"))
                {
                    refs.insert(name.to_string());
                }
            }
            for v in map.values() {
                collect_refs(v, refs);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                collect_refs(item, refs);
            }
        }
        _ => {}
    }
}

/// Filter schemas to only include root types and their transitive dependencies.
fn filter_schemas(schemas: Value, root_types: &[&str]) -> Value {
    let schemas_map = match &schemas {
        Value::Object(map) => map,
        _ => return schemas,
    };

    let mut needed: HashSet<String> = root_types.iter().map(|s| s.to_string()).collect();
    let mut to_process: Vec<String> = root_types.iter().map(|s| s.to_string()).collect();

    while let Some(type_name) = to_process.pop() {
        if let Some(schema) = schemas_map.get(&type_name) {
            let mut refs = HashSet::new();
            collect_refs(schema, &mut refs);
            for r in refs {
                if needed.insert(r.clone()) {
                    to_process.push(r);
                }
            }
        }
    }

    let filtered: serde_json::Map<String, Value> = schemas_map
        .iter()
        .filter(|(k, _)| needed.contains(*k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    Value::Object(filtered)
}

/// Extract inline `type` enum fields into separate named definitions.
///
/// This prevents typify from merging all `type` enum fields into a single enum,
/// which would cause serialization/deserialization issues.
fn extract_inline_type_enums(schemas: &mut Value) {
    let mut new_definitions: serde_json::Map<String, Value> = serde_json::Map::new();

    if let Value::Object(schemas_map) = schemas {
        // Collect modifications first
        let mut modifications: Vec<(String, String)> = Vec::new();

        for (type_name, schema) in schemas_map.iter() {
            if let Value::Object(obj) = schema
                && let Some(Value::Object(props)) = obj.get("properties")
                && let Some(Value::Object(type_prop)) = props.get("type")
            {
                // Check if it's a single-value enum
                if let Some(Value::Array(enum_vals)) = type_prop.get("enum")
                    && enum_vals.len() == 1
                {
                    // Create a unique type name
                    let unique_type_name = format!("{}Type", type_name);
                    modifications.push((type_name.clone(), unique_type_name.clone()));

                    // Create the new definition
                    let mut new_def = type_prop.clone();
                    new_def.insert("title".to_string(), Value::String(unique_type_name.clone()));
                    new_definitions.insert(unique_type_name, Value::Object(new_def));
                }
            }
        }

        // Apply modifications
        for (type_name, unique_type_name) in modifications {
            if let Some(Value::Object(obj)) = schemas_map.get_mut(&type_name)
                && let Some(Value::Object(props)) = obj.get_mut("properties")
                && props.contains_key("type")
            {
                // Replace inline enum with $ref
                props.insert(
                    "type".to_string(),
                    serde_json::json!({
                        "$ref": format!("#/definitions/{}", unique_type_name)
                    }),
                );
            }
        }

        // Add new definitions
        for (name, def) in new_definitions {
            schemas_map.insert(name, def);
        }
    }
}
