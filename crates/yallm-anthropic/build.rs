//! Build script for yallm-anthropic
//!
//! Fetches the OpenAPI spec URL from the Anthropic TypeScript SDK repository
//! and downloads the spec for code generation.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // The proc macro reads the vendored spec at expansion time; cargo cannot
    // see that dependency, so declare it or spec updates silently keep the
    // stale cached codegen.
    println!("cargo:rerun-if-changed=openapi.yml");

    // IMPORTANT: build scripts should not mutate tracked files by default.
    // Opt in explicitly when updating the vendored OpenAPI spec. Only "1"
    // (or any non-"0" value) opts in — unset AND "0" both mean off. The CI
    // offline job pins the vendored spec with YALLM_UPDATE_ANTHROPIC_OPENAPI=0,
    // so treating "0" as a refresh trigger would overwrite it with the
    // live spec on every fresh checkout and break deterministic builds.
    if std::env::var_os("YALLM_UPDATE_ANTHROPIC_OPENAPI")
        .filter(|v| v != "0")
        .is_none()
    {
        return;
    }

    if let Err(e) = fetch_anthropic_openapi_spec() {
        eprintln!("Warning: Failed to fetch Anthropic OpenAPI spec: {}", e);
        eprintln!("Falling back to existing spec if available");
    }
}

fn fetch_anthropic_openapi_spec() -> Result<(), Box<dyn std::error::Error>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("yallm")
        .join("anthropic");

    fs::create_dir_all(&cache_dir)?;

    // Step 1: Get .stats.yml from the Anthropic SDK repo
    let stats_yml_path = cache_dir.join(".stats.yml");
    let stats_yml_url =
        "https://raw.githubusercontent.com/anthropics/anthropic-sdk-typescript/main/.stats.yml";

    // Fetch .stats.yml (always refresh to check for updates)
    let stats_content = fetch_url(stats_yml_url)?;
    fs::write(&stats_yml_path, &stats_content)?;

    // Step 2: Parse the YAML to extract openapi_spec_url and hash
    let openapi_spec_url = extract_openapi_spec_url(&stats_content)?;
    let openapi_spec_hash = extract_openapi_spec_hash(&stats_content)?;

    // Step 3: Check if we already have this version cached
    let spec_filename = format!("openapi-{}.yml", &openapi_spec_hash[..16]);
    let spec_path = cache_dir.join(&spec_filename);
    let symlink_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("openapi.yml");

    if !spec_path.exists() {
        // Download the spec
        println!("cargo:warning=Downloading Anthropic OpenAPI spec...");
        let spec_content = fetch_url(&openapi_spec_url)?;
        fs::write(&spec_path, &spec_content)?;
        println!(
            "cargo:warning=Cached Anthropic OpenAPI spec: {}",
            spec_filename
        );
    }

    // Step 4: Create/update symlink to the spec file in the crate directory
    // Remove existing symlink or file
    if symlink_path.exists() || symlink_path.is_symlink() {
        fs::remove_file(&symlink_path)?;
    }

    // Copy the spec to the crate directory (symlinks can be problematic cross-platform)
    fs::copy(&spec_path, &symlink_path)?;

    println!(
        "cargo:warning=Using Anthropic OpenAPI spec hash: {}",
        &openapi_spec_hash[..16]
    );

    Ok(())
}

fn fetch_url(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Try curl first (most portable)
    let output = Command::new("curl")
        .args(["-fsSL", url])
        .output()
        .map_err(|e| format!("Failed to execute curl: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8(output.stdout)?)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("curl failed: {}", stderr).into())
    }
}

fn extract_openapi_spec_url(yaml_content: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Simple YAML parsing - look for openapi_spec_url: "..."
    for line in yaml_content.lines() {
        let line = line.trim();
        if line.starts_with("openapi_spec_url:") {
            let value = line
                .strip_prefix("openapi_spec_url:")
                .unwrap()
                .trim()
                .trim_matches('"')
                .trim_matches('\'');
            return Ok(value.to_string());
        }
    }
    Err("openapi_spec_url not found in .stats.yml".into())
}

fn extract_openapi_spec_hash(yaml_content: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Simple YAML parsing - look for openapi_spec_hash: ...
    for line in yaml_content.lines() {
        let line = line.trim();
        if line.starts_with("openapi_spec_hash:") {
            let value = line
                .strip_prefix("openapi_spec_hash:")
                .unwrap()
                .trim()
                .trim_matches('"')
                .trim_matches('\'');
            return Ok(value.to_string());
        }
    }
    Err("openapi_spec_hash not found in .stats.yml".into())
}
