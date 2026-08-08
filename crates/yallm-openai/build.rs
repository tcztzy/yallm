//! Build script for yallm-openai
//!
//! The vendored OpenAPI spec is read by the `include_openapi!` proc macro.
//! Cargo cannot see that dependency, so without this file a spec update
//! would silently keep using the stale cached codegen. Declare it here so
//! touching the spec forces the macro to re-run.

fn main() {
    println!("cargo:rerun-if-changed=openapi.documented.yml");
}
