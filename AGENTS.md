# AGENTS.md

## Dev environment tips
- This is a Rust workspace. Use `cargo metadata --no-deps` to list crates and targets.
- Build a specific crate with `cargo build -p <crate>` (e.g. `yallm-anthropic`).
- Run binaries via `cargo run -p <crate> -- <args>` when you need CLI behavior.
- The OpenAPI codegen macro may fetch specs at build time; set `YALLM_CACHE_DIR` if your cache directory is locked down.
- Workspace members live under `crates/*`; check each crate's `Cargo.toml` for names.

## Testing instructions
- Check CI steps in `.github/workflows` before changing test behavior.
- Run the full suite with `cargo test` from repo root.
- For a single crate: `cargo test -p <crate>`.
- To focus a single test: `cargo test -p <crate> <test_name>`.
- Run formatting and linting before committing: `cargo fmt` and `cargo clippy --workspace --all-targets`.
- Add or update tests for any behavior changes.

## PR instructions
- Title format: [<crate>] <Title>
- Ensure `cargo fmt`, `cargo clippy --workspace --all-targets`, and `cargo test` pass before committing.
