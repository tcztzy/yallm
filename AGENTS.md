# AGENTS.md

## Dev environment tips
- This is a Rust workspace. Use `cargo metadata --no-deps` to list crates and targets.
- Build a specific crate with `cargo build -p <crate>` (e.g. `yallm-anthropic`).
- Run binaries via `cargo run -p <crate> -- <args>` when you need CLI behavior.
- The OpenAPI codegen macro may fetch specs at build time; set `YALLM_CACHE_DIR` if your cache directory is locked down.
- Anthropic's vendored OpenAPI spec is only refreshed when `YALLM_UPDATE_ANTHROPIC_OPENAPI=1` is set (to avoid mutating tracked files during normal builds).
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

## Documentation conventions (agent-facing module docs)

Every source file in the core crates (`yallm-server`, `yallm-acp`,
`yallm-storage`, `yallm-ir`) carries a `//!` module doc written as the file's
entry point — read it before reading the code. It answers five questions:

1. **Purpose** — one-paragraph boundary of what the module does.
2. **Public surface** — key types/functions and their roles, one line each.
3. **Data flow** — who calls whom, what flows in and out.
4. **Invariants** — what must stay true (e.g. stream parsers must emit a
   `Stop` event).
5. **Gotchas** — the mistakes a reader is likely to make.

`#![warn(missing_docs)]` is on in those crates: every public item must carry
a `///` doc. Trivial fields get a terse line; anything non-obvious gets
context (env var names, defaults, failure modes). `cargo build` surfaces any
gap — keep it green.

Doctests: runnable examples belong in the doc comment of the entry-point
function (see `yallm-acp::ir_to_prompt`, `yallm-storage::LocalStore::open_database_url_sync`);
they are compiled by `cargo test --doc`, so they cannot rot. When writing
module docs, verify claims against the code — a stale doc costs the next
reader more tokens than no doc.

## Pre-commit and doc checks

- `pre-commit run --all-files` (after `pre-commit install`) runs: trailing
  whitespace, YAML checks, ruff for `python/`, and three Rust hooks —
  `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and `cargo doc` under `RUSTDOCFLAGS="-D warnings"` (fires the rustdoc
  lints from `[workspace.lints.rustdoc]`). The Rust hooks only run when
  `.rs`/`.toml` files changed.
- `[lints] workspace = true` is set in every crate manifest — inherited
  rustdoc lints (`broken-intra-doc-links`, `missing-crate-level-docs`) only
  fire during `cargo doc`, never during `cargo build`/`clippy`. If you add
  or move doc links, verify with
  `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`.
- Known: the `pytest` and `ty` hooks currently fail on the python side
  (pytest missing from the root venv; ty diagnostics in `python/yallm`) —
  the Rust hooks are the gate that matters for Rust changes.
