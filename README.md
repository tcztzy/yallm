# yallm

Yet another LLM: a high-performance API hub that sits between clients and model providers.

The goal is to connect to **any provider API** and expose **any downstream API surface**, while keeping
latency and overhead as low as possible (streaming-first, minimal allocations, strong typing where it
helps).

## Status

This project is **alpha / work in progress**.

What exists today:
- A Rust workspace with crates for an internal IR (`crates/yallm-ir`) and provider adapters
  (`crates/yallm-openai`, `crates/yallm-anthropic`, `crates/yallm-ollama`).
- OpenAPI-driven type generation via a proc-macro (`crates/yallm-macros`) for providers that ship
  OpenAPI specs.
- An axum server crate (`crates/yallm-server`) and a CLI crate (`crates/yallm-cli`), but the proxying
  routes are not implemented yet (endpoints currently return `501 Not Implemented`).

## Project Goals

- **Provider-agnostic**: integrate any upstream LLM provider.
- **Protocol-agnostic**: support multiple downstream APIs (e.g. OpenAI-compatible, provider-native,
  or custom) backed by the same upstream providers.
- **Performance-first**: keep translation overhead tiny; preserve streaming; avoid unnecessary
  buffering.
- **Strong types**: generate/derive types from OpenAPI where possible, and convert into a shared IR
  to reduce protocol cross-product complexity.

## Repository Layout

- `crates/yallm`: top-level binary (currently a placeholder that does not start the server yet)
- `crates/yallm-cli`: CLI argument parsing
- `crates/yallm-server`: axum HTTP server (routes currently stubbed)
- `crates/yallm-ir`: intermediate representation (IR) used for conversions
- `crates/yallm-openai`: OpenAI types + conversions (generated from OpenAPI + hand-written mapping)
- `crates/yallm-anthropic`: Anthropic types + conversions (generated from vendored OpenAPI spec)
- `crates/yallm-ollama`: Ollama types + conversions
- `crates/yallm-macros`: `include_openapi!` proc macro for compile-time type generation
- `python/yallm`: thin Python wrapper that locates and execs the `yallm` binary (packaging via
  `maturin` is configured in `pyproject.toml`)

For agent-specific workflow notes, see `AGENTS.md`.

## Development

Prereqs:
- Rust toolchain with **edition 2024** support.
- Network access may be needed for some OpenAPI-based builds unless specs are vendored locally
  (details below).

Common commands:
```bash
cargo build
cargo test
cargo fmt
cargo clippy --workspace --all-targets
```

## OpenAPI Type Generation

Provider crates use `yallm_macros::include_openapi! { ... }` to generate Rust types from OpenAPI
schemas at compile time.

Notes:
- The macro fetches and caches specs. If your default cache directory is not writable, set
  `YALLM_CACHE_DIR` to a writable path.
- `crates/yallm-anthropic/openapi.yml` is vendored in-repo so it can build offline. A `build.rs`
  exists to refresh the spec opportunistically (it should fall back to the vendored file when
  offline).
- `crates/yallm-openai` can fetch its OpenAPI spec at build time if a local copy is not present. If
  you need fully offline builds, place a local spec file in the crate (see the `include_openapi!`
  config in `crates/yallm-openai/src/lib.rs`).

## Roadmap (High Level)

- Implement the proxying logic in `crates/yallm-server` (routing + provider selection + streaming).
- Wire `crates/yallm` CLI to actually start the server.
- Add a provider/protocol registry to make “any provider” x “any API surface” composable.
- Add real-world docs and examples (curl requests, streaming examples, tool-calls).
- Add CI workflows (fmt/clippy/test).

## License

MIT. See `LICENSE`.

