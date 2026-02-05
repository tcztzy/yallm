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
- An axum server crate (`crates/yallm-server`) and a CLI crate (`crates/yallm-cli`).
- Compatibility endpoints exist and are validated with official Python SDKs.
- Real upstream provider proxying exists (OpenAI / Anthropic / Ollama), with an **auto** mode that
  falls back to deterministic mock responses when provider configuration is missing.

## Project Goals

- **Provider-agnostic**: integrate any upstream LLM provider.
- **Protocol-agnostic**: support multiple downstream APIs (e.g. OpenAI-compatible, provider-native,
  or custom) backed by the same upstream providers.
- **Performance-first**: keep translation overhead tiny; preserve streaming; avoid unnecessary
  buffering.
- **Strong types**: generate/derive types from OpenAPI where possible, and convert into a shared IR
  to reduce protocol cross-product complexity.

## Repository Layout

- `crates/yallm`: top-level binary (starts the HTTP server)
- `crates/yallm-cli`: CLI argument parsing
- `crates/yallm-server`: axum HTTP server (compat endpoints; real proxying + mock fallback)
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

## Run The Server (Mock Mode)

Start the server:
```bash
cargo run -p yallm -- serve --host 127.0.0.1 --port 8080
```

Compatibility endpoints:
- OpenAI-compatible: `POST /v1/chat/completions`
- Anthropic-compatible: `POST /v1/messages`
- Ollama-compatible: `POST /api/chat`

These currently return deterministic mock responses (useful for SDK integration tests). Wiring real
provider proxying is supported when configured (see below).

## Proxying To Real Providers

### Provider Selection (Model Prefix)

`yallm` chooses the upstream provider from the `model` field using an openrouter/litellm-style
prefix:
- `openai:<model>` or `openai/<model>`
- `anthropic:<model>` or `anthropic/<model>`
- `ollama:<model>` or `ollama/<model>`

If no prefix is present, `YALLM_DEFAULT_PROVIDER` is used (default: `openai`).

The prefix is stripped before calling upstream (e.g. `anthropic:claude-3-haiku-20240307` calls
Anthropic with model `claude-3-haiku-20240307`).

### Mode

`YALLM_MODE` controls behavior when provider configuration is missing:
- `auto` (default): proxy if configured, otherwise return deterministic mock responses
- `proxy`: always proxy (missing config returns an error)
- `mock`: always mock (never calls upstream)

### Provider Environment Variables

OpenAI:
- `OPENAI_API_KEY` (required for proxying)
- `OPENAI_BASE_URL` (default `https://api.openai.com`)

Anthropic:
- `ANTHROPIC_API_KEY` (required for proxying)
- `ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`)
- `ANTHROPIC_VERSION` (default `2023-06-01`)

Ollama:
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)

### Logging (Fluentd-Friendly JSON)

`yallm` emits JSON logs to stdout using `tracing`. The server logs:
- inbound HTTP requests (`event=http.in`)
- outbound HTTP responses (`event=http.out`)
- provider requests/responses (`event=provider.out` / `event=provider.in`)
- conversion stages (`event=convert.*`)

Redaction/truncation controls:
- `YALLM_LOG_REDACT_SECRETS=1` (default): redact auth headers and common secret JSON keys
- `YALLM_LOG_BODY_MAX_BYTES=0` (default): log full bodies; set to truncate and still log `body_len`

## OpenAPI Type Generation

Provider crates use `yallm_macros::include_openapi! { ... }` to generate Rust types from OpenAPI
schemas at compile time.

Notes:
- The macro fetches and caches specs. If your default cache directory is not writable, set
  `YALLM_CACHE_DIR` to a writable path.
- `crates/yallm-anthropic/openapi.yml` is vendored in-repo so it can build offline. A `build.rs`
  exists to refresh the spec, but is **opt-in** (set `YALLM_UPDATE_ANTHROPIC_OPENAPI=1`).
- `crates/yallm-openai` can fetch its OpenAPI spec at build time if a local copy is not present. If
  you need fully offline builds, place a local spec file in the crate (see the `include_openapi!`
  config in `crates/yallm-openai/src/lib.rs`).

## Roadmap (High Level)

- Implement streaming across all compat endpoints (currently `stream=true` returns 501).
- Add a provider/protocol registry to make “any provider” x “any API surface” composable.
- Add real-world docs and examples (curl requests, streaming examples, tool-calls).
- Add CI workflows (fmt/clippy/test).

## License

MIT. See `LICENSE`.
