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
- Real upstream provider proxying exists (OpenAI / Anthropic / Ollama / ACP), with an **auto** mode that
  falls back to deterministic mock responses when provider configuration is missing.
- `stream=true` is supported on all compat endpoints, including true upstream streaming in proxy
  mode (with cross-provider stream translation).

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
- `crates/yallm-acp`: ACP upstream/downstream bridge helpers
- `crates/yallm-ir`: intermediate representation (IR) used for conversions
- `crates/yallm-openai`: OpenAI types + conversions (generated from OpenAPI + hand-written mapping)
- `crates/yallm-responses`: OpenAI Responses/Conversations types + IR mapping helpers
- `crates/yallm-storage`: local JSON storage for responses/conversations history
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
- OpenAI Responses-compatible: `POST /v1/responses`
- OpenAI Conversations-compatible: `POST /v1/conversations`
- Anthropic-compatible: `POST /v1/messages`
- Ollama-compatible: `POST /api/chat`

These currently return deterministic mock responses (useful for SDK integration tests). Wiring real
provider proxying is supported when configured (see below).

Responses/conversations state is stored locally so history can continue across provider switches.

## Proxying To Real Providers

### Provider Selection (Model Prefix)

`yallm` chooses the upstream provider from the `model` field using an openrouter/litellm-style
prefix:
- `openai:<model>` or `openai/<model>`
- `anthropic:<model>` or `anthropic/<model>`
- `ollama:<model>` or `ollama/<model>`
- `acp:<model>` or `acp/<model>`

If no prefix is present, `YALLM_DEFAULT_PROVIDER` is used (default: `openai`).

The prefix is stripped before calling upstream (e.g. `anthropic:claude-3-haiku-20240307` calls
Anthropic with model `claude-3-haiku-20240307`).

### Mode

`YALLM_MODE` controls behavior when provider configuration is missing:
- `auto` (default): proxy if configured, otherwise return deterministic mock responses
- `proxy`: always proxy; if no auth/header is supplied, the upstream decides whether to reject it
- `mock`: always mock (never calls upstream)

### Configuration Layers

Config is loaded from multiple layers, highest priority first:

1. `.yallm/secrets.toml` — project secrets (gitignored)
2. `.yallm/*.local.toml` — per-machine overrides, applied in filename order (gitignored)
3. `.yallm/config.toml` — committed project config
4. `~/.yallm/config.toml` — user defaults
5. OS environment variables
6. `.env` — only fills keys absent from OS env (does not override)

Each TOML file uses an `[env]` table:

```toml
[env]
ANTHROPIC_API_KEY = "sk-ant-..."
YALLM_DEFAULT_PROVIDER = "anthropic"
```

### Provider Environment Variables

OpenAI:
- `OPENAI_API_KEY` (compatibility sugar for `Authorization: Bearer <key>`)
- `OPENAI_BASE_URL` (default `https://api.openai.com`)
- `YALLM_OPENAI_HEADERS` (optional JSON object of additional upstream headers)

Anthropic:
- `ANTHROPIC_API_KEY` (compatibility sugar for `x-api-key: <key>`)
- `ANTHROPIC_AUTH_TOKEN` (compatibility sugar for `Authorization: Bearer <token>`)
- `ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`)
- `ANTHROPIC_VERSION` (default `2023-06-01`)
- `YALLM_ANTHROPIC_HEADERS` (optional JSON object of additional upstream headers)

Ollama:
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `YALLM_OLLAMA_HEADERS` (optional JSON object of additional upstream headers)

ACP:
- `YALLM_ACP_COMMAND` command used to launch the ACP agent subprocess, for example
  `npx -y @agentclientprotocol/codex-acp`
- `YALLM_ACP_CWD` optional working directory sent in `session/new`; defaults to the current
  process directory

For HTTP `stream=true` requests routed to ACP, yallm streams downstream chunks from ACP
`session/update` agent message notifications.

`YALLM_*_HEADERS` values may use `os.environ/VAR` or `${VAR}` references, including interpolation
such as `"Authorization": "Bearer ${UPSTREAM_TOKEN}"`.

Request header forwarding:
- `YALLM_FORWARD_HEADERS` controls which downstream request headers may be forwarded upstream.
  Defaults to common provider headers:
  `authorization,x-api-key,api-key,openai-organization,openai-project,anthropic-version,anthropic-beta`.
  Set it to `none` to disable request header forwarding.
- Header precedence is: provider env/default headers, then model-route headers, then allowlisted request
  headers. If a higher-priority source supplies an auth header (`authorization`, `x-api-key`, or
  `api-key`), lower-priority auth headers are dropped.

Local storage:
- `YALLM_DB_URL` (optional): persistent database URL for responses, conversations, and dashboard
  monitoring events. Defaults to `sqlite://<cache-dir>/yallm/storage.sqlite3`.
- Supported URL schemes are currently `sqlite://`, `sqlite::memory:`, `file://`, and bare file
  paths. Other database URL schemes fail explicitly until a backend is added in `yallm-storage`.
- `YALLM_STORAGE_PATH` is a deprecated compatibility path for the old local JSON store.

### LiteLLM Config Compatibility

Point yallm at a LiteLLM `config.yaml` with `--litellm-config <path>` or
`YALLM_LITELLM_CONFIG=<path>`. The CLI flag wins when both are set.

Each model alias is reachable via **every** supported HTTP protocol — yallm converts
between OpenAI / Anthropic / Ollama formats on the fly. An alias whose upstream
is `openai/gpt-4o` can be called from `/v1/chat/completions`, `/v1/messages`,
or `/api/chat`. `GET /v1/models` therefore lists the same alias set under every
`?interface=` filter.

#### `yallm_params` (preferred) vs `litellm_params` (compatibility)

`litellm_params` exists for drop-in compatibility with existing LiteLLM
config files. New configs should use `yallm_params`, which makes the provider
explicit instead of prefix-encoding it in the model string. When both blocks
appear on the same entry, **`yallm_params` wins**.

```yaml
model_list:
  - model_name: gpt-alias
    yallm_params:                    # preferred
      provider: openai               # one of: openai | anthropic | ollama | acp
      model: gpt-4o
      api_base: https://openai-compatible.example/v1
      api_key: os.environ/OPENAI_API_KEY
      headers:
        OpenAI-Organization: ${OPENAI_ORG_ID}
      forward_headers:
        - authorization
        - x-request-id

  - model_name: claude-alias         # legacy LiteLLM-style entry still works
    litellm_params:
      model: anthropic/claude-3-haiku-20240307
      api_key: os.environ/ANTHROPIC_API_KEY
      headers:
        anthropic-beta: prompt-caching-2024-07-31
```

Supported `litellm_params` v1 fields: `model_name` and
`litellm_params.model/api_base/api_key/api_version/custom_llm_provider/headers/forward_headers` for
OpenAI-compatible, Anthropic, Ollama, and ACP upstreams. Exact aliases take priority
over prefix/default-provider routing. Advanced LiteLLM router features,
fallback rules, budgets, wildcard model groups, and Azure OpenAI routing are
not implemented.

Use env references for keys instead of committing secrets — both
`os.environ/VAR` and `${VAR}` syntaxes are accepted. Header values additionally support interpolation,
for example `Authorization: Bearer ${UPSTREAM_TOKEN}`.

### ACP Downstream Mode

Run yallm itself as an ACP agent over stdio:

```bash
yallm acp --model anthropic:claude-sonnet-4-5
```

ACP clients send `initialize`, `session/new`, and `session/prompt` to this process. yallm converts
the ACP prompt into its IR, routes it through the configured upstream provider or mock mode, and
streams the answer back with `session/update`. In ACP mode, logs are written to stderr so stdout
remains valid ACP JSON-RPC.

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

### Deterministic Offline Build Recipe

To make builds reproducible and offline-friendly:

1. Prefetch dependencies once:
   ```bash
   cargo fetch --locked
   ```
2. Keep vendored specs pinned (do not refresh Anthropic during normal builds):
   ```bash
   export YALLM_UPDATE_ANTHROPIC_OPENAPI=0
   ```
3. Build/test in offline mode:
   ```bash
   export CARGO_NET_OFFLINE=true
   cargo check --workspace --locked
   cargo test --workspace --locked
   ```
4. (Optional guard) verify vendored Anthropic spec was not mutated:
   ```bash
   git diff --exit-code -- crates/yallm-anthropic/openapi.yml
   ```

## Roadmap (High Level)

- Expand streaming fidelity (tool-call deltas, finer-grained usage/finish metadata, and broader
  protocol edge-case coverage).
- Add a provider/protocol registry to make “any provider” x “any API surface” composable.
- Add real-world docs and examples (curl requests, streaming examples, tool-calls).
- Expand CI coverage (e.g., matrix builds and additional checks).

## License

MIT. See `LICENSE`.
