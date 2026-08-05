//! yallm - Unified LLM API converter proxy server

use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use yallm_cli::{Cli, Commands};
use yallm_server::ServerConfig;

#[tokio::main]
async fn main() {
    let cli = Cli::parse_args();
    let acp_mode = matches!(&cli.command, Some(Commands::Acp { .. }));
    init_logging(acp_mode);

    let (host, port, tls_cert, tls_key, litellm_config) = match cli.command {
        Some(Commands::Serve {
            port,
            host,
            tls_cert,
            tls_key,
            litellm_config,
        }) => (host, port, tls_cert, tls_key, litellm_config),
        Some(Commands::Acp {
            model,
            litellm_config,
        }) => {
            if let Err(e) = run_acp(model, litellm_config).await {
                eprintln!("ACP error: {e}");
                std::process::exit(1);
            }
            return;
        }
        None => (
            cli.host,
            cli.port,
            cli.tls_cert,
            cli.tls_key,
            cli.litellm_config,
        ),
    };

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("invalid host/port combination");

    let scheme = if tls_cert.is_some() && tls_key.is_some() {
        "https"
    } else {
        "http"
    };
    println!("Starting yallm proxy server on {scheme}://{addr}");
    if let Err(e) = yallm_server::run(ServerConfig {
        addr,
        tls_cert,
        tls_key,
        litellm_config,
    })
    .await
    {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}

fn init_logging(acp_mode: bool) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    if acp_mode {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .with_writer(std::io::stderr)
            .init();
    } else {
        // JSON logs to stdout (fluentd-friendly). Control verbosity via `RUST_LOG`.
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .init();
    }
}

async fn run_acp(model: String, litellm_config: Option<PathBuf>) -> Result<(), String> {
    let loaded_config =
        yallm_config::load_with_options(yallm_config::LoadOptions { litellm_config });
    let state = Arc::new(yallm_server::AppState::from_loaded_config(loaded_config));

    yallm_acp::serve_stdio(model, move |request| {
        let state = state.clone();
        async move {
            let request_id = state.next_request_id();
            let headers = Default::default();
            yallm_server::complete_ir(&state, request_id, request, &headers)
                .await
                .map_err(|err| {
                    yallm_acp::internal_error(format!(
                        "yallm provider error (status {}): {}",
                        err.status, err.message
                    ))
                })
        }
    })
    .await
    .map_err(|e| format!("{e}"))
}
