//! yallm - Unified LLM API converter proxy server

use std::net::SocketAddr;

use yallm_cli::{Cli, Commands};
use yallm_server::ServerConfig;

#[tokio::main]
async fn main() {
    // JSON logs to stdout (fluentd-friendly). Control verbosity via `RUST_LOG`.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .json()
        .init();

    let cli = Cli::parse_args();

    let (host, port, tls_cert, tls_key, litellm_config) = match cli.command {
        Some(Commands::Serve {
            port,
            host,
            tls_cert,
            tls_key,
            litellm_config,
        }) => (host, port, tls_cert, tls_key, litellm_config),
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
