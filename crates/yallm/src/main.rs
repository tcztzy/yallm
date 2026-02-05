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

    let (host, port) = match cli.command {
        Some(Commands::Serve { port, host }) => (host, port),
        None => (cli.host, cli.port),
    };

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("invalid host/port combination");

    println!("Starting yallm proxy server on {addr}");
    if let Err(e) = yallm_server::run(ServerConfig { addr }).await {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}
