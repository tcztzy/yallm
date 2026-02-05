//! yallm - Unified LLM API converter proxy server

use std::net::SocketAddr;

use yallm_cli::{Cli, Commands};
use yallm_server::ServerConfig;

#[tokio::main]
async fn main() {
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
