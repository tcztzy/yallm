//! yallm - Unified LLM API converter proxy server

use yallm_cli::{Cli, Commands};

#[tokio::main]
async fn main() {
    let cli = Cli::parse_args();

    match cli.command {
        Some(Commands::Serve { port, host }) => {
            println!("Starting yallm proxy server on {}:{}", host, port);
            println!("Server implementation coming soon...");
        }
        None => {
            // Default behavior: start server with top-level args
            println!("Starting yallm proxy server on {}:{}", cli.host, cli.port);
            println!("Server implementation coming soon...");
        }
    }
}
