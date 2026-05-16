//! yallm CLI implementation

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Unified LLM API converter proxy server
#[derive(Parser)]
#[command(name = "yallm")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Port to listen on
    #[arg(short, long, default_value = "4000")]
    pub port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// TLS certificate path (PEM); enables HTTPS when set with --tls-key
    #[arg(long)]
    pub tls_cert: Option<String>,

    /// TLS private key path (PEM)
    #[arg(long)]
    pub tls_key: Option<String>,

    /// Path to a LiteLLM config.yaml file
    #[arg(long)]
    pub litellm_config: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the proxy server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "4000")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// TLS certificate path (PEM); enables HTTPS when set with --tls-key
        #[arg(long)]
        tls_cert: Option<String>,

        /// TLS private key path (PEM)
        #[arg(long)]
        tls_key: Option<String>,

        /// Path to a LiteLLM config.yaml file
        #[arg(long)]
        litellm_config: Option<PathBuf>,
    },
}

impl Cli {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_top_level_litellm_config() {
        let cli = Cli::parse_from(["yallm", "--litellm-config", "litellm.yaml"]);

        assert_eq!(cli.litellm_config, Some(PathBuf::from("litellm.yaml")));
    }

    #[test]
    fn parses_serve_litellm_config() {
        let cli = Cli::parse_from(["yallm", "serve", "--litellm-config", "litellm.yaml"]);

        match cli.command {
            Some(Commands::Serve { litellm_config, .. }) => {
                assert_eq!(litellm_config, Some(PathBuf::from("litellm.yaml")));
            }
            _ => panic!("expected serve command"),
        }
    }
}
