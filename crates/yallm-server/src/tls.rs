//! TLS listener wrapping a TCP listener for HTTPS support.

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::serve::Listener;
use rustls::ServerConfig;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

/// Build a rustls `TlsAcceptor` from PEM cert and key files.
pub fn tls_acceptor(cert_path: &str, key_path: &str) -> io::Result<TlsAcceptor> {
    let cert_file = std::fs::File::open(cert_path)?;
    let key_file = std::fs::File::open(key_path)?;

    let certs = rustls_pemfile::certs(&mut io::BufReader::new(cert_file))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let key = rustls_pemfile::private_key(&mut io::BufReader::new(key_file))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no private key found"))?;

    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    Ok(TlsAcceptor::from(Arc::new(config)))
}

/// A TLS-wrapped TCP listener implementing axum's `Listener` trait.
pub struct TlsListener {
    tcp: TcpListener,
    acceptor: TlsAcceptor,
}

impl TlsListener {
    pub async fn bind(addr: SocketAddr, acceptor: TlsAcceptor) -> io::Result<Self> {
        let tcp = TcpListener::bind(addr).await?;
        Ok(Self { tcp, acceptor })
    }
}

impl Listener for TlsListener {
    type Io = tokio_rustls::server::TlsStream<tokio::net::TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let (stream, addr) = match self.tcp.accept().await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("yallm: TCP accept error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    continue;
                }
            };
            match self.acceptor.accept(stream).await {
                Ok(tls) => return (tls, addr),
                Err(e) => {
                    tracing::debug!("yallm: TLS handshake failed from {addr}: {e}");
                    continue;
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.tcp.local_addr()
    }
}
