//! Starts the API. Everything it serves lives in the library next to it, so
//! the integration tests can build the same router without a socket.

use rust_nn_api::config::Config;
use tokio::net::TcpListener;
use tokio::signal;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_env("NN_API_LOG").unwrap_or_else(|_| EnvFilter::new("info,tower_http=info")))
        .init();

    let config = Config::from_env()?;
    let listener = TcpListener::bind(config.addr).await?;

    tracing::info!(
        address = %listener.local_addr()?,
        body_limit_mb = config.body_limit_bytes / (1024 * 1024),
        max_models = config.limits.max_models,
        "nn_api is listening"
    );

    axum::serve(listener, rust_nn_api::app_from_config(config))
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

/// Ctrl-C, or a SIGTERM from whatever is supervising the process.
async fn shutdown_signal() {
    let interrupt = async {
        signal::ctrl_c().await.expect("the interrupt handler can always be installed");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("the terminate handler can always be installed")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = interrupt => tracing::info!("interrupted, shutting down"),
        () = terminate => tracing::info!("terminated, shutting down"),
    }
}
