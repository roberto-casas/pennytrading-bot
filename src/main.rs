/// main.rs – Entry point for the PennyTrading Bot.
///
/// Orchestrates startup, the WebSocket feed, strategy loop, position
/// monitoring, and the live ratatui dashboard.
mod config;
mod dashboard;
mod database;
mod kelly;
mod models;
mod polymarket;
mod strategy;
mod trader;

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use crossterm::event::EventStream;
use futures_util::StreamExt;
use std::sync::{Arc, RwLock};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use config::Settings;
use database::Database;
use models::{AppState, BotSession};
use polymarket::PolymarketClient;
use strategy::Strategy;
use trader::Trader;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "pennytrading-bot",
    about = "Polymarket penny strategy trading bot – BTC/ETH 5-min & 15-min markets",
    version
)]
struct Cli {
    /// Run in dry-run mode: analyse markets and log signals but place no real orders.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Disable the interactive dashboard and print logs to stdout instead.
    #[arg(long, default_value_t = false)]
    no_dashboard: bool,

    /// Path to the YAML configuration file.
    #[arg(long, default_value = "config.yaml")]
    config: String,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load settings (YAML + env override)
    let settings = Settings::load(&cli.config, Some(cli.dry_run))?;

    // Logging – respects LOG_LEVEL env var; falls back to config
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&settings.bot.log_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    if settings.bot.dry_run {
        info!("🟡 DRY RUN mode – no real orders will be placed.");
    } else if !settings.has_credentials() {
        warn!("⚠️  No API credentials found – forcing dry-run mode.");
    }

    info!(
        "Watching assets: {:?}  resolutions: {:?}",
        settings.markets.assets, settings.markets.resolutions
    );

    // Open database
    let db = Arc::new(Database::open(&settings.bot.db_path)?);
    info!("Database opened: {}", settings.bot.db_path);

    // Build polymarket client
    let client = Arc::new(PolymarketClient::new(
        settings.clob.rest_url.clone(),
        settings.clob.ws_url.clone(),
        settings.clob.gamma_url.clone(),
        settings.poly_api_key.clone(),
        settings.poly_api_secret.clone(),
        settings.poly_api_passphrase.clone(),
        settings.bot.dry_run,
    )?);

    // Shared application state for the dashboard
    let state = Arc::new(RwLock::new(AppState {
        session_id: Uuid::new_v4().to_string(),
        started_at: Some(Utc::now()),
        dry_run: settings.bot.dry_run,
        ..AppState::default()
    }));

    let session_id = state.read().unwrap().session_id.clone();

    // Create and persist a new bot session
    let session = BotSession {
        session_id: session_id.clone(),
        started_at: Utc::now(),
        ended_at: None,
        dry_run: settings.bot.dry_run,
        trades_opened: 0,
        trades_closed: 0,
        total_pnl_usdc: 0.0,
        config_snapshot: serde_json::to_string(&settings.bot).unwrap_or_default(),
    };
    db.upsert_session(&session)?;

    // Load any positions that were open from a previous run
    {
        let mut st = state.write().unwrap();
        st.open_positions = db.get_open_positions().unwrap_or_default();
        info!(
            "Loaded {} open position(s) from previous run",
            st.open_positions.len()
        );
    }

    // Discover markets
    info!("Fetching BTC/ETH markets from Gamma API…");
    let markets = client
        .fetch_btc_eth_markets(
            &settings.markets.assets,
            &settings.markets.resolutions,
            settings.markets.max_markets_per_asset,
        )
        .await
        .unwrap_or_default();

    for m in &markets {
        if let Err(e) = db.upsert_market(m) {
            warn!("Failed to save market {}: {e}", m.condition_id);
        }
    }

    {
        let mut st = state.write().unwrap();
        st.markets = markets.clone();
        st.add_log(format!("Discovered {} market(s)", markets.len()));
    }

    // Fetch initial balance
    let balance = client.get_balance().await;
    {
        let mut st = state.write().unwrap();
        st.balance_usdc = if balance > 0.0 { balance } else { 1000.0 }; // default for dry-run
    }

    // Start WebSocket feed
    let (mut ws_rx, _ws_handle) = client.start_websocket(markets.clone());
    {
        let mut st = state.write().unwrap();
        st.ws_connected = true;
        st.add_log("WebSocket connection started".to_string());
    }

    // Build strategy and trader
    let strategy = Strategy::new(
        settings.strategy.clone(),
        settings.kelly.clone(),
        settings.risk.clone(),
    );
    let trader = Trader::new(
        settings.risk.clone(),
        settings.strategy.clone(),
        settings.bot.dry_run,
    );

    // Poll interval
    let poll_ms = (settings.bot.poll_interval_seconds * 1000.0) as u64;
    let mut poll_ticker = tokio::time::interval(std::time::Duration::from_millis(poll_ms));

    // Dashboard setup (unless --no-dashboard)
    let mut terminal = if !cli.no_dashboard {
        Some(dashboard::setup_terminal()?)
    } else {
        None
    };

    let refresh_ms = (settings.dashboard.refresh_rate * 1000.0) as u64;
    let mut dash_ticker =
        tokio::time::interval(std::time::Duration::from_millis(refresh_ms));

    let mut event_stream = EventStream::new();

    info!("Bot started.  Press 'q' to quit.");

    // -----------------------------------------------------------------------
    // Main event loop
    // -----------------------------------------------------------------------
    loop {
        // Render dashboard
        if let Some(ref mut term) = terminal {
            let st = state.read().unwrap().clone();
            term.draw(|f| dashboard::render(f, &st))?;
        }

        tokio::select! {
            // ── Dashboard keyboard events ──────────────────────────────────
            Some(Ok(event)) = event_stream.next() => {
                if terminal.is_some() && dashboard::handle_event(&event) {
                    break;
                }
            }

            // ── Dashboard refresh tick ─────────────────────────────────────
            _ = dash_ticker.tick() => {
                // State is already updated elsewhere; this tick just triggers a redraw.
            }

            // ── WebSocket order-book update ────────────────────────────────
            Some(update) = ws_rx.recv() => {
                let mut st = state.write().unwrap();
                st.order_books.insert(update.token_id.clone(), update.book);
                st.ws_connected = true;
            }

            // ── Strategy + position-management poll tick ───────────────────
            _ = poll_ticker.tick() => {
                if let Err(e) = run_cycle(
                    &state,
                    &client,
                    &db,
                    &session_id,
                    &strategy,
                    &trader,
                    &markets,
                ).await {
                    error!("Cycle error: {e}");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Graceful shutdown
    // -----------------------------------------------------------------------
    if let Some(ref mut term) = terminal {
        dashboard::teardown_terminal(term)?;
    }

    // Persist final session stats
    let st = state.read().unwrap();
    let final_session = BotSession {
        session_id: session_id.clone(),
        started_at: st.started_at.unwrap_or_else(Utc::now),
        ended_at: Some(Utc::now()),
        dry_run: st.dry_run,
        trades_opened: st.trades_opened,
        trades_closed: st.trades_closed,
        total_pnl_usdc: st.total_pnl,
        config_snapshot: session.config_snapshot.clone(),
    };
    drop(st);
    db.upsert_session(&final_session)?;

    info!(
        "Session {} ended – trades: {}/{} – total P&L: {:+.2} USDC",
        session_id,
        final_session.trades_opened,
        final_session.trades_closed,
        final_session.total_pnl_usdc,
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy + position management cycle (runs on each poll tick)
// ---------------------------------------------------------------------------

async fn run_cycle(
    state: &Arc<RwLock<AppState>>,
    client: &Arc<PolymarketClient>,
    db: &Arc<Database>,
    session_id: &str,
    strategy: &Strategy,
    trader: &Trader,
    markets: &[models::Market],
) -> Result<()> {
    // Snapshot what we need without holding the lock across awaits
    let (order_books, mut open_positions, balance_usdc) = {
        let st = state.read().unwrap();
        (
            st.order_books.clone(),
            st.open_positions.clone(),
            st.balance_usdc,
        )
    };

    // ── 1. Monitor existing positions (SL / TP / time-limit) ──────────────
    let closed = trader
        .monitor_positions(
            &mut open_positions,
            &order_books,
            markets,
            client,
            db,
            session_id,
        )
        .await?;

    // ── 2. Check market resolution ─────────────────────────────────────────
    // Re-fetch market states from DB to detect newly resolved markets
    let db_markets = db.get_all_markets().unwrap_or_default();
    let resolved = trader.check_resolution(&mut open_positions, &db_markets, db, session_id)?;

    // ── 3. Strategy: look for new opportunities ────────────────────────────
    let mut new_positions: Vec<models::Position> = Vec::new();
    let still_open: Vec<&models::Position> = open_positions
        .iter()
        .filter(|p| p.status.is_open())
        .collect();

    for market in markets {
        // Skip if we already hold a position in this market
        if still_open.iter().any(|p| p.market_id == market.condition_id) {
            continue;
        }

        // Try YES token book first, then NO
        for token_id in [&market.yes_token_id, &market.no_token_id] {
            if let Some(book) = order_books.get(token_id.as_str()) {
                if let Some(signal) = strategy.evaluate(
                    market,
                    book,
                    balance_usdc,
                    still_open.len() + new_positions.len(),
                ) {
                    match trader.open_position(&signal, client, db, session_id).await {
                        Ok(pos) => {
                            {
                                let mut st = state.write().unwrap();
                                st.add_log(format!(
                                    "Opened {} {} {:.4} edge={:.2}% bet=${:.2}",
                                    signal.market.label(),
                                    signal.side,
                                    signal.entry_price,
                                    signal.kelly.edge * 100.0,
                                    signal.bet_usdc,
                                ));
                                st.trades_opened += 1;
                            }
                            new_positions.push(pos);
                        }
                        Err(e) => {
                            let mut st = state.write().unwrap();
                            st.add_log(format!("ERROR opening position: {e}"));
                        }
                    }
                    break; // Only one signal per market per cycle
                }
            }
        }
    }

    // ── 4. Update shared state ─────────────────────────────────────────────
    {
        let mut st = state.write().unwrap();

        // Update positions list
        st.open_positions = open_positions
            .into_iter()
            .filter(|p| p.status.is_open())
            .chain(new_positions)
            .collect();

        // Move closed/resolved trades into recent_trades
        for pos in closed.iter().chain(resolved.iter()) {
            st.add_log(format!(
                "Closed {} {} P&L={:+.2} ({})",
                pos.asset,
                pos.side,
                pos.pnl_usdc.unwrap_or(0.0),
                pos.status,
            ));
            st.recent_trades.push_front(pos.clone());
            st.trades_closed += 1;
        }
        while st.recent_trades.len() > 50 {
            st.recent_trades.pop_back();
        }

        // Recalculate total P&L from DB
        st.total_pnl = db.total_pnl().unwrap_or(0.0);

        // Snapshot order_books from the client's shared cache
        if let Ok(books) = client.order_books.read() {
            st.order_books = books.clone();
        }
    }

    Ok(())
}
