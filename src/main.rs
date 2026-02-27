/// main.rs – Entry point for the PennyTrading Bot.
///
/// Orchestrates startup, the WebSocket feed, strategy loop, position
/// monitoring, and the live ratatui dashboard.
mod analytics;
mod config;
mod dashboard;
mod database;
mod events;
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
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

use analytics::{
    asset_pnl_correlation, default_stress_scenarios, evaluate_portfolio_stress, run_walk_forward,
};
use config::Settings;
use database::Database;
use events::ExternalEventMonitor;
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
        info!("DRY RUN mode – no real orders will be placed.");
    } else if !settings.has_credentials() {
        warn!("No API credentials found – forcing dry-run mode.");
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
    let mut markets = client
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
    let mut tracked_tokens: HashSet<String> = markets
        .iter()
        .flat_map(|m| [m.yes_token_id.clone(), m.no_token_id.clone()])
        .collect();

    // Fetch initial balance
    let balance = client.get_balance().await;
    {
        let mut st = state.write().unwrap();
        st.balance_usdc = if balance > 0.0 { balance } else { 1000.0 }; // default for dry-run
        st.session_start_equity = st.balance_usdc.max(0.0);
        st.session_peak_equity = st.session_start_equity;
        st.entries_paused = false;
        st.pause_reason = None;
    }

    // Start WebSocket feed
    let (mut ws_rx, mut ws_handle) = client.start_websocket(markets.clone());
    let mut last_ws_update = Instant::now();
    let ws_stale_after =
        Duration::from_secs_f64((settings.bot.poll_interval_seconds * 4.0).max(10.0));
    {
        let mut st = state.write().unwrap();
        st.ws_connected = false;
        st.add_log("WebSocket connection started".to_string());
    }

    // Build strategy and trader (wrapped for mutation during cycles)
    let strategy = Arc::new(RwLock::new(Strategy::new(
        settings.strategy.clone(),
        settings.kelly.clone(),
        settings.risk.clone(),
        settings.adaptive.clone(),
    )));
    let trader = Trader::new(
        settings.risk.clone(),
        settings.strategy.clone(),
        settings.execution.clone(),
        settings.bot.dry_run,
    );

    // Poll interval
    let poll_ms = (settings.bot.poll_interval_seconds * 1000.0) as u64;
    let mut poll_ticker = tokio::time::interval(std::time::Duration::from_millis(poll_ms));

    // Balance refresh counter (refresh every N cycles to avoid API spam)
    let mut cycle_count: u64 = 0;
    const BALANCE_REFRESH_CYCLES: u64 = 12; // every ~60s at 5s poll
    let calibration_refresh_cycles = settings.adaptive.calibration_interval_cycles;
    let mut event_monitor = ExternalEventMonitor::from_adaptive(&settings.adaptive);
    let event_poll_seconds = settings.adaptive.external_feed_poll_seconds.max(5.0);

    // Dashboard setup (unless --no-dashboard)
    let mut terminal = if !cli.no_dashboard {
        Some(dashboard::setup_terminal()?)
    } else {
        None
    };

    let refresh_ms = (settings.dashboard.refresh_rate * 1000.0) as u64;
    let mut dash_ticker = tokio::time::interval(std::time::Duration::from_millis(refresh_ms));
    let mut market_refresh_ticker = tokio::time::interval(Duration::from_secs(180));
    let mut event_feed_ticker = tokio::time::interval(Duration::from_secs_f64(event_poll_seconds));
    // Consume the immediate first tick so refresh happens after the interval.
    market_refresh_ticker.tick().await;
    event_feed_ticker.tick().await;

    let mut event_stream = EventStream::new();

    info!("Bot started.  Press 'q' to quit.");

    // -----------------------------------------------------------------------
    // Main event loop
    // -----------------------------------------------------------------------
    loop {
        // Render dashboard (read lock only, no clone of entire state)
        if let Some(ref mut term) = terminal {
            let st = state.read().unwrap();
            let st_ref = &*st;
            term.draw(|f| dashboard::render(f, st_ref))?;
            drop(st);
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
            update = ws_rx.recv() => {
                match update {
                    Some(update) => {
                        if tracked_tokens.contains(&update.token_id) {
                            let mut st = state.write().unwrap();
                            st.order_books.insert(update.token_id.clone(), update.book);
                            st.ws_connected = true;
                            last_ws_update = Instant::now();
                        }
                    }
                    None => {
                        {
                            let mut st = state.write().unwrap();
                            st.ws_connected = false;
                            st.add_log("WebSocket channel closed, restarting feed".to_string());
                        }
                        ws_handle.abort();
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        let (new_rx, new_handle) = client.start_websocket(markets.to_vec());
                        ws_rx = new_rx;
                        ws_handle = new_handle;
                        last_ws_update = Instant::now();
                    }
                }
            }

            // ── Periodic market metadata refresh ───────────────────────────
            _ = market_refresh_ticker.tick() => {
                let fresh = client.fetch_btc_eth_markets(
                    &settings.markets.assets,
                    &settings.markets.resolutions,
                    settings.markets.max_markets_per_asset,
                ).await.unwrap_or_default();

                if !fresh.is_empty() {
                    for m in &fresh {
                        if let Err(e) = db.upsert_market(m) {
                            warn!("Failed to refresh market {}: {e}", m.condition_id);
                        }
                    }

                    markets = fresh;
                    tracked_tokens = markets
                        .iter()
                        .flat_map(|m| [m.yes_token_id.clone(), m.no_token_id.clone()])
                        .collect();

                    {
                        let mut st = state.write().unwrap();
                        st.markets = markets.clone();
                        st.order_books.retain(|token_id, _| tracked_tokens.contains(token_id));
                        st.add_log(format!("Refreshed market universe: {} market(s)", markets.len()));
                        st.ws_connected = false;
                    }

                    ws_handle.abort();
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    let (new_rx, new_handle) = client.start_websocket(markets.clone());
                    ws_rx = new_rx;
                    ws_handle = new_handle;
                    last_ws_update = Instant::now();
                }
            }

            // ── External event/news feed refresh ───────────────────────────
            _ = event_feed_ticker.tick() => {
                if let Some(monitor) = event_monitor.as_mut() {
                    let snapshot = monitor.poll().await;
                    {
                        let mut strat = strategy.write().unwrap();
                        strat.set_external_event_state(snapshot.global_active, &snapshot.assets);
                    }
                    if snapshot.source_hits > 0 || snapshot.global_active || !snapshot.assets.is_empty() {
                        let mut assets: Vec<String> = snapshot.assets.iter().cloned().collect();
                        assets.sort();
                        let mut st = state.write().unwrap();
                        st.add_log(format!(
                            "External event state: global={} assets={:?} hits={} dup={} score={:.2}",
                            snapshot.global_active,
                            assets,
                            snapshot.source_hits,
                            snapshot.duplicate_hits,
                            snapshot.total_score,
                        ));
                    }
                }
            }

            // ── Strategy + position-management poll tick ───────────────────
            _ = poll_ticker.tick() => {
                cycle_count += 1;

                if last_ws_update.elapsed() > ws_stale_after {
                    let mut st = state.write().unwrap();
                    st.ws_connected = false;
                }

                // Periodically refresh balance from API
                if cycle_count % BALANCE_REFRESH_CYCLES == 0 {
                    let fresh_balance = client.get_balance().await;
                    if fresh_balance > 0.0 {
                        let mut st = state.write().unwrap();
                        st.balance_usdc = fresh_balance;
                    }
                }

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

                if settings.adaptive.enabled && cycle_count % calibration_refresh_cycles == 0 {
                    if let Ok(Some(stats)) = db.calibration_stats(settings.adaptive.sample_window) {
                        let mut st = state.write().unwrap();
                        st.add_log(format!(
                            "Calibration n={} brier(model={:.4}, market={:.4}) edge_hit={:.1}% avg_edge={:.2}%",
                            stats.sample_size,
                            stats.model_brier,
                            stats.market_brier,
                            stats.model_edge_hit_rate * 100.0,
                            stats.avg_edge * 100.0,
                        ));
                        drop(st);

                        if settings.adaptive.regime_enabled {
                            for regime in db.calibration_regimes(16).unwrap_or_default().into_iter() {
                                if let Ok(Some(regime_stats)) = db
                                    .calibration_stats_for_regime(settings.adaptive.sample_window, &regime)
                                {
                                    let mut st = state.write().unwrap();
                                    st.add_log(format!(
                                        "Calibration [{}] n={} brier(model={:.4}, market={:.4})",
                                        regime,
                                        regime_stats.sample_size,
                                        regime_stats.model_brier,
                                        regime_stats.market_brier,
                                    ));
                                    drop(st);

                                    let tune_msg = {
                                        let mut strat = strategy.write().unwrap();
                                        strat.adapt_from_calibration_for_regime(
                                            &regime_stats,
                                            &settings.adaptive,
                                            &regime,
                                        )
                                    };
                                    if let Some(msg) = tune_msg {
                                        let mut st = state.write().unwrap();
                                        st.add_log(msg);
                                    }
                                }
                            }
                        } else {
                            let tune_msg = {
                                let mut strat = strategy.write().unwrap();
                                strat.adapt_from_calibration(&stats, &settings.adaptive)
                            };
                            if let Some(msg) = tune_msg {
                                let mut st = state.write().unwrap();
                                st.add_log(msg);
                            }
                        }

                        let walkforward_limit = settings.adaptive.sample_window.saturating_mul(4).max(64);
                        if let Ok(samples) = db.resolved_trade_samples(walkforward_limit) {
                            if let Some(report) = run_walk_forward(
                                &samples,
                                4,
                                settings.adaptive.min_resolved_trades.max(16),
                            ) {
                                let mut st = state.write().unwrap();
                                st.add_log(format!(
                                    "Walk-forward n={} folds={} avg_test_pnl={:+.2} avg_win={:.1}% avg_trades={:.1}",
                                    report.sample_size,
                                    report.folds.len(),
                                    report.avg_test_pnl_usdc,
                                    report.avg_test_win_rate * 100.0,
                                    report.avg_test_trades,
                                ));
                            }
                            if let Some(corr) = asset_pnl_correlation(&samples, "BTC", "ETH") {
                                let mut st = state.write().unwrap();
                                st.add_log(format!(
                                    "Resolved BTC/ETH PnL correlation (minute buckets): {:.2}",
                                    corr
                                ));
                            }
                        }

                        let scenarios = default_stress_scenarios();
                        let stress = {
                            let st = state.read().unwrap();
                            evaluate_portfolio_stress(
                                st.balance_usdc,
                                &st.open_positions,
                                &st.order_books,
                                &scenarios,
                            )
                        };
                        if let Some(worst) = stress.first() {
                            let mut st = state.write().unwrap();
                            st.add_log(format!(
                                "Stress [{}]: projected_loss=${:.2} dd={:.1}%",
                                worst.name,
                                worst.projected_loss_usdc,
                                worst.projected_drawdown_pct * 100.0
                            ));
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Graceful shutdown
    // -----------------------------------------------------------------------
    ws_handle.abort();
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

const MAX_NEW_POSITIONS_PER_CYCLE: usize = 2;

async fn run_cycle(
    state: &Arc<RwLock<AppState>>,
    client: &Arc<PolymarketClient>,
    db: &Arc<Database>,
    session_id: &str,
    strategy: &Arc<RwLock<Strategy>>,
    trader: &Trader,
    markets: &[models::Market],
) -> Result<()> {
    let tracked_tokens: HashSet<&str> = markets
        .iter()
        .flat_map(|m| [m.yes_token_id.as_str(), m.no_token_id.as_str()])
        .collect();

    // Snapshot what we need without holding the lock across awaits
    let (
        order_books,
        mut open_positions,
        balance_usdc,
        prior_peak_equity,
        prior_entries_paused,
        prior_pause_reason,
    ) = {
        let st = state.read().unwrap();
        let books: HashMap<String, models::OrderBook> = st
            .order_books
            .iter()
            .filter(|(token_id, _)| tracked_tokens.contains(token_id.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        (
            books,
            st.open_positions.clone(),
            st.balance_usdc,
            st.session_peak_equity.max(st.session_start_equity),
            st.entries_paused,
            st.pause_reason.clone(),
        )
    };

    // ── 0. Update strategy spot prices from order book data ───────────────
    {
        let mut strat = strategy.write().unwrap();
        let active_tokens: HashSet<String> =
            tracked_tokens.iter().map(|s| (*s).to_string()).collect();
        strat.prune_price_history(&active_tokens);
        strat.update_spots_from_books(markets, &order_books);

        // Record mid-prices for adaptive volatility
        let now = Utc::now().timestamp() as f64;
        for market in markets {
            if let Some(book) = order_books.get(&market.yes_token_id) {
                if let Some(mid) = book.mid_price() {
                    strat.record_price(&market.yes_token_id, now, mid);
                }
            }
        }
    }

    // ── 1. Monitor existing positions (SL / TP / trailing-stop / time-limit)
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
    let db_markets = db.get_all_markets().unwrap_or_default();
    let resolved = trader.check_resolution(&mut open_positions, &db_markets, db, session_id)?;

    // ── 3. Strategy: look for new opportunities ────────────────────────────
    let mut new_positions: Vec<models::Position> = Vec::new();
    let still_open: Vec<&models::Position> = open_positions
        .iter()
        .filter(|p| p.status.is_open())
        .collect();
    let session_equity = estimate_session_equity(balance_usdc, &open_positions, &order_books);
    let peak_equity = prior_peak_equity.max(session_equity);
    let drawdown_usdc = (peak_equity - session_equity).max(0.0);
    let drawdown_pct = if peak_equity > 0.0 {
        drawdown_usdc / peak_equity
    } else {
        0.0
    };
    let drawdown_breach_usdc = trader.risk.max_session_drawdown_usdc > 0.0
        && drawdown_usdc >= trader.risk.max_session_drawdown_usdc;
    let drawdown_breach_pct = trader.risk.max_session_drawdown_pct > 0.0
        && drawdown_pct >= trader.risk.max_session_drawdown_pct;
    let drawdown_breached = drawdown_breach_usdc || drawdown_breach_pct;
    let entries_paused = prior_entries_paused || drawdown_breached;
    let pause_reason = if prior_entries_paused {
        prior_pause_reason
    } else if drawdown_breached {
        Some(format!(
            "Drawdown guard triggered: equity=${:.2}, peak=${:.2}, dd=${:.2} ({:.1}%)",
            session_equity,
            peak_equity,
            drawdown_usdc,
            drawdown_pct * 100.0
        ))
    } else {
        None
    };
    let pause_just_triggered = !prior_entries_paused && entries_paused;

    // Build per-asset position count for concentration limit
    let mut asset_position_counts: HashMap<String, usize> = HashMap::new();
    for p in &still_open {
        *asset_position_counts.entry(p.asset.clone()).or_insert(0) += 1;
    }

    // Deduct cost of open positions from available bankroll for Kelly sizing
    let open_cost: f64 = still_open.iter().map(|p| p.cost_usdc).sum();
    let available_bankroll = (balance_usdc - open_cost).max(0.0);

    let (mut candidate_signals, asset_sigmas) = {
        let strat = strategy.read().unwrap();
        let mut sigma_agg: HashMap<String, (f64, usize)> = HashMap::new();
        for market in markets {
            let default_sigma = match market.asset.as_str() {
                "BTC" => strat.btc_sigma,
                "ETH" => strat.eth_sigma,
                _ => continue,
            };
            let sigma = strat.estimate_volatility(&market.yes_token_id, default_sigma);
            let entry = sigma_agg.entry(market.asset.clone()).or_insert((0.0, 0));
            entry.0 += sigma;
            entry.1 += 1;
        }
        let asset_sigmas: HashMap<String, f64> = sigma_agg
            .into_iter()
            .map(|(asset, (sum, count))| {
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                (asset, avg)
            })
            .collect();

        let mut out = Vec::new();
        for market in markets {
            // Skip if we already hold a position in this market
            if still_open
                .iter()
                .any(|p| p.market_id == market.condition_id)
            {
                continue;
            }

            // Only evaluate from YES book; NO side is derived inside strategy.
            if let Some(book) = order_books.get(market.yes_token_id.as_str()) {
                if let Some(signal) = strat.evaluate(
                    market,
                    book,
                    available_bankroll,
                    still_open.len() + new_positions.len(),
                    &asset_position_counts,
                ) {
                    out.push(signal);
                }
            }
        }
        (out, asset_sigmas)
    };

    candidate_signals.sort_by(|a, b| {
        b.score()
            .partial_cmp(&a.score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let available_slots = trader
        .risk
        .max_open_positions
        .saturating_sub(still_open.len());
    let max_new_this_cycle = available_slots.min(MAX_NEW_POSITIONS_PER_CYCLE);

    if !entries_paused {
        for signal in candidate_signals {
            if new_positions.len() >= max_new_this_cycle {
                break;
            }
            if asset_position_counts
                .get(&signal.market.asset)
                .copied()
                .unwrap_or(0)
                >= 3
            {
                continue;
            }
            if trader.risk.max_asset_sigma > 0.0 {
                let sigma = asset_sigmas
                    .get(&signal.market.asset)
                    .copied()
                    .unwrap_or(0.0);
                if sigma > trader.risk.max_asset_sigma {
                    let mut st = state.write().unwrap();
                    st.add_log(format!(
                        "Skipped {} due to high sigma {:.2} > {:.2}",
                        signal.market.label(),
                        sigma,
                        trader.risk.max_asset_sigma
                    ));
                    continue;
                }
            }

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
                    *asset_position_counts
                        .entry(signal.market.asset.clone())
                        .or_insert(0) += 1;
                    new_positions.push(pos);
                }
                Err(e) => {
                    let mut st = state.write().unwrap();
                    st.add_log(format!("Skipped candidate (not filled/invalid): {e}"));
                }
            }
        }
    }

    // ── 4. Update shared state ─────────────────────────────────────────────
    {
        let mut st = state.write().unwrap();
        st.order_books
            .retain(|token_id, _| tracked_tokens.contains(token_id.as_str()));

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
        st.session_peak_equity = st.session_peak_equity.max(peak_equity);
        st.entries_paused = entries_paused;
        if pause_just_triggered {
            st.pause_reason = pause_reason.clone();
            if let Some(reason) = &pause_reason {
                st.add_log(format!("Entry pause activated: {reason}"));
            }
        }

        // Merge fresh order_books from the client's shared WS cache
        if let Ok(books) = client.order_books.read() {
            for (token_id, book) in books.iter() {
                if tracked_tokens.contains(token_id.as_str()) {
                    st.order_books.insert(token_id.clone(), book.clone());
                }
            }
        }
    }

    Ok(())
}

fn estimate_session_equity(
    balance_usdc: f64,
    positions: &[models::Position],
    order_books: &HashMap<String, models::OrderBook>,
) -> f64 {
    let mtm_value = positions
        .iter()
        .filter(|p| p.status.is_open())
        .map(|p| {
            let px = order_books
                .get(&p.token_id)
                .and_then(|b| b.best_bid())
                .unwrap_or(p.entry_price)
                .clamp(0.0, 1.0);
            px * p.size.max(0.0)
        })
        .sum::<f64>();
    (balance_usdc.max(0.0) + mtm_value).max(0.0)
}
