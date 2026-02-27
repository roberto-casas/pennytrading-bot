/// Integration tests for the database layer.
use chrono::Utc;
use tempfile::NamedTempFile;

// We reference the crate's modules via path re-includes used by integration tests.
// Build the crate and run via `cargo test --test database_tests`.

// ---------------------------------------------------------------------------
// Bring in the modules under test via the crate root
// ---------------------------------------------------------------------------
// Integration tests link against the compiled crate, so we use `pennytrading_bot::*`.
// Because the binary crate doesn't expose a lib target we use `#[path]` includes.

#[path = "../src/models.rs"]
mod models;

#[path = "../src/database.rs"]
mod database;

// kelly is required by models transitively
#[path = "../src/kelly.rs"]
mod kelly;

use database::Database;
use models::{
    BotSession, Market, OrderBook, OrderType, Position, PriceLevel, Side, TradeDiagnostics,
    TradeStatus,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tmp_db() -> (NamedTempFile, Database) {
    let f = NamedTempFile::new().expect("tempfile");
    let db = Database::open(f.path().to_str().unwrap()).expect("open db");
    (f, db)
}

fn sample_market(n: u32) -> Market {
    Market {
        condition_id: format!("cond_{n}"),
        question: format!("Will BTC be above $50,000 in 5 minutes? #{n}"),
        yes_token_id: format!("yes_{n}"),
        no_token_id: format!("no_{n}"),
        asset: "BTC".into(),
        resolution_minutes: 5,
        end_date_iso: "2099-01-01T00:00:00Z".into(),
        active: true,
        closed: false,
        resolved: false,
        resolution_price: None,
    }
}

fn sample_position(market_id: &str) -> Position {
    Position {
        position_id: format!("pos_{market_id}"),
        market_id: market_id.to_string(),
        token_id: format!("yes_{market_id}"),
        side: Side::YES,
        asset: "BTC".into(),
        entry_price: 0.35,
        size: 100.0,
        cost_usdc: 35.0,
        stop_loss_price: 0.175,
        take_profit_price: 0.70,
        high_water_mark: 0.35,
        status: TradeStatus::Open,
        exit_price: None,
        pnl_usdc: None,
        opened_at: Utc::now(),
        closed_at: None,
        dry_run: true,
        order_type: OrderType::Limit,
    }
}

fn sample_session(id: &str) -> BotSession {
    BotSession {
        session_id: id.to_string(),
        started_at: Utc::now(),
        ended_at: None,
        dry_run: true,
        trades_opened: 0,
        trades_closed: 0,
        total_pnl_usdc: 0.0,
        config_snapshot: "{}".into(),
    }
}

fn sample_diag(
    position_id: &str,
    edge: f64,
    model_prob: f64,
    market_prob: f64,
    regime: &str,
) -> TradeDiagnostics {
    TradeDiagnostics {
        position_id: position_id.to_string(),
        session_id: "sess_diag".into(),
        market_id: "cond_diag".into(),
        asset: "BTC".into(),
        side: Side::YES,
        model_prob,
        market_prob,
        edge,
        regime: regime.to_string(),
        entry_price: market_prob,
        bet_usdc: 50.0,
        created_at: Utc::now(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_schema_applied_on_open() {
    let (_f, db) = tmp_db();
    // If schema is not applied, get_active_markets() would panic.
    let markets = db.get_active_markets().expect("get_active_markets");
    assert!(markets.is_empty());
}

#[test]
fn test_upsert_and_get_market() {
    let (_f, db) = tmp_db();
    let m = sample_market(1);
    db.upsert_market(&m).expect("upsert");

    let markets = db.get_active_markets().expect("get");
    assert_eq!(markets.len(), 1);
    assert_eq!(markets[0].condition_id, "cond_1");
    assert_eq!(markets[0].asset, "BTC");
    assert_eq!(markets[0].resolution_minutes, 5);
}

#[test]
fn test_upsert_market_is_idempotent() {
    let (_f, db) = tmp_db();
    let m = sample_market(2);
    db.upsert_market(&m).expect("first upsert");
    db.upsert_market(&m).expect("second upsert (idempotent)");
    assert_eq!(db.get_active_markets().unwrap().len(), 1);
}

#[test]
fn test_market_resolution_update() {
    let (_f, db) = tmp_db();
    let mut m = sample_market(3);
    db.upsert_market(&m).expect("upsert");

    // Resolve the market
    m.resolved = true;
    m.active = false;
    m.resolution_price = Some(1.0);
    db.upsert_market(&m).expect("update");

    // Should not appear in active markets
    let active = db.get_active_markets().expect("get active");
    assert!(active.is_empty());

    // Should appear in all markets
    let all = db.get_all_markets().expect("get all");
    assert_eq!(all.len(), 1);
    assert!(all[0].resolved);
    assert_eq!(all[0].resolution_price, Some(1.0));
}

#[test]
fn test_upsert_and_get_position() {
    let (_f, db) = tmp_db();

    // Need a market and session first (FK constraints)
    let m = sample_market(10);
    db.upsert_market(&m).expect("upsert market");
    let s = sample_session("sess_1");
    db.upsert_session(&s).expect("upsert session");

    let pos = sample_position("cond_10");
    db.upsert_position(&pos, Some("sess_1"))
        .expect("upsert position");

    let positions = db.get_open_positions().expect("get open");
    assert_eq!(positions.len(), 1);
    assert_eq!(positions[0].market_id, "cond_10");
    assert_eq!(positions[0].side, Side::YES);
    assert!((positions[0].entry_price - 0.35).abs() < 1e-9);
}

#[test]
fn test_close_position_removed_from_open() {
    let (_f, db) = tmp_db();

    let m = sample_market(20);
    db.upsert_market(&m).expect("upsert market");
    let s = sample_session("sess_2");
    db.upsert_session(&s).expect("upsert session");

    let mut pos = sample_position("cond_20");
    db.upsert_position(&pos, Some("sess_2")).expect("insert");

    // Close it
    pos.status = TradeStatus::ClosedTakeProfit;
    pos.exit_price = Some(0.70);
    pos.pnl_usdc = Some(35.0);
    pos.closed_at = Some(Utc::now());
    db.upsert_position(&pos, Some("sess_2")).expect("update");

    // Should not be in open positions anymore
    let open = db.get_open_positions().expect("get open");
    assert!(open.is_empty());

    // Should be in all positions
    let all = db.get_all_positions(100).expect("get all");
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].status, TradeStatus::ClosedTakeProfit);
}

#[test]
fn test_total_pnl() {
    let (_f, db) = tmp_db();

    for i in 0..3u32 {
        let m = sample_market(30 + i);
        db.upsert_market(&m).unwrap();
        let s_id = format!("sess_{i}");
        let s = sample_session(&s_id);
        db.upsert_session(&s).unwrap();

        let mut pos = sample_position(&format!("cond_{}", 30 + i));
        pos.position_id = format!("pos_{i}");
        pos.status = TradeStatus::ClosedTakeProfit;
        pos.pnl_usdc = Some(10.0 * (i as f64 + 1.0));
        pos.exit_price = Some(0.70);
        pos.closed_at = Some(Utc::now());
        db.upsert_position(&pos, Some(&s_id)).unwrap();
    }

    let total = db.total_pnl().expect("total_pnl");
    // 10 + 20 + 30 = 60
    assert!((total - 60.0).abs() < 1e-9);
}

#[test]
fn test_upsert_session() {
    let (_f, db) = tmp_db();
    let mut s = sample_session("sess_abc");
    db.upsert_session(&s).expect("insert");

    // Update
    s.trades_opened = 5;
    s.total_pnl_usdc = 123.45;
    s.ended_at = Some(Utc::now());
    db.upsert_session(&s).expect("update");

    let sessions = db.get_sessions(10).expect("get sessions");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].trades_opened, 5);
    assert!((sessions[0].total_pnl_usdc - 123.45).abs() < 1e-9);
    assert!(sessions[0].ended_at.is_some());
}

#[test]
fn test_orderbook_snapshot_saved() {
    let (_f, db) = tmp_db();
    let m = sample_market(40);
    db.upsert_market(&m).expect("upsert market");

    let book = OrderBook {
        market_id: "cond_40".into(),
        token_id: "yes_40".into(),
        timestamp: Utc::now(),
        bids: vec![PriceLevel {
            price: 0.49,
            size: 100.0,
        }],
        asks: vec![PriceLevel {
            price: 0.51,
            size: 100.0,
        }],
    };

    db.save_orderbook_snapshot(&book).expect("save snapshot");
    // If no panic, the snapshot was saved successfully.
}

#[test]
fn test_dry_run_position_in_open() {
    let (_f, db) = tmp_db();
    let m = sample_market(50);
    db.upsert_market(&m).expect("upsert market");
    let s = sample_session("sess_dry");
    db.upsert_session(&s).expect("upsert session");

    let mut pos = sample_position("cond_50");
    pos.status = TradeStatus::DryRun;
    db.upsert_position(&pos, Some("sess_dry")).expect("upsert");

    // DRY_RUN status should count as open
    let open = db.get_open_positions().expect("get open");
    assert_eq!(open.len(), 1);
}

#[test]
fn test_calibration_stats_uses_resolved_positions() {
    let (_f, db) = tmp_db();
    let market = sample_market(60);
    db.upsert_market(&market).expect("upsert market");
    let session = sample_session("sess_diag");
    db.upsert_session(&session).expect("upsert session");

    let mut win = sample_position("cond_60");
    win.position_id = "pos_win".into();
    win.status = TradeStatus::ResolvedWin;
    win.closed_at = Some(Utc::now());
    win.pnl_usdc = Some(10.0);
    db.upsert_position(&win, Some("sess_diag"))
        .expect("upsert win");
    db.upsert_trade_diagnostics(&sample_diag(
        "pos_win",
        0.15,
        0.70,
        0.55,
        "low_vol|normal_spread|normal_liq|normal_event",
    ))
    .expect("diag win");

    let mut loss = sample_position("cond_60");
    loss.position_id = "pos_loss".into();
    loss.status = TradeStatus::ResolvedLoss;
    loss.closed_at = Some(Utc::now());
    loss.pnl_usdc = Some(-10.0);
    db.upsert_position(&loss, Some("sess_diag"))
        .expect("upsert loss");
    db.upsert_trade_diagnostics(&sample_diag(
        "pos_loss",
        -0.15,
        0.30,
        0.45,
        "high_vol|wide_spread|thin_liq|event",
    ))
    .expect("diag loss");

    let stats = db
        .calibration_stats(100)
        .expect("calibration stats")
        .expect("non-empty");
    assert_eq!(stats.sample_size, 2);
    assert!((stats.model_brier - 0.09).abs() < 1e-9);
    assert!((stats.market_brier - 0.2025).abs() < 1e-9);
    assert!((stats.avg_edge - 0.0).abs() < 1e-9);
    assert!((stats.model_edge_hit_rate - 0.5).abs() < 1e-9);

    let low = db
        .calibration_stats_for_regime(100, "low_vol|normal_spread|normal_liq|normal_event")
        .expect("low regime stats")
        .expect("low non-empty");
    assert_eq!(low.sample_size, 1);
    assert!((low.model_brier - 0.09).abs() < 1e-9);

    let regimes = db.calibration_regimes(10).expect("regime list");
    assert!(regimes
        .iter()
        .any(|r| r == "low_vol|normal_spread|normal_liq|normal_event"));
    assert!(regimes
        .iter()
        .any(|r| r == "high_vol|wide_spread|thin_liq|event"));
}

#[test]
fn test_resolved_trade_samples_returns_recent_chronological() {
    let (_f, db) = tmp_db();
    let market = sample_market(61);
    db.upsert_market(&market).expect("upsert market");
    let session = sample_session("sess_samples");
    db.upsert_session(&session).expect("upsert session");

    for i in 0..3 {
        let mut pos = sample_position("cond_61");
        pos.position_id = format!("pos_sample_{i}");
        pos.status = if i % 2 == 0 {
            TradeStatus::ResolvedWin
        } else {
            TradeStatus::ResolvedLoss
        };
        pos.closed_at = Some(Utc::now() + chrono::Duration::seconds(i as i64));
        pos.pnl_usdc = Some(if i % 2 == 0 { 5.0 } else { -4.0 });
        db.upsert_position(&pos, Some("sess_samples"))
            .expect("upsert position");

        db.upsert_trade_diagnostics(&sample_diag(
            &pos.position_id,
            0.05 + i as f64 * 0.01,
            0.60,
            0.52,
            "low_vol|normal_spread|normal_liq|normal_event",
        ))
        .expect("upsert diag");
    }

    let samples = db.resolved_trade_samples(2).expect("resolved samples");
    assert_eq!(samples.len(), 2);
    assert!(samples[0].closed_at <= samples[1].closed_at);
}
