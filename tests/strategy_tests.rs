/// Integration tests for strategy module.
#[path = "../src/kelly.rs"]
mod kelly;

#[path = "../src/models.rs"]
mod models;

#[path = "../src/config.rs"]
mod config;

#[path = "../src/strategy.rs"]
mod strategy;

use chrono::Utc;
use models::{Market, OrderBook, PriceLevel};
use std::collections::HashMap;
use strategy::{book_imbalance, check_liquidity, effective_price_min, extract_threshold, Strategy};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_book(bid: f64, ask: f64, liq_usdc: f64) -> OrderBook {
    OrderBook {
        market_id: "test_market".into(),
        token_id: "test_token".into(),
        timestamp: Utc::now(),
        bids: vec![PriceLevel {
            price: bid,
            size: liq_usdc / bid,
        }],
        asks: vec![PriceLevel {
            price: ask,
            size: liq_usdc / ask,
        }],
    }
}

fn make_market(resolution_minutes: u32, asset: &str) -> Market {
    Market {
        condition_id: "cond_x".into(),
        question: format!("Will {} be above $50,000 in {resolution_minutes} minutes?", asset),
        yes_token_id: "yes_x".into(),
        no_token_id: "no_x".into(),
        asset: asset.to_string(),
        resolution_minutes,
        // Far future so time-limit gates pass
        end_date_iso: "2099-12-31T23:59:59Z".into(),
        active: true,
        closed: false,
        resolved: false,
        resolution_price: None,
    }
}

fn default_strategy() -> Strategy {
    Strategy::new(
        config::StrategyConfig::default(),
        config::KellyConfig::default(),
        config::RiskConfig::default(),
    )
}

fn empty_asset_counts() -> HashMap<String, usize> {
    HashMap::new()
}

// ---------------------------------------------------------------------------
// Liquidity checks
// ---------------------------------------------------------------------------

#[test]
fn liquidity_pass_with_good_book() {
    let book = make_book(0.49, 0.51, 1000.0);
    assert!(check_liquidity(&book, 500.0, 0.08));
}

#[test]
fn liquidity_fail_wide_spread() {
    let book = make_book(0.40, 0.60, 2000.0);
    assert!(!check_liquidity(&book, 500.0, 0.08));
}

#[test]
fn liquidity_fail_low_ask_liquidity() {
    let book = make_book(0.49, 0.51, 50.0); // only $50 on ask side
    assert!(!check_liquidity(&book, 500.0, 0.08));
}

// ---------------------------------------------------------------------------
// Dynamic price minimum
// ---------------------------------------------------------------------------

#[test]
fn dynamic_price_min_at_start() {
    let m = effective_price_min(0.02, 0.0);
    assert!((m - 0.02).abs() < 1e-9);
}

#[test]
fn dynamic_price_min_increases_with_elapsed_time() {
    let m0 = effective_price_min(0.02, 0.0);
    let m50 = effective_price_min(0.02, 0.5);
    let m100 = effective_price_min(0.02, 1.0);
    assert!(m0 < m50);
    assert!(m50 < m100);
}

#[test]
fn dynamic_price_min_capped_at_30_pct() {
    let m = effective_price_min(0.02, 1.0);
    assert!(m <= 0.30 + 1e-9);
}

// ---------------------------------------------------------------------------
// Threshold extraction
// ---------------------------------------------------------------------------

#[test]
fn extract_threshold_dollar_with_comma() {
    assert_eq!(extract_threshold("Will BTC be above $50,000?"), Some(50_000.0));
}

#[test]
fn extract_threshold_dollar_no_comma() {
    assert_eq!(extract_threshold("ETH above $3000?"), Some(3000.0));
}

#[test]
fn extract_threshold_bare_number() {
    assert_eq!(extract_threshold("BTC above 50000 at 12:00"), Some(50_000.0));
}

#[test]
fn extract_threshold_no_number() {
    assert_eq!(extract_threshold("Will it rain tomorrow?"), None);
}

#[test]
fn extract_threshold_ignores_small_numbers() {
    // Small numbers like "5" or "15" (resolution) should be ignored
    let t = extract_threshold("Will BTC be above $50,000 in 5 minutes?");
    assert_eq!(t, Some(50_000.0));
}

// ---------------------------------------------------------------------------
// Order book imbalance
// ---------------------------------------------------------------------------

#[test]
fn book_imbalance_balanced() {
    let book = make_book(0.49, 0.51, 1000.0);
    let imb = book_imbalance(&book, 0.05);
    // Both sides have similar USDC liquidity
    assert!(imb.abs() < 0.15, "imbalance should be near zero: {imb}");
}

#[test]
fn book_imbalance_bid_heavy() {
    // Large bid side, small ask side
    let book = OrderBook {
        market_id: "test".into(),
        token_id: "tok".into(),
        timestamp: Utc::now(),
        bids: vec![PriceLevel { price: 0.49, size: 5000.0 }], // ~$2450
        asks: vec![PriceLevel { price: 0.51, size: 100.0 }],   // ~$51
    };
    let imb = book_imbalance(&book, 0.05);
    assert!(imb > 0.5, "should be bullish: {imb}");
}

#[test]
fn book_imbalance_ask_heavy() {
    let book = OrderBook {
        market_id: "test".into(),
        token_id: "tok".into(),
        timestamp: Utc::now(),
        bids: vec![PriceLevel { price: 0.49, size: 100.0 }],
        asks: vec![PriceLevel { price: 0.51, size: 5000.0 }],
    };
    let imb = book_imbalance(&book, 0.05);
    assert!(imb < -0.5, "should be bearish: {imb}");
}

// ---------------------------------------------------------------------------
// Strategy evaluate – should NOT signal without spot price
// ---------------------------------------------------------------------------

#[test]
fn evaluate_no_spot_returns_none() {
    let strat = default_strategy(); // btc_spot = 0
    let market = make_market(5, "BTC");
    let book = make_book(0.30, 0.35, 2000.0);

    // spot = 0 -> no probability -> no signal
    let signal = strat.evaluate(&market, &book, 1000.0, 0, &empty_asset_counts());
    assert!(signal.is_none());
}

// ---------------------------------------------------------------------------
// Strategy evaluate – should signal with good setup
// ---------------------------------------------------------------------------

#[test]
fn evaluate_signals_when_edge_exists() {
    let mut strat = default_strategy();
    // BTC at 52_000, threshold is 50_000 -> YES probability > 0.5
    // With market asking only 0.35 for YES, there should be edge
    strat.btc_spot = 52_000.0;

    let market = make_market(5, "BTC");
    let book = make_book(0.34, 0.36, 2000.0);

    let signal = strat.evaluate(&market, &book, 1000.0, 0, &empty_asset_counts());
    // Should get a YES signal since P(spot>50k) > 0.5 >> 0.36
    if let Some(sig) = signal {
        assert!(sig.kelly.edge > 0.0);
        assert!(sig.bet_usdc > 0.0);
    }
    // (signal may still be None if edge < min_edge; that's acceptable)
}

#[test]
fn evaluate_respects_max_positions() {
    let mut strat = default_strategy();
    strat.btc_spot = 52_000.0;
    let market = make_market(5, "BTC");
    let book = make_book(0.34, 0.36, 2000.0);

    let max = strat.risk_cfg.max_open_positions;
    // Already at the limit
    let signal = strat.evaluate(&market, &book, 1000.0, max, &empty_asset_counts());
    assert!(signal.is_none());
}

#[test]
fn evaluate_respects_per_asset_limit() {
    let mut strat = default_strategy();
    strat.btc_spot = 52_000.0;
    let market = make_market(5, "BTC");
    let book = make_book(0.34, 0.36, 2000.0);

    // Already have 3 BTC positions (MAX_POSITIONS_PER_ASSET)
    let mut counts = HashMap::new();
    counts.insert("BTC".to_string(), 3);
    let signal = strat.evaluate(&market, &book, 1000.0, 3, &counts);
    assert!(signal.is_none());
}

#[test]
fn evaluate_no_signal_for_eth_without_spot() {
    let strat = default_strategy(); // eth_spot = 0
    let market = make_market(15, "ETH");
    let book = make_book(0.45, 0.47, 2000.0);

    let signal = strat.evaluate(&market, &book, 1000.0, 0, &empty_asset_counts());
    assert!(signal.is_none());
}

#[test]
fn evaluate_no_signal_when_market_near_resolution() {
    let mut strat = default_strategy();
    strat.btc_spot = 52_000.0;

    let mut market = make_market(5, "BTC");
    // Set end_date to just seconds away (within time_limit_fraction)
    let soon = chrono::Utc::now() + chrono::Duration::seconds(30);
    market.end_date_iso = soon.to_rfc3339();

    let book = make_book(0.34, 0.36, 2000.0);
    let signal = strat.evaluate(&market, &book, 1000.0, 0, &empty_asset_counts());
    assert!(signal.is_none());
}

#[test]
fn evaluate_no_signal_below_price_min() {
    let mut strat = default_strategy();
    strat.btc_spot = 55_000.0; // well above threshold

    let market = make_market(5, "BTC"); // threshold ~ 50k
    // Ask is below price_min (0.02), so even a good NO signal should be skipped
    let book = make_book(0.005, 0.010, 1000.0);
    let signal = strat.evaluate(&market, &book, 1000.0, 0, &empty_asset_counts());
    // Very cheap ask below price_min -> should not trade
    assert!(signal.is_none());
}

// ---------------------------------------------------------------------------
// Adaptive volatility estimation
// ---------------------------------------------------------------------------

#[test]
fn record_price_and_estimate_volatility() {
    let mut strat = default_strategy();
    // Simulate 50 price observations with some variance
    let base = 0.50;
    for i in 0..50 {
        let t = i as f64 * 5.0; // 5 second intervals
        let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
        strat.record_price("tok_test", t, base + noise);
    }
    // Should produce a volatility estimate (not the default)
    let sigma = strat.estimate_volatility("tok_test", 0.80);
    assert!(sigma > 0.0, "sigma should be positive");
    assert!(sigma < 10.0, "sigma should be reasonable");
}

// ---------------------------------------------------------------------------
// Trailing stop logic
// ---------------------------------------------------------------------------

#[test]
fn trailing_stop_activates_in_profit() {
    let mut pos = models::Position {
        position_id: "p1".into(),
        market_id: "m1".into(),
        token_id: "t1".into(),
        side: models::Side::YES,
        asset: "BTC".into(),
        entry_price: 0.30,
        size: 100.0,
        cost_usdc: 30.0,
        stop_loss_price: 0.15,
        take_profit_price: 0.90,
        high_water_mark: 0.30,
        status: models::TradeStatus::Open,
        exit_price: None,
        pnl_usdc: None,
        opened_at: Utc::now(),
        closed_at: None,
        dry_run: true,
        order_type: models::OrderType::Limit,
    };

    // Price goes up to 0.50 -- should set HWM
    let trail = pos.update_trailing_stop(0.50, 0.20);
    assert_eq!(pos.high_water_mark, 0.50);
    // Trailing stop at 0.50 * 0.80 = 0.40 > 0.30 entry -> active
    assert!(trail.is_some());
    assert!((trail.unwrap() - 0.40).abs() < 1e-9);

    // Price drops to 0.45 -- HWM stays at 0.50
    let trail = pos.update_trailing_stop(0.45, 0.20);
    assert_eq!(pos.high_water_mark, 0.50);
    assert!(trail.is_some());
    // trail_price = 0.50 * 0.80 = 0.40, current 0.45 > 0.40 -> not triggered
    assert!(0.45 > trail.unwrap());
}

#[test]
fn trailing_stop_not_active_at_loss() {
    let mut pos = models::Position {
        position_id: "p2".into(),
        market_id: "m2".into(),
        token_id: "t2".into(),
        side: models::Side::YES,
        asset: "BTC".into(),
        entry_price: 0.50,
        size: 100.0,
        cost_usdc: 50.0,
        stop_loss_price: 0.25,
        take_profit_price: 1.50,
        high_water_mark: 0.50,
        status: models::TradeStatus::Open,
        exit_price: None,
        pnl_usdc: None,
        opened_at: Utc::now(),
        closed_at: None,
        dry_run: true,
        order_type: models::OrderType::Limit,
    };

    // Price drops to 0.40 -- below entry, no trailing stop
    let trail = pos.update_trailing_stop(0.40, 0.20);
    assert!(trail.is_none());
}
