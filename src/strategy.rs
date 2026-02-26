/// strategy.rs – Signal generation for the penny strategy.
///
/// For each active market the strategy:
///   1. Checks the order book (liquidity, spread).
///   2. Estimates fair value using a log-normal model with adaptive volatility.
///   3. Incorporates order-book imbalance as a directional signal.
///   4. Computes Kelly bet size (spread-adjusted) based on the edge.
///   5. Enforces price, time, and per-asset concentration limits.
///   6. Returns a `Signal` if all conditions are met.
use crate::{
    config::{KellyConfig, RiskConfig, StrategyConfig},
    kelly::{compute_bet, model_probability, BetResult},
    models::{Market, OrderBook, Side},
};
use std::collections::HashMap;
use tracing::debug;

// ---------------------------------------------------------------------------
// Signal
// ---------------------------------------------------------------------------

/// A trading signal produced by the strategy for a specific market/side.
#[derive(Debug, Clone)]
pub struct Signal {
    pub market: Market,
    pub side: Side,
    pub token_id: String,
    /// The price at which we want to enter (limit order price or best ask).
    pub entry_price: f64,
    /// Recommended bet size in USDC.
    pub bet_usdc: f64,
    pub kelly: BetResult,
    /// Model probability for this side.
    pub p_model: f64,
    /// Market-implied probability (best bid or ask depending on side).
    pub p_market: f64,
}

// ---------------------------------------------------------------------------
// Liquidity check
// ---------------------------------------------------------------------------

/// Check whether the order book has enough liquidity and a tight enough spread.
pub fn check_liquidity(
    book: &OrderBook,
    min_liquidity_usdc: f64,
    max_spread_pct: f64,
) -> bool {
    let Some(spread_pct) = book.spread_pct() else {
        return false;
    };
    if spread_pct > max_spread_pct {
        debug!(
            "Spread too wide: {:.2}% > {:.2}%",
            spread_pct * 100.0,
            max_spread_pct * 100.0
        );
        return false;
    }

    // Need sufficient ask-side liquidity to buy into
    let ask_liq = book.ask_liquidity(0.05);
    if ask_liq < min_liquidity_usdc {
        debug!(
            "Insufficient ask liquidity: ${ask_liq:.0} < ${min_liquidity_usdc:.0}"
        );
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Dynamic price limit
// ---------------------------------------------------------------------------

/// Adjust `price_min` upward near market resolution so we avoid buying
/// extremely cheap contracts that are unlikely to recover.
///
/// Formula: effective_min = price_min + (time_elapsed_fraction * 0.10)
/// so that with 80 % time elapsed the minimum is price_min + 0.08.
pub fn effective_price_min(base_min: f64, time_elapsed_fraction: f64) -> f64 {
    (base_min + time_elapsed_fraction * 0.10).min(0.30)
}

// ---------------------------------------------------------------------------
// Order book imbalance
// ---------------------------------------------------------------------------

/// Compute order-book imbalance ratio in [-1, 1].
/// Positive = more bid pressure (bullish for YES), negative = more ask pressure.
pub fn book_imbalance(book: &OrderBook, depth: f64) -> f64 {
    let bid_liq = book.bid_liquidity(depth);
    let ask_liq = book.ask_liquidity(depth);
    let total = bid_liq + ask_liq;
    if total < 1.0 {
        return 0.0;
    }
    (bid_liq - ask_liq) / total
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Maximum per-asset open positions to avoid correlated exposure.
const MAX_POSITIONS_PER_ASSET: usize = 3;

/// Weight of order-book imbalance signal (additive to model probability).
const IMBALANCE_WEIGHT: f64 = 0.02;

pub struct Strategy {
    pub cfg: StrategyConfig,
    pub kelly_cfg: KellyConfig,
    pub risk_cfg: RiskConfig,
    /// Current BTC spot price (updated from order book mid-prices or external feed).
    pub btc_spot: f64,
    /// Current ETH spot price.
    pub eth_spot: f64,
    /// Annualised volatility for BTC (adaptive: updated from recent price data).
    pub btc_sigma: f64,
    /// Annualised volatility for ETH (adaptive).
    pub eth_sigma: f64,
    /// Rolling mid-price history per token for realized volatility estimation.
    /// token_id -> Vec<(timestamp_secs, mid_price)>
    price_history: HashMap<String, Vec<(f64, f64)>>,
}

impl Strategy {
    pub fn new(cfg: StrategyConfig, kelly_cfg: KellyConfig, risk_cfg: RiskConfig) -> Self {
        Self {
            cfg,
            kelly_cfg,
            risk_cfg,
            btc_spot: 0.0,
            eth_spot: 0.0,
            btc_sigma: 0.80,
            eth_sigma: 0.90,
            price_history: HashMap::new(),
        }
    }

    /// Update spot prices (called whenever fresh price data arrives).
    pub fn update_spots(&mut self, btc: f64, eth: f64) {
        if btc > 0.0 {
            self.btc_spot = btc;
        }
        if eth > 0.0 {
            self.eth_spot = eth;
        }
    }

    /// Infer spot prices from order book mid-prices.
    /// Polymarket binary markets: if we know the threshold from the question
    /// and the mid-price, we can back out an implied spot.  As a simpler
    /// heuristic when we have no external feed, we use the order-book
    /// mid-price directly as the market-implied probability.
    pub fn update_spots_from_books(&mut self, markets: &[Market], books: &HashMap<String, OrderBook>) {
        // For each asset, gather the highest-liquidity market's implied spot.
        // The mid-price of the YES token IS the market's implied P(YES).
        // We don't need the actual BTC/ETH dollar price — the model compares
        // our model probability to the market's implied probability.
        // However, to run the GBM model we DO need the dollar spot.
        // We estimate it from the most liquid market's threshold + mid-price.
        for asset in &["BTC", "ETH"] {
            let mut best_liq = 0.0_f64;
            let mut best_spot = 0.0_f64;
            for m in markets.iter().filter(|m| m.asset == *asset) {
                if let Some(book) = books.get(&m.yes_token_id) {
                    let liq = book.ask_liquidity(0.05) + book.bid_liquidity(0.05);
                    if liq > best_liq {
                        if let Some(threshold) = extract_threshold(&m.question) {
                            if let Some(mid) = book.mid_price() {
                                // mid ≈ P(spot > threshold)
                                // For a rough inverse: spot ≈ threshold * (1 + (mid - 0.5) * factor)
                                // More accurate: use the GBM inverse, but for bootstrapping
                                // a simple linear interpolation works.
                                let offset_factor = 0.05; // 5% of threshold per 0.1 probability unit
                                let implied_spot = threshold * (1.0 + (mid - 0.5) * offset_factor * 2.0);
                                if implied_spot > 0.0 {
                                    best_spot = implied_spot;
                                    best_liq = liq;
                                }
                            }
                        }
                    }
                }
            }
            if best_spot > 0.0 {
                match *asset {
                    "BTC" => self.btc_spot = best_spot,
                    "ETH" => self.eth_spot = best_spot,
                    _ => {}
                }
            }
        }
    }

    /// Record a mid-price observation for adaptive volatility estimation.
    pub fn record_price(&mut self, token_id: &str, timestamp_secs: f64, mid_price: f64) {
        let history = self.price_history.entry(token_id.to_string()).or_default();
        history.push((timestamp_secs, mid_price));
        // Keep last 200 observations (enough for ~15 minutes at 5s poll)
        if history.len() > 200 {
            history.drain(..history.len() - 200);
        }
    }

    /// Estimate realized volatility from recent price history, annualized.
    pub fn estimate_volatility(&self, token_id: &str, default_sigma: f64) -> f64 {
        let Some(history) = self.price_history.get(token_id) else {
            return default_sigma;
        };
        if history.len() < 10 {
            return default_sigma;
        }

        // Compute log-returns and their standard deviation
        let mut log_returns = Vec::with_capacity(history.len() - 1);
        let mut avg_dt = 0.0;
        for i in 1..history.len() {
            let (t0, p0) = history[i - 1];
            let (t1, p1) = history[i];
            if p0 > 0.0 && p1 > 0.0 && t1 > t0 {
                log_returns.push((p1 / p0).ln());
                avg_dt += t1 - t0;
            }
        }
        if log_returns.len() < 5 {
            return default_sigma;
        }
        avg_dt /= log_returns.len() as f64;

        let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
        let variance = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (log_returns.len() - 1) as f64;
        let std_per_interval = variance.sqrt();

        // Annualize: sigma_annual = sigma_interval * sqrt(intervals_per_year)
        if avg_dt > 0.0 {
            let intervals_per_year = (365.25 * 24.0 * 3600.0) / avg_dt;
            let sigma = std_per_interval * intervals_per_year.sqrt();
            // Clamp to reasonable range
            sigma.clamp(0.20, 3.0)
        } else {
            default_sigma
        }
    }

    /// Count open positions per asset.
    fn count_per_asset(open_positions: &[(String, usize)], asset: &str) -> usize {
        open_positions.iter().filter(|(a, _)| a == asset).map(|(_, c)| c).sum()
    }

    /// Evaluate a market and return a `Signal` if there is a tradeable
    /// opportunity on either the YES or NO side.
    ///
    /// `yes_book` must be the YES token's order book for this market.
    /// Both YES and NO prices are derived from it (NO price = 1 - YES bid).
    ///
    /// `asset_position_counts` maps asset name to count of open positions.
    ///
    /// Returns `None` when no edge is found or conditions are not met.
    pub fn evaluate(
        &self,
        market: &Market,
        book: &OrderBook,
        bankroll_usdc: f64,
        open_positions: usize,
        asset_position_counts: &HashMap<String, usize>,
    ) -> Option<Signal> {
        // --- Gate 1: global position limit ----------------------------------
        if open_positions >= self.risk_cfg.max_open_positions {
            debug!("Max open positions reached");
            return None;
        }

        // --- Gate 2: per-asset concentration limit --------------------------
        let asset_count = asset_position_counts.get(&market.asset).copied().unwrap_or(0);
        if asset_count >= MAX_POSITIONS_PER_ASSET {
            debug!("Max positions for {} reached ({})", market.asset, asset_count);
            return None;
        }

        // --- Gate 3: time limit ---------------------------------------------
        if market.is_near_resolution(self.risk_cfg.time_limit_fraction) {
            debug!("Market {} too close to resolution", market.label());
            return None;
        }

        // --- Gate 4: liquidity and spread -----------------------------------
        if !check_liquidity(
            book,
            self.cfg.min_liquidity_usdc,
            self.cfg.max_spread_pct,
        ) {
            return None;
        }

        // --- Gate 5: select side and price ----------------------------------
        let best_ask = book.best_ask()?;
        let best_bid = book.best_bid()?;
        let spread = best_ask - best_bid;

        // Derive spot price and sigma for this asset
        let (spot, default_sigma) = match market.asset.as_str() {
            "BTC" => (self.btc_spot, self.btc_sigma),
            "ETH" => (self.eth_spot, self.eth_sigma),
            _ => return None,
        };
        if spot <= 0.0 {
            debug!("No spot price available for {}", market.asset);
            return None;
        }

        // Use adaptive volatility if we have enough price history
        let sigma = self.estimate_volatility(&market.yes_token_id, default_sigma);

        // Extract threshold from market question
        let threshold = extract_threshold(&market.question)?;
        let time_secs = market.seconds_to_resolution();

        // Calculate elapsed fraction for dynamic price adjustment
        let total_secs = market.resolution_minutes as f64 * 60.0;
        let elapsed_fraction = if total_secs > 0.0 {
            ((total_secs - time_secs) / total_secs).clamp(0.0, 1.0)
        } else {
            1.0
        };

        // Model probability that spot > threshold at resolution
        let p_yes_model_raw = model_probability(spot, threshold, sigma, time_secs);

        // Order book imbalance adjustment: strong bid pressure is bullish for YES
        let imbalance = book_imbalance(book, 0.05);
        let p_yes_model = (p_yes_model_raw + imbalance * IMBALANCE_WEIGHT).clamp(0.001, 0.999);
        let p_no_model = 1.0 - p_yes_model;

        // Spread-adjusted edge: the real cost of a round-trip is entry at ask,
        // exit at bid.  Half the spread is lost on each side.
        let half_spread = spread / 2.0;

        let dyn_min = effective_price_min(self.cfg.price_min, elapsed_fraction);

        // Try YES side (buy YES at best ask)
        let yes_entry = best_ask;
        // Effective market price accounts for spread cost on exit
        let yes_effective_market = yes_entry + half_spread;
        if yes_entry >= dyn_min
            && yes_entry <= self.cfg.price_max
            && p_yes_model > yes_effective_market + self.cfg.min_edge
        {
            let kelly = compute_bet(
                p_yes_model,
                yes_effective_market, // use spread-adjusted price for Kelly
                bankroll_usdc,
                self.kelly_cfg.fraction,
                self.kelly_cfg.max_position_pct,
                self.kelly_cfg.min_bet_usdc,
                self.kelly_cfg.max_bet_usdc,
            );
            if kelly.is_actionable {
                return Some(Signal {
                    market: market.clone(),
                    side: Side::YES,
                    token_id: market.yes_token_id.clone(),
                    entry_price: yes_entry,
                    bet_usdc: kelly.bet_usdc,
                    p_model: p_yes_model,
                    p_market: yes_entry,
                    kelly,
                });
            }
        }

        // Try NO side (buy NO at 1 - best_bid, because NO price = 1 - YES bid)
        let no_entry = 1.0 - best_bid;
        let no_effective_market = no_entry + half_spread;
        if no_entry >= dyn_min
            && no_entry <= self.cfg.price_max
            && p_no_model > no_effective_market + self.cfg.min_edge
        {
            let kelly = compute_bet(
                p_no_model,
                no_effective_market,
                bankroll_usdc,
                self.kelly_cfg.fraction,
                self.kelly_cfg.max_position_pct,
                self.kelly_cfg.min_bet_usdc,
                self.kelly_cfg.max_bet_usdc,
            );
            if kelly.is_actionable {
                return Some(Signal {
                    market: market.clone(),
                    side: Side::NO,
                    token_id: market.no_token_id.clone(),
                    entry_price: no_entry,
                    bet_usdc: kelly.bet_usdc,
                    p_model: p_no_model,
                    p_market: no_entry,
                    kelly,
                });
            }
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Threshold extraction
// ---------------------------------------------------------------------------

/// Extract the price threshold from a market question string.
///
/// Handles patterns like "Will BTC be above $50,000 at …" or
/// "BTC above 50000 USD at …".
pub fn extract_threshold(question: &str) -> Option<f64> {
    // Try to find a dollar amount like $50,000 or $50000
    let re_dollar = r"\$([0-9][0-9,]*)";
    if let Some(s) = first_regex_match(question, re_dollar) {
        let clean: String = s.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
        return clean.parse::<f64>().ok();
    }
    // Try bare number followed by "k" (e.g. "50k") or nothing
    let re_bare = r"\b([0-9][0-9,]*)(?:k)?\b";
    // Find the largest number in the question that looks like a price
    let mut candidates: Vec<f64> = question
        .split_whitespace()
        .filter_map(|w| {
            let clean: String = w.chars().filter(|c| c.is_ascii_digit() || *c == '.').collect();
            clean.parse::<f64>().ok()
        })
        .filter(|&v| v >= 100.0) // threshold must be at least $100
        .collect();
    let _ = re_bare; // used indirectly above
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.into_iter().last()
}

fn first_regex_match<'a>(text: &'a str, _pattern: &str) -> Option<&'a str> {
    // Minimal regex-free implementation: find the first $NNN pattern
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len()
                && (bytes[end].is_ascii_digit() || bytes[end] == b',' || bytes[end] == b'.')
            {
                end += 1;
            }
            if end > start {
                return Some(&text[start..end]);
            }
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::PriceLevel;
    use chrono::Utc;

    fn mock_book(bid: f64, ask: f64, liq: f64) -> OrderBook {
        OrderBook {
            market_id: "test".into(),
            token_id: "tok".into(),
            timestamp: Utc::now(),
            bids: vec![PriceLevel { price: bid, size: liq / bid }],
            asks: vec![PriceLevel { price: ask, size: liq / ask }],
        }
    }

    #[test]
    fn test_check_liquidity_pass() {
        let book = mock_book(0.49, 0.51, 1000.0);
        assert!(check_liquidity(&book, 500.0, 0.08));
    }

    #[test]
    fn test_check_liquidity_wide_spread() {
        let book = mock_book(0.40, 0.60, 1000.0);
        assert!(!check_liquidity(&book, 500.0, 0.08));
    }

    #[test]
    fn test_check_liquidity_low_liq() {
        let book = mock_book(0.49, 0.51, 10.0);
        assert!(!check_liquidity(&book, 500.0, 0.08));
    }

    #[test]
    fn test_effective_price_min_increases() {
        let base = 0.02_f64;
        let min_0 = effective_price_min(base, 0.0);
        let min_half = effective_price_min(base, 0.5);
        let min_full = effective_price_min(base, 1.0);
        assert!(min_0 < min_half);
        assert!(min_half < min_full);
    }

    #[test]
    fn test_extract_threshold_dollar() {
        assert_eq!(extract_threshold("Will BTC be above $50,000?"), Some(50000.0));
        assert_eq!(extract_threshold("Will ETH be above $3,000?"), Some(3000.0));
    }

    #[test]
    fn test_extract_threshold_bare() {
        let t = extract_threshold("BTC above 50000 at end");
        assert_eq!(t, Some(50000.0));
    }

    #[test]
    fn test_extract_threshold_none() {
        assert_eq!(extract_threshold("Will it rain today?"), None);
    }
}
