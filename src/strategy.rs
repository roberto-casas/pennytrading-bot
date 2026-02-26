/// strategy.rs – Signal generation for the penny strategy.
///
/// For each active market the strategy:
///   1. Checks the order book (liquidity, spread).
///   2. Estimates fair value using a log-normal model.
///   3. Computes Kelly bet size based on the edge.
///   4. Enforces price and time limits (dynamically adjusted).
///   5. Returns a `Signal` if all conditions are met.
use crate::{
    config::{KellyConfig, RiskConfig, StrategyConfig},
    kelly::{compute_bet, model_probability, BetResult},
    models::{Market, OrderBook, Side},
};
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
// Strategy
// ---------------------------------------------------------------------------

pub struct Strategy {
    pub cfg: StrategyConfig,
    pub kelly_cfg: KellyConfig,
    pub risk_cfg: RiskConfig,
    /// Current BTC spot price (updated externally, e.g. from a price feed).
    pub btc_spot: f64,
    /// Current ETH spot price.
    pub eth_spot: f64,
    /// Annualised volatility for BTC.
    pub btc_sigma: f64,
    /// Annualised volatility for ETH.
    pub eth_sigma: f64,
}

impl Strategy {
    pub fn new(cfg: StrategyConfig, kelly_cfg: KellyConfig, risk_cfg: RiskConfig) -> Self {
        Self {
            cfg,
            kelly_cfg,
            risk_cfg,
            btc_spot: 0.0,
            eth_spot: 0.0,
            btc_sigma: 0.80, // 80 % annualised – conservative for 5/15-min horizon
            eth_sigma: 0.90,
        }
    }

    /// Update spot prices (called whenever fresh price data arrives).
    pub fn update_spots(&mut self, btc: f64, eth: f64) {
        self.btc_spot = btc;
        self.eth_spot = eth;
    }

    /// Evaluate a market and return a `Signal` if there is a tradeable
    /// opportunity on either the YES or NO side.
    ///
    /// Returns `None` when no edge is found or conditions are not met.
    pub fn evaluate(
        &self,
        market: &Market,
        book: &OrderBook,
        bankroll_usdc: f64,
        open_positions: usize,
    ) -> Option<Signal> {
        // --- Gate 1: position limit -----------------------------------------
        if open_positions >= self.risk_cfg.max_open_positions {
            debug!("Max open positions reached");
            return None;
        }

        // --- Gate 2: time limit ---------------------------------------------
        if market.is_near_resolution(self.risk_cfg.time_limit_fraction) {
            debug!("Market {} too close to resolution", market.label());
            return None;
        }

        // --- Gate 3: liquidity and spread -----------------------------------
        if !check_liquidity(
            book,
            self.cfg.min_liquidity_usdc,
            self.cfg.max_spread_pct,
        ) {
            return None;
        }

        // --- Gate 4: select side and price ----------------------------------
        let best_ask = book.best_ask()?;
        let best_bid = book.best_bid()?;

        // Derive spot price and sigma for this asset
        let (spot, sigma) = match market.asset.as_str() {
            "BTC" => (self.btc_spot, self.btc_sigma),
            "ETH" => (self.eth_spot, self.eth_sigma),
            _ => return None,
        };
        if spot <= 0.0 {
            debug!("No spot price available for {}", market.asset);
            return None;
        }

        // Extract threshold from market question (very rough heuristic;
        // real implementation would parse structured market data)
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
        let p_yes_model = model_probability(spot, threshold, sigma, time_secs);
        let p_no_model = 1.0 - p_yes_model;

        // Try YES side (buy YES at best ask)
        let yes_entry = best_ask;
        let dyn_min = effective_price_min(self.cfg.price_min, elapsed_fraction);
        if yes_entry >= dyn_min
            && yes_entry <= self.cfg.price_max
            && p_yes_model > yes_entry + self.cfg.min_edge
        {
            let kelly = compute_bet(
                p_yes_model,
                yes_entry,
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

        // Try NO side (buy NO at 1 − best_bid, because NO price = 1 − YES bid)
        let no_entry = 1.0 - best_bid; // implied NO ask
        if no_entry >= dyn_min
            && no_entry <= self.cfg.price_max
            && p_no_model > no_entry + self.cfg.min_edge
        {
            let kelly = compute_bet(
                p_no_model,
                no_entry,
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
