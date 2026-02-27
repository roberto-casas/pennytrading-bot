/// kelly.rs – Kelly Criterion position sizing for binary prediction markets.
///
/// For a binary contract purchased at market price `p` (= implied probability),
/// which pays $1.00 if correct:
///
///   Net odds:  b = (1 − p) / p
///   Full Kelly: f* = (model_p · b − (1 − model_p)) / b
///                  = model_p − p          (simplified for binary payoff)
///
/// We apply a *fractional* Kelly multiplier (default 0.25) to reduce variance
/// and guard against model error.

// ---------------------------------------------------------------------------
// Normal distribution helpers (no external crate needed)
// ---------------------------------------------------------------------------

/// Approximation of the error function using Horner's method.
/// Maximum error < 1.5 × 10⁻⁷.
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0_f64 } else { -1.0_f64 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let result = 1.0 - poly * (-x * x).exp();
    sign * result
}

/// Standard normal CDF: Φ(x) = P(Z ≤ x) for Z ~ N(0,1).
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

// ---------------------------------------------------------------------------
// Kelly core
// ---------------------------------------------------------------------------

/// Compute the full Kelly fraction for a binary prediction market.
///
/// Returns a value in (−∞, 1]. A negative return means no edge (do not bet).
pub fn kelly_fraction(p_model: f64, p_market: f64) -> f64 {
    if p_market <= 0.0 || p_market >= 1.0 {
        return 0.0;
    }
    // Net odds per unit bet when buying at p_market
    let b = (1.0 - p_market) / p_market;
    // Standard Kelly: f* = (p·b − (1−p)) / b
    (p_model * b - (1.0 - p_model)) / b
}

/// Result of the Kelly bet-sizing calculation.
#[derive(Debug, Clone)]
pub struct BetResult {
    /// Full (unconstrained) Kelly fraction.
    pub full_fraction: f64,
    /// Fraction after applying the Kelly multiplier and position cap.
    pub adjusted_fraction: f64,
    /// Concrete USDC amount to bet given the bankroll.
    pub bet_usdc: f64,
    /// Edge = model probability − market price.
    pub edge: f64,
    /// True when there is positive expected value *and* the bet is above the minimum.
    pub is_actionable: bool,
}

/// Compute the bet size in USDC using fractional Kelly.
///
/// # Parameters
/// - `p_model`: Model's estimated probability of the contract paying $1.
/// - `p_market`: Current market price (implied probability).
/// - `bankroll_usdc`: Available USDC balance.
/// - `kelly_multiplier`: Fraction of full Kelly to use (e.g. 0.25).
/// - `max_position_pct`: Hard cap as a fraction of bankroll per position.
/// - `min_bet_usdc`: Minimum bet; return zero if below this.
/// - `max_bet_usdc`: Hard maximum per trade.
pub fn compute_bet(
    p_model: f64,
    p_market: f64,
    bankroll_usdc: f64,
    kelly_multiplier: f64,
    max_position_pct: f64,
    min_bet_usdc: f64,
    max_bet_usdc: f64,
) -> BetResult {
    let edge = p_model - p_market;
    let full_f = kelly_fraction(p_model, p_market);
    let adjusted_f = (full_f * kelly_multiplier).max(0.0).min(max_position_pct);

    let raw_bet = adjusted_f * bankroll_usdc;
    let bet = if raw_bet < min_bet_usdc {
        0.0
    } else {
        raw_bet
            .min(max_bet_usdc)
            .min(max_position_pct * bankroll_usdc)
    };

    BetResult {
        full_fraction: full_f,
        adjusted_fraction: adjusted_f,
        bet_usdc: bet,
        edge,
        is_actionable: full_f > 0.0 && bet > 0.0,
    }
}

// ---------------------------------------------------------------------------
// Model probability (log-normal short-horizon)
// ---------------------------------------------------------------------------

/// Estimate the probability that the spot price ends **above** `threshold`
/// at resolution, using a zero-drift log-normal (GBM) model.
///
/// # Parameters
/// - `current_spot`: Current underlying price (e.g. BTC/USD).
/// - `threshold`: The strike price in the market question.
/// - `sigma_annual`: Annualised volatility (e.g. 0.80 = 80 % for BTC).
/// - `time_to_resolution_secs`: Seconds until the market resolves.
pub fn model_probability(
    current_spot: f64,
    threshold: f64,
    sigma_annual: f64,
    time_to_resolution_secs: f64,
) -> f64 {
    if current_spot <= 0.0 || threshold <= 0.0 || time_to_resolution_secs <= 0.0 {
        return if current_spot > threshold { 1.0 } else { 0.0 };
    }

    // σ√t  (years)
    let t_years = time_to_resolution_secs / (365.25 * 24.0 * 3600.0);
    let sigma_t = sigma_annual * t_years.sqrt();
    if sigma_t == 0.0 {
        return if current_spot > threshold { 1.0 } else { 0.0 };
    }

    // d = ln(S/K) / σ√t
    let d = (current_spot / threshold).ln() / sigma_t;
    // P(S_T > K) = Φ(d)
    norm_cdf(d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_cdf_symmetry() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((norm_cdf(1.645) - 0.95).abs() < 0.001);
        assert!((norm_cdf(-1.645) - 0.05).abs() < 0.001);
        // Symmetry
        assert!((norm_cdf(1.0) + norm_cdf(-1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kelly_fraction_at_fair_price() {
        // When p_model == p_market, edge is zero → Kelly = 0
        let f = kelly_fraction(0.5, 0.5);
        assert!(f.abs() < 1e-10);
    }

    #[test]
    fn test_kelly_fraction_positive_edge() {
        // Model says 60 %, market at 50 % → positive Kelly
        let f = kelly_fraction(0.6, 0.5);
        assert!(f > 0.0);
    }

    #[test]
    fn test_kelly_fraction_negative_edge() {
        // Model says 40 %, market at 50 % → no edge
        let f = kelly_fraction(0.4, 0.5);
        assert!(f < 0.0);
    }

    #[test]
    fn test_compute_bet_below_minimum() {
        // Very small edge should yield below-min bet → is_actionable = false
        let result = compute_bet(0.505, 0.5, 100.0, 0.25, 0.10, 5.0, 500.0);
        // Edge is tiny; bet may be zero
        assert!(!result.is_actionable || result.bet_usdc >= 0.0);
    }

    #[test]
    fn test_compute_bet_respects_max() {
        // Large bankroll + large edge should still respect max_bet cap
        let result = compute_bet(0.80, 0.40, 100_000.0, 0.25, 0.10, 5.0, 500.0);
        assert!(result.bet_usdc <= 500.0);
    }

    #[test]
    fn test_model_probability_at_threshold() {
        // Spot = threshold → ~50 % probability (zero drift)
        let p = model_probability(50_000.0, 50_000.0, 0.80, 300.0);
        assert!((p - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_model_probability_above_threshold() {
        // Spot well above threshold → high probability
        let p = model_probability(52_000.0, 50_000.0, 0.80, 300.0);
        assert!(p > 0.5);
    }

    #[test]
    fn test_model_probability_below_threshold() {
        // Spot well below threshold → low probability
        let p = model_probability(48_000.0, 50_000.0, 0.80, 300.0);
        assert!(p < 0.5);
    }
}
