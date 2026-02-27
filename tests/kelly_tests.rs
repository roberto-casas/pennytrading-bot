/// Integration tests for the Kelly Criterion module.
///
/// These mirror the unit tests in src/kelly.rs but are run as integration
/// tests to verify the public API surface.
#[path = "../src/kelly.rs"]
mod kelly;

use kelly::{compute_bet, kelly_fraction, model_probability, norm_cdf};

#[test]
fn norm_cdf_at_zero_is_half() {
    assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
}

#[test]
fn norm_cdf_monotone() {
    assert!(norm_cdf(-1.0) < norm_cdf(0.0));
    assert!(norm_cdf(0.0) < norm_cdf(1.0));
}

#[test]
fn norm_cdf_symmetry() {
    for x in [-2.0_f64, -1.0, 0.5, 1.5, 2.5] {
        let sum = norm_cdf(x) + norm_cdf(-x);
        assert!((sum - 1.0).abs() < 1e-6, "symmetry failed at x={x}");
    }
}

#[test]
fn kelly_zero_edge() {
    // model == market → Kelly = 0
    assert!(kelly_fraction(0.5, 0.5).abs() < 1e-10);
    assert!(kelly_fraction(0.3, 0.3).abs() < 1e-10);
}

#[test]
fn kelly_positive_edge_is_positive() {
    // We estimate 60 %, market says 50 % → should bet
    assert!(kelly_fraction(0.60, 0.50) > 0.0);
    assert!(kelly_fraction(0.80, 0.30) > 0.0);
}

#[test]
fn kelly_negative_edge_is_negative() {
    assert!(kelly_fraction(0.40, 0.50) < 0.0);
}

#[test]
fn kelly_extreme_probabilities() {
    // Boundary: market at ~0 or ~1 should return 0
    assert_eq!(kelly_fraction(0.9, 0.0), 0.0);
    assert_eq!(kelly_fraction(0.1, 1.0), 0.0);
}

#[test]
fn compute_bet_no_edge() {
    let result = compute_bet(0.5, 0.5, 1000.0, 0.25, 0.10, 5.0, 500.0);
    assert!(!result.is_actionable);
    assert_eq!(result.bet_usdc, 0.0);
}

#[test]
fn compute_bet_respects_max_bet() {
    // Even with a huge bankroll the bet must not exceed max_bet_usdc
    let result = compute_bet(0.80, 0.40, 1_000_000.0, 0.25, 0.10, 5.0, 500.0);
    assert!(result.bet_usdc <= 500.0);
}

#[test]
fn compute_bet_respects_max_position_pct() {
    // 10 % of $100 = $10; with min_bet $5 this is actionable
    let result = compute_bet(0.80, 0.40, 100.0, 0.25, 0.10, 5.0, 500.0);
    assert!(result.bet_usdc <= 10.0 + 1e-9);
}

#[test]
fn compute_bet_below_minimum_is_not_actionable() {
    // Tiny bankroll → bet would be < min
    let result = compute_bet(0.60, 0.50, 10.0, 0.25, 0.10, 5.0, 500.0);
    // Adjusted bet = 0.25 * kelly_f * 10; kelly_f ≈ 0.20; adjusted ≈ 0.5 < 5 min
    assert!(!result.is_actionable);
}

#[test]
fn model_probability_at_threshold_is_half() {
    // spot == threshold → P ≈ 0.5 (GBM, zero drift)
    let p = model_probability(50_000.0, 50_000.0, 0.80, 300.0);
    assert!((p - 0.5).abs() < 0.01);
}

#[test]
fn model_probability_above_threshold_gt_half() {
    let p = model_probability(52_000.0, 50_000.0, 0.80, 300.0);
    assert!(p > 0.5);
}

#[test]
fn model_probability_below_threshold_lt_half() {
    let p = model_probability(48_000.0, 50_000.0, 0.80, 300.0);
    assert!(p < 0.5);
}

#[test]
fn model_probability_zero_time_is_deterministic() {
    // With 0 seconds remaining, the market has already resolved
    assert_eq!(model_probability(50_001.0, 50_000.0, 0.80, 0.0), 1.0);
    assert_eq!(model_probability(49_999.0, 50_000.0, 0.80, 0.0), 0.0);
}

#[test]
fn model_probability_longer_time_more_uncertainty() {
    // With spot only 0.2 % above threshold and a longer horizon,
    // uncertainty increases and probability falls back toward 0.5.
    // 30 s vs 300 s with very tight edge (50_100 vs 50_000).
    let spot = 50_100.0;
    let threshold = 50_000.0;
    let p_short = model_probability(spot, threshold, 0.80, 30.0);
    let p_long = model_probability(spot, threshold, 0.80, 300.0);
    // More time → d is smaller → probability closer to 0.5, i.e. p_long < p_short
    assert!(
        p_short > p_long,
        "p_short={p_short:.4} should be > p_long={p_long:.4}"
    );
}
