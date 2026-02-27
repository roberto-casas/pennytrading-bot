/// analytics.rs – Walk-forward validation and portfolio stress utilities.
use crate::models::{OrderBook, Position, ResolvedTradeSample};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone)]
pub struct WalkForwardFold {
    pub train_size: usize,
    pub test_size: usize,
    pub edge_threshold: f64,
    pub train_pnl_usdc: f64,
    pub test_pnl_usdc: f64,
    pub test_trades: usize,
    pub test_win_rate: f64,
}

#[derive(Debug, Clone)]
pub struct WalkForwardReport {
    pub sample_size: usize,
    pub folds: Vec<WalkForwardFold>,
    pub avg_test_pnl_usdc: f64,
    pub avg_test_win_rate: f64,
    pub avg_test_trades: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct StressScenario {
    pub name: &'static str,
    /// Per-position adverse mark-down applied to current mark-to-market.
    pub shock_pct: f64,
    /// Extra portfolio haircut when both BTC and ETH are exposed.
    pub cross_asset_haircut_pct: f64,
}

#[derive(Debug, Clone)]
pub struct StressResult {
    pub name: String,
    pub projected_loss_usdc: f64,
    pub projected_drawdown_pct: f64,
}

const EDGE_GRID_START: f64 = -0.02;
const EDGE_GRID_END: f64 = 0.20;
const EDGE_GRID_STEP: f64 = 0.01;

pub fn run_walk_forward(
    samples: &[ResolvedTradeSample],
    folds: usize,
    min_train_size: usize,
) -> Option<WalkForwardReport> {
    if folds == 0 || samples.len() < min_train_size + 2 {
        return None;
    }
    let n = samples.len();
    if min_train_size >= n {
        return None;
    }

    let mut fold_reports = Vec::new();
    let step = ((n - min_train_size) / folds).max(1);

    for fold_idx in 0..folds {
        let train_end = (min_train_size + fold_idx * step).min(n.saturating_sub(1));
        let test_end = if fold_idx + 1 == folds {
            n
        } else {
            (train_end + step).min(n)
        };
        if test_end <= train_end {
            continue;
        }

        let train = &samples[..train_end];
        let test = &samples[train_end..test_end];
        let threshold = choose_edge_threshold(train);
        let (train_pnl, _, _) = evaluate_threshold(train, threshold);
        let (test_pnl, test_trades, test_win_rate) = evaluate_threshold(test, threshold);

        fold_reports.push(WalkForwardFold {
            train_size: train.len(),
            test_size: test.len(),
            edge_threshold: threshold,
            train_pnl_usdc: train_pnl,
            test_pnl_usdc: test_pnl,
            test_trades,
            test_win_rate,
        });
    }

    if fold_reports.is_empty() {
        return None;
    }

    let folds_n = fold_reports.len() as f64;
    let avg_test_pnl_usdc = fold_reports.iter().map(|f| f.test_pnl_usdc).sum::<f64>() / folds_n;
    let avg_test_win_rate = fold_reports.iter().map(|f| f.test_win_rate).sum::<f64>() / folds_n;
    let avg_test_trades = fold_reports
        .iter()
        .map(|f| f.test_trades as f64)
        .sum::<f64>()
        / folds_n;

    Some(WalkForwardReport {
        sample_size: samples.len(),
        folds: fold_reports,
        avg_test_pnl_usdc,
        avg_test_win_rate,
        avg_test_trades,
    })
}

pub fn default_stress_scenarios() -> Vec<StressScenario> {
    vec![
        StressScenario {
            name: "mild_shock",
            shock_pct: 0.30,
            cross_asset_haircut_pct: 0.04,
        },
        StressScenario {
            name: "event_shock",
            shock_pct: 0.55,
            cross_asset_haircut_pct: 0.08,
        },
        StressScenario {
            name: "crash_shock",
            shock_pct: 0.80,
            cross_asset_haircut_pct: 0.12,
        },
    ]
}

pub fn evaluate_portfolio_stress(
    balance_usdc: f64,
    positions: &[Position],
    order_books: &HashMap<String, OrderBook>,
    scenarios: &[StressScenario],
) -> Vec<StressResult> {
    let marks: Vec<(String, f64, f64)> = positions
        .iter()
        .filter(|p| p.status.is_open())
        .map(|p| {
            let px = order_books
                .get(&p.token_id)
                .and_then(|b| b.best_bid())
                .unwrap_or(p.entry_price)
                .clamp(0.0, 1.0);
            (p.asset.clone(), px, p.size.max(0.0))
        })
        .collect();

    let mtm_value = marks.iter().map(|(_, px, sz)| px * sz).sum::<f64>();
    let equity = (balance_usdc.max(0.0) + mtm_value).max(1e-6);
    let has_cross_asset = {
        let mut assets = std::collections::HashSet::new();
        for (asset, _, _) in &marks {
            assets.insert(asset.as_str());
        }
        assets.len() >= 2
    };

    let mut out: Vec<StressResult> = scenarios
        .iter()
        .map(|s| {
            let base_loss = marks
                .iter()
                .map(|(_, px, sz)| {
                    let stressed_px = (*px * (1.0 - s.shock_pct)).max(0.0);
                    (px - stressed_px).max(0.0) * sz
                })
                .sum::<f64>();
            let cross_loss = if has_cross_asset {
                mtm_value * s.cross_asset_haircut_pct.max(0.0)
            } else {
                0.0
            };
            let loss = (base_loss + cross_loss).max(0.0);
            StressResult {
                name: s.name.to_string(),
                projected_loss_usdc: loss,
                projected_drawdown_pct: (loss / equity).min(1.0),
            }
        })
        .collect();

    out.sort_by(|a, b| {
        b.projected_loss_usdc
            .partial_cmp(&a.projected_loss_usdc)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

pub fn asset_pnl_correlation(
    samples: &[ResolvedTradeSample],
    asset_a: &str,
    asset_b: &str,
) -> Option<f64> {
    let a = asset_a.to_uppercase();
    let b = asset_b.to_uppercase();
    let mut bucket_a: BTreeMap<i64, f64> = BTreeMap::new();
    let mut bucket_b: BTreeMap<i64, f64> = BTreeMap::new();

    for s in samples {
        let key = s.closed_at.timestamp() / 60;
        match s.asset.to_uppercase().as_str() {
            x if x == a => *bucket_a.entry(key).or_insert(0.0) += s.pnl_usdc,
            x if x == b => *bucket_b.entry(key).or_insert(0.0) += s.pnl_usdc,
            _ => {}
        }
    }

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (k, xa) in &bucket_a {
        if let Some(yb) = bucket_b.get(k) {
            xs.push(*xa);
            ys.push(*yb);
        }
    }
    pearson_correlation(&xs, &ys)
}

fn choose_edge_threshold(train: &[ResolvedTradeSample]) -> f64 {
    let mut best_threshold = 0.0;
    let mut best_score = f64::NEG_INFINITY;
    let mut t = EDGE_GRID_START;
    while t <= EDGE_GRID_END + 1e-12 {
        let (pnl, trades, win_rate) = evaluate_threshold(train, t);
        let trade_penalty = if trades == 0 { -1_000_000.0 } else { 0.0 };
        let score = pnl + win_rate * 5.0 + trade_penalty;
        if score > best_score {
            best_score = score;
            best_threshold = t;
        }
        t += EDGE_GRID_STEP;
    }
    best_threshold
}

fn evaluate_threshold(samples: &[ResolvedTradeSample], edge_threshold: f64) -> (f64, usize, f64) {
    let mut pnl = 0.0;
    let mut trades = 0usize;
    let mut wins = 0usize;
    for s in samples {
        if s.edge >= edge_threshold {
            pnl += s.pnl_usdc;
            trades += 1;
            if s.pnl_usdc > 0.0 {
                wins += 1;
            }
        }
    }
    let win_rate = if trades > 0 {
        wins as f64 / trades as f64
    } else {
        0.0
    };
    (pnl, trades, win_rate)
}

fn pearson_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() < 3 || xs.len() != ys.len() {
        return None;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 1e-12 || var_y <= 1e-12 {
        return None;
    }
    Some((cov / (var_x.sqrt() * var_y.sqrt())).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{OrderType, Side, TradeStatus};
    use chrono::{Duration, Utc};

    fn sample_trade(i: i64, asset: &str, edge: f64, pnl: f64) -> ResolvedTradeSample {
        let t = Utc::now() + Duration::minutes(i);
        ResolvedTradeSample {
            position_id: format!("p_{i}"),
            asset: asset.to_string(),
            regime: "low_vol|normal_spread|normal_liq|normal_event".to_string(),
            edge,
            model_prob: 0.6,
            market_prob: 0.5,
            pnl_usdc: pnl,
            outcome: if pnl > 0.0 { 1.0 } else { 0.0 },
            created_at: t,
            closed_at: t,
        }
    }

    #[test]
    fn walk_forward_returns_report() {
        let samples = vec![
            sample_trade(0, "BTC", 0.02, -1.0),
            sample_trade(1, "BTC", 0.04, 1.0),
            sample_trade(2, "BTC", 0.05, 2.0),
            sample_trade(3, "ETH", 0.01, -0.5),
            sample_trade(4, "ETH", 0.08, 3.0),
            sample_trade(5, "BTC", 0.03, 0.5),
            sample_trade(6, "ETH", 0.07, 1.2),
            sample_trade(7, "BTC", 0.10, 2.2),
        ];
        let report = run_walk_forward(&samples, 3, 4).expect("report");
        assert_eq!(report.sample_size, samples.len());
        assert!(!report.folds.is_empty());
    }

    #[test]
    fn stress_eval_returns_worst_first() {
        let positions = vec![
            Position {
                position_id: "p1".into(),
                market_id: "m1".into(),
                token_id: "t1".into(),
                side: Side::YES,
                asset: "BTC".into(),
                entry_price: 0.5,
                size: 100.0,
                cost_usdc: 50.0,
                stop_loss_price: 0.2,
                take_profit_price: 0.8,
                high_water_mark: 0.5,
                status: TradeStatus::Open,
                exit_price: None,
                pnl_usdc: None,
                opened_at: Utc::now(),
                closed_at: None,
                dry_run: true,
                order_type: OrderType::Limit,
            },
            Position {
                position_id: "p2".into(),
                market_id: "m2".into(),
                token_id: "t2".into(),
                side: Side::YES,
                asset: "ETH".into(),
                entry_price: 0.5,
                size: 100.0,
                cost_usdc: 50.0,
                stop_loss_price: 0.2,
                take_profit_price: 0.8,
                high_water_mark: 0.5,
                status: TradeStatus::Open,
                exit_price: None,
                pnl_usdc: None,
                opened_at: Utc::now(),
                closed_at: None,
                dry_run: true,
                order_type: OrderType::Limit,
            },
        ];
        let books = HashMap::from([
            (
                "t1".to_string(),
                OrderBook {
                    market_id: "m1".into(),
                    token_id: "t1".into(),
                    timestamp: Utc::now(),
                    bids: vec![crate::models::PriceLevel {
                        price: 0.52,
                        size: 100.0,
                    }],
                    asks: vec![crate::models::PriceLevel {
                        price: 0.53,
                        size: 100.0,
                    }],
                },
            ),
            (
                "t2".to_string(),
                OrderBook {
                    market_id: "m2".into(),
                    token_id: "t2".into(),
                    timestamp: Utc::now(),
                    bids: vec![crate::models::PriceLevel {
                        price: 0.48,
                        size: 100.0,
                    }],
                    asks: vec![crate::models::PriceLevel {
                        price: 0.49,
                        size: 100.0,
                    }],
                },
            ),
        ]);
        let stress =
            evaluate_portfolio_stress(1000.0, &positions, &books, &default_stress_scenarios());
        assert!(!stress.is_empty());
        assert!(stress[0].projected_loss_usdc >= stress.last().unwrap().projected_loss_usdc);
    }

    #[test]
    fn correlation_returns_value_when_buckets_overlap() {
        let samples = vec![
            sample_trade(0, "BTC", 0.05, 1.0),
            sample_trade(0, "ETH", 0.05, 2.0),
            sample_trade(1, "BTC", 0.05, -1.0),
            sample_trade(1, "ETH", 0.05, -2.0),
            sample_trade(2, "BTC", 0.05, 1.5),
            sample_trade(2, "ETH", 0.05, 3.0),
        ];
        let corr = asset_pnl_correlation(&samples, "BTC", "ETH").expect("corr");
        assert!(corr > 0.9);
    }
}
