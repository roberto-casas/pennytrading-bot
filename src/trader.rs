/// trader.rs – Order management: open/close positions, SL/TP/trailing-stop
/// monitoring, trade resolution checking, and dry-run support.
use anyhow::{anyhow, Result};
use chrono::Utc;
use serde_json::Value;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    config::{ExecutionConfig, RiskConfig, StrategyConfig},
    database::Database,
    models::{Market, OrderBook, OrderType, Position, Side, TradeDiagnostics, TradeStatus},
    polymarket::PolymarketClient,
    strategy::Signal,
};

// ---------------------------------------------------------------------------
// Trader
// ---------------------------------------------------------------------------

/// Trailing stop percentage (how far below the high-water mark to trigger).
const TRAILING_STOP_PCT: f64 = 0.20; // 20% below peak

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionState {
    Filled,
    Partial,
    Accepted,
    Rejected,
}

pub struct Trader {
    pub risk: RiskConfig,
    pub strategy: StrategyConfig,
    pub execution: ExecutionConfig,
    pub dry_run: bool,
}

impl Trader {
    pub fn new(
        risk: RiskConfig,
        strategy: StrategyConfig,
        execution: ExecutionConfig,
        dry_run: bool,
    ) -> Self {
        Self {
            risk,
            strategy,
            execution,
            dry_run,
        }
    }

    // ------------------------------------------------------------------
    // Open a new position from a strategy signal
    // ------------------------------------------------------------------

    /// Open a position based on a strategy signal.
    ///
    /// Prefers a **limit order** at the best ask (or slightly inside the spread)
    /// when `strategy.prefer_limit_orders` is true.
    pub async fn open_position(
        &self,
        signal: &Signal,
        client: &PolymarketClient,
        db: &Database,
        session_id: &str,
    ) -> Result<Position> {
        let requested_size = signal.bet_usdc / signal.entry_price;
        let mut order_type = if self.strategy.prefer_limit_orders {
            OrderType::Limit
        } else {
            OrderType::Market
        };

        // Limit price: place AT the best ask for immediate fill potential.
        // Adding to the ask only makes us pay more; subtracting risks no fill.
        // Use the signal's entry_price directly (which is already best_ask or
        // derived NO ask).
        let limit_price = signal.entry_price;

        let mut resp = match order_type {
            OrderType::Limit => {
                client
                    .place_limit_order(
                        &signal.token_id,
                        signal.side.as_str(),
                        limit_price,
                        requested_size,
                        &signal.market.condition_id,
                    )
                    .await?
            }
            OrderType::Market => {
                client
                    .place_market_order(
                        &signal.token_id,
                        signal.side.as_str(),
                        signal.bet_usdc,
                        &signal.market.condition_id,
                    )
                    .await?
            }
        };

        let mut exec = classify_execution(&resp, order_type, self.dry_run);

        // Validate entries by requiring a fill confirmation. If a LIMIT order
        // is only accepted/pending, immediately fallback to MARKET.
        if matches!(exec, ExecutionState::Accepted)
            && order_type == OrderType::Limit
            && !self.dry_run
        {
            let fallback = client
                .place_market_order(
                    &signal.token_id,
                    signal.side.as_str(),
                    signal.bet_usdc,
                    &signal.market.condition_id,
                )
                .await?;
            resp = fallback;
            order_type = OrderType::Market;
            exec = classify_execution(&resp, order_type, false);
        }

        if matches!(exec, ExecutionState::Rejected | ExecutionState::Accepted) {
            return Err(anyhow!("entry order was not filled"));
        }
        let entry_price = extract_fill_price(&resp).unwrap_or(signal.entry_price);
        let has_actual_fill_price = extract_fill_price(&resp).is_some();
        let size = if matches!(exec, ExecutionState::Partial) {
            extract_filled_size(&resp)
                .unwrap_or(0.0)
                .min(requested_size)
        } else {
            extract_filled_size(&resp).unwrap_or(requested_size)
        };
        if size <= 0.0 {
            return Err(anyhow!("entry order reported zero fill size"));
        }
        let notional_usdc = extract_filled_notional(&resp)
            .unwrap_or(entry_price * size)
            .max(0.0);
        let entry_cost_usdc =
            self.execution_cost_usdc(notional_usdc, order_type, &resp, has_actual_fill_price);
        let cost_usdc = notional_usdc + entry_cost_usdc;

        // Compute stop-loss and take-profit levels
        let stop_loss_price = entry_price * (1.0 - self.risk.stop_loss_pct);
        let take_profit_price = compute_take_profit_price(entry_price, self.risk.take_profit_pct);

        let status = if self.dry_run {
            TradeStatus::DryRun
        } else {
            TradeStatus::Open
        };

        let pos = Position {
            position_id: Uuid::new_v4().to_string(),
            market_id: signal.market.condition_id.clone(),
            token_id: signal.token_id.clone(),
            side: signal.side,
            asset: signal.market.asset.clone(),
            entry_price,
            size,
            cost_usdc,
            stop_loss_price,
            take_profit_price,
            high_water_mark: entry_price,
            status,
            exit_price: None,
            pnl_usdc: None,
            opened_at: Utc::now(),
            closed_at: None,
            dry_run: self.dry_run,
            order_type,
        };

        db.upsert_position(&pos, Some(session_id))?;
        db.upsert_trade_diagnostics(&TradeDiagnostics {
            position_id: pos.position_id.clone(),
            session_id: session_id.to_string(),
            market_id: pos.market_id.clone(),
            asset: pos.asset.clone(),
            side: pos.side,
            model_prob: signal.p_model,
            market_prob: signal.p_market,
            edge: signal.p_model - signal.p_market,
            regime: signal.regime.clone(),
            entry_price: pos.entry_price,
            bet_usdc: signal.bet_usdc,
            created_at: Utc::now(),
        })?;

        if matches!(exec, ExecutionState::Partial) {
            warn!(
                "[{}] Partial entry fill {} {} {} @ {:.4} size={:.2}/{:.2}",
                if self.dry_run { "DRY" } else { "LIVE" },
                pos.asset,
                pos.side,
                signal.market.label(),
                pos.entry_price,
                pos.size,
                requested_size,
            );
        }

        info!(
            "[{}] Opened {} {} {} @ {:.4} SL={:.4} TP={:.4} size={:.2} cost=${:.2}",
            if self.dry_run { "DRY" } else { "LIVE" },
            pos.asset,
            pos.side,
            signal.market.label(),
            pos.entry_price,
            pos.stop_loss_price,
            pos.take_profit_price,
            pos.size,
            pos.cost_usdc,
        );

        Ok(pos)
    }

    // ------------------------------------------------------------------
    // Close a position
    // ------------------------------------------------------------------

    /// Close a position at `exit_price` with the given status.
    pub async fn close_position(
        &self,
        pos: &mut Position,
        exit_price: f64,
        status: TradeStatus,
        client: &PolymarketClient,
        db: &Database,
        session_id: &str,
    ) -> Result<bool> {
        // Place a sell order (limit preferred; market as fallback)
        let mut order_type = if self.strategy.prefer_limit_orders {
            OrderType::Limit
        } else {
            OrderType::Market
        };
        let sell_resp = if order_type == OrderType::Limit {
            client
                .place_limit_order(&pos.token_id, "SELL", exit_price, pos.size, &pos.market_id)
                .await
        } else {
            client
                .place_market_order(&pos.token_id, "SELL", pos.size * exit_price, &pos.market_id)
                .await
        };

        let mut resp = match sell_resp {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to place exit order for {}: {e}", pos.position_id);
                return Ok(false);
            }
        };
        let mut exec = classify_execution(&resp, order_type, self.dry_run);

        if matches!(exec, ExecutionState::Accepted)
            && order_type == OrderType::Limit
            && !self.dry_run
        {
            let fallback = client
                .place_market_order(&pos.token_id, "SELL", pos.size * exit_price, &pos.market_id)
                .await;
            match fallback {
                Ok(v) => {
                    resp = v;
                    order_type = OrderType::Market;
                    exec = classify_execution(&resp, order_type, false);
                }
                Err(e) => {
                    warn!(
                        "Exit LIMIT accepted but market fallback failed for {}: {e}",
                        pos.position_id
                    );
                    return Ok(false);
                }
            }
        }

        if matches!(exec, ExecutionState::Rejected | ExecutionState::Accepted) {
            warn!(
                "Exit order not filled for {}; keeping position open",
                pos.position_id
            );
            return Ok(false);
        }
        if matches!(exec, ExecutionState::Partial) {
            let partial_size = extract_filled_size(&resp).unwrap_or(0.0).min(pos.size);
            if partial_size <= 0.0 {
                warn!(
                    "Exit order partially filled for {} but no filled_size found; keeping position open",
                    pos.position_id
                );
                return Ok(false);
            }

            let partial_price = extract_fill_price(&resp).unwrap_or(exit_price);
            let has_actual_partial_price = extract_fill_price(&resp).is_some();
            let partial_notional = extract_filled_notional(&resp)
                .unwrap_or(partial_price * partial_size)
                .max(0.0);
            let exit_cost_usdc = self.execution_cost_usdc(
                partial_notional,
                order_type,
                &resp,
                has_actual_partial_price,
            );
            let basis_per_share = if pos.size > 0.0 {
                pos.cost_usdc / pos.size
            } else {
                pos.entry_price
            };
            let realized_prev = pos.pnl_usdc.unwrap_or(0.0);
            let realized_add = partial_notional - basis_per_share * partial_size - exit_cost_usdc;
            pos.pnl_usdc = Some(realized_prev + realized_add);
            let remaining_cost = (pos.cost_usdc - basis_per_share * partial_size).max(0.0);
            pos.size = (pos.size - partial_size).max(0.0);
            pos.cost_usdc = remaining_cost;

            if pos.size <= 1e-9 {
                pos.exit_price = Some(partial_price);
                pos.status = status;
                pos.closed_at = Some(Utc::now());
                pos.cost_usdc = 0.0;
                db.upsert_position(pos, Some(session_id))?;
                info!(
                    "[{}] Closed {} {} via partial completion @ {:.4} P&L={:+.2} ({})",
                    if self.dry_run { "DRY" } else { "LIVE" },
                    pos.asset,
                    pos.side,
                    partial_price,
                    pos.pnl_usdc.unwrap_or(0.0),
                    status,
                );
                return Ok(true);
            }

            pos.status = if pos.dry_run {
                TradeStatus::DryRun
            } else {
                TradeStatus::Open
            };
            db.upsert_position(pos, Some(session_id))?;
            warn!(
                "Exit partial fill for {}: sold {:.2}, remaining {:.2}",
                pos.position_id, partial_size, pos.size
            );
            return Ok(false);
        }

        let executed_price = extract_fill_price(&resp).unwrap_or(exit_price);
        let has_actual_close_price = extract_fill_price(&resp).is_some();
        let close_notional = extract_filled_notional(&resp)
            .unwrap_or(executed_price * pos.size)
            .max(0.0);
        let exit_cost_usdc =
            self.execution_cost_usdc(close_notional, order_type, &resp, has_actual_close_price);

        let pnl = pos.pnl_usdc.unwrap_or(0.0) + close_notional - pos.cost_usdc - exit_cost_usdc;
        pos.exit_price = Some(executed_price);
        pos.pnl_usdc = Some(pnl);
        pos.status = status;
        pos.closed_at = Some(Utc::now());
        pos.cost_usdc = 0.0;

        db.upsert_position(pos, Some(session_id))?;

        info!(
            "[{}] Closed {} {} @ {:.4} P&L={:+.2} USDC ({})",
            if self.dry_run { "DRY" } else { "LIVE" },
            pos.asset,
            pos.side,
            executed_price,
            pnl,
            status,
        );

        Ok(true)
    }

    // ------------------------------------------------------------------
    // Monitor open positions (SL / TP / trailing stop / time limit)
    // ------------------------------------------------------------------

    /// Check all open positions against stop-loss, take-profit, trailing stop,
    /// and time-limit rules.  Returns the list of positions that were closed.
    pub async fn monitor_positions(
        &self,
        positions: &mut Vec<Position>,
        order_books: &std::collections::HashMap<String, OrderBook>,
        markets: &[Market],
        client: &PolymarketClient,
        db: &Database,
        session_id: &str,
    ) -> Result<Vec<Position>> {
        let mut closed = Vec::new();

        for pos in positions.iter_mut() {
            if !pos.status.is_open() {
                continue;
            }

            // Resolve current price from order book (best bid for selling)
            let current_price = order_books
                .get(&pos.token_id)
                .and_then(|b| b.best_bid())
                .unwrap_or(pos.entry_price);

            // Find the associated market for time-limit check
            let market = markets.iter().find(|m| m.condition_id == pos.market_id);

            // Update trailing stop high-water mark
            let trailing_stop_price = pos.update_trailing_stop(current_price, TRAILING_STOP_PCT);

            let (trigger_close, close_status) = if pos.should_stop_loss(current_price) {
                (true, TradeStatus::ClosedStopLoss)
            } else if pos.should_take_profit(current_price) {
                (true, TradeStatus::ClosedTakeProfit)
            } else if let Some(trail_price) = trailing_stop_price {
                // Trailing stop: only triggers if price dropped below trail
                if current_price <= trail_price {
                    (true, TradeStatus::ClosedTakeProfit) // trailing stop is a profit-lock
                } else {
                    (false, TradeStatus::Open)
                }
            } else if market
                .map(|m| m.is_near_resolution(self.risk.time_limit_fraction))
                .unwrap_or(false)
            {
                (true, TradeStatus::ClosedTimeLimit)
            } else {
                (false, TradeStatus::Open)
            };

            if trigger_close {
                if self
                    .close_position(pos, current_price, close_status, client, db, session_id)
                    .await?
                {
                    closed.push(pos.clone());
                }
            }
        }

        Ok(closed)
    }

    // ------------------------------------------------------------------
    // Trade resolution checking
    // ------------------------------------------------------------------

    /// Check whether markets have resolved and update positions accordingly.
    pub fn check_resolution(
        &self,
        positions: &mut Vec<Position>,
        markets: &[Market],
        db: &Database,
        session_id: &str,
    ) -> Result<Vec<Position>> {
        let mut resolved = Vec::new();

        for pos in positions.iter_mut() {
            if !pos.status.is_open() {
                continue;
            }

            let Some(market) = markets.iter().find(|m| m.condition_id == pos.market_id) else {
                continue;
            };

            let Some(res_price) = market.resolution_price else {
                continue;
            };

            // In a binary market: YES pays 1.0 if res_price = 1.0, NO pays 1.0 if res_price = 0.0
            let payout_per_share = match pos.side {
                Side::YES => res_price,
                Side::NO => 1.0 - res_price,
            };

            let exit_price = payout_per_share;
            let settlement_notional = payout_per_share * pos.size;
            let pnl = pos.pnl_usdc.unwrap_or(0.0) + settlement_notional - pos.cost_usdc;
            let status = if pnl >= 0.0 {
                TradeStatus::ResolvedWin
            } else {
                TradeStatus::ResolvedLoss
            };

            pos.exit_price = Some(exit_price);
            pos.pnl_usdc = Some(pnl);
            pos.status = status;
            pos.closed_at = Some(Utc::now());
            pos.cost_usdc = 0.0;

            db.upsert_position(pos, Some(session_id))?;

            info!(
                "Resolved {} {} – payout={:.2} P&L={:+.2} ({})",
                pos.asset, pos.side, payout_per_share, pnl, status,
            );

            resolved.push(pos.clone());
        }

        Ok(resolved)
    }

    fn execution_cost_usdc(
        &self,
        notional_usdc: f64,
        order_type: OrderType,
        resp: &Value,
        has_actual_fill_price: bool,
    ) -> f64 {
        if notional_usdc <= 0.0 {
            return 0.0;
        }
        let fee_usdc = extract_fee_usdc(resp, notional_usdc)
            .unwrap_or_else(|| self.estimated_fee_usdc(notional_usdc, order_type));
        let slippage_usdc = if has_actual_fill_price {
            0.0
        } else {
            self.estimated_slippage_usdc(notional_usdc, order_type)
        };
        (fee_usdc + slippage_usdc).max(0.0)
    }

    fn estimated_fee_usdc(&self, notional_usdc: f64, order_type: OrderType) -> f64 {
        let fee_bps = match order_type {
            OrderType::Limit => self.execution.maker_fee_bps,
            OrderType::Market => self.execution.taker_fee_bps,
        };
        notional_usdc * fee_bps / 10_000.0
    }

    fn estimated_slippage_usdc(&self, notional_usdc: f64, order_type: OrderType) -> f64 {
        let (fee_bps, slippage_bps) = match order_type {
            OrderType::Limit => (
                self.execution.maker_fee_bps,
                self.execution.maker_slippage_bps,
            ),
            OrderType::Market => (
                self.execution.taker_fee_bps,
                self.execution.taker_slippage_bps,
            ),
        };
        let _ = fee_bps;
        notional_usdc * slippage_bps / 10_000.0
    }
}

fn compute_take_profit_price(entry_price: f64, take_profit_pct: f64) -> f64 {
    // Binary contracts settle in [0, 1], so cap TP below 1.0.
    let raw_tp = entry_price * (1.0 + take_profit_pct);
    let min_step = (entry_price + 0.03).min(0.98);
    raw_tp.min(0.98).max(min_step)
}

fn classify_execution(resp: &Value, order_type: OrderType, dry_run: bool) -> ExecutionState {
    if dry_run {
        return ExecutionState::Filled;
    }

    let status = extract_status(resp);
    if status.contains("partial") {
        return ExecutionState::Partial;
    }
    if status.contains("reject")
        || status.contains("fail")
        || status.contains("cancel")
        || status.contains("expire")
        || status.contains("error")
    {
        return ExecutionState::Rejected;
    }
    if status.contains("fill")
        || status.contains("match")
        || status.contains("execut")
        || status.contains("done")
        || status.contains("complete")
    {
        return ExecutionState::Filled;
    }
    if status.contains("open")
        || status.contains("accept")
        || status.contains("pending")
        || status.contains("new")
        || status.contains("live")
    {
        return ExecutionState::Accepted;
    }

    // Fallback: market orders are usually immediate, limit orders may rest.
    if order_type == OrderType::Market {
        ExecutionState::Filled
    } else {
        ExecutionState::Accepted
    }
}

fn extract_status(resp: &Value) -> String {
    for key in [
        "status",
        "state",
        "order_status",
        "orderState",
        "result",
        "message",
    ] {
        if let Some(s) = resp.get(key).and_then(|v| v.as_str()) {
            return s.to_lowercase();
        }
    }
    String::new()
}

fn extract_fill_price(resp: &Value) -> Option<f64> {
    for key in [
        "avg_price",
        "average_price",
        "filled_price",
        "execution_price",
        "price",
    ] {
        if let Some(v) = resp.get(key).and_then(as_f64) {
            return Some(v);
        }
    }
    None
}

fn extract_filled_size(resp: &Value) -> Option<f64> {
    for key in [
        "filled_size",
        "filledSize",
        "size_filled",
        "executed_size",
        "executedSize",
        "sizeFilled",
    ] {
        if let Some(v) = find_numeric_by_key(resp, key) {
            return Some(v);
        }
    }
    None
}

fn extract_filled_notional(resp: &Value) -> Option<f64> {
    for key in [
        "filled_notional",
        "filledNotional",
        "executed_notional",
        "executedNotional",
        "quote_amount",
        "quoteAmount",
        "filled_usdc",
        "filledUSDC",
        "notional",
        "cost",
        "value",
    ] {
        if let Some(v) = find_numeric_by_key(resp, key) {
            return Some(v);
        }
    }
    None
}

fn extract_fee_usdc(resp: &Value, notional_usdc: f64) -> Option<f64> {
    for key in [
        "fee",
        "fees",
        "fee_paid",
        "feePaid",
        "paid_fee",
        "paidFee",
        "fee_amount",
        "feeAmount",
        "total_fee",
        "totalFee",
    ] {
        if let Some(v) = find_numeric_by_key(resp, key) {
            return Some(v.abs().min(notional_usdc));
        }
    }
    for key in [
        "feeRateBps",
        "fee_rate_bps",
        "takerFeeRateBps",
        "makerFeeRateBps",
        "feeBps",
    ] {
        if let Some(v) = find_numeric_by_key(resp, key) {
            let bps = v.max(0.0);
            return Some((notional_usdc * bps / 10_000.0).min(notional_usdc));
        }
    }
    None
}

fn find_numeric_by_key(value: &Value, key: &str) -> Option<f64> {
    match value {
        Value::Object(map) => {
            if let Some(v) = map.get(key).and_then(as_f64) {
                return Some(v);
            }
            for child in map.values() {
                if let Some(v) = find_numeric_by_key(child, key) {
                    return Some(v);
                }
            }
            None
        }
        Value::Array(arr) => {
            let mut sum = 0.0;
            let mut any = false;
            for item in arr {
                if let Some(v) = find_numeric_by_key(item, key) {
                    sum += v;
                    any = true;
                }
            }
            if any {
                Some(sum)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classify_execution_partial_detected() {
        let resp = json!({"status":"partially_filled","filled_size":"12.5"});
        assert_eq!(
            classify_execution(&resp, OrderType::Limit, false),
            ExecutionState::Partial
        );
    }

    #[test]
    fn classify_execution_rejected_detected() {
        let resp = json!({"status":"rejected"});
        assert_eq!(
            classify_execution(&resp, OrderType::Market, false),
            ExecutionState::Rejected
        );
    }

    #[test]
    fn take_profit_is_capped_for_binary_markets() {
        // Raw TP would be 1.35, but binary cap enforces <= 0.98.
        let tp = compute_take_profit_price(0.45, 2.0);
        assert!(tp <= 0.98 + 1e-9);
    }

    #[test]
    fn extract_fee_uses_explicit_amount_when_present() {
        let resp = json!({"status":"filled","fee":"0.42"});
        let fee = extract_fee_usdc(&resp, 100.0).unwrap();
        assert!((fee - 0.42).abs() < 1e-9);
    }

    #[test]
    fn extract_fee_falls_back_to_bps_when_amount_missing() {
        let resp = json!({"status":"filled","feeRateBps": 7.5});
        let fee = extract_fee_usdc(&resp, 200.0).unwrap();
        assert!((fee - 0.15).abs() < 1e-9);
    }

    #[test]
    fn extract_filled_notional_reads_nested_fill_values() {
        let resp = json!({"fills":[{"notional":"12.3"},{"notional": 7.7}]});
        let notional = extract_filled_notional(&resp).unwrap();
        assert!((notional - 20.0).abs() < 1e-9);
    }
}
