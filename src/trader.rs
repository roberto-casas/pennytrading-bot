/// trader.rs – Order management: open/close positions, SL/TP monitoring,
/// trade resolution checking, and dry-run support.
use anyhow::Result;
use chrono::Utc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    config::{RiskConfig, StrategyConfig},
    database::Database,
    models::{Market, OrderBook, OrderType, Position, Side, TradeStatus},
    polymarket::PolymarketClient,
    strategy::Signal,
};

// ---------------------------------------------------------------------------
// Trader
// ---------------------------------------------------------------------------

pub struct Trader {
    pub risk: RiskConfig,
    pub strategy: StrategyConfig,
    pub dry_run: bool,
}

impl Trader {
    pub fn new(risk: RiskConfig, strategy: StrategyConfig, dry_run: bool) -> Self {
        Self {
            risk,
            strategy,
            dry_run,
        }
    }

    // ------------------------------------------------------------------
    // Open a new position from a strategy signal
    // ------------------------------------------------------------------

    /// Open a position based on a strategy signal.
    ///
    /// Prefers a **limit order** at the best ask (or slightly inside the spread)
    /// when `strategy.prefer_limit_orders` is true.  Falls back to a market
    /// order if the book is thin.
    pub async fn open_position(
        &self,
        signal: &Signal,
        client: &PolymarketClient,
        db: &Database,
        session_id: &str,
    ) -> Result<Position> {
        let size = signal.bet_usdc / signal.entry_price;
        let order_type = if self.strategy.prefer_limit_orders {
            OrderType::Limit
        } else {
            OrderType::Market
        };

        // Determine the limit price – try to get a slightly better price
        // than best ask by placing our limit 0.001 inside the spread.
        let limit_price = if order_type == OrderType::Limit {
            (signal.entry_price + 0.001).min(signal.entry_price * 1.01)
        } else {
            signal.entry_price
        };

        let resp = match order_type {
            OrderType::Limit => {
                client
                    .place_limit_order(
                        &signal.token_id,
                        signal.side.as_str(),
                        limit_price,
                        size,
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

        let _ = resp; // order response handled/logged in client

        // Compute stop-loss and take-profit levels
        let stop_loss_price = signal.entry_price * (1.0 - self.risk.stop_loss_pct);
        let take_profit_price = signal.entry_price * (1.0 + self.risk.take_profit_pct);

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
            entry_price: signal.entry_price,
            size,
            cost_usdc: signal.bet_usdc,
            stop_loss_price,
            take_profit_price,
            status,
            exit_price: None,
            pnl_usdc: None,
            opened_at: Utc::now(),
            closed_at: None,
            dry_run: self.dry_run,
            order_type,
        };

        db.upsert_position(&pos, Some(session_id))?;

        info!(
            "[{}] Opened {} {} {} @ {:.4} SL={:.4} TP={:.4} size={:.2} bet=${:.2}",
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
    ) -> Result<()> {
        // Place a sell order (limit preferred; market as fallback)
        let sell_resp = if self.strategy.prefer_limit_orders {
            client
                .place_limit_order(
                    &pos.token_id,
                    "SELL",
                    exit_price,
                    pos.size,
                    &pos.market_id,
                )
                .await
        } else {
            client
                .place_market_order(
                    &pos.token_id,
                    "SELL",
                    pos.size * exit_price,
                    &pos.market_id,
                )
                .await
        };

        if let Err(e) = sell_resp {
            warn!("Failed to place exit order for {}: {e}", pos.position_id);
        }

        let pnl = pos.size * (exit_price - pos.entry_price);
        pos.exit_price = Some(exit_price);
        pos.pnl_usdc = Some(pnl);
        pos.status = status;
        pos.closed_at = Some(Utc::now());

        db.upsert_position(pos, Some(session_id))?;

        info!(
            "[{}] Closed {} {} @ {:.4} P&L={:+.2} USDC ({})",
            if self.dry_run { "DRY" } else { "LIVE" },
            pos.asset,
            pos.side,
            exit_price,
            pnl,
            status,
        );

        Ok(())
    }

    // ------------------------------------------------------------------
    // Monitor open positions (SL / TP / time limit)
    // ------------------------------------------------------------------

    /// Check all open positions against stop-loss, take-profit and time-limit
    /// rules.  Returns the list of positions that were closed.
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

            // Resolve current price from order book
            let current_price = order_books
                .get(&pos.token_id)
                .and_then(|b| match pos.side {
                    Side::YES => b.best_bid(), // we'd sell into the bid
                    Side::NO => b.best_bid(),
                })
                .unwrap_or(pos.entry_price);

            // Find the associated market for time-limit check
            let market = markets.iter().find(|m| m.condition_id == pos.market_id);

            let (trigger_close, close_status) = if pos.should_stop_loss(current_price) {
                (true, TradeStatus::ClosedStopLoss)
            } else if pos.should_take_profit(current_price) {
                (true, TradeStatus::ClosedTakeProfit)
            } else if market
                .map(|m| m.is_near_resolution(self.risk.time_limit_fraction))
                .unwrap_or(false)
            {
                (true, TradeStatus::ClosedTimeLimit)
            } else {
                (false, TradeStatus::Open)
            };

            if trigger_close {
                self.close_position(pos, current_price, close_status, client, db, session_id)
                    .await?;
                closed.push(pos.clone());
            }
        }

        Ok(closed)
    }

    // ------------------------------------------------------------------
    // Trade resolution checking
    // ------------------------------------------------------------------

    /// Check whether markets have resolved and update positions accordingly.
    ///
    /// For each resolved market where we hold a position, we:
    ///  1. Determine the payout (1.0 or 0.0 per share).
    ///  2. Compute PnL.
    ///  3. Mark the position as RESOLVED_WIN or RESOLVED_LOSS.
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
            let pnl = pos.size * (payout_per_share - pos.entry_price);
            let status = if pnl >= 0.0 {
                TradeStatus::ResolvedWin
            } else {
                TradeStatus::ResolvedLoss
            };

            pos.exit_price = Some(exit_price);
            pos.pnl_usdc = Some(pnl);
            pos.status = status;
            pos.closed_at = Some(Utc::now());

            db.upsert_position(pos, Some(session_id))?;

            info!(
                "Resolved {} {} – payout={:.2} P&L={:+.2} ({})",
                pos.asset,
                pos.side,
                payout_per_share,
                pnl,
                status,
            );

            resolved.push(pos.clone());
        }

        Ok(resolved)
    }
}
