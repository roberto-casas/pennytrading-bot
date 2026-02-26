/// models.rs – Core data types shared across all bot modules.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    YES,
    NO,
}

impl Side {
    pub fn as_str(self) -> &'static str {
        match self {
            Side::YES => "YES",
            Side::NO => "NO",
        }
    }
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Side {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "YES" => Ok(Side::YES),
            "NO" => Ok(Side::NO),
            _ => Err(anyhow::anyhow!("Unknown side: {s}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Limit,
    Market,
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderType::Limit => write!(f, "LIMIT"),
            OrderType::Market => write!(f, "MARKET"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeStatus {
    Open,
    ClosedTakeProfit,
    ClosedStopLoss,
    ClosedTimeLimit,
    ResolvedWin,
    ResolvedLoss,
    DryRun,
}

impl TradeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            TradeStatus::Open => "OPEN",
            TradeStatus::ClosedTakeProfit => "CLOSED_TP",
            TradeStatus::ClosedStopLoss => "CLOSED_SL",
            TradeStatus::ClosedTimeLimit => "CLOSED_TIME",
            TradeStatus::ResolvedWin => "RESOLVED_WIN",
            TradeStatus::ResolvedLoss => "RESOLVED_LOSS",
            TradeStatus::DryRun => "DRY_RUN",
        }
    }

    pub fn is_open(self) -> bool {
        matches!(self, TradeStatus::Open | TradeStatus::DryRun)
    }
}

impl std::fmt::Display for TradeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for TradeStatus {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "OPEN" => Ok(TradeStatus::Open),
            "CLOSED_TP" => Ok(TradeStatus::ClosedTakeProfit),
            "CLOSED_SL" => Ok(TradeStatus::ClosedStopLoss),
            "CLOSED_TIME" => Ok(TradeStatus::ClosedTimeLimit),
            "RESOLVED_WIN" => Ok(TradeStatus::ResolvedWin),
            "RESOLVED_LOSS" => Ok(TradeStatus::ResolvedLoss),
            "DRY_RUN" => Ok(TradeStatus::DryRun),
            _ => Err(anyhow::anyhow!("Unknown status: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Order Book
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub market_id: String,
    pub token_id: String,
    pub timestamp: DateTime<Utc>,
    /// Sorted descending (best bid first).
    pub bids: Vec<PriceLevel>,
    /// Sorted ascending (best ask first).
    pub asks: Vec<PriceLevel>,
}

impl OrderBook {
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }

    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    pub fn spread_pct(&self) -> Option<f64> {
        let mid = self.mid_price()?;
        if mid == 0.0 {
            return None;
        }
        Some(self.spread()? / mid)
    }

    /// USDC value of bids within `depth` fraction of best bid.
    pub fn bid_liquidity(&self, depth: f64) -> f64 {
        let Some(best) = self.best_bid() else {
            return 0.0;
        };
        let cutoff = best * (1.0 - depth);
        self.bids
            .iter()
            .filter(|l| l.price >= cutoff)
            .map(|l| l.price * l.size)
            .sum()
    }

    /// USDC value of asks within `depth` fraction of best ask.
    pub fn ask_liquidity(&self, depth: f64) -> f64 {
        let Some(best) = self.best_ask() else {
            return 0.0;
        };
        let cutoff = best * (1.0 + depth);
        self.asks
            .iter()
            .filter(|l| l.price <= cutoff)
            .map(|l| l.price * l.size)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Market
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub condition_id: String,
    pub question: String,
    pub yes_token_id: String,
    pub no_token_id: String,
    /// "BTC" or "ETH"
    pub asset: String,
    /// Resolution time in minutes (5 or 15).
    pub resolution_minutes: u32,
    /// ISO-8601 string for when the market resolves.
    pub end_date_iso: String,
    pub active: bool,
    pub closed: bool,
    pub resolved: bool,
    /// 1.0 = YES won, 0.0 = NO won, None = unresolved.
    pub resolution_price: Option<f64>,
}

impl Market {
    pub fn end_datetime(&self) -> anyhow::Result<DateTime<Utc>> {
        let s = self.end_date_iso.replace('Z', "+00:00");
        let dt = DateTime::parse_from_rfc3339(&s)
            .map_err(|e| anyhow::anyhow!("Bad end_date_iso '{}': {e}", self.end_date_iso))?;
        Ok(dt.with_timezone(&Utc))
    }

    pub fn seconds_to_resolution(&self) -> f64 {
        self.end_datetime()
            .ok()
            .map(|end| {
                let delta = end - Utc::now();
                delta.num_milliseconds().max(0) as f64 / 1000.0
            })
            .unwrap_or(0.0)
    }

    /// Returns true if more than `fraction` of the total market lifetime has elapsed.
    pub fn is_near_resolution(&self, fraction: f64) -> bool {
        let total_secs = self.resolution_minutes as f64 * 60.0;
        let remaining = self.seconds_to_resolution();
        remaining < total_secs * (1.0 - fraction)
    }

    /// Short human-readable label, e.g. "BTC-5m".
    pub fn label(&self) -> String {
        format!("{}-{}m", self.asset, self.resolution_minutes)
    }
}

// ---------------------------------------------------------------------------
// Position / Trade
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub position_id: String,
    pub market_id: String,
    pub token_id: String,
    pub side: Side,
    pub asset: String,
    pub entry_price: f64,
    /// Number of shares held.
    pub size: f64,
    /// Total USDC spent to open the position.
    pub cost_usdc: f64,
    pub stop_loss_price: f64,
    pub take_profit_price: f64,
    /// Highest price observed since entry (for trailing stop).
    pub high_water_mark: f64,
    pub status: TradeStatus,
    pub exit_price: Option<f64>,
    pub pnl_usdc: Option<f64>,
    pub opened_at: DateTime<Utc>,
    pub closed_at: Option<DateTime<Utc>>,
    pub dry_run: bool,
    pub order_type: OrderType,
}

impl Position {
    pub fn unrealised_pnl(&self, current_price: f64) -> f64 {
        self.size * (current_price - self.entry_price)
    }

    pub fn should_stop_loss(&self, current_price: f64) -> bool {
        current_price <= self.stop_loss_price
    }

    pub fn should_take_profit(&self, current_price: f64) -> bool {
        current_price >= self.take_profit_price
    }

    /// Update the high-water mark and return the trailing stop price.
    /// Trailing stop activates once position is profitable (price > entry).
    /// Trail distance = `trail_pct` of the high-water mark.
    pub fn update_trailing_stop(&mut self, current_price: f64, trail_pct: f64) -> Option<f64> {
        if current_price > self.high_water_mark {
            self.high_water_mark = current_price;
        }
        // Only activate trailing stop once we're in profit
        if self.high_water_mark > self.entry_price {
            let trail_price = self.high_water_mark * (1.0 - trail_pct);
            // Trailing stop must be above entry to lock in profit
            if trail_price > self.entry_price {
                return Some(trail_price);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Bot session (one run)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotSession {
    pub session_id: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub dry_run: bool,
    pub trades_opened: u32,
    pub trades_closed: u32,
    pub total_pnl_usdc: f64,
    pub config_snapshot: String,
}

// ---------------------------------------------------------------------------
// Shared application state (for dashboard + async tasks)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct AppState {
    pub session_id: String,
    pub started_at: Option<DateTime<Utc>>,
    pub dry_run: bool,
    pub balance_usdc: f64,
    pub markets: Vec<Market>,
    /// token_id → OrderBook
    pub order_books: std::collections::HashMap<String, OrderBook>,
    pub open_positions: Vec<Position>,
    pub recent_trades: VecDeque<Position>,
    pub logs: VecDeque<String>,
    pub total_pnl: f64,
    pub trades_opened: u32,
    pub trades_closed: u32,
    pub ws_connected: bool,
}

impl AppState {
    pub fn add_log(&mut self, msg: impl Into<String>) {
        let entry = format!("[{}] {}", Utc::now().format("%H:%M:%S"), msg.into());
        self.logs.push_front(entry);
        // Keep a reasonable history
        while self.logs.len() > 200 {
            self.logs.pop_back();
        }
    }
}
