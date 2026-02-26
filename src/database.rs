/// database.rs – SQLite persistence layer using rusqlite.
///
/// All data (markets, positions, sessions, order-book snapshots) is stored in
/// a single SQLite file with WAL journaling so data survives bot restarts.
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::sync::Mutex;

use crate::models::{BotSession, Market, OrderBook, Position, Side, TradeStatus};

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS bot_sessions (
    session_id      TEXT PRIMARY KEY,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    dry_run         INTEGER NOT NULL DEFAULT 1,
    trades_opened   INTEGER NOT NULL DEFAULT 0,
    trades_closed   INTEGER NOT NULL DEFAULT 0,
    total_pnl_usdc  REAL    NOT NULL DEFAULT 0.0,
    config_snapshot TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS markets (
    condition_id        TEXT PRIMARY KEY,
    question            TEXT NOT NULL,
    yes_token_id        TEXT NOT NULL,
    no_token_id         TEXT NOT NULL,
    asset               TEXT NOT NULL,
    resolution_minutes  INTEGER NOT NULL,
    end_date_iso        TEXT NOT NULL,
    active              INTEGER NOT NULL DEFAULT 1,
    closed              INTEGER NOT NULL DEFAULT 0,
    resolved            INTEGER NOT NULL DEFAULT 0,
    resolution_price    REAL
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id    TEXT NOT NULL REFERENCES markets(condition_id),
    token_id        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    best_bid        REAL,
    best_ask        REAL,
    mid_price       REAL,
    spread          REAL,
    bid_liquidity   REAL,
    ask_liquidity   REAL
);

CREATE TABLE IF NOT EXISTS positions (
    position_id         TEXT PRIMARY KEY,
    session_id          TEXT REFERENCES bot_sessions(session_id),
    market_id           TEXT NOT NULL,
    token_id            TEXT NOT NULL,
    side                TEXT NOT NULL,
    asset               TEXT NOT NULL,
    entry_price         REAL NOT NULL,
    size                REAL NOT NULL,
    cost_usdc           REAL NOT NULL,
    stop_loss_price     REAL NOT NULL,
    take_profit_price   REAL NOT NULL,
    high_water_mark     REAL NOT NULL DEFAULT 0.0,
    status              TEXT NOT NULL DEFAULT 'OPEN',
    exit_price          REAL,
    pnl_usdc            REAL,
    opened_at           TEXT NOT NULL,
    closed_at           TEXT,
    dry_run             INTEGER NOT NULL DEFAULT 1,
    order_type          TEXT NOT NULL DEFAULT 'LIMIT'
);

CREATE INDEX IF NOT EXISTS idx_positions_status  ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_market  ON positions(market_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_cid     ON market_snapshots(condition_id);
";

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

pub struct Database {
    conn: Mutex<Connection>,
}

impl Database {
    /// Open (or create) the SQLite database at *path* and apply the schema.
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path).context("opening SQLite database")?;
        conn.execute_batch(SCHEMA).context("applying schema")?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn with_conn<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let conn = self.conn.lock().expect("database mutex poisoned");
        f(&conn)
    }

    // ------------------------------------------------------------------
    // Sessions
    // ------------------------------------------------------------------

    pub fn upsert_session(&self, s: &BotSession) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO bot_sessions
                    (session_id, started_at, ended_at, dry_run,
                     trades_opened, trades_closed, total_pnl_usdc, config_snapshot)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)
                 ON CONFLICT(session_id) DO UPDATE SET
                    ended_at       = excluded.ended_at,
                    trades_opened  = excluded.trades_opened,
                    trades_closed  = excluded.trades_closed,
                    total_pnl_usdc = excluded.total_pnl_usdc,
                    config_snapshot= excluded.config_snapshot",
                params![
                    s.session_id,
                    s.started_at.to_rfc3339(),
                    s.ended_at.map(|t| t.to_rfc3339()),
                    s.dry_run as i32,
                    s.trades_opened,
                    s.trades_closed,
                    s.total_pnl_usdc,
                    s.config_snapshot,
                ],
            )?;
            Ok(())
        })
    }

    pub fn get_sessions(&self, limit: usize) -> Result<Vec<BotSession>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT * FROM bot_sessions ORDER BY started_at DESC LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit as i64], |row| {
                Self::row_to_session(row)
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(anyhow::Error::from)
        })
    }

    fn row_to_session(row: &rusqlite::Row<'_>) -> Result<BotSession, rusqlite::Error> {
        Ok(BotSession {
            session_id: row.get("session_id")?,
            started_at: parse_dt(row.get::<_, String>("started_at")?),
            ended_at: row
                .get::<_, Option<String>>("ended_at")?
                .map(parse_dt),
            dry_run: row.get::<_, i32>("dry_run")? != 0,
            trades_opened: row.get("trades_opened")?,
            trades_closed: row.get("trades_closed")?,
            total_pnl_usdc: row.get("total_pnl_usdc")?,
            config_snapshot: row.get("config_snapshot")?,
        })
    }

    // ------------------------------------------------------------------
    // Markets
    // ------------------------------------------------------------------

    pub fn upsert_market(&self, m: &Market) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO markets
                    (condition_id, question, yes_token_id, no_token_id,
                     asset, resolution_minutes, end_date_iso,
                     active, closed, resolved, resolution_price)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
                 ON CONFLICT(condition_id) DO UPDATE SET
                    active           = excluded.active,
                    closed           = excluded.closed,
                    resolved         = excluded.resolved,
                    resolution_price = excluded.resolution_price",
                params![
                    m.condition_id,
                    m.question,
                    m.yes_token_id,
                    m.no_token_id,
                    m.asset,
                    m.resolution_minutes,
                    m.end_date_iso,
                    m.active as i32,
                    m.closed as i32,
                    m.resolved as i32,
                    m.resolution_price,
                ],
            )?;
            Ok(())
        })
    }

    pub fn get_active_markets(&self) -> Result<Vec<Market>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT * FROM markets WHERE active=1 AND closed=0 AND resolved=0",
            )?;
            let rows = stmt.query_map([], |row| Self::row_to_market(row))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(anyhow::Error::from)
        })
    }

    pub fn get_all_markets(&self) -> Result<Vec<Market>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT * FROM markets")?;
            let rows = stmt.query_map([], |row| Self::row_to_market(row))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(anyhow::Error::from)
        })
    }

    fn row_to_market(row: &rusqlite::Row<'_>) -> Result<Market, rusqlite::Error> {
        Ok(Market {
            condition_id: row.get("condition_id")?,
            question: row.get("question")?,
            yes_token_id: row.get("yes_token_id")?,
            no_token_id: row.get("no_token_id")?,
            asset: row.get("asset")?,
            resolution_minutes: row.get("resolution_minutes")?,
            end_date_iso: row.get("end_date_iso")?,
            active: row.get::<_, i32>("active")? != 0,
            closed: row.get::<_, i32>("closed")? != 0,
            resolved: row.get::<_, i32>("resolved")? != 0,
            resolution_price: row.get("resolution_price")?,
        })
    }

    // ------------------------------------------------------------------
    // Order-book snapshots
    // ------------------------------------------------------------------

    pub fn save_orderbook_snapshot(&self, book: &OrderBook) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO market_snapshots
                    (condition_id, token_id, timestamp,
                     best_bid, best_ask, mid_price, spread,
                     bid_liquidity, ask_liquidity)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                params![
                    book.market_id,
                    book.token_id,
                    book.timestamp.to_rfc3339(),
                    book.best_bid(),
                    book.best_ask(),
                    book.mid_price(),
                    book.spread(),
                    book.bid_liquidity(0.05),
                    book.ask_liquidity(0.05),
                ],
            )?;
            Ok(())
        })
    }

    // ------------------------------------------------------------------
    // Positions
    // ------------------------------------------------------------------

    pub fn upsert_position(&self, pos: &Position, session_id: Option<&str>) -> Result<()> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO positions
                    (position_id, session_id, market_id, token_id, side, asset,
                     entry_price, size, cost_usdc,
                     stop_loss_price, take_profit_price, high_water_mark,
                     status, exit_price, pnl_usdc,
                     opened_at, closed_at, dry_run, order_type)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)
                 ON CONFLICT(position_id) DO UPDATE SET
                    status          = excluded.status,
                    exit_price      = excluded.exit_price,
                    pnl_usdc        = excluded.pnl_usdc,
                    closed_at       = excluded.closed_at,
                    high_water_mark = excluded.high_water_mark",
                params![
                    pos.position_id,
                    session_id,
                    pos.market_id,
                    pos.token_id,
                    pos.side.as_str(),
                    pos.asset,
                    pos.entry_price,
                    pos.size,
                    pos.cost_usdc,
                    pos.stop_loss_price,
                    pos.take_profit_price,
                    pos.high_water_mark,
                    pos.status.as_str(),
                    pos.exit_price,
                    pos.pnl_usdc,
                    pos.opened_at.to_rfc3339(),
                    pos.closed_at.map(|t| t.to_rfc3339()),
                    pos.dry_run as i32,
                    pos.order_type.to_string(),
                ],
            )?;
            Ok(())
        })
    }

    pub fn get_open_positions(&self) -> Result<Vec<Position>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT * FROM positions WHERE status='OPEN' OR status='DRY_RUN'",
            )?;
            let rows = stmt.query_map([], |row| Self::row_to_position(row))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(anyhow::Error::from)
        })
    }

    pub fn get_all_positions(&self, limit: usize) -> Result<Vec<Position>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT * FROM positions ORDER BY opened_at DESC LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit as i64], |row| {
                Self::row_to_position(row)
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .map_err(anyhow::Error::from)
        })
    }

    pub fn total_pnl(&self) -> Result<f64> {
        self.with_conn(|conn| {
            let val: f64 = conn.query_row(
                "SELECT COALESCE(SUM(pnl_usdc),0.0) FROM positions WHERE pnl_usdc IS NOT NULL",
                [],
                |row| row.get(0),
            )?;
            Ok(val)
        })
    }

    fn row_to_position(row: &rusqlite::Row<'_>) -> Result<Position, rusqlite::Error> {
        use crate::models::OrderType;
        let side_str: String = row.get("side")?;
        let status_str: String = row.get("status")?;
        let order_type_str: String = row.get::<_, Option<String>>("order_type")?
            .unwrap_or_else(|| "LIMIT".into());
        let entry_price: f64 = row.get("entry_price")?;
        let high_water_mark: f64 = row.get::<_, Option<f64>>("high_water_mark")?
            .unwrap_or(entry_price);
        Ok(Position {
            position_id: row.get("position_id")?,
            market_id: row.get("market_id")?,
            token_id: row.get("token_id")?,
            side: side_str.parse().unwrap_or(Side::YES),
            asset: row.get("asset")?,
            entry_price,
            size: row.get("size")?,
            cost_usdc: row.get("cost_usdc")?,
            stop_loss_price: row.get("stop_loss_price")?,
            take_profit_price: row.get("take_profit_price")?,
            high_water_mark,
            status: status_str.parse().unwrap_or(TradeStatus::Open),
            exit_price: row.get("exit_price")?,
            pnl_usdc: row.get("pnl_usdc")?,
            opened_at: parse_dt(row.get::<_, String>("opened_at")?),
            closed_at: row
                .get::<_, Option<String>>("closed_at")?
                .map(parse_dt),
            dry_run: row.get::<_, i32>("dry_run")? != 0,
            order_type: match order_type_str.as_str() {
                "MARKET" => OrderType::Market,
                _ => OrderType::Limit,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_dt(s: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}
