/// dashboard.rs – ratatui live terminal dashboard.
///
/// Layout (5 panels):
///  ┌─ Header ──────────────────────────────────────────────────────────┐
///  │ PennyTrading Bot │ Session │ Balance │ P&L │ [DRY RUN] │ WS status │
///  ├─ Open Positions ──────────────┬─ Market Data ─────────────────────┤
///  │ table of open trades          │ order books per token             │
///  ├─ Recent Trades ───────────────┼─ Logs ────────────────────────────┤
///  │ last N closed trades          │ timestamped log lines             │
///  └───────────────────────────────┴───────────────────────────────────┘
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table},
    Frame, Terminal,
};
use std::io::{self, Stdout};

use crate::models::{AppState, Side, TradeStatus};

pub type CrossTerm = Terminal<CrosstermBackend<Stdout>>;

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

pub fn setup_terminal() -> anyhow::Result<CrossTerm> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend)?)
}

pub fn teardown_terminal(terminal: &mut CrossTerm) -> anyhow::Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Key event handling
// ---------------------------------------------------------------------------

/// Returns `true` when the user requests quit (q or Ctrl-C).
pub fn handle_event(event: &Event) -> bool {
    matches!(
        event,
        Event::Key(k)
            if k.code == KeyCode::Char('q')
            || k.code == KeyCode::Char('Q')
            || (k.code == KeyCode::Char('c')
                && k.modifiers.contains(crossterm::event::KeyModifiers::CONTROL))
    )
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

pub fn render(frame: &mut Frame, state: &AppState) {
    let area = frame.size();

    // Outer layout: header | body
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    render_header(frame, outer[0], state);

    // Body: left | right
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(outer[1]);

    // Left column: positions | trades
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(body[0]);

    // Right column: market data | logs
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(body[1]);

    render_positions(frame, left[0], state);
    render_trades(frame, left[1], state);
    render_markets(frame, right[0], state);
    render_logs(frame, right[1], state);
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

fn render_header(frame: &mut Frame, area: Rect, state: &AppState) {
    let mode = if state.dry_run {
        Span::styled(
            " [DRY RUN] ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            " [LIVE] ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )
    };

    let ws_status = if state.ws_connected {
        Span::styled("WS●", Style::default().fg(Color::Green))
    } else {
        Span::styled("WS○", Style::default().fg(Color::Red))
    };

    let pnl_color = if state.total_pnl >= 0.0 {
        Color::Green
    } else {
        Color::Red
    };

    let uptime = state
        .started_at
        .map(|t| {
            let secs = (chrono::Utc::now() - t).num_seconds();
            format!("{}h{:02}m", secs / 3600, (secs % 3600) / 60)
        })
        .unwrap_or_else(|| "—".into());

    let line = Line::from(vec![
        Span::styled(
            "  🪙 PennyTrading Bot  │ ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("Up: {}  │ ", uptime)),
        Span::raw(format!("Balance: ${:.2}  │ ", state.balance_usdc)),
        Span::styled(
            format!("P&L: {:+.2}  │ ", state.total_pnl),
            Style::default().fg(pnl_color).add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "Trades: {}/{} open  │ ",
            state.open_positions.len(),
            state.trades_opened,
        )),
        mode,
        Span::raw("  "),
        ws_status,
        Span::styled("  [q] quit", Style::default().fg(Color::DarkGray)),
    ]);

    let header = Paragraph::new(line).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" PennyTrading Bot "),
    );
    frame.render_widget(header, area);
}

// ---------------------------------------------------------------------------
// Open positions table
// ---------------------------------------------------------------------------

fn render_positions(frame: &mut Frame, area: Rect, state: &AppState) {
    let header_cells = ["Market", "Side", "Entry", "Size", "SL", "TP", "Unreal. P&L"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().add_modifier(Modifier::BOLD)));
    let header = Row::new(header_cells)
        .style(Style::default().bg(Color::DarkGray))
        .height(1);

    let rows: Vec<Row> = state
        .open_positions
        .iter()
        .map(|pos| {
            // Try to get current price from order books
            let current_price = state
                .order_books
                .get(&pos.token_id)
                .and_then(|b| b.best_bid())
                .unwrap_or(pos.entry_price);
            let upnl = pos.unrealised_pnl(current_price);
            let upnl_color = if upnl >= 0.0 {
                Color::Green
            } else {
                Color::Red
            };

            let label = format!(
                "{}-{}m",
                pos.asset,
                state
                    .markets
                    .iter()
                    .find(|m| m.condition_id == pos.market_id)
                    .map(|m| m.resolution_minutes)
                    .unwrap_or(0)
            );
            Row::new(vec![
                Cell::from(label),
                Cell::from(pos.side.as_str()).style(Style::default().fg(match pos.side {
                    Side::YES => Color::Green,
                    Side::NO => Color::Red,
                })),
                Cell::from(format!("{:.4}", pos.entry_price)),
                Cell::from(format!("{:.2}", pos.size)),
                Cell::from(format!("{:.4}", pos.stop_loss_price))
                    .style(Style::default().fg(Color::Red)),
                Cell::from(format!("{:.4}", pos.take_profit_price))
                    .style(Style::default().fg(Color::Green)),
                Cell::from(format!("{:+.2}", upnl))
                    .style(Style::default().fg(upnl_color).add_modifier(Modifier::BOLD)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Min(10),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Open Positions ({}) ", state.open_positions.len())),
    )
    .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    frame.render_widget(table, area);
}

// ---------------------------------------------------------------------------
// Recent trades table
// ---------------------------------------------------------------------------

fn render_trades(frame: &mut Frame, area: Rect, state: &AppState) {
    let header_cells = ["Time", "Market", "Side", "Entry", "Exit", "P&L", "Status"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().add_modifier(Modifier::BOLD)));
    let header = Row::new(header_cells)
        .style(Style::default().bg(Color::DarkGray))
        .height(1);

    let rows: Vec<Row> = state
        .recent_trades
        .iter()
        .take(20)
        .map(|pos| {
            let pnl = pos.pnl_usdc.unwrap_or(0.0);
            let pnl_color = if pnl >= 0.0 { Color::Green } else { Color::Red };
            let time_str = pos
                .closed_at
                .map(|t| t.format("%H:%M:%S").to_string())
                .unwrap_or_else(|| "—".into());
            let label = format!(
                "{}-{}m",
                pos.asset,
                state
                    .markets
                    .iter()
                    .find(|m| m.condition_id == pos.market_id)
                    .map(|m| m.resolution_minutes)
                    .unwrap_or(0)
            );
            let status_color = match pos.status {
                TradeStatus::ClosedTakeProfit | TradeStatus::ResolvedWin => Color::Green,
                TradeStatus::ClosedStopLoss | TradeStatus::ResolvedLoss => Color::Red,
                TradeStatus::ClosedTimeLimit => Color::Yellow,
                _ => Color::Gray,
            };
            Row::new(vec![
                Cell::from(time_str),
                Cell::from(label),
                Cell::from(pos.side.as_str()),
                Cell::from(format!("{:.4}", pos.entry_price)),
                Cell::from(
                    pos.exit_price
                        .map(|p| format!("{:.4}", p))
                        .unwrap_or_else(|| "—".into()),
                ),
                Cell::from(format!("{:+.2}", pnl))
                    .style(Style::default().fg(pnl_color).add_modifier(Modifier::BOLD)),
                Cell::from(pos.status.as_str()).style(Style::default().fg(status_color)),
            ])
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(5),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Min(10),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Recent Trades ({} total) ", state.trades_closed)),
    );

    frame.render_widget(table, area);
}

// ---------------------------------------------------------------------------
// Market data panel
// ---------------------------------------------------------------------------

fn render_markets(frame: &mut Frame, area: Rect, state: &AppState) {
    let header_cells = ["Market", "Token", "Bid", "Ask", "Mid", "Liq $"]
        .iter()
        .map(|h| Cell::from(*h).style(Style::default().add_modifier(Modifier::BOLD)));
    let header = Row::new(header_cells)
        .style(Style::default().bg(Color::DarkGray))
        .height(1);

    let mut rows: Vec<Row> = Vec::new();
    for market in &state.markets {
        for token_id in [&market.yes_token_id, &market.no_token_id] {
            let side_label = if token_id == &market.yes_token_id {
                "YES"
            } else {
                "NO"
            };
            let label = format!("{} {}", market.label(), side_label);
            if let Some(book) = state.order_books.get(token_id) {
                let liq = book.ask_liquidity(0.05);
                rows.push(Row::new(vec![
                    Cell::from(label),
                    Cell::from(format!("{:.8}…", &token_id[..token_id.len().min(8)])),
                    Cell::from(
                        book.best_bid()
                            .map(|p| format!("{:.4}", p))
                            .unwrap_or_else(|| "—".into()),
                    )
                    .style(Style::default().fg(Color::Green)),
                    Cell::from(
                        book.best_ask()
                            .map(|p| format!("{:.4}", p))
                            .unwrap_or_else(|| "—".into()),
                    )
                    .style(Style::default().fg(Color::Red)),
                    Cell::from(
                        book.mid_price()
                            .map(|p| format!("{:.4}", p))
                            .unwrap_or_else(|| "—".into()),
                    ),
                    Cell::from(format!("${:.0}", liq)),
                ]));
            } else {
                rows.push(Row::new(vec![
                    Cell::from(label),
                    Cell::from("…"),
                    Cell::from("—"),
                    Cell::from("—"),
                    Cell::from("—"),
                    Cell::from("—"),
                ]));
            }
        }
    }

    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Length(10),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Min(8),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Market Data ({} markets) ", state.markets.len())),
    );

    frame.render_widget(table, area);
}

// ---------------------------------------------------------------------------
// Logs panel
// ---------------------------------------------------------------------------

fn render_logs(frame: &mut Frame, area: Rect, state: &AppState) {
    let items: Vec<ListItem> = state
        .logs
        .iter()
        .take(area.height as usize)
        .map(|line| {
            let color = if line.contains("ERROR") || line.contains("CLOSED_SL") {
                Color::Red
            } else if line.contains("DRY") {
                Color::Yellow
            } else if line.contains("Opened") || line.contains("CLOSED_TP") || line.contains("WIN")
            {
                Color::Green
            } else {
                Color::Gray
            };
            ListItem::new(Line::from(Span::styled(
                line.clone(),
                Style::default().fg(color),
            )))
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Logs "))
        .style(Style::default().fg(Color::White));

    frame.render_widget(list, area);
}
