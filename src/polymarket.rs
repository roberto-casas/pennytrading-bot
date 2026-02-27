/// polymarket.rs – Async client for the Polymarket CLOB REST and WebSocket APIs.
///
/// Responsibilities:
///  - Discover BTC/ETH 5-min / 15-min markets via the Gamma Markets REST API
///  - Stream real-time order-book updates via the CLOB WebSocket feed
///  - Place / cancel limit and market orders (dry-run aware)
///  - Fetch account balance
use anyhow::{Context, Result};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, info, warn};

use crate::models::{Market, OrderBook, PriceLevel};

// ---------------------------------------------------------------------------
// Keyword helpers for market detection
// ---------------------------------------------------------------------------

fn detect_asset(question: &str) -> Option<&'static str> {
    let q = question.to_lowercase();
    if q.contains("btc") || q.contains("bitcoin") {
        Some("BTC")
    } else if q.contains("eth") || q.contains("ethereum") {
        Some("ETH")
    } else {
        None
    }
}

fn detect_resolution(question: &str) -> Option<u32> {
    let q = question.to_lowercase();
    for &(mins, kws) in &[
        (5u32, ["5-minute", "5 minute", "5min", "5 min"].as_slice()),
        (
            15u32,
            ["15-minute", "15 minute", "15min", "15 min"].as_slice(),
        ),
    ] {
        if kws.iter().any(|kw| q.contains(kw)) {
            return Some(mins);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// OrderBook update event sent through the channel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BookUpdate {
    pub token_id: String,
    pub market_id: String,
    pub book: OrderBook,
}

// ---------------------------------------------------------------------------
// PolymarketClient
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct PolymarketClient {
    pub rest_url: String,
    pub ws_url: String,
    pub gamma_url: String,
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub api_passphrase: Option<String>,
    pub dry_run: bool,
    http: Client,
    /// Shared order-book cache updated by the WebSocket task.
    pub order_books: Arc<RwLock<HashMap<String, OrderBook>>>,
}

impl PolymarketClient {
    pub fn new(
        rest_url: String,
        ws_url: String,
        gamma_url: String,
        api_key: Option<String>,
        api_secret: Option<String>,
        api_passphrase: Option<String>,
        dry_run: bool,
    ) -> Result<Self> {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .context("building HTTP client")?;
        Ok(Self {
            rest_url,
            ws_url,
            gamma_url,
            api_key,
            api_secret,
            api_passphrase,
            dry_run,
            http,
            order_books: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    // ------------------------------------------------------------------
    // Market discovery
    // ------------------------------------------------------------------

    /// Fetch active BTC/ETH 5-min / 15-min markets from the Gamma API.
    pub async fn fetch_btc_eth_markets(
        &self,
        assets: &[String],
        resolutions: &[u32],
        max_per_asset: usize,
    ) -> Result<Vec<Market>> {
        let url = format!("{}/markets", self.gamma_url);
        let resp = self
            .http
            .get(&url)
            .query(&[("active", "true"), ("closed", "false"), ("limit", "500")])
            .send()
            .await
            .context("fetching markets")?;

        let items: Value = resp.json().await.context("parsing market list")?;
        let items = match &items {
            Value::Array(a) => a.clone(),
            Value::Object(o) => o
                .get("markets")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default(),
            _ => vec![],
        };

        let mut markets = Vec::new();
        let mut counts: HashMap<&str, usize> = assets.iter().map(|a| (a.as_str(), 0)).collect();

        for item in &items {
            let question = item["question"].as_str().unwrap_or("");
            let asset = match detect_asset(question) {
                Some(a) if assets.iter().any(|x| x == a) => a,
                _ => continue,
            };
            let resolution = match detect_resolution(question) {
                Some(r) if resolutions.contains(&r) => r,
                _ => continue,
            };

            let count = counts.entry(asset).or_insert(0);
            if *count >= max_per_asset {
                continue;
            }

            let cid = item["conditionId"]
                .as_str()
                .or_else(|| item["condition_id"].as_str())
                .unwrap_or("");
            if cid.is_empty() {
                continue;
            }

            let (yes_tid, no_tid) = parse_token_ids(item);
            if yes_tid.is_empty() || no_tid.is_empty() {
                continue;
            }

            let end_date = item["endDateIso"]
                .as_str()
                .or_else(|| item["end_date_iso"].as_str())
                .unwrap_or("")
                .to_string();
            let resolved = item["resolved"]
                .as_bool()
                .or_else(|| item["isResolved"].as_bool())
                .or_else(|| item["is_resolved"].as_bool())
                .unwrap_or(false);
            let resolution_price = parse_resolution_price(item);

            markets.push(Market {
                condition_id: cid.to_string(),
                question: question.to_string(),
                yes_token_id: yes_tid,
                no_token_id: no_tid,
                asset: asset.to_string(),
                resolution_minutes: resolution,
                end_date_iso: end_date,
                active: item["active"].as_bool().unwrap_or(true),
                closed: item["closed"].as_bool().unwrap_or(false),
                resolved,
                resolution_price,
            });
            *count += 1;
        }

        info!("Discovered {} BTC/ETH markets", markets.len());
        Ok(markets)
    }

    // ------------------------------------------------------------------
    // REST order book
    // ------------------------------------------------------------------

    /// Fetch a single order-book snapshot via REST.
    pub async fn fetch_order_book(&self, token_id: &str, market_id: &str) -> Option<OrderBook> {
        let url = format!("{}/book", self.rest_url);
        let resp = self
            .http
            .get(&url)
            .query(&[("token_id", token_id)])
            .send()
            .await
            .ok()?;
        let data: Value = resp.json().await.ok()?;
        Some(parse_order_book(&data, market_id, token_id))
    }

    // ------------------------------------------------------------------
    // WebSocket feed
    // ------------------------------------------------------------------

    /// Start the WebSocket listener as a background tokio task.
    ///
    /// Returns a channel receiver that yields `BookUpdate` events, and a
    /// `JoinHandle` for the task.  The task reconnects automatically on
    /// disconnection.
    pub fn start_websocket(
        &self,
        markets: Vec<Market>,
    ) -> (mpsc::Receiver<BookUpdate>, tokio::task::JoinHandle<()>) {
        let (tx, rx) = mpsc::channel::<BookUpdate>(512);
        let ws_url = self.ws_url.clone();
        let api_key = self.api_key.clone().unwrap_or_default();
        let api_secret = self.api_secret.clone().unwrap_or_default();
        let api_passphrase = self.api_passphrase.clone().unwrap_or_default();
        let order_books = Arc::clone(&self.order_books);

        let handle = tokio::spawn(async move {
            let mut backoff = 1u64;
            loop {
                match ws_run(
                    &ws_url,
                    &markets,
                    &api_key,
                    &api_secret,
                    &api_passphrase,
                    &tx,
                    &order_books,
                )
                .await
                {
                    Ok(true) => {
                        // Connection was alive and received data; reset backoff
                        info!("WebSocket disconnected after receiving data – reconnecting");
                        backoff = 1;
                    }
                    Ok(false) => {
                        info!("WebSocket closed gracefully with no data.");
                        break;
                    }
                    Err(e) => {
                        warn!("WebSocket error: {e} – reconnecting in {backoff}s");
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(60);
            }
        });

        (rx, handle)
    }

    // ------------------------------------------------------------------
    // Balance
    // ------------------------------------------------------------------

    pub async fn get_balance(&self) -> f64 {
        if self.api_key.is_none() {
            return 0.0;
        }
        let url = format!("{}/balance-allowance", self.rest_url);
        match self
            .http
            .get(&url)
            .query(&[("asset_type", "USDC")])
            .send()
            .await
        {
            Ok(resp) => resp
                .json::<Value>()
                .await
                .ok()
                .and_then(|v| v["balance"].as_f64())
                .unwrap_or(0.0),
            Err(_) => 0.0,
        }
    }

    // ------------------------------------------------------------------
    // Order placement
    // ------------------------------------------------------------------

    /// Place a **limit** order. Returns the API response or a dry-run mock.
    pub async fn place_limit_order(
        &self,
        token_id: &str,
        side: &str,
        price: f64,
        size: f64,
        condition_id: &str,
    ) -> Result<Value> {
        let payload = json!({
            "tokenID": token_id,
            "side": side,
            "type": "LIMIT",
            "price": (price * 10_000.0).round() / 10_000.0,
            "size": (size * 100.0).round() / 100.0,
            "feeRateBps": 0,
            "condition_id": condition_id,
        });
        if self.dry_run {
            info!(
                "[DRY RUN] LIMIT {side} {:.8}… @ {price:.4} × {size:.2}",
                token_id
            );
            return Ok(json!({
                "status": "dry_run",
                "order_id": format!("dry_{}", Utc::now().timestamp_millis()),
            }));
        }
        info!("LIMIT {side} {:.8}… @ {price:.4} × {size:.2}", token_id);
        let resp = self
            .http
            .post(format!("{}/order", self.rest_url))
            .json(&payload)
            .send()
            .await?;
        Ok(resp.json().await?)
    }

    /// Place a **market** order for `amount_usdc` USDC worth.
    pub async fn place_market_order(
        &self,
        token_id: &str,
        side: &str,
        amount_usdc: f64,
        condition_id: &str,
    ) -> Result<Value> {
        let payload = json!({
            "tokenID": token_id,
            "side": side,
            "type": "MARKET",
            "amount": (amount_usdc * 100.0).round() / 100.0,
            "condition_id": condition_id,
        });
        if self.dry_run {
            info!("[DRY RUN] MARKET {side} {:.8}… ${amount_usdc:.2}", token_id);
            return Ok(json!({
                "status": "dry_run",
                "order_id": format!("dry_{}", Utc::now().timestamp_millis()),
            }));
        }
        info!("MARKET {side} {:.8}… ${amount_usdc:.2}", token_id);
        let resp = self
            .http
            .post(format!("{}/order", self.rest_url))
            .json(&payload)
            .send()
            .await?;
        Ok(resp.json().await?)
    }

    pub async fn cancel_order(&self, order_id: &str) -> Result<Value> {
        if self.dry_run {
            info!("[DRY RUN] Cancel {order_id}");
            return Ok(json!({"status": "dry_run"}));
        }
        let resp = self
            .http
            .post(format!("{}/cancel", self.rest_url))
            .json(&json!({"orderID": order_id}))
            .send()
            .await?;
        Ok(resp.json().await?)
    }
}

// ---------------------------------------------------------------------------
// WebSocket internals
// ---------------------------------------------------------------------------

/// Returns `Ok(true)` if at least one message was successfully received
/// (so caller can reset backoff), `Ok(false)` for graceful close with
/// no messages, and `Err` for connection failures.
async fn ws_run(
    ws_url: &str,
    markets: &[Market],
    api_key: &str,
    api_secret: &str,
    api_passphrase: &str,
    tx: &mpsc::Sender<BookUpdate>,
    order_books: &Arc<RwLock<HashMap<String, OrderBook>>>,
) -> Result<bool> {
    info!("Connecting to WebSocket: {ws_url}");

    let (mut ws, _) = connect_async(ws_url).await.context("WebSocket connect")?;

    // Build token_id → market_id map
    let mut token_to_market: HashMap<String, String> = HashMap::new();
    let mut all_token_ids: Vec<String> = Vec::new();
    for m in markets {
        token_to_market.insert(m.yes_token_id.clone(), m.condition_id.clone());
        token_to_market.insert(m.no_token_id.clone(), m.condition_id.clone());
        all_token_ids.push(m.yes_token_id.clone());
        all_token_ids.push(m.no_token_id.clone());
    }

    // Subscribe
    let sub = json!({
        "auth": {
            "apiKey": api_key,
            "secret": api_secret,
            "passphrase": api_passphrase,
        },
        "type": "subscribe",
        "channels": [{"name": "book", "token_ids": all_token_ids}],
    });
    ws.send(Message::Text(sub.to_string())).await?;
    info!("Subscribed to {} token order books", all_token_ids.len());

    let mut received_any = false;
    while let Some(msg) = ws.next().await {
        let msg = msg.context("WebSocket read error")?;
        match msg {
            Message::Text(text) => {
                received_any = true;
                handle_ws_text(&text, &token_to_market, tx, order_books);
            }
            Message::Ping(data) => {
                ws.send(Message::Pong(data)).await.ok();
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    Ok(received_any)
}

fn handle_ws_text(
    text: &str,
    token_to_market: &HashMap<String, String>,
    tx: &mpsc::Sender<BookUpdate>,
    order_books: &Arc<RwLock<HashMap<String, OrderBook>>>,
) {
    let Ok(msg) = serde_json::from_str::<Value>(text) else {
        return;
    };

    let event = msg.get("event_type").or_else(|| msg.get("type"));
    let is_book = matches!(
        event.and_then(|v| v.as_str()),
        Some("book" | "price_change" | "last_trade_price")
    );
    if !is_book {
        return;
    }

    let token_id = msg
        .get("asset_id")
        .or_else(|| msg.get("token_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if token_id.is_empty() {
        return;
    }

    let Some(market_id) = token_to_market.get(token_id).cloned() else {
        debug!("Ignoring update for unknown token: {token_id}");
        return;
    };

    let book = parse_order_book(&msg, &market_id, token_id);

    {
        if let Ok(mut ob) = order_books.write() {
            ob.insert(token_id.to_string(), book.clone());
        }
    }

    let update = BookUpdate {
        token_id: token_id.to_string(),
        market_id,
        book,
    };

    // Non-blocking send (drop if channel full)
    match tx.try_send(update) {
        Ok(_) => {}
        Err(TrySendError::Full(_)) => {
            debug!("Dropping WS book update (channel full) for token {token_id:.8}…");
        }
        Err(TrySendError::Closed(_)) => {}
    }

    debug!("Order book updated for token {token_id:.8}…");
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_order_book(data: &Value, market_id: &str, token_id: &str) -> OrderBook {
    let parse_levels = |key: &str| -> Vec<PriceLevel> {
        data[key]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|l| {
                let price = l["price"]
                    .as_f64()
                    .or_else(|| l["price"].as_str().and_then(|s| s.parse::<f64>().ok()))?;
                let size = l["size"]
                    .as_f64()
                    .or_else(|| l["size"].as_str().and_then(|s| s.parse::<f64>().ok()))?;
                Some(PriceLevel { price, size })
            })
            .collect()
    };

    let mut bids = parse_levels("bids");
    let mut asks = parse_levels("asks");
    bids.sort_by(|a, b| {
        b.price
            .partial_cmp(&a.price)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    asks.sort_by(|a, b| {
        a.price
            .partial_cmp(&b.price)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    OrderBook {
        market_id: market_id.to_string(),
        token_id: token_id.to_string(),
        timestamp: Utc::now(),
        bids,
        asks,
    }
}

fn parse_token_ids(item: &Value) -> (String, String) {
    // Try "tokens" array (array of objects with "token_id")
    if let Some(arr) = item["tokens"].as_array() {
        if arr.len() >= 2 {
            let yes = arr[0]["token_id"]
                .as_str()
                .or_else(|| arr[0].as_str())
                .unwrap_or("")
                .to_string();
            let no = arr[1]["token_id"]
                .as_str()
                .or_else(|| arr[1].as_str())
                .unwrap_or("")
                .to_string();
            return (yes, no);
        }
    }
    // Try "clobTokenIds" array
    if let Some(arr) = item["clobTokenIds"].as_array() {
        if arr.len() >= 2 {
            let yes = arr[0].as_str().unwrap_or("").to_string();
            let no = arr[1].as_str().unwrap_or("").to_string();
            return (yes, no);
        }
    }
    (String::new(), String::new())
}

fn parse_resolution_price(item: &Value) -> Option<f64> {
    for key in [
        "resolutionPrice",
        "resolution_price",
        "finalValue",
        "final_value",
    ] {
        if let Some(v) = item.get(key).and_then(value_as_f64) {
            return Some(v.clamp(0.0, 1.0));
        }
    }
    None
}

fn value_as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
}

// ---------------------------------------------------------------------------
// Serialisation helper for Gamma API response
// ---------------------------------------------------------------------------
#[allow(dead_code)]
#[derive(Deserialize)]
struct GammaMarket {
    #[serde(rename = "conditionId", default)]
    condition_id: String,
    question: String,
    #[serde(rename = "endDateIso", default)]
    end_date_iso: String,
    #[serde(default)]
    active: bool,
    #[serde(default)]
    closed: bool,
}
