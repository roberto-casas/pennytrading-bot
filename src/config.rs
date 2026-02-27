/// config.rs – Load settings from config.yaml + environment variables.
///
/// Environment variables always override YAML values.
/// API credentials are read exclusively from the environment / .env file.
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Sub-configs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct BotConfig {
    /// When true no real orders are placed.
    pub dry_run: bool,
    pub log_level: String,
    /// Path to the SQLite database file.
    pub db_path: String,
    /// How often (seconds) to poll positions for SL/TP and run strategy.
    pub poll_interval_seconds: f64,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            dry_run: true,
            log_level: "INFO".into(),
            db_path: "pennytrading.db".into(),
            poll_interval_seconds: 5.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct MarketsConfig {
    /// Assets to monitor (e.g. ["BTC", "ETH"]).
    pub assets: Vec<String>,
    /// Resolution times in minutes (e.g. [5, 15]).
    pub resolutions: Vec<u32>,
    /// Maximum simultaneous markets to watch per asset.
    pub max_markets_per_asset: usize,
}

impl Default for MarketsConfig {
    fn default() -> Self {
        Self {
            assets: vec!["BTC".into(), "ETH".into()],
            resolutions: vec![5, 15],
            max_markets_per_asset: 3,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct StrategyConfig {
    /// Minimum edge (probability difference) required to open a position.
    pub min_edge: f64,
    /// Minimum USDC liquidity on the best side of the order book.
    pub min_liquidity_usdc: f64,
    /// Maximum acceptable bid-ask spread as a fraction of mid price.
    pub max_spread_pct: f64,
    /// Prefer limit orders over market orders when possible.
    pub prefer_limit_orders: bool,
    /// Do not trade contracts priced below this (avoid extreme penny positions).
    pub price_min: f64,
    /// Do not trade contracts priced above this.
    pub price_max: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            min_edge: 0.06,
            min_liquidity_usdc: 1500.0,
            max_spread_pct: 0.04,
            prefer_limit_orders: false,
            price_min: 0.02,
            price_max: 0.98,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct KellyConfig {
    /// Fractional Kelly multiplier (0.25 = quarter Kelly for variance reduction).
    pub fraction: f64,
    /// Hard cap: maximum fraction of bankroll in a single position.
    pub max_position_pct: f64,
    /// Minimum bet in USDC; skip trades below this.
    pub min_bet_usdc: f64,
    /// Maximum bet in USDC per position.
    pub max_bet_usdc: f64,
}

impl Default for KellyConfig {
    fn default() -> Self {
        Self {
            fraction: 0.25,
            max_position_pct: 0.10,
            min_bet_usdc: 5.0,
            max_bet_usdc: 500.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct RiskConfig {
    /// Close position if price drops this fraction below entry (e.g. 0.50 = -50%).
    pub stop_loss_pct: f64,
    /// Baseline TP fraction over entry (final TP is binary-capped below 1.0).
    pub take_profit_pct: f64,
    /// Maximum number of simultaneously open positions.
    pub max_open_positions: usize,
    /// Close position if more than this fraction of total time has elapsed.
    pub time_limit_fraction: f64,
    /// Stop opening new positions once session drawdown exceeds this absolute amount.
    /// Set to 0 to disable.
    pub max_session_drawdown_usdc: f64,
    /// Stop opening new positions once session drawdown exceeds this fraction of
    /// peak equity. Set to 0 to disable.
    pub max_session_drawdown_pct: f64,
    /// Skip opening new positions for assets whose adaptive sigma is above this
    /// annualized volatility threshold. Set to 0 to disable.
    pub max_asset_sigma: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            stop_loss_pct: 0.50,
            take_profit_pct: 0.60,
            max_open_positions: 6,
            time_limit_fraction: 0.75,
            max_session_drawdown_usdc: 75.0,
            max_session_drawdown_pct: 0.12,
            max_asset_sigma: 2.20,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ClobConfig {
    pub rest_url: String,
    pub ws_url: String,
    pub gamma_url: String,
}

impl Default for ClobConfig {
    fn default() -> Self {
        Self {
            rest_url: "https://clob.polymarket.com".into(),
            ws_url: "wss://ws-subscriptions-clob.polymarket.com/ws/".into(),
            gamma_url: "https://gamma-api.polymarket.com".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ExecutionConfig {
    /// Estimated maker fee in basis points.
    pub maker_fee_bps: f64,
    /// Estimated taker fee in basis points.
    pub taker_fee_bps: f64,
    /// Estimated maker slippage in basis points.
    pub maker_slippage_bps: f64,
    /// Estimated taker slippage in basis points.
    pub taker_slippage_bps: f64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            maker_fee_bps: 0.0,
            taker_fee_bps: 2.0,
            maker_slippage_bps: 1.0,
            taker_slippage_bps: 8.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct AdaptiveConfig {
    /// Enable runtime parameter re-tuning from calibration feedback.
    pub enabled: bool,
    /// How often to evaluate calibration and retune parameters.
    pub calibration_interval_cycles: u64,
    /// Number of most-recent resolved trades used for calibration stats.
    pub sample_window: usize,
    /// Minimum number of resolved trades required before tuning can trigger.
    pub min_resolved_trades: usize,
    /// Minimum Brier-score gap required to trigger tuning.
    pub brier_margin: f64,
    /// Step size applied to strategy.min_edge during tuning.
    pub edge_step: f64,
    /// Lower bound for strategy.min_edge when loosening.
    pub min_edge_floor: f64,
    /// Upper bound for strategy.min_edge when tightening.
    pub max_edge_cap: f64,
    /// Step size applied to kelly.fraction during tuning.
    pub kelly_step: f64,
    /// Lower bound for kelly.fraction when tightening.
    pub min_kelly_fraction: f64,
    /// Upper bound for kelly.fraction when loosening.
    pub max_kelly_fraction: f64,
    /// Enable separate calibration/tuning tracks for volatility regimes.
    pub regime_enabled: bool,
    /// Sigma threshold separating `low_vol` and `high_vol` regimes.
    pub regime_sigma_threshold: f64,
    /// Spread-percentage threshold classifying a market as `wide_spread`.
    pub regime_wide_spread_pct: f64,
    /// Liquidity multiplier over `strategy.min_liquidity_usdc` below which a
    /// market is tagged `thin_liq` (after passing base liquidity gate).
    pub regime_thin_liquidity_mult: f64,
    /// Enable event/news regime tagging.
    pub event_regime_enabled: bool,
    /// Force all markets into event regime while true.
    pub global_event_active: bool,
    /// Assets currently under event/news risk (e.g. ["BTC"]).
    pub event_assets: Vec<String>,
    /// Optional keywords to detect event-sensitive markets from question text.
    pub event_keywords: Vec<String>,
    /// Enable external event/news feed polling.
    pub external_feed_enabled: bool,
    /// External event/news URLs (RSS or JSON endpoints).
    pub external_feed_urls: Vec<String>,
    /// Seconds between polling external feeds.
    pub external_feed_poll_seconds: f64,
    /// Per-request timeout for external feed fetches.
    pub external_feed_timeout_seconds: f64,
    /// Minutes to keep event regime active after a feed hit.
    pub external_event_cooldown_minutes: u64,
    /// Asset-specific keyword map for feed payload matching.
    pub external_asset_keywords: HashMap<String, Vec<String>>,
    /// Minimum aggregate weighted score required to activate event regime.
    pub external_event_min_score: f64,
    /// Minutes to retain payload fingerprints for cross-feed deduplication.
    pub external_dedup_window_minutes: u64,
    /// Optional source weights keyed by exact URL or hostname.
    pub external_source_weights: HashMap<String, f64>,
    /// Enable adaptive source reliability multipliers.
    pub external_reliability_enabled: bool,
    /// EMA smoothing factor for source reliability updates.
    pub external_reliability_alpha: f64,
    /// Lower bound for dynamic source reliability multiplier.
    pub external_reliability_min_mult: f64,
    /// Upper bound for dynamic source reliability multiplier.
    pub external_reliability_max_mult: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            calibration_interval_cycles: 24,
            sample_window: 200,
            min_resolved_trades: 30,
            brier_margin: 0.01,
            edge_step: 0.005,
            min_edge_floor: 0.04,
            max_edge_cap: 0.15,
            kelly_step: 0.02,
            min_kelly_fraction: 0.10,
            max_kelly_fraction: 0.35,
            regime_enabled: true,
            regime_sigma_threshold: 1.40,
            regime_wide_spread_pct: 0.025,
            regime_thin_liquidity_mult: 1.50,
            event_regime_enabled: false,
            global_event_active: false,
            event_assets: Vec::new(),
            event_keywords: vec![
                "fomc".to_string(),
                "cpi".to_string(),
                "sec".to_string(),
                "etf".to_string(),
            ],
            external_feed_enabled: false,
            external_feed_urls: vec![
                "https://www.coindesk.com/arc/outboundfeeds/rss/".to_string(),
                "https://www.theblock.co/rss.xml".to_string(),
            ],
            external_feed_poll_seconds: 90.0,
            external_feed_timeout_seconds: 8.0,
            external_event_cooldown_minutes: 30,
            external_asset_keywords: HashMap::from([
                (
                    "BTC".to_string(),
                    vec![
                        "bitcoin".to_string(),
                        "btc".to_string(),
                        "spot btc".to_string(),
                        "bitcoin etf".to_string(),
                    ],
                ),
                (
                    "ETH".to_string(),
                    vec![
                        "ethereum".to_string(),
                        "eth".to_string(),
                        "ether".to_string(),
                        "spot eth".to_string(),
                    ],
                ),
            ]),
            external_event_min_score: 1.0,
            external_dedup_window_minutes: 20,
            external_source_weights: HashMap::from([
                ("coindesk.com".to_string(), 1.0),
                ("theblock.co".to_string(), 1.0),
            ]),
            external_reliability_enabled: true,
            external_reliability_alpha: 0.20,
            external_reliability_min_mult: 0.50,
            external_reliability_max_mult: 1.50,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct DashboardConfig {
    /// Dashboard refresh rate in seconds.
    pub refresh_rate: f64,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self { refresh_rate: 1.0 }
    }
}

// ---------------------------------------------------------------------------
// Top-level settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct Settings {
    pub bot: BotConfig,
    pub markets: MarketsConfig,
    pub strategy: StrategyConfig,
    pub kelly: KellyConfig,
    pub risk: RiskConfig,
    pub clob: ClobConfig,
    pub execution: ExecutionConfig,
    pub adaptive: AdaptiveConfig,
    pub dashboard: DashboardConfig,

    // API credentials – populated from env, not from YAML.
    #[serde(skip)]
    pub poly_api_key: Option<String>,
    #[serde(skip)]
    pub poly_api_secret: Option<String>,
    #[serde(skip)]
    pub poly_api_passphrase: Option<String>,
    #[serde(skip)]
    pub poly_private_key: Option<String>,
    #[serde(skip)]
    pub poly_address: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            bot: BotConfig::default(),
            markets: MarketsConfig::default(),
            strategy: StrategyConfig::default(),
            kelly: KellyConfig::default(),
            risk: RiskConfig::default(),
            clob: ClobConfig::default(),
            execution: ExecutionConfig::default(),
            adaptive: AdaptiveConfig::default(),
            dashboard: DashboardConfig::default(),
            poly_api_key: None,
            poly_api_secret: None,
            poly_api_passphrase: None,
            poly_private_key: None,
            poly_address: None,
        }
    }
}

impl Settings {
    /// Load settings from *config_path* YAML file, then overlay env vars.
    pub fn load(config_path: &str, dry_run_override: Option<bool>) -> Result<Self> {
        // Try to load .env file (ignore error if absent)
        let _ = dotenvy::dotenv();

        let mut settings = if std::path::Path::new(config_path).exists() {
            let yaml = std::fs::read_to_string(config_path).context("reading config file")?;
            serde_yaml::from_str::<Settings>(&yaml).context("parsing config YAML")?
        } else {
            Settings::default()
        };

        // Credentials from environment
        settings.poly_api_key = std::env::var("POLY_API_KEY").ok();
        settings.poly_api_secret = std::env::var("POLY_API_SECRET").ok();
        settings.poly_api_passphrase = std::env::var("POLY_API_PASSPHRASE").ok();
        settings.poly_private_key = std::env::var("POLY_PRIVATE_KEY").ok();
        settings.poly_address = std::env::var("POLY_ADDRESS").ok();

        // Allow DRY_RUN env var to override YAML
        if let Ok(val) = std::env::var("DRY_RUN") {
            settings.bot.dry_run = matches!(val.to_lowercase().as_str(), "1" | "true" | "yes");
        }
        if let Some(dr) = dry_run_override {
            settings.bot.dry_run = dr;
        }

        settings.validate()?;
        Ok(settings)
    }

    pub fn has_credentials(&self) -> bool {
        self.poly_api_key.is_some()
            && self.poly_api_secret.is_some()
            && self.poly_api_passphrase.is_some()
    }

    fn validate(&self) -> Result<()> {
        validate_positive("bot.poll_interval_seconds", self.bot.poll_interval_seconds)?;
        validate_positive("dashboard.refresh_rate", self.dashboard.refresh_rate)?;

        if self.markets.max_markets_per_asset == 0 {
            bail!("markets.max_markets_per_asset must be > 0");
        }
        if self.strategy.price_min < 0.0 || self.strategy.price_min >= 1.0 {
            bail!("strategy.price_min must be in [0, 1)");
        }
        if self.strategy.price_max <= 0.0 || self.strategy.price_max > 1.0 {
            bail!("strategy.price_max must be in (0, 1]");
        }
        if self.strategy.price_min >= self.strategy.price_max {
            bail!("strategy.price_min must be < strategy.price_max");
        }
        if self.kelly.max_position_pct <= 0.0 || self.kelly.max_position_pct > 1.0 {
            bail!("kelly.max_position_pct must be in (0, 1]");
        }
        if self.adaptive.calibration_interval_cycles == 0 {
            bail!("adaptive.calibration_interval_cycles must be > 0");
        }
        if self.adaptive.sample_window == 0 {
            bail!("adaptive.sample_window must be > 0");
        }
        if self.adaptive.min_resolved_trades == 0 {
            bail!("adaptive.min_resolved_trades must be > 0");
        }
        if !self.adaptive.brier_margin.is_finite()
            || self.adaptive.brier_margin <= 0.0
            || self.adaptive.brier_margin >= 1.0
        {
            bail!("adaptive.brier_margin must be a finite number in (0, 1)");
        }
        if !self.adaptive.edge_step.is_finite()
            || self.adaptive.edge_step <= 0.0
            || self.adaptive.edge_step > 0.2
        {
            bail!("adaptive.edge_step must be a finite number in (0, 0.2]");
        }
        if !self.adaptive.min_edge_floor.is_finite()
            || self.adaptive.min_edge_floor < 0.0
            || self.adaptive.min_edge_floor >= 1.0
        {
            bail!("adaptive.min_edge_floor must be in [0, 1)");
        }
        if !self.adaptive.max_edge_cap.is_finite()
            || self.adaptive.max_edge_cap <= self.adaptive.min_edge_floor
            || self.adaptive.max_edge_cap > 1.0
        {
            bail!("adaptive.max_edge_cap must be in (min_edge_floor, 1]");
        }
        if !self.adaptive.kelly_step.is_finite()
            || self.adaptive.kelly_step <= 0.0
            || self.adaptive.kelly_step > 1.0
        {
            bail!("adaptive.kelly_step must be a finite number in (0, 1]");
        }
        if !self.adaptive.min_kelly_fraction.is_finite()
            || self.adaptive.min_kelly_fraction <= 0.0
            || self.adaptive.min_kelly_fraction > 1.0
        {
            bail!("adaptive.min_kelly_fraction must be in (0, 1]");
        }
        if !self.adaptive.max_kelly_fraction.is_finite()
            || self.adaptive.max_kelly_fraction < self.adaptive.min_kelly_fraction
            || self.adaptive.max_kelly_fraction > 1.0
        {
            bail!("adaptive.max_kelly_fraction must be in [min_kelly_fraction, 1]");
        }
        if !self.adaptive.regime_sigma_threshold.is_finite()
            || self.adaptive.regime_sigma_threshold <= 0.0
            || self.adaptive.regime_sigma_threshold > 10.0
        {
            bail!("adaptive.regime_sigma_threshold must be a finite number in (0, 10]");
        }
        if !self.adaptive.regime_wide_spread_pct.is_finite()
            || self.adaptive.regime_wide_spread_pct <= 0.0
            || self.adaptive.regime_wide_spread_pct > 1.0
        {
            bail!("adaptive.regime_wide_spread_pct must be a finite number in (0, 1]");
        }
        if !self.adaptive.regime_thin_liquidity_mult.is_finite()
            || self.adaptive.regime_thin_liquidity_mult < 1.0
            || self.adaptive.regime_thin_liquidity_mult > 20.0
        {
            bail!("adaptive.regime_thin_liquidity_mult must be in [1, 20]");
        }
        if self
            .adaptive
            .event_assets
            .iter()
            .any(|a| a.trim().is_empty() || a.len() > 16)
        {
            bail!("adaptive.event_assets entries must be non-empty and <= 16 chars");
        }
        if self
            .adaptive
            .event_keywords
            .iter()
            .any(|k| k.trim().is_empty() || k.len() > 64)
        {
            bail!("adaptive.event_keywords entries must be non-empty and <= 64 chars");
        }
        if self.adaptive.external_feed_enabled && self.adaptive.external_feed_urls.is_empty() {
            bail!("adaptive.external_feed_urls must be non-empty when external_feed_enabled=true");
        }
        if !self.adaptive.external_feed_poll_seconds.is_finite()
            || self.adaptive.external_feed_poll_seconds <= 0.0
        {
            bail!("adaptive.external_feed_poll_seconds must be a finite number > 0");
        }
        if !self.adaptive.external_feed_timeout_seconds.is_finite()
            || self.adaptive.external_feed_timeout_seconds <= 0.0
        {
            bail!("adaptive.external_feed_timeout_seconds must be a finite number > 0");
        }
        if self.adaptive.external_event_cooldown_minutes == 0
            || self.adaptive.external_event_cooldown_minutes > 24 * 60
        {
            bail!("adaptive.external_event_cooldown_minutes must be in [1, 1440]");
        }
        if !self.adaptive.external_event_min_score.is_finite()
            || self.adaptive.external_event_min_score <= 0.0
            || self.adaptive.external_event_min_score > 100.0
        {
            bail!("adaptive.external_event_min_score must be a finite number in (0, 100]");
        }
        if self.adaptive.external_dedup_window_minutes == 0
            || self.adaptive.external_dedup_window_minutes > 24 * 60
        {
            bail!("adaptive.external_dedup_window_minutes must be in [1, 1440]");
        }
        if self
            .adaptive
            .external_feed_urls
            .iter()
            .any(|u| u.trim().is_empty() || u.len() > 2048)
        {
            bail!("adaptive.external_feed_urls entries must be non-empty and <= 2048 chars");
        }
        if self
            .adaptive
            .external_asset_keywords
            .iter()
            .any(|(asset, kws)| {
                asset.trim().is_empty()
                    || asset.len() > 16
                    || kws.is_empty()
                    || kws.iter().any(|k| k.trim().is_empty() || k.len() > 64)
            })
        {
            bail!("adaptive.external_asset_keywords must map non-empty asset keys to non-empty keyword lists");
        }
        if self
            .adaptive
            .external_source_weights
            .iter()
            .any(|(src, w)| {
                src.trim().is_empty() || src.len() > 256 || !w.is_finite() || *w <= 0.0 || *w > 10.0
            })
        {
            bail!("adaptive.external_source_weights must map non-empty source keys to finite weights in (0, 10]");
        }
        if !self.adaptive.external_reliability_alpha.is_finite()
            || self.adaptive.external_reliability_alpha <= 0.0
            || self.adaptive.external_reliability_alpha > 1.0
        {
            bail!("adaptive.external_reliability_alpha must be a finite number in (0, 1]");
        }
        if !self.adaptive.external_reliability_min_mult.is_finite()
            || self.adaptive.external_reliability_min_mult <= 0.0
            || self.adaptive.external_reliability_min_mult > 10.0
        {
            bail!("adaptive.external_reliability_min_mult must be a finite number in (0, 10]");
        }
        if !self.adaptive.external_reliability_max_mult.is_finite()
            || self.adaptive.external_reliability_max_mult
                < self.adaptive.external_reliability_min_mult
            || self.adaptive.external_reliability_max_mult > 10.0
        {
            bail!(
                "adaptive.external_reliability_max_mult must be in [external_reliability_min_mult, 10]"
            );
        }
        if self.risk.max_session_drawdown_usdc < 0.0
            || !self.risk.max_session_drawdown_usdc.is_finite()
        {
            bail!("risk.max_session_drawdown_usdc must be a finite number >= 0");
        }
        if self.risk.max_session_drawdown_pct < 0.0 || self.risk.max_session_drawdown_pct >= 1.0 {
            bail!("risk.max_session_drawdown_pct must be in [0, 1)");
        }
        if self.risk.max_asset_sigma < 0.0 || !self.risk.max_asset_sigma.is_finite() {
            bail!("risk.max_asset_sigma must be a finite number >= 0");
        }
        for (name, v) in [
            ("execution.maker_fee_bps", self.execution.maker_fee_bps),
            ("execution.taker_fee_bps", self.execution.taker_fee_bps),
            (
                "execution.maker_slippage_bps",
                self.execution.maker_slippage_bps,
            ),
            (
                "execution.taker_slippage_bps",
                self.execution.taker_slippage_bps,
            ),
        ] {
            if !v.is_finite() || v < 0.0 || v > 500.0 {
                bail!("{name} must be a finite number in [0, 500]");
            }
        }

        Ok(())
    }
}

fn validate_positive(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        bail!("{name} must be a finite number > 0");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_zero_poll_interval() {
        let mut settings = Settings::default();
        settings.bot.poll_interval_seconds = 0.0;
        assert!(settings.validate().is_err());
    }

    #[test]
    fn validate_rejects_invalid_price_range() {
        let mut settings = Settings::default();
        settings.strategy.price_min = 0.90;
        settings.strategy.price_max = 0.40;
        assert!(settings.validate().is_err());
    }

    #[test]
    fn validate_rejects_invalid_adaptive_interval() {
        let mut settings = Settings::default();
        settings.adaptive.calibration_interval_cycles = 0;
        assert!(settings.validate().is_err());
    }

    #[test]
    fn validate_rejects_invalid_regime_liquidity_multiplier() {
        let mut settings = Settings::default();
        settings.adaptive.regime_thin_liquidity_mult = 0.8;
        assert!(settings.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty_event_keyword() {
        let mut settings = Settings::default();
        settings.adaptive.event_keywords = vec!["".to_string()];
        assert!(settings.validate().is_err());
    }

    #[test]
    fn validate_rejects_enabled_external_feeds_without_urls() {
        let mut settings = Settings::default();
        settings.adaptive.external_feed_enabled = true;
        settings.adaptive.external_feed_urls.clear();
        assert!(settings.validate().is_err());
    }

    #[test]
    fn validate_rejects_invalid_external_source_weight() {
        let mut settings = Settings::default();
        settings
            .adaptive
            .external_source_weights
            .insert("coindesk.com".to_string(), 0.0);
        assert!(settings.validate().is_err());
    }

    #[test]
    fn validate_rejects_invalid_reliability_bounds() {
        let mut settings = Settings::default();
        settings.adaptive.external_reliability_min_mult = 1.2;
        settings.adaptive.external_reliability_max_mult = 1.1;
        assert!(settings.validate().is_err());
    }
}
