/// config.rs – Load settings from config.yaml + environment variables.
///
/// Environment variables always override YAML values.
/// API credentials are read exclusively from the environment / .env file.
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

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
            min_edge: 0.04,
            min_liquidity_usdc: 500.0,
            max_spread_pct: 0.08,
            prefer_limit_orders: true,
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
    /// Close position if price rises this multiple above entry (e.g. 2.0 = +200%).
    pub take_profit_pct: f64,
    /// Maximum number of simultaneously open positions.
    pub max_open_positions: usize,
    /// Close position if more than this fraction of total time has elapsed.
    pub time_limit_fraction: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            stop_loss_pct: 0.50,
            take_profit_pct: 2.00,
            max_open_positions: 10,
            time_limit_fraction: 0.75,
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
            let yaml =
                std::fs::read_to_string(config_path).context("reading config file")?;
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

        Ok(settings)
    }

    pub fn has_credentials(&self) -> bool {
        self.poly_api_key.is_some()
            && self.poly_api_secret.is_some()
            && self.poly_api_passphrase.is_some()
    }
}
