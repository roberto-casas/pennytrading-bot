/// events.rs – External feed polling for event/news regime activation.
use crate::config::AdaptiveConfig;
use std::collections::{hash_map::DefaultHasher, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};
use tracing::debug;

const MAX_PAYLOAD_CHARS: usize = 200_000;

#[derive(Debug, Clone, Default)]
pub struct ExternalEventSnapshot {
    pub global_active: bool,
    pub assets: HashSet<String>,
    /// Number of unique, non-duplicate hit sources in this poll.
    pub source_hits: usize,
    /// Number of duplicate hit payloads dropped by dedup logic in this poll.
    pub duplicate_hits: usize,
    /// Aggregate weighted score across unique hit sources in this poll.
    pub total_score: f64,
}

#[derive(Debug, Default)]
struct PayloadEvaluation {
    global: bool,
    assets: HashSet<String>,
    score: f64,
    duplicate: bool,
}

impl PayloadEvaluation {
    fn has_hit(&self) -> bool {
        self.global || !self.assets.is_empty()
    }
}

#[derive(Debug)]
struct SourceObservation {
    source_key: String,
    unique_hit: bool,
}

pub struct ExternalEventMonitor {
    client: reqwest::Client,
    urls: Vec<String>,
    global_keywords: Vec<String>,
    asset_keywords: HashMap<String, Vec<String>>,
    source_weights: HashMap<String, f64>,
    min_score: f64,
    cooldown: Duration,
    dedup_window: Duration,
    reliability_enabled: bool,
    reliability_alpha: f64,
    reliability_min_mult: f64,
    reliability_max_mult: f64,
    source_reliability: HashMap<String, f64>,
    global_until: Option<Instant>,
    asset_until: HashMap<String, Instant>,
    recent_fingerprints: HashMap<u64, Instant>,
}

impl ExternalEventMonitor {
    pub fn from_adaptive(cfg: &AdaptiveConfig) -> Option<Self> {
        if !cfg.external_feed_enabled || cfg.external_feed_urls.is_empty() {
            return None;
        }
        let timeout = Duration::from_secs_f64(cfg.external_feed_timeout_seconds.max(0.1));
        let client = reqwest::Client::builder().timeout(timeout).build().ok()?;
        Some(Self {
            client,
            urls: cfg.external_feed_urls.clone(),
            global_keywords: cfg
                .event_keywords
                .iter()
                .map(|k| k.to_lowercase())
                .collect(),
            asset_keywords: cfg
                .external_asset_keywords
                .iter()
                .map(|(asset, kws)| {
                    (
                        asset.to_uppercase(),
                        kws.iter().map(|k| k.to_lowercase()).collect::<Vec<_>>(),
                    )
                })
                .collect(),
            source_weights: cfg
                .external_source_weights
                .iter()
                .map(|(src, w)| (src.trim().to_lowercase(), *w))
                .collect(),
            min_score: cfg.external_event_min_score,
            cooldown: Duration::from_secs(cfg.external_event_cooldown_minutes * 60),
            dedup_window: Duration::from_secs(cfg.external_dedup_window_minutes * 60),
            reliability_enabled: cfg.external_reliability_enabled,
            reliability_alpha: cfg.external_reliability_alpha,
            reliability_min_mult: cfg.external_reliability_min_mult,
            reliability_max_mult: cfg.external_reliability_max_mult,
            source_reliability: HashMap::new(),
            global_until: None,
            asset_until: HashMap::new(),
            recent_fingerprints: HashMap::new(),
        })
    }

    pub async fn poll(&mut self) -> ExternalEventSnapshot {
        let now = Instant::now();
        self.prune_fingerprints(now);

        let mut source_hits = 0usize;
        let mut duplicate_hits = 0usize;
        let mut total_score = 0.0_f64;
        let mut global_hit = false;
        let mut asset_hits = HashSet::new();
        let mut observations: Vec<SourceObservation> = Vec::new();

        let urls = self.urls.clone();
        for url in urls {
            let source_key = source_key_from_url(&url);
            match self.fetch_payload(&url).await {
                Some(payload) => {
                    let eval = self.evaluate_payload(&url, &payload, now);
                    let unique_hit = eval.has_hit() && !eval.duplicate;
                    observations.push(SourceObservation {
                        source_key,
                        unique_hit,
                    });
                    if eval.duplicate {
                        duplicate_hits += 1;
                        continue;
                    }
                    if eval.has_hit() {
                        source_hits += 1;
                        total_score += eval.score;
                        global_hit |= eval.global;
                        asset_hits.extend(eval.assets.into_iter());
                    }
                }
                None => {
                    debug!("External feed fetch failed: {url}");
                    observations.push(SourceObservation {
                        source_key,
                        unique_hit: false,
                    });
                }
            }
        }

        self.update_source_reliability(&observations, source_hits);
        self.prune_source_reliability();
        self.activate_from_hits(now, global_hit, &asset_hits, total_score);
        self.snapshot_at(now, source_hits, duplicate_hits, total_score)
    }

    async fn fetch_payload(&self, url: &str) -> Option<String> {
        let resp = self.client.get(url).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let text = resp.text().await.ok()?;
        let payload: String = text.chars().take(MAX_PAYLOAD_CHARS).collect();
        Some(payload)
    }

    fn evaluate_payload(
        &mut self,
        source_url: &str,
        payload: &str,
        now: Instant,
    ) -> PayloadEvaluation {
        let text = payload.to_lowercase();
        let global_hit = contains_any(&text, &self.global_keywords);
        let mut asset_hits = HashSet::new();
        for (asset, keywords) in &self.asset_keywords {
            if contains_any(&text, keywords) {
                asset_hits.insert(asset.clone());
            }
        }
        if !global_hit && asset_hits.is_empty() {
            return PayloadEvaluation::default();
        }
        let fp = fingerprint(&text);
        if self.is_duplicate_fingerprint(fp, now) {
            return PayloadEvaluation {
                global: global_hit,
                assets: asset_hits,
                score: 0.0,
                duplicate: true,
            };
        }
        self.recent_fingerprints.insert(fp, now + self.dedup_window);
        PayloadEvaluation {
            global: global_hit,
            assets: asset_hits,
            score: self.weight_for_url(source_url),
            duplicate: false,
        }
    }

    fn activate_from_hits(
        &mut self,
        now: Instant,
        global_hit: bool,
        asset_hits: &HashSet<String>,
        total_score: f64,
    ) -> bool {
        if total_score < self.min_score {
            return false;
        }
        if global_hit {
            self.global_until = Some(now + self.cooldown);
        }
        for asset in asset_hits {
            self.asset_until.insert(asset.clone(), now + self.cooldown);
        }
        true
    }

    fn snapshot_at(
        &mut self,
        now: Instant,
        source_hits: usize,
        duplicate_hits: usize,
        total_score: f64,
    ) -> ExternalEventSnapshot {
        if let Some(until) = self.global_until {
            if until <= now {
                self.global_until = None;
            }
        }
        self.asset_until.retain(|_, until| *until > now);
        self.prune_fingerprints(now);
        let assets: HashSet<String> = self.asset_until.keys().cloned().collect();
        ExternalEventSnapshot {
            global_active: self.global_until.is_some(),
            assets,
            source_hits,
            duplicate_hits,
            total_score,
        }
    }

    fn is_duplicate_fingerprint(&self, fp: u64, now: Instant) -> bool {
        self.recent_fingerprints
            .get(&fp)
            .map(|until| *until > now)
            .unwrap_or(false)
    }

    fn prune_fingerprints(&mut self, now: Instant) {
        self.recent_fingerprints.retain(|_, until| *until > now);
    }

    fn update_source_reliability(
        &mut self,
        observations: &[SourceObservation],
        unique_hits: usize,
    ) {
        if !self.reliability_enabled || observations.is_empty() {
            return;
        }
        let consensus_hit = unique_hits >= 2;
        for obs in observations {
            let prev = self
                .source_reliability
                .get(&obs.source_key)
                .copied()
                .unwrap_or(1.0);
            let target = if obs.unique_hit {
                if consensus_hit {
                    self.reliability_max_mult
                } else {
                    self.reliability_min_mult
                }
            } else {
                1.0
            };
            let next = prev + self.reliability_alpha * (target - prev);
            self.source_reliability.insert(
                obs.source_key.clone(),
                next.clamp(self.reliability_min_mult, self.reliability_max_mult),
            );
        }
    }

    fn prune_source_reliability(&mut self) {
        let keep: HashSet<String> = self.urls.iter().map(|u| source_key_from_url(u)).collect();
        self.source_reliability.retain(|k, _| keep.contains(k));
    }

    fn weight_for_url(&self, url: &str) -> f64 {
        let base = self.base_weight_for_url(url);
        let source_key = source_key_from_url(url);
        let dynamic = self
            .source_reliability
            .get(&source_key)
            .copied()
            .unwrap_or(1.0);
        (base * dynamic).max(0.0)
    }

    fn base_weight_for_url(&self, url: &str) -> f64 {
        if self.source_weights.is_empty() {
            return 1.0;
        }
        let url_lc = url.trim().to_lowercase();
        if let Some(w) = self.source_weights.get(&url_lc) {
            return *w;
        }
        let source_key = source_key_from_url(url);
        if let Some(w) = self.source_weights.get(&source_key) {
            return *w;
        }
        if let Some(stripped) = source_key.strip_prefix("www.") {
            if let Some(w) = self.source_weights.get(stripped) {
                return *w;
            }
        }
        for (pattern, w) in &self.source_weights {
            if !pattern.is_empty() && url_lc.contains(pattern) {
                return *w;
            }
        }
        1.0
    }
}

fn contains_any(text: &str, keywords: &[String]) -> bool {
    keywords.iter().any(|k| !k.is_empty() && text.contains(k))
}

fn fingerprint(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

fn source_key_from_url(url: &str) -> String {
    if let Ok(parsed) = reqwest::Url::parse(url) {
        if let Some(host) = parsed.host_str() {
            return host.trim().to_lowercase();
        }
    }
    url.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monitor_for_test() -> ExternalEventMonitor {
        ExternalEventMonitor {
            client: reqwest::Client::new(),
            urls: vec![],
            global_keywords: vec!["fomc".to_string(), "cpi".to_string()],
            asset_keywords: HashMap::from([
                (
                    "BTC".to_string(),
                    vec!["bitcoin".to_string(), "btc".to_string()],
                ),
                (
                    "ETH".to_string(),
                    vec!["ethereum".to_string(), "eth".to_string()],
                ),
            ]),
            source_weights: HashMap::from([
                ("news.example.com".to_string(), 1.0),
                ("high-trust.example.com".to_string(), 1.5),
            ]),
            min_score: 1.0,
            cooldown: Duration::from_secs(60),
            dedup_window: Duration::from_secs(120),
            reliability_enabled: true,
            reliability_alpha: 1.0,
            reliability_min_mult: 0.5,
            reliability_max_mult: 1.5,
            source_reliability: HashMap::new(),
            global_until: None,
            asset_until: HashMap::new(),
            recent_fingerprints: HashMap::new(),
        }
    }

    #[test]
    fn ingest_triggers_global_and_asset_hits() {
        let mut m = monitor_for_test();
        let now = Instant::now();
        let eval = m.evaluate_payload(
            "https://news.example.com/feed",
            "Breaking: FOMC impacts Bitcoin outlook",
            now,
        );
        assert!(eval.has_hit());
        assert!(eval.global);
        assert!(eval.assets.contains("BTC"));
        assert!(!eval.duplicate);
        assert!(eval.score >= 1.0);
        let activated = m.activate_from_hits(now, eval.global, &eval.assets, eval.score);
        assert!(activated);
        let snap = m.snapshot_at(now, 1, 0, eval.score);
        assert!(snap.global_active);
        assert!(snap.assets.contains("BTC"));
    }

    #[test]
    fn ingest_detects_multiple_assets() {
        let mut m = monitor_for_test();
        let now = Instant::now();
        let eval = m.evaluate_payload(
            "https://news.example.com/feed",
            "BTC rallies while Ethereum slips",
            now,
        );
        assert!(eval.has_hit());
        assert!(eval.assets.contains("BTC"));
        assert!(eval.assets.contains("ETH"));
        let activated = m.activate_from_hits(now, eval.global, &eval.assets, eval.score);
        assert!(activated);
        let snap = m.snapshot_at(now, 1, 0, eval.score);
        assert!(snap.assets.contains("BTC"));
        assert!(snap.assets.contains("ETH"));
    }

    #[test]
    fn duplicate_payload_is_suppressed() {
        let mut m = monitor_for_test();
        let t0 = Instant::now();
        let first = m.evaluate_payload("https://news.example.com/feed", "cpi and bitcoin", t0);
        assert!(first.has_hit());
        assert!(!first.duplicate);

        let second = m.evaluate_payload("https://news.example.com/feed", "cpi and bitcoin", t0);
        assert!(second.has_hit());
        assert!(second.duplicate);
        assert_eq!(second.score, 0.0);
    }

    #[test]
    fn activation_requires_minimum_score() {
        let mut m = monitor_for_test();
        m.min_score = 1.2;
        let now = Instant::now();
        let eval = m.evaluate_payload("https://news.example.com/feed", "fomc and bitcoin", now);
        assert!(eval.has_hit());
        let activated = m.activate_from_hits(now, eval.global, &eval.assets, eval.score);
        assert!(!activated);
        let snap = m.snapshot_at(now, 1, 0, eval.score);
        assert!(!snap.global_active);
        assert!(snap.assets.is_empty());
    }

    #[test]
    fn cooldown_expiry_clears_state() {
        let mut m = monitor_for_test();
        let t0 = Instant::now();
        let eval = m.evaluate_payload("https://high-trust.example.com/feed", "cpi and bitcoin", t0);
        let _ = m.activate_from_hits(t0, eval.global, &eval.assets, eval.score);
        let after = t0 + Duration::from_secs(61);
        let snap = m.snapshot_at(after, 0, 0, 0.0);
        assert!(!snap.global_active);
        assert!(snap.assets.is_empty());
    }

    #[test]
    fn reliability_downweights_unconfirmed_single_source_hits() {
        let mut m = monitor_for_test();
        m.update_source_reliability(
            &[
                SourceObservation {
                    source_key: "news.example.com".to_string(),
                    unique_hit: true,
                },
                SourceObservation {
                    source_key: "high-trust.example.com".to_string(),
                    unique_hit: false,
                },
            ],
            1,
        );
        let w = m.weight_for_url("https://news.example.com/feed");
        assert!((w - 0.5).abs() < 1e-9);
    }

    #[test]
    fn reliability_upweights_confirmed_multi_source_hits() {
        let mut m = monitor_for_test();
        m.update_source_reliability(
            &[
                SourceObservation {
                    source_key: "news.example.com".to_string(),
                    unique_hit: true,
                },
                SourceObservation {
                    source_key: "high-trust.example.com".to_string(),
                    unique_hit: true,
                },
            ],
            2,
        );
        let w_news = m.weight_for_url("https://news.example.com/feed");
        let w_high = m.weight_for_url("https://high-trust.example.com/feed");
        assert!((w_news - 1.5).abs() < 1e-9);
        assert!((w_high - 2.25).abs() < 1e-9);
    }
}
