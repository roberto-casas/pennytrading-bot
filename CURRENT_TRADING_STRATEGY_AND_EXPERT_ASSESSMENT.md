# Current Trading Strategy Dump + Expert Assessment

Generated on: 2026-02-27  
Scope: Current implementation in this repository (`src/strategy.rs`, `src/trader.rs`, `src/main.rs`, `src/kelly.rs`, `config.yaml`).

## 1. Strategy Dump (Current Implementation)

### Market Universe
- Venue: Polymarket CLOB.
- Instruments: BTC and ETH binary markets.
- Horizons: 5-minute and 15-minute markets.
- Max tracked markets: `max_markets_per_asset = 3` (per asset).

### Data + Runtime
- Primary feed: WebSocket order book stream (`book`, `price_change`, `last_trade_price` events).
- Poll cycle: `bot.poll_interval_seconds = 5`.
- Dashboard refresh: `dashboard.refresh_rate = 1s`.
- WS resilience:
  - reconnect with exponential backoff (up to 60s),
  - stale detection in main loop,
  - automatic feed restart when receiver channel closes.
- In-memory order books are pruned to tracked token IDs.

### Signal Generation Pipeline
For each market on each cycle:
1. Enforce position caps:
   - global max open positions (`risk.max_open_positions = 10`),
   - per-asset cap (`MAX_POSITIONS_PER_ASSET = 3`).
2. Skip markets too near resolution:
   - time filter based on `risk.time_limit_fraction = 0.75`.
3. Liquidity + spread gate:
   - min ask-side liquidity: `strategy.min_liquidity_usdc = 500`,
   - max spread %: `strategy.max_spread_pct = 0.08`.
4. Price model:
   - infer spot from highest-liquidity market midpoint + threshold heuristic,
   - estimate adaptive sigma from rolling token midpoint history (up to 200 points),
   - compute `P(YES)` from a log-normal model (`model_probability`),
   - adjust by order-book imbalance (`IMBALANCE_WEIGHT = 0.02`).
5. Entry pricing:
   - YES entry uses best ask.
   - NO entry derived as `1 - YES best bid`.
   - spread-adjusted effective entry for edge/Kelly checks.
6. Dynamic price floor:
   - `effective_price_min = price_min + elapsed_fraction * 0.10` (capped at `0.30`),
   - with base bounds `price_min = 0.02`, `price_max = 0.98`.
7. Open trade only if:
   - modeled probability edge exceeds `strategy.min_edge = 0.04`,
   - Kelly result is actionable.

### Position Sizing
- Kelly framework:
  - full Kelly for binary contract,
  - fractional multiplier `kelly.fraction = 0.25`,
  - cap per position `kelly.max_position_pct = 0.10`,
  - hard bet bounds: `min_bet_usdc = 5`, `max_bet_usdc = 500`.
- Available bankroll is balance minus cost of currently open positions.

### Execution
- Preferred entry type: limit (`strategy.prefer_limit_orders = true`).
- Exit order type follows same preference (limit sell when configured).
- One new position max per cycle (loop breaks after first open signal).

### Risk Controls and Exits
- Stop loss: `stop_loss_pct = 0.50` (price falls 50% from entry).
- Take profit: `take_profit_pct = 2.00` (price rises 200% from entry).
- Trailing stop: 20% below high-water mark after profitable move.
- Time-limit exit near resolution.
- Resolution settlement logic closes positions when market resolution price is available.

### Persistence
- SQLite persistence for sessions, markets, and positions.
- Open positions are restored at startup.

---

## 2. Expert-Agent Assessment (Trading + Financial Markets Panel)

Interpretation note: I assessed your implemented trading system (the bot) as "you".

### Agent A: Quant Research Lead
- Score: **6.5/10**
- Strengths:
  - clear, deterministic rules with explicit gates,
  - adaptive volatility and microstructure signal integration,
  - dynamic price floor is sensible for short-dated binaries.
- Risks:
  - spot inference from market thresholds is heuristic and can bias probability estimates,
  - no explicit calibration/validation loop on forecast quality (Brier score, calibration curves),
  - no regime switching for volatility shocks/news events.
- Recommendation:
  - add rolling model diagnostics and calibration adjustment before increasing capital.

### Agent B: Market Microstructure Specialist
- Score: **6.0/10**
- Strengths:
  - spread and liquidity filters prevent worst fills,
  - uses YES book consistently for derived NO pricing.
- Risks:
  - queue position/fill probability for limit orders is not modeled,
  - no slippage model in expected value checks,
  - dropped WS updates under pressure can reduce signal freshness.
- Recommendation:
  - add fill-rate and slippage telemetry; include expected adverse selection in edge threshold.

### Agent C: Risk Manager (Institutional)
- Score: **7.0/10**
- Strengths:
  - multiple independent risk brakes (global cap, per-asset cap, SL/TP, time limit, Kelly cap),
  - bankroll-aware sizing.
- Risks:
  - stop/take-profit percentages are wide for very short-horizon markets,
  - no portfolio-level drawdown circuit breaker,
  - correlated event risk across BTC/ETH snapshots is partially controlled but not stress-tested.
- Recommendation:
  - add daily/session max drawdown kill-switch and volatility-event throttle.

### Agent D: Portfolio Manager
- Score: **6.0/10**
- Strengths:
  - disciplined entry criteria and bounded exposure.
- Risks:
  - strategy currently opens at most one new position per cycle, potentially missing concurrent high-EV opportunities,
  - static per-asset cap may underuse capital in favorable regimes.
- Recommendation:
  - prioritize opportunities by expected value per risk unit and allow controlled multi-fill per cycle.

### Agent E: Trading Systems / Production Reliability
- Score: **7.5/10**
- Strengths:
  - reconnection logic, stale-feed detection, and state persistence are present,
  - bounded in-memory structures for tracked tokens.
- Risks:
  - no periodic market metadata refresh for resolution fields can delay settlement in some cases,
  - observability is still light (few hard metrics, mostly logs).
- Recommendation:
  - add periodic market refresh task and metrics dashboard (feed latency, drop counts, fill stats, PnL attribution).

---

## 3. Consolidated Verdict

- Current readiness: **promising prototype / controlled live-small-cap candidate**, not yet institution-grade.
- Composite score: **6.6/10**.
- Best next three upgrades:
  1. Model calibration + post-trade analytics (forecast quality, slippage, fill outcomes).
  2. Portfolio risk overlays (drawdown kill-switch, event/regime throttles).
  3. Execution intelligence (fill-probability + slippage-aware expected value).

---

## 4. Upgrade Progress (Implemented 2026-02-27)

- Completed:
  1. **Execution intelligence**:
     - fee/slippage-aware net PnL now uses real fill/notional/fee fields when available,
     - fallback fee/slippage estimates via `execution.*_bps` config,
     - partial-fill accounting keeps realized PnL and remaining cost basis consistent.
  2. **Portfolio risk overlays**:
     - session drawdown guard added: pauses new entries when drawdown breaches
       `risk.max_session_drawdown_usdc` or `risk.max_session_drawdown_pct`,
     - volatility throttle added: skips entries for assets above `risk.max_asset_sigma`.
  3. **Memory-growth hardening**:
     - strategy price-history cache now prunes stale token IDs when market universe rotates.
  4. **Model calibration baseline**:
     - entry-time diagnostics are persisted per position (`model_prob`, `market_prob`, `edge`),
     - periodic calibration telemetry now logs Brier score vs market baseline on resolved trades.
  5. **Adaptive re-tuning loop**:
     - runtime strategy parameters (`strategy.min_edge`, `kelly.fraction`) now auto-adjust
       based on calibration drift with bounded step sizes and floors/caps.
  6. **Regime-specific adaptation**:
     - calibration is now tracked per composite regime bucket
       (`volatility + spread + liquidity`),
     - adaptive tuning updates each observed regime track independently.
  7. **Event/news regime input**:
     - regime labels now include an explicit `event/normal_event` dimension,
     - event mode supports manual global toggle, asset list, and question-keyword tagging.
  8. **External event feed integration**:
     - external RSS/API feeds can now auto-activate event regimes with cooldown windows,
     - feed-driven event states are injected into runtime strategy in real time.
  9. **Feed quality hardening (source scoring + dedup)**:
     - event activation now uses aggregate weighted source confidence (`external_source_weights`,
       `external_event_min_score`) instead of raw hit count,
     - cross-feed payload fingerprint deduplication suppresses repeated noisy stories
       within a configurable window (`external_dedup_window_minutes`).
  10. **Adaptive feed source reliability**:
      - source weights are now dynamically adjusted via EMA reliability multipliers based on
        cross-source confirmation (single-source outliers are down-weighted; confirmed hits are up-weighted),
      - reliability state is bounded and pruned to active configured sources to prevent unbounded growth.
  11. **Walk-forward validation telemetry**:
      - periodic walk-forward analysis now runs on resolved diagnostics history, selecting
        edge thresholds on train windows and reporting out-of-sample test performance.
  12. **Portfolio stress + correlation telemetry**:
      - periodic stress scenarios now estimate projected portfolio loss/drawdown under
        mild/event/crash shocks for current open positions,
      - resolved-trade BTC/ETH PnL correlation is logged to monitor cross-asset coupling risk.

- Remaining primary gap:
  - none from the previously agreed batch list.
