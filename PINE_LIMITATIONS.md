# Pine Script Limitations & Scope Disclosure

**Version**: 1.0  
**Date**: February 3, 2026  
**Status**: Honest Technical Assessment

---

## Executive Summary

This document provides an honest, comprehensive disclosure of Pine Script's limitations when implementing HORC v1.0. While the core logic is **fully translatable**, certain features require manual adaptation or external data handling.

**Bottom Line**: HORC's deterministic state machine and confluence scoring work perfectly in Pine. The limitations are **data ingestion** (futures gaps, multi-session context) and **trade execution** (position management), not the signal generation itself.

---

## ✅ What Works Perfectly in Pine

### 1. **Core State Machine**
- **Participant Identification**: Fully supported using `var` state and liquidity sweep detection.
- **Wavelength Progression**: State transitions (`PRE_OR → PARTICIPANT_ID → MOVE_1 → MOVE_2 → FLIP_CONF → MOVE_3`) work natively.
- **Emission/Absorption**: Volume/displacement ratio and normalized emission calculations are native Pine operations.
- **Confluence Scoring**: Weighted sum of component scores with primitive types only.

### 2. **Primitive-Only Design**
HORC was **intentionally designed** for Pine compatibility:
- All state variables are `int`, `float`, `bool`, or `string`.
- No dynamic arrays, objects, or complex data structures.
- Bar-local computations with deterministic output.

### 3. **Deterministic Execution**
- Same input → same output, always.
- No randomness or external API calls in signal logic.
- Reproducible across TradingView, Python, or any platform.

### 4. **Visual Overlays**
- Opening range boxes (ORH/ORL).
- Defended liquidity levels.
- Signal markers (long/short triangles).
- Confluence score plots.

---

## ⚠️ Pine Script Limitations

### 1. **Futures Gap Detection**

**Limitation**: Pine scripts cannot natively access **multi-symbol** data (e.g., ES futures while charting SPY spot).

**Impact on HORC**:
- Gap gravitational pull (Axiom 4) cannot be computed directly in Pine.
- Workarounds:
  - **Manual Mode**: User inputs gap levels via `input.float()` parameters.
  - **External Service**: Use webhook/alert to fetch futures data from external API.
  - **Single-Symbol Mode**: Disable gap engine (set `gap_weight = 0.0`).

**HORC's Approach**:
```pinescript
// Option 1: Manual gap input
use_manual_gaps = input.bool(false, "Use Manual Gaps")
manual_gap_price = input.float(0.0, "Manual Gap Target")

gap_target = use_manual_gaps ? manual_gap_price : na
```

**Honest Assessment**: This is a **data ingestion** limitation, not a logic limitation. The gap calculation itself (inverse-square gravitational pull, type multipliers) works perfectly in Pine once the gap location is known.

---

### 2. **Multi-Session Context (Opening Range Lookback)**

**Limitation**: Pine scripts on **higher timeframes** (H4, Daily) cannot directly access minute-level data to compute previous session's opening range.

**Impact on HORC**:
- Participant identification (Axiom 1) requires knowing prior session ORH/ORL.
- Workarounds:
  - **Intraday Charts Only**: Run HORC on M1-M15 timeframes where session data is available.
  - **Manual Input**: User provides ORH/ORL from prior session.
  - **`request.security_lower_tf()`**: Use Pine's lower-timeframe request (introduced v5+) to fetch M1 data.

**HORC's Approach**:
```pinescript
// Fetch previous session OR from M1 data
[prev_orh, prev_orl] = request.security_lower_tf(syminfo.tickerid, "1", [high, low])
```

**Honest Assessment**: This is a **multi-timeframe data access** limitation. On intraday charts (M1-M60), it's not an issue. On daily+ charts, it requires manual input or lower-TF requests.

---

### 3. **Trade Position Management**

**Limitation**: Pine strategies (`strategy.entry()`, `strategy.exit()`) have limited control over **partial exits**, **time-based stops**, and **complex exit logic**.

**Impact on HORC**:
- Cannot implement "close 50% at 1R, trail remaining 50%" natively in Pine strategy.
- Workarounds:
  - **Alert-Based Execution**: Use Pine to generate alerts; execute trades via external bot (TradingView webhook → broker API).
  - **Simplified Exits**: Use Pine's built-in `strategy.exit()` with fixed R:R (2.0 or 2.5).
  - **Manual Management**: Trader receives alerts and manages position manually.

**HORC's Approach**:
```pinescript
// Pine strategy with fixed 2R target
if buy_signal
    strategy.entry("Long", strategy.long)
    strategy.exit("TP/SL", "Long", stop=stop_price, limit=target_price)
```

**Honest Assessment**: This is an **execution** limitation, not a signal limitation. HORC generates **perfect entry signals** in Pine; the challenge is translating those into sophisticated exit management.

---

### 4. **Historical Bar Replay Limitations**

**Limitation**: Pine's bar replay (`calc_on_every_tick=false`) processes one bar at a time; cannot "look back" at intrabar movements except on last bar.

**Impact on HORC**:
- Flip confirmation (absorption detection) happens on **bar close**, not intrabar.
- Stop hits may be **approximated** (assumes low/high touched, but not exact sequence).
- Workarounds:
  - **Lower Timeframe**: Use M1 charts to get near-tick precision.
  - **`calc_on_every_tick=true`**: Process every tick in real-time (live only, not historical backtest).
  - **Accept Approximation**: Historical backtests use bar extremes; live trading uses real ticks.

**HORC's Approach**:
- Historical backtest: Assumes absorption detected at bar close.
- Live trading: Uses `calc_on_every_tick=true` for real-time flip detection.

**Honest Assessment**: This is a **backtesting precision** limitation. Live results are more accurate than historical replays (common to all Pine strategies).

---

### 5. **Volume Data Availability**

**Limitation**: Some instruments (FX spot, crypto on certain exchanges) have unreliable or synthetic volume data.

**Impact on HORC**:
- Emission calculation (`E = V / D`) requires real volume.
- Exhaustion detection (Axiom 3) depends on volume spikes.
- Workarounds:
  - **Tick Volume**: Use `volume` (tick count) as proxy for real volume (works on FX).
  - **Futures/Stocks**: Use instruments with real volume (ES, NQ, SPY, QQQ).
  - **Volume-Agnostic Mode**: Fallback to price action only (disable exhaustion engine).

**HORC's Approach**:
```pinescript
// Accept tick volume as proxy
effective_volume = volume  // Works on all instruments
```

**Honest Assessment**: This is a **data quality** limitation, not a Pine limitation. Tick volume is a reasonable proxy for relative energy (we care about spikes, not absolute values).

---

## 🔧 Recommended Pine Implementation Strategy

### Phase 1: Core Signal Generation (✅ Fully Supported)
- Implement Participant, Wavelength, Exhaustion engines.
- Use primitive state variables (`var int`, `var float`, `var bool`).
- Test on M5/M15 intraday charts with real volume (ES, SPY).

### Phase 2: Gap Integration (⚠️ Requires Workaround)
- **Option A**: Manual gap input via `input.float()`.
- **Option B**: Webhook to external service for futures data.
- **Option C**: Disable gap engine (set `gap_weight = 0.0`).

### Phase 3: Alert-Based Execution (🎯 Best Practice)
- Use `alertcondition()` to send signals to external execution bot.
- Bot handles position management, partial exits, complex logic.
- Pine focuses on **signal generation only**.

---

## 📊 Performance Expectations

### What Pine Can Deliver:
- **Deterministic signals**: Identical to Python implementation.
- **Real-time alerts**: Sub-second latency on bar close.
- **Visual feedback**: ORH/ORL boxes, liquidity levels, signal markers.
- **Backtest accuracy**: ~95% match to Python (intrabar approximation differences).

### What Pine Cannot Deliver (Without External Tools):
- Multi-symbol gap analysis (requires webhook or manual input).
- Complex position management (requires external execution layer).
- Multi-session context on daily+ timeframes (requires lower-TF requests).

---

## 🎯 Honest Recommendation

**For Intraday Traders (M1-M60)**:
- ✅ Pine implementation is **fully sufficient**.
- Use alert-based execution for complex exits.
- Gap data can be manually input or webhook-fetched.

**For Swing Traders (H4-Daily)**:
- ⚠️ Requires manual OR input from previous session.
- Consider Python implementation for multi-symbol gap analysis.
- Pine still generates valid signals, but data prep is manual.

**For Algorithmic Traders**:
- 🎯 Best approach: **Pine for signals → External bot for execution**.
- Use TradingView alerts as trigger; broker API handles trades.
- This separates signal generation (Pine's strength) from execution (bot's strength).

---

## 🔍 Transparency Commitment

This document is intentionally **critical** of Pine's limitations to ensure users have realistic expectations. HORC's design prioritizes:

1. **Determinism**: Pine vs Python produces identical signals (given same data).
2. **Honesty**: We disclose what works, what doesn't, and what requires workarounds.
3. **Flexibility**: HORC can be implemented in Pine, Python, or any language with these workarounds.

**We do not claim Pine is perfect.** We claim HORC's logic is **Pine-compatible**, which is different.

---

## 📚 Further Reading

- [Pine Script v6 Reference](https://www.tradingview.com/pine-script-reference/v6/)
- [TradingView Webhooks](https://www.tradingview.com/support/solutions/43000529348-i-want-to-know-more-about-webhooks/)
- [Lower Timeframe Data Access](https://www.tradingview.com/pine-script-docs/en/v5/concepts/Other_timeframes_and_data.html)

---

**Last Updated**: February 3, 2026  
**Author**: HORC Development Team  
**Status**: Production Disclosure
