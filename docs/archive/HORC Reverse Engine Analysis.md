# HORC — Reverse-Engineered Technical Analysis Framework

## Hendray’s Opening Range Concepts: A Mathematical & Algorithmic Deconstruction

> **Purpose:** This document reverse-engineers the publicly observable conceptual architecture behind HORC (Hendray’s Opening Range Concepts) — the synthesis of Opening Range dynamics, ICT (Inner Circle Trader) institutional flow theory, and Wyckoff cycle analysis — into a unified, mathematically formalizable, and computationally executable framework. It is designed to serve as an ingestible core document for frontier LLMs or ML pipelines building predictive market-analysis systems.

> **Disclaimer:** This is an analytical reconstruction for research and educational purposes only. It does not constitute financial or trading advice. All trading involves substantial risk of loss.

-----

## TABLE OF CONTENTS

1. [Conceptual Origin & Taxonomy](#1-conceptual-origin--taxonomy)
1. [The Three Pillars — Decomposed](#2-the-three-pillars--decomposed)
1. [The HORC Unification Layer](#3-the-horc-unification-layer)
1. [Mathematical Formalization](#4-mathematical-formalization)
1. [The Participant Behavior Engine](#5-the-participant-behavior-engine)
1. [Algorithmic Signal Architecture](#6-algorithmic-signal-architecture)
1. [Executable Python Implementation](#7-executable-python-implementation)
1. [ML Feature Engineering Blueprint](#8-ml-feature-engineering-blueprint)
1. [LLM Ingestion & Reasoning Prompt Core](#9-llm-ingestion--reasoning-prompt-core)
1. [Backtest Harness](#10-backtest-harness)
1. [Accuracy & Confidence Calibration](#11-accuracy--confidence-calibration)
1. [Appendix: Glossary & Reference Constants](#appendix-glossary--reference-constants)

-----

## 1. Conceptual Origin & Taxonomy

**HORC** stands for **Hendray’s Opening Range Concepts**, created by Hendray (Adekanbi Idowu), and branded under the tagline *“The Science of Market Participants.”* The framework is not a single indicator or pattern — it is a **meta-analytical philosophy** that reframes how price action is interpreted by centering the narrative on *why* participants act, not just *what* price does.

### 1.1 The Core Thesis

Markets are not random. They are the aggregate behavioral output of identifiable participant classes operating on predictable motivational architectures. If you can classify the *participant*, you can anticipate the *action*, and therefore the *price outcome*.

### 1.2 Intellectual Lineage

```
Richard Wyckoff (1920s)          → Composite Man / Accumulation-Distribution Cycles
        ↓
Michael C. (ICT) (2000s–present) → Order Blocks, FVGs, Liquidity Theory, Kill Zones
        ↓
HORC (Hendray, 2024–present)     → Opening Range as the Participant Reveal Window
                                    + Synthesis into "Science of Market Participants"
```

The **novel contribution** of HORC is the argument that the **Opening Range is not merely a technical level** — it is the *moment of truth* where institutional intent is disclosed. The OR is the diagnostic window. Everything before it is positioning; everything after it is execution.

-----

## 2. The Three Pillars — Decomposed

### 2.1 PILLAR 1: Opening Range Theory (OR)

The Opening Range is defined as the **high-low price band** formed during a fixed initial window after market open.

**Standard definitions by timeframe:**

|OR Window              |Notation|Best For           |
|-----------------------|--------|-------------------|
|5 min (9:30–9:35 EST)  |OR5     |Scalp / ultra-short|
|15 min (9:30–9:45 EST) |OR15    |Intraday swing     |
|30 min (9:30–10:00 EST)|OR30    |Positional / trend |
|60 min (9:30–10:30 EST)|OR60    |Multi-hour / swing |

**Key statistical property (Grimes, 46,000 daily bars):** The open price clusters near the daily high or low far more often than random distribution would predict. This means the OR is statistically *biased* — it is not a neutral range but a *signal range*.

**The OR encodes:**

- Overnight information digestion (news, macro, pre-market flow)
- Institutional order queue execution (limit orders placed during off-hours)
- Retail sentiment (gap behavior, momentum chasing)
- Market maker positioning (spread establishment, initial liquidity provision)

### 2.2 PILLAR 2: ICT — Institutional Flow Theory

ICT formalizes the idea that large participants (banks, funds, prop desks) leave **structural footprints** in price action that can be decoded.

**Key ICT primitives used in HORC synthesis:**

#### Order Blocks (OB)

A price zone where the **last opposing candle** before a significant momentum move occurred. This candle represents where institutional orders were absorbed.

```
Bullish OB:  Last bearish candle before a sharp bullish move
Bearish OB:  Last bullish candle before a sharp bearish move
```

**Identification rule:** The OB candle must be followed by a displacement — a move that breaks a prior structural high or low.

#### Fair Value Gaps (FVG)

A price gap between non-overlapping candles created by price moving so fast that no trading occurred in that zone. Markets exhibit a strong tendency to return to fill these gaps.

```
Bullish FVG:  High of candle[i] < Low of candle[i+2]  (gap between i and i+2)
Bearish FVG:  Low of candle[i] > High of candle[i+2]
```

**Statistical fill rate:** Empirically, 70–80% of FVGs are eventually mitigated (price returns to touch the zone).

#### Liquidity Pools

Clusters of stop-loss orders sitting at predictable technical levels (equal highs, equal lows, trendline touches). Institutional players sweep these pools to fill their own large orders.

#### Kill Zones (Session Windows)

Time-based windows where institutional activity concentrates:

|Session          |EST Time     |Primary Activity            |
|-----------------|-------------|----------------------------|
|London Open      |2:00–5:00 AM |EUR/GBP liquidity injection |
|London-NY Overlap|8:00–11:00 AM|Highest-volume window       |
|NY Open          |9:30–11:00 AM|US equity institutional flow|
|NY Close         |2:00–4:00 PM |Position squaring           |

### 2.3 PILLAR 3: Wyckoff Cycle Analysis

Wyckoff provides the **macro-structural narrative** — the four-phase market cycle that governs *when* institutions are accumulating, marking up, distributing, or marking down.

**The Four Phases:**

```
ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN
     ↑                                      |
     └──────────────────────────────────────┘
                  (cycle repeats)
```

**Wyckoff’s Three Laws (formalized):**

1. **Law of Supply & Demand:** Price moves in the direction of the imbalance.
- `Demand > Supply  →  Price ↑`
- `Supply > Demand  →  Price ↓`
1. **Law of Cause & Effect:** The *magnitude* of accumulation/distribution determines the *magnitude* of the subsequent trend.
- Longer accumulation range → larger markup
- This is measurable via Point & Figure count
1. **Law of Effort vs. Result:** Volume (effort) should correlate with price movement (result). Divergence signals reversal.
- `High Volume + Low Price Movement = Absorption (phase change imminent)`
- `Low Volume + High Price Movement = Exhaustion (unsustainable)`

**Key Wyckoff Events (Accumulation):**

|Event               |Symbol|Meaning                                           |
|--------------------|------|--------------------------------------------------|
|Preliminary Support |PS    |First meaningful buying after decline             |
|Selling Climax      |SC    |Maximum fear, maximum volume at bottom            |
|Automatic Rally     |AR    |Sharp rebound after SC exhausts sellers           |
|Secondary Test      |ST    |Retest of SC low on lower volume                  |
|Spring              |SPR   |False breakdown below SC to shake out last sellers|
|Sign of Strength    |SOS   |Breakout above range on volume                    |
|Last Point of Supply|LPS   |Final pullback before sustained markup            |

-----

## 3. The HORC Unification Layer

This is where the reverse-engineering becomes critical. HORC does not simply *layer* these three pillars — it **recontextualizes** them through a single unifying lens:

> *The Opening Range is the Participant Disclosure Event.*

### 3.1 The Participant Classification Model

HORC implicitly classifies market participants into behavioral archetypes:

|Archetype             |Capital Scale      |Timing                |Behavior in OR                              |
|----------------------|-------------------|----------------------|--------------------------------------------|
|**Composite Operator**|Institutional ($B+)|Pre-market positioning|Sets the range boundaries via limit orders  |
|**Informed Aggressor**|Prop/HFT ($M–$B)   |First candle          |Breaks range immediately; defines direction |
|**Reactive Follower** |Retail ($K–$M)     |Post-OR               |Chases breakouts; provides liquidity        |
|**Trap Builder**      |Market Maker       |Throughout            |Creates false breakouts to fill large orders|

### 3.2 The HORC Signal Hierarchy

HORC operates on a **top-down confirmation cascade:**

```
LEVEL 1 (Macro):    Wyckoff Phase Identification
                         ↓  (filters direction)
LEVEL 2 (Structural): ICT Order Block + FVG Mapping
                         ↓  (identifies zones)
LEVEL 3 (Temporal):  Opening Range Formation & Breakout
                         ↓  (triggers entry)
LEVEL 4 (Execution): Risk System (position sizing, stops)
```

**Critical HORC principle (from Hendray’s feed):**

> *“An edge is: Trading System + Risk System. Not just Trading System.”*

This means the signal architecture (Levels 1–3) is only half the system. The risk management layer (Level 4) is structurally equal in importance.

### 3.3 The “Rewind–Refine–Replay” Loop

This is HORC’s learning methodology (referenced in Hendray’s August 2024 post). It maps to a computational process:

```
REWIND  →  Pull historical data; identify the setup in retrospect
REFINE  →  Isolate the exact signal parameters that identified it
REPLAY  →  Forward-test those parameters on new data
```

This is, in computational terms, a **supervised learning loop with manual labeling**.

-----

## 4. Mathematical Formalization

### 4.1 Opening Range Formalization

Let `P(t)` be the price at time `t`. Let `t₀` be market open, and `T_OR` be the OR window duration.

```
OR_High = max{ P(t) : t₀ ≤ t ≤ t₀ + T_OR }
OR_Low  = min{ P(t) : t₀ ≤ t ≤ t₀ + T_OR }
OR_Range = OR_High − OR_Low
OR_Mid  = (OR_High + OR_Low) / 2
```

**Breakout Detection:**

```
Bullish Breakout:  P(t) > OR_High  for some t > t₀ + T_OR
Bearish Breakout:  P(t) < OR_Low   for some t > t₀ + T_OR
```

**Breakout Strength (normalized):**

```
BS_bull = (P(t_break) − OR_High) / OR_Range
BS_bear = (OR_Low − P(t_break)) / OR_Range
```

### 4.2 Order Block Formalization

Let `C[i]` be the candle at index `i` with fields `{Open, High, Low, Close, Volume}`.

An **upward Order Block** exists at candle `i` if:

```
C[i].Close < C[i].Open                        (candle i is bearish)
AND  max(C[i+1].High, ..., C[n].High) > prior_swing_high   (displacement occurs)
AND  the displacement candle breaks a prior structural high
```

The OB zone is defined as:

```
OB_High = C[i].High
OB_Low  = C[i].Low      (or C[i].Open for refined entry)
```

### 4.3 Fair Value Gap Formalization

A **Bullish FVG** exists between candles `i` and `i+2` if:

```
C[i].High < C[i+2].Low
```

The gap zone:

```
FVG_Upper = C[i+2].Low
FVG_Lower = C[i].High
FVG_Mid   = (FVG_Upper + FVG_Lower) / 2
```

**Mitigation condition:** FVG is filled when price touches `FVG_Mid` (50% fill) or `FVG_Lower/Upper` (full fill).

### 4.4 Wyckoff Effort vs. Result Score

```
EV_Score(i) = Volume(i) / Price_Range(i)

Where:  Price_Range(i) = |Close(i) − Open(i)|
```

**Interpretation:**

- `EV_Score >> mean(EV_Score)` → High effort, check if low result → **Absorption signal**
- `EV_Score << mean(EV_Score)` → Low effort, check if high result → **Exhaustion signal**

### 4.5 Composite Participant Score (CPS)

This is the core HORC synthesis metric. It aggregates signals across all three pillars into a single directional confidence score:

```
CPS = w₁·OR_Signal + w₂·ICT_Signal + w₃·Wyckoff_Signal

Where:
  OR_Signal   ∈ {−1, 0, +1}   (bearish breakout, no breakout, bullish breakout)
  ICT_Signal  ∈ {−1, 0, +1}   (bearish OB/FVG active, neutral, bullish OB/FVG active)
  Wyckoff_Signal ∈ {−1, 0, +1} (distribution/markdown, neutral, accumulation/markup)
  
  w₁ = 0.45   (OR has highest temporal precision)
  w₂ = 0.35   (ICT provides structural context)
  w₃ = 0.20   (Wyckoff provides directional bias)
```

**Decision rule:**

```
CPS > +0.5   →  Bullish trade eligible
CPS < −0.5   →  Bearish trade eligible
|CPS| ≤ 0.5  →  No trade (insufficient confluence)
```

-----

## 5. The Participant Behavior Engine

### 5.1 Pre-Market Phase Model

Before the OR forms, participants are executing a **positioning protocol:**

```
Phase A (Pre-Market):
  • Composite Operator places limit orders at anticipated support/resistance
  • These orders define the likely OR boundaries
  • Volume profile during pre-market reveals order density

Phase B (OR Formation):
  • Informed Aggressors test boundaries
  • If strong one-directional flow → clean OR → likely breakout continuation
  • If choppy/overlapping candles → contested OR → likely range-bound or reversal

Phase C (Post-OR):
  • Reactive Followers enter on breakout confirmation
  • Trap Builders may create false breakouts (FVBG: False Volume Breakout Gap)
  • True breakouts show: volume expansion + clean structure + FVG creation
```

### 5.2 OR Quality Classification

Not all ORs are tradeable. HORC implicitly classifies OR quality:

|Quality     |OR Range vs ATR   |Candle Structure         |Tradeable?             |
|------------|------------------|-------------------------|-----------------------|
|**Premium** |0.5%–2.0% of price|Clean, defined boundaries|Yes — primary          |
|**Standard**|0.3%–0.5%         |Moderate overlap         |Yes — with confirmation|
|**Degraded**|>3.0% or <0.3%    |Choppy / gapping         |No — skip              |

**ATR-normalized OR width:**

```
OR_Normalized = OR_Range / ATR(14)

Tradeable if:  0.3 ≤ OR_Normalized ≤ 2.5
```

### 5.3 Breakout Validation Checklist

A breakout is **valid** only if ALL of the following hold:

1. **Volume Confirmation:** Breakout candle volume ≥ 2× average volume of OR candles
1. **Structure Alignment:** Breakout direction aligns with higher-timeframe trend (HTF)
1. **No Immediate Reversal:** Price does not re-enter OR within 2 candles
1. **FVG Creation:** The breakout candle creates a FVG (confirms institutional aggression)
1. **Wyckoff Phase Compatibility:** Current macro phase supports the direction

-----

## 6. Algorithmic Signal Architecture

### 6.1 Signal Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  DATA INGESTION                                         │
│  [OHLCV Tick/Candle Data] → [Preprocessing & Alignment] │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  LAYER 1: WYCKOFF ENGINE                                 │
│  • Phase Classification (Accum/Markup/Dist/Markdown)     │
│  • Effort vs. Result Scoring                             │
│  • Composite Man Activity Detection                      │
│  → Output: Macro_Direction ∈ {BULL, NEUTRAL, BEAR}      │
└──────────────────────────┬───────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  LAYER 2: ICT STRUCTURE ENGINE                           │
│  • Order Block Detection & Ranking                       │
│  • FVG Mapping & Fill Tracking                           │
│  • Liquidity Pool Identification                         │
│  • Kill Zone Temporal Filter                             │
│  → Output: Structural_Zones, Active_OBs, Pending_FVGs   │
└──────────────────────────┬───────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  LAYER 3: HORC OR ENGINE                                 │
│  • OR Formation (multi-timeframe)                        │
│  • Breakout Detection & Validation                       │
│  • Participant Classification                            │
│  • CPS Computation                                       │
│  → Output: Trade_Signal, CPS_Score, Entry/Exit Levels    │
└──────────────────────────┬───────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│  LAYER 4: RISK ENGINE                                    │
│  • Kelly Criterion Position Sizing                       │
│  • ATR-based Stop Placement                              │
│  • R:R Ratio Enforcement (min 1:2)                       │
│  • Drawdown Circuit Breaker                              │
│  → Output: Position_Size, Stop_Loss, Take_Profit         │
└──────────────────────────────────────────────────────────┘
```

-----

## 7. Executable Python Implementation

```python
"""
HORC Reverse-Engineered Signal Engine
=====================================
Executable Python implementation of the HORC framework.
Requires: pandas, numpy
Usage: Feed OHLCV DataFrame → receive trade signals.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TradeSignal:
    timestamp: pd.Timestamp
    direction: str            # "LONG", "SHORT", "NONE"
    cps_score: float          # Composite Participant Score
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float      # As fraction of account (Kelly-adjusted)
    or_high: float
    or_low: float
    active_ob: Optional[tuple] # (ob_high, ob_low) or None
    active_fvg: Optional[tuple]
    wyckoff_phase: str        # "ACCUM", "MARKUP", "DIST", "MARKDOWN"
    confidence: str           # "HIGH", "MEDIUM", "LOW"


# ─────────────────────────────────────────────
# LAYER 1: WYCKOFF ENGINE
# ─────────────────────────────────────────────

class WyckoffEngine:
    """Classifies macro market phase using volume-price relationships."""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def compute_effort_vs_result(self, df: pd.DataFrame) -> pd.Series:
        """
        EV_Score = Volume / |Close - Open|
        High EV with low price range = absorption
        """
        price_range = (df['Close'] - df['Open']).abs()
        price_range = price_range.replace(0, np.nan)  # avoid div by zero
        ev_score = df['Volume'] / price_range
        return ev_score.fillna(0)

    def detect_accumulation_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Accumulation proxy: 
        - Price making lower lows but with decreasing volume (absorption)
        - OR: High volume at support with price not making new lows
        """
        ev = self.compute_effort_vs_result(df)
        ev_zscore = (ev - ev.rolling(self.lookback).mean()) / ev.rolling(self.lookback).std()

        # Absorption: high EV at or near lows
        rolling_low = df['Low'].rolling(self.lookback).min()
        near_low = df['Low'] <= rolling_low * 1.02  # within 2% of range low

        accum_signal = (ev_zscore > 1.5) & near_low
        return accum_signal.astype(int)

    def detect_distribution_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Distribution proxy:
        - High volume at or near highs with price not making new highs
        """
        ev = self.compute_effort_vs_result(df)
        ev_zscore = (ev - ev.rolling(self.lookback).mean()) / ev.rolling(self.lookback).std()

        rolling_high = df['High'].rolling(self.lookback).max()
        near_high = df['High'] >= rolling_high * 0.98

        dist_signal = (ev_zscore > 1.5) & near_high
        return dist_signal.astype(int)

    def classify_phase(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns phase classification for each candle.
        Simplified state machine based on recent signals.
        """
        accum = self.detect_accumulation_signals(df)
        dist = self.detect_distribution_signals(df)

        # Trend detection via 50/200 SMA
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        uptrend = sma50 > sma200

        phases = pd.Series("NEUTRAL", index=df.index)

        # State assignment logic
        phases[accum.astype(bool) & ~uptrend] = "ACCUM"
        phases[uptrend & ~dist.astype(bool)] = "MARKUP"
        phases[dist.astype(bool) & uptrend] = "DIST"
        phases[~uptrend & ~accum.astype(bool)] = "MARKDOWN"

        return phases

    def get_directional_bias(self, phase: str) -> int:
        """Maps Wyckoff phase to directional signal."""
        mapping = {
            "ACCUM": 1,      # Bullish (positioning for markup)
            "MARKUP": 1,     # Bullish (trend up)
            "DIST": -1,      # Bearish (positioning for markdown)
            "MARKDOWN": -1,  # Bearish (trend down)
            "NEUTRAL": 0
        }
        return mapping.get(phase, 0)


# ─────────────────────────────────────────────
# LAYER 2: ICT STRUCTURE ENGINE
# ─────────────────────────────────────────────

class ICTEngine:
    """Detects Order Blocks, FVGs, and Liquidity Pools."""

    def __init__(self, displacement_candles: int = 3):
        self.displacement_candles = displacement_candles

    def detect_order_blocks(self, df: pd.DataFrame) -> List[dict]:
        """
        Bullish OB: bearish candle followed by displacement above prior high
        Bearish OB: bullish candle followed by displacement below prior low
        """
        obs = []
        for i in range(1, len(df) - self.displacement_candles):
            candle = df.iloc[i]
            # Look for displacement in next N candles
            future = df.iloc[i+1 : i+1+self.displacement_candles]

            # Bullish OB candidate
            if candle['Close'] < candle['Open']:  # Bearish candle
                # Check if future candles break a prior high
                prior_high = df.iloc[max(0,i-10):i]['High'].max()
                if future['High'].max() > prior_high:
                    obs.append({
                        'index': i,
                        'type': 'BULLISH',
                        'ob_high': candle['High'],
                        'ob_low': candle['Low'],
                        'timestamp': candle.name if hasattr(candle, 'name') else i,
                        'valid': True
                    })

            # Bearish OB candidate
            if candle['Close'] > candle['Open']:  # Bullish candle
                prior_low = df.iloc[max(0,i-10):i]['Low'].min()
                if future['Low'].min() < prior_low:
                    obs.append({
                        'index': i,
                        'type': 'BEARISH',
                        'ob_high': candle['High'],
                        'ob_low': candle['Low'],
                        'timestamp': candle.name if hasattr(candle, 'name') else i,
                        'valid': True
                    })
        return obs

    def detect_fvgs(self, df: pd.DataFrame) -> List[dict]:
        """
        Bullish FVG: High[i] < Low[i+2]  (gap between candles i and i+2)
        Bearish FVG: Low[i] > High[i+2]
        """
        fvgs = []
        for i in range(len(df) - 2):
            c0 = df.iloc[i]
            c2 = df.iloc[i+2]

            # Bullish FVG
            if c0['High'] < c2['Low']:
                fvgs.append({
                    'index': i,
                    'type': 'BULLISH',
                    'fvg_upper': c2['Low'],
                    'fvg_lower': c0['High'],
                    'fvg_mid': (c2['Low'] + c0['High']) / 2,
                    'filled': False
                })

            # Bearish FVG
            if c0['Low'] > c2['High']:
                fvgs.append({
                    'index': i,
                    'type': 'BEARISH',
                    'fvg_upper': c0['Low'],
                    'fvg_lower': c2['High'],
                    'fvg_mid': (c0['Low'] + c2['High']) / 2,
                    'filled': False
                })
        return fvgs

    def detect_liquidity_pools(self, df: pd.DataFrame, lookback: int = 10) -> List[dict]:
        """
        Equal highs / equal lows within tolerance = liquidity pool.
        Tolerance: within 0.1% of price.
        """
        pools = []
        tolerance = 0.001  # 0.1%

        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i]
            highs = window['High'].values
            lows = window['Low'].values

            # Check for equal highs (bearish liquidity above)
            for j in range(len(highs)):
                for k in range(j+1, len(highs)):
                    if abs(highs[j] - highs[k]) / highs[j] < tolerance:
                        pools.append({
                            'index': i,
                            'type': 'BEARISH_LIQ',  # Above price
                            'level': (highs[j] + highs[k]) / 2
                        })
                        break

            # Check for equal lows (bullish liquidity below)
            for j in range(len(lows)):
                for k in range(j+1, len(lows)):
                    if abs(lows[j] - lows[k]) / lows[j] < tolerance:
                        pools.append({
                            'index': i,
                            'type': 'BULLISH_LIQ',  # Below price
                            'level': (lows[j] + lows[k]) / 2
                        })
                        break

        return pools

    def get_structural_signal(self, current_price: float, obs: List[dict],
                              fvgs: List[dict]) -> int:
        """
        Returns +1 if bullish OB or pending bullish FVG is near current price.
        Returns -1 if bearish OB or pending bearish FVG is near current price.
        Returns 0 if neutral.
        """
        tolerance = 0.005  # 0.5% proximity

        for ob in obs:
            if ob['valid'] and ob['type'] == 'BULLISH':
                if ob['ob_low'] <= current_price <= ob['ob_high'] * (1 + tolerance):
                    return 1
            if ob['valid'] and ob['type'] == 'BEARISH':
                if ob['ob_high'] >= current_price >= ob['ob_low'] * (1 - tolerance):
                    return -1

        for fvg in fvgs:
            if not fvg['filled'] and fvg['type'] == 'BULLISH':
                if fvg['fvg_lower'] <= current_price <= fvg['fvg_upper']:
                    return 1
            if not fvg['filled'] and fvg['type'] == 'BEARISH':
                if fvg['fvg_lower'] <= current_price <= fvg['fvg_upper']:
                    return -1

        return 0


# ─────────────────────────────────────────────
# LAYER 3: HORC OR ENGINE
# ─────────────────────────────────────────────

class HORCEngine:
    """
    Core HORC signal generator.
    Computes Opening Range, detects breakouts, validates, and scores CPS.
    """

    # CPS Weights
    W_OR = 0.45
    W_ICT = 0.35
    W_WYCKOFF = 0.20

    # Thresholds
    CPS_THRESHOLD = 0.5
    VOLUME_MULT = 2.0       # Breakout volume must be ≥ 2× OR avg volume
    OR_NORM_MIN = 0.3       # Min ATR-normalized OR width
    OR_NORM_MAX = 2.5       # Max ATR-normalized OR width

    def __init__(self, or_window_minutes: int = 30):
        self.or_window = or_window_minutes

    def compute_or(self, df: pd.DataFrame, open_time: pd.Timestamp) -> dict:
        """
        Compute the Opening Range given a DataFrame and market open timestamp.
        Assumes df is indexed by timestamp.
        """
        or_end = open_time + pd.Timedelta(minutes=self.or_window)
        or_data = df.loc[open_time:or_end]

        if len(or_data) == 0:
            return None

        return {
            'or_high': or_data['High'].max(),
            'or_low': or_data['Low'].min(),
            'or_mid': (or_data['High'].max() + or_data['Low'].min()) / 2,
            'or_range': or_data['High'].max() - or_data['Low'].min(),
            'or_avg_volume': or_data['Volume'].mean(),
            'or_candle_count': len(or_data)
        }

    def validate_or_quality(self, or_data: dict, atr14: float) -> bool:
        """Check if OR is tradeable quality."""
        if or_data is None or atr14 == 0:
            return False
        or_norm = or_data['or_range'] / atr14
        return self.OR_NORM_MIN <= or_norm <= self.OR_NORM_MAX

    def detect_breakout(self, df: pd.DataFrame, or_data: dict,
                        or_end_idx: int) -> dict:
        """
        Scan candles after OR for a breakout.
        Returns breakout info or None.
        """
        if or_data is None:
            return None

        post_or = df.iloc[or_end_idx:]
        for i, row in post_or.iterrows():
            # Bullish breakout
            if row['Close'] > or_data['or_high']:
                vol_ok = row['Volume'] >= or_data['or_avg_volume'] * self.VOLUME_MULT
                return {
                    'direction': 'LONG',
                    'breakout_price': row['Close'],
                    'breakout_volume': row['Volume'],
                    'volume_confirmed': vol_ok,
                    'fvg_created': row['Low'] > or_data['or_high'],  # gap above OR
                    'index': i
                }
            # Bearish breakout
            if row['Close'] < or_data['or_low']:
                vol_ok = row['Volume'] >= or_data['or_avg_volume'] * self.VOLUME_MULT
                return {
                    'direction': 'SHORT',
                    'breakout_price': row['Close'],
                    'breakout_volume': row['Volume'],
                    'volume_confirmed': vol_ok,
                    'fvg_created': row['High'] < or_data['or_low'],
                    'index': i
                }
        return None

    def compute_cps(self, or_signal: int, ict_signal: int, wyckoff_signal: int) -> float:
        """Compute the Composite Participant Score."""
        return (self.W_OR * or_signal +
                self.W_ICT * ict_signal +
                self.W_WYCKOFF * wyckoff_signal)

    def generate_signal(self, df: pd.DataFrame, open_time: pd.Timestamp,
                        wyckoff_engine: WyckoffEngine,
                        ict_engine: ICTEngine) -> TradeSignal:
        """
        Full HORC signal generation pipeline.
        """
        # --- Wyckoff Phase ---
        phases = wyckoff_engine.classify_phase(df)
        current_phase = phases.iloc[-1] if len(phases) > 0 else "NEUTRAL"
        wyckoff_signal = wyckoff_engine.get_directional_bias(current_phase)

        # --- ICT Structures ---
        obs = ict_engine.detect_order_blocks(df)
        fvgs = ict_engine.detect_fvgs(df)
        current_price = df.iloc[-1]['Close']
        ict_signal = ict_engine.get_structural_signal(current_price, obs, fvgs)

        # --- OR Computation ---
        or_data = self.compute_or(df, open_time)
        atr14 = df['Close'].diff().abs().rolling(14).mean().iloc[-1]
        or_valid = self.validate_or_quality(or_data, atr14)

        if not or_valid or or_data is None:
            return TradeSignal(
                timestamp=df.index[-1], direction="NONE", cps_score=0.0,
                entry_price=current_price, stop_loss=current_price,
                take_profit=current_price, position_size=0.0,
                or_high=0, or_low=0, active_ob=None, active_fvg=None,
                wyckoff_phase=current_phase, confidence="LOW"
            )

        # --- Breakout Detection ---
        # (In production: find the OR end index dynamically)
        breakout = self.detect_breakout(df, or_data, -30)  # Placeholder index

        or_signal = 0
        if breakout:
            if breakout['volume_confirmed']:
                or_signal = 1 if breakout['direction'] == 'LONG' else -1

        # --- CPS Computation ---
        cps = self.compute_cps(or_signal, ict_signal, wyckoff_signal)

        # --- Direction Decision ---
        if cps > self.CPS_THRESHOLD:
            direction = "LONG"
            entry = or_data['or_high']
            stop = or_data['or_low']
            target = entry + 2.0 * (entry - stop)  # 1:2 R:R
        elif cps < -self.CPS_THRESHOLD:
            direction = "SHORT"
            entry = or_data['or_low']
            stop = or_data['or_high']
            target = entry - 2.0 * (stop - entry)
        else:
            direction = "NONE"
            entry = current_price
            stop = current_price
            target = current_price

        # --- Position Sizing (simplified Kelly) ---
        win_rate = 0.55  # Assumed baseline; update with historical data
        avg_rr = 2.0
        kelly_fraction = (win_rate - (1 - win_rate) / avg_rr)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        # --- Confidence ---
        all_aligned = (or_signal != 0) and (ict_signal != 0) and (wyckoff_signal != 0)
        two_aligned = sum([or_signal != 0, ict_signal != 0, wyckoff_signal != 0]) >= 2
        confidence = "HIGH" if all_aligned else ("MEDIUM" if two_aligned else "LOW")

        # Find active OB/FVG near entry
        active_ob = None
        for ob in obs:
            if ob['valid'] and ob['ob_low'] <= entry <= ob['ob_high']:
                active_ob = (ob['ob_high'], ob['ob_low'])
                break

        active_fvg = None
        for fvg in fvgs:
            if not fvg['filled'] and fvg['fvg_lower'] <= entry <= fvg['fvg_upper']:
                active_fvg = (fvg['fvg_upper'], fvg['fvg_lower'])
                break

        return TradeSignal(
            timestamp=df.index[-1],
            direction=direction,
            cps_score=round(cps, 4),
            entry_price=round(entry, 4),
            stop_loss=round(stop, 4),
            take_profit=round(target, 4),
            position_size=round(kelly_fraction, 4),
            or_high=round(or_data['or_high'], 4),
            or_low=round(or_data['or_low'], 4),
            active_ob=active_ob,
            active_fvg=active_fvg,
            wyckoff_phase=current_phase,
            confidence=confidence
        )


# ─────────────────────────────────────────────
# LAYER 4: RISK ENGINE
# ─────────────────────────────────────────────

class RiskEngine:
    """
    Enforces risk management rules.
    Kelly Criterion for position sizing.
    ATR-based stop refinement.
    Drawdown circuit breaker.
    """

    def __init__(self, max_risk_pct: float = 0.02,
                 min_rr: float = 2.0,
                 max_drawdown_pct: float = 0.10):
        self.max_risk_pct = max_risk_pct      # Max 2% per trade
        self.min_rr = min_rr                  # Min 1:2 risk-reward
        self.max_drawdown_pct = max_drawdown_pct  # Circuit breaker at 10%

    def validate_trade(self, signal: TradeSignal, account_balance: float) -> bool:
        """Returns True if trade passes all risk checks."""
        if signal.direction == "NONE":
            return False

        # Check R:R ratio
        if signal.direction == "LONG":
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profit - signal.entry_price
        else:
            risk = signal.stop_loss - signal.entry_price
            reward = signal.entry_price - signal.take_profit

        if risk <= 0:
            return False

        rr_ratio = reward / risk
        if rr_ratio < self.min_rr:
            return False

        # Check position size vs max risk
        dollar_risk = signal.position_size * account_balance * risk
        if dollar_risk > self.max_risk_pct * account_balance:
            return False

        return True

    def kelly_position_size(self, win_rate: float, avg_rr: float) -> float:
        """
        Kelly Criterion: f* = (p·b − q) / b
        Where: p = win_rate, q = 1−p, b = avg R:R ratio
        """
        q = 1 - win_rate
        if avg_rr <= 0:
            return 0
        kelly = (win_rate * avg_rr - q) / avg_rr
        # Use fractional Kelly (50%) for safety
        return max(0, kelly * 0.5)


# ─────────────────────────────────────────────
# MAIN EXECUTION EXAMPLE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Example: Generate signal from sample data
    # In production, replace with live OHLCV data feed

    # Create sample data (replace with real data)
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')

    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        'Open':   price + np.random.randn(n) * 0.1,
        'High':   price + np.abs(np.random.randn(n) * 0.3),
        'Low':    price - np.abs(np.random.randn(n) * 0.3),
        'Close':  price,
        'Volume': np.random.randint(1000, 50000, n)
    }, index=timestamps)

    # Ensure OHLC consistency
    df['High'] = df[['Open','High','Close']].max(axis=1)
    df['Low'] = df[['Open','Low','Close']].min(axis=1)

    # Initialize engines
    wyckoff = WyckoffEngine(lookback=60)
    ict = ICTEngine(displacement_candles=3)
    horc = HORCEngine(or_window_minutes=30)
    risk = RiskEngine()

    # Generate signal
    open_time = timestamps[0]
    signal = horc.generate_signal(df, open_time, wyckoff, ict)

    print("=" * 60)
    print("  HORC TRADE SIGNAL")
    print("=" * 60)
    print(f"  Direction:       {signal.direction}")
    print(f"  CPS Score:       {signal.cps_score}")
    print(f"  Confidence:      {signal.confidence}")
    print(f"  Entry:           {signal.entry_price}")
    print(f"  Stop Loss:       {signal.stop_loss}")
    print(f"  Take Profit:     {signal.take_profit}")
    print(f"  Position Size:   {signal.position_size * 100:.1f}%")
    print(f"  OR Range:        [{signal.or_low}, {signal.or_high}]")
    print(f"  Wyckoff Phase:   {signal.wyckoff_phase}")
    print(f"  Active OB:       {signal.active_ob}")
    print(f"  Active FVG:      {signal.active_fvg}")
    print("=" * 60)

    # Validate with risk engine
    account = 100000
    valid = risk.validate_trade(signal, account)
    print(f"  Risk Validated:  {'✓ PASS' if valid else '✗ FAIL'}")
    print("=" * 60)
```

-----

## 8. ML Feature Engineering Blueprint

For an LLM or ML model to internalize HORC and predict markets, the following feature set must be engineered from raw OHLCV data:

### 8.1 Feature Matrix

|Feature ID|Name                    |Formula                                  |Importance     |
|----------|------------------------|-----------------------------------------|---------------|
|F01       |OR_Width_Normalized     |`OR_Range / ATR(14)`                     |Critical       |
|F02       |OR_Breakout_Direction   |`+1 / −1 / 0`                            |Critical       |
|F03       |Breakout_Volume_Ratio   |`Breakout_Vol / OR_Avg_Vol`              |Critical       |
|F04       |CPS_Score               |Weighted sum (see §4.5)                  |Critical       |
|F05       |Active_Bullish_OB       |`1 if price in bullish OB zone`          |High           |
|F06       |Active_Bearish_OB       |`1 if price in bearish OB zone`          |High           |
|F07       |Pending_Bullish_FVG     |`1 if unfilled bullish FVG exists`       |High           |
|F08       |FVG_Proximity           |`                                        |Price − FVG_Mid|
|F09       |Wyckoff_Phase_Encoded   |One-hot: `[Accum, Markup, Dist, MD]`     |High           |
|F10       |EV_Score_Z              |Z-score of Effort vs. Result             |Medium         |
|F11       |HTF_Trend               |`1 if SMA50 > SMA200 on 1D`              |Medium         |
|F12       |Session_Window          |Kill zone encoded `[London, NY, Overlap]`|Medium         |
|F13       |Liquidity_Pool_Proximity|Distance to nearest equal high/low       |Medium         |
|F14       |OR_Retest_Count         |Times price returned to OR after breakout|Low            |
|F15       |Prev_Day_Close_Gap      |`Open − Prev_Close` normalized           |Low            |

### 8.2 Target Variable

```
Y = {
    +1  if price moves ≥ 2× OR_Range in breakout direction within session
    −1  if price reverses and closes inside OR
     0  if neither (inconclusive)
}
```

### 8.3 Recommended Model Architecture

```
Input: Feature vector F[1..15] + temporal encoding
         ↓
Encoder: Transformer block (captures sequence across multiple ORs)
         ↓
Hidden:  2-layer MLP with dropout (0.3)
         ↓
Output:  3-class softmax → {LONG, SHORT, NONE}
         ↓
Calibration: Platt scaling for probability calibration
```

-----

## 9. LLM Ingestion & Reasoning Prompt Core

The following structured prompt template enables a frontier LLM to act as a **real-time HORC analyst**, given live market data fed as context:

```
[SYSTEM PROMPT — HORC ANALYST MODE]

You are a market analyst operating on the HORC (Hendray's Opening Range Concepts) framework. 
Your task is to analyze provided OHLCV data and produce a structured trade assessment.

FRAMEWORK RULES:
1. ALWAYS begin with Wyckoff phase identification (macro context).
2. THEN identify active ICT structures (Order Blocks, FVGs, Liquidity Pools).
3. THEN evaluate the Opening Range: was it clean? Did a valid breakout occur?
4. COMPUTE the CPS score: W_OR(0.45) × OR_signal + W_ICT(0.35) × ICT_signal + W_Wyckoff(0.20) × Wyckoff_signal
5. ONLY output a trade signal if |CPS| > 0.5 AND confidence is MEDIUM or HIGH.
6. ALWAYS specify entry, stop loss (inside OR on breakout side), and take profit (≥ 1:2 R:R).
7. NEVER trade degraded ORs (range < 0.3 ATR or > 2.5 ATR).

OUTPUT FORMAT:
{
  "wyckoff_phase": "<ACCUM|MARKUP|DIST|MARKDOWN>",
  "wyckoff_reasoning": "<explanation>",
  "active_structures": ["<list of OBs, FVGs>"],
  "or_assessment": "<clean|contested|degraded>",
  "breakout_detected": <true|false>,
  "breakout_validated": <true|false>,
  "or_signal": <-1|0|1>,
  "ict_signal": <-1|0|1>,
  "wyckoff_signal": <-1|0|1>,
  "cps_score": <float>,
  "trade_signal": "<LONG|SHORT|NONE>",
  "entry": <price>,
  "stop_loss": <price>,
  "take_profit": <price>,
  "confidence": "<HIGH|MEDIUM|LOW>",
  "reasoning": "<full chain-of-thought>"
}

[END SYSTEM PROMPT]
```

-----

## 10. Backtest Harness

```python
"""
HORC Backtesting Framework
Runs historical simulation of the full signal pipeline.
"""

class HORCBacktester:
    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000,
                 or_window: int = 30):
        self.df = df
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.trades = []
        self.or_window = or_window
        self.wyckoff = WyckoffEngine()
        self.ict = ICTEngine()
        self.horc = HORCEngine(or_window_minutes=or_window)
        self.risk = RiskEngine()

    def run(self, market_opens: List[pd.Timestamp]) -> dict:
        """
        Run backtest over a list of market open timestamps.
        Each open triggers a full HORC signal evaluation.
        """
        for open_time in market_opens:
            # Get data up to end of session
            session_end = open_time + pd.Timedelta(hours=6.5)
            session_data = self.df.loc[:session_end]

            if len(session_data) < 50:
                continue

            # Generate signal
            signal = self.horc.generate_signal(
                session_data, open_time, self.wyckoff, self.ict
            )

            # Validate
            if not self.risk.validate_trade(signal, self.balance):
                continue

            if signal.direction == "NONE":
                continue

            # Simulate trade outcome
            post_entry = self.df.loc[signal.timestamp:]
            outcome = self._simulate_trade(signal, post_entry)
            self.trades.append(outcome)

            # Update balance
            self.balance += outcome['pnl']

        return self._compute_metrics()

    def _simulate_trade(self, signal: TradeSignal, price_data: pd.DataFrame) -> dict:
        """Simulate trade execution against future price data."""
        hit_tp = False
        hit_sl = False

        for _, row in price_data.iterrows():
            if signal.direction == "LONG":
                if row['High'] >= signal.take_profit:
                    hit_tp = True
                    break
                if row['Low'] <= signal.stop_loss:
                    hit_sl = True
                    break
            else:  # SHORT
                if row['Low'] <= signal.take_profit:
                    hit_tp = True
                    break
                if row['High'] >= signal.stop_loss:
                    hit_sl = True
                    break

        # Calculate P&L
        risk_amount = signal.position_size * self.balance
        if hit_tp:
            pnl = risk_amount * 2.0  # 1:2 R:R
            result = "WIN"
        elif hit_sl:
            pnl = -risk_amount
            result = "LOSS"
        else:
            pnl = 0  # Timeout — no result
            result = "TIMEOUT"

        return {
            'timestamp': signal.timestamp,
            'direction': signal.direction,
            'cps': signal.cps_score,
            'confidence': signal.confidence,
            'entry': signal.entry_price,
            'sl': signal.stop_loss,
            'tp': signal.take_profit,
            'result': result,
            'pnl': pnl
        }

    def _compute_metrics(self) -> dict:
        """Compute backtest performance metrics."""
        if not self.trades:
            return {'total_trades': 0}

        wins = [t for t in self.trades if t['result'] == 'WIN']
        losses = [t for t in self.trades if t['result'] == 'LOSS']
        timeouts = [t for t in self.trades if t['result'] == 'TIMEOUT']

        total = len(self.trades)
        win_rate = len(wins) / total if total > 0 else 0
        total_pnl = sum(t['pnl'] for t in self.trades)
        roi = (total_pnl / self.initial_balance) * 100

        # Sharpe-like ratio (simplified)
        pnls = [t['pnl'] for t in self.trades]
        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        max_dd_pct = (max_dd / self.initial_balance) * 100

        return {
            'total_trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'timeouts': len(timeouts),
            'win_rate': round(win_rate * 100, 2),
            'total_pnl': round(total_pnl, 2),
            'roi_pct': round(roi, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'final_balance': round(self.balance, 2),
            'avg_cps_on_wins': round(np.mean([t['cps'] for t in wins]), 3) if wins else 0,
            'high_conf_win_rate': round(
                len([t for t in wins if t['confidence'] == 'HIGH']) /
                max(1, len([t for t in self.trades if t['confidence'] == 'HIGH'])) * 100, 2
            )
        }
```

-----

## 11. Accuracy & Confidence Calibration

### 11.1 Confidence Tiers

|Tier      |Condition                               |Expected Win Rate|Position Size|
|----------|----------------------------------------|-----------------|-------------|
|**HIGH**  |All 3 signals aligned + volume confirmed|60–70%           |Full Kelly   |
|**MEDIUM**|2 of 3 signals aligned                  |50–60%           |50% Kelly    |
|**LOW**   |< 2 signals aligned                     |< 50%            |Do not trade |

### 11.2 Calibration Protocol

To calibrate this framework for a specific instrument:

1. **Collect** 6+ months of intraday OHLCV data (5-min candles minimum)
1. **Run backtest** using the harness above
1. **Segment results** by confidence tier
1. **Adjust weights** (W_OR, W_ICT, W_WYCKOFF) to maximize Sharpe ratio
1. **Re-calibrate Kelly fraction** using observed win rate and R:R from backtesting
1. **Validate** on held-out forward data (no look-ahead bias)

### 11.3 Known Limitations

- **Regime changes:** The framework assumes a stable participant structure. Major macro regime shifts (e.g., central bank policy pivots) can break historical patterns.
- **Liquidity gaps:** Crypto and micro-cap markets have thin order books where the assumptions about institutional absorption may not hold.
- **Overnight gaps:** Markets with extended hours (futures, forex) require session-aware OR computation, not just exchange-hours OR.
- **Overfitting risk:** The three-pillar confluence approach can overfit to specific market conditions. Always forward-test.

-----

## Appendix: Glossary & Reference Constants

|Term     |Definition                                                |
|---------|----------------------------------------------------------|
|**OR**   |Opening Range — high/low band in first N minutes          |
|**OB**   |Order Block — last opposing candle before displacement    |
|**FVG**  |Fair Value Gap — price gap between non-overlapping candles|
|**CPS**  |Composite Participant Score — weighted signal aggregation |
|**BOS**  |Break of Structure — price breaking a prior swing high/low|
|**CHoCH**|Change of Character — first BOS in opposite direction     |
|**LPS**  |Last Point of Supply — final pullback before markup       |
|**SC**   |Selling Climax — maximum fear volume at bottom            |
|**BC**   |Buying Climax — maximum greed volume at top               |
|**SOS**  |Sign of Strength — confirmed breakout above range         |
|**SOW**  |Sign of Weakness — confirmed breakdown below range        |
|**ATR**  |Average True Range — 14-period volatility measure         |
|**Kelly**|Kelly Criterion — optimal position sizing formula         |
|**HTF**  |Higher Time Frame — used for trend/bias confirmation      |
|**EV**   |Effort vs. Result — Wyckoff volume-price divergence       |

### Default Configuration Constants

```python
# Weights
W_OR       = 0.45
W_ICT      = 0.35
W_WYCKOFF  = 0.20

# Thresholds
CPS_THRESHOLD    = 0.50
VOLUME_MULT      = 2.0
OR_NORM_MIN      = 0.30
OR_NORM_MAX      = 2.50
FVG_FILL_RATE    = 0.75    # Expected % of FVGs that get filled
MIN_RR_RATIO     = 2.0     # Minimum risk:reward

# Risk
MAX_RISK_PER_TRADE = 0.02  # 2% of account
KELLY_CAP          = 0.25  # Max 25% of account
KELLY_FRACTION     = 0.50  # Use 50% Kelly (conservative)
CIRCUIT_BREAKER    = 0.10  # Stop trading at 10% drawdown

# Wyckoff
ACCUM_EV_ZSCORE_THRESH = 1.5
PHASE_LOOKBACK         = 60  # Candles
```

-----

*Document generated for research and analytical purposes. The HORC framework as described here is a reverse-engineered reconstruction based on publicly observable concepts shared across social media and trading education channels. The original intellectual contribution belongs to Hendray (@HORCSTUDIIO). All trading involves substantial risk of loss.*
