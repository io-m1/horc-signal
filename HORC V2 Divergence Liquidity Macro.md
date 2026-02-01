# HORC V2 — Divergence, Multi-Timeframe Liquidity & Macro Immunity

## Precision Amplification: The Three Missing Pillars

> **Context:** This document supplements the base HORC reverse-engineering (V1) with three concepts that were underrepresented or absent. These are the concepts that separate HORC from generic SMC: (1) **Participant-Based Divergence & Convergence** — not indicator divergence, but *who is in control* at swing points and whether raids confirm or flip control; (2) **Multi-Timeframe Liquidity Nesting** — how Daily, Sessional, and Hourly FVGs/liquidity points form a gravitational hierarchy; and (3) **Macro/News Immunity** — why the framework’s predictive thesis holds regardless of fundamental events.

-----

## PART A: DIVERGENCE & CONVERGENCE — THE PARTICIPANT VERSION

### A.1 Why This Is NOT Traditional RSI/MACD Divergence

Traditional divergence says: *“Price makes a new high, but the RSI makes a lower high — therefore momentum is weakening.”*

HORC’s divergence concept is fundamentally different. It is not about an oscillator disagreeing with price. It is about **which class of participant successfully controls a swing point**, and whether the next interaction with that same swing point reveals that control has *shifted*.

The question HORC asks at every swing point is not *“Is the RSI diverging?”* but rather:

> *“Who was sitting at this swing point? And when price returned to raid it — did the same participants hold, or did the opposite side take it?”*

This is **participant divergence**: the market’s revealed disagreement about who owns a price level.

-----

### A.2 The Divergence/Convergence Framework — Precise Definition

#### CONVERGENCE (Control Holds)

Convergence occurs when **the same participant class that controlled a swing point successfully defends it on the next test.**

```
BULLISH CONVERGENCE:
  • Swing Low formed (buyers absorbed selling at that level)
  • Price returns to test that swing low
  • Buyers defend again → price bounces
  • Result: The swing low HOLDS. Buyers are CONFIRMED in control.
  • Market signal: CONTINUATION of bullish bias

BEARISH CONVERGENCE:
  • Swing High formed (sellers absorbed buying at that level)  
  • Price returns to test that swing high
  • Sellers defend again → price drops
  • Result: The swing high HOLDS. Sellers are CONFIRMED in control.
  • Market signal: CONTINUATION of bearish bias
```

**Convergence = the market AGREES with the prior participant verdict.**

#### DIVERGENCE (Control Flips)

Divergence occurs when **the opposite participant class raids and takes a swing point that was previously controlled by the other side.**

```
BULLISH DIVERGENCE (Sellers were in control → Buyers raid and flip):
  • Swing High formed (sellers were sitting there, defending it)
  • Sellers are WAITING at that swing high (stop-loss cluster above = sell-side liquidity)
  • Price RAIDS that swing high — but instead of sellers holding,
    BUYERS sweep through it
  • The raid triggers all the sell stops, converting them into buy liquidity
  • Price REVERSES after the raid — buyers now own that level
  • Result: The swing high is BROKEN and FLIPPED. Control DIVERGED from sellers → buyers.
  • Market signal: REVERSAL to bullish

BEARISH DIVERGENCE (Buyers were in control → Sellers raid and flip):
  • Swing Low formed (buyers were sitting there, defending it)
  • Buyers are WAITING at that swing low (stop-loss cluster below = buy-side liquidity)
  • Price RAIDS that swing low — but instead of buyers holding,
    SELLERS sweep through it
  • The raid triggers all the buy stops, converting them into sell liquidity
  • Price REVERSES after the raid — sellers now own that level
  • Result: The swing low is BROKEN and FLIPPED. Control DIVERGED from buyers → sellers.
  • Market signal: REVERSAL to bearish
```

**Divergence = the market DISAGREES with the prior participant verdict. A raid flips ownership.**

-----

### A.3 The Critical Distinction: Raid vs. Breakout

This is where most traders fail. Not every breach of a swing point is a divergence event. There are exactly **two outcomes** when price reaches a swing point:

```
SWING POINT INTERACTION
         │
         ├── Price WICKS through (wick-only penetration)
         │       │
         │       ├── Reversal candle closes BACK inside prior range
         │       │       → THIS IS A RAID (liquidity grab)
         │       │       → Prior control HOLDS if same side defends after
         │       │       → Prior control FLIPS if opposite side takes over after
         │       │
         │       └── Continuation candle closes BEYOND prior range  
         │               → This is structural BREAKOUT, not a raid
         │               → Evaluate on next timeframe
         │
         └── Price CLOSES through (body close beyond swing)
                 → This is a STRUCTURAL BREAK (BOS / Break of Structure)
                 → The swing point is invalidated entirely
                 → No divergence analysis needed — structure has shifted
```

**The raid (wick-only penetration followed by reversal) is the DIAGNOSTIC EVENT.** It reveals whether participants on one side were genuinely absorbed or were simply waiting to be swept.

-----

### A.4 Mathematical Formalization of Participant Divergence

Let `S[i]` be a swing point at index `i`, with:

- `S[i].level` = price level of the swing
- `S[i].type` = `HIGH` or `LOW`
- `S[i].controller` = `BUYERS` or `SELLERS` (the side that originally formed and defended this swing)

When price returns to `S[i]` at time `t`:

```
Define:
  penetration_depth = |P(t) − S[i].level|
  candle_close = Close(t)
  prior_range_boundary = S[i].level (the swing extreme)

RAID CONDITION:
  For S[i].type = HIGH:
    raid = (High(t) > S[i].level) AND (Close(t) < S[i].level)
    
  For S[i].type = LOW:
    raid = (Low(t) < S[i].level) AND (Close(t) > S[i].level)

If raid = TRUE, evaluate DIVERGENCE:

  DIVERGENCE_SCORE = 0

  Factor 1 — Volume at raid candle:
    If Volume(t) > 1.5 × avg_volume(lookback=10):
      DIVERGENCE_SCORE += 1    // Significant participant activity at raid

  Factor 2 — Post-raid displacement:
    displacement_candles = next 2 candles after t
    If S[i].type = LOW:
      // Bullish divergence check: did buyers take over?
      If max(High(displacement_candles)) > prior_swing_high:
        DIVERGENCE_SCORE += 2  // Buyers successfully displaced — confirmed flip
    If S[i].type = HIGH:
      // Bearish divergence check: did sellers take over?  
      If min(Low(displacement_candles)) < prior_swing_low:
        DIVERGENCE_SCORE += 2

  Factor 3 — FVG creation during raid:
    If raid candle creates a Fair Value Gap in the reversal direction:
      DIVERGENCE_SCORE += 1    // Institutional aggression confirmed

DIVERGENCE CLASSIFICATION:
  DIVERGENCE_SCORE >= 3  →  CONFIRMED DIVERGENCE (high-probability reversal)
  DIVERGENCE_SCORE == 2  →  PROBABLE DIVERGENCE (await next-candle confirmation)
  DIVERGENCE_SCORE <= 1  →  NO DIVERGENCE (convergence — prior control holds)
```

-----

### A.5 Executable Python: Divergence Engine

```python
class DivergenceEngine:
    """
    Detects participant-based divergence at swing points.
    This is NOT RSI/MACD divergence. This is structural control-flip detection.
    """

    def __init__(self, volume_mult: float = 1.5, lookback: int = 10):
        self.volume_mult = volume_mult
        self.lookback = lookback

    def identify_swing_points(self, df: pd.DataFrame, order: int = 3) -> list:
        """
        Identify swing highs and lows using a rolling window.
        A swing high: highest point in a window of `order` candles on each side.
        """
        swings = []
        for i in range(order, len(df) - order):
            window = df.iloc[i - order: i + order + 1]
            
            # Swing High
            if df.iloc[i]['High'] == window['High'].max():
                swings.append({
                    'index': i,
                    'type': 'HIGH',
                    'level': df.iloc[i]['High'],
                    'controller': 'SELLERS',  # Sellers defended at highs
                    'timestamp': df.index[i]
                })
            
            # Swing Low
            if df.iloc[i]['Low'] == window['Low'].min():
                swings.append({
                    'index': i,
                    'type': 'LOW',
                    'level': df.iloc[i]['Low'],
                    'controller': 'BUYERS',   # Buyers defended at lows
                    'timestamp': df.index[i]
                })
        return swings

    def detect_raid(self, df: pd.DataFrame, swing: dict) -> dict:
        """
        Scan forward from swing point for a raid event.
        Raid = wick penetration WITHOUT body close beyond the swing level.
        """
        swing_idx = swing['index']
        swing_level = swing['level']

        # Look at the next 30 candles for a raid
        for i in range(swing_idx + 1, min(swing_idx + 30, len(df))):
            candle = df.iloc[i]

            if swing['type'] == 'HIGH':
                # Raid: wick goes above swing high, but closes below it
                if candle['High'] > swing_level and candle['Close'] < swing_level:
                    return {
                        'raid_index': i,
                        'raid_candle': candle,
                        'penetration': candle['High'] - swing_level,
                        'volume': candle['Volume']
                    }

            elif swing['type'] == 'LOW':
                # Raid: wick goes below swing low, but closes above it
                if candle['Low'] < swing_level and candle['Close'] > swing_level:
                    return {
                        'raid_index': i,
                        'raid_candle': candle,
                        'penetration': swing_level - candle['Low'],
                        'volume': candle['Volume']
                    }

        return None  # No raid detected

    def score_divergence(self, df: pd.DataFrame, swing: dict, raid: dict) -> dict:
        """
        Score whether a raid constitutes a confirmed divergence (control flip).
        """
        score = 0
        details = []
        raid_idx = raid['raid_index']

        # Factor 1: Volume significance at raid
        avg_vol = df.iloc[max(0, raid_idx - self.lookback):raid_idx]['Volume'].mean()
        if raid['volume'] > avg_vol * self.volume_mult:
            score += 1
            details.append("VOLUME_CONFIRMED: Raid candle volume exceeds threshold")

        # Factor 2: Post-raid displacement (the critical test)
        if raid_idx + 2 < len(df):
            post_raid = df.iloc[raid_idx + 1: raid_idx + 3]

            # Find prior swing in opposite direction for displacement reference
            if swing['type'] == 'LOW':
                # Bullish divergence: buyers should displace ABOVE a prior swing high
                prior_high = df.iloc[max(0, swing['index'] - 20):swing['index']]['High'].max()
                if post_raid['High'].max() > prior_high:
                    score += 2
                    details.append("DISPLACEMENT_CONFIRMED: Buyers displaced above prior high")
            
            elif swing['type'] == 'HIGH':
                # Bearish divergence: sellers should displace BELOW a prior swing low
                prior_low = df.iloc[max(0, swing['index'] - 20):swing['index']]['Low'].min()
                if post_raid['Low'].min() < prior_low:
                    score += 2
                    details.append("DISPLACEMENT_CONFIRMED: Sellers displaced below prior low")

        # Factor 3: FVG creation in reversal direction
        if raid_idx + 2 < len(df):
            c0 = df.iloc[raid_idx]
            c2 = df.iloc[raid_idx + 2]
            
            if swing['type'] == 'LOW':
                # Bullish FVG after bullish divergence
                if c0['High'] < c2['Low']:
                    score += 1
                    details.append("FVG_CREATED: Bullish FVG confirms institutional aggression")
            elif swing['type'] == 'HIGH':
                # Bearish FVG after bearish divergence
                if c0['Low'] > c2['High']:
                    score += 1
                    details.append("FVG_CREATED: Bearish FVG confirms institutional aggression")

        # Classification
        if score >= 3:
            classification = "CONFIRMED_DIVERGENCE"
            direction = "BULLISH" if swing['type'] == 'LOW' else "BEARISH"
        elif score == 2:
            classification = "PROBABLE_DIVERGENCE"
            direction = "BULLISH" if swing['type'] == 'LOW' else "BEARISH"
        else:
            classification = "CONVERGENCE"  # Control holds
            direction = "CONTINUATION"

        return {
            'swing': swing,
            'raid': raid,
            'score': score,
            'classification': classification,
            'reversal_direction': direction,
            'details': details
        }

    def scan(self, df: pd.DataFrame) -> list:
        """
        Full scan: find all swing points, check for raids, score divergence.
        Returns list of all divergence/convergence events.
        """
        swings = self.identify_swing_points(df)
        results = []

        for swing in swings:
            raid = self.detect_raid(df, swing)
            if raid is not None:
                result = self.score_divergence(df, swing, raid)
                results.append(result)

        return results
```

-----

## PART B: MULTI-TIMEFRAME LIQUIDITY HIERARCHY

### B.1 The Core Thesis: Liquidity Forms a Gravitational Hierarchy

Price does not move randomly between levels. It moves **toward liquidity**, and liquidity exists at multiple timeframe scales simultaneously. The critical insight is that **higher-timeframe liquidity DOMINATES lower-timeframe liquidity.**

This creates a nested gravitational structure:

```
DAILY LIQUIDITY (Strongest Pull)
  ├── Daily FVGs
  ├── Previous Day High / Low
  ├── Daily Order Blocks
  └── Daily Swing Points
        │
        ▼  (filters and attracts toward)
        
SESSION LIQUIDITY (Medium Pull)  
  ├── Session FVGs (London / NY AM / NY PM)
  ├── Previous Session High / Low
  ├── Session Order Blocks
  └── Session Swing Points (Asian High/Low)
        │
        ▼  (filters and attracts toward)

HOURLY LIQUIDITY (Entry Precision)
  ├── Hourly FVGs
  ├── Previous Hour High / Low
  ├── Hourly Order Blocks
  └── Hourly Swing Points (15m / 5m confirmation)
```

**The rule:** Price will ALWAYS seek to satisfy higher-timeframe liquidity obligations before completing lower-timeframe moves. A daily FVG is a magnet that hourly price action cannot ignore — it will eventually pull price toward it regardless of what the 5-minute chart shows.

-----

### B.2 How Timeframes Nest — The Delivery Model

Price delivery follows a **top-down cascade**, not a bottom-up build:

```
STEP 1: DAILY BIAS DETERMINATION
  • Identify the unfilled Daily FVG, unmitigated Daily OB, or 
    previous Day High/Low that has NOT been raided
  • This establishes the SESSION's gravitational target
  • Example: An unfilled Bullish Daily FVG above current price
    → The day's job is to DELIVER price up to that FVG

STEP 2: SESSION ROUTING
  • Within the daily target, identify which SESSION will do the delivery
  • London session (2:00–5:00 AM EST): Often initiates the move or creates
    the manipulation (fake move opposite direction to grab liquidity)
  • NY AM session (9:30–11:00 AM EST): Often executes the true delivery
  • NY PM session (2:00–4:00 PM EST): Often completes or closes gaps
  • The session creates its OWN FVGs and liquidity points EN ROUTE to the daily target

STEP 3: HOURLY EXECUTION
  • Within the session's routing, the hourly chart reveals the 
    EXACT entry points
  • Hourly FVGs, OBs, and swing points become the precision entry zones
  • A 15m or 5m confirmation candle triggers the actual entry
```

-----

### B.3 Confluence Points — Where Timeframes Align

The highest-probability trade setups occur where liquidity from MULTIPLE timeframes converges at the SAME price zone:

```
TIER 1 CONFLUENCE (Maximum Probability):
  Daily FVG + Session FVG + Hourly OB at same zone
  → All three timeframes agree this price level needs to be visited
  → Entry here has the backing of the entire gravitational hierarchy

TIER 2 CONFLUENCE:
  Daily OB + Session FVG overlap
  OR
  Session OB + Hourly FVG + Divergence confirmation
  → Two timeframes aligned with one confirmation signal

TIER 3 CONFLUENCE:
  Single timeframe signal with cross-timeframe trend alignment
  → Tradeable but lower conviction
```

-----

### B.4 The FVG Fill Hierarchy — Which Fills First?

When multiple unfilled FVGs exist across timeframes, they fill in a **predictable order**:

```
PRIORITY ORDER (fills from closest to furthest from current price):

1. Hourly FVGs closest to current price fill FIRST (within minutes/hours)
2. Session FVGs fill during the active session (within hours)
3. Daily FVGs fill within the trading day (may take the full session)
4. Weekly FVGs can take multiple days

BUT: If a Daily FVG is between current price and a Session FVG target,
the Daily FVG will be visited EN ROUTE — it cannot be skipped.

This means: Track ALL unfilled FVGs across timeframes. The daily ones
are the DESTINATION. The hourly ones are the WAYPOINTS.
```

-----

### B.5 Session Liquidity Points — The Specific Levels

Each session leaves behind specific liquidity points that become targets for subsequent sessions:

```
ASIAN SESSION (8:00 PM – 12:00 AM EST)
  Creates: Asian High (AH), Asian Low (AL)
  These become: London's manipulation targets
  London often raids AH or AL to grab liquidity before the real move

LONDON SESSION (2:00 – 5:00 AM EST)  
  Creates: London High (LH), London Low (LL)
  These become: NY's reference points
  NY often extends beyond London's range in the true direction

NY AM SESSION (9:30 – 11:00 AM EST)
  Creates: NY AM High, NY AM Low
  The Opening Range (OR) forms here — the HORC diagnostic window
  This is where the daily bias is CONFIRMED or INVALIDATED

NY PM SESSION (2:00 – 4:00 PM EST)
  Creates: Completion moves, gap fills
  Often revisits unfilled FVGs from earlier sessions
  The "cleanup" session
```

-----

### B.6 Mathematical Formalization: Liquidity Gravity Score

```
For each price level L, define its Liquidity Gravity Score (LGS):

LGS(L) = Σ [ w_tf × proximity_factor(L, target_tf) × type_weight(target_type) ]

Where:
  tf ∈ {DAILY, SESSION, HOURLY}
  
  w_tf = timeframe weight:
    w_DAILY   = 3.0
    w_SESSION = 2.0  
    w_HOURLY  = 1.0

  proximity_factor(L, target) = 1 / (1 + |L − target.level| / ATR)
    → Approaches 1.0 as L approaches the target
    → Decays toward 0 as L moves away

  type_weight(target_type):
    FVG (unfilled)  = 1.5   // Strongest magnet — imbalance MUST be resolved
    Order Block     = 1.2   // Strong — institutional zone
    Swing Point     = 1.0   // Standard — liquidity resting point
    Session High/Low = 0.8  // Reference level

TRADE TARGET SELECTION:
  Target = argmax { LGS(L) } over all identified unfilled liquidity levels
  in the direction of the current bias

  If LGS(target) > 4.0  →  HIGH CONVICTION target
  If LGS(target) > 2.5  →  MEDIUM CONVICTION target
  If LGS(target) ≤ 2.5  →  LOW CONVICTION — wait for more confluence
```

-----

### B.7 Executable Python: Multi-Timeframe Liquidity Engine

```python
class MTFLiquidityEngine:
    """
    Multi-Timeframe Liquidity Hierarchy Engine.
    Tracks FVGs, OBs, and swing points across Daily, Session, and Hourly timeframes.
    Computes Liquidity Gravity Score for target selection.
    """

    # Timeframe weights
    W_DAILY = 3.0
    W_SESSION = 2.0
    W_HOURLY = 1.0

    # Type weights
    TYPE_WEIGHTS = {
        'FVG': 1.5,
        'ORDER_BLOCK': 1.2,
        'SWING': 1.0,
        'SESSION_EXTREME': 0.8
    }

    def __init__(self):
        self.daily_targets = []
        self.session_targets = []
        self.hourly_targets = []

    def add_target(self, level: float, timeframe: str, target_type: str,
                   direction: str, filled: bool = False):
        """
        Register a liquidity target.
        direction: 'ABOVE' or 'BELOW' current price
        """
        target = {
            'level': level,
            'timeframe': timeframe,
            'type': target_type,
            'direction': direction,
            'filled': filled
        }

        if timeframe == 'DAILY':
            self.daily_targets.append(target)
        elif timeframe == 'SESSION':
            self.session_targets.append(target)
        elif timeframe == 'HOURLY':
            self.hourly_targets.append(target)

    def compute_lgs(self, current_price: float, atr: float) -> list:
        """
        Compute Liquidity Gravity Score for all unfilled targets.
        Returns sorted list of targets by LGS (highest first).
        """
        all_targets = (
            [(t, self.W_DAILY) for t in self.daily_targets if not t['filled']] +
            [(t, self.W_SESSION) for t in self.session_targets if not t['filled']] +
            [(t, self.W_HOURLY) for t in self.hourly_targets if not t['filled']]
        )

        scored = []
        for target, w_tf in all_targets:
            # Proximity factor: decays with distance, normalized by ATR
            distance = abs(current_price - target['level'])
            proximity = 1.0 / (1.0 + distance / max(atr, 0.0001))

            # Type weight
            type_w = self.TYPE_WEIGHTS.get(target['type'], 1.0)

            # LGS
            lgs = w_tf * proximity * type_w

            scored.append({
                **target,
                'lgs': round(lgs, 3),
                'proximity': round(proximity, 3),
                'distance': round(distance, 4)
            })

        # Sort by LGS descending
        scored.sort(key=lambda x: x['lgs'], reverse=True)
        return scored

    def get_primary_target(self, current_price: float, atr: float,
                           bias_direction: str) -> dict:
        """
        Get the highest-priority target aligned with the current bias.
        bias_direction: 'BULLISH' or 'BEARISH'
        """
        all_scored = self.compute_lgs(current_price, atr)

        target_direction = 'ABOVE' if bias_direction == 'BULLISH' else 'BELOW'

        for target in all_scored:
            if target['direction'] == target_direction:
                # Classify conviction
                if target['lgs'] > 4.0:
                    target['conviction'] = 'HIGH'
                elif target['lgs'] > 2.5:
                    target['conviction'] = 'MEDIUM'
                else:
                    target['conviction'] = 'LOW'
                return target

        return None  # No aligned target found

    def check_confluence(self, current_price: float, atr: float,
                         tolerance: float = 0.005) -> list:
        """
        Find price zones where multiple timeframe targets overlap.
        tolerance: percentage proximity to consider as 'same zone'
        """
        all_scored = self.compute_lgs(current_price, atr)
        confluences = []

        for i, t1 in enumerate(all_scored):
            zone_targets = [t1]
            for t2 in all_scored[i+1:]:
                # Check if within tolerance
                if abs(t1['level'] - t2['level']) / max(t1['level'], 0.0001) < tolerance:
                    if t1['timeframe'] != t2['timeframe']:  # Different timeframes
                        zone_targets.append(t2)

            if len(zone_targets) >= 2:
                avg_level = sum(t['level'] for t in zone_targets) / len(zone_targets)
                total_lgs = sum(t['lgs'] for t in zone_targets)
                timeframes_involved = list(set(t['timeframe'] for t in zone_targets))

                tier = 'TIER_1' if len(timeframes_involved) >= 3 else (
                    'TIER_2' if len(timeframes_involved) >= 2 else 'TIER_3'
                )

                confluences.append({
                    'zone_level': round(avg_level, 4),
                    'total_lgs': round(total_lgs, 3),
                    'tier': tier,
                    'timeframes': timeframes_involved,
                    'targets': zone_targets,
                    'direction': zone_targets[0]['direction']
                })

        # Remove duplicates and sort by total LGS
        confluences.sort(key=lambda x: x['total_lgs'], reverse=True)
        return confluences

    def mark_filled(self, price_touched: float, tolerance: float = 0.001):
        """
        Mark targets as filled when price touches them.
        """
        for target_list in [self.daily_targets, self.session_targets, self.hourly_targets]:
            for target in target_list:
                if not target['filled']:
                    if abs(price_touched - target['level']) / max(target['level'], 0.0001) < tolerance:
                        target['filled'] = True
```

-----

## PART C: MACRO/NEWS IMMUNITY — WHY THE FRAMEWORK IS PREDICTIVE

### C.1 The Thesis (Stated Precisely)

The HORC framework — and the broader SMC/ICT methodology it builds upon — operates on the following foundational claim:

> **Institutional participants know the news BEFORE the news is released. Their positioning (visible in price action) IS the news. Therefore, reading price action correctly makes the actual news release irrelevant to bias determination.**

This is not a conspiracy theory. It is a structural observation about information asymmetry in financial markets.

-----

### C.2 Why This Holds — The Logical Chain

```
1. INFORMATION ASYMMETRY IS STRUCTURAL
   • Central banks telegraph policy through forward guidance
   • Institutional research teams have dedicated economists
   • Large banks have proprietary data feeds, relationships, 
     and analytical capacity that retail traders cannot match
   • By the time news hits Reuters/Bloomberg, institutions
     have ALREADY positioned

2. POSITIONING LEAVES FOOTPRINTS
   • Large orders cannot be placed instantly — they leave traces:
     → Order Blocks (the candle where they loaded)
     → FVGs (price moved too fast because they were absorbing)
     → Liquidity raids (they needed to trigger stops to fill their size)
   • These footprints are VISIBLE on the chart BEFORE the news

3. THE NEWS CONFIRMS WHAT PRICE ALREADY SHOWED
   • When CPI is released, if institutions were already long
     (visible via bullish OBs and FVGs forming pre-release),
     the news simply provides the "reason" for a move that 
     was structurally telegraphed
   • A "surprise" news event that moves price in the OPPOSITE
     direction of the pre-release structure is actually the 
     institutions using the news as a MANIPULATION event —
     they raid liquidity during the spike, then reverse

4. THEREFORE: PRICE ACTION IS THE LEADING INDICATOR
   • News is a LAGGING confirmation (or a manipulation opportunity)
   • The bias was already determined by participant positioning
   • Reading the structure correctly → you already know the direction
     before the news drops
```

-----

### C.3 How HORC Handles News Events Specifically

```
SCENARIO A: News aligns with pre-release structure
  Pre-release: Bullish OBs forming, FVGs pointing up, buyers accumulating
  News: Positive economic data
  Result: Price continues in the direction structure indicated
  HORC action: Trade was already entered based on structure. News is irrelevant.

SCENARIO B: News appears to contradict pre-release structure  
  Pre-release: Bullish structure (buyers accumulating)
  News: Negative surprise data → price spikes DOWN
  Result: THIS IS THE MANIPULATION PHASE
  • The spike down raids the buy-side liquidity (stop losses below)
  • Institutions FILL their long positions during this spike
    (they needed sell-side liquidity to buy at lower prices)
  • After the raid, price REVERSES sharply higher
  HORC action: Wait for the raid to complete → enter on the reversal
  • The OR post-news will show a clean bullish breakout
  • The pre-news structure was CORRECT — it just needed the manipulation first

SCENARIO C: No pre-release structure (thin/unclear positioning)
  Pre-release: No clear OBs or FVGs, ambiguous structure
  News: Major event
  Result: NEWS CREATES the structure — don't trade during the event
  • Wait for the post-news OR to form
  • The first 15-30 minutes after release will establish the new bias
  • Trade the OR breakout, not the news itself
```

-----

### C.4 The “Power of Three” Connection

The SMC concept known as “Power of Three” directly maps to the news immunity framework:

```
POWER OF THREE SEQUENCE:

  PHASE 1: ACCUMULATION (Pre-News)
    • Price consolidates in a range
    • Institutions are quietly loading positions
    • FVGs and OBs form but don't resolve yet
    • This is the "positioning" phase

  PHASE 2: MANIPULATION (News Release / Spike)
    • Price makes a sharp move in the OPPOSITE direction of the 
      true institutional intent
    • This grabs liquidity (triggers stops) to FUEL the real move
    • Retail traders chase this spike — they are the liquidity
    • News headlines justify this move in retail minds

  PHASE 3: DISTRIBUTION / DELIVERY (Post-News Reversal)
    • Price reverses sharply in the TRUE direction
    • This is the move that was telegraphed by pre-release structure
    • Institutions have now filled their positions during Phases 1 & 2
    • Price delivers to the daily/session liquidity targets

HORC APPLICATION:
  • If you correctly identified the accumulation phase (Phase 1)
    via OBs and FVGs, you KNOW the manipulation is coming
  • You do NOT trade the manipulation spike
  • You enter AFTER the reversal confirms (Phase 3)
  • The news was irrelevant to your analysis — it was just fuel
```

-----

### C.5 Mathematical Framework: News Event Handling

```python
class NewsEventHandler:
    """
    Handles price behavior around news/macro events.
    Implements the manipulation detection and post-event bias logic.
    """

    def __init__(self, pre_event_window: int = 30,    # candles before event
                 post_event_window: int = 60,          # candles after event
                 manipulation_threshold: float = 2.0): # ATR multiplier for spike
        self.pre_event_window = pre_event_window
        self.post_event_window = post_event_window
        self.manipulation_threshold = manipulation_threshold

    def detect_pre_event_structure(self, df: pd.DataFrame, event_idx: int) -> dict:
        """
        Analyze structure BEFORE the event to determine pre-positioning bias.
        """
        pre_data = df.iloc[max(0, event_idx - self.pre_event_window):event_idx]
        
        if len(pre_data) < 5:
            return {'bias': 'NEUTRAL', 'confidence': 'LOW'}

        # Simple bias: are OBs / FVGs pointing up or down?
        # Count bullish vs bearish structure in pre-event window
        bullish_count = 0
        bearish_count = 0

        for i in range(len(pre_data) - 2):
            # Bullish FVG
            if pre_data.iloc[i]['High'] < pre_data.iloc[i+2]['Low']:
                bullish_count += 1
            # Bearish FVG
            if pre_data.iloc[i]['Low'] > pre_data.iloc[i+2]['High']:
                bearish_count += 1

        # Trend bias
        if pre_data.iloc[-1]['Close'] > pre_data.iloc[0]['Open']:
            bullish_count += 1
        else:
            bearish_count += 1

        if bullish_count > bearish_count + 1:
            return {'bias': 'BULLISH', 'confidence': 'MEDIUM' if bullish_count > 2 else 'LOW'}
        elif bearish_count > bullish_count + 1:
            return {'bias': 'BEARISH', 'confidence': 'MEDIUM' if bearish_count > 2 else 'LOW'}
        else:
            return {'bias': 'NEUTRAL', 'confidence': 'LOW'}

    def detect_manipulation(self, df: pd.DataFrame, event_idx: int, atr: float) -> dict:
        """
        Detect if the post-event move is a manipulation spike.
        Manipulation = sharp move that reverses within 2-3 candles.
        """
        if event_idx + 3 >= len(df):
            return {'manipulation_detected': False}

        event_candle = df.iloc[event_idx]
        next_candles = df.iloc[event_idx + 1: event_idx + 4]

        # Check for spike: event candle range > manipulation_threshold × ATR
        event_range = event_candle['High'] - event_candle['Low']
        is_spike = event_range > self.manipulation_threshold * atr

        if not is_spike:
            return {'manipulation_detected': False}

        # Check for reversal: did price come back?
        if event_candle['Close'] > event_candle['Open']:
            # Bullish spike — check if next candles reverse (bearish manipulation)
            reversal = next_candles['Low'].min() < event_candle['Open']
        else:
            # Bearish spike — check if next candles reverse (bullish manipulation)
            reversal = next_candles['High'].max() > event_candle['Open']

        return {
            'manipulation_detected': is_spike and reversal,
            'spike_direction': 'BULLISH' if event_candle['Close'] > event_candle['Open'] else 'BEARISH',
            'reversal_confirmed': reversal,
            'true_direction': 'BEARISH' if event_candle['Close'] > event_candle['Open'] else 'BULLISH'
        }

    def post_event_bias(self, df: pd.DataFrame, event_idx: int, atr: float) -> dict:
        """
        Determine the true post-event bias using structure, not headlines.
        """
        pre_structure = self.detect_pre_event_structure(df, event_idx)
        manipulation = self.detect_manipulation(df, event_idx, atr)

        if manipulation['manipulation_detected']:
            # Manipulation flips the apparent direction
            # The TRUE direction is opposite the spike
            return {
                'bias': manipulation['true_direction'],
                'source': 'POST_MANIPULATION_REVERSAL',
                'confidence': 'HIGH',
                'note': 'News spike was manipulation. True bias is the reversal direction.'
            }
        elif pre_structure['bias'] != 'NEUTRAL':
            # No manipulation detected — pre-event structure holds
            return {
                'bias': pre_structure['bias'],
                'source': 'PRE_EVENT_STRUCTURE',
                'confidence': pre_structure['confidence'],
                'note': 'News aligned with pre-event positioning. Structure bias holds.'
            }
        else:
            # Ambiguous — wait for post-event OR
            return {
                'bias': 'WAIT_FOR_OR',
                'source': 'POST_EVENT_OR_PENDING',
                'confidence': 'LOW',
                'note': 'No clear pre-event structure. Wait for post-event OR to form.'
            }
```

-----

## INTEGRATION: HOW THESE THREE CONCEPTS CONNECT IN THE SIGNAL PIPELINE

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: MACRO/NEWS ASSESSMENT                              │
│  • Is there a news event? If yes → run NewsEventHandler     │
│  • Determine if bias is pre-positioned, manipulated, or     │
│    pending post-event OR                                     │
│  → Output: Macro_Bias ∈ {BULL, BEAR, WAIT}                 │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: MTF LIQUIDITY TARGETING                            │
│  • Load Daily, Session, Hourly FVGs/OBs/Swings             │
│  • Compute LGS for all unfilled targets                     │
│  • Identify confluence zones (Tier 1/2/3)                   │
│  → Output: Primary_Target, Confluence_Zones                 │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: DIVERGENCE SCAN                                    │
│  • Scan recent swing points for raid events                 │
│  • Score each raid for divergence (control flip)            │
│  • If confirmed divergence found → this CONFIRMS or         │
│    INVALIDATES the bias from Steps 1 & 2                    │
│  → Output: Divergence_Events, Bias_Confirmation             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: HORC OR ENGINE (from V1)                           │
│  • Form OR, detect breakout, validate with volume           │
│  • Compute CPS (now incorporating divergence as a factor)   │
│  • Entry toward the identified MTF confluence target        │
│  → Output: TradeSignal with full context                    │
└─────────────────────────────────────────────────────────────┘
```

-----

## UPDATED CPS FORMULA (V2)

The original CPS had three inputs. Version 2 adds divergence confirmation:

```
CPS_V2 = w₁·OR_Signal + w₂·ICT_Signal + w₃·Wyckoff_Signal + w₄·Divergence_Signal

Where:
  w₁ = 0.35  (OR — reduced slightly to accommodate divergence)
  w₂ = 0.25  (ICT structural zones)
  w₃ = 0.15  (Wyckoff macro phase)
  w₄ = 0.25  (Divergence — high weight because it's the CONFIRMATION layer)

  Divergence_Signal:
    CONFIRMED_DIVERGENCE aligned with bias  →  +1 or -1
    PROBABLE_DIVERGENCE aligned             →  +0.5 or -0.5
    CONVERGENCE (holds prior structure)     →  0
    DIVERGENCE contradicts current bias     →  -0.5 (warning: reconsider)

DECISION THRESHOLDS (unchanged):
  |CPS_V2| > 0.5   →  Trade eligible
  |CPS_V2| ≤ 0.5   →  No trade
```

-----

*This document is a precision supplement to the HORC V1 reverse-engineering. The three concepts herein — participant divergence, multi-timeframe liquidity nesting, and macro immunity — represent the analytical depth that separates HORC from surface-level pattern matching. Original intellectual contribution: Hendray (@HORCSTUDIIO). All trading involves substantial risk.*
