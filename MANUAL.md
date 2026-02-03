## HORC Manual

### System Overview

HORC identifies high-probability market reversals by unifying four independent market axioms into a single confluence score. Unlike indicator-based systems, HORC operates on first-principles market mechanics.

### The Four Axioms

#### Axiom 1: Participant Control

The first participant to sweep opening range liquidity controls the session.

When buyers sweep seller stops at opening range low, they own the session until proven otherwise. We identify WHO is in control, then wait for them to exhaust.

Implementation: ParticipantIdentifier class detects OR sweeps within first 3 candles after OR close.

#### Axiom 2: Wavelength Structure

All tradeable moves progress through exactly 3 phases before completion.

Move 1: Initial thrust from OR sweep
Move 2: Retest of defended liquidity  
Move 3: Final directional move (the trade)

Implementation: WavelengthEngine with 7-state FSM tracking progression.

States: PRE_OR, PARTICIPANT_ID, MOVE_1, MOVE_2, FLIP_CONFIRMED, MOVE_3, COMPLETE

#### Axiom 3: Exhaustion Detection

High effort (volume) with low result (price displacement) signals exhaustion.

Emission = Volume / Displacement

When normalized emission exceeds 1.5 at defended liquidity with minimal price movement, reversal is imminent.

Implementation: ExhaustionDetector calculating volume absorption, body rejection, price stagnation, reversal patterns.

#### Axiom 4: Gap Mechanics

Unfilled gaps act as gravitational attractors pulling price toward them.

Gravitational Pull = min(1.0, 100/distance^2) x type_multiplier

Gap types: COMMON, BREAKAWAY, EXHAUSTION, MEASURING

Implementation: FuturesGapEngine tracking gap detection, classification, and fill progress.

### Signal Generation

#### Confluence Calculation

```
confluence = participant_strength * 0.50 +
             wavelength_progress * 0.20 +
             exhaustion_strength * 0.20 +
             gap_strength * 0.10
```

#### Actionable Requirements

1. Confluence score >= threshold (0.30)
2. Clear directional bias (non-zero)
3. Strategic validation (if enabled)

#### Bias Determination

Voting from 3 sources:
- Participant: BUYERS = +1, SELLERS = -1, NONE = 0
- Wavelength: Structure direction based on move extremes
- Gap: Gap direction if gravitational pull > 0.5

Final bias: Sum of votes (positive = LONG, negative = SHORT)

### Calibration Results

Testing on 486 days EURUSD M1 RTH data:

Win Rate: 50.2% with 1:1 R:R
Direction Accuracy (5 bars): 49.6%
Direction Accuracy (60 bars): 44.1%

Baseline config (39.6% WR, PF 0.99) performs best.
All additional filters (LE concepts, pullback entries) degraded performance.

### Pine Script Implementation

horc_signal.pine provides TradingView indicator with:

- MACD/RSI-style histogram (green=bullish, red=bearish)
- Signal line overlay (5-period EMA)
- Info table showing state, participant, CPS, entry/target
- Configurable OR window and signal threshold
- Alert conditions for automated trading

Pine Limitations:
- Cannot access multi-symbol data for gap detection
- Multi-session context requires lower-TF requests
- Complex position management requires external execution

### Recommended Deployment

For Intraday (M1-M60):
- Pine implementation fully sufficient
- Use alert-based execution for complex exits
- Gap data manually input or webhook-fetched

For Swing (H4-Daily):
- Requires manual OR input from previous session
- Consider Python for multi-symbol gap analysis

For Algorithmic:
- Pine for signals, external bot for execution
- TradingView alerts trigger broker API

### Configuration Reference

OrchestratorConfig:
- confluence_threshold: float (0.30)
- participant_weight: float (0.50)
- wavelength_weight: float (0.20)
- exhaustion_weight: float (0.20)
- gap_weight: float (0.10)
- require_agreement: bool (False)
- min_wavelength_moves: int (1)
- require_strategic_context: bool (False)

WavelengthConfig:
- min_move_1_size_atr: float (0.5)
- max_move_2_retracement: float (0.786)
- exhaustion_threshold: float (0.70)
- max_move_duration_candles: int (50)
- flip_confirmation_candles: int (3)

ExhaustionConfig:
- volume_weight: float (0.30)
- body_weight: float (0.30)
- price_weight: float (0.25)
- reversal_weight: float (0.15)
- threshold: float (0.70)

### Testing

```
pytest tests/
```

200 tests covering all engines and edge cases.
