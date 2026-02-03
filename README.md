## HORC Signal System

High Order Range Confluence trading signal system built on four market axioms.

### Core Axioms

1. Participant Control via opening range liquidity sweeps
2. Wavelength Structure tracking 3-phase price progression
3. Exhaustion Detection through emission absorption analysis
4. Gap Mechanics using futures gap gravitational targeting

### Installation

```
pip install -r requirements.txt
```

### Quick Start

```
python run_validation.py
python realistic_m1_test.py
pytest tests/
```

### Architecture

```
Market Data -> 4 Engines -> Orchestrator -> Signal IR -> Output
                                 |
                         Confluence Scoring
```

Engines: Participant, Wavelength, Exhaustion, Gap

### Configuration

OrchestratorConfig parameters:
- confluence_threshold: 0.30 (minimum score for signal)
- participant_weight: 0.50
- wavelength_weight: 0.20
- exhaustion_weight: 0.20
- gap_weight: 0.10
- require_agreement: False

### Pine Script

TradingView implementation: horc_signal.pine

MACD/RSI-style histogram display with signal line overlay.

### Data

Real data in data/ folder:
- EURUSD_M1_RTH.csv (486 days, 546MB)
- GBPUSD_M1.csv (28MB)

Supported adapters: Interactive Brokers, Polygon.io, Historical CSV

### Project Structure

```
src/core/       Core engine implementations
src/engines/    High-level engine wrappers
src/data/       Data adapters
src/validation/ Backtesting and validation
tests/          Unit tests (200 passing)
```

### Performance

Baseline HORC on EURUSD M1:
- Win Rate: 39.6%
- Profit Factor: 0.99
- Break-even R:R: 1.5:1 at 40% WR

System designed for multi-TF confluence, not single-TF standalone.

### License

Proprietary
