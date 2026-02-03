# HORC Signal v1.0

High Order Range Confluence - Deterministic Trading Signal System

## Core Axioms

1. **Participant Control** - Market participant identification via liquidity sweeps
2. **Wavelength Structure** - Multi-phase price progression tracking (3-move cycle)
3. **Exhaustion Detection** - Emission-based absorption analysis
4. **Gap Mechanics** - Futures gap gravitational targeting

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python demo_orchestrator.py
python demo_participant.py
python demo_wavelength.py
python demo_exhaustion.py
python demo_gaps.py
python run_validation.py
```

## Architecture

```
Market Data → [4 Core Engines] → Orchestrator → Signal IR → Output
                                      ↑
                              Confluence Scoring
```

### Core Engines

- **Participant Engine** - Opening range sweeps
- **Wavelength Engine** - Structural progression  
- **Exhaustion Engine** - Emission absorption
- **Gap Engine** - Futures gap tracking

### Orchestrator

Unified signal generation with confluence scoring:
- Weighted contribution from each engine
- Majority vote bias determination
- Regime-aware filtering (optional)
- Pine-safe state management

## Signal Generation

Requirements for actionable signals:
- Confluence score ≥ 0.75 (configurable)
- Clear directional bias (≥2 engines agree)
- Absorption confirmation at defended liquidity
- Regime filter pass (if enabled)

## Configuration

Key parameters:
- `confluence_threshold` - Minimum score (default 0.75)
- `participant_weight` - Weight for participant control (default 0.30)
- `wavelength_weight` - Weight for wavelength (default 0.25)
- `exhaustion_weight` - Weight for exhaustion (default 0.25)
- `gap_weight` - Weight for gap pull (default 0.20)

## Pine Script

TradingView implementation: `horc_signal_lite.pine`

## Data Sources

Supported adapters:
- Interactive Brokers
- Polygon.io
- Historical CSV

## Testing

```bash
pytest tests/
```

## License

Proprietary - All Rights Reserved
