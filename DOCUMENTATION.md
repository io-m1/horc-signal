# HORC Signal v1.0

High Order Range Confluence Signal System

## Overview

HORC is a deterministic trading signal system built on four core axioms:
1. **Participant Control**: Market participant identification via liquidity sweeps
2. **Wavelength Structure**: Multi-phase price progression tracking
3. **Exhaustion Detection**: Emission-based absorption analysis
4. **Gap Mechanics**: Futures gap gravitational targeting

## Quick Start

```bash
pip install -r requirements.txt
python demo_orchestrator.py
```

## Architecture

```
Raw Data → 4 Engines → Orchestrator → Signal IR → (Backtest/Pine)
                            ↑
                    Confluence + Gating
```

### Core Components

- **Participant Engine**: Identifies market control via opening range sweeps
- **Wavelength Engine**: Tracks structural progression (Move 1/2/3, Flip)
- **Exhaustion Engine**: Detects absorption via emission analysis
- **Gap Engine**: Tracks futures gap targets and fills
- **Orchestrator**: Unified confluence scoring and signal generation

### Signal Generation

Signals require:
- Confluence score >= 0.75 (default threshold)
- Clear directional bias from multiple engines
- Regime filter pass (optional)
- Absorption confirmation at defended liquidity

## Configuration

Key parameters in `OrchestratorConfig`:
- `confluence_threshold`: Minimum score for signal (default 0.75)
- `require_absorption`: Require exhaustion confirmation (default True)
- `enable_regime_filter`: Filter by volatility regime (default False)
- Engine-specific configs for tuning individual components

## Validation

Run comprehensive validation:
```bash
python run_validation.py
```

## Pine Script

TradingView implementations:
- `horc_signal_lite.pine`: Lightweight v4.3 implementation
- `horc_signal.pine`: Full-featured version

## Data Sources

Supported adapters:
- Interactive Brokers (IB)
- Polygon.io
- Historical CSV

See `docs/DATA_SOURCES.md` for API setup.

## Development

Core modules:
- `/src/core/`: Engine implementations
- `/src/engines/`: High-level engine wrappers
- `/src/data/`: Data adapters
- `/src/validation/`: Backtesting and validation
- `/tests/`: Unit and integration tests

Run tests:
```bash
pytest tests/
```

## License

Proprietary - All Rights Reserved
