Minimal logging instructions for HORC trade-exporter

- Default path: `logs/trade_logs.csv`
- Enable logging: set environment variable `HORC_TRADE_LOG_ENABLE=1` (default enabled)
- Disable logging: set `HORC_TRADE_LOG_ENABLE=0`
- To change path: set `HORC_TRADE_LOG_PATH` to desired file (directory will be created)
- Format: CSV by default; JSONL supported via code (call `init_trade_logger(..., fmt="json")`)

This file is intentionally minimal — the logger is opt-in via environment variables and
used by the demo harness `demo_orchestrator.py`.
