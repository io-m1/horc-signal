import os
import csv
import json
import threading
from datetime import datetime, timezone
from typing import Optional

DEFAULT_PATH = os.environ.get("HORC_TRADE_LOG_PATH", "logs/trade_logs.csv")
DEFAULT_ENABLE = os.environ.get("HORC_TRADE_LOG_ENABLE", "1")


class _TradeLogger:
    def __init__(self, path: str = DEFAULT_PATH, fmt: str = "csv", enable: bool = True):
        self.path = path
        self.fmt = fmt.lower()
        self.enable = enable
        self._lock = threading.Lock()

        if not self.enable:
            return

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # If CSV, write header if file doesn't exist
        if self.fmt == "csv":
            file_exists = os.path.exists(self.path)
            self._f = open(self.path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._f)
            if not file_exists:
                self._writer.writerow(self._header())
                self._f.flush()
        else:
            # JSONL (one JSON per line)
            self._f = open(self.path, "a", encoding="utf-8")

    def _header(self):
        return [
            "timestamp_ms",
            "iso",
            "bias",
            "actionable",
            "confidence",
            "participant_control",
            "wavelength_state",
            "moves_completed",
            "exhaustion_score",
            "in_exhaustion_zone",
            "active_gap_type",
            "gap_fill_progress",
            "has_futures_target",
            "futures_target",
            "liquidity_direction",
            "liquidity_level",
            "market_control",
            "market_control_conclusive",
            "strategic_alignment",
            "debug_flags",
            "bars_processed",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    def log(self, signal_ir, candle, bars_processed: Optional[int] = None, extra: Optional[dict] = None):
        if not self.enable:
            return

        row = {
            "timestamp_ms": int(signal_ir.timestamp),
            "iso": datetime.fromtimestamp(signal_ir.timestamp / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "bias": int(signal_ir.bias),
            "actionable": bool(signal_ir.actionable),
            "confidence": float(signal_ir.confidence),
            "participant_control": int(signal_ir.participant_control),
            "wavelength_state": int(signal_ir.wavelength_state),
            "moves_completed": int(signal_ir.moves_completed),
            "exhaustion_score": float(signal_ir.exhaustion_score),
            "in_exhaustion_zone": bool(signal_ir.in_exhaustion_zone),
            "active_gap_type": int(signal_ir.active_gap_type),
            "gap_fill_progress": float(signal_ir.gap_fill_progress),
            "has_futures_target": bool(signal_ir.has_futures_target),
            "futures_target": float(signal_ir.futures_target) if not (signal_ir.futures_target is None) else None,
            "liquidity_direction": int(signal_ir.liquidity_direction),
            "liquidity_level": float(signal_ir.liquidity_level) if signal_ir.liquidity_level is not None else None,
            "market_control": int(signal_ir.market_control),
            "market_control_conclusive": bool(signal_ir.market_control_conclusive),
            "strategic_alignment": float(signal_ir.strategic_alignment),
            "debug_flags": int(signal_ir.debug_flags),
            "bars_processed": int(bars_processed) if bars_processed is not None else None,
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": float(getattr(candle, "volume", 0.0)),
        }

        if extra:
            row.update(extra)

        with self._lock:
            if self.fmt == "csv":
                # Preserve header ordering
                hdr = self._header()
                self._writer.writerow([row.get(k, "") for k in hdr])
                self._f.flush()
            else:
                self._f.write(json.dumps(row, default=str) + "\n")
                self._f.flush()

    def close(self):
        try:
            if hasattr(self, "_f") and not self._f.closed:
                self._f.close()
        except Exception:
            pass


# Module-level singleton
_global_logger = None


def init_trade_logger(path: Optional[str] = None, fmt: str = "csv", enable: Optional[bool] = None):
    global _global_logger
    if _global_logger is not None:
        return _global_logger

    if enable is None:
        enable = DEFAULT_ENABLE != "0"

    path = path or DEFAULT_PATH
    _global_logger = _TradeLogger(path=path, fmt=fmt, enable=enable)
    return _global_logger


def get_logger():
    global _global_logger
    if _global_logger is None:
        init_trade_logger()
    return _global_logger


# Convenience singleton instance exposed
trade_logger = get_logger()
