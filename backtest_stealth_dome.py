"""
HORC-CRT Stealth Dome Backtester
=================================

Replays historical 1-minute FX data bar-by-bar through the Stealth Dome engine,
simulates trades with SL/TP evaluation, and generates a full accuracy & calibration report.

Data format (auto-detected):
    <TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
    EURUSD,20030506,000000,1.12921,1.1293,1.1291,1.12921,592300001

Usage:
    python backtest_stealth_dome.py                                # all 3 pairs, recent 1 year
    python backtest_stealth_dome.py --pair EURUSD                  # single pair
    python backtest_stealth_dome.py --pair EURUSD --days 365       # last 365 days of data
    python backtest_stealth_dome.py --pair EURUSD --from 20240101 --to 20250101
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.engines.stealth_dome import (
    BUYER, SELLER, NEUTRAL,
    OHLCV, StealthSignal, StealthDomeConfig, StealthDomeEngine,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent / "src" / "data"

PAIR_FILES = {
    "EURUSD": DATA_DIR / "EURUSD.txt",
    "GBPUSD": DATA_DIR / "GBPUSD.txt",
    "USDJPY": DATA_DIR / "USDJPY.txt",
}


def load_bars(
    filepath: Path,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    max_bars: Optional[int] = None,
) -> List[OHLCV]:
    """
    Load 1-minute OHLCV bars from the data file.

    Args:
        filepath:  path to the .txt data file
        from_date: YYYYMMDD filter start (inclusive)
        to_date:   YYYYMMDD filter end (exclusive)
        max_bars:  safety cap on number of bars loaded
    """
    bars: List[OHLCV] = []
    print(f"[LOAD] Loading {filepath.name}...", end=" ", flush=True)
    t0 = time.time()

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row in reader:
            if len(row) < 7:
                continue
            # Row: TICKER,DTYYYYMMDD,TIME,OPEN,HIGH,LOW,CLOSE,VOL
            date_str = row[1]

            # Date filter
            if from_date and date_str < from_date:
                continue
            if to_date and date_str >= to_date:
                continue

            time_str = row[2]
            ts = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")

            bars.append(OHLCV(
                timestamp=ts,
                open=float(row[3]),
                high=float(row[4]),
                low=float(row[5]),
                close=float(row[6]),
                volume=float(row[7]) if len(row) > 7 else 0.0,
            ))

            if max_bars and len(bars) >= max_bars:
                break

    elapsed = time.time() - t0
    print(f"OK {len(bars):,} bars in {elapsed:.1f}s")
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Trade simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Completed trade result."""
    signal: StealthSignal
    exit_price: float
    exit_time: datetime
    outcome: str           # "WIN", "LOSS", "TIMEOUT"
    pnl_pips: float
    r_multiple: float
    bars_held: int


def evaluate_trades(
    signals: List[StealthSignal],
    bars: List[OHLCV],
    bar_index_map: Dict[datetime, int],
    timeout_bars: int = 200,
) -> List[Trade]:
    """
    For each signal, walk forward through subsequent bars to see if
    TP or SL is hit first (or timeout).
    """
    trades: List[Trade] = []

    for sig in signals:
        idx = bar_index_map.get(sig.timestamp)
        if idx is None:
            continue

        entry = sig.entry_price
        sl = sig.sl_price
        tp = sig.tp_price
        direction = sig.direction

        risk = abs(entry - sl)
        if risk < 1e-10:
            continue

        outcome = "TIMEOUT"
        exit_price = entry
        exit_time = sig.timestamp
        bars_held = 0

        for j in range(idx + 1, min(idx + 1 + timeout_bars, len(bars))):
            b = bars[j]
            bars_held += 1

            if direction == BUYER:
                # Check SL first (conservative)
                if b.low <= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    exit_time = b.timestamp
                    break
                if b.high >= tp:
                    outcome = "WIN"
                    exit_price = tp
                    exit_time = b.timestamp
                    break
            else:  # SELLER
                if b.high >= sl:
                    outcome = "LOSS"
                    exit_price = sl
                    exit_time = b.timestamp
                    break
                if b.low <= tp:
                    outcome = "WIN"
                    exit_price = tp
                    exit_time = b.timestamp
                    break

        if direction == BUYER:
            pnl_pips = (exit_price - entry) * _pip_mult(sig)
        else:
            pnl_pips = (entry - exit_price) * _pip_mult(sig)

        r_mult = pnl_pips / (risk * _pip_mult(sig)) if risk > 1e-10 else 0.0

        trades.append(Trade(
            signal=sig,
            exit_price=exit_price,
            exit_time=exit_time,
            outcome=outcome,
            pnl_pips=pnl_pips,
            r_multiple=r_mult,
            bars_held=bars_held,
        ))

    return trades


def _pip_mult(sig: StealthSignal) -> float:
    """Rough pip multiplier — for JPY pairs use 100, else 10000."""
    if sig.entry_price > 50:  # JPY pair
        return 100.0
    return 10000.0


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    pair: str,
    trades: List[Trade],
    signals: List[StealthSignal],
    total_bars: int,
    config: StealthDomeConfig,
    elapsed_sec: float,
) -> str:
    """Generate a comprehensive text report."""

    lines: List[str] = []

    def line(s=""):
        lines.append(s)

    def section(title):
        # Use only ASCII characters for Windows console compatibility
        line()
        line(f"{'-' * 70}")
        line(f"  {title}")
        line(f"{'-' * 70}")

    line("=" * 70)
    line("  HORC-CRT STEALTH DOME v1 -- BACKTEST REPORT")
    line("=" * 70)
    line(f"  Pair:          {pair}")
    line(f"  Bars:          {total_bars:,}")
    line(f"  Signals:       {len(signals):,}")
    line(f"  Trades:        {len(trades):,}")
    line(f"  Runtime:       {elapsed_sec:.1f}s")
    line(f"  Config:        LTF={config.ltf_minutes}m | CRT={'ON' if config.crt_active else 'OFF'} "
         f"| MinDiv={config.min_divergence} | R:R={config.risk_reward} | KZ={'ON' if config.use_kill_zones else 'OFF'}")

    if not trades:
        line()
        line("  [!] No trades generated. Try relaxing filters (--no-kz or --min-div 1).")
        return "\n".join(lines)

    # ── Overall stats ────────────────────────────────────────────────────
    section("OVERALL PERFORMANCE")

    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]
    timeouts = [t for t in trades if t.outcome == "TIMEOUT"]

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    total_pnl = sum(t.pnl_pips for t in trades)
    avg_win = sum(t.pnl_pips for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pips for t in losses) / len(losses) if losses else 0
    gross_profit = sum(t.pnl_pips for t in trades if t.pnl_pips > 0)
    gross_loss = abs(sum(t.pnl_pips for t in trades if t.pnl_pips < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_r = sum(t.r_multiple for t in trades) / len(trades) if trades else 0
    avg_bars = sum(t.bars_held for t in trades) / len(trades) if trades else 0

    line(f"  Win Rate:        {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L / {len(timeouts)}T)")
    line(f"  Total PnL:       {total_pnl:+.1f} pips")
    line(f"  Profit Factor:   {profit_factor:.2f}")
    line(f"  Avg Win:         {avg_win:+.1f} pips")
    line(f"  Avg Loss:        {avg_loss:+.1f} pips")
    line(f"  Avg R-Multiple:  {avg_r:+.2f}R")
    line(f"  Avg Bars Held:   {avg_bars:.0f}")

    # ── Equity curve stats ───────────────────────────────────────────────
    section("EQUITY CURVE")

    equity = [0.0]
    for t in trades:
        equity.append(equity[-1] + t.pnl_pips)

    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        dd = peak - e
        max_dd = max(max_dd, dd)

    # Sharpe-like ratio (daily returns approximation)
    if len(equity) > 1:
        returns = [equity[i] - equity[i - 1] for i in range(1, len(equity))]
        mean_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / len(returns)) if len(returns) > 1 else 1
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0
    else:
        sharpe = 0

    line(f"  Max Drawdown:    {max_dd:.1f} pips")
    line(f"  Sharpe Ratio:    {sharpe:.2f}  (annualized, trade-level)")
    line(f"  Final Equity:    {equity[-1]:+.1f} pips")

    # ── Streak analysis ──────────────────────────────────────────────────
    section("STREAK ANALYSIS")

    max_con_wins = max_con_losses = 0
    cur_wins = cur_losses = 0
    for t in trades:
        if t.outcome == "WIN":
            cur_wins += 1
            cur_losses = 0
            max_con_wins = max(max_con_wins, cur_wins)
        elif t.outcome == "LOSS":
            cur_losses += 1
            cur_wins = 0
            max_con_losses = max(max_con_losses, cur_losses)
        else:
            cur_wins = cur_losses = 0

    line(f"  Max Consecutive Wins:   {max_con_wins}")
    line(f"  Max Consecutive Losses: {max_con_losses}")

    # ── Direction split ──────────────────────────────────────────────────
    section("DIRECTION SPLIT")

    longs = [t for t in trades if t.signal.direction == BUYER]
    shorts = [t for t in trades if t.signal.direction == SELLER]

    for label, subset in [("LONG", longs), ("SHORT", shorts)]:
        if not subset:
            line(f"  {label}:  No trades")
            continue
        w = sum(1 for t in subset if t.outcome == "WIN")
        wr = w / len(subset) * 100
        pnl = sum(t.pnl_pips for t in subset)
        line(f"  {label}:  {len(subset)} trades | WR {wr:.1f}% | PnL {pnl:+.1f} pips")

    # ── Kill zone breakdown ──────────────────────────────────────────────
    section("KILL ZONE BREAKDOWN")

    kz_groups: Dict[str, List[Trade]] = defaultdict(list)
    for t in trades:
        kz_groups[t.signal.kill_zone].append(t)

    for kz in ["LONDON", "NY", "ASIA", "OFF"]:
        subset = kz_groups.get(kz, [])
        if not subset:
            continue
        w = sum(1 for t in subset if t.outcome == "WIN")
        wr = w / len(subset) * 100
        pnl = sum(t.pnl_pips for t in subset)
        line(f"  {kz:8s}:  {len(subset):4d} trades | WR {wr:.1f}% | PnL {pnl:+.1f} pips")

    # ── Timeframe activation ─────────────────────────────────────────────
    section("TIMEFRAME ACTIVATION")

    tf_groups: Dict[str, List[Trade]] = defaultdict(list)
    for t in trades:
        tf_groups[t.signal.bias_tf].append(t)

    for tf, subset in sorted(tf_groups.items(), key=lambda x: -len(x[1])):
        w = sum(1 for t in subset if t.outcome == "WIN")
        wr = w / len(subset) * 100
        pnl = sum(t.pnl_pips for t in subset)
        line(f"  TF {tf:>5s}:  {len(subset):4d} trades | WR {wr:.1f}% | PnL {pnl:+.1f} pips")

    # ── Monthly breakdown ────────────────────────────────────────────────
    section("MONTHLY BREAKDOWN (most recent 24 months)")

    monthly: Dict[str, List[Trade]] = defaultdict(list)
    for t in trades:
        key = t.signal.timestamp.strftime("%Y-%m")
        monthly[key].append(t)

    sorted_months = sorted(monthly.keys(), reverse=True)[:24]
    for mo in sorted(sorted_months):
        subset = monthly[mo]
        w = sum(1 for t in subset if t.outcome == "WIN")
        wr = w / len(subset) * 100
        pnl = sum(t.pnl_pips for t in subset)
        line(f"  {mo}:  {len(subset):4d} trades | WR {wr:.1f}% | PnL {pnl:+.1f} pips")

    # ── Calibration recommendations ──────────────────────────────────────
    section("CALIBRATION RECOMMENDATIONS")

    if win_rate < 40:
        line("  [!] Win rate below 40% -- consider:")
        line("       • Increasing min_divergence to require stronger confluence")
        line("       • Tightening kill zone windows")
        line("       • Verifying CRT TF matches instrument volatility")
    elif win_rate > 60:
        line("  [OK] Win rate above 60% -- signals are selective and accurate")
    else:
        line("  [..] Win rate 40-60% -- typical for trend-following with 2R target")

    if profit_factor < 1.0:
        line("  [!] Profit factor < 1.0 -- system is net negative")
        line("       • Consider increasing R:R ratio")
        line("       • Check if SL is too tight (atr_mult too low)")
    elif profit_factor > 1.5:
        line("  [OK] Profit factor > 1.5 -- healthy edge")

    if max_dd > total_pnl * 2:
        line("  [!] Drawdown exceeds 2x total PnL -- risk/reward needs adjustment")

    line()
    line("=" * 70)
    line("  END OF REPORT")
    line("=" * 70)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    pair: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    max_bars: Optional[int] = None,
    config: Optional[StealthDomeConfig] = None,
) -> Tuple[str, List[Trade], List[StealthSignal]]:
    """Run a full backtest for one pair and return (report, trades, signals)."""

    filepath = PAIR_FILES.get(pair.upper())
    if not filepath or not filepath.exists():
        print(f"❌ Data file not found for {pair}. Available: {list(PAIR_FILES.keys())}")
        sys.exit(1)

    cfg = config or StealthDomeConfig()
    engine = StealthDomeEngine(cfg)

    # Load bars
    bars = load_bars(filepath, from_date=from_date, to_date=to_date, max_bars=max_bars)
    if not bars:
        print("❌ No bars loaded. Check date range.")
        sys.exit(1)

    # Build index map
    bar_index_map: Dict[datetime, int] = {b.timestamp: i for i, b in enumerate(bars)}

    # Replay
    print(f"[RUN] Running Stealth Dome on {len(bars):,} bars...", end=" ", flush=True)
    t0 = time.time()

    signals: List[StealthSignal] = []
    for bar in bars:
        sig = engine.process_bar(bar)
        if sig is not None:
            signals.append(sig)

    replay_time = time.time() - t0
    print(f"OK {len(signals):,} signals in {replay_time:.1f}s")

    # Evaluate trades
    print(f"[EVAL] Evaluating trades...", end=" ", flush=True)
    trades = evaluate_trades(signals, bars, bar_index_map, timeout_bars=200)
    print(f"OK {len(trades):,} trades evaluated")

    total_time = time.time() - t0 + replay_time
    report = generate_report(pair, trades, signals, len(bars), cfg, total_time)

    return report, trades, signals


def main():
    parser = argparse.ArgumentParser(description="HORC-CRT Stealth Dome Backtester")
    parser.add_argument("--pair", default="EURUSD", help="Currency pair (EURUSD, GBPUSD, USDJPY). Use ALL for all pairs.")
    parser.add_argument("--from", dest="from_date", help="Start date YYYYMMDD")
    parser.add_argument("--to", dest="to_date", help="End date YYYYMMDD (exclusive)")
    parser.add_argument("--days", type=int, help="Use last N days of data (alternative to --from/--to)")
    parser.add_argument("--max-bars", type=int, help="Max bars to load (safety cap)")
    parser.add_argument("--all-pairs", action="store_true", help="Run backtest on all 3 pairs")

    # Config overrides
    parser.add_argument("--ltf", type=int, default=5, help="LTF raid pulse (minutes)")
    parser.add_argument("--no-crt", action="store_true", help="Disable CRT filter")
    parser.add_argument("--no-kz", action="store_true", help="Disable kill zone filter")
    parser.add_argument("--min-div", type=int, default=2, help="Min divergence score")
    parser.add_argument("--rr", type=float, default=2.0, help="Risk:Reward ratio")
    parser.add_argument("--atr-mult", type=float, default=0.1, help="ATR SL buffer multiplier")
    parser.add_argument("--crt-tf", type=int, default=5, help="CRT candle timeframe (minutes)")
    parser.add_argument("--v88", action="store_true", help="Enable v8.8 Logic Layering (Premium/Discount Filter)")

    args = parser.parse_args()

    # Resolve date range
    from_date = args.from_date
    to_date = args.to_date
    if args.days and not from_date:
        # Approximate: we don't know data end date, just use a recent window
        # The loader will naturally filter
        end = datetime(2025, 12, 31)
        start = end - timedelta(days=args.days)
        from_date = start.strftime("%Y%m%d")
        to_date = end.strftime("%Y%m%d")

    cfg = StealthDomeConfig(
        ltf_minutes=args.ltf,
        crt_active=not args.no_crt,
        crt_tf_minutes=args.crt_tf,
        min_divergence=args.min_div,
        atr_sl_buffer=args.atr_mult,
        use_kill_zones=not args.no_kz,
        risk_reward=args.rr,
        chart_tf_minutes=args.crt_tf if args.crt_tf > 0 else 5,
        use_coordinates_filter=args.v88,
    )

    pairs = list(PAIR_FILES.keys()) if args.all_pairs else [args.pair.upper()]

    for pair in pairs:
        print(f"\n{'=' * 70}")
        print(f"  BACKTESTING: {pair}")
        print(f"{'=' * 70}\n")

        report, trades, signals = run_backtest(
            pair, from_date=from_date, to_date=to_date,
            max_bars=args.max_bars, config=cfg,
        )
        print(report)

        # Save report to file
        report_path = Path(__file__).resolve().parent / f"report_{pair}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n[SAVED] Report saved to {report_path}")


if __name__ == "__main__":
    main()
