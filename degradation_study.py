"""Degradation Study: Systematic Axiom Removal
Measures contribution of each axiom by disabling them one-by-one and comparing performance.

Usage:
    python3 degradation_study.py
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

from src.engines import (
    ParticipantIdentifier,
    WavelengthEngine,
    WavelengthConfig,
    ExhaustionDetector,
    ExhaustionConfig,
    FuturesGapEngine,
    GapConfig,
    Candle,
)
from src.core import HORCOrchestrator
from src.core.orchestrator import OrchestratorConfig


@dataclass
class BacktestResult:
    config_name: str
    total_trades: int
    win_rate: float
    avg_rr: float
    sharpe: float
    total_return: float
    
    def __repr__(self):
        return (f"{self.config_name}: {self.total_trades} trades, "
                f"WR={self.win_rate:.1%}, AvgRR={self.avg_rr:.2f}, "
                f"Sharpe={self.sharpe:.2f}, Return={self.total_return:.1%}")


def simple_backtest(data: pd.DataFrame, config: OrchestratorConfig, 
                   disable_participant=False, disable_wavelength=False,
                   disable_exhaustion=False, disable_gaps=False) -> BacktestResult:
    """Run lightweight backtest with optional axiom disabling."""
    
    # Initialize engines
    part = ParticipantIdentifier()
    wave = WavelengthEngine(WavelengthConfig())
    exh = ExhaustionDetector(ExhaustionConfig())
    gap = FuturesGapEngine(GapConfig())
    
    # Override weights if disabling
    if disable_participant:
        config.participant_weight = 0.0
    if disable_wavelength:
        config.wavelength_weight = 0.0
    if disable_exhaustion:
        config.exhaustion_weight = 0.0
    if disable_gaps:
        config.gap_weight = 0.0
    
    # Normalize weights to sum to 1.0
    total_w = (config.participant_weight + config.wavelength_weight + 
               config.exhaustion_weight + config.gap_weight)
    if total_w > 0:
        config.participant_weight /= total_w
        config.wavelength_weight /= total_w
        config.exhaustion_weight /= total_w
        config.gap_weight /= total_w
    
    orch = HORCOrchestrator(part, wave, exh, gap, config)
    
    # Build candles from close series
    closes = data["close"].values
    idx = list(data.index)
    candles = []
    prev = None
    for t, c in zip(idx, closes):
        o = float(prev) if prev is not None else float(c)
        h = max(o, float(c)) + 0.2
        l = min(o, float(c)) - 0.2
        vol = 1000.0
        candle = Candle(timestamp=t.to_pydatetime(), open=o, high=h, 
                       low=l, close=float(c), volume=vol)
        candles.append(candle)
        prev = c
    
    # Simple trade sim
    in_pos = False
    pos_dir = 0
    entry = 0.0
    stop = 0.0
    target = 0.0
    trades = []
    
    ranges = [c.high - c.low for c in candles]
    avg_range = max(0.001, np.mean(ranges))
    
    for c in candles:
        sig = orch.process_bar(candle=c)
        
        if not in_pos and sig.actionable:
            entry = c.close
            pos_dir = sig.bias
            if pos_dir > 0:
                stop = entry - 2.0 * avg_range
                target = entry + 2.0 * (entry - stop)
            else:
                stop = entry + 2.0 * avg_range
                target = entry - 2.0 * (stop - entry)
            in_pos = True
        
        elif in_pos:
            if pos_dir > 0:
                if c.high >= target:
                    rr = (target - entry) / max(1e-6, abs(entry - stop))
                    trades.append(rr)
                    in_pos = False
                elif c.low <= stop:
                    trades.append(-1.0)
                    in_pos = False
            else:
                if c.low <= target:
                    rr = (entry - target) / max(1e-6, abs(entry - stop))
                    trades.append(rr)
                    in_pos = False
                elif c.high >= stop:
                    trades.append(-1.0)
                    in_pos = False
    
    # Compute metrics
    if not trades:
        return BacktestResult(
            config_name="Unknown",
            total_trades=0,
            win_rate=0.0,
            avg_rr=0.0,
            sharpe=0.0,
            total_return=0.0
        )
    
    trades_arr = np.array(trades)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr < 0]
    
    win_rate = len(wins) / len(trades)
    avg_rr = trades_arr.mean()
    sharpe = (trades_arr.mean() / trades_arr.std()) * np.sqrt(252) if trades_arr.std() > 0 else 0
    total_return = trades_arr.sum()
    
    return BacktestResult(
        config_name="Unknown",
        total_trades=len(trades),
        win_rate=win_rate,
        avg_rr=avg_rr,
        sharpe=sharpe,
        total_return=total_return
    )


def run_degradation_study():
    """Run degradation study on all axioms."""
    
    print("=" * 80)
    print("  🔬 HORC DEGRADATION STUDY")
    print("  Systematic Axiom Removal Analysis")
    print("=" * 80)
    print()
    
    # Load data
    import os
    csv_path = "data/GBPUSD_M1_H4.csv"
    
    if not os.path.exists(csv_path):
        print("❌ Data file not found. Generate synthetic data for demo...")
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="4h")
        data = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        }, index=dates)
    else:
        print(f"✅ Loading data from {csv_path}")
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        data = data[['close']]
    
    print(f"   Bars: {len(data)}")
    print(f"   Range: {data.index.min().date()} to {data.index.max().date()}")
    print()
    
    # Baseline config
    base_config = OrchestratorConfig(
        confluence_threshold=0.75,
        participant_weight=0.30,
        wavelength_weight=0.25,
        exhaustion_weight=0.25,
        gap_weight=0.20
    )
    
    results = []
    
    print("🔄 Running configurations...")
    print()
    
    # Baseline (all axioms)
    print("   1/5 Baseline (all axioms enabled)...")
    r = simple_backtest(data, OrchestratorConfig(**base_config.__dict__))
    r.config_name = "Baseline (All)"
    results.append(r)
    
    # Disable Participant
    print("   2/5 Without Participant axiom...")
    r = simple_backtest(data, OrchestratorConfig(**base_config.__dict__), 
                       disable_participant=True)
    r.config_name = "No Participant"
    results.append(r)
    
    # Disable Wavelength
    print("   3/5 Without Wavelength axiom...")
    r = simple_backtest(data, OrchestratorConfig(**base_config.__dict__), 
                       disable_wavelength=True)
    r.config_name = "No Wavelength"
    results.append(r)
    
    # Disable Exhaustion
    print("   4/5 Without Exhaustion axiom...")
    r = simple_backtest(data, OrchestratorConfig(**base_config.__dict__), 
                       disable_exhaustion=True)
    r.config_name = "No Exhaustion"
    results.append(r)
    
    # Disable Gaps
    print("   5/5 Without Gaps axiom...")
    r = simple_backtest(data, OrchestratorConfig(**base_config.__dict__), 
                       disable_gaps=True)
    r.config_name = "No Gaps"
    results.append(r)
    
    print()
    print("=" * 80)
    print("  📊 RESULTS")
    print("=" * 80)
    print()
    
    baseline = results[0]
    
    print(f"{'Configuration':<20} {'Trades':<8} {'WinRate':<10} {'AvgRR':<8} {'Sharpe':<8} {'Return':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r.config_name:<20} {r.total_trades:<8} {r.win_rate:>8.1%} "
              f"{r.avg_rr:>7.2f} {r.sharpe:>7.2f} {r.total_return:>9.1%}")
    
    print()
    print("=" * 80)
    print("  📉 DEGRADATION ANALYSIS")
    print("=" * 80)
    print()
    
    if baseline.sharpe == 0:
        print("⚠️  Baseline Sharpe is 0 — no actionable signals generated.")
        print("   This may indicate:")
        print("   • Data lacks opening range context (intraday required)")
        print("   • Confluence threshold too high")
        print("   • Insufficient volatility for signal generation")
        print()
        print("   Recommendation: Use M1 or M5 data with session markers.")
    else:
        print(f"Baseline Sharpe: {baseline.sharpe:.2f}")
        print()
        
        for r in results[1:]:
            delta = baseline.sharpe - r.sharpe
            pct = (delta / baseline.sharpe * 100) if baseline.sharpe != 0 else 0
            
            impact = "🔴 CRITICAL" if abs(pct) > 50 else "🟡 MODERATE" if abs(pct) > 25 else "🟢 MINOR"
            
            print(f"{r.config_name:<20} Sharpe: {r.sharpe:>6.2f} | "
                  f"Delta: {delta:>+6.2f} ({pct:>+5.1f}%) | {impact}")
    
    print()
    print("=" * 80)
    print("  ✅ DEGRADATION STUDY COMPLETE")
    print("=" * 80)
    print()
    
    return results


if __name__ == "__main__":
    run_degradation_study()
