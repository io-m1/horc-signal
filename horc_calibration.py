#!/usr/bin/env python3
"""
HORC Parameter Optimization - Calibrate for Maximum Accuracy

Objective: Find parameters that generate rare, ultra-high-precision signals
Testing: 6 strategic configurations on real historical data
"""
import sys
sys.path.insert(0, '/workspaces/horc-signal')

import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

from src.core.orchestrator import HORCOrchestrator, OrchestratorConfig
from src.engines import (
    Candle,
    ParticipantIdentifier,
    WavelengthEngine,
    WavelengthConfig,
    ExhaustionDetector,
    ExhaustionConfig,
    FuturesGapEngine,
    GapConfig,
)

@dataclass
class CalibrationConfig:
    name: str
    confluence_threshold: float
    participant_weight: float
    wavelength_weight: float
    exhaustion_weight: float
    gap_weight: float
    participant_conviction: float
    wavelength_move1_atr: float
    exhaustion_threshold: float

@dataclass
class TestResult:
    config: CalibrationConfig
    signals: int
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_r: float
    avg_pnl_r: float
    max_drawdown_r: float
    sharpe_ratio: float
    quality_score: float

def add_session_markers(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['is_rth'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] > 9) & (df['hour'] < 16))
    df['date'] = df['timestamp'].dt.date
    df['session_id'] = df.apply(lambda row: f"{row['date']}_RTH" if row['is_rth'] else '', axis=1)
    return df

def backtest_with_config(df, config: CalibrationConfig) -> TestResult:
    participant = ParticipantIdentifier({'opening_range_minutes': 30, 'min_conviction_threshold': config.participant_conviction})
    wavelength = WavelengthEngine(WavelengthConfig(min_move_1_size_atr=config.wavelength_move1_atr, max_move_duration_candles=20))
    exhaustion = ExhaustionDetector(ExhaustionConfig(volume_lookback=3, threshold=config.exhaustion_threshold))
    gap_engine = FuturesGapEngine(GapConfig(min_gap_size_percent=0.001, gap_fill_tolerance=0.5))
    
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=config.confluence_threshold,
        participant_weight=config.participant_weight,
        wavelength_weight=config.wavelength_weight,
        exhaustion_weight=config.exhaustion_weight,
        gap_weight=config.gap_weight,
    )
    
    orchestrator = HORCOrchestrator(participant, wavelength, exhaustion, gap_engine, orchestrator_config)
    
    signals = 0
    trades = []
    position = None
    equity_curve = [0.0]
    
    df['tr'] = df[['high', 'low']].diff(axis=1).abs()['low'].fillna(0)
    df['atr'] = df['tr'].rolling(14).mean().bfill()
    
    sessions = df[df['session_id'] != ''].groupby('session_id')
    prev_session_candles = []
    
    for session_id, session_df in sessions:
        session_candles = []
        for idx, row in session_df.iterrows():
            candle = Candle(
                timestamp=row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']) if pd.notna(row['volume']) else 0
            )
            session_candles.append(candle)
        
        if not prev_session_candles:
            prev_session_candles = session_candles
            continue
        
        participant.prev_session_candles = prev_session_candles
        
        for i, candle in enumerate(session_candles):
            participant_candles = session_candles[:i+1]
            signal = orchestrator.process_bar(candle, participant_candles=participant_candles)
            
            if signal and signal.bias != 0 and signal.actionable:
                signals += 1
                row = session_df.iloc[i]
                
                if position is None:
                    entry_price = candle.close
                    stop_distance = row['atr'] * 1.5
                    
                    if signal.bias == 1:
                        stop_loss = entry_price - stop_distance
                        target = entry_price + (2 * stop_distance)
                    else:
                        stop_loss = entry_price + stop_distance
                        target = entry_price - (2 * stop_distance)
                    
                    position = {
                        'side': 'LONG' if signal.bias == 1 else 'SHORT',
                        'entry': entry_price,
                        'stop': stop_loss,
                        'target': target,
                        'entry_time': candle.timestamp,
                        'confidence': signal.confidence
                    }
            
            if position:
                row = session_df.iloc[i]
                
                if position['side'] == 'LONG':
                    hit_stop = candle.low <= position['stop']
                    hit_target = candle.high >= position['target']
                else:
                    hit_stop = candle.high >= position['stop']
                    hit_target = candle.low <= position['target']
                
                if hit_target:
                    trades.append({'entry_time': position['entry_time'], 'exit_time': candle.timestamp, 'side': position['side'], 'pnl_r': 2.0, 'confidence': position['confidence'], 'outcome': 'WIN'})
                    equity_curve.append(equity_curve[-1] + 2.0)
                    position = None
                elif hit_stop:
                    trades.append({'entry_time': position['entry_time'], 'exit_time': candle.timestamp, 'side': position['side'], 'pnl_r': -1.0, 'confidence': position['confidence'], 'outcome': 'LOSS'})
                    equity_curve.append(equity_curve[-1] - 1.0)
                    position = None
        
        prev_session_candles = session_candles
    
    if not trades:
        return TestResult(config=config, signals=signals, trades=0, wins=0, losses=0, win_rate=0.0, total_pnl_r=0.0, avg_pnl_r=0.0, max_drawdown_r=0.0, sharpe_ratio=0.0, quality_score=0.0)
    
    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    win_rate = len(wins) / len(trades) * 100
    total_pnl = sum(t['pnl_r'] for t in trades)
    avg_pnl = total_pnl / len(trades)
    
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    
    returns = [t['pnl_r'] for t in trades]
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    
    quality_score = (win_rate / 100.0) * max(avg_pnl, 0) * np.sqrt(len(trades)) - (max_dd * 0.1)
    
    return TestResult(config=config, signals=signals, trades=len(trades), wins=len(wins), losses=len(losses), win_rate=win_rate, total_pnl_r=total_pnl, avg_pnl_r=avg_pnl, max_drawdown_r=max_dd, sharpe_ratio=sharpe, quality_score=quality_score)

def generate_configs():
    return [
        CalibrationConfig("Participant-Heavy", 0.40, 0.50, 0.20, 0.20, 0.10, 0.3, 0.3, 0.6),
        CalibrationConfig("Wavelength-Heavy", 0.45, 0.25, 0.50, 0.15, 0.10, 0.4, 0.4, 0.7),
        CalibrationConfig("Exhaustion-Heavy", 0.40, 0.25, 0.20, 0.45, 0.10, 0.4, 0.3, 0.5),
        CalibrationConfig("Balanced-Low", 0.35, 0.30, 0.25, 0.25, 0.20, 0.3, 0.3, 0.6),
        CalibrationConfig("Ultra-Selective", 0.75, 0.30, 0.25, 0.25, 0.20, 0.5, 0.5, 0.7),
        CalibrationConfig("Aggressive-Entry", 0.30, 0.40, 0.30, 0.20, 0.10, 0.2, 0.2, 0.5),
    ]

def main():
    print("=" * 90)
    print("🎯 HORC PARAMETER CALIBRATION - Optimize for Quality Signals")
    print("=" * 90)
    print("\n📋 Goal: Find optimal parameters for rare, high-precision signals\n")
    
    print("📥 Loading EURUSD M1 data...")
    df = pd.read_csv('/workspaces/horc-signal/data/EURUSD_M1_RTH.csv', nrows=200000)
    print(f"✅ Loaded {len(df):,} bars\n")
    
    print("🕐 Processing RTH sessions...")
    df = add_session_markers(df)
    df_rth = df[df['is_rth']].copy().reset_index(drop=True)
    date_range = pd.to_datetime(df_rth['timestamp'])
    print(f"✅ RTH bars: {len(df_rth):,} | Range: {date_range.min()} to {date_range.max()} ({(date_range.max() - date_range.min()).days} days)\n")
    
    configs = generate_configs()
    print(f"🔬 Testing {len(configs)} strategic configurations...\n")
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {config.name:<22} | Conf:{config.confluence_threshold:.2f} P:{config.participant_weight:.2f} W:{config.wavelength_weight:.2f} E:{config.exhaustion_weight:.2f} G:{config.gap_weight:.2f}", end=" ", flush=True)
        result = backtest_with_config(df_rth, config)
        results.append(result)
        print(f"→ Signals:{result.signals} Trades:{result.trades} WR:{result.win_rate:.1f}% AvgR:{result.avg_pnl_r:+.3f}")
    
    results.sort(key=lambda x: x.quality_score, reverse=True)
    
    print("\n" + "=" * 90)
    print("📊 RESULTS (Ranked by Quality Score)")
    print("=" * 90)
    print(f"\n{'Rank':<6}{'Configuration':<22}{'Trades':<8}{'WR%':<8}{'Avg R':<10}{'Total R':<10}{'Quality':<10}")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r.config.name:<22}{r.trades:<8}{r.win_rate:<8.1f}{r.avg_pnl_r:<10.3f}{r.total_pnl_r:<10.2f}{r.quality_score:<10.3f}")
    
    best = results[0]
    print("\n" + "=" * 90)
    print("🏆 OPTIMAL CONFIGURATION")
    print("=" * 90)
    print(f"\nName: {best.config.name}\n")
    print("Parameters:")
    print(f"  confluence_threshold = {best.config.confluence_threshold:.2f}")
    print(f"  participant_weight = {best.config.participant_weight:.2f}")
    print(f"  wavelength_weight = {best.config.wavelength_weight:.2f}")
    print(f"  exhaustion_weight = {best.config.exhaustion_weight:.2f}")
    print(f"  gap_weight = {best.config.gap_weight:.2f}")
    print(f"  participant_conviction = {best.config.participant_conviction:.2f}")
    print(f"  wavelength_move1_atr = {best.config.wavelength_move1_atr:.2f}")
    print(f"  exhaustion_threshold = {best.config.exhaustion_threshold:.2f}")
    print(f"\nPerformance:")
    print(f"  Signals: {best.signals} | Trades: {best.trades}")
    print(f"  Win Rate: {best.win_rate:.1f}% | Avg R: {best.avg_pnl_r:+.3f} | Total R: {best.total_pnl_r:+.2f}")
    print(f"  Max DD: {best.max_drawdown_r:.2f}R | Sharpe: {best.sharpe_ratio:.2f} | Quality: {best.quality_score:.3f}")
    
    best_dict = {'name': best.config.name, 'confluence_threshold': best.config.confluence_threshold, 'participant_weight': best.config.participant_weight, 'wavelength_weight': best.config.wavelength_weight, 'exhaustion_weight': best.config.exhaustion_weight, 'gap_weight': best.config.gap_weight, 'participant_conviction': best.config.participant_conviction, 'wavelength_move1_atr': best.config.wavelength_move1_atr, 'exhaustion_threshold': best.config.exhaustion_threshold, 'performance': {'signals': best.signals, 'trades': best.trades, 'win_rate': best.win_rate, 'avg_pnl_r': best.avg_pnl_r, 'total_pnl_r': best.total_pnl_r, 'max_drawdown_r': best.max_drawdown_r, 'sharpe_ratio': best.sharpe_ratio, 'quality_score': best.quality_score}}
    
    with open('/workspaces/horc-signal/results/optimal_config.json', 'w') as f:
        json.dump(best_dict, f, indent=2)
    
    print(f"\n✅ Optimal configuration saved to results/optimal_config.json")
    print("=" * 90)

if __name__ == "__main__":
    main()
