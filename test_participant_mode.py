#!/usr/bin/env python3
"""
HORC Participant-Only Mode - Use OR Sweep Detection Alone

This configuration lets Participant engine drive bias without requiring
wavelength/gap votes. This is the most reliable signal on FX data.
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

def add_session_markers(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['is_rth'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] > 9) & (df['hour'] < 16))
    df['date'] = df['timestamp'].dt.date
    df['session_id'] = df.apply(lambda row: f"{row['date']}_RTH" if row['is_rth'] else '', axis=1)
    return df

def backtest_participant_mode(df, confluence_threshold=0.30, participant_conviction=0.3):
    """Run backtest with Participant-driven bias"""
    
    participant = ParticipantIdentifier({'opening_range_minutes': 30, 'min_conviction_threshold': participant_conviction})
    wavelength = WavelengthEngine(WavelengthConfig(min_move_1_size_atr=0.3, max_move_duration_candles=20))
    exhaustion = ExhaustionDetector(ExhaustionConfig(volume_lookback=3, threshold=0.6))
    gap_engine = FuturesGapEngine(GapConfig(min_gap_size_percent=0.001, gap_fill_tolerance=0.5))
    
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=confluence_threshold,
        participant_weight=0.50,  # Participant-heavy
        wavelength_weight=0.20,
        exhaustion_weight=0.20,
        gap_weight=0.10,
        require_agreement=False  # KEY: Allow single-engine bias
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
        return {'signals': signals, 'trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'avg_pnl': 0.0, 'max_dd': 0.0, 'sharpe': 0.0}
    
    wins = [t for t in trades if t['outcome'] == 'WIN']
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
    
    return {'signals': signals, 'trades': len(trades), 'wins': len(wins), 'losses': len(trades) - len(wins), 'win_rate': win_rate, 'total_pnl': total_pnl, 'avg_pnl': avg_pnl, 'max_dd': max_dd, 'sharpe': sharpe}

def main():
    print("=" * 90)
    print("🎯 HORC PARTICIPANT-ONLY MODE - OR Sweep Detection")
    print("=" * 90)
    print("\nConfiguration: Participant engine drives bias without requiring confluence")
    print("Strategy: Trade Opening Range sweeps with exhaustion confirmation\n")
    
    print("📥 Loading EURUSD M1 data...")
    df = pd.read_csv('/workspaces/horc-signal/data/EURUSD_M1_RTH.csv', nrows=200000)
    print(f"✅ Loaded {len(df):,} bars\n")
    
    print("🕐 Processing RTH sessions...")
    df = add_session_markers(df)
    df_rth = df[df['is_rth']].copy().reset_index(drop=True)
    date_range = pd.to_datetime(df_rth['timestamp'])
    print(f"✅ RTH bars: {len(df_rth):,} | Range: {date_range.min()} to {date_range.max()} ({(date_range.max() - date_range.min()).days} days)\n")
    
    print("🔬 Testing Participant-Only Mode...\n")
    
    result = backtest_participant_mode(df_rth, confluence_threshold=0.30, participant_conviction=0.3)
    
    print("=" * 90)
    print("📊 RESULTS - Participant-Driven Signals")
    print("=" * 90)
    print(f"\nTotal Signals: {result['signals']}")
    print(f"Total Trades: {result['trades']}")
    
    if result['trades'] > 0:
        print(f"Wins: {result['wins']} | Losses: {result['losses']}")
        print(f"Win Rate: {result['win_rate']:.1f}%")
        print(f"Average P&L: {result['avg_pnl']:+.3f}R")
        print(f"Total P&L: {result['total_pnl']:+.2f}R")
        print(f"Max Drawdown: {result['max_dd']:.2f}R")
        print(f"Sharpe Ratio: {result['sharpe']:.2f}")
        
        print("\n✅ SUCCESS: HORC now generates tradeable signals!")
        print(f"   → {result['trades']} trades over {(date_range.max() - date_range.min()).days} days")
        print(f"   → {result['win_rate']:.1f}% win rate with {result['avg_pnl']:+.3f}R average")
        
        # Save config
        config = {
            'name': 'Participant-Only Mode',
            'confluence_threshold': 0.30,
            'participant_weight': 0.50,
            'wavelength_weight': 0.20,
            'exhaustion_weight': 0.20,
            'gap_weight': 0.10,
            'require_agreement': False,
            'participant_conviction': 0.3,
            'wavelength_move1_atr': 0.3,
            'exhaustion_threshold': 0.6,
            'performance': result
        }
        
        with open('/workspaces/horc-signal/results/participant_only_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuration saved to results/participant_only_config.json")
    else:
        print("\n⚠️  Still no trades - investigating further...")
    
    print("=" * 90)

if __name__ == "__main__":
    main()
