#!/usr/bin/env python3
"""
HORC Fine-Tuning - Find optimal threshold for rare, high-accuracy signals
"""
import sys
sys.path.insert(0, '/workspaces/horc-signal')

import pandas as pd
import numpy as np
import json

from src.core.orchestrator import HORCOrchestrator, OrchestratorConfig
from src.engines import Candle, ParticipantIdentifier, WavelengthEngine, WavelengthConfig, ExhaustionDetector, ExhaustionConfig, FuturesGapEngine, GapConfig

def add_session_markers(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['is_rth'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] > 9) & (df['hour'] < 16))
    df['date'] = df['timestamp'].dt.date
    df['session_id'] = df.apply(lambda row: f"{row['date']}_RTH" if row['is_rth'] else '', axis=1)
    return df

def backtest(df, confluence_threshold, participant_conviction):
    participant = ParticipantIdentifier({'opening_range_minutes': 30, 'min_conviction_threshold': participant_conviction})
    wavelength = WavelengthEngine(WavelengthConfig(min_move_1_size_atr=0.5, max_move_duration_candles=15))
    exhaustion = ExhaustionDetector(ExhaustionConfig(volume_lookback=3, threshold=0.7))
    gap_engine = FuturesGapEngine(GapConfig(min_gap_size_percent=0.001, gap_fill_tolerance=0.5))
    
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=confluence_threshold,
        participant_weight=0.50,
        wavelength_weight=0.20,
        exhaustion_weight=0.20,
        gap_weight=0.10,
        require_agreement=False,
        require_strategic_context=False
    )
    
    orchestrator = HORCOrchestrator(participant, wavelength, exhaustion, gap_engine, orchestrator_config)
    
    signals = 0
    trades = []
    position = None
    
    df['tr'] = df[['high', 'low']].diff(axis=1).abs()['low'].fillna(0)
    df['atr'] = df['tr'].rolling(14).mean().bfill()
    
    sessions = df[df['session_id'] != ''].groupby('session_id')
    prev_session_candles = []
    
    for session_id, session_df in sessions:
        session_candles = []
        for idx, row in session_df.iterrows():
            candle = Candle(timestamp=row['timestamp'], open=float(row['open']), high=float(row['high']), low=float(row['low']), close=float(row['close']), volume=int(row['volume']) if pd.notna(row['volume']) else 0)
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
                    
                    position = {'side': 'LONG' if signal.bias == 1 else 'SHORT', 'entry': entry_price, 'stop': stop_loss, 'target': target, 'confidence': signal.confidence}
            
            if position:
                row = session_df.iloc[i]
                
                if position['side'] == 'LONG':
                    hit_stop = candle.low <= position['stop']
                    hit_target = candle.high >= position['target']
                else:
                    hit_stop = candle.high >= position['stop']
                    hit_target = candle.low <= position['target']
                
                if hit_target:
                    trades.append({'pnl_r': 2.0, 'outcome': 'WIN', 'confidence': position['confidence']})
                    position = None
                elif hit_stop:
                    trades.append({'pnl_r': -1.0, 'outcome': 'LOSS', 'confidence': position['confidence']})
                    position = None
        
        prev_session_candles = session_candles
    
    if not trades:
        return {'signals': signals, 'trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'avg_pnl': 0.0}
    
    wins = [t for t in trades if t['outcome'] == 'WIN']
    return {'signals': signals, 'trades': len(trades), 'wins': len(wins), 'losses': len(trades) - len(wins), 'win_rate': len(wins) / len(trades) * 100, 'total_pnl': sum(t['pnl_r'] for t in trades), 'avg_pnl': sum(t['pnl_r'] for t in trades) / len(trades)}

def main():
    print("=" * 90)
    print("🎯 HORC FINE-TUNING - Optimizing for Quality Over Quantity")
    print("=" * 90)
    
    df = pd.read_csv('/workspaces/horc-signal/data/EURUSD_M1_RTH.csv', nrows=200000)
    df = add_session_markers(df)
    df_rth = df[df['is_rth']].copy().reset_index(drop=True)
    
    date_range = pd.to_datetime(df_rth['timestamp'])
    days = (date_range.max() - date_range.min()).days
    
    print(f"\n📊 Data: {len(df_rth):,} RTH bars over {days} days\n")
    
    thresholds = [(0.40, 0.4), (0.45, 0.4), (0.50, 0.5), (0.55, 0.5), (0.60, 0.5), (0.65, 0.6), (0.70, 0.6)]
    
    print(f"{'Conf':<8}{'Conv':<8}{'Signals':<10}{'Trades':<10}{'WR%':<8}{'Avg R':<10}{'Total R':<10}{'Sig/Week':<10}")
    print("-" * 90)
    
    best = None
    
    for conf_thresh, conviction in thresholds:
        result = backtest(df_rth, conf_thresh, conviction)
        sig_per_week = result['signals'] / days * 7
        
        print(f"{conf_thresh:<8.2f}{conviction:<8.2f}{result['signals']:<10}{result['trades']:<10}{result['win_rate']:<8.1f}{result['avg_pnl']:<10.3f}{result['total_pnl']:<10.2f}{sig_per_week:<10.1f}")
        
        if result['win_rate'] > 50 and result['avg_pnl'] > 0:
            if best is None or result['avg_pnl'] > best['avg_pnl']:
                best = {'conf': conf_thresh, 'conv': conviction, **result}
    
    print("\n" + "=" * 90)
    
    if best:
        print(f"🏆 OPTIMAL: Conf={best['conf']:.2f} Conv={best['conv']:.2f}")
        print(f"   Win Rate: {best['win_rate']:.1f}% | Avg R: {best['avg_pnl']:+.3f} | Trades: {best['trades']}")
        config = {'confluence_threshold': best['conf'], 'participant_conviction': best['conv'], 'performance': best}
        with open('/workspaces/horc-signal/results/optimal_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✅ Saved to results/optimal_config.json")
    else:
        print("⚠️  No configuration achieved >50% win rate with positive expectancy")
    
    print("=" * 90)

if __name__ == "__main__":
    main()
