#!/usr/bin/env python3
"""
HORC Smart Calibration - Filter based on time-of-day and session structure
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
    df['minutes_from_930'] = (df['hour'] - 9) * 60 + df['minute'] - 30
    df['is_rth'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] > 9) & (df['hour'] < 16))
    df['date'] = df['timestamp'].dt.date
    df['session_id'] = df.apply(lambda row: f"{row['date']}_RTH" if row['is_rth'] else '', axis=1)
    return df

def backtest_with_filters(df, signal_window_start, signal_window_end, require_sweep, min_range_size):
    """
    Filter signals by time window and opening range quality
    signal_window_start/end: Minutes from 9:30 (e.g., 30 = 10:00 AM)
    require_sweep: Only trade when there's an OR sweep
    min_range_size: Minimum ATR multiplier for opening range
    """
    participant = ParticipantIdentifier({'opening_range_minutes': 30, 'min_conviction_threshold': 0.3})
    wavelength = WavelengthEngine(WavelengthConfig(min_move_1_size_atr=0.5, max_move_duration_candles=15))
    exhaustion = ExhaustionDetector(ExhaustionConfig(volume_lookback=3, threshold=0.7))
    gap_engine = FuturesGapEngine(GapConfig(min_gap_size_percent=0.001, gap_fill_tolerance=0.5))
    
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=0.30,
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
    
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                    abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean().bfill()
    
    sessions = df[df['session_id'] != ''].groupby('session_id')
    prev_session_candles = []
    
    for session_id, session_df in sessions:
        session_candles = []
        session_df_reset = session_df.reset_index(drop=True)
        
        for idx, row in session_df_reset.iterrows():
            candle = Candle(
                timestamp=row['timestamp'], 
                open=float(row['open']), high=float(row['high']), 
                low=float(row['low']), close=float(row['close']), 
                volume=int(row['volume']) if pd.notna(row['volume']) else 0
            )
            session_candles.append(candle)
        
        if not prev_session_candles:
            prev_session_candles = session_candles
            continue
        
        participant.prev_session_candles = prev_session_candles
        
        # Calculate opening range size (first 30 candles = 30 min)
        if len(session_candles) >= 30:
            or_candles = session_candles[:30]
            or_high = max(c.high for c in or_candles)
            or_low = min(c.low for c in or_candles)
            or_size = or_high - or_low
            session_atr = session_df_reset.iloc[30]['atr'] if len(session_df_reset) > 30 else session_df_reset['atr'].mean()
            or_size_atr = or_size / session_atr if session_atr > 0 else 0
        else:
            or_size_atr = 0
        
        for i, candle in enumerate(session_candles):
            minutes_from_open = i  # Each candle is 1 min
            participant_candles = session_candles[:i+1]
            
            # Reset orchestrator wavelength state for each session
            if i == 0:
                orchestrator.wavelength.reset()
            
            signal = orchestrator.process_bar(candle, participant_candles=participant_candles)
            
            if signal and signal.bias != 0 and signal.actionable:
                row = session_df_reset.iloc[i]
                
                # FILTER 1: Time window
                in_time_window = signal_window_start <= minutes_from_open <= signal_window_end
                
                # FILTER 2: Opening range size (quality filter)
                range_quality = or_size_atr >= min_range_size
                
                # FILTER 3: Sweep requirement
                has_sweep = participant.prev_session_candles and i >= 30  # After OR complete
                
                if in_time_window and range_quality and (not require_sweep or has_sweep):
                    signals += 1
                    
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
                            'confidence': signal.confidence,
                            'or_size': or_size_atr,
                            'time': minutes_from_open
                        }
            
            if position:
                if position['side'] == 'LONG':
                    hit_stop = candle.low <= position['stop']
                    hit_target = candle.high >= position['target']
                else:
                    hit_stop = candle.high >= position['stop']
                    hit_target = candle.low <= position['target']
                
                if hit_target:
                    trades.append({
                        'pnl_r': 2.0, 'outcome': 'WIN', 
                        'confidence': position['confidence'],
                        'or_size': position['or_size'],
                        'entry_time': position['time']
                    })
                    position = None
                elif hit_stop:
                    trades.append({
                        'pnl_r': -1.0, 'outcome': 'LOSS',
                        'confidence': position['confidence'],
                        'or_size': position['or_size'],
                        'entry_time': position['time']
                    })
                    position = None
        
        prev_session_candles = session_candles
    
    if not trades:
        return {'signals': signals, 'trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'avg_pnl': 0.0}
    
    wins = [t for t in trades if t['outcome'] == 'WIN']
    return {
        'signals': signals, 
        'trades': len(trades), 
        'wins': len(wins), 
        'losses': len(trades) - len(wins), 
        'win_rate': len(wins) / len(trades) * 100, 
        'total_pnl': sum(t['pnl_r'] for t in trades), 
        'avg_pnl': sum(t['pnl_r'] for t in trades) / len(trades)
    }

def main():
    print("=" * 100)
    print("🎯 HORC SMART CALIBRATION - Time-Based & Quality Filtering")
    print("=" * 100)
    
    df = pd.read_csv('/workspaces/horc-signal/data/EURUSD_M1_RTH.csv', nrows=200000)
    df = add_session_markers(df)
    df_rth = df[df['is_rth']].copy().reset_index(drop=True)
    
    date_range = pd.to_datetime(df_rth['timestamp'])
    days = (date_range.max() - date_range.min()).days
    
    print(f"\n📊 Data: {len(df_rth):,} RTH bars over {days} days\n")
    
    # Test configurations: (window_start, window_end, require_sweep, min_range_atr)
    configs = [
        # After OR (30 min) - traditional setup
        (30, 60, False, 0.5),   # 10:00-10:30, min OR 0.5 ATR
        (30, 60, False, 1.0),   # 10:00-10:30, min OR 1.0 ATR
        (30, 60, False, 1.5),   # 10:00-10:30, min OR 1.5 ATR
        (30, 90, False, 1.0),   # 10:00-11:00, min OR 1.0 ATR
        (30, 120, False, 1.0),  # 10:00-11:30, min OR 1.0 ATR
        # First 2 hours only
        (30, 120, False, 0.5),  # 10:00-11:30, relaxed range
        (30, 120, False, 1.5),  # 10:00-11:30, tight range
        # Morning session only
        (30, 180, False, 1.0),  # 10:00-12:30
        # All day (control)
        (30, 360, False, 0.5),  # 10:00-3:30, relaxed
    ]
    
    print(f"{'Start':<8}{'End':<8}{'MinOR':<8}{'Signals':<10}{'Trades':<10}{'WR%':<8}{'AvgR':<10}{'TotR':<10}{'T/Wk':<8}")
    print("-" * 100)
    
    best = None
    
    for start, end, req_sweep, min_or in configs:
        result = backtest_with_filters(df_rth, start, end, req_sweep, min_or)
        trades_per_week = result['trades'] / days * 7 if days > 0 else 0
        
        start_time = f"{9 + start // 60}:{start % 60:02d}"
        end_time = f"{9 + end // 60}:{end % 60:02d}"
        
        marker = ""
        if result['win_rate'] > 50 and result['avg_pnl'] > 0:
            marker = " ⭐"
            if best is None or result['avg_pnl'] > best['avg_pnl']:
                best = {
                    'start': start, 'end': end, 'min_or': min_or, 
                    **result
                }
        
        print(f"{start_time:<8}{end_time:<8}{min_or:<8.1f}{result['signals']:<10}{result['trades']:<10}{result['win_rate']:<8.1f}{result['avg_pnl']:<10.3f}{result['total_pnl']:<10.2f}{trades_per_week:<8.1f}{marker}")
    
    print("\n" + "=" * 100)
    
    if best:
        start_t = f"{9 + best['start'] // 60}:{best['start'] % 60:02d}"
        end_t = f"{9 + best['end'] // 60}:{best['end'] % 60:02d}"
        print(f"🏆 OPTIMAL: Window {start_t}-{end_t}, Min OR Size: {best['min_or']} ATR")
        print(f"   Win Rate: {best['win_rate']:.1f}% | Avg R: {best['avg_pnl']:+.3f} | Trades: {best['trades']} | {best['trades']/days*7:.1f}/week")
        
        config = {
            'signal_window_start': best['start'],
            'signal_window_end': best['end'],
            'min_opening_range_atr': best['min_or'],
            'confluence_threshold': 0.30,
            'performance': {
                'win_rate': best['win_rate'],
                'avg_pnl_r': best['avg_pnl'],
                'total_trades': best['trades'],
                'trades_per_week': best['trades'] / days * 7
            }
        }
        with open('/workspaces/horc-signal/results/optimal_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✅ Saved to results/optimal_config.json")
    else:
        print("⚠️  No configuration achieved >50% win rate with positive expectancy")
        print("   Consider adjusting stop/target ratios or using trailing stops")
    
    print("=" * 100)

if __name__ == "__main__":
    main()
