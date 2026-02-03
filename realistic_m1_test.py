#!/usr/bin/env python3
"""
Realistic M1 Accuracy Test with Session Markers
Tests HORC on intraday FX data with proper RTH session boundaries
"""
import sys
sys.path.insert(0, '/workspaces/horc-signal')

import pandas as pd
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
    """
    Mark RTH sessions for FX data
    Session: 9:30 AM - 4:00 PM ET (standard equity hours for consistency)
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    
    # Mark RTH: 09:30 to 16:00
    df['is_rth'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | \
                   ((df['hour'] > 9) & (df['hour'] < 16))
    
    # Create session_id for each trading day's RTH period
    df['date'] = df['timestamp'].dt.date
    df['session_id'] = df.apply(
        lambda row: f"{row['date']}_RTH" if row['is_rth'] else '',
        axis=1
    )
    
    return df

def simple_backtest(df, symbol="EURUSD", confluence_threshold=0.75):
    """Run simple backtest with 2R targets and ATR-based stops"""
    # Initialize engines
    participant = ParticipantIdentifier({'opening_range_minutes': 30, 'min_conviction_threshold': 0.5})
    wavelength = WavelengthEngine(WavelengthConfig(min_move_1_size_atr=0.5, max_move_duration_candles=10))
    exhaustion = ExhaustionDetector(ExhaustionConfig(volume_lookback=3, threshold=0.7))
    gap_engine = FuturesGapEngine(GapConfig(min_gap_size_percent=0.001, gap_fill_tolerance=0.5))
    
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=confluence_threshold,
        participant_weight=0.30,
        wavelength_weight=0.25,
        exhaustion_weight=0.25,
        gap_weight=0.20,
    )
    orchestrator = HORCOrchestrator(
        participant,
        wavelength,
        exhaustion,
        gap_engine,
        orchestrator_config
    )
    
    signals_generated = 0
    trades = []
    position = None
    atr_period = 14
    
    # Debug counters
    actionable_count = 0
    bias_nonzero_count = 0
    total_signals = 0
    
    # Calculate ATR for stops
    df['tr'] = df[['high', 'low']].diff(axis=1).abs()['low'].fillna(0)
    df['atr'] = df['tr'].rolling(atr_period).mean().bfill()
    
    print(f"📊 Processing {len(df):,} bars...")
    
    # Group by session
    sessions = df[df['session_id'] != ''].groupby('session_id')
    print(f"   Sessions found: {len(sessions)}")
    
    prev_session_candles = []
    
    for session_id, session_df in sessions:
        session_candles = []
        
        for idx, row in session_df.iterrows():
            # Build candle
            candle = Candle(
                timestamp=row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']) if pd.notna(row['volume']) else 0
            )
            session_candles.append(candle)
        
        # Skip first session (need prev session for participant)
        if not prev_session_candles:
            prev_session_candles = session_candles
            continue
        
        # Set prev session for participant
        participant.prev_session_candles = prev_session_candles
        
        # Process each bar in this session
        for i, candle in enumerate(session_candles):
            # Get participant candles (all bars so far in this session)
            participant_candles = session_candles[:i+1]
            
            # Process bar with participant context
            signal = orchestrator.process_bar(
                candle,
                participant_candles=participant_candles
            )
            
            # Debug tracking
            total_signals += 1
            if signal.actionable:
                actionable_count += 1
            if signal.bias != 0:
                bias_nonzero_count += 1
            
            # Check for signal
            if signal and signal.bias != 0 and signal.actionable:
                signals_generated += 1
                
                # Get row for ATR
                row = session_df.iloc[i]
                
                # Open position if flat
                if position is None:
                    entry_price = candle.close
                    stop_distance = row['atr'] * 1.5  # 1.5 ATR stop
                    
                    if signal.bias == 1:  # Long
                        stop_loss = entry_price - stop_distance
                        target = entry_price + (2 * stop_distance)  # 2R target
                    else:  # Short (bias == -1)
                        stop_loss = entry_price + stop_distance
                        target = entry_price - (2 * stop_distance)
                    
                    position = {
                        'side': 'LONG' if signal.bias == 1 else 'SHORT',
                        'entry': entry_price,
                        'stop': stop_loss,
                        'target': target,
                        'entry_time': candle.timestamp,
                        'cps': signal.confidence
                    }
            
            # Check exit conditions
            if position:
                row = session_df.iloc[i]
                hit_stop = False
                hit_target = False
                
                if position['side'] == 'LONG':
                    hit_stop = candle.low <= position['stop']
                    hit_target = candle.high >= position['target']
                else:  # SHORT
                    hit_stop = candle.high >= position['stop']
                    hit_target = candle.low <= position['target']
                
                if hit_target:
                    exit_price = position['target']
                    pnl = 2.0  # 2R win
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': candle.timestamp,
                        'side': position['side'],
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl_r': pnl,
                        'cps': position['cps'],
                        'outcome': 'WIN'
                    })
                    position = None
                
                elif hit_stop:
                    exit_price = position['stop']
                    pnl = -1.0  # 1R loss
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': candle.timestamp,
                        'side': position['side'],
                        'entry': position['entry'],
                        'exit': exit_price,
                        'pnl_r': pnl,
                        'cps': position['cps'],
                        'outcome': 'LOSS'
                    })
                    position = None
        
        # Save this session for next iteration
        prev_session_candles = session_candles
    
    print(f"\n🔍 DEBUG INFO:")
    print(f"   Total bars processed: {total_signals:,}")
    print(f"   Actionable signals: {actionable_count}")
    print(f"   Non-zero bias: {bias_nonzero_count}")
    print(f"   Both (tradeable): {signals_generated}")
    
    return signals_generated, trades

def main():
    print("=" * 70)
    print("🔬 HORC M1 Accuracy Test - Session-Aware")
    print("=" * 70)
    
    # Load EURUSD M1 data
    print("\n📥 Loading EURUSD M1 data...")
    df = pd.read_csv('/workspaces/horc-signal/data/EURUSD_M1_RTH.csv', nrows=50000)  # Sample first 50k bars
    
    print(f"✅ Loaded EURUSD M1 sample")
    print(f"   Bars: {len(df):,}")
    
    # Add session markers
    print("\n🕐 Adding RTH session markers (09:30 - 16:00 ET)...")
    df = add_session_markers(df)
    
    # Filter to RTH only for meaningful test
    df_rth = df[df['is_rth']].copy().reset_index(drop=True)
    print(f"   RTH bars: {len(df_rth):,}")
    
    # Test on first month of RTH data
    test_size = min(20000, len(df_rth))  # ~1 month of RTH bars
    df_test = df_rth.head(test_size).copy()
    
    date_range = pd.to_datetime(df_test['timestamp'])
    print(f"   Range: {date_range.min()} to {date_range.max()}")
    print(f"   Period: {(date_range.max() - date_range.min()).days} days\n")
    
    print(f"🎯 Testing on first {test_size:,} RTH bars (~1 month)")
    print()
    
    # Run backtest
    signals, trades = simple_backtest(df_test, symbol="EURUSD", confluence_threshold=0.50)  # Lower threshold
    
    # Results
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Total Signals Generated: {signals}")
    print(f"Total Trades Executed: {len(trades)}")
    
    if trades:
        wins = [t for t in trades if t['outcome'] == 'WIN']
        losses = [t for t in trades if t['outcome'] == 'LOSS']
        total_r = sum(t['pnl_r'] for t in trades)
        win_rate = len(wins) / len(trades) * 100
        
        print(f"\n✅ TRADE PERFORMANCE")
        print(f"   Wins: {len(wins)}")
        print(f"   Losses: {len(losses)}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total P&L: {total_r:+.2f}R")
        print(f"   Avg P&L per Trade: {total_r/len(trades):+.3f}R")
        
        # Show first 10 trades
        print(f"\n📋 Sample Trades (first 10):")
        print(f"{'Entry':<20} {'Side':<6} {'CPS':<5} {'Outcome':<6} {'P&L':<8}")
        print("-" * 70)
        for t in trades[:10]:
            print(f"{str(t['entry_time']):<20} {t['side']:<6} {t['cps']:<5.2f} {t['outcome']:<6} {t['pnl_r']:+.2f}R")
    else:
        print("\n⚠️  NO TRADES EXECUTED")
        print("   Possible reasons:")
        print("   • Confluence threshold too high (0.75)")
        print("   • Sample period lacks clear setups")
        print("   • Opening range not establishing properly")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
