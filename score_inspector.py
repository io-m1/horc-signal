#!/usr/bin/env python3
"""
HORC Score Inspector - See actual confidence scores from each engine

This shows the real problem: engines detect but scores are too low to meet confluence
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
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['is_rth'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] > 9) & (df['hour'] < 16))
    df['date'] = df['timestamp'].dt.date
    df['session_id'] = df.apply(lambda row: f"{row['date']}_RTH" if row['is_rth'] else '', axis=1)
    return df

def main():
    print("=" * 90)
    print("📊 HORC SCORE INSPECTOR - Analyzing Engine Confidence Levels")
    print("=" * 90)
    
    df = pd.read_csv('/workspaces/horc-signal/data/EURUSD_M1_RTH.csv', nrows=10000)
    df = add_session_markers(df)
    df_rth = df[df['is_rth']].copy().reset_index(drop=True)
    
    print(f"\n✅ Loaded {len(df_rth):,} RTH bars\n")
    
    # Ultra-permissive settings
    participant = ParticipantIdentifier({'opening_range_minutes': 30, 'min_conviction_threshold': 0.1})
    wavelength = WavelengthEngine(WavelengthConfig(min_move_1_size_atr=0.1, max_move_duration_candles=50))
    exhaustion = ExhaustionDetector(ExhaustionConfig(volume_lookback=3, threshold=0.3))
    gap_engine = FuturesGapEngine(GapConfig(min_gap_size_percent=0.001, gap_fill_tolerance=0.5))
    
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=0.10,
        participant_weight=0.25,
        wavelength_weight=0.25,
        exhaustion_weight=0.25,
        gap_weight=0.25,
    )
    
    orchestrator = HORCOrchestrator(participant, wavelength, exhaustion, gap_engine, orchestrator_config)
    
    sessions = df_rth[df_rth['session_id'] != ''].groupby('session_id')
    prev_session_candles = []
    
    sample_signals = []
    
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
            
            # Collect samples
            if len(sample_signals) < 20:
                sample_signals.append({
                    'timestamp': candle.timestamp,
                    'bias': signal.bias,
                    'confidence': signal.confidence,
                    'actionable': signal.actionable
                })
        
        prev_session_candles = session_candles
        
        if len(sample_signals) >= 20:
            break
    
    print("📋 SAMPLE SIGNALS (first 20 bars after OR):\n")
    print(f"{'Timestamp':<25}{'Bias':<8}{'Confidence':<15}{'Actionable':<12}")
    print("-" * 90)
    
    for s in sample_signals:
        bias_str = "LONG" if s['bias'] == 1 else ("SHORT" if s['bias'] == -1 else "NEUTRAL")
        action_str = "YES" if s['actionable'] else "NO"
        print(f"{str(s['timestamp']):<25}{bias_str:<8}{s['confidence']:<15.4f}{action_str:<12}")
    
    # Summary
    avg_conf = sum(s['confidence'] for s in sample_signals) / len(sample_signals)
    max_conf = max(s['confidence'] for s in sample_signals)
    actionable_count = sum(1 for s in sample_signals if s['actionable'])
    
    print("\n" + "=" * 90)
    print("📊 SUMMARY")
    print("=" * 90)
    print(f"Average Confidence: {avg_conf:.4f}")
    print(f"Maximum Confidence: {max_conf:.4f}")
    print(f"Confluence Threshold: {orchestrator_config.confluence_threshold:.2f}")
    print(f"Actionable Signals: {actionable_count} / {len(sample_signals)}")
    
    print("\n💡 KEY INSIGHT:")
    if max_conf < orchestrator_config.confluence_threshold:
        print(f"   ❌ Max confidence ({max_conf:.4f}) never reaches threshold ({orchestrator_config.confluence_threshold:.2f})")
        print(f"   → Solution: Lower confluence_threshold to ~{max_conf * 0.8:.2f} or")
        print(f"   → Solution: Increase engine weights to boost confidence calculation")
    else:
        print(f"   ✅ Confidence CAN reach threshold but rarely does")
        print(f"   → This is normal selective behavior")
    
    print("=" * 90)

if __name__ == "__main__":
    main()
