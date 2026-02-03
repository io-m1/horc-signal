#!/usr/bin/env python3
"""
HORC Engine Diagnostic - Debug why no signals are generated

This script inspects each engine's internal state to understand
why confluence conditions are never met on real market data.
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
    print("🔍 HORC ENGINE DIAGNOSTIC - Understanding Signal Generation")
    print("=" * 90)
    print("\nLoading data...")
    
    df = pd.read_csv('/workspaces/horc-signal/data/EURUSD_M1_RTH.csv', nrows=10000)
    df = add_session_markers(df)
    df_rth = df[df['is_rth']].copy().reset_index(drop=True)
    
    print(f"✅ Loaded {len(df_rth):,} RTH bars\n")
    
    # Initialize with most permissive settings
    participant = ParticipantIdentifier({'opening_range_minutes': 30, 'min_conviction_threshold': 0.1})
    wavelength = WavelengthEngine(WavelengthConfig(min_move_1_size_atr=0.1, max_move_duration_candles=50))
    exhaustion = ExhaustionDetector(ExhaustionConfig(volume_lookback=3, threshold=0.3))
    gap_engine = FuturesGapEngine(GapConfig(min_gap_size_percent=0.001, gap_fill_tolerance=0.5))
    
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=0.10,  # Extremely low
        participant_weight=0.25,
        wavelength_weight=0.25,
        exhaustion_weight=0.25,
        gap_weight=0.25,
    )
    
    orchestrator = HORCOrchestrator(participant, wavelength, exhaustion, gap_engine, orchestrator_config)
    
    # Engine state tracking
    participant_detections = 0
    wavelength_detections = 0
    exhaustion_detections = 0
    gap_detections = 0
    actionable_signals = 0
    
    sessions = df_rth[df_rth['session_id'] != ''].groupby('session_id')
    prev_session_candles = []
    
    total_bars = 0
    
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
            
            # Get individual engine results before orchestrator
            participant_res = participant.identify(participant_candles)
            wavelength_res = wavelength.process_candle(candle, participant_res)
            exhaustion_res = exhaustion.detect_exhaustion(participant_candles, direction="LONG")
            
            # Track detections
            if participant_res.participant_type != 0:  # Not NONE
                participant_detections += 1
            
            if wavelength_res.state != 0:  # Not INIT/PRE_OR
                wavelength_detections += 1
            
            if exhaustion_res.threshold_met:
                exhaustion_detections += 1
            
            # Process through orchestrator
            signal = orchestrator.process_bar(candle, participant_candles=participant_candles)
            
            if signal.actionable:
                actionable_signals += 1
            
            total_bars += 1
        
        prev_session_candles = session_candles
        
        # Stop after 5 sessions for quick diagnostic
        if sessions.ngroups >= 5:
            session_count = sum(1 for _ in sessions)
            if session_count >= 5:
                break
    
    print(f"📊 ENGINE DIAGNOSTICS (first {total_bars:,} bars)")
    print("=" * 90)
    print(f"\nParticipant Engine:")
    print(f"  Detections: {participant_detections} ({participant_detections/total_bars*100:.2f}%)")
    print(f"  Status: {'✅ Active' if participant_detections > 0 else '❌ No detections'}")
    
    print(f"\nWavelength Engine:")
    print(f"  State Changes: {wavelength_detections} ({wavelength_detections/total_bars*100:.2f}%)")
    print(f"  Status: {'✅ Active' if wavelength_detections > 0 else '❌ No state progression'}")
    
    print(f"\nExhaustion Engine:")
    print(f"  Exhaustion Zones: {exhaustion_detections} ({exhaustion_detections/total_bars*100:.2f}%)")
    print(f"  Status: {'✅ Active' if exhaustion_detections > 0 else '❌ No exhaustion detected'}")
    
    print(f"\nOrchestrator:")
    print(f"  Actionable Signals: {actionable_signals} ({actionable_signals/total_bars*100:.2f}%)")
    print(f"  Status: {'✅ Generating signals' if actionable_signals > 0 else '❌ No actionable signals'}")
    
    print("\n" + "=" * 90)
    
    if actionable_signals == 0:
        print("\n💡 DIAGNOSIS:")
        if participant_detections == 0:
            print("   ❌ Participant engine not detecting OR sweeps")
            print("      → Check: ORH/ORL from prev session, sweep detection logic")
        if wavelength_detections == 0:
            print("   ❌ Wavelength engine not progressing beyond INIT state")
            print("      → Check: Move1 size requirements, absorption detection")
        if exhaustion_detections == 0:
            print("   ❌ Exhaustion engine not finding exhaustion zones")
            print("      → Check: Volume/displacement ratio calculation")
        if all([participant_detections > 0, wavelength_detections > 0, exhaustion_detections > 0]):
            print("   ⚠️  Engines detecting individually but confluence not met")
            print("      → Check: Timing alignment, weight calculations")
    
    print("=" * 90)

if __name__ == "__main__":
    results = []
    main()
