#!/usr/bin/env python3
"""
HORC Bias Voting Inspector - See why bias stays at 0

Shows which engines are voting and what direction
"""
import sys
sys.path.insert(0, '/workspaces/horc-signal')

import pandas as pd
from src.core.orchestrator import HORCOrchestrator, OrchestratorConfig
from src.engines import (
    Candle,
    ParticipantIdentifier,
    ParticipantType,
    WavelengthEngine,
    WavelengthConfig,
    WavelengthState,
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
    print("=" * 100)
    print("🗳️  HORC BIAS VOTING INSPECTOR - Understanding Why Bias = 0")
    print("=" * 100)
    
    df = pd.read_csv('/workspaces/horc-signal/data/EURUSD_M1_RTH.csv', nrows=10000)
    df = add_session_markers(df)
    df_rth = df[df['is_rth']].copy().reset_index(drop=True)
    
    print(f"\n✅ Loaded {len(df_rth):,} RTH bars\n")
    
    # Ultra-permissive
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
        require_agreement=False  # Disable majority vote requirement
    )
    
    orchestrator = HORCOrchestrator(participant, wavelength, exhaustion, gap_engine, orchestrator_config)
    
    sessions = df_rth[df_rth['session_id'] != ''].groupby('session_id')
    prev_session_candles = []
    
    samples = []
    
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
            
            # Get individual results
            participant_res = participant.identify(participant_candles)
            wavelength_res = wavelength.process_candle(candle, participant_res)
            
            # Determine votes manually
            p_vote = "NEUTRAL"
            if participant_res.participant_type == ParticipantType.BUYERS:
                p_vote = "LONG"
            elif participant_res.participant_type == ParticipantType.SELLERS:
                p_vote = "SHORT"
            
            w_vote = "NEUTRAL"
            w_state = WavelengthState(wavelength_res.state).name if hasattr(wavelength_res, 'state') else "UNKNOWN"
            if wavelength_res.state in [WavelengthState.MOVE_1.value, WavelengthState.MOVE_3.value]:
                if wavelength_res.move_1_extreme and wavelength_res.move_2_extreme:
                    if wavelength_res.move_1_extreme > wavelength_res.move_2_extreme:
                        w_vote = "LONG"
                    else:
                        w_vote = "SHORT"
            elif wavelength_res.state == WavelengthState.MOVE_2.value:
                if wavelength_res.move_1_extreme and wavelength_res.move_2_extreme:
                    if wavelength_res.move_1_extreme > wavelength_res.move_2_extreme:
                        w_vote = "SHORT"
                    else:
                        w_vote = "LONG"
            
            if len(samples) < 30:
                samples.append({
                    'timestamp': candle.timestamp,
                    'p_type': participant_res.participant_type.name if hasattr(participant_res.participant_type, 'name') else str(participant_res.participant_type),
                    'p_vote': p_vote,
                    'w_state': w_state,
                    'w_vote': w_vote,
                    'w_m1': wavelength_res.move_1_extreme,
                    'w_m2': wavelength_res.move_2_extreme
                })
        
        prev_session_candles = session_candles
        
        if len(samples) >= 30:
            break
    
    print("📋 VOTING BREAKDOWN (first 30 bars):\n")
    print(f"{'Timestamp':<25}{'Participant':<18}{'Vote':<10}{'Wavelength':<20}{'Vote':<10}")
    print("-" * 100)
    
    for s in samples:
        print(f"{str(s['timestamp']):<25}{s['p_type']:<18}{s['p_vote']:<10}{s['w_state']:<20}{s['w_vote']:<10}")
    
    # Summary
    p_votes = [s['p_vote'] for s in samples]
    w_votes = [s['w_vote'] for s in samples]
    
    print("\n" + "=" * 100)
    print("📊 VOTING SUMMARY")
    print("=" * 100)
    print(f"\nParticipant Engine Votes:")
    print(f"  LONG: {p_votes.count('LONG')} ({p_votes.count('LONG')/len(p_votes)*100:.1f}%)")
    print(f"  SHORT: {p_votes.count('SHORT')} ({p_votes.count('SHORT')/len(p_votes)*100:.1f}%)")
    print(f"  NEUTRAL: {p_votes.count('NEUTRAL')} ({p_votes.count('NEUTRAL')/len(p_votes)*100:.1f}%)")
    
    print(f"\nWavelength Engine Votes:")
    print(f"  LONG: {w_votes.count('LONG')} ({w_votes.count('LONG')/len(w_votes)*100:.1f}%)")
    print(f"  SHORT: {w_votes.count('SHORT')} ({w_votes.count('SHORT')/len(w_votes)*100:.1f}%)")
    print(f"  NEUTRAL: {w_votes.count('NEUTRAL')} ({w_votes.count('NEUTRAL')/len(w_votes)*100:.1f}%)")
    
    print("\n💡 KEY INSIGHT:")
    if p_votes.count('NEUTRAL') > len(p_votes) * 0.8:
        print("   ❌ Participant engine stuck in NEUTRAL - not detecting BUYERS/SELLERS")
        print("      → Check: ORH/ORL sweep detection logic")
    if w_votes.count('NEUTRAL') > len(w_votes) * 0.8:
        print("   ❌ Wavelength engine not progressing to voteable states")
        print("      → Check: Move1/Move2 detection, state progression")
    
    print("=" * 100)

if __name__ == "__main__":
    main()
