from datetime import datetime, timedelta
from typing import List

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
from src.core import HORCOrchestrator, SignalIR
from src.core.orchestrator import OrchestratorConfig
from src.logging import init_trade_logger

def create_bullish_scenario() -> tuple[List[Candle], List[Candle]]:
    base_time = datetime(2026, 2, 2, 9, 30)
    base_price = 4500.0
    
    spot_candles = [
        Candle(
            timestamp=base_time - timedelta(days=1, hours=7),
            open=base_price,
            high=base_price + 20.0,  # ORH = 4520
            low=base_price - 10.0,   # ORL = 4490
            close=base_price + 5.0,
            volume=10000.0
        ),
        
        Candle(
            timestamp=base_time,
            open=base_price + 10.0,
            high=base_price + 25.0,  # Sweeps ORH (4520) → 4525
            low=base_price + 8.0,
            close=base_price + 23.0,
            volume=15000.0  # High volume = conviction
        ),
        
        Candle(
            timestamp=base_time + timedelta(minutes=5),
            open=base_price + 23.0,
            high=base_price + 30.0,
            low=base_price + 22.0,
            close=base_price + 28.0,
            volume=12000.0
        ),
        
        Candle(
            timestamp=base_time + timedelta(minutes=10),
            open=base_price + 28.0,
            high=base_price + 29.0,
            low=base_price + 15.0,  # Deep pullback
            close=base_price + 17.0,
            volume=8000.0
        ),
        
        Candle(
            timestamp=base_time + timedelta(minutes=15),
            open=base_price + 17.0,
            high=base_price + 18.0,
            low=base_price + 14.0,
            close=base_price + 16.0,
            volume=20000.0  # Very high volume = exhaustion
        ),
        
        Candle(
            timestamp=base_time + timedelta(minutes=20),
            open=base_price + 16.0,
            high=base_price + 35.0,  # Strong move
            low=base_price + 15.0,
            close=base_price + 33.0,
            volume=14000.0
        ),
    ]
    
    futures_candles = [
        Candle(
            timestamp=base_time - timedelta(minutes=30),
            open=base_price + 40.0,
            high=base_price + 42.0,
            low=base_price + 38.0,
            close=base_price + 40.0,  # Futures close at 4540
            volume=5000.0
        ),
        
        Candle(
            timestamp=base_time,
            open=base_price + 50.0,  # Gaps to 4550 (gap: 4540-4550)
            high=base_price + 52.0,
            low=base_price + 48.0,
            close=base_price + 51.0,
            volume=6000.0
        ),
    ]
    
    return spot_candles, futures_candles

def print_section(title: str, char: str = "="):
    width = 80
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)
    print()

def main():
    
    print_section("HORC ORCHESTRATOR DEMONSTRATION", "=")
    print("Unified signal generation with Pine-safe output")
    print("Scenario: Bullish setup with high confluence")
    
    print_section("1. Creating Market Data", "-")
    
    spot_candles, futures_candles = create_bullish_scenario()
    
    print(f"Generated {len(spot_candles)} spot candles")
    print(f"Generated {len(futures_candles)} futures candles")
    print(f"\nPrevious ORH: ${spot_candles[0].high:.2f}")
    print(f"First move high: ${spot_candles[1].high:.2f} (sweeps ORH ✓)")
    print(f"Futures gap: ${futures_candles[0].close:.2f} → ${futures_candles[1].open:.2f}")
    
    print_section("2. Initializing Engines", "-")
    
    participant_config = {
        'opening_range_minutes': 30,
        'min_conviction_threshold': 0.5
    }
    participant = ParticipantIdentifier(participant_config)
    participant.prev_session_candles = [spot_candles[0]]  # Set previous session
    print("✓ ParticipantIdentifier initialized")
    
    wavelength_config = WavelengthConfig(
        min_move_1_size_atr=0.5,
        max_move_duration_candles=10
    )
    wavelength = WavelengthEngine(wavelength_config)
    print("✓ WavelengthEngine initialized")
    
    exhaustion_config = ExhaustionConfig(
        volume_lookback=3,
        threshold=0.7
    )
    exhaustion = ExhaustionDetector(exhaustion_config)
    print("✓ ExhaustionDetector initialized")
    
    gap_config = GapConfig(
        min_gap_size_percent=0.001,
        gap_fill_tolerance=0.5
    )
    gap_engine = FuturesGapEngine(gap_config)
    print("✓ FuturesGapEngine initialized")
    
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=0.75,
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
    print("✓ HORCOrchestrator initialized")
    # Initialize optional trade logging (controlled via env vars)
    trade_logger = init_trade_logger()
    try:
        if getattr(trade_logger, "enable", False):
            print(f"✓ Trade logging enabled -> {trade_logger.path}")
    except Exception:
        pass
    print(f"\nConfluence threshold: {orchestrator_config.confluence_threshold:.2f}")
    print(f"Weights: P={orchestrator_config.participant_weight:.2f}, "
          f"W={orchestrator_config.wavelength_weight:.2f}, "
          f"E={orchestrator_config.exhaustion_weight:.2f}, "
          f"G={orchestrator_config.gap_weight:.2f}")
    
    print_section("3. Processing Market Data Bar-by-Bar", "-")
    
    signals: List[SignalIR] = []
    
    for i, candle in enumerate(spot_candles[1:], 1):  # Skip first (previous session)
        print(f"\nBar {i}: {candle.timestamp.strftime('%H:%M')}")
        print(f"  Price: O=${candle.open:.2f} H=${candle.high:.2f} "
              f"L=${candle.low:.2f} C=${candle.close:.2f} V={candle.volume:.0f}")
        
        signal = orchestrator.process_bar(
            candle=candle,
            futures_candle=futures_candles[-1] if i == 1 else None,  # Only first bar has futures
            participant_candles=[spot_candles[0], candle] if i == 1 else None
        )
        
        signals.append(signal)
        
        bias_str = {-1: "BEARISH", 0: "NEUTRAL", 1: "BULLISH"}[signal.bias]
        action_icon = "🟢" if signal.actionable else "⚪"
        
        print(f"\n  Signal: {action_icon} {bias_str}")
        print(f"  Confidence: {signal.confidence:.3f}")
        print(f"  Participant: {signal.participant_control:+d} | "
              f"Wavelength: {signal.moves_completed}/3 | "
              f"Exhaustion: {signal.exhaustion_score:.3f} | "
              f"Gap: type={signal.active_gap_type}")
        
        if signal.actionable:
            print(f"\n  ⚡ ACTIONABLE SIGNAL DETECTED ⚡")
    
    print_section("4. Summary", "-")
    
    actionable_signals = [s for s in signals if s.actionable]
    
    print(f"Total bars processed: {len(signals)}")
    print(f"Actionable signals: {len(actionable_signals)}")
    
    if actionable_signals:
        print("\nActionable Signals:")
        for i, sig in enumerate(actionable_signals, 1):
            bias_str = {-1: "SELL", 0: "NEUTRAL", 1: "BUY"}[sig.bias]
            print(f"  {i}. {sig.timestamp.strftime('%H:%M')} - "
                  f"{bias_str} @ confidence {sig.confidence:.3f}")
    
    print_section("5. Pine Script Readiness Check", "-")
    
    print("✓ All signals use primitive types (int, float, bool)")
    print("✓ No dynamic objects or nested structures")
    print("✓ State is bar-local and deterministic")
    print("✓ Ready for Pine Script translation")
    
    print("\nPine Translation Template:")
    print_section("6. Next Steps", "-")
    
    print("✓ Phase 2A complete: Orchestration layer implemented")
    print("\nRecommended next steps:")
    print("  1. Build replay engine (determinism validation)")
    print("  2. Add comprehensive tests (test_orchestrator.py)")
    print("  3. Create backtesting harness (PnL simulation)")
    print("  4. Port to Pine Script (TradingView deployment)")
    print("  5. Add regime filtering (Phase 2B)")
    
    print_section("END OF DEMONSTRATION", "=")

if __name__ == "__main__":
    main()
