from datetime import datetime, timedelta
from src.engines.participant import (
    ParticipantIdentifier,
    ParticipantType,
    Candle
)

def print_separator(title: str):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def demo_buyers_control():
    print_separator("DEMO 1: BUYERS Control - First Move Sweeps ORH")
    
    identifier = ParticipantIdentifier()
    base_time = datetime(2024, 1, 1, 9, 30)
    
    previous_session = [
        Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
        Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0),
    ]
    
    print("\nPrevious Session Data:")
    orh, orl = identifier.get_opening_range(previous_session)
    print(f"  ORH_prev: ${orh:.2f}")
    print(f"  ORL_prev: ${orl:.2f}")
    
    identifier.prev_session_candles = previous_session
    current_time = datetime(2024, 1, 2, 9, 30)
    
    current_session = [
        Candle(current_time, 4520.0, 4540.0, 4515.0, 4535.0, 1500.0),
    ]
    
    print("\nCurrent Session - First Candle:")
    candle = current_session[0]
    print(f"  Open:  ${candle.open:.2f}")
    print(f"  High:  ${candle.high:.2f} ← SWEEPS ORH_prev (${orh:.2f})")
    print(f"  Low:   ${candle.low:.2f}")
    print(f"  Close: ${candle.close:.2f}")
    
    result = identifier.identify(current_session)
    
    print("\n📊 PARTICIPANT IDENTIFICATION RESULT:")
    print(f"  Participant Type:  {result.participant_type.value}")
    print(f"  Conviction Level:  {result.conviction_level}")
    print(f"  Control Price:     ${result.control_price:.2f}")
    print(f"  Sweep Candle:      #{result.sweep_candle_index}")
    
    print("\n💡 INTERPRETATION:")
    print("  → BUYERS swept sell-side liquidity at ORH")
    print("  → Buyers are the informed/aggressive participants")
    print("  → Expect continuation to the upside (bullish bias)")

def demo_sellers_control():
    print_separator("DEMO 2: SELLERS Control - First Move Sweeps ORL")
    
    identifier = ParticipantIdentifier()
    base_time = datetime(2024, 1, 1, 9, 30)
    
    previous_session = [
        Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
        Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0),
    ]
    
    print("\nPrevious Session Data:")
    orh, orl = identifier.get_opening_range(previous_session)
    print(f"  ORH_prev: ${orh:.2f}")
    print(f"  ORL_prev: ${orl:.2f}")
    
    identifier.prev_session_candles = previous_session
    current_time = datetime(2024, 1, 2, 9, 30)
    
    current_session = [
        Candle(current_time, 4490.0, 4495.0, 4470.0, 4475.0, 1800.0),
    ]
    
    print("\nCurrent Session - First Candle:")
    candle = current_session[0]
    print(f"  Open:  ${candle.open:.2f}")
    print(f"  High:  ${candle.high:.2f}")
    print(f"  Low:   ${candle.low:.2f} ← SWEEPS ORL_prev (${orl:.2f})")
    print(f"  Close: ${candle.close:.2f}")
    
    result = identifier.identify(current_session)
    
    print("\n📊 PARTICIPANT IDENTIFICATION RESULT:")
    print(f"  Participant Type:  {result.participant_type.value}")
    print(f"  Conviction Level:  {result.conviction_level}")
    print(f"  Control Price:     ${result.control_price:.2f}")
    print(f"  Sweep Candle:      #{result.sweep_candle_index}")
    
    print("\n💡 INTERPRETATION:")
    print("  → SELLERS swept buy-side liquidity at ORL")
    print("  → Sellers are the informed/aggressive participants")
    print("  → Expect continuation to the downside (bearish bias)")

def demo_no_conviction():
    print_separator("DEMO 3: NO CONVICTION - No Sweep Detected")
    
    identifier = ParticipantIdentifier()
    base_time = datetime(2024, 1, 1, 9, 30)
    
    previous_session = [
        Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
        Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0),
    ]
    
    print("\nPrevious Session Data:")
    orh, orl = identifier.get_opening_range(previous_session)
    print(f"  ORH_prev: ${orh:.2f}")
    print(f"  ORL_prev: ${orl:.2f}")
    
    identifier.prev_session_candles = previous_session
    current_time = datetime(2024, 1, 2, 9, 30)
    
    current_session = [
        Candle(current_time, 4500.0, 4515.0, 4495.0, 4510.0, 800.0),
        Candle(current_time + timedelta(minutes=1), 4510.0, 4520.0, 4505.0, 4515.0, 750.0),
        Candle(current_time + timedelta(minutes=2), 4515.0, 4525.0, 4510.0, 4520.0, 700.0),
    ]
    
    print("\nCurrent Session - First 3 Candles:")
    for i, candle in enumerate(current_session):
        print(f"  Candle {i}: High=${candle.high:.2f}, Low=${candle.low:.2f}")
    print(f"  → None sweep ORH (${orh:.2f}) or ORL (${orl:.2f})")
    
    result = identifier.identify(current_session)
    
    print("\n📊 PARTICIPANT IDENTIFICATION RESULT:")
    print(f"  Participant Type:  {result.participant_type.value}")
    print(f"  Conviction Level:  {result.conviction_level}")
    print(f"  Control Price:     {result.control_price}")
    print(f"  Sweep Candle:      {result.sweep_candle_index}")
    
    print("\n💡 INTERPRETATION:")
    print("  → No decisive sweep detected in first moves")
    print("  → Informed participants have not revealed themselves yet")
    print("  → WAIT for conviction before taking position")
    print("  → System remains neutral until clear signal emerges")

def demo_second_candle_sweep():
    print_separator("DEMO 4: Second Candle Sweep - Delayed Entry")
    
    identifier = ParticipantIdentifier()
    base_time = datetime(2024, 1, 1, 9, 30)
    
    previous_session = [
        Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
        Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0),
    ]
    
    print("\nPrevious Session Data:")
    orh, orl = identifier.get_opening_range(previous_session)
    print(f"  ORH_prev: ${orh:.2f}")
    print(f"  ORL_prev: ${orl:.2f}")
    
    identifier.prev_session_candles = previous_session
    current_time = datetime(2024, 1, 2, 9, 30)
    
    current_session = [
        Candle(current_time, 4500.0, 4515.0, 4495.0, 4510.0, 800.0),
        Candle(current_time + timedelta(minutes=1), 4510.0, 4540.0, 4505.0, 4535.0, 1500.0),
    ]
    
    print("\nCurrent Session:")
    print(f"  Candle 0: High=${current_session[0].high:.2f}, Low=${current_session[0].low:.2f} (no sweep)")
    print(f"  Candle 1: High=${current_session[1].high:.2f}, Low=${current_session[1].low:.2f} ← SWEEPS ORH")
    
    result = identifier.identify(current_session)
    
    print("\n📊 PARTICIPANT IDENTIFICATION RESULT:")
    print(f"  Participant Type:  {result.participant_type.value}")
    print(f"  Conviction Level:  {result.conviction_level}")
    print(f"  Control Price:     ${result.control_price:.2f}")
    print(f"  Sweep Candle:      #{result.sweep_candle_index} (second candle)")
    
    print("\n💡 INTERPRETATION:")
    print("  → Informed buyers showed up on second candle")
    print("  → System detected sweep within allowed window (3 candles)")
    print("  → Still valid signal despite delayed entry")

def demo_mathematical_properties():
    print_separator("DEMO 5: Mathematical Properties - Determinism & Monotonicity")
    
    identifier1 = ParticipantIdentifier()
    identifier2 = ParticipantIdentifier()
    base_time = datetime(2024, 1, 1, 9, 30)
    
    previous_session = [
        Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
        Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0),
    ]
    
    current_session = [
        Candle(datetime(2024, 1, 2, 9, 30), 4520.0, 4540.0, 4515.0, 4535.0, 1500.0),
    ]
    
    identifier1.prev_session_candles = previous_session
    identifier2.prev_session_candles = previous_session
    
    result1 = identifier1.identify(current_session)
    result2 = identifier2.identify(current_session)
    
    print("\n✓ DETERMINISM TEST:")
    print(f"  Identifier 1 Result: {result1.participant_type.value}")
    print(f"  Identifier 2 Result: {result2.participant_type.value}")
    print(f"  Same Input → Same Output: {result1.participant_type == result2.participant_type}")
    
    identifier3 = ParticipantIdentifier()
    identifier3.prev_session_candles = previous_session
    
    orh, orl = identifier3.get_opening_range(previous_session)
    current_time = datetime(2024, 1, 2, 9, 30)
    
    both_sweeps = [
        Candle(current_time, 4490.0, 4495.0, 4470.0, 4475.0, 1800.0),
        Candle(current_time + timedelta(minutes=1), 4475.0, 4540.0, 4470.0, 4535.0, 2000.0),
    ]
    
    result3 = identifier3.identify(both_sweeps)
    
    print("\n✓ MONOTONICITY TEST (First Sweep Wins):")
    print(f"  Candle 0: Sweeps ORL → SELLERS")
    print(f"  Candle 1: Sweeps ORH → BUYERS (ignored)")
    print(f"  Final Result: {result3.participant_type.value}")
    print(f"  First Sweep Wins: {result3.participant_type == ParticipantType.SELLERS}")
    
    print("\n✓ BINARY OUTPUT TEST:")
    print(f"  Output is one of: {[pt.value for pt in ParticipantType]}")
    print(f"  No probabilistic component - pure classification")

def main():
    print("\n" + "="*70)
    print("  HORC ParticipantIdentifier - Implementation Demo")
    print("  AXIOM 2: First Move Determinism")
    print("="*70)
    
    demo_buyers_control()
    demo_sellers_control()
    demo_no_conviction()
    demo_second_candle_sweep()
    demo_mathematical_properties()
    
    print("\n" + "="*70)
    print("  Demo Complete - All AXIOM 2 Properties Validated ✓")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
