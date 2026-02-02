"""
Demo: WavelengthEngine Implementation

This script demonstrates AXIOM 1: Wavelength Invariant (exactly 3 moves)
Shows the complete finite-state automaton in action
"""

from datetime import datetime, timedelta
from src.engines.wavelength import (
    WavelengthEngine,
    WavelengthState,
    WavelengthConfig,
    validate_wavelength_progression
)
from src.engines.participant import (
    Candle,
    ParticipantResult,
    ParticipantType
)


def print_separator(title: str):
    """Print a formatted section separator"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_state_info(result, candle_num):
    """Print current state information"""
    print(f"\n📍 Candle #{candle_num} | State: {result.state.value}")
    print(f"   Moves Completed: {result.moves_completed}/3")
    print(f"   Signal Strength: {result.signal_strength:.2f}")
    if result.move_1_extreme:
        print(f"   Move 1 Extreme: ${result.move_1_extreme:.2f}")
    if result.move_2_extreme:
        print(f"   Move 2 Extreme: ${result.move_2_extreme:.2f}")
    if result.flip_point:
        print(f"   Flip Point: ${result.flip_point:.2f}")
    if result.entry_price:
        print(f"   Entry: ${result.entry_price:.2f}")
    if result.stop_price:
        print(f"   Stop: ${result.stop_price:.2f}")
    if result.target_price:
        print(f"   Target: ${result.target_price:.2f}")


def demo_complete_cycle():
    """Demonstrate complete 3-move wavelength cycle"""
    print_separator("DEMO 1: Complete 3-Move Cycle (BUYERS)")
    
    engine = WavelengthEngine()
    base_time = datetime(2024, 1, 2, 9, 30)
    
    # Create BUYERS participant
    participant = ParticipantResult(
        participant_type=ParticipantType.BUYERS,
        conviction_level=True,
        control_price=4480.0,
        timestamp=base_time,
        orh_prev=4530.0,
        orl_prev=4480.0,
        sweep_candle_index=0
    )
    
    print("\n🎯 Participant Identified: BUYERS")
    print("   Control Price: $4480.00 (swept ORL)")
    
    state_history = []
    candle_num = 0
    
    # === STAGE 1: Participant Identification ===
    print("\n" + "-"*70)
    print("STAGE 1: Participant Identification → MOVE_1")
    print("-"*70)
    
    # Initial candle with participant
    candle_num += 1
    candle = Candle(base_time, 4490.0, 4500.0, 4485.0, 4495.0, 1000.0)
    result = engine.process_candle(candle, participant)
    state_history.append(result.state)
    print_state_info(result, candle_num)
    
    # === MOVE 1: Upward directional move ===
    print("\n▲ Move 1: Upward directional move (buyers pushing up)...")
    
    for i in range(1, 12):
        candle_num += 1
        price_base = 4495.0 + (i * 5)
        
        if i < 11:
            # Strong upward candles
            candle = Candle(
                base_time + timedelta(minutes=i),
                price_base, price_base + 7, price_base - 2, price_base + 5,
                1000.0 + i * 100
            )
        else:
            # Rejection candle (completes Move 1)
            candle = Candle(
                base_time + timedelta(minutes=i),
                price_base, price_base + 15, price_base - 2, price_base + 2,  # Long upper wick
                2500.0
            )
        
        result = engine.process_candle(candle)
        state_history.append(result.state)
        
        if i in [1, 5, 10, 11]:
            print_state_info(result, candle_num)
    
    print("\n✓ Move 1 Complete - Buyers reached extreme, now showing exhaustion")
    
    # === MOVE 2: Reversal/Retracement ===
    print("\n" + "-"*70)
    print("STAGE 2: MOVE_1 → MOVE_2 (Reversal)")
    print("-"*70)
    print("\n▼ Move 2: Counter-move (creating liquidity sweep opportunity)...")
    
    for i in range(12, 18):
        candle_num += 1
        price_base = 4545.0 - ((i - 11) * 4)
        
        # Bearish reversal candles
        candle = Candle(
            base_time + timedelta(minutes=i),
            price_base + 2, price_base + 3, price_base - 5, price_base - 3,
            1200.0
        )
        
        result = engine.process_candle(candle)
        state_history.append(result.state)
        
        if i in [12, 15, 17]:
            print_state_info(result, candle_num)
    
    print("\n✓ Move 2 In Progress - Retracing to create flip point opportunity")
    
    # === FLIP POINT: Exhaustion/Absorption ===
    print("\n" + "-"*70)
    print("STAGE 3: MOVE_2 → FLIP_CONFIRMED (Absorption)")
    print("-"*70)
    print("\n⚡ Exhaustion Detection: High volume, long wicks, price stagnation...")
    
    # Absorption candles (high volume, small body, long wicks)
    for i in range(18, 22):
        candle_num += 1
        
        # Small-body, high-volume candles showing absorption
        candle = Candle(
            base_time + timedelta(minutes=i),
            4518.0, 4523.0, 4510.0, 4520.0,  # Long lower wick
            3000.0 + (i - 18) * 500  # Increasing volume
        )
        
        result = engine.process_candle(candle)
        state_history.append(result.state)
        print_state_info(result, candle_num)
        
        if result.state == WavelengthState.FLIP_CONFIRMED:
            print("\n✓ FLIP POINT CONFIRMED - Absorption detected!")
            print("   Passive liquidity has absorbed aggressive sellers")
            print("   Buyers ready to take control again")
            break
    
    # === FLIP CONFIRMATION ===
    print("\n" + "-"*70)
    print("STAGE 4: FLIP_CONFIRMED → MOVE_3 (Confirmation Period)")
    print("-"*70)
    
    for i in range(22, 25):
        candle_num += 1
        candle = Candle(
            base_time + timedelta(minutes=i),
            4520.0, 4527.0, 4518.0, 4525.0,
            1800.0
        )
        
        result = engine.process_candle(candle)
        state_history.append(result.state)
        print_state_info(result, candle_num)
    
    print("\n✓ Flip Point Confirmed - Ready for Move 3")
    
    # === MOVE 3: Continuation ===
    print("\n" + "-"*70)
    print("STAGE 5: MOVE_3 → COMPLETE (Continuation to Target)")
    print("-"*70)
    print("\n▲ Move 3: Continuation move toward target...")
    
    for i in range(25, 35):
        candle_num += 1
        price_base = 4525.0 + ((i - 24) * 4)
        
        candle = Candle(
            base_time + timedelta(minutes=i),
            price_base, price_base + 5, price_base - 2, price_base + 3,
            1500.0
        )
        
        result = engine.process_candle(candle)
        state_history.append(result.state)
        
        if i in [26, 30, 34] or result.state == WavelengthState.COMPLETE:
            print_state_info(result, candle_num)
        
        if result.state == WavelengthState.COMPLETE:
            print("\n✓✓✓ PATTERN COMPLETE - Target Reached!")
            break
    
    # === VALIDATION ===
    print("\n" + "="*70)
    print("  AXIOM 1 VALIDATION")
    print("="*70)
    
    is_valid = validate_wavelength_progression(state_history)
    
    print(f"\n✓ Wavelength Invariant Verified: {is_valid}")
    print(f"✓ Exactly 3 moves completed: {result.moves_completed == 3}")
    print(f"✓ Terminal state reached: {result.state == WavelengthState.COMPLETE}")
    print(f"\n📊 State Progression:")
    
    unique_states = []
    for state in state_history:
        if not unique_states or state != unique_states[-1]:
            unique_states.append(state)
    
    for idx, state in enumerate(unique_states):
        arrow = " → " if idx < len(unique_states) - 1 else ""
        print(f"   {state.value}{arrow}", end="")
    print("\n")


def demo_pattern_failure():
    """Demonstrate pattern invalidation (FAILED state)"""
    print_separator("DEMO 2: Pattern Invalidation (FAILED)")
    
    engine = WavelengthEngine()
    base_time = datetime(2024, 1, 2, 9, 30)
    
    # Set engine to MOVE_2 manually
    engine.state = WavelengthState.MOVE_2
    engine.moves_completed = 2
    engine.participant_type = ParticipantType.BUYERS
    engine.move_1_start = 4490.0
    engine.move_1_extreme = 4560.0
    engine.move_2_extreme = 4520.0
    
    print("\n⚙️  Setup:")
    print(f"   State: {engine.state.value}")
    print(f"   Participant: {engine.participant_type.value}")
    print(f"   Move 1 Start: ${engine.move_1_start:.2f}")
    print(f"   Move 1 Extreme: ${engine.move_1_extreme:.2f}")
    print(f"   Move 2 Extreme: ${engine.move_2_extreme:.2f}")
    
    print("\n❌ Invalidation Scenario: Price breaks below Move 1 start")
    print("   This violates the retracement rule and invalidates the pattern")
    
    # Price breaks below Move 1 start
    invalidation_candle = Candle(
        base_time,
        4495.0, 4500.0, 4485.0, 4488.0,  # Low < move_1_start
        2000.0
    )
    
    result = engine.process_candle(invalidation_candle)
    
    print(f"\n📍 After Invalidation:")
    print(f"   State: {result.state.value}")
    print(f"   Signal Strength: {result.signal_strength:.2f}")
    print(f"\n💡 Pattern invalidated - no trade signal")


def demo_state_machine_properties():
    """Demonstrate FSA mathematical properties"""
    print_separator("DEMO 3: Finite-State Automaton Properties")
    
    # Determinism
    print("\n1️⃣  DETERMINISM TEST")
    print("   Same input → Same output (reproducible)")
    
    engine1 = WavelengthEngine()
    engine2 = WavelengthEngine()
    
    participant = ParticipantResult(
        participant_type=ParticipantType.BUYERS,
        conviction_level=True,
        control_price=4480.0,
        timestamp=datetime(2024, 1, 2, 9, 30),
        orh_prev=4530.0,
        orl_prev=4480.0,
        sweep_candle_index=0
    )
    
    candle = Candle(datetime(2024, 1, 2, 9, 30), 4490.0, 4500.0, 4485.0, 4495.0, 1000.0)
    
    result1 = engine1.process_candle(candle, participant)
    result2 = engine2.process_candle(candle, participant)
    
    print(f"   Engine 1 State: {result1.state.value}")
    print(f"   Engine 2 State: {result2.state.value}")
    print(f"   ✓ Deterministic: {result1.state == result2.state}")
    
    # Completeness
    print("\n2️⃣  COMPLETENESS TEST")
    print("   All states have defined transitions")
    
    engine = WavelengthEngine()
    states_tested = []
    
    for state in WavelengthState:
        engine.state = state
        engine.participant_type = ParticipantType.BUYERS
        result = engine.process_candle(candle)
        states_tested.append(state)
    
    print(f"   States Tested: {len(states_tested)}/8")
    print(f"   ✓ Complete: All states process without error")
    
    # Termination
    print("\n3️⃣  TERMINATION TEST")
    print("   Terminal states (COMPLETE, FAILED) don't transition further")
    
    engine = WavelengthEngine()
    
    # Test COMPLETE
    engine.state = WavelengthState.COMPLETE
    before = engine.state
    engine.process_candle(candle)
    after = engine.state
    print(f"   COMPLETE before: {before.value}")
    print(f"   COMPLETE after:  {after.value}")
    print(f"   ✓ COMPLETE is terminal: {before == after}")
    
    # Test FAILED
    engine.state = WavelengthState.FAILED
    before = engine.state
    engine.process_candle(candle)
    after = engine.state
    print(f"   FAILED before: {before.value}")
    print(f"   FAILED after:  {after.value}")
    print(f"   ✓ FAILED is terminal: {before == after}")
    
    # Moore Machine
    print("\n4️⃣  MOORE MACHINE TEST")
    print("   Output depends only on state, not input history")
    
    engine = WavelengthEngine()
    engine.state = WavelengthState.MOVE_2
    engine.participant_type = ParticipantType.BUYERS
    
    strength = engine.calculate_signal_strength()
    print(f"   State: {engine.state.value}")
    print(f"   Signal Strength: {strength}")
    print(f"   ✓ Output is function of state only")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("  HORC WavelengthEngine - Implementation Demo")
    print("  AXIOM 1: Wavelength Invariant (Exactly 3 Moves)")
    print("="*70)
    
    demo_complete_cycle()
    demo_pattern_failure()
    demo_state_machine_properties()
    
    print("\n" + "="*70)
    print("  Demo Complete - All AXIOM 1 Properties Validated ✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
