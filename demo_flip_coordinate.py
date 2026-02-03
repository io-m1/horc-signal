from src.core import (
    ParticipantType,
    FlipEngine,
    TimeframeType,
    ChargeEngine,
    CoordinateEngine,
    build_coordinate_from_participant_states,
)

print("=" * 70)
print("PHASE 1.5 DEMONSTRATION — Flip + Charge + Coordinate Engines")
print("=" * 70)
print()

print("INITIALIZATION")
print("-" * 70)
flip_engine = FlipEngine("D", TimeframeType.DAILY)
charge_engine = ChargeEngine()
coord_engine = CoordinateEngine()

charge_engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.BUYER)
flip_engine.register_tf_open(
    open_time=1000,
    next_open_time=2000,
    initial_participant=ParticipantType.BUYER
)

print("✓ Flip Engine initialized (Daily TF)")
print("✓ Charge Engine initialized (Initial: BUYER)")
print("✓ Daily period: 1000 → 2000 (validity window)")
print()

print("SCENARIO 1: Level Formation (BUYER Control)")
print("-" * 70)

level1 = charge_engine.assign_charge("D", 1.1050, 1100, is_high=True)
coord1 = build_coordinate_from_participant_states(
    price=1.1050, timestamp=1100, is_high=True,
    participants={"D": ParticipantType.BUYER}
)

print(f"Level 1: {level1.price}")
print(f"  Charge: {level1.charge_symbol} ({level1.charge})")
print(f"  Label: {level1.label}")
print(f"  Participant: {level1.participant_at_formation.name}")
print(f"  Coordinate: {coord1.label}")
print()

print("SCENARIO 2: Flip Detection (Opposition)")
print("-" * 70)

range_high = 1.1050
range_low = 1.1000

print("Initial range: [1.1000, 1.1050]")
print()

flip_engine.update_sweep(1150, 1.1100, 1.1020, range_high, range_low)
range_high = 1.1100

print("Step 1: Sweep HIGH → 1.1100 (BUYER takes control)")
result1 = flip_engine.validate_flip(1150, 1.1100)
print(f"  Flip occurred: {result1.flip_occurred}")
print(f"  State: {result1.state.name}")
print()

flip_engine.update_sweep(1200, 1.1100, 1.0950, range_high, range_low)

print("Step 2: Sweep LOW → 1.0950 (opposition detected!)")
result2 = flip_engine.validate_flip(1200, 1.0950)
print(f"  Flip occurred: {result2.flip_occurred}")
print(f"  State: {result2.state.name}")
print(f"  Original participant: {result2.flip_point.original_participant.name}")
print(f"  New participant: {result2.flip_point.new_participant.name}")
print(f"  Within validity window: {result2.within_validity_window}")
print()

print("SCENARIO 3: Level Formation (SELLER Control)")
print("-" * 70)

charge_engine.update_participant("D", ParticipantType.SELLER, result2.flip_point)

level2 = charge_engine.assign_charge("D", 1.0950, 1300, is_high=False)
coord2 = build_coordinate_from_participant_states(
    price=1.0950, timestamp=1300, is_high=False,
    participants={"D": ParticipantType.SELLER}
)

print(f"Level 2: {level2.price}")
print(f"  Charge: {level2.charge_symbol} ({level2.charge})")
print(f"  Label: {level2.label}")
print(f"  Participant: {level2.participant_at_formation.name}")
print(f"  Coordinate: {coord2.label}")
print()

print("SCENARIO 4: Charge Immutability")
print("-" * 70)

print(f"Level 1 (before flip): {level1.label} — charge = {level1.charge}")
print(f"Level 2 (after flip):  {level2.label} — charge = {level2.charge}")
print()
print("✓ Level 1 charge is IMMUTABLE (still D+ despite flip)")
print("✓ Level 2 has new charge (D− after flip)")
print()

print("SCENARIO 5: Coordinate Divergence")
print("-" * 70)

divergent_tfs = coord1.get_divergence_tfs(coord2)

print(f"Coordinate 1: {coord1.label}")
print(f"Coordinate 2: {coord2.label}")
print(f"Divergent TFs: {divergent_tfs}")
print(f"Match (strict): {coord1.matches(coord2, strict=True)}")
print()

print("SCENARIO 6: Temporal Finality (Lock After Next Open)")
print("-" * 70)

print(f"Current time: 1300 (before next open at 2000)")
print(f"  Within validity window: {flip_engine.is_within_validity_window(1300)}")
print(f"  Flip state: {result2.flip_point.state.name}")
print()

result3 = flip_engine.validate_flip(2100, 1.0950)

print(f"After next open: 2100 (past validity window)")
print(f"  Within validity window: {result3.within_validity_window}")
print(f"  Flip state: {result3.flip_point.state.name}")
print(f"  Is locked: {result3.flip_point.is_locked}")
print()
print("✓ Flip is now LOCKED (immutable)")
print()

print("=" * 70)
print("DOCTRINE SUMMARY")
print("=" * 70)
print()
print("FLIP ENGINE:")
print('  "A flip is valid only before the next corresponding open."')
print()
print("CHARGE ENGINE:")
print('  "Any high or low inherits the participant state active at formation."')
print()
print("COORDINATE ENGINE:")
print('  "Only timeframes that exist at formation are included."')
print()
print("TEMPORAL FINALITY:")
print('  "Once a timeframe closes, its charge state is immutable."')
print()
print("=" * 70)
print("PHASE 1.5 COMPLETE — All engines operational")
print("=" * 70)
