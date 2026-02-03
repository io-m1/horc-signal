import pytest
from src.core import (
    ParticipantType,
    ParticipantEngine,
    ParentPeriod,
    FlipEngine,
    FlipState,
    TimeframeType,
    ChargeEngine,
    Charge,
    CoordinateEngine,
    Coordinate,
    build_coordinate_from_participant_states,
)

class TestFlipEngine:
    
    def test_flip_detection_basic(self):
        engine = FlipEngine(timeframe="D", tf_type=TimeframeType.DAILY)
        
        engine.register_tf_open(
            open_time=1000,
            next_open_time=2000,  # Next day at 2000
            initial_participant=ParticipantType.BUYER
        )
        
        range_high = 1.1050
        range_low = 1.1010
        
        engine.update_sweep(
            current_time=1100,
            current_high=1.1100,  # New high
            current_low=1.1020,
            range_high=range_high,
            range_low=range_low,
        )
        
        range_high = 1.1100
        
        result1 = engine.validate_flip(current_time=1100, current_price=1.1100)
        assert result1.flip_occurred is False
        assert result1.state == FlipState.ACTIVE
        assert result1.within_validity_window is True
        
        engine.update_sweep(
            current_time=1200,
            current_high=1.1100,
            current_low=1.0950,  # New low below previous range
            range_high=range_high,
            range_low=range_low,
        )
        
        result2 = engine.validate_flip(current_time=1200, current_price=1.0950)
        assert result2.flip_occurred is True
        assert result2.state == FlipState.CONFIRMED
        assert result2.flip_point is not None
        assert result2.flip_point.original_participant == ParticipantType.BUYER
        assert result2.flip_point.new_participant == ParticipantType.SELLER
        assert result2.current_participant == ParticipantType.SELLER
    
    def test_flip_temporal_finality(self):
        engine = FlipEngine(timeframe="D", tf_type=TimeframeType.DAILY)
        
        engine.register_tf_open(
            open_time=1000,
            next_open_time=2000,
            initial_participant=ParticipantType.BUYER
        )
        
        range_high = 1.1050
        range_low = 1.1000
        
        engine.update_sweep(1100, 1.1100, 1.1020, range_high, range_low)
        range_high = 1.1100
        
        engine.update_sweep(1150, 1.1100, 1.0950, range_high, range_low)
        
        result1 = engine.validate_flip(1150, 1.0950)
        assert result1.within_validity_window is True
        
        result2 = engine.validate_flip(2100, 1.0950)  # time > next_open_time
        assert result2.within_validity_window is False
        assert result2.flip_point.is_locked is True
        assert result2.flip_point.state == FlipState.LOCKED
    
    def test_no_flip_without_opposition(self):
        engine = FlipEngine(timeframe="D", tf_type=TimeframeType.DAILY)
        
        engine.register_tf_open(1000, 2000, ParticipantType.BUYER)
        
        engine.update_sweep(
            current_time=1100,
            current_high=1.1100,  # New high
            current_low=1.1020,
            range_high=1.1050,
            range_low=1.1010
        )
        
        result = engine.validate_flip(1100, 1.1100)
        assert result.flip_occurred is False
        assert result.state == FlipState.ACTIVE

class TestChargeEngine:
    
    def test_charge_assignment_buyer(self):
        engine = ChargeEngine()
        engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.BUYER)
        
        level = engine.assign_charge(
            timeframe="D",
            price=1.1050,
            timestamp=1000,
            is_high=True
        )
        
        assert level.charge == Charge.POSITIVE
        assert level.charge_symbol == "+"
        assert level.label == "D+"
        assert level.participant_at_formation == ParticipantType.BUYER
    
    def test_charge_assignment_seller(self):
        engine = ChargeEngine()
        engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.SELLER)
        
        level = engine.assign_charge(
            timeframe="D",
            price=1.0950,
            timestamp=1000,
            is_high=False
        )
        
        assert level.charge == Charge.NEGATIVE
        assert level.charge_symbol == "−"
        assert level.label == "D−"
        assert level.participant_at_formation == ParticipantType.SELLER
    
    def test_charge_flip_inheritance(self):
        engine = ChargeEngine()
        engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.BUYER)
        
        level1 = engine.assign_charge("D", 1.1050, 1000, True)
        assert level1.charge == Charge.POSITIVE
        
        engine.update_participant("D", ParticipantType.SELLER)
        
        level2 = engine.assign_charge("D", 1.0950, 2000, False)
        assert level2.charge == Charge.NEGATIVE
        
        assert level1.charge == Charge.POSITIVE
        assert level1.participant_at_formation == ParticipantType.BUYER

class TestCoordinateEngine:
    
    def test_coordinate_build_single_tf(self):
        participants = {"S": ParticipantType.BUYER}
        
        coord = build_coordinate_from_participant_states(
            price=1.1000,
            timestamp=1000,
            is_high=True,
            participants=participants
        )
        
        assert coord.S == Charge.POSITIVE
        assert coord.D is None
        assert coord.W is None
        assert coord.M is None
        assert coord.label == "(S+)"
        assert coord.active_tfs == ('S',)
    
    def test_coordinate_build_multi_tf(self):
        participants = {
            "M": ParticipantType.SELLER,
            "W": ParticipantType.BUYER,
            "D": ParticipantType.BUYER,
            "S": ParticipantType.SELLER,
        }
        
        coord = build_coordinate_from_participant_states(
            price=1.1000,
            timestamp=1000,
            is_high=True,
            participants=participants
        )
        
        assert coord.M == Charge.NEGATIVE
        assert coord.W == Charge.POSITIVE
        assert coord.D == Charge.POSITIVE
        assert coord.S == Charge.NEGATIVE
        assert coord.label == "(M−, W+, D+, S−)"
        assert set(coord.active_tfs) == {'M', 'W', 'D', 'S'}
    
    def test_coordinate_matching_strict(self):
        coord1 = Coordinate(
            price=1.1000, timestamp=1000, is_high=True,
            M=-1, W=+1, D=+1, S=-1,
            active_tfs=('M', 'W', 'D', 'S')
        )
        
        coord2 = Coordinate(
            price=1.1000, timestamp=2000, is_high=True,
            M=-1, W=+1, D=+1, S=-1,
            active_tfs=('M', 'W', 'D', 'S')
        )
        
        assert coord1.matches(coord2, strict=True)
    
    def test_coordinate_divergence_detection(self):
        coord1 = Coordinate(
            price=1.1000, timestamp=1000, is_high=True,
            M=-1, W=+1, D=+1, S=-1,
            active_tfs=('M', 'W', 'D', 'S')
        )
        
        coord2 = Coordinate(
            price=1.1000, timestamp=2000, is_high=True,
            M=-1, W=+1, D=-1, S=-1,  # Daily flipped
            active_tfs=('M', 'W', 'D', 'S')
        )
        
        divergent = coord1.get_divergence_tfs(coord2)
        assert divergent == ['D']
        assert not coord1.matches(coord2, strict=True)

class TestIntegratedFlipChargeCoordinate:
    
    def test_full_workflow(self):
        flip_engine = FlipEngine("D", TimeframeType.DAILY)
        charge_engine = ChargeEngine()
        coord_engine = CoordinateEngine()
        
        charge_engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.BUYER)
        
        flip_engine.register_tf_open(1000, 2000, ParticipantType.BUYER)
        
        level1 = charge_engine.assign_charge("D", 1.1050, 1100, is_high=True)
        coord1 = build_coordinate_from_participant_states(
            price=1.1050, timestamp=1100, is_high=True,
            participants={"D": ParticipantType.BUYER}
        )
        
        assert level1.charge == Charge.POSITIVE
        assert coord1.label == "(D+)"
        
        range_high = 1.1050
        range_low = 1.1000
        
        flip_engine.update_sweep(1150, 1.1100, 1.1020, range_high, range_low)
        range_high = 1.1100
        
        flip_engine.update_sweep(1200, 1.1100, 1.0950, range_high, range_low)
        
        flip_result = flip_engine.validate_flip(1200, 1.0950)
        
        assert flip_result.flip_occurred is True
        assert flip_result.current_participant == ParticipantType.SELLER
        
        charge_engine.update_participant("D", ParticipantType.SELLER, flip_result.flip_point)
        
        level2 = charge_engine.assign_charge("D", 1.0950, 1300, is_high=False)
        coord2 = build_coordinate_from_participant_states(
            price=1.0950, timestamp=1300, is_high=False,
            participants={"D": ParticipantType.SELLER}
        )
        
        assert level2.charge == Charge.NEGATIVE
        assert coord2.label == "(D−)"
        
        assert level1.charge == Charge.POSITIVE
        
        divergent = coord1.get_divergence_tfs(coord2)
        assert divergent == ['D']

class TestTemporalFinality:
    
    def test_state_lock_after_next_open(self):
        flip_engine = FlipEngine("D", TimeframeType.DAILY)
        charge_engine = ChargeEngine()
        
        flip_engine.register_tf_open(1000, 2000, ParticipantType.BUYER)
        charge_engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.BUYER)
        
        range_high = 1.1050
        range_low = 1.1000
        
        flip_engine.update_sweep(1150, 1.1100, 1.1020, range_high, range_low)
        range_high = 1.1100
        
        flip_engine.update_sweep(1200, 1.1100, 1.0950, range_high, range_low)
        
        result = flip_engine.validate_flip(1200, 1.0950)
        
        assert result.within_validity_window is True
        assert result.flip_point.state == FlipState.CONFIRMED
        
        result_locked = flip_engine.validate_flip(2100, 1.0950)
        
        assert result_locked.within_validity_window is False
        assert result_locked.flip_point.state == FlipState.LOCKED
        assert result_locked.flip_point.is_locked is True
        
        flip_point_locked = flip_engine.get_current_flip()
        assert flip_point_locked.state == FlipState.LOCKED
