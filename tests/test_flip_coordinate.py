"""
Test Flip + Charge + Coordinate Engines (PHASE 1.5)

Validates temporal finality, charge inheritance, and multi-TF state vectors.
"""

import pytest
from src.core import (
    # PHASE 1
    ParticipantType,
    ParticipantEngine,
    ParentPeriod,
    # PHASE 1.5
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
    """Test flip validation and temporal finality."""
    
    def test_flip_detection_basic(self):
        """Test basic flip detection with opposition."""
        engine = FlipEngine(timeframe="D", tf_type=TimeframeType.DAILY)
        
        # Register daily open
        engine.register_tf_open(
            open_time=1000,
            next_open_time=2000,  # Next day at 2000
            initial_participant=ParticipantType.BUYER
        )
        
        # Track range progression
        range_high = 1.1050
        range_low = 1.1010
        
        # Sweep high (buyer active)
        engine.update_sweep(
            current_time=1100,
            current_high=1.1100,  # New high
            current_low=1.1020,
            range_high=range_high,
            range_low=range_low,
        )
        
        # Update range
        range_high = 1.1100
        
        # No flip yet (only high swept)
        result1 = engine.validate_flip(current_time=1100, current_price=1.1100)
        assert result1.flip_occurred is False
        assert result1.state == FlipState.ACTIVE
        assert result1.within_validity_window is True
        
        # Sweep low (opposition detected)
        engine.update_sweep(
            current_time=1200,
            current_high=1.1100,
            current_low=1.0950,  # New low below previous range
            range_high=range_high,
            range_low=range_low,
        )
        
        # Flip should be registered
        result2 = engine.validate_flip(current_time=1200, current_price=1.0950)
        assert result2.flip_occurred is True
        assert result2.state == FlipState.CONFIRMED
        assert result2.flip_point is not None
        assert result2.flip_point.original_participant == ParticipantType.BUYER
        assert result2.flip_point.new_participant == ParticipantType.SELLER
        assert result2.current_participant == ParticipantType.SELLER
    
    def test_flip_temporal_finality(self):
        """Test that flip locks after next open."""
        engine = FlipEngine(timeframe="D", tf_type=TimeframeType.DAILY)
        
        engine.register_tf_open(
            open_time=1000,
            next_open_time=2000,
            initial_participant=ParticipantType.BUYER
        )
        
        # Trigger flip within validity window
        range_high = 1.1050
        range_low = 1.1000
        
        # Sweep high first
        engine.update_sweep(1100, 1.1100, 1.1020, range_high, range_low)
        range_high = 1.1100
        
        # Then sweep low (flip)
        engine.update_sweep(1150, 1.1100, 1.0950, range_high, range_low)
        
        result1 = engine.validate_flip(1150, 1.0950)
        assert result1.within_validity_window is True
        
        # Check after next open (past validity window)
        result2 = engine.validate_flip(2100, 1.0950)  # time > next_open_time
        assert result2.within_validity_window is False
        assert result2.flip_point.is_locked is True
        assert result2.flip_point.state == FlipState.LOCKED
    
    def test_no_flip_without_opposition(self):
        """Test that no flip occurs without opposition."""
        engine = FlipEngine(timeframe="D", tf_type=TimeframeType.DAILY)
        
        engine.register_tf_open(1000, 2000, ParticipantType.BUYER)
        
        # Only sweep high (no opposition)
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
    """Test charge assignment and inheritance."""
    
    def test_charge_assignment_buyer(self):
        """Test charge assignment for buyer-controlled level."""
        engine = ChargeEngine()
        engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.BUYER)
        
        # Assign charge to high (buyer active)
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
        """Test charge assignment for seller-controlled level."""
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
        """Test that charge changes for new levels after flip."""
        engine = ChargeEngine()
        engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.BUYER)
        
        # Level 1: Before flip (buyer active)
        level1 = engine.assign_charge("D", 1.1050, 1000, True)
        assert level1.charge == Charge.POSITIVE
        
        # Flip occurs (seller takes control)
        engine.update_participant("D", ParticipantType.SELLER)
        
        # Level 2: After flip (seller active)
        level2 = engine.assign_charge("D", 1.0950, 2000, False)
        assert level2.charge == Charge.NEGATIVE
        
        # Level 1 charge is IMMUTABLE (still positive)
        assert level1.charge == Charge.POSITIVE
        assert level1.participant_at_formation == ParticipantType.BUYER


class TestCoordinateEngine:
    """Test coordinate building and comparison."""
    
    def test_coordinate_build_single_tf(self):
        """Test building coordinate for single timeframe."""
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
        """Test building coordinate for multiple timeframes."""
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
        """Test strict coordinate matching."""
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
        
        # Identical coordinates
        assert coord1.matches(coord2, strict=True)
    
    def test_coordinate_divergence_detection(self):
        """Test divergence detection between coordinates."""
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
    """Test integrated workflow: Flip → Charge → Coordinate."""
    
    def test_full_workflow(self):
        """Test complete workflow from flip detection to coordinate building."""
        # Initialize engines
        flip_engine = FlipEngine("D", TimeframeType.DAILY)
        charge_engine = ChargeEngine()
        coord_engine = CoordinateEngine()
        
        # Register timeframes
        charge_engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.BUYER)
        
        # Open daily period
        flip_engine.register_tf_open(1000, 2000, ParticipantType.BUYER)
        
        # Level 1: Before flip (buyer active)
        level1 = charge_engine.assign_charge("D", 1.1050, 1100, is_high=True)
        coord1 = build_coordinate_from_participant_states(
            price=1.1050, timestamp=1100, is_high=True,
            participants={"D": ParticipantType.BUYER}
        )
        
        assert level1.charge == Charge.POSITIVE
        assert coord1.label == "(D+)"
        
        # Trigger flip (opposition detected)
        range_high = 1.1050
        range_low = 1.1000
        
        # Sweep high
        flip_engine.update_sweep(1150, 1.1100, 1.1020, range_high, range_low)
        range_high = 1.1100
        
        # Sweep low (flip)
        flip_engine.update_sweep(1200, 1.1100, 1.0950, range_high, range_low)
        
        flip_result = flip_engine.validate_flip(1200, 1.0950)
        
        assert flip_result.flip_occurred is True
        assert flip_result.current_participant == ParticipantType.SELLER
        
        # Update charge engine after flip
        charge_engine.update_participant("D", ParticipantType.SELLER, flip_result.flip_point)
        
        # Level 2: After flip (seller active)
        level2 = charge_engine.assign_charge("D", 1.0950, 1300, is_high=False)
        coord2 = build_coordinate_from_participant_states(
            price=1.0950, timestamp=1300, is_high=False,
            participants={"D": ParticipantType.SELLER}
        )
        
        assert level2.charge == Charge.NEGATIVE
        assert coord2.label == "(D−)"
        
        # Verify immutability: level1 still has original charge
        assert level1.charge == Charge.POSITIVE
        
        # Verify divergence detection
        divergent = coord1.get_divergence_tfs(coord2)
        assert divergent == ['D']


class TestTemporalFinality:
    """Test temporal finality across all engines."""
    
    def test_state_lock_after_next_open(self):
        """Test that all states lock after next open."""
        flip_engine = FlipEngine("D", TimeframeType.DAILY)
        charge_engine = ChargeEngine()
        
        flip_engine.register_tf_open(1000, 2000, ParticipantType.BUYER)
        charge_engine.register_timeframe("D", TimeframeType.DAILY, ParticipantType.BUYER)
        
        # Flip occurs at 1200 (within window)
        range_high = 1.1050
        range_low = 1.1000
        
        # Sweep high
        flip_engine.update_sweep(1150, 1.1100, 1.1020, range_high, range_low)
        range_high = 1.1100
        
        # Sweep low (flip)
        flip_engine.update_sweep(1200, 1.1100, 1.0950, range_high, range_low)
        
        result = flip_engine.validate_flip(1200, 1.0950)
        
        assert result.within_validity_window is True
        assert result.flip_point.state == FlipState.CONFIRMED
        
        # Check state at 2100 (after next open)
        result_locked = flip_engine.validate_flip(2100, 1.0950)
        
        assert result_locked.within_validity_window is False
        assert result_locked.flip_point.state == FlipState.LOCKED
        assert result_locked.flip_point.is_locked is True
        
        # Flip is immutable now
        flip_point_locked = flip_engine.get_current_flip()
        assert flip_point_locked.state == FlipState.LOCKED
