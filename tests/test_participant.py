"""
Unit tests for Participant Identification Engine

Tests AXIOM 2: First Move Determinism
Validates all edge cases and mathematical properties
"""

import pytest
from datetime import datetime, timedelta
from src.engines.participant import (
    ParticipantIdentifier,
    ParticipantType,
    ParticipantResult,
    Candle,
    create_test_candles_sweep_high,
    create_test_candles_sweep_low,
    create_test_candles_no_sweep
)


class TestCandle:
    """Test Candle dataclass validation"""
    
    def test_valid_candle_creation(self):
        """Test creating a valid candle"""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=4500.0,
            high=4510.0,
            low=4490.0,
            close=4505.0,
            volume=1000.0
        )
        assert candle.high == 4510.0
        assert candle.low == 4490.0
    
    def test_invalid_high_raises_error(self):
        """Test that invalid high value raises ValueError"""
        with pytest.raises(ValueError, match="High.*cannot be less"):
            Candle(
                timestamp=datetime(2024, 1, 1, 9, 30),
                open=4500.0,
                high=4490.0,  # High < open (invalid)
                low=4480.0,
                close=4505.0,
                volume=1000.0
            )
    
    def test_invalid_low_raises_error(self):
        """Test that invalid low value raises ValueError"""
        with pytest.raises(ValueError, match="Low.*cannot be greater"):
            Candle(
                timestamp=datetime(2024, 1, 1, 9, 30),
                open=4500.0,
                high=4510.0,
                low=4505.0,  # Low > close (invalid)
                close=4500.0,
                volume=1000.0
            )
    
    def test_negative_volume_raises_error(self):
        """Test that negative volume raises ValueError"""
        with pytest.raises(ValueError, match="Volume.*cannot be negative"):
            Candle(
                timestamp=datetime(2024, 1, 1, 9, 30),
                open=4500.0,
                high=4510.0,
                low=4490.0,
                close=4505.0,
                volume=-100.0  # Negative volume (invalid)
            )


class TestParticipantIdentifier:
    """Test ParticipantIdentifier core functionality"""
    
    @pytest.fixture
    def identifier(self):
        """Create a fresh identifier for each test"""
        return ParticipantIdentifier()
    
    @pytest.fixture
    def previous_session_candles(self):
        """Create standard previous session candles"""
        base_time = datetime(2024, 1, 1, 9, 30)
        return [
            Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4510.0, 4525.0, 4505.0, 4520.0, 1100.0),
            Candle(base_time + timedelta(minutes=2), 4520.0, 4530.0, 4515.0, 4525.0, 1200.0),
        ]
    
    def test_initialization_default_config(self, identifier):
        """Test identifier initializes with default configuration"""
        assert identifier.or_lookback_sessions == 1
        assert identifier.min_conviction_threshold == 0.8
        assert identifier.max_first_move_candles == 3
        assert identifier.prev_session_candles == []
    
    def test_initialization_custom_config(self):
        """Test identifier initializes with custom configuration"""
        config = {
            'or_lookback_sessions': 2,
            'min_conviction_threshold': 0.9,
            'max_first_move_candles': 5
        }
        identifier = ParticipantIdentifier(config)
        assert identifier.or_lookback_sessions == 2
        assert identifier.min_conviction_threshold == 0.9
        assert identifier.max_first_move_candles == 5
    
    def test_get_opening_range_success(self, identifier, previous_session_candles):
        """Test opening range calculation returns correct ORH and ORL"""
        orh, orl = identifier.get_opening_range(previous_session_candles)
        
        # ORH should be max high (4530.0)
        # ORL should be min low (4480.0)
        assert orh == 4530.0
        assert orl == 4480.0
    
    def test_get_opening_range_empty_list_raises_error(self, identifier):
        """Test that empty candle list raises ValueError"""
        with pytest.raises(ValueError, match="Cannot calculate opening range from empty"):
            identifier.get_opening_range([])
    
    def test_get_opening_range_single_candle(self, identifier):
        """Test opening range with single candle"""
        candle = Candle(datetime(2024, 1, 1), 4500.0, 4510.0, 4490.0, 4505.0, 1000.0)
        orh, orl = identifier.get_opening_range([candle])
        
        assert orh == 4510.0
        assert orl == 4490.0


class TestIdentifyParticipant:
    """Test the core identify_participant logic (AXIOM 2)"""
    
    @pytest.fixture
    def identifier(self):
        return ParticipantIdentifier()
    
    def test_sellers_identified_sweep_low(self, identifier):
        """Test SELLERS identified when first candle sweeps ORL_prev"""
        orh_prev = 4530.0
        orl_prev = 4480.0
        
        # Create candle that sweeps orl_prev
        candles = create_test_candles_sweep_low(orl_prev)
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.SELLERS
        assert conviction is True
        assert sweep_idx == 0
    
    def test_buyers_identified_sweep_high(self, identifier):
        """Test BUYERS identified when first candle sweeps ORH_prev"""
        orh_prev = 4530.0
        orl_prev = 4480.0
        
        # Create candle that sweeps orh_prev
        candles = create_test_candles_sweep_high(orh_prev)
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.BUYERS
        assert conviction is True
        assert sweep_idx == 0
    
    def test_none_identified_no_sweep(self, identifier):
        """Test NONE returned when no sweep occurs"""
        orh_prev = 4530.0
        orl_prev = 4480.0
        
        # Create candles that stay within range
        candles = create_test_candles_no_sweep(orh_prev, orl_prev)
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.NONE
        assert conviction is False
        assert sweep_idx is None
    
    def test_empty_candles_returns_none(self, identifier):
        """Test that empty candle list returns NONE"""
        participant, conviction, sweep_idx = identifier.identify_participant(
            [], 4530.0, 4480.0
        )
        
        assert participant == ParticipantType.NONE
        assert conviction is False
        assert sweep_idx is None
    
    def test_exact_level_touch_counts_as_sweep(self, identifier):
        """Test that exact touch of level (<=, >=) counts as sweep"""
        orh_prev = 4530.0
        orl_prev = 4480.0
        
        # Candle that exactly touches ORL (not below, exactly at)
        candle = Candle(
            timestamp=datetime(2024, 1, 2, 9, 30),
            open=4500.0,
            high=4510.0,
            low=4480.0,  # Exactly at ORL_prev
            close=4505.0,
            volume=1000.0
        )
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            [candle], orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.SELLERS
        assert conviction is True
    
    def test_second_candle_sweep_detected(self, identifier):
        """Test that sweep in second candle is detected"""
        orh_prev = 4530.0
        orl_prev = 4480.0
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            # First candle: no sweep
            Candle(base_time, 4500.0, 4510.0, 4490.0, 4505.0, 1000.0),
            # Second candle: sweeps ORH
            Candle(base_time + timedelta(minutes=1), 4505.0, 4535.0, 4500.0, 4530.0, 1100.0)
        ]
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.BUYERS
        assert conviction is True
        assert sweep_idx == 1  # Second candle (index 1)
    
    def test_first_sweep_wins_if_both_levels_swept(self, identifier):
        """Test that first sweep determines outcome (order matters)"""
        orh_prev = 4530.0
        orl_prev = 4480.0
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            # First candle: sweeps ORL (sellers)
            Candle(base_time, 4500.0, 4510.0, 4475.0, 4505.0, 1000.0),
            # Second candle: sweeps ORH (buyers)
            Candle(base_time + timedelta(minutes=1), 4505.0, 4535.0, 4500.0, 4530.0, 1100.0)
        ]
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        # First sweep (SELLERS) should win
        assert participant == ParticipantType.SELLERS
        assert sweep_idx == 0
    
    def test_respects_max_first_move_candles_config(self):
        """Test that only configured number of candles are analyzed"""
        identifier = ParticipantIdentifier({'max_first_move_candles': 2})
        orh_prev = 4530.0
        orl_prev = 4480.0
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            # First two candles: no sweep
            Candle(base_time, 4500.0, 4510.0, 4490.0, 4505.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4505.0, 4515.0, 4495.0, 4510.0, 1100.0),
            # Third candle: sweeps ORH (but should be ignored)
            Candle(base_time + timedelta(minutes=2), 4510.0, 4535.0, 4505.0, 4530.0, 1200.0)
        ]
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        # Third candle sweep should be ignored (max_first_move_candles=2)
        assert participant == ParticipantType.NONE
        assert conviction is False


class TestIdentifyMethod:
    """Test the full identify() method pipeline"""
    
    @pytest.fixture
    def identifier(self):
        return ParticipantIdentifier()
    
    @pytest.fixture
    def setup_previous_session(self, identifier):
        """Setup previous session data"""
        base_time = datetime(2024, 1, 1, 9, 30)
        identifier.prev_session_candles = [
            Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0),
        ]
        # ORH = 4530.0, ORL = 4480.0
    
    def test_identify_raises_error_without_previous_session(self, identifier):
        """Test that identify() raises error if prev_session_candles not set"""
        current_candles = [
            Candle(datetime(2024, 1, 2, 9, 30), 4500.0, 4510.0, 4475.0, 4505.0, 1000.0)
        ]
        
        with pytest.raises(ValueError, match="Previous session candles not set"):
            identifier.identify(current_candles)
    
    def test_identify_buyers_complete_result(self, identifier, setup_previous_session):
        """Test full identify() pipeline for BUYERS"""
        current_candles = create_test_candles_sweep_high(4530.0)
        
        result = identifier.identify(current_candles)
        
        assert isinstance(result, ParticipantResult)
        assert result.participant_type == ParticipantType.BUYERS
        assert result.conviction_level is True
        assert result.control_price == 4530.0  # ORH_prev
        assert result.orh_prev == 4530.0
        assert result.orl_prev == 4480.0
        assert result.sweep_candle_index == 0
        assert result.timestamp == current_candles[0].timestamp
    
    def test_identify_sellers_complete_result(self, identifier, setup_previous_session):
        """Test full identify() pipeline for SELLERS"""
        current_candles = create_test_candles_sweep_low(4480.0)
        
        result = identifier.identify(current_candles)
        
        assert result.participant_type == ParticipantType.SELLERS
        assert result.conviction_level is True
        assert result.control_price == 4480.0  # ORL_prev
        assert result.orh_prev == 4530.0
        assert result.orl_prev == 4480.0
        assert result.sweep_candle_index == 0
    
    def test_identify_none_complete_result(self, identifier, setup_previous_session):
        """Test full identify() pipeline for NONE (no sweep)"""
        current_candles = create_test_candles_no_sweep(4530.0, 4480.0)
        
        result = identifier.identify(current_candles)
        
        assert result.participant_type == ParticipantType.NONE
        assert result.conviction_level is False
        assert result.control_price is None  # No control price when NONE
        assert result.sweep_candle_index is None
    
    def test_identify_empty_current_candles(self, identifier, setup_previous_session):
        """Test identify() with empty current candles returns neutral result"""
        result = identifier.identify([])
        
        assert result.participant_type == ParticipantType.NONE
        assert result.conviction_level is False
        assert result.control_price is None
        assert result.orh_prev == 4530.0  # Still calculated from prev session
        assert result.orl_prev == 4480.0


class TestSessionManagement:
    """Test session data management methods"""
    
    def test_update_session_data(self):
        """Test updating session data for next cycle"""
        identifier = ParticipantIdentifier()
        base_time = datetime(2024, 1, 1, 9, 30)
        
        new_session = [
            Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0),
        ]
        
        identifier.update_session_data(new_session)
        
        assert len(identifier.prev_session_candles) == 2
        assert identifier.prev_session_candles[0].high == 4520.0
    
    def test_update_session_data_creates_copy(self):
        """Test that update_session_data creates a copy (doesn't reference)"""
        identifier = ParticipantIdentifier()
        base_time = datetime(2024, 1, 1, 9, 30)
        
        original = [
            Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
        ]
        
        identifier.update_session_data(original)
        
        # Modify original
        original.append(Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0))
        
        # Identifier's copy should not be affected
        assert len(identifier.prev_session_candles) == 1
    
    def test_reset_clears_session_data(self):
        """Test reset() clears all session data"""
        identifier = ParticipantIdentifier()
        base_time = datetime(2024, 1, 1, 9, 30)
        
        identifier.prev_session_candles = [
            Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
        ]
        
        identifier.reset()
        
        assert identifier.prev_session_candles == []


class TestMathematicalProperties:
    """Test mathematical properties of the implementation"""
    
    def test_determinism(self):
        """Test that same input produces same output (deterministic)"""
        identifier1 = ParticipantIdentifier()
        identifier2 = ParticipantIdentifier()
        
        base_time = datetime(2024, 1, 1, 9, 30)
        prev_session = [Candle(base_time, 4500.0, 4530.0, 4480.0, 4520.0, 1000.0)]
        current = create_test_candles_sweep_high(4530.0)
        
        identifier1.prev_session_candles = prev_session
        identifier2.prev_session_candles = prev_session
        
        result1 = identifier1.identify(current)
        result2 = identifier2.identify(current)
        
        # Both should produce identical results
        assert result1.participant_type == result2.participant_type
        assert result1.conviction_level == result2.conviction_level
        assert result1.control_price == result2.control_price
    
    def test_binary_output(self):
        """Test that output is always one of exactly three states"""
        identifier = ParticipantIdentifier()
        base_time = datetime(2024, 1, 1, 9, 30)
        identifier.prev_session_candles = [
            Candle(base_time, 4500.0, 4530.0, 4480.0, 4520.0, 1000.0)
        ]
        
        # Test multiple scenarios
        test_cases = [
            create_test_candles_sweep_high(4530.0),
            create_test_candles_sweep_low(4480.0),
            create_test_candles_no_sweep(4530.0, 4480.0),
        ]
        
        for candles in test_cases:
            result = identifier.identify(candles)
            # Must be one of three enum values
            assert result.participant_type in [
                ParticipantType.BUYERS,
                ParticipantType.SELLERS,
                ParticipantType.NONE
            ]
    
    def test_monotonicity_first_sweep_wins(self):
        """Test that first sweep is permanent (monotonic property)"""
        identifier = ParticipantIdentifier()
        orh_prev = 4530.0
        orl_prev = 4480.0
        base_time = datetime(2024, 1, 2, 9, 30)
        
        # Sellers sweep first, buyers sweep later
        candles = [
            Candle(base_time, 4500.0, 4510.0, 4475.0, 4505.0, 1000.0),  # Sellers sweep
            Candle(base_time + timedelta(minutes=1), 4505.0, 4535.0, 4500.0, 4530.0, 1100.0),  # Buyers sweep
        ]
        
        participant, _, _ = identifier.identify_participant(candles, orh_prev, orl_prev)
        
        # First sweep (SELLERS) wins - later sweep ignored
        assert participant == ParticipantType.SELLERS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
