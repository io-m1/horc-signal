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
    
    def test_valid_candle_creation(self):
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
    
    @pytest.fixture
    def identifier(self):
        return ParticipantIdentifier()
    
    @pytest.fixture
    def previous_session_candles(self):
        base_time = datetime(2024, 1, 1, 9, 30)
        return [
            Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4510.0, 4525.0, 4505.0, 4520.0, 1100.0),
            Candle(base_time + timedelta(minutes=2), 4520.0, 4530.0, 4515.0, 4525.0, 1200.0),
        ]
    
    def test_initialization_default_config(self, identifier):
        assert identifier.or_lookback_sessions == 1
        assert identifier.min_conviction_threshold == 0.8
        assert identifier.max_first_move_candles == 3
        assert identifier.prev_session_candles == []
    
    def test_initialization_custom_config(self):
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
        orh, orl = identifier.get_opening_range(previous_session_candles)
        
        assert orh == 4530.0
        assert orl == 4480.0
    
    def test_get_opening_range_empty_list_raises_error(self, identifier):
        with pytest.raises(ValueError, match="Cannot calculate opening range from empty"):
            identifier.get_opening_range([])
    
    def test_get_opening_range_single_candle(self, identifier):
        candle = Candle(datetime(2024, 1, 1), 4500.0, 4510.0, 4490.0, 4505.0, 1000.0)
        orh, orl = identifier.get_opening_range([candle])
        
        assert orh == 4510.0
        assert orl == 4490.0

class TestIdentifyParticipant:
    
    @pytest.fixture
    def identifier(self):
        return ParticipantIdentifier()
    
    def test_sellers_identified_sweep_low(self, identifier):
        orh_prev = 4530.0
        orl_prev = 4480.0
        
        candles = create_test_candles_sweep_low(orl_prev)
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.SELLERS
        assert conviction is True
        assert sweep_idx == 0
    
    def test_buyers_identified_sweep_high(self, identifier):
        orh_prev = 4530.0
        orl_prev = 4480.0
        
        candles = create_test_candles_sweep_high(orh_prev)
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.BUYERS
        assert conviction is True
        assert sweep_idx == 0
    
    def test_none_identified_no_sweep(self, identifier):
        orh_prev = 4530.0
        orl_prev = 4480.0
        
        candles = create_test_candles_no_sweep(orh_prev, orl_prev)
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.NONE
        assert conviction is False
        assert sweep_idx is None
    
    def test_empty_candles_returns_none(self, identifier):
        participant, conviction, sweep_idx = identifier.identify_participant(
            [], 4530.0, 4480.0
        )
        
        assert participant == ParticipantType.NONE
        assert conviction is False
        assert sweep_idx is None
    
    def test_exact_level_touch_counts_as_sweep(self, identifier):
        orh_prev = 4530.0
        orl_prev = 4480.0
        
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
        orh_prev = 4530.0
        orl_prev = 4480.0
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time, 4500.0, 4510.0, 4490.0, 4505.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4505.0, 4535.0, 4500.0, 4530.0, 1100.0)
        ]
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.BUYERS
        assert conviction is True
        assert sweep_idx == 1  # Second candle (index 1)
    
    def test_first_sweep_wins_if_both_levels_swept(self, identifier):
        orh_prev = 4530.0
        orl_prev = 4480.0
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time, 4500.0, 4510.0, 4475.0, 4505.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4505.0, 4535.0, 4500.0, 4530.0, 1100.0)
        ]
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.SELLERS
        assert sweep_idx == 0
    
    def test_respects_max_first_move_candles_config(self):
        identifier = ParticipantIdentifier({'max_first_move_candles': 2})
        orh_prev = 4530.0
        orl_prev = 4480.0
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time, 4500.0, 4510.0, 4490.0, 4505.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4505.0, 4515.0, 4495.0, 4510.0, 1100.0),
            Candle(base_time + timedelta(minutes=2), 4510.0, 4535.0, 4505.0, 4530.0, 1200.0)
        ]
        
        participant, conviction, sweep_idx = identifier.identify_participant(
            candles, orh_prev, orl_prev
        )
        
        assert participant == ParticipantType.NONE
        assert conviction is False

class TestIdentifyMethod:
    
    @pytest.fixture
    def identifier(self):
        return ParticipantIdentifier()
    
    @pytest.fixture
    def setup_previous_session(self, identifier):
        base_time = datetime(2024, 1, 1, 9, 30)
        identifier.prev_session_candles = [
            Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0),
        ]
    
    def test_identify_raises_error_without_previous_session(self, identifier):
        current_candles = [
            Candle(datetime(2024, 1, 2, 9, 30), 4500.0, 4510.0, 4475.0, 4505.0, 1000.0)
        ]
        
        with pytest.raises(ValueError, match="Previous session candles not set"):
            identifier.identify(current_candles)
    
    def test_identify_buyers_complete_result(self, identifier, setup_previous_session):
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
        current_candles = create_test_candles_sweep_low(4480.0)
        
        result = identifier.identify(current_candles)
        
        assert result.participant_type == ParticipantType.SELLERS
        assert result.conviction_level is True
        assert result.control_price == 4480.0  # ORL_prev
        assert result.orh_prev == 4530.0
        assert result.orl_prev == 4480.0
        assert result.sweep_candle_index == 0
    
    def test_identify_none_complete_result(self, identifier, setup_previous_session):
        current_candles = create_test_candles_no_sweep(4530.0, 4480.0)
        
        result = identifier.identify(current_candles)
        
        assert result.participant_type == ParticipantType.NONE
        assert result.conviction_level is False
        assert result.control_price is None  # No control price when NONE
        assert result.sweep_candle_index is None
    
    def test_identify_empty_current_candles(self, identifier, setup_previous_session):
        result = identifier.identify([])
        
        assert result.participant_type == ParticipantType.NONE
        assert result.conviction_level is False
        assert result.control_price is None
        assert result.orh_prev == 4530.0  # Still calculated from prev session
        assert result.orl_prev == 4480.0

class TestSessionManagement:
    
    def test_update_session_data(self):
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
        identifier = ParticipantIdentifier()
        base_time = datetime(2024, 1, 1, 9, 30)
        
        original = [
            Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
        ]
        
        identifier.update_session_data(original)
        
        original.append(Candle(base_time + timedelta(minutes=1), 4510.0, 4530.0, 4505.0, 4525.0, 1100.0))
        
        assert len(identifier.prev_session_candles) == 1
    
    def test_reset_clears_session_data(self):
        identifier = ParticipantIdentifier()
        base_time = datetime(2024, 1, 1, 9, 30)
        
        identifier.prev_session_candles = [
            Candle(base_time, 4500.0, 4520.0, 4480.0, 4510.0, 1000.0),
        ]
        
        identifier.reset()
        
        assert identifier.prev_session_candles == []

class TestMathematicalProperties:
    
    def test_determinism(self):
        identifier1 = ParticipantIdentifier()
        identifier2 = ParticipantIdentifier()
        
        base_time = datetime(2024, 1, 1, 9, 30)
        prev_session = [Candle(base_time, 4500.0, 4530.0, 4480.0, 4520.0, 1000.0)]
        current = create_test_candles_sweep_high(4530.0)
        
        identifier1.prev_session_candles = prev_session
        identifier2.prev_session_candles = prev_session
        
        result1 = identifier1.identify(current)
        result2 = identifier2.identify(current)
        
        assert result1.participant_type == result2.participant_type
        assert result1.conviction_level == result2.conviction_level
        assert result1.control_price == result2.control_price
    
    def test_binary_output(self):
        identifier = ParticipantIdentifier()
        base_time = datetime(2024, 1, 1, 9, 30)
        identifier.prev_session_candles = [
            Candle(base_time, 4500.0, 4530.0, 4480.0, 4520.0, 1000.0)
        ]
        
        test_cases = [
            create_test_candles_sweep_high(4530.0),
            create_test_candles_sweep_low(4480.0),
            create_test_candles_no_sweep(4530.0, 4480.0),
        ]
        
        for candles in test_cases:
            result = identifier.identify(candles)
            assert result.participant_type in [
                ParticipantType.BUYERS,
                ParticipantType.SELLERS,
                ParticipantType.NONE
            ]
    
    def test_monotonicity_first_sweep_wins(self):
        identifier = ParticipantIdentifier()
        orh_prev = 4530.0
        orl_prev = 4480.0
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time, 4500.0, 4510.0, 4475.0, 4505.0, 1000.0),  # Sellers sweep
            Candle(base_time + timedelta(minutes=1), 4505.0, 4535.0, 4500.0, 4530.0, 1100.0),  # Buyers sweep
        ]
        
        participant, _, _ = identifier.identify_participant(candles, orh_prev, orl_prev)
        
        assert participant == ParticipantType.SELLERS

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
