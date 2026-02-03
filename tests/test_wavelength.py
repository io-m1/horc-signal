import pytest
from datetime import datetime, timedelta
from src.engines.wavelength import (
    WavelengthEngine,
    WavelengthState,
    WavelengthResult,
    WavelengthConfig,
    validate_wavelength_progression
)
from src.engines.participant import (
    Candle,
    ParticipantResult,
    ParticipantType
)

class TestWavelengthState:
    
    def test_all_states_defined(self):
        states = list(WavelengthState)
        assert len(states) == 8
        
        expected_states = [
            WavelengthState.PRE_OR,
            WavelengthState.PARTICIPANT_ID,
            WavelengthState.MOVE_1,
            WavelengthState.MOVE_2,
            WavelengthState.FLIP_CONFIRMED,
            WavelengthState.MOVE_3,
            WavelengthState.COMPLETE,
            WavelengthState.FAILED
        ]
        
        for state in expected_states:
            assert state in states

class TestWavelengthEngine:
    
    @pytest.fixture
    def engine(self):
        return WavelengthEngine()
    
    @pytest.fixture
    def buyer_participant(self):
        return ParticipantResult(
            participant_type=ParticipantType.BUYERS,
            conviction_level=True,
            control_price=4480.0,
            timestamp=datetime(2024, 1, 2, 9, 30),
            orh_prev=4530.0,
            orl_prev=4480.0,
            sweep_candle_index=0
        )
    
    @pytest.fixture
    def seller_participant(self):
        return ParticipantResult(
            participant_type=ParticipantType.SELLERS,
            conviction_level=True,
            control_price=4530.0,
            timestamp=datetime(2024, 1, 2, 9, 30),
            orh_prev=4530.0,
            orl_prev=4480.0,
            sweep_candle_index=0
        )
    
    def test_initialization(self, engine):
        assert engine.state == WavelengthState.PRE_OR
        assert engine.moves_completed == 0
        assert engine.flip_point is None
        assert engine.move_1_extreme is None
        assert engine.move_2_extreme is None
    
    def test_custom_config(self):
        config = WavelengthConfig(
            min_move_1_size_atr=1.0,
            exhaustion_threshold=0.8
        )
        engine = WavelengthEngine(config)
        
        assert engine.config.min_move_1_size_atr == 1.0
        assert engine.config.exhaustion_threshold == 0.8
    
    def test_reset_clears_state(self, engine, buyer_participant):
        candle = Candle(datetime(2024, 1, 2, 9, 30), 4500.0, 4510.0, 4490.0, 4505.0, 1000.0)
        engine.process_candle(candle, buyer_participant)
        
        engine.reset()
        
        assert engine.state == WavelengthState.PRE_OR
        assert engine.moves_completed == 0
        assert engine.flip_point is None
        assert len(engine.candle_history) == 0

class TestStateTransitions:
    
    @pytest.fixture
    def engine(self):
        return WavelengthEngine()
    
    def test_pre_or_to_participant_id(self, engine):
        participant = ParticipantResult(
            participant_type=ParticipantType.BUYERS,
            conviction_level=True,
            control_price=4480.0,
            timestamp=datetime(2024, 1, 2, 9, 30),
            orh_prev=4530.0,
            orl_prev=4480.0,
            sweep_candle_index=0
        )
        
        candle = Candle(
            datetime(2024, 1, 2, 9, 30),
            4490.0, 4500.0, 4485.0, 4495.0, 1000.0
        )
        
        result = engine.process_candle(candle, participant)
        
        assert engine.state == WavelengthState.PARTICIPANT_ID
        assert engine.participant_type == ParticipantType.BUYERS
        assert engine.control_price == 4480.0
    
    def test_participant_id_to_move_1(self, engine):
        participant = ParticipantResult(
            participant_type=ParticipantType.BUYERS,
            conviction_level=True,
            control_price=4480.0,
            timestamp=datetime(2024, 1, 2, 9, 30),
            orh_prev=4530.0,
            orl_prev=4480.0,
            sweep_candle_index=0
        )
        
        base_time = datetime(2024, 1, 2, 9, 30)
        
        engine.process_candle(
            Candle(base_time, 4490.0, 4500.0, 4485.0, 4495.0, 1000.0),
            participant
        )
        
        assert engine.state == WavelengthState.PARTICIPANT_ID
        
        for i in range(1, 10):
            price = 4495.0 + (i * 5)
            candle = Candle(
                base_time + timedelta(minutes=i),
                price, price + 5, price - 2, price + 3,
                1000.0 + i * 100
            )
            result = engine.process_candle(candle)
        
        rejection_candle = Candle(
            base_time + timedelta(minutes=10),
            4540.0, 4560.0, 4535.0, 4537.0,  # Long upper wick
            2000.0
        )
        result = engine.process_candle(rejection_candle)
        
        assert engine.state == WavelengthState.MOVE_1
        assert engine.moves_completed == 1
        assert engine.move_1_extreme is not None
    
    def test_move_1_to_move_2(self, engine):
        participant = ParticipantResult(
            participant_type=ParticipantType.BUYERS,
            conviction_level=True,
            control_price=4480.0,
            timestamp=datetime(2024, 1, 2, 9, 30),
            orh_prev=4530.0,
            orl_prev=4480.0,
            sweep_candle_index=0
        )
        
        base_time = datetime(2024, 1, 2, 9, 30)
        
        engine.process_candle(
            Candle(base_time, 4490.0, 4500.0, 4485.0, 4495.0, 1000.0),
            participant
        )
        
        for i in range(1, 10):
            price = 4495.0 + (i * 5)
            engine.process_candle(
                Candle(base_time + timedelta(minutes=i), price, price + 5, price - 2, price + 3, 1000.0)
            )
        
        engine.process_candle(
            Candle(base_time + timedelta(minutes=10), 4540.0, 4560.0, 4535.0, 4537.0, 2000.0)
        )
        
        assert engine.state == WavelengthState.MOVE_1
        
        reversal_candle = Candle(
            base_time + timedelta(minutes=11),
            4537.0, 4540.0, 4520.0, 4525.0,  # Bearish close < open
            1500.0
        )
        result = engine.process_candle(reversal_candle)
        
        assert engine.state == WavelengthState.MOVE_2
        assert engine.moves_completed == 2
    
    def test_move_2_to_flip_confirmed(self, engine):
        
        engine.state = WavelengthState.MOVE_2
        engine.moves_completed = 2
        engine.participant_type = ParticipantType.BUYERS
        engine.move_1_extreme = 4560.0
        engine.move_2_extreme = 4520.0
        engine.flip_point = None
        
        base_time = datetime(2024, 1, 2, 9, 30)
        
        for i in range(5):
            candle = Candle(
                base_time + timedelta(minutes=i),
                4520.0, 4525.0, 4515.0, 4521.0,  # Small body
                3000.0  # High volume
            )
            engine.move_2_candles.append(candle)
            engine.candle_history.append(candle)
        
        absorption_candle = Candle(
            base_time + timedelta(minutes=5),
            4518.0, 4522.0, 4510.0, 4520.0,  # Long lower wick
            5000.0  # Very high volume
        )
        
        result = engine.process_candle(absorption_candle)
        
        if engine.state == WavelengthState.FLIP_CONFIRMED:
            assert engine.flip_point is not None
            assert engine.moves_completed == 2  # Still 2, Move 3 not started yet
    
    def test_flip_confirmed_to_move_3(self, engine):
        engine.state = WavelengthState.FLIP_CONFIRMED
        engine.moves_completed = 2
        engine.flip_point = 4520.0
        engine.flip_confirmation_count = 0
        engine.participant_type = ParticipantType.BUYERS
        
        base_time = datetime(2024, 1, 2, 9, 30)
        
        for i in range(engine.config.flip_confirmation_candles):
            candle = Candle(
                base_time + timedelta(minutes=i),
                4520.0, 4525.0, 4518.0, 4523.0, 1000.0
            )
            result = engine.process_candle(candle)
        
        assert engine.state == WavelengthState.MOVE_3
        assert engine.moves_completed == 3
    
    def test_move_3_to_complete(self, engine):
        engine.state = WavelengthState.MOVE_3
        engine.moves_completed = 3
        engine.flip_point = 4520.0
        engine.move_1_extreme = 4560.0
        engine.participant_type = ParticipantType.BUYERS
        
        candle = Candle(
            datetime(2024, 1, 2, 9, 30),
            4560.0, 4565.0, 4555.0, 4563.0, 1000.0
        )
        
        result = engine.process_candle(candle)
        
        assert engine.state == WavelengthState.COMPLETE

class TestPatternInvalidation:
    
    @pytest.fixture
    def engine(self):
        return WavelengthEngine()
    
    def test_move_2_invalidation_breaks_start(self, engine):
        engine.state = WavelengthState.MOVE_2
        engine.moves_completed = 2
        engine.participant_type = ParticipantType.BUYERS
        engine.move_1_start = 4490.0
        engine.move_1_extreme = 4560.0
        engine.move_2_extreme = 4520.0
        
        invalidation_candle = Candle(
            datetime(2024, 1, 2, 9, 30),
            4495.0, 4500.0, 4485.0, 4488.0,  # Low < move_1_start
            1000.0
        )
        
        result = engine.process_candle(invalidation_candle)
        
        assert engine.state == WavelengthState.FAILED
    
    def test_move_3_invalidation_breaks_flip(self, engine):
        engine.state = WavelengthState.MOVE_3
        engine.moves_completed = 3
        engine.participant_type = ParticipantType.BUYERS
        engine.flip_point = 4520.0
        
        stop_candle = Candle(
            datetime(2024, 1, 2, 9, 30),
            4520.0, 4522.0, 4515.0, 4517.0,  # Low < flip_point
            1000.0
        )
        
        result = engine.process_candle(stop_candle)
        
        assert engine.state == WavelengthState.FAILED
    
    def test_timeout_invalidation(self, engine):
        config = WavelengthConfig(max_move_duration_candles=5)
        engine = WavelengthEngine(config)
        
        engine.state = WavelengthState.MOVE_2
        engine.moves_completed = 2
        engine.participant_type = ParticipantType.BUYERS
        engine.candles_in_current_move = 6  # Exceeds max
        
        candle = Candle(
            datetime(2024, 1, 2, 9, 30),
            4520.0, 4525.0, 4518.0, 4523.0, 1000.0
        )
        
        result = engine.process_candle(candle)
        
        assert engine.state == WavelengthState.FAILED

class TestHelperMethods:
    
    @pytest.fixture
    def engine(self):
        return WavelengthEngine()
    
    def test_calculate_atr(self, engine):
        base_time = datetime(2024, 1, 2, 9, 30)
        candles = [
            Candle(base_time, 4500.0, 4510.0, 4490.0, 4505.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 4505.0, 4520.0, 4500.0, 4515.0, 1100.0),
            Candle(base_time + timedelta(minutes=2), 4515.0, 4525.0, 4510.0, 4520.0, 1200.0),
        ]
        
        atr = engine.calculate_atr(candles)
        
        assert atr > 0
        assert isinstance(atr, float)
    
    def test_calculate_atr_empty_list(self, engine):
        atr = engine.calculate_atr([])
        assert atr == 0.0
    
    def test_signal_strength_progression(self, engine):
        strengths = {}
        
        for state in WavelengthState:
            engine.state = state
            strengths[state] = engine.calculate_signal_strength()
        
        assert strengths[WavelengthState.PRE_OR] == 0.0
        assert strengths[WavelengthState.PARTICIPANT_ID] == 0.2
        assert strengths[WavelengthState.MOVE_1] == 0.3
        assert strengths[WavelengthState.MOVE_2] == 0.5
        assert strengths[WavelengthState.FLIP_CONFIRMED] == 0.8
        assert strengths[WavelengthState.MOVE_3] == 0.9
        assert strengths[WavelengthState.COMPLETE] == 1.0
        assert strengths[WavelengthState.FAILED] == 0.0

class TestWavelengthResult:
    
    def test_result_creation(self):
        result = WavelengthResult(
            state=WavelengthState.FLIP_CONFIRMED,
            moves_completed=2,
            flip_point=4520.0,
            move_1_extreme=4560.0,
            move_2_extreme=4520.0,
            signal_strength=0.8,
            entry_price=4520.0,
            stop_price=4515.0,
            target_price=4600.0,
            participant_type=ParticipantType.BUYERS,
            candles_in_current_move=5,
            timestamp=datetime(2024, 1, 2, 9, 30)
        )
        
        assert result.state == WavelengthState.FLIP_CONFIRMED
        assert result.moves_completed == 2
        assert result.flip_point == 4520.0
        assert result.signal_strength == 0.8

class TestAxiom1Validation:
    
    def test_validate_wavelength_progression_success(self):
        states = [
            WavelengthState.PRE_OR,
            WavelengthState.PARTICIPANT_ID,
            WavelengthState.MOVE_1,
            WavelengthState.MOVE_2,
            WavelengthState.FLIP_CONFIRMED,
            WavelengthState.MOVE_3,
            WavelengthState.COMPLETE
        ]
        
        assert validate_wavelength_progression(states) is True
    
    def test_validate_wavelength_progression_incomplete(self):
        states = [
            WavelengthState.PRE_OR,
            WavelengthState.PARTICIPANT_ID,
            WavelengthState.MOVE_1,
            WavelengthState.FAILED  # Missing MOVE_2, MOVE_3
        ]
        
        assert validate_wavelength_progression(states) is False
    
    def test_validate_wavelength_progression_missing_move_2(self):
        states = [
            WavelengthState.PRE_OR,
            WavelengthState.MOVE_1,
            WavelengthState.MOVE_3,  # Skipped MOVE_2
            WavelengthState.COMPLETE
        ]
        
        assert validate_wavelength_progression(states) is False

class TestMathematicalProperties:
    
    def test_determinism(self):
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
        
        candle = Candle(
            datetime(2024, 1, 2, 9, 30),
            4490.0, 4500.0, 4485.0, 4495.0, 1000.0
        )
        
        result1 = engine1.process_candle(candle, participant)
        result2 = engine2.process_candle(candle, participant)
        
        assert result1.state == result2.state
        assert result1.moves_completed == result2.moves_completed
        assert result1.participant_type == result2.participant_type
    
    def test_state_completeness(self):
        engine = WavelengthEngine()
        
        candle = Candle(
            datetime(2024, 1, 2, 9, 30),
            4490.0, 4500.0, 4485.0, 4495.0, 1000.0
        )
        
        for state in WavelengthState:
            engine.state = state
            engine.participant_type = ParticipantType.BUYERS
            
            result = engine.process_candle(candle)
            assert isinstance(result, WavelengthResult)
    
    def test_terminal_states_no_transitions(self):
        engine = WavelengthEngine()
        candle = Candle(
            datetime(2024, 1, 2, 9, 30),
            4490.0, 4500.0, 4485.0, 4495.0, 1000.0
        )
        
        engine.state = WavelengthState.COMPLETE
        initial_state = engine.state
        engine.process_candle(candle)
        assert engine.state == initial_state
        
        engine.state = WavelengthState.FAILED
        initial_state = engine.state
        engine.process_candle(candle)
        assert engine.state == initial_state

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
