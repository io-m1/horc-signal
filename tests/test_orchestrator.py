import pytest
import math
from datetime import datetime, timedelta
from typing import List

from src.engines import (
    ParticipantIdentifier,
    ParticipantType,
    ParticipantResult,
    WavelengthEngine,
    WavelengthConfig,
    WavelengthState,
    WavelengthResult,
    ExhaustionDetector,
    ExhaustionConfig,
    ExhaustionResult,
    FuturesGapEngine,
    GapConfig,
    GapAnalysisResult,
    Candle,
)
from src.core import (
    SignalIR,
    HORCOrchestrator,
    WAVELENGTH_STATE,
    GAP_TYPE,
    BIAS,
    PARTICIPANT_CONTROL,
)
from src.core.orchestrator import OrchestratorConfig

@pytest.fixture
def base_timestamp():
    return datetime(2026, 2, 2, 9, 30)

@pytest.fixture
def base_timestamp_ms():
    dt = datetime(2026, 2, 2, 9, 30)
    return int(dt.timestamp() * 1000)

@pytest.fixture
def sample_candle():
    dt = datetime(2026, 2, 2, 9, 30)
    return Candle(
        timestamp=dt,
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=10000.0
    )

@pytest.fixture
def orchestrator_config():
    return OrchestratorConfig(
        confluence_threshold=0.75,
        participant_weight=0.30,
        wavelength_weight=0.25,
        exhaustion_weight=0.25,
        gap_weight=0.20,
    )

@pytest.fixture
def orchestrator(orchestrator_config):
    participant = ParticipantIdentifier()  # Uses default dict config
    wavelength = WavelengthEngine(WavelengthConfig())
    exhaustion = ExhaustionDetector(ExhaustionConfig())
    gap_engine = FuturesGapEngine(GapConfig())
    
    return HORCOrchestrator(
        participant,
        wavelength,
        exhaustion,
        gap_engine,
        orchestrator_config
    )

class TestSignalIR:
    
    def test_valid_signal_ir_creation(self, base_timestamp_ms):
        signal = SignalIR(
            timestamp=base_timestamp_ms,
            bias=1,
            actionable=True,
            confidence=0.85,
            participant_control=1,
            wavelength_state=0,
            moves_completed=2,
            current_extreme_high=105.0,
            current_extreme_low=95.0,
            exhaustion_score=0.7,
            in_exhaustion_zone=True,
            active_gap_type=1,
            gap_fill_progress=0.5,
            has_futures_target=True,
            futures_target=110.0,
            debug_flags=0x03,
        )
        
        assert signal.bias == 1
        assert signal.actionable is True
        assert signal.confidence == 0.85
        assert signal.has_futures_target is True
        assert signal.futures_target == 110.0
    
    def test_bias_validation(self, base_timestamp_ms):
        for bias in [-1, 0, 1]:
            signal = SignalIR(
                timestamp=base_timestamp_ms,
                bias=bias,
                actionable=False,
                confidence=0.0,
                participant_control=0,
                wavelength_state=0,
                moves_completed=0,
                current_extreme_high=100.0,
                current_extreme_low=90.0,
                exhaustion_score=0.0,
                in_exhaustion_zone=False,
                active_gap_type=0,
                gap_fill_progress=0.0,
                has_futures_target=False,
                futures_target=math.nan,
                debug_flags=0,
            )
            assert signal.bias == bias
        
        with pytest.raises(ValueError, match="bias must be -1, 0, or 1"):
            SignalIR(
                timestamp=base_timestamp_ms,
                bias=2,  # Invalid
                actionable=False,
                confidence=0.0,
                participant_control=0,
                wavelength_state=0,
                moves_completed=0,
                current_extreme_high=100.0,
                current_extreme_low=90.0,
                exhaustion_score=0.0,
                in_exhaustion_zone=False,
                active_gap_type=0,
                gap_fill_progress=0.0,
                has_futures_target=False,
                futures_target=math.nan,
                debug_flags=0,
            )
    
    def test_confidence_bounds(self, base_timestamp_ms):
        for conf in [0.0, 0.5, 1.0]:
            signal = SignalIR(
                timestamp=base_timestamp_ms,
                bias=0,
                actionable=False,
                confidence=conf,
                participant_control=0,
                wavelength_state=0,
                moves_completed=0,
                current_extreme_high=100.0,
                current_extreme_low=90.0,
                exhaustion_score=0.0,
                in_exhaustion_zone=False,
                active_gap_type=0,
                gap_fill_progress=0.0,
                has_futures_target=False,
                futures_target=math.nan,
                debug_flags=0,
            )
            assert signal.confidence == conf
        
        with pytest.raises(ValueError, match="confidence must be"):
            SignalIR(
                timestamp=base_timestamp_ms,
                bias=0,
                actionable=False,
                confidence=1.5,  # Invalid
                participant_control=0,
                wavelength_state=0,
                moves_completed=0,
                current_extreme_high=100.0,
                current_extreme_low=90.0,
                exhaustion_score=0.0,
                in_exhaustion_zone=False,
                active_gap_type=0,
                gap_fill_progress=0.0,
                has_futures_target=False,
                futures_target=math.nan,
                debug_flags=0,
            )
    
    def test_wavelength_state_validation(self, base_timestamp_ms):
        for state in range(8):
            signal = SignalIR(
                timestamp=base_timestamp_ms,
                bias=0,
                actionable=False,
                confidence=0.0,
                participant_control=0,
                wavelength_state=state,
                moves_completed=0,
                current_extreme_high=100.0,
                current_extreme_low=90.0,
                exhaustion_score=0.0,
                in_exhaustion_zone=False,
                active_gap_type=0,
                gap_fill_progress=0.0,
                has_futures_target=False,
                futures_target=math.nan,
                debug_flags=0,
            )
            assert signal.wavelength_state == state
        
        with pytest.raises(ValueError, match="wavelength_state must be 0-7"):
            SignalIR(
                timestamp=base_timestamp_ms,
                bias=0,
                actionable=False,
                confidence=0.0,
                participant_control=0,
                wavelength_state=99,  # Invalid
                moves_completed=0,
                current_extreme_high=100.0,
                current_extreme_low=90.0,
                exhaustion_score=0.0,
                in_exhaustion_zone=False,
                active_gap_type=0,
                gap_fill_progress=0.0,
                has_futures_target=False,
                futures_target=math.nan,
                debug_flags=0,
            )
    
    def test_moves_completed_bounds(self, base_timestamp_ms):
        for moves in [0, 1, 2, 3]:
            signal = SignalIR(
                timestamp=base_timestamp_ms,
                bias=0,
                actionable=False,
                confidence=0.0,
                participant_control=0,
                wavelength_state=0,
                moves_completed=moves,
                current_extreme_high=100.0,
                current_extreme_low=90.0,
                exhaustion_score=0.0,
                in_exhaustion_zone=False,
                active_gap_type=0,
                gap_fill_progress=0.0,
                has_futures_target=False,
                futures_target=math.nan,
                debug_flags=0,
            )
            assert signal.moves_completed == moves
    
    def test_signal_ir_immutable(self, base_timestamp_ms):
        signal = SignalIR(
            timestamp=base_timestamp_ms,
            bias=1,
            actionable=True,
            confidence=0.85,
            participant_control=1,
            wavelength_state=0,
            moves_completed=0,
            current_extreme_high=100.0,
            current_extreme_low=90.0,
            exhaustion_score=0.0,
            in_exhaustion_zone=False,
            active_gap_type=0,
            gap_fill_progress=0.0,
            has_futures_target=False,
            futures_target=math.nan,
            debug_flags=0,
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            signal.bias = -1
    
    def test_signal_ir_to_dict(self, base_timestamp_ms):
        signal = SignalIR(
            timestamp=base_timestamp_ms,
            bias=1,
            actionable=True,
            confidence=0.85,
            participant_control=1,
            wavelength_state=2,
            moves_completed=1,
            current_extreme_high=105.0,
            current_extreme_low=95.0,
            exhaustion_score=0.6,
            in_exhaustion_zone=False,
            active_gap_type=1,
            gap_fill_progress=0.3,
            has_futures_target=True,
            futures_target=110.0,
            debug_flags=0x05,
        )
        
        d = signal.to_dict()
        
        assert d["bias"] == 1
        assert d["actionable"] is True
        assert d["confidence"] == 0.85
        assert d["wavelength_state"] == 2
        assert d["has_futures_target"] is True
        assert d["futures_target"] == 110.0

class TestOrchestratorConfig:
    
    def test_default_config(self):
        config = OrchestratorConfig()
        
        assert config.confluence_threshold == 0.30
        assert config.participant_weight == 0.50
        assert config.wavelength_weight == 0.20
        assert config.exhaustion_weight == 0.20
        assert config.gap_weight == 0.10
        assert config.require_agreement == False
        assert config.require_strategic_context == False
    
    def test_weights_must_sum_to_one(self):
        config = OrchestratorConfig(
            participant_weight=0.4,
            wavelength_weight=0.3,
            exhaustion_weight=0.2,
            gap_weight=0.1,
        )
        assert abs(sum([
            config.participant_weight,
            config.wavelength_weight,
            config.exhaustion_weight,
            config.gap_weight
        ]) - 1.0) < 0.001
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            OrchestratorConfig(
                participant_weight=0.5,
                wavelength_weight=0.5,
                exhaustion_weight=0.5,
                gap_weight=0.5,  # Sum > 1.0
            )
    
    def test_threshold_bounds(self):
        for threshold in [0.0, 0.5, 0.75, 1.0]:
            config = OrchestratorConfig(confluence_threshold=threshold)
            assert config.confluence_threshold == threshold
        
        with pytest.raises(ValueError, match="confluence_threshold must be"):
            OrchestratorConfig(confluence_threshold=1.5)

class TestOrchestrator:
    
    def test_orchestrator_initialization(self, orchestrator):
        assert orchestrator.participant is not None
        assert orchestrator.wavelength is not None
        assert orchestrator.exhaustion is not None
        assert orchestrator.gap_engine is not None
        assert orchestrator.config.confluence_threshold == 0.75
    
    def test_process_single_bar(self, orchestrator, sample_candle):
        signal = orchestrator.process_bar(sample_candle)
        
        assert isinstance(signal, SignalIR)
        expected_ts = int(sample_candle.timestamp.timestamp() * 1000)
        assert signal.timestamp == expected_ts
        assert signal.bias in [-1, 0, 1]
        assert 0.0 <= signal.confidence <= 1.0
        assert isinstance(signal.actionable, bool)
    
    def test_deterministic_processing(self, orchestrator, sample_candle):
        orchestrator.reset()
        
        signal1 = orchestrator.process_bar(sample_candle)
        
        orchestrator.reset()
        signal2 = orchestrator.process_bar(sample_candle)
        
        assert signal1.bias == signal2.bias
        assert signal1.confidence == signal2.confidence
        assert signal1.actionable == signal2.actionable
        assert signal1.participant_control == signal2.participant_control
    
    def test_confluence_scoring(self, orchestrator):
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=10000.0
        )
        
        signal = orchestrator.process_bar(candle)
        
        assert 0.0 <= signal.confidence <= 1.0
    
    def test_bias_determination(self, orchestrator, sample_candle):
        signal = orchestrator.process_bar(sample_candle)
        
        assert signal.bias in [-1, 0, 1]
    
    def test_actionability_gating(self, orchestrator):
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=10000.0
        )
        
        signal = orchestrator.process_bar(candle)
        
        if signal.actionable:
            assert signal.confidence >= orchestrator.config.confluence_threshold
            assert signal.bias != 0
    
    def test_state_persistence(self, orchestrator, base_timestamp):
        candles = [
            Candle(
                timestamp=base_timestamp + timedelta(minutes=i),
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=102.0 + i,
                volume=10000.0
            )
            for i in range(5)
        ]
        
        signals = []
        for candle in candles:
            signal = orchestrator.process_bar(candle)
            signals.append(signal)
        
        assert len(signals) == 5
        assert orchestrator.bars_processed == 5
        
        for i in range(1, len(signals)):
            assert signals[i].timestamp > signals[i-1].timestamp
    
    def test_reset(self, orchestrator, sample_candle):
        orchestrator.process_bar(sample_candle)
        orchestrator.process_bar(sample_candle)
        
        assert orchestrator.bars_processed == 2
        assert orchestrator.prev_signal is not None
        
        orchestrator.reset()
        
        assert orchestrator.bars_processed == 0
        assert orchestrator.prev_signal is None

class TestPineCompatibility:
    
    def test_no_dynamic_objects(self, orchestrator, sample_candle):
        signal = orchestrator.process_bar(sample_candle)
        
        for field_name in signal.__dataclass_fields__:
            value = getattr(signal, field_name)
            assert isinstance(value, (int, float, bool)), \
                f"Field {field_name} has non-primitive type {type(value)}"
    
    def test_bounded_state(self, orchestrator, sample_candle):
        signal = orchestrator.process_bar(sample_candle)
        
        assert 0.0 <= signal.confidence <= 1.0
        assert 0.0 <= signal.exhaustion_score <= 1.0
        assert 0.0 <= signal.gap_fill_progress <= 1.0
        assert 0 <= signal.wavelength_state <= 7
        assert 0 <= signal.moves_completed <= 3
        assert signal.bias in [-1, 0, 1]
        assert signal.participant_control in [-1, 0, 1]
    
    def test_bar_local_state(self, orchestrator, base_timestamp):
        candles = [
            Candle(
                timestamp=base_timestamp + timedelta(minutes=i),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=10000.0
            )
            for i in range(3)
        ]
        
        signals_forward = []
        orchestrator.reset()
        for candle in candles:
            signals_forward.append(orchestrator.process_bar(candle))
        
        assert len(signals_forward) == 3

class TestPerformance:
    
    def test_fast_processing(self, orchestrator, base_timestamp):
        import time
        
        candles = [
            Candle(
                timestamp=base_timestamp + timedelta(minutes=i),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=10000.0
            )
            for i in range(100)
        ]
        
        start = time.time()
        for candle in candles:
            orchestrator.process_bar(candle)
        elapsed = time.time() - start
        
        assert elapsed < 1.0
        
        per_bar = elapsed / len(candles)
        assert per_bar < 0.01  # Under 10ms per bar

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
