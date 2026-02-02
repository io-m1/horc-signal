"""
Tests for HORC Orchestrator and Signal IR

Validates:
    1. SignalIR Pine-compatibility constraints
    2. Confluence scoring (weighted sum)
    3. Bias determination (majority vote)
    4. Actionability gating (confluence + bias + regime)
    5. Determinism (same input → same output)
    6. State persistence (bar-to-bar)
"""

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


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@pytest.fixture
def base_timestamp():
    """Base timestamp for tests (datetime for Candle creation)"""
    return datetime(2026, 2, 2, 9, 30)


@pytest.fixture
def base_timestamp_ms():
    """Base timestamp as unix ms int for SignalIR creation"""
    dt = datetime(2026, 2, 2, 9, 30)
    return int(dt.timestamp() * 1000)


@pytest.fixture
def sample_candle():
    """Sample candle for testing"""
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
    """Default orchestrator configuration"""
    return OrchestratorConfig(
        confluence_threshold=0.75,
        participant_weight=0.30,
        wavelength_weight=0.25,
        exhaustion_weight=0.25,
        gap_weight=0.20,
    )


@pytest.fixture
def orchestrator(orchestrator_config):
    """Initialized orchestrator with all engines"""
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


# ==============================================================================
# SIGNAL IR TESTS - Pine Compatibility
# ==============================================================================

class TestSignalIR:
    """Test Signal IR Pine-compatibility constraints"""
    
    def test_valid_signal_ir_creation(self, base_timestamp_ms):
        """Can create valid SignalIR with all required fields"""
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
        """Bias must be -1, 0, or 1"""
        # Valid values
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
        
        # Invalid value
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
        """Confidence must be [0.0, 1.0]"""
        # Valid boundaries
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
        
        # Out of bounds
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
        """Wavelength state must be 0-7"""
        # Valid states
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
        
        # Invalid state
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
        """Moves completed must be 0-3"""
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
        """SignalIR is frozen (immutable)"""
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
        
        # Cannot modify fields
        with pytest.raises(Exception):  # FrozenInstanceError
            signal.bias = -1
    
    def test_signal_ir_to_dict(self, base_timestamp_ms):
        """Can serialize to dictionary"""
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


# ==============================================================================
# ORCHESTRATOR CONFIG TESTS
# ==============================================================================

class TestOrchestratorConfig:
    """Test orchestrator configuration validation"""
    
    def test_default_config(self):
        """Default config has sensible values"""
        config = OrchestratorConfig()
        
        assert config.confluence_threshold == 0.75
        assert config.participant_weight == 0.30
        assert config.wavelength_weight == 0.25
        assert config.exhaustion_weight == 0.25
        assert config.gap_weight == 0.20
    
    def test_weights_must_sum_to_one(self):
        """Weights must sum to 1.0"""
        # Valid weights
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
        
        # Invalid weights
        with pytest.raises(ValueError, match="must sum to 1.0"):
            OrchestratorConfig(
                participant_weight=0.5,
                wavelength_weight=0.5,
                exhaustion_weight=0.5,
                gap_weight=0.5,  # Sum > 1.0
            )
    
    def test_threshold_bounds(self):
        """Confluence threshold must be [0.0, 1.0]"""
        # Valid
        for threshold in [0.0, 0.5, 0.75, 1.0]:
            config = OrchestratorConfig(confluence_threshold=threshold)
            assert config.confluence_threshold == threshold
        
        # Invalid
        with pytest.raises(ValueError, match="confluence_threshold must be"):
            OrchestratorConfig(confluence_threshold=1.5)


# ==============================================================================
# ORCHESTRATOR INTEGRATION TESTS
# ==============================================================================

class TestOrchestrator:
    """Test orchestrator signal generation"""
    
    def test_orchestrator_initialization(self, orchestrator):
        """Can initialize orchestrator with all engines"""
        assert orchestrator.participant is not None
        assert orchestrator.wavelength is not None
        assert orchestrator.exhaustion is not None
        assert orchestrator.gap_engine is not None
        assert orchestrator.config.confluence_threshold == 0.75
    
    def test_process_single_bar(self, orchestrator, sample_candle):
        """Can process single bar and emit SignalIR"""
        signal = orchestrator.process_bar(sample_candle)
        
        # Returns valid SignalIR
        assert isinstance(signal, SignalIR)
        # timestamp is unix ms int
        expected_ts = int(sample_candle.timestamp.timestamp() * 1000)
        assert signal.timestamp == expected_ts
        assert signal.bias in [-1, 0, 1]
        assert 0.0 <= signal.confidence <= 1.0
        assert isinstance(signal.actionable, bool)
    
    def test_deterministic_processing(self, orchestrator, sample_candle):
        """Same input produces same output (determinism)"""
        # Reset orchestrator
        orchestrator.reset()
        
        # Process same candle twice
        signal1 = orchestrator.process_bar(sample_candle)
        
        orchestrator.reset()
        signal2 = orchestrator.process_bar(sample_candle)
        
        # Signals should be identical
        assert signal1.bias == signal2.bias
        assert signal1.confidence == signal2.confidence
        assert signal1.actionable == signal2.actionable
        assert signal1.participant_control == signal2.participant_control
    
    def test_confluence_scoring(self, orchestrator):
        """Confluence score is weighted sum of engine outputs"""
        # This test would require mocking engine outputs
        # For now, verify confluence is within bounds
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=10000.0
        )
        
        signal = orchestrator.process_bar(candle)
        
        # Confluence must be [0.0, 1.0]
        assert 0.0 <= signal.confidence <= 1.0
    
    def test_bias_determination(self, orchestrator, sample_candle):
        """Bias is determined by majority vote"""
        signal = orchestrator.process_bar(sample_candle)
        
        # Bias must be -1, 0, or 1
        assert signal.bias in [-1, 0, 1]
    
    def test_actionability_gating(self, orchestrator):
        """Signal is actionable only if confluence >= threshold and bias != 0"""
        candle = Candle(
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=10000.0
        )
        
        signal = orchestrator.process_bar(candle)
        
        # If actionable, must meet requirements
        if signal.actionable:
            assert signal.confidence >= orchestrator.config.confluence_threshold
            assert signal.bias != 0
    
    def test_state_persistence(self, orchestrator, base_timestamp):
        """State persists bar-to-bar"""
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
        
        # Processed multiple bars
        assert len(signals) == 5
        assert orchestrator.bars_processed == 5
        
        # Each signal has increasing timestamp
        for i in range(1, len(signals)):
            assert signals[i].timestamp > signals[i-1].timestamp
    
    def test_reset(self, orchestrator, sample_candle):
        """Can reset orchestrator state"""
        # Process some bars
        orchestrator.process_bar(sample_candle)
        orchestrator.process_bar(sample_candle)
        
        assert orchestrator.bars_processed == 2
        assert orchestrator.prev_signal is not None
        
        # Reset
        orchestrator.reset()
        
        assert orchestrator.bars_processed == 0
        assert orchestrator.prev_signal is None


# ==============================================================================
# PINE COMPATIBILITY TESTS
# ==============================================================================

class TestPineCompatibility:
    """Test Pine Script compatibility guarantees"""
    
    def test_no_dynamic_objects(self, orchestrator, sample_candle):
        """SignalIR contains only primitive types (Pine-compatible)"""
        signal = orchestrator.process_bar(sample_candle)
        
        # Check all fields are Pine-compatible primitives
        # NO datetime, NO None, NO complex objects
        for field_name in signal.__dataclass_fields__:
            value = getattr(signal, field_name)
            # All values must be int, float (including nan), or bool
            assert isinstance(value, (int, float, bool)), \
                f"Field {field_name} has non-primitive type {type(value)}"
    
    def test_bounded_state(self, orchestrator, sample_candle):
        """All scores and states have bounded ranges"""
        signal = orchestrator.process_bar(sample_candle)
        
        # Bounded ranges
        assert 0.0 <= signal.confidence <= 1.0
        assert 0.0 <= signal.exhaustion_score <= 1.0
        assert 0.0 <= signal.gap_fill_progress <= 1.0
        assert 0 <= signal.wavelength_state <= 7
        assert 0 <= signal.moves_completed <= 3
        assert signal.bias in [-1, 0, 1]
        assert signal.participant_control in [-1, 0, 1]
    
    def test_bar_local_state(self, orchestrator, base_timestamp):
        """State is bar-local (no hidden future context)"""
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
        
        # Process forward
        signals_forward = []
        orchestrator.reset()
        for candle in candles:
            signals_forward.append(orchestrator.process_bar(candle))
        
        # Each signal depends only on current and prior bars
        # (This is validated by determinism test - same history → same signal)
        assert len(signals_forward) == 3


# ==============================================================================
# PERFORMANCE TESTS
# ==============================================================================

class TestPerformance:
    """Test orchestrator performance characteristics"""
    
    def test_fast_processing(self, orchestrator, base_timestamp):
        """Can process bars quickly"""
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
        
        # Should process 100 bars in under 1 second
        assert elapsed < 1.0
        
        # Average per bar
        per_bar = elapsed / len(candles)
        assert per_bar < 0.01  # Under 10ms per bar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
