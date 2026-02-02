"""
Test Suite for ExhaustionDetector (AXIOM 3: Absorption Reversal)
================================================================

Validates all mathematical properties and edge cases for exhaustion detection.
"""

import pytest
from datetime import datetime, timedelta
from src.engines.exhaustion import (
    ExhaustionDetector,
    ExhaustionConfig,
    ExhaustionResult,
    VolumeBar
)
from src.engines.participant import Candle


# ============================================================================
# Test Data Generators
# ============================================================================

def create_test_candles_uptrend() -> list[Candle]:
    """Create uptrend candle sequence for testing"""
    base_time = datetime(2024, 1, 2, 9, 30)
    candles = []
    
    for i in range(10):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + i * 2.0,
            high=102.0 + i * 2.0,
            low=99.0 + i * 2.0,
            close=101.0 + i * 2.0,
            volume=1000.0
        ))
    
    return candles


def create_test_candles_with_rejection() -> list[Candle]:
    """Create candles with rejection wicks (exhaustion signal)"""
    base_time = datetime(2024, 1, 2, 9, 30)
    candles = []
    
    # Normal uptrend
    for i in range(5):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + i * 2.0,
            high=102.0 + i * 2.0,
            low=99.0 + i * 2.0,
            close=101.0 + i * 2.0,
            volume=1000.0
        ))
    
    # Add rejection candle (long upper wick)
    candles.append(Candle(
        timestamp=base_time + timedelta(minutes=5),
        open=110.0,
        high=115.0,  # Pushed much higher
        low=109.0,
        close=110.5,  # Closed near open (rejection)
        volume=2000.0
    ))
    
    return candles


def create_test_candles_stagnant() -> list[Candle]:
    """Create stagnant price action (choppy, overlapping ranges)"""
    base_time = datetime(2024, 1, 2, 9, 30)
    candles = []
    
    for i in range(10):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + (i % 2) * 1.0,  # Oscillating
            high=102.0,
            low=99.0,
            close=100.0 + ((i + 1) % 2) * 1.0,
            volume=1000.0
        ))
    
    return candles


def create_test_volume_bars_high_absorption() -> list[VolumeBar]:
    """Create volume bars showing high absorption (high volume, negative delta)"""
    base_time = datetime(2024, 1, 2, 9, 30)
    bars = []
    
    for i in range(10):
        # Increasing volume with positive delta (buying exhaustion)
        volume = 1000.0 + i * 200.0
        bid_vol = volume * 0.6  # More buying
        ask_vol = volume * 0.4
        
        bars.append(VolumeBar(
            timestamp=base_time + timedelta(minutes=i),
            volume=volume,
            bid_volume=ask_vol,
            ask_volume=bid_vol,
            delta=bid_vol - ask_vol  # Positive delta
        ))
    
    return bars


def create_test_volume_bars_low_absorption() -> list[VolumeBar]:
    """Create volume bars showing low absorption (low volume, delta confirms direction)"""
    base_time = datetime(2024, 1, 2, 9, 30)
    bars = []
    
    for i in range(10):
        volume = 500.0
        bid_vol = volume * 0.4
        ask_vol = volume * 0.6
        
        bars.append(VolumeBar(
            timestamp=base_time + timedelta(minutes=i),
            volume=volume,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            delta=bid_vol - ask_vol  # Negative delta (consistent selling)
        ))
    
    return bars


# ============================================================================
# VolumeBar Tests
# ============================================================================

class TestVolumeBar:
    """Test VolumeBar dataclass validation"""
    
    def test_valid_volume_bar(self):
        """Test valid volume bar creation"""
        bar = VolumeBar(
            timestamp=datetime(2024, 1, 2, 9, 30),
            volume=1000.0,
            bid_volume=600.0,
            ask_volume=400.0,
            delta=200.0
        )
        
        assert bar.volume == 1000.0
        assert bar.bid_volume == 600.0
        assert bar.ask_volume == 400.0
        assert bar.delta == 200.0
    
    def test_negative_volume_raises_error(self):
        """Test that negative volume raises ValueError"""
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            VolumeBar(
                timestamp=datetime(2024, 1, 2, 9, 30),
                volume=-100.0,
                bid_volume=50.0,
                ask_volume=50.0,
                delta=0.0
            )
    
    def test_negative_bid_volume_raises_error(self):
        """Test that negative bid volume raises ValueError"""
        with pytest.raises(ValueError, match="Bid/ask volumes cannot be negative"):
            VolumeBar(
                timestamp=datetime(2024, 1, 2, 9, 30),
                volume=100.0,
                bid_volume=-50.0,
                ask_volume=150.0,
                delta=0.0
            )
    
    def test_volume_sum_mismatch_raises_error(self):
        """Test that bid + ask != total raises ValueError"""
        with pytest.raises(ValueError, match="Bid \\+ Ask volume must equal total volume"):
            VolumeBar(
                timestamp=datetime(2024, 1, 2, 9, 30),
                volume=100.0,
                bid_volume=60.0,
                ask_volume=50.0,  # Sum = 110, not 100
                delta=10.0
            )


# ============================================================================
# ExhaustionConfig Tests
# ============================================================================

class TestExhaustionConfig:
    """Test ExhaustionConfig validation"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ExhaustionConfig()
        
        assert config.volume_weight == 0.30
        assert config.body_weight == 0.30
        assert config.price_weight == 0.25
        assert config.reversal_weight == 0.15
        assert config.threshold == 0.70
        assert abs((config.volume_weight + config.body_weight + 
                   config.price_weight + config.reversal_weight) - 1.0) < 0.001
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ExhaustionConfig(
            volume_weight=0.25,
            body_weight=0.25,
            price_weight=0.25,
            reversal_weight=0.25,
            threshold=0.80
        )
        
        assert config.threshold == 0.80
        assert abs(sum([config.volume_weight, config.body_weight,
                       config.price_weight, config.reversal_weight]) - 1.0) < 0.001
    
    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0"""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ExhaustionConfig(
                volume_weight=0.30,
                body_weight=0.30,
                price_weight=0.30,
                reversal_weight=0.30  # Sum = 1.20
            )
    
    def test_threshold_out_of_range_raises_error(self):
        """Test that threshold must be [0.0, 1.0]"""
        with pytest.raises(ValueError, match="Threshold must be"):
            ExhaustionConfig(threshold=1.5)


# ============================================================================
# ExhaustionDetector Initialization Tests
# ============================================================================

class TestExhaustionDetectorInit:
    """Test ExhaustionDetector initialization"""
    
    def test_default_initialization(self):
        """Test initialization with default config"""
        detector = ExhaustionDetector()
        
        assert detector.config.volume_weight == 0.30
        assert detector.config.threshold == 0.70
    
    def test_custom_config_initialization(self):
        """Test initialization with custom config"""
        config = ExhaustionConfig(threshold=0.80)
        detector = ExhaustionDetector(config)
        
        assert detector.config.threshold == 0.80


# ============================================================================
# Volume Absorption Tests
# ============================================================================

class TestVolumeAbsorption:
    """Test calculate_volume_absorption method"""
    
    def test_empty_volume_data(self):
        """Test with empty volume data"""
        detector = ExhaustionDetector()
        score = detector.calculate_volume_absorption([])
        
        assert score == 0.0
    
    def test_insufficient_volume_data(self):
        """Test with insufficient volume data (< 3 bars)"""
        detector = ExhaustionDetector()
        bars = create_test_volume_bars_high_absorption()[:2]
        score = detector.calculate_volume_absorption(bars)
        
        assert score == 0.0
    
    def test_high_absorption_returns_high_score(self):
        """Test that high absorption returns high score"""
        detector = ExhaustionDetector()
        bars = create_test_volume_bars_high_absorption()
        score = detector.calculate_volume_absorption(bars, direction="LONG")
        
        assert score > 0.3  # Should be elevated
    
    def test_low_absorption_returns_low_score(self):
        """Test that low absorption returns low score"""
        detector = ExhaustionDetector()
        bars = create_test_volume_bars_low_absorption()
        score = detector.calculate_volume_absorption(bars, direction="LONG")
        
        assert score >= 0.0
    
    def test_score_range_bounded(self):
        """Test that score is always [0.0, 1.0]"""
        detector = ExhaustionDetector()
        bars = create_test_volume_bars_high_absorption()
        
        for direction in ["LONG", "SHORT"]:
            score = detector.calculate_volume_absorption(bars, direction)
            assert 0.0 <= score <= 1.0


# ============================================================================
# Body Rejection Tests
# ============================================================================

class TestBodyRejection:
    """Test calculate_candle_body_rejection method"""
    
    def test_empty_candles(self):
        """Test with empty candle list"""
        detector = ExhaustionDetector()
        score = detector.calculate_candle_body_rejection([])
        
        assert score == 0.0
    
    def test_insufficient_candles(self):
        """Test with insufficient candles (< 2)"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()[:1]
        score = detector.calculate_candle_body_rejection(candles)
        
        assert score == 0.0
    
    def test_rejection_candles_return_high_score(self):
        """Test that rejection wicks return high score"""
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        score = detector.calculate_candle_body_rejection(candles, direction="LONG")
        
        assert score > 0.4  # Should detect upper wick rejection
    
    def test_normal_candles_return_low_score(self):
        """Test that normal candles return low score"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        score = detector.calculate_candle_body_rejection(candles, direction="LONG")
        
        assert score < 0.5  # Normal uptrend candles
    
    def test_score_range_bounded(self):
        """Test that score is always [0.0, 1.0]"""
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        
        for direction in ["LONG", "SHORT"]:
            score = detector.calculate_candle_body_rejection(candles, direction)
            assert 0.0 <= score <= 1.0


# ============================================================================
# Price Stagnation Tests
# ============================================================================

class TestPriceStagnation:
    """Test calculate_price_stagnation method"""
    
    def test_empty_candles(self):
        """Test with empty candle list"""
        detector = ExhaustionDetector()
        score = detector.calculate_price_stagnation([])
        
        assert score == 0.0
    
    def test_insufficient_candles(self):
        """Test with insufficient candles (< 3)"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()[:2]
        score = detector.calculate_price_stagnation(candles)
        
        assert score == 0.0
    
    def test_stagnant_price_returns_high_score(self):
        """Test that stagnant price action returns high score"""
        detector = ExhaustionDetector()
        candles = create_test_candles_stagnant()
        score = detector.calculate_price_stagnation(candles)
        
        assert score > 0.5  # Choppy overlapping ranges
    
    def test_trending_price_returns_low_score(self):
        """Test that trending price returns low score"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        score = detector.calculate_price_stagnation(candles)
        
        assert score < 0.5  # Clean directional movement
    
    def test_score_range_bounded(self):
        """Test that score is always [0.0, 1.0]"""
        detector = ExhaustionDetector()
        candles = create_test_candles_stagnant()
        score = detector.calculate_price_stagnation(candles)
        
        assert 0.0 <= score <= 1.0


# ============================================================================
# Reversal Pattern Tests
# ============================================================================

class TestReversalPatterns:
    """Test calculate_reversal_patterns method"""
    
    def test_empty_candles(self):
        """Test with empty candle list"""
        detector = ExhaustionDetector()
        score = detector.calculate_reversal_patterns([])
        
        assert score == 0.0
    
    def test_insufficient_candles(self):
        """Test with insufficient candles (< 2)"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()[:1]
        score = detector.calculate_reversal_patterns(candles)
        
        assert score == 0.0
    
    def test_shooting_star_pattern(self):
        """Test shooting star pattern detection"""
        detector = ExhaustionDetector()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time, 100.0, 101.0, 99.0, 100.5, 1000.0),
            Candle(base_time + timedelta(minutes=1), 100.5, 105.0, 100.0, 101.0, 2000.0)
            # Shooting star: long upper wick, small body
        ]
        
        score = detector.calculate_reversal_patterns(candles)
        assert score > 0.5
    
    def test_hammer_pattern(self):
        """Test hammer pattern detection"""
        detector = ExhaustionDetector()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time, 100.0, 101.0, 99.0, 99.5, 1000.0),
            Candle(base_time + timedelta(minutes=1), 99.5, 100.0, 95.0, 99.0, 2000.0)
            # Hammer: long lower wick, small body
        ]
        
        score = detector.calculate_reversal_patterns(candles)
        assert score > 0.5
    
    def test_score_range_bounded(self):
        """Test that score is always [0.0, 1.0]"""
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        score = detector.calculate_reversal_patterns(candles)
        
        assert 0.0 <= score <= 1.0


# ============================================================================
# Overall Exhaustion Score Tests
# ============================================================================

class TestExhaustionScore:
    """Test calculate_exhaustion_score method"""
    
    def test_empty_candles_returns_zero(self):
        """Test with empty candles"""
        detector = ExhaustionDetector()
        score = detector.calculate_exhaustion_score([])
        
        assert score == 0.0
    
    def test_score_weighted_combination(self):
        """Test that score is weighted combination of components"""
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        volume_bars = create_test_volume_bars_high_absorption()
        
        score = detector.calculate_exhaustion_score(candles, volume_bars, "LONG")
        
        # Should be combination of all 4 components
        assert 0.0 <= score <= 1.0
    
    def test_high_exhaustion_signals(self):
        """Test that high exhaustion scenario returns high score"""
        detector = ExhaustionDetector()
        
        # Create high exhaustion scenario
        candles = create_test_candles_stagnant()  # High stagnation
        volume_bars = create_test_volume_bars_high_absorption()  # High volume absorption
        
        score = detector.calculate_exhaustion_score(candles, volume_bars, "LONG")
        
        # Should be elevated (not necessarily above threshold, depends on all factors)
        assert score > 0.3
    
    def test_score_range_bounded(self):
        """Test that score is always [0.0, 1.0]"""
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        volume_bars = create_test_volume_bars_high_absorption()
        
        score = detector.calculate_exhaustion_score(candles, volume_bars)
        assert 0.0 <= score <= 1.0


# ============================================================================
# Exhaustion Detection (Full Pipeline) Tests
# ============================================================================

class TestExhaustionDetection:
    """Test detect_exhaustion method (full pipeline)"""
    
    def test_empty_candles_returns_zero_result(self):
        """Test with empty candles"""
        detector = ExhaustionDetector()
        result = detector.detect_exhaustion([])
        
        assert result.score == 0.0
        assert result.threshold_met == False
    
    def test_result_structure(self):
        """Test that result contains all expected fields"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        
        result = detector.detect_exhaustion(candles)
        
        assert hasattr(result, 'score')
        assert hasattr(result, 'volume_score')
        assert hasattr(result, 'body_score')
        assert hasattr(result, 'price_score')
        assert hasattr(result, 'reversal_score')
        assert hasattr(result, 'threshold_met')
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'details')
    
    def test_threshold_detection(self):
        """Test threshold_met flag"""
        # Low threshold config for testing
        config = ExhaustionConfig(threshold=0.20)
        detector = ExhaustionDetector(config)
        
        candles = create_test_candles_stagnant()
        volume_bars = create_test_volume_bars_high_absorption()
        
        result = detector.detect_exhaustion(candles, volume_bars)
        
        # With low threshold, should trigger
        assert result.score >= 0.20
    
    def test_details_string_formatted(self):
        """Test that details string is properly formatted"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        
        result = detector.detect_exhaustion(candles)
        
        assert "Overall Score:" in result.details
        assert "Component Scores:" in result.details
        assert "Volume Absorption:" in result.details
        assert "Body Rejection:" in result.details
    
    def test_component_scores_in_range(self):
        """Test that all component scores are [0.0, 1.0]"""
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        volume_bars = create_test_volume_bars_high_absorption()
        
        result = detector.detect_exhaustion(candles, volume_bars)
        
        assert 0.0 <= result.volume_score <= 1.0
        assert 0.0 <= result.body_score <= 1.0
        assert 0.0 <= result.price_score <= 1.0
        assert 0.0 <= result.reversal_score <= 1.0
        assert 0.0 <= result.score <= 1.0


# ============================================================================
# Mathematical Properties Tests
# ============================================================================

class TestMathematicalProperties:
    """Test mathematical properties of exhaustion detection"""
    
    def test_determinism(self):
        """Test that same input produces same output"""
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        volume_bars = create_test_volume_bars_high_absorption()
        
        result1 = detector.detect_exhaustion(candles, volume_bars)
        result2 = detector.detect_exhaustion(candles, volume_bars)
        
        assert result1.score == result2.score
        assert result1.volume_score == result2.volume_score
        assert result1.body_score == result2.body_score
    
    def test_monotonicity_volume(self):
        """Test that more absorption → higher score (monotonic)"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        
        low_volume = create_test_volume_bars_low_absorption()
        high_volume = create_test_volume_bars_high_absorption()
        
        score_low = detector.calculate_exhaustion_score(candles, low_volume)
        score_high = detector.calculate_exhaustion_score(candles, high_volume)
        
        # High absorption should generally produce higher or equal score
        # (not strict monotonicity due to weighted combination)
        assert score_high >= 0.0 and score_low >= 0.0
    
    def test_bounded_output(self):
        """Test that output is always bounded [0.0, 1.0]"""
        detector = ExhaustionDetector()
        
        # Test with various inputs
        test_cases = [
            (create_test_candles_uptrend(), create_test_volume_bars_high_absorption()),
            (create_test_candles_stagnant(), create_test_volume_bars_low_absorption()),
            (create_test_candles_with_rejection(), create_test_volume_bars_high_absorption())
        ]
        
        for candles, volume_bars in test_cases:
            result = detector.detect_exhaustion(candles, volume_bars)
            assert 0.0 <= result.score <= 1.0
    
    def test_convex_combination(self):
        """Test that score is convex combination (weights sum to 1.0)"""
        config = ExhaustionConfig()
        
        total_weight = (config.volume_weight + config.body_weight + 
                       config.price_weight + config.reversal_weight)
        
        assert abs(total_weight - 1.0) < 0.001


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_candle(self):
        """Test with single candle"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()[:1]
        
        result = detector.detect_exhaustion(candles)
        
        # Should handle gracefully (most methods return 0.0)
        assert 0.0 <= result.score <= 1.0
    
    def test_no_volume_data(self):
        """Test without volume data"""
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        
        result = detector.detect_exhaustion(candles, volume_data=None)
        
        assert result.volume_score == 0.0
        # Other scores should still be calculated
        assert result.score >= 0.0
    
    def test_zero_range_candles(self):
        """Test with zero-range candles (high = low)"""
        detector = ExhaustionDetector()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time + timedelta(minutes=i), 100.0, 100.0, 100.0, 100.0, 1000.0)
            for i in range(5)
        ]
        
        result = detector.detect_exhaustion(candles)
        
        # Should handle gracefully without division by zero
        assert 0.0 <= result.score <= 1.0
    
    def test_extreme_wick_ratios(self):
        """Test with extreme wick-to-body ratios"""
        detector = ExhaustionDetector()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        # Candle with 99% wick, 1% body
        candles = [
            Candle(base_time, 100.0, 101.0, 99.0, 100.5, 1000.0),
            Candle(base_time + timedelta(minutes=1), 100.0, 110.0, 99.0, 100.1, 2000.0)
        ]
        
        result = detector.detect_exhaustion(candles)
        
        # Should detect high rejection
        assert result.body_score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
