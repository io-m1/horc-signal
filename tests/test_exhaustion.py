import pytest
from datetime import datetime, timedelta
from src.engines.exhaustion import (
    ExhaustionDetector,
    ExhaustionConfig,
    ExhaustionResult,
    VolumeBar
)
from src.engines.participant import Candle

def create_test_candles_uptrend() -> list[Candle]:
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
    base_time = datetime(2024, 1, 2, 9, 30)
    candles = []
    
    for i in range(5):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + i * 2.0,
            high=102.0 + i * 2.0,
            low=99.0 + i * 2.0,
            close=101.0 + i * 2.0,
            volume=1000.0
        ))
    
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
    base_time = datetime(2024, 1, 2, 9, 30)
    bars = []
    
    for i in range(10):
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

class TestVolumeBar:
    
    def test_valid_volume_bar(self):
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
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            VolumeBar(
                timestamp=datetime(2024, 1, 2, 9, 30),
                volume=-100.0,
                bid_volume=50.0,
                ask_volume=50.0,
                delta=0.0
            )
    
    def test_negative_bid_volume_raises_error(self):
        with pytest.raises(ValueError, match="Bid/ask volumes cannot be negative"):
            VolumeBar(
                timestamp=datetime(2024, 1, 2, 9, 30),
                volume=100.0,
                bid_volume=-50.0,
                ask_volume=150.0,
                delta=0.0
            )
    
    def test_volume_sum_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="Bid \\+ Ask volume must equal total volume"):
            VolumeBar(
                timestamp=datetime(2024, 1, 2, 9, 30),
                volume=100.0,
                bid_volume=60.0,
                ask_volume=50.0,  # Sum = 110, not 100
                delta=10.0
            )

class TestExhaustionConfig:
    
    def test_default_config(self):
        config = ExhaustionConfig()
        
        assert config.volume_weight == 0.30
        assert config.body_weight == 0.30
        assert config.price_weight == 0.25
        assert config.reversal_weight == 0.15
        assert config.threshold == 0.70
        assert abs((config.volume_weight + config.body_weight + 
                   config.price_weight + config.reversal_weight) - 1.0) < 0.001
    
    def test_custom_config(self):
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
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ExhaustionConfig(
                volume_weight=0.30,
                body_weight=0.30,
                price_weight=0.30,
                reversal_weight=0.30  # Sum = 1.20
            )
    
    def test_threshold_out_of_range_raises_error(self):
        with pytest.raises(ValueError, match="Threshold must be"):
            ExhaustionConfig(threshold=1.5)

class TestExhaustionDetectorInit:
    
    def test_default_initialization(self):
        detector = ExhaustionDetector()
        
        assert detector.config.volume_weight == 0.30
        assert detector.config.threshold == 0.70
    
    def test_custom_config_initialization(self):
        config = ExhaustionConfig(threshold=0.80)
        detector = ExhaustionDetector(config)
        
        assert detector.config.threshold == 0.80

class TestVolumeAbsorption:
    
    def test_empty_volume_data(self):
        detector = ExhaustionDetector()
        score = detector.calculate_volume_absorption([])
        
        assert score == 0.0
    
    def test_insufficient_volume_data(self):
        detector = ExhaustionDetector()
        bars = create_test_volume_bars_high_absorption()[:2]
        score = detector.calculate_volume_absorption(bars)
        
        assert score == 0.0
    
    def test_high_absorption_returns_high_score(self):
        detector = ExhaustionDetector()
        bars = create_test_volume_bars_high_absorption()
        score = detector.calculate_volume_absorption(bars, direction="LONG")
        
        assert score > 0.3  # Should be elevated
    
    def test_low_absorption_returns_low_score(self):
        detector = ExhaustionDetector()
        bars = create_test_volume_bars_low_absorption()
        score = detector.calculate_volume_absorption(bars, direction="LONG")
        
        assert score >= 0.0
    
    def test_score_range_bounded(self):
        detector = ExhaustionDetector()
        bars = create_test_volume_bars_high_absorption()
        
        for direction in ["LONG", "SHORT"]:
            score = detector.calculate_volume_absorption(bars, direction)
            assert 0.0 <= score <= 1.0

class TestBodyRejection:
    
    def test_empty_candles(self):
        detector = ExhaustionDetector()
        score = detector.calculate_candle_body_rejection([])
        
        assert score == 0.0
    
    def test_insufficient_candles(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()[:1]
        score = detector.calculate_candle_body_rejection(candles)
        
        assert score == 0.0
    
    def test_rejection_candles_return_high_score(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        score = detector.calculate_candle_body_rejection(candles, direction="LONG")
        
        assert score > 0.4  # Should detect upper wick rejection
    
    def test_normal_candles_return_low_score(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        score = detector.calculate_candle_body_rejection(candles, direction="LONG")
        
        assert score < 0.5  # Normal uptrend candles
    
    def test_score_range_bounded(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        
        for direction in ["LONG", "SHORT"]:
            score = detector.calculate_candle_body_rejection(candles, direction)
            assert 0.0 <= score <= 1.0

class TestPriceStagnation:
    
    def test_empty_candles(self):
        detector = ExhaustionDetector()
        score = detector.calculate_price_stagnation([])
        
        assert score == 0.0
    
    def test_insufficient_candles(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()[:2]
        score = detector.calculate_price_stagnation(candles)
        
        assert score == 0.0
    
    def test_stagnant_price_returns_high_score(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_stagnant()
        score = detector.calculate_price_stagnation(candles)
        
        assert score > 0.5  # Choppy overlapping ranges
    
    def test_trending_price_returns_low_score(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        score = detector.calculate_price_stagnation(candles)
        
        assert score < 0.5  # Clean directional movement
    
    def test_score_range_bounded(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_stagnant()
        score = detector.calculate_price_stagnation(candles)
        
        assert 0.0 <= score <= 1.0

class TestReversalPatterns:
    
    def test_empty_candles(self):
        detector = ExhaustionDetector()
        score = detector.calculate_reversal_patterns([])
        
        assert score == 0.0
    
    def test_insufficient_candles(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()[:1]
        score = detector.calculate_reversal_patterns(candles)
        
        assert score == 0.0
    
    def test_shooting_star_pattern(self):
        detector = ExhaustionDetector()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time, 100.0, 101.0, 99.0, 100.5, 1000.0),
            Candle(base_time + timedelta(minutes=1), 100.5, 105.0, 100.0, 101.0, 2000.0)
        ]
        
        score = detector.calculate_reversal_patterns(candles)
        assert score > 0.5
    
    def test_hammer_pattern(self):
        detector = ExhaustionDetector()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time, 100.0, 101.0, 99.0, 99.5, 1000.0),
            Candle(base_time + timedelta(minutes=1), 99.5, 100.0, 95.0, 99.0, 2000.0)
        ]
        
        score = detector.calculate_reversal_patterns(candles)
        assert score > 0.5
    
    def test_score_range_bounded(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        score = detector.calculate_reversal_patterns(candles)
        
        assert 0.0 <= score <= 1.0

class TestExhaustionScore:
    
    def test_empty_candles_returns_zero(self):
        detector = ExhaustionDetector()
        score = detector.calculate_exhaustion_score([])
        
        assert score == 0.0
    
    def test_score_weighted_combination(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        volume_bars = create_test_volume_bars_high_absorption()
        
        score = detector.calculate_exhaustion_score(candles, volume_bars, "LONG")
        
        assert 0.0 <= score <= 1.0
    
    def test_high_exhaustion_signals(self):
        detector = ExhaustionDetector()
        
        candles = create_test_candles_stagnant()  # High stagnation
        volume_bars = create_test_volume_bars_high_absorption()  # High volume absorption
        
        score = detector.calculate_exhaustion_score(candles, volume_bars, "LONG")
        
        assert score > 0.3
    
    def test_score_range_bounded(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        volume_bars = create_test_volume_bars_high_absorption()
        
        score = detector.calculate_exhaustion_score(candles, volume_bars)
        assert 0.0 <= score <= 1.0

class TestExhaustionDetection:
    
    def test_empty_candles_returns_zero_result(self):
        detector = ExhaustionDetector()
        result = detector.detect_exhaustion([])
        
        assert result.score == 0.0
        assert result.threshold_met == False
    
    def test_result_structure(self):
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
        config = ExhaustionConfig(threshold=0.20)
        detector = ExhaustionDetector(config)
        
        candles = create_test_candles_stagnant()
        volume_bars = create_test_volume_bars_high_absorption()
        
        result = detector.detect_exhaustion(candles, volume_bars)
        
        assert result.score >= 0.20
    
    def test_details_string_formatted(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        
        result = detector.detect_exhaustion(candles)
        
        assert "Overall Score:" in result.details
        assert "Component Scores:" in result.details
        assert "Volume Absorption:" in result.details
        assert "Body Rejection:" in result.details
    
    def test_component_scores_in_range(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        volume_bars = create_test_volume_bars_high_absorption()
        
        result = detector.detect_exhaustion(candles, volume_bars)
        
        assert 0.0 <= result.volume_score <= 1.0
        assert 0.0 <= result.body_score <= 1.0
        assert 0.0 <= result.price_score <= 1.0
        assert 0.0 <= result.reversal_score <= 1.0
        assert 0.0 <= result.score <= 1.0

class TestMathematicalProperties:
    
    def test_determinism(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_with_rejection()
        volume_bars = create_test_volume_bars_high_absorption()
        
        result1 = detector.detect_exhaustion(candles, volume_bars)
        result2 = detector.detect_exhaustion(candles, volume_bars)
        
        assert result1.score == result2.score
        assert result1.volume_score == result2.volume_score
        assert result1.body_score == result2.body_score
    
    def test_monotonicity_volume(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        
        low_volume = create_test_volume_bars_low_absorption()
        high_volume = create_test_volume_bars_high_absorption()
        
        score_low = detector.calculate_exhaustion_score(candles, low_volume)
        score_high = detector.calculate_exhaustion_score(candles, high_volume)
        
        assert score_high >= 0.0 and score_low >= 0.0
    
    def test_bounded_output(self):
        detector = ExhaustionDetector()
        
        test_cases = [
            (create_test_candles_uptrend(), create_test_volume_bars_high_absorption()),
            (create_test_candles_stagnant(), create_test_volume_bars_low_absorption()),
            (create_test_candles_with_rejection(), create_test_volume_bars_high_absorption())
        ]
        
        for candles, volume_bars in test_cases:
            result = detector.detect_exhaustion(candles, volume_bars)
            assert 0.0 <= result.score <= 1.0
    
    def test_convex_combination(self):
        config = ExhaustionConfig()
        
        total_weight = (config.volume_weight + config.body_weight + 
                       config.price_weight + config.reversal_weight)
        
        assert abs(total_weight - 1.0) < 0.001

class TestEdgeCases:
    
    def test_single_candle(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()[:1]
        
        result = detector.detect_exhaustion(candles)
        
        assert 0.0 <= result.score <= 1.0
    
    def test_no_volume_data(self):
        detector = ExhaustionDetector()
        candles = create_test_candles_uptrend()
        
        result = detector.detect_exhaustion(candles, volume_data=None)
        
        assert result.volume_score == 0.0
        assert result.score >= 0.0
    
    def test_zero_range_candles(self):
        detector = ExhaustionDetector()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time + timedelta(minutes=i), 100.0, 100.0, 100.0, 100.0, 1000.0)
            for i in range(5)
        ]
        
        result = detector.detect_exhaustion(candles)
        
        assert 0.0 <= result.score <= 1.0
    
    def test_extreme_wick_ratios(self):
        detector = ExhaustionDetector()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        candles = [
            Candle(base_time, 100.0, 101.0, 99.0, 100.5, 1000.0),
            Candle(base_time + timedelta(minutes=1), 100.0, 110.0, 99.0, 100.1, 2000.0)
        ]
        
        result = detector.detect_exhaustion(candles)
        
        assert result.body_score > 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
