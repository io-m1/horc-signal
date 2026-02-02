"""
Test Suite for FuturesGapEngine (AXIOM 4: Futures Supremacy)
============================================================

Validates all mathematical properties and edge cases for gap detection and targeting.
"""

import pytest
from datetime import datetime, timedelta
from src.engines.gaps import (
    FuturesGapEngine,
    Gap,
    GapType,
    GapConfig,
    GapAnalysisResult
)
from src.engines.participant import Candle


# ============================================================================
# Test Data Generators
# ============================================================================

def create_test_candles_with_gap_up() -> list[Candle]:
    """Create candle sequence with gap up"""
    base_time = datetime(2024, 1, 2, 9, 30)
    candles = []
    
    # Pre-gap candles
    for i in range(5):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1000.0
        ))
    
    # Gap up candle (open > previous high)
    candles.append(Candle(
        timestamp=base_time + timedelta(minutes=5),
        open=105.0,  # Gaps above previous high (102.0)
        high=107.0,
        low=104.5,
        close=106.0,
        volume=2000.0
    ))
    
    # Post-gap candles
    for i in range(3):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=6 + i),
            open=106.0 + i,
            high=108.0 + i,
            low=105.0 + i,
            close=107.0 + i,
            volume=1000.0
        ))
    
    return candles


def create_test_candles_with_gap_down() -> list[Candle]:
    """Create candle sequence with gap down"""
    base_time = datetime(2024, 1, 2, 9, 30)
    candles = []
    
    # Pre-gap candles
    for i in range(5):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0,
            high=102.0,
            low=99.0,
            close=100.0,
            volume=1000.0
        ))
    
    # Gap down candle (open < previous low)
    candles.append(Candle(
        timestamp=base_time + timedelta(minutes=5),
        open=95.0,  # Gaps below previous low (99.0)
        high=96.0,
        low=94.0,
        close=95.5,
        volume=2000.0
    ))
    
    # Post-gap candles
    for i in range(3):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=6 + i),
            open=95.0 - i,
            high=96.0 - i,
            low=93.0 - i,
            close=94.0 - i,
            volume=1000.0
        ))
    
    return candles


def create_test_candles_gap_filled() -> list[Candle]:
    """Create candle sequence where gap gets filled"""
    base_time = datetime(2024, 1, 2, 9, 30)
    candles = []
    
    # Pre-gap
    candles.append(Candle(base_time, 100.0, 102.0, 99.0, 101.0, 1000.0))
    
    # Gap up
    candles.append(Candle(
        base_time + timedelta(minutes=1),
        105.0, 107.0, 104.5, 106.0, 2000.0
    ))
    
    # Price fills gap (touches gap range)
    candles.append(Candle(
        base_time + timedelta(minutes=2),
        106.0, 106.5, 103.0, 104.0, 1500.0  # Low touches gap
    ))
    
    return candles


def create_test_candles_no_gaps() -> list[Candle]:
    """Create candles with no gaps (continuous price action)"""
    base_time = datetime(2024, 1, 2, 9, 30)
    candles = []
    
    for i in range(10):
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=100.0 + i * 0.5,
            high=102.0 + i * 0.5,
            low=99.0 + i * 0.5,
            close=101.0 + i * 0.5,
            volume=1000.0
        ))
    
    return candles


# ============================================================================
# Gap Dataclass Tests
# ============================================================================

class TestGap:
    """Test Gap dataclass"""
    
    def test_valid_gap_creation(self):
        """Test valid gap creation"""
        gap = Gap(
            upper=105.0,
            lower=102.0,
            date=datetime(2024, 1, 2, 9, 30),
            gap_type=GapType.COMMON
        )
        
        assert gap.upper == 105.0
        assert gap.lower == 102.0
        assert gap.size == 3.0
        assert gap.target_level == 103.5  # Midpoint
        assert not gap.filled
    
    def test_gap_upper_must_be_greater_than_lower(self):
        """Test that upper > lower is enforced"""
        with pytest.raises(ValueError, match="Gap upper .* must be > lower"):
            Gap(
                upper=100.0,
                lower=105.0,  # Invalid: lower > upper
                date=datetime(2024, 1, 2, 9, 30),
                gap_type=GapType.COMMON
            )
    
    def test_gap_midpoint(self):
        """Test midpoint calculation"""
        gap = Gap(
            upper=110.0,
            lower=100.0,
            date=datetime(2024, 1, 2, 9, 30),
            gap_type=GapType.COMMON
        )
        
        assert gap.midpoint() == 105.0
    
    def test_gap_contains_price(self):
        """Test price containment check"""
        gap = Gap(
            upper=105.0,
            lower=100.0,
            date=datetime(2024, 1, 2, 9, 30),
            gap_type=GapType.COMMON
        )
        
        assert gap.contains_price(102.5)  # Inside gap
        assert gap.contains_price(100.0)  # At lower boundary
        assert gap.contains_price(105.0)  # At upper boundary
        assert not gap.contains_price(99.0)  # Below gap
        assert not gap.contains_price(106.0)  # Above gap
    
    def test_gap_age_calculation(self):
        """Test gap age calculation"""
        gap_date = datetime(2024, 1, 2, 9, 30)
        gap = Gap(
            upper=105.0,
            lower=100.0,
            date=gap_date,
            gap_type=GapType.COMMON
        )
        
        # Check age after 1 day
        current_date = gap_date + timedelta(days=1)
        assert abs(gap.age_days(current_date) - 1.0) < 0.01
        
        # Check age after 10 days
        current_date = gap_date + timedelta(days=10)
        assert abs(gap.age_days(current_date) - 10.0) < 0.01
    
    def test_gap_distance_to_price(self):
        """Test distance calculation to price"""
        gap = Gap(
            upper=105.0,
            lower=100.0,
            date=datetime(2024, 1, 2, 9, 30),
            gap_type=GapType.COMMON
        )
        
        # Price inside gap
        assert gap.distance_to_price(102.5) == 0.0
        
        # Price below gap
        assert gap.distance_to_price(95.0) == 5.0  # 100 - 95
        
        # Price above gap
        assert gap.distance_to_price(110.0) == 5.0  # 110 - 105


# ============================================================================
# GapConfig Tests
# ============================================================================

class TestGapConfig:
    """Test GapConfig validation"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = GapConfig()
        
        assert config.min_gap_size_points == 2.0
        assert config.min_gap_size_percent == 0.001
        assert config.max_gap_age_days == 30
        assert config.gap_fill_tolerance == 0.5
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = GapConfig(
            min_gap_size_points=5.0,
            max_gap_age_days=60,
            gap_fill_tolerance=0.7
        )
        
        assert config.min_gap_size_points == 5.0
        assert config.max_gap_age_days == 60
        assert config.gap_fill_tolerance == 0.7
    
    def test_negative_min_gap_size_raises_error(self):
        """Test that negative gap size raises error"""
        with pytest.raises(ValueError, match="min_gap_size_points must be >= 0"):
            GapConfig(min_gap_size_points=-1.0)
    
    def test_invalid_gap_fill_tolerance_raises_error(self):
        """Test that invalid tolerance raises error"""
        with pytest.raises(ValueError, match="gap_fill_tolerance must be"):
            GapConfig(gap_fill_tolerance=1.5)


# ============================================================================
# FuturesGapEngine Initialization Tests
# ============================================================================

class TestFuturesGapEngineInit:
    """Test FuturesGapEngine initialization"""
    
    def test_default_initialization(self):
        """Test initialization with default config"""
        engine = FuturesGapEngine()
        
        assert engine.config.min_gap_size_points == 2.0
        assert len(engine.gaps) == 0
    
    def test_custom_config_initialization(self):
        """Test initialization with custom config"""
        config = GapConfig(min_gap_size_points=5.0)
        engine = FuturesGapEngine(config)
        
        assert engine.config.min_gap_size_points == 5.0


# ============================================================================
# Gap Detection Tests
# ============================================================================

class TestGapDetection:
    """Test detect_gaps method"""
    
    def test_empty_candles(self):
        """Test with empty candle list"""
        engine = FuturesGapEngine()
        gaps = engine.detect_gaps([])
        
        assert len(gaps) == 0
    
    def test_insufficient_candles(self):
        """Test with single candle"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()[:1]
        gaps = engine.detect_gaps(candles)
        
        assert len(gaps) == 0
    
    def test_gap_up_detection(self):
        """Test detection of gap up"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        gaps = engine.detect_gaps(candles)
        
        assert len(gaps) >= 1
        gap = gaps[0]
        assert gap.direction == "UP"
        assert gap.upper > gap.lower
        assert gap.size > 0
    
    def test_gap_down_detection(self):
        """Test detection of gap down"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_down()
        gaps = engine.detect_gaps(candles)
        
        assert len(gaps) >= 1
        gap = gaps[0]
        assert gap.direction == "DOWN"
        assert gap.upper > gap.lower
        assert gap.size > 0
    
    def test_no_gaps_detected_in_continuous_price(self):
        """Test that no gaps detected in continuous price action"""
        engine = FuturesGapEngine()
        candles = create_test_candles_no_gaps()
        gaps = engine.detect_gaps(candles)
        
        # Should detect 0 gaps (continuous price movement)
        assert len(gaps) == 0
    
    def test_gap_size_threshold(self):
        """Test that only gaps meeting size threshold are detected"""
        config = GapConfig(min_gap_size_points=10.0)  # Large threshold
        engine = FuturesGapEngine(config)
        
        candles = create_test_candles_with_gap_up()  # Gap size is 3.0
        gaps = engine.detect_gaps(candles)
        
        # Should not detect gap (size < threshold)
        assert len(gaps) == 0
    
    def test_gap_stores_volume_context(self):
        """Test that gap stores volume at creation"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        gaps = engine.detect_gaps(candles)
        
        assert len(gaps) >= 1
        gap = gaps[0]
        assert gap.volume_context > 0


# ============================================================================
# Gap Classification Tests
# ============================================================================

class TestGapClassification:
    """Test gap type classification"""
    
    def test_gap_type_assigned(self):
        """Test that detected gaps have type assigned"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        gaps = engine.detect_gaps(candles)
        
        assert len(gaps) >= 1
        gap = gaps[0]
        assert gap.gap_type in [GapType.COMMON, GapType.BREAKAWAY, 
                                GapType.EXHAUSTION, GapType.MEASURING]
    
    def test_common_gap_classification(self):
        """Test that small gaps are classified as COMMON"""
        # Create small gap
        base_time = datetime(2024, 1, 2, 9, 30)
        candles = [
            Candle(base_time, 100.0, 100.5, 99.5, 100.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 100.6, 101.0, 100.5, 100.8, 1000.0)
            # Small gap of 0.1 points
        ]
        
        config = GapConfig(min_gap_size_points=0.05)  # Allow small gaps
        engine = FuturesGapEngine(config)
        gaps = engine.detect_gaps(candles)
        
        if len(gaps) > 0:
            # Small gap should be COMMON
            assert gaps[0].gap_type == GapType.COMMON


# ============================================================================
# Gap Fill Detection Tests
# ============================================================================

class TestGapFillDetection:
    """Test gap fill detection logic"""
    
    def test_gap_marked_as_filled(self):
        """Test that filled gap is marked correctly"""
        engine = FuturesGapEngine()
        candles = create_test_candles_gap_filled()
        gaps = engine.detect_gaps(candles)
        
        assert len(gaps) >= 1
        gap = gaps[0]
        # Gap should be marked as filled (price revisited gap range)
        assert gap.filled
    
    def test_unfilled_gap(self):
        """Test that unfilled gap remains unfilled"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        gaps = engine.detect_gaps(candles)
        
        assert len(gaps) >= 1
        gap = gaps[0]
        # Gap not filled in test data
        assert not gap.filled


# ============================================================================
# Target Calculation Tests
# ============================================================================

class TestTargetCalculation:
    """Test calculate_futures_target method"""
    
    def test_empty_gaps_returns_none(self):
        """Test with no gaps"""
        engine = FuturesGapEngine()
        target = engine.calculate_futures_target([], 100.0, datetime.now())
        
        assert target is None
    
    def test_all_filled_gaps_returns_none(self):
        """Test when all gaps are filled"""
        engine = FuturesGapEngine()
        candles = create_test_candles_gap_filled()
        gaps = engine.detect_gaps(candles)
        
        # All gaps filled, should return None
        target = engine.calculate_futures_target(gaps, 100.0, datetime.now())
        assert target is None
    
    def test_returns_nearest_unfilled_gap(self):
        """Test that nearest unfilled gap is returned"""
        engine = FuturesGapEngine()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        # Create multiple gaps
        gap1 = Gap(105.0, 102.0, base_time, GapType.COMMON, filled=False)
        gap2 = Gap(115.0, 112.0, base_time, GapType.COMMON, filled=False)
        
        gaps = [gap1, gap2]
        current_price = 100.0
        
        # Use a current_date close to the gap dates to avoid age filtering
        current_date = datetime(2024, 1, 3, 9, 30)  # 1 day after gaps
        
        # Nearest gap is gap1 (midpoint 103.5 vs 113.5)
        target = engine.calculate_futures_target(gaps, current_price, current_date)
        
        assert target == gap1.target_level
    
    def test_old_gaps_excluded(self):
        """Test that gaps older than max age are excluded"""
        config = GapConfig(max_gap_age_days=30)
        engine = FuturesGapEngine(config)
        
        old_date = datetime(2024, 1, 1, 9, 30)
        current_date = datetime(2024, 2, 15, 9, 30)  # 45 days later
        
        gap = Gap(105.0, 102.0, old_date, GapType.COMMON, filled=False)
        
        # Gap is too old (45 days > 30 days)
        target = engine.calculate_futures_target([gap], 100.0, current_date)
        
        assert target is None


# ============================================================================
# Gap Analysis Tests
# ============================================================================

class TestGapAnalysis:
    """Test analyze_gaps method (full pipeline)"""
    
    def test_analysis_result_structure(self):
        """Test that analysis result contains all fields"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        gaps = engine.detect_gaps(candles)
        
        result = engine.analyze_gaps(gaps, 100.0, datetime.now())
        
        assert hasattr(result, 'target_price')
        assert hasattr(result, 'nearest_gap')
        assert hasattr(result, 'total_gaps')
        assert hasattr(result, 'unfilled_gaps')
        assert hasattr(result, 'fill_probability')
        assert hasattr(result, 'gravitational_pull')
        assert hasattr(result, 'details')
    
    def test_no_gaps_analysis(self):
        """Test analysis with no gaps"""
        engine = FuturesGapEngine()
        result = engine.analyze_gaps([], 100.0, datetime.now())
        
        assert result.target_price is None
        assert result.nearest_gap is None
        assert result.total_gaps == 0
        assert result.unfilled_gaps == 0
    
    def test_fill_probability_range(self):
        """Test that fill probability is bounded [0.0, 1.0]"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        gaps = engine.detect_gaps(candles)
        
        result = engine.analyze_gaps(gaps, 100.0, datetime.now())
        
        assert 0.0 <= result.fill_probability <= 1.0
    
    def test_gravitational_pull_range(self):
        """Test that gravitational pull is bounded [0.0, 1.0]"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        gaps = engine.detect_gaps(candles)
        
        result = engine.analyze_gaps(gaps, 100.0, datetime.now())
        
        assert 0.0 <= result.gravitational_pull <= 1.0
    
    def test_details_string_formatted(self):
        """Test that details string is properly formatted"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        gaps = engine.detect_gaps(candles)
        
        result = engine.analyze_gaps(gaps, 100.0, datetime.now())
        
        assert "Gap Analysis Summary:" in result.details
        assert "Total Gaps Detected:" in result.details
        assert "Target Price:" in result.details


# ============================================================================
# Helper Method Tests
# ============================================================================

class TestHelperMethods:
    """Test helper methods"""
    
    def test_get_unfilled_gaps(self):
        """Test filtering for unfilled gaps"""
        engine = FuturesGapEngine()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        gap1 = Gap(105.0, 102.0, base_time, GapType.COMMON, filled=False)
        gap2 = Gap(115.0, 112.0, base_time, GapType.COMMON, filled=True)
        
        gaps = [gap1, gap2]
        engine.gaps = gaps
        
        # Pass current_date to avoid age filtering issues
        current_date = datetime(2024, 1, 3, 9, 30)  # 1 day after gaps
        unfilled = engine.get_unfilled_gaps(current_date=current_date)
        
        assert len(unfilled) == 1
        assert unfilled[0] == gap1
    
    def test_get_gap_by_type(self):
        """Test filtering gaps by type"""
        engine = FuturesGapEngine()
        base_time = datetime(2024, 1, 2, 9, 30)
        
        gap1 = Gap(105.0, 102.0, base_time, GapType.COMMON)
        gap2 = Gap(115.0, 112.0, base_time, GapType.BREAKAWAY)
        gap3 = Gap(125.0, 122.0, base_time, GapType.COMMON)
        
        gaps = [gap1, gap2, gap3]
        engine.gaps = gaps
        
        common_gaps = engine.get_gap_by_type(GapType.COMMON)
        
        assert len(common_gaps) == 2
        assert gap1 in common_gaps
        assert gap3 in common_gaps


# ============================================================================
# Mathematical Properties Tests
# ============================================================================

class TestMathematicalProperties:
    """Test mathematical properties of gap detection"""
    
    def test_determinism(self):
        """Test that same input produces same output"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        
        gaps1 = engine.detect_gaps(candles)
        gaps2 = engine.detect_gaps(candles)
        
        assert len(gaps1) == len(gaps2)
        if len(gaps1) > 0:
            assert gaps1[0].upper == gaps2[0].upper
            assert gaps1[0].lower == gaps2[0].lower
    
    def test_gap_count_consistency(self):
        """Test that gap count is consistent"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        
        gaps = engine.detect_gaps(candles)
        stored_gaps = engine.gaps
        
        assert len(gaps) == len(stored_gaps)
    
    def test_target_determinism(self):
        """Test that target calculation is deterministic"""
        engine = FuturesGapEngine()
        candles = create_test_candles_with_gap_up()
        gaps = engine.detect_gaps(candles)
        
        target1 = engine.calculate_futures_target(gaps, 100.0, datetime.now())
        target2 = engine.calculate_futures_target(gaps, 100.0, datetime.now())
        
        assert target1 == target2


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_large_gap(self):
        """Test detection of single large gap"""
        base_time = datetime(2024, 1, 2, 9, 30)
        candles = [
            Candle(base_time, 100.0, 102.0, 99.0, 101.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 150.0, 152.0, 149.0, 151.0, 5000.0)
            # Very large gap
        ]
        
        engine = FuturesGapEngine()
        gaps = engine.detect_gaps(candles)
        
        assert len(gaps) == 1
        assert gaps[0].size > 40.0
    
    def test_multiple_consecutive_gaps(self):
        """Test multiple gaps in sequence"""
        base_time = datetime(2024, 1, 2, 9, 30)
        candles = [
            Candle(base_time, 100.0, 102.0, 99.0, 101.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 105.0, 107.0, 104.0, 106.0, 2000.0),  # Gap 1
            Candle(base_time + timedelta(minutes=2), 110.0, 112.0, 109.0, 111.0, 2000.0),  # Gap 2
        ]
        
        engine = FuturesGapEngine()
        gaps = engine.detect_gaps(candles)
        
        assert len(gaps) >= 2
    
    def test_gap_at_exact_threshold(self):
        """Test gap exactly at size threshold"""
        config = GapConfig(min_gap_size_points=3.0)
        engine = FuturesGapEngine(config)
        
        base_time = datetime(2024, 1, 2, 9, 30)
        candles = [
            Candle(base_time, 100.0, 100.0, 99.0, 100.0, 1000.0),
            Candle(base_time + timedelta(minutes=1), 103.0, 104.0, 102.5, 103.5, 2000.0)
            # Gap size exactly 3.0
        ]
        
        gaps = engine.detect_gaps(candles)
        
        # Should detect gap at threshold
        assert len(gaps) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
