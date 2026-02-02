"""
AXIOM 4: Futures Supremacy Implementation
==========================================

Theoretical Foundation:
    Futures gaps act as "structural magnets" or gravitational anchors for price.
    Based on market microstructure theory: gaps represent information asymmetry
    and unfilled limit orders that create persistent imbalances in the order book.

Mathematical Model:
    Target Price: T = nearest(G_unfilled)
    
    where:
        G_unfilled = set of all unfilled gaps
        nearest(·) = gap with minimum distance to current price
        
    Gap Fill Probability: P(fill) ∝ 1 / (age_days × distance)
    
    Gap Gravitational Pull:
        F = k / d²
        
        where:
            k = gap strength constant (based on gap type)
            d = distance from current price to gap midpoint

Gap Classification:
    1. Common Gap: Low volume, often fills quickly
    2. Breakaway Gap: High volume, start of new trend
    3. Exhaustion Gap: Very high volume, end of trend
    4. Measuring Gap: Mid-trend, continuation signal

References:
    - Kyle, A. S. (1985). "Continuous Auctions and Insider Trading"
    - Market microstructure theory on price discovery
    - Technical analysis: Gap theory (Edwards & Magee)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum
import math


# Import Candle from participant module
from .participant import Candle


class GapType(Enum):
    """
    Classification of gap types based on context and volume.
    
    Types:
        COMMON: Low-significance gap, often fills quickly
        BREAKAWAY: High-significance, start of new trend
        EXHAUSTION: End-of-trend gap, reversal signal
        MEASURING: Mid-trend continuation gap
    """
    COMMON = "common"
    BREAKAWAY = "breakaway"
    EXHAUSTION = "exhaustion"
    MEASURING = "measuring"


@dataclass
class Gap:
    """
    Futures gap data structure.
    
    A gap occurs when the opening price of one period is significantly
    different from the closing price of the previous period, leaving
    an unfilled price range on the chart.
    
    Attributes:
        upper: Upper boundary of gap (higher price)
        lower: Lower boundary of gap (lower price)
        date: Timestamp when gap was created
        gap_type: Classification of gap (common, breakaway, exhaustion, measuring)
        filled: Whether gap has been filled (price revisited the gap range)
        target_level: Target price for gap fill (typically midpoint)
        size: Size of gap in points
        volume_context: Volume at gap creation (for classification)
        direction: "UP" for gap up, "DOWN" for gap down
    """
    upper: float
    lower: float
    date: datetime
    gap_type: GapType
    filled: bool = False
    target_level: Optional[float] = None
    size: float = field(init=False)
    volume_context: float = 0.0
    direction: str = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields and validate"""
        if self.upper <= self.lower:
            raise ValueError(f"Gap upper ({self.upper}) must be > lower ({self.lower})")
        
        self.size = self.upper - self.lower
        
        # Set target level as midpoint if not specified
        if self.target_level is None:
            self.target_level = (self.upper + self.lower) / 2.0
        
        # Determine direction (simplified - assumes gap up if not specified)
        # In practice, this would be set during gap detection
        self.direction = "UP"  # Will be set properly in detect_gaps()
    
    def midpoint(self) -> float:
        """Calculate gap midpoint"""
        return (self.upper + self.lower) / 2.0
    
    def contains_price(self, price: float) -> bool:
        """Check if price is within gap range"""
        return self.lower <= price <= self.upper
    
    def age_days(self, current_date: datetime) -> float:
        """Calculate age of gap in days"""
        return (current_date - self.date).total_seconds() / 86400.0
    
    def distance_to_price(self, price: float) -> float:
        """
        Calculate distance from price to gap.
        
        Returns:
            0.0 if price is within gap
            Positive distance if price is outside gap
        """
        if self.contains_price(price):
            return 0.0
        elif price < self.lower:
            return self.lower - price
        else:
            return price - self.upper


@dataclass
class GapConfig:
    """
    Configuration for gap detection and analysis.
    
    Attributes:
        min_gap_size_points: Minimum gap size to detect (default: 2.0)
        min_gap_size_percent: Minimum gap as % of price (default: 0.001 = 0.1%)
        max_gap_age_days: Maximum age for gap consideration (default: 30)
        gap_fill_tolerance: % tolerance for gap fill (default: 0.5 = 50%)
        volume_multiplier_breakaway: Volume threshold for breakaway gaps (default: 1.5x)
        volume_multiplier_exhaustion: Volume threshold for exhaustion gaps (default: 2.0x)
        common_gap_max_size_percent: Max size for common gaps (default: 0.02 = 2%)
    """
    min_gap_size_points: float = 2.0
    min_gap_size_percent: float = 0.001  # 0.1%
    max_gap_age_days: int = 30
    gap_fill_tolerance: float = 0.5  # 50% of gap must be filled
    volume_multiplier_breakaway: float = 1.5
    volume_multiplier_exhaustion: float = 2.0
    common_gap_max_size_percent: float = 0.02  # 2%
    
    def __post_init__(self):
        """Validate configuration"""
        if self.min_gap_size_points < 0:
            raise ValueError("min_gap_size_points must be >= 0")
        if not (0 <= self.min_gap_size_percent <= 1):
            raise ValueError("min_gap_size_percent must be [0, 1]")
        if not (0 <= self.gap_fill_tolerance <= 1):
            raise ValueError("gap_fill_tolerance must be [0, 1]")


@dataclass
class GapAnalysisResult:
    """
    Result of gap analysis for target calculation.
    
    Attributes:
        target_price: Calculated target price (nearest unfilled gap)
        nearest_gap: The nearest unfilled gap
        total_gaps: Total number of gaps detected
        unfilled_gaps: Number of unfilled gaps
        fill_probability: Estimated probability of gap fill [0.0, 1.0]
        gravitational_pull: Strength of gravitational pull [0.0, 1.0]
        details: Human-readable analysis breakdown
    """
    target_price: Optional[float]
    nearest_gap: Optional[Gap]
    total_gaps: int
    unfilled_gaps: int
    fill_probability: float
    gravitational_pull: float
    details: str


class FuturesGapEngine:
    """
    AXIOM 4: Futures Supremacy - Gap Detection and Target Calculation
    
    Detects gaps in futures price data and calculates target prices based
    on the gravitational anchor principle: unfilled gaps act as magnets
    attracting price back to fill the gap.
    
    Mathematical Properties:
        - Gap detection: deterministic (same data → same gaps)
        - Target calculation: deterministic (nearest unfilled gap)
        - Fill detection: threshold-based (50% overlap default)
        - Gravitational pull: inverse square law (1/d²)
    
    Usage:
        engine = FuturesGapEngine()
        gaps = engine.detect_gaps(futures_data)
        result = engine.analyze_gaps(gaps, current_price)
        
        if result.target_price:
            print(f"Target: ${result.target_price:.2f}")
            print(f"Probability: {result.fill_probability:.1%}")
    """
    
    def __init__(self, config: Optional[GapConfig] = None):
        """
        Initialize futures gap engine.
        
        Args:
            config: Optional configuration. Uses defaults if None.
        """
        self.config = config if config is not None else GapConfig()
        self.gaps: List[Gap] = []
    
    def detect_gaps(self, candles: List[Candle]) -> List[Gap]:
        """
        Detect all futures gaps in candle data.
        
        A gap occurs when:
            curr_open > prev_high (gap up) or
            curr_open < prev_low (gap down)
        
        Methodology:
            1. Iterate through candles comparing open to previous high/low
            2. Classify gap type based on volume and context
            3. Set initial target level (midpoint)
            4. Track gaps for fill detection
        
        Args:
            candles: List of futures candles (OHLCV data)
            
        Returns:
            List of detected Gap objects
            
        Example:
            gaps = engine.detect_gaps(futures_candles)
            print(f"Found {len(gaps)} gaps")
            
            for gap in gaps:
                if not gap.filled:
                    print(f"Unfilled gap at ${gap.target_level:.2f}")
        """
        if not candles or len(candles) < 2:
            return []
        
        detected_gaps: List[Gap] = []
        
        # Calculate average volume for classification
        volumes = [c.volume for c in candles if c.volume > 0]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        for i in range(1, len(candles)):
            prev_candle = candles[i - 1]
            curr_candle = candles[i]
            
            gap_upper = None
            gap_lower = None
            gap_direction = None
            
            # Detect gap up (current open > previous high)
            if curr_candle.open > prev_candle.high:
                gap_lower = prev_candle.high
                gap_upper = curr_candle.open
                gap_direction = "UP"
                
            # Detect gap down (current open < previous low)
            elif curr_candle.open < prev_candle.low:
                gap_upper = prev_candle.low
                gap_lower = curr_candle.open
                gap_direction = "DOWN"
            
            # If gap detected, validate and classify
            if gap_upper is not None and gap_lower is not None:
                gap_size = gap_upper - gap_lower
                gap_size_percent = gap_size / prev_candle.close if prev_candle.close > 0 else 0
                
                # Check minimum gap size threshold
                # Use AND logic: gap must meet BOTH absolute and percentage thresholds
                # This prevents tiny percentage gaps on large prices or large absolute gaps on small prices
                if (gap_size >= self.config.min_gap_size_points and
                    gap_size_percent >= self.config.min_gap_size_percent):
                    
                    # Classify gap type
                    gap_type = self._classify_gap_type(
                        gap_size=gap_size,
                        gap_size_percent=gap_size_percent,
                        volume=curr_candle.volume,
                        avg_volume=avg_volume,
                        candles=candles,
                        gap_index=i,
                        direction=gap_direction
                    )
                    
                    # Create gap object
                    gap = Gap(
                        upper=gap_upper,
                        lower=gap_lower,
                        date=curr_candle.timestamp,
                        gap_type=gap_type,
                        filled=False,
                        target_level=(gap_upper + gap_lower) / 2.0,
                        volume_context=curr_candle.volume
                    )
                    gap.direction = gap_direction
                    
                    detected_gaps.append(gap)
        
        # Update gap fill status for all detected gaps
        self._update_gap_fills(detected_gaps, candles)
        
        # Store gaps for future reference
        self.gaps = detected_gaps
        
        return detected_gaps
    
    def _classify_gap_type(self,
                          gap_size: float,
                          gap_size_percent: float,
                          volume: float,
                          avg_volume: float,
                          candles: List[Candle],
                          gap_index: int,
                          direction: str) -> GapType:
        """
        Classify gap into one of four types.
        
        Classification Logic:
            1. COMMON: Small gap, normal volume, random occurrence
            2. BREAKAWAY: Large gap, high volume, after consolidation
            3. EXHAUSTION: Very large gap, extreme volume, after extended move
            4. MEASURING: Medium gap, elevated volume, mid-trend
        
        Args:
            gap_size: Absolute gap size in points
            gap_size_percent: Gap size as percentage of price
            volume: Volume at gap creation
            avg_volume: Average volume baseline
            candles: Full candle history for context
            gap_index: Index of gap in candles list
            direction: "UP" or "DOWN"
            
        Returns:
            GapType classification
        """
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Check if gap is small (likely common gap)
        if gap_size_percent < self.config.common_gap_max_size_percent:
            return GapType.COMMON
        
        # Check for exhaustion gap (extreme volume + large size)
        if volume_ratio >= self.config.volume_multiplier_exhaustion:
            # Exhaustion gaps occur after extended moves
            # Check if previous candles show extended trend
            lookback = min(20, gap_index)
            if lookback >= 5:
                prev_candles = candles[gap_index - lookback:gap_index]
                trend_strength = self._calculate_trend_strength(prev_candles)
                
                if trend_strength > 0.7:  # Strong trend preceding gap
                    return GapType.EXHAUSTION
        
        # Check for breakaway gap (high volume + after consolidation)
        if volume_ratio >= self.config.volume_multiplier_breakaway:
            # Breakaway gaps occur after consolidation/range
            lookback = min(20, gap_index)
            if lookback >= 5:
                prev_candles = candles[gap_index - lookback:gap_index]
                volatility = self._calculate_volatility(prev_candles)
                
                if volatility < 0.015:  # Low volatility = consolidation
                    return GapType.BREAKAWAY
        
        # Default to measuring gap (mid-trend continuation)
        return GapType.MEASURING
    
    def _calculate_trend_strength(self, candles: List[Candle]) -> float:
        """
        Calculate trend strength [0.0, 1.0].
        
        Strong trend = 1.0 (consistent directional movement)
        Weak trend = 0.0 (choppy/sideways)
        
        Args:
            candles: List of candles to analyze
            
        Returns:
            Trend strength [0.0, 1.0]
        """
        if len(candles) < 3:
            return 0.0
        
        # Calculate net price change
        net_change = abs(candles[-1].close - candles[0].open)
        
        # Calculate total price movement
        total_movement = sum(c.high - c.low for c in candles)
        
        if total_movement == 0:
            return 0.0
        
        # Trend strength = efficiency of movement
        efficiency = net_change / total_movement
        
        return min(1.0, efficiency)
    
    def _calculate_volatility(self, candles: List[Candle]) -> float:
        """
        Calculate volatility as average true range percentage.
        
        Args:
            candles: List of candles to analyze
            
        Returns:
            Volatility as average ATR / price ratio
        """
        if len(candles) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(candles)):
            prev_close = candles[i - 1].close
            curr_high = candles[i].high
            curr_low = candles[i].low
            
            tr = max(
                curr_high - curr_low,
                abs(curr_high - prev_close),
                abs(curr_low - prev_close)
            )
            true_ranges.append(tr)
        
        avg_tr = sum(true_ranges) / len(true_ranges) if true_ranges else 0
        avg_price = sum(c.close for c in candles) / len(candles)
        
        volatility = avg_tr / avg_price if avg_price > 0 else 0
        
        return volatility
    
    def _update_gap_fills(self, gaps: List[Gap], candles: List[Candle]) -> None:
        """
        Update fill status for all gaps based on subsequent price action.
        
        A gap is considered filled when:
            - Price touches at least 50% (configurable) of the gap range
            - Or price fully crosses through the gap
        
        Args:
            gaps: List of gaps to check
            candles: Full candle history
        """
        for gap in gaps:
            if gap.filled:
                continue  # Already filled
            
            # Find candles after gap creation
            gap_date = gap.date
            subsequent_candles = [c for c in candles if c.timestamp > gap_date]
            
            for candle in subsequent_candles:
                # Check if candle touches gap range
                if self._check_gap_fill(gap, candle):
                    gap.filled = True
                    break
    
    def _check_gap_fill(self, gap: Gap, candle: Candle) -> bool:
        """
        Check if a candle fills the gap.
        
        Fill criteria:
            - Candle range overlaps with gap by at least fill_tolerance %
            - Or candle fully crosses through gap
        
        Args:
            gap: Gap to check
            candle: Candle to test against gap
            
        Returns:
            True if gap is considered filled
        """
        # Check if candle range overlaps with gap
        overlap_low = max(gap.lower, candle.low)
        overlap_high = min(gap.upper, candle.high)
        
        if overlap_high <= overlap_low:
            return False  # No overlap
        
        overlap_size = overlap_high - overlap_low
        gap_size = gap.upper - gap.lower
        
        overlap_percent = overlap_size / gap_size if gap_size > 0 else 0
        
        # Gap is filled if overlap >= tolerance threshold
        return overlap_percent >= self.config.gap_fill_tolerance
    
    def calculate_futures_target(self,
                                gaps: List[Gap],
                                current_price: float,
                                current_date: datetime) -> Optional[float]:
        """
        Calculate target price based on nearest unfilled gap.
        
        Methodology:
            1. Filter gaps: unfilled and within max age
            2. Find nearest gap to current price (by distance to midpoint)
            3. Return gap's target level (midpoint)
        
        Args:
            gaps: List of detected gaps
            current_price: Current market price
            current_date: Current timestamp
            
        Returns:
            Target price (gap midpoint) or None if no valid gaps
            
        Example:
            target = engine.calculate_futures_target(gaps, 4500.0, datetime.now())
            
            if target:
                print(f"Target: ${target:.2f}")
                distance = abs(target - current_price)
                print(f"Distance: ${distance:.2f}")
        """
        if not gaps:
            return None
        
        # Filter for unfilled gaps within age limit
        valid_gaps = [
            gap for gap in gaps
            if not gap.filled and gap.age_days(current_date) <= self.config.max_gap_age_days
        ]
        
        if not valid_gaps:
            return None
        
        # Find nearest gap by distance to midpoint
        nearest_gap = min(valid_gaps, key=lambda g: abs(g.midpoint() - current_price))
        
        return nearest_gap.target_level
    
    def analyze_gaps(self,
                    gaps: List[Gap],
                    current_price: float,
                    current_date: datetime) -> GapAnalysisResult:
        """
        Complete gap analysis with target calculation and probability assessment.
        
        This is the primary interface for gap analysis. It provides:
            - Target price calculation
            - Gap fill probability estimation
            - Gravitational pull strength
            - Human-readable analysis breakdown
        
        Args:
            gaps: List of detected gaps
            current_price: Current market price
            current_date: Current timestamp
            
        Returns:
            GapAnalysisResult with complete analysis
            
        Example:
            result = engine.analyze_gaps(gaps, 4500.0, datetime.now())
            
            if result.target_price:
                print(result.details)
                print(f"Fill probability: {result.fill_probability:.1%}")
        """
        total_gaps = len(gaps)
        unfilled_gaps_list = [g for g in gaps if not g.filled]
        unfilled_count = len(unfilled_gaps_list)
        
        # Calculate target
        target = self.calculate_futures_target(gaps, current_price, current_date)
        
        # Find nearest gap
        nearest_gap = None
        if unfilled_gaps_list:
            nearest_gap = min(unfilled_gaps_list, 
                            key=lambda g: abs(g.midpoint() - current_price))
        
        # Calculate fill probability and gravitational pull
        fill_prob = 0.0
        grav_pull = 0.0
        
        if nearest_gap:
            distance = nearest_gap.distance_to_price(current_price)
            age = nearest_gap.age_days(current_date)
            
            # Fill probability: inverse of (age × distance)
            # Newer and closer gaps have higher probability
            if distance > 0 and age > 0:
                # Normalize to [0, 1] range
                fill_prob = 1.0 / (1.0 + (age * distance / 100.0))
            else:
                fill_prob = 1.0  # Price is at the gap
            
            # Gravitational pull: inverse square law
            if distance > 0:
                grav_pull = min(1.0, 100.0 / (distance ** 2))
            else:
                grav_pull = 1.0
            
            # Adjust for gap type
            type_multipliers = {
                GapType.EXHAUSTION: 1.5,   # Highest pull
                GapType.BREAKAWAY: 1.3,
                GapType.MEASURING: 1.1,
                GapType.COMMON: 0.8        # Lowest pull
            }
            grav_pull *= type_multipliers[nearest_gap.gap_type]
            grav_pull = min(1.0, grav_pull)
        
        # Build detailed breakdown
        target_str = f"${target:.2f}" if target is not None else "N/A"
        nearest_str = f"${nearest_gap.target_level:.2f}" if nearest_gap else "N/A"
        gap_type_str = nearest_gap.gap_type.value if nearest_gap else "N/A"
        gap_age_str = f"{nearest_gap.age_days(current_date):.1f} days" if nearest_gap else "N/A"
        distance_str = f"${distance:.2f}" if nearest_gap else "N/A"
        
        details = f"""Gap Analysis Summary:
  Total Gaps Detected:     {total_gaps}
  Unfilled Gaps:           {unfilled_count}
  Current Price:           ${current_price:.2f}
  
  Target Analysis:
    Target Price:          {target_str}
    Nearest Gap:           {nearest_str}
    Gap Type:              {gap_type_str}
    Gap Age:               {gap_age_str}
    Distance to Gap:       {distance_str}
    
  Probability Metrics:
    Fill Probability:      {fill_prob:.1%}
    Gravitational Pull:    {grav_pull:.1%}
    
  Interpretation:
    {'High probability gap fill expected' if fill_prob > 0.6 else 'Moderate gap fill probability' if fill_prob > 0.3 else 'Low gap fill probability'}
    {'Strong gravitational pull toward gap' if grav_pull > 0.6 else 'Moderate gravitational influence' if grav_pull > 0.3 else 'Weak gravitational influence'}
"""
        
        return GapAnalysisResult(
            target_price=target,
            nearest_gap=nearest_gap,
            total_gaps=total_gaps,
            unfilled_gaps=unfilled_count,
            fill_probability=fill_prob,
            gravitational_pull=grav_pull,
            details=details
        )
    
    def get_unfilled_gaps(self,
                         gaps: Optional[List[Gap]] = None,
                         current_date: Optional[datetime] = None) -> List[Gap]:
        """
        Get list of unfilled gaps within age limit.
        
        Args:
            gaps: Optional gap list. Uses self.gaps if None.
            current_date: Optional date for age calculation. Uses now() if None.
            
        Returns:
            List of unfilled gaps within max age
        """
        gap_list = gaps if gaps is not None else self.gaps
        check_date = current_date if current_date is not None else datetime.now()
        
        return [
            gap for gap in gap_list
            if not gap.filled and gap.age_days(check_date) <= self.config.max_gap_age_days
        ]
    
    def get_gap_by_type(self, gap_type: GapType, gaps: Optional[List[Gap]] = None) -> List[Gap]:
        """
        Filter gaps by type.
        
        Args:
            gap_type: Type to filter for
            gaps: Optional gap list. Uses self.gaps if None.
            
        Returns:
            List of gaps matching the specified type
        """
        gap_list = gaps if gaps is not None else self.gaps
        return [gap for gap in gap_list if gap.gap_type == gap_type]
