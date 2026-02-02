"""
AXIOM 3: Absorption Reversal Implementation
============================================

Theoretical Foundation:
    Based on Kyle (1985) informed trader theory and Glosten-Milgrom (1985) 
    information asymmetry model. When aggressive liquidity (market orders) 
    is absorbed by passive liquidity (limit orders) without price continuation,
    it signals exhaustion of the current move and high probability reversal.

Mathematical Model:
    Exhaustion Score: E(t) = w₁·V(t) + w₂·B(t) + w₃·P(t) + w₄·R(t)
    
    where:
        V(t) = Volume absorption score [0.0, 1.0]
        B(t) = Candle body rejection score [0.0, 1.0]
        P(t) = Price stagnation score [0.0, 1.0]
        R(t) = Reversal pattern score [0.0, 1.0]
        
        w₁ = 0.30 (volume weight)
        w₂ = 0.30 (body weight)
        w₃ = 0.25 (price weight)
        w₄ = 0.15 (reversal weight)
        
    Threshold: E(t) ≥ 0.70 indicates absorption reversal likely

References:
    - Kyle, A. S. (1985). "Continuous Auctions and Insider Trading"
    - Glosten, L. R., & Milgrom, P. R. (1985). "Bid, Ask and Transaction Prices"
    - Rosu, I. (2009). "A Dynamic Model of the Limit Order Book"
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum


@dataclass
class VolumeBar:
    """
    Volume data with bid/ask breakdown for absorption analysis.
    
    Attributes:
        timestamp: Bar timestamp
        volume: Total volume
        bid_volume: Volume executed at bid (selling pressure)
        ask_volume: Volume executed at ask (buying pressure)
        delta: Net volume delta (bid_volume - ask_volume)
    """
    timestamp: datetime
    volume: float
    bid_volume: float
    ask_volume: float
    delta: float
    
    def __post_init__(self):
        """Validate volume data integrity"""
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")
        if self.bid_volume < 0 or self.ask_volume < 0:
            raise ValueError("Bid/ask volumes cannot be negative")
        if abs(self.bid_volume + self.ask_volume - self.volume) > 0.01:
            raise ValueError("Bid + Ask volume must equal total volume")


# Import Candle from participant module
from .participant import Candle


@dataclass
class ExhaustionConfig:
    """
    Configuration for exhaustion detection parameters.
    
    Weights must sum to 1.0 (convex combination).
    Optimized via walk-forward analysis per README Section 4.1.
    """
    volume_weight: float = 0.30
    body_weight: float = 0.30
    price_weight: float = 0.25
    reversal_weight: float = 0.15
    threshold: float = 0.70
    volume_lookback: int = 20  # Candles for volume analysis
    price_lookback: int = 10   # Candles for price stagnation
    reversal_lookback: int = 5  # Candles for reversal patterns
    
    def __post_init__(self):
        """Validate configuration constraints"""
        total_weight = (self.volume_weight + self.body_weight + 
                       self.price_weight + self.reversal_weight)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"Threshold must be [0.0, 1.0], got {self.threshold}")


@dataclass
class ExhaustionResult:
    """
    Complete exhaustion analysis result with score breakdown.
    
    Attributes:
        score: Overall exhaustion score [0.0, 1.0]
        volume_score: Volume absorption component [0.0, 1.0]
        body_score: Candle body rejection component [0.0, 1.0]
        price_score: Price stagnation component [0.0, 1.0]
        reversal_score: Reversal pattern component [0.0, 1.0]
        threshold_met: True if score >= threshold
        timestamp: Analysis timestamp
        details: Human-readable breakdown of scoring factors
    """
    score: float
    volume_score: float
    body_score: float
    price_score: float
    reversal_score: float
    threshold_met: bool
    timestamp: datetime
    details: str
    
    def __post_init__(self):
        """Validate result ranges"""
        for name, value in [
            ('score', self.score),
            ('volume_score', self.volume_score),
            ('body_score', self.body_score),
            ('price_score', self.price_score),
            ('reversal_score', self.reversal_score)
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be [0.0, 1.0], got {value}")


def calculate_exhaustion_score(candles: List[Candle], 
                              volume_data: List[VolumeBar]) -> float:
    """
    Calculate exhaustion score from candles and volume data.
    
    This is a module-level wrapper function that matches the README.md
    Section 3 interface specification. It creates a detector instance
    and returns the score.
    
    Returns: Exhaustion score [0.0, 1.0]
    >= 0.70 indicates absorption reversal likely
    
    Args:
        candles: List of OHLCV candles
        volume_data: List of volume bars with bid/ask breakdown
        
    Returns:
        float: Exhaustion score [0.0, 1.0]
        
    Example:
        >>> score = calculate_exhaustion_score(candles, volume_bars)
        >>> if score >= 0.70:
        ...     print("Absorption detected - reversal likely")
    """
    detector = ExhaustionDetector()
    result = detector.detect(candles, volume_data)
    return result.score


class ExhaustionDetector:
    """
    AXIOM 3: Absorption Reversal Detector
    
    Detects when aggressive liquidity is absorbed by passive liquidity,
    indicating exhaustion of current move and high-probability reversal.
    
    Mathematical Properties:
        - Output range: [0.0, 1.0] (bounded)
        - Monotonic: More absorption factors → higher score
        - Weighted linear combination (convex optimization space)
        - Threshold-based binary classification (score >= 0.70)
    
    Usage:
        detector = ExhaustionDetector()
        result = detector.detect_exhaustion(candles, volume_data)
        
        if result.threshold_met:
            print(f"Absorption detected! Score: {result.score:.3f}")
            print(result.details)
    """
    
    def __init__(self, config: Optional[ExhaustionConfig] = None):
        """
        Initialize exhaustion detector.
        
        Args:
            config: Optional configuration. Uses defaults if None.
        """
        self.config = config if config is not None else ExhaustionConfig()
    
    def calculate_volume_absorption(self, 
                                   volume_data: List[VolumeBar],
                                   direction: str = "LONG") -> float:
        """
        Calculate volume absorption score [0.0, 1.0].
        
        Theory:
            When large volume occurs without price continuation, it indicates
            absorption by passive liquidity (limit orders). Volume delta 
            divergence from price movement signals exhaustion.
        
        Methodology:
            1. Calculate volume trend (increasing = more aggressive orders)
            2. Calculate delta divergence (delta vs price direction mismatch)
            3. Normalize to [0.0, 1.0] range
        
        Args:
            volume_data: List of VolumeBar data
            direction: "LONG" for uptrend exhaustion, "SHORT" for downtrend
            
        Returns:
            Volume absorption score [0.0, 1.0]
            1.0 = Maximum absorption (high volume, negative delta divergence)
            0.0 = No absorption (low volume or delta confirms direction)
        """
        if not volume_data:
            return 0.0
        
        lookback = min(self.config.volume_lookback, len(volume_data))
        if lookback < 3:
            return 0.0
        
        recent_bars = volume_data[-lookback:]
        
        # Factor 1: Volume trend (increasing volume = more aggression)
        volumes = [bar.volume for bar in recent_bars]
        avg_volume = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-3:]) / 3  # Last 3 bars
        volume_increase_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Factor 2: Delta divergence
        # LONG exhaustion: positive delta with high volume = buyers exhausted
        # SHORT exhaustion: negative delta with high volume = sellers exhausted
        deltas = [bar.delta for bar in recent_bars]
        avg_delta = sum(deltas) / len(deltas)
        
        if direction == "LONG":
            # For uptrend exhaustion, positive delta is exhaustion signal
            delta_score = max(0.0, avg_delta) / max(1.0, avg_volume)
        else:  # SHORT
            # For downtrend exhaustion, negative delta is exhaustion signal
            delta_score = abs(min(0.0, avg_delta)) / max(1.0, avg_volume)
        
        # Factor 3: Volume concentration (spikes indicate climax)
        max_volume = max(volumes)
        volume_concentration = max_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Combine factors
        volume_component = min(1.0, (volume_increase_ratio - 1.0) * 2.0)  # 0-1 range
        delta_component = min(1.0, delta_score * 10.0)  # Normalize
        concentration_component = min(1.0, (volume_concentration - 1.0) / 2.0)  # 0-1 range
        
        # Weighted combination
        score = (0.40 * volume_component +
                0.40 * delta_component +
                0.20 * concentration_component)
        
        return max(0.0, min(1.0, score))
    
    def calculate_candle_body_rejection(self, 
                                       candles: List[Candle],
                                       direction: str = "LONG") -> float:
        """
        Calculate candle body rejection score [0.0, 1.0].
        
        Theory:
            Long wicks relative to body indicate price rejection - market
            attempted to push price further but was absorbed by liquidity.
            Classic exhaustion signal in candlestick analysis.
        
        Methodology:
            1. Calculate wick-to-body ratios for recent candles
            2. Identify rejection patterns (long wick in direction of trend)
            3. Normalize to [0.0, 1.0] range
        
        Args:
            candles: List of Candle data
            direction: "LONG" for uptrend exhaustion, "SHORT" for downtrend
            
        Returns:
            Body rejection score [0.0, 1.0]
            1.0 = Strong rejection (long wicks, small bodies)
            0.0 = No rejection (small wicks, large bodies)
        """
        if not candles:
            return 0.0
        
        lookback = min(5, len(candles))  # Focus on recent candles
        if lookback < 2:
            return 0.0
        
        recent_candles = candles[-lookback:]
        rejection_scores = []
        
        for candle in recent_candles:
            body_size = abs(candle.close - candle.open)
            upper_wick = candle.high - max(candle.open, candle.close)
            lower_wick = min(candle.open, candle.close) - candle.low
            total_range = candle.high - candle.low
            
            if total_range == 0:
                rejection_scores.append(0.0)
                continue
            
            # Calculate wick ratios
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            body_ratio = body_size / total_range
            
            if direction == "LONG":
                # For uptrend exhaustion, look for long upper wicks
                wick_score = upper_wick_ratio
                # Bearish rejection: upper wick > 2x body
                if upper_wick > 2.0 * body_size:
                    wick_score *= 1.5  # Boost score for strong rejection
            else:  # SHORT
                # For downtrend exhaustion, look for long lower wicks
                wick_score = lower_wick_ratio
                # Bullish rejection: lower wick > 2x body
                if lower_wick > 2.0 * body_size:
                    wick_score *= 1.5  # Boost score for strong rejection
            
            rejection_scores.append(min(1.0, wick_score))
        
        # Weight recent candles more heavily
        weights = [i + 1 for i in range(len(rejection_scores))]
        weighted_score = sum(s * w for s, w in zip(rejection_scores, weights))
        total_weight = sum(weights)
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return max(0.0, min(1.0, final_score))
    
    def calculate_price_stagnation(self, candles: List[Candle]) -> float:
        """
        Calculate price stagnation score [0.0, 1.0].
        
        Theory:
            When price makes less progress despite continued attempts to push
            higher/lower, it indicates absorption. Measured as ratio of
            net price change to total price movement (efficiency metric).
        
        Methodology:
            1. Calculate net price change (end - start)
            2. Calculate total price movement (sum of ranges)
            3. Efficiency = net_change / total_movement
            4. Stagnation = 1.0 - efficiency
        
        Args:
            candles: List of Candle data
            
        Returns:
            Price stagnation score [0.0, 1.0]
            1.0 = Maximum stagnation (high movement, low progress)
            0.0 = No stagnation (efficient directional movement)
        """
        if not candles:
            return 0.0
        
        lookback = min(self.config.price_lookback, len(candles))
        if lookback < 3:
            return 0.0
        
        recent_candles = candles[-lookback:]
        
        # Net price change (directional progress)
        start_price = recent_candles[0].open
        end_price = recent_candles[-1].close
        net_change = abs(end_price - start_price)
        
        # Total price movement (sum of all ranges)
        total_movement = sum(c.high - c.low for c in recent_candles)
        
        if total_movement == 0:
            return 0.0
        
        # Efficiency: how much net progress per unit of movement
        efficiency = net_change / total_movement
        
        # Stagnation is inverse of efficiency
        stagnation = 1.0 - efficiency
        
        # Also factor in overlapping ranges (choppy price action)
        overlaps = 0
        for i in range(1, len(recent_candles)):
            prev = recent_candles[i - 1]
            curr = recent_candles[i]
            
            # Check if ranges overlap significantly
            overlap_low = max(prev.low, curr.low)
            overlap_high = min(prev.high, curr.high)
            
            if overlap_high > overlap_low:
                overlap_range = overlap_high - overlap_low
                avg_range = ((prev.high - prev.low) + (curr.high - curr.low)) / 2
                if avg_range > 0:
                    overlap_ratio = overlap_range / avg_range
                    overlaps += overlap_ratio
        
        # Normalize overlap score
        avg_overlaps = overlaps / (len(recent_candles) - 1) if len(recent_candles) > 1 else 0
        
        # Combine stagnation and overlap
        final_score = 0.70 * stagnation + 0.30 * min(1.0, avg_overlaps)
        
        return max(0.0, min(1.0, final_score))
    
    def calculate_reversal_patterns(self, candles: List[Candle]) -> float:
        """
        Calculate reversal pattern score [0.0, 1.0].
        
        Theory:
            Classic reversal candlestick patterns (engulfing, hammer, shooting star)
            represent absorption events where one side overwhelms the other.
        
        Patterns Detected:
            - Engulfing patterns (bullish/bearish)
            - Hammer / Hanging Man (long lower wick)
            - Shooting Star / Inverted Hammer (long upper wick)
            - Doji patterns (indecision)
        
        Args:
            candles: List of Candle data
            
        Returns:
            Reversal pattern score [0.0, 1.0]
            1.0 = Strong reversal pattern detected
            0.0 = No reversal patterns
        """
        if not candles:
            return 0.0
        
        lookback = min(self.config.reversal_lookback, len(candles))
        if lookback < 2:
            return 0.0
        
        recent_candles = candles[-lookback:]
        pattern_scores = []
        
        for i in range(1, len(recent_candles)):
            prev = recent_candles[i - 1]
            curr = recent_candles[i]
            
            curr_body = abs(curr.close - curr.open)
            curr_range = curr.high - curr.low
            curr_upper_wick = curr.high - max(curr.open, curr.close)
            curr_lower_wick = min(curr.open, curr.close) - curr.low
            
            prev_body = abs(prev.close - prev.open)
            
            score = 0.0
            
            # Pattern 1: Engulfing
            if curr_body > prev_body * 1.5:  # Current candle significantly larger
                # Bearish engulfing (after uptrend)
                if prev.close > prev.open and curr.close < curr.open:
                    if curr.open >= prev.close and curr.close <= prev.open:
                        score = max(score, 0.9)
                # Bullish engulfing (after downtrend)
                elif prev.close < prev.open and curr.close > curr.open:
                    if curr.open <= prev.close and curr.close >= prev.open:
                        score = max(score, 0.9)
            
            # Pattern 2: Hammer / Hanging Man
            if curr_range > 0:
                if curr_lower_wick > 2.0 * curr_body:  # Long lower wick
                    if curr_lower_wick > 0.6 * curr_range:
                        score = max(score, 0.75)
            
            # Pattern 3: Shooting Star / Inverted Hammer
            if curr_range > 0:
                if curr_upper_wick > 2.0 * curr_body:  # Long upper wick
                    if curr_upper_wick > 0.6 * curr_range:
                        score = max(score, 0.75)
            
            # Pattern 4: Doji (indecision)
            if curr_range > 0:
                if curr_body < 0.1 * curr_range:  # Very small body
                    score = max(score, 0.5)
            
            pattern_scores.append(score)
        
        # Return maximum pattern score found (any strong pattern triggers signal)
        return max(pattern_scores) if pattern_scores else 0.0
    
    def calculate_exhaustion_score(self,
                                  candles: List[Candle],
                                  volume_data: Optional[List[VolumeBar]] = None,
                                  direction: str = "LONG") -> float:
        """
        Calculate overall exhaustion score using weighted linear combination.
        
        Mathematical Formula:
            E(t) = w₁·V(t) + w₂·B(t) + w₃·P(t) + w₄·R(t)
            
            where weights are from config (default: 0.30, 0.30, 0.25, 0.15)
        
        Args:
            candles: List of OHLCV candles
            volume_data: Optional volume bar data with bid/ask breakdown
            direction: "LONG" for uptrend exhaustion, "SHORT" for downtrend
            
        Returns:
            Exhaustion score [0.0, 1.0]
            >= 0.70 indicates high probability absorption reversal
        """
        if not candles:
            return 0.0
        
        # Calculate component scores
        volume_score = 0.0
        if volume_data and len(volume_data) > 0:
            volume_score = self.calculate_volume_absorption(volume_data, direction)
        
        body_score = self.calculate_candle_body_rejection(candles, direction)
        price_score = self.calculate_price_stagnation(candles)
        reversal_score = self.calculate_reversal_patterns(candles)
        
        # Weighted linear combination
        total_score = (
            self.config.volume_weight * volume_score +
            self.config.body_weight * body_score +
            self.config.price_weight * price_score +
            self.config.reversal_weight * reversal_score
        )
        
        return max(0.0, min(1.0, total_score))
    
    def detect_exhaustion(self,
                         candles: List[Candle],
                         volume_data: Optional[List[VolumeBar]] = None,
                         direction: str = "LONG") -> ExhaustionResult:
        """
        Main exhaustion detection method - returns complete result.
        
        This is the primary interface for exhaustion detection. It calculates
        all component scores and returns detailed breakdown.
        
        Args:
            candles: List of OHLCV candles
            volume_data: Optional volume bar data with bid/ask breakdown
            direction: "LONG" for uptrend exhaustion, "SHORT" for downtrend
            
        Returns:
            ExhaustionResult with complete score breakdown
            
        Example:
            detector = ExhaustionDetector()
            result = detector.detect_exhaustion(candles, volume_data)
            
            if result.threshold_met:
                print(f"EXHAUSTION DETECTED!")
                print(f"Score: {result.score:.3f}")
                print(result.details)
        """
        if not candles:
            return ExhaustionResult(
                score=0.0,
                volume_score=0.0,
                body_score=0.0,
                price_score=0.0,
                reversal_score=0.0,
                threshold_met=False,
                timestamp=datetime.now(),
                details="No candle data provided"
            )
        
        # Calculate all component scores
        volume_score = 0.0
        if volume_data and len(volume_data) > 0:
            volume_score = self.calculate_volume_absorption(volume_data, direction)
        
        body_score = self.calculate_candle_body_rejection(candles, direction)
        price_score = self.calculate_price_stagnation(candles)
        reversal_score = self.calculate_reversal_patterns(candles)
        
        # Calculate overall score
        total_score = (
            self.config.volume_weight * volume_score +
            self.config.body_weight * body_score +
            self.config.price_weight * price_score +
            self.config.reversal_weight * reversal_score
        )
        
        # Check threshold
        threshold_met = total_score >= self.config.threshold
        
        # Build detailed breakdown
        details = f"""Exhaustion Analysis Breakdown:
  Overall Score: {total_score:.3f} (Threshold: {self.config.threshold:.2f})
  
  Component Scores:
    • Volume Absorption:  {volume_score:.3f} (weight: {self.config.volume_weight:.2f})
    • Body Rejection:     {body_score:.3f} (weight: {self.config.body_weight:.2f})
    • Price Stagnation:   {price_score:.3f} (weight: {self.config.price_weight:.2f})
    • Reversal Patterns:  {reversal_score:.3f} (weight: {self.config.reversal_weight:.2f})
  
  Weighted Contributions:
    • Volume:    {self.config.volume_weight * volume_score:.3f}
    • Body:      {self.config.body_weight * body_score:.3f}
    • Price:     {self.config.price_weight * price_score:.3f}
    • Reversal:  {self.config.reversal_weight * reversal_score:.3f}
  
  Result: {'EXHAUSTION DETECTED' if threshold_met else 'No exhaustion'}
"""
        
        return ExhaustionResult(
            score=total_score,
            volume_score=volume_score,
            body_score=body_score,
            price_score=price_score,
            reversal_score=reversal_score,
            threshold_met=threshold_met,
            timestamp=candles[-1].timestamp if candles else datetime.now(),
            details=details
        )
