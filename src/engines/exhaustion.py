from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from enum import Enum

@dataclass
class VolumeBar:
    timestamp: datetime
    volume: float
    bid_volume: float
    ask_volume: float
    delta: float
    
    def __post_init__(self):
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")
        if self.bid_volume < 0 or self.ask_volume < 0:
            raise ValueError("Bid/ask volumes cannot be negative")
        if abs(self.bid_volume + self.ask_volume - self.volume) > 0.01:
            raise ValueError("Bid + Ask volume must equal total volume")

from .participant import Candle

@dataclass
class ExhaustionConfig:
    volume_weight: float = 0.30
    body_weight: float = 0.30
    price_weight: float = 0.25
    reversal_weight: float = 0.15
    threshold: float = 0.70
    volume_lookback: int = 20  # Candles for volume analysis
    price_lookback: int = 10   # Candles for price stagnation
    reversal_lookback: int = 5  # Candles for reversal patterns
    
    def __post_init__(self):
        total_weight = (self.volume_weight + self.body_weight + 
                       self.price_weight + self.reversal_weight)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"Threshold must be [0.0, 1.0], got {self.threshold}")

@dataclass
class ExhaustionResult:
    score: float
    volume_score: float
    body_score: float
    price_score: float
    reversal_score: float
    threshold_met: bool
    timestamp: datetime
    details: str
    
    def __post_init__(self):
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
    detector = ExhaustionDetector()
    result = detector.detect(candles, volume_data)
    return result.score

class ExhaustionDetector:
    def __init__(self, config: Optional[ExhaustionConfig] = None):
        self.config = config if config is not None else ExhaustionConfig()
    
    def calculate_volume_absorption(self, 
                                   volume_data: List[VolumeBar],
                                   direction: str = "LONG") -> float:
        if not volume_data:
            return 0.0
        
        lookback = min(self.config.volume_lookback, len(volume_data))
        if lookback < 3:
            return 0.0
        
        recent_bars = volume_data[-lookback:]
        
        volumes = [bar.volume for bar in recent_bars]
        avg_volume = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-3:]) / 3  # Last 3 bars
        volume_increase_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        deltas = [bar.delta for bar in recent_bars]
        avg_delta = sum(deltas) / len(deltas)
        
        if direction == "LONG":
            delta_score = max(0.0, avg_delta) / max(1.0, avg_volume)
        else:  # SHORT
            delta_score = abs(min(0.0, avg_delta)) / max(1.0, avg_volume)
        
        max_volume = max(volumes)
        volume_concentration = max_volume / avg_volume if avg_volume > 0 else 1.0
        
        volume_component = min(1.0, (volume_increase_ratio - 1.0) * 2.0)  # 0-1 range
        delta_component = min(1.0, delta_score * 10.0)  # Normalize
        concentration_component = min(1.0, (volume_concentration - 1.0) / 2.0)  # 0-1 range
        
        score = (0.40 * volume_component +
                0.40 * delta_component +
                0.20 * concentration_component)
        
        return max(0.0, min(1.0, score))
    
    def calculate_candle_body_rejection(self, 
                                       candles: List[Candle],
                                       direction: str = "LONG") -> float:
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
            
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            body_ratio = body_size / total_range
            
            if direction == "LONG":
                wick_score = upper_wick_ratio
                if upper_wick > 2.0 * body_size:
                    wick_score *= 1.5  # Boost score for strong rejection
            else:  # SHORT
                wick_score = lower_wick_ratio
                if lower_wick > 2.0 * body_size:
                    wick_score *= 1.5  # Boost score for strong rejection
            
            rejection_scores.append(min(1.0, wick_score))
        
        weights = [i + 1 for i in range(len(rejection_scores))]
        weighted_score = sum(s * w for s, w in zip(rejection_scores, weights))
        total_weight = sum(weights)
        
        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return max(0.0, min(1.0, final_score))
    
    def calculate_price_stagnation(self, candles: List[Candle]) -> float:
        if not candles:
            return 0.0
        
        lookback = min(self.config.price_lookback, len(candles))
        if lookback < 3:
            return 0.0
        
        recent_candles = candles[-lookback:]
        
        start_price = recent_candles[0].open
        end_price = recent_candles[-1].close
        net_change = abs(end_price - start_price)
        
        total_movement = sum(c.high - c.low for c in recent_candles)
        
        if total_movement == 0:
            return 0.0
        
        efficiency = net_change / total_movement
        
        stagnation = 1.0 - efficiency
        
        overlaps = 0
        for i in range(1, len(recent_candles)):
            prev = recent_candles[i - 1]
            curr = recent_candles[i]
            
            overlap_low = max(prev.low, curr.low)
            overlap_high = min(prev.high, curr.high)
            
            if overlap_high > overlap_low:
                overlap_range = overlap_high - overlap_low
                avg_range = ((prev.high - prev.low) + (curr.high - curr.low)) / 2
                if avg_range > 0:
                    overlap_ratio = overlap_range / avg_range
                    overlaps += overlap_ratio
        
        avg_overlaps = overlaps / (len(recent_candles) - 1) if len(recent_candles) > 1 else 0
        
        final_score = 0.70 * stagnation + 0.30 * min(1.0, avg_overlaps)
        
        return max(0.0, min(1.0, final_score))
    
    def calculate_reversal_patterns(self, candles: List[Candle]) -> float:
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
            
            if curr_body > prev_body * 1.5:  # Current candle significantly larger
                if prev.close > prev.open and curr.close < curr.open:
                    if curr.open >= prev.close and curr.close <= prev.open:
                        score = max(score, 0.9)
                elif prev.close < prev.open and curr.close > curr.open:
                    if curr.open <= prev.close and curr.close >= prev.open:
                        score = max(score, 0.9)
            
            if curr_range > 0:
                if curr_lower_wick > 2.0 * curr_body:  # Long lower wick
                    if curr_lower_wick > 0.6 * curr_range:
                        score = max(score, 0.75)
            
            if curr_range > 0:
                if curr_upper_wick > 2.0 * curr_body:  # Long upper wick
                    if curr_upper_wick > 0.6 * curr_range:
                        score = max(score, 0.75)
            
            if curr_range > 0:
                if curr_body < 0.1 * curr_range:  # Very small body
                    score = max(score, 0.5)
            
            pattern_scores.append(score)
        
        return max(pattern_scores) if pattern_scores else 0.0
    
    def calculate_exhaustion_score(self,
                                  candles: List[Candle],
                                  volume_data: Optional[List[VolumeBar]] = None,
                                  direction: str = "LONG") -> float:
        if not candles:
            return 0.0
        
        volume_score = 0.0
        if volume_data and len(volume_data) > 0:
            volume_score = self.calculate_volume_absorption(volume_data, direction)
        
        body_score = self.calculate_candle_body_rejection(candles, direction)
        price_score = self.calculate_price_stagnation(candles)
        reversal_score = self.calculate_reversal_patterns(candles)
        
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
        
        volume_score = 0.0
        if volume_data and len(volume_data) > 0:
            volume_score = self.calculate_volume_absorption(volume_data, direction)
        
        body_score = self.calculate_candle_body_rejection(candles, direction)
        price_score = self.calculate_price_stagnation(candles)
        reversal_score = self.calculate_reversal_patterns(candles)
        
        total_score = (
            self.config.volume_weight * volume_score +
            self.config.body_weight * body_score +
            self.config.price_weight * price_score +
            self.config.reversal_weight * reversal_score
        )
        
        threshold_met = total_score >= self.config.threshold
        
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
