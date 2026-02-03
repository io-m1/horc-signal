from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum
import math

from .participant import Candle

class GapType(Enum):
    COMMON = "common"
    BREAKAWAY = "breakaway"
    EXHAUSTION = "exhaustion"
    MEASURING = "measuring"

@dataclass
class Gap:
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
        if self.upper <= self.lower:
            raise ValueError(f"Gap upper ({self.upper}) must be > lower ({self.lower})")
        
        self.size = self.upper - self.lower
        
        if self.target_level is None:
            self.target_level = (self.upper + self.lower) / 2.0
        
        self.direction = "UP"  # Will be set properly in detect_gaps()
    
    def midpoint(self) -> float:
        return (self.upper + self.lower) / 2.0
    
    def contains_price(self, price: float) -> bool:
        return self.lower <= price <= self.upper
    
    def age_days(self, current_date: datetime) -> float:
        return (current_date - self.date).total_seconds() / 86400.0
    
    def distance_to_price(self, price: float) -> float:
        if self.contains_price(price):
            return 0.0
        elif price < self.lower:
            return self.lower - price
        else:
            return price - self.upper

@dataclass
class GapConfig:
    min_gap_size_points: float = 2.0
    min_gap_size_percent: float = 0.001  # 0.1%
    max_gap_age_days: int = 30
    gap_fill_tolerance: float = 0.5  # 50% of gap must be filled
    volume_multiplier_breakaway: float = 1.5
    volume_multiplier_exhaustion: float = 2.0
    common_gap_max_size_percent: float = 0.02  # 2%
    
    def __post_init__(self):
        if self.min_gap_size_points < 0:
            raise ValueError("min_gap_size_points must be >= 0")
        if not (0 <= self.min_gap_size_percent <= 1):
            raise ValueError("min_gap_size_percent must be [0, 1]")
        if not (0 <= self.gap_fill_tolerance <= 1):
            raise ValueError("gap_fill_tolerance must be [0, 1]")

@dataclass
class GapAnalysisResult:
    target_price: Optional[float]
    nearest_gap: Optional[Gap]
    total_gaps: int
    unfilled_gaps: int
    fill_probability: float
    gravitational_pull: float
    details: str

class FuturesGapEngine:
    def __init__(self, config: Optional[GapConfig] = None):
        self.config = config if config is not None else GapConfig()
        self.gaps: List[Gap] = []
    
    def detect_gaps(self, candles: List[Candle]) -> List[Gap]:
        if not candles or len(candles) < 2:
            return []
        
        detected_gaps: List[Gap] = []
        
        volumes = [c.volume for c in candles if c.volume > 0]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        for i in range(1, len(candles)):
            prev_candle = candles[i - 1]
            curr_candle = candles[i]
            
            gap_upper = None
            gap_lower = None
            gap_direction = None
            
            if curr_candle.open > prev_candle.high:
                gap_lower = prev_candle.high
                gap_upper = curr_candle.open
                gap_direction = "UP"
                
            elif curr_candle.open < prev_candle.low:
                gap_upper = prev_candle.low
                gap_lower = curr_candle.open
                gap_direction = "DOWN"
            
            if gap_upper is not None and gap_lower is not None:
                gap_size = gap_upper - gap_lower
                gap_size_percent = gap_size / prev_candle.close if prev_candle.close > 0 else 0
                
                if (gap_size >= self.config.min_gap_size_points and
                    gap_size_percent >= self.config.min_gap_size_percent):
                    
                    gap_type = self._classify_gap_type(
                        gap_size=gap_size,
                        gap_size_percent=gap_size_percent,
                        volume=curr_candle.volume,
                        avg_volume=avg_volume,
                        candles=candles,
                        gap_index=i,
                        direction=gap_direction
                    )
                    
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
        
        self._update_gap_fills(detected_gaps, candles)
        
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
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        if gap_size_percent < self.config.common_gap_max_size_percent:
            return GapType.COMMON
        
        if volume_ratio >= self.config.volume_multiplier_exhaustion:
            lookback = min(20, gap_index)
            if lookback >= 5:
                prev_candles = candles[gap_index - lookback:gap_index]
                trend_strength = self._calculate_trend_strength(prev_candles)
                
                if trend_strength > 0.7:  # Strong trend preceding gap
                    return GapType.EXHAUSTION
        
        if volume_ratio >= self.config.volume_multiplier_breakaway:
            lookback = min(20, gap_index)
            if lookback >= 5:
                prev_candles = candles[gap_index - lookback:gap_index]
                volatility = self._calculate_volatility(prev_candles)
                
                if volatility < 0.015:  # Low volatility = consolidation
                    return GapType.BREAKAWAY
        
        return GapType.MEASURING
    
    def _calculate_trend_strength(self, candles: List[Candle]) -> float:
        if len(candles) < 3:
            return 0.0
        
        net_change = abs(candles[-1].close - candles[0].open)
        
        total_movement = sum(c.high - c.low for c in candles)
        
        if total_movement == 0:
            return 0.0
        
        efficiency = net_change / total_movement
        
        return min(1.0, efficiency)
    
    def _calculate_volatility(self, candles: List[Candle]) -> float:
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
        for gap in gaps:
            if gap.filled:
                continue  # Already filled
            
            gap_date = gap.date
            subsequent_candles = [c for c in candles if c.timestamp > gap_date]
            
            for candle in subsequent_candles:
                if self._check_gap_fill(gap, candle):
                    gap.filled = True
                    break
    
    def _check_gap_fill(self, gap: Gap, candle: Candle) -> bool:
        overlap_low = max(gap.lower, candle.low)
        overlap_high = min(gap.upper, candle.high)
        
        if overlap_high <= overlap_low:
            return False  # No overlap
        
        overlap_size = overlap_high - overlap_low
        gap_size = gap.upper - gap.lower
        
        overlap_percent = overlap_size / gap_size if gap_size > 0 else 0
        
        return overlap_percent >= self.config.gap_fill_tolerance
    
    def calculate_futures_target(self,
                                gaps: List[Gap],
                                current_price: float,
                                current_date: datetime) -> Optional[float]:
        if not gaps:
            return None
        
        valid_gaps = [
            gap for gap in gaps
            if not gap.filled and gap.age_days(current_date) <= self.config.max_gap_age_days
        ]
        
        if not valid_gaps:
            return None
        
        nearest_gap = min(valid_gaps, key=lambda g: abs(g.midpoint() - current_price))
        
        return nearest_gap.target_level
    
    def analyze_gaps(self,
                    gaps: List[Gap],
                    current_price: float,
                    current_date: datetime) -> GapAnalysisResult:
        total_gaps = len(gaps)
        unfilled_gaps_list = [g for g in gaps if not g.filled]
        unfilled_count = len(unfilled_gaps_list)
        
        target = self.calculate_futures_target(gaps, current_price, current_date)
        
        nearest_gap = None
        if unfilled_gaps_list:
            nearest_gap = min(unfilled_gaps_list, 
                            key=lambda g: abs(g.midpoint() - current_price))
        
        fill_prob = 0.0
        grav_pull = 0.0
        
        if nearest_gap:
            distance = nearest_gap.distance_to_price(current_price)
            age = nearest_gap.age_days(current_date)
            
            if distance > 0 and age > 0:
                fill_prob = 1.0 / (1.0 + (age * distance / 100.0))
            else:
                fill_prob = 1.0  # Price is at the gap
            
            if distance > 0:
                grav_pull = min(1.0, 100.0 / (distance ** 2))
            else:
                grav_pull = 1.0
            
            type_multipliers = {
                GapType.EXHAUSTION: 1.5,   # Highest pull
                GapType.BREAKAWAY: 1.3,
                GapType.MEASURING: 1.1,
                GapType.COMMON: 0.8        # Lowest pull
            }
            grav_pull *= type_multipliers[nearest_gap.gap_type]
            grav_pull = min(1.0, grav_pull)
        
        target_str = f"${target:.2f}" if target is not None else "N/A"
        nearest_str = f"${nearest_gap.target_level:.2f}" if nearest_gap else "N/A"
        gap_type_str = nearest_gap.gap_type.value if nearest_gap else "N/A"
        gap_age_str = f"{nearest_gap.age_days(current_date):.1f} days" if nearest_gap else "N/A"
        distance_str = f"${distance:.2f}" if nearest_gap else "N/A"
        
        details = (
            f"Gap Analysis Summary:\n"
            f"  Total Gaps Detected: {total_gaps}\n"
            f"  Unfilled Gaps: {unfilled_count}\n"
            f"  Target Price: {target_str}\n"
            f"  Nearest Gap: {nearest_str} ({gap_type_str})\n"
            f"  Gap Age: {gap_age_str}\n"
            f"  Distance to Gap: {distance_str}\n"
            f"  Fill Probability: {fill_prob:.1%}\n"
            f"  Gravitational Pull: {grav_pull:.2f}"
        )
        
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
        gap_list = gaps if gaps is not None else self.gaps
        check_date = current_date if current_date is not None else datetime.now()
        
        return [
            gap for gap in gap_list
            if not gap.filled and gap.age_days(check_date) <= self.config.max_gap_age_days
        ]
    
    def get_gap_by_type(self, gap_type: GapType, gaps: Optional[List[Gap]] = None) -> List[Gap]:
        gap_list = gaps if gaps is not None else self.gaps
        return [gap for gap in gap_list if gap.gap_type == gap_type]
