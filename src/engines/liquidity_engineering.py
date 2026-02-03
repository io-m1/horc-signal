#!/usr/bin/env python3
"""
Liquidity Engineering Module

Based on iSpeculatefx Journal concepts:
- Area of Liquidity (AOL) Types 1, 2, 3
- Liquidity Engineering (ELQ) - swept liquidity or inducement
- VV Analysis (Validation-Violation)
- Single Candle Zones

These concepts enhance HORC by providing:
1. Better signal filtering (AOL type classification)
2. Precise entry timing (wait for return to ELQ)
3. Clearer structure identification (VV Analysis)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
from datetime import datetime


@dataclass
class Candle:
    """Basic candle representation."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    @property
    def range(self) -> float:
        return self.high - self.low


class AOLType(Enum):
    """Area of Liquidity Types."""
    NONE = 0
    TYPE_1_SWEEP_ENGULF = 1      # Sweep + engulfing (highest probability)
    TYPE_2_REJECTION_AOL = 2     # Rejection at KL + Type 1
    TYPE_3_SAME_COLOR = 3        # Same-color engulfing (continuation)


class Direction(Enum):
    """Trade direction."""
    BULLISH = 1
    BEARISH = -1
    NEUTRAL = 0


@dataclass
class AOLResult:
    """Result of AOL detection."""
    aol_type: AOLType
    direction: Direction
    entry_zone_high: float
    entry_zone_low: float
    swept_liquidity: bool
    confidence: float  # 0.0 to 1.0
    
    @property
    def is_valid(self) -> bool:
        return self.aol_type != AOLType.NONE


@dataclass
class LiquidityEngineering:
    """
    A swing point that has been used to mitigate into a level,
    making it a valid entry point when price returns.
    
    The ELQ (Engineering Liquidity) is your POI (Point of Interest).
    """
    price: float
    swing_type: str  # 'HIGH' or 'LOW'
    timestamp: datetime
    used_for_mitigation: bool  # Was it used to tag a level?
    broke_structure: bool      # Did it break structure? (REQUIRED)
    candle_index: int
    
    @property
    def is_valid_entry(self) -> bool:
        """ELQ must have broken structure to be valid."""
        return self.used_for_mitigation and self.broke_structure


@dataclass
class VVAnalysis:
    """
    Validation-Violation Analysis.
    
    Validation = The point of structure break (BOS)
    Violation = The swing that led to the BOS
    """
    validation_price: float      # BOS level
    violation_price: float       # Swing that caused BOS
    direction: Direction
    is_liquidation_engulf: bool  # Did engulfing sweep wicks?
    timestamp: datetime


@dataclass
class SingleCandleZone:
    """
    Zone created by single candle rejection.
    The opening price of the rejected candle = future key level.
    """
    level: float  # The opening price of rejection candle
    zone_type: str  # 'SUPPORT' or 'RESISTANCE'
    rejection_candle_index: int
    rejection_strength: float  # Wick to body ratio


class LiquidityEngineeringEngine:
    """
    Engine for detecting liquidity engineering setups.
    
    Trading Model: KL → BOS (body) → AOL → ELQ
    
    Entry: At the liquidity engineering within the AOL
    Stop: Above/below the AOL
    Target: Previous structure or opposing key level
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_wick_body_ratio = self.config.get('min_wick_body_ratio', 2.0)
        self.engulf_threshold = self.config.get('engulf_threshold', 0.5)
        self.structure_lookback = self.config.get('structure_lookback', 20)
        
        # State
        self.swing_highs: List[Tuple[int, float]] = []
        self.swing_lows: List[Tuple[int, float]] = []
        self.bos_events: List[Tuple[int, Direction, float]] = []
        self.elq_points: List[LiquidityEngineering] = []
    
    def detect_aol_type(
        self, 
        candles: List[Candle], 
        key_level: Optional[float] = None
    ) -> AOLResult:
        """
        Detect Area of Liquidity type from candle pattern.
        
        Type 1: Engulfing that sweeps the prior candle's wick
        Type 2: Rejection at key_level followed by Type 1
        Type 3: Same-color engulfing
        """
        if len(candles) < 2:
            return AOLResult(AOLType.NONE, Direction.NEUTRAL, 0, 0, False, 0.0)
        
        prev, curr = candles[-2], candles[-1]
        
        # Check for bullish engulfing
        is_bullish_engulf = (
            curr.is_bullish and 
            curr.close > prev.open and 
            curr.open <= prev.close
        )
        
        # Check for bearish engulfing
        is_bearish_engulf = (
            curr.is_bearish and 
            curr.close < prev.open and 
            curr.open >= prev.close
        )
        
        if is_bullish_engulf:
            # Did it sweep the prior low? (Liquidation)
            swept_low = curr.low < prev.low
            same_color = prev.is_bullish
            
            entry_zone = (prev.open, prev.close) if prev.is_bearish else (prev.close, prev.open)
            
            if swept_low:
                # Check proximity to key level for Type 2
                if key_level and abs(curr.low - key_level) / key_level < 0.002:
                    return AOLResult(
                        AOLType.TYPE_2_REJECTION_AOL, Direction.BULLISH,
                        max(entry_zone), min(entry_zone), True, 0.9
                    )
                return AOLResult(
                    AOLType.TYPE_1_SWEEP_ENGULF, Direction.BULLISH,
                    max(entry_zone), min(entry_zone), True, 0.85
                )
            elif same_color:
                return AOLResult(
                    AOLType.TYPE_3_SAME_COLOR, Direction.BULLISH,
                    max(entry_zone), min(entry_zone), False, 0.6
                )
        
        if is_bearish_engulf:
            # Did it sweep the prior high? (Liquidation)
            swept_high = curr.high > prev.high
            same_color = prev.is_bearish
            
            entry_zone = (prev.close, prev.open) if prev.is_bullish else (prev.open, prev.close)
            
            if swept_high:
                if key_level and abs(curr.high - key_level) / key_level < 0.002:
                    return AOLResult(
                        AOLType.TYPE_2_REJECTION_AOL, Direction.BEARISH,
                        max(entry_zone), min(entry_zone), True, 0.9
                    )
                return AOLResult(
                    AOLType.TYPE_1_SWEEP_ENGULF, Direction.BEARISH,
                    max(entry_zone), min(entry_zone), True, 0.85
                )
            elif same_color:
                return AOLResult(
                    AOLType.TYPE_3_SAME_COLOR, Direction.BEARISH,
                    max(entry_zone), min(entry_zone), False, 0.6
                )
        
        return AOLResult(AOLType.NONE, Direction.NEUTRAL, 0, 0, False, 0.0)
    
    def detect_bos(
        self, 
        candles: List[Candle], 
        require_body_break: bool = True
    ) -> Optional[Tuple[int, Direction, float]]:
        """
        Detect Break of Structure.
        
        BOS must be with body (not just wick) for validity.
        Returns: (index, direction, level)
        """
        if len(candles) < self.structure_lookback:
            return None
        
        # Find recent swing points
        recent = candles[-self.structure_lookback:]
        
        # Simple swing detection
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent) - 2):
            # Swing high
            if (recent[i].high > recent[i-1].high and 
                recent[i].high > recent[i-2].high and
                recent[i].high > recent[i+1].high and 
                recent[i].high > recent[i+2].high):
                swing_highs.append((i, recent[i].high))
            
            # Swing low
            if (recent[i].low < recent[i-1].low and 
                recent[i].low < recent[i-2].low and
                recent[i].low < recent[i+1].low and 
                recent[i].low < recent[i+2].low):
                swing_lows.append((i, recent[i].low))
        
        if not swing_highs and not swing_lows:
            return None
        
        curr = recent[-1]
        
        # Check for bullish BOS (breaking above swing high)
        if swing_highs:
            last_high_idx, last_high = swing_highs[-1]
            if require_body_break:
                if curr.close > last_high:  # Body above
                    return (len(candles) - 1, Direction.BULLISH, last_high)
            else:
                if curr.high > last_high:
                    return (len(candles) - 1, Direction.BULLISH, last_high)
        
        # Check for bearish BOS (breaking below swing low)
        if swing_lows:
            last_low_idx, last_low = swing_lows[-1]
            if require_body_break:
                if curr.close < last_low:  # Body below
                    return (len(candles) - 1, Direction.BEARISH, last_low)
            else:
                if curr.low < last_low:
                    return (len(candles) - 1, Direction.BEARISH, last_low)
        
        return None
    
    def find_engineering_liquidity(
        self,
        candles: List[Candle],
        bos_index: int,
        direction: Direction
    ) -> Optional[LiquidityEngineering]:
        """
        After BOS, find the liquidity engineering point:
        1. The actual swept liquidity before BOS, OR
        2. The first pullback (inducement) after BOS
        
        This becomes the POI for entry.
        
        CRITICAL: The ELQ picked must have broken structure to be valid.
        """
        if bos_index < 5:
            return None
        
        lookback = candles[max(0, bos_index - 15):bos_index]
        
        if direction == Direction.BULLISH:
            # For bullish BOS, look for the swing low that was swept
            # This is the liquidity that fueled the move up
            swing_lows = []
            for i in range(1, len(lookback) - 1):
                if (lookback[i].low < lookback[i-1].low and 
                    lookback[i].low < lookback[i+1].low):
                    swing_lows.append((i, lookback[i]))
            
            if swing_lows:
                # The most recent swing low before BOS is the ELQ
                idx, candle = swing_lows[-1]
                
                # Check if this swing was swept (used for mitigation)
                was_swept = any(c.low < candle.low for c in lookback[idx+1:])
                
                return LiquidityEngineering(
                    price=candle.low,
                    swing_type='LOW',
                    timestamp=candle.timestamp,
                    used_for_mitigation=was_swept,
                    broke_structure=True,  # BOS already confirmed
                    candle_index=bos_index - len(lookback) + idx
                )
        
        elif direction == Direction.BEARISH:
            # For bearish BOS, look for swing high that was swept
            swing_highs = []
            for i in range(1, len(lookback) - 1):
                if (lookback[i].high > lookback[i-1].high and 
                    lookback[i].high > lookback[i+1].high):
                    swing_highs.append((i, lookback[i]))
            
            if swing_highs:
                idx, candle = swing_highs[-1]
                was_swept = any(c.high > candle.high for c in lookback[idx+1:])
                
                return LiquidityEngineering(
                    price=candle.high,
                    swing_type='HIGH',
                    timestamp=candle.timestamp,
                    used_for_mitigation=was_swept,
                    broke_structure=True,
                    candle_index=bos_index - len(lookback) + idx
                )
        
        return None
    
    def analyze_vv(
        self, 
        candles: List[Candle], 
        bos_index: int,
        direction: Direction
    ) -> Optional[VVAnalysis]:
        """
        Validation-Violation Analysis.
        
        Validation = BOS level
        Violation = The swing that led to the BOS
        
        Best used when engulfing did NOT sweep engulfed wicks.
        """
        if bos_index < 3:
            return None
        
        bos_candle = candles[bos_index]
        lookback = candles[max(0, bos_index - 10):bos_index]
        
        if direction == Direction.BULLISH:
            validation = bos_candle.close  # Body break level
            
            # Find the swing low that led to this BOS (violation)
            lowest = min(lookback, key=lambda c: c.low)
            violation = lowest.low
            
            # Check if the BOS candle is a liquidation engulf
            if bos_index > 0:
                prev = candles[bos_index - 1]
                is_liquidation = bos_candle.low < prev.low
            else:
                is_liquidation = False
            
            return VVAnalysis(
                validation_price=validation,
                violation_price=violation,
                direction=direction,
                is_liquidation_engulf=is_liquidation,
                timestamp=bos_candle.timestamp
            )
        
        elif direction == Direction.BEARISH:
            validation = bos_candle.close
            highest = max(lookback, key=lambda c: c.high)
            violation = highest.high
            
            if bos_index > 0:
                prev = candles[bos_index - 1]
                is_liquidation = bos_candle.high > prev.high
            else:
                is_liquidation = False
            
            return VVAnalysis(
                validation_price=validation,
                violation_price=violation,
                direction=direction,
                is_liquidation_engulf=is_liquidation,
                timestamp=bos_candle.timestamp
            )
        
        return None
    
    def find_single_candle_zone(
        self, 
        candles: List[Candle],
        lookback: int = 50
    ) -> List[SingleCandleZone]:
        """
        Find zones created by single candle rejection.
        
        The opening price of the rejected candle = future key level.
        For multiple rejections at same level, use FIRST rejection's open.
        """
        zones = []
        recent = candles[-lookback:] if len(candles) > lookback else candles
        
        for i, c in enumerate(recent):
            if c.body_size == 0:
                continue
            
            wick_body_ratio_upper = c.upper_wick / c.body_size if c.body_size > 0 else 0
            wick_body_ratio_lower = c.lower_wick / c.body_size if c.body_size > 0 else 0
            
            # Bullish rejection (long lower wick = support)
            if wick_body_ratio_lower >= self.min_wick_body_ratio:
                zones.append(SingleCandleZone(
                    level=c.open,
                    zone_type='SUPPORT',
                    rejection_candle_index=i,
                    rejection_strength=wick_body_ratio_lower
                ))
            
            # Bearish rejection (long upper wick = resistance)
            elif wick_body_ratio_upper >= self.min_wick_body_ratio:
                zones.append(SingleCandleZone(
                    level=c.open,
                    zone_type='RESISTANCE',
                    rejection_candle_index=i,
                    rejection_strength=wick_body_ratio_upper
                ))
        
        # Deduplicate by level (keep first)
        seen_levels = set()
        unique_zones = []
        for zone in zones:
            level_key = round(zone.level, 5)
            if level_key not in seen_levels:
                seen_levels.add(level_key)
                unique_zones.append(zone)
        
        return unique_zones
    
    def get_complete_setup(
        self,
        candles: List[Candle],
        htf_key_level: Optional[float] = None
    ) -> Optional[dict]:
        """
        Complete setup detection: KL → BOS → AOL → ELQ
        
        Returns setup dict if valid, None otherwise.
        """
        if len(candles) < 20:
            return None
        
        # 1. Detect BOS with body break
        bos = self.detect_bos(candles, require_body_break=True)
        if not bos:
            return None
        
        bos_index, direction, bos_level = bos
        
        # 2. Detect AOL Type
        aol = self.detect_aol_type(candles[-5:], htf_key_level)
        
        # 3. Find Engineering Liquidity
        elq = self.find_engineering_liquidity(candles, bos_index, direction)
        
        # 4. VV Analysis
        vv = self.analyze_vv(candles, bos_index, direction)
        
        # 5. Check for single candle zones as key levels
        zones = self.find_single_candle_zone(candles)
        
        # Validate setup
        if not elq or not elq.is_valid_entry:
            return None
        
        # Build setup
        setup = {
            'direction': direction,
            'bos_level': bos_level,
            'bos_index': bos_index,
            'aol_type': aol.aol_type,
            'aol_confidence': aol.confidence,
            'entry_zone': (aol.entry_zone_low, aol.entry_zone_high) if aol.is_valid else None,
            'elq_price': elq.price,
            'elq_valid': elq.is_valid_entry,
            'vv_validation': vv.validation_price if vv else None,
            'vv_violation': vv.violation_price if vv else None,
            'single_candle_zones': [(z.level, z.zone_type) for z in zones],
            'timestamp': candles[-1].timestamp
        }
        
        # Calculate confidence score
        confidence = 0.0
        if aol.aol_type == AOLType.TYPE_2_REJECTION_AOL:
            confidence += 0.4
        elif aol.aol_type == AOLType.TYPE_1_SWEEP_ENGULF:
            confidence += 0.3
        elif aol.aol_type == AOLType.TYPE_3_SAME_COLOR:
            confidence += 0.15
        
        if elq.is_valid_entry:
            confidence += 0.3
        
        if vv and not vv.is_liquidation_engulf:
            confidence += 0.2  # VV is best for non-liquidation engulfs
        
        if htf_key_level:
            confidence += 0.1  # HTF confluence
        
        setup['confidence'] = min(confidence, 1.0)
        
        return setup


# Convenience function for integration with HORC
def enhance_horc_signal(
    horc_signal: dict,
    candles: List[Candle],
    htf_key_level: Optional[float] = None
) -> dict:
    """
    Enhance a HORC signal with liquidity engineering analysis.
    
    This adds:
    - AOL type classification
    - ELQ entry point
    - VV analysis
    - Enhanced confidence score
    """
    engine = LiquidityEngineeringEngine()
    setup = engine.get_complete_setup(candles, htf_key_level)
    
    if setup:
        horc_signal['liquidity_engineering'] = {
            'elq_price': setup['elq_price'],
            'aol_type': setup['aol_type'].name,
            'vv_validation': setup['vv_validation'],
            'vv_violation': setup['vv_violation'],
            'le_confidence': setup['confidence']
        }
        
        # Adjust HORC confidence based on LE analysis
        base_confidence = horc_signal.get('confidence', 0.5)
        le_boost = setup['confidence'] * 0.3  # LE can boost up to 30%
        horc_signal['enhanced_confidence'] = min(base_confidence + le_boost, 1.0)
        
        # Add entry refinement
        if setup['direction'].value == horc_signal.get('bias', 0):
            horc_signal['refined_entry'] = setup['elq_price']
            horc_signal['entry_type'] = 'ELQ_RETURN'  # Wait for return to ELQ
        else:
            horc_signal['refined_entry'] = None
            horc_signal['entry_type'] = 'CONFLICT'  # HORC and LE disagree
    
    return horc_signal


if __name__ == "__main__":
    # Simple test
    from datetime import datetime, timedelta
    
    # Create test candles simulating a bullish setup
    base_time = datetime.now()
    candles = [
        Candle(base_time + timedelta(minutes=i*5), 
               open=100 + i*0.1, 
               high=100 + i*0.1 + 0.15,
               low=100 + i*0.1 - 0.05,
               close=100 + i*0.1 + 0.1,
               volume=1000)
        for i in range(20)
    ]
    
    # Add a sweep + engulfing pattern at the end
    candles[-2] = Candle(
        candles[-2].timestamp,
        open=102.0, high=102.1, low=101.5, close=101.6, volume=1500
    )
    candles[-1] = Candle(
        candles[-1].timestamp,
        open=101.5, high=102.5, low=101.4, close=102.4, volume=2000  # Sweeps low, engulfs
    )
    
    engine = LiquidityEngineeringEngine()
    
    # Test AOL detection
    aol = engine.detect_aol_type(candles)
    print(f"AOL Type: {aol.aol_type.name}")
    print(f"Direction: {aol.direction.name}")
    print(f"Swept Liquidity: {aol.swept_liquidity}")
    print(f"Confidence: {aol.confidence}")
    
    # Test single candle zones
    zones = engine.find_single_candle_zone(candles)
    print(f"\nSingle Candle Zones: {len(zones)}")
    for z in zones:
        print(f"  {z.zone_type} at {z.level:.4f} (strength: {z.rejection_strength:.2f})")
