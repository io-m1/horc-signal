from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math

class Participant(IntEnum):
    NEUTRAL = 0
    BUYER = 1   # D+ / W+ / M+
    SELLER = -1 # D- / W- / M-

@dataclass
class Coordinate:
    """
    Represents a significant High or Low in the HORC system,
    tagged with its dominant Participant and Timeframe.
    """
    time: datetime
    price: float
    participant: Participant
    timeframe: str  # e.g., 'D', 'W', 'M'
    is_liquidity: bool = False  # True if Highest Conclusive TF says so
    is_imbalance: bool = False  # True if Lower Conclusive TF says so

@dataclass
class PeriodState:
    """Tracks the state of a single binary period (e.g., a specific Day)."""
    start_time: datetime
    end_time: Optional[datetime]
    
    # Opening Range (First Candle)
    open_price: float = 0.0
    open_range_high: float = -1.0
    open_range_low: float = float('inf')
    
    # Closing Range (Last Candle of Prev Period) - passed in
    prev_close_signal: Participant = Participant.NEUTRAL
    
    # Current Signal Status
    first_raid: Participant = Participant.NEUTRAL  # The "Open" signal (D+ or D-)
    current_signal: Participant = Participant.NEUTRAL # The live running signal
    is_conclusive: bool = False  # True if First Raid == Opposite of Prev Close
    
    # Extremes for the period
    period_high: float = -1.0
    period_low: float = float('inf')

class OppositionRule:
    """
    Implements the Binary Logic for Signal Validation.
    Rule: New Period Signal must OPEN Opposite to Previous Period Close to be Conclusive.
    """
    @staticmethod
    def is_conclusive(prev_close: Participant, new_open: Participant) -> bool:
        if prev_close == Participant.NEUTRAL:
            return True # No history, assume conclusive
        if new_open == Participant.NEUTRAL:
            return False
        return prev_close != new_open

class CoordinateTracker:
    """
    Engine for tracking HORC Coordinates across timeframes.
    Manages PeriodStates and applies Logic Layering.
    """
    def __init__(self):
        # Map timeframe -> List[Coordinate]
        self.coordinates: Dict[str, List[Coordinate]] = {}
        # Map timeframe -> Current PeriodState
        self.active_periods: Dict[str, PeriodState] = {}
        # History of PeriodStates for lookback (e.g., getting Prev Close)
        self.period_history: Dict[str, List[PeriodState]] = {}

    def on_bar(self, timeframe: str, time: datetime, open_: float, high: float, low: float, close: float, is_new_period: bool):
        """
        Process a bar for a specific timeframe (e.g. 'D', 'W').
        Must be called with the *Chart* resolution bar, but triggered on TF boundaries.
        Argument 'is_new_period' is True when this bar represents the start of a new TF period.
        """
        if timeframe not in self.coordinates:
            self.coordinates[timeframe] = []
            self.period_history[timeframe] = []

        # Handle Period Transition
        if is_new_period:
            self._finalize_period(timeframe)
            self._start_new_period(timeframe, time, open_)

        period = self.active_periods.get(timeframe)
        if not period:
            return

        # Update Extremes
        period.period_high = max(period.period_high, high)
        period.period_low = min(period.period_low, low)

        # Logic 1.1: Detect First Raid (Open Signal)
        # Using Closing Range Logic (CRL) implies referencing Prev Close High/Low.
        # Here we simplify to: Did we break the Opening Range High/Low first?
        # NOTE: Full CRL requires access to the PREVIOUS period's High/Low.
        
        # We need to refine this. The "First Raid" determines the Open Signal.
        # If signal not set, check raids.
        if period.first_raid == Participant.NEUTRAL:
            # Check against Opening Range (First Candle) or Prev Close Range (CRL)
            # Defaulting to OPL (Open Price Logic) for simplicity if CRL data missing,
            # but aiming for CRL if we have History.
            
            # Simple Logic: First break of Open Range High/Low
            if high > period.open_range_high:
                period.first_raid = Participant.BUYER
                period.current_signal = Participant.BUYER
                # Apply Opposition Rule
                period.is_conclusive = OppositionRule.is_conclusive(period.prev_close_signal, Participant.BUYER)
            elif low < period.open_range_low:
                period.first_raid = Participant.SELLER
                period.current_signal = Participant.SELLER
                period.is_conclusive = OppositionRule.is_conclusive(period.prev_close_signal, Participant.SELLER)

        # Logic 2.2: Flipping (Intra-Period)
        # If cycle complete (Target Met) + Opposite Raid -> Flip
        # (Simplified flip logic: if we hold Buyer, but break significant low, flip)
        # For v8.8 MVP, we stick to the primary trend.

        # Update Open Range if this is the first candle
        if period.open_range_high == -1.0:
            period.open_range_high = high
            period.open_range_low = low

    def _start_new_period(self, timeframe: str, time: datetime, open_price: float):
        prev_signal = Participant.NEUTRAL
        if self.period_history[timeframe]:
            prev_signal = self.period_history[timeframe][-1].current_signal

        new_period = PeriodState(
            start_time=time,
            end_time=None,
            open_price=open_price,
            prev_close_signal=prev_signal
        )
        self.active_periods[timeframe] = new_period

    def _finalize_period(self, timeframe: str):
        period = self.active_periods.get(timeframe)
        if period:
            period.end_time = datetime.now() # Should use actual close time
            self.period_history[timeframe].append(period)
            
            # Record Coordinate (Peak/Trough)
            # If Buyer dominant, the High is the coordinate.
            # If Seller dominant, the Low is the coordinate.
            if period.current_signal == Participant.BUYER:
                 # Buyers defend the Low (Origin)
                 coord = Coordinate(
                     time=period.start_time,
                     price=period.period_low,
                     participant=Participant.BUYER,
                     timeframe=timeframe,
                     is_liquidity=period.is_conclusive
                 )
                 self.coordinates[timeframe].append(coord)
            elif period.current_signal == Participant.SELLER:
                 # Sellers defend the High (Origin)
                 coord = Coordinate(
                     time=period.start_time,
                     price=period.period_high,
                     participant=Participant.SELLER,
                     timeframe=timeframe,
                     is_liquidity=period.is_conclusive
                 )
                 self.coordinates[timeframe].append(coord)

    def find_premium_liquidity(self, timeframe: str) -> Optional[Coordinate]:
        """
        Finds the 'Premium Liquidity' in the current range context.
        Definition: The *First* liquidity formed (Origin) of the current major move.
        """
        coords = self.coordinates.get(timeframe, [])
        if not coords:
            return None
        
        # Simple Logic: The last conclusive liquidity establishes the current 'Range' origin.
        # If we are in a selling leg (current signal Seller), the origin was the last Buyer liquidity?
        # No, the user says: "Range starts at D- (Seller). That D- is the Premium Liquidity."
        # This implies the Premium Liquidity is the coordinate that *started* the current dominant trend.
        
        # Iterate backwards to find the last Conclusive Liquidity.
        for coord in reversed(coords):
            if coord.is_liquidity:
                return coord
        return None

class RangeType(IntEnum):
    UNDEFINED = 0
    PREMIUM = 1
    DISCOUNT = 2
    EQUILIBRIUM = 3

@dataclass
class RangeContext:
    origin_coord: Optional[Coordinate]
    range_high: float
    range_low: float
    current_price: float
    range_type: RangeType = RangeType.UNDEFINED

class RangeAnalysis:
    """
    Analyzes price position within the active range defined by Coordinates.
    """
    @staticmethod
    def analyze(tracker: CoordinateTracker, timeframe: str, current_price: float) -> RangeContext:
        origin = tracker.find_premium_liquidity(timeframe)
        if not origin:
            return RangeContext(None, current_price, current_price, current_price)
        
        # Define Range High/Low based on Origin + Current Price extension
        # If Origin is SELLER (High), Range is [Current Low, Origin High]
        # If Origin is BUYER (Low), Range is [Origin Low, Current High]
        
        # We need the extrema since the origin.
        # Ideally tracker keeps track of "Current Swing High/Low".
        # For now, we approximate using the period's current high/low.
        
        period = tracker.active_periods.get(timeframe)
        period_high = period.period_high if period else current_price
        period_low = period.period_low if period else current_price

        r_high = period_high
        r_low = period_low

        if origin.participant == Participant.SELLER:
            r_high = origin.price
            # Range Low is the lowest low since then.
            r_low = min(r_low, current_price) 
        elif origin.participant == Participant.BUYER:
            r_low = origin.price
            r_high = max(r_high, current_price)

        # Calculate Position
        if r_high == r_low:
             return RangeContext(origin, r_high, r_low, current_price, RangeType.EQUILIBRIUM)

        midpoint = (r_high + r_low) / 2
        
        # Premium > 50%, Discount < 50%
        # But interpretation depends on Direction.
        # If Selling (Origin=Seller), we want to sell in Premium.
        # If Buying (Origin=Buyer), we want to buy in Discount.
        
        # Generic classification:
        # Upper half = Premium
        # Lower half = Discount
        
        r_type = RangeType.PREMIUM if current_price > midpoint else RangeType.DISCOUNT
        
        return RangeContext(origin, r_high, r_low, current_price, r_type)
