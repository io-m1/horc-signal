from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

from src.engines import Candle
from src.core import HORCOrchestrator
from src.core.strategic_context import LiquidityIntent, MarketControlState, LIQUIDITY_DIRECTION
from src.core.opposition import SignalState, PeriodType

@dataclass
class SessionBoundary:
    timestamp: datetime
    session_type: str  # "DAILY", "WEEKLY"
    prev_high: float
    prev_low: float
    prev_close: float
    open_price: float

class SessionManager:
    def __init__(self, orchestrator: HORCOrchestrator):
        self.orchestrator = orchestrator
        
        self.current_session_high = None
        self.current_session_low = None
        self.session_open_time = None
        self.prev_day_high = None
        self.prev_day_low = None
        self.prev_day_close = None
        
        self.bars: List[Candle] = []
        self.max_history = 200  # Keep last 200 bars
        
    def is_new_session(self, current: datetime, previous: Optional[datetime]) -> str:
        if previous is None:
            return "DAILY"  # First bar starts a session
        
        if current.date() != previous.date():
            if current.weekday() == 0 and previous.weekday() != 0:
                return "WEEKLY"
            return "DAILY"
        
        return ""
    
    def detect_market_control(self, lookback: int = 50) -> MarketControlState:
        if len(self.bars) < lookback:
            return MarketControlState(
                passive=0,
                aggressor=0,
                control=0,
                conclusive=False
            )
        
        recent = self.bars[-lookback:]
        closes = [b.close for b in recent]
        
        sma = sum(closes) / len(closes)
        current_price = closes[-1]
        
        trend_strength = (current_price - sma) / sma
        
        up_bars = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        up_ratio = up_bars / (len(closes) - 1)
        
        if trend_strength > 0.002 and up_ratio > 0.55:
            return MarketControlState(
                passive=-1,  # Sellers defending
                aggressor=1,  # Buyers attacking
                control=1,  # BUYERS_IN_CONTROL
                conclusive=True
            )
        elif trend_strength < -0.002 and up_ratio < 0.45:
            return MarketControlState(
                passive=1,  # Buyers defending
                aggressor=-1,  # Sellers attacking
                control=-1,  # SELLERS_IN_CONTROL
                conclusive=True
            )
        else:
            return MarketControlState(
                passive=0,
                aggressor=0,
                control=0,
                conclusive=False
            )
    
    def identify_liquidity_intent(self, candle: Candle) -> LiquidityIntent:
        if self.current_session_high is None or self.current_session_low is None:
            return LiquidityIntent.null()
        
        if candle.high > self.current_session_high:
            self.current_session_high = candle.high
            return LiquidityIntent.from_level(
                level=candle.high,
                direction=-1,  # SELL_SIDE
                timeframe="15T",
                current_price=candle.close,
                atr=abs(candle.high - candle.low)
            )
        elif candle.low < self.current_session_low:
            self.current_session_low = candle.low
            return LiquidityIntent.from_level(
                level=candle.low,
                direction=1,  # BUY_SIDE
                timeframe="15T",
                current_price=candle.close,
                atr=abs(candle.high - candle.low)
            )
        
        return LiquidityIntent.null()
    
    def setup_strategic_context(self, candle: Candle):
        control = self.detect_market_control()
        
        liquidity = self.identify_liquidity_intent(candle)
        
        self.orchestrator.set_strategic_context(liquidity, control)
    
    def update_opposition_state(
        self,
        boundary: SessionBoundary,
        prev_close_signal: SignalState = SignalState.INCONCLUSIVE
    ):
        period_type = PeriodType.DAILY if boundary.session_type == "DAILY" else PeriodType.WEEKLY
        
        self.orchestrator.update_opposition(
            period_type=period_type,
            current_open=boundary.open_price,
            prev_close_high=boundary.prev_high,
            prev_close_low=boundary.prev_low,
            prev_close_signal=prev_close_signal,
            timestamp=int(boundary.timestamp.timestamp() * 1000),
        )
    
    def process_bar(self, candle: Candle, prev_candle: Optional[Candle]) -> None:
        self.bars.append(candle)
        if len(self.bars) > self.max_history:
            self.bars.pop(0)
        
        prev_time = prev_candle.timestamp if prev_candle else None
        boundary_type = self.is_new_session(candle.timestamp, prev_time)
        
        if boundary_type:
            
            if prev_candle and self.prev_day_high and self.prev_day_low:
                boundary = SessionBoundary(
                    timestamp=candle.timestamp,
                    session_type=boundary_type,
                    prev_high=self.prev_day_high,
                    prev_low=self.prev_day_low,
                    prev_close=self.prev_day_close or prev_candle.close,
                    open_price=candle.open
                )
                
                self.update_opposition_state(boundary)
            
            if self.current_session_high and self.current_session_low:
                self.prev_day_high = self.current_session_high
                self.prev_day_low = self.current_session_low
                self.prev_day_close = self.bars[-2].close if len(self.bars) >= 2 else candle.open
            
            self.current_session_high = candle.high
            self.current_session_low = candle.low
            self.session_open_time = candle.timestamp
            
            self.setup_strategic_context(candle)
        
        else:
            if self.current_session_high is None:
                self.current_session_high = candle.high
            else:
                self.current_session_high = max(self.current_session_high, candle.high)
            
            if self.current_session_low is None:
                self.current_session_low = candle.low
            else:
                self.current_session_low = min(self.current_session_low, candle.low)
            
            liquidity = self.identify_liquidity_intent(candle)
            if liquidity.valid:  # Changed from != UNKNOWN
                control = self.detect_market_control()
                self.orchestrator.set_strategic_context(liquidity, control)

def create_managed_orchestrator(conf_thresh: float = 0.55) -> Tuple[HORCOrchestrator, SessionManager]:
    from run_validation import create_default_orchestrator
    
    orchestrator = create_default_orchestrator(conf_thresh=conf_thresh)
    manager = SessionManager(orchestrator)
    
    return orchestrator, manager
