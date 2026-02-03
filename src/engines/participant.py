from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

class ParticipantType(Enum):
    BUYERS = "BUYERS"
    SELLERS = "SELLERS"
    NONE = "NONE"

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __post_init__(self):
        if self.high < max(self.open, self.close):
            raise ValueError(f"High {self.high} cannot be less than max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError(f"Low {self.low} cannot be greater than min(open, close)")
        if self.volume < 0:
            raise ValueError(f"Volume {self.volume} cannot be negative")

@dataclass
class ParticipantResult:
    participant_type: ParticipantType
    conviction_level: bool
    control_price: Optional[float]
    timestamp: datetime
    orh_prev: float
    orl_prev: float
    sweep_candle_index: Optional[int]

class ParticipantIdentifier:
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
            
        self.or_lookback_sessions: int = config.get('or_lookback_sessions', 1)
        self.min_conviction_threshold: float = config.get('min_conviction_threshold', 0.8)
        self.max_first_move_candles: int = config.get('max_first_move_candles', 3)
        
        self.prev_session_candles: List[Candle] = []
        
    def get_opening_range(self, candles: List[Candle]) -> Tuple[float, float]:
        if not candles:
            raise ValueError("Cannot calculate opening range from empty candle list")
        
        orh = max(candle.high for candle in candles)
        orl = min(candle.low for candle in candles)
        
        return orh, orl
    
    def identify_participant(
        self, 
        candles: List[Candle], 
        orh_prev: float, 
        orl_prev: float
    ) -> Tuple[ParticipantType, bool, Optional[int]]:
        if not candles:
            return ParticipantType.NONE, False, None
        
        first_moves = candles[:self.max_first_move_candles]
        
        for idx, candle in enumerate(first_moves):
            if candle.low <= orl_prev:
                return ParticipantType.SELLERS, True, idx
            
            if candle.high >= orh_prev:
                return ParticipantType.BUYERS, True, idx
        
        return ParticipantType.NONE, False, None
    
    def identify(self, current_candles: List[Candle]) -> ParticipantResult:
        if not self.prev_session_candles:
            raise ValueError(
                "Previous session candles not set. "
                "Must set identifier.prev_session_candles before calling identify()"
            )
        
        if not current_candles:
            orh_prev, orl_prev = self.get_opening_range(self.prev_session_candles)
            return ParticipantResult(
                participant_type=ParticipantType.NONE,
                conviction_level=False,
                control_price=None,
                timestamp=datetime.now(),
                orh_prev=orh_prev,
                orl_prev=orl_prev,
                sweep_candle_index=None
            )
        
        orh_prev, orl_prev = self.get_opening_range(self.prev_session_candles)
        
        participant_type, conviction, sweep_idx = self.identify_participant(
            current_candles, orh_prev, orl_prev
        )
        
        control_price: Optional[float] = None
        if participant_type == ParticipantType.BUYERS:
            control_price = orh_prev  # Buyers swept this level
        elif participant_type == ParticipantType.SELLERS:
            control_price = orl_prev  # Sellers swept this level
        
        return ParticipantResult(
            participant_type=participant_type,
            conviction_level=conviction,
            control_price=control_price,
            timestamp=current_candles[0].timestamp,
            orh_prev=orh_prev,
            orl_prev=orl_prev,
            sweep_candle_index=sweep_idx
        )
    
    def update_session_data(self, new_session_candles: List[Candle]) -> None:
        self.prev_session_candles = new_session_candles.copy()
    
    def reset(self) -> None:
        self.prev_session_candles = []

def create_test_candles_sweep_high(orh: float) -> List[Candle]:
    base_time = datetime(2024, 1, 1, 9, 30)
    
    return [
        Candle(
            timestamp=base_time,
            open=orh - 5.0,
            high=orh + 2.0,  # Sweeps ORH
            low=orh - 6.0,
            close=orh - 3.0,
            volume=1000.0
        )
    ]

def create_test_candles_sweep_low(orl: float) -> List[Candle]:
    base_time = datetime(2024, 1, 1, 9, 30)
    
    return [
        Candle(
            timestamp=base_time,
            open=orl + 5.0,
            high=orl + 6.0,
            low=orl - 2.0,  # Sweeps ORL
            close=orl + 3.0,
            volume=1000.0
        )
    ]

def create_test_candles_no_sweep(orh: float, orl: float) -> List[Candle]:
    base_time = datetime(2024, 1, 1, 9, 30)
    midpoint = (orh + orl) / 2.0
    
    return [
        Candle(
            timestamp=base_time,
            open=midpoint,
            high=midpoint + 1.0,  # Stays below ORH
            low=midpoint - 1.0,   # Stays above ORL
            close=midpoint,
            volume=1000.0
        )
    ]
