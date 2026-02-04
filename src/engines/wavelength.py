    @staticmethod
    def apply_crl(prev_high: float, prev_low: float, new_open: float, prev_participant: ParticipantType) -> ParticipantType:
        if new_open > prev_high:
            return ParticipantType.BUYERS
        elif new_open < prev_low:
            return ParticipantType.SELLERS
        else:
            return prev_participant

    @staticmethod
    def label(participant: ParticipantType, open_val: float, close_val: float) -> str:
        sponsor = "B" if participant == ParticipantType.BUYERS else "S" if participant == ParticipantType.SELLERS else "?"
        charge = "+" if close_val > open_val else "-" if close_val < open_val else "0"
        return f"{sponsor}{charge}"

    @staticmethod
    def convergence(states: List[ParticipantType], session_state: ParticipantType) -> int:
        return sum(1 for s in states if s == session_state)

    @staticmethod
    def divergence(states: List[ParticipantType], session_state: ParticipantType) -> int:
        return sum(1 for s in states if s != session_state and s != ParticipantType.NONE)
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum

from .participant import Candle, ParticipantResult, ParticipantType

class WavelengthState(Enum):
    PRE_OR = "pre_opening_range"
    PARTICIPANT_ID = "participant_identified"
    MOVE_1 = "first_move_complete"
    MOVE_2 = "second_move_complete"
    FLIP_CONFIRMED = "flip_point_confirmed"
    MOVE_3 = "third_move_complete"
    COMPLETE = "signal_complete"
    FAILED = "signal_failed"

@dataclass
class WavelengthResult:
    state: WavelengthState
    moves_completed: int
    flip_point: Optional[float]
    move_1_extreme: Optional[float]
    move_2_extreme: Optional[float]
    signal_strength: float
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    participant_type: ParticipantType
    candles_in_current_move: int
    timestamp: datetime

@dataclass
class WavelengthConfig:
    min_move_1_size_atr: float = 0.5
    max_move_2_retracement: float = 0.786
    exhaustion_threshold: float = 0.70
    max_move_duration_candles: int = 50
    flip_confirmation_candles: int = 3
    use_simple_exhaustion: bool = True

class WavelengthEngine:
    def __init__(self, config: Optional[WavelengthConfig] = None):
        self.config = config or WavelengthConfig()
        
        self.state = WavelengthState.PRE_OR
        self.moves_completed = 0
        
        self.flip_point: Optional[float] = None
        self.move_1_extreme: Optional[float] = None
        self.move_2_extreme: Optional[float] = None
        self.move_1_start: Optional[float] = None
        
        self.participant_type = ParticipantType.NONE
        self.control_price: Optional[float] = None
        
        self.candles_in_current_move = 0
        self.flip_confirmation_count = 0
        
        self.candle_history: List[Candle] = []
        self.move_1_candles: List[Candle] = []
        self.move_2_candles: List[Candle] = []
        
    def reset(self) -> None:
        self.state = WavelengthState.PRE_OR
        self.moves_completed = 0
        self.flip_point = None
        self.move_1_extreme = None
        self.move_2_extreme = None
        self.move_1_start = None
        self.participant_type = ParticipantType.NONE
        self.control_price = None
        self.candles_in_current_move = 0
        self.flip_confirmation_count = 0
        self.candle_history = []
        self.move_1_candles = []
        self.move_2_candles = []
    
    def calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        if len(candles) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, min(len(candles), period + 1)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    
    def detect_move_1_completion(self, candle: Candle) -> bool:
        if self.control_price is None or self.move_1_start is None:
            return False
        
        if self.participant_type == ParticipantType.BUYERS:
            move_size = candle.high - self.move_1_start
            body_size = abs(candle.close - candle.open)
            upper_wick = candle.high - max(candle.open, candle.close)
            rejection = upper_wick > 2 * body_size if body_size > 0 else True
        else:  # SELLERS
            move_size = self.move_1_start - candle.low
            body_size = abs(candle.close - candle.open)
            lower_wick = min(candle.open, candle.close) - candle.low
            rejection = lower_wick > 2 * body_size if body_size > 0 else True
        
        atr = self.calculate_atr(self.candle_history) if self.candle_history else move_size * 0.5
        is_significant = move_size >= self.config.min_move_1_size_atr * atr
        
        return is_significant and rejection
    
    def detect_move_2_reversal(self, candle: Candle) -> bool:
        if self.move_1_extreme is None:
            return False
        
        if self.participant_type == ParticipantType.BUYERS:
            reversal = candle.close < candle.open  # Bearish candle
        else:  # SELLERS
            reversal = candle.close > candle.open  # Bullish candle
        
        return reversal
    
    def detect_exhaustion(self, candle: Candle) -> float:
        if not self.move_2_candles:
            return 0.0
        
        body_size = abs(candle.close - candle.open)
        candle_range = candle.high - candle.low
        
        volume_score = 0.0
        if len(self.move_2_candles) > 1:
            avg_volume = sum(c.volume for c in self.move_2_candles[-5:]) / len(self.move_2_candles[-5:])
            if avg_volume > 0:
                volume_score = min(candle.volume / avg_volume, 1.0) * (1.0 - body_size / candle_range if candle_range > 0 else 1.0)
        
        if self.participant_type == ParticipantType.BUYERS:
            lower_wick = min(candle.open, candle.close) - candle.low
            wick_score = lower_wick / candle_range if candle_range > 0 else 0.0
        else:
            upper_wick = candle.high - max(candle.open, candle.close)
            wick_score = upper_wick / candle_range if candle_range > 0 else 0.0
        
        stagnation_score = 0.0
        if len(self.move_2_candles) >= 3:
            recent_highs = [c.high for c in self.move_2_candles[-3:]]
            recent_lows = [c.low for c in self.move_2_candles[-3:]]
            price_range = max(recent_highs) - min(recent_lows)
            atr = self.calculate_atr(self.candle_history) if self.candle_history else 1.0
            stagnation_score = 1.0 - min(price_range / atr, 1.0) if atr > 0 else 0.0
        
        exhaustion_score = (
            0.30 * volume_score +
            0.30 * wick_score +
            0.40 * stagnation_score
        )
        
        return exhaustion_score
    
    def check_pattern_invalidation(self, candle: Candle) -> bool:
        if self.candles_in_current_move > self.config.max_move_duration_candles:
            return True
        
        if self.state == WavelengthState.MOVE_2:
            if self.move_1_extreme is None:
                return False
            
            if self.participant_type == ParticipantType.BUYERS:
                if self.move_1_start and candle.low < self.move_1_start:
                    return True
            else:  # SELLERS
                if self.move_1_start and candle.high > self.move_1_start:
                    return True
        
        elif self.state == WavelengthState.MOVE_3:
            if self.flip_point is None:
                return False
            
            if self.participant_type == ParticipantType.BUYERS:
                if candle.low < self.flip_point:
                    return True
            else:  # SELLERS
                if candle.high > self.flip_point:
                    return True
        
        return False
    
    def calculate_signal_strength(self) -> float:
        strength_map = {
            WavelengthState.PRE_OR: 0.0,
            WavelengthState.PARTICIPANT_ID: 0.2,
            WavelengthState.MOVE_1: 0.3,
            WavelengthState.MOVE_2: 0.5,
            WavelengthState.FLIP_CONFIRMED: 0.8,
            WavelengthState.MOVE_3: 0.9,
            WavelengthState.COMPLETE: 1.0,
            WavelengthState.FAILED: 0.0,
        }
        return strength_map.get(self.state, 0.0)
    
    def process_candle(
        self, 
        candle: Candle, 
        participant_result: Optional[ParticipantResult] = None
    ) -> WavelengthResult:
        self.candle_history.append(candle)
        self.candles_in_current_move += 1
        
        if self.state in [WavelengthState.MOVE_2, WavelengthState.MOVE_3]:
            if self.check_pattern_invalidation(candle):
                self.state = WavelengthState.FAILED
                return self._build_result(candle)
        
        if self.state == WavelengthState.PRE_OR:
            self._transition_pre_or(candle, participant_result)
            
        elif self.state == WavelengthState.PARTICIPANT_ID:
            self._transition_participant_id(candle)
            
        elif self.state == WavelengthState.MOVE_1:
            self._transition_move_1(candle)
            
        elif self.state == WavelengthState.MOVE_2:
            self._transition_move_2(candle)
            
        elif self.state == WavelengthState.FLIP_CONFIRMED:
            self._transition_flip_confirmed(candle)
            
        elif self.state == WavelengthState.MOVE_3:
            self._transition_move_3(candle)
        
        return self._build_result(candle)
    
    def _transition_pre_or(self, candle: Candle, participant_result: Optional[ParticipantResult]) -> None:
        if participant_result and participant_result.participant_type != ParticipantType.NONE:
            self.participant_type = participant_result.participant_type
            self.control_price = participant_result.control_price
            self.move_1_start = candle.close
            self.state = WavelengthState.PARTICIPANT_ID
            self.candles_in_current_move = 0
    
    def _transition_participant_id(self, candle: Candle) -> None:
        if self.participant_type == ParticipantType.BUYERS:
            if self.move_1_extreme is None or candle.high > self.move_1_extreme:
                self.move_1_extreme = candle.high
        else:  # SELLERS
            if self.move_1_extreme is None or candle.low < self.move_1_extreme:
                self.move_1_extreme = candle.low
        
        self.move_1_candles.append(candle)
        
        if self.detect_move_1_completion(candle):
            self.moves_completed = 1
            self.state = WavelengthState.MOVE_1
            self.candles_in_current_move = 0
    
    def _transition_move_1(self, candle: Candle) -> None:
        if self.participant_type == ParticipantType.BUYERS:
            if candle.high > self.move_1_extreme:
                self.move_1_extreme = candle.high
        else:  # SELLERS
            if candle.low < self.move_1_extreme:
                self.move_1_extreme = candle.low
        
        if self.detect_move_2_reversal(candle):
            self.moves_completed = 2
            self.state = WavelengthState.MOVE_2
            self.move_2_candles = [candle]
            self.move_2_extreme = None
            self.candles_in_current_move = 0
    
    def _transition_move_2(self, candle: Candle) -> None:
        self.move_2_candles.append(candle)
        
        if self.participant_type == ParticipantType.BUYERS:
            if self.move_2_extreme is None or candle.low < self.move_2_extreme:
                self.move_2_extreme = candle.low
        else:  # SELLERS
            if self.move_2_extreme is None or candle.high > self.move_2_extreme:
                self.move_2_extreme = candle.high
        
        exhaustion_score = self.detect_exhaustion(candle)
        if exhaustion_score >= self.config.exhaustion_threshold:
            self.flip_point = self.move_2_extreme
            self.state = WavelengthState.FLIP_CONFIRMED
            self.flip_confirmation_count = 0
            self.candles_in_current_move = 0
    
    def _transition_flip_confirmed(self, candle: Candle) -> None:
        self.flip_confirmation_count += 1
        
        if self.flip_confirmation_count >= self.config.flip_confirmation_candles:
            self.moves_completed = 3
            self.state = WavelengthState.MOVE_3
            self.candles_in_current_move = 0
    
    def _transition_move_3(self, candle: Candle) -> None:
        if self.flip_point is None or self.move_1_extreme is None:
            return
        
        if self.participant_type == ParticipantType.BUYERS:
            if candle.high > self.move_1_extreme:
                self.state = WavelengthState.COMPLETE
        else:  # SELLERS
            if candle.low < self.move_1_extreme:
                self.state = WavelengthState.COMPLETE
    
    def _build_result(self, candle: Candle) -> WavelengthResult:
        entry_price = self.flip_point
        stop_price = None
        
        if self.flip_point and self.move_2_extreme:
            if self.participant_type == ParticipantType.BUYERS:
                stop_price = self.move_2_extreme - (self.calculate_atr(self.candle_history) * 0.5 if self.candle_history else 0)
            else:
                stop_price = self.move_2_extreme + (self.calculate_atr(self.candle_history) * 0.5 if self.candle_history else 0)
        
        target_price = None
        if self.move_1_extreme and self.flip_point:
            move_1_size = abs(self.move_1_extreme - (self.move_1_start or 0))
            if self.participant_type == ParticipantType.BUYERS:
                target_price = self.move_1_extreme + move_1_size
            else:
                target_price = self.move_1_extreme - move_1_size
        
        return WavelengthResult(
            state=self.state,
            moves_completed=self.moves_completed,
            flip_point=self.flip_point,
            move_1_extreme=self.move_1_extreme,
            move_2_extreme=self.move_2_extreme,
            signal_strength=self.calculate_signal_strength(),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            participant_type=self.participant_type,
            candles_in_current_move=self.candles_in_current_move,
            timestamp=candle.timestamp
        )

def validate_wavelength_progression(states: List[WavelengthState]) -> bool:
    required_sequence = [
        WavelengthState.MOVE_1,
        WavelengthState.MOVE_2,
        WavelengthState.MOVE_3
    ]
    return all(state in states for state in required_sequence)
