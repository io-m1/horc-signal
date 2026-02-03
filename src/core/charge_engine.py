from dataclasses import dataclass
from typing import Dict, Optional

from .participant_engine import ParticipantType
from .flip_engine import FlipEngine, FlipPoint, TimeframeType

class Charge:
    POSITIVE = +1   # Buyer-born
    NEGATIVE = -1   # Seller-born
    NEUTRAL = 0     # Inconclusive
    
    @staticmethod
    def from_participant(participant: ParticipantType) -> int:
        if participant == ParticipantType.BUYER:
            return Charge.POSITIVE
        elif participant == ParticipantType.SELLER:
            return Charge.NEGATIVE
        else:
            return Charge.NEUTRAL
    
    @staticmethod
    def to_symbol(charge: int) -> str:
        if charge == Charge.POSITIVE:
            return "+"
        elif charge == Charge.NEGATIVE:
            return "−"  # Unicode minus
        else:
            return "0"

@dataclass(frozen=True)
class ChargedLevel:
    price: float
    timestamp: int
    timeframe: str              # "S", "D", "W", "M"
    charge: int                 # +1 (buyer-born), -1 (seller-born), 0 (neutral)
    participant_at_formation: ParticipantType   # Participant when level formed
    
    is_high: bool               # True if swing high, False if swing low
    flip_point: Optional[FlipPoint] = None  # Flip that caused charge assignment (if any)
    
    @property
    def charge_symbol(self) -> str:
        return Charge.to_symbol(self.charge)
    
    @property
    def label(self) -> str:
        return f"{self.timeframe}{self.charge_symbol}"

@dataclass
class ChargeState:
    timeframe: str
    tf_type: TimeframeType
    current_charge: int             # Current charge (+1, -1, 0)
    current_participant: ParticipantType    # Current participant in control
    
    levels_assigned: int = 0        # Count of levels assigned with this charge

class ChargeEngine:
    def __init__(self):
        self._charge_states: Dict[str, ChargeState] = {}
    
    def register_timeframe(
        self,
        timeframe: str,
        tf_type: TimeframeType,
        initial_participant: ParticipantType
    ) -> None:
        initial_charge = Charge.from_participant(initial_participant)
        
        self._charge_states[timeframe] = ChargeState(
            timeframe=timeframe,
            tf_type=tf_type,
            current_charge=initial_charge,
            current_participant=initial_participant,
            levels_assigned=0,
        )
    
    def update_participant(
        self,
        timeframe: str,
        new_participant: ParticipantType,
        flip_point: Optional[FlipPoint] = None
    ) -> None:
        if timeframe not in self._charge_states:
            raise ValueError(f"Timeframe {timeframe} not registered")
        
        state = self._charge_states[timeframe]
        new_charge = Charge.from_participant(new_participant)
        
        self._charge_states[timeframe] = ChargeState(
            timeframe=state.timeframe,
            tf_type=state.tf_type,
            current_charge=new_charge,
            current_participant=new_participant,
            levels_assigned=state.levels_assigned,  # Preserve count
        )
    
    def assign_charge(
        self,
        timeframe: str,
        price: float,
        timestamp: int,
        is_high: bool,
        flip_point: Optional[FlipPoint] = None
    ) -> ChargedLevel:
        if timeframe not in self._charge_states:
            raise ValueError(f"Timeframe {timeframe} not registered")
        
        state = self._charge_states[timeframe]
        
        level = ChargedLevel(
            price=price,
            timestamp=timestamp,
            timeframe=timeframe,
            charge=state.current_charge,
            participant_at_formation=state.current_participant,
            is_high=is_high,
            flip_point=flip_point,
        )
        
        self._charge_states[timeframe].levels_assigned += 1
        
        return level
    
    def get_current_charge(self, timeframe: str) -> int:
        if timeframe not in self._charge_states:
            return Charge.NEUTRAL
        return self._charge_states[timeframe].current_charge
    
    def get_current_participant(self, timeframe: str) -> ParticipantType:
        if timeframe not in self._charge_states:
            return ParticipantType.NONE
        return self._charge_states[timeframe].current_participant
    
    def get_charge_symbol(self, timeframe: str) -> str:
        charge = self.get_current_charge(timeframe)
        return Charge.to_symbol(charge)

class ChargeValidator:
    @staticmethod
    def validate_charge_assignment(
        level: ChargedLevel,
        expected_participant: ParticipantType
    ) -> bool:
        expected_charge = Charge.from_participant(expected_participant)
        return level.charge == expected_charge
    
    @staticmethod
    def validate_charge_immutability(
        original_level: ChargedLevel,
        updated_level: ChargedLevel
    ) -> bool:
        return (
            original_level.charge == updated_level.charge and
            original_level.participant_at_formation == updated_level.participant_at_formation
        )
    
    @staticmethod
    def validate_flip_charge_change(
        before_flip: ChargedLevel,
        after_flip: ChargedLevel,
        flip_point: FlipPoint
    ) -> bool:
        before_charge = Charge.from_participant(flip_point.original_participant)
        if before_flip.charge != before_charge:
            return False
        
        after_charge = Charge.from_participant(flip_point.new_participant)
        if after_flip.charge != after_charge:
            return False
        
        return True

def build_multi_tf_charge_label(charges: Dict[str, int]) -> str:
    tf_order = ['M', 'W', 'D', 'S']
    
    labels = []
    for tf in tf_order:
        if tf in charges:
            symbol = Charge.to_symbol(charges[tf])
            labels.append(f"{tf}{symbol}")
    
    return f"({', '.join(labels)})"

def compare_charges(charge1: Dict[str, int], charge2: Dict[str, int]) -> Dict[str, bool]:
    result = {}
    all_tfs = set(charge1.keys()) | set(charge2.keys())
    
    for tf in all_tfs:
        c1 = charge1.get(tf, Charge.NEUTRAL)
        c2 = charge2.get(tf, Charge.NEUTRAL)
        result[tf] = (c1 == c2)
    
    return result
