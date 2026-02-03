from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple

from .opposition import SignalState, LogicType, PeriodType, compute_signal_from_crl
from .enums import TIMEFRAME_RANK

class ParticipantType(IntEnum):
    NONE = 0
    BUYER = 1
    SELLER = -1

class ConfidenceState(IntEnum):
    INCONCLUSIVE = 0
    NORMAL = 1
    GAP_OVERRIDE = 2

class ParentPeriod(IntEnum):
    SESSIONAL = 1
    DAILY = 2
    WEEKLY = 3
    MONTHLY = 4

DIVISIBLE_TFS = {
    "W1": ["D1"],  # Weekly → Daily only
    "D1": ["H12", "H8", "H6", "H4"],  # Daily → 12H, 8H, 6H, 4H (NOT 3H, 2H, 1H)
    "H12": ["H6", "H4", "H3"],  # 12H → 6H, 4H, 3H
    "H8": ["H4", "H2"],  # 8H → 4H, 2H
    "H4": ["H2", "H1"],  # 4H → 2H, 1H
    "H1": ["M30", "M15", "M5"],  # 1H → 30M, 15M, 5M
}

PARENT_TF_MAP = {
    ParentPeriod.SESSIONAL: "H1",  # Session uses 1H as parent
    ParentPeriod.DAILY: "D1",
    ParentPeriod.WEEKLY: "W1",
    ParentPeriod.MONTHLY: "MN",
}

def get_divisible_tfs(parent_tf: str) -> List[str]:
    return DIVISIBLE_TFS.get(parent_tf, [])

@dataclass
class ParticipantState:
    participant_type: ParticipantType = ParticipantType.NONE
    participant_tf: str = ""  # The TF that registered the participant
    conclusive_tf: str = ""  # The first TF showing opposition
    parent_tf: str = ""  # The parent TF being analyzed
    confidence_state: ConfidenceState = ConfidenceState.INCONCLUSIVE
    locked: bool = False
    
    lock_timestamp: int = 0
    lock_price: float = 0.0
    
    prev_participant_type: ParticipantType = ParticipantType.NONE
    
    @property
    def is_buyer(self) -> bool:
        return self.participant_type == ParticipantType.BUYER
    
    @property
    def is_seller(self) -> bool:
        return self.participant_type == ParticipantType.SELLER
    
    @property
    def is_conclusive(self) -> bool:
        return self.locked and self.participant_type != ParticipantType.NONE
    
    @property
    def is_gap_override(self) -> bool:
        return self.confidence_state == ConfidenceState.GAP_OVERRIDE
    
    def to_pine_vars(self) -> dict:
        return {
            "participant_type": int(self.participant_type),
            "participant_tf": self.participant_tf,
            "conclusive_tf": self.conclusive_tf,
            "confidence_state": int(self.confidence_state),
            "participant_locked": 1 if self.locked else 0,
            "is_gap_override": 1 if self.is_gap_override else 0,
        }

def check_opposition_on_tf(
    tf: str,
    current_open: float,
    prev_close_high: float,
    prev_close_low: float,
    prev_participant: ParticipantType,
) -> ParticipantType:
    signal = compute_signal_from_crl(
        current_open=current_open,
        prev_close_high=prev_close_high,
        prev_close_low=prev_close_low,
    )
    
    if signal == SignalState.BUY:
        current_participant = ParticipantType.BUYER
    elif signal == SignalState.SELL:
        current_participant = ParticipantType.SELLER
    else:
        return ParticipantType.NONE  # Inconclusive
    
    if current_participant == prev_participant:
        return ParticipantType.NONE  # No opposition (same participant)
    
    return current_participant

@dataclass
class GapInfo:
    exists: bool = False
    direction: int = 0  # +1 bullish, -1 bearish
    size: float = 0.0

def detect_gap(
    current_open: float,
    prev_close: float,
    threshold: float = 0.0001,  # Minimum gap size
) -> GapInfo:
    gap_size = abs(current_open - prev_close)
    
    if gap_size < threshold:
        return GapInfo(exists=False)
    
    direction = 1 if current_open > prev_close else -1
    
    return GapInfo(
        exists=True,
        direction=direction,
        size=gap_size,
    )

def gap_implied_participant(gap: GapInfo) -> ParticipantType:
    if gap.direction == 1:
        return ParticipantType.BUYER
    elif gap.direction == -1:
        return ParticipantType.SELLER
    else:
        return ParticipantType.NONE

class ParticipantEngine:
    def __init__(self, parent_period: ParentPeriod):
        self.parent_period = parent_period
        self.parent_tf = PARENT_TF_MAP[parent_period]
        self.current_state: Optional[ParticipantState] = None
        
    def register_participant(
        self,
        current_open: float,
        prev_close: float,
        prev_high: float,
        prev_low: float,
        prev_participant: ParticipantType = ParticipantType.NONE,
        timestamp: int = 0,
    ) -> ParticipantState:
        if self.current_state and self.current_state.locked:
            return self.current_state
        
        scan_tfs = get_divisible_tfs(self.parent_tf)
        
        conclusive_tf = None
        participant_type = ParticipantType.NONE
        
        for tf in scan_tfs:
            current_participant = check_opposition_on_tf(
                tf=tf,
                current_open=current_open,
                prev_close_high=prev_high,
                prev_close_low=prev_low,
                prev_participant=prev_participant,
            )
            
            if current_participant != ParticipantType.NONE:
                conclusive_tf = tf
                participant_type = current_participant
                break
        
        confidence = ConfidenceState.NORMAL
        
        if conclusive_tf is None:
            gap = detect_gap(current_open, prev_close)
            
            if gap.exists:
                participant_type = gap_implied_participant(gap)
                conclusive_tf = self.parent_tf
                confidence = ConfidenceState.GAP_OVERRIDE
            else:
                confidence = ConfidenceState.INCONCLUSIVE
        
        locked = participant_type != ParticipantType.NONE
        
        self.current_state = ParticipantState(
            participant_type=participant_type,
            participant_tf=conclusive_tf or "",
            conclusive_tf=conclusive_tf or "",
            parent_tf=self.parent_tf,
            confidence_state=confidence,
            locked=locked,
            lock_timestamp=timestamp if locked else 0,
            lock_price=current_open if locked else 0.0,
            prev_participant_type=prev_participant,
        )
        
        return self.current_state
    
    def reset(self):
        self.current_state = None
    
    def is_locked(self) -> bool:
        return self.current_state is not None and self.current_state.locked

def validate_tf_eligibility(participant_tf: str, parent_tf: str) -> bool:
    if participant_tf == parent_tf:
        return False
    
    divisible = get_divisible_tfs(parent_tf)
    return participant_tf in divisible
