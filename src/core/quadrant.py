from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Tuple

from .opposition import SignalState, LogicType, PeriodType, AggressorState
from .enums import TIMEFRAME_RANK

class SignalRole(IntEnum):
    UNRESOLVED = 0
    IMBALANCE = 1
    LIQUIDITY = 2

class ParticipantScope(IntEnum):
    SESSIONAL = 1   # Asia, London, NY sessions
    DAILY = 2       # Daily participant
    WEEKLY = 3      # Weekly participant
    MONTHLY = 4     # Monthly participant

@dataclass
class TimeframeSignal:
    tf: str
    conclusive: bool
    direction: int  # +1 or -1
    logic_type: LogicType = LogicType.CRL
    role: SignalRole = SignalRole.UNRESOLVED
    aggressor: Optional[AggressorState] = None
    
    liquidity_high: float = 0.0
    liquidity_low: float = 0.0
    
    @property
    def rank(self) -> int:
        return TIMEFRAME_RANK.get(self.tf, 0)
    
    @property
    def is_liquidity_owner(self) -> bool:
        return self.role == SignalRole.LIQUIDITY
    
    @property
    def is_imbalance_only(self) -> bool:
        return self.role == SignalRole.IMBALANCE
    
    def to_pine_vars(self) -> dict:
        return {
            f"tf_{self.tf}_conclusive": 1 if self.conclusive else 0,
            f"tf_{self.tf}_direction": self.direction,
            f"tf_{self.tf}_role": int(self.role),
            f"tf_{self.tf}_liq_high": self.liquidity_high,
            f"tf_{self.tf}_liq_low": self.liquidity_low,
        }

MAX_TF_BY_SCOPE: Dict[ParticipantScope, str] = {
    ParticipantScope.SESSIONAL: "M30",   # Sessional max = 30m
    ParticipantScope.DAILY: "H12",       # Daily max = 12H (often 8H)
    ParticipantScope.WEEKLY: "D1",       # Weekly max = Daily
    ParticipantScope.MONTHLY: "W1",      # Monthly max = Weekly
}

LOGIC_BY_SCOPE: Dict[ParticipantScope, LogicType] = {
    ParticipantScope.SESSIONAL: LogicType.OPL,  # Sessions use Open Price Logic
    ParticipantScope.DAILY: LogicType.CRL,      # Daily uses Closing Range Logic
    ParticipantScope.WEEKLY: LogicType.CRL,
    ParticipantScope.MONTHLY: LogicType.CRL,
}

def is_tf_eligible(
    tf: str,
    scope: ParticipantScope,
) -> bool:
    max_tf = MAX_TF_BY_SCOPE.get(scope, "D1")
    max_rank = TIMEFRAME_RANK.get(max_tf, 9)
    tf_rank = TIMEFRAME_RANK.get(tf, 0)
    
    return tf_rank <= max_rank

def get_preferred_logic(scope: ParticipantScope) -> LogicType:
    return LOGIC_BY_SCOPE.get(scope, LogicType.CRL)

@dataclass
class QuadrantResult:
    hct: Optional[TimeframeSignal] = None  # Highest Conclusive Timeframe
    signals: List[TimeframeSignal] = field(default_factory=list)
    resolved: bool = False
    
    @property
    def liquidity_direction(self) -> int:
        if self.hct:
            return self.hct.direction
        return 0
    
    @property
    def liquidity_tf(self) -> Optional[str]:
        if self.hct:
            return self.hct.tf
        return None
    
    @property
    def imbalance_signals(self) -> List[TimeframeSignal]:
        return [s for s in self.signals if s.role == SignalRole.IMBALANCE]
    
    @property
    def has_conflict(self) -> bool:
        conclusive = [s for s in self.signals if s.conclusive]
        if len(conclusive) < 2:
            return False
        
        directions = set(s.direction for s in conclusive)
        return len(directions) > 1
    
    def to_pine_vars(self) -> dict:
        return {
            "quadrant_resolved": 1 if self.resolved else 0,
            "quadrant_hct_tf": self.hct.tf if self.hct else "",
            "quadrant_liq_direction": self.liquidity_direction,
            "quadrant_has_conflict": 1 if self.has_conflict else 0,
            "quadrant_imbalance_count": len(self.imbalance_signals),
        }

def resolve_quadrant(
    signals: List[TimeframeSignal],
) -> QuadrantResult:
    conclusive = [s for s in signals if s.conclusive]
    
    if not conclusive:
        return QuadrantResult(
            hct=None,
            signals=signals,
            resolved=False,
        )
    
    hct = max(conclusive, key=lambda s: s.rank)
    
    for s in signals:
        if not s.conclusive:
            s.role = SignalRole.UNRESOLVED
        elif s.tf == hct.tf:
            s.role = SignalRole.LIQUIDITY
        else:
            s.role = SignalRole.IMBALANCE
    
    return QuadrantResult(
        hct=hct,
        signals=signals,
        resolved=True,
    )

@dataclass
class MultiScopeResult:
    daily_result: Optional[QuadrantResult] = None
    sessional_result: Optional[QuadrantResult] = None
    
    @property
    def master_direction(self) -> int:
        if self.daily_result and self.daily_result.hct:
            return self.daily_result.liquidity_direction
        if self.sessional_result and self.sessional_result.hct:
            return self.sessional_result.liquidity_direction
        return 0
    
    @property
    def sessional_aligned(self) -> bool:
        if not self.daily_result or not self.sessional_result:
            return True  # No conflict if one is missing
        
        if not self.daily_result.hct or not self.sessional_result.hct:
            return True
        
        return (
            self.daily_result.liquidity_direction == 
            self.sessional_result.liquidity_direction
        )
    
    def to_pine_vars(self) -> dict:
        vars = {
            "multi_master_direction": self.master_direction,
            "multi_sessional_aligned": 1 if self.sessional_aligned else 0,
        }
        
        if self.daily_result:
            vars.update({f"daily_{k}": v for k, v in self.daily_result.to_pine_vars().items()})
        if self.sessional_result:
            vars.update({f"sess_{k}": v for k, v in self.sessional_result.to_pine_vars().items()})
        
        return vars

def resolve_multi_scope(
    daily_signals: List[TimeframeSignal],
    sessional_signals: List[TimeframeSignal],
) -> MultiScopeResult:
    daily_result = resolve_quadrant(daily_signals) if daily_signals else None
    sessional_result = resolve_quadrant(sessional_signals) if sessional_signals else None
    
    return MultiScopeResult(
        daily_result=daily_result,
        sessional_result=sessional_result,
    )

@dataclass
class ImbalanceZone:
    high: float
    low: float
    direction: int  # +1 = bullish imbalance, -1 = bearish imbalance
    source_tf: str
    filled: bool = False
    
    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2
    
    def contains_price(self, price: float) -> bool:
        return self.low <= price <= self.high
    
    def to_pine_vars(self) -> dict:
        return {
            "imb_high": self.high,
            "imb_low": self.low,
            "imb_direction": self.direction,
            "imb_filled": 1 if self.filled else 0,
        }

def extract_imbalance_zones(
    result: QuadrantResult,
) -> List[ImbalanceZone]:
    zones = []
    
    for signal in result.signals:
        if signal.role == SignalRole.IMBALANCE:
            zones.append(ImbalanceZone(
                high=signal.liquidity_high,
                low=signal.liquidity_low,
                direction=signal.direction,
                source_tf=signal.tf,
            ))
    
    return zones
