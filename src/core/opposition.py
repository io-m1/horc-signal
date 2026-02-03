from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple

class SignalState(IntEnum):
    SELL = -1           # Sellers dominant (D-)
    INCONCLUSIVE = 0    # Not tradable — seek another TF or sub-logic
    BUY = 1             # Buyers dominant (D+)
    
    @property
    def is_conclusive(self) -> bool:
        return self != SignalState.INCONCLUSIVE
    
    def oppose(self) -> SignalState:
        if self == SignalState.BUY:
            return SignalState.SELL
        elif self == SignalState.SELL:
            return SignalState.BUY
        return SignalState.INCONCLUSIVE

class LogicType(IntEnum):
    CRL = 1   # Closing Range Logic (reference = H/L of last candle of prev period)
    OPL = 2   # Open Price Logic (reference = open price only)
    ORL = 3   # Opening Range Logic (reference = H/L of first candle of new period)

class PeriodType(IntEnum):
    DAILY = 1
    WEEKLY = 2
    MONTHLY = 3
    YEARLY = 4

@dataclass
class PeriodSignal:
    period: PeriodType
    logic: LogicType
    open_signal: SignalState
    close_signal: SignalState
    conclusive: bool = False
    
    reference_high: float = 0.0
    reference_low: float = 0.0
    
    timestamp: int = 0
    
    def __post_init__(self):
        if self.conclusive and self.open_signal == SignalState.INCONCLUSIVE:
            raise ValueError("Conclusive signal cannot have INCONCLUSIVE open_signal")
    
    def to_pine_vars(self) -> dict:
        return {
            "period_type": int(self.period),
            "logic_type": int(self.logic),
            "open_signal": int(self.open_signal),
            "close_signal": int(self.close_signal),
            "conclusive": 1 if self.conclusive else 0,
            "ref_high": self.reference_high,
            "ref_low": self.reference_low,
        }

def validate_opposition(
    prev_close: SignalState,
    new_open: SignalState,
) -> bool:
    if prev_close == SignalState.INCONCLUSIVE:
        return False
    if new_open == SignalState.INCONCLUSIVE:
        return False
    
    return new_open == -prev_close

def compute_signal_from_crl(
    current_open: float,
    prev_close_high: float,
    prev_close_low: float,
) -> SignalState:
    if current_open > prev_close_high:
        return SignalState.BUY
    elif current_open < prev_close_low:
        return SignalState.SELL
    else:
        return SignalState.INCONCLUSIVE

def compute_signal_from_opl(
    current_open: float,
    prev_open: float,
) -> SignalState:
    if current_open > prev_open:
        return SignalState.BUY
    elif current_open < prev_open:
        return SignalState.SELL
    else:
        return SignalState.INCONCLUSIVE

@dataclass
class AggressorState:
    signal: SignalState = SignalState.INCONCLUSIVE
    conclusive: bool = False
    source_period: Optional[PeriodType] = None
    source_logic: LogicType = LogicType.CRL
    
    locked_at: int = 0
    
    @property
    def is_actionable(self) -> bool:
        return self.conclusive and self.signal != SignalState.INCONCLUSIVE
    
    def to_pine_vars(self) -> dict:
        return {
            "aggressor_signal": int(self.signal),
            "aggressor_conclusive": 1 if self.conclusive else 0,
            "aggressor_period": int(self.source_period) if self.source_period else 0,
            "aggressor_locked": 1 if self.locked_at > 0 else 0,
        }

def resolve_aggressor(
    prev_period: PeriodSignal,
    new_period: PeriodSignal,
    current_aggressor: Optional[AggressorState] = None,
) -> AggressorState:
    if current_aggressor and current_aggressor.conclusive:
        return current_aggressor
    
    if prev_period.logic != new_period.logic:
        raise ValueError(
            f"Logic mismatch: prev={prev_period.logic}, new={new_period.logic}. "
            "Logic must be consistent — NEVER mixed."
        )
    
    opposition_satisfied = validate_opposition(
        prev_close=prev_period.close_signal,
        new_open=new_period.open_signal,
    )
    
    if opposition_satisfied:
        return AggressorState(
            signal=new_period.open_signal,
            conclusive=True,
            source_period=new_period.period,
            source_logic=new_period.logic,
            locked_at=new_period.timestamp,
        )
    else:
        return AggressorState(
            signal=SignalState.INCONCLUSIVE,
            conclusive=False,
            source_period=new_period.period,
            source_logic=new_period.logic,
            locked_at=0,
        )

@dataclass
class OppositionChain:
    periods: List[PeriodSignal]
    logic: LogicType = LogicType.CRL
    
    def __post_init__(self):
        for p in self.periods:
            if p.logic != self.logic:
                raise ValueError(
                    f"Logic mismatch in chain: expected {self.logic}, got {p.logic}"
                )
    
    @property
    def first_conclusive(self) -> Optional[PeriodSignal]:
        for period in self.periods:
            if period.conclusive:
                return period
        return None
    
    def resolve_chain_aggressor(self) -> AggressorState:
        conclusive = self.first_conclusive
        if conclusive:
            return AggressorState(
                signal=conclusive.open_signal,
                conclusive=True,
                source_period=conclusive.period,
                source_logic=self.logic,
                locked_at=conclusive.timestamp,
            )
        return AggressorState(
            signal=SignalState.INCONCLUSIVE,
            conclusive=False,
            source_logic=self.logic,
        )
    
    def to_pine_vars(self) -> dict:
        first = self.first_conclusive
        return {
            "chain_logic": int(self.logic),
            "chain_conclusive": 1 if first else 0,
            "chain_signal": int(first.open_signal) if first else 0,
            "chain_period": int(first.period) if first else 0,
        }

def is_new_period(
    current_timestamp: int,
    last_period_timestamp: int,
    period_type: PeriodType,
) -> bool:
    period_ms = {
        PeriodType.DAILY: 24 * 60 * 60 * 1000,
        PeriodType.WEEKLY: 7 * 24 * 60 * 60 * 1000,
        PeriodType.MONTHLY: 30 * 24 * 60 * 60 * 1000,  # Approximate
        PeriodType.YEARLY: 365 * 24 * 60 * 60 * 1000,  # Approximate
    }
    
    length = period_ms.get(period_type, period_ms[PeriodType.DAILY])
    return (current_timestamp - last_period_timestamp) >= length

def get_period_start(
    timestamp: int,
    period_type: PeriodType,
) -> int:
    period_ms = {
        PeriodType.DAILY: 24 * 60 * 60 * 1000,
        PeriodType.WEEKLY: 7 * 24 * 60 * 60 * 1000,
        PeriodType.MONTHLY: 30 * 24 * 60 * 60 * 1000,
        PeriodType.YEARLY: 365 * 24 * 60 * 60 * 1000,
    }
    
    length = period_ms.get(period_type, period_ms[PeriodType.DAILY])
    return (timestamp // length) * length
