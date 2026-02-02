"""
Opposition Rule Engine — THE CORE INVARIANT

This is the spine of HORC. Everything else is support structure.

THE INVARIANT:
    A higher-order participant signal is ONLY valid if the new period
    FIRST opens in OPPOSITION to the previous period's closing signal
    — on a consistent logic.

BOOLEAN FORM:
    if new_open_signal == -previous_close_signal:
        signal.state = CONCLUSIVE
    else:
        signal.state = INCONCLUSIVE

CRITICAL RULES:
    1. Once conclusive → NEVER overridden
    2. Passive TFs must align TO the conclusive signal, not override it
    3. Gap logic is SUBORDINATE — read-only, not authoritative
    4. Timeframe does not matter once opposition is satisfied
    5. Logic type (CRL/OPL) must be consistent — NEVER mixed

Pine Translation:
    All structures here are Pine-safe.
    SignalState maps to int values.
    Opposition check is a single comparison.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple


# ==============================================================================
# SIGNAL STATE — TRI-STATE (NOT BINARY)
# ==============================================================================

class SignalState(IntEnum):
    """
    Tri-state signal — the missing piece that fixes false signals.
    
    CONCLUSIVE states are IMMUTABLE once set.
    INCONCLUSIVE is not false — it's unusable until resolved.
    
    Pine: Maps directly to const int.
    """
    SELL = -1           # Sellers dominant (D-)
    INCONCLUSIVE = 0    # Not tradable — seek another TF or sub-logic
    BUY = 1             # Buyers dominant (D+)
    
    @property
    def is_conclusive(self) -> bool:
        return self != SignalState.INCONCLUSIVE
    
    def oppose(self) -> SignalState:
        """Return the opposite signal state."""
        if self == SignalState.BUY:
            return SignalState.SELL
        elif self == SignalState.SELL:
            return SignalState.BUY
        return SignalState.INCONCLUSIVE


# ==============================================================================
# LOGIC TYPE — DEFINES THE REFERENCE (MUST BE CONSISTENT)
# ==============================================================================

class LogicType(IntEnum):
    """
    The defined logic used to determine signal state.
    
    RULE: Once chosen → NEVER mixed in the same analysis chain.
    
    CRL is preferred — it's clean, no overlap issues, fully deterministic.
    """
    CRL = 1   # Closing Range Logic (reference = H/L of last candle of prev period)
    OPL = 2   # Open Price Logic (reference = open price only)
    ORL = 3   # Opening Range Logic (reference = H/L of first candle of new period)


# ==============================================================================
# PERIOD TYPE — THE BINARY PERIODS
# ==============================================================================

class PeriodType(IntEnum):
    """
    Binary periods — one closes fully before the next opens.
    
    Sessions (Asia/London/NY) are NOT binary periods — they overlap.
    """
    DAILY = 1
    WEEKLY = 2
    MONTHLY = 3
    YEARLY = 4


# ==============================================================================
# PERIOD SIGNAL — CAPTURES OPEN/CLOSE STATE
# ==============================================================================

@dataclass
class PeriodSignal:
    """
    A single period's signal state.
    
    Captures both the opening signal and closing signal,
    and whether opposition was satisfied (conclusive).
    
    Pine: This becomes a set of simple variables per period.
    """
    period: PeriodType
    logic: LogicType
    open_signal: SignalState
    close_signal: SignalState
    conclusive: bool = False
    
    # Reference levels used to determine signals
    reference_high: float = 0.0
    reference_low: float = 0.0
    
    # Timestamp for ordering
    timestamp: int = 0
    
    def __post_init__(self):
        """Validate consistency."""
        if self.conclusive and self.open_signal == SignalState.INCONCLUSIVE:
            raise ValueError("Conclusive signal cannot have INCONCLUSIVE open_signal")
    
    def to_pine_vars(self) -> dict:
        """Export as Pine-compatible dict."""
        return {
            "period_type": int(self.period),
            "logic_type": int(self.logic),
            "open_signal": int(self.open_signal),
            "close_signal": int(self.close_signal),
            "conclusive": 1 if self.conclusive else 0,
            "ref_high": self.reference_high,
            "ref_low": self.reference_low,
        }


# ==============================================================================
# OPPOSITION VALIDATOR — THE CORE ENGINE
# ==============================================================================

def validate_opposition(
    prev_close: SignalState,
    new_open: SignalState,
) -> bool:
    """
    THE CORE INVARIANT.
    
    A new period's signal is CONCLUSIVE & TRUE
    IFF the new period FIRST opens as the OPPOSITE signal
    to the previous period's closing signal.
    
    This is the entire signal philosophy in one function.
    
    Args:
        prev_close: Previous period's closing signal
        new_open: New period's opening signal
        
    Returns:
        True if opposition is satisfied (signal is conclusive)
        False if same direction (signal is inconclusive)
    """
    # Can't validate if either is inconclusive
    if prev_close == SignalState.INCONCLUSIVE:
        return False
    if new_open == SignalState.INCONCLUSIVE:
        return False
    
    # THE RULE: new_open must equal negative of prev_close
    return new_open == -prev_close


def compute_signal_from_crl(
    current_open: float,
    prev_close_high: float,
    prev_close_low: float,
) -> SignalState:
    """
    Compute signal state using Closing Range Logic (CRL).
    
    CRL uses the high/low of the LAST candle of the previous period
    as the reference range.
    
    Rules:
        - Open above prev_close_high → BUY signal
        - Open below prev_close_low → SELL signal  
        - Open inside range → INCONCLUSIVE
    
    This is deterministic with no ambiguity.
    """
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
    """
    Compute signal state using Open Price Logic (OPL).
    
    OPL uses only the open price of the previous period
    as the reference.
    
    Rules:
        - Current open > prev_open → BUY signal
        - Current open < prev_open → SELL signal
        - Current open == prev_open → INCONCLUSIVE
    """
    if current_open > prev_open:
        return SignalState.BUY
    elif current_open < prev_open:
        return SignalState.SELL
    else:
        return SignalState.INCONCLUSIVE


# ==============================================================================
# AGGRESSOR RESOLUTION — ONCE CONCLUSIVE, NEVER OVERRIDDEN
# ==============================================================================

@dataclass
class AggressorState:
    """
    The resolved aggressor state.
    
    Once conclusive:
        - Signal is IMMUTABLE
        - All passive TFs must align TO this
        - Gap/sub-logic cannot override
        
    This is exactly what the edge teaches.
    """
    signal: SignalState = SignalState.INCONCLUSIVE
    conclusive: bool = False
    source_period: Optional[PeriodType] = None
    source_logic: LogicType = LogicType.CRL
    
    # Lock timestamp — when this became conclusive
    locked_at: int = 0
    
    @property
    def is_actionable(self) -> bool:
        """A signal is only actionable when conclusive."""
        return self.conclusive and self.signal != SignalState.INCONCLUSIVE
    
    def to_pine_vars(self) -> dict:
        """Export as Pine-compatible dict."""
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
    """
    Resolve the aggressor from period signals.
    
    RULES:
        1. If current aggressor is already conclusive → return unchanged
        2. Validate opposition between prev_close and new_open
        3. If opposition satisfied → lock new aggressor as conclusive
        4. Otherwise → return inconclusive state
        
    This is the gatekeeper for all downstream signals.
    """
    # Rule 1: Once conclusive, never override
    if current_aggressor and current_aggressor.conclusive:
        return current_aggressor
    
    # Logics must match
    if prev_period.logic != new_period.logic:
        raise ValueError(
            f"Logic mismatch: prev={prev_period.logic}, new={new_period.logic}. "
            "Logic must be consistent — NEVER mixed."
        )
    
    # Validate opposition
    opposition_satisfied = validate_opposition(
        prev_close=prev_period.close_signal,
        new_open=new_period.open_signal,
    )
    
    if opposition_satisfied:
        # Signal is conclusive — lock it
        return AggressorState(
            signal=new_period.open_signal,
            conclusive=True,
            source_period=new_period.period,
            source_logic=new_period.logic,
            locked_at=new_period.timestamp,
        )
    else:
        # Signal is inconclusive — not tradable yet
        return AggressorState(
            signal=SignalState.INCONCLUSIVE,
            conclusive=False,
            source_period=new_period.period,
            source_logic=new_period.logic,
            locked_at=0,
        )


# ==============================================================================
# MULTI-PERIOD CHAIN — FIND FIRST CONCLUSIVE
# ==============================================================================

@dataclass
class OppositionChain:
    """
    Chain of period signals across timeframes.
    
    KEY INSIGHT: Once opposition is satisfied on ANY period,
    ALL timeframes converge to the same final signal.
    
    This chain finds the FIRST conclusive signal.
    """
    periods: List[PeriodSignal]
    logic: LogicType = LogicType.CRL
    
    def __post_init__(self):
        # Validate all periods use the same logic
        for p in self.periods:
            if p.logic != self.logic:
                raise ValueError(
                    f"Logic mismatch in chain: expected {self.logic}, got {p.logic}"
                )
    
    @property
    def first_conclusive(self) -> Optional[PeriodSignal]:
        """Find the first conclusive signal in the chain."""
        for period in self.periods:
            if period.conclusive:
                return period
        return None
    
    def resolve_chain_aggressor(self) -> AggressorState:
        """
        Resolve aggressor from the chain.
        
        Returns the first conclusive signal, or inconclusive if none found.
        """
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
        """Export chain state as Pine-compatible dict."""
        first = self.first_conclusive
        return {
            "chain_logic": int(self.logic),
            "chain_conclusive": 1 if first else 0,
            "chain_signal": int(first.open_signal) if first else 0,
            "chain_period": int(first.period) if first else 0,
        }


# ==============================================================================
# PERIOD BOUNDARY HELPERS
# ==============================================================================

def is_new_period(
    current_timestamp: int,
    last_period_timestamp: int,
    period_type: PeriodType,
) -> bool:
    """
    Check if we've crossed into a new period.
    
    This is essential for detecting when to re-evaluate opposition.
    
    Args:
        current_timestamp: Current bar timestamp (ms)
        last_period_timestamp: Last known period start (ms)
        period_type: The period type to check
        
    Returns:
        True if new period has started
    """
    # Period lengths in milliseconds
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
    """
    Get the start timestamp of the period containing the given timestamp.
    
    Used to align signals to period boundaries.
    """
    # Period lengths in milliseconds
    period_ms = {
        PeriodType.DAILY: 24 * 60 * 60 * 1000,
        PeriodType.WEEKLY: 7 * 24 * 60 * 60 * 1000,
        PeriodType.MONTHLY: 30 * 24 * 60 * 60 * 1000,
        PeriodType.YEARLY: 365 * 24 * 60 * 60 * 1000,
    }
    
    length = period_ms.get(period_type, period_ms[PeriodType.DAILY])
    return (timestamp // length) * length
