"""
Participant Engine — THE CORE DECISION SYSTEM

This is PHASE 1. Everything else depends on this being correct.

PURPOSE:
    Determine WHO is in control (buyer/seller) and LOCK that state
    for the entire session/day/week until invalidated.

CORE OUTPUTS (NON-NEGOTIABLE):
    - participant_type: BUYER / SELLER / NONE
    - participant_tf: The timeframe that registered the participant
    - conclusive_tf: The first TF showing opposition
    - confidence_state: NORMAL / GAP_OVERRIDE
    - locked: Boolean - once locked, immutable until next period

THE KEY RULES:
    1. A timeframe is "conclusive" IFF it shows OPPOSITION to prior participant
    2. Opposition = different participant (buyer ↔ seller) from previous period
    3. Liquidity uses HIGHEST conclusive TF (low frequency)
    4. Imbalance uses LOWEST conclusive TF (intrinsic inefficiency)
    5. A TF CANNOT register itself (Daily TF ≠ daily liquidity)
    6. If no opposition at any TF → "all is true" (gap override)

DIVISIBLE TIMEFRAMES ONLY:
    Weekly → Daily (D1)
    Daily → 12H, 8H, 6H, 4H (NOT 3H, 2H, 1H — not divisible)
    12H → 6H, 4H, 3H
    8H → 4H, 2H
    4H → 2H, 1H
    1H → 30M, 15M, 5M

ALGORITHM (FORMAL):
    
    # STEP 1: Determine parent TF and scan direction
    IF analyzing_weekly:
        parent_tf = "W1"
        scan_range = ["D1"]  # Only Daily divisible from Weekly
        
    IF analyzing_daily:
        parent_tf = "D1"
        scan_range = ["H12", "H8", "H6", "H4"]  # Divisible only
        
    IF analyzing_sessional:
        parent_tf = "SESSION"
        scan_range = ["M30", "M15", "M5"]
    
    # STEP 2: Get previous period's participant
    prev_participant = get_previous_participant(parent_tf)
    
    # STEP 3: Scan for opposition (HIGH → LOW for liquidity)
    conclusive_tf = NONE
    FOR tf IN scan_range (HIGH to LOW):
        current_participant = check_opposition(tf, prev_participant)
        
        IF current_participant != NONE AND current_participant != prev_participant:
            conclusive_tf = tf
            participant_type = current_participant
            confidence_state = NORMAL
            BREAK
    
    # STEP 4: Handle "all is true" (gap override)
    IF conclusive_tf == NONE:
        # No opposition found at any TF
        IF gap_exists(parent_tf):
            # Gap overrides opposition requirement
            participant_type = gap_implied_participant()
            conclusive_tf = parent_tf
            confidence_state = GAP_OVERRIDE
        ELSE:
            # No participant can be determined
            participant_type = NONE
            confidence_state = INCONCLUSIVE
    
    # STEP 5: Lock participant
    IF participant_type != NONE:
        lock_participant(
            type=participant_type,
            tf=conclusive_tf,
            parent=parent_tf,
            confidence=confidence_state
        )
    
    RETURN ParticipantState(
        type=participant_type,
        tf=conclusive_tf,
        parent=parent_tf,
        confidence=confidence_state,
        locked=True if participant_type != NONE else False
    )

PINE TRANSLATION:
    All logic here translates to:
    - var int participant_type = 0
    - var string participant_tf = ""
    - var string conclusive_tf = ""
    - var int confidence_state = 0
    - var bool participant_locked = false
    
    On new period boundary:
    - Reset if period changed
    - Run scan
    - Lock state

NO VISUALS. NO ENTRIES. ONLY STRUCTURE DETERMINATION.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple

from .opposition import SignalState, LogicType, PeriodType, compute_signal_from_crl
from .enums import TIMEFRAME_RANK


# ==============================================================================
# PARTICIPANT TYPE — WHO IS IN CONTROL
# ==============================================================================

class ParticipantType(IntEnum):
    """
    The participant in control.
    
    Once locked, this is IMMUTABLE until next period boundary.
    """
    NONE = 0
    BUYER = 1
    SELLER = -1


# ==============================================================================
# CONFIDENCE STATE — HOW WAS PARTICIPANT DETERMINED
# ==============================================================================

class ConfidenceState(IntEnum):
    """
    How the participant was determined.
    
    NORMAL: Standard opposition rule satisfied
    GAP_OVERRIDE: No opposition found, gap implied participant
    INCONCLUSIVE: No participant could be determined
    """
    INCONCLUSIVE = 0
    NORMAL = 1
    GAP_OVERRIDE = 2


# ==============================================================================
# PARENT PERIOD — WHAT ARE WE ANALYZING
# ==============================================================================

class ParentPeriod(IntEnum):
    """
    The parent period being analyzed.
    
    Determines which divisible TFs to scan.
    """
    SESSIONAL = 1
    DAILY = 2
    WEEKLY = 3
    MONTHLY = 4


# ==============================================================================
# DIVISIBLE TIMEFRAMES — THE ONLY TFS THAT MATTER
# ==============================================================================

# Divisible TF relationships
DIVISIBLE_TFS = {
    "W1": ["D1"],  # Weekly → Daily only
    "D1": ["H12", "H8", "H6", "H4"],  # Daily → 12H, 8H, 6H, 4H (NOT 3H, 2H, 1H)
    "H12": ["H6", "H4", "H3"],  # 12H → 6H, 4H, 3H
    "H8": ["H4", "H2"],  # 8H → 4H, 2H
    "H4": ["H2", "H1"],  # 4H → 2H, 1H
    "H1": ["M30", "M15", "M5"],  # 1H → 30M, 15M, 5M
}

# Parent period to parent TF mapping
PARENT_TF_MAP = {
    ParentPeriod.SESSIONAL: "H1",  # Session uses 1H as parent
    ParentPeriod.DAILY: "D1",
    ParentPeriod.WEEKLY: "W1",
    ParentPeriod.MONTHLY: "MN",
}


def get_divisible_tfs(parent_tf: str) -> List[str]:
    """
    Get the divisible timeframes for a parent TF.
    
    These are the ONLY timeframes that matter for participant registration.
    
    Args:
        parent_tf: The parent timeframe (e.g., "D1", "W1")
    
    Returns:
        List of divisible TFs, ordered HIGH to LOW
    """
    return DIVISIBLE_TFS.get(parent_tf, [])


# ==============================================================================
# PARTICIPANT STATE — THE CORE OUTPUT
# ==============================================================================

@dataclass
class ParticipantState:
    """
    The participant state output.
    
    This is what locks for the session/day/week.
    Once locked, it is IMMUTABLE until next period boundary.
    """
    participant_type: ParticipantType = ParticipantType.NONE
    participant_tf: str = ""  # The TF that registered the participant
    conclusive_tf: str = ""  # The first TF showing opposition
    parent_tf: str = ""  # The parent TF being analyzed
    confidence_state: ConfidenceState = ConfidenceState.INCONCLUSIVE
    locked: bool = False
    
    # Metadata
    lock_timestamp: int = 0
    lock_price: float = 0.0
    
    # Previous period state (for opposition check)
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
        """Export as Pine-compatible dict."""
        return {
            "participant_type": int(self.participant_type),
            "participant_tf": self.participant_tf,
            "conclusive_tf": self.conclusive_tf,
            "confidence_state": int(self.confidence_state),
            "participant_locked": 1 if self.locked else 0,
            "is_gap_override": 1 if self.is_gap_override else 0,
        }


# ==============================================================================
# OPPOSITION CHECK — CORE VALIDATION
# ==============================================================================

def check_opposition_on_tf(
    tf: str,
    current_open: float,
    prev_close_high: float,
    prev_close_low: float,
    prev_participant: ParticipantType,
) -> ParticipantType:
    """
    Check for opposition on a specific timeframe.
    
    Opposition = current participant ≠ previous participant
    
    Args:
        tf: The timeframe to check
        current_open: Opening price of new period on this TF
        prev_close_high: High of last candle of previous period
        prev_close_low: Low of last candle of previous period
        prev_participant: Previous period's participant
    
    Returns:
        ParticipantType if opposition found, NONE otherwise
    """
    # Use CRL (Closing Range Logic) to determine current signal
    signal = compute_signal_from_crl(
        current_open=current_open,
        prev_close_high=prev_close_high,
        prev_close_low=prev_close_low,
    )
    
    # Map signal to participant
    if signal == SignalState.BUY:
        current_participant = ParticipantType.BUYER
    elif signal == SignalState.SELL:
        current_participant = ParticipantType.SELLER
    else:
        return ParticipantType.NONE  # Inconclusive
    
    # Check for opposition
    if current_participant == prev_participant:
        return ParticipantType.NONE  # No opposition (same participant)
    
    return current_participant


# ==============================================================================
# GAP DETECTION — "ALL IS TRUE" OVERRIDE
# ==============================================================================

@dataclass
class GapInfo:
    """
    Gap information for override logic.
    """
    exists: bool = False
    direction: int = 0  # +1 bullish, -1 bearish
    size: float = 0.0


def detect_gap(
    current_open: float,
    prev_close: float,
    threshold: float = 0.0001,  # Minimum gap size
) -> GapInfo:
    """
    Detect if a gap exists and its direction.
    
    A gap is a price discontinuity between periods.
    
    Args:
        current_open: Opening price of new period
        prev_close: Closing price of previous period
        threshold: Minimum gap size to register
    
    Returns:
        GapInfo with existence and direction
    """
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
    """
    Determine participant implied by gap direction.
    
    Bullish gap → Buyers in control
    Bearish gap → Sellers in control
    """
    if gap.direction == 1:
        return ParticipantType.BUYER
    elif gap.direction == -1:
        return ParticipantType.SELLER
    else:
        return ParticipantType.NONE


# ==============================================================================
# PARTICIPANT ENGINE — THE CORE ALGORITHM
# ==============================================================================

class ParticipantEngine:
    """
    The Participant Engine — determines WHO is in control.
    
    This is PHASE 1. Everything else depends on this.
    
    Usage:
        engine = ParticipantEngine(parent_period=ParentPeriod.DAILY)
        
        # On new period boundary (e.g., daily open)
        state = engine.register_participant(
            current_candles={...},  # Current bar data per TF
            prev_candles={...},     # Previous period last bar per TF
        )
        
        if state.locked:
            print(f"Participant: {state.participant_type.name}")
            print(f"Conclusive TF: {state.conclusive_tf}")
            print(f"Confidence: {state.confidence_state.name}")
    """
    
    def __init__(self, parent_period: ParentPeriod):
        """
        Initialize engine for a specific parent period.
        
        Args:
            parent_period: DAILY, WEEKLY, etc.
        """
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
        """
        Register participant using the core algorithm.
        
        ALGORITHM:
            1. Get divisible TFs for parent
            2. Scan HIGH → LOW for opposition
            3. Lock on first opposition found
            4. If no opposition, check gap override
            5. Return locked state
        
        Args:
            current_open: Opening price of new period (parent TF)
            prev_close: Closing price of previous period
            prev_high: High of last candle of previous period
            prev_low: Low of last candle of previous period
            prev_participant: Previous period's participant
            timestamp: Current timestamp
        
        Returns:
            ParticipantState with locked participant
        """
        # If already locked, return current state
        if self.current_state and self.current_state.locked:
            return self.current_state
        
        # STEP 1: Get divisible TFs (HIGH → LOW order)
        scan_tfs = get_divisible_tfs(self.parent_tf)
        
        # STEP 2: Scan for opposition
        conclusive_tf = None
        participant_type = ParticipantType.NONE
        
        for tf in scan_tfs:
            # Check opposition on this TF
            # Note: In real implementation, you'd need actual OHLC data per TF
            # For now, we use parent TF data as proxy
            current_participant = check_opposition_on_tf(
                tf=tf,
                current_open=current_open,
                prev_close_high=prev_high,
                prev_close_low=prev_low,
                prev_participant=prev_participant,
            )
            
            if current_participant != ParticipantType.NONE:
                # Opposition found!
                conclusive_tf = tf
                participant_type = current_participant
                break
        
        # STEP 3: Handle "all is true" (gap override)
        confidence = ConfidenceState.NORMAL
        
        if conclusive_tf is None:
            # No opposition found at any TF
            gap = detect_gap(current_open, prev_close)
            
            if gap.exists:
                # Gap overrides opposition requirement
                participant_type = gap_implied_participant(gap)
                conclusive_tf = self.parent_tf
                confidence = ConfidenceState.GAP_OVERRIDE
            else:
                # No participant can be determined
                confidence = ConfidenceState.INCONCLUSIVE
        
        # STEP 4: Lock participant
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
        """
        Reset participant state.
        
        Called on new period boundary (e.g., new day, new week).
        """
        self.current_state = None
    
    def is_locked(self) -> bool:
        """Check if participant is locked."""
        return self.current_state is not None and self.current_state.locked


# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================

def validate_tf_eligibility(participant_tf: str, parent_tf: str) -> bool:
    """
    Validate that a TF can register participant for parent.
    
    RULE: A TF CANNOT register itself.
        - Daily TF ❌ cannot register daily liquidity
        - Weekly TF ❌ cannot register weekly liquidity
    
    Args:
        participant_tf: The TF that registered participant
        parent_tf: The parent TF being analyzed
    
    Returns:
        True if valid pairing
    """
    # TF cannot register itself
    if participant_tf == parent_tf:
        return False
    
    # TF must be in divisible list
    divisible = get_divisible_tfs(parent_tf)
    return participant_tf in divisible


# ==============================================================================
# DOCTRINE SUMMARY
# ==============================================================================

PARTICIPANT_ENGINE_DOCTRINE = """
THE CORE ALGORITHM (PHASE 1):

PURPOSE:
    Determine WHO is in control and LOCK that state until next period.

OUTPUTS:
    - participant_type (BUYER / SELLER / NONE)
    - conclusive_tf (first TF showing opposition)
    - confidence_state (NORMAL / GAP_OVERRIDE / INCONCLUSIVE)
    - locked (boolean - immutable once set)

RULES:
    1. "Conclusive" = first TF showing OPPOSITION to prior participant
    2. Opposition = different participant from previous period
    3. Scan divisible TFs HIGH → LOW (low frequency first)
    4. Lock on first opposition found
    5. If no opposition → check gap override ("all is true")
    6. A TF CANNOT register itself (Daily ≠ daily liquidity)

DIVISIBLE TFS ONLY:
    Weekly → Daily
    Daily → 12H, 8H, 6H, 4H (NOT 3H, 2H, 1H)
    12H → 6H, 4H, 3H
    8H → 4H, 2H
    4H → 2H, 1H
    1H → 30M, 15M, 5M

GAP OVERRIDE:
    If no opposition at any TF + gap exists:
        → Gap direction implies participant
        → Confidence = GAP_OVERRIDE
    
    "If all is true, then all is true."

NO VISUALS. NO ENTRIES. ONLY STRUCTURE DETERMINATION.
This must work flawlessly before anything else.
"""
