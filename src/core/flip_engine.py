"""
Flip Engine — PHASE 1.5 (Temporal Finality)

DOCTRINE:
    "A flip is valid only before the next corresponding open.
     Whichever participant is dominant at overlap time becomes authoritative."

PURPOSE:
    Validates WHEN participant control changes and enforces temporal finality.
    A Flip Point (FP) marks state transition, but only within a validity window.

CRITICAL RULE:
    Before next open → flip can be registered/invalidated
    After next open → flip is LOCKED (immutable)

CORRESPONDING OPENS:
    - Session ↔ next session open
    - Day ↔ next day open
    - Week ↔ next week open
    - Month ↔ next month open

INTEGRATION:
    - Depends on: ParticipantEngine (PHASE 1)
    - Feeds into: ChargeEngine (assigns +/− to levels)
    - Used by: CoordinateEngine (multi-TF state vectors)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from .participant_engine import ParticipantType


class FlipState(IntEnum):
    """State of a flip point within its validity window."""
    ACTIVE = 0          # Participant in control, no flip yet
    PENDING = 1         # Opposition detected, flip not confirmed
    CONFIRMED = 2       # Flip confirmed, within validity window
    LOCKED = 3          # Next open reached, state immutable
    INVALID = 4         # Flip invalidated (no opposition by next open)


class TimeframeType(IntEnum):
    """Timeframe classifications for flip validation."""
    SESSION = 0
    DAILY = 1
    WEEKLY = 2
    MONTHLY = 3


@dataclass(frozen=True)
class FlipPoint:
    """
    Represents a validated flip point.
    
    A flip is the moment participant control changes on a timeframe.
    Validity is constrained by the next corresponding open.
    """
    timeframe: str                      # "S", "D", "W", "M"
    tf_type: TimeframeType              # Session/Daily/Weekly/Monthly
    flip_price: float                   # Price where flip occurred
    flip_time: int                      # Timestamp of flip
    next_open_time: int                 # Next corresponding open (validity boundary)
    
    original_participant: ParticipantType   # Participant before flip
    new_participant: ParticipantType        # Participant after flip
    
    state: FlipState                    # Current flip state
    
    # Opposition detection metadata
    opposition_detected: bool           # True if opposite side swept
    cycles_elapsed: int                 # How many cycles since TF open (3-cycle allowance)
    
    @property
    def is_valid(self) -> bool:
        """Check if flip is still within validity window."""
        return self.state in (FlipState.PENDING, FlipState.CONFIRMED)
    
    @property
    def is_locked(self) -> bool:
        """Check if flip is locked (past next open)."""
        return self.state == FlipState.LOCKED
    
    @property
    def is_active(self) -> bool:
        """Check if flip is active (before any flip detected)."""
        return self.state == FlipState.ACTIVE


@dataclass(frozen=True)
class FlipValidationResult:
    """Result of flip validation at current timestamp."""
    flip_occurred: bool                 # Did a flip happen?
    flip_point: Optional[FlipPoint]     # FlipPoint if flip occurred
    current_participant: ParticipantType    # Current participant in control
    state: FlipState                    # Current flip state
    within_validity_window: bool        # Is current time before next open?
    

class FlipEngine:
    """
    Validates participant control changes with temporal finality.
    
    CRITICAL RULES:
    1. Flip is valid ONLY before next corresponding open
    2. After next open → state is LOCKED (immutable)
    3. 3-cycle allowance for participant establishment
    4. Whichever participant is dominant at next open = authoritative
    
    WORKFLOW:
    1. Detect TF open (session/day/week/month boundary)
    2. Track which side swept first (high or low)
    3. Monitor for opposition (opposite side sweep)
    4. Register flip if opposition detected before next open
    5. Lock state at next open (immutable)
    """
    
    def __init__(self, timeframe: str, tf_type: TimeframeType):
        """
        Initialize flip engine for a specific timeframe.
        
        Args:
            timeframe: "S", "D", "W", "M"
            tf_type: Session/Daily/Weekly/Monthly classification
        """
        self.timeframe = timeframe
        self.tf_type = tf_type
        
        # Current state tracking
        self._current_flip: Optional[FlipPoint] = None
        self._current_participant: ParticipantType = ParticipantType.NONE
        self._tf_open_time: int = 0
        self._next_open_time: int = 0
        self._cycles_elapsed: int = 0
        
        # Opposition tracking
        self._high_swept: bool = False
        self._low_swept: bool = False
        self._first_sweep_participant: ParticipantType = ParticipantType.NONE
    
    def register_tf_open(
        self,
        open_time: int,
        next_open_time: int,
        initial_participant: ParticipantType
    ) -> None:
        """
        Register a new timeframe open (session/day/week/month boundary).
        
        This resets flip tracking for the new period.
        Locks any previous flip if it existed.
        
        Args:
            open_time: Timestamp of TF open
            next_open_time: Timestamp of next corresponding open (validity boundary)
            initial_participant: Participant in control at open
        """
        # Lock previous flip if it existed
        if self._current_flip and not self._current_flip.is_locked:
            self._current_flip = FlipPoint(
                timeframe=self._current_flip.timeframe,
                tf_type=self._current_flip.tf_type,
                flip_price=self._current_flip.flip_price,
                flip_time=self._current_flip.flip_time,
                next_open_time=self._current_flip.next_open_time,
                original_participant=self._current_flip.original_participant,
                new_participant=self._current_flip.new_participant,
                state=FlipState.LOCKED,  # LOCK
                opposition_detected=self._current_flip.opposition_detected,
                cycles_elapsed=self._current_flip.cycles_elapsed,
            )
        
        # Reset for new period
        self._tf_open_time = open_time
        self._next_open_time = next_open_time
        self._current_participant = initial_participant
        self._cycles_elapsed = 0
        self._high_swept = False
        self._low_swept = False
        self._first_sweep_participant = ParticipantType.NONE
        
        # No flip yet in this period
        self._current_flip = None
    
    def update_sweep(
        self,
        current_time: int,
        current_high: float,
        current_low: float,
        range_high: float,  # Highest point since TF open
        range_low: float     # Lowest point since TF open
    ) -> None:
        """
        Update sweep tracking for opposition detection.
        
        Sweep means taking out the previous range (high or low).
        
        Args:
            current_time: Current timestamp
            current_high: Current bar high
            current_low: Current bar low
            range_high: Highest point in the current TF period so far
            range_low: Lowest point in the current TF period so far
        """
        # Check if we're past validity window
        if current_time >= self._next_open_time:
            # Lock any existing flip
            if self._current_flip and not self._current_flip.is_locked:
                self._lock_flip()
            return
        
        # Increment cycle counter
        self._cycles_elapsed += 1
        
        # Track high sweep (new high created)
        if current_high > range_high and not self._high_swept:
            self._high_swept = True
            if self._first_sweep_participant == ParticipantType.NONE:
                self._first_sweep_participant = ParticipantType.BUYER
        
        # Track low sweep (new low created)
        if current_low < range_low and not self._low_swept:
            self._low_swept = True
            if self._first_sweep_participant == ParticipantType.NONE:
                self._first_sweep_participant = ParticipantType.SELLER
    
    def validate_flip(
        self,
        current_time: int,
        current_price: float
    ) -> FlipValidationResult:
        """
        Validate if a flip has occurred at current timestamp.
        
        LOGIC:
        1. Check if we're within validity window (before next open)
        2. Check if opposition detected (both high and low swept)
        3. Determine if flip should be registered/confirmed/locked
        
        Args:
            current_time: Current timestamp
            current_price: Current price
        
        Returns:
            FlipValidationResult with flip status
        """
        within_validity_window = current_time < self._next_open_time
        
        # If past validity window, lock existing flip
        if not within_validity_window:
            if self._current_flip and not self._current_flip.is_locked:
                self._lock_flip()
            
            return FlipValidationResult(
                flip_occurred=self._current_flip is not None,
                flip_point=self._current_flip,
                current_participant=self._current_participant,
                state=self._current_flip.state if self._current_flip else FlipState.ACTIVE,
                within_validity_window=False,
            )
        
        # Check for opposition (both high and low swept)
        opposition_detected = self._high_swept and self._low_swept
        
        # No opposition yet
        if not opposition_detected:
            return FlipValidationResult(
                flip_occurred=False,
                flip_point=None,
                current_participant=self._current_participant,
                state=FlipState.ACTIVE,
                within_validity_window=True,
            )
        
        # Opposition detected - determine new participant
        # If BUYER was first → flip to SELLER
        # If SELLER was first → flip to BUYER
        if self._first_sweep_participant == ParticipantType.BUYER:
            new_participant = ParticipantType.SELLER
        elif self._first_sweep_participant == ParticipantType.SELLER:
            new_participant = ParticipantType.BUYER
        else:
            # No clear first sweep (shouldn't happen)
            return FlipValidationResult(
                flip_occurred=False,
                flip_point=None,
                current_participant=self._current_participant,
                state=FlipState.ACTIVE,
                within_validity_window=True,
            )
        
        # Register flip if not already registered
        if self._current_flip is None:
            self._current_flip = FlipPoint(
                timeframe=self.timeframe,
                tf_type=self.tf_type,
                flip_price=current_price,
                flip_time=current_time,
                next_open_time=self._next_open_time,
                original_participant=self._current_participant,
                new_participant=new_participant,
                state=FlipState.CONFIRMED,
                opposition_detected=True,
                cycles_elapsed=self._cycles_elapsed,
            )
            
            # Update current participant
            self._current_participant = new_participant
        
        return FlipValidationResult(
            flip_occurred=True,
            flip_point=self._current_flip,
            current_participant=new_participant,
            state=FlipState.CONFIRMED,
            within_validity_window=True,
        )
    
    def _lock_flip(self) -> None:
        """Lock current flip (make immutable)."""
        if self._current_flip:
            self._current_flip = FlipPoint(
                timeframe=self._current_flip.timeframe,
                tf_type=self._current_flip.tf_type,
                flip_price=self._current_flip.flip_price,
                flip_time=self._current_flip.flip_time,
                next_open_time=self._current_flip.next_open_time,
                original_participant=self._current_flip.original_participant,
                new_participant=self._current_flip.new_participant,
                state=FlipState.LOCKED,
                opposition_detected=self._current_flip.opposition_detected,
                cycles_elapsed=self._current_flip.cycles_elapsed,
            )
    
    def get_current_participant(self) -> ParticipantType:
        """Get current participant in control (after any flips)."""
        return self._current_participant
    
    def get_current_flip(self) -> Optional[FlipPoint]:
        """Get current flip point (if any)."""
        return self._current_flip
    
    def is_within_validity_window(self, current_time: int) -> bool:
        """Check if current time is before next open (validity window)."""
        return current_time < self._next_open_time


# Utility functions

def get_next_open_time(
    current_time: int,
    tf_type: TimeframeType,
    session_duration: int = 86400,  # 24 hours default
) -> int:
    """
    Calculate next corresponding open time for a timeframe.
    
    Args:
        current_time: Current timestamp
        tf_type: Session/Daily/Weekly/Monthly
        session_duration: Session duration in seconds (default 24h)
    
    Returns:
        Timestamp of next open
    """
    if tf_type == TimeframeType.SESSION:
        return current_time + session_duration
    elif tf_type == TimeframeType.DAILY:
        return current_time + 86400  # 24 hours
    elif tf_type == TimeframeType.WEEKLY:
        return current_time + 604800  # 7 days
    elif tf_type == TimeframeType.MONTHLY:
        return current_time + 2592000  # ~30 days (simplified)
    else:
        return current_time + 86400  # Default to daily


def detect_opposition(
    high_swept: bool,
    low_swept: bool,
    cycles_elapsed: int,
    max_cycles: int = 3
) -> bool:
    """
    Detect if opposition has occurred (3-cycle allowance).
    
    Args:
        high_swept: Has high been swept?
        low_swept: Has low been swept?
        cycles_elapsed: How many cycles since TF open
        max_cycles: Maximum cycles for allowance (default 3)
    
    Returns:
        True if opposition detected
    """
    # Both sides swept = opposition
    if high_swept and low_swept:
        return True
    
    # If max cycles reached without opposition → no opposition
    if cycles_elapsed >= max_cycles:
        return False
    
    return False
