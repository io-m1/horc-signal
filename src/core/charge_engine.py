"""
Charge Engine — PHASE 1.5 (Participant State Inheritance)

DOCTRINE:
    "Any high or low inherits the participant state active at the time it was formed."

PURPOSE:
    Assigns +/− charge to price levels based on participant control at formation.
    Charge is IMMUTABLE once assigned (no retroactive changes).

CRITICAL RULE:
    - Before flip: all levels = original participant charge
    - After flip: all NEW levels = new participant charge
    - Previous charges remain unchanged (immutability)

CHARGE ENCODING:
    - BUYER control → `+` (positive charge, +1)
    - SELLER control → `−` (negative charge, -1)
    - No control → `0` (neutral/inconclusive)

INTEGRATION:
    - Depends on: ParticipantEngine (WHO), FlipEngine (WHEN)
    - Feeds into: CoordinateEngine (multi-TF state vectors)
    - Used by: Liquidity Registration (marking levels with charge)
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .participant_engine import ParticipantType
from .flip_engine import FlipEngine, FlipPoint, TimeframeType


class Charge:
    """Charge state representation (+1, -1, 0)."""
    POSITIVE = +1   # Buyer-born
    NEGATIVE = -1   # Seller-born
    NEUTRAL = 0     # Inconclusive
    
    @staticmethod
    def from_participant(participant: ParticipantType) -> int:
        """Convert ParticipantType to charge value."""
        if participant == ParticipantType.BUYER:
            return Charge.POSITIVE
        elif participant == ParticipantType.SELLER:
            return Charge.NEGATIVE
        else:
            return Charge.NEUTRAL
    
    @staticmethod
    def to_symbol(charge: int) -> str:
        """Convert charge to symbol (+, -, 0)."""
        if charge == Charge.POSITIVE:
            return "+"
        elif charge == Charge.NEGATIVE:
            return "−"  # Unicode minus
        else:
            return "0"


@dataclass(frozen=True)
class ChargedLevel:
    """
    A price level with participant charge assigned.
    
    Charge is assigned at formation time and is IMMUTABLE.
    """
    price: float
    timestamp: int
    timeframe: str              # "S", "D", "W", "M"
    charge: int                 # +1 (buyer-born), -1 (seller-born), 0 (neutral)
    participant_at_formation: ParticipantType   # Participant when level formed
    
    is_high: bool               # True if swing high, False if swing low
    flip_point: Optional[FlipPoint] = None  # Flip that caused charge assignment (if any)
    
    @property
    def charge_symbol(self) -> str:
        """Get charge as symbol (+, -, 0)."""
        return Charge.to_symbol(self.charge)
    
    @property
    def label(self) -> str:
        """Get human-readable label (e.g., 'D+', 'W−')."""
        return f"{self.timeframe}{self.charge_symbol}"


@dataclass
class ChargeState:
    """
    Tracks charge state for a single timeframe.
    
    Maintains current participant and charge assignment logic.
    """
    timeframe: str
    tf_type: TimeframeType
    current_charge: int             # Current charge (+1, -1, 0)
    current_participant: ParticipantType    # Current participant in control
    
    # Historical tracking (for immutability validation)
    levels_assigned: int = 0        # Count of levels assigned with this charge


class ChargeEngine:
    """
    Assigns +/− charge to price levels based on participant state at formation.
    
    CRITICAL RULES:
    1. Charge assigned at formation time (not retroactively)
    2. Charge is IMMUTABLE once assigned
    3. Flip changes charge for NEW levels only
    4. Each timeframe has independent charge state
    
    WORKFLOW:
    1. Monitor participant control via FlipEngine
    2. When level forms (high/low), assign current participant charge
    3. On flip, update charge for FUTURE levels
    4. Previous charges remain unchanged
    """
    
    def __init__(self):
        """Initialize charge engine with state tracking per timeframe."""
        self._charge_states: Dict[str, ChargeState] = {}
    
    def register_timeframe(
        self,
        timeframe: str,
        tf_type: TimeframeType,
        initial_participant: ParticipantType
    ) -> None:
        """
        Register a timeframe for charge tracking.
        
        Args:
            timeframe: "S", "D", "W", "M"
            tf_type: Session/Daily/Weekly/Monthly
            initial_participant: Initial participant in control
        """
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
        """
        Update participant for a timeframe (typically after a flip).
        
        This changes the charge for FUTURE levels only.
        Previous levels remain unchanged.
        
        Args:
            timeframe: "S", "D", "W", "M"
            new_participant: New participant in control
            flip_point: FlipPoint that caused the change (if any)
        """
        if timeframe not in self._charge_states:
            raise ValueError(f"Timeframe {timeframe} not registered")
        
        state = self._charge_states[timeframe]
        new_charge = Charge.from_participant(new_participant)
        
        # Update state for future assignments
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
        """
        Assign charge to a price level (high or low).
        
        Charge is determined by current participant state at formation time.
        Once assigned, charge is IMMUTABLE.
        
        Args:
            timeframe: "S", "D", "W", "M"
            price: Price of the level
            timestamp: Time when level formed
            is_high: True if swing high, False if swing low
            flip_point: FlipPoint active at formation (if any)
        
        Returns:
            ChargedLevel with assigned charge
        """
        if timeframe not in self._charge_states:
            raise ValueError(f"Timeframe {timeframe} not registered")
        
        state = self._charge_states[timeframe]
        
        # Assign current charge (at formation time)
        level = ChargedLevel(
            price=price,
            timestamp=timestamp,
            timeframe=timeframe,
            charge=state.current_charge,
            participant_at_formation=state.current_participant,
            is_high=is_high,
            flip_point=flip_point,
        )
        
        # Increment assignment counter
        self._charge_states[timeframe].levels_assigned += 1
        
        return level
    
    def get_current_charge(self, timeframe: str) -> int:
        """Get current charge for a timeframe (+1, -1, 0)."""
        if timeframe not in self._charge_states:
            return Charge.NEUTRAL
        return self._charge_states[timeframe].current_charge
    
    def get_current_participant(self, timeframe: str) -> ParticipantType:
        """Get current participant for a timeframe."""
        if timeframe not in self._charge_states:
            return ParticipantType.NONE
        return self._charge_states[timeframe].current_participant
    
    def get_charge_symbol(self, timeframe: str) -> str:
        """Get current charge as symbol (+, -, 0)."""
        charge = self.get_current_charge(timeframe)
        return Charge.to_symbol(charge)


class ChargeValidator:
    """
    Validates charge assignment rules and immutability.
    
    Used for testing and verification of charge logic.
    """
    
    @staticmethod
    def validate_charge_assignment(
        level: ChargedLevel,
        expected_participant: ParticipantType
    ) -> bool:
        """
        Validate that level charge matches expected participant.
        
        Args:
            level: ChargedLevel to validate
            expected_participant: Expected participant at formation
        
        Returns:
            True if charge is correct
        """
        expected_charge = Charge.from_participant(expected_participant)
        return level.charge == expected_charge
    
    @staticmethod
    def validate_charge_immutability(
        original_level: ChargedLevel,
        updated_level: ChargedLevel
    ) -> bool:
        """
        Validate that charge has not changed (immutability check).
        
        Args:
            original_level: Original ChargedLevel
            updated_level: Updated ChargedLevel (should be same)
        
        Returns:
            True if charge is unchanged
        """
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
        """
        Validate that charge changed correctly after flip.
        
        Levels before flip should have original charge.
        Levels after flip should have new charge.
        
        Args:
            before_flip: Level formed before flip
            after_flip: Level formed after flip
            flip_point: FlipPoint that occurred
        
        Returns:
            True if charges are correct
        """
        # Before flip: should match original participant
        before_charge = Charge.from_participant(flip_point.original_participant)
        if before_flip.charge != before_charge:
            return False
        
        # After flip: should match new participant
        after_charge = Charge.from_participant(flip_point.new_participant)
        if after_flip.charge != after_charge:
            return False
        
        return True


# Utility functions

def build_multi_tf_charge_label(charges: Dict[str, int]) -> str:
    """
    Build multi-timeframe charge label.
    
    Example: {'M': -1, 'W': +1, 'D': +1, 'S': -1} → "(M−, W+, D+, S−)"
    
    Args:
        charges: Dict mapping timeframe to charge
    
    Returns:
        Formatted charge label
    """
    # Order: M, W, D, S
    tf_order = ['M', 'W', 'D', 'S']
    
    labels = []
    for tf in tf_order:
        if tf in charges:
            symbol = Charge.to_symbol(charges[tf])
            labels.append(f"{tf}{symbol}")
    
    return f"({', '.join(labels)})"


def compare_charges(charge1: Dict[str, int], charge2: Dict[str, int]) -> Dict[str, bool]:
    """
    Compare two charge states and identify differences.
    
    Args:
        charge1: First charge state
        charge2: Second charge state
    
    Returns:
        Dict mapping timeframe to whether charges match
    """
    result = {}
    all_tfs = set(charge1.keys()) | set(charge2.keys())
    
    for tf in all_tfs:
        c1 = charge1.get(tf, Charge.NEUTRAL)
        c2 = charge2.get(tf, Charge.NEUTRAL)
        result[tf] = (c1 == c2)
    
    return result
