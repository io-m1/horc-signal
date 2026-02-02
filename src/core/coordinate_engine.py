"""
Coordinate Engine — PHASE 1.5 (Multi-TF State Vectors)

DOCTRINE:
    "Only timeframes that exist at the moment of formation are included."
    "Once a timeframe closes, its charge state is immutable."

PURPOSE:
    Encodes multi-timeframe state vectors at price levels.
    Enables precise zone identification via coordinate comparison.

CRITICAL RULE:
    - Coordinate = ordered participant state across ACTIVE timeframes
    - HVO Rule: Only TFs that exist at formation are included
    - No retroactive changes (immutability)

COORDINATE FORMAT:
    (M±, W±, D±, S±)
    
    Example: (M−, W+, D+, S−)
    - Monthly: seller-born
    - Weekly: buyer-born
    - Daily: buyer-born
    - Session: seller-born

INTEGRATION:
    - Depends on: ParticipantEngine (WHO), FlipEngine (WHEN), ChargeEngine (+/−)
    - Feeds into: Liquidity Registration (marking zones with coordinates)
    - Used by: Zone targeting, divergence detection
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .participant_engine import ParticipantType
from .flip_engine import TimeframeType, FlipPoint
from .charge_engine import ChargedLevel, Charge


@dataclass(frozen=True)
class Coordinate:
    """
    Multi-timeframe state vector for a price level.
    
    Encodes participant charge across all active timeframes at formation.
    Immutable once assigned (no retroactive changes).
    """
    price: float
    timestamp: int
    
    # Charge state per timeframe (ordered M → W → D → S)
    M: Optional[int] = None     # Monthly charge (+1, -1, None)
    W: Optional[int] = None     # Weekly charge
    D: Optional[int] = None     # Daily charge
    S: Optional[int] = None     # Session charge
    
    # Metadata
    active_tfs: Tuple[str, ...] = ()    # Which TFs were active at formation
    is_high: bool = False               # True if swing high, False if low
    
    @property
    def label(self) -> str:
        """
        Get coordinate as formatted string.
        
        Example: "(M−, W+, D+, S−)"
        """
        parts = []
        for tf in ['M', 'W', 'D', 'S']:
            charge = getattr(self, tf)
            if charge is not None:
                symbol = Charge.to_symbol(charge)
                parts.append(f"{tf}{symbol}")
        
        if not parts:
            return "()"
        
        return f"({', '.join(parts)})"
    
    @property
    def vector(self) -> Tuple[Optional[int], ...]:
        """Get coordinate as numeric vector (M, W, D, S)."""
        return (self.M, self.W, self.D, self.S)
    
    def matches(self, other: 'Coordinate', strict: bool = True) -> bool:
        """
        Check if this coordinate matches another.
        
        Args:
            other: Coordinate to compare against
            strict: If True, all TFs must match. If False, only active TFs.
        
        Returns:
            True if coordinates match
        """
        if strict:
            return self.vector == other.vector
        else:
            # Only compare active TFs
            for tf in self.active_tfs:
                if getattr(self, tf) != getattr(other, tf):
                    return False
            return True
    
    def get_divergence_tfs(self, other: 'Coordinate') -> List[str]:
        """
        Get list of timeframes where charges diverge.
        
        Args:
            other: Coordinate to compare against
        
        Returns:
            List of TF strings where charges differ
        """
        divergent = []
        for tf in ['M', 'W', 'D', 'S']:
            self_charge = getattr(self, tf)
            other_charge = getattr(other, tf)
            
            # Only compare if both have values
            if self_charge is not None and other_charge is not None:
                if self_charge != other_charge:
                    divergent.append(tf)
        
        return divergent


class CoordinateEngine:
    """
    Builds multi-timeframe state vectors (coordinates) for price levels.
    
    CRITICAL RULES:
    1. HVO Rule: Only TFs that exist at formation are included
    2. Coordinates are IMMUTABLE once assigned
    3. No retroactive recalculation
    4. State vectors enable precise zone targeting
    
    WORKFLOW:
    1. Level forms (high or low)
    2. Determine which TFs are active at formation time
    3. Query charge state from ChargeEngine for each active TF
    4. Build coordinate tuple (M±, W±, D±, S±)
    5. Coordinate is immutable forever
    """
    
    def __init__(self):
        """Initialize coordinate engine."""
        self._coordinates: List[Coordinate] = []
    
    def build_coordinate(
        self,
        price: float,
        timestamp: int,
        is_high: bool,
        charged_levels: Dict[str, ChargedLevel]
    ) -> Coordinate:
        """
        Build coordinate from charged levels.
        
        Args:
            price: Price of the level
            timestamp: Formation time
            is_high: True if swing high, False if swing low
            charged_levels: Dict mapping TF → ChargedLevel for this price
        
        Returns:
            Coordinate with multi-TF state vector
        """
        # Extract charges from charged levels
        M_charge = charged_levels.get('M').charge if 'M' in charged_levels else None
        W_charge = charged_levels.get('W').charge if 'W' in charged_levels else None
        D_charge = charged_levels.get('D').charge if 'D' in charged_levels else None
        S_charge = charged_levels.get('S').charge if 'S' in charged_levels else None
        
        # Determine active TFs (only those with charges)
        active_tfs = tuple([
            tf for tf in ['M', 'W', 'D', 'S']
            if charged_levels.get(tf) is not None
        ])
        
        coordinate = Coordinate(
            price=price,
            timestamp=timestamp,
            M=M_charge,
            W=W_charge,
            D=D_charge,
            S=S_charge,
            active_tfs=active_tfs,
            is_high=is_high,
        )
        
        # Store for history
        self._coordinates.append(coordinate)
        
        return coordinate
    
    def build_from_charges(
        self,
        price: float,
        timestamp: int,
        is_high: bool,
        charges: Dict[str, int]
    ) -> Coordinate:
        """
        Build coordinate directly from charge dict.
        
        Args:
            price: Price of the level
            timestamp: Formation time
            is_high: True if swing high, False if swing low
            charges: Dict mapping TF → charge value
        
        Returns:
            Coordinate with multi-TF state vector
        """
        active_tfs = tuple(charges.keys())
        
        return Coordinate(
            price=price,
            timestamp=timestamp,
            M=charges.get('M'),
            W=charges.get('W'),
            D=charges.get('D'),
            S=charges.get('S'),
            active_tfs=active_tfs,
            is_high=is_high,
        )
    
    def find_matching_coordinates(
        self,
        target: Coordinate,
        strict: bool = True
    ) -> List[Coordinate]:
        """
        Find all coordinates matching the target.
        
        Args:
            target: Target coordinate to match
            strict: If True, all TFs must match. If False, only active TFs.
        
        Returns:
            List of matching coordinates
        """
        matches = []
        for coord in self._coordinates:
            if coord.matches(target, strict=strict):
                matches.append(coord)
        return matches
    
    def get_all_coordinates(self) -> List[Coordinate]:
        """Get all registered coordinates."""
        return self._coordinates.copy()


class HVOValidator:
    """
    Validates Highest Volume Open (HVO) rule.
    
    "Only timeframes that exist at the moment of formation are included."
    """
    
    @staticmethod
    def get_active_timeframes(
        timestamp: int,
        session_start: int,
        day_start: int,
        week_start: int,
        month_start: int
    ) -> List[str]:
        """
        Determine which timeframes are active at given timestamp.
        
        HVO RULE LOGIC:
        - Same session → S only
        - Same day, different session → S, D
        - Same week, different day → S, D, W
        - Different month → S, D, W, M
        
        Args:
            timestamp: Current timestamp
            session_start: Start of current session
            day_start: Start of current day
            week_start: Start of current week
            month_start: Start of current month
        
        Returns:
            List of active TF strings
        """
        active = ['S']  # Session always active
        
        # Check if day has opened (different session)
        if timestamp >= day_start and day_start > session_start:
            active.append('D')
        
        # Check if week has opened (different day)
        if timestamp >= week_start and week_start > day_start:
            active.append('W')
        
        # Check if month has opened (different week)
        if timestamp >= month_start and month_start > week_start:
            active.append('M')
        
        return active
    
    @staticmethod
    def validate_coordinate_tfs(
        coordinate: Coordinate,
        expected_tfs: List[str]
    ) -> bool:
        """
        Validate that coordinate only includes expected TFs.
        
        Args:
            coordinate: Coordinate to validate
            expected_tfs: Expected active TFs at formation
        
        Returns:
            True if coordinate TFs match expected
        """
        coord_tfs = set(coordinate.active_tfs)
        expected_tfs_set = set(expected_tfs)
        
        return coord_tfs == expected_tfs_set


class CoordinateComparator:
    """
    Compares coordinates for divergence and targeting.
    """
    
    @staticmethod
    def calculate_divergence_score(coord1: Coordinate, coord2: Coordinate) -> float:
        """
        Calculate divergence score between two coordinates.
        
        Score = number of divergent TFs / number of comparable TFs
        
        Args:
            coord1: First coordinate
            coord2: Second coordinate
        
        Returns:
            Divergence score (0.0 = identical, 1.0 = all divergent)
        """
        divergent_tfs = coord1.get_divergence_tfs(coord2)
        
        # Count comparable TFs (both have values)
        comparable = 0
        for tf in ['M', 'W', 'D', 'S']:
            if getattr(coord1, tf) is not None and getattr(coord2, tf) is not None:
                comparable += 1
        
        if comparable == 0:
            return 0.0
        
        return len(divergent_tfs) / comparable
    
    @staticmethod
    def find_highest_divergent_tf(coord1: Coordinate, coord2: Coordinate) -> Optional[str]:
        """
        Find highest timeframe where coordinates diverge.
        
        Returns:
            Highest divergent TF (M > W > D > S), or None if no divergence
        """
        divergent = coord1.get_divergence_tfs(coord2)
        
        if not divergent:
            return None
        
        # Return highest TF (M > W > D > S)
        tf_priority = ['M', 'W', 'D', 'S']
        for tf in tf_priority:
            if tf in divergent:
                return tf
        
        return None
    
    @staticmethod
    def is_flip_coordinate(
        before_coord: Coordinate,
        after_coord: Coordinate,
        tf: str
    ) -> bool:
        """
        Check if a flip occurred on specific TF between two coordinates.
        
        Args:
            before_coord: Coordinate before potential flip
            after_coord: Coordinate after potential flip
            tf: Timeframe to check
        
        Returns:
            True if flip detected on that TF
        """
        before_charge = getattr(before_coord, tf)
        after_charge = getattr(after_coord, tf)
        
        if before_charge is None or after_charge is None:
            return False
        
        # Flip = charge reversal (+1 → -1 or -1 → +1)
        return before_charge != after_charge and abs(before_charge - after_charge) == 2


# Utility functions

def format_coordinate_comparison(coord1: Coordinate, coord2: Coordinate) -> str:
    """
    Format coordinate comparison for display.
    
    Example:
        Target:  (M−, W+, D+, S−)
        Current: (M−, W+, D−, S−)
                        ↑
                  Daily flip
    
    Args:
        coord1: First coordinate (e.g., target)
        coord2: Second coordinate (e.g., current)
    
    Returns:
        Formatted comparison string
    """
    divergent_tfs = coord1.get_divergence_tfs(coord2)
    
    result = f"Coord 1: {coord1.label}\n"
    result += f"Coord 2: {coord2.label}\n"
    
    if divergent_tfs:
        result += f"Divergent TFs: {', '.join(divergent_tfs)}"
    else:
        result += "No divergence (identical)"
    
    return result


def build_coordinate_from_participant_states(
    price: float,
    timestamp: int,
    is_high: bool,
    participants: Dict[str, ParticipantType]
) -> Coordinate:
    """
    Build coordinate from participant states.
    
    Args:
        price: Price level
        timestamp: Formation time
        is_high: True if high, False if low
        participants: Dict mapping TF → ParticipantType
    
    Returns:
        Coordinate with charges derived from participants
    """
    charges = {
        tf: Charge.from_participant(participant)
        for tf, participant in participants.items()
    }
    
    active_tfs = tuple(charges.keys())
    
    return Coordinate(
        price=price,
        timestamp=timestamp,
        M=charges.get('M'),
        W=charges.get('W'),
        D=charges.get('D'),
        S=charges.get('S'),
        active_tfs=active_tfs,
        is_high=is_high,
    )
