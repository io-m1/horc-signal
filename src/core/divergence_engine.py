"""
Divergence Engine — PHASE 1.75

Compares passive coordinates (historical levels) with aggressor coordinates
(current market momentum) to detect divergence patterns.

DOCTRINE:
    "Divergence is when present momentum (aggressors) and historical levels (passive)
     show opposite signs."

PURPOSE:
    - Calculate divergence scores between coordinate pairs
    - Identify full divergence (all TFs opposite)
    - Detect partial divergence (some TFs opposite)
    - Provide quantitative divergence strength

INTEGRATION:
    CoordinateEngine → DivergenceEngine → AbsorptionEngine → Opposition
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from src.core.coordinate_engine import Coordinate


class DivergenceType(Enum):
    """Type of divergence detected."""
    NONE = "none"           # No divergence
    PARTIAL = "partial"     # Some TFs diverge
    FULL = "full"          # All TFs diverge


@dataclass(frozen=True)
class DivergenceResult:
    """
    Result of divergence analysis between passive and aggressor coordinates.
    
    Attributes:
        passive: Historical/passive coordinate
        aggressor: Current/aggressive coordinate
        divergent_tfs: List of timeframes showing divergence
        divergence_score: Ratio of divergent TFs (0.0-1.0)
        divergence_type: Classification of divergence strength
        comparable_tfs: Number of TFs that could be compared
    """
    passive: Coordinate
    aggressor: Coordinate
    divergent_tfs: tuple[str, ...]
    divergence_score: float
    divergence_type: DivergenceType
    comparable_tfs: int


class DivergenceEngine:
    """
    Detects divergence between passive and aggressor coordinates.
    
    RULES:
        1. Divergence = opposite charge signs on same TF
        2. Full divergence = ALL comparable TFs opposite
        3. Partial divergence = SOME comparable TFs opposite
        4. No divergence = NO opposite TFs
        5. Score = divergent_tfs / comparable_tfs
    
    IMMUTABILITY:
        DivergenceResult is frozen (cannot be modified after creation)
    """
    
    @staticmethod
    def calculate_divergence(
        passive: Coordinate,
        aggressor: Coordinate
    ) -> DivergenceResult:
        """
        Calculate divergence between passive and aggressor coordinates.
        
        Args:
            passive: Historical/passive coordinate (resting orders)
            aggressor: Current/aggressive coordinate (present momentum)
        
        Returns:
            DivergenceResult with complete analysis
        
        Algorithm:
            1. Find TFs active in BOTH coordinates
            2. Compare charges on each common TF
            3. Divergence = opposite signs (+/− or −/+)
            4. Calculate score = divergent / comparable
            5. Classify as NONE / PARTIAL / FULL
        
        Example:
            >>> passive = Coordinate(price=100, M=None, W=-1, D=-1, S=-1)
            >>> aggressor = Coordinate(price=105, M=None, W=+1, D=+1, S=+1)
            >>> result = DivergenceEngine.calculate_divergence(passive, aggressor)
            >>> result.divergence_type
            DivergenceType.FULL  # All TFs (W, D, S) opposite
        """
        # Get TFs that exist in both coordinates
        comparable_tfs = DivergenceEngine._get_comparable_tfs(passive, aggressor)
        
        if not comparable_tfs:
            # No TFs to compare
            return DivergenceResult(
                passive=passive,
                aggressor=aggressor,
                divergent_tfs=tuple(),
                divergence_score=0.0,
                divergence_type=DivergenceType.NONE,
                comparable_tfs=0
            )
        
        # Check each TF for divergence
        divergent_tfs = []
        for tf in comparable_tfs:
            passive_charge = getattr(passive, tf)
            aggressor_charge = getattr(aggressor, tf)
            
            # Divergence = opposite signs
            if DivergenceEngine._is_divergent(passive_charge, aggressor_charge):
                divergent_tfs.append(tf)
        
        # Calculate score
        divergence_score = len(divergent_tfs) / len(comparable_tfs)
        
        # Classify divergence type
        if divergence_score == 0.0:
            divergence_type = DivergenceType.NONE
        elif divergence_score == 1.0:
            divergence_type = DivergenceType.FULL
        else:
            divergence_type = DivergenceType.PARTIAL
        
        return DivergenceResult(
            passive=passive,
            aggressor=aggressor,
            divergent_tfs=tuple(divergent_tfs),
            divergence_score=divergence_score,
            divergence_type=divergence_type,
            comparable_tfs=len(comparable_tfs)
        )
    
    @staticmethod
    def is_full_divergence(passive: Coordinate, aggressor: Coordinate) -> bool:
        """
        Check if all comparable TFs show divergence.
        
        Args:
            passive: Historical coordinate
            aggressor: Current coordinate
        
        Returns:
            True if ALL comparable TFs have opposite charges
        
        Example:
            >>> passive = Coordinate(price=100, W=-1, D=-1, S=-1)
            >>> aggressor = Coordinate(price=105, W=+1, D=+1, S=+1)
            >>> DivergenceEngine.is_full_divergence(passive, aggressor)
            True
        """
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        return result.divergence_type == DivergenceType.FULL
    
    @staticmethod
    def get_divergence_score(passive: Coordinate, aggressor: Coordinate) -> float:
        """
        Get divergence score as ratio (0.0 - 1.0).
        
        Args:
            passive: Historical coordinate
            aggressor: Current coordinate
        
        Returns:
            Ratio of divergent TFs to comparable TFs
            0.0 = no divergence
            1.0 = full divergence
        
        Example:
            >>> passive = Coordinate(price=100, W=-1, D=-1, S=+1)
            >>> aggressor = Coordinate(price=105, W=+1, D=+1, S=+1)
            >>> DivergenceEngine.get_divergence_score(passive, aggressor)
            0.666...  # 2 of 3 TFs divergent
        """
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        return result.divergence_score
    
    @staticmethod
    def get_divergent_timeframes(
        passive: Coordinate,
        aggressor: Coordinate
    ) -> List[str]:
        """
        Get list of timeframes showing divergence.
        
        Args:
            passive: Historical coordinate
            aggressor: Current coordinate
        
        Returns:
            List of TF strings (e.g., ['W', 'D', 'S'])
        
        Example:
            >>> passive = Coordinate(price=100, W=-1, D=-1, S=+1)
            >>> aggressor = Coordinate(price=105, W=+1, D=+1, S=+1)
            >>> DivergenceEngine.get_divergent_timeframes(passive, aggressor)
            ['W', 'D']
        """
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        return list(result.divergent_tfs)
    
    # ==================== PRIVATE HELPERS ====================
    
    @staticmethod
    def _get_comparable_tfs(coord1: Coordinate, coord2: Coordinate) -> List[str]:
        """
        Get timeframes that exist in BOTH coordinates.
        
        Returns:
            List of TF strings where both have non-None charges
        """
        comparable = []
        for tf in ['M', 'W', 'D', 'S']:
            charge1 = getattr(coord1, tf)
            charge2 = getattr(coord2, tf)
            
            # Both must have non-None charges
            if charge1 is not None and charge2 is not None:
                comparable.append(tf)
        
        return comparable
    
    @staticmethod
    def _is_divergent(charge1: Optional[int], charge2: Optional[int]) -> bool:
        """
        Check if two charges show divergence (opposite signs).
        
        Args:
            charge1: First charge (+1 or -1 or None)
            charge2: Second charge (+1 or -1 or None)
        
        Returns:
            True if charges are opposite (+/− or −/+)
            False if same sign, zero, or None
        
        Algorithm:
            Divergence = charge1 * charge2 < 0
            (Negative product means opposite signs)
        """
        if charge1 is None or charge2 is None:
            return False
        
        if charge1 == 0 or charge2 == 0:
            return False
        
        # Opposite signs → negative product
        return charge1 * charge2 < 0
