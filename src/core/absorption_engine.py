"""
Absorption Engine — PHASE 1.75

Determines absorption type (internal vs external) and calculates absorption strength
based on divergence patterns and liquidity targets.

DOCTRINE:
    "Exhaustion absorption = passive overwhelms aggressor → reversal"
    "Internal absorption → trend continuation"
    "External absorption → trend reversal"

PURPOSE:
    - Classify absorption as internal or external
    - Calculate absorption strength
    - Identify liquidity targets
    - Determine reversal vs continuation signals

INTEGRATION:
    DivergenceEngine → AbsorptionEngine → Opposition → Quadrant
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

from src.core.coordinate_engine import Coordinate
from src.core.divergence_engine import DivergenceEngine, DivergenceResult


class AbsorptionType(Enum):
    """Type of absorption detected."""
    NONE = "none"               # No absorption
    INTERNAL = "internal"       # Trend continuation
    EXTERNAL = "external"       # Trend reversal
    EXHAUSTION = "exhaustion"   # Passive overwhelms aggressor


@dataclass(frozen=True)
class AbsorptionResult:
    """
    Result of absorption analysis.
    
    Attributes:
        divergence: Underlying divergence result
        absorption_type: Classification (internal/external/exhaustion/none)
        absorption_strength: Quantitative strength (0.0-1.0)
        target_coordinate: Expected price target (if identifiable)
        is_reversal_signal: Whether this signals reversal
        is_continuation_signal: Whether this signals continuation
    """
    divergence: DivergenceResult
    absorption_type: AbsorptionType
    absorption_strength: float
    target_coordinate: Optional[Coordinate]
    is_reversal_signal: bool
    is_continuation_signal: bool


class AbsorptionEngine:
    """
    Analyzes absorption patterns to determine market direction.
    
    RULES:
        1. Exhaustion absorption → passive strength > aggressor strength
        2. Internal absorption → targets internal liquidity (continuation)
        3. External absorption → targets external liquidity (reversal)
        4. Absorption strength ≥ 0.5 required for valid signal
        5. Full divergence (1.0) → maximum absorption strength
    
    IMMUTABILITY:
        AbsorptionResult is frozen (cannot be modified after creation)
    """
    
    # Minimum divergence score for valid absorption
    MIN_DIVERGENCE_THRESHOLD = 0.5
    
    @staticmethod
    def analyze_absorption(
        passive: Coordinate,
        aggressor: Coordinate,
        external_target: Optional[Coordinate] = None,
        passive_volume: float = 1.0,
        aggressor_volume: float = 1.0
    ) -> AbsorptionResult:
        """
        Analyze absorption type and strength between passive and aggressor.
        
        Args:
            passive: Historical/passive coordinate (AOI level)
            aggressor: Current/aggressive coordinate (present momentum)
            external_target: Optional external liquidity target
            passive_volume: Volume at passive level (default 1.0)
            aggressor_volume: Volume in aggressive move (default 1.0)
        
        Returns:
            AbsorptionResult with complete analysis
        
        Algorithm:
            1. Calculate divergence via DivergenceEngine
            2. Check if divergence meets threshold (≥0.5)
            3. Compare passive vs aggressor strength (volume-weighted)
            4. If passive > aggressor → exhaustion absorption
            5. If external target exists → external absorption (reversal)
            6. Otherwise → internal absorption (continuation)
        
        Example:
            >>> passive = Coordinate(price=100, D=-1, S=-1)
            >>> aggressor = Coordinate(price=105, D=+1, S=+1)
            >>> external = Coordinate(price=110, D=+1, S=+1)
            >>> result = AbsorptionEngine.analyze_absorption(
            ...     passive, aggressor, external, 
            ...     passive_volume=1000, aggressor_volume=500
            ... )
            >>> result.absorption_type
            AbsorptionType.EXTERNAL  # Reversal signal
        """
        # Step 1: Calculate divergence
        divergence = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        # Step 2: Check threshold
        if divergence.divergence_score < AbsorptionEngine.MIN_DIVERGENCE_THRESHOLD:
            # Not enough divergence for absorption
            return AbsorptionResult(
                divergence=divergence,
                absorption_type=AbsorptionType.NONE,
                absorption_strength=0.0,
                target_coordinate=None,
                is_reversal_signal=False,
                is_continuation_signal=False
            )
        
        # Step 3: Calculate absorption strength
        absorption_strength = AbsorptionEngine._calculate_absorption_strength(
            divergence_score=divergence.divergence_score,
            passive_volume=passive_volume,
            aggressor_volume=aggressor_volume
        )
        
        # Step 4: Determine absorption type
        absorption_type = AbsorptionEngine._determine_absorption_type(
            passive=passive,
            aggressor=aggressor,
            external_target=external_target,
            passive_volume=passive_volume,
            aggressor_volume=aggressor_volume
        )
        
        # Step 5: Determine signals
        is_reversal = absorption_type in [AbsorptionType.EXTERNAL, AbsorptionType.EXHAUSTION]
        is_continuation = absorption_type == AbsorptionType.INTERNAL
        
        # Step 6: Identify target
        target = AbsorptionEngine._identify_target(
            absorption_type=absorption_type,
            passive=passive,
            aggressor=aggressor,
            external_target=external_target
        )
        
        return AbsorptionResult(
            divergence=divergence,
            absorption_type=absorption_type,
            absorption_strength=absorption_strength,
            target_coordinate=target,
            is_reversal_signal=is_reversal,
            is_continuation_signal=is_continuation
        )
    
    @staticmethod
    def is_exhaustion_absorption(
        passive: Coordinate,
        aggressor: Coordinate,
        passive_volume: float,
        aggressor_volume: float
    ) -> bool:
        """
        Check if passive liquidity overwhelms aggressor (exhaustion).
        
        Args:
            passive: Passive coordinate
            aggressor: Aggressor coordinate
            passive_volume: Volume at passive level
            aggressor_volume: Volume in aggressive move
        
        Returns:
            True if passive strength > aggressor strength
        
        Rule:
            Exhaustion = passive_volume > aggressor_volume
        """
        return passive_volume > aggressor_volume
    
    @staticmethod
    def is_internal_absorption(
        passive: Coordinate,
        aggressor: Coordinate,
        external_target: Optional[Coordinate]
    ) -> bool:
        """
        Check if absorption targets internal liquidity (continuation).
        
        Args:
            passive: Passive coordinate
            aggressor: Aggressor coordinate
            external_target: External liquidity target (if exists)
        
        Returns:
            True if no external target (internal continuation)
        """
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor,
            external_target=external_target
        )
        return result.absorption_type == AbsorptionType.INTERNAL
    
    @staticmethod
    def is_external_absorption(
        passive: Coordinate,
        aggressor: Coordinate,
        external_target: Optional[Coordinate]
    ) -> bool:
        """
        Check if absorption targets external liquidity (reversal).
        
        Args:
            passive: Passive coordinate
            aggressor: Aggressor coordinate
            external_target: External liquidity target
        
        Returns:
            True if external target exists (reversal signal)
        """
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor,
            external_target=external_target
        )
        return result.absorption_type == AbsorptionType.EXTERNAL
    
    # ==================== PRIVATE HELPERS ====================
    
    @staticmethod
    def _calculate_absorption_strength(
        divergence_score: float,
        passive_volume: float,
        aggressor_volume: float
    ) -> float:
        """
        Calculate absorption strength combining divergence and volume.
        
        Args:
            divergence_score: Divergence ratio (0.0-1.0)
            passive_volume: Volume at passive level
            aggressor_volume: Volume in aggressive move
        
        Returns:
            Absorption strength (0.0-1.0)
        
        Algorithm:
            strength = divergence_score * (passive_volume / total_volume)
            
            Higher passive volume → stronger absorption
            Higher divergence → stronger absorption
        """
        if passive_volume <= 0 or aggressor_volume <= 0:
            return divergence_score
        
        total_volume = passive_volume + aggressor_volume
        volume_ratio = passive_volume / total_volume
        
        # Combine divergence with volume weighting
        return divergence_score * volume_ratio
    
    @staticmethod
    def _determine_absorption_type(
        passive: Coordinate,
        aggressor: Coordinate,
        external_target: Optional[Coordinate],
        passive_volume: float,
        aggressor_volume: float
    ) -> AbsorptionType:
        """
        Determine type of absorption based on conditions.
        
        Algorithm:
            1. If passive_volume > aggressor_volume → EXHAUSTION
            2. Else if external_target exists → EXTERNAL
            3. Else → INTERNAL
        """
        # Check exhaustion first
        if AbsorptionEngine.is_exhaustion_absorption(
            passive, aggressor, passive_volume, aggressor_volume
        ):
            return AbsorptionType.EXHAUSTION
        
        # Check for external target
        if external_target is not None:
            # If external target has same charges as aggressor → external absorption
            if AbsorptionEngine._is_moving_toward(aggressor, external_target):
                return AbsorptionType.EXTERNAL
        
        # Default to internal absorption (continuation)
        return AbsorptionType.INTERNAL
    
    @staticmethod
    def _identify_target(
        absorption_type: AbsorptionType,
        passive: Coordinate,
        aggressor: Coordinate,
        external_target: Optional[Coordinate]
    ) -> Optional[Coordinate]:
        """
        Identify target coordinate based on absorption type.
        
        Returns:
            Target coordinate (if identifiable)
        """
        if absorption_type == AbsorptionType.EXTERNAL:
            return external_target
        elif absorption_type == AbsorptionType.EXHAUSTION:
            # Reversal toward opposite of aggressor
            return None  # Would need historical context
        elif absorption_type == AbsorptionType.INTERNAL:
            # Continuation in aggressor direction
            return None  # Would need future projections
        else:
            return None
    
    @staticmethod
    def _is_moving_toward(current: Coordinate, target: Coordinate) -> bool:
        """
        Check if current coordinate is moving toward target.
        
        Algorithm:
            Moving toward = majority of charges match
        """
        matches = 0
        total = 0
        
        for tf in ['M', 'W', 'D', 'S']:
            current_charge = getattr(current, tf)
            target_charge = getattr(target, tf)
            
            if current_charge is not None and target_charge is not None:
                total += 1
                if current_charge == target_charge:
                    matches += 1
        
        if total == 0:
            return False
        
        # Moving toward if >50% charges match
        return matches / total > 0.5
