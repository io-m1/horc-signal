from dataclasses import dataclass
from typing import Optional
from enum import Enum

from src.core.coordinate_engine import Coordinate
from src.core.divergence_engine import DivergenceEngine, DivergenceResult

class AbsorptionType(Enum):
    NONE = "none"               # No absorption
    INTERNAL = "internal"       # Trend continuation
    EXTERNAL = "external"       # Trend reversal
    EXHAUSTION = "exhaustion"   # Passive overwhelms aggressor

@dataclass(frozen=True)
class AbsorptionResult:
    divergence: DivergenceResult
    absorption_type: AbsorptionType
    absorption_strength: float
    target_coordinate: Optional[Coordinate]
    is_reversal_signal: bool
    is_continuation_signal: bool

class AbsorptionEngine:
    MIN_DIVERGENCE_THRESHOLD = 0.5
    
    @staticmethod
    def analyze_absorption(
        passive: Coordinate,
        aggressor: Coordinate,
        external_target: Optional[Coordinate] = None,
        passive_volume: float = 1.0,
        aggressor_volume: float = 1.0
    ) -> AbsorptionResult:
        divergence = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        if divergence.divergence_score < AbsorptionEngine.MIN_DIVERGENCE_THRESHOLD:
            return AbsorptionResult(
                divergence=divergence,
                absorption_type=AbsorptionType.NONE,
                absorption_strength=0.0,
                target_coordinate=None,
                is_reversal_signal=False,
                is_continuation_signal=False
            )
        
        absorption_strength = AbsorptionEngine._calculate_absorption_strength(
            divergence_score=divergence.divergence_score,
            passive_volume=passive_volume,
            aggressor_volume=aggressor_volume
        )
        
        absorption_type = AbsorptionEngine._determine_absorption_type(
            passive=passive,
            aggressor=aggressor,
            external_target=external_target,
            passive_volume=passive_volume,
            aggressor_volume=aggressor_volume
        )
        
        is_reversal = absorption_type in [AbsorptionType.EXTERNAL, AbsorptionType.EXHAUSTION]
        is_continuation = absorption_type == AbsorptionType.INTERNAL
        
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
        return passive_volume > aggressor_volume
    
    @staticmethod
    def is_internal_absorption(
        passive: Coordinate,
        aggressor: Coordinate,
        external_target: Optional[Coordinate]
    ) -> bool:
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
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor,
            external_target=external_target
        )
        return result.absorption_type == AbsorptionType.EXTERNAL
    
    @staticmethod
    def _calculate_absorption_strength(
        divergence_score: float,
        passive_volume: float,
        aggressor_volume: float
    ) -> float:
        if passive_volume <= 0 or aggressor_volume <= 0:
            return divergence_score
        
        total_volume = passive_volume + aggressor_volume
        volume_ratio = passive_volume / total_volume
        
        return divergence_score * volume_ratio
    
    @staticmethod
    def _determine_absorption_type(
        passive: Coordinate,
        aggressor: Coordinate,
        external_target: Optional[Coordinate],
        passive_volume: float,
        aggressor_volume: float
    ) -> AbsorptionType:
        if AbsorptionEngine.is_exhaustion_absorption(
            passive, aggressor, passive_volume, aggressor_volume
        ):
            return AbsorptionType.EXHAUSTION
        
        if external_target is not None:
            if AbsorptionEngine._is_moving_toward(aggressor, external_target):
                return AbsorptionType.EXTERNAL
        
        return AbsorptionType.INTERNAL
    
    @staticmethod
    def _identify_target(
        absorption_type: AbsorptionType,
        passive: Coordinate,
        aggressor: Coordinate,
        external_target: Optional[Coordinate]
    ) -> Optional[Coordinate]:
        if absorption_type == AbsorptionType.EXTERNAL:
            return external_target
        elif absorption_type == AbsorptionType.EXHAUSTION:
            return None  # Would need historical context
        elif absorption_type == AbsorptionType.INTERNAL:
            return None  # Would need future projections
        else:
            return None
    
    @staticmethod
    def _is_moving_toward(current: Coordinate, target: Coordinate) -> bool:
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
        
        return matches / total > 0.5
