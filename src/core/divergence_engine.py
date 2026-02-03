from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from src.core.coordinate_engine import Coordinate

class DivergenceType(Enum):
    NONE = "none"           # No divergence
    PARTIAL = "partial"     # Some TFs diverge
    FULL = "full"          # All TFs diverge

@dataclass(frozen=True)
class DivergenceResult:
    passive: Coordinate
    aggressor: Coordinate
    divergent_tfs: tuple[str, ...]
    divergence_score: float
    divergence_type: DivergenceType
    comparable_tfs: int

class DivergenceEngine:
    @staticmethod
    def calculate_divergence(
        passive: Coordinate,
        aggressor: Coordinate
    ) -> DivergenceResult:
        comparable_tfs = DivergenceEngine._get_comparable_tfs(passive, aggressor)
        
        if not comparable_tfs:
            return DivergenceResult(
                passive=passive,
                aggressor=aggressor,
                divergent_tfs=tuple(),
                divergence_score=0.0,
                divergence_type=DivergenceType.NONE,
                comparable_tfs=0
            )
        
        divergent_tfs = []
        for tf in comparable_tfs:
            passive_charge = getattr(passive, tf)
            aggressor_charge = getattr(aggressor, tf)
            
            if DivergenceEngine._is_divergent(passive_charge, aggressor_charge):
                divergent_tfs.append(tf)
        
        divergence_score = len(divergent_tfs) / len(comparable_tfs)
        
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
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        return result.divergence_type == DivergenceType.FULL
    
    @staticmethod
    def get_divergence_score(passive: Coordinate, aggressor: Coordinate) -> float:
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        return result.divergence_score
    
    @staticmethod
    def get_divergent_timeframes(
        passive: Coordinate,
        aggressor: Coordinate
    ) -> List[str]:
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        return list(result.divergent_tfs)
    
    @staticmethod
    def _get_comparable_tfs(coord1: Coordinate, coord2: Coordinate) -> List[str]:
        comparable = []
        for tf in ['M', 'W', 'D', 'S']:
            charge1 = getattr(coord1, tf)
            charge2 = getattr(coord2, tf)
            
            if charge1 is not None and charge2 is not None:
                comparable.append(tf)
        
        return comparable
    
    @staticmethod
    def _is_divergent(charge1: Optional[int], charge2: Optional[int]) -> bool:
        if charge1 is None or charge2 is None:
            return False
        
        if charge1 == 0 or charge2 == 0:
            return False
        
        return charge1 * charge2 < 0
