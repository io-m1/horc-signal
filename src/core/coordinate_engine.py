from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .participant_engine import ParticipantType
from .flip_engine import TimeframeType, FlipPoint
from .charge_engine import ChargedLevel, Charge

@dataclass(frozen=True)
class Coordinate:
    price: float
    timestamp: int
    
    M: Optional[int] = None     # Monthly charge (+1, -1, None)
    W: Optional[int] = None     # Weekly charge
    D: Optional[int] = None     # Daily charge
    S: Optional[int] = None     # Session charge
    
    active_tfs: Tuple[str, ...] = ()    # Which TFs were active at formation
    is_high: bool = False               # True if swing high, False if low
    
    @property
    def label(self) -> str:
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
        return (self.M, self.W, self.D, self.S)
    
    def matches(self, other: 'Coordinate', strict: bool = True) -> bool:
        if strict:
            return self.vector == other.vector
        else:
            for tf in self.active_tfs:
                if getattr(self, tf) != getattr(other, tf):
                    return False
            return True
    
    def get_divergence_tfs(self, other: 'Coordinate') -> List[str]:
        divergent = []
        for tf in ['M', 'W', 'D', 'S']:
            self_charge = getattr(self, tf)
            other_charge = getattr(other, tf)
            
            if self_charge is not None and other_charge is not None:
                if self_charge != other_charge:
                    divergent.append(tf)
        
        return divergent

class CoordinateEngine:
    def __init__(self):
        self._coordinates: List[Coordinate] = []
    
    def build_coordinate(
        self,
        price: float,
        timestamp: int,
        is_high: bool,
        charged_levels: Dict[str, ChargedLevel]
    ) -> Coordinate:
        M_charge = charged_levels.get('M').charge if 'M' in charged_levels else None
        W_charge = charged_levels.get('W').charge if 'W' in charged_levels else None
        D_charge = charged_levels.get('D').charge if 'D' in charged_levels else None
        S_charge = charged_levels.get('S').charge if 'S' in charged_levels else None
        
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
        
        self._coordinates.append(coordinate)
        
        return coordinate
    
    def build_from_charges(
        self,
        price: float,
        timestamp: int,
        is_high: bool,
        charges: Dict[str, int]
    ) -> Coordinate:
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
        matches = []
        for coord in self._coordinates:
            if coord.matches(target, strict=strict):
                matches.append(coord)
        return matches
    
    def get_all_coordinates(self) -> List[Coordinate]:
        return self._coordinates.copy()

class HVOValidator:
    @staticmethod
    def get_active_timeframes(
        timestamp: int,
        session_start: int,
        day_start: int,
        week_start: int,
        month_start: int
    ) -> List[str]:
        active = ['S']  # Session always active
        
        if timestamp >= day_start and day_start > session_start:
            active.append('D')
        
        if timestamp >= week_start and week_start > day_start:
            active.append('W')
        
        if timestamp >= month_start and month_start > week_start:
            active.append('M')
        
        return active
    
    @staticmethod
    def validate_coordinate_tfs(
        coordinate: Coordinate,
        expected_tfs: List[str]
    ) -> bool:
        coord_tfs = set(coordinate.active_tfs)
        expected_tfs_set = set(expected_tfs)
        
        return coord_tfs == expected_tfs_set

class CoordinateComparator:
    @staticmethod
    def calculate_divergence_score(coord1: Coordinate, coord2: Coordinate) -> float:
        divergent_tfs = coord1.get_divergence_tfs(coord2)
        
        comparable = 0
        for tf in ['M', 'W', 'D', 'S']:
            if getattr(coord1, tf) is not None and getattr(coord2, tf) is not None:
                comparable += 1
        
        if comparable == 0:
            return 0.0
        
        return len(divergent_tfs) / comparable
    
    @staticmethod
    def find_highest_divergent_tf(coord1: Coordinate, coord2: Coordinate) -> Optional[str]:
        divergent = coord1.get_divergence_tfs(coord2)
        
        if not divergent:
            return None
        
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
        before_charge = getattr(before_coord, tf)
        after_charge = getattr(after_coord, tf)
        
        if before_charge is None or after_charge is None:
            return False
        
        return before_charge != after_charge and abs(before_charge - after_charge) == 2

def format_coordinate_comparison(coord1: Coordinate, coord2: Coordinate) -> str:
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
