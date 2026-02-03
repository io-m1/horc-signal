from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime

from src.core.coordinate_engine import Coordinate

class LiquidityType(Enum):
    INTERNAL = "internal"   # Internal to current trend
    EXTERNAL = "external"   # External to current trend

class SessionType(Enum):
    FRANKFURT = "frankfurt"
    LONDON = "london"
    NEW_YORK = "new_york"
    ASIA = "asia"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class AOI:
    coordinate: Coordinate
    price: float
    liquidity_type: LiquidityType
    volume: float
    session: SessionType
    timestamp: datetime
    is_mitigated: bool = False
    mitigation_timestamp: Optional[datetime] = None
    target_coordinate: Optional[Coordinate] = None

@dataclass
class AOIRegistry:
    aois: List[AOI] = field(default_factory=list)
    active_aois: Dict[SessionType, List[AOI]] = field(default_factory=dict)
    
    def __post_init__(self):
        for session in SessionType:
            self.active_aois[session] = []

class AOIManager:
    def __init__(self):
        self.registry = AOIRegistry()
    
    def register_aoi(
        self,
        coordinate: Coordinate,
        price: float,
        liquidity_type: LiquidityType,
        volume: float,
        session: SessionType,
        target_coordinate: Optional[Coordinate] = None
    ) -> AOI:
        aoi = AOI(
            coordinate=coordinate,
            price=price,
            liquidity_type=liquidity_type,
            volume=volume,
            session=session,
            timestamp=datetime.now(),
            is_mitigated=False,
            mitigation_timestamp=None,
            target_coordinate=target_coordinate
        )
        
        self.registry.aois.append(aoi)
        self.registry.active_aois[session].append(aoi)
        
        return aoi
    
    def is_mitigated(self, aoi: AOI, current_price: float, tolerance: float = 0.0001) -> bool:
        if aoi.is_mitigated:
            return True
        
        price_diff = abs(current_price - aoi.price)
        threshold = aoi.price * tolerance
        
        return price_diff <= threshold
    
    def mark_mitigated(self, aoi: AOI) -> AOI:
        if aoi in self.registry.active_aois[aoi.session]:
            self.registry.active_aois[aoi.session].remove(aoi)
        
        mitigated_aoi = AOI(
            coordinate=aoi.coordinate,
            price=aoi.price,
            liquidity_type=aoi.liquidity_type,
            volume=aoi.volume,
            session=aoi.session,
            timestamp=aoi.timestamp,
            is_mitigated=True,
            mitigation_timestamp=datetime.now(),
            target_coordinate=aoi.target_coordinate
        )
        
        if aoi in self.registry.aois:
            idx = self.registry.aois.index(aoi)
            self.registry.aois[idx] = mitigated_aoi
        
        return mitigated_aoi
    
    def get_active_aois(
        self,
        session: Optional[SessionType] = None,
        liquidity_type: Optional[LiquidityType] = None
    ) -> List[AOI]:
        if session is not None:
            aois = self.registry.active_aois[session]
        else:
            aois = []
            for session_aois in self.registry.active_aois.values():
                aois.extend(session_aois)
        
        if liquidity_type is not None:
            aois = [aoi for aoi in aois if aoi.liquidity_type == liquidity_type]
        
        return aois
    
    def get_highest_volume_aoi(
        self,
        session: Optional[SessionType] = None,
        liquidity_type: Optional[LiquidityType] = None
    ) -> Optional[AOI]:
        aois = self.get_active_aois(session=session, liquidity_type=liquidity_type)
        
        if not aois:
            return None
        
        return max(aois, key=lambda aoi: aoi.volume)
    
    def get_all_aois(self) -> List[AOI]:
        return self.registry.aois.copy()
    
    def get_session_chain(self) -> List[SessionType]:
        return [
            SessionType.FRANKFURT,
            SessionType.LONDON,
            SessionType.NEW_YORK,
            SessionType.ASIA
        ]
    
    def clear_session(self, session: SessionType):
        self.registry.active_aois[session] = []
    
    def clear_all(self):
        self.registry.aois.clear()
        for session in SessionType:
            self.registry.active_aois[session] = []
