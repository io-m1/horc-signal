from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from datetime import datetime
from enum import Enum

from src.core.coordinate_engine import Coordinate
from src.core.participant_engine import ParticipantType
from src.core.divergence_engine import DivergenceEngine
from src.core.absorption_engine import AbsorptionEngine, AbsorptionType
from src.core.aoi_manager import AOI, LiquidityType

class ZoneStatus(Enum):
    ACTIVE = "active"           # Valid, unmitigated
    MITIGATED = "mitigated"     # Price returned, invalidated
    TARGETED = "targeted"       # Currently being targeted
    DEFENDED = "defended"       # Defended by participants

@dataclass(frozen=True)
class LiquidityZone:
    coordinate: Coordinate
    price: float
    participant: ParticipantType
    liquidity_type: LiquidityType
    volume: float
    timestamp: datetime
    status: ZoneStatus
    aoi: Optional[AOI] = None
    is_first_valid: bool = False

@dataclass
class ZoneHierarchy:
    first_valid: Optional[LiquidityZone] = None
    active_zones: List[LiquidityZone] = field(default_factory=list)
    mitigated_zones: List[LiquidityZone] = field(default_factory=list)
    zone_relationships: Dict[float, List[float]] = field(default_factory=dict)

class LiquidityRegistration:
    def __init__(self, mitigation_tolerance: float = 0.0001):
        self.hierarchy = ZoneHierarchy()
        self.mitigation_tolerance = mitigation_tolerance
        self.zones_by_price: Dict[float, LiquidityZone] = {}
    
    def register_zone(
        self,
        coordinate: Coordinate,
        price: float,
        participant: ParticipantType,
        liquidity_type: LiquidityType,
        volume: float,
        aoi: Optional[AOI] = None
    ) -> LiquidityZone:
        is_first_valid = self.hierarchy.first_valid is None
        
        zone = LiquidityZone(
            coordinate=coordinate,
            price=price,
            participant=participant,
            liquidity_type=liquidity_type,
            volume=volume,
            timestamp=datetime.now(),
            status=ZoneStatus.ACTIVE,
            aoi=aoi,
            is_first_valid=is_first_valid
        )
        
        if is_first_valid:
            self.hierarchy.first_valid = zone
        
        self.hierarchy.active_zones.append(zone)
        self.zones_by_price[price] = zone
        
        return zone
    
    def is_mitigated(self, zone: LiquidityZone, current_price: float) -> bool:
        if zone.status == ZoneStatus.MITIGATED:
            return True
        
        price_diff = abs(current_price - zone.price)
        threshold = zone.price * self.mitigation_tolerance
        
        return price_diff <= threshold
    
    def mark_mitigated(self, zone: LiquidityZone) -> LiquidityZone:
        if zone in self.hierarchy.active_zones:
            self.hierarchy.active_zones.remove(zone)
        
        mitigated_zone = LiquidityZone(
            coordinate=zone.coordinate,
            price=zone.price,
            participant=zone.participant,
            liquidity_type=zone.liquidity_type,
            volume=zone.volume,
            timestamp=zone.timestamp,
            status=ZoneStatus.MITIGATED,
            aoi=zone.aoi,
            is_first_valid=zone.is_first_valid
        )
        
        self.hierarchy.mitigated_zones.append(mitigated_zone)
        
        if zone.price in self.zones_by_price:
            self.zones_by_price[zone.price] = mitigated_zone
        
        return mitigated_zone
    
    def get_target_zones(
        self,
        current_coordinate: Coordinate,
        absorption_type: AbsorptionType,
        participant: Optional[ParticipantType] = None
    ) -> List[LiquidityZone]:
        targets = []
        
        for zone in self.hierarchy.active_zones:
            if zone.status == ZoneStatus.MITIGATED:
                continue
            
            if absorption_type == AbsorptionType.INTERNAL:
                if zone.liquidity_type == LiquidityType.INTERNAL:
                    targets.append(zone)
            
            elif absorption_type in [AbsorptionType.EXTERNAL, AbsorptionType.EXHAUSTION]:
                if zone.liquidity_type == LiquidityType.EXTERNAL:
                    if participant:
                        if zone.participant != participant:
                            targets.append(zone)
                    else:
                        targets.append(zone)
        
        targets.sort(key=lambda z: z.volume, reverse=True)
        
        return targets
    
    def get_first_valid_liquidity(self) -> Optional[LiquidityZone]:
        return self.hierarchy.first_valid
    
    def get_active_zones(
        self,
        liquidity_type: Optional[LiquidityType] = None,
        participant: Optional[ParticipantType] = None
    ) -> List[LiquidityZone]:
        zones = self.hierarchy.active_zones.copy()
        
        if liquidity_type:
            zones = [z for z in zones if z.liquidity_type == liquidity_type]
        
        if participant:
            zones = [z for z in zones if z.participant == participant]
        
        return zones
    
    def get_zone_by_price(self, price: float) -> Optional[LiquidityZone]:
        return self.zones_by_price.get(price)
    
    def update_zones(self, current_price: float):
        zones_to_mitigate = []
        
        for zone in self.hierarchy.active_zones:
            if self.is_mitigated(zone, current_price):
                zones_to_mitigate.append(zone)
        
        for zone in zones_to_mitigate:
            self.mark_mitigated(zone)
    
    def add_zone_relationship(self, zone1_price: float, zone2_price: float):
        if zone1_price not in self.hierarchy.zone_relationships:
            self.hierarchy.zone_relationships[zone1_price] = []
        
        if zone2_price not in self.hierarchy.zone_relationships[zone1_price]:
            self.hierarchy.zone_relationships[zone1_price].append(zone2_price)
    
    def get_related_zones(self, price: float) -> List[LiquidityZone]:
        related_prices = self.hierarchy.zone_relationships.get(price, [])
        return [self.zones_by_price[p] for p in related_prices if p in self.zones_by_price]
    
    def clear_all(self):
        self.hierarchy = ZoneHierarchy()
        self.zones_by_price.clear()
    
    def get_zone_count(self) -> Dict[str, int]:
        return {
            "active": len(self.hierarchy.active_zones),
            "mitigated": len(self.hierarchy.mitigated_zones),
            "total": len(self.zones_by_price)
        }
