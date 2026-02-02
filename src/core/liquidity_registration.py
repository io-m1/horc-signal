"""
Liquidity Registration Engine — PHASE 2

Registers price levels with coordinates, tracks zone hierarchies, and manages
liquidity invalidation based on participant state and absorption patterns.

DOCTRINE:
    "Liquidity is a relationship, not a price."
    "First valid liquidity controls all others."
    "Continuation depends on reversal."

PURPOSE:
    - Register liquidity zones with multi-TF coordinates
    - Track zone invalidation via mitigation
    - Manage zone hierarchies (first valid controls)
    - Identify target zones based on absorption type

INTEGRATION:
    CoordinateEngine → DivergenceEngine → AbsorptionEngine → LiquidityRegistration → SignalIR
"""

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
    """Status of liquidity zone."""
    ACTIVE = "active"           # Valid, unmitigated
    MITIGATED = "mitigated"     # Price returned, invalidated
    TARGETED = "targeted"       # Currently being targeted
    DEFENDED = "defended"       # Defended by participants


@dataclass(frozen=True)
class LiquidityZone:
    """
    Registered liquidity zone with coordinate encoding.
    
    A zone represents a price level with specific participant state (coordinate)
    and liquidity characteristics. Zones are immutable once registered.
    
    Attributes:
        coordinate: Multi-TF state vector at formation
        price: Price level of zone
        participant: WHO created this zone (BUYER/SELLER)
        liquidity_type: INTERNAL or EXTERNAL
        volume: Volume at formation
        timestamp: When zone was registered
        status: Current zone status
        aoi: Associated Area of Interest (if any)
        is_first_valid: Is this the first valid liquidity?
    """
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
    """
    Tracks zone relationships and control.
    
    THREE LAWS enforcement:
    - LAW 1: Liquidity is a relationship
    - LAW 2: First valid liquidity controls all others
    - LAW 3: Continuation depends on reversal
    
    Attributes:
        first_valid: First valid liquidity zone (controls all)
        active_zones: Currently valid zones
        mitigated_zones: Invalidated zones
        zone_relationships: Map of zone to related zones
    """
    first_valid: Optional[LiquidityZone] = None
    active_zones: List[LiquidityZone] = field(default_factory=list)
    mitigated_zones: List[LiquidityZone] = field(default_factory=list)
    zone_relationships: Dict[float, List[float]] = field(default_factory=dict)


class LiquidityRegistration:
    """
    Manages liquidity zone registration and targeting.
    
    RULES:
        1. Zones registered with complete coordinates
        2. First valid zone controls all subsequent zones
        3. Mitigation = price returns within tolerance
        4. Internal zones target continuation
        5. External zones target reversal
        6. Zone hierarchy tracked via relationships
    
    USAGE:
        registry = LiquidityRegistration()
        
        # Register zone
        zone = registry.register_zone(
            coordinate=coord,
            price=100.0,
            participant=ParticipantType.BUYER,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0
        )
        
        # Check targeting
        targets = registry.get_target_zones(
            current_coordinate=current_coord,
            absorption_type=AbsorptionType.INTERNAL
        )
    """
    
    def __init__(self, mitigation_tolerance: float = 0.0001):
        """
        Initialize liquidity registration system.
        
        Args:
            mitigation_tolerance: Price tolerance for mitigation (0.01% default)
        """
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
        """
        Register a new liquidity zone.
        
        Args:
            coordinate: Multi-TF state vector
            price: Zone price level
            participant: WHO created zone
            liquidity_type: INTERNAL or EXTERNAL
            volume: Volume at formation
            aoi: Associated Area of Interest
        
        Returns:
            Newly registered LiquidityZone
        
        Algorithm:
            1. Create zone with coordinate
            2. Check if this is first valid zone
            3. Add to active zones
            4. Update hierarchy relationships
        """
        # Determine if first valid
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
        
        # Set first valid if none exists
        if is_first_valid:
            self.hierarchy.first_valid = zone
        
        # Add to active zones
        self.hierarchy.active_zones.append(zone)
        self.zones_by_price[price] = zone
        
        return zone
    
    def is_mitigated(self, zone: LiquidityZone, current_price: float) -> bool:
        """
        Check if zone has been mitigated by price return.
        
        Args:
            zone: Zone to check
            current_price: Current market price
        
        Returns:
            True if price within tolerance of zone price
        
        Algorithm:
            |current_price - zone_price| / zone_price <= tolerance
        """
        if zone.status == ZoneStatus.MITIGATED:
            return True
        
        price_diff = abs(current_price - zone.price)
        threshold = zone.price * self.mitigation_tolerance
        
        return price_diff <= threshold
    
    def mark_mitigated(self, zone: LiquidityZone) -> LiquidityZone:
        """
        Mark zone as mitigated (invalidated).
        
        Args:
            zone: Zone to invalidate
        
        Returns:
            New zone with updated status
        
        Note:
            Creates new frozen zone with MITIGATED status
        """
        # Remove from active
        if zone in self.hierarchy.active_zones:
            self.hierarchy.active_zones.remove(zone)
        
        # Create mitigated zone
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
        
        # Add to mitigated list
        self.hierarchy.mitigated_zones.append(mitigated_zone)
        
        # Update zones_by_price
        if zone.price in self.zones_by_price:
            self.zones_by_price[zone.price] = mitigated_zone
        
        return mitigated_zone
    
    def get_target_zones(
        self,
        current_coordinate: Coordinate,
        absorption_type: AbsorptionType,
        participant: Optional[ParticipantType] = None
    ) -> List[LiquidityZone]:
        """
        Get zones that should be targeted based on absorption type.
        
        Args:
            current_coordinate: Current market state
            absorption_type: Type of absorption detected
            participant: Current participant (filter by opposite)
        
        Returns:
            List of zones to target
        
        Algorithm:
            - INTERNAL absorption → target internal zones (continuation)
            - EXTERNAL absorption → target external zones (reversal)
            - EXHAUSTION absorption → target opposite participant zones
        """
        targets = []
        
        for zone in self.hierarchy.active_zones:
            # Skip mitigated zones
            if zone.status == ZoneStatus.MITIGATED:
                continue
            
            # Internal absorption targets internal zones
            if absorption_type == AbsorptionType.INTERNAL:
                if zone.liquidity_type == LiquidityType.INTERNAL:
                    targets.append(zone)
            
            # External/Exhaustion absorption targets external zones
            elif absorption_type in [AbsorptionType.EXTERNAL, AbsorptionType.EXHAUSTION]:
                if zone.liquidity_type == LiquidityType.EXTERNAL:
                    # Target opposite participant
                    if participant:
                        if zone.participant != participant:
                            targets.append(zone)
                    else:
                        targets.append(zone)
        
        # Sort by volume (highest first)
        targets.sort(key=lambda z: z.volume, reverse=True)
        
        return targets
    
    def get_first_valid_liquidity(self) -> Optional[LiquidityZone]:
        """
        Get the first valid liquidity zone (controls all others).
        
        Returns:
            First valid zone or None
        
        Rule:
            "First valid liquidity controls all others" (LAW 2)
        """
        return self.hierarchy.first_valid
    
    def get_active_zones(
        self,
        liquidity_type: Optional[LiquidityType] = None,
        participant: Optional[ParticipantType] = None
    ) -> List[LiquidityZone]:
        """
        Get active (unmitigated) zones with optional filters.
        
        Args:
            liquidity_type: Filter by INTERNAL or EXTERNAL
            participant: Filter by BUYER or SELLER
        
        Returns:
            List of active zones matching filters
        """
        zones = self.hierarchy.active_zones.copy()
        
        if liquidity_type:
            zones = [z for z in zones if z.liquidity_type == liquidity_type]
        
        if participant:
            zones = [z for z in zones if z.participant == participant]
        
        return zones
    
    def get_zone_by_price(self, price: float) -> Optional[LiquidityZone]:
        """Get zone at specific price level."""
        return self.zones_by_price.get(price)
    
    def update_zones(self, current_price: float):
        """
        Update zone statuses based on current price.
        
        Checks all active zones for mitigation and updates status.
        
        Args:
            current_price: Current market price
        """
        zones_to_mitigate = []
        
        for zone in self.hierarchy.active_zones:
            if self.is_mitigated(zone, current_price):
                zones_to_mitigate.append(zone)
        
        for zone in zones_to_mitigate:
            self.mark_mitigated(zone)
    
    def add_zone_relationship(self, zone1_price: float, zone2_price: float):
        """
        Add relationship between two zones.
        
        Args:
            zone1_price: First zone price
            zone2_price: Second zone price
        
        Usage:
            Track that zone1 depends on zone2, or zone1 targets zone2
        """
        if zone1_price not in self.hierarchy.zone_relationships:
            self.hierarchy.zone_relationships[zone1_price] = []
        
        if zone2_price not in self.hierarchy.zone_relationships[zone1_price]:
            self.hierarchy.zone_relationships[zone1_price].append(zone2_price)
    
    def get_related_zones(self, price: float) -> List[LiquidityZone]:
        """
        Get zones related to given price level.
        
        Args:
            price: Zone price to look up
        
        Returns:
            List of related zones
        """
        related_prices = self.hierarchy.zone_relationships.get(price, [])
        return [self.zones_by_price[p] for p in related_prices if p in self.zones_by_price]
    
    def clear_all(self):
        """Clear all zones and reset hierarchy."""
        self.hierarchy = ZoneHierarchy()
        self.zones_by_price.clear()
    
    def get_zone_count(self) -> Dict[str, int]:
        """
        Get count of zones by status.
        
        Returns:
            Dict with counts: {"active": N, "mitigated": M, "total": T}
        """
        return {
            "active": len(self.hierarchy.active_zones),
            "mitigated": len(self.hierarchy.mitigated_zones),
            "total": len(self.zones_by_price)
        }
