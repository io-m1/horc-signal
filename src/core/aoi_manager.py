"""
AOI (Area of Interest) Manager — PHASE 1.75

Tracks and validates Areas of Interest with liquidity for divergence/absorption analysis.

DOCTRINE:
    "AOI selection depends on which liquidity is calling price"
    "Highest volume levels reveal divergence patterns"
    "Session differences matter (Frankfurt low vs London high)"

PURPOSE:
    - Register AOIs with coordinates and liquidity
    - Track mitigation status (has price returned?)
    - Manage multi-session absorption patterns
    - Identify active AOIs by session and type

INTEGRATION:
    ChargeEngine → CoordinateEngine → AOI Manager → DivergenceEngine → AbsorptionEngine
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime

from src.core.coordinate_engine import Coordinate


class LiquidityType(Enum):
    """Type of liquidity at AOI."""
    INTERNAL = "internal"   # Internal to current trend
    EXTERNAL = "external"   # External to current trend


class SessionType(Enum):
    """Trading session for AOI."""
    FRANKFURT = "frankfurt"
    LONDON = "london"
    NEW_YORK = "new_york"
    ASIA = "asia"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AOI:
    """
    Area of Interest with liquidity validation.
    
    An AOI represents a price level with significant liquidity that may act as
    a magnet for price action. AOIs are validated by divergence/absorption patterns.
    
    Attributes:
        coordinate: Passive coordinate at AOI formation
        price: Price level of AOI
        liquidity_type: Internal or external liquidity
        volume: Volume at formation (higher = stronger AOI)
        session: Trading session when AOI formed
        timestamp: When AOI was registered
        is_mitigated: Has price returned to this level?
        mitigation_timestamp: When AOI was mitigated (if applicable)
        target_coordinate: What this AOI is targeting (if known)
    """
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
    """
    Registry of all AOIs with state tracking.
    
    Manages AOI lifecycle:
        1. Registration (new AOI added)
        2. Active (AOI is valid and unmitigated)
        3. Mitigated (price has returned to AOI)
    
    Attributes:
        aois: List of all registered AOIs
        active_aois: Dict of active (unmitigated) AOIs by session
    """
    aois: List[AOI] = field(default_factory=list)
    active_aois: Dict[SessionType, List[AOI]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize active AOI tracking for all sessions."""
        for session in SessionType:
            self.active_aois[session] = []


class AOIManager:
    """
    Manages Areas of Interest for divergence/absorption analysis.
    
    RULES:
        1. AOIs validated by volume (higher = stronger)
        2. Mitigation = price returns to AOI level
        3. Once mitigated, AOI becomes inactive
        4. Multi-session tracking (Frankfurt → London → NY)
        5. Internal vs external classification critical
    
    USAGE:
        manager = AOIManager()
        aoi = manager.register_aoi(
            coordinate=coord,
            price=100.50,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0,
            session=SessionType.FRANKFURT
        )
        if manager.is_mitigated(aoi, current_price=100.52):
            manager.mark_mitigated(aoi)
    """
    
    def __init__(self):
        """Initialize AOI manager with empty registry."""
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
        """
        Register a new Area of Interest.
        
        Args:
            coordinate: Passive coordinate at AOI
            price: Price level
            liquidity_type: Internal or external
            volume: Volume at formation
            session: Trading session
            target_coordinate: Optional target this AOI points to
        
        Returns:
            Newly created AOI
        
        Example:
            >>> manager = AOIManager()
            >>> coord = Coordinate(price=100, D=-1, S=-1)
            >>> aoi = manager.register_aoi(
            ...     coordinate=coord,
            ...     price=100.0,
            ...     liquidity_type=LiquidityType.INTERNAL,
            ...     volume=1000.0,
            ...     session=SessionType.FRANKFURT
            ... )
        """
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
        
        # Add to registry
        self.registry.aois.append(aoi)
        self.registry.active_aois[session].append(aoi)
        
        return aoi
    
    def is_mitigated(self, aoi: AOI, current_price: float, tolerance: float = 0.0001) -> bool:
        """
        Check if price has returned to AOI level (mitigation).
        
        Args:
            aoi: AOI to check
            current_price: Current market price
            tolerance: Price tolerance (default 0.01%)
        
        Returns:
            True if current price is within tolerance of AOI price
        
        Algorithm:
            Mitigated if |current_price - aoi_price| / aoi_price <= tolerance
        """
        if aoi.is_mitigated:
            return True
        
        price_diff = abs(current_price - aoi.price)
        threshold = aoi.price * tolerance
        
        return price_diff <= threshold
    
    def mark_mitigated(self, aoi: AOI) -> AOI:
        """
        Mark an AOI as mitigated (price has returned).
        
        Args:
            aoi: AOI to mark as mitigated
        
        Returns:
            New AOI with updated mitigation status
        
        Note:
            Creates new frozen AOI with updated status
        """
        # Remove from active list
        if aoi in self.registry.active_aois[aoi.session]:
            self.registry.active_aois[aoi.session].remove(aoi)
        
        # Create new mitigated AOI (frozen dataclass requires recreation)
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
        
        # Update registry
        if aoi in self.registry.aois:
            idx = self.registry.aois.index(aoi)
            self.registry.aois[idx] = mitigated_aoi
        
        return mitigated_aoi
    
    def get_active_aois(
        self,
        session: Optional[SessionType] = None,
        liquidity_type: Optional[LiquidityType] = None
    ) -> List[AOI]:
        """
        Get list of active (unmitigated) AOIs.
        
        Args:
            session: Filter by session (None = all sessions)
            liquidity_type: Filter by liquidity type (None = all types)
        
        Returns:
            List of active AOIs matching filters
        
        Example:
            >>> manager = AOIManager()
            >>> # ... register some AOIs ...
            >>> frankfurt_aois = manager.get_active_aois(
            ...     session=SessionType.FRANKFURT
            ... )
        """
        if session is not None:
            aois = self.registry.active_aois[session]
        else:
            # All sessions
            aois = []
            for session_aois in self.registry.active_aois.values():
                aois.extend(session_aois)
        
        # Filter by liquidity type if specified
        if liquidity_type is not None:
            aois = [aoi for aoi in aois if aoi.liquidity_type == liquidity_type]
        
        return aois
    
    def get_highest_volume_aoi(
        self,
        session: Optional[SessionType] = None,
        liquidity_type: Optional[LiquidityType] = None
    ) -> Optional[AOI]:
        """
        Get AOI with highest volume matching filters.
        
        Args:
            session: Filter by session
            liquidity_type: Filter by liquidity type
        
        Returns:
            AOI with highest volume, or None if no AOIs match
        
        Rule:
            "Highest volume levels reveal divergence patterns"
        """
        aois = self.get_active_aois(session=session, liquidity_type=liquidity_type)
        
        if not aois:
            return None
        
        return max(aois, key=lambda aoi: aoi.volume)
    
    def get_all_aois(self) -> List[AOI]:
        """Get all registered AOIs (active and mitigated)."""
        return self.registry.aois.copy()
    
    def get_session_chain(self) -> List[SessionType]:
        """
        Get ordered list of trading sessions for multi-session tracking.
        
        Returns:
            [FRANKFURT, LONDON, NEW_YORK, ASIA]
        
        Usage:
            Track absorption patterns across sessions:
            Frankfurt Low → London High → NY Reversal
        """
        return [
            SessionType.FRANKFURT,
            SessionType.LONDON,
            SessionType.NEW_YORK,
            SessionType.ASIA
        ]
    
    def clear_session(self, session: SessionType):
        """
        Clear all AOIs for a specific session.
        
        Args:
            session: Session to clear
        
        Usage:
            Reset at end of session or start of new session
        """
        self.registry.active_aois[session] = []
    
    def clear_all(self):
        """Clear all AOIs from registry."""
        self.registry.aois.clear()
        for session in SessionType:
            self.registry.active_aois[session] = []
