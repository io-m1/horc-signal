"""
Liquidity Chain - The Hierarchy Model

DOCTRINE (from the edge owner):
    "Liquidity is a hierarchy, not a location.
     Continuation is chosen by the counterparty.
     The first valid liquidity in a chain controls all others."

THREE LAWS:
    1. Liquidity is NOT a point, it is a RELATIONSHIP
       - Position in hierarchy matters
       - Formation order matters
       - Participant class matters
    
    2. The FIRST valid liquidity in a chain CONTROLS all others
       - TF does NOT matter (sessional can dominate monthly)
       - Age does NOT matter
       - Size does NOT matter
       - Formation order + participant interaction > timeframe
    
    3. Continuation is DEPENDENT on reversal
       - Sell continuation requires buyers to reverse
       - Buyers reverse only at their most discounted liquidity
       - Counterparty decides where continuation happens

DEFINITIONS:
    Internal Liquidity: Formed on current aggressor trend (can be premium/discount/trapped)
    External Liquidity: Temporary reference, loses relevance once crossed
    Premium: Higher in formation order (formed first)
    Discount: Lower in formation order (formed later on pullback)
    Trapped: Enclosed by higher-TF range, requires parent TF exhaustion to activate

PINE-SAFETY:
    All state is primitive-compatible. LiquidityChain can be serialized to arrays.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto


class LiquidityType(Enum):
    """Type of liquidity level."""
    INTERNAL = auto()   # Formed on current trend (tradable)
    EXTERNAL = auto()   # Reference only (not tradable until internalized)


class LiquiditySide(Enum):
    """Which side of price the liquidity sits."""
    BUY_SIDE = 1       # Highs - targets for sellers
    SELL_SIDE = -1     # Lows - targets for buyers


class TrappedState(Enum):
    """Trapped liquidity state."""
    FREE = 0           # Not trapped, can be activated normally
    TRAPPED = 1        # Enclosed by higher TF, needs parent exhaustion


@dataclass
class LiquidityNode:
    """
    A single liquidity level in the chain hierarchy.
    
    KEY INSIGHT: Liquidity is defined by its relationship to other nodes,
    not by its absolute price level or timeframe.
    
    Attributes:
        price: The price level
        timeframe: TF where liquidity was formed (string key)
        formation_index: Order of formation in the chain (0 = first = most premium/discount)
        participant: Who formed this liquidity (+1 buyers, -1 sellers)
        side: BUY_SIDE (highs) or SELL_SIDE (lows)
        liquidity_type: INTERNAL (tradable) or EXTERNAL (reference)
        rank: Premium/discount rank (lower = more premium, higher = more discounted)
        trapped_by: TF that trapped this liquidity (None if free)
        timestamp: Unix ms when formed
        invalidated: Whether this node has been crossed/filled
        
    Pine Translation:
        // Arrays for each field
        var float[] liq_prices = array.new_float()
        var int[] liq_formation = array.new_int()
        var int[] liq_participant = array.new_int()
        var int[] liq_rank = array.new_int()
        var int[] liq_trapped_tf = array.new_int()  // -1 = free
    """
    price: float
    timeframe: str
    formation_index: int
    participant: int              # +1 buyers, -1 sellers
    side: LiquiditySide
    liquidity_type: LiquidityType = LiquidityType.INTERNAL
    rank: int = 0                 # 0 = most premium
    trapped_by: Optional[str] = None
    timestamp: int = 0            # Unix ms
    invalidated: bool = False
    
    @property
    def is_trapped(self) -> bool:
        """Check if this liquidity is trapped by a higher TF."""
        return self.trapped_by is not None
    
    @property
    def trapped_state(self) -> TrappedState:
        """Get trapped state enum."""
        return TrappedState.TRAPPED if self.is_trapped else TrappedState.FREE
    
    @property
    def is_premium(self) -> bool:
        """Premium = formed first in the trend direction."""
        return self.rank == 0
    
    @property
    def is_active(self) -> bool:
        """Active = internal, not invalidated, not trapped."""
        return (
            self.liquidity_type == LiquidityType.INTERNAL and
            not self.invalidated and
            not self.is_trapped
        )
    
    def to_pine_array_row(self) -> Tuple[float, int, int, int, int, int, int]:
        """Export as tuple for Pine array storage."""
        return (
            self.price,
            self.formation_index,
            self.participant,
            self.side.value,
            self.rank,
            -1 if self.trapped_by is None else hash(self.trapped_by) % 100,
            1 if self.invalidated else 0,
        )


@dataclass
class LiquidityChain:
    """
    Ordered chain of liquidity levels forming a hierarchy.
    
    CORE PRINCIPLE: The first valid liquidity in the chain controls all others.
    
    This is NOT a flat list of levels. It is a hierarchy where:
        - Formation order determines premium/discount rank
        - Earlier = more premium (for trend direction)
        - Earlier = more discounted (for pullback direction)
        - Trapped nodes require parent TF exhaustion
    
    Attributes:
        direction: Overall trend direction (+1 bullish, -1 bearish)
        aggressor: Who controls the trend (+1 buyers, -1 sellers)
        nodes: Ordered list of liquidity nodes (formation order)
        fair_value_area: Dynamic FVA = most discounted internal liquidity
    
    Pine Translation:
        var int chain_direction = 0
        var int chain_aggressor = 0
        var float chain_fva = na
        // Plus arrays for nodes (see LiquidityNode.to_pine_array_row)
    """
    direction: int = 0            # +1 bullish, -1 bearish
    aggressor: int = 0            # +1 buyers aggressive, -1 sellers aggressive
    nodes: List[LiquidityNode] = field(default_factory=list)
    
    @property
    def fair_value_area(self) -> Optional[float]:
        """
        Dynamic Fair Value Area = Most Discounted Internal Liquidity.
        
        NOT: Midpoint, Fibonacci, Average, Fixed range
        
        This is where the counterparty exhausts.
        This is where continuation is decided.
        """
        # Find most discounted active node on the pullback side
        pullback_nodes = [
            n for n in self.nodes
            if n.is_active and 
            n.side.value == -self.direction  # Pullback side
        ]
        
        if not pullback_nodes:
            return None
        
        # Most discounted = highest formation index (formed last in pullback)
        most_discounted = max(pullback_nodes, key=lambda n: n.formation_index)
        return most_discounted.price
    
    @property
    def most_premium_liquidity(self) -> Optional[LiquidityNode]:
        """
        Most premium = first formed on the trend.
        Controls all lower liquidity.
        Can reverse price even if lower liquidity zones untouched.
        """
        trend_nodes = [
            n for n in self.nodes
            if n.is_active and
            n.side.value == self.direction  # Trend side
        ]
        
        if not trend_nodes:
            return None
        
        # Most premium = lowest formation index (formed first)
        return min(trend_nodes, key=lambda n: n.formation_index)
    
    @property
    def most_discounted_liquidity(self) -> Optional[LiquidityNode]:
        """
        Most discounted = first formed on the pullback.
        Determines where buyers/sellers exhaust.
        This is the true Fair Value Area.
        """
        pullback_nodes = [
            n for n in self.nodes
            if n.is_active and
            n.side.value == -self.direction  # Pullback side
        ]
        
        if not pullback_nodes:
            return None
        
        # Most discounted = lowest formation index in pullback direction
        return min(pullback_nodes, key=lambda n: n.formation_index)
    
    @property
    def controlling_liquidity(self) -> Optional[LiquidityNode]:
        """
        The liquidity that controls price behavior.
        
        LAW 2: The first valid liquidity in a chain controls all others.
        """
        active_nodes = [n for n in self.nodes if n.is_active]
        if not active_nodes:
            return None
        
        # First formed = controls
        return min(active_nodes, key=lambda n: n.formation_index)
    
    def add_node(
        self,
        price: float,
        timeframe: str,
        participant: int,
        side: LiquiditySide,
        timestamp: int,
        liquidity_type: LiquidityType = LiquidityType.INTERNAL,
    ) -> LiquidityNode:
        """
        Add a new liquidity node to the chain.
        
        Formation index is assigned automatically based on insertion order.
        Rank is computed relative to other nodes on the same side.
        """
        formation_index = len(self.nodes)
        
        # Compute rank relative to same-side nodes
        same_side = [n for n in self.nodes if n.side == side and not n.invalidated]
        rank = len(same_side)  # New node gets next rank
        
        node = LiquidityNode(
            price=price,
            timeframe=timeframe,
            formation_index=formation_index,
            participant=participant,
            side=side,
            liquidity_type=liquidity_type,
            rank=rank,
            timestamp=timestamp,
        )
        
        self.nodes.append(node)
        return node
    
    def invalidate_crossed(self, current_price: float) -> List[LiquidityNode]:
        """
        Invalidate liquidity nodes that have been crossed.
        
        External liquidity loses relevance once crossed.
        Internal liquidity becomes "filled".
        
        Returns list of invalidated nodes.
        """
        invalidated = []
        
        for node in self.nodes:
            if node.invalidated:
                continue
            
            # Check if price crossed this level
            if node.side == LiquiditySide.BUY_SIDE:
                # Buy-side liquidity (highs) - crossed if price went above
                if current_price >= node.price:
                    node.invalidated = True
                    invalidated.append(node)
            else:
                # Sell-side liquidity (lows) - crossed if price went below
                if current_price <= node.price:
                    node.invalidated = True
                    invalidated.append(node)
        
        return invalidated
    
    def trap_by_range(
        self,
        range_high: float,
        range_low: float,
        trapping_tf: str,
    ) -> List[LiquidityNode]:
        """
        Trap liquidity nodes enclosed by a higher-TF range.
        
        RULE: Only lower TF liquidity can be trapped.
        Once trapped:
            - Cannot be activated by its own TF imbalance
            - Requires parent TF exhaustion imbalance
        
        Args:
            range_high: Upper bound of trapping range
            range_low: Lower bound of trapping range
            trapping_tf: The TF that creates the trap
        
        Returns:
            List of newly trapped nodes
        """
        trapped = []
        
        for node in self.nodes:
            if node.is_trapped or node.invalidated:
                continue
            
            # Check if node is inside the range
            if range_low < node.price < range_high:
                # Node is trapped by this range
                object.__setattr__(node, 'trapped_by', trapping_tf)
                trapped.append(node)
        
        return trapped
    
    def get_active_targets(self, from_price: float) -> List[LiquidityNode]:
        """
        Get active liquidity targets from current price.
        
        Returns nodes ordered by proximity (nearest first).
        """
        active = [n for n in self.nodes if n.is_active]
        
        # Sort by distance from current price
        active.sort(key=lambda n: abs(n.price - from_price))
        
        return active
    
    def rerank_nodes(self) -> None:
        """
        Recompute premium/discount ranks after invalidations.
        
        Rank is relative to remaining active nodes on each side.
        """
        for side in [LiquiditySide.BUY_SIDE, LiquiditySide.SELL_SIDE]:
            side_nodes = [
                n for n in self.nodes
                if n.side == side and not n.invalidated
            ]
            # Sort by formation index (original order)
            side_nodes.sort(key=lambda n: n.formation_index)
            
            # Assign new ranks
            for i, node in enumerate(side_nodes):
                object.__setattr__(node, 'rank', i)
    
    def to_pine_vars(self) -> dict:
        """Export chain state as Pine-compatible dict."""
        return {
            "chain_direction": self.direction,
            "chain_aggressor": self.aggressor,
            "chain_fva": self.fair_value_area or float('nan'),
            "chain_node_count": len([n for n in self.nodes if n.is_active]),
            "chain_controlling_price": (
                self.controlling_liquidity.price 
                if self.controlling_liquidity else float('nan')
            ),
        }
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __iter__(self):
        return iter(self.nodes)


def build_chain_from_swings(
    swing_highs: List[Tuple[float, int, str]],  # (price, timestamp, timeframe)
    swing_lows: List[Tuple[float, int, str]],
    direction: int,
    aggressor: int,
) -> LiquidityChain:
    """
    Build a liquidity chain from detected swing points.
    
    Args:
        swing_highs: List of (price, timestamp_ms, timeframe) for highs
        swing_lows: List of (price, timestamp_ms, timeframe) for lows
        direction: Trend direction (+1 bullish, -1 bearish)
        aggressor: Who controls (+1 buyers, -1 sellers)
    
    Returns:
        LiquidityChain with nodes ranked by formation order
    """
    chain = LiquidityChain(direction=direction, aggressor=aggressor)
    
    # Combine all swings with their side
    all_swings = []
    for price, ts, tf in swing_highs:
        all_swings.append((ts, price, tf, LiquiditySide.BUY_SIDE))
    for price, ts, tf in swing_lows:
        all_swings.append((ts, price, tf, LiquiditySide.SELL_SIDE))
    
    # Sort by timestamp (formation order)
    all_swings.sort(key=lambda x: x[0])
    
    # Add to chain in formation order
    for ts, price, tf, side in all_swings:
        # Determine participant based on side and direction
        if side == LiquiditySide.BUY_SIDE:
            participant = 1  # Buyers created this high
        else:
            participant = -1  # Sellers created this low
        
        chain.add_node(
            price=price,
            timeframe=tf,
            participant=participant,
            side=side,
            timestamp=ts,
            liquidity_type=LiquidityType.INTERNAL,
        )
    
    return chain


def continuation_permitted(
    chain: LiquidityChain,
    continuation_side: int,
) -> Tuple[bool, Optional[LiquidityNode]]:
    """
    Check if continuation is permitted (LAW 3).
    
    LAW 3: Continuation is dependent on reversal.
        - Sell continuation requires buyers to reverse
        - Buyers reverse only at their most discounted liquidity
        - Counterparty decides where continuation happens
    
    Args:
        chain: The current liquidity chain
        continuation_side: +1 for bullish continuation, -1 for bearish
    
    Returns:
        (permitted, controlling_node)
        - permitted: True if counterparty has exhausted
        - controlling_node: The liquidity level that permits/blocks
    """
    # Find counterparty's most premium liquidity (first formed = controlling)
    counterparty_side = -continuation_side
    
    # Get ALL counterparty nodes (including invalidated)
    counterparty_nodes = [
        n for n in chain.nodes
        if n.participant == counterparty_side and not n.is_trapped
    ]
    
    if not counterparty_nodes:
        # No counterparty liquidity = continuation permitted by default
        return True, None
    
    # Most premium (controlling) = first formed = lowest formation_index
    controlling = min(counterparty_nodes, key=lambda n: n.formation_index)
    
    # LAW 2: First valid liquidity controls
    # If controlling liquidity is invalidated, counterparty has exhausted there
    # Continuation is now permitted
    if controlling.invalidated:
        return True, controlling
    
    # Counterparty has not exhausted at their controlling level yet
    return False, controlling
