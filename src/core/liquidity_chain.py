from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto

class LiquidityType(Enum):
    INTERNAL = auto()   # Formed on current trend (tradable)
    EXTERNAL = auto()   # Reference only (not tradable until internalized)

class LiquiditySide(Enum):
    BUY_SIDE = 1       # Highs - targets for sellers
    SELL_SIDE = -1     # Lows - targets for buyers

class TrappedState(Enum):
    FREE = 0           # Not trapped, can be activated normally
    TRAPPED = 1        # Enclosed by higher TF, needs parent exhaustion

@dataclass
class LiquidityNode:
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
        return self.trapped_by is not None
    
    @property
    def trapped_state(self) -> TrappedState:
        return TrappedState.TRAPPED if self.is_trapped else TrappedState.FREE
    
    @property
    def is_premium(self) -> bool:
        return self.rank == 0
    
    @property
    def is_active(self) -> bool:
        return (
            self.liquidity_type == LiquidityType.INTERNAL and
            not self.invalidated and
            not self.is_trapped
        )
    
    def to_pine_array_row(self) -> Tuple[float, int, int, int, int, int, int]:
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
    direction: int = 0            # +1 bullish, -1 bearish
    aggressor: int = 0            # +1 buyers aggressive, -1 sellers aggressive
    nodes: List[LiquidityNode] = field(default_factory=list)
    
    @property
    def fair_value_area(self) -> Optional[float]:
        pullback_nodes = [
            n for n in self.nodes
            if n.is_active and 
            n.side.value == -self.direction  # Pullback side
        ]
        
        if not pullback_nodes:
            return None
        
        most_discounted = max(pullback_nodes, key=lambda n: n.formation_index)
        return most_discounted.price
    
    @property
    def most_premium_liquidity(self) -> Optional[LiquidityNode]:
        trend_nodes = [
            n for n in self.nodes
            if n.is_active and
            n.side.value == self.direction  # Trend side
        ]
        
        if not trend_nodes:
            return None
        
        return min(trend_nodes, key=lambda n: n.formation_index)
    
    @property
    def most_discounted_liquidity(self) -> Optional[LiquidityNode]:
        pullback_nodes = [
            n for n in self.nodes
            if n.is_active and
            n.side.value == -self.direction  # Pullback side
        ]
        
        if not pullback_nodes:
            return None
        
        return min(pullback_nodes, key=lambda n: n.formation_index)
    
    @property
    def controlling_liquidity(self) -> Optional[LiquidityNode]:
        active_nodes = [n for n in self.nodes if n.is_active]
        if not active_nodes:
            return None
        
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
        formation_index = len(self.nodes)
        
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
        invalidated = []
        
        for node in self.nodes:
            if node.invalidated:
                continue
            
            if node.side == LiquiditySide.BUY_SIDE:
                if current_price >= node.price:
                    node.invalidated = True
                    invalidated.append(node)
            else:
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
        trapped = []
        
        for node in self.nodes:
            if node.is_trapped or node.invalidated:
                continue
            
            if range_low < node.price < range_high:
                object.__setattr__(node, 'trapped_by', trapping_tf)
                trapped.append(node)
        
        return trapped
    
    def get_active_targets(self, from_price: float) -> List[LiquidityNode]:
        active = [n for n in self.nodes if n.is_active]
        
        active.sort(key=lambda n: abs(n.price - from_price))
        
        return active
    
    def rerank_nodes(self) -> None:
        for side in [LiquiditySide.BUY_SIDE, LiquiditySide.SELL_SIDE]:
            side_nodes = [
                n for n in self.nodes
                if n.side == side and not n.invalidated
            ]
            side_nodes.sort(key=lambda n: n.formation_index)
            
            for i, node in enumerate(side_nodes):
                object.__setattr__(node, 'rank', i)
    
    def to_pine_vars(self) -> dict:
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
    chain = LiquidityChain(direction=direction, aggressor=aggressor)
    
    all_swings = []
    for price, ts, tf in swing_highs:
        all_swings.append((ts, price, tf, LiquiditySide.BUY_SIDE))
    for price, ts, tf in swing_lows:
        all_swings.append((ts, price, tf, LiquiditySide.SELL_SIDE))
    
    all_swings.sort(key=lambda x: x[0])
    
    for ts, price, tf, side in all_swings:
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
    counterparty_side = -continuation_side
    
    counterparty_nodes = [
        n for n in chain.nodes
        if n.participant == counterparty_side and not n.is_trapped
    ]
    
    if not counterparty_nodes:
        return True, None
    
    controlling = min(counterparty_nodes, key=lambda n: n.formation_index)
    
    if controlling.invalidated:
        return True, controlling
    
    return False, controlling
