"""
Strategic Context - Top of Decision Stack

PHILOSOPHY (from the edge owner):
    "The first thing you want to do when you come to your chart is 
     what is the liquidity you want to target."

This module implements the two FIRST-CLASS state objects that must be
resolved BEFORE any signal generation:

    1. LiquidityIntent - The target liquidity level that anchors analysis
    2. MarketControlState - Who is aggressor vs passive, and at what TF

DECISION HIERARCHY:
    LiquidityEngine → MarketControlResolver → IF valid → ZoneEngine → Execution
                                            → IF NOT valid → NullIR (stand down)

PINE-SAFETY:
    All state is primitive (int, float, bool) for 1:1 Pine translation.
    
    Pine equivalent:
        var int liquidity_direction = 0
        var float liquidity_level = na
        var int liquidity_tf = 0
        var int market_control = 0
        var int control_tf = 0
        var bool context_valid = false

RULE: Liquidity selection > Market control > Everything else
      If liquidity and control disagree → stand down (no signal)
"""

from dataclasses import dataclass
from typing import Optional, List
from .enums import LIQUIDITY_DIRECTION, MARKET_CONTROL, TIMEFRAME_RANK


@dataclass
class LiquidityIntent:
    """
    The liquidity target that anchors the entire analysis.
    
    This is resolved FIRST, before any other engine runs.
    Everything downstream is subordinate to this intent.
    
    Attributes:
        direction: SELL_SIDE (-1) or BUY_SIDE (+1) or NONE (0)
        level: The price level of the liquidity target
        timeframe: The TF where liquidity was identified (string key)
        priority: Higher = more authoritative (derived from TF rank)
        distance_atr: Distance to target in ATR units
        valid: Whether this intent is actionable
    
    Pine Translation:
        var int liq_direction = 0
        var float liq_level = na
        var int liq_tf_rank = 0
        var float liq_distance = na
        var bool liq_valid = false
    """
    direction: int = 0          # LIQUIDITY_DIRECTION value
    level: float = 0.0          # Price level
    timeframe: str = "D1"       # TF string key
    priority: int = 0           # Derived from TIMEFRAME_RANK
    distance_atr: float = 0.0   # Distance in ATR units
    valid: bool = False         # Is this intent actionable?
    
    @classmethod
    def from_level(
        cls,
        level: float,
        direction: int,
        timeframe: str,
        current_price: float,
        atr: float,
    ) -> "LiquidityIntent":
        """
        Create intent from a detected liquidity level.
        
        Args:
            level: The liquidity price level
            direction: SELL_SIDE or BUY_SIDE
            timeframe: TF where detected
            current_price: Current market price
            atr: Current ATR for distance calculation
        
        Returns:
            LiquidityIntent with computed fields
        """
        priority = TIMEFRAME_RANK.get(timeframe, 0)
        distance = abs(level - current_price)
        distance_atr = distance / atr if atr > 0 else 0.0
        
        # Valid if direction is set and level is reachable
        valid = direction != 0 and distance_atr > 0.5 and distance_atr < 20.0
        
        return cls(
            direction=direction,
            level=level,
            timeframe=timeframe,
            priority=priority,
            distance_atr=distance_atr,
            valid=valid,
        )
    
    @classmethod
    def null(cls) -> "LiquidityIntent":
        """Return a null/invalid intent (stand down)"""
        return cls(
            direction=0,
            level=0.0,
            timeframe="",
            priority=0,
            distance_atr=0.0,
            valid=False,
        )
    
    def to_pine_vars(self) -> dict:
        """Export as Pine-compatible variable dict"""
        return {
            "liq_direction": self.direction,
            "liq_level": self.level if self.valid else float('nan'),
            "liq_tf_rank": self.priority,
            "liq_distance": self.distance_atr,
            "liq_valid": 1 if self.valid else 0,
        }
    
    @classmethod
    def from_chain(
        cls,
        chain: "LiquidityChain",
        current_price: float,
        atr: float,
    ) -> "LiquidityIntent":
        """
        Create intent from a LiquidityChain's controlling liquidity.
        
        Uses the chain's fair_value_area (most discounted internal liquidity)
        as the target level.
        
        Args:
            chain: The liquidity chain
            current_price: Current market price
            atr: Current ATR
        
        Returns:
            LiquidityIntent derived from chain hierarchy
        """
        # Get controlling liquidity from chain
        controlling = chain.controlling_liquidity
        if controlling is None:
            return cls.null()
        
        # Direction based on chain direction
        # Bullish chain → target sell-side liquidity (lows)
        # Bearish chain → target buy-side liquidity (highs)
        if chain.direction > 0:
            direction = -1  # SELL_SIDE
        elif chain.direction < 0:
            direction = 1   # BUY_SIDE
        else:
            return cls.null()
        
        return cls.from_level(
            level=controlling.price,
            direction=direction,
            timeframe=controlling.timeframe,
            current_price=current_price,
            atr=atr,
        )


# Forward reference for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .liquidity_chain import LiquidityChain


@dataclass
class MarketControlState:
    """
    Who controls the market - aggressor vs passive identification.
    
    Process (from video):
        1. Start from higher TF (12H → 8H → 6H)
        2. Identify passive participant (D+/D-)
        3. Identify aggressor
        4. Highest TF where aggressor is conclusive = control TF
    
    Attributes:
        passive: The passive participant (BUYERS or SELLERS)
        aggressor: The aggressive participant (BUYERS or SELLERS)
        control: Combined control state (MARKET_CONTROL value)
        control_tf: The TF where control was determined
        conclusive: Whether control is definitively established
    
    Pine Translation:
        var int mkt_passive = 0
        var int mkt_aggressor = 0
        var int mkt_control = 0
        var int mkt_control_tf = 0
        var bool mkt_conclusive = false
    """
    passive: int = 0            # PARTICIPANT_CONTROL value (-1, 0, +1)
    aggressor: int = 0          # PARTICIPANT_CONTROL value (-1, 0, +1)
    control: int = 0            # MARKET_CONTROL value
    control_tf: str = ""        # TF string key where control resolved
    control_tf_rank: int = 0    # Numeric rank of control TF
    conclusive: bool = False    # Is control definitively established?
    
    @classmethod
    def from_participant_analysis(
        cls,
        participant_type: int,
        conviction: bool,
        sweep_direction: int,
        timeframe: str,
    ) -> "MarketControlState":
        """
        Derive market control from participant identification.
        
        The participant who swept first is the AGGRESSOR.
        The other side becomes PASSIVE (defending).
        
        Args:
            participant_type: From ParticipantResult (+1 buyers, -1 sellers)
            conviction: Was the sweep decisive?
            sweep_direction: Direction of the sweep
            timeframe: TF of analysis
        
        Returns:
            MarketControlState with derived fields
        """
        if participant_type == 0 or not conviction:
            return cls.inconclusive()
        
        # Aggressor is the one who swept
        aggressor = participant_type
        passive = -participant_type  # Other side
        
        # Derive control state
        if aggressor == 1:  # Buyers aggressive
            control = MARKET_CONTROL["BUYERS_AGGRESSIVE"]
        else:  # Sellers aggressive
            control = MARKET_CONTROL["SELLERS_AGGRESSIVE"]
        
        return cls(
            passive=passive,
            aggressor=aggressor,
            control=control,
            control_tf=timeframe,
            control_tf_rank=TIMEFRAME_RANK.get(timeframe, 0),
            conclusive=conviction,
        )
    
    @classmethod
    def inconclusive(cls) -> "MarketControlState":
        """Return inconclusive state (cannot determine control)"""
        return cls(
            passive=0,
            aggressor=0,
            control=MARKET_CONTROL["INCONCLUSIVE"],
            control_tf="",
            control_tf_rank=0,
            conclusive=False,
        )
    
    def is_aligned_with(self, liquidity: LiquidityIntent) -> bool:
        """
        Check if market control is aligned with liquidity intent.
        
        Alignment rules:
            - Targeting BUY_SIDE liquidity → need SELLERS_AGGRESSIVE
              (sellers pushing up into buy-side liquidity)
            - Targeting SELL_SIDE liquidity → need BUYERS_AGGRESSIVE
              (buyers pushing down into sell-side liquidity)
        
        Returns:
            True if aligned, False if misaligned (stand down)
        """
        if not self.conclusive or not liquidity.valid:
            return False
        
        # Sellers aggressive → price moving UP → targets BUY_SIDE
        if self.aggressor == -1 and liquidity.direction == 1:
            return True
        
        # Buyers aggressive → price moving DOWN → targets SELL_SIDE
        if self.aggressor == 1 and liquidity.direction == -1:
            return True
        
        return False
    
    def to_pine_vars(self) -> dict:
        """Export as Pine-compatible variable dict"""
        return {
            "mkt_passive": self.passive,
            "mkt_aggressor": self.aggressor,
            "mkt_control": self.control,
            "mkt_control_tf": self.control_tf_rank,
            "mkt_conclusive": 1 if self.conclusive else 0,
        }


@dataclass
class StrategicContext:
    """
    Top-level strategic context that gates all downstream processing.
    
    This is the FIRST thing computed on each session/day.
    If context is invalid → all downstream engines return NullIR.
    
    DECISION FLOW:
        1. Identify target liquidity (LiquidityIntent)
        2. Determine market control (MarketControlState)
        3. Check alignment (control must support intent)
        4. IF aligned → proceed to zone/execution
        5. IF NOT aligned → stand down (valid=False)
    
    Attributes:
        liquidity: The liquidity intent
        control: The market control state
        valid: Whether context permits signal generation
        alignment_score: How well control aligns with intent [0, 1]
        reason: Human-readable reason for validity state
    
    Pine Translation:
        // All nested vars flattened
        var int ctx_liq_direction = 0
        var float ctx_liq_level = na
        var int ctx_mkt_control = 0
        var bool ctx_valid = false
    """
    liquidity: LiquidityIntent
    control: MarketControlState
    valid: bool = False
    alignment_score: float = 0.0
    reason: str = ""
    
    @classmethod
    def resolve(
        cls,
        liquidity: LiquidityIntent,
        control: MarketControlState,
    ) -> "StrategicContext":
        """
        Resolve strategic context from liquidity and control.
        
        This is the GATING function - determines if trading is permitted.
        
        Args:
            liquidity: The target liquidity intent
            control: The market control state
        
        Returns:
            StrategicContext with validity determined
        """
        # Both must be valid individually
        if not liquidity.valid:
            return cls(
                liquidity=liquidity,
                control=control,
                valid=False,
                alignment_score=0.0,
                reason="Liquidity intent not valid",
            )
        
        if not control.conclusive:
            return cls(
                liquidity=liquidity,
                control=control,
                valid=False,
                alignment_score=0.0,
                reason="Market control inconclusive",
            )
        
        # Check alignment
        aligned = control.is_aligned_with(liquidity)
        
        if not aligned:
            return cls(
                liquidity=liquidity,
                control=control,
                valid=False,
                alignment_score=0.0,
                reason=f"Control ({control.aggressor}) misaligned with liquidity ({liquidity.direction})",
            )
        
        # Calculate alignment score
        # Higher TF control + closer liquidity = better alignment
        tf_factor = min(control.control_tf_rank / 10.0, 1.0)  # Normalize TF
        dist_factor = 1.0 - min(liquidity.distance_atr / 10.0, 1.0)  # Closer = better
        alignment_score = (tf_factor * 0.6) + (dist_factor * 0.4)
        
        return cls(
            liquidity=liquidity,
            control=control,
            valid=True,
            alignment_score=alignment_score,
            reason="Liquidity and control aligned",
        )
    
    @classmethod
    def null(cls) -> "StrategicContext":
        """Return a null context (stand down)"""
        return cls(
            liquidity=LiquidityIntent.null(),
            control=MarketControlState.inconclusive(),
            valid=False,
            alignment_score=0.0,
            reason="No strategic context",
        )
    
    def to_pine_vars(self) -> dict:
        """Export all state as flat Pine-compatible dict"""
        result = {}
        result.update(self.liquidity.to_pine_vars())
        result.update(self.control.to_pine_vars())
        result["ctx_valid"] = 1 if self.valid else 0
        result["ctx_alignment"] = self.alignment_score
        return result


def resolve_market_control_from_timeframes(
    participant_results: List[tuple],  # [(timeframe, ParticipantResult), ...]
) -> MarketControlState:
    """
    Resolve market control by walking TFs top-down.
    
    Process:
        1. Sort by TF (highest first)
        2. Find highest TF where aggressor is conclusive
        3. Return that control state
    
    Args:
        participant_results: List of (timeframe, result) tuples
    
    Returns:
        MarketControlState from highest conclusive TF
    """
    # Sort by TF rank descending (highest first)
    sorted_results = sorted(
        participant_results,
        key=lambda x: TIMEFRAME_RANK.get(x[0], 0),
        reverse=True,
    )
    
    for timeframe, result in sorted_results:
        if result.conviction_level and result.participant_type != 0:
            # Found conclusive control at this TF
            return MarketControlState.from_participant_analysis(
                participant_type=result.participant_type,
                conviction=result.conviction_level,
                sweep_direction=result.participant_type,  # Sweep = aggressor direction
                timeframe=timeframe,
            )
    
    # No conclusive control found
    return MarketControlState.inconclusive()
