from dataclasses import dataclass
from typing import Optional, List
from .enums import LIQUIDITY_DIRECTION, MARKET_CONTROL, TIMEFRAME_RANK

@dataclass
class LiquidityIntent:
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
        priority = TIMEFRAME_RANK.get(timeframe, 0)
        distance = abs(level - current_price)
        distance_atr = distance / atr if atr > 0 else 0.0
        
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
        return cls(
            direction=0,
            level=0.0,
            timeframe="",
            priority=0,
            distance_atr=0.0,
            valid=False,
        )
    
    def to_pine_vars(self) -> dict:
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
        controlling = chain.controlling_liquidity
        if controlling is None:
            return cls.null()
        
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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .liquidity_chain import LiquidityChain

@dataclass
class MarketControlState:
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
        if participant_type == 0 or not conviction:
            return cls.inconclusive()
        
        aggressor = participant_type
        passive = -participant_type  # Other side
        
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
        return cls(
            passive=0,
            aggressor=0,
            control=MARKET_CONTROL["INCONCLUSIVE"],
            control_tf="",
            control_tf_rank=0,
            conclusive=False,
        )
    
    def is_aligned_with(self, liquidity: LiquidityIntent) -> bool:
        if not self.conclusive or not liquidity.valid:
            return False
        
        if self.aggressor == -1 and liquidity.direction == 1:
            return True
        
        if self.aggressor == 1 and liquidity.direction == -1:
            return True
        
        return False
    
    def to_pine_vars(self) -> dict:
        return {
            "mkt_passive": self.passive,
            "mkt_aggressor": self.aggressor,
            "mkt_control": self.control,
            "mkt_control_tf": self.control_tf_rank,
            "mkt_conclusive": 1 if self.conclusive else 0,
        }

@dataclass
class StrategicContext:
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
        
        aligned = control.is_aligned_with(liquidity)
        
        if not aligned:
            return cls(
                liquidity=liquidity,
                control=control,
                valid=False,
                alignment_score=0.0,
                reason=f"Control ({control.aggressor}) misaligned with liquidity ({liquidity.direction})",
            )
        
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
        return cls(
            liquidity=LiquidityIntent.null(),
            control=MarketControlState.inconclusive(),
            valid=False,
            alignment_score=0.0,
            reason="No strategic context",
        )
    
    def to_pine_vars(self) -> dict:
        result = {}
        result.update(self.liquidity.to_pine_vars())
        result.update(self.control.to_pine_vars())
        result["ctx_valid"] = 1 if self.valid else 0
        result["ctx_alignment"] = self.alignment_score
        return result

def resolve_market_control_from_timeframes(
    participant_results: List[tuple],  # [(timeframe, ParticipantResult), ...]
) -> MarketControlState:
    sorted_results = sorted(
        participant_results,
        key=lambda x: TIMEFRAME_RANK.get(x[0], 0),
        reverse=True,
    )
    
    for timeframe, result in sorted_results:
        if result.conviction_level and result.participant_type != 0:
            return MarketControlState.from_participant_analysis(
                participant_type=result.participant_type,
                conviction=result.conviction_level,
                sweep_direction=result.participant_type,  # Sweep = aggressor direction
                timeframe=timeframe,
            )
    
    return MarketControlState.inconclusive()
