from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Tuple

from .quadrant import ParticipantScope, TimeframeSignal, SignalRole
from .enums import TIMEFRAME_RANK

class ImbalanceState(IntEnum):
    PENDING = 0
    VALID = 1
    INVALID = 2
    CONDITIONAL = 3

class ImbalanceType(IntEnum):
    UNKNOWN = 0
    FEAR = 1       # Buy zone — extreme pessimism before reversal
    EUPHORIA = 2   # Sell zone — extreme optimism before reversal

class Tier(IntEnum):
    SESSIONAL = 1   # M1 - M30
    INTRADAY = 2    # H1 - H8
    DAILY = 3       # H12 - D1
    WEEKLY = 4      # W1
    MONTHLY = 5     # MN

def get_tier(tf: str) -> Tier:
    rank = TIMEFRAME_RANK.get(tf, 0)
    
    if rank <= 3:      # M1 - M30
        return Tier.SESSIONAL
    elif rank <= 7:    # H1 - H8
        return Tier.INTRADAY
    elif rank <= 9:    # H12 - D1
        return Tier.DAILY
    elif rank == 10:   # W1
        return Tier.WEEKLY
    else:              # MN
        return Tier.MONTHLY

@dataclass
class Imbalance:
    high: float
    low: float
    timeframe: str
    imbalance_type: ImbalanceType
    
    state: ImbalanceState = ImbalanceState.PENDING
    
    creation_timestamp: int = 0
    creation_candle_index: int = 0
    
    created_by_liquidity: Optional[float] = None
    
    cut_by_liquidity: Optional[float] = None
    cut_timestamp: int = 0
    
    is_defense_liquidity: bool = False
    is_creator_liquidity: bool = False
    
    @property
    def tier(self) -> Tier:
        return get_tier(self.timeframe)
    
    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2
    
    @property
    def range_size(self) -> float:
        return self.high - self.low
    
    @property
    def is_valid(self) -> bool:
        return self.state == ImbalanceState.VALID
    
    @property
    def is_invalid(self) -> bool:
        return self.state == ImbalanceState.INVALID
    
    def contains_price(self, price: float) -> bool:
        return self.low <= price <= self.high
    
    def to_pine_vars(self) -> dict:
        return {
            "imb_high": self.high,
            "imb_low": self.low,
            "imb_state": int(self.state),
            "imb_type": int(self.imbalance_type),
            "imb_valid": 1 if self.is_valid else 0,
            "imb_cut": 1 if self.cut_by_liquidity else 0,
        }

@dataclass
class LiquidityLevel:
    price: float
    timeframe: str
    direction: int  # +1 = buy-side (high), -1 = sell-side (low)
    timestamp: int = 0
    
    is_defense: bool = False
    
    @property
    def tier(self) -> Tier:
        return get_tier(self.timeframe)
    
    @property
    def is_buy_side(self) -> bool:
        return self.direction == 1
    
    @property
    def is_sell_side(self) -> bool:
        return self.direction == -1

def validate_same_tier(
    imbalance: Imbalance,
    liquidity: LiquidityLevel,
) -> bool:
    return imbalance.tier == liquidity.tier

def has_liquidity_cut(
    imbalance: Imbalance,
    liquidity: LiquidityLevel,
    current_price: float,
) -> bool:
    if liquidity.is_buy_side and imbalance.imbalance_type == ImbalanceType.FEAR:
        if liquidity.price > imbalance.high:
            if current_price < imbalance.low:
                return True
    
    if liquidity.is_sell_side and imbalance.imbalance_type == ImbalanceType.EUPHORIA:
        if liquidity.price < imbalance.low:
            if current_price > imbalance.high:
                return True
    
    return False

def is_defense_liquidity(
    liquidity: LiquidityLevel,
    trend_direction: int,
    all_liquidity: List[LiquidityLevel],
) -> bool:
    same_tier = [l for l in all_liquidity if l.tier == liquidity.tier]
    
    if not same_tier:
        return False
    
    if trend_direction == 1:  # Bullish trend
        sell_side = [l for l in same_tier if l.is_sell_side]
        if sell_side:
            most_discounted = min(sell_side, key=lambda l: l.price)
            return liquidity.price == most_discounted.price
    
    elif trend_direction == -1:  # Bearish trend
        buy_side = [l for l in same_tier if l.is_buy_side]
        if buy_side:
            most_premium = max(buy_side, key=lambda l: l.price)
            return liquidity.price == most_premium.price
    
    return False

def is_creator_liquidity(
    imbalance: Imbalance,
    liquidity: LiquidityLevel,
) -> bool:
    if imbalance.created_by_liquidity is None:
        return False
    
    tolerance = imbalance.range_size * 0.1  # 10% of zone size
    return abs(liquidity.price - imbalance.created_by_liquidity) <= tolerance

def validate_imbalance(
    imbalance: Imbalance,
    liquidity_levels: List[LiquidityLevel],
    current_price: float,
    trend_direction: int,
) -> Imbalance:
    same_tier_liquidity = [
        l for l in liquidity_levels 
        if validate_same_tier(imbalance, l)
    ]
    
    if not same_tier_liquidity:
        imbalance.state = ImbalanceState.VALID
        return imbalance
    
    cutting_liquidity = None
    for liq in same_tier_liquidity:
        if has_liquidity_cut(imbalance, liq, current_price):
            cutting_liquidity = liq
            imbalance.cut_by_liquidity = liq.price
            break
    
    if cutting_liquidity is None:
        imbalance.state = ImbalanceState.VALID
        return imbalance
    
    if is_defense_liquidity(cutting_liquidity, trend_direction, same_tier_liquidity):
        imbalance.is_defense_liquidity = True
        imbalance.state = ImbalanceState.VALID
        return imbalance
    
    if is_creator_liquidity(imbalance, cutting_liquidity):
        imbalance.is_creator_liquidity = True
        imbalance.state = ImbalanceState.VALID
        return imbalance
    
    imbalance.state = ImbalanceState.INVALID
    return imbalance

@dataclass
class TrapSetup:
    bigger_zone: Imbalance
    smaller_zone: Imbalance
    trapped_liquidity: LiquidityLevel
    
    valid: bool = False
    
    @property
    def target_price(self) -> float:
        return self.smaller_zone.midpoint
    
    def to_pine_vars(self) -> dict:
        return {
            "trap_valid": 1 if self.valid else 0,
            "trap_bigger_high": self.bigger_zone.high,
            "trap_bigger_low": self.bigger_zone.low,
            "trap_smaller_high": self.smaller_zone.high,
            "trap_smaller_low": self.smaller_zone.low,
            "trap_target": self.target_price,
        }

def validate_trap_setup(
    bigger_zone: Imbalance,
    smaller_zone: Imbalance,
    trapped_liquidity: LiquidityLevel,
    liquidity_levels: List[LiquidityLevel],
    current_price: float,
    trend_direction: int,
) -> TrapSetup:
    setup = TrapSetup(
        bigger_zone=bigger_zone,
        smaller_zone=smaller_zone,
        trapped_liquidity=trapped_liquidity,
        valid=False,
    )
    
    if bigger_zone.tier <= smaller_zone.tier:
        return setup  # Invalid — bigger must be higher TF
    
    if smaller_zone.tier != trapped_liquidity.tier:
        return setup  # Invalid — tier mismatch
    
    if not (bigger_zone.low <= smaller_zone.low and 
            smaller_zone.high <= bigger_zone.high):
        return setup  # Invalid — not properly nested
    
    validated_smaller = validate_imbalance(
        smaller_zone,
        liquidity_levels,
        current_price,
        trend_direction,
    )
    
    if validated_smaller.state != ImbalanceState.VALID:
        return setup  # Invalid — smaller zone is cut
    
    setup.valid = True
    return setup

def validate_all_imbalances(
    imbalances: List[Imbalance],
    liquidity_levels: List[LiquidityLevel],
    current_price: float,
    trend_direction: int,
) -> List[Imbalance]:
    return [
        validate_imbalance(imb, liquidity_levels, current_price, trend_direction)
        for imb in imbalances
    ]

def get_valid_imbalances(
    imbalances: List[Imbalance],
) -> List[Imbalance]:
    return [imb for imb in imbalances if imb.is_valid]

def get_invalid_imbalances(
    imbalances: List[Imbalance],
) -> List[Imbalance]:
    return [imb for imb in imbalances if imb.is_invalid]
