"""
Imbalance-Liquidity Relationship Engine — THE MECHANICAL VALIDATION LAYER

This module implements the 6 GOVERNING RULES for imbalance validity.

THE ONE-SENTENCE SUMMARY:
    Liquidity invalidates imbalance by default —
    except when it is defending trend, or when it created the zone.
    Trapped liquidity only works when nested inside a higher-range zone.

THE 6 RULES:
    RULE 1: Same tier must match same tier (Daily Liq → Daily Imb)
    RULE 2: Imbalance is "extreme value", not random candles
    RULE 3: Liquidity cuts invalidate imbalance (DEFAULT)
    RULE 4: Only TWO exceptions exist (defense OR creator)
    RULE 5: Trapped liquidity needs TWO zones (bigger + smaller)
    RULE 6: Price targets the trap, not the noise

DECISION FLOW:
    1. Identify same-tier liquidity & imbalance
    2. Has liquidity cut this imbalance?
        - No → zone valid
        - Yes → check exceptions
    3. Is cutting liquidity defense OR creator?
        - Yes → zone valid
        - No → zone INVALID
    4. For traps: confirm bigger + smaller range zones exist

Pine Translation:
    All structures here are Pine-safe.
    Validation is a series of boolean checks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Tuple

from .quadrant import ParticipantScope, TimeframeSignal, SignalRole
from .enums import TIMEFRAME_RANK


# ==============================================================================
# IMBALANCE STATE — VALID, INVALID, OR CONDITIONAL
# ==============================================================================

class ImbalanceState(IntEnum):
    """
    The validation state of an imbalance zone.
    
    VALID: Zone can be used for entries
    INVALID: Zone has been cut by liquidity (no exception applies)
    CONDITIONAL: Zone is cut but exception may apply (needs confirmation)
    PENDING: Not yet validated
    """
    PENDING = 0
    VALID = 1
    INVALID = 2
    CONDITIONAL = 3


# ==============================================================================
# IMBALANCE TYPE — WHAT KIND OF EXTREME VALUE
# ==============================================================================

class ImbalanceType(IntEnum):
    """
    The type of imbalance based on participant control shift.
    
    FEAR: Seller exhaustion → buyer aggression (buy zone)
    EUPHORIA: Buyer exhaustion → seller aggression (sell zone)
    """
    UNKNOWN = 0
    FEAR = 1       # Buy zone — extreme pessimism before reversal
    EUPHORIA = 2   # Sell zone — extreme optimism before reversal


# ==============================================================================
# TIER — FOR SAME-TIER MATCHING (RULE 1)
# ==============================================================================

class Tier(IntEnum):
    """
    Timeframe tier for same-tier matching.
    
    RULE 1: Liquidity and Imbalance must be merged on the SAME tier.
    Mixing tiers = invalid inference.
    """
    SESSIONAL = 1   # M1 - M30
    INTRADAY = 2    # H1 - H8
    DAILY = 3       # H12 - D1
    WEEKLY = 4      # W1
    MONTHLY = 5     # MN


def get_tier(tf: str) -> Tier:
    """
    Get the tier for a timeframe.
    
    Used for RULE 1 validation.
    """
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


# ==============================================================================
# IMBALANCE ZONE — THE CORE STRUCTURE
# ==============================================================================

@dataclass
class Imbalance:
    """
    An imbalance zone with full validation state.
    
    An imbalance is the EXTREME OF VALUE where control shifts:
        - Fear value (for buys) — seller exhaustion
        - Euphoria value (for sells) — buyer exhaustion
    
    If price hasn't moved away aggressively, it's NOT imbalance.
    """
    high: float
    low: float
    timeframe: str
    imbalance_type: ImbalanceType
    
    # Validation state
    state: ImbalanceState = ImbalanceState.PENDING
    
    # Creation metadata
    creation_timestamp: int = 0
    creation_candle_index: int = 0
    
    # The liquidity that created this zone (for RULE 4 exception 2)
    created_by_liquidity: Optional[float] = None
    
    # Cut tracking
    cut_by_liquidity: Optional[float] = None
    cut_timestamp: int = 0
    
    # Exception tracking
    is_defense_liquidity: bool = False
    is_creator_liquidity: bool = False
    
    @property
    def tier(self) -> Tier:
        """Get the tier of this imbalance."""
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


# ==============================================================================
# LIQUIDITY LEVEL — FOR CUT DETECTION
# ==============================================================================

@dataclass
class LiquidityLevel:
    """
    A liquidity level that may cut imbalances.
    """
    price: float
    timeframe: str
    direction: int  # +1 = buy-side (high), -1 = sell-side (low)
    timestamp: int = 0
    
    # Is this the most premium/discounted defense?
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


# ==============================================================================
# RULE 1: SAME TIER VALIDATION
# ==============================================================================

def validate_same_tier(
    imbalance: Imbalance,
    liquidity: LiquidityLevel,
) -> bool:
    """
    RULE 1: Same tier must match same tier.
    
    Daily Liquidity → Daily Imbalance
    Weekly Liquidity → Weekly Imbalance
    Sessional Liquidity → Sessional Imbalance
    
    Mixing tiers = invalid inference.
    
    Returns:
        True if same tier (valid pairing)
        False if different tiers (invalid pairing)
    """
    return imbalance.tier == liquidity.tier


# ==============================================================================
# RULE 3: LIQUIDITY CUT DETECTION
# ==============================================================================

def has_liquidity_cut(
    imbalance: Imbalance,
    liquidity: LiquidityLevel,
    current_price: float,
) -> bool:
    """
    RULE 3: Check if liquidity has cut through an imbalance.
    
    A cut occurs when:
        - Price moved from liquidity level
        - Through the imbalance zone
        - Without the liquidity being "taken"
    
    This is the DEFAULT invalidation check.
    
    Args:
        imbalance: The imbalance zone to check
        liquidity: The liquidity level
        current_price: Current price
    
    Returns:
        True if liquidity has cut through the imbalance
    """
    # For buy-side liquidity (high) cutting a fear zone (buy imbalance)
    if liquidity.is_buy_side and imbalance.imbalance_type == ImbalanceType.FEAR:
        # If price is below liquidity and liquidity is above imbalance
        if liquidity.price > imbalance.high:
            # Liquidity sits above the zone — potential cut
            if current_price < imbalance.low:
                # Price has moved through the zone — CUT
                return True
    
    # For sell-side liquidity (low) cutting a euphoria zone (sell imbalance)
    if liquidity.is_sell_side and imbalance.imbalance_type == ImbalanceType.EUPHORIA:
        # If liquidity is below imbalance
        if liquidity.price < imbalance.low:
            # Liquidity sits below the zone — potential cut
            if current_price > imbalance.high:
                # Price has moved through the zone — CUT
                return True
    
    return False


# ==============================================================================
# RULE 4 EXCEPTION 1: DEFENSE LIQUIDITY
# ==============================================================================

def is_defense_liquidity(
    liquidity: LiquidityLevel,
    trend_direction: int,
    all_liquidity: List[LiquidityLevel],
) -> bool:
    """
    RULE 4 EXCEPTION 1: Defense Liquidity.
    
    Liquidity is allowed to cut the zone ONLY IF it is:
        - Most DISCOUNTED liquidity → defense for BUY trend
        - Most PREMIUM liquidity → defense for SELL trend
    
    If that liquidity is still the last defended line, it can sponsor price.
    
    Args:
        liquidity: The liquidity level to check
        trend_direction: +1 for bullish, -1 for bearish
        all_liquidity: All known liquidity levels (same tier)
    
    Returns:
        True if this is defense liquidity
    """
    same_tier = [l for l in all_liquidity if l.tier == liquidity.tier]
    
    if not same_tier:
        return False
    
    if trend_direction == 1:  # Bullish trend
        # Defense = most DISCOUNTED (lowest low)
        sell_side = [l for l in same_tier if l.is_sell_side]
        if sell_side:
            most_discounted = min(sell_side, key=lambda l: l.price)
            return liquidity.price == most_discounted.price
    
    elif trend_direction == -1:  # Bearish trend
        # Defense = most PREMIUM (highest high)
        buy_side = [l for l in same_tier if l.is_buy_side]
        if buy_side:
            most_premium = max(buy_side, key=lambda l: l.price)
            return liquidity.price == most_premium.price
    
    return False


# ==============================================================================
# RULE 4 EXCEPTION 2: LIQUIDITY-CREATED ZONE
# ==============================================================================

def is_creator_liquidity(
    imbalance: Imbalance,
    liquidity: LiquidityLevel,
) -> bool:
    """
    RULE 4 EXCEPTION 2: Liquidity-Created Zone.
    
    If an imbalance is CREATED BY the liquidity range itself, then
    that same liquidity CANNOT invalidate its own zone.
    
    Why? Because it is the SOURCE, not an external attacker.
    Only EXTERNAL liquidity matters for invalidation.
    
    Args:
        imbalance: The imbalance zone
        liquidity: The liquidity level
    
    Returns:
        True if this liquidity created the zone
    """
    if imbalance.created_by_liquidity is None:
        return False
    
    # Check if the liquidity price matches the creator
    # (with small tolerance for floating point)
    tolerance = imbalance.range_size * 0.1  # 10% of zone size
    return abs(liquidity.price - imbalance.created_by_liquidity) <= tolerance


# ==============================================================================
# MAIN VALIDATOR: VALIDATE IMBALANCE (ALL RULES)
# ==============================================================================

def validate_imbalance(
    imbalance: Imbalance,
    liquidity_levels: List[LiquidityLevel],
    current_price: float,
    trend_direction: int,
) -> Imbalance:
    """
    Validate an imbalance against all 6 rules.
    
    DECISION FLOW:
        1. Filter to same-tier liquidity (RULE 1)
        2. Check if any liquidity cuts the imbalance (RULE 3)
        3. If cut, check exceptions (RULE 4)
        4. Update state accordingly
    
    Args:
        imbalance: The imbalance zone to validate
        liquidity_levels: All known liquidity levels
        current_price: Current price
        trend_direction: +1 bullish, -1 bearish
    
    Returns:
        Updated Imbalance with validation state
    """
    # RULE 1: Filter to same tier only
    same_tier_liquidity = [
        l for l in liquidity_levels 
        if validate_same_tier(imbalance, l)
    ]
    
    # If no same-tier liquidity, zone is valid by default
    if not same_tier_liquidity:
        imbalance.state = ImbalanceState.VALID
        return imbalance
    
    # RULE 3: Check for liquidity cuts
    cutting_liquidity = None
    for liq in same_tier_liquidity:
        if has_liquidity_cut(imbalance, liq, current_price):
            cutting_liquidity = liq
            imbalance.cut_by_liquidity = liq.price
            break
    
    # No cut → zone is valid
    if cutting_liquidity is None:
        imbalance.state = ImbalanceState.VALID
        return imbalance
    
    # RULE 4: Check exceptions
    
    # Exception 1: Defense liquidity
    if is_defense_liquidity(cutting_liquidity, trend_direction, same_tier_liquidity):
        imbalance.is_defense_liquidity = True
        imbalance.state = ImbalanceState.VALID
        return imbalance
    
    # Exception 2: Creator liquidity
    if is_creator_liquidity(imbalance, cutting_liquidity):
        imbalance.is_creator_liquidity = True
        imbalance.state = ImbalanceState.VALID
        return imbalance
    
    # No exception applies → zone is INVALID
    imbalance.state = ImbalanceState.INVALID
    return imbalance


# ==============================================================================
# RULE 5: TRAPPED LIQUIDITY NESTING
# ==============================================================================

@dataclass
class TrapSetup:
    """
    A trapped liquidity setup requiring TWO zones.
    
    RULE 5: Trapped liquidity needs:
        1. A BIGGER-RANGE zone (higher TF, may have liquidity cuts)
        2. A SMALLER-RANGE zone (same TF, must be clean)
    
    No bigger-range zone = no valid trap.
    """
    bigger_zone: Imbalance
    smaller_zone: Imbalance
    trapped_liquidity: LiquidityLevel
    
    valid: bool = False
    
    @property
    def target_price(self) -> float:
        """The true target — the trapped liquidity's zone."""
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
    """
    RULE 5: Validate a trapped liquidity setup.
    
    Requirements:
        1. Bigger-range zone exists (may have liquidity cuts — allowed)
        2. Smaller-range zone is CLEAN (no cuts, or exception applies)
        3. Zones are properly nested
    
    Args:
        bigger_zone: The higher-TF zone (e.g., daily)
        smaller_zone: The same-TF zone as trapped liquidity
        trapped_liquidity: The trapped liquidity level
        liquidity_levels: All known liquidity levels
        current_price: Current price
        trend_direction: +1 bullish, -1 bearish
    
    Returns:
        TrapSetup with validity state
    """
    setup = TrapSetup(
        bigger_zone=bigger_zone,
        smaller_zone=smaller_zone,
        trapped_liquidity=trapped_liquidity,
        valid=False,
    )
    
    # Check 1: Bigger zone must be higher tier
    if bigger_zone.tier <= smaller_zone.tier:
        return setup  # Invalid — bigger must be higher TF
    
    # Check 2: Smaller zone must be same tier as trapped liquidity
    if smaller_zone.tier != trapped_liquidity.tier:
        return setup  # Invalid — tier mismatch
    
    # Check 3: Zones must be nested (smaller inside bigger)
    if not (bigger_zone.low <= smaller_zone.low and 
            smaller_zone.high <= bigger_zone.high):
        return setup  # Invalid — not properly nested
    
    # Check 4: Smaller zone must be valid (RULE 3 + 4)
    validated_smaller = validate_imbalance(
        smaller_zone,
        liquidity_levels,
        current_price,
        trend_direction,
    )
    
    if validated_smaller.state != ImbalanceState.VALID:
        return setup  # Invalid — smaller zone is cut
    
    # All checks passed
    setup.valid = True
    return setup


# ==============================================================================
# BATCH VALIDATION — VALIDATE ALL IMBALANCES
# ==============================================================================

def validate_all_imbalances(
    imbalances: List[Imbalance],
    liquidity_levels: List[LiquidityLevel],
    current_price: float,
    trend_direction: int,
) -> List[Imbalance]:
    """
    Validate all imbalances in a list.
    
    Returns list with updated validation states.
    """
    return [
        validate_imbalance(imb, liquidity_levels, current_price, trend_direction)
        for imb in imbalances
    ]


def get_valid_imbalances(
    imbalances: List[Imbalance],
) -> List[Imbalance]:
    """
    Filter to only valid imbalances.
    """
    return [imb for imb in imbalances if imb.is_valid]


def get_invalid_imbalances(
    imbalances: List[Imbalance],
) -> List[Imbalance]:
    """
    Filter to only invalid imbalances.
    """
    return [imb for imb in imbalances if imb.is_invalid]


# ==============================================================================
# DOCTRINE SUMMARY
# ==============================================================================

IMBALANCE_LIQUIDITY_DOCTRINE = """
THE 6 GOVERNING RULES:

RULE 1: Same tier must match same tier
    - Daily Liquidity → Daily Imbalance
    - Mixing tiers = invalid inference

RULE 2: Imbalance is "extreme value", not random candles
    - Fear value (for buys) — seller exhaustion
    - Euphoria value (for sells) — buyer exhaustion
    - If price hasn't moved away aggressively, it's NOT imbalance

RULE 3: Liquidity cuts invalidate imbalance (DEFAULT)
    - If liquidity cuts through zone → zone is INVALID
    - This explains "why my zone didn't hold"

RULE 4: Only TWO exceptions exist
    - EXCEPTION 1: Defense liquidity (most discounted/premium)
    - EXCEPTION 2: Creator liquidity (source can't invalidate itself)

RULE 5: Trapped liquidity needs TWO zones
    - Bigger-range zone (higher TF, may have cuts)
    - Smaller-range zone (same TF, must be clean)
    - No bigger zone = no valid trap

RULE 6: Price targets the trap, not the noise
    - Other liquidities are "bonus reactions"
    - The true target is the trapped liquidity's zone

ONE SENTENCE:
    Liquidity invalidates imbalance by default —
    except when it is defending trend, or when it created the zone.
    Trapped liquidity only works when nested inside a higher-range zone.
"""
