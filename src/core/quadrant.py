"""
Quadrant Rule Engine — THE AUTHORITY LAYER

Opposition Rule decides ELIGIBILITY.
Quadrant Rule decides AUTHORITY.

PURPOSE:
    When two timeframes are BOTH conclusive under Opposition Rule
    but DISAGREE, Quadrant Rule decides which one owns the truth.

THE CORE AXIOM:
    The highest conclusive timeframe retains LIQUIDITY (truth).
    Lower conclusive timeframes retain only IMBALANCE (expression).

TRUTH ASSIGNMENT (NON-NEGOTIABLE):
    | Property         | Owned By              |
    |------------------|----------------------|
    | Liquidity        | Highest Conclusive TF |
    | True Direction   | Highest Conclusive TF |
    | Structure Truth  | Highest Conclusive TF |
    | Imbalance        | Lower Conclusive TFs  |
    | Entry Refinement | Lower TFs only        |

CRITICAL CONSEQUENCE:
    If lower TF says SELL but higher TF says BUY:
        - That high is NOT liquidity
        - That move is SELL IMBALANCE
        - Price will NOT respect that high as liquidity
        - It may USE it to sponsor a move, but won't defend it

LOGIC TYPE BY PERIOD:
    Daily & Above:
        - Primary: Closing Range Logic (CRL)
        - Backup: Close-Open Logic (only if visible gap)
    
    Sessional:
        - Primary: Open Price Logic (OPL)
        - Backup: Close-Open Logic (if gap)

TIMEFRAME ELIGIBILITY:
    You CANNOT register a participant on its own timeframe.
    | Participant | Max TF allowed       |
    |-------------|---------------------|
    | Daily       | ≤ 12H (often 8H)    |
    | Weekly      | ≤ Daily             |
    | Sessional   | ≤ 30m (often 5m-15m)|

Pine Translation:
    All structures here are Pine-safe.
    Quadrant resolution is a simple max() over conclusive TFs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Tuple

from .opposition import SignalState, LogicType, PeriodType, AggressorState
from .enums import TIMEFRAME_RANK


# ==============================================================================
# SIGNAL ROLE — LIQUIDITY vs IMBALANCE
# ==============================================================================

class SignalRole(IntEnum):
    """
    The role assigned to a timeframe's signal after quadrant resolution.
    
    LIQUIDITY: This TF owns truth — its direction is THE direction
    IMBALANCE: This TF only marks imbalance — price may use but not defend
    UNRESOLVED: Not yet processed through quadrant resolution
    """
    UNRESOLVED = 0
    IMBALANCE = 1
    LIQUIDITY = 2


# ==============================================================================
# PARTICIPANT SCOPE — WHAT PERIOD DOES THIS PARTICIPANT BELONG TO
# ==============================================================================

class ParticipantScope(IntEnum):
    """
    The scope/period a participant operates in.
    
    Determines which logic type and max TF are valid.
    """
    SESSIONAL = 1   # Asia, London, NY sessions
    DAILY = 2       # Daily participant
    WEEKLY = 3      # Weekly participant
    MONTHLY = 4     # Monthly participant


# ==============================================================================
# TIMEFRAME SIGNAL — A SINGLE TF's CONCLUSIVE STATE
# ==============================================================================

@dataclass
class TimeframeSignal:
    """
    A single timeframe's signal state after Opposition Rule validation.
    
    This is the input to Quadrant Rule resolution.
    
    Attributes:
        tf: Timeframe string (e.g., "H4", "D1", "W1")
        conclusive: Whether this TF passed Opposition Rule
        direction: +1 (buy) or -1 (sell) — only meaningful if conclusive
        logic_type: The logic used to determine conclusiveness
        role: Assigned after quadrant resolution (LIQUIDITY or IMBALANCE)
        aggressor: The underlying aggressor state from Opposition Rule
    """
    tf: str
    conclusive: bool
    direction: int  # +1 or -1
    logic_type: LogicType = LogicType.CRL
    role: SignalRole = SignalRole.UNRESOLVED
    aggressor: Optional[AggressorState] = None
    
    # Reference levels for this TF
    liquidity_high: float = 0.0
    liquidity_low: float = 0.0
    
    @property
    def rank(self) -> int:
        """Get the hierarchical rank of this timeframe."""
        return TIMEFRAME_RANK.get(self.tf, 0)
    
    @property
    def is_liquidity_owner(self) -> bool:
        """Check if this TF owns liquidity (is HCT)."""
        return self.role == SignalRole.LIQUIDITY
    
    @property
    def is_imbalance_only(self) -> bool:
        """Check if this TF only provides imbalance."""
        return self.role == SignalRole.IMBALANCE
    
    def to_pine_vars(self) -> dict:
        """Export as Pine-compatible dict."""
        return {
            f"tf_{self.tf}_conclusive": 1 if self.conclusive else 0,
            f"tf_{self.tf}_direction": self.direction,
            f"tf_{self.tf}_role": int(self.role),
            f"tf_{self.tf}_liq_high": self.liquidity_high,
            f"tf_{self.tf}_liq_low": self.liquidity_low,
        }


# ==============================================================================
# TIMEFRAME ELIGIBILITY — VALIDATION RULES
# ==============================================================================

# Maximum allowed TF for participant registration by scope
MAX_TF_BY_SCOPE: Dict[ParticipantScope, str] = {
    ParticipantScope.SESSIONAL: "M30",   # Sessional max = 30m
    ParticipantScope.DAILY: "H12",       # Daily max = 12H (often 8H)
    ParticipantScope.WEEKLY: "D1",       # Weekly max = Daily
    ParticipantScope.MONTHLY: "W1",      # Monthly max = Weekly
}

# Preferred logic type by scope
LOGIC_BY_SCOPE: Dict[ParticipantScope, LogicType] = {
    ParticipantScope.SESSIONAL: LogicType.OPL,  # Sessions use Open Price Logic
    ParticipantScope.DAILY: LogicType.CRL,      # Daily uses Closing Range Logic
    ParticipantScope.WEEKLY: LogicType.CRL,
    ParticipantScope.MONTHLY: LogicType.CRL,
}


def is_tf_eligible(
    tf: str,
    scope: ParticipantScope,
) -> bool:
    """
    Check if a timeframe is eligible for participant registration.
    
    RULE: You CANNOT register a participant on its own timeframe.
    
    Args:
        tf: The timeframe to check
        scope: The participant's scope
    
    Returns:
        True if TF is eligible for this scope
    """
    max_tf = MAX_TF_BY_SCOPE.get(scope, "D1")
    max_rank = TIMEFRAME_RANK.get(max_tf, 9)
    tf_rank = TIMEFRAME_RANK.get(tf, 0)
    
    return tf_rank <= max_rank


def get_preferred_logic(scope: ParticipantScope) -> LogicType:
    """
    Get the preferred logic type for a participant scope.
    
    Daily & Above: CRL (Closing Range Logic)
    Sessional: OPL (Open Price Logic)
    """
    return LOGIC_BY_SCOPE.get(scope, LogicType.CRL)


# ==============================================================================
# QUADRANT RESOLUTION — THE CORE ALGORITHM
# ==============================================================================

@dataclass
class QuadrantResult:
    """
    Result of quadrant resolution.
    
    Identifies the Highest Conclusive Timeframe (HCT) and assigns roles.
    """
    hct: Optional[TimeframeSignal] = None  # Highest Conclusive Timeframe
    signals: List[TimeframeSignal] = field(default_factory=list)
    resolved: bool = False
    
    @property
    def liquidity_direction(self) -> int:
        """The true direction — owned by HCT."""
        if self.hct:
            return self.hct.direction
        return 0
    
    @property
    def liquidity_tf(self) -> Optional[str]:
        """The timeframe that owns liquidity."""
        if self.hct:
            return self.hct.tf
        return None
    
    @property
    def imbalance_signals(self) -> List[TimeframeSignal]:
        """All signals that are imbalance-only."""
        return [s for s in self.signals if s.role == SignalRole.IMBALANCE]
    
    @property
    def has_conflict(self) -> bool:
        """Check if there was a conflict (disagreement between conclusive TFs)."""
        conclusive = [s for s in self.signals if s.conclusive]
        if len(conclusive) < 2:
            return False
        
        directions = set(s.direction for s in conclusive)
        return len(directions) > 1
    
    def to_pine_vars(self) -> dict:
        """Export as Pine-compatible dict."""
        return {
            "quadrant_resolved": 1 if self.resolved else 0,
            "quadrant_hct_tf": self.hct.tf if self.hct else "",
            "quadrant_liq_direction": self.liquidity_direction,
            "quadrant_has_conflict": 1 if self.has_conflict else 0,
            "quadrant_imbalance_count": len(self.imbalance_signals),
        }


def resolve_quadrant(
    signals: List[TimeframeSignal],
) -> QuadrantResult:
    """
    Resolve quadrant — determine which TF owns truth.
    
    THE FORMAL RULE:
        If two timeframes are conclusive but disagree:
        The higher timeframe decides liquidity & bias.
        The lower timeframe is reclassified as imbalance only.
    
    Args:
        signals: List of TimeframeSignal from Opposition Rule validation
    
    Returns:
        QuadrantResult with HCT and role assignments
    
    Example:
        signals = [
            TimeframeSignal(tf="H4", conclusive=True, direction=1),
            TimeframeSignal(tf="H1", conclusive=True, direction=-1),
            TimeframeSignal(tf="M15", conclusive=False, direction=0),
        ]
        
        result = resolve_quadrant(signals)
        # result.hct.tf == "H4" (highest conclusive)
        # result.liquidity_direction == 1 (buy)
        # H1 is reclassified as IMBALANCE only
    """
    # Filter to conclusive signals only
    conclusive = [s for s in signals if s.conclusive]
    
    if not conclusive:
        # No conclusive signals — cannot resolve
        return QuadrantResult(
            hct=None,
            signals=signals,
            resolved=False,
        )
    
    # Find Highest Conclusive Timeframe (HCT)
    hct = max(conclusive, key=lambda s: s.rank)
    
    # Assign roles
    for s in signals:
        if not s.conclusive:
            s.role = SignalRole.UNRESOLVED
        elif s.tf == hct.tf:
            s.role = SignalRole.LIQUIDITY
        else:
            # Lower conclusive TF = imbalance only
            s.role = SignalRole.IMBALANCE
    
    return QuadrantResult(
        hct=hct,
        signals=signals,
        resolved=True,
    )


# ==============================================================================
# MULTI-SCOPE RESOLVER — HANDLE DAILY + SESSIONAL TOGETHER
# ==============================================================================

@dataclass
class MultiScopeResult:
    """
    Result of multi-scope quadrant resolution.
    
    Handles the case where both daily and sessional participants
    have conclusive signals.
    """
    daily_result: Optional[QuadrantResult] = None
    sessional_result: Optional[QuadrantResult] = None
    
    @property
    def master_direction(self) -> int:
        """
        The master direction — daily takes precedence over sessional.
        """
        if self.daily_result and self.daily_result.hct:
            return self.daily_result.liquidity_direction
        if self.sessional_result and self.sessional_result.hct:
            return self.sessional_result.liquidity_direction
        return 0
    
    @property
    def sessional_aligned(self) -> bool:
        """Check if sessional aligns with daily."""
        if not self.daily_result or not self.sessional_result:
            return True  # No conflict if one is missing
        
        if not self.daily_result.hct or not self.sessional_result.hct:
            return True
        
        return (
            self.daily_result.liquidity_direction == 
            self.sessional_result.liquidity_direction
        )
    
    def to_pine_vars(self) -> dict:
        """Export as Pine-compatible dict."""
        vars = {
            "multi_master_direction": self.master_direction,
            "multi_sessional_aligned": 1 if self.sessional_aligned else 0,
        }
        
        if self.daily_result:
            vars.update({f"daily_{k}": v for k, v in self.daily_result.to_pine_vars().items()})
        if self.sessional_result:
            vars.update({f"sess_{k}": v for k, v in self.sessional_result.to_pine_vars().items()})
        
        return vars


def resolve_multi_scope(
    daily_signals: List[TimeframeSignal],
    sessional_signals: List[TimeframeSignal],
) -> MultiScopeResult:
    """
    Resolve quadrant across multiple scopes.
    
    Daily always takes precedence over sessional for master direction.
    
    Args:
        daily_signals: Signals for daily participant (H1-H12 typically)
        sessional_signals: Signals for sessional participant (M5-M30 typically)
    
    Returns:
        MultiScopeResult with both resolutions and master direction
    """
    daily_result = resolve_quadrant(daily_signals) if daily_signals else None
    sessional_result = resolve_quadrant(sessional_signals) if sessional_signals else None
    
    return MultiScopeResult(
        daily_result=daily_result,
        sessional_result=sessional_result,
    )


# ==============================================================================
# IMBALANCE QUALIFICATION — WHAT LOWER TFs PROVIDE
# ==============================================================================

@dataclass
class ImbalanceZone:
    """
    An imbalance zone identified from a lower conclusive TF.
    
    This is NOT liquidity — price may use it but won't defend it.
    """
    high: float
    low: float
    direction: int  # +1 = bullish imbalance, -1 = bearish imbalance
    source_tf: str
    filled: bool = False
    
    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2
    
    def contains_price(self, price: float) -> bool:
        return self.low <= price <= self.high
    
    def to_pine_vars(self) -> dict:
        return {
            "imb_high": self.high,
            "imb_low": self.low,
            "imb_direction": self.direction,
            "imb_filled": 1 if self.filled else 0,
        }


def extract_imbalance_zones(
    result: QuadrantResult,
) -> List[ImbalanceZone]:
    """
    Extract imbalance zones from lower conclusive TFs.
    
    RULE: Lower conclusive TFs don't own liquidity — they mark imbalance.
    
    Args:
        result: QuadrantResult from resolve_quadrant()
    
    Returns:
        List of ImbalanceZone from all IMBALANCE-role signals
    """
    zones = []
    
    for signal in result.signals:
        if signal.role == SignalRole.IMBALANCE:
            # This TF's range is imbalance, not liquidity
            zones.append(ImbalanceZone(
                high=signal.liquidity_high,
                low=signal.liquidity_low,
                direction=signal.direction,
                source_tf=signal.tf,
            ))
    
    return zones


# ==============================================================================
# DOCTRINE SUMMARY (FOR REFERENCE)
# ==============================================================================

QUADRANT_DOCTRINE = """
Opposition Rule decides ELIGIBILITY.
Quadrant Rule decides AUTHORITY.
Highest conclusive timeframe owns TRUTH.
Lower timeframes only EXPRESS it.

CRITICAL CONSEQUENCE:
    If lower TF says SELL but higher TF says BUY:
        ❌ That high is NOT liquidity
        ✅ That move is SELL IMBALANCE
        → Price will NOT respect that high
        → It may USE it to sponsor a move
        → But it will NOT defend it as liquidity

This is institutional-grade logic.
"""
