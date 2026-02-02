"""
Signal Intermediate Representation (IR)

Pine-compatible signal schema - enforces portability constraints from day one.

DESIGN PRINCIPLES:
- Every field MUST be expressible in Pine Script v5
- Only simple types: int, float, bool (NO datetime, NO None)
- No classes, no dynamic objects, no unbounded memory
- State persists using Pine's var/varip primitives only
- Frozen dataclass = immutable = safe for deterministic replay

PORTABILITY GUARANTEE:
If it compiles to SignalIR, it can be ported to Pine Script.
This is the enforcement firewall for deployment compatibility.

CRITICAL RULES:
1. datetime → int (unix milliseconds)
2. Optional[float] → float (use math.nan) + bool flag
3. No None values anywhere (Pine has no None, only na)
"""

from dataclasses import dataclass
import math

# Import locked enum contracts
# CRITICAL: These numbers must NEVER change after deployment
from .enums import WAVELENGTH_STATE, GAP_TYPE, BIAS, PARTICIPANT_CONTROL, DEBUG_FLAGS, LIQUIDITY_DIRECTION, MARKET_CONTROL


@dataclass(frozen=True)
class SignalIR:
    """
    Pine-compatible intermediate representation for HORC signals.
    
    CONTRACT GUARANTEES:
    1. All fields are primitive types (int, float, bool) - NO datetime, NO None
    2. No nested objects or dynamic collections
    3. All state fits in Pine's var/varip persistence model
    4. Deterministic: same input → same output, always
    5. Bar-local: no hidden future context
    
    CRITICAL PATTERNS:
        datetime → int (unix ms)
        Optional[float] → float (math.nan) + bool flag
        None → math.nan or False (never use None)
    """
    
    # ===================================================================
    # METADATA
    # ===================================================================
    timestamp: int
    """
    Bar close time as UNIX milliseconds.
    Pine: int(time)
    Convert from datetime: int(dt.timestamp() * 1000)
    """
    
    # ===================================================================
    # CORE SIGNAL - The Decision Layer
    # ===================================================================
    bias: int
    """
    Overall directional bias after multi-engine confluence.
    -1 = bearish, 0 = neutral, +1 = bullish
    Pine: var int bias = 0
    """
    
    actionable: bool
    """
    True only if confluence threshold met AND bias is non-zero.
    Pine: var bool actionable = false
    """
    
    confidence: float
    """
    Final weighted confluence score [0.0, 1.0].
    Pine: var float confidence = 0.0
    """
    
    # ===================================================================
    # PARTICIPANT CONTROL (AXIOM 2: First Move Determinism)
    # ===================================================================
    participant_control: int
    """
    Who controls the opening range?
    -1 = sellers swept low, +1 = buyers swept high, 0 = undetermined
    Pine: var int participant_control = 0
    """
    
    # ===================================================================
    # WAVELENGTH STATE (AXIOM 1: Three-Move Invariant)
    # ===================================================================
    wavelength_state: int
    """
    Current state in wavelength FSM (0-7, see WAVELENGTH_STATE enum).
    Pine: var int wavelength_state = 0
    """
    
    moves_completed: int
    """
    Number of moves completed in current wavelength cycle (0-3).
    Pine: var int moves_completed = 0
    """
    
    current_extreme_high: float
    """
    Current high extreme in wavelength tracking.
    Pine: var float current_extreme_high = na
    """
    
    current_extreme_low: float
    """
    Current low extreme in wavelength tracking.
    Pine: var float current_extreme_low = na
    """
    
    # ===================================================================
    # EXHAUSTION / ABSORPTION (AXIOM 3: Absorption Reversal)
    # ===================================================================
    exhaustion_score: float
    """
    Weighted exhaustion score [0.0, 1.0].
    Pine: var float exhaustion_score = 0.0
    """
    
    in_exhaustion_zone: bool
    """
    True if exhaustion_score >= threshold (typically 0.7).
    Pine: var bool in_exhaustion_zone = false
    """
    
    # ===================================================================
    # FUTURES GAP CONTEXT (AXIOM 4: Futures Supremacy)
    # ===================================================================
    active_gap_type: int
    """
    Currently active gap type (0-8, see GAP_TYPE enum).
    Pine: var int active_gap_type = 0
    """
    
    gap_fill_progress: float
    """
    How much of the active gap has been filled [0.0, 1.0].
    Pine: var float gap_fill_progress = 0.0
    """
    
    has_futures_target: bool
    """
    True if futures_target is active (not na).
    REQUIRED: Pine cannot test for na directly in all contexts.
    Pine: var bool has_futures_target = false
    """
    
    futures_target: float
    """
    Projected gravitational target from futures gap analysis.
    math.nan if no active unfilled gaps (mirrors Pine's na).
    Pine: var float futures_target = na
    CRITICAL: Always check has_futures_target before using this value.
    """
    
    # ===================================================================
    # DEBUG / VISUALIZATION (Optional - can be dropped in production Pine)
    # ===================================================================
    debug_flags: int
    """
    Bitfield for debug/visualization flags.
    See DEBUG_FLAGS enum for bit meanings.
    Pine: var int debug_flags = 0
    """
    
    # ===================================================================
    # STRATEGIC CONTEXT (TOP OF DECISION STACK)
    # These fields gate all other signals. If invalid, actionable=False.
    # ===================================================================
    liquidity_direction: int = 0
    """
    Target liquidity direction.
    -1 = SELL_SIDE, +1 = BUY_SIDE, 0 = NONE
    Pine: var int liq_direction = 0
    """
    
    liquidity_level: float = 0.0
    """
    Target liquidity price level.
    Pine: var float liq_level = na
    """
    
    liquidity_valid: bool = False
    """
    Whether liquidity intent is valid/actionable.
    Pine: var bool liq_valid = false
    """
    
    market_control: int = 0
    """
    Market control state (see MARKET_CONTROL enum).
    Pine: var int mkt_control = 0
    """
    
    market_control_conclusive: bool = False
    """
    Whether market control is conclusively determined.
    Pine: var bool mkt_conclusive = false
    """
    
    strategic_alignment: float = 0.0
    """
    Alignment score between liquidity and control [0.0, 1.0].
    Higher = better alignment = higher confidence.
    Pine: var float ctx_alignment = 0.0
    """
    
    strategic_valid: bool = False
    """
    Whether strategic context permits signal generation.
    If False, actionable MUST be False regardless of confluence.
    Pine: var bool ctx_valid = false
    """
    
    def __post_init__(self):
        """
        Validate Pine-compatibility constraints.
        These checks ensure the IR contract is never violated.
        CRITICAL: This prevents silent Pine-breaking drift.
        """
        # Timestamp must be positive unix ms
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be positive unix ms, got {self.timestamp}")
        
        # Bias must be -1, 0, or 1
        if self.bias not in (-1, 0, 1):
            raise ValueError(f"bias must be -1, 0, or 1, got {self.bias}")
        
        # Confidence must be [0.0, 1.0]
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be [0.0, 1.0], got {self.confidence}")
        
        # Participant control must be -1, 0, or 1
        if self.participant_control not in (-1, 0, 1):
            raise ValueError(
                f"participant_control must be -1, 0, or 1, got {self.participant_control}"
            )
        
        # Wavelength state must be valid enum value (0-7)
        if not (0 <= self.wavelength_state <= 7):
            raise ValueError(
                f"wavelength_state must be 0-7, got {self.wavelength_state}"
            )
        
        # Moves completed must be [0, 3]
        if not (0 <= self.moves_completed <= 3):
            raise ValueError(
                f"moves_completed must be 0-3, got {self.moves_completed}"
            )
        
        # Exhaustion score must be [0.0, 1.0]
        if not (0.0 <= self.exhaustion_score <= 1.0):
            raise ValueError(
                f"exhaustion_score must be [0.0, 1.0], got {self.exhaustion_score}"
            )
        
        # Gap fill progress must be [0.0, 1.0]
        if not (0.0 <= self.gap_fill_progress <= 1.0):
            raise ValueError(
                f"gap_fill_progress must be [0.0, 1.0], got {self.gap_fill_progress}"
            )
        
        # Active gap type must be valid enum value (0-8)
        if not (0 <= self.active_gap_type <= 8):
            raise ValueError(
                f"active_gap_type must be 0-8, got {self.active_gap_type}"
            )
        
        # futures_target consistency check
        if self.has_futures_target and math.isnan(self.futures_target):
            raise ValueError("has_futures_target=True but futures_target is nan")
        if not self.has_futures_target and not math.isnan(self.futures_target):
            raise ValueError("has_futures_target=False but futures_target is not nan")
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization/logging.
        Useful for CSV export, database storage, or debugging.
        """
        return {
            "timestamp": self.timestamp,
            "bias": self.bias,
            "actionable": self.actionable,
            "confidence": self.confidence,
            "participant_control": self.participant_control,
            "wavelength_state": self.wavelength_state,
            "moves_completed": self.moves_completed,
            "current_extreme_high": self.current_extreme_high,
            "current_extreme_low": self.current_extreme_low,
            "exhaustion_score": self.exhaustion_score,
            "in_exhaustion_zone": self.in_exhaustion_zone,
            "active_gap_type": self.active_gap_type,
            "gap_fill_progress": self.gap_fill_progress,
            "has_futures_target": self.has_futures_target,
            "futures_target": self.futures_target if self.has_futures_target else None,
            "debug_flags": self.debug_flags,
            # Strategic context
            "liquidity_direction": self.liquidity_direction,
            "liquidity_level": self.liquidity_level,
            "liquidity_valid": self.liquidity_valid,
            "market_control": self.market_control,
            "market_control_conclusive": self.market_control_conclusive,
            "strategic_alignment": self.strategic_alignment,
            "strategic_valid": self.strategic_valid,
        }
    
    def __str__(self) -> str:
        """Human-readable signal summary for logging."""
        from datetime import datetime
        
        bias_str = {-1: "BEARISH", 0: "NEUTRAL", 1: "BULLISH"}[self.bias]
        action_str = "ACTIONABLE" if self.actionable else "NO ACTION"
        dt = datetime.fromtimestamp(self.timestamp / 1000.0)
        
        target_str = f"{self.futures_target:.2f}" if self.has_futures_target else "none"
        liq_dir = {-1: "SELL", 0: "NONE", 1: "BUY"}[self.liquidity_direction]
        ctx_str = "ALIGNED" if self.strategic_valid else "STANDBY"
        
        return (
            f"SignalIR @ {dt.strftime('%Y-%m-%d %H:%M')}\n"
            f"  Strategic: {ctx_str} | Liq: {liq_dir} | Alignment: {self.strategic_alignment:.2f}\n"
            f"  Bias: {bias_str} | {action_str}\n"
            f"  Confidence: {self.confidence:.2f}\n"
            f"  Participant: {self.participant_control:+d} | "
            f"Wavelength: {self.moves_completed}/3 moves | "
            f"Exhaustion: {self.exhaustion_score:.2f} | "
            f"Gap: {self.active_gap_type} Target: {target_str}"
        )
