"""
Signal Intermediate Representation (IR)

Pine-compatible signal schema - enforces portability constraints from day one.

DESIGN PRINCIPLES:
- Every field MUST be expressible in Pine Script v5
- Only simple types: int, float, bool, string
- No classes, no dynamic objects, no unbounded memory
- State persists using Pine's var/varip primitives only
- Frozen dataclass = immutable = safe for deterministic replay

PORTABILITY GUARANTEE:
If it compiles to SignalIR, it can be ported to Pine Script.
This is the enforcement firewall for deployment compatibility.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Optional


class WavelengthStateEnum(IntEnum):
    """
    Wavelength state machine states.
    IntEnum enables direct Pine Script translation as integer constants.
    
    Pine equivalent:
        const int WL_INIT = 0
        const int WL_MOVE_1_AGGRESSIVE = 1
        ...
    """
    INIT = 0
    MOVE_1_AGGRESSIVE = 1
    MOVE_1_PASSIVE = 2
    MOVE_2_REVERSAL = 3
    MOVE_3_CONTINUATION = 4
    EXHAUSTION_ABSORPTION = 5
    COMPLETE = 6
    INVALIDATED = 7


class GapTypeEnum(IntEnum):
    """
    Futures gap classification types.
    
    Pine equivalent:
        const int GAP_NONE = 0
        const int GAP_COMMON_UP = 1
        ...
    """
    NONE = 0
    COMMON_UP = 1
    COMMON_DOWN = 2
    BREAKAWAY_UP = 3
    BREAKAWAY_DOWN = 4
    RUNAWAY_UP = 5
    RUNAWAY_DOWN = 6
    EXHAUSTION_UP = 7
    EXHAUSTION_DOWN = 8


@dataclass(frozen=True)
class SignalIR:
    """
    Pine-compatible intermediate representation for HORC signals.
    
    CONTRACT GUARANTEES:
    1. All fields are primitive types (int, float, bool) or datetime
    2. No nested objects or dynamic collections
    3. All state fits in Pine's var/varip persistence model
    4. Deterministic: same input → same output, always
    5. Bar-local: no hidden future context
    
    FIELD CATEGORIES:
    - Core Signal: bias, actionable, confidence
    - Participant Control: who's in charge (Axiom 2)
    - Wavelength: 3-move cycle state (Axiom 1)
    - Exhaustion: absorption reversal (Axiom 3)
    - Futures Gap: gravitational targeting (Axiom 4)
    - Debug: optional flags for visualization
    
    Pine Translation Strategy:
        Python SignalIR → var-persisted primitives in Pine
        Each field becomes a var float/int/bool
        Orchestration logic ports to Pine functions
        No object creation on each bar (Pine doesn't have classes)
    """
    
    # ===================================================================
    # METADATA
    # ===================================================================
    timestamp: datetime
    """Bar close time - Pine: time(timeframe.period)"""
    
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
    This is the final gate - "should we act on this signal?"
    
    Pine: var bool actionable = false
    """
    
    confidence: float
    """
    Final weighted confluence score [0.0, 1.0].
    Higher = more engines agree + stronger signals.
    
    Pine: var float confidence = 0.0
    Must satisfy: 0.0 <= confidence <= 1.0
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
    Current state in wavelength FSM (0-7, see WavelengthStateEnum).
    
    Pine: var int wavelength_state = 0
    """
    
    moves_completed: int
    """
    Number of moves completed in current wavelength cycle (0-3).
    
    Pine: var int moves_completed = 0
    Must satisfy: 0 <= moves_completed <= 3
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
    Combines: volume absorption, body rejection, price stagnation, reversal patterns.
    
    Pine: var float exhaustion_score = 0.0
    Must satisfy: 0.0 <= exhaustion_score <= 1.0
    """
    
    in_exhaustion_zone: bool
    """
    True if exhaustion_score >= threshold (typically 0.7).
    Binary flag for easy regime filtering.
    
    Pine: var bool in_exhaustion_zone = false
    """
    
    # ===================================================================
    # FUTURES GAP CONTEXT (AXIOM 4: Futures Supremacy)
    # ===================================================================
    active_gap_type: int
    """
    Currently active gap type (0-8, see GapTypeEnum).
    0 = no gap, 1-8 = specific gap classifications.
    
    Pine: var int active_gap_type = 0
    """
    
    gap_fill_progress: float
    """
    How much of the active gap has been filled [0.0, 1.0].
    0.0 = unfilled, 1.0 = completely filled.
    
    Pine: var float gap_fill_progress = 0.0
    Must satisfy: 0.0 <= gap_fill_progress <= 1.0
    """
    
    futures_target: Optional[float]
    """
    Projected gravitational target from futures gap analysis.
    None if no active unfilled gaps.
    
    Pine: var float futures_target = na
    Uses na (not available) for None equivalent.
    """
    
    # ===================================================================
    # DEBUG / VISUALIZATION (Optional - can be dropped in production Pine)
    # ===================================================================
    debug_flags: int
    """
    Bitfield for debug/visualization flags.
    Bit 0 (0x01): Participant sweep detected
    Bit 1 (0x02): Wavelength reset occurred
    Bit 2 (0x04): Exhaustion zone entry
    Bit 3 (0x08): Gap fill completed
    Bit 4 (0x10): Confluence threshold crossed
    
    Pine: var int debug_flags = 0
    Use bitwise operations: bitwise_and(), bitwise_or()
    """
    
    def __post_init__(self):
        """
        Validate Pine-compatibility constraints.
        These checks ensure the IR contract is never violated.
        """
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
        
        # Wavelength state must be valid enum value
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
        
        # Active gap type must be valid enum value
        if not (0 <= self.active_gap_type <= 8):
            raise ValueError(
                f"active_gap_type must be 0-8, got {self.active_gap_type}"
            )
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization/logging.
        Useful for CSV export, database storage, or debugging.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
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
            "futures_target": self.futures_target,
            "debug_flags": self.debug_flags,
        }
    
    def __str__(self) -> str:
        """Human-readable signal summary for logging."""
        bias_str = {-1: "BEARISH", 0: "NEUTRAL", 1: "BULLISH"}[self.bias]
        action_str = "🟢 ACTIONABLE" if self.actionable else "⚪ NO ACTION"
        
        return (
            f"SignalIR @ {self.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"  Bias: {bias_str} | {action_str}\n"
            f"  Confidence: {self.confidence:.2f}\n"
            f"  Participant: {self.participant_control:+d} | "
            f"Wavelength: {self.moves_completed}/3 moves | "
            f"Exhaustion: {self.exhaustion_score:.2f} | "
            f"Gap: {self.active_gap_type}"
        )


# ===================================================================
# PINE TRANSLATION REFERENCE
# ===================================================================
"""
Pine Script v5 Translation Template:

//@version=5
indicator("HORC Signal IR", overlay=true)

// === Signal IR State Variables (var = persisted across bars) ===
var int bias = 0
var bool actionable = false
var float confidence = 0.0

var int participant_control = 0

var int wavelength_state = 0
var int moves_completed = 0
var float current_extreme_high = na
var float current_extreme_low = na

var float exhaustion_score = 0.0
var bool in_exhaustion_zone = false

var int active_gap_type = 0
var float gap_fill_progress = 0.0
var float futures_target = na

var int debug_flags = 0

// === Orchestration Logic (per bar) ===
process_bar() =>
    // Run all engine logic (participant, wavelength, exhaustion, gaps)
    // ... (engine functions omitted for brevity)
    
    // Compute confluence
    conf = participant_strength * 0.30 + 
           wavelength_progress * 0.25 + 
           exhaustion_score * 0.25 + 
           gap_pull * 0.20
    
    // Determine bias (majority vote)
    bias_votes = array.new_int(0)
    array.push(bias_votes, participant_control)
    array.push(bias_votes, wavelength_direction)
    array.push(bias_votes, gap_direction)
    
    bullish_votes = array.sum(array.new_int().from(array.filter(bias_votes, x => x > 0)))
    bearish_votes = array.sum(array.new_int().from(array.filter(bias_votes, x => x < 0)))
    
    signal_bias = bullish_votes >= 2 ? 1 : bearish_votes >= 2 ? -1 : 0
    
    // Gate actionable
    signal_actionable = conf >= 0.75 and signal_bias != 0
    
    // Update state
    bias := signal_bias
    actionable := signal_actionable
    confidence := conf
    
    [bias, actionable, confidence]

// === Run on each bar ===
[sig_bias, sig_action, sig_conf] = process_bar()

// === Visualization ===
bgcolor(sig_action ? (sig_bias > 0 ? color.new(color.green, 90) : color.new(color.red, 90)) : na)
plotchar(sig_action, "Signal", "▲", location.belowbar, sig_bias > 0 ? color.green : color.red)
"""
