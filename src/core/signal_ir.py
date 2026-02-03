from dataclasses import dataclass
import math

from .enums import WAVELENGTH_STATE, GAP_TYPE, BIAS, PARTICIPANT_CONTROL, DEBUG_FLAGS, LIQUIDITY_DIRECTION, MARKET_CONTROL

@dataclass(frozen=True)
class SignalIR:
    timestamp: int
    bias: int
    actionable: bool
    confidence: float
    participant_control: int
    wavelength_state: int
    moves_completed: int
    current_extreme_high: float
    current_extreme_low: float
    exhaustion_score: float
    in_exhaustion_zone: bool
    active_gap_type: int
    gap_fill_progress: float
    has_futures_target: bool
    futures_target: float
    debug_flags: int
    liquidity_direction: int = 0
    liquidity_level: float = 0.0
    liquidity_valid: bool = False
    market_control: int = 0
    market_control_conclusive: bool = False
    strategic_alignment: float = 0.0
    strategic_valid: bool = False
    def __post_init__(self):
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be positive unix ms, got {self.timestamp}")
        
        if self.bias not in (-1, 0, 1):
            raise ValueError(f"bias must be -1, 0, or 1, got {self.bias}")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be [0.0, 1.0], got {self.confidence}")
        
        if self.participant_control not in (-1, 0, 1):
            raise ValueError(
                f"participant_control must be -1, 0, or 1, got {self.participant_control}"
            )
        
        if not (0 <= self.wavelength_state <= 7):
            raise ValueError(
                f"wavelength_state must be 0-7, got {self.wavelength_state}"
            )
        
        if not (0 <= self.moves_completed <= 3):
            raise ValueError(
                f"moves_completed must be 0-3, got {self.moves_completed}"
            )
        
        if not (0.0 <= self.exhaustion_score <= 1.0):
            raise ValueError(
                f"exhaustion_score must be [0.0, 1.0], got {self.exhaustion_score}"
            )
        
        if not (0.0 <= self.gap_fill_progress <= 1.0):
            raise ValueError(
                f"gap_fill_progress must be [0.0, 1.0], got {self.gap_fill_progress}"
            )
        
        if not (0 <= self.active_gap_type <= 8):
            raise ValueError(
                f"active_gap_type must be 0-8, got {self.active_gap_type}"
            )
        
        if self.has_futures_target and math.isnan(self.futures_target):
            raise ValueError("has_futures_target=True but futures_target is nan")
        if not self.has_futures_target and not math.isnan(self.futures_target):
            raise ValueError("has_futures_target=False but futures_target is not nan")
    
    def to_dict(self) -> dict:
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
            "liquidity_direction": self.liquidity_direction,
            "liquidity_level": self.liquidity_level,
            "liquidity_valid": self.liquidity_valid,
            "market_control": self.market_control,
            "market_control_conclusive": self.market_control_conclusive,
            "strategic_alignment": self.strategic_alignment,
            "strategic_valid": self.strategic_valid,
        }
    
    def __str__(self) -> str:
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
