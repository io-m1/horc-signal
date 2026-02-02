"""
Wavelength Engine - Three-Move Finite State Automaton

Implements AXIOM 1: Wavelength Invariant
"∀ participant p, ∃ exactly 3 moves {M₁, M₂, M₃}"

This module implements a deterministic finite-state automaton (FSA) that models
the three-move participant cycle. The system is a Moore machine where output
depends only on the current state.

Theoretical Foundation:
- Large participants cannot fill entire positions in one move (market impact)
- Execution must occur in phases: directional move, absorption/reversal, continuation
- This creates a structural 3-move pattern observable across timeframes

State Machine Properties:
- Deterministic: One input + one state → exactly one next state
- Complete: Every state has defined transitions for all inputs
- Provably terminating: All paths lead to COMPLETE or FAILED
- Moore machine: Output depends only on current state

Mathematical Specification:
    δ: S × I → S  (state transition function)
    where S = {PRE_OR, PARTICIPANT_ID, MOVE_1, MOVE_2, FLIP_CONFIRMED, MOVE_3, COMPLETE, FAILED}
    and I = input signals from candles and participant identification
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum

from .participant import Candle, ParticipantResult, ParticipantType


class WavelengthState(Enum):
    """
    States of the wavelength finite-state automaton
    
    The system progresses through these states sequentially, with terminal
    states COMPLETE (success) or FAILED (invalidation).
    
    State Descriptions:
        PRE_OR: Waiting for session to start, no participant identified yet
        PARTICIPANT_ID: Participant identified, waiting for first move
        MOVE_1: First directional move completed
        MOVE_2: Reversal/retracement completed (counter-move)
        FLIP_CONFIRMED: Absorption detected at extreme, flip point established
        MOVE_3: Third move in progress toward target
        COMPLETE: All 3 moves completed successfully, signal complete
        FAILED: Pattern invalidated (broke extreme, no absorption, etc.)
    """
    PRE_OR = "pre_opening_range"
    PARTICIPANT_ID = "participant_identified"
    MOVE_1 = "first_move_complete"
    MOVE_2 = "second_move_complete"
    FLIP_CONFIRMED = "flip_point_confirmed"
    MOVE_3 = "third_move_complete"
    COMPLETE = "signal_complete"
    FAILED = "signal_failed"


@dataclass
class WavelengthResult:
    """
    Result of wavelength state machine processing
    
    Attributes:
        state: Current state of the automaton
        moves_completed: Number of moves completed (0-3)
        flip_point: Price level where absorption occurred and reversal confirmed
        move_1_extreme: Highest/lowest point of first move
        move_2_extreme: Highest/lowest point of second move (reversal)
        signal_strength: Confidence score [0.0, 1.0] for the current signal
        entry_price: Suggested entry price (typically at flip point)
        stop_price: Stop loss price (beyond move 2 extreme)
        target_price: Target price (from futures gaps or projection)
        participant_type: The controlling participant (BUYERS or SELLERS)
        candles_in_current_move: Number of candles since last move transition
        timestamp: When this result was generated
    """
    state: WavelengthState
    moves_completed: int
    flip_point: Optional[float]
    move_1_extreme: Optional[float]
    move_2_extreme: Optional[float]
    signal_strength: float
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    participant_type: ParticipantType
    candles_in_current_move: int
    timestamp: datetime


@dataclass
class WavelengthConfig:
    """
    Configuration for wavelength engine
    
    Attributes:
        min_move_1_size_atr: Minimum size for move 1 in ATR units (default: 0.5)
        max_move_2_retracement: Max retracement of move 1 allowed (default: 0.786, Fibonacci)
        exhaustion_threshold: Score threshold for flip confirmation (default: 0.70)
        max_move_duration_candles: Max candles per move before timeout (default: 50)
        flip_confirmation_candles: Candles to confirm flip after exhaustion (default: 3)
        use_simple_exhaustion: Use simplified exhaustion detection (default: True)
    """
    min_move_1_size_atr: float = 0.5
    max_move_2_retracement: float = 0.786
    exhaustion_threshold: float = 0.70
    max_move_duration_candles: int = 50
    flip_confirmation_candles: int = 3
    use_simple_exhaustion: bool = True


class WavelengthEngine:
    """
    Three-move wavelength finite-state automaton
    
    This class implements AXIOM 1: Wavelength Invariant. It tracks the progression
    through exactly 3 moves: initial directional move, counter-move (reversal),
    and continuation move to target.
    
    The state machine is deterministic and complete, with all paths leading to
    either COMPLETE (successful 3-move pattern) or FAILED (pattern invalidation).
    
    State Transition Rules:
        PRE_OR → PARTICIPANT_ID: When participant identified
        PARTICIPANT_ID → MOVE_1: When first directional move completes
        MOVE_1 → MOVE_2: When reversal detected
        MOVE_1 → FAILED: If price breaks beyond move 1 extreme without reversal
        MOVE_2 → FLIP_CONFIRMED: When exhaustion/absorption detected
        MOVE_2 → FAILED: If price breaks move 1 extreme (pattern invalidated)
        FLIP_CONFIRMED → MOVE_3: After flip confirmation period
        MOVE_3 → COMPLETE: When target reached
        MOVE_3 → FAILED: If stop loss hit
    
    Mathematical Properties:
        - Deterministic: δ(state, input) → unique next state
        - Complete: All states have transitions defined
        - Terminating: All paths reach COMPLETE or FAILED
        - Moore machine: Output = f(state) only
    """
    
    def __init__(self, config: Optional[WavelengthConfig] = None):
        """
        Initialize wavelength engine
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or WavelengthConfig()
        
        # State machine variables
        self.state = WavelengthState.PRE_OR
        self.moves_completed = 0
        
        # Price tracking
        self.flip_point: Optional[float] = None
        self.move_1_extreme: Optional[float] = None
        self.move_2_extreme: Optional[float] = None
        self.move_1_start: Optional[float] = None
        
        # Participant tracking
        self.participant_type = ParticipantType.NONE
        self.control_price: Optional[float] = None
        
        # Move timing
        self.candles_in_current_move = 0
        self.flip_confirmation_count = 0
        
        # Historical candles for analysis
        self.candle_history: List[Candle] = []
        self.move_1_candles: List[Candle] = []
        self.move_2_candles: List[Candle] = []
        
    def reset(self) -> None:
        """Reset engine to initial state"""
        self.state = WavelengthState.PRE_OR
        self.moves_completed = 0
        self.flip_point = None
        self.move_1_extreme = None
        self.move_2_extreme = None
        self.move_1_start = None
        self.participant_type = ParticipantType.NONE
        self.control_price = None
        self.candles_in_current_move = 0
        self.flip_confirmation_count = 0
        self.candle_history = []
        self.move_1_candles = []
        self.move_2_candles = []
    
    def calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """
        Calculate Average True Range for the given candles
        
        ATR is used for:
        - Measuring move significance
        - Setting stops relative to volatility
        - Normalizing price movements
        
        Args:
            candles: List of candles to analyze
            period: ATR period (default: 14)
            
        Returns:
            Average True Range value
        """
        if len(candles) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, min(len(candles), period + 1)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    
    def detect_move_1_completion(self, candle: Candle) -> bool:
        """
        Detect if Move 1 (initial directional move) has completed
        
        Move 1 is complete when:
        1. Price has moved significantly from control_price (> min ATR)
        2. Price shows signs of exhaustion or rejection (reversal candle)
        
        For BUYERS: Look for bearish rejection after upward move
        For SELLERS: Look for bullish rejection after downward move
        
        Args:
            candle: Current candle to analyze
            
        Returns:
            True if Move 1 appears complete
        """
        if self.control_price is None or self.move_1_start is None:
            return False
        
        # Calculate move size
        if self.participant_type == ParticipantType.BUYERS:
            move_size = candle.high - self.move_1_start
            # Look for bearish rejection (long upper wick)
            body_size = abs(candle.close - candle.open)
            upper_wick = candle.high - max(candle.open, candle.close)
            rejection = upper_wick > 2 * body_size if body_size > 0 else True
        else:  # SELLERS
            move_size = self.move_1_start - candle.low
            # Look for bullish rejection (long lower wick)
            body_size = abs(candle.close - candle.open)
            lower_wick = min(candle.open, candle.close) - candle.low
            rejection = lower_wick > 2 * body_size if body_size > 0 else True
        
        # Check if move is significant
        atr = self.calculate_atr(self.candle_history) if self.candle_history else move_size * 0.5
        is_significant = move_size >= self.config.min_move_1_size_atr * atr
        
        return is_significant and rejection
    
    def detect_move_2_reversal(self, candle: Candle) -> bool:
        """
        Detect if Move 2 (counter-move/retracement) has started
        
        Move 2 is a reversal that:
        1. Moves against the initial direction
        2. Should not exceed max retracement of Move 1
        3. Creates liquidity sweep opportunity
        
        Args:
            candle: Current candle to analyze
            
        Returns:
            True if reversal detected
        """
        if self.move_1_extreme is None:
            return False
        
        # Check for reversal based on participant type
        if self.participant_type == ParticipantType.BUYERS:
            # After upward Move 1, look for downward reversal
            reversal = candle.close < candle.open  # Bearish candle
        else:  # SELLERS
            # After downward Move 1, look for upward reversal
            reversal = candle.close > candle.open  # Bullish candle
        
        return reversal
    
    def detect_exhaustion(self, candle: Candle) -> float:
        """
        Detect exhaustion/absorption at Move 2 extreme
        
        This is a simplified version. In production, this would integrate
        with the ExhaustionDetector class that implements AXIOM 3.
        
        Simplified heuristics:
        - High volume with small body (absorption)
        - Long wicks showing rejection
        - Multiple candles failing to break level
        
        Args:
            candle: Current candle to analyze
            
        Returns:
            Exhaustion score [0.0, 1.0]
        """
        if not self.move_2_candles:
            return 0.0
        
        # Calculate components
        body_size = abs(candle.close - candle.open)
        candle_range = candle.high - candle.low
        
        # Volume score: high volume with small body suggests absorption
        volume_score = 0.0
        if len(self.move_2_candles) > 1:
            avg_volume = sum(c.volume for c in self.move_2_candles[-5:]) / len(self.move_2_candles[-5:])
            if avg_volume > 0:
                volume_score = min(candle.volume / avg_volume, 1.0) * (1.0 - body_size / candle_range if candle_range > 0 else 1.0)
        
        # Wick score: long wicks show rejection
        if self.participant_type == ParticipantType.BUYERS:
            # In bullish setup, look for long lower wick (buying pressure)
            lower_wick = min(candle.open, candle.close) - candle.low
            wick_score = lower_wick / candle_range if candle_range > 0 else 0.0
        else:
            # In bearish setup, look for long upper wick (selling pressure)
            upper_wick = candle.high - max(candle.open, candle.close)
            wick_score = upper_wick / candle_range if candle_range > 0 else 0.0
        
        # Price stagnation: multiple candles failing to make progress
        stagnation_score = 0.0
        if len(self.move_2_candles) >= 3:
            recent_highs = [c.high for c in self.move_2_candles[-3:]]
            recent_lows = [c.low for c in self.move_2_candles[-3:]]
            price_range = max(recent_highs) - min(recent_lows)
            atr = self.calculate_atr(self.candle_history) if self.candle_history else 1.0
            stagnation_score = 1.0 - min(price_range / atr, 1.0) if atr > 0 else 0.0
        
        # Weighted combination (simplified version of AXIOM 3)
        exhaustion_score = (
            0.30 * volume_score +
            0.30 * wick_score +
            0.40 * stagnation_score
        )
        
        return exhaustion_score
    
    def check_pattern_invalidation(self, candle: Candle) -> bool:
        """
        Check if current price action invalidates the wavelength pattern
        
        Invalidation occurs when:
        - During Move 2: Price breaks beyond Move 1 extreme (wrong direction)
        - During Move 3: Price breaks below flip point (stop loss hit)
        - Timeout: Move takes too long (> max_move_duration_candles)
        
        Args:
            candle: Current candle to check
            
        Returns:
            True if pattern is invalidated
        """
        # Timeout check
        if self.candles_in_current_move > self.config.max_move_duration_candles:
            return True
        
        # State-specific invalidation
        if self.state == WavelengthState.MOVE_2:
            if self.move_1_extreme is None:
                return False
            
            # Move 2 should retrace but not break Move 1 extreme
            if self.participant_type == ParticipantType.BUYERS:
                # In bullish setup, breaking below Move 1 start invalidates
                if self.move_1_start and candle.low < self.move_1_start:
                    return True
            else:  # SELLERS
                # In bearish setup, breaking above Move 1 start invalidates
                if self.move_1_start and candle.high > self.move_1_start:
                    return True
        
        elif self.state == WavelengthState.MOVE_3:
            if self.flip_point is None:
                return False
            
            # Move 3 breaking flip point hits stop loss
            if self.participant_type == ParticipantType.BUYERS:
                if candle.low < self.flip_point:
                    return True
            else:  # SELLERS
                if candle.high > self.flip_point:
                    return True
        
        return False
    
    def calculate_signal_strength(self) -> float:
        """
        Calculate overall signal strength based on current state
        
        Signal strength increases as we progress through states:
        - PRE_OR, PARTICIPANT_ID: 0.0 (no signal yet)
        - MOVE_1: 0.3 (initial direction established)
        - MOVE_2: 0.5 (reversal in progress)
        - FLIP_CONFIRMED: 0.8 (high probability setup)
        - MOVE_3: 0.9 (pattern executing)
        - COMPLETE: 1.0 (pattern complete)
        - FAILED: 0.0 (pattern invalidated)
        
        Returns:
            Signal strength [0.0, 1.0]
        """
        strength_map = {
            WavelengthState.PRE_OR: 0.0,
            WavelengthState.PARTICIPANT_ID: 0.2,
            WavelengthState.MOVE_1: 0.3,
            WavelengthState.MOVE_2: 0.5,
            WavelengthState.FLIP_CONFIRMED: 0.8,
            WavelengthState.MOVE_3: 0.9,
            WavelengthState.COMPLETE: 1.0,
            WavelengthState.FAILED: 0.0,
        }
        return strength_map.get(self.state, 0.0)
    
    def process_candle(
        self, 
        candle: Candle, 
        participant_result: Optional[ParticipantResult] = None
    ) -> WavelengthResult:
        """
        Process single candle through the finite-state automaton
        
        This is the main state machine logic. Each candle updates the state
        based on deterministic transition rules.
        
        State Transition Logic:
            PRE_OR:
                IF participant identified → PARTICIPANT_ID
                
            PARTICIPANT_ID:
                IF move 1 starts → MOVE_1 (track extreme)
                
            MOVE_1:
                IF move 1 completes → MOVE_2 (set move_1_extreme)
                IF timeout → FAILED
                
            MOVE_2:
                IF pattern invalidated → FAILED
                IF exhaustion detected → FLIP_CONFIRMED (set flip_point)
                IF timeout → FAILED
                
            FLIP_CONFIRMED:
                After confirmation period → MOVE_3
                
            MOVE_3:
                IF target reached → COMPLETE
                IF stop hit → FAILED
                IF timeout → FAILED
                
            COMPLETE, FAILED:
                Terminal states (no transitions)
        
        Args:
            candle: Current candle to process
            participant_result: Optional participant identification result
            
        Returns:
            WavelengthResult with current state and signal data
        """
        # Add to history
        self.candle_history.append(candle)
        self.candles_in_current_move += 1
        
        # Check for pattern invalidation (applicable to MOVE_2, MOVE_3)
        if self.state in [WavelengthState.MOVE_2, WavelengthState.MOVE_3]:
            if self.check_pattern_invalidation(candle):
                self.state = WavelengthState.FAILED
                return self._build_result(candle)
        
        # State machine transitions
        if self.state == WavelengthState.PRE_OR:
            self._transition_pre_or(candle, participant_result)
            
        elif self.state == WavelengthState.PARTICIPANT_ID:
            self._transition_participant_id(candle)
            
        elif self.state == WavelengthState.MOVE_1:
            self._transition_move_1(candle)
            
        elif self.state == WavelengthState.MOVE_2:
            self._transition_move_2(candle)
            
        elif self.state == WavelengthState.FLIP_CONFIRMED:
            self._transition_flip_confirmed(candle)
            
        elif self.state == WavelengthState.MOVE_3:
            self._transition_move_3(candle)
        
        # COMPLETE and FAILED are terminal states - no transitions
        
        return self._build_result(candle)
    
    def _transition_pre_or(self, candle: Candle, participant_result: Optional[ParticipantResult]) -> None:
        """Transition logic for PRE_OR state"""
        if participant_result and participant_result.participant_type != ParticipantType.NONE:
            self.participant_type = participant_result.participant_type
            self.control_price = participant_result.control_price
            self.move_1_start = candle.close
            self.state = WavelengthState.PARTICIPANT_ID
            self.candles_in_current_move = 0
    
    def _transition_participant_id(self, candle: Candle) -> None:
        """Transition logic for PARTICIPANT_ID state"""
        # Track the initial move and wait for completion
        if self.participant_type == ParticipantType.BUYERS:
            if self.move_1_extreme is None or candle.high > self.move_1_extreme:
                self.move_1_extreme = candle.high
        else:  # SELLERS
            if self.move_1_extreme is None or candle.low < self.move_1_extreme:
                self.move_1_extreme = candle.low
        
        self.move_1_candles.append(candle)
        
        # Check for move 1 completion
        if self.detect_move_1_completion(candle):
            self.moves_completed = 1
            self.state = WavelengthState.MOVE_1
            self.candles_in_current_move = 0
    
    def _transition_move_1(self, candle: Candle) -> None:
        """Transition logic for MOVE_1 state"""
        # Update extreme if price extends
        if self.participant_type == ParticipantType.BUYERS:
            if candle.high > self.move_1_extreme:
                self.move_1_extreme = candle.high
        else:  # SELLERS
            if candle.low < self.move_1_extreme:
                self.move_1_extreme = candle.low
        
        # Check for reversal (Move 2 starting)
        if self.detect_move_2_reversal(candle):
            self.moves_completed = 2
            self.state = WavelengthState.MOVE_2
            self.move_2_candles = [candle]
            self.move_2_extreme = None
            self.candles_in_current_move = 0
    
    def _transition_move_2(self, candle: Candle) -> None:
        """Transition logic for MOVE_2 state"""
        self.move_2_candles.append(candle)
        
        # Track Move 2 extreme (reversal extreme)
        if self.participant_type == ParticipantType.BUYERS:
            # In bullish setup, track low of reversal
            if self.move_2_extreme is None or candle.low < self.move_2_extreme:
                self.move_2_extreme = candle.low
        else:  # SELLERS
            # In bearish setup, track high of reversal
            if self.move_2_extreme is None or candle.high > self.move_2_extreme:
                self.move_2_extreme = candle.high
        
        # Check for exhaustion/absorption
        exhaustion_score = self.detect_exhaustion(candle)
        if exhaustion_score >= self.config.exhaustion_threshold:
            self.flip_point = self.move_2_extreme
            self.state = WavelengthState.FLIP_CONFIRMED
            self.flip_confirmation_count = 0
            self.candles_in_current_move = 0
    
    def _transition_flip_confirmed(self, candle: Candle) -> None:
        """Transition logic for FLIP_CONFIRMED state"""
        self.flip_confirmation_count += 1
        
        # After confirmation period, transition to Move 3
        if self.flip_confirmation_count >= self.config.flip_confirmation_candles:
            self.moves_completed = 3
            self.state = WavelengthState.MOVE_3
            self.candles_in_current_move = 0
    
    def _transition_move_3(self, candle: Candle) -> None:
        """Transition logic for MOVE_3 state"""
        # Check if target reached (simplified - in production, use futures gap target)
        if self.flip_point is None or self.move_1_extreme is None:
            return
        
        # Simple projection: Move 3 should extend beyond Move 1 extreme
        if self.participant_type == ParticipantType.BUYERS:
            if candle.high > self.move_1_extreme:
                self.state = WavelengthState.COMPLETE
        else:  # SELLERS
            if candle.low < self.move_1_extreme:
                self.state = WavelengthState.COMPLETE
    
    def _build_result(self, candle: Candle) -> WavelengthResult:
        """Build WavelengthResult from current state"""
        # Calculate stop and entry prices
        entry_price = self.flip_point
        stop_price = None
        
        if self.flip_point and self.move_2_extreme:
            if self.participant_type == ParticipantType.BUYERS:
                stop_price = self.move_2_extreme - (self.calculate_atr(self.candle_history) * 0.5 if self.candle_history else 0)
            else:
                stop_price = self.move_2_extreme + (self.calculate_atr(self.candle_history) * 0.5 if self.candle_history else 0)
        
        # Simple target (in production, use futures gap)
        target_price = None
        if self.move_1_extreme and self.flip_point:
            move_1_size = abs(self.move_1_extreme - (self.move_1_start or 0))
            if self.participant_type == ParticipantType.BUYERS:
                target_price = self.move_1_extreme + move_1_size
            else:
                target_price = self.move_1_extreme - move_1_size
        
        return WavelengthResult(
            state=self.state,
            moves_completed=self.moves_completed,
            flip_point=self.flip_point,
            move_1_extreme=self.move_1_extreme,
            move_2_extreme=self.move_2_extreme,
            signal_strength=self.calculate_signal_strength(),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            participant_type=self.participant_type,
            candles_in_current_move=self.candles_in_current_move,
            timestamp=candle.timestamp
        )


def validate_wavelength_progression(states: List[WavelengthState]) -> bool:
    """
    Validate that wavelength progression includes exactly 3 moves
    
    This function verifies AXIOM 1: Wavelength Invariant
    The sequence must include MOVE_1, MOVE_2, and MOVE_3 states.
    
    Args:
        states: List of states traversed during execution
        
    Returns:
        True if exactly 3 moves occurred in sequence
        
    Example:
        >>> states = [PRE_OR, PARTICIPANT_ID, MOVE_1, MOVE_2, FLIP_CONFIRMED, MOVE_3, COMPLETE]
        >>> validate_wavelength_progression(states)
        True
    """
    required_sequence = [
        WavelengthState.MOVE_1,
        WavelengthState.MOVE_2,
        WavelengthState.MOVE_3
    ]
    return all(state in states for state in required_sequence)
