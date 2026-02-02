"""
HORC Signal Orchestrator

Unified signal generation layer - the confluence engine.
Integrates all four HORC axioms into actionable signals with Pine-safe output.

ARCHITECTURE:
    Raw Data → 4 Engines → Orchestrator → Signal IR → (Backtest/Pine)
                                ↑
                        Confluence + Gating

DESIGN PRINCIPLES:
1. Pine-safe state: Only primitives, no dynamic objects
2. Deterministic: Same input → same output, always (MANDATORY)
3. Bar-local: No hidden future context
4. Aggressive gating: High confluence threshold → fewer, better signals
5. Regime-aware: Optional filtering by market conditions

DETERMINISM RULE (NON-NEGOTIABLE):
    Given the same bar sequence → identical IR sequence
    - No randomness
    - No clock access (use bar timestamps only)
    - No external state leaks
    
    This enables: replay validation, Pine parity, walk-forward trust

CONFLUENCE SCORING:
    confidence = w₁·participant + w₂·wavelength + w₃·exhaustion + w₄·gap
    
    Default weights:
        - Participant control: 30% (who's in charge matters most)
        - Wavelength progress: 25% (structural positioning)
        - Exhaustion absorption: 25% (reversal probability)
        - Futures gap pull: 20% (gravitational targeting)

BIAS DETERMINATION:
    Requires multi-engine agreement (majority vote):
        - Participant: -1 (sellers) / +1 (buyers)
        - Wavelength: directional signal from state
        - Gap context: implied direction from unfilled gaps
    
    Signal is actionable ONLY if:
        - Confluence >= threshold (default 0.75)
        - Bias != 0 (clear directional agreement)
        - Regime filter passes (optional)

PINE TRANSLATION STRATEGY:
    This orchestrator is designed to be 1:1 portable to Pine Script:
        - All state is var-persisted primitives
        - No classes or objects (just functions)
        - Engines become Pine functions
        - IR fields become var floats/ints/bools
        - Confluence logic ports directly
"""

from dataclasses import dataclass
from typing import Optional, List
import math

from ..engines import (
    ParticipantIdentifier,
    ParticipantResult,
    ParticipantType,
    WavelengthEngine,
    WavelengthResult,
    WavelengthState,
    WavelengthConfig,
    ExhaustionDetector,
    ExhaustionResult,
    ExhaustionConfig,
    FuturesGapEngine,
    GapAnalysisResult,
    GapConfig,
    Candle,
)
from .signal_ir import SignalIR
from .enums import WAVELENGTH_STATE, GAP_TYPE, BIAS, PARTICIPANT_CONTROL


@dataclass
class OrchestratorConfig:
    """
    Orchestrator configuration - controls confluence and gating.
    
    Attributes:
        confluence_threshold: Minimum confidence for actionable signals [0.0, 1.0]
        participant_weight: Weight for participant control (default 0.30)
        wavelength_weight: Weight for wavelength progress (default 0.25)
        exhaustion_weight: Weight for exhaustion score (default 0.25)
        gap_weight: Weight for gap gravitational pull (default 0.20)
        require_agreement: Require majority vote for bias (default True)
        regime_filter_enabled: Enable regime-based gating (default False, Phase 2)
        min_wavelength_moves: Minimum moves for wavelength contribution (default 1)
    """
    confluence_threshold: float = 0.75
    participant_weight: float = 0.30
    wavelength_weight: float = 0.25
    exhaustion_weight: float = 0.25
    gap_weight: float = 0.20
    require_agreement: bool = True
    regime_filter_enabled: bool = False
    min_wavelength_moves: int = 1
    
    def __post_init__(self):
        """Validate configuration"""
        # Threshold must be [0.0, 1.0]
        if not (0.0 <= self.confluence_threshold <= 1.0):
            raise ValueError(
                f"confluence_threshold must be [0.0, 1.0], got {self.confluence_threshold}"
            )
        
        # Weights must be positive and sum to 1.0
        total_weight = (
            self.participant_weight + 
            self.wavelength_weight + 
            self.exhaustion_weight + 
            self.gap_weight
        )
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight:.3f}"
            )
        
        if any(w < 0 for w in [
            self.participant_weight,
            self.wavelength_weight,
            self.exhaustion_weight,
            self.gap_weight
        ]):
            raise ValueError("All weights must be non-negative")
        
        # Min moves must be [0, 3]
        if not (0 <= self.min_wavelength_moves <= 3):
            raise ValueError(
                f"min_wavelength_moves must be 0-3, got {self.min_wavelength_moves}"
            )


class HORCOrchestrator:
    """
    Unified signal orchestrator - the confluence engine.
    
    Integrates all four HORC axioms into actionable Pine-safe signals:
        - AXIOM 1: Wavelength Invariant (3-move cycle)
        - AXIOM 2: First Move Determinism (participant identification)
        - AXIOM 3: Absorption Reversal (exhaustion detection)
        - AXIOM 4: Futures Supremacy (gap targeting)
    
    Usage Pattern:
        orchestrator = HORCOrchestrator(
            participant_identifier,
            wavelength_engine,
            exhaustion_detector,
            gap_engine,
            OrchestratorConfig(confluence_threshold=0.75)
        )
        
        # Bar-by-bar processing
        for candle in candles:
            signal = orchestrator.process_bar(candle, futures_candle)
            
            if signal.actionable:
                print(f"Signal: {signal.bias} @ {signal.confidence:.2f}")
    
    Pine Translation:
        This class becomes a collection of Pine functions:
            - process_bar() → main orchestration logic
            - _calculate_confluence() → weighted sum
            - _determine_bias() → majority vote
            - State persists in var primitives (see signal_ir.py Pine template)
    """
    
    def __init__(
        self,
        participant: ParticipantIdentifier,
        wavelength: WavelengthEngine,
        exhaustion: ExhaustionDetector,
        gap_engine: FuturesGapEngine,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize orchestrator with engine instances.
        
        Args:
            participant: Participant identification engine
            wavelength: Wavelength state machine engine
            exhaustion: Exhaustion detection engine
            gap_engine: Futures gap analysis engine
            config: Orchestrator configuration (uses defaults if None)
        """
        self.participant = participant
        self.wavelength = wavelength
        self.exhaustion = exhaustion
        self.gap_engine = gap_engine
        self.config = config or OrchestratorConfig()
        
        # Persistent state (Pine-compatible - only primitives)
        self.prev_signal: Optional[SignalIR] = None
        self.bars_processed: int = 0
    
    def process_bar(
        self,
        candle: Candle,
        futures_candle: Optional[Candle] = None,
        participant_candles: Optional[List[Candle]] = None,
    ) -> SignalIR:
        """
        Process single bar through full orchestration pipeline.
        
        This is the main entry point - processes one bar and emits Signal IR.
        
        Pipeline:
            1. Run all four engines
            2. Calculate confluence score (weighted sum)
            3. Determine bias (majority vote)
            4. Gate actionability (confluence + bias + regime)
            5. Emit Pine-safe Signal IR
        
        Args:
            candle: Current OHLCV candle
            futures_candle: Futures candle for gap analysis (optional)
            participant_candles: Candles for participant identification (optional)
        
        Returns:
            SignalIR with complete signal state
        
        Example:
            signal = orchestrator.process_bar(
                candle=Candle(...),
                futures_candle=Candle(...)
            )
            
            if signal.actionable:
                if signal.bias > 0:
                    print(f"BUY signal @ {signal.confidence:.2f}")
                else:
                    print(f"SELL signal @ {signal.confidence:.2f}")
        """
        self.bars_processed += 1
        
        # ===================================================================
        # STEP 1: Run All Engines
        # ===================================================================
        
        # Participant identification
        if participant_candles:
            participant_res = self.participant.identify(participant_candles)
        else:
            # Use previous session if not provided
            participant_res = ParticipantResult(
                participant_type=ParticipantType.NONE,
                conviction_level=False,
                control_price=None,
                timestamp=candle.timestamp,
                orh_prev=0.0,
                orl_prev=0.0,
                sweep_candle_index=None,
            )
        
        # Wavelength state machine
        wavelength_res = self.wavelength.process_candle(candle, participant_res)
        
        # Exhaustion detection (use last N candles if available)
        exhaustion_res = self.exhaustion.detect_exhaustion(
            candles=[candle],  # Single candle analysis
            volume_data=None,
            direction="LONG" if wavelength_res.state in [
                WavelengthState.MOVE_1,
                WavelengthState.MOVE_3,
            ] else "SHORT"
        )
        
        # Futures gap analysis
        if futures_candle:
            gaps = self.gap_engine.detect_gaps([futures_candle])
            gap_res = self.gap_engine.analyze_gaps(
                gaps,
                current_price=candle.close,
                current_date=candle.timestamp
            )
        else:
            # No futures data - neutral gap contribution
            from ..engines.gaps import Gap
            gap_res = GapAnalysisResult(
                target_price=None,
                nearest_gap=None,
                total_gaps=0,
                unfilled_gaps=0,
                fill_probability=0.0,
                gravitational_pull=0.0,
                details="No futures data provided"
            )
        
        # ===================================================================
        # STEP 2: Calculate Confluence Score
        # ===================================================================
        
        confluence = self._calculate_confluence(
            participant_res,
            wavelength_res,
            exhaustion_res,
            gap_res
        )
        
        # ===================================================================
        # STEP 3: Determine Bias (Majority Vote)
        # ===================================================================
        
        bias = self._determine_bias(
            participant_res,
            wavelength_res,
            gap_res
        )
        
        # ===================================================================
        # STEP 4: Gate Actionability
        # ===================================================================
        
        actionable = self._is_actionable(confluence, bias)
        
        # ===================================================================
        # STEP 5: Emit Pine-Safe Signal IR
        # ===================================================================
        
        # Convert timestamp to unix ms (Pine-safe)
        timestamp_ms = int(candle.timestamp.timestamp() * 1000)
        
        # Handle futures_target Pine na pattern
        has_target = gap_res.target_price is not None
        target_value = gap_res.target_price if has_target else math.nan
        
        signal = SignalIR(
            timestamp=timestamp_ms,
            bias=bias,
            actionable=actionable,
            confidence=confluence,
            
            # Participant
            participant_control=self._participant_to_control(participant_res),
            
            # Wavelength
            wavelength_state=self._wavelength_state_to_int(wavelength_res.state),
            moves_completed=wavelength_res.moves_completed,
            current_extreme_high=wavelength_res.move_1_extreme or candle.high,
            current_extreme_low=wavelength_res.move_2_extreme or candle.low,
            
            # Exhaustion
            exhaustion_score=exhaustion_res.score,
            in_exhaustion_zone=exhaustion_res.threshold_met,
            
            # Gap
            active_gap_type=self._gap_to_type(gap_res),
            gap_fill_progress=self._calculate_gap_fill_progress(gap_res),
            has_futures_target=has_target,
            futures_target=target_value,
            
            # Debug flags
            debug_flags=self._compute_debug_flags(
                participant_res,
                wavelength_res,
                exhaustion_res,
                gap_res
            ),
        )
        
        # Validate IR before returning (catch Pine-breaking drift)
        self._validate_ir(signal)
        
        self.prev_signal = signal
        return signal
    
    def _calculate_confluence(
        self,
        participant: ParticipantResult,
        wavelength: WavelengthResult,
        exhaustion: ExhaustionResult,
        gap: GapAnalysisResult,
    ) -> float:
        """
        Calculate weighted confluence score [0.0, 1.0].
        
        Formula:
            confidence = w₁·P + w₂·W + w₃·E + w₄·G
        
        Where:
            P = participant strength [0.0, 1.0]
            W = wavelength progress [0.0, 1.0]
            E = exhaustion score [0.0, 1.0]
            G = gap gravitational pull [0.0, 1.0]
        
        Pine Translation:
            float conf = 
                participant_strength * 0.30 + 
                wavelength_progress * 0.25 + 
                exhaustion_score * 0.25 + 
                gap_pull * 0.20
        """
        # Participant strength: 1.0 if identified with conviction, 0.5 without, 0.0 if none
        if participant.participant_type == ParticipantType.NONE:
            participant_strength = 0.0
        elif participant.conviction_level:
            participant_strength = 1.0
        else:
            participant_strength = 0.5
        
        # Wavelength progress: signal_strength from engine, adjusted by moves completed
        wavelength_progress = wavelength.signal_strength
        if wavelength.moves_completed < self.config.min_wavelength_moves:
            wavelength_progress *= 0.5  # Penalize early signals
        
        # Exhaustion score: direct from engine
        exhaustion_strength = exhaustion.score
        
        # Gap gravitational pull: direct from engine
        gap_strength = gap.gravitational_pull
        
        # Weighted sum
        confluence = (
            participant_strength * self.config.participant_weight +
            wavelength_progress * self.config.wavelength_weight +
            exhaustion_strength * self.config.exhaustion_weight +
            gap_strength * self.config.gap_weight
        )
        
        return max(0.0, min(1.0, confluence))
    
    def _determine_bias(
        self,
        participant: ParticipantResult,
        wavelength: WavelengthResult,
        gap: GapAnalysisResult,
    ) -> int:
        """
        Determine overall directional bias via majority vote.
        
        Returns:
            -1 = bearish, 0 = neutral, +1 = bullish
        
        Logic:
            Each engine "votes" -1, 0, or +1
            Bias is determined by majority (>=2 engines agree)
            If no majority, bias = 0 (neutral)
        
        Pine Translation:
            int bias_votes[3]
            bias_votes[0] = participant_control
            bias_votes[1] = wavelength_direction
            bias_votes[2] = gap_direction
            
            int bullish = 0
            int bearish = 0
            for i = 0 to 2
                if bias_votes[i] > 0
                    bullish := bullish + 1
                if bias_votes[i] < 0
                    bearish := bearish + 1
            
            int signal_bias = bullish >= 2 ? 1 : bearish >= 2 ? -1 : 0
        """
        # Collect votes
        votes = []
        
        # Participant vote
        if participant.participant_type == ParticipantType.BUYERS:
            votes.append(1)
        elif participant.participant_type == ParticipantType.SELLERS:
            votes.append(-1)
        else:
            votes.append(0)
        
        # Wavelength vote (based on state)
        if wavelength.state in [
            WavelengthState.MOVE_1,
            WavelengthState.MOVE_3,
        ]:
            # Directional state - infer from participant or extremes
            if wavelength.move_1_extreme and wavelength.move_2_extreme:
                if wavelength.move_1_extreme > wavelength.move_2_extreme:
                    votes.append(1)  # Bullish structure
                else:
                    votes.append(-1)  # Bearish structure
            else:
                votes.append(0)
        elif wavelength.state == WavelengthState.MOVE_2:
            # Reversal - opposite of move 1
            if wavelength.move_1_extreme and wavelength.move_2_extreme:
                if wavelength.move_1_extreme > wavelength.move_2_extreme:
                    votes.append(-1)  # Reversing down
                else:
                    votes.append(1)  # Reversing up
            else:
                votes.append(0)
        else:
            votes.append(0)
        
        # Gap vote (based on target direction)
        if gap.target_price and gap.nearest_gap:
            # Assuming current price is stored or can be inferred
            # For now, use gravitational pull sign
            if gap.gravitational_pull > 0.5:
                # Strong pull - infer direction from gap type
                if gap.nearest_gap.gap_type.name.endswith("_UP"):
                    votes.append(1)
                elif gap.nearest_gap.gap_type.name.endswith("_DOWN"):
                    votes.append(-1)
                else:
                    votes.append(0)
            else:
                votes.append(0)
        else:
            votes.append(0)
        
        # Count votes (improved clarity for Pine translation)
        pos = sum(1 for v in votes if v > 0)
        neg = sum(1 for v in votes if v < 0)
        
        # Determine bias (require majority if configured)
        if self.config.require_agreement:
            if pos >= 2:
                return 1
            elif neg >= 2:
                return -1
            else:
                return 0
        else:
            # Simple sum (not recommended)
            total = sum(votes)
            if total > 0:
                return 1
            elif total < 0:
                return -1
            else:
                return 0
    
    def _is_actionable(self, confluence: float, bias: int) -> bool:
        """
        Determine if signal is actionable.
        
        Requirements:
            1. Confluence >= threshold
            2. Bias != 0 (clear direction)
            3. Regime filter passes (if enabled)
        
        Pine Translation:
            bool actionable = conf >= 0.75 and signal_bias != 0
            if regime_filter_enabled
                actionable := actionable and regime_ok()
        """
        # Base requirements
        if confluence < self.config.confluence_threshold:
            return False
        
        if bias == 0:
            return False
        
        # Regime filter (Phase 2 - placeholder)
        if self.config.regime_filter_enabled:
            regime_ok = self._check_regime()
            if not regime_ok:
                return False
        
        return True
    
    def _check_regime(self) -> bool:
        """
        Placeholder for regime-based filtering.
        
        Phase 2 Implementation:
            - Trend detection (moving averages, ADX)
            - Volatility regime (ATR, Bollinger Bands)
            - Volume regime (relative volume)
            - Time-of-day filtering (avoid low liquidity periods)
        
        Pine Translation:
            bool regime_ok()
                // Placeholder - always true for Phase 1
                true
        """
        # Phase 2: Add regime detection logic
        return True
    
    def _participant_to_control(self, participant: ParticipantResult) -> int:
        """Convert participant type to control signal (-1, 0, +1)"""
        if participant.participant_type == ParticipantType.BUYERS:
            return 1
        elif participant.participant_type == ParticipantType.SELLERS:
            return -1
        else:
            return 0
    
    def _wavelength_state_to_int(self, state: WavelengthState) -> int:
        """
        Map WavelengthState enum to integer for Pine compatibility.
        
        Mapping:
            PRE_OR -> 0 (INIT)
            PARTICIPANT_ID -> 1
            MOVE_1 -> 2
            MOVE_2 -> 3
            FLIP_CONFIRMED -> 4
            MOVE_3 -> 5
            COMPLETE -> 6
            FAILED -> 7 (INVALIDATED)
        """
        mapping = {
            WavelengthState.PRE_OR: 0,
            WavelengthState.PARTICIPANT_ID: 1,
            WavelengthState.MOVE_1: 2,
            WavelengthState.MOVE_2: 3,
            WavelengthState.FLIP_CONFIRMED: 4,
            WavelengthState.MOVE_3: 5,
            WavelengthState.COMPLETE: 6,
            WavelengthState.FAILED: 7,
        }
        return mapping.get(state, 0)
    
    def _gap_to_type(self, gap: GapAnalysisResult) -> int:
        """Convert gap result to type enum integer"""
        if gap.nearest_gap:
            return gap.nearest_gap.gap_type.value
        return GAP_TYPE["NONE"]
    
    def _validate_ir(self, ir: SignalIR):
        """
        Validate Signal IR constraints at runtime.
        
        This prevents silent Pine-breaking drift by catching violations
        before they propagate to backtesting or deployment.
        
        CRITICAL: Call this before returning IR from process_bar.
        
        Checked constraints:
            - bias in [-1, 0, 1]
            - confidence in [0.0, 1.0]
            - wavelength_state in [0, 7]
            - gap_fill_progress in [0.0, 1.0]
            - exhaustion_score in [0.0, 1.0]
            - futures_target consistency with has_futures_target
        
        Raises:
            AssertionError if any constraint violated
        """
        assert -1 <= ir.bias <= 1, f"bias out of range: {ir.bias}"
        assert 0.0 <= ir.confidence <= 1.0, f"confidence out of range: {ir.confidence}"
        assert 0 <= ir.wavelength_state <= 7, f"wavelength_state out of range: {ir.wavelength_state}"
        assert 0.0 <= ir.gap_fill_progress <= 1.0, f"gap_fill_progress out of range: {ir.gap_fill_progress}"
        assert 0.0 <= ir.exhaustion_score <= 1.0, f"exhaustion_score out of range: {ir.exhaustion_score}"
        assert 0 <= ir.moves_completed <= 3, f"moves_completed out of range: {ir.moves_completed}"
        
        # futures_target consistency
        if ir.has_futures_target:
            assert not math.isnan(ir.futures_target), "has_futures_target=True but target is nan"
        else:
            assert math.isnan(ir.futures_target), "has_futures_target=False but target is not nan"
    
    def _calculate_gap_fill_progress(self, gap: GapAnalysisResult) -> float:
        """
        Calculate gap fill progress [0.0, 1.0].
        
        If gap is filled, progress = 1.0
        If partially filled, progress = (filled_size / total_size)
        If unfilled, progress = 0.0
        """
        if not gap.nearest_gap:
            return 0.0
        
        # Use fill probability as proxy for progress
        # (Actual fill tracking would require price history)
        return gap.fill_probability
    
    def _compute_debug_flags(
        self,
        participant: ParticipantResult,
        wavelength: WavelengthResult,
        exhaustion: ExhaustionResult,
        gap: GapAnalysisResult,
    ) -> int:
        """
        Compute debug flags as bitfield.
        
        Bits:
            0 (0x01): Participant sweep detected
            1 (0x02): Wavelength reset/state change
            2 (0x04): Exhaustion zone entry
            3 (0x08): Gap fill completed
            4 (0x10): Confluence threshold crossed
        
        Pine Translation:
            int flags = 0
            if participant_sweep
                flags := bitwise_or(flags, 0x01)
            if wavelength_reset
                flags := bitwise_or(flags, 0x02)
            // ... etc
        """
        flags = 0
        
        # Bit 0: Participant sweep
        if participant.conviction_level:
            flags |= 0x01
        
        # Bit 1: Wavelength state transition
        if self.prev_signal and self.prev_signal.wavelength_state != wavelength.state.value:
            flags |= 0x02
        
        # Bit 2: Exhaustion zone
        if exhaustion.threshold_met:
            flags |= 0x04
        
        # Bit 3: Gap fill completed
        if gap.nearest_gap and gap.fill_probability >= 0.95:
            flags |= 0x08
        
        # Bit 4: Confluence threshold crossed
        # (Would need previous confluence to detect crossing)
        
        return flags
    
    def reset(self):
        """
        Reset orchestrator state.
        
        Useful for:
            - Starting new trading session
            - Switching instruments
            - Reprocessing historical data
        
        Pine Translation:
            reset_state() =>
                bias := 0
                actionable := false
                confidence := 0.0
                // ... reset all var state
        """
        self.prev_signal = None
        self.bars_processed = 0
        # Note: Individual engines maintain their own state
        # Call engine.reset() if they implement it
