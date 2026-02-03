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
from .enums import WAVELENGTH_STATE, GAP_TYPE, BIAS, PARTICIPANT_CONTROL, LIQUIDITY_DIRECTION, MARKET_CONTROL
from .strategic_context import LiquidityIntent, MarketControlState, StrategicContext
from .opposition import (
    SignalState,
    LogicType,
    PeriodType,
    PeriodSignal,
    AggressorState,
    validate_opposition,
    resolve_aggressor,
    compute_signal_from_crl,
)
from .quadrant import (
    SignalRole,
    ParticipantScope,
    TimeframeSignal,
    QuadrantResult,
    resolve_quadrant,
    is_tf_eligible,
    get_preferred_logic,
)
from ..logging.trade_logger import trade_logger

@dataclass
class OrchestratorConfig:
    confluence_threshold: float = 0.30
    participant_weight: float = 0.50
    wavelength_weight: float = 0.20
    exhaustion_weight: float = 0.20
    gap_weight: float = 0.10
    require_agreement: bool = False
    regime_filter_enabled: bool = False
    min_wavelength_moves: int = 1
    require_strategic_context: bool = False
    
    def __post_init__(self):
        if not (0.0 <= self.confluence_threshold <= 1.0):
            raise ValueError(
                f"confluence_threshold must be [0.0, 1.0], got {self.confluence_threshold}"
            )
        
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
        
        if not (0 <= self.min_wavelength_moves <= 3):
            raise ValueError(
                f"min_wavelength_moves must be 0-3, got {self.min_wavelength_moves}"
            )

class HORCOrchestrator:
    def __init__(
        self,
        participant: ParticipantIdentifier,
        wavelength: WavelengthEngine,
        exhaustion: ExhaustionDetector,
        gap_engine: FuturesGapEngine,
        config: Optional[OrchestratorConfig] = None,
    ):
        self.participant = participant
        self.wavelength = wavelength
        self.exhaustion = exhaustion
        self.gap_engine = gap_engine
        self.config = config or OrchestratorConfig()
        
        self.prev_signal: Optional[SignalIR] = None
        self.bars_processed: int = 0
        
        self.strategic_context: StrategicContext = StrategicContext.null()
        
        self.aggressor_state: AggressorState = AggressorState()
        self.prev_period_signal: Optional[PeriodSignal] = None
        self.logic_type: LogicType = LogicType.CRL  # Default to CRL (cleanest)
        
        self.tf_signals: List[TimeframeSignal] = []
        self.quadrant_result: Optional[QuadrantResult] = None
        self.participant_scope: ParticipantScope = ParticipantScope.DAILY
    
    def set_strategic_context(
        self,
        liquidity: LiquidityIntent,
        control: MarketControlState,
    ) -> StrategicContext:
        self.strategic_context = StrategicContext.resolve(liquidity, control)
        return self.strategic_context
    
    def update_opposition(
        self,
        period_type: PeriodType,
        current_open: float,
        prev_close_high: float,
        prev_close_low: float,
        prev_close_signal: SignalState,
        timestamp: int,
    ) -> AggressorState:
        if self.aggressor_state.conclusive:
            return self.aggressor_state
        
        new_open_signal = compute_signal_from_crl(
            current_open=current_open,
            prev_close_high=prev_close_high,
            prev_close_low=prev_close_low,
        )
        
        new_period = PeriodSignal(
            period=period_type,
            logic=self.logic_type,
            open_signal=new_open_signal,
            close_signal=SignalState.INCONCLUSIVE,  # Unknown yet
            reference_high=prev_close_high,
            reference_low=prev_close_low,
            timestamp=timestamp,
        )
        
        prev_period = PeriodSignal(
            period=period_type,
            logic=self.logic_type,
            open_signal=SignalState.INCONCLUSIVE,  # Not relevant
            close_signal=prev_close_signal,
            timestamp=timestamp - 1,  # Placeholder
        )
        
        self.aggressor_state = resolve_aggressor(
            prev_period=prev_period,
            new_period=new_period,
            current_aggressor=self.aggressor_state,
        )
        
        self.prev_period_signal = new_period
        return self.aggressor_state

    def register_tf_signal(
        self,
        tf: str,
        conclusive: bool,
        direction: int,
        liquidity_high: float = 0.0,
        liquidity_low: float = 0.0,
    ) -> TimeframeSignal:
        if not is_tf_eligible(tf, self.participant_scope):
            raise ValueError(
                f"TF {tf} not eligible for scope {self.participant_scope.name}. "
                f"Max allowed: {self.participant_scope}"
            )
        
        signal = TimeframeSignal(
            tf=tf,
            conclusive=conclusive,
            direction=direction,
            logic_type=get_preferred_logic(self.participant_scope),
            liquidity_high=liquidity_high,
            liquidity_low=liquidity_low,
        )
        
        existing = [s for s in self.tf_signals if s.tf == tf]
        if existing:
            self.tf_signals.remove(existing[0])
        
        self.tf_signals.append(signal)
        return signal

    def resolve_quadrant_authority(self) -> QuadrantResult:
        self.quadrant_result = resolve_quadrant(self.tf_signals)
        return self.quadrant_result

    def get_authority_direction(self) -> int:
        if self.quadrant_result and self.quadrant_result.hct:
            return self.quadrant_result.liquidity_direction
        return 0

    def is_signal_aligned(self, signal_direction: int) -> bool:
        authority = self.get_authority_direction()
        if authority == 0:
            return False  # No authority resolved
        return signal_direction == authority

    def process_bar(
        self,
        candle: Candle,
        futures_candle: Optional[Candle] = None,
        participant_candles: Optional[List[Candle]] = None,
    ) -> SignalIR:
        self.bars_processed += 1
        
        if participant_candles:
            participant_res = self.participant.identify(participant_candles)
        else:
            participant_res = ParticipantResult(
                participant_type=ParticipantType.NONE,
                conviction_level=False,
                control_price=None,
                timestamp=candle.timestamp,
                orh_prev=0.0,
                orl_prev=0.0,
                sweep_candle_index=None,
            )
        
        wavelength_res = self.wavelength.process_candle(candle, participant_res)
        
        exhaustion_res = self.exhaustion.detect_exhaustion(
            candles=[candle],  # Single candle analysis
            volume_data=None,
            direction="LONG" if wavelength_res.state in [
                WavelengthState.MOVE_1,
                WavelengthState.MOVE_3,
            ] else "SHORT"
        )
        
        if futures_candle:
            gaps = self.gap_engine.detect_gaps([futures_candle])
            gap_res = self.gap_engine.analyze_gaps(
                gaps,
                current_price=candle.close,
                current_date=candle.timestamp
            )
        else:
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
        
        confluence = self._calculate_confluence(
            participant_res,
            wavelength_res,
            exhaustion_res,
            gap_res
        )
        
        bias = self._determine_bias(
            participant_res,
            wavelength_res,
            gap_res
        )
        
        strategic_valid = self.strategic_context.valid if self.config.require_strategic_context else True
        
        actionable = self._is_actionable(confluence, bias) and strategic_valid
        
        timestamp_ms = int(candle.timestamp.timestamp() * 1000)
        
        has_target = gap_res.target_price is not None
        target_value = gap_res.target_price if has_target else math.nan
        
        signal = SignalIR(
            timestamp=timestamp_ms,
            bias=bias,
            actionable=actionable,
            confidence=confluence,
            
            participant_control=self._participant_to_control(participant_res),
            
            wavelength_state=self._wavelength_state_to_int(wavelength_res.state),
            moves_completed=wavelength_res.moves_completed,
            current_extreme_high=wavelength_res.move_1_extreme or candle.high,
            current_extreme_low=wavelength_res.move_2_extreme or candle.low,
            
            exhaustion_score=exhaustion_res.score,
            in_exhaustion_zone=exhaustion_res.threshold_met,
            
            active_gap_type=self._gap_to_type(gap_res),
            gap_fill_progress=self._calculate_gap_fill_progress(gap_res),
            has_futures_target=has_target,
            futures_target=target_value,
            
            debug_flags=self._compute_debug_flags(
                participant_res,
                wavelength_res,
                exhaustion_res,
                gap_res
            ),
            
            liquidity_direction=self.strategic_context.liquidity.direction,
            liquidity_level=self.strategic_context.liquidity.level,
            liquidity_valid=self.strategic_context.liquidity.valid,
            market_control=self.strategic_context.control.control,
            market_control_conclusive=self.strategic_context.control.conclusive,
            strategic_alignment=self.strategic_context.alignment_score,
            strategic_valid=self.strategic_context.valid,
        )
        
        self._validate_ir(signal)
        
        self.prev_signal = signal
        # Emit an optional trade-log row for auditing/validation
        try:
            if trade_logger and getattr(trade_logger, "enable", False):
                try:
                    trade_logger.log(signal, candle, bars_processed=self.bars_processed)
                except Exception:
                    # Do not let logging interfere with core logic
                    pass
        except Exception:
            pass

        return signal
    
    def _calculate_confluence(
        self,
        participant: ParticipantResult,
        wavelength: WavelengthResult,
        exhaustion: ExhaustionResult,
        gap: GapAnalysisResult,
    ) -> float:
        if participant.participant_type == ParticipantType.NONE:
            participant_strength = 0.0
        elif participant.conviction_level:
            participant_strength = 1.0
        else:
            participant_strength = 0.5
        
        wavelength_progress = wavelength.signal_strength
        if wavelength.moves_completed < self.config.min_wavelength_moves:
            wavelength_progress *= 0.5  # Penalize early signals
        
        exhaustion_strength = exhaustion.score
        
        gap_strength = gap.gravitational_pull
        
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
        votes = []
        
        if participant.participant_type == ParticipantType.BUYERS:
            votes.append(1)
        elif participant.participant_type == ParticipantType.SELLERS:
            votes.append(-1)
        else:
            votes.append(0)
        
        if wavelength.state in [
            WavelengthState.MOVE_1,
            WavelengthState.MOVE_3,
        ]:
            if wavelength.move_1_extreme and wavelength.move_2_extreme:
                if wavelength.move_1_extreme > wavelength.move_2_extreme:
                    votes.append(1)  # Bullish structure
                else:
                    votes.append(-1)  # Bearish structure
            else:
                votes.append(0)
        elif wavelength.state == WavelengthState.MOVE_2:
            if wavelength.move_1_extreme and wavelength.move_2_extreme:
                if wavelength.move_1_extreme > wavelength.move_2_extreme:
                    votes.append(-1)  # Reversing down
                else:
                    votes.append(1)  # Reversing up
            else:
                votes.append(0)
        else:
            votes.append(0)
        
        if gap.target_price and gap.nearest_gap:
            if gap.gravitational_pull > 0.5:
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
        
        pos = sum(1 for v in votes if v > 0)
        neg = sum(1 for v in votes if v < 0)
        
        if self.config.require_agreement:
            if pos >= 2:
                return 1
            elif neg >= 2:
                return -1
            else:
                return 0
        else:
            total = sum(votes)
            if total > 0:
                return 1
            elif total < 0:
                return -1
            else:
                return 0
    
    def _is_actionable(self, confluence: float, bias: int) -> bool:
        if confluence < self.config.confluence_threshold:
            return False
        
        if bias == 0:
            return False
        
        if self.config.regime_filter_enabled:
            regime_ok = self._check_regime()
            if not regime_ok:
                return False
        
        return True
    
    def _check_regime(self) -> bool:
        return True
    
    def _participant_to_control(self, participant: ParticipantResult) -> int:
        if participant.participant_type == ParticipantType.BUYERS:
            return 1
        elif participant.participant_type == ParticipantType.SELLERS:
            return -1
        else:
            return 0
    
    def _wavelength_state_to_int(self, state: WavelengthState) -> int:
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
        if gap.nearest_gap:
            return gap.nearest_gap.gap_type.value
        return GAP_TYPE["NONE"]
    
    def _validate_ir(self, ir: SignalIR):
        assert -1 <= ir.bias <= 1, f"bias out of range: {ir.bias}"
        assert 0.0 <= ir.confidence <= 1.0, f"confidence out of range: {ir.confidence}"
        assert 0 <= ir.wavelength_state <= 7, f"wavelength_state out of range: {ir.wavelength_state}"
        assert 0.0 <= ir.gap_fill_progress <= 1.0, f"gap_fill_progress out of range: {ir.gap_fill_progress}"
        assert 0.0 <= ir.exhaustion_score <= 1.0, f"exhaustion_score out of range: {ir.exhaustion_score}"
        assert 0 <= ir.moves_completed <= 3, f"moves_completed out of range: {ir.moves_completed}"
        
        if ir.has_futures_target:
            assert not math.isnan(ir.futures_target), "has_futures_target=True but target is nan"
        else:
            assert math.isnan(ir.futures_target), "has_futures_target=False but target is not nan"
    
    def _calculate_gap_fill_progress(self, gap: GapAnalysisResult) -> float:
        if not gap.nearest_gap:
            return 0.0
        
        return gap.fill_probability
    
    def _compute_debug_flags(
        self,
        participant: ParticipantResult,
        wavelength: WavelengthResult,
        exhaustion: ExhaustionResult,
        gap: GapAnalysisResult,
    ) -> int:
        flags = 0
        
        if participant.conviction_level:
            flags |= 0x01
        
        if self.prev_signal and self.prev_signal.wavelength_state != wavelength.state.value:
            flags |= 0x02
        
        if exhaustion.threshold_met:
            flags |= 0x04
        
        if gap.nearest_gap and gap.fill_probability >= 0.95:
            flags |= 0x08
        
        return flags
    
    def reset(self):
        self.prev_signal = None
        self.bars_processed = 0
