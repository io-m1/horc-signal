"""
HORC Core Components

Core orchestration and signal generation layer.
Implements Pine-safe intermediate representation (IR) and unified signal orchestration.

DECISION HIERARCHY (from edge philosophy):
    ParticipantEngine → Opposition → Quadrant → Imbalance/Liquidity → LiquidityChain → StrategicContext → Engines → SignalIR
    
PHASE 1 (PARTICIPANT ENGINE):
    "Determine WHO is in control and LOCK that state.
     Scan divisible TFs HIGH → LOW for opposition.
     Lock on first opposition. Gap overrides if no opposition.
     A TF cannot register itself."
    
THE CORE INVARIANT (OPPOSITION RULE):
    "A signal is only true when a new period first opens in opposition
     to the previous period's close — on a single, consistent logic.
     Anything else is inconclusive noise."

THE AUTHORITY LAYER (QUADRANT RULE):
    "Opposition Rule decides eligibility.
     Quadrant Rule decides authority.
     Highest conclusive timeframe owns truth.
     Lower timeframes only express it."

THE MECHANICAL LAYER (IMBALANCE-LIQUIDITY):
    "Liquidity invalidates imbalance by default —
     except when it is defending trend, or when it created the zone.
     Trapped liquidity only works when nested inside a higher-range zone."
    
DOCTRINE (LIQUIDITY CHAIN):
    "Liquidity is a hierarchy, not a location.
     Continuation is chosen by the counterparty.
     The first valid liquidity in a chain controls all others."
"""

from .signal_ir import SignalIR
from .orchestrator import HORCOrchestrator
from .enums import (
    WAVELENGTH_STATE,
    GAP_TYPE,
    BIAS,
    PARTICIPANT_CONTROL,
    DEBUG_FLAGS,
    TIMEFRAME,
    TIMEFRAME_RANK,
    LIQUIDITY_DIRECTION,
    MARKET_CONTROL,
)
from .strategic_context import (
    LiquidityIntent,
    MarketControlState,
    StrategicContext,
    resolve_market_control_from_timeframes,
)
from .participant_engine import (
    ParticipantType,
    ConfidenceState,
    ParentPeriod,
    ParticipantState,
    GapInfo,
    ParticipantEngine,
    get_divisible_tfs,
    check_opposition_on_tf,
    detect_gap,
    gap_implied_participant,
    validate_tf_eligibility,
    DIVISIBLE_TFS,
    PARENT_TF_MAP,
)
from .flip_engine import (
    FlipState,
    TimeframeType,
    FlipPoint,
    FlipValidationResult,
    FlipEngine,
    get_next_open_time,
    detect_opposition as flip_detect_opposition,
)
from .charge_engine import (
    Charge,
    ChargedLevel,
    ChargeState,
    ChargeEngine,
    ChargeValidator,
    build_multi_tf_charge_label,
    compare_charges,
)
from .coordinate_engine import (
    Coordinate,
    CoordinateEngine,
    HVOValidator,
    CoordinateComparator,
    format_coordinate_comparison,
    build_coordinate_from_participant_states,
)
from .liquidity_chain import (
    LiquidityNode,
    LiquidityChain,
    LiquidityType,
    LiquiditySide,
    TrappedState,
    build_chain_from_swings,
    continuation_permitted,
)
from .opposition import (
    SignalState,
    LogicType,
    PeriodType,
    PeriodSignal,
    AggressorState,
    OppositionChain,
    validate_opposition,
    resolve_aggressor,
    compute_signal_from_crl,
    compute_signal_from_opl,
    is_new_period,
    get_period_start,
)
from .quadrant import (
    SignalRole,
    ParticipantScope,
    TimeframeSignal,
    QuadrantResult,
    MultiScopeResult,
    ImbalanceZone,
    resolve_quadrant,
    resolve_multi_scope,
    extract_imbalance_zones,
    is_tf_eligible,
    get_preferred_logic,
    MAX_TF_BY_SCOPE,
    LOGIC_BY_SCOPE,
)
from .imbalance_liquidity import (
    ImbalanceState,
    ImbalanceType,
    Tier,
    Imbalance,
    LiquidityLevel,
    TrapSetup,
    get_tier,
    validate_same_tier,
    has_liquidity_cut,
    is_defense_liquidity,
    is_creator_liquidity,
    validate_imbalance,
    validate_trap_setup,
    validate_all_imbalances,
    get_valid_imbalances,
    get_invalid_imbalances,
)

__all__ = [
    # Signal IR
    "SignalIR",
    # Orchestrator
    "HORCOrchestrator",
    # Participant Engine (PHASE 1 - THE CORE)
    "ParticipantType",
    "ConfidenceState",
    "ParentPeriod",
    "ParticipantState",
    "GapInfo",
    "ParticipantEngine",
    "get_divisible_tfs",
    "check_opposition_on_tf",
    "detect_gap",
    "gap_implied_participant",
    "validate_tf_eligibility",
    "DIVISIBLE_TFS",
    "PARENT_TF_MAP",
    # Flip Engine (PHASE 1.5 - TEMPORAL FINALITY)
    "FlipState",
    "TimeframeType",
    "FlipPoint",
    "FlipValidationResult",
    "FlipEngine",
    "get_next_open_time",
    "flip_detect_opposition",
    # Charge Engine (PHASE 1.5 - +/− LABELING)
    "Charge",
    "ChargedLevel",
    "ChargeState",
    "ChargeEngine",
    "ChargeValidator",
    "build_multi_tf_charge_label",
    "compare_charges",
    # Coordinate Engine (PHASE 1.5 - MULTI-TF STATE VECTORS)
    "Coordinate",
    "CoordinateEngine",
    "HVOValidator",
    "CoordinateComparator",
    "format_coordinate_comparison",
    "build_coordinate_from_participant_states",
    # Liquidity Chain (THE HIERARCHY MODEL)
    "LiquidityNode",
    "LiquidityChain",
    "LiquidityType",
    "LiquiditySide",
    "TrappedState",
    "build_chain_from_swings",
    "continuation_permitted",
    # Strategic Context
    "LiquidityIntent",
    "MarketControlState",
    "StrategicContext",
    "resolve_market_control_from_timeframes",
    # Opposition Rule (THE CORE INVARIANT)
    "SignalState",
    "LogicType",
    "PeriodType",
    "PeriodSignal",
    "AggressorState",
    "OppositionChain",
    "validate_opposition",
    "resolve_aggressor",
    "compute_signal_from_crl",
    "compute_signal_from_opl",
    "is_new_period",
    "get_period_start",
    # Quadrant Rule (THE AUTHORITY LAYER)
    "SignalRole",
    "ParticipantScope",
    "TimeframeSignal",
    "QuadrantResult",
    "MultiScopeResult",
    "ImbalanceZone",
    "resolve_quadrant",
    "resolve_multi_scope",
    "extract_imbalance_zones",
    "is_tf_eligible",
    "get_preferred_logic",
    "MAX_TF_BY_SCOPE",
    "LOGIC_BY_SCOPE",
    # Imbalance-Liquidity (THE MECHANICAL LAYER)
    "ImbalanceState",
    "ImbalanceType",
    "Tier",
    "Imbalance",
    "LiquidityLevel",
    "TrapSetup",
    "get_tier",
    "validate_same_tier",
    "has_liquidity_cut",
    "is_defense_liquidity",
    "is_creator_liquidity",
    "validate_imbalance",
    "validate_trap_setup",
    "validate_all_imbalances",
    "get_valid_imbalances",
    "get_invalid_imbalances",
    # Enums
    "WAVELENGTH_STATE",
    "GAP_TYPE",
    "BIAS",
    "PARTICIPANT_CONTROL",
    "DEBUG_FLAGS",
    "TIMEFRAME",
    "TIMEFRAME_RANK",
    "LIQUIDITY_DIRECTION",
    "MARKET_CONTROL",
]
