"""
HORC Engine Components

Core engines implementing the four axioms of HORC:
1. Wavelength Invariant (3-move cycle)
2. First Move Determinism (participant identification)
3. Absorption Reversal (exhaustion detection)
4. Futures Supremacy (gap targeting)
"""

from .participant import (
    ParticipantIdentifier,
    ParticipantType,
    ParticipantResult,
    Candle,
)
from .wavelength import (
    WavelengthEngine,
    WavelengthState,
    WavelengthResult,
    WavelengthConfig,
    validate_wavelength_progression,
)
from .exhaustion import (
    ExhaustionDetector,
    ExhaustionConfig,
    ExhaustionResult,
    VolumeBar,
)
from .gaps import (
    FuturesGapEngine,
    Gap,
    GapType,
    GapConfig,
    GapAnalysisResult,
)

__all__ = [
    "ParticipantIdentifier",
    "ParticipantType",
    "ParticipantResult",
    "Candle",
    "WavelengthEngine",
    "WavelengthState",
    "WavelengthResult",
    "WavelengthConfig",
    "validate_wavelength_progression",
    "ExhaustionDetector",
    "ExhaustionConfig",
    "ExhaustionResult",
    "VolumeBar",
    "FuturesGapEngine",
    "Gap",
    "GapType",
    "GapConfig",
    "GapAnalysisResult",
]
