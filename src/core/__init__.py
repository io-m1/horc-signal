"""
HORC Core Components

Core orchestration and signal generation layer.
Implements Pine-safe intermediate representation (IR) and unified signal orchestration.
"""

from .signal_ir import SignalIR
from .orchestrator import HORCOrchestrator
from .enums import (
    WAVELENGTH_STATE,
    GAP_TYPE,
    BIAS,
    PARTICIPANT_CONTROL,
    DEBUG_FLAGS,
)

__all__ = [
    "SignalIR",
    "HORCOrchestrator",
    "WAVELENGTH_STATE",
    "GAP_TYPE",
    "BIAS",
    "PARTICIPANT_CONTROL",
    "DEBUG_FLAGS",
]
