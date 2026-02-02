"""
HORC Core Components

Core orchestration and signal generation layer.
Implements Pine-safe intermediate representation (IR) and unified signal orchestration.
"""

from .signal_ir import SignalIR
from .orchestrator import HORCOrchestrator

__all__ = [
    "SignalIR",
    "HORCOrchestrator",
]
