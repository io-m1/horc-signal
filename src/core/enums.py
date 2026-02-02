"""
Pine-Safe Enum Contracts

CRITICAL: These numeric mappings are LOCKED FOREVER.
Once deployed to Pine Script, changing these numbers breaks deployed indicators.

RULE: Never renumber. Never reorder. Only append.

Pine Translation:
    These constants map 1:1 to Pine Script const int declarations.
    
    Pine example:
        const int WL_INIT = 0
        const int WL_MOVE_1_AGGRESSIVE = 1
        ...
"""

# ==============================================================================
# WAVELENGTH STATE MACHINE
# ==============================================================================

WAVELENGTH_STATE = {
    "INIT": 0,                      # PRE_OR in engine
    "PARTICIPANT_ID": 1,            # Participant identified
    "MOVE_1": 2,                    # First move complete
    "MOVE_2": 3,                    # Second move (reversal)
    "FLIP_CONFIRMED": 4,            # Absorption detected
    "MOVE_3": 5,                    # Third move to target
    "COMPLETE": 6,                  # Signal complete
    "INVALIDATED": 7,               # Pattern failed
}

# Reverse mapping for display
WAVELENGTH_STATE_NAMES = {v: k for k, v in WAVELENGTH_STATE.items()}


# ==============================================================================
# GAP TYPE CLASSIFICATION
# ==============================================================================

GAP_TYPE = {
    "NONE": 0,
    "COMMON_UP": 1,
    "COMMON_DOWN": 2,
    "BREAKAWAY_UP": 3,
    "BREAKAWAY_DOWN": 4,
    "RUNAWAY_UP": 5,
    "RUNAWAY_DOWN": 6,
    "EXHAUSTION_UP": 7,
    "EXHAUSTION_DOWN": 8,
}

# Reverse mapping for display
GAP_TYPE_NAMES = {v: k for k, v in GAP_TYPE.items()}


# ==============================================================================
# BIAS SIGNALS
# ==============================================================================

BIAS = {
    "BEARISH": -1,
    "NEUTRAL": 0,
    "BULLISH": 1,
}


# ==============================================================================
# PARTICIPANT CONTROL
# ==============================================================================

PARTICIPANT_CONTROL = {
    "SELLERS": -1,
    "NEUTRAL": 0,
    "BUYERS": 1,
}


# ==============================================================================
# DEBUG FLAGS (BITFIELD)
# ==============================================================================

DEBUG_FLAGS = {
    "PARTICIPANT_SWEEP": 0x01,      # Bit 0: Sweep detected
    "WAVELENGTH_TRANSITION": 0x02,  # Bit 1: State change
    "EXHAUSTION_ENTRY": 0x04,       # Bit 2: Entered exhaustion zone
    "GAP_FILL_COMPLETE": 0x08,      # Bit 3: Gap filled
    "CONFLUENCE_THRESHOLD": 0x10,   # Bit 4: Crossed threshold
}
