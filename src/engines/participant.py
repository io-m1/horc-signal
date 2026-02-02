"""
Participant Identification Engine

Implements AXIOM 2: First Move Determinism
"Control(session) = f(First_Sweep_Direction)"

This module identifies which participant class (buyers or sellers) controls a session
based on the first aggressive move that sweeps prior liquidity levels (ORH/ORL).

Theoretical Foundation:
- Kyle (1985): Informed traders act before market moves
- First aggressive sweep reveals informed side
- No probabilistic component - pure decision logic

Mathematical Specification:
    identify_participant: Candles × ℝ × ℝ → {BUYERS, SELLERS, NONE} × 𝔹
    
    where:
        - First parameter: sequence of candles
        - Second parameter: ORH_prev (previous opening range high)
        - Third parameter: ORL_prev (previous opening range low)
        - Returns: (participant_type, conviction_confirmed)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ParticipantType(Enum):
    """Enumeration of participant classifications"""
    BUYERS = "BUYERS"
    SELLERS = "SELLERS"
    NONE = "NONE"


@dataclass
class Candle:
    """
    OHLCV candle data structure
    
    Attributes:
        timestamp: Candle open time
        open: Opening price
        high: Highest price during period
        low: Lowest price during period
        close: Closing price
        volume: Trading volume
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __post_init__(self):
        """Validate candle data integrity"""
        if self.high < max(self.open, self.close):
            raise ValueError(f"High {self.high} cannot be less than max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError(f"Low {self.low} cannot be greater than min(open, close)")
        if self.volume < 0:
            raise ValueError(f"Volume {self.volume} cannot be negative")


@dataclass
class ParticipantResult:
    """
    Result of participant identification analysis
    
    Attributes:
        participant_type: Classified participant (BUYERS, SELLERS, or NONE)
        conviction_level: True if first move swept prior liquidity with conviction
        control_price: The specific price level that was swept (ORH or ORL)
        timestamp: When the identification occurred
        orh_prev: Previous session's opening range high (for reference)
        orl_prev: Previous session's opening range low (for reference)
        sweep_candle_index: Index of candle that performed the sweep (0-2, or None)
    """
    participant_type: ParticipantType
    conviction_level: bool
    control_price: Optional[float]
    timestamp: datetime
    orh_prev: float
    orl_prev: float
    sweep_candle_index: Optional[int]


class ParticipantIdentifier:
    """
    Identifies controlling participant based on first move sweep dynamics
    
    This class implements AXIOM 2: First Move Determinism
    
    The logic is deterministic and binary:
    1. Examine first 1-3 candles of current session
    2. If any candle sweeps ORL_prev → SELLERS control (swept buy-side liquidity)
    3. If any candle sweeps ORH_prev → BUYERS control (swept sell-side liquidity)
    4. If no sweep detected → NONE (wait for conviction)
    
    Theoretical Justification:
    - Informed traders with private information act decisively at session open
    - First sweep reveals which side has structural edge
    - No ambiguity: Either liquidity is swept or it isn't
    
    Mathematical Properties:
    - Deterministic: Same input always produces same output
    - Binary: Exactly one of {BUYERS, SELLERS, NONE} is returned
    - Order-sensitive: Processes candles sequentially (first sweep wins)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize participant identifier
        
        Args:
            config: Configuration dictionary with optional parameters:
                - or_lookback_sessions: Number of prior sessions for OR calculation (default: 1)
                - min_conviction_threshold: Minimum conviction level 0-1 (default: 0.8)
                - max_first_move_candles: Maximum candles to analyze for first move (default: 3)
        """
        if config is None:
            config = {}
            
        self.or_lookback_sessions: int = config.get('or_lookback_sessions', 1)
        self.min_conviction_threshold: float = config.get('min_conviction_threshold', 0.8)
        self.max_first_move_candles: int = config.get('max_first_move_candles', 3)
        
        # Storage for previous session data
        self.prev_session_candles: List[Candle] = []
        
    def get_opening_range(self, candles: List[Candle]) -> Tuple[float, float]:
        """
        Calculate Opening Range High (ORH) and Opening Range Low (ORL) from candles
        
        The opening range is typically the first 30-60 minutes of the trading session,
        but this implementation takes the high/low of all provided candles to allow
        flexibility in OR definition (could be first candle, first hour, etc.)
        
        Mathematical Definition:
            ORH = max(candles.high)
            ORL = min(candles.low)
        
        Args:
            candles: List of candles from previous session(s)
            
        Returns:
            Tuple of (ORH, ORL) as (float, float)
            
        Raises:
            ValueError: If candles list is empty
            
        Example:
            >>> candles = [Candle(..., high=4500, low=4480, ...),
            ...            Candle(..., high=4510, low=4485, ...)]
            >>> orh, orl = identifier.get_opening_range(candles)
            >>> # orh = 4510, orl = 4480
        """
        if not candles:
            raise ValueError("Cannot calculate opening range from empty candle list")
        
        orh = max(candle.high for candle in candles)
        orl = min(candle.low for candle in candles)
        
        return orh, orl
    
    def identify_participant(
        self, 
        candles: List[Candle], 
        orh_prev: float, 
        orl_prev: float
    ) -> Tuple[ParticipantType, bool, Optional[int]]:
        """
        Core first move detection logic (AXIOM 2 implementation)
        
        Examines first 1-3 candles to determine which participant swept prior liquidity.
        This is the mathematical heart of participant identification.
        
        Decision Logic:
            FOR each candle in first_moves[0:3]:
                IF candle.low <= ORL_prev:
                    RETURN (SELLERS, True, candle_index)  # Swept buy-side
                ELIF candle.high >= ORH_prev:
                    RETURN (BUYERS, True, candle_index)   # Swept sell-side
            
            IF no sweep detected:
                RETURN (NONE, False, None)
        
        Edge Cases Handled:
        1. Empty candle list → (NONE, False, None)
        2. Both levels swept in sequence → First sweep wins (order matters)
        3. Price exactly at level → Counts as sweep (<=, >= not <, >)
        4. No sweep detected → (NONE, False, None)
        
        Args:
            candles: Current session candles (typically real-time data)
            orh_prev: Previous session's opening range high
            orl_prev: Previous session's opening range low
            
        Returns:
            Tuple of (participant_type, conviction_confirmed, sweep_candle_index)
            
        Mathematical Properties:
            - Monotonic: Once identified, participant doesn't change
            - Deterministic: Same input → same output
            - Order-sensitive: First sweep determines outcome
        """
        if not candles:
            return ParticipantType.NONE, False, None
        
        # Analyze first 1-3 candles only (informed traders act first)
        first_moves = candles[:self.max_first_move_candles]
        
        for idx, candle in enumerate(first_moves):
            # Check for sweep of ORL_prev (sellers sweeping buy-side liquidity)
            if candle.low <= orl_prev:
                return ParticipantType.SELLERS, True, idx
            
            # Check for sweep of ORH_prev (buyers sweeping sell-side liquidity)
            if candle.high >= orh_prev:
                return ParticipantType.BUYERS, True, idx
        
        # No decisive first move detected - remain neutral
        return ParticipantType.NONE, False, None
    
    def identify(self, current_candles: List[Candle]) -> ParticipantResult:
        """
        Main identification logic - full participant identification pipeline
        
        This is the primary public interface for participant identification.
        It orchestrates:
        1. Opening range calculation from previous session
        2. First move detection in current session
        3. Result packaging with full context
        
        Usage Pattern:
            identifier = ParticipantIdentifier(config)
            identifier.prev_session_candles = [...]  # Set previous session data
            result = identifier.identify(current_candles)
            
            if result.participant_type == ParticipantType.BUYERS:
                # Buyers control - expect bullish follow-through
                ...
        
        Args:
            current_candles: Candles from current session (real-time or historical)
            
        Returns:
            ParticipantResult with complete identification data
            
        Raises:
            ValueError: If prev_session_candles not set or empty
            
        Example:
            >>> identifier = ParticipantIdentifier()
            >>> identifier.prev_session_candles = yesterday_candles
            >>> result = identifier.identify(today_candles)
            >>> print(f"Participant: {result.participant_type}")
            >>> print(f"Conviction: {result.conviction_level}")
            >>> print(f"Control Price: {result.control_price}")
        """
        # Validate preconditions
        if not self.prev_session_candles:
            raise ValueError(
                "Previous session candles not set. "
                "Must set identifier.prev_session_candles before calling identify()"
            )
        
        if not current_candles:
            # No current data - return neutral result
            orh_prev, orl_prev = self.get_opening_range(self.prev_session_candles)
            return ParticipantResult(
                participant_type=ParticipantType.NONE,
                conviction_level=False,
                control_price=None,
                timestamp=datetime.now(),
                orh_prev=orh_prev,
                orl_prev=orl_prev,
                sweep_candle_index=None
            )
        
        # Step 1: Calculate opening range from previous session
        orh_prev, orl_prev = self.get_opening_range(self.prev_session_candles)
        
        # Step 2: Identify participant via first move detection
        participant_type, conviction, sweep_idx = self.identify_participant(
            current_candles, orh_prev, orl_prev
        )
        
        # Step 3: Determine control price (the level that was swept)
        control_price: Optional[float] = None
        if participant_type == ParticipantType.BUYERS:
            control_price = orh_prev  # Buyers swept this level
        elif participant_type == ParticipantType.SELLERS:
            control_price = orl_prev  # Sellers swept this level
        
        # Step 4: Package complete result
        return ParticipantResult(
            participant_type=participant_type,
            conviction_level=conviction,
            control_price=control_price,
            timestamp=current_candles[0].timestamp,
            orh_prev=orh_prev,
            orl_prev=orl_prev,
            sweep_candle_index=sweep_idx
        )
    
    def update_session_data(self, new_session_candles: List[Candle]) -> None:
        """
        Update the previous session candles for next identification cycle
        
        This method should be called at the end of each session to update
        the reference data for the next session's participant identification.
        
        Args:
            new_session_candles: Candles from the session that just completed
            
        Example:
            >>> # End of trading session
            >>> identifier.update_session_data(today_candles)
            >>> # Now today's data becomes the reference for tomorrow
        """
        self.prev_session_candles = new_session_candles.copy()
    
    def reset(self) -> None:
        """
        Reset identifier state (clear all session data)
        
        Useful for:
        - Starting fresh backtests
        - Error recovery
        - Testing
        """
        self.prev_session_candles = []


# Utility functions for testing and validation

def create_test_candles_sweep_high(orh: float) -> List[Candle]:
    """
    Create test candles that sweep a given ORH level
    
    Args:
        orh: The opening range high to sweep
        
    Returns:
        List of candles with first candle sweeping the ORH
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    
    return [
        Candle(
            timestamp=base_time,
            open=orh - 5.0,
            high=orh + 2.0,  # Sweeps ORH
            low=orh - 6.0,
            close=orh - 3.0,
            volume=1000.0
        )
    ]


def create_test_candles_sweep_low(orl: float) -> List[Candle]:
    """
    Create test candles that sweep a given ORL level
    
    Args:
        orl: The opening range low to sweep
        
    Returns:
        List of candles with first candle sweeping the ORL
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    
    return [
        Candle(
            timestamp=base_time,
            open=orl + 5.0,
            high=orl + 6.0,
            low=orl - 2.0,  # Sweeps ORL
            close=orl + 3.0,
            volume=1000.0
        )
    ]


def create_test_candles_no_sweep(orh: float, orl: float) -> List[Candle]:
    """
    Create test candles that don't sweep either level
    
    Args:
        orh: The opening range high
        orl: The opening range low
        
    Returns:
        List of candles staying within the range
    """
    base_time = datetime(2024, 1, 1, 9, 30)
    midpoint = (orh + orl) / 2.0
    
    return [
        Candle(
            timestamp=base_time,
            open=midpoint,
            high=midpoint + 1.0,  # Stays below ORH
            low=midpoint - 1.0,   # Stays above ORL
            close=midpoint,
            volume=1000.0
        )
    ]
