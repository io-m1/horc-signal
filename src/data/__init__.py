"""
Data adapters for HORC signal system.

Available adapters:
    - IBDataAdapter: Interactive Brokers (free with account)
    - MassiveAdapter: Massive.com (formerly Polygon.io, $79-199/month)
    - AlpacaAdapter: Alpaca (free for stocks, limited futures)
    - Historical loader: CSV replay for backtesting
"""

# Historical loader (always available)
from .historical_loader import (
    load_historical_csv,
    generate_synthetic_data,
    candle_to_pine_timestamp,
    LoaderConfig,
)

__all__ = [
    "load_historical_csv",
    "generate_synthetic_data", 
    "candle_to_pine_timestamp",
    "LoaderConfig",
]

try:
    from .ib_adapter import IBDataAdapter, IBConfig
    __all__.extend(["IBDataAdapter", "IBConfig"])
except ImportError:
    # ib_insync not installed
    pass

try:
    from .polygon_adapter import MassiveAdapter
    # Alias for backward compatibility
    PolygonAdapter = MassiveAdapter
    __all__.extend(["MassiveAdapter", "PolygonAdapter"])
except ImportError:
    # requests not installed
    pass
