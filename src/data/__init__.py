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
    pass

try:
    from .polygon_adapter import MassiveAdapter
    PolygonAdapter = MassiveAdapter
    __all__.extend(["MassiveAdapter", "PolygonAdapter"])
except ImportError:
    pass
