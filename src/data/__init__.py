"""
Data adapters for HORC signal system.

Available adapters:
    - IBDataAdapter: Interactive Brokers (free with account)
    - MassiveAdapter: Massive.com (formerly Polygon.io, $79-199/month)
    - AlpacaAdapter: Alpaca (free for stocks, limited futures)
"""

try:
    from .ib_adapter import IBDataAdapter, IBConfig
    __all__ = ["IBDataAdapter", "IBConfig"]
except ImportError:
    # ib_insync not installed
    __all__ = []

try:
    from .polygon_adapter import MassiveAdapter
    # Alias for backward compatibility
    PolygonAdapter = MassiveAdapter
    __all__.extend(["MassiveAdapter", "PolygonAdapter"])
except ImportError:
    # requests not installed
    pass
