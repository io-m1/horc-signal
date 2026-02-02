"""
Massive.com Data Adapter for HORC
(formerly Polygon.io - rebranded January 2026)

AFFORDABLE institutional-grade real-time and historical data.

PRICING (as of February 2026):
    - Stocks Basic: $0/month (delayed, learning only)
    - Stocks Starter: $29/month (15-min delayed)
    - Stocks Developer: $79/month (15-min delayed, more history)
    - Stocks Advanced: $199/month (real-time stocks + futures)
    
SETUP:
    1. Sign up at polygon.io
    2. Get API key
    3. pip install polygon-api-client websockets
    
USAGE:
    from src.data.polygon_adapter import PolygonAdapter
    
    adapter = PolygonAdapter(api_key="YOUR_KEY")
    
    # Historical data
    candles = adapter.get_historical_bars("C:ES", "1", "minute", days=1)
    
    # Live stream
    async for candle in adapter.stream_bars("C:ES"):
        signal = orchestrator.process_bar(candle)
"""

import asyncio
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional, List
import requests
import websockets
import json

from ..engines import Candle


class MassiveAdapter:
    """
    Massive.com data adapter for HORC (formerly Polygon.io).
    
    Institutional-grade data, more affordable than CME direct feeds.
    Excellent for backtesting with 20+ years of historical data.
    Trusted by Google, Revolut, and other major institutions.
    
    Symbol formats:
        - Futures: "C:ES" (continuous ES contract)
        - Stocks: "AAPL"
        - Forex: "C:EURUSD"
        - Options: "O:AAPL251219C00150000"
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.massive.com"
        self._session = requests.Session()
    
    def get_historical_bars(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        days: Optional[int] = None
    ) -> List[Candle]:
        """
        Get historical bars from Polygon.
        
        Args:
            symbol: Symbol (e.g., "C:ES" for ES futures, "AAPL" for stocks)
            multiplier: Bar multiplier (1, 5, 15, etc.)
            timespan: Bar timespan ("minute", "hour", "day")
            from_date: Start date (YYYY-MM-DD) or datetime
            to_date: End date (YYYY-MM-DD) or datetime
            days: Alternative to from_date: get last N days
            
        Returns:
            List of Candle objects
            
        Example:
            # Get 1 day of 1-minute ES bars
            candles = adapter.get_historical_bars("C:ES", 1, "minute", days=1)
            
            # Get specific date range
            candles = adapter.get_historical_bars(
                "C:ES", 5, "minute",
                from_date="2026-02-01",
                to_date="2026-02-02"
            )
        """
        # Handle date parameters
        if days is not None:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
        
        if isinstance(from_date, datetime):
            from_date = from_date.strftime("%Y-%m-%d")
        if isinstance(to_date, datetime):
            to_date = to_date.strftime("%Y-%m-%d")
        
        # Build URL
        url = (
            f"{self.base_url}/v2/aggs/ticker/{symbol}/range/"
            f"{multiplier}/{timespan}/{from_date}/{to_date}"
        )
        
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
        
        response = self._session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("resultsCount", 0) == 0:
            return []
        
        candles = [
            Candle(
                timestamp=datetime.fromtimestamp(bar["t"] / 1000),
                open=bar["o"],
                high=bar["h"],
                low=bar["l"],
                close=bar["c"],
                volume=bar["v"]
            )
            for bar in data.get("results", [])
        ]
        
        print(f"✓ Retrieved {len(candles)} bars for {symbol}")
        return candles
    
    async def stream_bars(
        self,
        symbol: str,
        timeframe: str = "1min"
    ) -> AsyncIterator[Candle]:
        """
        Stream real-time aggregated bars via WebSocket.
        
        Requires Advanced plan ($199/month) for real-time futures.
        
        Args:
            symbol: Symbol (e.g., "C:ES")
            timeframe: Aggregation timeframe ("1min", "5min", etc.)
            
        Yields:
            Candle objects
            
        Example:
            async for candle in adapter.stream_bars("C:ES"):
                signal = orchestrator.process_bar(candle)
        """
        uri = f"wss://socket.massive.com/futures"
        
        async with websockets.connect(uri) as ws:
            # Authenticate
            auth_msg = {"action": "auth", "params": self.api_key}
            await ws.send(json.dumps(auth_msg))
            
            # Subscribe to aggregates
            sub_msg = {
                "action": "subscribe",
                "params": f"CA.{symbol}"  # Continuous aggregate
            }
            await ws.send(json.dumps(sub_msg))
            
            print(f"✓ Streaming {symbol} via Polygon WebSocket...")
            
            async for message in ws:
                data = json.loads(message)
                
                for item in data:
                    if item.get("ev") == "CA":  # Continuous Aggregate
                        yield Candle(
                            timestamp=datetime.fromtimestamp(item["s"] / 1000),
                            open=item["o"],
                            high=item["h"],
                            low=item["l"],
                            close=item["c"],
                            volume=item["v"]
                        )


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

def example_backtest():
    """Example: Backtest with Massive historical data"""
    from src.core import HORCOrchestrator
    
    # Initialize
    adapter = MassiveAdapter(api_key="YOUR_API_KEY_HERE")
    orchestrator = HORCOrchestrator()
    
    # Get 1 day of ES 1-minute bars
    candles = adapter.get_historical_bars("C:ES", 1, "minute", days=1)
    
    # Run HORC
    signals = []
    for candle in candles:
        signal = orchestrator.process_bar(candle)
        signals.append(signal)
        
        if signal.actionable:
            print(f"[{candle.timestamp}] SIGNAL: {signal.bias:+d} conf={signal.confidence:.2f}")
    
    # Stats
    actionable = [s for s in signals if s.actionable]
    print(f"\nBacktest complete:")
    print(f"  Bars processed: {len(signals)}")
    print(f"  Actionable signals: {len(actionable)}")


if __name__ == "__main__":
    example_backtest()
