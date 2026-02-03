import asyncio
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional, List
import requests
import websockets
import json

from ..engines import Candle

class MassiveAdapter:
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
        if days is not None:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
        
        if isinstance(from_date, datetime):
            from_date = from_date.strftime("%Y-%m-%d")
        if isinstance(to_date, datetime):
            to_date = to_date.strftime("%Y-%m-%d")
        
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
        uri = f"wss://socket.massive.com/futures"
        
        async with websockets.connect(uri) as ws:
            auth_msg = {"action": "auth", "params": self.api_key}
            await ws.send(json.dumps(auth_msg))
            
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

def example_backtest():
    from src.core import HORCOrchestrator
    
    adapter = MassiveAdapter(api_key="YOUR_API_KEY_HERE")
    orchestrator = HORCOrchestrator()
    
    candles = adapter.get_historical_bars("C:ES", 1, "minute", days=1)
    
    signals = []
    for candle in candles:
        signal = orchestrator.process_bar(candle)
        signals.append(signal)
        
        if signal.actionable:
            print(f"[{candle.timestamp}] SIGNAL: {signal.bias:+d} conf={signal.confidence:.2f}")
    
    actionable = [s for s in signals if s.actionable]
    print(f"\nBacktest complete:")
    print(f"  Bars processed: {len(signals)}")
    print(f"  Actionable signals: {len(actionable)}")

if __name__ == "__main__":
    example_backtest()
