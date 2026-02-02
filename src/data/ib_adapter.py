"""
Interactive Brokers Data Adapter for HORC

FREE real-time futures data via IB account.

SETUP:
    1. Open IB account (funded with minimum balance)
    2. Install TWS or IB Gateway
    3. Enable API in TWS: File → Global Configuration → API → Settings
    4. pip install ib_insync
    
USAGE:
    from src.data.ib_adapter import IBDataAdapter
    from src.core import HORCOrchestrator
    
    adapter = IBDataAdapter()
    await adapter.connect()
    
    orchestrator = HORCOrchestrator()
    
    async for candle in adapter.stream_bars("ES", "1min"):
        signal = orchestrator.process_bar(candle)
        if signal.actionable:
            print(f"SIGNAL: {signal}")
"""

import asyncio
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional
from dataclasses import dataclass

try:
    from ib_insync import IB, Future, util
except ImportError:
    raise ImportError(
        "ib_insync not installed. Install with: pip install ib_insync"
    )

from ..engines import Candle


@dataclass
class IBConfig:
    """Interactive Brokers connection configuration"""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper trading (7496 for live)
    client_id: int = 1
    timeout: int = 10
    

class IBDataAdapter:
    """
    Interactive Brokers data adapter for HORC.
    
    Provides real-time futures bars with NO additional data fees
    (included with funded IB account).
    
    Supports:
        - ES (E-mini S&P 500)
        - NQ (E-mini NASDAQ)
        - YM (E-mini Dow)
        - RTY (E-mini Russell 2000)
        - All CME futures
    """
    
    def __init__(self, config: Optional[IBConfig] = None):
        self.config = config or IBConfig()
        self.ib = IB()
        self._connected = False
    
    async def connect(self) -> bool:
        """
        Connect to TWS/IB Gateway.
        
        Returns:
            True if connected successfully
            
        Raises:
            ConnectionError if TWS not running or API not enabled
        """
        try:
            await self.ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout
            )
            self._connected = True
            print(f"✓ Connected to IB ({self.config.host}:{self.config.port})")
            return True
            
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to IB. Is TWS/Gateway running?\n"
                f"Error: {e}\n"
                f"Troubleshooting:\n"
                f"  1. Start TWS or IB Gateway\n"
                f"  2. Enable API: File → Global Config → API → Settings\n"
                f"  3. Check port: {self.config.port} (7497 paper, 7496 live)"
            )
    
    def disconnect(self):
        """Disconnect from IB"""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            print("✓ Disconnected from IB")
    
    def _create_futures_contract(
        self,
        symbol: str,
        exchange: str = "CME",
        currency: str = "USD"
    ) -> Future:
        """
        Create IB futures contract.
        
        Args:
            symbol: Root symbol (ES, NQ, YM, RTY, etc.)
            exchange: Exchange (CME, CBOT, NYMEX, COMEX)
            currency: Currency (USD)
            
        Returns:
            Qualified IB Future contract
            
        Example:
            contract = adapter._create_futures_contract("ES")
            # Returns front month ES contract
        """
        # Get front month contract
        contract = Future(symbol, exchange=exchange, currency=currency)
        
        # Request contract details to get actual expiry
        details = self.ib.reqContractDetails(contract)
        
        if not details:
            raise ValueError(
                f"No contract found for {symbol} on {exchange}. "
                f"Valid symbols: ES, NQ, YM, RTY, GC, CL, etc."
            )
        
        # Return the front month (first detail)
        return details[0].contract
    
    async def stream_bars(
        self,
        symbol: str,
        bar_size: str = "1 min",
        what_to_show: str = "TRADES"
    ) -> AsyncIterator[Candle]:
        """
        Stream real-time bars from IB.
        
        Args:
            symbol: Futures symbol (ES, NQ, YM, RTY)
            bar_size: Bar size ("1 min", "5 mins", "15 mins", "1 hour")
            what_to_show: Data type ("TRADES", "MIDPOINT", "BID", "ASK")
            
        Yields:
            Candle objects compatible with HORC engines
            
        Example:
            async for candle in adapter.stream_bars("ES", "1 min"):
                signal = orchestrator.process_bar(candle)
                print(signal)
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Create contract
        contract = self._create_futures_contract(symbol)
        
        # Request real-time bars (5-second bars, most granular IB offers)
        bars = self.ib.reqRealTimeBars(
            contract,
            5,  # 5-second bars
            what_to_show,
            useRTH=False  # Include extended hours
        )
        
        print(f"✓ Streaming {symbol} bars ({bar_size})...")
        
        # Aggregate 5-second bars into requested bar_size
        bar_seconds = self._parse_bar_size(bar_size)
        current_bar = None
        bar_start = None
        
        async for bar in bars:
            timestamp = datetime.fromtimestamp(bar.time)
            
            # Initialize new bar
            if current_bar is None:
                bar_start = self._floor_timestamp(timestamp, bar_seconds)
                current_bar = {
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                continue
            
            # Check if we need to emit completed bar
            if timestamp >= bar_start + timedelta(seconds=bar_seconds):
                # Emit completed bar
                yield Candle(
                    timestamp=bar_start,
                    open=current_bar['open'],
                    high=current_bar['high'],
                    low=current_bar['low'],
                    close=current_bar['close'],
                    volume=current_bar['volume']
                )
                
                # Start new bar
                bar_start = self._floor_timestamp(timestamp, bar_seconds)
                current_bar = {
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], bar.high)
                current_bar['low'] = min(current_bar['low'], bar.low)
                current_bar['close'] = bar.close
                current_bar['volume'] += bar.volume
    
    async def get_historical_bars(
        self,
        symbol: str,
        duration: str = "1 D",
        bar_size: str = "1 min",
        end_datetime: Optional[datetime] = None
    ) -> list[Candle]:
        """
        Get historical bars for backtesting.
        
        Args:
            symbol: Futures symbol (ES, NQ, etc.)
            duration: How far back ("1 D", "1 W", "1 M", "1 Y")
            bar_size: Bar size ("1 min", "5 mins", "15 mins", "1 hour", "1 day")
            end_datetime: End time (default: now)
            
        Returns:
            List of Candle objects
            
        Example:
            # Get last 1000 1-minute ES bars
            candles = await adapter.get_historical_bars("ES", "1 D", "1 min")
            
            for candle in candles:
                signal = orchestrator.process_bar(candle)
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        contract = self._create_futures_contract(symbol)
        
        end_dt = end_datetime or datetime.now()
        
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_dt,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False
        )
        
        candles = [
            Candle(
                timestamp=bar.date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume
            )
            for bar in bars
        ]
        
        print(f"✓ Retrieved {len(candles)} historical bars for {symbol}")
        return candles
    
    def _parse_bar_size(self, bar_size: str) -> int:
        """Convert bar size string to seconds"""
        size_map = {
            "1 min": 60,
            "5 mins": 300,
            "15 mins": 900,
            "1 hour": 3600
        }
        
        if bar_size not in size_map:
            raise ValueError(
                f"Invalid bar_size: {bar_size}. "
                f"Valid: {list(size_map.keys())}"
            )
        
        return size_map[bar_size]
    
    def _floor_timestamp(self, dt: datetime, seconds: int) -> datetime:
        """Floor timestamp to nearest bar boundary"""
        timestamp = int(dt.timestamp())
        floored = (timestamp // seconds) * seconds
        return datetime.fromtimestamp(floored)


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

async def example_live_stream():
    """Example: Stream live ES bars and generate HORC signals"""
    from src.core import HORCOrchestrator
    
    adapter = IBDataAdapter()
    await adapter.connect()
    
    orchestrator = HORCOrchestrator()
    
    try:
        async for candle in adapter.stream_bars("ES", "1 min"):
            signal = orchestrator.process_bar(candle)
            
            if signal.actionable:
                print(f"\n{'='*60}")
                print(f"🚨 ACTIONABLE SIGNAL")
                print(signal)
                print(f"{'='*60}\n")
            else:
                print(f"[{candle.timestamp}] {signal.bias:+d} conf={signal.confidence:.2f}")
                
    finally:
        adapter.disconnect()


async def example_backtest():
    """Example: Backtest HORC on historical data"""
    from src.core import HORCOrchestrator
    
    adapter = IBDataAdapter()
    await adapter.connect()
    
    # Get 1 day of 1-minute ES bars
    candles = await adapter.get_historical_bars("ES", "1 D", "1 min")
    
    adapter.disconnect()
    
    # Backtest
    orchestrator = HORCOrchestrator()
    signals = []
    
    for candle in candles:
        signal = orchestrator.process_bar(candle)
        signals.append(signal)
        
        if signal.actionable:
            print(f"[{candle.timestamp}] SIGNAL: {signal.bias:+d} conf={signal.confidence:.2f}")
    
    # Statistics
    actionable = [s for s in signals if s.actionable]
    print(f"\nBacktest Results:")
    print(f"  Total bars: {len(signals)}")
    print(f"  Actionable signals: {len(actionable)}")
    print(f"  Signal rate: {len(actionable)/len(signals)*100:.1f}%")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_live_stream())
