# Quick Start: Get Real Futures Data for HORC

**Bottom line**: Massive.com (the link you shared) is **Polygon.io rebranded**. They're the same company - Polygon announced the rebrand to "Massive" in January 2026.

## Best Option for You: Interactive Brokers

Since you want **free** futures data, here's your path:

### Why Interactive Brokers Wins

| Feature | IB | Massive.com | CME Direct |
|---------|----|-----------|---------| 
| **Cost** | $0 (with account) | $199/month | $1,000s/month |
| **Real-time ES/NQ** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Historical data** | ✅ Yes | ✅ Yes (20+ years) | ✅ Yes |
| **Can execute trades** | ✅ Yes | ❌ No | ❌ No |
| **Setup complexity** | Medium | Easy | Very Hard |

## 5-Minute Setup (Interactive Brokers)

### 1. Install IB Software
```bash
# Download TWS or IB Gateway from:
# https://www.interactivebrokers.com/en/trading/tws.php

# Or use paper trading (no account needed):
# https://www.interactivebrokers.com/en/trading/tws-updateable-latest-demo.php
```

### 2. Enable API Access
- Launch TWS
- **File → Global Configuration → API → Settings**
- Check "Enable ActiveX and Socket Clients"
- Add `127.0.0.1` to trusted IPs
- Note port: **7497** (paper) or **7496** (live)

### 3. Install Python Library
```bash
cd c:\Users\Dell\Documents\horc-signal
.venv\Scripts\activate
pip install ib_insync
```

### 4. Test Connection
```python
from src.data import IBDataAdapter
import asyncio

async def test():
    adapter = IBDataAdapter()
    await adapter.connect()
    print("✅ Connected!")
    
    # Get 1 day of historical ES bars
    candles = await adapter.get_historical_bars("ES", "1 D", "1 min")
    print(f"✅ Got {len(candles)} bars")
    
    adapter.disconnect()

asyncio.run(test())
```

### 5. Run Your HORC System
```python
from src.data import IBDataAdapter
from src.core import HORCOrchestrator
import asyncio

async def trade_es():
    # Initialize
    adapter = IBDataAdapter()
    await adapter.connect()
    orchestrator = HORCOrchestrator()
    
    # Stream live bars
    async for candle in adapter.stream_bars("ES", "1 min"):
        signal = orchestrator.process_bar(candle)
        
        if signal.actionable:
            print(f"\n{'='*60}")
            print(f"🚨 SIGNAL: {signal.bias:+d}")
            print(f"   Confidence: {signal.confidence:.0%}")
            print(f"   Participant: {signal.participant_control:+d}")
            print(f"   Wavelength: {signal.moves_completed}/3")
            print(f"   Exhaustion: {signal.exhaustion_score:.2f}")
            print(f"   Gap Target: {signal.futures_target if signal.has_futures_target else 'None'}")
            print(f"{'='*60}\n")

asyncio.run(trade_es())
```

## Alternative: Massive.com (if you prefer cloud)

Massive.com (the link you shared) is the **same company** as Polygon.io - just rebranded.

### Pricing
- **Stocks Advanced**: $199/month (real-time + futures)
- **Stocks Developer**: $79/month (15-min delayed, good for backtesting)
- **Stocks Basic**: $0/month (delayed, 5 calls/min - learning only)

### When to use Massive instead of IB:
- ✅ Don't want to run TWS locally
- ✅ Need 20+ years of historical data
- ✅ Only doing backtesting (no live trading)
- ✅ Want cloud-based solution

### Quick Start with Massive
```bash
pip install requests websockets
```

```python
from src.data import MassiveAdapter
from src.core import HORCOrchestrator

# Initialize
adapter = MassiveAdapter(api_key="YOUR_KEY")
orchestrator = HORCOrchestrator()

# Get 30 days of ES 1-minute bars
candles = adapter.get_historical_bars("C:ES", 1, "minute", days=30)

# Backtest
for candle in candles:
    signal = orchestrator.process_bar(candle)
    if signal.actionable:
        print(f"[{candle.timestamp}] SIGNAL: {signal.bias:+d} @ {signal.confidence:.0%}")
```

Sign up: https://massive.com/dashboard

## What About CME MDP 3.0 Handlers?

Those are **not for you**. Here's why:

| Requirement | You Need | Institutional Traders |
|-------------|----------|---------------------|
| **Data feed cost** | $0-200/month | $1,000-10,000/month |
| **Latency needs** | Seconds OK | Microseconds critical |
| **Infrastructure** | Laptop | Colocated servers |
| **Volume** | 1 symbol | Thousands of symbols |
| **Setup time** | Minutes | Months |

The handlers (EPAM Java, C++) only **decode** the binary format. You still need to:
1. Pay CME for the feed ($1,000s/month)
2. Set up multicast networking
3. Handle failover/redundancy
4. Colocate servers (for low latency)

**For retail trading, use IB or Massive.**

## Recommended Path

1. **Start with IB Paper Trading** (100% free)
   - Test your HORC system
   - Get comfortable with the API
   - No money at risk

2. **Backtest with historical data** (free from IB)
   - Get 60 days of 1-minute bars
   - Validate your strategy
   - Tune your confluence threshold

3. **Go live when ready** (fund IB account)
   - Real-time data included
   - Can execute trades
   - Professional platform

4. **Optional: Add Massive for deep history**
   - If you need 20+ years of data
   - Great for strategy research
   - $79-199/month

## Files I Created for You

All ready to use:

1. **[src/data/ib_adapter.py](src/data/ib_adapter.py)** - Interactive Brokers adapter
2. **[src/data/polygon_adapter.py](src/data/polygon_adapter.py)** - Massive.com adapter (formerly Polygon)
3. **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** - Complete comparison guide

Both integrate seamlessly with your HORC orchestrator. No changes needed to your existing code.

## Still Have Questions?

- **IB Paper Trading**: https://www.interactivebrokers.com/en/trading/tws-updateable-latest-demo.php
- **Massive Pricing**: https://massive.com/pricing
- **IB API Guide**: https://interactivebrokers.github.io/tws-api/

Your HORC system is production-ready. Just pick a data source and connect!
