# Data Sources for HORC Signal System

## Reality Check: Free Live Futures Data Doesn't Exist

CME Group (Chicago Mercantile Exchange) charges for real-time futures data. The open-source handlers you found (EPAM Java MDP 3.0, C++ handlers) only **decode** the binary format - you still need to pay CME for the actual feed.

## Recommended Solutions (Ranked by Cost)

### 1. **Interactive Brokers** ⭐ BEST FOR LIVE TRADING
**Cost**: Free with funded account (minimum $2,000-10,000 depending on account type)

**What You Get**:
- Real-time ES, NQ, YM, RTY, all CME futures
- NO additional market data fees if actively trading futures
- Historical data for backtesting
- Professional-grade API

**Setup**:
```bash
pip install ib_insync
```

**Python Code**:
```python
from src.data.ib_adapter import IBDataAdapter
from src.core import HORCOrchestrator

adapter = IBDataAdapter()
await adapter.connect()

orchestrator = HORCOrchestrator()

# Live stream
async for candle in adapter.stream_bars("ES", "1 min"):
    signal = orchestrator.process_bar(candle)
    if signal.actionable:
        print(f"SIGNAL: {signal}")
```

**Pros**:
- ✅ FREE real-time data with account
- ✅ Professional quality (same data as institutions)
- ✅ Best for live trading (can execute directly)
- ✅ Historical data included

**Cons**:
- ❌ Requires funded account
- ❌ Need to run TWS/IB Gateway locally

---

### 2. **Massive.com** (formerly Polygon.io) ⭐ BEST FOR BACKTESTING
**Cost**: $79-199/month (Advanced plan for real-time futures)

**What You Get**:
- Real-time and historical futures, stocks, forex, options
- REST API + WebSocket streaming
- Excellent historical data (20+ years for stocks, 7+ years for futures)
- Nanosecond timestamped tick data
- No software installation required

**Setup**:
```bash
pip install massive websockets
```

**Python Code**:
```python
from src.data.massive_adapter import MassiveAdapter

adapter = MassiveAdapter(api_key="YOUR_KEY")

# Historical backtest
candles = adapter.get_historical_bars("C:ES", 1, "minute", days=30)

for candle in candles:
    signal = orchestrator.process_bar(candle)
```

**Pricing Tiers**:
- **Stocks Basic**: $0/month (5 calls/min, 2 years history, delayed)
- **Stocks Starter**: $29/month (15-min delayed, 5 years history)
- **Stocks Developer**: $79/month (15-min delayed, 10 years history)
- **Stocks Advanced**: $199/month (real-time, 20+ years history)
- **Futures available with Advanced plan**

**Pros**:
- ✅ Institutional-grade data quality
- ✅ Excellent historical data (20+ years)
- ✅ Simple REST API + WebSocket
- ✅ No local software needed
- ✅ Free tier available for learning
- ✅ Trusted by Google, Revolut, Motley Fool

**Cons**:
- ❌ Real-time futures requires $199/month
- ❌ 15-minute delay on cheaper tiers

**Website**: https://massive.com/pricing

---

### 3. **TD Ameritrade API** (Free-ish)
**Cost**: Free with funded account

**What You Get**:
- Real-time futures quotes (with delays on some contracts)
- Historical data
- REST API

**Setup**:
```bash
pip install tda-api
```

**Pros**:
- ✅ Free with account
- ✅ No additional data fees
- ✅ Good for backtesting

**Cons**:
- ❌ Some futures data is delayed
- ❌ API less reliable than IB
- ❌ TDA being merged with Schwab (future uncertain)

---

### 4. **Databento** (Institutional Quality, Affordable)
**Cost**: $99-299/month (pay-as-you-go also available)

**What You Get**:
- Professional-grade historical data
- CME futures (ES, NQ, all products)
- Tick-level precision
- Python SDK

**Setup**:
```bash
pip install databento
```

**Pros**:
- ✅ Institutional data quality
- ✅ Excellent for backtesting
- ✅ Pay-as-you-go option (good for research)

**Cons**:
- ❌ Not free
- ❌ Historical only (no real-time streaming)

**Website**: https://databento.com

---

## Free Alternatives (With Limitations)

### Yahoo Finance (Delayed)
**Cost**: Free

**Limitations**:
- 15-20 minute delayed data
- Futures coverage limited
- No tick data

**Use Case**: Learning, experimentation only

```python
import yfinance as yf
data = yf.download("ES=F", period="1d", interval="1m")
```

### Alpha Vantage (Limited)
**Cost**: Free tier available

**Limitations**:
- 5 API calls per minute (free tier)
- Limited futures coverage
- Delayed data on free tier

---

## What I Recommend for Your HORC System

### For Live Trading:
**Interactive Brokers** - No contest. Free real-time data, can execute trades directly from same API.

### For Backtesting/Development:
**Polygon.io** ($99-199/month) - Best historical data, simple API, perfect for strategy development.

### For Learning (No Budget):
1. Use **IB Paper Trading** (free, no funded account needed for delayed data)
2. Get **60 days of historical data** from IB for free
3. Test your HORC system thoroughly before going live

---

## What About Those CME MDP 3.0 Handlers?

The tools you mentioned (EPAM Java, C++ handlers) are for **institutional users** who:
1. Already pay CME for direct feed ($1,000s/month)
2. Need ultra-low latency (microseconds matter)
3. Process millions of messages/second
4. Run colocated servers at CME data centers

**For retail traders like us, these are overkill and impractical.**

---

## Setup Instructions for IB (Recommended)

### Step 1: Open Account
1. Go to interactivebrokers.com
2. Open account (paper trading or live)
3. Fund account (if live)

### Step 2: Install TWS
1. Download TWS or IB Gateway
2. Install and launch
3. Login with credentials

### Step 3: Enable API
1. In TWS: **File → Global Configuration → API → Settings**
2. Check "Enable ActiveX and Socket Clients"
3. Add "127.0.0.1" to trusted IPs
4. Note the port (7497 for paper, 7496 for live)

### Step 4: Install Python Library
```bash
pip install ib_insync
```

### Step 5: Test Connection
```python
from src.data.ib_adapter import IBDataAdapter

adapter = IBDataAdapter()
await adapter.connect()  # Should print "✓ Connected to IB"
```

### Step 6: Run Your HORC System
```python
# See src/data/ib_adapter.py for full example
async for candle in adapter.stream_bars("ES", "1 min"):
    signal = orchestrator.process_bar(candle)
```

---

## Need Help?

The data adapters are in `src/data/`:
- `ib_adapter.py` - Interactive Brokers (recommended)
- `polygon_adapter.py` - Polygon.io

Both integrate seamlessly with your HORC orchestrator. No changes needed to your existing code.
