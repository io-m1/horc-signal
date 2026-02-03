## Data Sources

### Interactive Brokers (Recommended for Live Trading)

Free with funded account. Real-time ES, NQ, all CME futures. Historical data included.

Setup:
```
pip install ib_insync
```

Usage:
```python
from src.data.ib_adapter import IBDataAdapter

adapter = IBDataAdapter()
await adapter.connect()
candles = await adapter.get_historical_bars("ES", "1 D", "1 min")
```

### Polygon.io / Massive.com (Recommended for Backtesting)

$79-199/month. 20+ years historical data. REST API + WebSocket.

Setup:
```
pip install requests websockets
```

Usage:
```python
from src.data.polygon_adapter import PolygonAdapter

adapter = PolygonAdapter(api_key="YOUR_KEY")
candles = adapter.get_historical_bars("C:ES", 1, "minute", days=30)
```

### Historical CSV

Local CSV files in data/ folder.

Usage:
```python
from src.data.historical_loader import load_historical_csv

candles = load_historical_csv("data/EURUSD_M1_RTH.csv")
```

### Data Folder Contents

- EURUSD_M1_RTH.csv: 486 days, 546MB
- GBPUSD_M1.csv: 28MB
- convert_histdata.py: Conversion utility
