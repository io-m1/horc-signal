# Historical Data Directory

Place your CSV files here. Supported formats:
- HistData.com M1 files
- Dukascopy exports
- Generic OHLCV CSVs

Example download commands:

```bash
# Download EURUSD M1 from HistData (free)
# 1. Go to: http://www.histdata.com/download-free-forex-data/?/metatrader/1-minute-bar-quotes/eurusd
# 2. Download yearly ZIP files
# 3. Extract CSVs here

# Example structure:
# data/
#   EURUSD_M1_2023.csv
#   EURUSD_M1_2024.csv
#   GBPUSD_M1_2023.csv
#   XAUUSD_H1_2024.csv
```

Then run:
```bash
python replay_historical.py --file data/EURUSD_M1_2024.csv --timeframe 15T
```

