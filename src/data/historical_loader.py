import os
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas required for historical data loading. "
        "Install with: pip install pandas"
    )

from ..engines import Candle

@dataclass
class LoaderConfig:
    timeframe: str = "15T"  # Pandas resample string (15T = 15 min, 1H = 1 hour)
    tz: str = "UTC"
    date_format: str = "auto"  # auto-detect or specify
    skip_volume: bool = True  # Forex volume is tick-count, often unreliable

def detect_csv_format(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        header = f.readline().strip().lower()
        first_row = f.readline().strip()
    
    if ';' in first_row:
        sep = ';'
    elif ',' in first_row:
        sep = ','
    else:
        sep = '\t'
    
    has_header = any(col in header for col in ['date', 'time', 'open', 'high', 'low', 'close'])
    
    sample_row = first_row if not has_header else f.readline().strip() if f else first_row
    parts = sample_row.split(sep)
    
    if len(parts) >= 7:  # date, time, O, H, L, C, V
        if '.' not in parts[0] and len(parts[0]) == 8:
            return {
                'sep': sep,
                'has_header': has_header,
                'date_col': 0,
                'time_col': 1,
                'ohlcv_cols': [2, 3, 4, 5, 6],
                'date_format': '%Y%m%d',
                'time_format': '%H%M%S',
                'combined': False
            }
        elif '.' in parts[0]:
            return {
                'sep': sep,
                'has_header': has_header,
                'date_col': 0,
                'time_col': 1,
                'ohlcv_cols': [2, 3, 4, 5, 6],
                'date_format': '%Y.%m.%d',
                'time_format': '%H:%M',
                'combined': False
            }
    
    if len(parts) >= 6:
        return {
            'sep': sep,
            'has_header': has_header,
            'datetime_col': 0,
            'ohlcv_cols': [1, 2, 3, 4, 5],
            'date_format': 'auto',
            'combined': True
        }
    
    raise ValueError(f"Unknown CSV format in {file_path}")

def load_historical_csv(
    file_path: str,
    timeframe: str = "15T",
    tz: str = "UTC",
    skip_volume: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Candle]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    fmt = detect_csv_format(file_path)
    
    df = pd.read_csv(
        file_path,
        sep=fmt['sep'],
        header=0 if fmt['has_header'] else None
    )
    
    if not fmt['has_header']:
        if fmt.get('combined'):
            df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume'][:len(df.columns)]
        else:
            df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume'][:len(df.columns)]
    
    if fmt.get('combined'):
        df['datetime'] = pd.to_datetime(df.iloc[:, 0], utc=True)
    else:
        date_col = df.columns[fmt['date_col']]
        time_col = df.columns[fmt['time_col']]
        
        date_str = df[date_col].astype(str)
        time_str = df[time_col].astype(str)
        
        if fmt['time_format'] == '%H%M%S':
            time_str = time_str.str.zfill(6)
        
        combined_format = f"{fmt['date_format']} {fmt['time_format']}"
        df['datetime'] = pd.to_datetime(
            date_str + ' ' + time_str,
            format=combined_format,
            utc=True
        )
    
    df = df.set_index('datetime')
    
    ohlcv_cols = fmt['ohlcv_cols']
    ohlcv_names = ['open', 'high', 'low', 'close', 'volume']
    
    rename_map = {}
    for i, col_idx in enumerate(ohlcv_cols):
        if col_idx < len(df.columns) + 1:  # +1 because datetime is now index
            old_name = df.columns[col_idx - (0 if fmt.get('combined') else 2)]
            rename_map[old_name] = ohlcv_names[i]
    
    df = df[['open', 'high', 'low', 'close', 'volume']] if 'open' in df.columns else df.iloc[:, :5]
    df.columns = ohlcv_names[:len(df.columns)]
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz='UTC')]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz='UTC')]
    
    if timeframe != '1T':
        df = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    candles: List[Candle] = []
    
    for ts, row in df.iterrows():
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        
        h = max(h, o, c)
        l = min(l, o, c)
        
        candles.append(
            Candle(
                timestamp=ts.to_pydatetime(),  # datetime object
                open=o,
                high=h,
                low=l,
                close=c,
                volume=0.0 if skip_volume else float(row['volume']),
            )
        )
    
    return candles

def candle_to_pine_timestamp(candle: Candle) -> int:
    if isinstance(candle.timestamp, datetime):
        return int(candle.timestamp.timestamp() * 1000)
    return int(candle.timestamp)

def generate_synthetic_data(
    symbol: str = "EURUSD",
    days: int = 30,
    timeframe_minutes: int = 15,
    base_price: float = 1.0850,
    volatility: float = 0.0005,
) -> List[Candle]:
    import random
    
    candles = []
    current_price = base_price
    
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    bars_per_day = (24 * 60) // timeframe_minutes
    total_bars = days * bars_per_day
    
    random.seed(42)  # Deterministic for testing
    
    for i in range(total_bars):
        ts = start + pd.Timedelta(minutes=i * timeframe_minutes)
        
        change = random.gauss(0, volatility)
        mean_reversion = (base_price - current_price) * 0.01
        change += mean_reversion
        
        o = current_price
        c = o + change
        
        range_ext = abs(change) + random.uniform(0, volatility)
        if change >= 0:
            h = c + random.uniform(0, range_ext * 0.5)
            l = o - random.uniform(0, range_ext * 0.5)
        else:
            h = o + random.uniform(0, range_ext * 0.5)
            l = c - random.uniform(0, range_ext * 0.5)
        
        candles.append(Candle(
            timestamp=ts,
            open=round(o, 5),
            high=round(h, 5),
            low=round(l, 5),
            close=round(c, 5),
            volume=round(abs(change) * 1000000 + random.uniform(50000, 150000), 0)
        ))
        
        current_price = c
    
    return candles

if __name__ == "__main__":
    print("Generating synthetic EURUSD data...")
    candles = generate_synthetic_data(
        symbol="EURUSD",
        days=30,
        timeframe_minutes=15,
        base_price=1.0850
    )
    
    print(f"Generated {len(candles)} candles")
    print(f"First: {candles[0].timestamp} O={candles[0].open:.5f}")
    print(f"Last:  {candles[-1].timestamp} C={candles[-1].close:.5f}")
    
    prices = [c.close for c in candles]
    print(f"Price range: {min(prices):.5f} - {max(prices):.5f}")
