"""
Historical Data Loader for HORC Backtesting

Pine-safe historical replay from CSV files.
Supports HistData.com, Dukascopy, and similar formats.

DATA SOURCES (Free):
    1. HistData.com - M1 CSV for all major pairs (2000-present)
       http://www.histdata.com/download-free-forex-historical-data/
       
    2. Dukascopy - Tick + M1/H1 (bank-level quality)
       https://www.dukascopy.com/datafeed/historical/
       
    3. Forexite - Daily + H1
       https://www.forexite.com/free_forex_quotes/

USAGE:
    from src.data.historical_loader import load_historical_csv
    
    candles = load_historical_csv(
        "data/EURUSD_M1_2024.csv",
        timeframe="15T"
    )
    
    for candle in candles:
        signal = orchestrator.process_bar(candle)

SUPPORTED FORMATS:
    - HistData ASCII: date,time,open,high,low,close,volume
    - Dukascopy: timestamp,open,high,low,close,volume
    - Generic OHLCV: datetime,open,high,low,close,volume
"""

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
    """Configuration for historical data loader"""
    timeframe: str = "15T"  # Pandas resample string (15T = 15 min, 1H = 1 hour)
    tz: str = "UTC"
    date_format: str = "auto"  # auto-detect or specify
    skip_volume: bool = True  # Forex volume is tick-count, often unreliable


def detect_csv_format(file_path: str) -> dict:
    """
    Auto-detect CSV format from header and first few rows.
    
    Returns:
        Dict with parsing instructions
    """
    with open(file_path, 'r') as f:
        header = f.readline().strip().lower()
        first_row = f.readline().strip()
    
    # HistData format: no header, columns are date;time;open;high;low;close;volume
    # Or with header: Date,Time,Open,High,Low,Close,Volume
    
    if ';' in first_row:
        sep = ';'
    elif ',' in first_row:
        sep = ','
    else:
        sep = '\t'
    
    # Check if first line is header
    has_header = any(col in header for col in ['date', 'time', 'open', 'high', 'low', 'close'])
    
    # Detect date format from first data row
    sample_row = first_row if not has_header else f.readline().strip() if f else first_row
    parts = sample_row.split(sep)
    
    # Common formats
    if len(parts) >= 7:  # date, time, O, H, L, C, V
        # HistData: 20240101;000000;1.10234;1.10250;1.10200;1.10245;100
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
        # HistData alt: 2024.01.01,00:00,1.10234,...
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
    
    # Combined datetime column
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
    """
    Load historical CSV and return Pine-safe Candle list.
    
    Args:
        file_path: Path to CSV file
        timeframe: Resample timeframe (1T=1min, 5T=5min, 15T=15min, 1H=1hour, 4H=4hour)
        tz: Timezone (default UTC)
        skip_volume: If True, set volume to 0 (forex tick volume unreliable)
        start_date: Filter start (YYYY-MM-DD)
        end_date: Filter end (YYYY-MM-DD)
        
    Returns:
        List of Candle objects with datetime timestamps
        
    Note:
        For Pine-safe replay, convert candle.timestamp to int(timestamp() * 1000)
        in your orchestrator or replay script.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Auto-detect format
    fmt = detect_csv_format(file_path)
    
    # Load CSV
    df = pd.read_csv(
        file_path,
        sep=fmt['sep'],
        header=0 if fmt['has_header'] else None
    )
    
    # Assign column names if no header
    if not fmt['has_header']:
        if fmt.get('combined'):
            df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume'][:len(df.columns)]
        else:
            df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume'][:len(df.columns)]
    
    # Parse datetime
    if fmt.get('combined'):
        df['datetime'] = pd.to_datetime(df.iloc[:, 0], utc=True)
    else:
        date_col = df.columns[fmt['date_col']]
        time_col = df.columns[fmt['time_col']]
        
        # Convert to string and combine
        date_str = df[date_col].astype(str)
        time_str = df[time_col].astype(str)
        
        # Pad time if needed (e.g., 0 -> 000000)
        if fmt['time_format'] == '%H%M%S':
            time_str = time_str.str.zfill(6)
        
        combined_format = f"{fmt['date_format']} {fmt['time_format']}"
        df['datetime'] = pd.to_datetime(
            date_str + ' ' + time_str,
            format=combined_format,
            utc=True
        )
    
    # Set index
    df = df.set_index('datetime')
    
    # Select OHLCV columns
    ohlcv_cols = fmt['ohlcv_cols']
    ohlcv_names = ['open', 'high', 'low', 'close', 'volume']
    
    # Rename columns to standard names
    rename_map = {}
    for i, col_idx in enumerate(ohlcv_cols):
        if col_idx < len(df.columns) + 1:  # +1 because datetime is now index
            old_name = df.columns[col_idx - (0 if fmt.get('combined') else 2)]
            rename_map[old_name] = ohlcv_names[i]
    
    # Just select the columns we need by position after datetime parsing
    df = df[['open', 'high', 'low', 'close', 'volume']] if 'open' in df.columns else df.iloc[:, :5]
    df.columns = ohlcv_names[:len(df.columns)]
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaN rows
    df = df.dropna()
    
    # Date filtering
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz='UTC')]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz='UTC')]
    
    # Resample to target timeframe
    if timeframe != '1T':
        df = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    
    # Build Candle list
    candles: List[Candle] = []
    
    for ts, row in df.iterrows():
        # Validate OHLC relationships before creating candle
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        
        # Fix any OHLC inconsistencies from resampling
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
    """Convert Candle timestamp to Pine-safe UNIX milliseconds"""
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
    """
    Generate synthetic OHLCV data for testing.
    
    Useful when you don't have real data yet.
    
    Args:
        symbol: Symbol name (for reference only)
        days: Number of days to generate
        timeframe_minutes: Candle period in minutes
        base_price: Starting price
        volatility: Price volatility per candle
        
    Returns:
        List of synthetic Candles
    """
    import random
    
    candles = []
    current_price = base_price
    
    # Start from a fixed date for reproducibility
    start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    bars_per_day = (24 * 60) // timeframe_minutes
    total_bars = days * bars_per_day
    
    random.seed(42)  # Deterministic for testing
    
    for i in range(total_bars):
        ts = start + pd.Timedelta(minutes=i * timeframe_minutes)
        
        # Random walk with mean reversion
        change = random.gauss(0, volatility)
        mean_reversion = (base_price - current_price) * 0.01
        change += mean_reversion
        
        o = current_price
        c = o + change
        
        # Generate high/low
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
            volume=0.0
        ))
        
        current_price = c
    
    return candles


if __name__ == "__main__":
    # Demo: Generate synthetic data and show stats
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
    
    # Price stats
    prices = [c.close for c in candles]
    print(f"Price range: {min(prices):.5f} - {max(prices):.5f}")
