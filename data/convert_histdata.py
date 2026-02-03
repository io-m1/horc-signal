"""Convert HistData/Dukascopy-style TXT to OHLCV CSV and optional resampling.

Usage:
    python3 data/convert_histdata.py --input data/GBPUSD.txt --symbol GBPUSD --out data/GBPUSD_M1.csv --resample H4

This script is minimal and robust for large files (streaming read).
"""
import argparse
import csv
from datetime import datetime
import pandas as pd


def parse_line(line: str):
    # Expected: TICKER,DTYYYYMMDD,TIME,OPEN,HIGH,LOW,CLOSE,VOL
    parts = line.strip().split(',')
    if len(parts) < 8:
        return None
    ticker, dts, timestr, o, h, l, c, v = parts[:8]
    try:
        ts = datetime.strptime(dts + ' ' + timestr, '%Y%m%d %H%M%S')
    except Exception:
        return None
    return {
        'ticker': ticker,
        'timestamp': ts,
        'open': float(o),
        'high': float(h),
        'low': float(l),
        'close': float(c),
        'volume': float(v)
    }


def convert(input_path: str, symbol: str, out_csv: str, resample: str = None):
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        header = f.readline()  # skip header
        for i, line in enumerate(f):
            parsed = parse_line(line)
            if not parsed:
                continue
            if parsed['ticker'] != symbol:
                continue
            rows.append(parsed)

    if not rows:
        raise SystemExit('No rows parsed for symbol')

    df = pd.DataFrame(rows)
    df = df.set_index('timestamp').sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.to_csv(out_csv)
    print(f'Wrote {len(df)} rows to {out_csv}')

    if resample:
        # Normalize common alias like H4 -> 4H for pandas
        rs = resample.upper()
        if rs.startswith('H') and rs[1:].isdigit():
            rs = (rs[1:] + 'h').lower()
        # Resample OHLCV
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        r = df.resample(rs).agg(agg).dropna()
        out_rs = out_csv.replace('.csv', f'_{resample}.csv')
        r.to_csv(out_rs)
        print(f'Wrote resampled ({resample} -> {rs}) {len(r)} rows to {out_rs}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--symbol', default='GBPUSD')
    p.add_argument('--out', required=True)
    p.add_argument('--resample', default=None, help='Pandas offset alias, e.g. H4, H1, D')
    args = p.parse_args()

    convert(args.input, args.symbol, args.out, args.resample)


if __name__ == '__main__':
    main()
