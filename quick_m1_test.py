"""Quick M1 backtest to assess HORC accuracy on intraday historical data."""
import pandas as pd
import numpy as np
from datetime import datetime
from src.engines import (
    ParticipantIdentifier, WavelengthEngine, WavelengthConfig,
    ExhaustionDetector, ExhaustionConfig, FuturesGapEngine, GapConfig, Candle
)
from src.core import HORCOrchestrator
from src.core.orchestrator import OrchestratorConfig

print("=" * 80)
print("  🔬 HORC M1 HISTORICAL BACKTEST")
print("  Testing signal generation on real intraday tick data")
print("=" * 80)
print()

# Load M1 data
df = pd.read_csv("data/GBPUSD_M1.csv", index_col=0, parse_dates=True)
print(f"✅ Loaded GBPUSD M1 data")
print(f"   Bars: {len(df):,}")
print(f"   Range: {df.index.min()} to {df.index.max()}")
print(f"   Period: {(df.index.max() - df.index.min()).days} days")
print()

# Take first 10,000 bars for quick test (about 1 week of data)
df = df.head(10000)
print(f"🎯 Testing on first {len(df):,} bars (~7 days)")
print()

# Initialize engines
part = ParticipantIdentifier()
wave = WavelengthEngine(WavelengthConfig())
exh = ExhaustionDetector(ExhaustionConfig())
gap = FuturesGapEngine(GapConfig())

config = OrchestratorConfig(
    confluence_threshold=0.75,
    participant_weight=0.30,
    wavelength_weight=0.25,
    exhaustion_weight=0.25,
    gap_weight=0.20
)

orch = HORCOrchestrator(part, wave, exh, gap, config)

# Build candles
candles = []
for ts, row in df.iterrows():
    c = Candle(
        timestamp=ts.to_pydatetime(),
        open=float(row['open']),
        high=float(row['high']),
        low=float(row['low']),
        close=float(row['close']),
        volume=float(row['volume'])
    )
    candles.append(c)

print("🔄 Processing bars...")
print()

# Simple backtest
in_pos = False
pos_dir = 0
entry = 0.0
stop = 0.0
target = 0.0
trades = []
signals = []

atr = df['high'] - df['low']
avg_atr = atr.rolling(14).mean().bfill()

for i, c in enumerate(candles):
    sig = orch.process_bar(candle=c)
    
    if sig.actionable:
        signals.append({
            'time': c.timestamp,
            'bias': 'LONG' if sig.bias > 0 else 'SHORT',
            'confidence': sig.confidence,
            'price': c.close
        })
    
    if not in_pos and sig.actionable:
        entry = c.close
        pos_dir = sig.bias
        atr_val = float(avg_atr.iloc[i]) if i < len(avg_atr) else 0.0002
        
        if pos_dir > 0:
            stop = entry - 2.0 * atr_val
            target = entry + 2.0 * (entry - stop)
        else:
            stop = entry + 2.0 * atr_val
            target = entry - 2.0 * (stop - entry)
        
        in_pos = True
    
    elif in_pos:
        if pos_dir > 0:
            if c.high >= target:
                rr = (target - entry) / max(1e-6, abs(entry - stop))
                trades.append({'rr': rr, 'outcome': 'WIN', 'bars': i})
                in_pos = False
            elif c.low <= stop:
                trades.append({'rr': -1.0, 'outcome': 'LOSS', 'bars': i})
                in_pos = False
        else:
            if c.low <= target:
                rr = (entry - target) / max(1e-6, abs(entry - stop))
                trades.append({'rr': rr, 'outcome': 'WIN', 'bars': i})
                in_pos = False
            elif c.high >= stop:
                trades.append({'rr': -1.0, 'outcome': 'LOSS', 'bars': i})
                in_pos = False

print("=" * 80)
print("  📊 RESULTS")
print("=" * 80)
print()

print(f"Total Signals Generated: {len(signals)}")
print(f"Total Trades Executed: {len(trades)}")
print()

if signals:
    print("Sample Signals:")
    for s in signals[:5]:
        print(f"  {s['time'].strftime('%Y-%m-%d %H:%M')} | {s['bias']} @ {s['price']:.5f} (conf: {s['confidence']:.2%})")
    if len(signals) > 5:
        print(f"  ... and {len(signals) - 5} more")
    print()

if trades:
    wins = [t for t in trades if t['outcome'] == 'WIN']
    losses = [t for t in trades if t['outcome'] == 'LOSS']
    
    win_rate = len(wins) / len(trades)
    avg_rr = np.mean([t['rr'] for t in trades])
    avg_win = np.mean([t['rr'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['rr'] for t in losses]) if losses else 0
    
    print("Performance Metrics:")
    print(f"  Win Rate: {win_rate:.1%} ({len(wins)}W / {len(losses)}L)")
    print(f"  Average R:R: {avg_rr:.2f}")
    print(f"  Average Win: {avg_win:.2f}R")
    print(f"  Average Loss: {avg_loss:.2f}R")
    print(f"  Expectancy: {avg_rr:.2f}R per trade")
    
    total_return = sum(t['rr'] for t in trades)
    print(f"  Total Return: {total_return:.2f}R")
    
    # Sharpe approximation
    returns = np.array([t['rr'] for t in trades])
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    
    print()
    
    if win_rate >= 0.55 and avg_rr > 0.3:
        print("✅ PASSING: System shows positive edge on historical data")
    elif win_rate >= 0.50 and avg_rr > 0.0:
        print("⚠️  MARGINAL: System is profitable but edge is thin")
    else:
        print("❌ FAILING: System shows no edge on this sample")
else:
    print("⚠️  NO TRADES EXECUTED")
    print("   Possible reasons:")
    print("   • Confluence threshold too high (0.75)")
    print("   • Sample period lacks clear setups")
    print("   • Opening range not establishing properly")

print()
print("=" * 80)
