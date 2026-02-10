from datetime import datetime, timedelta
import sys
import os

# Ensure src is in path to find engines
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from engines.stealth_dome import StealthDomeEngine, OHLCV
    from engines.horc_coordinates import Participant, RangeType, RangeAnalysis
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.path.append(os.getcwd()) # Try adding root
    try:
        from src.engines.stealth_dome import StealthDomeEngine, OHLCV
        print("Imports successful from root.")
    except ImportError as e2:
        print(f"Import failed from root: {e2}")
        sys.exit(1)

def run_test():
    print("Initializing Engine v8.8...")
    engine = StealthDomeEngine()
    engine.cfg.use_coordinates_filter = True # Enable v8.8 logic
    
    print("Generating dummy data...")
    bars = []
    start_dt = datetime(2025, 1, 1, 9, 30)
    price = 1.1000
    for i in range(200): # 200 minutes
        # Simulate some movement
        if i % 20 < 10: price += 0.0005
        else: price -= 0.0005
        
        bars.append(OHLCV(
            timestamp=start_dt + timedelta(minutes=i),
            open=price, high=price+0.0002, low=price-0.0002, close=price, volume=1000
        ))

    print("Processing bars...")
    signals = 0
    for bar in bars:
        sig = engine.process_bar(bar)
        if sig:
            signals += 1
            # print(f"Signal at {bar.timestamp}: {sig}")
    
    print(f"Processed {len(bars)} bars. Signals generated: {signals}")
    
    # Check Coordinate Tracker state
    print("Checking Coordinate Tracker...")
    # 'D' timeframe (1440 min) won't have closed in 200 mins.
    # '60' (1 hour) should have closed ~3 times.
    
    coords_60 = engine.coord_tracker.coordinates.get('60', [])
    print(f"60-min Coordinates: {len(coords_60)}")
    
    active_60 = engine.coord_tracker.active_periods.get('60')
    if active_60:
        print(f"Active 60-min period: Start {active_60.start_time}, High {active_60.period_high}, Low {active_60.period_low}")
    else:
        print("No active 60-min period found.")

    if len(coords_60) > 0:
        print(f"SUCCESS: {len(coords_60)} Coordinates tracked.")
        for i, c in enumerate(coords_60):
            p_str = "BUYER" if c.participant == Participant.BUYER else "SELLER"
            print(f"  [{i}] {c.time} | {p_str} @ {c.price:.5f} | Liq: {c.is_liquidity}")
            
        # Test Premium Liquidity Detection
        print("\nTesting Premium Liquidity Logic for '60' TF:")
        premium = engine.coord_tracker.find_premium_liquidity('60')
        if premium:
            p_str = "BUYER" if premium.participant == Participant.BUYER else "SELLER"
            print(f"  Found Premium Liquidity: {premium.time} | {p_str} @ {premium.price:.5f}")
        else:
            print("  No Premium Liquidity found (maybe no conclusive signals).")
            
        # Test Range Context
        current_price = bars[-1].close
        print(f"\nTesting Range Analysis at Price {current_price:.5f}:")
        ctx = RangeAnalysis.analyze(engine.coord_tracker, '60', current_price)
        print(f"  Range: {ctx.range_low:.5f} - {ctx.range_high:.5f}")
        print(f"  Type: {ctx.range_type}")
    else:
        print("No '60' coordinates found. (Try running more bars?)")

if __name__ == "__main__":
    run_test()
