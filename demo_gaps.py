"""
FuturesGapEngine Demonstration Script
======================================

Demonstrates AXIOM 4: Futures Supremacy - gap detection and target calculation.

This script shows:
1. Gap up detection
2. Gap down detection
3. Gap classification (common, breakaway, exhaustion, measuring)
4. Gap fill detection
5. Target price calculation (nearest unfilled gap)
6. Gravitational pull and fill probability
"""

from datetime import datetime, timedelta
from src.engines.gaps import (
    FuturesGapEngine,
    Gap,
    GapType,
    GapConfig
)
from src.engines.participant import Candle


def print_separator(title: str):
    """Print formatted section separator"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_gap_up_detection():
    """Demonstrate gap up detection"""
    print_separator("DEMO 1: Gap Up Detection")
    
    engine = FuturesGapEngine()
    base_time = datetime(2024, 1, 2, 9, 30)
    
    print("Scenario: Futures open gap up after news event")
    print("-" * 80)
    
    candles = []
    
    # Pre-gap trading
    print("\nPre-Gap Trading:")
    for i in range(3):
        candle = Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=4500.0,
            high=4505.0,
            low=4495.0,
            close=4500.0,
            volume=1000.0
        )
        candles.append(candle)
        print(f"  Candle {i+1}: O={candle.open:.1f}, H={candle.high:.1f}, "
              f"L={candle.low:.1f}, C={candle.close:.1f}")
    
    # Gap up candle
    print("\n📈 GAP UP EVENT:")
    gap_candle = Candle(
        timestamp=base_time + timedelta(minutes=3),
        open=4520.0,  # Opens above previous high (4505.0)
        high=4530.0,
        low=4518.0,
        close=4525.0,
        volume=5000.0
    )
    candles.append(gap_candle)
    print(f"  Candle 4: O={gap_candle.open:.1f}, H={gap_candle.high:.1f}, "
          f"L={gap_candle.low:.1f}, C={gap_candle.close:.1f}")
    print(f"  Volume: {gap_candle.volume:.0f} (5x normal)")
    
    # Post-gap trading
    print("\nPost-Gap Trading:")
    for i in range(2):
        candle = Candle(
            timestamp=base_time + timedelta(minutes=4 + i),
            open=4525.0 + i * 5.0,
            high=4535.0 + i * 5.0,
            low=4523.0 + i * 5.0,
            close=4530.0 + i * 5.0,
            volume=2000.0
        )
        candles.append(candle)
        print(f"  Candle {5+i}: O={candle.open:.1f}, H={candle.high:.1f}, "
              f"L={candle.low:.1f}, C={candle.close:.1f}")
    
    # Detect gaps
    gaps = engine.detect_gaps(candles)
    
    print(f"\n✓ Gaps Detected: {len(gaps)}")
    if gaps:
        gap = gaps[0]
        print(f"\nGap Details:")
        print(f"  Upper:         ${gap.upper:.2f}")
        print(f"  Lower:         ${gap.lower:.2f}")
        print(f"  Size:          ${gap.size:.2f} points")
        print(f"  Target:        ${gap.target_level:.2f} (midpoint)")
        print(f"  Direction:     {gap.direction}")
        print(f"  Type:          {gap.gap_type.value}")
        print(f"  Filled:        {'Yes' if gap.filled else 'No'}")


def demo_gap_down_detection():
    """Demonstrate gap down detection"""
    print_separator("DEMO 2: Gap Down Detection")
    
    engine = FuturesGapEngine()
    base_time = datetime(2024, 1, 2, 9, 30)
    
    print("Scenario: Futures open gap down on negative news")
    print("-" * 80)
    
    candles = []
    
    # Pre-gap trading
    print("\nPre-Gap Trading:")
    for i in range(3):
        candle = Candle(
            timestamp=base_time + timedelta(minutes=i),
            open=4500.0,
            high=4510.0,
            low=4495.0,
            close=4500.0,
            volume=1000.0
        )
        candles.append(candle)
        print(f"  Candle {i+1}: O={candle.open:.1f}, H={candle.high:.1f}, "
              f"L={candle.low:.1f}, C={candle.close:.1f}")
    
    # Gap down candle
    print("\n📉 GAP DOWN EVENT:")
    gap_candle = Candle(
        timestamp=base_time + timedelta(minutes=3),
        open=4475.0,  # Opens below previous low (4495.0)
        high=4480.0,
        low=4470.0,
        close=4473.0,
        volume=6000.0
    )
    candles.append(gap_candle)
    print(f"  Candle 4: O={gap_candle.open:.1f}, H={gap_candle.high:.1f}, "
          f"L={gap_candle.low:.1f}, C={gap_candle.close:.1f}")
    print(f"  Volume: {gap_candle.volume:.0f} (6x normal)")
    
    # Post-gap trading
    print("\nPost-Gap Trading:")
    for i in range(2):
        candle = Candle(
            timestamp=base_time + timedelta(minutes=4 + i),
            open=4473.0 - i * 3.0,
            high=4478.0 - i * 3.0,
            low=4468.0 - i * 3.0,
            close=4470.0 - i * 3.0,
            volume=2000.0
        )
        candles.append(candle)
        print(f"  Candle {5+i}: O={candle.open:.1f}, H={candle.high:.1f}, "
              f"L={candle.low:.1f}, C={candle.close:.1f}")
    
    # Detect gaps
    gaps = engine.detect_gaps(candles)
    
    print(f"\n✓ Gaps Detected: {len(gaps)}")
    if gaps:
        gap = gaps[0]
        print(f"\nGap Details:")
        print(f"  Upper:         ${gap.upper:.2f}")
        print(f"  Lower:         ${gap.lower:.2f}")
        print(f"  Size:          ${gap.size:.2f} points")
        print(f"  Target:        ${gap.target_level:.2f} (midpoint)")
        print(f"  Direction:     {gap.direction}")
        print(f"  Type:          {gap.gap_type.value}")


def demo_gap_classification():
    """Demonstrate gap type classification"""
    print_separator("DEMO 3: Gap Classification")
    
    engine = FuturesGapEngine()
    base_time = datetime(2024, 1, 2, 9, 30)
    
    print("The Four Gap Types:")
    print("-" * 80)
    
    print("\n1. COMMON GAP:")
    print("   • Small size, normal volume")
    print("   • Random occurrence, often fills quickly")
    print("   • Low trading significance")
    
    print("\n2. BREAKAWAY GAP:")
    print("   • Large size, high volume")
    print("   • Occurs after consolidation/range")
    print("   • Signals start of new trend")
    
    print("\n3. EXHAUSTION GAP:")
    print("   • Very large size, extreme volume")
    print("   • Occurs after extended trend")
    print("   • Signals trend exhaustion, likely reversal")
    
    print("\n4. MEASURING GAP (Continuation):")
    print("   • Medium size, elevated volume")
    print("   • Occurs mid-trend")
    print("   • Signals trend continuation")
    
    # Create varied gaps
    print("\n" + "-" * 80)
    print("Creating test gaps with different characteristics...\n")
    
    # Small gap (COMMON)
    candles = [
        Candle(base_time, 4500.0, 4505.0, 4495.0, 4500.0, 1000.0),
        Candle(base_time + timedelta(minutes=1), 4507.0, 4510.0, 4506.0, 4508.0, 1000.0)
        # Small gap, normal volume
    ]
    
    config = GapConfig(min_gap_size_points=1.0)
    engine = FuturesGapEngine(config)
    gaps = engine.detect_gaps(candles)
    
    if gaps:
        print(f"Gap 1: Size=${gaps[0].size:.2f}, Type={gaps[0].gap_type.value.upper()}")
        print("  → Small gap, likely fills quickly")


def demo_gap_fill_detection():
    """Demonstrate gap fill detection"""
    print_separator("DEMO 4: Gap Fill Detection")
    
    engine = FuturesGapEngine()
    base_time = datetime(2024, 1, 2, 9, 30)
    
    print("Scenario: Gap up followed by pullback that fills the gap")
    print("-" * 80)
    
    candles = []
    
    # Pre-gap
    print("\n1. Pre-Gap Trading:")
    candle1 = Candle(base_time, 4500.0, 4505.0, 4495.0, 4500.0, 1000.0)
    candles.append(candle1)
    print(f"   High: ${candle1.high:.2f}")
    
    # Gap up
    print("\n2. Gap Up:")
    candle2 = Candle(
        base_time + timedelta(minutes=1),
        4520.0, 4530.0, 4518.0, 4525.0, 3000.0
    )
    candles.append(candle2)
    print(f"   Open: ${candle2.open:.2f} (above previous high)")
    print(f"   Gap Range: ${candle1.high:.2f} - ${candle2.open:.2f}")
    
    # Rally continues
    print("\n3. Continuation:")
    for i in range(2):
        candle = Candle(
            base_time + timedelta(minutes=2 + i),
            4525.0 + i * 5.0,
            4535.0 + i * 5.0,
            4522.0 + i * 5.0,
            4530.0 + i * 5.0,
            2000.0
        )
        candles.append(candle)
        print(f"   Candle {3+i}: C=${candle.close:.2f}")
    
    # Check gap status
    gaps_before_fill = engine.detect_gaps(candles)
    print(f"\n   Gap Status: {'Filled' if gaps_before_fill[0].filled else 'UNFILLED'}")
    
    # Pullback fills gap
    print("\n4. Pullback Fills Gap:")
    fill_candle = Candle(
        base_time + timedelta(minutes=4),
        4535.0, 4536.0, 4510.0, 4515.0, 4000.0
    )
    candles.append(fill_candle)
    print(f"   High: ${fill_candle.high:.2f}")
    print(f"   Low:  ${fill_candle.low:.2f} (touches gap range)")
    
    # Detect gaps again
    gaps_after_fill = engine.detect_gaps(candles)
    
    if gaps_after_fill:
        gap = gaps_after_fill[0]
        print(f"\n✓ Gap Status: {'FILLED ✓' if gap.filled else 'UNFILLED'}")
        if gap.filled:
            print(f"   Price revisited gap range (${gap.lower:.2f} - ${gap.upper:.2f})")
            print(f"   Gap fill complete!")


def demo_target_calculation():
    """Demonstrate target price calculation"""
    print_separator("DEMO 5: Target Price Calculation (Nearest Unfilled Gap)")
    
    engine = FuturesGapEngine()
    base_time = datetime(2024, 1, 2, 9, 30)
    
    print("Scenario: Multiple unfilled gaps at different price levels")
    print("-" * 80)
    
    candles = []
    current_price = 4500.0
    
    # Create multiple gaps
    print("\nCreating price history with multiple gaps...")
    
    # Gap 1 (down)
    candles.extend([
        Candle(base_time, 4600.0, 4605.0, 4595.0, 4600.0, 1000.0),
        Candle(base_time + timedelta(minutes=1), 4580.0, 4585.0, 4575.0, 4580.0, 2000.0)
    ])
    
    # Gap 2 (down)
    candles.extend([
        Candle(base_time + timedelta(minutes=2), 4580.0, 4585.0, 4575.0, 4580.0, 1000.0),
        Candle(base_time + timedelta(minutes=3), 4520.0, 4525.0, 4515.0, 4520.0, 2000.0)
    ])
    
    # Current level (around 4500)
    candles.append(
        Candle(base_time + timedelta(minutes=4), 4520.0, 4525.0, 4495.0, current_price, 1500.0)
    )
    
    # Gap 3 (down, below current price)
    candles.extend([
        Candle(base_time + timedelta(minutes=5), 4500.0, 4505.0, 4495.0, 4500.0, 1000.0),
        Candle(base_time + timedelta(minutes=6), 4450.0, 4455.0, 4445.0, 4450.0, 2000.0)
    ])
    
    # Detect gaps
    gaps = engine.detect_gaps(candles)
    
    print(f"\n✓ Total Gaps Detected: {len(gaps)}")
    print(f"  Current Price: ${current_price:.2f}\n")
    
    print("Gap Inventory:")
    for i, gap in enumerate(gaps, 1):
        distance = abs(gap.target_level - current_price)
        direction_marker = "↑" if gap.target_level > current_price else "↓"
        filled_marker = "FILLED" if gap.filled else "UNFILLED"
        
        print(f"  Gap {i}: ${gap.target_level:.2f} {direction_marker} "
              f"(distance: ${distance:.2f}) - {filled_marker}")
    
    # Calculate target
    result = engine.analyze_gaps(gaps, current_price, datetime.now())
    
    print("\n" + "-" * 80)
    print("TARGET CALCULATION:")
    print("-" * 80)
    
    if result.target_price:
        print(f"\n🎯 Target Price: ${result.target_price:.2f}")
        print(f"   (Nearest unfilled gap)")
        
        if result.nearest_gap:
            distance = abs(result.target_price - current_price)
            direction = "ABOVE" if result.target_price > current_price else "BELOW"
            
            print(f"\n   Gap Details:")
            print(f"     Range:  ${result.nearest_gap.lower:.2f} - ${result.nearest_gap.upper:.2f}")
            print(f"     Type:   {result.nearest_gap.gap_type.value}")
            print(f"     Position: ${distance:.2f} points {direction} current price")
    else:
        print("\n   No valid target (all gaps filled or expired)")


def demo_gravitational_pull():
    """Demonstrate gravitational pull and fill probability"""
    print_separator("DEMO 6: Gravitational Pull & Fill Probability")
    
    engine = FuturesGapEngine()
    base_time = datetime(2024, 1, 2, 9, 30)
    
    print("Concept: Gaps act as 'gravitational anchors' attracting price")
    print("Formula: Pull ∝ 1/d² (inverse square law)")
    print("-" * 80)
    
    # Create gap
    candles = [
        Candle(base_time, 4500.0, 4505.0, 4495.0, 4500.0, 1000.0),
        Candle(base_time + timedelta(minutes=1), 4520.0, 4530.0, 4518.0, 4525.0, 3000.0)
    ]
    
    gaps = engine.detect_gaps(candles)
    
    if gaps:
        gap = gaps[0]
        test_prices = [4530.0, 4520.0, 4515.0, 4510.0, 4505.0]
        
        print(f"\nGap Range: ${gap.lower:.2f} - ${gap.upper:.2f}")
        print(f"Gap Target: ${gap.target_level:.2f}\n")
        
        print("Price Distance Analysis:")
        print(f"{'Price':<12} {'Distance':<12} {'Pull':<12} {'Fill Prob':<12}")
        print("-" * 48)
        
        for price in test_prices:
            result = engine.analyze_gaps(gaps, price, datetime.now())
            distance = abs(price - gap.target_level)
            
            print(f"${price:<11.2f} ${distance:<11.2f} "
                  f"{result.gravitational_pull:<11.1%} "
                  f"{result.fill_probability:<11.1%}")
        
        print("\nInterpretation:")
        print("  • Closer to gap → Stronger pull")
        print("  • Stronger pull → Higher fill probability")
        print("  • Gaps act as price magnets (structural necessity)")


def demo_complete_analysis():
    """Demonstrate complete gap analysis"""
    print_separator("DEMO 7: Complete Gap Analysis")
    
    engine = FuturesGapEngine()
    base_time = datetime(2024, 1, 2, 9, 30)
    current_date = datetime(2024, 1, 5, 9, 30)  # 3 days later
    
    print("Scenario: Complete analysis with multiple gaps over several days")
    print("-" * 80)
    
    # Build complex gap scenario
    candles = []
    
    # Day 1: Gap up
    candles.extend([
        Candle(base_time, 4500.0, 4505.0, 4495.0, 4500.0, 1000.0),
        Candle(base_time + timedelta(minutes=1), 4520.0, 4530.0, 4518.0, 4525.0, 3000.0)
    ])
    
    # Day 2: Continue higher, then gap down
    candles.extend([
        Candle(base_time + timedelta(days=1), 4550.0, 4560.0, 4545.0, 4550.0, 2000.0),
        Candle(base_time + timedelta(days=1, minutes=1), 4520.0, 4525.0, 4515.0, 4518.0, 3500.0)
    ])
    
    # Day 3: Current price
    current_price = 4510.0
    candles.append(
        Candle(base_time + timedelta(days=2), 4518.0, 4522.0, 4505.0, current_price, 2000.0)
    )
    
    # Detect and analyze
    gaps = engine.detect_gaps(candles)
    result = engine.analyze_gaps(gaps, current_price, current_date)
    
    print("\n" + result.details)
    
    print("\n" + "-" * 80)
    print("TRADING IMPLICATIONS:")
    print("-" * 80)
    
    if result.target_price:
        if result.fill_probability > 0.6:
            print(f"\n✓ HIGH PROBABILITY GAP FILL SETUP")
            print(f"  Target: ${result.target_price:.2f}")
            print(f"  Probability: {result.fill_probability:.1%}")
            print(f"  Action: Consider mean-reversion trade toward gap")
        elif result.fill_probability > 0.3:
            print(f"\n⚠ MODERATE GAP FILL PROBABILITY")
            print(f"  Target: ${result.target_price:.2f}")
            print(f"  Probability: {result.fill_probability:.1%}")
            print(f"  Action: Monitor for confirmation before entry")
        else:
            print(f"\n✗ LOW GAP FILL PROBABILITY")
            print(f"  Probability: {result.fill_probability:.1%}")
            print(f"  Action: Gap may remain unfilled, avoid mean-reversion trades")


def main():
    """Run all demonstrations"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "    AXIOM 4: FUTURES SUPREMACY - FuturesGapEngine Demonstration".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\nThis demonstration validates the complete AXIOM 4 implementation:")
    print("  • Gap detection (up and down)")
    print("  • Gap classification (common, breakaway, exhaustion, measuring)")
    print("  • Gap fill detection")
    print("  • Target calculation (nearest unfilled gap)")
    print("  • Gravitational pull and fill probability")
    
    input("\nPress Enter to begin demonstrations...")
    
    # Run all demos
    demo_gap_up_detection()
    input("\nPress Enter to continue...")
    
    demo_gap_down_detection()
    input("\nPress Enter to continue...")
    
    demo_gap_classification()
    input("\nPress Enter to continue...")
    
    demo_gap_fill_detection()
    input("\nPress Enter to continue...")
    
    demo_target_calculation()
    input("\nPress Enter to continue...")
    
    demo_gravitational_pull()
    input("\nPress Enter to continue...")
    
    demo_complete_analysis()
    
    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Gaps act as structural price magnets (gravitational anchors)")
    print("  2. Unfilled gaps create persistent order book imbalances")
    print("  3. Nearest unfilled gap = highest probability target")
    print("  4. Gravitational pull follows inverse square law (1/d²)")
    print("  5. Gap classification helps predict fill probability")
    print("  6. Based on market microstructure theory (information asymmetry)")
    print("\n")


if __name__ == "__main__":
    main()
