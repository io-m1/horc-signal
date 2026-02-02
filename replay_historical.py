"""
Historical Replay Engine for HORC Backtesting

Deterministic replay through historical data → Pine-safe SignalIR stream.
This is the validation checkpoint before live trading or Pine translation.

REQUIREMENTS:
    1. Run the same data twice → identical signal counts (determinism test)
    2. All signals use only Pine-safe primitives
    3. No forward-looking bias (bar-local processing)

USAGE:
    # With real data:
    python replay_historical.py --file data/EURUSD_M1_2024.csv --timeframe 15T
    
    # With synthetic data (no download needed):
    python replay_historical.py --synthetic --days 60

EXPECTED OUTPUT:
    On EURUSD M15 with properly tuned thresholds:
    - Actionable signals: 1-5% of bars (if more → weak gating)
    - Bias distribution: Roughly balanced (not 90% one side)
    - Determinism: Identical counts on rerun
"""

import argparse
import sys
from datetime import datetime

from src.data.historical_loader import (
    load_historical_csv,
    generate_synthetic_data,
    candle_to_pine_timestamp,
)
from src.engines import (
    ParticipantIdentifier,
    WavelengthEngine,
    WavelengthConfig,
    ExhaustionDetector,
    ExhaustionConfig,
    FuturesGapEngine,
    GapConfig,
    Candle,
)
from src.core import HORCOrchestrator, SignalIR
from src.core.orchestrator import OrchestratorConfig


def create_orchestrator(confluence_threshold: float = 0.70) -> HORCOrchestrator:
    """Create orchestrator with default configs suitable for forex replay"""
    
    # Participant identifier
    participant = ParticipantIdentifier()
    
    # Wavelength engine - tuned for 15-minute forex
    wavelength_config = WavelengthConfig(
        min_move_1_size_atr=0.3,  # Lower for forex (smaller moves)
        max_move_2_retracement=0.786,
        exhaustion_threshold=0.65,
        max_move_duration_candles=30,
        flip_confirmation_candles=2,
    )
    wavelength = WavelengthEngine(wavelength_config)
    
    # Exhaustion detector
    exhaustion_config = ExhaustionConfig(
        volume_weight=0.20,  # Lower weight - forex volume unreliable
        body_weight=0.35,
        price_weight=0.30,
        reversal_weight=0.15,
        threshold=0.65,
        volume_lookback=10,
        price_lookback=8,
        reversal_lookback=4,
    )
    exhaustion = ExhaustionDetector(exhaustion_config)
    
    # Gap engine (less relevant for spot forex, but keep for structure)
    gap_config = GapConfig(
        min_gap_size_percent=0.002,  # 0.2% for forex
        max_gap_age_days=5,
        gap_fill_tolerance=0.5,
    )
    gap_engine = FuturesGapEngine(gap_config)
    
    # Orchestrator with forex-appropriate threshold
    # require_agreement=False for testing to allow single-engine signals
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=confluence_threshold,
        participant_weight=0.30,
        wavelength_weight=0.30,  # Higher for forex (structure matters)
        exhaustion_weight=0.25,
        gap_weight=0.15,  # Lower for forex (gaps less frequent)
        require_agreement=False,  # Allow signals with single-engine bias
    )
    
    return HORCOrchestrator(
        participant,
        wavelength,
        exhaustion,
        gap_engine,
        orchestrator_config
    )


def run_replay(
    candles: list[Candle],
    orchestrator: HORCOrchestrator,
    verbose: bool = False,
    session_bars: int = 96,  # 96 x 15min = 24 hours
) -> dict:
    """
    Run deterministic replay through historical data.
    
    Args:
        candles: List of Candle objects
        orchestrator: Configured HORCOrchestrator
        verbose: Print each signal if True
        session_bars: Number of bars per session (for participant ID context)
        
    Returns:
        Dict with replay statistics
    """
    stats = {
        'total_bars': 0,
        'actionable_signals': 0,
        'bullish_signals': 0,
        'bearish_signals': 0,
        'neutral_signals': 0,
        'avg_confidence': 0.0,
        'max_confidence': 0.0,
        'wavelength_moves': {0: 0, 1: 0, 2: 0, 3: 0},
        'exhaustion_triggers': 0,
        'timestamps': [],  # For determinism check
    }
    
    confidence_sum = 0.0
    
    # Track session windows for participant ID
    # We need at least 2 sessions worth of data before we can identify participants
    warmup_bars = session_bars * 2
    
    for i, candle in enumerate(candles):
        # During warmup, we're just building history
        if i < warmup_bars:
            # Set previous session data on the participant identifier
            if i >= session_bars:
                # We have enough history - set prev_session
                prev_start = max(0, i - session_bars)
                orchestrator.participant.prev_session_candles = candles[prev_start:i]
            continue
        
        # Set up participant context
        prev_start = i - session_bars
        prev_end = i
        orchestrator.participant.prev_session_candles = candles[prev_start:prev_end]
        
        # Current session candles (last few for first move detection)
        current_session_start = max(warmup_bars, i - 10)  # Last 10 candles for analysis
        current_candles = candles[current_session_start:i+1]
        
        # Process bar
        signal = orchestrator.process_bar(
            candle=candle,
            futures_candle=None,  # No futures for forex
            participant_candles=current_candles
        )
        
        stats['total_bars'] += 1
        stats['timestamps'].append(signal.timestamp)
        
        # Track wavelength state
        moves = signal.moves_completed
        if moves in stats['wavelength_moves']:
            stats['wavelength_moves'][moves] += 1
        
        # Track exhaustion
        if signal.in_exhaustion_zone:
            stats['exhaustion_triggers'] += 1
        
        # Track confidence
        confidence_sum += signal.confidence
        if signal.confidence > stats['max_confidence']:
            stats['max_confidence'] = signal.confidence
        
        # Count signals by type
        if signal.actionable:
            stats['actionable_signals'] += 1
            
            if signal.bias > 0:
                stats['bullish_signals'] += 1
            elif signal.bias < 0:
                stats['bearish_signals'] += 1
            else:
                stats['neutral_signals'] += 1
            
            if verbose:
                ts = datetime.fromtimestamp(signal.timestamp / 1000)
                direction = "🟢 LONG" if signal.bias > 0 else "🔴 SHORT" if signal.bias < 0 else "⚪ FLAT"
                print(
                    f"{ts.strftime('%Y-%m-%d %H:%M')} | {direction} | "
                    f"Conf: {signal.confidence:.1%} | "
                    f"Moves: {signal.moves_completed}/3 | "
                    f"Exh: {signal.exhaustion_score:.2f}"
                )
    
    # Calculate averages
    if stats['total_bars'] > 0:
        stats['avg_confidence'] = confidence_sum / stats['total_bars']
        stats['signal_rate'] = stats['actionable_signals'] / stats['total_bars']
    else:
        stats['signal_rate'] = 0.0
    
    return stats


def print_summary(stats: dict, run_name: str = ""):
    """Print formatted replay summary"""
    
    print("\n" + "=" * 70)
    print(f"  HORC HISTORICAL REPLAY {run_name}")
    print("=" * 70)
    
    print(f"\n📊 BARS PROCESSED: {stats['total_bars']:,}")
    
    print(f"\n📈 SIGNAL SUMMARY:")
    print(f"   Actionable signals: {stats['actionable_signals']:,} ({stats['signal_rate']:.2%} of bars)")
    print(f"   🟢 Bullish: {stats['bullish_signals']:,}")
    print(f"   🔴 Bearish: {stats['bearish_signals']:,}")
    print(f"   ⚪ Neutral: {stats['neutral_signals']:,}")
    
    if stats['bullish_signals'] + stats['bearish_signals'] > 0:
        bull_pct = stats['bullish_signals'] / (stats['bullish_signals'] + stats['bearish_signals'])
        print(f"   📊 Bull/Bear ratio: {bull_pct:.1%} / {1-bull_pct:.1%}")
    
    print(f"\n🎯 CONFIDENCE METRICS:")
    print(f"   Average: {stats['avg_confidence']:.1%}")
    print(f"   Maximum: {stats['max_confidence']:.1%}")
    
    print(f"\n🌊 WAVELENGTH DISTRIBUTION:")
    for moves, count in sorted(stats['wavelength_moves'].items()):
        pct = count / stats['total_bars'] * 100 if stats['total_bars'] > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   Move {moves}: {count:,} ({pct:.1f}%) {bar}")
    
    print(f"\n💥 EXHAUSTION TRIGGERS: {stats['exhaustion_triggers']:,}")
    
    # Health checks
    print(f"\n✅ HEALTH CHECKS:")
    
    # Signal rate sanity
    if 0.01 <= stats['signal_rate'] <= 0.10:
        print(f"   ✓ Signal rate {stats['signal_rate']:.2%} is in healthy range (1-10%)")
    elif stats['signal_rate'] > 0.10:
        print(f"   ⚠️ Signal rate {stats['signal_rate']:.2%} too high - consider raising threshold")
    else:
        print(f"   ⚠️ Signal rate {stats['signal_rate']:.2%} very low - may need tuning")
    
    # Bias balance
    if stats['bullish_signals'] + stats['bearish_signals'] > 0:
        bull_pct = stats['bullish_signals'] / (stats['bullish_signals'] + stats['bearish_signals'])
        if 0.35 <= bull_pct <= 0.65:
            print(f"   ✓ Directional balance {bull_pct:.0%}/{1-bull_pct:.0%} is healthy")
        else:
            print(f"   ⚠️ Directional bias {bull_pct:.0%}/{1-bull_pct:.0%} may indicate overfitting")
    
    print("\n" + "=" * 70)


def determinism_check(candles: list[Candle]) -> bool:
    """
    Run replay twice and verify identical results.
    
    Returns True if deterministic (MANDATORY for Pine translation).
    """
    print("\n🔒 DETERMINISM CHECK...")
    
    orchestrator1 = create_orchestrator()
    orchestrator2 = create_orchestrator()
    
    stats1 = run_replay(candles, orchestrator1, verbose=False)
    stats2 = run_replay(candles, orchestrator2, verbose=False)
    
    # Compare key metrics
    checks = [
        ('actionable_signals', stats1['actionable_signals'] == stats2['actionable_signals']),
        ('bullish_signals', stats1['bullish_signals'] == stats2['bullish_signals']),
        ('bearish_signals', stats1['bearish_signals'] == stats2['bearish_signals']),
        ('timestamps', stats1['timestamps'] == stats2['timestamps']),
    ]
    
    all_pass = all(c[1] for c in checks)
    
    if all_pass:
        print("   ✅ DETERMINISM VERIFIED - identical results on both runs")
    else:
        print("   ❌ DETERMINISM FAILED:")
        for name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"      {status} {name}")
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="HORC Historical Replay Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Synthetic data (no download needed):
    python replay_historical.py --synthetic --days 60
    
    # Real data:
    python replay_historical.py --file data/EURUSD_M1_2024.csv --timeframe 15T
    
    # Verbose mode (print each signal):
    python replay_historical.py --synthetic --verbose
    
    # Custom threshold:
    python replay_historical.py --synthetic --threshold 0.80
        """
    )
    
    parser.add_argument('--file', '-f', help='Path to CSV data file')
    parser.add_argument('--synthetic', '-s', action='store_true', 
                        help='Use synthetic data (no file needed)')
    parser.add_argument('--days', '-d', type=int, default=60,
                        help='Days of synthetic data (default: 60)')
    parser.add_argument('--timeframe', '-t', default='15T',
                        help='Resample timeframe (default: 15T = 15 minutes)')
    parser.add_argument('--threshold', type=float, default=0.70,
                        help='Confluence threshold (default: 0.70)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print each signal')
    parser.add_argument('--determinism-only', action='store_true',
                        help='Only run determinism check')
    
    args = parser.parse_args()
    
    # Load or generate data
    if args.synthetic:
        print(f"🔧 Generating {args.days} days of synthetic EURUSD data...")
        candles = generate_synthetic_data(
            symbol="EURUSD",
            days=args.days,
            timeframe_minutes=15,
            base_price=1.0850,
            volatility=0.0005
        )
        run_name = f"(Synthetic {args.days}d M15)"
    elif args.file:
        print(f"📂 Loading {args.file}...")
        candles = load_historical_csv(
            args.file,
            timeframe=args.timeframe
        )
        run_name = f"({args.file} {args.timeframe})"
    else:
        print("❌ Must specify --file or --synthetic")
        parser.print_help()
        sys.exit(1)
    
    print(f"   Loaded {len(candles):,} candles")
    print(f"   First: {candles[0].timestamp}")
    print(f"   Last:  {candles[-1].timestamp}")
    
    # Determinism check (always run first)
    is_deterministic = determinism_check(candles)
    
    if args.determinism_only:
        sys.exit(0 if is_deterministic else 1)
    
    if not is_deterministic:
        print("\n⚠️  WARNING: Continuing despite determinism failure...")
    
    # Main replay
    print(f"\n🚀 Running full replay (threshold={args.threshold})...")
    orchestrator = create_orchestrator(confluence_threshold=args.threshold)
    stats = run_replay(candles, orchestrator, verbose=args.verbose)
    
    print_summary(stats, run_name)
    
    # Pine readiness
    print("\n🎯 PINE SCRIPT READINESS:")
    if is_deterministic and stats['signal_rate'] <= 0.10:
        print("   ✅ Ready for Pine Script translation")
        print("   Next: Implement IR → Pine field mapping")
    else:
        issues = []
        if not is_deterministic:
            issues.append("fix determinism issues first")
        if stats['signal_rate'] > 0.10:
            issues.append("reduce signal rate (raise threshold)")
        print(f"   ⚠️  Address issues before Pine: {', '.join(issues)}")


if __name__ == "__main__":
    main()
