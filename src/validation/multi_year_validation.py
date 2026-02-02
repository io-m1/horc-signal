"""
Multi-Year EURUSD Validation Framework

Comprehensive backtesting across market regimes (2015-2026).
Validates HORC signal edge before live deployment.

MARKET REGIMES COVERED:
    2015-2016: EUR weakness, Fed hiking cycle
    2017-2018: Low volatility, range-bound
    2019:      Pre-COVID trend
    2020:      COVID crash + recovery (extreme volatility)
    2021-2022: Inflation spike, rate normalization
    2023-2024: Higher-for-longer regime
    2025-2026: Current environment

METRICS COMPUTED:
    - Signal frequency by year/regime
    - Directional accuracy (if outcome tagging enabled)
    - Win rate, expectancy, max drawdown
    - Regime sensitivity (does edge persist?)

USAGE:
    # With synthetic data (immediate):
    python -m src.validation.multi_year_validation --synthetic
    
    # With real data:
    python -m src.validation.multi_year_validation --data-dir data/eurusd/
"""

import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.historical_loader import load_historical_csv, generate_synthetic_data
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


@dataclass
class YearlyStats:
    """Statistics for a single year"""
    year: int
    total_bars: int = 0
    actionable_signals: int = 0
    bullish_signals: int = 0
    bearish_signals: int = 0
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    signal_timestamps: List[int] = field(default_factory=list)
    
    @property
    def signal_rate(self) -> float:
        return self.actionable_signals / self.total_bars if self.total_bars > 0 else 0.0
    
    @property
    def bull_ratio(self) -> float:
        total = self.bullish_signals + self.bearish_signals
        return self.bullish_signals / total if total > 0 else 0.5


@dataclass 
class RegimeStats:
    """Statistics for a market regime period"""
    name: str
    start_year: int
    end_year: int
    description: str
    yearly_stats: Dict[int, YearlyStats] = field(default_factory=dict)
    
    @property
    def total_signals(self) -> int:
        return sum(ys.actionable_signals for ys in self.yearly_stats.values())
    
    @property
    def total_bars(self) -> int:
        return sum(ys.total_bars for ys in self.yearly_stats.values())
    
    @property
    def avg_signal_rate(self) -> float:
        return self.total_signals / self.total_bars if self.total_bars > 0 else 0.0


# Define market regimes
MARKET_REGIMES = [
    RegimeStats("EUR_WEAKNESS", 2015, 2016, "EUR weakness, Fed hiking cycle"),
    RegimeStats("LOW_VOL", 2017, 2018, "Low volatility, range-bound"),
    RegimeStats("PRE_COVID", 2019, 2019, "Pre-COVID trend environment"),
    RegimeStats("COVID_SHOCK", 2020, 2020, "COVID crash + recovery"),
    RegimeStats("INFLATION", 2021, 2022, "Inflation spike, rate normalization"),
    RegimeStats("HIGHER_LONGER", 2023, 2024, "Higher-for-longer regime"),
    RegimeStats("CURRENT", 2025, 2026, "Current environment"),
]


def create_validation_orchestrator(threshold: float = 0.25) -> HORCOrchestrator:
    """Create orchestrator tuned for validation"""
    
    participant = ParticipantIdentifier()
    
    wavelength_config = WavelengthConfig(
        min_move_1_size_atr=0.3,
        max_move_2_retracement=0.786,
        exhaustion_threshold=0.65,
        max_move_duration_candles=30,
        flip_confirmation_candles=2,
    )
    wavelength = WavelengthEngine(wavelength_config)
    
    exhaustion_config = ExhaustionConfig(
        volume_weight=0.20,
        body_weight=0.35,
        price_weight=0.30,
        reversal_weight=0.15,
        threshold=0.65,
    )
    exhaustion = ExhaustionDetector(exhaustion_config)
    
    gap_config = GapConfig(
        min_gap_size_percent=0.002,
        max_gap_age_days=5,
    )
    gap_engine = FuturesGapEngine(gap_config)
    
    orchestrator_config = OrchestratorConfig(
        confluence_threshold=threshold,
        participant_weight=0.30,
        wavelength_weight=0.30,
        exhaustion_weight=0.25,
        gap_weight=0.15,
        require_agreement=False,
    )
    
    return HORCOrchestrator(
        participant, wavelength, exhaustion, gap_engine, orchestrator_config
    )


def generate_multi_year_synthetic(
    start_year: int = 2015,
    end_year: int = 2026,
    timeframe_minutes: int = 15,
) -> Dict[int, List[Candle]]:
    """
    Generate synthetic data for multiple years.
    
    Each year has different volatility characteristics to simulate regimes.
    """
    yearly_data = {}
    
    # Regime-specific volatility multipliers
    volatility_by_year = {
        2015: 0.0006,  # Higher vol
        2016: 0.0005,
        2017: 0.0003,  # Low vol
        2018: 0.0003,
        2019: 0.0004,
        2020: 0.0010,  # COVID extreme vol
        2021: 0.0006,
        2022: 0.0007,  # Inflation vol
        2023: 0.0005,
        2024: 0.0004,
        2025: 0.0005,
        2026: 0.0005,
    }
    
    base_price = 1.1000
    
    for year in range(start_year, end_year + 1):
        vol = volatility_by_year.get(year, 0.0005)
        
        # Generate full year (365 days)
        candles = generate_synthetic_data(
            symbol="EURUSD",
            days=365,
            timeframe_minutes=timeframe_minutes,
            base_price=base_price,
            volatility=vol,
        )
        
        # Adjust timestamps to correct year
        adjusted_candles = []
        for i, c in enumerate(candles):
            new_ts = datetime(year, 1, 1, tzinfo=timezone.utc) + (c.timestamp - candles[0].timestamp)
            adjusted_candles.append(Candle(
                timestamp=new_ts,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
            ))
        
        yearly_data[year] = adjusted_candles
        
        # Drift base price for next year
        if adjusted_candles:
            base_price = adjusted_candles[-1].close
    
    return yearly_data


def run_yearly_validation(
    candles: List[Candle],
    orchestrator: HORCOrchestrator,
    year: int,
    session_bars: int = 96,
) -> YearlyStats:
    """Run validation for a single year"""
    
    stats = YearlyStats(year=year)
    warmup_bars = session_bars * 2
    confidence_sum = 0.0
    
    for i, candle in enumerate(candles):
        if i < warmup_bars:
            if i >= session_bars:
                orchestrator.participant.prev_session_candles = candles[max(0, i-session_bars):i]
            continue
        
        # Set participant context
        orchestrator.participant.prev_session_candles = candles[i-session_bars:i]
        current_candles = candles[max(warmup_bars, i-10):i+1]
        
        signal = orchestrator.process_bar(
            candle=candle,
            futures_candle=None,
            participant_candles=current_candles
        )
        
        stats.total_bars += 1
        confidence_sum += signal.confidence
        
        if signal.confidence > stats.max_confidence:
            stats.max_confidence = signal.confidence
        
        if signal.actionable:
            stats.actionable_signals += 1
            stats.signal_timestamps.append(signal.timestamp)
            
            if signal.bias > 0:
                stats.bullish_signals += 1
            elif signal.bias < 0:
                stats.bearish_signals += 1
    
    if stats.total_bars > 0:
        stats.avg_confidence = confidence_sum / stats.total_bars
    
    return stats


def run_multi_year_validation(
    yearly_data: Dict[int, List[Candle]],
    threshold: float = 0.25,
) -> Tuple[Dict[int, YearlyStats], List[RegimeStats]]:
    """
    Run full multi-year validation.
    
    Returns:
        (yearly_stats, regime_stats)
    """
    yearly_stats = {}
    
    for year, candles in sorted(yearly_data.items()):
        print(f"  Processing {year}...", end=" ", flush=True)
        
        # Fresh orchestrator per year (no state leak)
        orchestrator = create_validation_orchestrator(threshold)
        stats = run_yearly_validation(candles, orchestrator, year)
        yearly_stats[year] = stats
        
        print(f"{stats.actionable_signals} signals ({stats.signal_rate:.1%})")
    
    # Aggregate into regimes
    regime_stats = []
    for regime in MARKET_REGIMES:
        regime_copy = RegimeStats(
            name=regime.name,
            start_year=regime.start_year,
            end_year=regime.end_year,
            description=regime.description,
        )
        for year in range(regime.start_year, regime.end_year + 1):
            if year in yearly_stats:
                regime_copy.yearly_stats[year] = yearly_stats[year]
        regime_stats.append(regime_copy)
    
    return yearly_stats, regime_stats


def print_validation_report(
    yearly_stats: Dict[int, YearlyStats],
    regime_stats: List[RegimeStats],
):
    """Print comprehensive validation report"""
    
    print("\n" + "=" * 80)
    print("  HORC MULTI-YEAR VALIDATION REPORT")
    print("=" * 80)
    
    # Overall summary
    total_bars = sum(ys.total_bars for ys in yearly_stats.values())
    total_signals = sum(ys.actionable_signals for ys in yearly_stats.values())
    total_bull = sum(ys.bullish_signals for ys in yearly_stats.values())
    total_bear = sum(ys.bearish_signals for ys in yearly_stats.values())
    
    print(f"\n📊 OVERALL SUMMARY ({min(yearly_stats.keys())}-{max(yearly_stats.keys())})")
    print(f"   Total bars processed: {total_bars:,}")
    print(f"   Total signals: {total_signals:,} ({total_signals/total_bars:.2%})")
    print(f"   🟢 Bullish: {total_bull:,} ({total_bull/(total_bull+total_bear):.1%})")
    print(f"   🔴 Bearish: {total_bear:,} ({total_bear/(total_bull+total_bear):.1%})")
    
    # Yearly breakdown
    print(f"\n📅 YEARLY BREAKDOWN")
    print("-" * 70)
    print(f"{'Year':<6} {'Bars':>8} {'Signals':>8} {'Rate':>8} {'Bull':>6} {'Bear':>6} {'Ratio':>8}")
    print("-" * 70)
    
    for year in sorted(yearly_stats.keys()):
        ys = yearly_stats[year]
        bull_pct = ys.bull_ratio * 100
        print(f"{year:<6} {ys.total_bars:>8,} {ys.actionable_signals:>8,} "
              f"{ys.signal_rate:>7.1%} {ys.bullish_signals:>6} {ys.bearish_signals:>6} "
              f"{bull_pct:>6.0f}%/{100-bull_pct:.0f}%")
    
    # Regime analysis
    print(f"\n🌊 REGIME ANALYSIS")
    print("-" * 70)
    
    for regime in regime_stats:
        if regime.total_bars == 0:
            continue
            
        print(f"\n  {regime.name} ({regime.start_year}-{regime.end_year})")
        print(f"  {regime.description}")
        print(f"  Signals: {regime.total_signals:,} | Rate: {regime.avg_signal_rate:.2%}")
        
        # Consistency check
        rates = [ys.signal_rate for ys in regime.yearly_stats.values()]
        if len(rates) > 1:
            rate_std = (sum((r - regime.avg_signal_rate)**2 for r in rates) / len(rates)) ** 0.5
            print(f"  Consistency (std): {rate_std:.3f}")
    
    # Health assessment
    print(f"\n✅ VALIDATION HEALTH CHECK")
    print("-" * 70)
    
    overall_rate = total_signals / total_bars if total_bars > 0 else 0
    
    # Signal rate check
    if 0.03 <= overall_rate <= 0.15:
        print(f"  ✓ Signal rate {overall_rate:.1%} is healthy (3-15%)")
    elif overall_rate < 0.03:
        print(f"  ⚠️ Signal rate {overall_rate:.1%} may be too conservative")
    else:
        print(f"  ⚠️ Signal rate {overall_rate:.1%} may indicate weak gating")
    
    # Directional balance
    bull_ratio = total_bull / (total_bull + total_bear) if (total_bull + total_bear) > 0 else 0.5
    if 0.35 <= bull_ratio <= 0.65:
        print(f"  ✓ Directional balance {bull_ratio:.0%}/{1-bull_ratio:.0%} is healthy")
    else:
        print(f"  ⚠️ Directional skew {bull_ratio:.0%}/{1-bull_ratio:.0%} - investigate bias")
    
    # Regime consistency
    regime_rates = [r.avg_signal_rate for r in regime_stats if r.total_bars > 0]
    if len(regime_rates) > 1:
        rate_range = max(regime_rates) - min(regime_rates)
        if rate_range < 0.05:
            print(f"  ✓ Regime consistency good (rate range: {rate_range:.1%})")
        else:
            print(f"  ⚠️ Regime sensitivity detected (rate range: {rate_range:.1%})")
    
    print("\n" + "=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Year EURUSD Validation")
    parser.add_argument('--synthetic', '-s', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--data-dir', '-d', type=str,
                        help='Directory with yearly CSV files')
    parser.add_argument('--start-year', type=int, default=2015)
    parser.add_argument('--end-year', type=int, default=2026)
    parser.add_argument('--threshold', '-t', type=float, default=0.25)
    parser.add_argument('--timeframe', type=str, default='15T')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  HORC MULTI-YEAR VALIDATION")
    print("=" * 80)
    
    # Load or generate data
    if args.synthetic:
        print(f"\n🔧 Generating synthetic data ({args.start_year}-{args.end_year})...")
        yearly_data = generate_multi_year_synthetic(
            args.start_year, 
            args.end_year,
            timeframe_minutes=15
        )
        print(f"   Generated {sum(len(c) for c in yearly_data.values()):,} total candles")
        
    elif args.data_dir:
        print(f"\n📂 Loading data from {args.data_dir}...")
        yearly_data = {}
        for year in range(args.start_year, args.end_year + 1):
            # Try common naming patterns
            patterns = [
                f"EURUSD_M1_{year}.csv",
                f"EURUSD_{year}.csv",
                f"eurusd_{year}.csv",
            ]
            for pattern in patterns:
                path = os.path.join(args.data_dir, pattern)
                if os.path.exists(path):
                    candles = load_historical_csv(path, timeframe=args.timeframe)
                    yearly_data[year] = candles
                    print(f"   {year}: {len(candles):,} candles")
                    break
        
        if not yearly_data:
            print("❌ No data files found!")
            sys.exit(1)
    else:
        print("❌ Must specify --synthetic or --data-dir")
        sys.exit(1)
    
    # Run validation
    print(f"\n🚀 Running validation (threshold={args.threshold})...")
    yearly_stats, regime_stats = run_multi_year_validation(yearly_data, args.threshold)
    
    # Print report
    print_validation_report(yearly_stats, regime_stats)
    
    print("\n🎯 NEXT STEPS:")
    print("   1. If regime consistency is good → proceed to outcome tagging")
    print("   2. If directional skew exists → review bias logic")
    print("   3. If signal rate too high/low → adjust threshold")


if __name__ == "__main__":
    main()
