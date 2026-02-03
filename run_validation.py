import argparse
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.data.historical_loader import (
    load_historical_csv,
    generate_synthetic_data,
)
from realistic_market_generator import generate_realistic_synthetic_data
from session_manager import SessionManager
from src.core import HORCOrchestrator
from src.core.orchestrator import OrchestratorConfig
from src.core.signal_ir import SignalIR
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

from stress_test_military import (
    MilitaryGradeStressTester,
    Trade,
    StressTestResult,
)

class HORCBacktester:
    def __init__(
        self,
        orchestrator: HORCOrchestrator,
        session_manager: SessionManager,
        stop_loss_atr: float = 1.5,
        take_profit_atr: float = 3.0,
        max_bars_in_trade: int = 96,  # 24 hours on 15m
    ):
        self.orchestrator = orchestrator
        self.session_manager = session_manager
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.max_bars_in_trade = max_bars_in_trade
        
        self.signals: List[SignalIR] = []
        self.trades: List[Trade] = []
        
    def process_bar(
        self,
        candle: Candle,
        htf_candle: Optional[Candle] = None
    ) -> Optional[SignalIR]:
        signal = self.orchestrator.process_bar(candle, htf_candle)
        
        if signal and signal.actionable:
            self.signals.append(signal)
            return signal
        
        return None
    
    def run_backtest(
        self,
        candles: List[Candle],
        htf_candles: Optional[List[Candle]] = None,
        verbose: bool = False
    ) -> List[SignalIR]:
        print(f"🔄 Processing {len(candles)} bars through HORC engine...")
        
        total_ir_generated = 0
        total_actionable = 0
        confidence_scores = []
        
        for i, candle in enumerate(candles):
            htf = htf_candles[i] if htf_candles and i < len(htf_candles) else None
            
            prev_candle = candles[i-1] if i > 0 else None
            self.session_manager.process_bar(candle, prev_candle)
            
            signal = self.orchestrator.process_bar(candle, htf)
            
            if signal:
                total_ir_generated += 1
                confidence_scores.append(signal.confidence)
                
                if signal.actionable:
                    total_actionable += 1
                    self.signals.append(signal)
                    
                    if verbose and len(self.signals) <= 5:
                        print(f"\n   ✅ Signal #{len(self.signals)} at bar {i}")
                        print(f"      Time: {candle.timestamp}")
                        print(f"      Bias: {signal.bias}")
                        print(f"      Confidence: {signal.confidence:.3f}")
                        print(f"      Price: {candle.close:.5f}")
            
            if (i + 1) % 2000 == 0:
                avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                print(f"   Processed {i+1}/{len(candles)} bars | "
                      f"IR: {total_ir_generated} | Actionable: {total_actionable} | "
                      f"Avg Conf: {avg_conf:.3f}")
        
        avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        print(f"\n   📊 Backtest Statistics:")
        print(f"      Total IRs generated: {total_ir_generated}")
        print(f"      Actionable signals: {total_actionable}")
        print(f"      Signal rate: {(total_actionable/len(candles)*100):.3f}%")
        print(f"      Average confidence: {avg_conf:.3f}")
        
        if total_ir_generated == 0:
            print(f"\n   ⚠️  WARNING: No SignalIR generated at all")
            print(f"      - Orchestrator may not be receiving valid bar data")
            print(f"      - Check that engines are properly initialized")
        elif total_actionable == 0:
            print(f"\n   ⚠️  WARNING: IRs generated but none actionable")
            print(f"      - Confidence threshold may be too high: {self.orchestrator.config.confluence_threshold}")
            print(f"      - Max confidence seen: {max(confidence_scores) if confidence_scores else 0:.3f}")
            print(f"      - Recommendation: Lower threshold to {max(confidence_scores)*0.9 if confidence_scores else 0.5:.2f}")
        
        print(f"\n✅ Backtest complete: {len(self.signals)} actionable signals")
        return self.signals
    
    def signals_to_trades(
        self,
        candles: List[Candle],
        atr: float
    ) -> List[Trade]:
        if not self.signals:
            print("⚠️  No signals to convert to trades")
            return []
        
        print(f"📊 Converting {len(self.signals)} signals to trades...")
        
        trades = []
        
        for signal in self.signals:
            entry_idx = None
            for i, candle in enumerate(candles):
                if candle.timestamp == signal.timestamp:
                    entry_idx = i
                    break
            
            if entry_idx is None or entry_idx >= len(candles) - 1:
                continue
            
            entry_candle = candles[entry_idx]
            entry_price = entry_candle.close
            entry_time = entry_candle.timestamp
            
            direction = "LONG" if signal.bias > 0 else "SHORT"
            
            if direction == "LONG":
                stop_price = entry_price - (self.stop_loss_atr * atr)
                target_price = entry_price + (self.take_profit_atr * atr)
            else:
                stop_price = entry_price + (self.stop_loss_atr * atr)
                target_price = entry_price - (self.take_profit_atr * atr)
            
            exit_idx = entry_idx + 1
            exit_reason = "TIMEOUT"
            exit_price = entry_price
            
            for i in range(entry_idx + 1, min(entry_idx + self.max_bars_in_trade, len(candles))):
                bar = candles[i]
                
                if direction == "LONG" and bar.low <= stop_price:
                    exit_idx = i
                    exit_price = stop_price
                    exit_reason = "STOP"
                    break
                elif direction == "SHORT" and bar.high >= stop_price:
                    exit_idx = i
                    exit_price = stop_price
                    exit_reason = "STOP"
                    break
                
                if direction == "LONG" and bar.high >= target_price:
                    exit_idx = i
                    exit_price = target_price
                    exit_reason = "TARGET"
                    break
                elif direction == "SHORT" and bar.low <= target_price:
                    exit_idx = i
                    exit_price = target_price
                    exit_reason = "TARGET"
                    break
            
            if exit_reason == "TIMEOUT":
                exit_idx = min(entry_idx + self.max_bars_in_trade, len(candles) - 1)
                exit_price = candles[exit_idx].close
            
            exit_time = candles[exit_idx].timestamp
            
            if direction == "LONG":
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
            
            pnl_pct = (pnl / entry_price) * 100
            
            trade = Trade(
                entry_time=entry_time,
                exit_time=exit_time,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                size=1.0,  # Standard lot
                pnl=pnl,
                pnl_pct=pnl_pct,
                bars_held=exit_idx - entry_idx,
                exit_reason=exit_reason,
                cps_at_entry=signal.confidence,
                emission_at_entry=getattr(signal, 'emission_strength', 0.0),
                absorption_type=getattr(signal, 'absorption_type', 'NONE')
            )
            
            trades.append(trade)
        
        print(f"✅ {len(trades)} trades simulated")
        return trades

def create_default_orchestrator(
    conf_thresh: float = 0.55,
    exhaustion_thresh: float = 1.5,
    internal_thresh: float = 1.2,
    external_thresh: float = 1.0
) -> HORCOrchestrator:
    participant = ParticipantIdentifier()
    wavelength = WavelengthEngine()
    exhaustion = ExhaustionDetector()
    gap_engine = FuturesGapEngine()
    
    config = OrchestratorConfig(
        confluence_threshold=conf_thresh,
        participant_weight=0.30,
        wavelength_weight=0.25,
        exhaustion_weight=0.25,
        gap_weight=0.20,
    )
    
    orchestrator = HORCOrchestrator(
        participant=participant,
        wavelength=wavelength,
        exhaustion=exhaustion,
        gap_engine=gap_engine,
        config=config
    )
    
    return orchestrator

def calculate_atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period:
        period = len(candles)
    
    trs = []
    for i in range(1, len(candles)):
        prev_close = candles[i-1].close
        high = candles[i].high
        low = candles[i].low
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)
    
    return sum(trs[-period:]) / period if trs else 0.01

def run_validation(
    data_file: Optional[str] = None,
    synthetic: bool = True,
    synthetic_days: int = 365,
    conf_thresh: float = 0.55,
    walk_forward: bool = False,
    output_dir: str = "results/validation"
) -> Dict:
    print("=" * 80)
    print("  🎖️  HORC MILITARY-GRADE VALIDATION")
    print("=" * 80)
    print()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("📊 PHASE 1: DATA PREPARATION")
    print()
    
    if synthetic:
        print(f"   Generating {synthetic_days} days of synthetic data...")
        candles = generate_realistic_synthetic_data(
            symbol="EURUSD",
            days=synthetic_days,
            timeframe_minutes=15,
            base_price=1.1000,
            volatility=0.0005
        )
        print(f"   ✅ Generated {len(candles)} bars")
    else:
        print(f"   Loading data from: {data_file}")
        candles = load_historical_csv(data_file)
        print(f"   ✅ Loaded {len(candles)} bars")
    
    print()
    
    print("📊 PHASE 2: HORC BACKTESTING")
    print()
    
    orchestrator = create_default_orchestrator(conf_thresh=conf_thresh)
    session_manager = SessionManager(orchestrator)
    backtester = HORCBacktester(orchestrator, session_manager)
    
    signals = backtester.run_backtest(candles, verbose=True)
    
    if len(signals) == 0:
        print()
        print("❌ VALIDATION FAILED: No signals generated")
        print("   Possible causes:")
        print("   - Confidence threshold too high")
        print("   - Data quality issues")
        print("   - Engine configuration error")
        print()
        return {"status": "FAILED", "reason": "NO_SIGNALS"}
    
    signal_rate = (len(signals) / len(candles)) * 100
    print(f"   Signal Rate: {signal_rate:.2f}% of bars")
    print()
    
    print("📊 PHASE 3: TRADE SIMULATION")
    print()
    
    atr = calculate_atr(candles)
    print(f"   ATR: {atr:.5f}")
    
    trades = backtester.signals_to_trades(candles, atr)
    
    if len(trades) == 0:
        print()
        print("❌ VALIDATION FAILED: No trades simulated")
        return {"status": "FAILED", "reason": "NO_TRADES"}
    
    print()
    
    print("📊 PHASE 4: MILITARY-GRADE STRESS TEST")
    print()
    
    tester = MilitaryGradeStressTester(
        initial_capital=100000,
        position_size_pct=0.02,
        slippage_bps=2.0,
        commission_per_trade=1.0
    )
    
    price_data = pd.DataFrame({
        'close': [c.close for c in candles],
        'timestamp': [c.timestamp for c in candles]
    })
    price_data.set_index('timestamp', inplace=True)
    
    benchmark_returns = pd.Series([0.0] * len(trades))  # Mock benchmark
    
    stress_result = tester.run_full_stress_test(
        trades=trades,
        price_data=price_data,
        benchmark_returns=benchmark_returns,
        rf_rate=0.02
    )
    
    print("=" * 80)
    print("  💾 SAVING RESULTS")
    print("=" * 80)
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    trades_file = f"{output_dir}/trades_{timestamp}.csv"
    trades_df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'pnl': t.pnl,
        'pnl_pct': t.pnl_pct,
        'bars_held': t.bars_held,
        'exit_reason': t.exit_reason,
        'cps': t.cps_at_entry,
    } for t in trades])
    trades_df.to_csv(trades_file, index=False)
    print(f"   ✅ Trades saved: {trades_file}")
    
    results_file = f"{output_dir}/stress_test_{timestamp}.json"
    results = {
        'timestamp': timestamp,
        'data_source': 'synthetic' if synthetic else data_file,
        'total_bars': len(candles),
        'total_signals': len(signals),
        'signal_rate_pct': signal_rate,
        'total_trades': stress_result.total_trades,
        'win_rate': stress_result.win_rate,
        'sharpe_ratio': stress_result.sharpe_ratio,
        'sortino_ratio': stress_result.sortino_ratio,
        'calmar_ratio': stress_result.calmar_ratio,
        'max_drawdown': stress_result.max_drawdown,
        'profit_factor': stress_result.profit_factor,
        'expectancy': stress_result.expectancy,
        'grade': stress_result.grade,
        'passes_institutional': stress_result.passes_institutional_grade,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✅ Results saved: {results_file}")
    
    print()
    print("=" * 80)
    print(f"  🎖️  VALIDATION GRADE: {stress_result.grade}")
    print(f"  {'✅ PASSES' if stress_result.passes_institutional_grade else '❌ FAILS'} INSTITUTIONAL STANDARDS")
    print("=" * 80)
    print()
    
    print("📋 QUICK SUMMARY:")
    print(f"   Sharpe Ratio: {stress_result.sharpe_ratio:.2f}")
    print(f"   Win Rate: {stress_result.win_rate*100:.1f}%")
    print(f"   Profit Factor: {stress_result.profit_factor:.2f}")
    print(f"   Max Drawdown: {stress_result.max_drawdown*100:.1f}%")
    print()
    
    if stress_result.passes_institutional_grade:
        print("🎉 CONGRATULATIONS! HORC achieves institutional quality.")
        print("   Next steps:")
        print("   1. Run on real multi-year data")
        print("   2. Walk-forward optimization")
        print("   3. Paper trading validation")
        print("   4. Publish results")
    else:
        print("⚠️  HORC needs improvement to reach institutional grade.")
        print("   Recommendations:")
        if stress_result.sharpe_ratio < 1.5:
            print("   - Lower confidence threshold to increase signal frequency")
        if stress_result.win_rate < 0.52:
            print("   - Adjust stop/target ratio (currently 1:2)")
        if stress_result.max_drawdown > 0.20:
            print("   - Reduce position size or tighten stops")
    
    print()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="HORC Military-Grade Validation")
    
    parser.add_argument(
        '--file',
        type=str,
        help='Path to historical CSV file'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic data (no CSV needed)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Days of synthetic data (default: 365)'
    )
    parser.add_argument(
        '--conf-thresh',
        type=float,
        default=0.55,
        help='Confidence threshold (default: 0.55)'
    )
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Run walk-forward optimization'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/validation',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    if not args.synthetic and not args.file:
        print("❌ Error: Must specify either --synthetic or --file")
        sys.exit(1)
    
    try:
        results = run_validation(
            data_file=args.file,
            synthetic=args.synthetic,
            synthetic_days=args.days,
            conf_thresh=args.conf_thresh,
            walk_forward=args.walk_forward,
            output_dir=args.output
        )
        
        if results.get('status') == 'FAILED':
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
