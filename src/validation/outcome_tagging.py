import os
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

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

class OutcomeType(Enum):
    WIN = "WIN"           # Hit target before stop
    LOSS = "LOSS"         # Hit stop before target
    TIMEOUT = "TIMEOUT"   # Neither hit within lookforward
    PENDING = "PENDING"   # Not enough future data

@dataclass
class SignalOutcome:
    timestamp: int
    bias: int  # +1 bullish, -1 bearish
    confidence: float
    entry_price: float
    
    mfe: float = 0.0       # Maximum Favorable Excursion
    mae: float = 0.0       # Maximum Adverse Excursion  
    final_move: float = 0.0  # Price change at lookforward end
    
    hit_target: bool = False
    hit_stop: bool = False
    bars_to_target: int = 0
    bars_to_stop: int = 0
    
    outcome: OutcomeType = OutcomeType.PENDING
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'bias': self.bias,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'mfe': self.mfe,
            'mae': self.mae,
            'final_move': self.final_move,
            'hit_target': self.hit_target,
            'hit_stop': self.hit_stop,
            'bars_to_target': self.bars_to_target,
            'bars_to_stop': self.bars_to_stop,
            'outcome': self.outcome.value,
        }

@dataclass
class OutcomeStats:
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    max_mfe: float = 0.0
    max_mae: float = 0.0
    
    win_rate_10pip: float = 0.0
    win_rate_20pip: float = 0.0
    win_rate_30pip: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    
    bull_wins: int = 0
    bull_losses: int = 0
    bear_wins: int = 0
    bear_losses: int = 0
    
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0
    
    @property
    def bull_win_rate(self) -> float:
        total = self.bull_wins + self.bull_losses
        return self.bull_wins / total if total > 0 else 0.0
    
    @property
    def bear_win_rate(self) -> float:
        total = self.bear_wins + self.bear_losses
        return self.bear_wins / total if total > 0 else 0.0

def create_tagging_orchestrator(threshold: float = 0.25) -> HORCOrchestrator:
    
    participant = ParticipantIdentifier()
    wavelength = WavelengthEngine(WavelengthConfig(
        min_move_1_size_atr=0.3,
        max_move_duration_candles=30,
    ))
    exhaustion = ExhaustionDetector(ExhaustionConfig())
    gap_engine = FuturesGapEngine(GapConfig(min_gap_size_percent=0.002))
    
    config = OrchestratorConfig(
        confluence_threshold=threshold,
        require_agreement=False,
    )
    
    return HORCOrchestrator(participant, wavelength, exhaustion, gap_engine, config)

def calculate_outcome(
    signal: SignalIR,
    entry_candle: Candle,
    future_candles: List[Candle],
    target_pips: float = 20.0,
    stop_pips: float = 20.0,
    pip_value: float = 0.0001,  # For forex majors
) -> SignalOutcome:
    entry_price = entry_candle.close
    direction = signal.bias  # +1 or -1
    
    outcome = SignalOutcome(
        timestamp=signal.timestamp,
        bias=direction,
        confidence=signal.confidence,
        entry_price=entry_price,
    )
    
    if not future_candles:
        outcome.outcome = OutcomeType.PENDING
        return outcome
    
    target_price_diff = target_pips * pip_value
    stop_price_diff = stop_pips * pip_value
    
    for i, candle in enumerate(future_candles):
        if direction > 0:  # Bullish
            favorable = (candle.high - entry_price) / pip_value
            adverse = (entry_price - candle.low) / pip_value
            current_move = (candle.close - entry_price) / pip_value
        else:  # Bearish
            favorable = (entry_price - candle.low) / pip_value
            adverse = (candle.high - entry_price) / pip_value
            current_move = (entry_price - candle.close) / pip_value
        
        if favorable > outcome.mfe:
            outcome.mfe = favorable
        if adverse > outcome.mae:
            outcome.mae = adverse
        
        if not outcome.hit_target and favorable >= target_pips:
            outcome.hit_target = True
            outcome.bars_to_target = i + 1
        
        if not outcome.hit_stop and adverse >= stop_pips:
            outcome.hit_stop = True
            outcome.bars_to_stop = i + 1
        
    last_candle = future_candles[-1]
    if direction > 0:
        outcome.final_move = (last_candle.close - entry_price) / pip_value
    else:
        outcome.final_move = (entry_price - last_candle.close) / pip_value
    
    if outcome.hit_target and not outcome.hit_stop:
        outcome.outcome = OutcomeType.WIN
    elif outcome.hit_stop and not outcome.hit_target:
        outcome.outcome = OutcomeType.LOSS
    elif outcome.hit_target and outcome.hit_stop:
        if outcome.bars_to_target <= outcome.bars_to_stop:
            outcome.outcome = OutcomeType.WIN
        else:
            outcome.outcome = OutcomeType.LOSS
    else:
        outcome.outcome = OutcomeType.TIMEOUT
    
    return outcome

def run_outcome_tagging(
    candles: List[Candle],
    threshold: float = 0.25,
    target_pips: float = 20.0,
    stop_pips: float = 20.0,
    lookforward_bars: int = 48,  # 12 hours at M15
    session_bars: int = 96,
    pip_value: float = 0.0001,
) -> Tuple[List[SignalOutcome], OutcomeStats]:
    orchestrator = create_tagging_orchestrator(threshold)
    outcomes: List[SignalOutcome] = []
    
    warmup = session_bars * 2
    
    signals_with_index: List[Tuple[int, SignalIR, Candle]] = []
    
    for i, candle in enumerate(candles):
        if i < warmup:
            if i >= session_bars:
                orchestrator.participant.prev_session_candles = candles[max(0, i-session_bars):i]
            continue
        
        orchestrator.participant.prev_session_candles = candles[i-session_bars:i]
        current = candles[max(warmup, i-10):i+1]
        
        signal = orchestrator.process_bar(candle, None, current)
        
        if signal.actionable:
            signals_with_index.append((i, signal, candle))
    
    for idx, signal, entry_candle in signals_with_index:
        future_start = idx + 1
        future_end = min(idx + 1 + lookforward_bars, len(candles))
        future_candles = candles[future_start:future_end]
        
        outcome = calculate_outcome(
            signal, entry_candle, future_candles,
            target_pips, stop_pips, pip_value
        )
        outcomes.append(outcome)
    
    stats = calculate_outcome_stats(outcomes, target_pips)
    
    return outcomes, stats

def calculate_outcome_stats(
    outcomes: List[SignalOutcome],
    base_target: float = 20.0,
) -> OutcomeStats:
    
    stats = OutcomeStats()
    stats.total_signals = len(outcomes)
    
    if not outcomes:
        return stats
    
    mfe_sum = 0.0
    mae_sum = 0.0
    win_moves = []
    loss_moves = []
    
    for o in outcomes:
        mfe_sum += o.mfe
        mae_sum += o.mae
        
        if o.mfe > stats.max_mfe:
            stats.max_mfe = o.mfe
        if o.mae > stats.max_mae:
            stats.max_mae = o.mae
        
        if o.outcome == OutcomeType.WIN:
            stats.wins += 1
            win_moves.append(o.mfe)
            if o.bias > 0:
                stats.bull_wins += 1
            else:
                stats.bear_wins += 1
                
        elif o.outcome == OutcomeType.LOSS:
            stats.losses += 1
            loss_moves.append(o.mae)
            if o.bias > 0:
                stats.bull_losses += 1
            else:
                stats.bear_losses += 1
                
        elif o.outcome == OutcomeType.TIMEOUT:
            stats.timeouts += 1
    
    stats.avg_mfe = mfe_sum / len(outcomes)
    stats.avg_mae = mae_sum / len(outcomes)
    
    if win_moves:
        stats.avg_win = sum(win_moves) / len(win_moves)
    if loss_moves:
        stats.avg_loss = sum(loss_moves) / len(loss_moves)
    
    stats.win_rate_10pip = sum(1 for o in outcomes if o.mfe >= 10) / len(outcomes)
    stats.win_rate_20pip = sum(1 for o in outcomes if o.mfe >= 20) / len(outcomes)
    stats.win_rate_30pip = sum(1 for o in outcomes if o.mfe >= 30) / len(outcomes)
    
    if stats.wins + stats.losses > 0:
        wr = stats.win_rate
        stats.expectancy = (wr * stats.avg_win) - ((1 - wr) * stats.avg_loss)
    
    return stats

def print_outcome_report(
    outcomes: List[SignalOutcome],
    stats: OutcomeStats,
    target_pips: float,
    stop_pips: float,
):
    
    print("\n" + "=" * 80)
    print("  HORC OUTCOME TAGGING REPORT")
    print("=" * 80)
    
    print(f"\n⚙️  PARAMETERS")
    print(f"   Target: {target_pips:.0f} pips | Stop: {stop_pips:.0f} pips")
    print(f"   Risk/Reward: 1:{target_pips/stop_pips:.1f}")
    
    print(f"\n📊 SIGNAL OUTCOMES (n={stats.total_signals})")
    print("-" * 60)
    print(f"   ✅ Wins:     {stats.wins:>5} ({stats.wins/stats.total_signals*100:.1f}%)")
    print(f"   ❌ Losses:   {stats.losses:>5} ({stats.losses/stats.total_signals*100:.1f}%)")
    print(f"   ⏱️  Timeouts: {stats.timeouts:>5} ({stats.timeouts/stats.total_signals*100:.1f}%)")
    
    print(f"\n🎯 WIN RATE ANALYSIS")
    print("-" * 60)
    
    actual_wr = stats.win_rate * 100
    required_wr = stop_pips / (target_pips + stop_pips) * 100  # Breakeven WR
    edge = actual_wr - required_wr
    
    print(f"   Actual Win Rate:   {actual_wr:.1f}%")
    print(f"   Required (BE):     {required_wr:.1f}%")
    print(f"   Edge:              {edge:+.1f}%")
    
    if edge > 5:
        print(f"   Assessment:        🟢 POSITIVE EDGE")
    elif edge > 0:
        print(f"   Assessment:        🟡 MARGINAL EDGE")
    else:
        print(f"   Assessment:        🔴 NO EDGE")
    
    print(f"\n📈 EXCURSION ANALYSIS")
    print("-" * 60)
    print(f"   Avg MFE:  {stats.avg_mfe:>6.1f} pips (max: {stats.max_mfe:.1f})")
    print(f"   Avg MAE:  {stats.avg_mae:>6.1f} pips (max: {stats.max_mae:.1f})")
    print(f"   MFE/MAE:  {stats.avg_mfe/stats.avg_mae:.2f}x" if stats.avg_mae > 0 else "   MFE/MAE:  N/A")
    
    print(f"\n📊 WIN RATE BY THRESHOLD")
    print("-" * 60)
    print(f"   Hit 10 pips: {stats.win_rate_10pip*100:.1f}%")
    print(f"   Hit 20 pips: {stats.win_rate_20pip*100:.1f}%")
    print(f"   Hit 30 pips: {stats.win_rate_30pip*100:.1f}%")
    
    print(f"\n🧭 DIRECTIONAL BREAKDOWN")
    print("-" * 60)
    print(f"   🟢 Bullish: {stats.bull_wins}W / {stats.bull_losses}L = {stats.bull_win_rate*100:.1f}%")
    print(f"   🔴 Bearish: {stats.bear_wins}W / {stats.bear_losses}L = {stats.bear_win_rate*100:.1f}%")
    
    print(f"\n💰 EXPECTANCY (Per Signal)")
    print("-" * 60)
    print(f"   Avg Win:   +{stats.avg_win:.1f} pips")
    print(f"   Avg Loss:  -{stats.avg_loss:.1f} pips")
    print(f"   Expectancy: {stats.expectancy:+.2f} pips/signal")
    
    if stats.expectancy > 0:
        print(f"   Assessment: 🟢 POSITIVE EXPECTANCY")
    else:
        print(f"   Assessment: 🔴 NEGATIVE EXPECTANCY")
    
    print("\n" + "=" * 80)
    
    print("\n🎯 ACTIONABLE INSIGHTS")
    print("-" * 60)
    
    if stats.avg_mfe > target_pips * 1.5:
        print(f"   → MFE suggests target could be increased to {stats.avg_mfe*0.8:.0f} pips")
    
    if stats.avg_mae < stop_pips * 0.7:
        print(f"   → MAE suggests stop could be tightened to {stats.avg_mae*1.2:.0f} pips")
    
    if abs(stats.bull_win_rate - stats.bear_win_rate) > 0.1:
        better = "BULLISH" if stats.bull_win_rate > stats.bear_win_rate else "BEARISH"
        print(f"   → {better} signals show stronger edge - consider directional filter")
    
    if stats.timeouts > stats.wins:
        print(f"   → Many timeouts - consider extending lookforward or reducing target")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="HORC Outcome Tagging")
    parser.add_argument('--synthetic', '-s', action='store_true')
    parser.add_argument('--file', '-f', type=str, help='CSV file path')
    parser.add_argument('--days', '-d', type=int, default=90)
    parser.add_argument('--threshold', '-t', type=float, default=0.25)
    parser.add_argument('--target', type=float, default=20.0, help='Target in pips')
    parser.add_argument('--stop', type=float, default=20.0, help='Stop in pips')
    parser.add_argument('--lookforward', type=int, default=48, help='Bars to look forward')
    parser.add_argument('--timeframe', type=str, default='15T')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  HORC OUTCOME TAGGING")
    print("=" * 80)
    
    if args.synthetic:
        print(f"\n🔧 Generating {args.days} days of synthetic data...")
        candles = generate_synthetic_data(
            days=args.days,
            timeframe_minutes=15,
            base_price=1.0850,
            volatility=0.0005,
        )
    elif args.file:
        print(f"\n📂 Loading {args.file}...")
        candles = load_historical_csv(args.file, timeframe=args.timeframe)
    else:
        print("❌ Must specify --synthetic or --file")
        sys.exit(1)
    
    print(f"   Loaded {len(candles):,} candles")
    
    print(f"\n🚀 Running outcome tagging...")
    print(f"   Target: {args.target} pips | Stop: {args.stop} pips")
    print(f"   Lookforward: {args.lookforward} bars")
    
    outcomes, stats = run_outcome_tagging(
        candles,
        threshold=args.threshold,
        target_pips=args.target,
        stop_pips=args.stop,
        lookforward_bars=args.lookforward,
    )
    
    print_outcome_report(outcomes, stats, args.target, args.stop)
    
    print("\n🎯 NEXT STEPS:")
    if stats.expectancy > 0:
        print("   ✅ Positive expectancy found!")
        print("   1. Validate on more data (multi-year)")
        print("   2. Test different target/stop combinations")
        print("   3. Proceed to Pine Script mapping")
    else:
        print("   ⚠️ No clear edge yet")
        print("   1. Adjust confluence threshold")
        print("   2. Review signal gating logic")
        print("   3. Try different target/stop ratios")

if __name__ == "__main__":
    main()
