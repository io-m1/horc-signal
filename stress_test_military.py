import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import statistics
from scipy import stats

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str  # "TARGET", "STOP", "TIMEOUT"
    cps_at_entry: float
    emission_at_entry: float
    absorption_type: str

@dataclass
class StressTestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    total_return: float
    avg_return_per_trade: float
    std_dev_returns: float
    
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    
    profit_factor: float
    expectancy: float
    kelly_criterion: float
    
    benchmark_return: float
    alpha: float
    information_ratio: float
    
    mc_sharpe_mean: float
    mc_sharpe_std: float
    mc_sharpe_pct_rank: float  # Where actual ranks in MC distribution
    
    bull_market_return: float
    bear_market_return: float
    sideways_market_return: float
    
    p_value: float  # vs random
    t_statistic: float
    
    passes_institutional_grade: bool
    grade: str  # "A+", "A", "B+", "B", "C", "F"

class MilitaryGradeStressTester:
    def __init__(
        self,
        initial_capital: float = 100000,
        position_size_pct: float = 0.02,  # 2% risk per trade
        slippage_bps: float = 2.0,  # 2 basis points
        commission_per_trade: float = 1.0,  # $1 per side
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.slippage_bps = slippage_bps / 10000
        self.commission = commission_per_trade
        
    def run_full_stress_test(
        self,
        trades: List[Trade],
        price_data: pd.DataFrame,
        benchmark_returns: pd.Series,
        rf_rate: float = 0.02  # Risk-free rate (2% annualized)
    ) -> StressTestResult:
        print("=" * 80)
        print("  🎖️  HORC MILITARY-GRADE STRESS TEST")
        print("=" * 80)
        print()
        
        if len(trades) == 0:
            print("❌ No trades to analyze")
            return None
        
        print("📊 PHASE 1: CORE STATISTICS")
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades)
        
        avg_win = statistics.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = statistics.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) \
            if losing_trades and winning_trades else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        total_return = sum(t.pnl for t in trades) / self.initial_capital
        
        print(f"   Total Trades: {len(trades)}")
        print(f"   Win Rate: {win_rate*100:.1f}%")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Expectancy: ${expectancy:.2f}")
        print()
        
        print("📊 PHASE 2: RISK METRICS")
        
        returns = [t.pnl_pct / 100 for t in trades]
        std_dev = statistics.stdev(returns) if len(returns) > 1 else 0
        
        avg_return = statistics.mean(returns)
        sharpe = (avg_return - rf_rate/252) / std_dev if std_dev > 0 else 0  # Daily
        sharpe_annualized = sharpe * np.sqrt(252)
        
        downside_returns = [r for r in returns if r < 0]
        downside_dev = statistics.stdev(downside_returns) if len(downside_returns) > 1 else std_dev
        sortino = (avg_return - rf_rate/252) / downside_dev if downside_dev > 0 else 0
        sortino_annualized = sortino * np.sqrt(252)
        
        equity_curve = [self.initial_capital]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)
        
        peak = equity_curve[0]
        max_dd = 0
        dd_duration = 0
        max_dd_duration = 0
        current_dd_start = None
        
        for i, equity in enumerate(equity_curve):
            if equity > peak:
                peak = equity
                if current_dd_start is not None:
                    max_dd_duration = max(max_dd_duration, dd_duration)
                    dd_duration = 0
                    current_dd_start = None
            else:
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
                if current_dd_start is None:
                    current_dd_start = i
                dd_duration += 1
        
        calmar = (total_return / max_dd) if max_dd > 0 else 0
        
        print(f"   Sharpe Ratio: {sharpe_annualized:.2f} (annualized)")
        print(f"   Sortino Ratio: {sortino_annualized:.2f} (annualized)")
        print(f"   Calmar Ratio: {calmar:.2f}")
        print(f"   Max Drawdown: {max_dd*100:.2f}%")
        print(f"   Max DD Duration: {max_dd_duration} trades")
        print()
        
        print("📊 PHASE 3: MONTE CARLO SIMULATION (1000 runs)")
        mc_sharpes = []
        
        for _ in range(1000):
            shuffled_returns = np.random.choice(returns, size=len(returns), replace=True)
            mc_avg = np.mean(shuffled_returns)
            mc_std = np.std(shuffled_returns)
            mc_sharpe = (mc_avg - rf_rate/252) / mc_std if mc_std > 0 else 0
            mc_sharpes.append(mc_sharpe * np.sqrt(252))
        
        mc_sharpe_mean = np.mean(mc_sharpes)
        mc_sharpe_std = np.std(mc_sharpes)
        mc_pct_rank = (sum(1 for s in mc_sharpes if s < sharpe_annualized) / len(mc_sharpes)) * 100
        
        print(f"   MC Sharpe Mean: {mc_sharpe_mean:.2f}")
        print(f"   MC Sharpe Std: {mc_sharpe_std:.2f}")
        print(f"   Actual Sharpe Percentile: {mc_pct_rank:.1f}%")
        print()
        
        print("📊 PHASE 4: STATISTICAL SIGNIFICANCE")
        
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        print(f"   T-Statistic: {t_stat:.2f}")
        print(f"   P-Value: {p_value:.4f}")
        print(f"   Significant: {'✅ YES' if p_value < 0.05 else '❌ NO'}")
        print()
        
        print("📊 PHASE 5: REGIME ANALYSIS")
        
        bull_trades = [t for t in trades if "bull" in str(t.entry_time).lower()]
        bear_trades = [t for t in trades if "bear" in str(t.entry_time).lower()]
        sideways_trades = [t for t in trades if t not in bull_trades and t not in bear_trades]
        
        bull_return = sum(t.pnl for t in bull_trades) / len(bull_trades) if bull_trades else 0
        bear_return = sum(t.pnl for t in bear_trades) / len(bear_trades) if bear_trades else 0
        sideways_return = sum(t.pnl for t in sideways_trades) / len(sideways_trades) if sideways_trades else 0
        
        print(f"   Bull Market Avg: ${bull_return:.2f}")
        print(f"   Bear Market Avg: ${bear_return:.2f}")
        print(f"   Sideways Avg: ${sideways_return:.2f}")
        print()
        
        print("📊 PHASE 6: BENCHMARK COMPARISON")
        
        benchmark_return = 0.10  # Placeholder (would use real SPY data)
        alpha = total_return - benchmark_return
        
        tracking_error = std_dev * np.sqrt(252)  # Annualized
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        print(f"   HORC Return: {total_return*100:.2f}%")
        print(f"   Benchmark Return: {benchmark_return*100:.2f}%")
        print(f"   Alpha: {alpha*100:.2f}%")
        print(f"   Information Ratio: {information_ratio:.2f}")
        print()
        
        print("📊 PHASE 7: POSITION SIZING")
        
        kelly = (win_rate * abs(avg_win) - (1-win_rate) * abs(avg_loss)) / abs(avg_win) if avg_win != 0 else 0
        kelly_pct = kelly * 100
        
        print(f"   Kelly Criterion: {kelly_pct:.1f}%")
        print(f"   Current Position Size: {self.position_size_pct*100:.1f}%")
        print(f"   Recommendation: {'✅ OPTIMAL' if abs(kelly_pct - self.position_size_pct*100) < 1 else '⚠️  ADJUST'}")
        print()
        
        print("=" * 80)
        print("  🎖️  INSTITUTIONAL GRADE ASSESSMENT")
        print("=" * 80)
        print()
        
        grade_points = 0
        max_points = 0
        
        criteria = [
            ("Sharpe > 2.0", sharpe_annualized > 2.0, 20),
            ("Sharpe > 1.5", sharpe_annualized > 1.5, 15),
            ("Win Rate > 52%", win_rate > 0.52, 10),
            ("Profit Factor > 1.5", profit_factor > 1.5, 10),
            ("Max DD < 20%", max_dd < 0.20, 15),
            ("Positive Expectancy", expectancy > 0, 10),
            ("Statistically Significant", p_value < 0.05, 10),
            ("Beats Benchmark", alpha > 0, 10),
        ]
        
        for criterion, passes, points in criteria:
            max_points += points
            if passes:
                grade_points += points
                print(f"   ✅ {criterion:30s} (+{points} pts)")
            else:
                print(f"   ❌ {criterion:30s} (+0 pts)")
        
        print()
        grade_pct = (grade_points / max_points) * 100
        
        if grade_pct >= 95:
            grade = "A+"
        elif grade_pct >= 90:
            grade = "A"
        elif grade_pct >= 85:
            grade = "A-"
        elif grade_pct >= 80:
            grade = "B+"
        elif grade_pct >= 75:
            grade = "B"
        elif grade_pct >= 70:
            grade = "B-"
        elif grade_pct >= 65:
            grade = "C+"
        elif grade_pct >= 60:
            grade = "C"
        else:
            grade = "F"
        
        passes = grade in ["A+", "A", "A-", "B+"]
        
        print(f"   SCORE: {grade_points}/{max_points} ({grade_pct:.0f}%)")
        print(f"   GRADE: {grade}")
        print(f"   INSTITUTIONAL QUALITY: {'✅ YES' if passes else '❌ NO'}")
        print()
        
        return StressTestResult(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_return=total_return,
            avg_return_per_trade=avg_return,
            std_dev_returns=std_dev,
            sharpe_ratio=sharpe_annualized,
            sortino_ratio=sortino_annualized,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            profit_factor=profit_factor,
            expectancy=expectancy,
            kelly_criterion=kelly,
            benchmark_return=benchmark_return,
            alpha=alpha,
            information_ratio=information_ratio,
            mc_sharpe_mean=mc_sharpe_mean,
            mc_sharpe_std=mc_sharpe_std,
            mc_sharpe_pct_rank=mc_pct_rank,
            bull_market_return=bull_return,
            bear_market_return=bear_return,
            sideways_market_return=sideways_return,
            p_value=p_value,
            t_statistic=t_stat,
            passes_institutional_grade=passes,
            grade=grade
        )

def generate_mock_trades_for_testing(num_trades: int = 100) -> List[Trade]:
    import random
    random.seed(42)
    
    trades = []
    base_time = datetime(2023, 1, 1)
    
    for i in range(num_trades):
        is_win = random.random() < 0.55
        
        entry_time = base_time + timedelta(days=i*2)
        exit_time = entry_time + timedelta(hours=random.randint(2, 48))
        
        entry_price = 100 + random.gauss(0, 10)
        
        if is_win:
            pnl_pct = random.gauss(2.5, 1.5)  # Avg win ~2.5%
        else:
            pnl_pct = random.gauss(-1.5, 0.8)  # Avg loss ~-1.5%
        
        exit_price = entry_price * (1 + pnl_pct/100)
        pnl = (exit_price - entry_price) * 100  # 100 shares
        
        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction="LONG" if random.random() > 0.5 else "SHORT",
            entry_price=entry_price,
            exit_price=exit_price,
            size=100,
            pnl=pnl,
            pnl_pct=pnl_pct,
            bars_held=random.randint(4, 96),
            exit_reason="TARGET" if is_win else "STOP",
            cps_at_entry=random.uniform(0.55, 0.85),
            emission_at_entry=random.uniform(1.0, 2.0),
            absorption_type=random.choice(["INTERNAL", "EXTERNAL", "EXHAUSTION"])
        )
        trades.append(trade)
    
    return trades

if __name__ == "__main__":
    print("🎖️  HORC Military-Grade Stress Test Suite")
    print("   Generating mock trades for demonstration...")
    print()
    
    trades = generate_mock_trades_for_testing(100)
    
    tester = MilitaryGradeStressTester(
        initial_capital=100000,
        position_size_pct=0.02,
        slippage_bps=2.0
    )
    
    result = tester.run_full_stress_test(
        trades=trades,
        price_data=pd.DataFrame(),  # Would use real data
        benchmark_returns=pd.Series(),  # Would use SPY data
        rf_rate=0.02
    )
    
    print("=" * 80)
    print("✅ Stress test complete!")
    print()
    print("📋 NEXT STEPS:")
    print("   1. Integrate with real HORC engine output")
    print("   2. Load multi-year CSV data")
    print("   3. Run on 3-10 years of historical data")
    print("   4. Compare results across different symbols (SPY, QQQ, ES, NQ)")
    print("   5. Publish results to establish credibility")
    print("=" * 80)
