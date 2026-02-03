import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
import itertools

@dataclass
class WalkForwardSegment:
    segment_num: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    best_params: Dict
    train_sharpe: float
    train_trades: int
    
    test_sharpe: float
    test_trades: int
    test_return: float
    test_max_dd: float
    
    sharpe_degradation: float  # train - test

@dataclass
class ParameterSet:
    conf_thresh: float
    exhaustion_thresh: float
    internal_thresh: float
    external_thresh: float
    min_cps: float

class WalkForwardOptimizer:
    def __init__(
        self,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3  # How much to roll forward
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        
    def create_segments(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        segments = []
        current = start_date
        segment_num = 0
        
        while True:
            train_start = current
            train_end = train_start + timedelta(days=30 * self.train_months)
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.test_months)
            
            if test_end > end_date:
                break
            
            segments.append((segment_num, train_start, train_end, test_start, test_end))
            segment_num += 1
            
            current = current + timedelta(days=30 * self.step_months)
        
        return segments
    
    def generate_parameter_space(self) -> List[ParameterSet]:
        conf_thresholds = [0.50, 0.55, 0.60, 0.65]
        exhaustion_thresholds = [1.3, 1.5, 1.7, 2.0]
        internal_thresholds = [1.0, 1.2, 1.4, 1.6]
        external_thresholds = [0.8, 1.0, 1.2, 1.4]
        min_cps_values = [0.50, 0.55, 0.60]
        
        param_sets = []
        
        for conf, exh, int_t, ext, min_cps in itertools.product(
            conf_thresholds,
            exhaustion_thresholds,
            internal_thresholds,
            external_thresholds,
            min_cps_values
        ):
            if exh > int_t > ext:
                param_sets.append(ParameterSet(
                    conf_thresh=conf,
                    exhaustion_thresh=exh,
                    internal_thresh=int_t,
                    external_thresh=ext,
                    min_cps=min_cps
                ))
        
        return param_sets
    
    def optimize_segment(
        self,
        train_data: pd.DataFrame,
        param_space: List[ParameterSet]
    ) -> Tuple[ParameterSet, float]:
        best_params = None
        best_sharpe = -999
        
        for params in param_space:
            sharpe = self._backtest_with_params(train_data, params)
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        
        return best_params, best_sharpe
    
    def test_parameters(
        self,
        test_data: pd.DataFrame,
        params: ParameterSet
    ) -> Dict:
        sharpe = self._backtest_with_params(test_data, params)
        
        return {
            "sharpe": sharpe,
            "trades": np.random.randint(10, 50),  # Mock
            "return": np.random.uniform(-0.1, 0.2),  # Mock
            "max_dd": np.random.uniform(0.05, 0.15)  # Mock
        }
    
    def _backtest_with_params(
        self,
        data: pd.DataFrame,
        params: ParameterSet
    ) -> float:
        base_sharpe = 1.5
        
        if params.conf_thresh < 0.52:
            base_sharpe -= 0.3
        if params.exhaustion_thresh > 2.0:
            base_sharpe -= 0.2
        
        noise = np.random.normal(0, 0.3)
        return base_sharpe + noise
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardSegment]:
        print("=" * 80)
        print("  🔄 WALK-FORWARD OPTIMIZATION")
        print("=" * 80)
        print()
        print(f"Training Window: {self.train_months} months")
        print(f"Testing Window: {self.test_months} months")
        print(f"Roll-Forward Step: {self.step_months} months")
        print()
        
        segments = self.create_segments(start_date, end_date)
        print(f"Total Segments: {len(segments)}")
        print()
        
        param_space = self.generate_parameter_space()
        print(f"Parameter Space: {len(param_space)} combinations")
        print()
        
        results = []
        
        for segment_num, train_start, train_end, test_start, test_end in segments:
            print(f"📊 SEGMENT {segment_num + 1}/{len(segments)}")
            print(f"   Training: {train_start.date()} to {train_end.date()}")
            print(f"   Testing:  {test_start.date()} to {test_end.date()}")
            
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            best_params, train_sharpe = self.optimize_segment(train_data, param_space)
            
            print(f"   Best Params: conf={best_params.conf_thresh:.2f}, "
                  f"exh={best_params.exhaustion_thresh:.1f}, "
                  f"int={best_params.internal_thresh:.1f}")
            print(f"   Train Sharpe: {train_sharpe:.2f}")
            
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            test_results = self.test_parameters(test_data, best_params)
            
            test_sharpe = test_results["sharpe"]
            degradation = train_sharpe - test_sharpe
            
            print(f"   Test Sharpe: {test_sharpe:.2f}")
            print(f"   Degradation: {degradation:.2f} ({'✅ STABLE' if degradation < 0.5 else '⚠️  OVERFITTED'})")
            print()
            
            segment_result = WalkForwardSegment(
                segment_num=segment_num,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params.__dict__,
                train_sharpe=train_sharpe,
                train_trades=np.random.randint(30, 100),
                test_sharpe=test_sharpe,
                test_trades=test_results["trades"],
                test_return=test_results["return"],
                test_max_dd=test_results["max_dd"],
                sharpe_degradation=degradation
            )
            
            results.append(segment_result)
        
        print("=" * 80)
        print("  📊 WALK-FORWARD SUMMARY")
        print("=" * 80)
        print()
        
        avg_train_sharpe = np.mean([s.train_sharpe for s in results])
        avg_test_sharpe = np.mean([s.test_sharpe for s in results])
        avg_degradation = np.mean([s.sharpe_degradation for s in results])
        
        print(f"Average Train Sharpe: {avg_train_sharpe:.2f}")
        print(f"Average Test Sharpe: {avg_test_sharpe:.2f}")
        print(f"Average Degradation: {avg_degradation:.2f}")
        print()
        
        if avg_degradation < 0.3:
            print("✅ STABLE: Parameters generalize well to out-of-sample data")
        elif avg_degradation < 0.5:
            print("⚠️  MODERATE: Some overfitting, acceptable for deployment")
        else:
            print("❌ OVERFITTED: Parameters don't generalize, DO NOT DEPLOY")
        
        print()
        
        print("📊 PARAMETER STABILITY")
        print()
        
        conf_values = [s.best_params["conf_thresh"] for s in results]
        exh_values = [s.best_params["exhaustion_thresh"] for s in results]
        
        print(f"   Confidence Threshold: {np.mean(conf_values):.2f} ± {np.std(conf_values):.2f}")
        print(f"   Exhaustion Threshold: {np.mean(exh_values):.2f} ± {np.std(exh_values):.2f}")
        print()
        
        if np.std(conf_values) < 0.05:
            print("✅ STABLE: Parameters consistent across segments")
        else:
            print("⚠️  UNSTABLE: Parameters vary significantly (market regime dependency)")
        
        print()
        
        return results

def run_example():
    
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="1h")
    prices = pd.DataFrame({
        "close": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    }, index=dates)
    
    optimizer = WalkForwardOptimizer(
        train_months=12,
        test_months=3,
        step_months=3
    )
    
    results = optimizer.run_walk_forward(
        data=prices,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31)
    )
    
    print("=" * 80)
    print("✅ Walk-forward optimization complete!")
    print()
    print("📋 INTEGRATION STEPS:")
    print("   1. Replace mock backtest with real HORC engine")
    print("   2. Load multi-year CSV data")
    print("   3. Run optimization on historical data")
    print("   4. Deploy parameters with lowest degradation")
    print("   5. Monitor live performance vs test period performance")
    print("=" * 80)

if __name__ == "__main__":
    run_example()
