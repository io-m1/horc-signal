import csv
import itertools
from dataclasses import asdict
from typing import List, Dict
import sys

# Import engine and backtester
try:
    from src.engines.stealth_dome import StealthDomeConfig
    from backtest_stealth_dome import run_backtest, Trade
except ImportError:
    print("❌ Error: Run this script from the project root (c:\\Users\\Dell\\Documents\\horc-signal)")
    sys.exit(1)

def calculate_metrics(trades: List[Trade]) -> Dict:
    if not trades:
        return {
            "Trades": 0, "WinRate": 0.0, "ProfitFactor": 0.0, 
            "TotalPnL": 0.0, "MaxDD": 0.0, "AvgR": 0.0
        }
    
    wins = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]
    
    win_rate = len(wins) / len(trades) * 100
    total_pnl = sum(t.pnl_pips for t in trades)
    
    gross_profit = sum(t.pnl_pips for t in trades if t.pnl_pips > 0)
    gross_loss = abs(sum(t.pnl_pips for t in trades if t.pnl_pips < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    
    # Max Drawdown
    equity = [0.0]
    for t in trades:
        equity.append(equity[-1] + t.pnl_pips)
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)
        
    avg_r = sum(t.r_multiple for t in trades) / len(trades)

    return {
        "Trades": len(trades),
        "WinRate": round(win_rate, 1),
        "ProfitFactor": round(pf, 2),
        "TotalPnL": round(total_pnl, 1),
        "MaxDD": round(max_dd, 1),
        "AvgR": round(avg_r, 2)
    }

import contextlib
import os

def main():
    pairs = ["EURUSD", "GBPUSD"]
    
    # Calibration Grid
    # Focused on finding "Beginner Friendly" settings (High WR, Low DD)
    param_grid = {
        "risk_reward": [1.5, 2.0, 2.5],
        "min_divergence": [1, 2],
        "divergence_mode": ["conflict", "consensus"],
        "use_kill_zones": [True, False],
        # Always use v8.8 coordinates
        "use_coordinates_filter": [True]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    print(f"🔬 Starting Calibration: {len(combinations)} configs x {len(pairs)} pairs = {len(combinations)*len(pairs)} runs")
    print("-" * 80)
    print(f"{'PAIR':<6} | {'R:R':<3} | {'DIV':<9} | {'KZ':<5} | {'WO':<9} | {'WR%':<5} | {'PF':<4} | {'TRDS':<6} | {'MaxDD':<6}")
    print("-" * 80)

    for pair in pairs:
        for params in combinations:
            # Create config
            cfg = StealthDomeConfig(**params)
            
            try:
                # Suppress output
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        _, trades, _ = run_backtest(pair, config=cfg)
                
                metrics = calculate_metrics(trades)
                
                # Score for "Beginner Friendliness"
                # Heavy weight on Win Rate and Profit Factor (Efficiency), Penalize DD
                # Score = WR * 2 + PF * 10 - DD / 5
                score = (metrics["WinRate"] * 2) + (metrics["ProfitFactor"] * 10) - (metrics["MaxDD"] * 0.2)
                
                res = {
                    "Pair": pair,
                    **params,
                    **metrics,
                    "Score": round(score, 1)
                }
                results.append(res)
                
                # Print row (Short format)
                div_str = f"{params['min_divergence']}({params['divergence_mode'][:4]})"
                print(f"{pair:<6} | {params['risk_reward']:<3} | {div_str:<9} | {str(params['use_kill_zones'])[0]:<1} | {metrics['WinRate']:<5} | {metrics['ProfitFactor']:<4} | {metrics['Trades']:<6} | {metrics['MaxDD']:<6}")
                
            except Exception as e:
                print(f"❌ Failed run {pair} {params}: {e}")

    # Save CSV
    keys = results[0].keys()
    with open("calibration_results.csv", "w", newline="") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        
    print("-" * 80)
    print("✅ Calibration Complete. Results saved to calibration_results.csv")
    
    # Find Best Per Pair
    print("\n🏆 BEST SETTINGS FOR BEGINNERS (Sorted by Score):")
    for pair in pairs:
        pair_res = [r for r in results if r["Pair"] == pair]
        if not pair_res: continue
        
        # Sort by Score desc
        pair_res.sort(key=lambda x: x["Score"], reverse=True)
        best = pair_res[0]
        
        print(f"\nExample Strategy for {pair}:")
        print(f"  • R:R: {best['risk_reward']}")
        print(f"  • Divergence: Min {best['min_divergence']} ({best['divergence_mode']})")
        print(f"  • Kill Zones: {best['use_kill_zones']}")
        print(f"  -> Win Rate: {best['WinRate']}% | PF: {best['ProfitFactor']} | Trades: {best['Trades']}")

if __name__ == "__main__":
    main()
