from src.data.historical_loader import generate_synthetic_data
from src.core.emission_engine import EmissionEngine, AbsorptionType
import statistics
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SignalOutcome:
    bar_index: int
    signal_type: str  # "BUY" or "SELL"
    entry_price: float
    stop_price: float
    target_price: float
    emission_norm: float
    absorption_type: AbsorptionType
    divergence_axes: int
    intent_balance: float
    conf_score: float
    outcome: Optional[str] = None  # "WIN", "LOSS", or None (open)
    bars_held: int = 0
    pnl: float = 0.0

def simulate_horc_signals(candles, engine, confluence_threshold=0.55):
    engine.EXHAUSTION_THRESHOLD = 1.5
    engine.INTERNAL_THRESHOLD = 1.2
    engine.EXTERNAL_THRESHOLD = 1.0
    
    signals = []
    intent_balance = 0.0
    current_participant = 0  # INCONCLUSIVE
    expected_dir = 0
    defended_liq = None
    w_state = "PRE_OR"
    orh = None
    orl = None
    or_formed = False
    or_end_bar = None
    move1_bar = None
    move2_bar = None
    flip_bar = None
    wavelength_bias = 0.0
    sess_fail_count = 0
    
    active_signal = None
    
    for i, candle in enumerate(candles):
        if i < 20:  # Need history for ATR/emission
            continue
        
        recent_ranges = [c.high - c.low for c in candles[max(0, i-14):i+1]]
        atr = statistics.mean(recent_ranges) if recent_ranges else 0.01
        
        if i % 96 == 0 and i > 0:  # 96 bars = 1 day at 15min
            orh = None
            orl = None
            or_formed = False
            or_end_bar = None
            current_participant = 0
            w_state = "PRE_OR"
            wavelength_bias *= 0.5
            sess_fail_count = 0
        
        if not or_formed and i % 96 < 4:
            orh = candle.high if orh is None else max(orh, candle.high)
            orl = candle.low if orl is None else min(orl, candle.low)
        
        if i % 96 == 4 and not or_formed and orh and orl:
            or_formed = True
            or_end_bar = i
            if candle.low < orl:
                current_participant = 1  # BUYER
                defended_liq = orl
            elif candle.high > orh:
                current_participant = -1  # SELLER
                defended_liq = orh
        
        if not or_formed or current_participant == 0:
            continue
        
        result = engine.calculate_emission(
            close=candle.close,
            open_price=candle.open,
            volume=candle.volume,
            atr=atr,
            defended_liq=defended_liq,
            intent_balance=intent_balance,
            current_participant=current_participant,
            close_prev=candles[i-1].close if i > 0 else None
        )
        
        div_result = None
        if i >= 3:
            div_result = engine.calculate_divergence(
                close=candle.close,
                close_3bars_ago=candles[i-3].close,
                atr=atr,
                emission_current=result.emission,
                emission_1bar=engine._emission_history[-2] if len(engine._emission_history) >= 2 else result.emission,
                emission_2bar=engine._emission_history[-3] if len(engine._emission_history) >= 3 else result.emission,
                expected_dir=expected_dir,
                intent_balance=intent_balance
            )
        
        vec_D_agg = (candle.volume / statistics.mean([c.volume for c in candles[max(0,i-20):i+1]]) - 1.0) if candle.volume > 0 else 0
        vec_D_pass = (candle.close - defended_liq) / atr if defended_liq else 0
        
        agg_dir = 1 if vec_D_agg > 0 else -1 if vec_D_agg < 0 else 0
        pass_dir = 1 if vec_D_pass > 0 else -1 if vec_D_pass < 0 else 0
        
        alignment = abs(vec_D_agg) if agg_dir == pass_dir and agg_dir != 0 else 0.0
        conflict = -abs(vec_D_pass) if agg_dir != pass_dir and agg_dir != 0 and pass_dir != 0 else 0.0
        
        intent_balance += (alignment * 0.6 + conflict * 0.4)
        intent_balance *= 0.995
        x = intent_balance / 3.0
        intent_balance = (x / (1.0 + abs(x))) * 3.0
        
        if w_state == "PRE_OR" and or_formed and current_participant != 0:
            w_state = "PART_ID"
        
        if w_state == "PART_ID":
            if (current_participant == 1 and candle.close > defended_liq + atr * 0.5) or \
               (current_participant == -1 and candle.close < defended_liq - atr * 0.5):
                w_state = "MOVE_1"
                move1_bar = i
        
        if w_state == "MOVE_1":
            if (current_participant == 1 and candle.low <= defended_liq + atr * 0.3) or \
               (current_participant == -1 and candle.high >= defended_liq - atr * 0.3):
                w_state = "MOVE_2"
                move2_bar = i
        
        if w_state == "MOVE_2":
            if result.absorption_type != AbsorptionType.NONE or (div_result and div_result.divergence_axes > 0):
                if (current_participant == 1 and candle.close < defended_liq - atr * 0.2) or \
                   (current_participant == -1 and candle.close > defended_liq + atr * 0.2):
                    w_state = "FLIP_CONF"
                    flip_bar = i
                    expected_dir = -current_participant
        
        if w_state == "FLIP_CONF":
            if i - flip_bar > 10:
                w_state = "FAILED"
                sess_fail_count += 1
                wavelength_bias -= 0.15
            else:
                if (expected_dir == 1 and candle.close > candles[move2_bar].close + atr * 0.5) or \
                   (expected_dir == -1 and candle.close < candles[move2_bar].close - atr * 0.5):
                    w_state = "MOVE_3"
        
        conf = 0.5
        if div_result:
            if div_result.is_full_divergence:
                conf += 0.12
            elif div_result.is_partial_divergence:
                conf += 0.06
        
        regime_mult = 1.0
        conf *= regime_mult
        
        intent_mult = 1.15 if abs(intent_balance) > 0.5 else 0.85 if abs(intent_balance) < 0.2 else 1.0
        conf *= intent_mult
        
        if result.absorption_type == AbsorptionType.INTERNAL:
            conf *= 1.10
        elif result.absorption_type == AbsorptionType.EXTERNAL:
            conf *= 1.05
        elif result.absorption_type == AbsorptionType.EXHAUSTION:
            conf *= 0.70
        
        conf *= max(0.7, 1.0 + wavelength_bias)
        conf = min(0.95, max(0.35, conf))
        
        valid = w_state == "MOVE_3"
        high_conf = conf >= confluence_threshold
        abs_trade = result.absorption_type != AbsorptionType.NONE
        within_window = flip_bar is not None and i - flip_bar <= 5
        intent_allows = (expected_dir == 1 and intent_balance > 0.2) or (expected_dir == -1 and intent_balance < -0.2)
        
        if valid and high_conf and abs_trade and within_window and intent_allows and active_signal is None:
            if expected_dir == 1:  # BUY
                entry = candle.close
                stop = min(orl, defended_liq - atr * 0.5) if defended_liq else entry - atr
                rr_mult = 2.5 if abs(intent_balance) > 0.5 else 2.0
                target = entry + (entry - stop) * rr_mult
                
                sig = SignalOutcome(
                    bar_index=i,
                    signal_type="BUY",
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    emission_norm=result.emission_norm,
                    absorption_type=result.absorption_type,
                    divergence_axes=div_result.divergence_axes if div_result else 0,
                    intent_balance=intent_balance,
                    conf_score=conf
                )
                signals.append(sig)
                active_signal = sig
                w_state = "COMPLETE"
                
            elif expected_dir == -1:  # SELL
                entry = candle.close
                stop = max(orh, defended_liq + atr * 0.5) if defended_liq else entry + atr
                rr_mult = 2.5 if abs(intent_balance) > 0.5 else 2.0
                target = entry - (stop - entry) * rr_mult
                
                sig = SignalOutcome(
                    bar_index=i,
                    signal_type="SELL",
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    emission_norm=result.emission_norm,
                    absorption_type=result.absorption_type,
                    divergence_axes=div_result.divergence_axes if div_result else 0,
                    intent_balance=intent_balance,
                    conf_score=conf
                )
                signals.append(sig)
                active_signal = sig
                w_state = "COMPLETE"
        
        if active_signal:
            active_signal.bars_held += 1
            
            if active_signal.signal_type == "BUY":
                if candle.high >= active_signal.target_price:
                    active_signal.outcome = "WIN"
                    active_signal.pnl = active_signal.target_price - active_signal.entry_price
                    active_signal = None
                    expected_dir = 0
                    w_state = "PRE_OR"
                elif candle.low <= active_signal.stop_price:
                    active_signal.outcome = "LOSS"
                    active_signal.pnl = active_signal.stop_price - active_signal.entry_price
                    active_signal = None
                    expected_dir = 0
                    w_state = "PRE_OR"
            else:  # SELL
                if candle.low <= active_signal.target_price:
                    active_signal.outcome = "WIN"
                    active_signal.pnl = active_signal.entry_price - active_signal.target_price
                    active_signal = None
                    expected_dir = 0
                    w_state = "PRE_OR"
                elif candle.high >= active_signal.stop_price:
                    active_signal.outcome = "LOSS"
                    active_signal.pnl = active_signal.entry_price - active_signal.stop_price
                    active_signal = None
                    expected_dir = 0
                    w_state = "PRE_OR"
    
    return signals

def analyze_outcomes(signals: List[SignalOutcome]):
    
    print("=" * 70)
    print("  HORC v4.3 OUTCOME ANALYSIS & CALIBRATION")
    print("=" * 70)
    print()
    
    if not signals:
        print("⚠️  No signals generated - thresholds too strict")
        print()
        print("📊 CALIBRATION RECOMMENDATIONS:")
        print("   - LOWER confluence_threshold: 0.62 → 0.55")
        print("   - LOWER emission thresholds:")
        print("     * EXHAUSTION: 1.8 → 1.5")
        print("     * INTERNAL: 1.4 → 1.2")
        print("     * EXTERNAL: 1.2 → 1.0")
        return
    
    wins = [s for s in signals if s.outcome == "WIN"]
    losses = [s for s in signals if s.outcome == "LOSS"]
    open_trades = [s for s in signals if s.outcome is None]
    
    total = len(signals)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / (win_count + loss_count) * 100) if (win_count + loss_count) > 0 else 0
    
    print(f"📈 SIGNAL SUMMARY:")
    print(f"   Total signals: {total}")
    print(f"   Wins: {win_count}  Losses: {loss_count}  Open: {len(open_trades)}")
    print(f"   Win rate: {win_rate:.1f}%")
    print()
    
    print("💥 EMISSION ANALYSIS:")
    if wins:
        win_emissions = [s.emission_norm for s in wins]
        print(f"   WINNERS - Avg emission: {statistics.mean(win_emissions):.2f}")
        print(f"             Range: {min(win_emissions):.2f} - {max(win_emissions):.2f}")
    
    if losses:
        loss_emissions = [s.emission_norm for s in losses]
        print(f"   LOSERS  - Avg emission: {statistics.mean(loss_emissions):.2f}")
        print(f"             Range: {min(loss_emissions):.2f} - {max(loss_emissions):.2f}")
    print()
    
    print("🌊 ABSORPTION ANALYSIS:")
    for abs_type in [AbsorptionType.INTERNAL, AbsorptionType.EXTERNAL, AbsorptionType.EXHAUSTION]:
        type_signals = [s for s in signals if s.absorption_type == abs_type and s.outcome in ["WIN", "LOSS"]]
        if type_signals:
            type_wins = [s for s in type_signals if s.outcome == "WIN"]
            type_wr = len(type_wins) / len(type_signals) * 100
            print(f"   {abs_type.value:12s}: {len(type_signals):2d} signals, {type_wr:.1f}% win rate")
    print()
    
    print("🎯 CONFIDENCE ANALYSIS:")
    if wins:
        win_confs = [s.conf_score for s in wins]
        print(f"   WINNERS - Avg CPS: {statistics.mean(win_confs)*100:.1f}%")
    if losses:
        loss_confs = [s.conf_score for s in losses]
        print(f"   LOSERS  - Avg CPS: {statistics.mean(loss_confs)*100:.1f}%")
    print()
    
    print("📊 DIVERGENCE ANALYSIS:")
    for axes in [1, 2, 3]:
        axes_signals = [s for s in signals if s.divergence_axes == axes and s.outcome in ["WIN", "LOSS"]]
        if axes_signals:
            axes_wins = [s for s in axes_signals if s.outcome == "WIN"]
            axes_wr = len(axes_wins) / len(axes_signals) * 100
            print(f"   {axes} axes: {len(axes_signals):2d} signals, {axes_wr:.1f}% win rate")
    print()
    
    closed_signals = [s for s in signals if s.outcome in ["WIN", "LOSS"]]
    if closed_signals:
        total_pnl = sum(s.pnl for s in closed_signals)
        avg_win = statistics.mean([s.pnl for s in wins]) if wins else 0
        avg_loss = statistics.mean([s.pnl for s in losses]) if losses else 0
        profit_factor = abs(sum(s.pnl for s in wins) / sum(s.pnl for s in losses)) if losses and wins else 0
        
        print("💰 PNL ANALYSIS:")
        print(f"   Total PnL: {total_pnl:+.4f}")
        print(f"   Avg Win: {avg_win:.4f}  Avg Loss: {avg_loss:.4f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print()
    
    print("=" * 70)
    print("  📊 CALIBRATION RECOMMENDATIONS FOR PINE SCRIPT")
    print("=" * 70)
    print()
    
    if win_rate >= 60:
        print("✅ PERFORMANCE: GOOD")
        print("   System is well-calibrated")
        print()
    elif 50 <= win_rate < 60:
        print("⚠️  PERFORMANCE: ACCEPTABLE")
        print("   Minor tuning recommended")
        print()
    else:
        print("❌ PERFORMANCE: NEEDS IMPROVEMENT")
        print("   Significant recalibration needed")
        print()
    
    if total < 10:
        print("🔧 SIGNAL FREQUENCY:")
        print("   Too few signals - LOOSEN thresholds:")
        print("   - confluence_threshold: 0.62 → 0.58")
        print("   - EXHAUSTION: 1.8 → 1.6")
        print("   - INTERNAL: 1.4 → 1.3")
        print()
    
    if win_rate < 50:
        print("🔧 WIN RATE:")
        print("   Below breakeven - TIGHTEN filters:")
        if losses:
            bad_abs = max([(abs_type, len([s for s in losses if s.absorption_type == abs_type])) 
                          for abs_type in [AbsorptionType.INTERNAL, AbsorptionType.EXTERNAL, AbsorptionType.EXHAUSTION]],
                         key=lambda x: x[1])
            print(f"   - Most losses from {bad_abs[0].value} absorption - reduce weight")
        print("   - RAISE confluence_threshold: 0.62 → 0.68")
        print()
    
    if wins and losses:
        optimal_emission = statistics.mean([s.emission_norm for s in wins])
        print("🔧 OPTIMAL EMISSION THRESHOLDS:")
        print(f"   Winners cluster around: {optimal_emission:.2f}")
        print(f"   Suggested EXHAUSTION threshold: {optimal_emission * 1.3:.2f}")
        print(f"   Suggested INTERNAL threshold: {optimal_emission * 1.0:.2f}")
        print(f"   Suggested EXTERNAL threshold: {optimal_emission * 0.8:.2f}")
        print()
    
    print("=" * 70)

def main():
    print("🔧 Generating test data...")
    candles = generate_synthetic_data(days=30, timeframe_minutes=15, volatility=0.0008)
    print(f"   Generated {len(candles)} candles")
    print()
    
    print("🚀 Running simulation...")
    engine = EmissionEngine(lookback=20)
    signals = simulate_horc_signals(candles, engine, confluence_threshold=0.55)
    print(f"   Generated {len(signals)} signals")
    print()
    
    analyze_outcomes(signals)

if __name__ == "__main__":
    main()
