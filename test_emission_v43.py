from src.data.historical_loader import generate_synthetic_data
from src.core.emission_engine import EmissionEngine, AbsorptionType
from src.engines import Candle
import statistics

def test_emission_engine():
    print("=" * 70)
    print("  HORC v4.3 EMISSION ENGINE TEST")
    print("=" * 70)
    print()
    
    print("📊 Generating synthetic data...")
    candles = generate_synthetic_data(days=5, timeframe_minutes=15, volatility=0.02)
    print(f"   Generated {len(candles)} 15-minute candles")
    print(f"   Date range: {candles[0].timestamp} to {candles[-1].timestamp}")
    print()
    
    engine = EmissionEngine(lookback=20)
    
    emissions = []
    absorptions = {
        AbsorptionType.NONE: 0,
        AbsorptionType.INTERNAL: 0,
        AbsorptionType.EXTERNAL: 0,
        AbsorptionType.EXHAUSTION: 0
    }
    divergences_full = 0
    divergences_partial = 0
    
    or_high = max(c.high for c in candles[:4])
    or_low = min(c.low for c in candles[:4])
    defended_liq = or_low  # Assume buyer participant
    current_participant = 1  # BUYER
    
    print("🔄 Processing bars...")
    print()
    
    intent_balance = 0.0
    
    for i, candle in enumerate(candles):
        if i < 3:
            continue  # Need history for divergence
        
        recent_ranges = [c.high - c.low for c in candles[max(0, i-14):i+1]]
        atr = statistics.mean(recent_ranges) if recent_ranges else 0.01
        
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
        
        emissions.append(result.emission_norm)
        absorptions[result.absorption_type] += 1
        
        if i >= 3:
            div_result = engine.calculate_divergence(
                close=candle.close,
                close_3bars_ago=candles[i-3].close,
                atr=atr,
                emission_current=result.emission,
                emission_1bar=engine._emission_history[-2] if len(engine._emission_history) >= 2 else result.emission,
                emission_2bar=engine._emission_history[-3] if len(engine._emission_history) >= 3 else result.emission,
                expected_dir=current_participant,
                intent_balance=intent_balance
            )
            
            if div_result.is_full_divergence:
                divergences_full += 1
            elif div_result.is_partial_divergence:
                divergences_partial += 1
        
        intent_balance = intent_balance * 0.995 + (0.1 if candle.close > candle.open else -0.1)
        intent_balance = max(-3, min(3, intent_balance))
    
    print("=" * 70)
    print("  EMISSION ANALYSIS RESULTS")
    print("=" * 70)
    print()
    
    print("📊 EMISSION STATISTICS:")
    print(f"   Average normalized emission: {statistics.mean(emissions):.2f}")
    print(f"   Max normalized emission: {max(emissions):.2f}")
    print(f"   Min normalized emission: {min(emissions):.2f}")
    print(f"   Std deviation: {statistics.stdev(emissions):.2f}")
    print()
    
    print("💥 ABSORPTION DETECTION:")
    total_abs = sum(absorptions.values())
    for abs_type, count in absorptions.items():
        pct = (count / total_abs * 100) if total_abs > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {abs_type.value:12s}: {count:3d} ({pct:5.1f}%) {bar}")
    print()
    
    print("🌊 DIVERGENCE DETECTION:")
    print(f"   Full divergence (2+ axes): {divergences_full}")
    print(f"   Partial divergence (1 axis): {divergences_partial}")
    print(f"   Divergence rate: {(divergences_full + divergences_partial) / len(candles) * 100:.1f}%")
    print()
    
    print("✅ VALIDATION CHECKS:")
    high_emission_bars = sum(1 for e in emissions if e > 1.8)
    print(f"   Bars above exhaustion threshold (1.8): {high_emission_bars} ({high_emission_bars/len(emissions)*100:.1f}%)")
    
    moderate_emission = sum(1 for e in emissions if 1.2 <= e <= 1.8)
    print(f"   Bars in absorption range (1.2-1.8): {moderate_emission} ({moderate_emission/len(emissions)*100:.1f}%)")
    
    low_emission = sum(1 for e in emissions if e < 0.8)
    print(f"   Bars below efficiency threshold (0.8): {low_emission} ({low_emission/len(emissions)*100:.1f}%)")
    print()
    
    print("=" * 70)
    print("✅ Emission engine test complete!")
    print("=" * 70)
    print()
    print("🎯 NEXT STEPS:")
    print("   1. Test with real CSV data: python replay_historical.py --file data/YOUR_FILE.csv")
    print("   2. Validate emission thresholds match market behavior")
    print("   3. Compare with Pine Script v4.3 on TradingView")
    print()

if __name__ == "__main__":
    test_emission_engine()
