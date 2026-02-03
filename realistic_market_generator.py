import random
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
import pandas as pd

from src.engines import Candle

class RealisticMarketSimulator:
    def __init__(
        self,
        base_price: float = 1.1000,
        volatility: float = 0.0005,
        seed: int = 42
    ):
        self.base_price = base_price
        self.volatility = volatility
        self.current_price = base_price
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.session_high = None
        self.session_low = None
        self.last_liquidity_sweep = None
        self.wavelength_move = 0  # Track which move we're in
        self.regime = "RANGING"  # TRENDING_UP, TRENDING_DOWN, RANGING
        
    def generate_bars(
        self,
        num_bars: int,
        timeframe_minutes: int = 15
    ) -> List[Candle]:
        candles = []
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        
        self.session_high = self.current_price + (self.volatility * 20)
        self.session_low = self.current_price - (self.volatility * 20)
        
        bars_since_sweep = 0
        
        for i in range(num_bars):
            ts = start + timedelta(minutes=i * timeframe_minutes)
            
            if i % 96 == 0 and i > 0:
                self._start_new_session()
            
            if i % 500 == 0:
                self._change_regime()
            
            candle = self._generate_structured_bar(ts, bars_since_sweep)
            candles.append(candle)
            
            if candle.high > self.session_high:
                self.last_liquidity_sweep = "HIGH"
                self.session_high = candle.high
                bars_since_sweep = 0
                self.wavelength_move = 1  # Start move 1
            elif candle.low < self.session_low:
                self.last_liquidity_sweep = "LOW"
                self.session_low = candle.low
                bars_since_sweep = 0
                self.wavelength_move = 1
            else:
                bars_since_sweep += 1
            
            if self.wavelength_move > 0:
                if bars_since_sweep % 8 == 0:  # Move completes every ~8 bars
                    self.wavelength_move += 1
                if self.wavelength_move > 3:
                    self.wavelength_move = 0  # Reset after move 3
            
            self.current_price = candle.close
        
        return candles
    
    def _generate_structured_bar(
        self,
        timestamp: datetime,
        bars_since_sweep: int
    ) -> Candle:
        o = self.current_price
        
        if self.regime == "TRENDING_UP":
            trend_bias = self.volatility * 0.5
        elif self.regime == "TRENDING_DOWN":
            trend_bias = -self.volatility * 0.5
        else:
            trend_bias = 0
        
        noise = random.gauss(0, self.volatility)
        
        mean_reversion = (self.base_price - self.current_price) * 0.005
        
        wavelength_bias = 0
        if self.wavelength_move == 1:
            wavelength_bias = self.volatility * 1.5 * (1 if self.last_liquidity_sweep == "HIGH" else -1)
        elif self.wavelength_move == 2:
            wavelength_bias = -self.volatility * 0.8 * (1 if self.last_liquidity_sweep == "HIGH" else -1)
        elif self.wavelength_move == 3:
            wavelength_bias = self.volatility * 1.2 * (1 if self.last_liquidity_sweep == "HIGH" else -1)
        
        change = trend_bias + noise + mean_reversion + wavelength_bias
        c = o + change
        
        range_mult = 1.0
        
        if bars_since_sweep < 3:
            range_mult = 1.5  # Bigger wicks during sweep
        
        is_absorption = random.random() < 0.05  # 5% of bars show absorption
        if is_absorption:
            body = (c - o)
            c = o + (body * 0.3)  # Compress the body
            range_mult = 2.0  # But large wicks
        
        range_ext = abs(change) * range_mult + random.uniform(0, self.volatility * 0.5)
        
        if change >= 0:
            h = max(o, c) + random.uniform(0, range_ext)
            l = min(o, c) - random.uniform(0, range_ext * 0.5)
        else:
            h = max(o, c) + random.uniform(0, range_ext * 0.5)
            l = min(o, c) - random.uniform(0, range_ext)
        
        base_volume = 100000
        
        if bars_since_sweep < 2:
            volume_mult = 2.5
        elif bars_since_sweep < 5:
            volume_mult = 1.5
        else:
            volume_mult = 1.0
        
        if is_absorption:
            volume_mult *= 2.0
        
        volume = base_volume * volume_mult * random.uniform(0.8, 1.2)
        
        return Candle(
            timestamp=timestamp,
            open=round(o, 5),
            high=round(h, 5),
            low=round(l, 5),
            close=round(c, 5),
            volume=round(volume, 0)
        )
    
    def _start_new_session(self):
        gap = random.gauss(0, self.volatility * 2)
        self.current_price += gap
        
        self.session_high = self.current_price + (self.volatility * 20)
        self.session_low = self.current_price - (self.volatility * 20)
        self.wavelength_move = 0
    
    def _change_regime(self):
        regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING"]
        self.regime = random.choice(regimes)
        print(f"   Regime change → {self.regime}")

def generate_realistic_synthetic_data(
    symbol: str = "EURUSD",
    days: int = 30,
    timeframe_minutes: int = 15,
    base_price: float = 1.1000,
    volatility: float = 0.0005,
    seed: int = 42
) -> List[Candle]:
    simulator = RealisticMarketSimulator(
        base_price=base_price,
        volatility=volatility,
        seed=seed
    )
    
    bars_per_day = (24 * 60) // timeframe_minutes
    total_bars = days * bars_per_day
    
    print(f"   Generating {total_bars} bars with realistic market structure...")
    print(f"   - Liquidity sweeps and session levels")
    print(f"   - 3-move wavelength structures")
    print(f"   - Volume absorption patterns")
    print(f"   - Regime changes (trending/ranging)")
    
    candles = simulator.generate_bars(total_bars, timeframe_minutes)
    
    return candles

if __name__ == "__main__":
    print("Testing realistic synthetic data generator...")
    print()
    
    candles = generate_realistic_synthetic_data(
        symbol="EURUSD",
        days=30,
        timeframe_minutes=15,
        base_price=1.1000
    )
    
    print()
    print(f"Generated {len(candles)} candles")
    print(f"First: {candles[0].timestamp} O={candles[0].open:.5f} V={candles[0].volume:.0f}")
    print(f"Last:  {candles[-1].timestamp} C={candles[-1].close:.5f}")
    
    prices = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    
    print()
    print(f"Price range: {min(prices):.5f} - {max(prices):.5f}")
    print(f"Volume range: {min(volumes):.0f} - {max(volumes):.0f}")
    print(f"Avg volume: {np.mean(volumes):.0f}")
