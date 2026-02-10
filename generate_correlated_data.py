import csv
import random
import math
from datetime import datetime, timedelta
from pathlib import Path

def generate_correlated_data(days=60):
    start_dt = datetime(2025, 1, 1, 0, 0)
    
    # Base parameters
    eur_price = 1.0800
    gbp_price = 1.2700
    
    # Shared drift component (Market Factor)
    drift = 0.0
    
    eur_bars = []
    gbp_bars = []
    
    current_dt = start_dt
    
    print(f"Generating {days} days of correlated data...")
    
    for day in range(days):
        # Daily volatility regime
        daily_vol = random.uniform(0.0020, 0.0080)
        
        for minute in range(0, 1440, 5): # 5-minute bars
            # Time of day effect (London/NY overlap volatility)
            hour = current_dt.hour
            tod_mult = 1.0
            if 8 <= hour <= 16: # London/NY
                tod_mult = 2.0
            elif 22 <= hour <= 2: # Asia low vol
                tod_mult = 0.5
                
            # Shared shock
            shock = random.gauss(0, 1) * (daily_vol / math.sqrt(288)) * tod_mult
            
            # Idiosyncratic shocks
            eur_shock = random.gauss(0, 1) * (daily_vol / math.sqrt(288)) * 0.3
            gbp_shock = random.gauss(0, 1) * (daily_vol / math.sqrt(288)) * 0.4 # GBP slightly more volatile
            
            # Update Prices
            eur_ret = shock * 0.8 + eur_shock # 0.8 correlation factor roughly
            gbp_ret = shock * 0.9 + gbp_shock
            
            eur_open = eur_price
            gbp_open = gbp_price
            
            eur_price *= (1 + eur_ret)
            gbp_price *= (1 + gbp_ret)
            
            # Generate OHLC
            # Intra-bar volatility
            eur_high = max(eur_open, eur_price) * (1 + random.uniform(0, 0.0002 * tod_mult))
            eur_low = min(eur_open, eur_price) * (1 - random.uniform(0, 0.0002 * tod_mult))
            
            gbp_high = max(gbp_open, gbp_price) * (1 + random.uniform(0, 0.0003 * tod_mult))
            gbp_low = min(gbp_open, gbp_price) * (1 - random.uniform(0, 0.0003 * tod_mult))
            
            # Volume
            vol = int(random.lognormvariate(8, 1) * tod_mult)
            
            # Append Row: TICKER,DTYYYYMMDD,TIME,OPEN,HIGH,LOW,CLOSE,VOL
            dt_str = current_dt.strftime("%Y%m%d")
            time_str = current_dt.strftime("%H%M%S")
            
            eur_bars.append(["EURUSD", dt_str, time_str, 
                             f"{eur_open:.5f}", f"{eur_high:.5f}", f"{eur_low:.5f}", f"{eur_price:.5f}", vol])
            
            gbp_bars.append(["GBPUSD", dt_str, time_str,
                             f"{gbp_open:.5f}", f"{gbp_high:.5f}", f"{gbp_low:.5f}", f"{gbp_price:.5f}", vol])
            
            current_dt += timedelta(minutes=5)
            
    # Save to files
    data_dir = Path("src/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    header = ["TICKER","DTYYYYMMDD","TIME","OPEN","HIGH","LOW","CLOSE","VOL"]
    
    with open(data_dir / "EURUSD.txt", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(eur_bars)
    
    with open(data_dir / "GBPUSD.txt", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(gbp_bars)
        
    print(f"Saved {len(eur_bars)} bars each to src/data/EURUSD.txt and src/data/GBPUSD.txt")

if __name__ == "__main__":
    generate_correlated_data()
