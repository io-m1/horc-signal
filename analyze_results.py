import re
from pathlib import Path

def parse_report(filepath):
    metrics = {
        "Win Rate": "N/A",
        "Total PnL": "N/A",
        "Profit Factor": "N/A",
        "Avg R-Multiple": "N/A",
        "Max Drawdown": "N/A",
        "Trades": "0"
    }
    
    if not filepath.exists():
        return metrics

    try:
        # PowerShell > redirection often produces UTF-16LE with BOM
        content = filepath.read_text(encoding='utf-16le')
    except:
        try:
            content = filepath.read_text(encoding='utf-8')
        except:
            return metrics

    # Regex patterns
    patterns = {
        "Win Rate": r"Win Rate:\s+([\d\.]+)%",
        "Total PnL": r"Total PnL:\s+([+\-\d\.]+) pips",
        "Profit Factor": r"Profit Factor:\s+([\d\.]+)",
        "Avg R-Multiple": r"Avg R-Multiple:\s+([+\-\d\.]+)R",
        "Max Drawdown": r"Max Drawdown:\s+([\d\.]+) pips",
        "Trades": r"Trades:\s+(\d+)"
    }

    for key, pat in patterns.items():
        match = re.search(pat, content)
        if match:
            metrics[key] = match.group(1)
            
    return metrics

def main():
    pairs = ["EURUSD", "GBPUSD"]
    versions = ["v87", "v88"]
    
    print(f"{'PAIR':<8} {'VER':<5} | {'TRADES':<6} | {'WIN%':<6} | {'PF':<5} | {'PNL':<8} | {'MAX DD':<8}")
    print("-" * 65)
    
    for pair in pairs:
        for ver in versions:
            fname = f"report_{pair}_{ver}.txt"
            p = Path(fname)
            m = parse_report(p)
            
            print(f"{pair:<8} {ver:<5} | {m['Trades']:<6} | {m['Win Rate']:<6} | {m['Profit Factor']:<5} | {m['Total PnL']:<8} | {m['Max Drawdown']:<8}")
            
if __name__ == "__main__":
    main()
