import pandas as pd
from datetime import datetime

def generate_report(trade_log, performance_metrics, filename=None):
    if filename is None:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("--- Günlük/Haftalık Performans Raporu ---\n\n")
        f.write("Portföy Değeri:\n")
        f.write(str(trade_log['portfolio_value'].describe()) + "\n\n")
        f.write("Performans Metrikleri:\n")
        for k, v in performance_metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Rapor kaydedildi: {filename}") 