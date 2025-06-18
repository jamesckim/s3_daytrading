#!/usr/bin/env python3
"""
Train MLX on James's Curated Tickers
====================================
Focus on the 32 high-quality tickers for optimal training.
"""

import json
import subprocess
import sys

# Use the all tickers training script with James's tickers
with open('../tickers/james_tickers.json', 'r') as f:
    james_tickers = json.load(f)

# Write temp file with these tickers
with open('../tickers/temp_training_tickers.json', 'w') as f:
    json.dump(james_tickers, f)

print(f"🎯 Training on {len(james_tickers)} curated tickers:")
print(f"   {', '.join(james_tickers[:10])}...")
print(f"\nThese are high-volume, liquid stocks perfect for intraday trading.\n")

# Run the training with appropriate parameters
cmd = [
    sys.executable,
    'train_mlx_all_tickers.py',
    '--days', '15',           # 15 days of minute data
    '--epochs', '20',         # More epochs for better training
    '--batch-size', '32',
    '--sample-interval', '20' # Every 20 minutes for more samples
]

# Replace the ticker file temporarily
import shutil
shutil.copy('../tickers/all_tickers.json', '../tickers/all_tickers_backup.json')
shutil.copy('../tickers/temp_training_tickers.json', '../tickers/all_tickers.json')

try:
    subprocess.run(cmd, check=True)
finally:
    # Restore original
    shutil.copy('../tickers/all_tickers_backup.json', '../tickers/all_tickers.json')
    import os
    os.remove('../tickers/all_tickers_backup.json')
    os.remove('../tickers/temp_training_tickers.json')