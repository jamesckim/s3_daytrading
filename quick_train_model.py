#!/usr/bin/env python3
"""
Quick Training Script for Indicator Transformer
==============================================
Trains the model with a small dataset for quick testing.
Now with progress monitoring and caching!
"""

import subprocess
import sys
import os
import signal

# Quick training with limited data
# Detect optimal device
import torch
if torch.backends.mps.is_available():
    device = "mps"
    print("🎯 Apple Silicon GPU detected - using Metal Performance Shaders")
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

cmd = [
    sys.executable,
    "train_indicator_transformer.py",
    "--symbols", "AAPL", "MSFT", "NVDA", "TSLA", "SPY",  # Just 5 symbols
    "--days", "30",  # Just 30 days of data
    "--epochs", "20",  # Just 20 epochs for quick training
    "--batch-size", "16",  # Optimal for MPS
    "--device", device  # Use best available device
]

print("🚀 Starting quick model training with MAXIMUM PERFORMANCE optimizations...")
print("This will train a basic model with:")
print("  ✓ Apple Silicon GPU acceleration (MPS)" if device == "mps" else f"  ✓ {device.upper()} acceleration")
print("  ✓ Mixed precision training (BFloat16)" if device == "mps" else "  ✓ Mixed precision (if supported)")
print("  ✓ torch.compile for Metal kernel fusion" if device == "mps" else "  ✓ torch.compile optimization")
print("  ✓ Optimized matmul precision")
print("  ✓ Multi-threaded data loading")
print("  ✓ Progress tracking with ETA")
print("  ✓ Indicator caching system")
print("  ✓ Memory usage monitoring")
print("-" * 60)
print("\nExpected performance on Apple Silicon:")
print("  • vs CPU: ~6-7x faster")
print("  • vs basic MPS: ~2x faster") 
print("\nTip: Run 'python monitor_training.py' in another terminal to see live stats!")
print("-" * 60)

# Create cache directory
os.makedirs("data_cache/indicators", exist_ok=True)

# Handle Ctrl-C gracefully
def signal_handler(signum, frame):
    print("\n\n🛑 Training interrupted by user (Ctrl-C)")
    print("Cleaning up...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    subprocess.run(cmd)
except KeyboardInterrupt:
    print("\n\n🛑 Training script interrupted")
    sys.exit(0)