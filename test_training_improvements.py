#!/usr/bin/env python3
"""
Test Training Improvements
==========================
Quick test to verify the new progress monitoring and caching work correctly.
"""

import subprocess
import sys

# Quick test with minimal data
cmd = [
    sys.executable,
    "train_indicator_transformer.py",
    "--symbols", "AAPL", "MSFT",  # Just 2 symbols
    "--days", "7",  # Just 7 days of data
    "--epochs", "2",  # Just 2 epochs
    "--batch-size", "8",
    "--device", "cpu"  # Use CPU for compatibility
]

print("🧪 Testing training improvements...")
print("This will train a minimal model to test the new features")
print("-" * 60)

subprocess.run(cmd)