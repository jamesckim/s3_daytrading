#!/usr/bin/env python3
"""
Test GPU Speedup
================
Quick test to show the performance improvement.
"""

import time
import numpy as np
import pandas as pd
from mlx_indicator_library import MLXIndicatorLibrary, GPUData
import mlx.core as mx

# Force GPU
mx.set_default_device(mx.gpu)

def create_test_data(n_bars=2600):
    """Create test price data."""
    return pd.DataFrame({
        'open': np.random.randn(n_bars).cumsum() + 100,
        'high': np.random.randn(n_bars).cumsum() + 101,
        'low': np.random.randn(n_bars).cumsum() + 99,
        'close': np.random.randn(n_bars).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, n_bars).astype(float)
    })

def test_cpu_computation(df):
    """Simulate CPU-based indicator computation."""
    start = time.time()
    
    # Simple indicators (CPU)
    sma_7 = df['close'].rolling(7).mean()
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    
    # RSI (CPU)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # VWAP (CPU)
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    cpu_time = time.time() - start
    return cpu_time

def test_gpu_computation(df):
    """GPU-based indicator computation."""
    start = time.time()
    
    # Convert to GPU
    gpu_data = GPUData(
        open=mx.array(df['open'].values, dtype=mx.float32),
        high=mx.array(df['high'].values, dtype=mx.float32),
        low=mx.array(df['low'].values, dtype=mx.float32),
        close=mx.array(df['close'].values, dtype=mx.float32),
        volume=mx.array(df['volume'].values, dtype=mx.float32),
        timestamps=mx.array(np.arange(len(df)), dtype=mx.int32)
    )
    
    # Compute all indicators on GPU
    mlx_lib = MLXIndicatorLibrary()
    indicators = mlx_lib.compute_all_indicators_gpu(gpu_data)
    
    # Force evaluation
    mx.eval(indicators)
    
    gpu_time = time.time() - start
    return gpu_time

def main():
    """Run speedup test."""
    print("🚀 Apple Silicon GPU Speedup Test")
    print("="*50)
    
    # Test different data sizes
    sizes = [1000, 2600, 5000, 10000]
    
    for n_bars in sizes:
        print(f"\n📊 Testing with {n_bars} bars:")
        
        # Create test data
        df = create_test_data(n_bars)
        
        # Test CPU
        cpu_time = test_cpu_computation(df)
        print(f"   CPU time: {cpu_time:.3f}s")
        
        # Test GPU
        gpu_time = test_gpu_computation(df)
        print(f"   GPU time: {gpu_time:.3f}s")
        
        # Speedup
        speedup = cpu_time / gpu_time
        print(f"   Speedup: {speedup:.1f}x")
    
    print("\n✅ Test complete!")
    print("\nNote: First GPU run includes initialization overhead.")
    print("Subsequent runs will be even faster.")

if __name__ == "__main__":
    main()