#!/usr/bin/env python3
"""
Simple GPU Test
===============
Test basic GPU functionality.
"""

import time
import numpy as np
import mlx.core as mx

# Force GPU
mx.set_default_device(mx.gpu)

def test_basic_operations():
    """Test basic MLX operations."""
    print("🧪 Testing basic MLX operations...")
    
    # Create test data
    n = 1000
    data = np.random.randn(n).cumsum() + 100
    
    # Transfer to GPU
    start = time.time()
    gpu_data = mx.array(data, dtype=mx.float32)
    mx.eval(gpu_data)  # Force evaluation
    transfer_time = time.time() - start
    print(f"✅ Data transfer to GPU: {transfer_time:.3f}s")
    
    # Simple operations
    start = time.time()
    
    # SMA calculation
    window = 20
    cumsum = mx.cumsum(mx.pad(gpu_data, (1, 0), constant_values=0))
    sma = (cumsum[window:] - cumsum[:-window]) / window
    
    # Force evaluation
    mx.eval(sma)
    
    compute_time = time.time() - start
    print(f"✅ SMA computation on GPU: {compute_time:.3f}s")
    
    # Multiple operations
    start = time.time()
    
    # Various computations
    ema_alpha = 0.1
    ema = mx.zeros_like(gpu_data)
    ema[0] = gpu_data[0]
    
    for i in range(1, len(gpu_data)):
        ema[i] = ema_alpha * gpu_data[i] + (1 - ema_alpha) * ema[i-1]
    
    # Stats
    mean = mx.mean(gpu_data)
    std = mx.std(gpu_data)
    max_val = mx.max(gpu_data)
    min_val = mx.min(gpu_data)
    
    # Force evaluation
    mx.eval([ema, mean, std, max_val, min_val])
    
    multi_time = time.time() - start
    print(f"✅ Multiple operations on GPU: {multi_time:.3f}s")
    
    print(f"\n📊 Results:")
    print(f"   Mean: {mean.item():.2f}")
    print(f"   Std: {std.item():.2f}")
    print(f"   Max: {max_val.item():.2f}")
    print(f"   Min: {min_val.item():.2f}")
    
    return True

def test_memory_limits():
    """Test memory limits."""
    print("\n🧪 Testing memory limits...")
    
    sizes = [1000, 5000, 10000, 50000]
    
    for size in sizes:
        try:
            # Create large array
            data = mx.random.normal((size, 100))
            mx.eval(data)
            
            # Compute
            result = mx.mean(data, axis=1)
            mx.eval(result)
            
            print(f"✅ Size {size}x100: OK")
            
        except Exception as e:
            print(f"❌ Size {size}x100: {str(e)}")
            break
    
    return True

def main():
    """Run tests."""
    print("🚀 Simple GPU Test")
    print("="*50)
    
    # Test basic operations
    test_basic_operations()
    
    # Test memory limits
    test_memory_limits()
    
    print("\n✅ Tests complete!")

if __name__ == "__main__":
    main()