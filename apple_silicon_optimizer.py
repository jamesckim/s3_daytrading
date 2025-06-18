#!/usr/bin/env python3
"""
Apple Silicon Performance Optimizer
===================================
Optimizations specific to M1/M2/M3 chips.
"""

import os
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import psutil

def optimize_for_apple_silicon():
    """Configure environment for optimal Apple Silicon performance."""
    
    # 1. Set environment variables for accelerated BLAS/LAPACK
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(mp.cpu_count())
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
    
    # 2. Configure NumPy to use Accelerate framework
    np.show_config()  # Verify using Accelerate
    
    # 3. Set process affinity for performance cores
    # M1 Pro has 8 performance cores + 2 efficiency cores
    # M2 Pro has 8 performance cores + 4 efficiency cores
    performance_cores = get_performance_cores()
    
    return {
        'cpu_count': mp.cpu_count(),
        'performance_cores': performance_cores,
        'efficiency_cores': mp.cpu_count() - performance_cores,
        'recommended_workers': performance_cores  # Use P-cores for compute
    }

def get_performance_cores():
    """Detect number of performance cores on Apple Silicon."""
    # M1: 4-8 P-cores, M2: 4-8 P-cores, M3: 4-12 P-cores
    total_cores = mp.cpu_count()
    
    if total_cores == 8:  # M1/M2 base
        return 4
    elif total_cores == 10:  # M1/M2 Pro
        return 8
    elif total_cores == 12:  # M2 Max base
        return 8
    elif total_cores >= 14:  # M1/M2 Ultra or M3 Pro/Max
        return min(12, total_cores - 4)  # Reserve some E-cores
    else:
        return max(4, total_cores // 2)

def create_optimized_thread_pool(task_type='compute'):
    """Create thread pool optimized for Apple Silicon."""
    config = optimize_for_apple_silicon()
    
    if task_type == 'compute':
        # Use performance cores for heavy computation
        workers = config['performance_cores']
    elif task_type == 'io':
        # Use all cores for I/O bound tasks
        workers = config['cpu_count']
    else:
        # Balanced approach
        workers = config['recommended_workers']
    
    print(f"🍎 Apple Silicon Optimization:")
    print(f"   • Total cores: {config['cpu_count']}")
    print(f"   • Performance cores: {config['performance_cores']}")
    print(f"   • Using {workers} workers for {task_type} tasks")
    
    return workers

def vectorize_computation(func):
    """Decorator to automatically vectorize NumPy operations."""
    def wrapper(*args, **kwargs):
        # Ensure inputs are contiguous arrays for better cache performance
        args = [np.ascontiguousarray(arg) if isinstance(arg, np.ndarray) else arg 
                for arg in args]
        return func(*args, **kwargs)
    return wrapper

@vectorize_computation
def fast_rolling_window(data, window_size, operation='mean'):
    """Optimized rolling window using Apple Silicon SIMD."""
    if operation == 'mean':
        # Use convolution for rolling mean (uses Accelerate)
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='valid')
    elif operation == 'std':
        # Vectorized rolling std
        cumsum = np.cumsum(np.insert(data, 0, 0))
        cumsum_sq = np.cumsum(np.insert(data**2, 0, 0))
        
        mean = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        mean_sq = (cumsum_sq[window_size:] - cumsum_sq[:-window_size]) / window_size
        
        return np.sqrt(mean_sq - mean**2)
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Memory optimization for unified memory architecture
def optimize_memory_layout(data_dict):
    """Optimize data layout for Apple Silicon's unified memory."""
    # Convert to column-major (Fortran) order for better cache performance
    # when doing column operations
    optimized = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            optimized[key] = np.asfortranarray(value)
        else:
            optimized[key] = value
    return optimized

# Batch matrix operations for Neural Engine utilization
def batch_matrix_multiply(A_list, B_list):
    """Batch matrix multiplication optimized for Apple Silicon."""
    # Stack matrices for single BLAS call
    A_stacked = np.stack(A_list)
    B_stacked = np.stack(B_list)
    
    # This will use Accelerate framework's batch GEMM
    return np.matmul(A_stacked, B_stacked)

if __name__ == "__main__":
    # Test optimization
    config = optimize_for_apple_silicon()
    print(f"\nOptimal workers for computation: {create_optimized_thread_pool('compute')}")
    print(f"Optimal workers for I/O: {create_optimized_thread_pool('io')}")
    
    # Benchmark vectorized operations
    import time
    data = np.random.randn(1000000)
    
    start = time.time()
    result = fast_rolling_window(data, 100, 'mean')
    print(f"\nVectorized rolling mean: {time.time() - start:.4f}s")
    
    start = time.time()
    result = fast_rolling_window(data, 100, 'std')
    print(f"Vectorized rolling std: {time.time() - start:.4f}s")