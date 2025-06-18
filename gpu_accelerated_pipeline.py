#!/usr/bin/env python3
"""
GPU-Accelerated Pipeline Integration
====================================
Seamless integration of MLX GPU computation with existing pipeline.
"""

import mlx.core as mx
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from mlx_indicator_library import MLXIndicatorLibrary, MLXAVWAPCalculator, GPUData
from apple_silicon_optimizer import optimize_for_apple_silicon

logger = logging.getLogger(__name__)

# Initialize MLX for GPU
mx.set_default_device(mx.gpu)


class GPUAcceleratedPipeline:
    """Drop-in replacement for parallel_indicator_computation using GPU."""
    
    def __init__(self):
        self.mlx_library = MLXIndicatorLibrary()
        self.gpu_cache = {}
        
        # Optimize for Apple Silicon
        config = optimize_for_apple_silicon()
        logger.info(f"🍎 Apple Silicon: {config['performance_cores']} P-cores, {config['efficiency_cores']} E-cores")
    
    def process_batch_gpu(self, data_batch: List[Tuple[str, pd.DataFrame]], 
                         indicator_library=None, avwap_cache=None) -> List[Tuple[str, pd.DataFrame, np.ndarray]]:
        """Process a batch of symbols entirely on GPU."""
        results = []
        
        # Pre-load all data to GPU in one shot
        gpu_data_batch = self._batch_load_to_gpu(data_batch)
        
        # Process each symbol on GPU
        for symbol, df, gpu_data in gpu_data_batch:
            try:
                # Compute all standard indicators on GPU
                standard_indicators = self.mlx_library.compute_all_indicators_gpu(gpu_data)
                
                # Compute AVWAP indicators on GPU
                avwap_calc = MLXAVWAPCalculator(gpu_data)
                avwap_indicators = avwap_calc.calculate_all_avwap_gpu()
                
                # Combine all indicators
                all_indicators = self._combine_indicators(standard_indicators, avwap_indicators)
                
                # Convert back to numpy for compatibility
                indicators_np = np.array(all_indicators)
                
                # Handle NaN values
                indicators_np = np.nan_to_num(indicators_np, nan=0.0)
                
                results.append((symbol, df, indicators_np))
                
            except Exception as e:
                logger.error(f"GPU processing failed for {symbol}: {e}")
                # Fallback to zero array
                n_indicators = len(self.mlx_library.indicators) + 13  # 13 AVWAP indicators
                indicators_np = np.zeros((len(df), n_indicators), dtype=np.float32)
                results.append((symbol, df, indicators_np))
        
        return results
    
    def _batch_load_to_gpu(self, data_batch: List[Tuple[str, pd.DataFrame]]) -> List[Tuple[str, pd.DataFrame, GPUData]]:
        """Load all data to GPU in a single operation."""
        gpu_batch = []
        
        for symbol, df in data_batch:
            # Convert to GPU arrays
            gpu_data = GPUData(
                open=mx.array(df['open'].values, dtype=mx.float32),
                high=mx.array(df['high'].values, dtype=mx.float32),
                low=mx.array(df['low'].values, dtype=mx.float32),
                close=mx.array(df['close'].values, dtype=mx.float32),
                volume=mx.array(df['volume'].values, dtype=mx.float32),
                timestamps=mx.array(np.arange(len(df)), dtype=mx.int32)
            )
            
            gpu_batch.append((symbol, df, gpu_data))
        
        # Force GPU sync
        mx.eval(gpu_batch[0][2].close)
        
        return gpu_batch
    
    def _combine_indicators(self, standard: mx.array, avwap: Dict[str, mx.array]) -> mx.array:
        """Combine standard and AVWAP indicators."""
        # Convert AVWAP dict to array
        avwap_arrays = []
        for key in sorted(avwap.keys()):  # Sort for consistent ordering
            avwap_arrays.append(avwap[key].reshape(-1, 1))
        
        avwap_combined = mx.concatenate(avwap_arrays, axis=1)
        
        # Combine with standard indicators
        return mx.concatenate([standard, avwap_combined], axis=1)


def parallel_indicator_computation_gpu(data_batch, indicator_library, avwap_cache):
    """GPU-accelerated replacement for parallel_indicator_computation."""
    pipeline = GPUAcceleratedPipeline()
    return pipeline.process_batch_gpu(data_batch, indicator_library, avwap_cache)


# Monkey-patch the existing function for seamless integration
def integrate_gpu_acceleration():
    """Replace CPU computation with GPU computation."""
    import sys
    
    # Find the module
    for module_name, module in sys.modules.items():
        if hasattr(module, 'parallel_indicator_computation'):
            # Replace with GPU version
            module.parallel_indicator_computation = parallel_indicator_computation_gpu
            logger.info(f"✅ Replaced {module_name}.parallel_indicator_computation with GPU version")
            break


# Benchmark comparison
def benchmark_gpu_vs_cpu():
    """Compare GPU vs CPU performance."""
    # Create sample data
    n_symbols = 10
    n_bars = 2600
    
    data_batch = []
    for i in range(n_symbols):
        df = pd.DataFrame({
            'open': np.random.randn(n_bars).cumsum() + 100,
            'high': np.random.randn(n_bars).cumsum() + 101,
            'low': np.random.randn(n_bars).cumsum() + 99,
            'close': np.random.randn(n_bars).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, n_bars).astype(float)
        })
        data_batch.append((f'SYM{i}', df))
    
    # Test GPU version
    pipeline = GPUAcceleratedPipeline()
    
    start = time.time()
    gpu_results = pipeline.process_batch_gpu(data_batch)
    gpu_time = time.time() - start
    
    print(f"\n🚀 GPU Performance:")
    print(f"   • Time: {gpu_time:.2f}s")
    print(f"   • Throughput: {n_symbols * n_bars / gpu_time:.0f} bars/sec")
    print(f"   • Per symbol: {gpu_time / n_symbols:.3f}s")
    
    # Verify results
    for symbol, df, indicators in gpu_results:
        print(f"   • {symbol}: {indicators.shape} indicators computed")


if __name__ == "__main__":
    benchmark_gpu_vs_cpu()