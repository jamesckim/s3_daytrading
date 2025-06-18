#!/usr/bin/env python3
"""
Benchmark MPS Optimizations
===========================
Compare different optimization levels for PyTorch on Apple Silicon.
Tests the cumulative effect of MPS, mixed precision, and torch.compile.
"""

import torch
import torch.nn as nn
import time
import numpy as np
import os
from indicator_transformer import IndicatorTransformer, IndicatorLibrary
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_batch(batch_size, num_indicators, device):
    """Create dummy data for benchmarking."""
    indicator_values = torch.randn(batch_size, num_indicators).to(device)
    market_context = torch.randn(batch_size, 64).to(device)
    indicator_indices = torch.arange(num_indicators).repeat(batch_size, 1).to(device)
    
    # Dummy targets
    indicator_usefulness = torch.rand(batch_size, num_indicators).to(device)
    
    return indicator_values, market_context, indicator_indices, indicator_usefulness


def benchmark_configuration(
    device: str, 
    num_epochs: int = 5, 
    batch_size: int = 16,
    use_amp: bool = False,
    use_compile: bool = False,
    config_name: str = None
) -> Dict:
    """Benchmark a specific configuration."""
    
    if config_name is None:
        config_name = f"{device.upper()}"
        if use_amp:
            config_name += " + AMP"
        if use_compile:
            config_name += " + Compile"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {config_name}")
    logger.info(f"{'='*60}")
    
    # Set optimizations for MPS
    if device == 'mps':
        torch.set_float32_matmul_precision('medium')
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        # Set OMP threads
        try:
            import subprocess
            physical_cpus = int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).decode().strip())
            os.environ['OMP_NUM_THREADS'] = str(physical_cpus)
        except:
            pass
    
    # Initialize model
    indicator_library = IndicatorLibrary()
    num_indicators = len(indicator_library.indicators)
    
    model = IndicatorTransformer(
        num_indicators=num_indicators,
        d_model=256,
        num_heads=8,
        num_layers=6
    ).to(device)
    
    # Compile model if requested
    compiled = False
    if use_compile and device in ['cuda', 'mps']:
        try:
            logger.info("  Compiling model with torch.compile...")
            model = torch.compile(model, backend="inductor", mode="max-autotune")
            compiled = True
            logger.info("  ✓ Model compiled successfully")
        except Exception as e:
            logger.warning(f"  ✗ Compile failed: {e}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4 if device == 'mps' else 1e-4)
    criterion = nn.BCELoss()
    
    # Mixed precision settings
    amp_dtype = torch.bfloat16 if device == 'mps' else torch.float16
    
    # Create dummy data
    batches_per_epoch = 50
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(5):
        data = create_dummy_batch(batch_size, num_indicators, device)
        
        if use_amp and device in ['cuda', 'mps']:
            with torch.autocast(device_type=device, dtype=amp_dtype):
                outputs = model(data[0], data[1], data[2])
                loss = criterion(outputs['selection_probs'], data[3])
        else:
            outputs = model(data[0], data[1], data[2])
            loss = criterion(outputs['selection_probs'], data[3])
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Actual benchmark
    logger.info(f"Running {num_epochs} epochs with {batches_per_epoch} batches each...")
    
    epoch_times = []
    batch_times = []
    
    start_total = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for batch in range(batches_per_epoch):
            batch_start = time.time()
            
            # Create batch
            data = create_dummy_batch(batch_size, num_indicators, device)
            
            # Forward pass with or without AMP
            if use_amp and device in ['cuda', 'mps']:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    outputs = model(data[0], data[1], data[2])
                    loss = criterion(outputs['selection_probs'], data[3])
            else:
                outputs = model(data[0], data[1], data[2])
                loss = criterion(outputs['selection_probs'], data[3])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Synchronize for accurate timing
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        logger.info(f"  Epoch {epoch+1}/{num_epochs}: {epoch_time:.2f}s "
                   f"(avg batch: {np.mean(batch_times[-batches_per_epoch:])*1000:.1f}ms)")
    
    total_time = time.time() - start_total
    
    # Summary statistics
    avg_epoch_time = np.mean(epoch_times)
    avg_batch_time = np.mean(batch_times)
    throughput = (batch_size * batches_per_epoch) / avg_epoch_time
    
    logger.info(f"\n{config_name} Summary:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Avg epoch time: {avg_epoch_time:.2f}s")
    logger.info(f"  Avg batch time: {avg_batch_time*1000:.1f}ms")
    logger.info(f"  Throughput: {throughput:.1f} samples/sec")
    
    return {
        'config': config_name,
        'device': device,
        'use_amp': use_amp,
        'use_compile': use_compile,
        'compiled': compiled,
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'avg_batch_time': avg_batch_time,
        'throughput': throughput
    }


def main():
    """Run comprehensive optimization benchmarks."""
    logger.info("🏃 PyTorch M3 Optimization Benchmark")
    logger.info("Testing cumulative performance improvements on Apple Silicon")
    
    # Check if MPS is available
    if not torch.backends.mps.is_available():
        logger.error("MPS not available! This benchmark is designed for Apple Silicon.")
        return
    
    # Test configurations in order of optimization
    configurations = [
        # Baseline
        {'device': 'cpu', 'use_amp': False, 'use_compile': False, 'config_name': '1. Baseline (CPU)'},
        
        # MPS only
        {'device': 'mps', 'use_amp': False, 'use_compile': False, 'config_name': '2. +MPS'},
        
        # MPS + Mixed Precision
        {'device': 'mps', 'use_amp': True, 'use_compile': False, 'config_name': '3. +MPS +AMP (BF16)'},
        
        # MPS + Mixed Precision + Compile
        {'device': 'mps', 'use_amp': True, 'use_compile': True, 'config_name': '4. +MPS +AMP +Compile'},
    ]
    
    results = []
    
    for config in configurations:
        try:
            result = benchmark_configuration(**config)
            results.append(result)
        except Exception as e:
            logger.error(f"Error benchmarking {config['config_name']}: {e}")
    
    # Display comparison
    if results:
        logger.info(f"\n{'='*80}")
        logger.info("OPTIMIZATION COMPARISON (based on pytorch_m3_o3.md)")
        logger.info(f"{'='*80}")
        
        # Use CPU as baseline
        baseline_throughput = results[0]['throughput'] if results else 1.0
        
        logger.info(f"\n{'Configuration':<30} {'Throughput':<20} {'Speedup':<15} {'Epoch Time':<15}")
        logger.info("-" * 80)
        
        for result in results:
            speedup = result['throughput'] / baseline_throughput
            logger.info(
                f"{result['config']:<30} "
                f"{result['throughput']:<20.1f} "
                f"{speedup:<15.2f}x "
                f"{result['avg_epoch_time']:<15.2f}s"
            )
        
        # Show cumulative improvements
        logger.info(f"\n{'='*80}")
        logger.info("CUMULATIVE IMPROVEMENTS")
        logger.info(f"{'='*80}")
        
        if len(results) >= 2:
            mps_speedup = results[1]['throughput'] / results[0]['throughput']
            logger.info(f"MPS alone: {mps_speedup:.2f}x faster than CPU")
        
        if len(results) >= 3:
            amp_speedup = results[2]['throughput'] / results[1]['throughput']
            total_speedup = results[2]['throughput'] / results[0]['throughput']
            logger.info(f"AMP improvement: {amp_speedup:.2f}x faster than MPS alone")
            logger.info(f"MPS + AMP: {total_speedup:.2f}x faster than CPU")
        
        if len(results) >= 4:
            compile_speedup = results[3]['throughput'] / results[2]['throughput']
            final_speedup = results[3]['throughput'] / results[0]['throughput']
            logger.info(f"Compile improvement: {compile_speedup:.2f}x faster than MPS+AMP")
            logger.info(f"TOTAL (all optimizations): {final_speedup:.2f}x faster than CPU")
        
        # Recommendations
        logger.info(f"\n{'='*80}")
        logger.info("RECOMMENDATIONS")
        logger.info(f"{'='*80}")
        logger.info("✅ Use all optimizations for maximum performance:")
        logger.info("   python train_indicator_transformer.py --device mps")
        logger.info("   (AMP and compile are now enabled by default)")
        logger.info("\n✅ Expected real-world speedup: 5-7x over CPU")
        logger.info("✅ BFloat16 provides better numerical stability than Float16")
        logger.info("✅ torch.compile provides Metal kernel fusion")
        
        # Check PyTorch version
        logger.info(f"\n📌 Your PyTorch version: {torch.__version__}")
        if not torch.__version__.startswith(('2.5', '2.6', '2.7')):
            logger.warning("⚠️  Consider upgrading to PyTorch 2.5+ for best MPS performance:")
            logger.warning("   uv pip install --pre torch -U")


if __name__ == "__main__":
    main()