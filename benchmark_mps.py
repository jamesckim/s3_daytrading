#!/usr/bin/env python3
"""
Benchmark MPS vs CPU Performance
=================================
Compare training performance between CPU and Apple Silicon GPU (MPS).
"""

import torch
import torch.nn as nn
import time
import numpy as np
from indicator_transformer import IndicatorTransformer, IndicatorLibrary
import logging

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


def benchmark_device(device, num_epochs=10, batch_size=16):
    """Benchmark training on a specific device."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking on {device.upper()}")
    logger.info(f"{'='*60}")
    
    # Initialize model
    indicator_library = IndicatorLibrary()
    num_indicators = len(indicator_library.indicators)
    
    model = IndicatorTransformer(
        num_indicators=num_indicators,
        d_model=256,
        num_heads=8,
        num_layers=6
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    # Create dummy data
    batches_per_epoch = 50
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(5):
        data = create_dummy_batch(batch_size, num_indicators, device)
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
            
            # Forward pass
            data = create_dummy_batch(batch_size, num_indicators, device)
            outputs = model(data[0], data[1], data[2])
            
            # Loss calculation
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
    
    logger.info(f"\n{device.upper()} Performance Summary:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Avg epoch time: {avg_epoch_time:.2f}s")
    logger.info(f"  Avg batch time: {avg_batch_time*1000:.1f}ms")
    logger.info(f"  Throughput: {throughput:.1f} samples/sec")
    
    return {
        'device': device,
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'avg_batch_time': avg_batch_time,
        'throughput': throughput
    }


def main():
    """Run benchmarks on available devices."""
    logger.info("🏃 Indicator Transformer Performance Benchmark")
    
    # Check available devices
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')
    
    logger.info(f"Available devices: {devices}")
    
    # Run benchmarks
    results = []
    for device in devices:
        try:
            result = benchmark_device(device)
            results.append(result)
        except Exception as e:
            logger.error(f"Error benchmarking {device}: {e}")
    
    # Compare results
    if len(results) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("PERFORMANCE COMPARISON")
        logger.info(f"{'='*60}")
        
        # Sort by throughput
        results.sort(key=lambda x: x['throughput'], reverse=True)
        baseline = results[-1]['throughput']  # Slowest device as baseline
        
        for i, result in enumerate(results):
            speedup = result['throughput'] / baseline
            logger.info(f"\n{i+1}. {result['device'].upper()}:")
            logger.info(f"   Throughput: {result['throughput']:.1f} samples/sec")
            logger.info(f"   Speedup: {speedup:.2f}x")
            logger.info(f"   Avg batch time: {result['avg_batch_time']*1000:.1f}ms")
    
    # Recommendations
    logger.info(f"\n{'='*60}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*60}")
    
    if 'mps' in [r['device'] for r in results]:
        mps_result = next(r for r in results if r['device'] == 'mps')
        cpu_result = next(r for r in results if r['device'] == 'cpu')
        speedup = mps_result['throughput'] / cpu_result['throughput']
        
        logger.info(f"✅ MPS provides {speedup:.1f}x speedup over CPU")
        logger.info("✅ Recommended batch size for MPS: 16-32")
        logger.info("✅ No code changes needed - just use --device mps")
    
    logger.info("\nTo use MPS in training:")
    logger.info("  python train_indicator_transformer.py --device mps")


if __name__ == "__main__":
    main()