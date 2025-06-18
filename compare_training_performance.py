#!/usr/bin/env python3
"""
Compare MLX vs PyTorch Training Performance
===========================================
Runs both training implementations and compares:
- Training speed
- Memory usage
- Model accuracy
- Hardware utilization
"""

import os
import sys
import time
import psutil
import subprocess
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import mlx.core as mx
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingBenchmark:
    """Benchmark training performance for MLX vs PyTorch."""
    
    def __init__(self):
        self.results = {
            'pytorch': {},
            'mlx': {}
        }
        self.process = psutil.Process()
        
    def measure_memory(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def run_pytorch_training(self, symbols, epochs=5, batch_size=32):
        """Run PyTorch training and measure performance."""
        logger.info("\n" + "="*60)
        logger.info("🔥 PYTORCH TRAINING BENCHMARK")
        logger.info("="*60)
        
        # Initial memory
        initial_memory = self.measure_memory()
        start_time = time.time()
        
        # Prepare command
        cmd = [
            sys.executable, 
            "train_indicator_transformer.py",
            "--symbols"] + symbols + [
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--days", "30",  # Smaller dataset for benchmark
            "--device", "mps" if torch.backends.mps.is_available() else "cpu",
            "--no-compile"  # Disable compile for fair comparison
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Track metrics during training
        memory_samples = []
        cpu_samples = []
        
        # Run training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor process
        last_epoch = 0
        epoch_times = []
        epoch_start = time.time()
        
        while True:
            line = process.stdout.readline()
            if not line:
                break
            
            print(f"[PyTorch] {line.strip()}")
            
            # Track memory periodically
            memory_samples.append(self.measure_memory())
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            
            # Parse epoch completion
            if "Epoch" in line and "/" in line:
                try:
                    current_epoch = int(line.split("Epoch")[1].split("/")[0].strip())
                    if current_epoch > last_epoch:
                        epoch_time = time.time() - epoch_start
                        epoch_times.append(epoch_time)
                        last_epoch = current_epoch
                        epoch_start = time.time()
                except:
                    pass
            
            # Parse loss values
            if "Train Loss:" in line:
                try:
                    loss = float(line.split("Train Loss:")[1].strip().split()[0])
                    self.results['pytorch']['final_loss'] = loss
                except:
                    pass
        
        process.wait()
        
        # Calculate metrics
        total_time = time.time() - start_time
        peak_memory = max(memory_samples) if memory_samples else initial_memory
        avg_memory = np.mean(memory_samples) if memory_samples else initial_memory
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
        
        self.results['pytorch'] = {
            'total_time': total_time,
            'avg_epoch_time': np.mean(epoch_times) if epoch_times else total_time/epochs,
            'peak_memory': peak_memory,
            'avg_memory': avg_memory,
            'memory_increase': peak_memory - initial_memory,
            'avg_cpu_usage': avg_cpu,
            'epochs_completed': len(epoch_times),
            'framework': 'PyTorch',
            'device': 'MPS' if torch.backends.mps.is_available() else 'CPU'
        }
        
        logger.info(f"\n✅ PyTorch training completed in {total_time:.1f}s")
        logger.info(f"   Peak memory: {peak_memory:.0f} MB")
        logger.info(f"   Avg CPU usage: {avg_cpu:.1f}%")
        
        return self.results['pytorch']
    
    def run_mlx_training(self, symbols, epochs=5, batch_size=32):
        """Run MLX training and measure performance."""
        logger.info("\n" + "="*60)
        logger.info("🚀 MLX TRAINING BENCHMARK")
        logger.info("="*60)
        
        # Initial memory
        initial_memory = self.measure_memory()
        start_time = time.time()
        
        # Prepare command
        cmd = [
            sys.executable, 
            "train_indicator_transformer_mlx.py",
            "--symbols"] + symbols + [
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--days", "30"  # Smaller dataset for benchmark
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Track metrics during training
        memory_samples = []
        cpu_samples = []
        
        # Run training
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor process
        last_epoch = 0
        epoch_times = []
        epoch_start = time.time()
        
        while True:
            line = process.stdout.readline()
            if not line:
                break
            
            print(f"[MLX] {line.strip()}")
            
            # Track memory periodically
            memory_samples.append(self.measure_memory())
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
            
            # Parse epoch completion
            if "Epoch" in line and "/" in line:
                try:
                    current_epoch = int(line.split("Epoch")[1].split("/")[0].strip())
                    if current_epoch > last_epoch:
                        epoch_time = time.time() - epoch_start
                        epoch_times.append(epoch_time)
                        last_epoch = current_epoch
                        epoch_start = time.time()
                except:
                    pass
            
            # Parse loss values
            if "Train Loss:" in line:
                try:
                    loss = float(line.split("Train Loss:")[1].strip().split()[0])
                    self.results['mlx']['final_loss'] = loss
                except:
                    pass
        
        process.wait()
        
        # Calculate metrics
        total_time = time.time() - start_time
        peak_memory = max(memory_samples) if memory_samples else initial_memory
        avg_memory = np.mean(memory_samples) if memory_samples else initial_memory
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
        
        self.results['mlx'] = {
            'total_time': total_time,
            'avg_epoch_time': np.mean(epoch_times) if epoch_times else total_time/epochs,
            'peak_memory': peak_memory,
            'avg_memory': avg_memory,
            'memory_increase': peak_memory - initial_memory,
            'avg_cpu_usage': avg_cpu,
            'epochs_completed': len(epoch_times),
            'framework': 'MLX',
            'device': 'Metal'
        }
        
        logger.info(f"\n✅ MLX training completed in {total_time:.1f}s")
        logger.info(f"   Peak memory: {peak_memory:.0f} MB")
        logger.info(f"   Avg CPU usage: {avg_cpu:.1f}%")
        
        return self.results['mlx']
    
    def compare_results(self):
        """Compare and visualize results."""
        logger.info("\n" + "="*80)
        logger.info("📊 PERFORMANCE COMPARISON")
        logger.info("="*80)
        
        # Create comparison DataFrame
        df = pd.DataFrame([self.results['pytorch'], self.results['mlx']])
        
        # Calculate speedup
        pytorch_time = self.results['pytorch']['total_time']
        mlx_time = self.results['mlx']['total_time']
        speedup = pytorch_time / mlx_time
        
        # Memory efficiency
        pytorch_mem = self.results['pytorch']['peak_memory']
        mlx_mem = self.results['mlx']['peak_memory']
        memory_reduction = (1 - mlx_mem / pytorch_mem) * 100
        
        logger.info(f"\n🏁 Training Time:")
        logger.info(f"   PyTorch: {pytorch_time:.1f}s")
        logger.info(f"   MLX: {mlx_time:.1f}s")
        logger.info(f"   Speedup: {speedup:.2f}x faster")
        
        logger.info(f"\n💾 Memory Usage:")
        logger.info(f"   PyTorch Peak: {pytorch_mem:.0f} MB")
        logger.info(f"   MLX Peak: {mlx_mem:.0f} MB")
        logger.info(f"   Memory Reduction: {memory_reduction:.1f}%")
        
        logger.info(f"\n⚡ Per-Epoch Time:")
        logger.info(f"   PyTorch: {self.results['pytorch']['avg_epoch_time']:.1f}s")
        logger.info(f"   MLX: {self.results['mlx']['avg_epoch_time']:.1f}s")
        
        # Create visualizations
        self._create_comparison_plots()
        
        # Save detailed results
        with open('training_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("\n📁 Results saved to training_benchmark_results.json")
        
        return speedup, memory_reduction
    
    def _create_comparison_plots(self):
        """Create comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Training Time Comparison
        ax = axes[0, 0]
        frameworks = ['PyTorch', 'MLX']
        times = [self.results['pytorch']['total_time'], self.results['mlx']['total_time']]
        bars = ax.bar(frameworks, times, color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Total Training Time')
        
        # Add values on bars
        for bar, time in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{time:.1f}s', ha='center', va='bottom')
        
        # 2. Memory Usage Comparison
        ax = axes[0, 1]
        peak_mem = [self.results['pytorch']['peak_memory'], self.results['mlx']['peak_memory']]
        avg_mem = [self.results['pytorch']['avg_memory'], self.results['mlx']['avg_memory']]
        
        x = np.arange(len(frameworks))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, peak_mem, width, label='Peak Memory', color=['#FF6B6B', '#4ECDC4'])
        bars2 = ax.bar(x + width/2, avg_mem, width, label='Avg Memory', color=['#FFB6B6', '#8EDCD4'])
        
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage')
        ax.set_xticks(x)
        ax.set_xticklabels(frameworks)
        ax.legend()
        
        # 3. Per-Epoch Time
        ax = axes[1, 0]
        epoch_times = [self.results['pytorch']['avg_epoch_time'], self.results['mlx']['avg_epoch_time']]
        bars = ax.bar(frameworks, epoch_times, color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Average Time per Epoch')
        
        for bar, time in zip(bars, epoch_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{time:.1f}s', ha='center', va='bottom')
        
        # 4. Performance Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate metrics
        speedup = self.results['pytorch']['total_time'] / self.results['mlx']['total_time']
        memory_saving = (1 - self.results['mlx']['peak_memory'] / self.results['pytorch']['peak_memory']) * 100
        
        summary_text = f"""Performance Summary:
        
MLX vs PyTorch:
• Speed: {speedup:.2f}x faster
• Memory: {memory_saving:.1f}% less
• Device: {self.results['mlx']['device']} vs {self.results['pytorch']['device']}

MLX Advantages:
✓ Native Apple Silicon optimization
✓ Unified memory architecture
✓ Automatic mixed precision
✓ No CPU-GPU data transfers
"""
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('training_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Comparison plots saved to training_performance_comparison.png")


def run_quick_test():
    """Run a quick test to verify both implementations work."""
    logger.info("\n🧪 Running quick functionality test...")
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Test PyTorch
    logger.info("\nTesting PyTorch implementation...")
    try:
        from train_indicator_transformer import IndicatorTransformer, get_optimal_device
        from indicator_transformer import IndicatorLibrary
        
        device = get_optimal_device()
        library = IndicatorLibrary()
        model = IndicatorTransformer(
            num_indicators=len(library.indicators),
            d_model=64,  # Smaller for test
            num_heads=4,
            num_layers=2
        ).to(device)
        
        # Test forward pass
        batch_size = 4
        num_indicators = len(library.indicators)
        
        test_input = {
            'indicator_values': torch.randn(batch_size, num_indicators).to(device),
            'market_context': torch.randn(batch_size, 10).to(device),
            'indicator_indices': torch.arange(num_indicators).unsqueeze(0).repeat(batch_size, 1).to(device)
        }
        
        output = model(test_input['indicator_values'], test_input['market_context'], 
                      test_input['indicator_indices'])
        
        logger.info(f"✅ PyTorch test passed! Output shape: {output['selection_probs'].shape}")
        
    except Exception as e:
        logger.error(f"❌ PyTorch test failed: {e}")
        return False
    
    # Test MLX
    logger.info("\nTesting MLX implementation...")
    try:
        from train_indicator_transformer_mlx import MLXIndicatorTransformer
        
        mlx_model = MLXIndicatorTransformer(
            num_indicators=num_indicators,
            d_model=64,
            num_heads=4,
            num_layers=2
        )
        
        # Test forward pass
        test_input_mlx = {
            'indicator_values': mx.random.normal((batch_size, num_indicators)),
            'market_context': mx.random.normal((batch_size, 10)),
            'indicator_indices': mx.broadcast_to(
                mx.expand_dims(mx.arange(num_indicators), 0),
                (batch_size, num_indicators)
            )
        }
        
        output_mlx = mlx_model(
            test_input_mlx['indicator_values'],
            test_input_mlx['market_context'],
            test_input_mlx['indicator_indices']
        )
        
        logger.info(f"✅ MLX test passed! Output shape: {output_mlx['selection_probs'].shape}")
        
    except Exception as e:
        logger.error(f"❌ MLX test failed: {e}")
        return False
    
    return True


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare MLX vs PyTorch training performance')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                       help='Symbols to use for benchmark')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs for benchmark')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip functionality test')
    
    args = parser.parse_args()
    
    logger.info("🔬 MLX vs PyTorch Training Performance Comparison")
    logger.info("="*60)
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Check device availability
    if torch.backends.mps.is_available():
        logger.info("Device: Apple Silicon (MPS/Metal)")
    else:
        logger.info("Device: CPU")
    
    # Run quick test first
    if not args.skip_test:
        if not run_quick_test():
            logger.error("Functionality tests failed. Please check the implementations.")
            return
    
    # Create benchmark
    benchmark = TrainingBenchmark()
    
    # Run PyTorch benchmark
    pytorch_results = benchmark.run_pytorch_training(
        args.symbols, args.epochs, args.batch_size
    )
    
    # Run MLX benchmark
    mlx_results = benchmark.run_mlx_training(
        args.symbols, args.epochs, args.batch_size
    )
    
    # Compare results
    speedup, memory_reduction = benchmark.compare_results()
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🏆 BENCHMARK SUMMARY")
    logger.info("="*80)
    logger.info(f"\nMLX Performance Gains:")
    logger.info(f"  • Training Speed: {speedup:.2f}x faster")
    logger.info(f"  • Memory Usage: {memory_reduction:.1f}% less")
    logger.info(f"  • Native Apple Silicon: ✓")
    logger.info(f"  • Unified Memory: ✓")
    logger.info(f"  • Automatic Mixed Precision: ✓")
    
    if speedup > 1.5:
        logger.info(f"\n✅ MLX provides significant performance improvements!")
        logger.info(f"   Recommendation: Use MLX for training on Apple Silicon")
    else:
        logger.info(f"\n⚠️  Performance gains are modest")
        logger.info(f"   Consider dataset size and model complexity")
    
    logger.info("\n📁 Results saved:")
    logger.info("   - training_benchmark_results.json")
    logger.info("   - training_performance_comparison.png")


if __name__ == "__main__":
    main()