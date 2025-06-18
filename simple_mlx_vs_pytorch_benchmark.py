#!/usr/bin/env python3
"""
Simple MLX vs PyTorch MPS Benchmark
====================================
Standalone benchmark comparing MLX and PyTorch MPS performance
without external dependencies.
"""

import time
import numpy as np
import logging
import os
from typing import Dict, List

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available")

# MLX imports
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as mlx_optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available. Install with: pip install mlx")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# PyTorch Transformer Implementation
if PYTORCH_AVAILABLE:
    class SimpleTransformerPyTorch(nn.Module):
        def __init__(self, input_dim=128, d_model=256, num_heads=8, num_layers=4):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=1024,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            self.output_proj = nn.Linear(d_model, 1)
            
        def forward(self, x):
            x = self.input_proj(x)
            x = self.transformer(x)
            x = self.output_proj(x)
            return torch.sigmoid(x.squeeze(-1))


# MLX Transformer Implementation
if MLX_AVAILABLE:
    class TransformerBlockMLX(mlx_nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.norm1 = mlx_nn.LayerNorm(d_model)
            self.norm2 = mlx_nn.LayerNorm(d_model)
            
            # Self-attention
            self.d_k = d_model // num_heads
            self.num_heads = num_heads
            self.W_q = mlx_nn.Linear(d_model, d_model)
            self.W_k = mlx_nn.Linear(d_model, d_model)
            self.W_v = mlx_nn.Linear(d_model, d_model)
            self.W_o = mlx_nn.Linear(d_model, d_model)
            
            # Feed-forward
            self.ff = mlx_nn.Sequential(
                mlx_nn.Linear(d_model, 1024),
                mlx_nn.ReLU(),
                mlx_nn.Linear(1024, d_model)
            )
            
        def __call__(self, x):
            # Self-attention
            batch_size, seq_len, _ = x.shape
            
            Q = self.W_q(x).reshape(batch_size, seq_len, self.num_heads, self.d_k)
            K = self.W_k(x).reshape(batch_size, seq_len, self.num_heads, self.d_k)
            V = self.W_v(x).reshape(batch_size, seq_len, self.num_heads, self.d_k)
            
            Q = Q.transpose(0, 2, 1, 3)
            K = K.transpose(0, 2, 1, 3)
            V = V.transpose(0, 2, 1, 3)
            
            scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.d_k))
            attn = mx.softmax(scores, axis=-1)
            out = mx.matmul(attn, V)
            
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            out = self.W_o(out)
            
            # Residual + norm
            x = self.norm1(x + out)
            
            # Feed-forward
            out = self.ff(x)
            x = self.norm2(x + out)
            
            return x
    
    class SimpleTransformerMLX(mlx_nn.Module):
        def __init__(self, input_dim=128, d_model=256, num_heads=8, num_layers=4):
            super().__init__()
            self.input_proj = mlx_nn.Linear(input_dim, d_model)
            self.layers = [TransformerBlockMLX(d_model, num_heads) for _ in range(num_layers)]
            self.output_proj = mlx_nn.Linear(d_model, 1)
            
        def __call__(self, x):
            x = self.input_proj(x)
            for layer in self.layers:
                x = layer(x)
            x = self.output_proj(x)
            return mx.sigmoid(x.squeeze(-1))


def benchmark_pytorch_mps(batch_size=16, seq_len=100, num_epochs=5):
    """Benchmark PyTorch MPS performance."""
    if not PYTORCH_AVAILABLE:
        return None
        
    logger.info("\n" + "="*60)
    logger.info("Benchmarking PyTorch MPS")
    logger.info("="*60)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'cpu':
        logger.warning("MPS not available, using CPU")
    
    # Model
    model = SimpleTransformerPyTorch().to(device)
    if device == 'mps':
        torch.set_float32_matmul_precision('medium')
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.BCELoss()
    
    # Benchmark
    batches_per_epoch = 50
    epoch_times = []
    batch_times = []
    
    logger.info("Warming up...")
    for _ in range(5):
        x = torch.randn(batch_size, seq_len, 128).to(device)
        y = torch.rand(batch_size, seq_len).to(device)
        
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    logger.info(f"Running {num_epochs} epochs...")
    start_total = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for _ in range(batches_per_epoch):
            batch_start = time.time()
            
            x = torch.randn(batch_size, seq_len, 128).to(device)
            y = torch.rand(batch_size, seq_len).to(device)
            
            # Forward pass (skip mixed precision for now due to MPS issues)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if device == 'mps':
                torch.mps.synchronize()
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        logger.info(f"  Epoch {epoch+1}: {epoch_time:.2f}s")
    
    total_time = time.time() - start_total
    
    return {
        'framework': f'PyTorch {device.upper()}',
        'total_time': total_time,
        'avg_epoch_time': np.mean(epoch_times),
        'avg_batch_time': np.mean(batch_times),
        'throughput': (batch_size * seq_len * batches_per_epoch) / np.mean(epoch_times)
    }


def benchmark_mlx(batch_size=16, seq_len=100, num_epochs=5):
    """Benchmark MLX performance."""
    if not MLX_AVAILABLE:
        return None
        
    logger.info("\n" + "="*60)
    logger.info("Benchmarking Apple MLX")
    logger.info("="*60)
    
    # Model
    model = SimpleTransformerMLX()
    
    def loss_fn(model, x, y):
        outputs = model(x)
        # BCE loss
        loss = -mx.mean(y * mx.log(outputs + 1e-8) + (1 - y) * mx.log(1 - outputs + 1e-8))
        return loss
    
    loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)
    optimizer = mlx_optim.AdamW(learning_rate=2e-4, weight_decay=1e-5)
    
    # Benchmark
    batches_per_epoch = 50
    epoch_times = []
    batch_times = []
    
    logger.info("Warming up...")
    for _ in range(5):
        x = mx.random.normal([batch_size, seq_len, 128])
        y = mx.random.uniform(shape=[batch_size, seq_len])
        
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(loss)
    
    logger.info(f"Running {num_epochs} epochs...")
    start_total = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for _ in range(batches_per_epoch):
            batch_start = time.time()
            
            x = mx.random.normal([batch_size, seq_len, 128])
            y = mx.random.uniform(shape=[batch_size, seq_len])
            
            loss, grads = loss_and_grad_fn(model, x, y)
            optimizer.update(model, grads)
            mx.eval(loss)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        logger.info(f"  Epoch {epoch+1}: {epoch_time:.2f}s")
    
    total_time = time.time() - start_total
    
    return {
        'framework': 'MLX',
        'total_time': total_time,
        'avg_epoch_time': np.mean(epoch_times),
        'avg_batch_time': np.mean(batch_times),
        'throughput': (batch_size * seq_len * batches_per_epoch) / np.mean(epoch_times)
    }


def main():
    """Run simple benchmark comparison."""
    logger.info("🏁 Simple PyTorch MPS vs MLX Benchmark")
    logger.info("Transformer training performance comparison")
    
    results = []
    
    # Run PyTorch benchmark
    if PYTORCH_AVAILABLE:
        pytorch_result = benchmark_pytorch_mps()
        if pytorch_result:
            results.append(pytorch_result)
    
    # Run MLX benchmark
    if MLX_AVAILABLE:
        mlx_result = benchmark_mlx()
        if mlx_result:
            results.append(mlx_result)
    
    # Compare results
    if len(results) >= 2:
        print("\n" + "="*80)
        print("RESULTS COMPARISON")
        print("="*80)
        print(f"\n{'Framework':<20} {'Avg Epoch':<15} {'Avg Batch':<15} {'Throughput':<20}")
        print("-"*70)
        
        for r in results:
            print(f"{r['framework']:<20} {r['avg_epoch_time']:<15.2f}s "
                  f"{r['avg_batch_time']*1000:<15.1f}ms {r['throughput']:<20.1f}")
        
        # Calculate speedup
        if 'PyTorch' in results[0]['framework'] and 'MLX' in results[1]['framework']:
            speedup = results[0]['throughput'] / results[1]['throughput']
            print(f"\n📊 PyTorch MPS is {speedup:.2f}x faster than MLX")
        elif 'MLX' in results[0]['framework'] and 'PyTorch' in results[1]['framework']:
            speedup = results[1]['throughput'] / results[0]['throughput']
            print(f"\n📊 PyTorch MPS is {speedup:.2f}x faster than MLX")
    
    print("\n✅ Benchmark complete!")
    
    if not MLX_AVAILABLE:
        print("\n💡 To test MLX, install it with:")
        print("   python install_mlx.py")
        print("   # or")
        print("   pip install mlx")


if __name__ == "__main__":
    main()