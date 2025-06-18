#!/usr/bin/env python3
"""
Indicator Transformer - MLX Implementation
==========================================
MLX-based transformer for indicator selection on Apple Silicon.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer in MLX."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(self.d_k))
        
        if mask is not None:
            scores = mx.where(mask == 0, mx.array(-1e9), scores)
        
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = mx.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def __call__(self, x, mask=None):
        # Self-attention with residual
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class IndicatorTransformerMLX(nn.Module):
    """MLX implementation of the Indicator Transformer."""
    
    def __init__(
        self,
        num_indicators: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        self.num_indicators = num_indicators
        self.d_model = d_model
        
        # Embeddings
        self.indicator_embedding = nn.Embedding(num_indicators, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.temporal_embedding = nn.Linear(1, d_model)
        
        # Market context encoder
        self.market_context_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_model)
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        # Output heads
        self.selection_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.weight_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def __call__(self, indicator_values, market_context, indicator_indices):
        batch_size = indicator_values.shape[0]
        seq_len = indicator_values.shape[1]
        
        # Indicator embeddings
        indicator_embeds = self.indicator_embedding(indicator_indices)
        
        # Position embeddings
        positions = mx.arange(seq_len)
        position_embeds = self.position_embedding(positions)
        position_embeds = mx.expand_dims(position_embeds, 0)
        position_embeds = mx.repeat(position_embeds, batch_size, axis=0)
        
        # Temporal embeddings from indicator values
        temporal_embeds = self.temporal_embedding(mx.expand_dims(indicator_values, -1))
        
        # Market context
        context_embeds = self.market_context_encoder(market_context)
        context_embeds = mx.expand_dims(context_embeds, 1)
        context_embeds = mx.repeat(context_embeds, seq_len, axis=1)
        
        # Combine embeddings
        embeddings = indicator_embeds + position_embeds + temporal_embeds + context_embeds
        
        # Pass through transformer blocks
        attention_weights = []
        x = embeddings
        for block in self.transformer_blocks:
            x, attn = block(x)
            attention_weights.append(attn)
        
        # Output predictions
        selection_logits = self.selection_head(x).squeeze(-1)
        selection_probs = mx.sigmoid(selection_logits)
        
        weight_logits = self.weight_head(x).squeeze(-1)
        weights = mx.softmax(weight_logits, axis=-1)
        
        return {
            'selection_probs': selection_probs,
            'indicator_weights': weights,
            'attention_weights': attention_weights
        }


class MLXIndicatorSelector:
    """MLX-based indicator selector."""
    
    def __init__(self, model_path: Optional[str] = None):
        from indicator_transformer import IndicatorLibrary
        self.indicator_library = IndicatorLibrary()
        self.num_indicators = len(self.indicator_library.indicators)
        
        # Initialize model
        self.model = IndicatorTransformerMLX(
            num_indicators=self.num_indicators,
            d_model=256,
            num_heads=8,
            num_layers=6
        )
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        logger.info(f"MLX Indicator Selector initialized with {self.num_indicators} indicators")
    
    def load_model(self, model_path: str):
        """Load model weights."""
        # MLX uses different format, would need conversion from PyTorch
        logger.warning("Model loading not implemented for MLX version")
    
    def select_indicators(self, data: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """Select indicators using the MLX model."""
        # Compute indicator values
        indicator_values = []
        for indicator_name in self.indicator_library.indicators:
            try:
                series = self.indicator_library.compute_indicator(data, indicator_name)
                if len(series) > 0:
                    indicator_values.append(float(series.iloc[-1]))
                else:
                    indicator_values.append(0.0)
            except:
                indicator_values.append(0.0)
        
        # Convert to MLX array
        indicator_tensor = mx.array(indicator_values).reshape(1, -1)
        
        # Create dummy market context
        market_context = mx.zeros((1, 10))
        
        # Create indicator indices
        indicator_indices = mx.arange(self.num_indicators).reshape(1, -1)
        
        # Forward pass
        outputs = self.model(indicator_tensor, market_context, indicator_indices)
        
        # Get selection probabilities
        selection_probs = outputs['selection_probs'][0]
        weights = outputs['indicator_weights'][0]
        
        # Convert to numpy for processing
        probs_np = np.array(selection_probs)
        weights_np = np.array(weights)
        
        # Select top-k
        top_indices = np.argsort(probs_np)[-top_k:][::-1]
        
        selected = {}
        indicator_names = list(self.indicator_library.indicators.keys())
        
        for idx in top_indices:
            name = indicator_names[idx]
            selected[name] = {
                'selection_prob': float(probs_np[idx]),
                'weight': float(weights_np[idx]),
                'value': indicator_values[idx]
            }
        
        return {
            'selected_indicators': selected,
            'selection_method': 'mlx_transformer'
        }


def create_mlx_optimizer(model, learning_rate=1e-4):
    """Create MLX optimizer."""
    return optim.AdamW(learning_rate=learning_rate, weight_decay=1e-5)


def train_step(model, optimizer, batch, loss_fn):
    """Single training step in MLX."""
    def loss_fn_wrapper(model, batch):
        outputs = model(
            batch['indicator_values'],
            batch['market_context'],
            batch['indicator_indices']
        )
        
        # Compute losses
        selection_probs = outputs['selection_probs']
        indicator_weights = outputs['indicator_weights']
        
        # BCE loss for selection
        selection_loss = mx.mean(
            -batch['targets'] * mx.log(selection_probs + 1e-8) 
            - (1 - batch['targets']) * mx.log(1 - selection_probs + 1e-8)
        )
        
        # MSE loss for weights
        weight_loss = mx.mean((indicator_weights - batch['weight_targets']) ** 2)
        
        # Total loss
        loss = selection_loss + 0.5 * weight_loss
        
        return loss, outputs
    
    # Compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn_wrapper)
    (loss, outputs), grads = loss_and_grad_fn(model, batch)
    
    # Update weights
    optimizer.update(model, grads)
    
    return loss, outputs


def benchmark_mlx_transformer():
    """Benchmark MLX transformer performance."""
    import time
    
    logger.info("Benchmarking MLX Transformer...")
    
    # Initialize model
    num_indicators = 111  # From PyTorch version
    model = IndicatorTransformerMLX(
        num_indicators=num_indicators,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    # Create optimizer
    optimizer = create_mlx_optimizer(model)
    
    # Benchmark parameters
    batch_size = 16
    num_epochs = 5
    batches_per_epoch = 50
    
    # Create dummy data
    def create_batch():
        return {
            'indicator_values': mx.random.normal((batch_size, num_indicators)),
            'market_context': mx.random.normal((batch_size, 10)),
            'indicator_indices': mx.tile(mx.arange(num_indicators), (batch_size, 1)),
            'targets': mx.random.uniform((batch_size, num_indicators)),
            'weight_targets': mx.softmax(mx.random.normal((batch_size, num_indicators)), axis=-1)
        }
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(5):
        batch = create_batch()
        loss, _ = train_step(model, optimizer, batch, None)
        mx.eval(loss)  # Force evaluation
    
    # Benchmark
    logger.info(f"Running {num_epochs} epochs...")
    epoch_times = []
    batch_times = []
    
    start_total = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for batch_idx in range(batches_per_epoch):
            batch_start = time.time()
            
            batch = create_batch()
            loss, _ = train_step(model, optimizer, batch, None)
            mx.eval(loss)  # Force evaluation
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        avg_batch_time = np.mean(batch_times[-batches_per_epoch:])
        logger.info(f"  Epoch {epoch+1}/{num_epochs}: {epoch_time:.2f}s "
                   f"(avg batch: {avg_batch_time*1000:.1f}ms)")
    
    total_time = time.time() - start_total
    
    # Summary
    avg_epoch_time = np.mean(epoch_times)
    avg_batch_time = np.mean(batch_times)
    throughput = (batch_size * batches_per_epoch) / avg_epoch_time
    
    logger.info(f"\nMLX Performance Summary:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Avg epoch time: {avg_epoch_time:.2f}s")
    logger.info(f"  Avg batch time: {avg_batch_time*1000:.1f}ms")
    logger.info(f"  Throughput: {throughput:.1f} samples/sec")
    
    return {
        'framework': 'MLX',
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'avg_batch_time': avg_batch_time,
        'throughput': throughput
    }


if __name__ == "__main__":
    # Test the MLX implementation
    logger.info("Testing MLX Indicator Transformer...")
    
    # Run benchmark
    results = benchmark_mlx_transformer()
    
    # Test inference
    logger.info("\nTesting inference...")
    selector = MLXIndicatorSelector()
    
    # Create dummy data
    dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
    test_data = pd.DataFrame({
        'open': 100 + np.random.randn(500).cumsum() * 0.1,
        'high': 100.5 + np.random.randn(500).cumsum() * 0.1,
        'low': 99.5 + np.random.randn(500).cumsum() * 0.1,
        'close': 100 + np.random.randn(500).cumsum() * 0.1,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    result = selector.select_indicators(test_data)
    logger.info(f"\nSelected {len(result['selected_indicators'])} indicators")
    for name, info in list(result['selected_indicators'].items())[:3]:
        logger.info(f"  {name}: prob={info['selection_prob']:.3f}")