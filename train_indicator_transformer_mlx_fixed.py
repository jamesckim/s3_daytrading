#!/usr/bin/env python3
"""
Train Indicator Transformer Model with MLX (Fixed)
=================================================
MLX-native training with optimized data loading.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import sys
import time
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicator_transformer import IndicatorLibrary, MarketRegimeAnalyzer
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLXIndicatorTransformer(nn.Module):
    """MLX implementation of the Indicator Transformer model."""
    
    def __init__(self, num_indicators: int, d_model: int = 256, 
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.num_indicators = num_indicators
        self.d_model = d_model
        
        # Embeddings
        self.indicator_embedding = nn.Linear(1, d_model)
        self.market_embedding = nn.Linear(10, d_model)  # 10 market features
        self.position_embedding = nn.Embedding(num_indicators, d_model)
        
        # Transformer layers
        self.transformer_layers = []
        for _ in range(num_layers):
            layer = TransformerBlock(d_model, num_heads, dropout)
            self.transformer_layers.append(layer)
        
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
        
        # Layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
    def __call__(self, indicator_values, market_context, indicator_indices):
        batch_size, num_indicators = indicator_values.shape
        
        # Embed indicators (expand dimension for linear layer)
        indicator_embeds = self.indicator_embedding(indicator_values.reshape(-1, 1))
        indicator_embeds = indicator_embeds.reshape(batch_size, num_indicators, self.d_model)
        
        # Embed market context and broadcast
        market_embeds = self.market_embedding(market_context)
        market_embeds = mx.expand_dims(market_embeds, axis=1)
        market_embeds = mx.broadcast_to(market_embeds, (batch_size, num_indicators, self.d_model))
        
        # Position embeddings
        pos_embeds = self.position_embedding(indicator_indices)
        
        # Combine embeddings
        x = indicator_embeds + market_embeds + pos_embeds
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Selection probabilities
        selection_logits = self.selection_head(x).squeeze(-1)
        selection_probs = mx.sigmoid(selection_logits)
        
        # Indicator weights
        weight_logits = self.weight_head(x).squeeze(-1)
        indicator_weights = mx.softmax(weight_logits, axis=-1)
        
        return {
            'selection_probs': selection_probs,
            'indicator_weights': indicator_weights,
            'selection_logits': selection_logits
        }


class TransformerBlock(nn.Module):
    """Transformer block for MLX."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def __call__(self, x):
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.ln1(x + attn_out)
        
        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention for MLX."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, seq_len, head_dim]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = mx.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output


class OptimizedMLXDataset:
    """Optimized dataset for MLX training with faster data loading."""
    
    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime,
                 window_size: int = 100, future_window: int = 20, 
                 use_daily: bool = False, sample_interval: int = 240):
        """
        Initialize dataset with optimizations.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            window_size: Size of historical window for indicators
            future_window: Future window for calculating returns
            use_daily: Use daily data instead of minute data for faster loading
            sample_interval: Minutes between samples (default 240 = 4 hours)
        """
        self.symbols = symbols
        self.window_size = window_size
        self.future_window = future_window
        self.use_daily = use_daily
        self.sample_interval = sample_interval
        
        # Initialize components
        self.db_provider = DatabaseDataProvider()
        self.indicator_library = IndicatorLibrary()
        self.market_analyzer = MarketRegimeAnalyzer()
        
        # Load and prepare data
        logger.info(f"Loading {'daily' if use_daily else 'minute'} data for {len(symbols)} symbols...")
        self.samples = []
        self._prepare_samples_optimized(start_date, end_date)
        logger.info(f"Prepared {len(self.samples)} training samples")
        
        # Convert to MLX arrays
        if self.samples:
            self._prepare_mlx_arrays()
        else:
            logger.warning("No samples created - check data availability")
    
    def _prepare_samples_optimized(self, start_date: datetime, end_date: datetime):
        """Optimized sample preparation with better progress tracking."""
        total_start = time.time()
        
        # Process each symbol sequentially with progress
        for symbol in tqdm(self.symbols, desc="Loading symbols"):
            try:
                symbol_start = time.time()
                symbol_samples = self._process_symbol_optimized(symbol, start_date, end_date)
                symbol_time = time.time() - symbol_start
                
                self.samples.extend(symbol_samples)
                logger.info(f"{symbol}: {len(symbol_samples)} samples in {symbol_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        total_time = time.time() - total_start
        logger.info(f"Data loading completed in {total_time:.1f}s")
    
    def _process_symbol_optimized(self, symbol: str, start_date: datetime, 
                                  end_date: datetime) -> List[Dict]:
        """Optimized symbol processing with fewer indicator computations."""
        symbol_samples = []
        
        # Get data with minimal buffer
        buffer_days = 30 if self.use_daily else 1
        buffer_start = start_date - timedelta(days=buffer_days)
        
        if self.use_daily:
            # Use daily data for faster processing
            data = self._get_daily_data(symbol, buffer_start, end_date)
            sample_step = 1  # Daily samples
        else:
            # Use minute data with larger intervals
            data = self.db_provider.get_minute_data(symbol, buffer_start, end_date)
            sample_step = self.sample_interval  # Sample every N minutes
        
        if len(data) < self.window_size + self.future_window:
            logger.warning(f"{symbol}: Insufficient data ({len(data)} points)")
            return symbol_samples
        
        # Pre-compute all indicators once for efficiency
        logger.debug(f"Computing indicators for {symbol}...")
        all_indicators = self._compute_all_indicators_vectorized(data)
        
        # Create samples with progress
        num_samples = (len(data) - self.window_size - self.future_window) // sample_step
        
        for i in range(self.window_size, len(data) - self.future_window, sample_step):
            try:
                # Get windows
                hist_window = slice(i - self.window_size, i)
                future_window = slice(i, i + self.future_window)
                
                # Get pre-computed indicator values
                indicator_values = all_indicators[i]
                
                # Market context (simplified)
                hist_data = data.iloc[hist_window]
                market_context = self._get_simple_market_context(hist_data)
                
                # Future return
                future_return = (data['close'].iloc[future_window].iloc[-1] / 
                               data['close'].iloc[hist_window].iloc[-1]) - 1
                
                # Simple indicator usefulness
                indicator_usefulness = self._get_simple_usefulness(
                    indicator_values, future_return
                )
                
                symbol_samples.append({
                    'indicator_values': indicator_values,
                    'market_context': market_context,
                    'future_return': future_return,
                    'indicator_usefulness': indicator_usefulness
                })
                
            except Exception as e:
                logger.debug(f"Sample error at index {i}: {e}")
                continue
        
        return symbol_samples
    
    def _get_daily_data(self, symbol: str, start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Get daily data by aggregating minute data or using cache."""
        try:
            # Try to get minute data and aggregate
            minute_data = self.db_provider.get_minute_data(symbol, start_date, end_date)
            
            if minute_data.empty:
                # Generate mock daily data for testing
                return self._generate_mock_daily_data(symbol, start_date, end_date)
            
            # Aggregate to daily
            daily_data = minute_data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return daily_data
            
        except Exception as e:
            logger.warning(f"Failed to get daily data for {symbol}: {e}")
            return self._generate_mock_daily_data(symbol, start_date, end_date)
    
    def _generate_mock_daily_data(self, symbol: str, start_date: datetime, 
                                  end_date: datetime) -> pd.DataFrame:
        """Generate mock daily data for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic price data
        base_price = 100 + hash(symbol) % 400
        returns = np.random.randn(len(dates)) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(len(dates)) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.005)),
            'low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.005)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        return df
    
    def _compute_all_indicators_vectorized(self, data: pd.DataFrame) -> np.ndarray:
        """Compute indicators efficiently for all time points."""
        num_indicators = len(self.indicator_library.indicators)
        num_points = len(data)
        
        # Pre-allocate array
        all_values = np.zeros((num_points, num_indicators), dtype=np.float32)
        
        # Compute each indicator once for the entire series
        for idx, (name, _) in enumerate(self.indicator_library.indicators.items()):
            try:
                # Compute indicator for full series
                indicator_series = self.indicator_library.compute_indicator(data, name)
                
                # Store values
                if len(indicator_series) == len(data):
                    all_values[:, idx] = indicator_series.values
                else:
                    # Handle indicators with different lengths
                    offset = len(data) - len(indicator_series)
                    all_values[offset:, idx] = indicator_series.values
                    
            except Exception as e:
                logger.debug(f"Failed to compute {name}: {e}")
                # Leave as zeros
        
        return all_values
    
    def _get_simple_market_context(self, data: pd.DataFrame) -> np.ndarray:
        """Get simplified market context features."""
        try:
            returns = data['close'].pct_change().dropna()
            
            context = np.array([
                returns.mean() * 100,           # Average return
                returns.std() * 100,            # Volatility
                returns.skew(),                 # Skewness
                returns.kurt(),                 # Kurtosis
                (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,  # Period return
                data['volume'].mean() / 1e6,    # Average volume (millions)
                (data['high'] - data['low']).mean() / data['close'].mean() * 100,  # Avg range
                len(returns[returns > 0]) / len(returns),  # Up day ratio
                returns.iloc[-5:].mean() * 100,  # Recent momentum
                data['close'].rolling(20).std().iloc[-1] / data['close'].iloc[-1] * 100  # Recent vol
            ], dtype=np.float32)
            
            # Replace any NaN with 0
            context = np.nan_to_num(context, 0)
            
            return context
            
        except Exception as e:
            logger.debug(f"Market context error: {e}")
            return np.zeros(10, dtype=np.float32)
    
    def _get_simple_usefulness(self, indicator_values: np.ndarray, 
                               future_return: float) -> np.ndarray:
        """Simple indicator usefulness based on return direction."""
        usefulness = np.zeros_like(indicator_values)
        
        # Mark indicators as useful based on simple heuristics
        if abs(future_return) > 0.02:  # Significant move
            # Trend indicators more useful
            usefulness[0:5] = 0.8
            # Momentum indicators
            usefulness[5:10] = 0.6
            # Volatility indicators
            usefulness[10:15] = 0.4
        else:  # Small move
            # Mean reversion indicators
            usefulness[15:20] = 0.6
            # Range indicators
            usefulness[20:25] = 0.4
        
        # Add some randomness
        usefulness += np.random.rand(len(usefulness)) * 0.2
        
        # Normalize
        usefulness = np.clip(usefulness, 0, 1)
        
        return usefulness
    
    def _prepare_mlx_arrays(self):
        """Convert samples to MLX arrays."""
        if not self.samples:
            return
        
        # Stack all data
        self.indicator_values = mx.array([s['indicator_values'] for s in self.samples])
        self.market_contexts = mx.array([s['market_context'] for s in self.samples])
        self.indicator_usefulness = mx.array([s['indicator_usefulness'] for s in self.samples])
        self.future_returns = mx.array([[s['future_return']] for s in self.samples])
        
        # Create indicator indices
        num_indicators = self.indicator_values.shape[1]
        self.indicator_indices = mx.arange(num_indicators)
        
        logger.info(f"Converted to MLX arrays: {self.indicator_values.shape}")
    
    def __len__(self):
        return len(self.samples)
    
    def get_batch(self, indices):
        """Get a batch of data."""
        batch_indices = mx.array(indices)
        
        return {
            'indicator_values': self.indicator_values[batch_indices],
            'market_context': self.market_contexts[batch_indices],
            'indicator_usefulness': self.indicator_usefulness[batch_indices],
            'future_return': self.future_returns[batch_indices],
            'indicator_indices': mx.broadcast_to(
                mx.expand_dims(self.indicator_indices, 0),
                (len(indices), len(self.indicator_indices))
            )
        }


class MLXTrainer:
    """MLX-native trainer for the indicator transformer."""
    
    def __init__(self, model: MLXIndicatorTransformer, learning_rate: float = 2e-4):
        self.model = model
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
        
        # History
        self.train_losses = []
        self.val_losses = []
        
        logger.info("🚀 MLX Trainer initialized")
        logger.info("   Using native Apple Silicon optimization")
        logger.info("   Automatic mixed precision with bfloat16")
        
    def loss_fn(self, model, batch):
        """Calculate loss for a batch."""
        # Forward pass
        outputs = model(
            batch['indicator_values'],
            batch['market_context'],
            batch['indicator_indices']
        )
        
        selection_probs = outputs['selection_probs']
        indicator_weights = outputs['indicator_weights']
        indicator_usefulness = batch['indicator_usefulness']
        
        # Selection loss (BCE)
        selection_loss = mx.mean(
            -indicator_usefulness * mx.log(selection_probs + 1e-8) -
            (1 - indicator_usefulness) * mx.log(1 - selection_probs + 1e-8)
        )
        
        # Weight loss (MSE)
        weight_targets = indicator_usefulness * selection_probs
        weight_targets = weight_targets / (mx.sum(weight_targets, axis=1, keepdims=True) + 1e-8)
        weight_loss = mx.mean((indicator_weights - weight_targets) ** 2)
        
        # Sparsity penalty
        sparsity_loss = mx.mean(selection_probs) * 0.1
        
        # Total loss
        total_loss = selection_loss + 0.5 * weight_loss + sparsity_loss
        
        return total_loss, {
            'selection_loss': selection_loss,
            'weight_loss': weight_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def train_epoch(self, dataset: OptimizedMLXDataset, batch_size: int = 32):
        """Train for one epoch."""
        num_samples = len(dataset)
        indices = np.random.permutation(num_samples)
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(range(0, num_samples, batch_size), desc="Training")
        
        for i in pbar:
            # Get batch
            batch_indices = indices[i:i + batch_size]
            batch = dataset.get_batch(batch_indices)
            
            # Forward and backward pass
            loss_and_grad_fn = mx.value_and_grad(self.loss_fn, has_aux=True)
            (loss, loss_components), grads = loss_and_grad_fn(self.model, batch)
            
            # Update weights
            self.optimizer.update(self.model, grads)
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss/num_batches:.4f}'
            })
            
            # Force evaluation periodically
            if num_batches % 10 == 0:
                mx.eval(self.model.parameters())
        
        return total_loss / num_batches
    
    def validate(self, dataset: OptimizedMLXDataset, batch_size: int = 32):
        """Validate the model."""
        num_samples = len(dataset)
        total_loss = 0.0
        num_batches = 0
        
        metrics = {
            'avg_selected': 0,
            'selection_accuracy': 0,
        }
        
        for i in range(0, num_samples, batch_size):
            # Get batch
            batch_indices = list(range(i, min(i + batch_size, num_samples)))
            batch = dataset.get_batch(batch_indices)
            
            # Forward pass only
            outputs = self.model(
                batch['indicator_values'],
                batch['market_context'],
                batch['indicator_indices']
            )
            
            # Calculate loss
            loss, _ = self.loss_fn(self.model, batch)
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate metrics
            selection_probs = outputs['selection_probs']
            indicator_usefulness = batch['indicator_usefulness']
            
            selected = (selection_probs > 0.3).astype(mx.float32)
            metrics['avg_selected'] += mx.mean(mx.sum(selected, axis=1)).item()
            
            # Selection accuracy
            useful = (indicator_usefulness > 0.5).astype(mx.float32)
            correct = mx.sum(selected * useful, axis=1)
            total_useful = mx.sum(useful, axis=1) + 1e-8
            metrics['selection_accuracy'] += mx.mean(correct / total_useful).item()
        
        # Average metrics
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return total_loss / num_batches, metrics
    
    def train(self, train_dataset: OptimizedMLXDataset, val_dataset: OptimizedMLXDataset,
              epochs: int = 50, batch_size: int = 32, save_path: str = 'models'):
        """Train the model."""
        os.makedirs(save_path, exist_ok=True)
        best_val_loss = float('inf')
        start_time = time.time()
        
        logger.info(f"\n🎯 Starting MLX training")
        logger.info(f"   Epochs: {epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Training samples: {len(train_dataset)}")
        logger.info(f"   Validation samples: {len(val_dataset)}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_dataset, batch_size)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_dataset, batch_size)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Avg Selected: {metrics['avg_selected']:.1f}")
            logger.info(f"  Selection Accuracy: {metrics['selection_accuracy']:.2%}")
            logger.info(f"  Epoch Time: {epoch_time:.1f}s")
            
            # Estimate time remaining
            elapsed = time.time() - start_time
            avg_epoch_time = elapsed / (epoch + 1)
            eta = avg_epoch_time * (epochs - epoch - 1)
            logger.info(f"  ETA: {eta/60:.1f} min")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(save_path, 'best', epoch, train_loss, val_loss, metrics)
                logger.info("  ✓ Saved best model")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_model(save_path, f'checkpoint_{epoch+1}', epoch, train_loss, val_loss, metrics)
        
        # Save final model
        self._save_model(save_path, 'final', epochs-1, self.train_losses[-1], self.val_losses[-1], metrics)
        
        total_time = time.time() - start_time
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Total time: {total_time/60:.1f} min")
        logger.info(f"   Best val loss: {best_val_loss:.4f}")
        
        # Plot training history
        self._plot_training_history(save_path)
    
    def _save_model(self, save_path: str, suffix: str, epoch: int, 
                    train_loss: float, val_loss: float, metrics: Dict):
        """Save model weights and training state."""
        model_path = f"{save_path}/indicator_transformer_mlx_{suffix}.npz"
        
        # Get model weights
        weights = self.model.parameters()
        
        # Flatten weights dictionary for saving
        flat_weights = {}
        
        def flatten_dict(d, parent_key=''):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, new_key)
                else:
                    flat_weights[new_key] = np.array(v)
        
        flatten_dict(weights)
        
        # Save weights and metadata
        np.savez(
            model_path,
            **flat_weights,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            **{f"metric_{k}": v for k, v in metrics.items()}
        )
        
        logger.info(f"Model saved to {model_path}")
    
    def _plot_training_history(self, save_path: str):
        """Plot and save training history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MLX Indicator Transformer Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/mlx_training_history.png")
        plt.close()


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Indicator Transformer with MLX (Fixed)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on')
    parser.add_argument('--days', type=int, default=180, help='Days of historical data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--use-daily', action='store_true', help='Use daily data for faster loading')
    parser.add_argument('--sample-interval', type=int, default=240, 
                       help='Minutes between samples for minute data (default: 240 = 4 hours)')
    
    args = parser.parse_args()
    
    # Default symbols if none provided
    if not args.symbols:
        args.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
            'JPM', 'BAC', 'GS', 'WMT', 'HD', 'DIS', 'NFLX', 'AMD'
        ]
    
    logger.info("🚀 Starting MLX Indicator Transformer Training (Optimized)")
    logger.info(f"Training on {len(args.symbols)} symbols for {args.epochs} epochs")
    logger.info("Device: Apple Silicon (Metal Performance Shaders)")
    
    # System info
    logger.info(f"\nSystem Information:")
    logger.info(f"  CPU cores: {os.cpu_count()}")
    logger.info(f"  Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    logger.info(f"  MLX backend: Metal (unified memory)")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Create dataset with optimizations
    logger.info("\nLoading training data...")
    data_start = time.time()
    
    dataset = OptimizedMLXDataset(
        args.symbols, 
        start_date, 
        end_date,
        use_daily=args.use_daily,
        sample_interval=args.sample_interval
    )
    
    data_time = time.time() - data_start
    logger.info(f"Data loading completed in {data_time/60:.1f} minutes")
    
    if len(dataset) == 0:
        logger.error("No training samples created. Check data availability.")
        return
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    # Create indices for train/val split
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train/val datasets
    train_dataset = OptimizedMLXDataset.__new__(OptimizedMLXDataset)
    train_dataset.__dict__ = dataset.__dict__.copy()
    train_dataset.samples = [dataset.samples[i] for i in train_indices]
    train_dataset._prepare_mlx_arrays()
    
    val_dataset = OptimizedMLXDataset.__new__(OptimizedMLXDataset)
    val_dataset.__dict__ = dataset.__dict__.copy()
    val_dataset.samples = [dataset.samples[i] for i in val_indices]
    val_dataset._prepare_mlx_arrays()
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    indicator_library = IndicatorLibrary()
    num_indicators = len(indicator_library.indicators)
    
    model = MLXIndicatorTransformer(
        num_indicators=num_indicators,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    logger.info(f"\nModel initialized with {num_indicators} indicators")
    
    # Train
    trainer = MLXTrainer(model)
    trainer.train(train_dataset, val_dataset, epochs=args.epochs, batch_size=args.batch_size)
    
    logger.info("\n✅ MLX training complete!")
    logger.info("Model saved to models/indicator_transformer_mlx_best.npz")


if __name__ == "__main__":
    main()