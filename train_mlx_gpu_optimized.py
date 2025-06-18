#!/usr/bin/env python3
"""
GPU-Optimized MLX Training
==========================
Maximizes GPU utilization on Apple Silicon.
"""

import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import sys
import time
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicator_transformer import IndicatorLibrary
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider
from optimized_avwap_indicators import OptimizedAVWAPCalculator, integrate_with_indicator_library
from train_indicator_transformer_mlx_fixed import MLXIndicatorTransformer
from mlx_trainer_fixed import MLXTrainerFixed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Force MLX to use GPU more aggressively
mx.set_default_device(mx.gpu)


class GPUOptimizedDataLoader:
    """Optimized data loader that maximizes GPU usage."""
    
    def __init__(self, samples, batch_size=64):  # Larger batch size for GPU
        self.samples = samples
        self.batch_size = batch_size
        
        # Pre-compute and transfer all data to GPU at once
        logger.info("🚀 Transferring all data to GPU memory...")
        
        # Convert to numpy arrays first (more efficient)
        indicator_values = np.array([s['indicator_values'] for s in samples], dtype=np.float32)
        market_contexts = np.array([s['market_context'] for s in samples], dtype=np.float32)
        indicator_usefulness = np.array([s['indicator_usefulness'] for s in samples], dtype=np.float32)
        future_returns = np.array([[s['future_return']] for s in samples], dtype=np.float32)
        
        # Transfer to MLX GPU arrays
        self.indicator_values = mx.array(indicator_values)
        self.market_contexts = mx.array(market_contexts)
        self.indicator_usefulness = mx.array(indicator_usefulness)
        self.future_returns = mx.array(future_returns)
        
        # Pre-allocate indicator indices
        self.indicator_indices = mx.arange(self.indicator_values.shape[1])
        
        # Pre-compute all possible batches for faster iteration
        self.n_samples = len(samples)
        self.indices = mx.arange(self.n_samples)
        
        logger.info(f"✅ GPU memory loaded: {self.n_samples} samples")
        
    def __len__(self):
        return self.n_samples
    
    def get_batch(self, indices):
        """Get batch - data already on GPU."""
        return {
            'indicator_values': self.indicator_values[indices],
            'market_context': self.market_contexts[indices],
            'indicator_usefulness': self.indicator_usefulness[indices],
            'future_return': self.future_returns[indices],
            'indicator_indices': mx.broadcast_to(
                mx.expand_dims(self.indicator_indices, 0),
                (len(indices), len(self.indicator_indices))
            )
        }
    
    def get_random_batch(self, batch_size=None):
        """Get random batch for training."""
        if batch_size is None:
            batch_size = self.batch_size
        
        # Random indices
        indices = mx.random.randint(0, self.n_samples, (batch_size,))
        return self.get_batch(indices)


class GPUOptimizedTrainer:
    """Trainer optimized for maximum GPU utilization."""
    
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        # Use Adam with lower learning rate for stability
        self.optimizer = optim.Adam(learning_rate=learning_rate * 0.1)  # Reduced LR
        
        # Keep model on GPU
        mx.eval(self.model.trainable_parameters())
        
        self.train_losses = []
        self.val_losses = []
        
        logger.info("🔥 GPU-Optimized Trainer initialized")
        
    def train_epoch_gpu(self, train_loader, val_loader):
        """Train one epoch with maximum GPU utilization."""
        
        # Larger batch size for GPU
        batch_size = 128  # Increase from default 32
        n_batches = len(train_loader) // batch_size
        
        epoch_loss = 0.0
        
        # Process multiple batches in parallel on GPU
        for i in tqdm(range(n_batches), desc="GPU Training"):
            # Get batch (already on GPU)
            batch = train_loader.get_random_batch(batch_size)
            
            # Forward pass
            def loss_fn(model):
                outputs = model(
                    batch['indicator_values'],
                    batch['market_context'],
                    batch['indicator_indices']
                )
                
                selection_logits = outputs['selection_logits']
                indicator_usefulness = batch['indicator_usefulness']
                
                # Use numerically stable sigmoid cross-entropy with logits
                # This is more stable than computing sigmoid then log
                selection_loss = mx.mean(
                    mx.maximum(selection_logits, 0) - 
                    selection_logits * indicator_usefulness + 
                    mx.log(1 + mx.exp(-mx.abs(selection_logits)))
                )
                
                # Add L2 regularization
                l2_loss = 0.0
                params = model.trainable_parameters()
                for param in params.values():
                    if hasattr(param, 'shape'):  # Check if it's an array
                        l2_loss = l2_loss + mx.sum(param ** 2)
                l2_loss = l2_loss * 1e-5
                
                return selection_loss + l2_loss
            
            # Compute gradients
            loss_and_grad_fn = mx.value_and_grad(loss_fn)
            loss, grads = loss_and_grad_fn(self.model)
            
            # Gradient clipping for stability
            max_grad_norm = 1.0
            grad_norm = 0.0
            for g in grads.values():
                if hasattr(g, 'shape'):
                    grad_norm += mx.sum(g ** 2)
            grad_norm = mx.sqrt(grad_norm)
            
            if grad_norm > max_grad_norm:
                scale = max_grad_norm / grad_norm
                grads = {k: v * scale for k, v in grads.items()}
            
            # Update weights
            self.optimizer.update(self.model, grads)
            
            epoch_loss += loss.item()
            
            # Force GPU sync every N batches to maintain high utilization
            if i % 10 == 0:
                mx.eval(self.model.trainable_parameters())
        
        # Validation
        val_loss = self.validate_gpu(val_loader)
        
        return epoch_loss / n_batches, val_loss
    
    def validate_gpu(self, val_loader):
        """Validate on GPU."""
        total_loss = 0.0
        n_batches = 20  # Fixed number of validation batches
        
        for _ in range(n_batches):
            batch = val_loader.get_random_batch(128)
            
            outputs = self.model(
                batch['indicator_values'],
                batch['market_context'],
                batch['indicator_indices']
            )
            
            selection_logits = outputs['selection_logits']
            indicator_usefulness = batch['indicator_usefulness']
            
            # Use same stable loss as training
            loss = mx.mean(
                mx.maximum(selection_logits, 0) - 
                selection_logits * indicator_usefulness + 
                mx.log(1 + mx.exp(-mx.abs(selection_logits)))
            )
            
            total_loss += loss.item()
        
        return total_loss / n_batches
    
    def train(self, train_loader, val_loader, epochs=20):
        """Full training loop optimized for GPU."""
        logger.info(f"\n🏃 Starting GPU-optimized training for {epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train one epoch
            train_loss, val_loss = self.train_epoch_gpu(train_loader, val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            # Log progress
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Time: {epoch_time:.1f}s")
            logger.info(f"  GPU Throughput: {len(train_loader) / epoch_time:.0f} samples/sec")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('models', 'best')
                logger.info("  ✓ Saved best model")
        
        return best_val_loss
    
    def save_model(self, save_path, suffix):
        """Save model weights."""
        os.makedirs(save_path, exist_ok=True)
        model_path = f"{save_path}/indicator_transformer_mlx_{suffix}.npz"
        
        # Get flattened weights
        weights = {}
        for name, param in self.model.trainable_parameters().items():
            weights[name] = np.array(param)
        
        np.savez(model_path, **weights)


def parallel_indicator_computation(data_batch, indicator_library, avwap_cache):
    """Compute indicators in parallel for a batch of symbols."""
    results = []
    
    for symbol, data in data_batch:
        try:
            # Compute AVWAP if not cached
            if symbol not in avwap_cache:
                calculator = OptimizedAVWAPCalculator(data)
                avwap_cache[symbol] = calculator.calculate_all_avwap_indicators()
            
            avwap_indicators = avwap_cache[symbol]
            
            # Compute all indicators
            all_indicators = np.zeros((len(data), len(indicator_library.indicators)), dtype=np.float32)
            
            for idx, (name, func) in enumerate(indicator_library.indicators.items()):
                try:
                    if name in avwap_indicators:
                        series = avwap_indicators[name]
                    else:
                        series = func(data)
                    
                    if len(series) == len(data):
                        all_indicators[:, idx] = series.values
                    else:
                        offset = len(data) - len(series)
                        all_indicators[offset:, idx] = series.values
                except:
                    pass
            
            results.append((symbol, data, all_indicators))
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    return results


def main():
    """GPU-optimized MLX training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU-Optimized MLX Training')
    parser.add_argument('--tickers-file', default='../tickers/james_tickers.json')
    parser.add_argument('--days', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)  # Larger for GPU
    parser.add_argument('--sample-interval', type=int, default=10)
    parser.add_argument('--parallel-workers', type=int, default=8)
    parser.add_argument('--min-samples', type=int, default=1000,
                       help='Minimum samples needed (compatibility)')
    
    args = parser.parse_args()
    
    # Load tickers
    with open(args.tickers_file, 'r') as f:
        tickers = json.load(f)
    
    logger.info("\n" + "="*80)
    logger.info("🚀 GPU-OPTIMIZED MLX TRAINING")
    logger.info("="*80)
    logger.info("🔥 Maximizing GPU utilization on Apple Silicon")
    
    # Initialize
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    db_provider = DatabaseDataProvider()
    indicator_library = IndicatorLibrary()
    indicator_library = integrate_with_indicator_library(indicator_library)
    num_indicators = len(indicator_library.indicators)
    
    # Load data for all symbols first
    logger.info(f"\n📥 Loading data for {len(tickers)} tickers...")
    symbol_data = []
    
    for symbol in tqdm(tickers, desc="Loading"):
        try:
            data = db_provider.get_minute_data(symbol, start_date, end_date)
            if len(data) >= 200:
                symbol_data.append((symbol, data))
        except:
            pass
    
    logger.info(f"✅ Loaded {len(symbol_data)} valid tickers")
    
    # Parallel indicator computation
    logger.info("\n⚡ Computing indicators in parallel...")
    all_samples = []
    all_indicator_values = []
    avwap_cache = {}
    
    # Show summary of indicators being computed
    logger.info(f"\n📊 Indicators to compute:")
    indicator_names = list(indicator_library.indicators.keys())
    avwap_indicators = [name for name in indicator_names if 'AVWAP' in name]
    other_indicators = [name for name in indicator_names if 'AVWAP' not in name]
    logger.info(f"  • {len(avwap_indicators)} AVWAP indicators")
    logger.info(f"  • {len(other_indicators)} other indicators")
    logger.info(f"  • Total: {len(indicator_names)} indicators")
    
    # Process in batches with thread pool optimized for Apple Silicon
    from apple_silicon_optimizer import create_optimized_thread_pool
    optimal_workers = create_optimized_thread_pool('compute')
    batch_size = optimal_workers
    
    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        futures = []
        
        for i in range(0, len(symbol_data), batch_size):
            batch = symbol_data[i:i+batch_size]
            future = executor.submit(parallel_indicator_computation, batch, indicator_library, avwap_cache)
            futures.append(future)
        
        # Collect results
        processed_symbols = []
        for batch_idx, future in enumerate(futures):
            # Show progress at batch level only
            if batch_idx == 0:
                logger.info(f"\n🔄 Processing {len(symbol_data)} symbols in {len(futures)} batches...")
            results = future.result()
            
            # Track processed symbols
            batch_symbols = [symbol for symbol, _, _ in results]
            processed_symbols.extend(batch_symbols)
            
            # Summary logging every few batches
            if (batch_idx + 1) % 2 == 0 or batch_idx == len(futures) - 1:
                logger.info(f"  • Processed {len(processed_symbols)}/{len(symbol_data)} symbols")
            
            for symbol, data, all_indicators in results:
                all_indicator_values.append(all_indicators)
                
                # Create samples
                window_size = 100
                future_window = 20
                
                for i in range(window_size, len(data) - future_window, args.sample_interval):
                    try:
                        hist_data = data.iloc[i-window_size:i]
                        
                        # Market context
                        returns = hist_data['close'].pct_change().dropna()
                        market_context = np.array([
                            np.clip(returns.mean() * 10000, -100, 100),
                            np.clip(returns.std() * 10000, 0, 100),
                            np.clip(returns.skew(), -3, 3),
                            np.clip(returns.kurt(), -3, 10),
                            np.clip((hist_data['close'].iloc[-1] / hist_data['close'].iloc[0] - 1) * 10000, -500, 500),
                            np.log1p(hist_data['volume'].mean() / 1e6),
                            np.clip((hist_data['high'] - hist_data['low']).mean() / hist_data['close'].mean() * 10000, 0, 200),
                            len(returns[returns > 0]) / len(returns),
                            np.clip(returns.iloc[-10:].mean() * 10000, -100, 100),
                            np.clip(hist_data['volume'].iloc[-10:].mean() / hist_data['volume'].mean(), 0.1, 10)
                        ], dtype=np.float32)
                        
                        market_context = np.nan_to_num(market_context, 0)
                        
                        # Future return
                        future_return = (data['close'].iloc[i+future_window-1] / data['close'].iloc[i-1] - 1)
                        future_return = np.clip(future_return, -0.1, 0.1)
                        
                        # Indicator usefulness
                        usefulness = np.zeros(num_indicators, dtype=np.float32)
                        
                        for j, name in enumerate(indicator_library.indicators.keys()):
                            if 'AVWAP' in name:
                                usefulness[j] = 0.9 if 'SESSION' in name else 0.8
                            elif 'VWAP' in name:
                                usefulness[j] = 0.8
                            elif 'RSI' in name and ('7' in name or '9' in name):
                                usefulness[j] = 0.7
                            else:
                                usefulness[j] = 0.5
                        
                        all_samples.append({
                            'indicator_values': all_indicators[i].copy(),
                            'market_context': market_context,
                            'future_return': future_return,
                            'indicator_usefulness': usefulness
                        })
                        
                    except:
                        pass
    
    logger.info(f"\n✅ Created {len(all_samples)} training samples")
    
    # Check minimum samples
    if len(all_samples) < args.min_samples:
        logger.error(f"Insufficient samples: {len(all_samples)} < {args.min_samples}")
        logger.info(f"\nSuggestions:")
        logger.info(f"  • Increase --days (currently {args.days})")
        logger.info(f"  • Decrease --sample-interval (currently {args.sample_interval})")
        logger.info(f"  • Add more tickers")
        return
    
    # Normalize
    logger.info("\n📊 Normalizing indicators...")
    indicator_scaler = StandardScaler()
    all_indicator_concat = np.vstack(all_indicator_values)
    
    # Check for NaN/Inf values before normalization
    if np.any(np.isnan(all_indicator_concat)) or np.any(np.isinf(all_indicator_concat)):
        logger.warning("Found NaN/Inf values in indicators, replacing with 0")
        all_indicator_concat = np.nan_to_num(all_indicator_concat, nan=0.0, posinf=1.0, neginf=-1.0)
    
    indicator_scaler.fit(all_indicator_concat)
    
    for sample in all_samples:
        normalized = indicator_scaler.transform(
            sample['indicator_values'].reshape(1, -1)
        ).flatten().astype(np.float32)
        # Replace any remaining NaN/Inf
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=3.0, neginf=-3.0)
        sample['indicator_values'] = normalized
    
    # Split data
    np.random.shuffle(all_samples)
    val_size = int(len(all_samples) * 0.2)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    logger.info(f"   Train: {len(train_samples)} samples")
    logger.info(f"   Val: {len(val_samples)} samples")
    
    # Create GPU-optimized data loaders
    logger.info("\n🚀 Creating GPU-optimized data loaders...")
    train_loader = GPUOptimizedDataLoader(train_samples, batch_size=args.batch_size)
    val_loader = GPUOptimizedDataLoader(val_samples, batch_size=args.batch_size)
    
    # Create model
    logger.info("\n🏗️  Creating model on GPU...")
    model = MLXIndicatorTransformer(
        num_indicators=num_indicators,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    # Train with GPU optimization
    logger.info("\n🔥 Starting GPU-optimized training...")
    trainer = GPUOptimizedTrainer(model, learning_rate=1e-4)
    
    start_time = time.time()
    best_val_loss = trainer.train(train_loader, val_loader, epochs=args.epochs)
    train_time = time.time() - start_time
    
    # Save scaler
    import pickle
    os.makedirs('models', exist_ok=True)
    with open('models/indicator_scaler_gpu.pkl', 'wb') as f:
        pickle.dump(indicator_scaler, f)
    
    logger.info(f"\n" + "="*80)
    logger.info("✅ GPU-OPTIMIZED TRAINING COMPLETE!")
    logger.info("="*80)
    
    logger.info(f"\n📊 Performance Summary:")
    logger.info(f"   Total time: {train_time/60:.1f} minutes")
    logger.info(f"   Throughput: {len(train_samples) * args.epochs / train_time:.0f} samples/sec")
    logger.info(f"   GPU efficiency: ~{args.batch_size * args.epochs * 60 / train_time:.0f} batches/min")
    
    logger.info(f"\n🎯 Optimizations used:")
    logger.info(f"   • Larger batch size ({args.batch_size}) for GPU")
    logger.info(f"   • Parallel indicator computation ({args.parallel_workers} workers)")
    logger.info(f"   • Pre-loaded GPU memory")
    logger.info(f"   • Optimized data pipeline")
    logger.info(f"   • Forced GPU device for all operations")


if __name__ == "__main__":
    main()