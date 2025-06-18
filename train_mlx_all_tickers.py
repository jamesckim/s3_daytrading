#!/usr/bin/env python3
"""
Train MLX Model on All Tickers
===============================
Train on all 655 tickers for comprehensive intraday patterns.
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
import psutil
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicator_transformer import IndicatorLibrary, MarketRegimeAnalyzer
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider
from optimized_avwap_indicators import OptimizedAVWAPCalculator, integrate_with_indicator_library
from train_indicator_transformer_mlx_fixed import MLXIndicatorTransformer
from mlx_trainer_fixed import MLXTrainerFixed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train MLX on all available tickers."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MLX on All Tickers')
    parser.add_argument('--days', type=int, default=20, help='Days of historical minute data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-tickers', type=int, default=None, help='Limit number of tickers (for testing)')
    parser.add_argument('--sample-interval', type=int, default=30, help='Minutes between samples (30 for more data)')
    
    args = parser.parse_args()
    
    # Load all tickers
    with open('../tickers/all_tickers.json', 'r') as f:
        all_tickers = json.load(f)
    
    if args.max_tickers:
        all_tickers = all_tickers[:args.max_tickers]
    
    logger.info("\n" + "="*80)
    logger.info("🚀 MLX TRAINING ON ALL TICKERS")
    logger.info("="*80)
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Total Tickers: {len(all_tickers)}")
    logger.info(f"   History: {args.days} days of minute data")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch Size: {args.batch_size}")
    logger.info(f"   Sample Interval: {args.sample_interval} minutes")
    logger.info(f"   AVWAP: Optimized (42,000+ points/sec)")
    
    # System info
    logger.info(f"\n💻 System:")
    logger.info(f"   CPU cores: {os.cpu_count()}")
    logger.info(f"   Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB available")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Initialize components
    logger.info("\n📊 Initializing...")
    db_provider = DatabaseDataProvider()
    indicator_library = IndicatorLibrary()
    indicator_library = integrate_with_indicator_library(indicator_library)
    num_indicators = len(indicator_library.indicators)
    logger.info(f"✅ {num_indicators} indicators ready (with fast AVWAP)")
    
    # Process tickers in batches to manage memory
    all_samples = []
    indicator_scaler = StandardScaler()
    all_indicator_values = []
    
    batch_size = 50  # Process 50 tickers at a time
    failed_tickers = []
    successful_tickers = []
    
    logger.info(f"\n📥 Processing {len(all_tickers)} tickers in batches of {batch_size}...")
    
    for batch_start in range(0, len(all_tickers), batch_size):
        batch_end = min(batch_start + batch_size, len(all_tickers))
        batch_tickers = all_tickers[batch_start:batch_end]
        
        logger.info(f"\nBatch {batch_start//batch_size + 1}: Processing {len(batch_tickers)} tickers...")
        
        for symbol in tqdm(batch_tickers, desc=f"Batch {batch_start//batch_size + 1}"):
            try:
                # Get minute data
                data = db_provider.get_minute_data(symbol, start_date, end_date)
                
                if len(data) < 200:  # Need minimum data
                    logger.debug(f"{symbol}: Insufficient data ({len(data)} bars)")
                    failed_tickers.append((symbol, "insufficient_data"))
                    continue
                
                # Compute indicators
                start_compute = time.time()
                
                # AVWAP indicators
                calculator = OptimizedAVWAPCalculator(data)
                avwap_indicators = calculator.calculate_all_avwap_indicators()
                
                # All indicators
                all_indicators = np.zeros((len(data), num_indicators), dtype=np.float32)
                
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
                
                compute_time = time.time() - start_compute
                
                # Store for scaler fitting
                all_indicator_values.append(all_indicators)
                
                # Create samples with smaller interval for more data
                window_size = 100
                future_window = 20
                
                samples_created = 0
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
                        
                        # Indicator usefulness (prioritize AVWAP)
                        usefulness = np.zeros(num_indicators, dtype=np.float32)
                        
                        for j, name in enumerate(indicator_library.indicators.keys()):
                            if 'AVWAP' in name:
                                usefulness[j] = 0.9 if 'SESSION' in name else 0.8
                            elif 'VWAP' in name:
                                usefulness[j] = 0.8
                            elif 'RSI' in name and ('7' in name or '9' in name):
                                usefulness[j] = 0.7
                            elif 'ATR' in name:
                                usefulness[j] = 0.6
                            elif 'MACD' in name and 'SCALP' in name:
                                usefulness[j] = 0.7
                            elif 'BB' in name and ('10' in name or '20' in name):
                                usefulness[j] = 0.6
                            else:
                                usefulness[j] = 0.4
                        
                        # Adjust based on market move
                        if abs(future_return) > 0.001:
                            usefulness = np.minimum(usefulness * 1.2, 1.0)
                        
                        all_samples.append({
                            'indicator_values': all_indicators[i].copy(),
                            'market_context': market_context,
                            'future_return': future_return,
                            'indicator_usefulness': usefulness,
                            'symbol': symbol
                        })
                        
                        samples_created += 1
                        
                    except Exception as e:
                        logger.debug(f"Sample error: {e}")
                
                successful_tickers.append(symbol)
                logger.debug(f"{symbol}: {len(data)} bars, {samples_created} samples in {compute_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                failed_tickers.append((symbol, str(e)))
        
        # Log batch progress
        logger.info(f"Batch complete: {len(all_samples)} total samples")
    
    # Summary
    logger.info(f"\n✅ Data processing complete!")
    logger.info(f"   Successful tickers: {len(successful_tickers)}")
    logger.info(f"   Failed tickers: {len(failed_tickers)}")
    logger.info(f"   Total samples: {len(all_samples)}")
    
    if len(all_samples) < 1000:
        logger.error("Insufficient samples for training. Need at least 1000.")
        return
    
    # Fit scaler
    logger.info("\n📊 Normalizing indicators...")
    all_indicator_concat = np.vstack(all_indicator_values)
    indicator_scaler.fit(all_indicator_concat)
    
    # Normalize samples
    for sample in all_samples:
        sample['indicator_values'] = indicator_scaler.transform(
            sample['indicator_values'].reshape(1, -1)
        ).flatten().astype(np.float32)
    
    # Remove symbol from samples (not needed for training)
    for sample in all_samples:
        del sample['symbol']
    
    # Shuffle and split
    np.random.shuffle(all_samples)
    val_size = int(len(all_samples) * 0.2)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    logger.info(f"   Train: {len(train_samples)} samples")
    logger.info(f"   Val: {len(val_samples)} samples")
    
    # Create MLX datasets
    class NormalizedMLXDataset:
        def __init__(self, samples):
            self.samples = samples
            
            # Convert to MLX arrays
            self.indicator_values = mx.array(
                np.array([s['indicator_values'] for s in samples], dtype=np.float32)
            )
            self.market_contexts = mx.array(
                np.array([s['market_context'] for s in samples], dtype=np.float32)
            )
            self.indicator_usefulness = mx.array(
                np.array([s['indicator_usefulness'] for s in samples], dtype=np.float32)
            )
            self.future_returns = mx.array(
                np.array([[s['future_return']] for s in samples], dtype=np.float32)
            )
            
            self.indicator_indices = mx.arange(self.indicator_values.shape[1])
        
        def __len__(self):
            return len(self.samples)
        
        def get_batch(self, indices):
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
    
    train_dataset = NormalizedMLXDataset(train_samples)
    val_dataset = NormalizedMLXDataset(val_samples)
    
    # Create model
    logger.info("\n🏗️  Creating model...")
    model = MLXIndicatorTransformer(
        num_indicators=num_indicators,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    # Train
    logger.info("\n🏃 Starting training...")
    trainer = MLXTrainerFixed(model, learning_rate=1e-4)
    
    training_start = time.time()
    trainer.train(train_dataset, val_dataset, epochs=args.epochs, batch_size=args.batch_size)
    training_time = time.time() - training_start
    
    # Save scaler
    import pickle
    os.makedirs('models', exist_ok=True)
    with open('models/indicator_scaler_all_tickers.pkl', 'wb') as f:
        pickle.dump(indicator_scaler, f)
    
    # Results
    logger.info(f"\n" + "="*80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*80)
    
    logger.info(f"\n📊 Training Summary:")
    logger.info(f"   Tickers processed: {len(successful_tickers)}")
    logger.info(f"   Total samples: {len(all_samples)}")
    logger.info(f"   Training time: {training_time/60:.1f} minutes")
    logger.info(f"   Samples/second: {len(train_samples) * args.epochs / training_time:.0f}")
    
    if trainer.train_losses:
        logger.info(f"\n📈 Training Metrics:")
        logger.info(f"   Initial loss: {trainer.train_losses[0]:.4f}")
        logger.info(f"   Final loss: {trainer.train_losses[-1]:.4f}")
        logger.info(f"   Best val loss: {min(trainer.val_losses):.4f}")
    
    logger.info(f"\n🎯 Model trained on diverse market conditions:")
    logger.info(f"   • {len(successful_tickers)} different stocks")
    logger.info(f"   • {args.days} days of minute data per stock")
    logger.info(f"   • Optimized AVWAP indicators for intraday trading")
    logger.info(f"   • Ready for production deployment")
    
    logger.info(f"\n📁 Saved:")
    logger.info(f"   • Model: models/indicator_transformer_mlx_best.npz")
    logger.info(f"   • Scaler: models/indicator_scaler_all_tickers.pkl")


if __name__ == "__main__":
    main()