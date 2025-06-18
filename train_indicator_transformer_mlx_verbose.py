#!/usr/bin/env python3
"""
Train Indicator Transformer Model with MLX (Verbose)
===================================================
MLX training with detailed progress reporting.
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

# Enhanced logging for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Make logging immediate
import functools
for handler in logger.handlers:
    handler.flush = functools.partial(handler.flush)


class VerboseMLXDataset:
    """MLX Dataset with detailed progress reporting."""
    
    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime,
                 window_size: int = 100, future_window: int = 20, 
                 use_daily: bool = False, sample_interval: int = 240):
        """Initialize with verbose logging."""
        
        self.symbols = symbols
        self.window_size = window_size
        self.future_window = future_window
        self.use_daily = use_daily
        self.sample_interval = sample_interval
        
        logger.info(f"\n📋 Dataset Configuration:")
        logger.info(f"   Window size: {window_size} {'days' if use_daily else 'minutes'}")
        logger.info(f"   Future window: {future_window} {'days' if use_daily else 'minutes'}")
        logger.info(f"   Sample interval: {'daily' if use_daily else f'{sample_interval} minutes'}")
        logger.info(f"   Date range: {start_date.date()} to {end_date.date()}")
        
        # Initialize components
        logger.info(f"\n🔧 Initializing components...")
        
        logger.info("   Loading database provider...")
        self.db_provider = DatabaseDataProvider()
        
        logger.info("   Loading indicator library...")
        self.indicator_library = IndicatorLibrary()
        logger.info(f"   Found {len(self.indicator_library.indicators)} indicators")
        
        logger.info("   Loading market analyzer...")
        self.market_analyzer = MarketRegimeAnalyzer()
        
        # Load and prepare data
        logger.info(f"\n📊 Starting data loading process...")
        self.samples = []
        self._prepare_samples_verbose(start_date, end_date)
        
        # Convert to MLX arrays
        if self.samples:
            logger.info(f"\n🔄 Converting {len(self.samples)} samples to MLX arrays...")
            self._prepare_mlx_arrays()
            logger.info("✅ MLX arrays ready")
        else:
            logger.warning("⚠️  No samples created - check data availability")
    
    def _prepare_samples_verbose(self, start_date: datetime, end_date: datetime):
        """Prepare samples with detailed progress reporting."""
        total_start = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(self.symbols)} symbols")
        logger.info(f"{'='*60}")
        
        for idx, symbol in enumerate(self.symbols):
            logger.info(f"\n[{idx+1}/{len(self.symbols)}] Processing {symbol}")
            logger.info(f"{'─'*40}")
            
            try:
                symbol_start = time.time()
                
                # Process symbol with detailed progress
                symbol_samples = self._process_symbol_verbose(symbol, start_date, end_date)
                
                symbol_time = time.time() - symbol_start
                
                if symbol_samples:
                    self.samples.extend(symbol_samples)
                    logger.info(f"✅ {symbol}: {len(symbol_samples)} samples created in {symbol_time:.1f}s")
                else:
                    logger.warning(f"⚠️  {symbol}: No samples created")
                
            except Exception as e:
                logger.error(f"❌ {symbol}: Failed - {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        total_time = time.time() - total_start
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Data Loading Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Total samples: {len(self.samples)}")
        logger.info(f"Average time per symbol: {total_time/len(self.symbols):.1f}s")
    
    def _process_symbol_verbose(self, symbol: str, start_date: datetime, 
                                end_date: datetime) -> List[Dict]:
        """Process symbol with detailed progress updates."""
        symbol_samples = []
        
        # Calculate data requirements
        logger.info(f"   📅 Date range: {start_date.date()} to {end_date.date()}")
        
        # Get data with buffer
        buffer_days = 30 if self.use_daily else 1
        buffer_start = start_date - timedelta(days=buffer_days)
        logger.info(f"   📅 Buffer start: {buffer_start.date()}")
        
        # Load data
        logger.info(f"   📥 Loading {'daily' if self.use_daily else 'minute'} data from database...")
        data_start = time.time()
        
        if self.use_daily:
            data = self._get_daily_data_verbose(symbol, buffer_start, end_date)
            sample_step = 1
        else:
            logger.info(f"   🔍 Querying minute data...")
            data = self.db_provider.get_minute_data(symbol, buffer_start, end_date)
            logger.info(f"   ✅ Retrieved {len(data)} minute data points in {time.time()-data_start:.1f}s")
            sample_step = self.sample_interval
        
        # Check data sufficiency
        min_required = self.window_size + self.future_window
        logger.info(f"   📏 Data points: {len(data)} (minimum required: {min_required})")
        
        if len(data) < min_required:
            logger.warning(f"   ⚠️  Insufficient data: {len(data)} < {min_required}")
            return symbol_samples
        
        # Pre-compute indicators
        logger.info(f"   🧮 Computing all indicators...")
        indicator_start = time.time()
        
        all_indicators = self._compute_all_indicators_verbose(data, symbol)
        
        indicator_time = time.time() - indicator_start
        logger.info(f"   ✅ Indicators computed in {indicator_time:.1f}s")
        
        # Create samples
        num_possible_samples = (len(data) - self.window_size - self.future_window) // sample_step
        logger.info(f"   🎯 Creating up to {num_possible_samples} samples...")
        
        sample_creation_start = time.time()
        samples_created = 0
        
        # Progress bar for sample creation
        sample_indices = range(self.window_size, len(data) - self.future_window, sample_step)
        
        with tqdm(sample_indices, desc=f"   Creating samples for {symbol}", 
                  ncols=80, leave=False) as pbar:
            for i in pbar:
                try:
                    # Get windows
                    hist_window = slice(i - self.window_size, i)
                    future_window = slice(i, i + self.future_window)
                    
                    # Get pre-computed indicator values
                    indicator_values = all_indicators[i]
                    
                    # Market context
                    hist_data = data.iloc[hist_window]
                    market_context = self._get_simple_market_context(hist_data)
                    
                    # Future return
                    future_return = (data['close'].iloc[future_window].iloc[-1] / 
                                   data['close'].iloc[hist_window].iloc[-1]) - 1
                    
                    # Indicator usefulness
                    indicator_usefulness = self._get_simple_usefulness(
                        indicator_values, future_return
                    )
                    
                    symbol_samples.append({
                        'indicator_values': indicator_values,
                        'market_context': market_context,
                        'future_return': future_return,
                        'indicator_usefulness': indicator_usefulness
                    })
                    
                    samples_created += 1
                    pbar.set_postfix({'created': samples_created})
                    
                except Exception as e:
                    logger.debug(f"Sample error at index {i}: {e}")
                    continue
        
        sample_time = time.time() - sample_creation_start
        logger.info(f"   ✅ Created {samples_created} samples in {sample_time:.1f}s")
        
        return symbol_samples
    
    def _get_daily_data_verbose(self, symbol: str, start_date: datetime, 
                                end_date: datetime) -> pd.DataFrame:
        """Get daily data with progress updates."""
        try:
            logger.info(f"   🔍 Attempting to load minute data for aggregation...")
            minute_data = self.db_provider.get_minute_data(symbol, start_date, end_date)
            
            if minute_data.empty:
                logger.warning(f"   ⚠️  No minute data found, using mock data")
                return self._generate_mock_daily_data(symbol, start_date, end_date)
            
            logger.info(f"   📊 Aggregating {len(minute_data)} minute records to daily...")
            
            # Aggregate to daily
            daily_data = minute_data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            logger.info(f"   ✅ Created {len(daily_data)} daily records")
            return daily_data
            
        except Exception as e:
            logger.warning(f"   ❌ Failed to get daily data: {e}")
            logger.info(f"   🔄 Using mock data instead")
            return self._generate_mock_daily_data(symbol, start_date, end_date)
    
    def _compute_all_indicators_verbose(self, data: pd.DataFrame, symbol: str) -> np.ndarray:
        """Compute indicators with progress reporting."""
        num_indicators = len(self.indicator_library.indicators)
        num_points = len(data)
        
        logger.info(f"      Computing {num_indicators} indicators for {num_points} data points...")
        
        # Pre-allocate array
        all_values = np.zeros((num_points, num_indicators), dtype=np.float32)
        
        # Progress bar for indicators
        with tqdm(enumerate(self.indicator_library.indicators.items()), 
                  total=num_indicators, desc="      Computing indicators", 
                  ncols=80, leave=False) as pbar:
            
            for idx, (name, _) in pbar:
                pbar.set_postfix({'indicator': name[:20]})
                
                try:
                    # Compute indicator
                    indicator_series = self.indicator_library.compute_indicator(data, name)
                    
                    # Store values
                    if len(indicator_series) == len(data):
                        all_values[:, idx] = indicator_series.values
                    else:
                        offset = len(data) - len(indicator_series)
                        all_values[offset:, idx] = indicator_series.values
                        
                except Exception as e:
                    logger.debug(f"Failed to compute {name}: {e}")
        
        return all_values
    
    def _generate_mock_daily_data(self, symbol: str, start_date: datetime, 
                                  end_date: datetime) -> pd.DataFrame:
        """Generate mock daily data."""
        logger.info(f"   🎲 Generating mock daily data...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
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
        
        logger.info(f"   ✅ Generated {len(df)} days of mock data")
        return df
    
    def _get_simple_market_context(self, data: pd.DataFrame) -> np.ndarray:
        """Get market context features."""
        try:
            returns = data['close'].pct_change().dropna()
            
            context = np.array([
                returns.mean() * 100,
                returns.std() * 100,
                returns.skew(),
                returns.kurt(),
                (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
                data['volume'].mean() / 1e6,
                (data['high'] - data['low']).mean() / data['close'].mean() * 100,
                len(returns[returns > 0]) / len(returns),
                returns.iloc[-5:].mean() * 100,
                data['close'].rolling(20).std().iloc[-1] / data['close'].iloc[-1] * 100
            ], dtype=np.float32)
            
            return np.nan_to_num(context, 0)
            
        except Exception:
            return np.zeros(10, dtype=np.float32)
    
    def _get_simple_usefulness(self, indicator_values: np.ndarray, 
                               future_return: float) -> np.ndarray:
        """Calculate indicator usefulness."""
        usefulness = np.zeros_like(indicator_values)
        
        if abs(future_return) > 0.02:
            usefulness[0:5] = 0.8
            usefulness[5:10] = 0.6
            usefulness[10:15] = 0.4
        else:
            usefulness[15:20] = 0.6
            usefulness[20:25] = 0.4
        
        usefulness += np.random.rand(len(usefulness)) * 0.2
        usefulness = np.clip(usefulness, 0, 1)
        
        return usefulness
    
    def _prepare_mlx_arrays(self):
        """Convert samples to MLX arrays."""
        if not self.samples:
            return
        
        # Convert to lists first for MLX compatibility
        indicator_values_list = [s['indicator_values'].tolist() for s in self.samples]
        market_contexts_list = [s['market_context'].tolist() for s in self.samples]
        indicator_usefulness_list = [s['indicator_usefulness'].tolist() for s in self.samples]
        future_returns_list = [[s['future_return']] for s in self.samples]
        
        # Create MLX arrays
        self.indicator_values = mx.array(indicator_values_list)
        self.market_contexts = mx.array(market_contexts_list)
        self.indicator_usefulness = mx.array(indicator_usefulness_list)
        self.future_returns = mx.array(future_returns_list)
        
        # Create indicator indices
        num_indicators = self.indicator_values.shape[1]
        self.indicator_indices = mx.arange(num_indicators)
        
        logger.info(f"📦 MLX arrays created: {self.indicator_values.shape}")
    
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


# Import the model and trainer from the fixed version
from train_indicator_transformer_mlx_fixed import (
    MLXIndicatorTransformer,
    TransformerBlock,
    MultiHeadAttention,
    MLXTrainer
)


def main():
    """Main training function with verbose logging."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Indicator Transformer with MLX (Verbose)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on')
    parser.add_argument('--days', type=int, default=180, help='Days of historical data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--use-daily', action='store_true', help='Use daily data')
    parser.add_argument('--sample-interval', type=int, default=240, help='Minutes between samples')
    
    args = parser.parse_args()
    
    # Default symbols
    if not args.symbols:
        args.symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    logger.info("\n" + "="*80)
    logger.info("🚀 MLX INDICATOR TRANSFORMER TRAINING (VERBOSE)")
    logger.info("="*80)
    
    logger.info(f"\n📋 Training Configuration:")
    logger.info(f"   Symbols: {', '.join(args.symbols)}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Days of history: {args.days}")
    logger.info(f"   Data type: {'Daily' if args.use_daily else 'Minute'}")
    
    # System info
    logger.info(f"\n💻 System Information:")
    logger.info(f"   CPU cores: {os.cpu_count()}")
    logger.info(f"   Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    logger.info(f"   Python version: {sys.version.split()[0]}")
    logger.info(f"   MLX backend: Metal (unified memory)")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Create dataset
    logger.info(f"\n📥 STARTING DATA LOADING PHASE")
    logger.info(f"{'='*60}")
    
    data_start = time.time()
    
    dataset = VerboseMLXDataset(
        args.symbols, 
        start_date, 
        end_date,
        use_daily=args.use_daily,
        sample_interval=args.sample_interval
    )
    
    data_time = time.time() - data_start
    logger.info(f"\n⏱️  Total data loading time: {data_time/60:.1f} minutes")
    
    if len(dataset) == 0:
        logger.error("\n❌ No training samples created!")
        logger.error("Possible issues:")
        logger.error("   1. Insufficient historical data in database")
        logger.error("   2. Window size too large for available data")
        logger.error("   3. Database connection issues")
        logger.error("\nSolutions:")
        logger.error("   1. Use --days 250 for more history")
        logger.error("   2. Use --use-daily flag")
        logger.error("   3. Check database connectivity")
        return
    
    # Split into train/val
    logger.info(f"\n📊 Splitting data into train/validation sets...")
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train/val datasets
    train_dataset = VerboseMLXDataset.__new__(VerboseMLXDataset)
    train_dataset.__dict__ = dataset.__dict__.copy()
    train_dataset.samples = [dataset.samples[i] for i in train_indices]
    train_dataset._prepare_mlx_arrays()
    
    val_dataset = VerboseMLXDataset.__new__(VerboseMLXDataset)
    val_dataset.__dict__ = dataset.__dict__.copy()
    val_dataset.samples = [dataset.samples[i] for i in val_indices]
    val_dataset._prepare_mlx_arrays()
    
    logger.info(f"✅ Train samples: {len(train_dataset)}")
    logger.info(f"✅ Validation samples: {len(val_dataset)}")
    
    # Initialize model
    logger.info(f"\n🏗️  Creating MLX model...")
    indicator_library = IndicatorLibrary()
    num_indicators = len(indicator_library.indicators)
    
    model = MLXIndicatorTransformer(
        num_indicators=num_indicators,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    logger.info(f"✅ Model created with {num_indicators} indicators")
    
    # Train
    logger.info(f"\n🏃 STARTING TRAINING PHASE")
    logger.info(f"{'='*60}")
    
    trainer = MLXTrainer(model)
    trainer.train(train_dataset, val_dataset, epochs=args.epochs, batch_size=args.batch_size)
    
    logger.info("\n✅ Training complete!")
    logger.info("📁 Model saved to models/indicator_transformer_mlx_best.npz")


if __name__ == "__main__":
    main()