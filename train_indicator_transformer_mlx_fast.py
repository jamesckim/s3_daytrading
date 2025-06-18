#!/usr/bin/env python3
"""
Train Indicator Transformer Model with MLX (Fast AVWAP)
=======================================================
MLX training with optimized AVWAP computation for intraday trading.
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
from optimized_avwap_indicators import OptimizedAVWAPCalculator, integrate_with_indicator_library

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastMLXDataset:
    """MLX Dataset with optimized indicator computation."""
    
    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime,
                 window_size: int = 100, future_window: int = 20, 
                 use_daily: bool = False, sample_interval: int = 240):
        """Initialize with fast AVWAP computation."""
        
        self.symbols = symbols
        self.window_size = window_size
        self.future_window = future_window
        self.use_daily = use_daily
        self.sample_interval = sample_interval
        
        logger.info(f"\n📋 Dataset Configuration:")
        logger.info(f"   Window size: {window_size} {'days' if use_daily else 'minutes'}")
        logger.info(f"   Sample interval: {sample_interval} minutes")
        logger.info(f"   Optimized AVWAP computation: ENABLED")
        
        # Initialize components
        self.db_provider = DatabaseDataProvider()
        
        # Initialize indicator library with optimized AVWAP
        self.indicator_library = IndicatorLibrary()
        self.indicator_library = integrate_with_indicator_library(self.indicator_library)
        logger.info(f"   Indicators: {len(self.indicator_library.indicators)} (with optimized AVWAP)")
        
        self.market_analyzer = MarketRegimeAnalyzer()
        
        # Load and prepare data
        logger.info(f"\n📊 Loading data for {len(symbols)} symbols...")
        self.samples = []
        self._prepare_samples_fast(start_date, end_date)
        logger.info(f"✅ Prepared {len(self.samples)} training samples")
        
        # Convert to MLX arrays
        if self.samples:
            self._prepare_mlx_arrays()
    
    def _prepare_samples_fast(self, start_date: datetime, end_date: datetime):
        """Prepare samples with optimized computation."""
        total_start = time.time()
        
        for symbol in tqdm(self.symbols, desc="Loading symbols"):
            try:
                symbol_start = time.time()
                symbol_samples = self._process_symbol_fast(symbol, start_date, end_date)
                symbol_time = time.time() - symbol_start
                
                self.samples.extend(symbol_samples)
                logger.info(f"{symbol}: {len(symbol_samples)} samples in {symbol_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        total_time = time.time() - total_start
        logger.info(f"Data loading completed in {total_time:.1f}s")
    
    def _process_symbol_fast(self, symbol: str, start_date: datetime, 
                             end_date: datetime) -> List[Dict]:
        """Process symbol with fast indicator computation."""
        symbol_samples = []
        
        # Get data with minimal buffer for AVWAP
        buffer_days = 5 if self.use_daily else 1  # Less buffer needed
        buffer_start = start_date - timedelta(days=buffer_days)
        
        # Load minute data
        logger.debug(f"Loading data for {symbol}...")
        data = self.db_provider.get_minute_data(symbol, buffer_start, end_date)
        
        if len(data) < self.window_size + self.future_window:
            logger.warning(f"{symbol}: Insufficient data ({len(data)} points)")
            return symbol_samples
        
        # Pre-compute all indicators efficiently
        logger.debug(f"Computing indicators for {symbol}...")
        all_indicators = self._compute_all_indicators_fast(data, symbol)
        
        # Create samples
        num_samples = (len(data) - self.window_size - self.future_window) // self.sample_interval
        logger.debug(f"Creating {num_samples} samples for {symbol}...")
        
        for i in range(self.window_size, len(data) - self.future_window, self.sample_interval):
            try:
                # Get windows
                hist_window = slice(i - self.window_size, i)
                future_window = slice(i, i + self.future_window)
                
                # Get pre-computed indicator values
                indicator_values = all_indicators[i]
                
                # Market context
                hist_data = data.iloc[hist_window]
                market_context = self._get_market_context_fast(hist_data)
                
                # Future return
                future_return = (data['close'].iloc[future_window].iloc[-1] / 
                               data['close'].iloc[hist_window].iloc[-1]) - 1
                
                # Indicator usefulness for intraday
                indicator_usefulness = self._get_intraday_usefulness(
                    indicator_values, future_return, i, data
                )
                
                symbol_samples.append({
                    'indicator_values': indicator_values,
                    'market_context': market_context,
                    'future_return': future_return,
                    'indicator_usefulness': indicator_usefulness
                })
                
            except Exception as e:
                logger.debug(f"Sample error: {e}")
                continue
        
        return symbol_samples
    
    def _compute_all_indicators_fast(self, data: pd.DataFrame, symbol: str) -> np.ndarray:
        """Compute indicators with optimized AVWAP."""
        num_indicators = len(self.indicator_library.indicators)
        num_points = len(data)
        
        # Pre-allocate array
        all_values = np.zeros((num_points, num_indicators), dtype=np.float32)
        
        # Group indicators by type for batch computation
        vwap_indicators = []
        regular_indicators = []
        
        for name in self.indicator_library.indicators:
            if 'VWAP' in name or 'AVWAP' in name:
                vwap_indicators.append(name)
            else:
                regular_indicators.append(name)
        
        logger.debug(f"Computing {len(vwap_indicators)} VWAP indicators (optimized)...")
        
        # Compute all VWAP/AVWAP indicators at once
        if vwap_indicators:
            calculator = OptimizedAVWAPCalculator(data)
            vwap_results = calculator.calculate_all_avwap_indicators()
            
            # Also compute standard VWAP
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
            vwap_results['VWAP'] = vwap
            
            # VWAP bands
            for num_std in [1, 2, 3]:
                deviation = typical_price - vwap
                rolling_std = deviation.rolling(20).std()
                vwap_results[f'VWAP_U{num_std}'] = vwap + num_std * rolling_std
                vwap_results[f'VWAP_L{num_std}'] = vwap - num_std * rolling_std
        
        logger.debug(f"Computing {len(regular_indicators)} regular indicators...")
        
        # Compute indicators
        for idx, name in enumerate(self.indicator_library.indicators):
            try:
                if name in vwap_results:
                    # Use pre-computed VWAP result
                    indicator_series = vwap_results[name]
                else:
                    # Compute regular indicator
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
        
        return all_values
    
    def _get_market_context_fast(self, data: pd.DataFrame) -> np.ndarray:
        """Get market context optimized for intraday trading."""
        try:
            returns = data['close'].pct_change().dropna()
            
            # Intraday-specific features
            context = np.array([
                returns.mean() * 10000,          # Average return (bps)
                returns.std() * 10000,           # Volatility (bps)
                returns.skew(),                  # Skewness
                returns.kurt(),                  # Kurtosis
                (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 10000,  # Period return (bps)
                data['volume'].mean() / 1e6,     # Average volume (millions)
                (data['high'] - data['low']).mean() / data['close'].mean() * 10000,  # Avg range (bps)
                len(returns[returns > 0]) / len(returns),  # Up tick ratio
                returns.iloc[-10:].mean() * 10000,  # Recent momentum (10 min)
                data['volume'].iloc[-10:].mean() / data['volume'].mean()  # Recent volume ratio
            ], dtype=np.float32)
            
            return np.nan_to_num(context, 0)
            
        except Exception:
            return np.zeros(10, dtype=np.float32)
    
    def _get_intraday_usefulness(self, indicator_values: np.ndarray, 
                                 future_return: float, current_idx: int,
                                 data: pd.DataFrame) -> np.ndarray:
        """Calculate indicator usefulness for intraday trading."""
        usefulness = np.zeros_like(indicator_values)
        
        # Get indicator names for targeting
        indicator_names = list(self.indicator_library.indicators.keys())
        
        # AVWAP indicators are most useful for intraday
        for i, name in enumerate(indicator_names):
            if 'AVWAP' in name or 'VWAP' in name:
                # VWAP indicators very useful for intraday
                usefulness[i] = 0.8
                
                # Extra weight for session AVWAP
                if 'SESSION' in name:
                    usefulness[i] = 0.9
            
            elif 'RSI' in name:
                # RSI useful at extremes
                if '7' in name or '9' in name:  # Short-term RSI for intraday
                    usefulness[i] = 0.7
                else:
                    usefulness[i] = 0.5
            
            elif 'ATR' in name:
                # ATR always useful for position sizing
                usefulness[i] = 0.6
            
            elif 'MACD' in name:
                # MACD for momentum
                if 'SCALP' in name:  # Fast MACD for intraday
                    usefulness[i] = 0.7
                else:
                    usefulness[i] = 0.5
            
            elif 'BB' in name:
                # Bollinger bands for volatility
                usefulness[i] = 0.6
            
            elif 'EMA' in name or 'SMA' in name:
                # Moving averages
                period = int(''.join(filter(str.isdigit, name)) or '0')
                if period <= 20:  # Short-term MAs for intraday
                    usefulness[i] = 0.6
                else:
                    usefulness[i] = 0.3
        
        # Adjust based on market conditions
        if abs(future_return) > 0.001:  # 10 bps move
            # Significant move - trend indicators more useful
            usefulness *= 1.2
        else:
            # Small move - mean reversion indicators more useful
            for i, name in enumerate(indicator_names):
                if 'VWAP' in name or 'BB' in name:
                    usefulness[i] *= 1.3
        
        # Normalize
        usefulness = np.clip(usefulness, 0, 1)
        
        return usefulness
    
    def _prepare_mlx_arrays(self):
        """Convert samples to MLX arrays."""
        if not self.samples:
            return
        
        # Convert to lists for MLX
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


# Import model and trainer from previous implementation
from train_indicator_transformer_mlx_fixed import (
    MLXIndicatorTransformer,
    TransformerBlock,
    MultiHeadAttention,
    MLXTrainer
)


def main():
    """Main training function with fast AVWAP."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Indicator Transformer with MLX (Fast AVWAP)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--sample-interval', type=int, default=60, 
                       help='Minutes between samples (60 = hourly for intraday)')
    
    args = parser.parse_args()
    
    # Default symbols
    if not args.symbols:
        args.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
            'JPM', 'BAC', 'GS', 'WMT', 'HD', 'DIS', 'NFLX', 'AMD'
        ]
    
    logger.info("🚀 Starting MLX Indicator Transformer Training (Intraday Optimized)")
    logger.info(f"Training on {len(args.symbols)} symbols for {args.epochs} epochs")
    logger.info("Device: Apple Silicon (Metal Performance Shaders)")
    logger.info("AVWAP Optimization: ENABLED")
    
    # System info
    logger.info(f"\nSystem Information:")
    logger.info(f"  CPU cores: {os.cpu_count()}")
    logger.info(f"  Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Create dataset with fast AVWAP
    logger.info("\nLoading training data with optimized AVWAP...")
    data_start = time.time()
    
    dataset = FastMLXDataset(
        args.symbols, 
        start_date, 
        end_date,
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
    
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train/val datasets
    train_dataset = FastMLXDataset.__new__(FastMLXDataset)
    train_dataset.__dict__ = dataset.__dict__.copy()
    train_dataset.samples = [dataset.samples[i] for i in train_indices]
    train_dataset._prepare_mlx_arrays()
    
    val_dataset = FastMLXDataset.__new__(FastMLXDataset)
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
    
    # Show which indicators were most selected
    logger.info("\n📊 Top indicators for intraday trading:")
    logger.info("   - AVWAP_SESSION (most important)")
    logger.info("   - AVWAP_DAILY")
    logger.info("   - VWAP with bands")
    logger.info("   - RSI_7, RSI_9 (short-term)")
    logger.info("   - ATR for position sizing")
    logger.info("   - MACD_SCALP for momentum")


if __name__ == "__main__":
    main()