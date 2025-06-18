#!/usr/bin/env python3
"""
MLX-Accelerated Indicator Computation
======================================
Uses Apple Silicon GPU for indicator calculations.
"""

import mlx.core as mx
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MLXIndicatorComputer:
    """Compute indicators using MLX for GPU acceleration."""
    
    def __init__(self):
        # Force GPU usage
        mx.set_default_device(mx.gpu)
        
    def compute_sma(self, prices: mx.array, period: int) -> mx.array:
        """Simple Moving Average on GPU."""
        # Use convolution for efficient rolling window
        kernel = mx.ones(period) / period
        # Pad the input to handle edges
        padded = mx.pad(prices, (period-1, 0), mode='edge')
        return mx.convolve(padded, kernel, mode='valid')
    
    def compute_ema(self, prices: mx.array, period: int) -> mx.array:
        """Exponential Moving Average on GPU."""
        alpha = 2.0 / (period + 1)
        
        # Initialize with first value
        ema = mx.zeros_like(prices)
        ema[0] = prices[0]
        
        # Vectorized EMA computation
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def compute_rsi(self, prices: mx.array, period: int = 14) -> mx.array:
        """RSI on GPU."""
        # Calculate price changes
        deltas = mx.diff(prices, prepend=prices[0])
        
        # Separate gains and losses
        gains = mx.maximum(deltas, 0)
        losses = -mx.minimum(deltas, 0)
        
        # Calculate average gains and losses
        avg_gains = self.compute_sma(gains, period)
        avg_losses = self.compute_sma(losses, period)
        
        # RSI calculation
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def compute_vwap(self, prices: mx.array, volumes: mx.array) -> mx.array:
        """VWAP on GPU."""
        cumulative_pv = mx.cumsum(prices * volumes)
        cumulative_volume = mx.cumsum(volumes)
        return cumulative_pv / (cumulative_volume + 1e-10)
    
    def compute_bollinger_bands(self, prices: mx.array, period: int = 20, std_dev: float = 2.0) -> Tuple[mx.array, mx.array, mx.array]:
        """Bollinger Bands on GPU."""
        sma = self.compute_sma(prices, period)
        
        # Rolling standard deviation using convolution
        squared_diff = (prices - sma) ** 2
        variance = self.compute_sma(squared_diff, period)
        std = mx.sqrt(variance)
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        return upper, sma, lower
    
    def batch_compute_indicators(self, data: pd.DataFrame, indicator_configs: Dict) -> mx.array:
        """Compute all indicators in batch on GPU."""
        # Transfer price/volume data to GPU once
        prices = mx.array(data['close'].values, dtype=mx.float32)
        volumes = mx.array(data['volume'].values, dtype=mx.float32)
        high = mx.array(data['high'].values, dtype=mx.float32)
        low = mx.array(data['low'].values, dtype=mx.float32)
        
        indicators = {}
        
        # Compute indicators in parallel on GPU
        if 'sma' in indicator_configs:
            for period in indicator_configs['sma']:
                indicators[f'SMA_{period}'] = self.compute_sma(prices, period)
        
        if 'ema' in indicator_configs:
            for period in indicator_configs['ema']:
                indicators[f'EMA_{period}'] = self.compute_ema(prices, period)
        
        if 'rsi' in indicator_configs:
            for period in indicator_configs['rsi']:
                indicators[f'RSI_{period}'] = self.compute_rsi(prices, period)
        
        if 'vwap' in indicator_configs:
            indicators['VWAP'] = self.compute_vwap(prices, volumes)
        
        if 'bollinger' in indicator_configs:
            for period in indicator_configs['bollinger']:
                upper, middle, lower = self.compute_bollinger_bands(prices, period)
                indicators[f'BB_UPPER_{period}'] = upper
                indicators[f'BB_MIDDLE_{period}'] = middle
                indicators[f'BB_LOWER_{period}'] = lower
        
        # Stack all indicators
        indicator_arrays = list(indicators.values())
        return mx.stack(indicator_arrays, axis=1)


def accelerate_indicator_computation(data_batch, indicator_library, avwap_cache):
    """GPU-accelerated version of parallel_indicator_computation."""
    mlx_computer = MLXIndicatorComputer()
    results = []
    
    # Process all symbols in batch on GPU
    for symbol, data in data_batch:
        try:
            # Define which indicators to compute
            indicator_configs = {
                'sma': [7, 20, 50, 200],
                'ema': [9, 21, 50],
                'rsi': [7, 14, 21],
                'vwap': True,
                'bollinger': [20]
            }
            
            # Compute on GPU
            gpu_indicators = mlx_computer.batch_compute_indicators(data, indicator_configs)
            
            # Convert back to numpy for compatibility
            indicators_np = np.array(gpu_indicators)
            
            # Combine with AVWAP indicators
            if symbol not in avwap_cache:
                from optimized_avwap_indicators import OptimizedAVWAPCalculator
                calculator = OptimizedAVWAPCalculator(data)
                avwap_cache[symbol] = calculator.calculate_all_avwap_indicators()
            
            results.append((symbol, data, indicators_np))
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    return results