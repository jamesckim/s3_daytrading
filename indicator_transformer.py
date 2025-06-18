#!/usr/bin/env python3
"""
Indicator Transformer - AI-Driven Dynamic Indicator Selection
============================================================
Transformer-based architecture that learns to select and weight technical indicators
based on market conditions, discovering complex relationships and interactions.

Features:
- Multi-head self-attention for indicator relationships
- Dynamic indicator selection based on market regime
- Performance-based learning and adaptation
- Interpretable attention weights
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
from collections import deque
import warnings
warnings.filterwarnings('ignore')


@dataclass
class IndicatorConfig:
    """Configuration for an indicator."""
    name: str
    category: str  # 'trend', 'momentum', 'volatility', 'volume', 'market_structure'
    params: Dict[str, Any]
    compute_function: str
    lookback_required: int
    normalization: str  # 'standard', 'minmax', 'none'


class IndicatorLibrary:
    """
    Comprehensive library of technical indicators organized by category.
    """
    
    def __init__(self):
        self.indicators = self._build_indicator_library()
        self.logger = logging.getLogger('IndicatorLibrary')
    
    def _build_indicator_library(self) -> Dict[str, IndicatorConfig]:
        """Build comprehensive indicator library."""
        indicators = {}
        
        # Trend Indicators
        # Optimized EMA periods - removed 1,2,3,7,500 as they add noise
        ema_periods = [5, 10, 20, 50, 100, 150, 200, 300, 1000]
        sma_periods = [10, 20, 50, 200]  # Keep SMA periods as original
        
        for period in sma_periods:
            indicators[f'SMA_{period}'] = IndicatorConfig(
                name=f'SMA_{period}',
                category='trend',
                params={'period': period},
                compute_function='sma',
                lookback_required=period,
                normalization='none'
            )
        
        for period in ema_periods:
            indicators[f'EMA_{period}'] = IndicatorConfig(
                name=f'EMA_{period}',
                category='trend',
                params={'period': period},
                compute_function='ema',
                lookback_required=period * 2,
                normalization='none'
            )
        
        # Momentum Indicators - Multiple RSI periods for different timeframes
        rsi_periods = [7, 9, 14, 21, 25]  # Short to long-term momentum
        for period in rsi_periods:
            indicators[f'RSI_{period}'] = IndicatorConfig(
                name=f'RSI_{period}',
                category='momentum',
                params={'period': period},
                compute_function='rsi',
                lookback_required=period * 2,
                normalization='none'  # Already 0-100
            )
        
        # MACD Variations for different market conditions
        macd_configs = [
            ('MACD_FAST', {'fast': 5, 'slow': 13, 'signal': 5}),      # Faster signals
            ('MACD', {'fast': 12, 'slow': 26, 'signal': 9}),          # Standard
            ('MACD_SLOW', {'fast': 19, 'slow': 39, 'signal': 9}),     # Slower, stronger signals
            ('MACD_SCALP', {'fast': 3, 'slow': 10, 'signal': 16})     # Ultra-fast for scalping
        ]
        
        for name, params in macd_configs:
            indicators[name] = IndicatorConfig(
                name=name,
                category='momentum',
                params=params,
                compute_function='macd',
                lookback_required=params['slow'] * 2,
                normalization='standard'
            )
        
        # Volatility Indicators - Multiple ATR periods
        atr_periods = [7, 10, 14, 20, 30]  # Quick to longer-term volatility
        for period in atr_periods:
            indicators[f'ATR_{period}'] = IndicatorConfig(
                name=f'ATR_{period}',
                category='volatility',
                params={'period': period},
                compute_function='atr',
                lookback_required=period * 2,
                normalization='standard'
            )
        
        # Bollinger Bands - Multiple periods for squeeze detection
        bb_periods = [10, 15, 20, 30, 50]  # Fast to long-term bands
        for period in bb_periods:
            indicators[f'BB_{period}'] = IndicatorConfig(
                name=f'BB_{period}',
                category='volatility',
                params={'period': period, 'std': 2},
                compute_function='bollinger_bands',
                lookback_required=period * 2,
                normalization='none'
            )
        
        # Volume Indicators
        indicators['OBV'] = IndicatorConfig(
            name='OBV',
            category='volume',
            params={},
            compute_function='obv',
            lookback_required=1,
            normalization='standard'
        )
        
        indicators['VWAP'] = IndicatorConfig(
            name='VWAP',
            category='volume',
            params={},
            compute_function='vwap',
            lookback_required=1,
            normalization='none'
        )
        
        # Add VWAP bands for standard VWAP
        for std_mult in [1, 2, 3]:
            # Upper band
            indicators[f'VWAP_U{std_mult}'] = IndicatorConfig(
                name=f'VWAP_U{std_mult}',
                category='volume',
                params={'base': 'VWAP', 'std_mult': std_mult, 'direction': 'upper'},
                compute_function='vwap_band',
                lookback_required=20,
                normalization='none'
            )
            
            # Lower band
            indicators[f'VWAP_L{std_mult}'] = IndicatorConfig(
                name=f'VWAP_L{std_mult}',
                category='volume',
                params={'base': 'VWAP', 'std_mult': std_mult, 'direction': 'lower'},
                compute_function='vwap_band',
                lookback_required=20,
                normalization='none'
            )
        
        # Anchored VWAP indicators - anchored to different time points
        anchored_vwap_configs = [
            ('AVWAP_SESSION', {'anchor': 'session'}),      # From market open
            ('AVWAP_DAILY', {'anchor': 'daily'}),          # From start of day
            ('AVWAP_WEEKLY', {'anchor': 'weekly'}),        # From start of week
            ('AVWAP_MONTHLY', {'anchor': 'monthly'}),      # From start of month
            # Dynamic anchors based on price/volume
            ('AVWAP_HIGH', {'anchor': 'high'}),            # From highest price of day
            ('AVWAP_LOW', {'anchor': 'low'}),              # From lowest price of day
            ('AVWAP_HVOL', {'anchor': 'high_volume'}),     # From highest volume minute
            ('AVWAP_PREV_HIGH', {'anchor': 'prev_high'}),  # From previous day's high
            ('AVWAP_PREV_LOW', {'anchor': 'prev_low'}),    # From previous day's low
            ('AVWAP_PREV_HVOL', {'anchor': 'prev_hvol'}),  # From previous day's high volume
        ]
        
        for name, params in anchored_vwap_configs:
            indicators[name] = IndicatorConfig(
                name=name,
                category='volume',
                params=params,
                compute_function='anchored_vwap',
                lookback_required=1,
                normalization='none'
            )
            
            # Add VWAP bands (standard deviations) for each VWAP
            for std_mult in [1, 2, 3]:
                # Upper band
                indicators[f'{name}_U{std_mult}'] = IndicatorConfig(
                    name=f'{name}_U{std_mult}',
                    category='volume',
                    params={'base': name, 'std_mult': std_mult, 'direction': 'upper'},
                    compute_function='vwap_band',
                    lookback_required=20,  # Need some data for std calculation
                    normalization='none'
                )
                
                # Lower band
                indicators[f'{name}_L{std_mult}'] = IndicatorConfig(
                    name=f'{name}_L{std_mult}',
                    category='volume',
                    params={'base': name, 'std_mult': std_mult, 'direction': 'lower'},
                    compute_function='vwap_band',
                    lookback_required=20,
                    normalization='none'
                )
        
        # Market Structure
        indicators['SUPPORT_RESISTANCE'] = IndicatorConfig(
            name='SUPPORT_RESISTANCE',
            category='market_structure',
            params={'lookback': 50},
            compute_function='support_resistance',
            lookback_required=50,
            normalization='none'
        )
        
        return indicators
    
    def compute_indicator(self, data: pd.DataFrame, indicator_name: str) -> pd.Series:
        """Compute a specific indicator."""
        if indicator_name not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        
        config = self.indicators[indicator_name]
        
        # Call appropriate computation function
        if config.compute_function == 'sma':
            result = data['close'].rolling(config.params['period']).mean()
        elif config.compute_function == 'ema':
            result = data['close'].ewm(span=config.params['period']).mean()
        elif config.compute_function == 'rsi':
            result = self._compute_rsi(data['close'], config.params['period'])
        elif config.compute_function == 'macd':
            result = self._compute_macd(data['close'], **config.params)
        elif config.compute_function == 'atr':
            result = self._compute_atr(data, config.params['period'])
        elif config.compute_function == 'bollinger_bands':
            result = self._compute_bollinger_bands(data['close'], **config.params)
        elif config.compute_function == 'obv':
            result = self._compute_obv(data)
        elif config.compute_function == 'vwap':
            result = self._compute_vwap(data)
        elif config.compute_function == 'anchored_vwap':
            result = self._compute_anchored_vwap(data, config.params['anchor'])
        elif config.compute_function == 'vwap_band':
            result = self._compute_vwap_band(data, config.params['base'], 
                                           config.params['std_mult'], 
                                           config.params['direction'])
        elif config.compute_function == 'support_resistance':
            result = self._compute_support_resistance(data, config.params['lookback'])
        else:
            raise NotImplementedError(f"Compute function not implemented: {config.compute_function}")
        
        # Apply normalization
        if config.normalization == 'standard':
            result = (result - result.mean()) / (result.std() + 1e-8)
        elif config.normalization == 'minmax':
            result = (result - result.min()) / (result.max() - result.min() + 1e-8)
        
        return result
    
    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        """Compute MACD line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        return macd_line
    
    def _compute_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _compute_bollinger_bands(self, prices: pd.Series, period: int, std: int) -> pd.Series:
        """Compute Bollinger Band width."""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        bb_width = (upper_band - lower_band) / sma
        return bb_width
    
    def _compute_obv(self, data: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume."""
        obv = (data['volume'] * (~data['close'].diff().le(0) * 2 - 1)).cumsum()
        return obv
    
    def _compute_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Compute Volume Weighted Average Price."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap
    
    def _compute_anchored_vwap(self, data: pd.DataFrame, anchor: str) -> pd.Series:
        """Compute Anchored VWAP from specific anchor points."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Initialize result series
        avwap = pd.Series(index=data.index, dtype=float)
        
        if anchor == 'session':
            # Anchor at market open (9:30 AM ET)
            for date in data.index.date:
                day_data = data[data.index.date == date]
                if len(day_data) > 0:
                    # Find first bar after 9:30 AM ET
                    day_start = day_data.between_time('09:30', '16:00').index[0] if len(day_data.between_time('09:30', '16:00')) > 0 else day_data.index[0]
                    day_mask = data.index >= day_start
                    day_data_from_open = data[day_mask & (data.index.date == date)]
                    
                    if len(day_data_from_open) > 0:
                        cum_vol = day_data_from_open['volume'].cumsum()
                        cum_pv = (typical_price[day_data_from_open.index] * day_data_from_open['volume']).cumsum()
                        avwap[day_data_from_open.index] = cum_pv / cum_vol
                        
        elif anchor == 'daily':
            # Anchor at start of each day
            for date in data.index.date:
                day_mask = data.index.date == date
                day_data = data[day_mask]
                if len(day_data) > 0:
                    cum_vol = day_data['volume'].cumsum()
                    cum_pv = (typical_price[day_mask] * day_data['volume']).cumsum()
                    avwap[day_mask] = cum_pv / cum_vol
                    
        elif anchor == 'weekly':
            # Anchor at start of each week (Monday)
            data['week'] = data.index.isocalendar().week
            for week in data['week'].unique():
                week_mask = data['week'] == week
                week_data = data[week_mask]
                if len(week_data) > 0:
                    cum_vol = week_data['volume'].cumsum()
                    cum_pv = (typical_price[week_mask] * week_data['volume']).cumsum()
                    avwap[week_mask] = cum_pv / cum_vol
                    
        elif anchor == 'monthly':
            # Anchor at start of each month
            for month_start in pd.date_range(data.index[0].replace(day=1), data.index[-1], freq='MS'):
                month_end = month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                month_mask = (data.index >= month_start) & (data.index <= month_end)
                month_data = data[month_mask]
                if len(month_data) > 0:
                    cum_vol = month_data['volume'].cumsum()
                    cum_pv = (typical_price[month_mask] * month_data['volume']).cumsum()
                    avwap[month_mask] = cum_pv / cum_vol
                    
        # Dynamic anchors based on price/volume events
        elif anchor in ['high', 'low', 'high_volume', 'prev_high', 'prev_low', 'prev_hvol']:
            # Process each day separately
            unique_dates = pd.unique(data.index.date)
            
            for i, date in enumerate(unique_dates):
                day_mask = data.index.date == date
                day_data = data[day_mask]
                
                if len(day_data) == 0:
                    continue
                
                # Determine anchor point based on type
                anchor_idx = None
                
                if anchor == 'high':
                    # Anchor at highest price of current day
                    anchor_idx = day_data['high'].idxmax()
                elif anchor == 'low':
                    # Anchor at lowest price of current day
                    anchor_idx = day_data['low'].idxmin()
                elif anchor == 'high_volume':
                    # Anchor at highest volume minute of current day
                    anchor_idx = day_data['volume'].idxmax()
                elif anchor in ['prev_high', 'prev_low', 'prev_hvol'] and i > 0:
                    # Use previous day's data
                    prev_date = unique_dates[i-1]
                    prev_day_mask = data.index.date == prev_date
                    prev_day_data = data[prev_day_mask]
                    
                    if len(prev_day_data) > 0:
                        if anchor == 'prev_high':
                            anchor_idx = prev_day_data['high'].idxmax()
                        elif anchor == 'prev_low':
                            anchor_idx = prev_day_data['low'].idxmin()
                        elif anchor == 'prev_hvol':
                            anchor_idx = prev_day_data['volume'].idxmax()
                
                # Calculate VWAP from anchor point forward
                if anchor_idx is not None:
                    # Get all data from anchor point to end of current day
                    mask_from_anchor = (data.index >= anchor_idx) & day_mask
                    data_from_anchor = data[mask_from_anchor]
                    
                    if len(data_from_anchor) > 0:
                        cum_vol = data_from_anchor['volume'].cumsum()
                        cum_pv = (typical_price[mask_from_anchor] * data_from_anchor['volume']).cumsum()
                        avwap[mask_from_anchor] = cum_pv / cum_vol
        
        return avwap
    
    def _compute_vwap_band(self, data: pd.DataFrame, base_vwap: str, 
                          std_mult: int, direction: str) -> pd.Series:
        """Compute VWAP bands using standard deviation."""
        # First compute the base VWAP
        base_vwap_values = self.compute_indicator(data, base_vwap)
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate squared deviations from VWAP
        squared_deviations = (typical_price - base_vwap_values) ** 2
        
        # Weight by volume for volume-weighted standard deviation
        weighted_squared_dev = squared_deviations * data['volume']
        
        # Calculate rolling sums with a 20-period window
        window = 20
        rolling_weighted_sum = weighted_squared_dev.rolling(window=window, min_periods=1).sum()
        rolling_volume_sum = data['volume'].rolling(window=window, min_periods=1).sum()
        
        # Calculate volume-weighted standard deviation
        vwstd = np.sqrt(rolling_weighted_sum / (rolling_volume_sum + 1e-8))
        
        # Calculate bands
        if direction == 'upper':
            band = base_vwap_values + (std_mult * vwstd)
        else:  # lower
            band = base_vwap_values - (std_mult * vwstd)
        
        return band
    
    def _compute_support_resistance(self, data: pd.DataFrame, lookback: int) -> pd.Series:
        """Compute distance to nearest support/resistance."""
        # Simplified version - distance to recent high/low
        rolling_high = data['high'].rolling(window=lookback).max()
        rolling_low = data['low'].rolling(window=lookback).min()
        
        # Distance to resistance (positive) or support (negative)
        dist_to_resistance = (rolling_high - data['close']) / data['close']
        dist_to_support = (data['close'] - rolling_low) / data['close']
        
        # Return whichever is closer
        result = pd.Series(index=data.index)
        mask = dist_to_resistance < dist_to_support
        result[mask] = dist_to_resistance[mask]
        result[~mask] = -dist_to_support[~mask]
        
        return result


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for learning indicator relationships.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)
        
        # Add and norm
        output = self.layer_norm(x + self.dropout(output))
        
        return output, attention_weights


class IndicatorTransformer(nn.Module):
    """
    Transformer architecture for dynamic indicator selection and weighting.
    """
    
    def __init__(self, 
                 num_indicators: int,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 100):
        super().__init__()
        
        self.num_indicators = num_indicators
        self.d_model = d_model
        
        # Indicator embeddings
        self.indicator_embedding = nn.Embedding(num_indicators, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.temporal_embedding = nn.Linear(1, d_model)  # For time-based features
        
        # Market context encoder
        self.market_context_encoder = nn.Sequential(
            nn.Linear(10, 128),  # 10 market features
            nn.ReLU(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Transformer layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.indicator_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # Selection probability
        )
        
        self.indicator_weighter = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Positive weights
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                indicator_values: torch.Tensor,
                market_context: torch.Tensor,
                indicator_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to select and weight indicators.
        
        Args:
            indicator_values: [batch_size, num_indicators] - Current indicator values
            market_context: [batch_size, 10] - Market regime features
            indicator_indices: [batch_size, num_indicators] - Indicator IDs
            
        Returns:
            Dictionary with selection probabilities, weights, and attention maps
        """
        batch_size = indicator_values.size(0)
        num_indicators = indicator_values.size(1)
        
        # Embed indicators
        indicator_embeds = self.indicator_embedding(indicator_indices)
        
        # Add indicator values as features
        value_embeds = self.temporal_embedding(indicator_values.unsqueeze(-1))
        
        # Combine embeddings
        x = indicator_embeds + value_embeds
        
        # Add market context
        context_embed = self.market_context_encoder(market_context)
        context_embed = context_embed.unsqueeze(1).expand(-1, num_indicators, -1)
        x = x + context_embed
        
        # Store attention weights for interpretability
        attention_weights_list = []
        
        # Pass through transformer layers
        for i, (attention_layer, ff_layer) in enumerate(zip(self.attention_layers, self.feed_forward_layers)):
            # Self-attention
            x, attention_weights = attention_layer(x)
            attention_weights_list.append(attention_weights)
            
            # Feed-forward
            x = x + ff_layer(x)
        
        # Generate selection probabilities and weights
        selection_probs = self.indicator_selector(x).squeeze(-1)
        indicator_weights = self.indicator_weighter(x).squeeze(-1)
        
        # Normalize weights for selected indicators
        selected_mask = selection_probs > 0.5
        indicator_weights = indicator_weights * selected_mask.float()
        weight_sum = indicator_weights.sum(dim=1, keepdim=True)
        indicator_weights = indicator_weights / (weight_sum + 1e-8)
        
        return {
            'selection_probs': selection_probs,
            'indicator_weights': indicator_weights,
            'selected_mask': selected_mask,
            'attention_weights': attention_weights_list,
            'representations': x
        }


class MarketRegimeAnalyzer:
    """
    Analyzes market conditions to provide context for indicator selection.
    """
    
    def __init__(self):
        self.regime_window = 50
        self.features = [
            'trend_strength',
            'volatility_regime',
            'volume_profile',
            'market_breadth',
            'momentum_state',
            'mean_reversion_potential',
            'breakout_probability',
            'time_of_day_pattern',
            'day_of_week_effect',
            'market_efficiency'
        ]
    
    def analyze_market_context(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract market context features from recent data.
        
        Returns:
            Array of shape [10] with normalized market features
        """
        if len(data) < self.regime_window:
            return np.zeros(10)
        
        recent_data = data.iloc[-self.regime_window:]
        features = []
        
        # 1. Trend strength (ADX-like)
        returns = recent_data['close'].pct_change()
        trend_strength = abs(returns.mean()) / (returns.std() + 1e-8)
        features.append(np.tanh(trend_strength))
        
        # 2. Volatility regime (normalized ATR)
        high_low_range = (recent_data['high'] - recent_data['low']) / recent_data['close']
        volatility_regime = high_low_range.mean() / high_low_range.std()
        features.append(np.tanh(volatility_regime))
        
        # 3. Volume profile (relative to average)
        volume_ratio = recent_data['volume'].iloc[-5:].mean() / recent_data['volume'].mean()
        features.append(np.log1p(volume_ratio))
        
        # 4. Market breadth (up vs down days)
        up_days = (returns > 0).sum()
        market_breadth = (up_days / len(returns)) * 2 - 1  # Scale to [-1, 1]
        features.append(market_breadth)
        
        # 5. Momentum state (rate of change)
        momentum = recent_data['close'].iloc[-1] / recent_data['close'].iloc[-10] - 1
        features.append(np.tanh(momentum * 10))
        
        # 6. Mean reversion potential
        zscore = (recent_data['close'].iloc[-1] - recent_data['close'].mean()) / recent_data['close'].std()
        mean_reversion = -np.tanh(zscore)
        features.append(mean_reversion)
        
        # 7. Breakout probability (price near range extremes)
        price_position = (recent_data['close'].iloc[-1] - recent_data['low'].min()) / \
                        (recent_data['high'].max() - recent_data['low'].min() + 1e-8)
        breakout_prob = 2 * abs(price_position - 0.5)
        features.append(breakout_prob)
        
        # 8. Time of day pattern (assuming datetime index)
        if hasattr(data.index, 'hour'):
            hour = data.index[-1].hour
            time_feature = np.sin(2 * np.pi * hour / 24)
        else:
            time_feature = 0
        features.append(time_feature)
        
        # 9. Day of week effect
        if hasattr(data.index, 'dayofweek'):
            day = data.index[-1].dayofweek
            day_feature = np.sin(2 * np.pi * day / 7)
        else:
            day_feature = 0
        features.append(day_feature)
        
        # 10. Market efficiency (serial correlation)
        if len(returns.dropna()) > 20:
            autocorr = returns.dropna().autocorr(lag=1)
            efficiency = 1 - abs(autocorr)
        else:
            efficiency = 0.5
        features.append(efficiency)
        
        return np.array(features, dtype=np.float32)


class AIIndicatorSelector:
    """
    Main class for AI-driven indicator selection using transformer architecture.
    """
    
    def __init__(self, 
                 model_path: Optional[Path] = None,
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.logger = logging.getLogger('AIIndicatorSelector')
        
        # Initialize components
        self.indicator_library = IndicatorLibrary()
        self.market_analyzer = MarketRegimeAnalyzer()
        
        # Get indicator list
        self.indicator_names = list(self.indicator_library.indicators.keys())
        self.num_indicators = len(self.indicator_names)
        
        # Initialize transformer model
        self.model = IndicatorTransformer(
            num_indicators=self.num_indicators,
            d_model=256,
            num_heads=8,
            num_layers=6
        ).to(self.device)
        
        # Load pre-trained weights if available
        if model_path and model_path.exists():
            self.load_model(model_path)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.indicator_usage_stats = {name: {'selected': 0, 'total': 0, 'performance': []} 
                                     for name in self.indicator_names}
        
        # Training components
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
    
    def select_indicators(self, 
                         data: pd.DataFrame,
                         top_k: int = 20) -> Dict[str, Any]:
        """
        Select optimal indicators for current market conditions.
        
        Args:
            data: Historical price data
            top_k: Maximum number of indicators to select
            
        Returns:
            Dictionary with selected indicators and weights
        """
        self.model.eval()
        
        with torch.no_grad():
            # Compute all indicators
            indicator_values = self._compute_all_indicators(data)
            
            # Get market context
            market_context = self.market_analyzer.analyze_market_context(data)
            
            # Prepare inputs
            indicator_tensor = torch.FloatTensor(indicator_values).unsqueeze(0).to(self.device)
            context_tensor = torch.FloatTensor(market_context).unsqueeze(0).to(self.device)
            indices_tensor = torch.LongTensor(range(self.num_indicators)).unsqueeze(0).to(self.device)
            
            # Forward pass
            outputs = self.model(indicator_tensor, context_tensor, indices_tensor)
            
            # Extract results
            selection_probs = outputs['selection_probs'].squeeze().cpu().numpy()
            indicator_weights = outputs['indicator_weights'].squeeze().cpu().numpy()
            
            # Select top-k indicators
            if top_k < self.num_indicators:
                top_indices = np.argsort(selection_probs)[-top_k:]
            else:
                top_indices = np.where(selection_probs > 0.5)[0]
            
            # Prepare results
            selected_indicators = {}
            for idx in top_indices:
                if selection_probs[idx] > 0.3:  # Confidence threshold
                    indicator_name = self.indicator_names[idx]
                    selected_indicators[indicator_name] = {
                        'weight': float(indicator_weights[idx]),
                        'selection_prob': float(selection_probs[idx]),
                        'value': float(indicator_values[idx])
                    }
            
            # Update usage statistics
            for i, name in enumerate(self.indicator_names):
                self.indicator_usage_stats[name]['total'] += 1
                if i in top_indices:
                    self.indicator_usage_stats[name]['selected'] += 1
            
            # Get attention weights for interpretability
            attention_maps = self._extract_attention_patterns(outputs['attention_weights'])
            
            return {
                'selected_indicators': selected_indicators,
                'market_context': market_context.tolist(),
                'attention_patterns': attention_maps,
                'regime_detection': self._interpret_market_regime(market_context)
            }
    
    def _compute_all_indicators(self, data: pd.DataFrame) -> np.ndarray:
        """Compute all indicators from the library."""
        indicator_values = []
        
        for indicator_name in self.indicator_names:
            try:
                value = self.indicator_library.compute_indicator(data, indicator_name)
                # Use the last value
                if len(value) > 0 and not pd.isna(value.iloc[-1]):
                    indicator_values.append(value.iloc[-1])
                else:
                    indicator_values.append(0.0)
            except Exception as e:
                self.logger.warning(f"Failed to compute {indicator_name}: {e}")
                indicator_values.append(0.0)
        
        return np.array(indicator_values, dtype=np.float32)
    
    def _extract_attention_patterns(self, attention_weights_list: List[torch.Tensor]) -> Dict[str, List[str]]:
        """Extract interpretable attention patterns from transformer."""
        # Get the last layer's attention weights
        last_layer_attention = attention_weights_list[-1].squeeze().mean(dim=0).cpu().numpy()
        
        # Find strongly connected indicator pairs
        strong_connections = []
        threshold = np.percentile(last_layer_attention.flatten(), 90)
        
        for i in range(self.num_indicators):
            for j in range(i+1, self.num_indicators):
                if last_layer_attention[i, j] > threshold:
                    strong_connections.append({
                        'indicator1': self.indicator_names[i],
                        'indicator2': self.indicator_names[j],
                        'strength': float(last_layer_attention[i, j])
                    })
        
        # Sort by strength
        strong_connections.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'strong_connections': strong_connections[:10],  # Top 10 connections
            'attention_summary': 'Indicators showing strong relationships'
        }
    
    def _interpret_market_regime(self, market_context: np.ndarray) -> str:
        """Interpret market regime from context features."""
        trend_strength = market_context[0]
        volatility = market_context[1]
        momentum = market_context[4]
        
        if trend_strength > 0.5 and momentum > 0.3:
            regime = "Strong Uptrend"
        elif trend_strength > 0.5 and momentum < -0.3:
            regime = "Strong Downtrend"
        elif volatility > 0.7:
            regime = "High Volatility"
        elif abs(trend_strength) < 0.2:
            regime = "Ranging/Sideways"
        else:
            regime = "Transitional"
        
        return regime
    
    def update_performance(self, 
                          selected_indicators: Dict[str, Any],
                          trade_result: float):
        """
        Update model based on trade performance.
        
        Args:
            selected_indicators: The indicators that were selected
            trade_result: Profit/loss from the trade
        """
        # Store performance data
        self.performance_history.append({
            'indicators': list(selected_indicators.keys()),
            'result': trade_result,
            'timestamp': datetime.now()
        })
        
        # Update indicator-specific performance
        for indicator_name in selected_indicators:
            if indicator_name in self.indicator_usage_stats:
                self.indicator_usage_stats[indicator_name]['performance'].append(trade_result)
        
        # Periodically retrain model (every 100 trades)
        if len(self.performance_history) % 100 == 0:
            self._retrain_on_performance()
    
    def _retrain_on_performance(self):
        """Retrain model based on accumulated performance data."""
        # This would involve creating training batches from performance history
        # and running gradient descent to improve indicator selection
        self.logger.info("Retraining model on performance data...")
        # Implementation depends on specific training strategy
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for indicator selection."""
        report = {
            'total_selections': sum(stats['selected'] for stats in self.indicator_usage_stats.values()),
            'unique_indicators_used': sum(1 for stats in self.indicator_usage_stats.values() if stats['selected'] > 0),
            'top_selected_indicators': [],
            'best_performing_indicators': [],
            'selection_efficiency': 0.0
        }
        
        # Top selected indicators
        selection_rates = [(name, stats['selected'] / max(stats['total'], 1)) 
                          for name, stats in self.indicator_usage_stats.items()]
        selection_rates.sort(key=lambda x: x[1], reverse=True)
        report['top_selected_indicators'] = selection_rates[:10]
        
        # Best performing indicators
        indicator_performance = []
        for name, stats in self.indicator_usage_stats.items():
            if len(stats['performance']) > 0:
                avg_performance = np.mean(stats['performance'])
                indicator_performance.append((name, avg_performance))
        
        indicator_performance.sort(key=lambda x: x[1], reverse=True)
        report['best_performing_indicators'] = indicator_performance[:10]
        
        # Selection efficiency (selected indicators that performed well)
        if report['total_selections'] > 0:
            positive_selections = sum(1 for record in self.performance_history if record['result'] > 0)
            report['selection_efficiency'] = positive_selections / len(self.performance_history)
        
        return report
    
    def save_model(self, path: Path):
        """Save model weights and configuration."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'indicator_names': self.indicator_names,
            'usage_stats': self.indicator_usage_stats,
            'performance_history': list(self.performance_history)
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load model weights and configuration."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'indicator_names' in checkpoint:
            self.indicator_names = checkpoint['indicator_names']
        
        if 'usage_stats' in checkpoint:
            self.indicator_usage_stats = checkpoint['usage_stats']
        
        if 'performance_history' in checkpoint:
            self.performance_history = deque(checkpoint['performance_history'], maxlen=1000)
        
        self.model.eval()  # Set to evaluation mode
        self.logger.info(f"Model loaded from {path}")


def main():
    """Example usage of the AI Indicator Selector."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🤖 AI Indicator Selector - Transformer Architecture")
    print("=" * 60)
    
    # Initialize selector
    selector = AIIndicatorSelector(device='cpu')
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1min')
    sample_data = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
        'high': 100.5 + np.random.randn(len(dates)).cumsum() * 0.1,
        'low': 99.5 + np.random.randn(len(dates)).cumsum() * 0.1,
        'close': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Select indicators
    print("\n📊 Selecting optimal indicators for current market...")
    selection = selector.select_indicators(sample_data.iloc[-500:], top_k=15)
    
    print(f"\n🎯 Market Regime: {selection['regime_detection']}")
    print(f"\n✅ Selected {len(selection['selected_indicators'])} indicators:")
    
    for indicator, info in selection['selected_indicators'].items():
        print(f"   {indicator}: weight={info['weight']:.3f}, confidence={info['selection_prob']:.3f}")
    
    # Show attention patterns
    print("\n🔗 Strong Indicator Relationships:")
    for conn in selection['attention_patterns']['strong_connections'][:5]:
        print(f"   {conn['indicator1']} <-> {conn['indicator2']}: {conn['strength']:.3f}")
    
    # Performance report
    report = selector.get_performance_report()
    print(f"\n📈 Performance Summary:")
    print(f"   Total selections: {report['total_selections']}")
    print(f"   Unique indicators used: {report['unique_indicators_used']}")


if __name__ == "__main__":
    main()