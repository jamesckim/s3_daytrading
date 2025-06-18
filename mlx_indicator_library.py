#!/usr/bin/env python3
"""
MLX Indicator Library - Complete GPU Implementation
==================================================
All indicators computed on Apple Silicon GPU using MLX.
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# Force GPU usage
mx.set_default_device(mx.gpu)


@dataclass
class GPUData:
    """Holds all price/volume data on GPU."""
    open: mx.array
    high: mx.array
    low: mx.array
    close: mx.array
    volume: mx.array
    timestamps: mx.array
    
    def __len__(self):
        return len(self.close)


class MLXIndicatorLibrary:
    """Complete indicator library implemented in MLX for GPU computation."""
    
    def __init__(self):
        self.indicators = {}
        self._register_all_indicators()
        
    def _register_all_indicators(self):
        """Register all indicator computation functions."""
        # Price indicators
        for period in [7, 9, 14, 20, 50, 100, 200]:
            self.indicators[f'SMA_{period}'] = lambda d, p=period: self._sma(d.close, p)
            self.indicators[f'EMA_{period}'] = lambda d, p=period: self._ema(d.close, p)
        
        # RSI variants
        for period in [7, 9, 14, 21]:
            self.indicators[f'RSI_{period}'] = lambda d, p=period: self._rsi(d.close, p)
        
        # MACD
        self.indicators['MACD'] = lambda d: self._macd(d.close)[0]
        self.indicators['MACD_SIGNAL'] = lambda d: self._macd(d.close)[1]
        self.indicators['MACD_HIST'] = lambda d: self._macd(d.close)[2]
        
        # Bollinger Bands
        for period in [20, 50]:
            self.indicators[f'BB_UPPER_{period}'] = lambda d, p=period: self._bollinger(d.close, p)[0]
            self.indicators[f'BB_MIDDLE_{period}'] = lambda d, p=period: self._bollinger(d.close, p)[1]
            self.indicators[f'BB_LOWER_{period}'] = lambda d, p=period: self._bollinger(d.close, p)[2]
        
        # Volume indicators
        self.indicators['OBV'] = lambda d: self._obv(d.close, d.volume)
        self.indicators['AD'] = lambda d: self._accumulation_distribution(d)
        self.indicators['MFI_14'] = lambda d: self._mfi(d, 14)
        
        # Volatility
        self.indicators['ATR_14'] = lambda d: self._atr(d, 14)
        self.indicators['NATR_14'] = lambda d: self._natr(d, 14)
        
        # VWAP (session)
        self.indicators['VWAP'] = lambda d: self._vwap(d.close, d.volume)
        
        # Stochastic
        self.indicators['STOCH_K'] = lambda d: self._stochastic(d, 14)[0]
        self.indicators['STOCH_D'] = lambda d: self._stochastic(d, 14)[1]
        
        # Williams %R
        self.indicators['WILLIAMS_R'] = lambda d: self._williams_r(d, 14)
        
        # CCI
        self.indicators['CCI_20'] = lambda d: self._cci(d, 20)
        
        # ADX
        self.indicators['ADX_14'] = lambda d: self._adx(d, 14)
        
    def compute_all_indicators_gpu(self, data: GPUData) -> mx.array:
        """Compute all indicators in parallel on GPU."""
        n_bars = len(data)
        n_indicators = len(self.indicators)
        
        # Pre-allocate output array on GPU
        all_indicators = mx.zeros((n_bars, n_indicators), dtype=mx.float32)
        
        # Compute each indicator
        for idx, (name, func) in enumerate(self.indicators.items()):
            try:
                indicator_values = func(data)
                
                # Handle different output lengths
                if len(indicator_values) == n_bars:
                    all_indicators[:, idx] = indicator_values
                else:
                    # Pad with NaN for shorter indicators
                    pad_length = n_bars - len(indicator_values)
                    padded = mx.pad(indicator_values, (pad_length, 0), constant_values=mx.nan)
                    all_indicators[:, idx] = padded
                    
            except Exception as e:
                logger.warning(f"Failed to compute {name}: {e}")
                all_indicators[:, idx] = mx.nan
        
        return all_indicators
    
    # Core indicator implementations
    
    def _sma(self, data: mx.array, period: int) -> mx.array:
        """Simple Moving Average - O(n) using cumsum."""
        cumsum = mx.cumsum(mx.pad(data, (1, 0), constant_values=0))
        sma = (cumsum[period:] - cumsum[:-period]) / period
        
        # Pad with NaN for consistency
        return mx.pad(sma, (period-1, 0), constant_values=mx.nan)
    
    def _ema(self, data: mx.array, period: int) -> mx.array:
        """Exponential Moving Average - O(n) iterative."""
        alpha = 2.0 / (period + 1)
        ema = mx.zeros_like(data)
        ema[0] = data[0]
        
        # Use scan operation for efficiency
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            
        return ema
    
    def _diff(self, arr: mx.array, prepend=None) -> mx.array:
        """Compute first differences."""
        if prepend is not None:
            arr = mx.concatenate([mx.array([prepend]), arr])
            return arr[1:] - arr[:-1]
        else:
            return arr[1:] - arr[:-1]
    
    @lru_cache(maxsize=10)
    def _rsi(self, close: mx.array, period: int = 14) -> mx.array:
        """RSI using Wilder's smoothing."""
        deltas = self._diff(close, prepend=close[0])
        gains = mx.maximum(deltas, 0)
        losses = -mx.minimum(deltas, 0)
        
        # Wilder's smoothing (EMA with alpha = 1/period)
        avg_gain = self._wilder_smooth(gains, period)
        avg_loss = self._wilder_smooth(losses, period)
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _wilder_smooth(self, data: mx.array, period: int) -> mx.array:
        """Wilder's smoothing (RMA)."""
        alpha = 1.0 / period
        smoothed = mx.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            
        return smoothed
    
    def _macd(self, close: mx.array, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[mx.array, mx.array, mx.array]:
        """MACD indicator."""
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _bollinger(self, close: mx.array, period: int = 20, std_dev: float = 2.0) -> Tuple[mx.array, mx.array, mx.array]:
        """Bollinger Bands."""
        middle = self._sma(close, period)
        
        # Rolling standard deviation
        std = self._rolling_std(close, period)
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    def _rolling_std(self, data: mx.array, period: int) -> mx.array:
        """Rolling standard deviation - memory efficient."""
        # Use cumulative sum approach for efficiency
        cumsum = mx.cumsum(mx.pad(data, (1, 0), constant_values=0))
        cumsum_sq = mx.cumsum(mx.pad(data**2, (1, 0), constant_values=0))
        
        # Calculate rolling mean and mean of squares
        count = period
        rolling_sum = cumsum[period:] - cumsum[:-period]
        rolling_sum_sq = cumsum_sq[period:] - cumsum_sq[:-period]
        
        rolling_mean = rolling_sum / count
        rolling_mean_sq = rolling_sum_sq / count
        
        # Standard deviation
        variance = rolling_mean_sq - rolling_mean**2
        # Ensure non-negative variance
        variance = mx.maximum(variance, 0)
        std = mx.sqrt(variance)
        
        # Pad with NaN
        return mx.pad(std, (period-1, 0), constant_values=mx.nan)
    
    def _vwap(self, close: mx.array, volume: mx.array) -> mx.array:
        """Volume Weighted Average Price."""
        cum_pv = mx.cumsum(close * volume)
        cum_vol = mx.cumsum(volume)
        return cum_pv / (cum_vol + 1e-10)
    
    def _obv(self, close: mx.array, volume: mx.array) -> mx.array:
        """On Balance Volume."""
        price_diff = self._diff(close, prepend=close[0])
        volume_direction = mx.sign(price_diff) * volume
        return mx.cumsum(volume_direction)
    
    def _accumulation_distribution(self, data: GPUData) -> mx.array:
        """Accumulation/Distribution indicator."""
        clv = ((data.close - data.low) - (data.high - data.close)) / (data.high - data.low + 1e-10)
        ad = mx.cumsum(clv * data.volume)
        return ad
    
    def _mfi(self, data: GPUData, period: int = 14) -> mx.array:
        """Money Flow Index."""
        typical_price = (data.high + data.low + data.close) / 3
        raw_money_flow = typical_price * data.volume
        
        # Positive and negative money flow
        price_diff = self._diff(typical_price, prepend=typical_price[0])
        positive_flow = mx.where(price_diff > 0, raw_money_flow, 0)
        negative_flow = mx.where(price_diff < 0, raw_money_flow, 0)
        
        # Rolling sums
        positive_mf = self._rolling_sum(positive_flow, period)
        negative_mf = self._rolling_sum(negative_flow, period)
        
        mfr = positive_mf / (negative_mf + 1e-10)
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi
    
    def _rolling_sum(self, data: mx.array, period: int) -> mx.array:
        """Rolling sum using cumsum for O(n) complexity."""
        cumsum = mx.cumsum(mx.pad(data, (1, 0), constant_values=0))
        rolling = cumsum[period:] - cumsum[:-period]
        return mx.pad(rolling, (period-1, 0), constant_values=mx.nan)
    
    def _atr(self, data: GPUData, period: int = 14) -> mx.array:
        """Average True Range."""
        # True Range calculation
        high_low = data.high - data.low
        
        # Use roll for previous close
        prev_close = mx.roll(data.close, 1)
        prev_close[0] = data.close[0]
        
        high_close = mx.abs(data.high - prev_close)
        low_close = mx.abs(data.low - prev_close)
        
        true_range = mx.maximum(high_low, mx.maximum(high_close, low_close))
        
        # ATR using Wilder's smoothing
        return self._wilder_smooth(true_range, period)
    
    def _natr(self, data: GPUData, period: int = 14) -> mx.array:
        """Normalized ATR."""
        atr = self._atr(data, period)
        return (atr / data.close) * 100
    
    def _stochastic(self, data: GPUData, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[mx.array, mx.array]:
        """Stochastic oscillator."""
        # Rolling high/low
        rolling_high = self._rolling_max(data.high, period)
        rolling_low = self._rolling_min(data.low, period)
        
        # %K calculation
        k_percent = 100 * (data.close - rolling_low) / (rolling_high - rolling_low + 1e-10)
        
        # Smooth %K to get %K
        k_smooth = self._sma(k_percent, smooth_k)
        
        # %D is SMA of %K
        d_percent = self._sma(k_smooth, smooth_d)
        
        return k_smooth, d_percent
    
    def _rolling_max(self, data: mx.array, period: int) -> mx.array:
        """Rolling maximum - memory efficient using strided approach."""
        n = len(data)
        if n < period:
            return mx.full(n, mx.nan)
        
        # Process in chunks to avoid memory issues
        chunk_size = min(1000, n - period + 1)
        result = []
        
        for start in range(0, n - period + 1, chunk_size):
            end = min(start + chunk_size, n - period + 1)
            chunk_max = []
            
            for i in range(start, end):
                chunk_max.append(mx.max(data[i:i+period]))
            
            result.extend(chunk_max)
        
        # Pad with NaN
        result_array = mx.array(result)
        return mx.pad(result_array, (period-1, 0), constant_values=mx.nan)
    
    def _rolling_min(self, data: mx.array, period: int) -> mx.array:
        """Rolling minimum - memory efficient using strided approach."""
        n = len(data)
        if n < period:
            return mx.full(n, mx.nan)
        
        # Process in chunks to avoid memory issues
        chunk_size = min(1000, n - period + 1)
        result = []
        
        for start in range(0, n - period + 1, chunk_size):
            end = min(start + chunk_size, n - period + 1)
            chunk_min = []
            
            for i in range(start, end):
                chunk_min.append(mx.min(data[i:i+period]))
            
            result.extend(chunk_min)
        
        # Pad with NaN
        result_array = mx.array(result)
        return mx.pad(result_array, (period-1, 0), constant_values=mx.nan)
    
    def _williams_r(self, data: GPUData, period: int = 14) -> mx.array:
        """Williams %R."""
        highest_high = self._rolling_max(data.high, period)
        lowest_low = self._rolling_min(data.low, period)
        
        wr = -100 * (highest_high - data.close) / (highest_high - lowest_low + 1e-10)
        return wr
    
    def _cci(self, data: GPUData, period: int = 20) -> mx.array:
        """Commodity Channel Index."""
        typical_price = (data.high + data.low + data.close) / 3
        sma_tp = self._sma(typical_price, period)
        
        # Mean absolute deviation
        mad = self._rolling_mad(typical_price, period)
        
        cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
        return cci
    
    def _rolling_mad(self, data: mx.array, period: int) -> mx.array:
        """Rolling mean absolute deviation - memory efficient."""
        n = len(data)
        if n < period:
            return mx.full(n, mx.nan)
        
        # Use rolling mean for efficiency
        rolling_mean = self._sma(data, period)
        
        # Process in chunks
        chunk_size = min(1000, n - period + 1)
        result = []
        
        for start in range(0, n - period + 1, chunk_size):
            end = min(start + chunk_size, n - period + 1)
            
            for i in range(start, end):
                window = data[i:i+period]
                mean = rolling_mean[i+period-1]
                if not mx.isnan(mean):
                    mad = mx.mean(mx.abs(window - mean))
                    result.append(mad)
                else:
                    result.append(mx.nan)
        
        # Pad with NaN
        result_array = mx.array(result)
        return mx.pad(result_array, (period-1, 0), constant_values=mx.nan)
    
    def _adx(self, data: GPUData, period: int = 14) -> mx.array:
        """Average Directional Index."""
        # Calculate +DI and -DI
        plus_dm = mx.maximum(data.high - mx.roll(data.high, 1), 0)
        minus_dm = mx.maximum(mx.roll(data.low, 1) - data.low, 0)
        
        # Zero out where opposite is larger
        plus_dm = mx.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm = mx.where(minus_dm > plus_dm, minus_dm, 0)
        
        # ATR for normalization
        atr = self._atr(data, period)
        
        # Smoothed DI
        plus_di = 100 * self._wilder_smooth(plus_dm, period) / atr
        minus_di = 100 * self._wilder_smooth(minus_dm, period) / atr
        
        # DX and ADX
        dx = 100 * mx.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._wilder_smooth(dx, period)
        
        return adx


class MLXAVWAPCalculator:
    """GPU-accelerated AVWAP calculations."""
    
    def __init__(self, data: GPUData):
        self.data = data
        self.n = len(data)
        
        # Pre-compute cumulative values
        self.cum_pv = mx.cumsum(data.close * data.volume)
        self.cum_vol = mx.cumsum(data.volume)
        
        # Detect session/day/week boundaries
        self._detect_anchors()
    
    def _detect_anchors(self):
        """Detect anchor points for AVWAP calculations."""
        # For now, use simple modulo for sessions
        # In production, use actual timestamp data
        session_size = 390  # Regular trading session
        
        self.session_starts = mx.arange(0, self.n, session_size)
        self.day_starts = mx.arange(0, self.n, session_size)
        self.week_starts = mx.arange(0, self.n, session_size * 5)
        self.month_starts = mx.arange(0, self.n, session_size * 20)
    
    def calculate_all_avwap_gpu(self) -> Dict[str, mx.array]:
        """Calculate all AVWAP indicators on GPU."""
        indicators = {}
        
        # Session AVWAP
        indicators['AVWAP_SESSION'] = self._calculate_anchored_vwap(self.session_starts)
        
        # Daily AVWAP
        indicators['AVWAP_DAILY'] = self._calculate_anchored_vwap(self.day_starts)
        
        # Weekly AVWAP
        indicators['AVWAP_WEEKLY'] = self._calculate_anchored_vwap(self.week_starts)
        
        # Monthly AVWAP
        indicators['AVWAP_MONTHLY'] = self._calculate_anchored_vwap(self.month_starts)
        
        # Rolling AVWAP variants
        for period in [50, 100, 200]:
            indicators[f'AVWAP_ROLLING_{period}'] = self._calculate_rolling_avwap(period)
        
        # Add bands for session AVWAP
        session_avwap = indicators['AVWAP_SESSION']
        for std in [1, 2, 3]:
            upper, lower = self._calculate_avwap_bands(session_avwap, std)
            indicators[f'AVWAP_SESSION_U{std}'] = upper
            indicators[f'AVWAP_SESSION_L{std}'] = lower
        
        return indicators
    
    def _calculate_anchored_vwap(self, anchor_points: mx.array) -> mx.array:
        """Calculate AVWAP from anchor points."""
        avwap = mx.full(self.n, mx.nan)
        
        for i in range(len(anchor_points)):
            start_idx = anchor_points[i]
            end_idx = anchor_points[i+1] if i < len(anchor_points)-1 else self.n
            
            # VWAP from anchor point
            pv_from_anchor = self.cum_pv[start_idx:end_idx] - self.cum_pv[start_idx]
            vol_from_anchor = self.cum_vol[start_idx:end_idx] - self.cum_vol[start_idx]
            
            avwap[start_idx:end_idx] = pv_from_anchor / (vol_from_anchor + 1e-10)
        
        return avwap
    
    def _calculate_rolling_avwap(self, period: int) -> mx.array:
        """Rolling AVWAP using sliding window."""
        avwap = mx.full(self.n, mx.nan)
        
        for i in range(period, self.n):
            start_idx = i - period
            
            pv_window = self.cum_pv[i] - self.cum_pv[start_idx]
            vol_window = self.cum_vol[i] - self.cum_vol[start_idx]
            
            avwap[i] = pv_window / (vol_window + 1e-10)
        
        return avwap
    
    def _calculate_avwap_bands(self, avwap: mx.array, num_std: float) -> Tuple[mx.array, mx.array]:
        """Calculate AVWAP bands."""
        # Calculate rolling standard deviation of price from AVWAP
        price_diff = self.data.close - avwap
        
        # 20-period rolling std
        std = mx.zeros_like(avwap)
        for i in range(20, len(avwap)):
            window = price_diff[i-20:i]
            std[i] = mx.std(window)
        
        upper = avwap + num_std * std
        lower = avwap - num_std * std
        
        return upper, lower


def create_mlx_indicator_library():
    """Factory function to create MLX indicator library."""
    return MLXIndicatorLibrary()