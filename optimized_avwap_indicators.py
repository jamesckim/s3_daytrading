#!/usr/bin/env python3
"""
Optimized AVWAP Indicators for Intraday Trading
===============================================
Fast computation of Anchored VWAP indicators using cumulative sums.
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


class OptimizedAVWAPCalculator:
    """Optimized AVWAP calculation for intraday trading."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price/volume data.
        
        Args:
            data: DataFrame with OHLCV data and datetime index
        """
        self.data = data
        self.close = data['close'].values
        self.volume = data['volume'].values
        self.timestamps = data.index
        
        # Pre-compute cumulative sums for efficiency
        self._precompute_cumulative_values()
        
    def _precompute_cumulative_values(self):
        """Pre-compute cumulative sums for fast AVWAP calculation."""
        # Cumulative volume
        self.cum_volume = np.cumsum(self.volume)
        
        # Cumulative price * volume (using typical price)
        typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        self.cum_pv = np.cumsum(typical_price * self.volume)
        
        # Pre-compute session boundaries for intraday anchors
        self._find_session_boundaries()
        
    def _find_session_boundaries(self):
        """Find market session boundaries for anchoring."""
        self.session_starts = []
        self.day_starts = []
        self.week_starts = []
        self.month_starts = []
        
        # Market hours (adjust for your market)
        market_open = time(9, 30)  # 9:30 AM
        market_close = time(16, 0)  # 4:00 PM
        
        prev_date = None
        prev_week = None
        prev_month = None
        
        for i, ts in enumerate(self.timestamps):
            current_date = ts.date()
            current_week = ts.isocalendar()[1]
            current_month = ts.month
            
            # New trading day
            if current_date != prev_date:
                self.day_starts.append(i)
                prev_date = current_date
                
                # Check for new week
                if current_week != prev_week:
                    self.week_starts.append(i)
                    prev_week = current_week
                
                # Check for new month
                if current_month != prev_month:
                    self.month_starts.append(i)
                    prev_month = current_month
            
            # New session (for futures/extended hours)
            if ts.time() == market_open:
                self.session_starts.append(i)
    
    def calculate_avwap_from_index(self, anchor_idx: int) -> np.ndarray:
        """
        Calculate AVWAP from a specific index (anchor point).
        
        This is the key optimization - O(n) instead of O(n²)
        """
        n = len(self.close)
        avwap = np.full(n, np.nan)
        
        if anchor_idx >= n:
            return avwap
        
        # Get cumulative values at anchor
        if anchor_idx > 0:
            anchor_cum_pv = self.cum_pv[anchor_idx - 1]
            anchor_cum_vol = self.cum_volume[anchor_idx - 1]
        else:
            anchor_cum_pv = 0
            anchor_cum_vol = 0
        
        # Calculate AVWAP from anchor point onwards
        # AVWAP = (Cumulative PV - Anchor PV) / (Cumulative Volume - Anchor Volume)
        cum_pv_from_anchor = self.cum_pv[anchor_idx:] - anchor_cum_pv
        cum_vol_from_anchor = self.cum_volume[anchor_idx:] - anchor_cum_vol
        
        # Avoid division by zero
        mask = cum_vol_from_anchor > 0
        avwap[anchor_idx:][mask] = cum_pv_from_anchor[mask] / cum_vol_from_anchor[mask]
        
        return avwap
    
    def calculate_session_avwap(self) -> pd.Series:
        """Calculate AVWAP from session start (most important for intraday)."""
        avwap = np.full(len(self.close), np.nan)
        
        for i in range(len(self.session_starts)):
            start_idx = self.session_starts[i]
            
            # Find end of this session
            if i < len(self.session_starts) - 1:
                end_idx = self.session_starts[i + 1]
            else:
                end_idx = len(self.close)
            
            # Calculate AVWAP for this session
            session_avwap = self.calculate_avwap_from_index(start_idx)
            avwap[start_idx:end_idx] = session_avwap[start_idx:end_idx]
        
        return pd.Series(avwap, index=self.timestamps, name='AVWAP_SESSION')
    
    def calculate_daily_avwap(self) -> pd.Series:
        """Calculate AVWAP from day start."""
        avwap = np.full(len(self.close), np.nan)
        
        for i in range(len(self.day_starts)):
            start_idx = self.day_starts[i]
            
            # Find end of this day
            if i < len(self.day_starts) - 1:
                end_idx = self.day_starts[i + 1]
            else:
                end_idx = len(self.close)
            
            # Calculate AVWAP for this day
            daily_avwap = self.calculate_avwap_from_index(start_idx)
            avwap[start_idx:end_idx] = daily_avwap[start_idx:end_idx]
        
        return pd.Series(avwap, index=self.timestamps, name='AVWAP_DAILY')
    
    def calculate_weekly_avwap(self) -> pd.Series:
        """Calculate AVWAP from week start."""
        avwap = np.full(len(self.close), np.nan)
        
        for i in range(len(self.week_starts)):
            start_idx = self.week_starts[i]
            
            # Find end of this week
            if i < len(self.week_starts) - 1:
                end_idx = self.week_starts[i + 1]
            else:
                end_idx = len(self.close)
            
            # Calculate AVWAP for this week
            weekly_avwap = self.calculate_avwap_from_index(start_idx)
            avwap[start_idx:end_idx] = weekly_avwap[start_idx:end_idx]
        
        return pd.Series(avwap, index=self.timestamps, name='AVWAP_WEEKLY')
    
    def calculate_monthly_avwap(self) -> pd.Series:
        """Calculate AVWAP from month start."""
        avwap = np.full(len(self.close), np.nan)
        
        for i in range(len(self.month_starts)):
            start_idx = self.month_starts[i]
            
            # Find end of this month
            if i < len(self.month_starts) - 1:
                end_idx = self.month_starts[i + 1]
            else:
                end_idx = len(self.close)
            
            # Calculate AVWAP for this month
            monthly_avwap = self.calculate_avwap_from_index(start_idx)
            avwap[start_idx:end_idx] = monthly_avwap[start_idx:end_idx]
        
        return pd.Series(avwap, index=self.timestamps, name='AVWAP_MONTHLY')
    
    def calculate_rolling_avwap(self, periods: int) -> pd.Series:
        """Calculate rolling AVWAP (e.g., last N bars)."""
        n = len(self.close)
        avwap = np.full(n, np.nan)
        
        # For each point, calculate AVWAP from N bars ago
        for i in range(periods, n):
            anchor_idx = i - periods
            
            # Use pre-computed cumulative values
            if anchor_idx > 0:
                cum_pv_window = self.cum_pv[i] - self.cum_pv[anchor_idx - 1]
                cum_vol_window = self.cum_volume[i] - self.cum_volume[anchor_idx - 1]
            else:
                cum_pv_window = self.cum_pv[i]
                cum_vol_window = self.cum_volume[i]
            
            if cum_vol_window > 0:
                avwap[i] = cum_pv_window / cum_vol_window
        
        return pd.Series(avwap, index=self.timestamps, name=f'AVWAP_ROLLING_{periods}')
    
    def calculate_avwap_bands(self, avwap: pd.Series, num_std: float = 1.0) -> tuple:
        """Calculate AVWAP bands for support/resistance."""
        # Calculate deviation from AVWAP
        deviation = self.close - avwap.values
        
        # Use rolling window for standard deviation
        window = 20  # Adjust based on timeframe
        rolling_std = pd.Series(deviation).rolling(window).std()
        
        upper_band = avwap + num_std * rolling_std
        lower_band = avwap - num_std * rolling_std
        
        return upper_band, lower_band
    
    def calculate_all_avwap_indicators(self) -> dict:
        """Calculate all AVWAP indicators efficiently."""
        # Removed verbose logging - calculation in progress
        start_time = datetime.now()
        
        indicators = {}
        
        # Core AVWAP indicators for intraday trading
        indicators['AVWAP_SESSION'] = self.calculate_session_avwap()
        indicators['AVWAP_DAILY'] = self.calculate_daily_avwap()
        indicators['AVWAP_WEEKLY'] = self.calculate_weekly_avwap()
        indicators['AVWAP_MONTHLY'] = self.calculate_monthly_avwap()
        
        # Rolling AVWAP for different timeframes
        indicators['AVWAP_ROLLING_100'] = self.calculate_rolling_avwap(100)
        indicators['AVWAP_ROLLING_200'] = self.calculate_rolling_avwap(200)
        indicators['AVWAP_ROLLING_500'] = self.calculate_rolling_avwap(500)
        
        # AVWAP bands (1, 2, 3 standard deviations)
        session_avwap = indicators['AVWAP_SESSION']
        
        upper1, lower1 = self.calculate_avwap_bands(session_avwap, 1.0)
        indicators['AVWAP_SESSION_U1'] = upper1
        indicators['AVWAP_SESSION_L1'] = lower1
        
        upper2, lower2 = self.calculate_avwap_bands(session_avwap, 2.0)
        indicators['AVWAP_SESSION_U2'] = upper2
        indicators['AVWAP_SESSION_L2'] = lower2
        
        upper3, lower3 = self.calculate_avwap_bands(session_avwap, 3.0)
        indicators['AVWAP_SESSION_U3'] = upper3
        indicators['AVWAP_SESSION_L3'] = lower3
        
        # Silent calculation - removed per-symbol logging
        
        return indicators


def integrate_with_indicator_library(indicator_library):
    """Integrate optimized AVWAP calculation with existing indicator library."""
    
    # Store original AVWAP functions
    original_avwap_funcs = {}
    avwap_indicators = []
    
    # Find all AVWAP indicators
    for name, func in indicator_library.indicators.items():
        if 'AVWAP' in name or 'VWAP' in name:
            original_avwap_funcs[name] = func
            avwap_indicators.append(name)
    
    logger.info(f"Found {len(avwap_indicators)} VWAP/AVWAP indicators to optimize")
    
    # Create optimized computation function
    def compute_all_vwap_optimized(data: pd.DataFrame) -> dict:
        """Compute all VWAP indicators at once."""
        # Regular VWAP (non-anchored)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        results = {'VWAP': vwap}
        
        # VWAP bands
        for num_std in [1, 2, 3]:
            deviation = typical_price - vwap
            rolling_std = deviation.rolling(20).std()
            results[f'VWAP_U{num_std}'] = vwap + num_std * rolling_std
            results[f'VWAP_L{num_std}'] = vwap - num_std * rolling_std
        
        # Optimized AVWAP
        calculator = OptimizedAVWAPCalculator(data)
        avwap_results = calculator.calculate_all_avwap_indicators()
        results.update(avwap_results)
        
        return results
    
    # Replace individual AVWAP functions with cached lookup
    _vwap_cache = {}
    _vwap_cache_key = None
    
    def create_optimized_vwap_func(indicator_name):
        def optimized_func(data: pd.DataFrame) -> pd.Series:
            nonlocal _vwap_cache, _vwap_cache_key
            
            # Create cache key from data
            cache_key = (len(data), data.index[0], data.index[-1])
            
            # Check if we need to recompute
            if cache_key != _vwap_cache_key or indicator_name not in _vwap_cache:
                # Compute all VWAP indicators at once
                _vwap_cache = compute_all_vwap_optimized(data)
                _vwap_cache_key = cache_key
            
            # Return the specific indicator
            return _vwap_cache.get(indicator_name, pd.Series(np.nan, index=data.index))
        
        return optimized_func
    
    # Replace AVWAP functions with optimized versions
    for name in avwap_indicators:
        indicator_library.indicators[name] = create_optimized_vwap_func(name)
    
    logger.info("✅ Integrated optimized AVWAP computation")
    
    return indicator_library


# Example usage for testing
if __name__ == "__main__":
    # Generate test data
    dates = pd.date_range(start='2024-01-01 09:30', end='2024-01-05 16:00', freq='1min')
    
    # Filter for market hours only
    market_hours = []
    for ts in dates:
        if 9 <= ts.hour < 16 or (ts.hour == 16 and ts.minute == 0):
            market_hours.append(ts)
    
    # Create test OHLCV data
    n = len(market_hours)
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(n) * 0.1)
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(n) * 0.05,
        'high': prices + np.abs(np.random.randn(n) * 0.1),
        'low': prices - np.abs(np.random.randn(n) * 0.1),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    }, index=pd.DatetimeIndex(market_hours))
    
    # Test optimized calculation
    print("Testing optimized AVWAP calculation...")
    calculator = OptimizedAVWAPCalculator(data)
    
    import time
    start = time.time()
    indicators = calculator.calculate_all_avwap_indicators()
    elapsed = time.time() - start
    
    print(f"\nCalculated {len(indicators)} indicators in {elapsed:.3f}s")
    print(f"Speed: {len(data) / elapsed:.0f} data points/second")
    
    # Show sample results
    print("\nSample AVWAP values:")
    for name, series in list(indicators.items())[:5]:
        last_values = series.dropna().tail(3)
        print(f"{name}: {last_values.values}")