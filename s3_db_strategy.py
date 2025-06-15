#!/usr/bin/env python3
"""
S3 Database Strategy - Real data version using PostgreSQL
=========================================================
Production-ready S3 strategy that uses real market data from the database
for backtesting and live trading applications.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import torch
import time
warnings.filterwarnings('ignore')

# Import components
from indicator_transformer import AIIndicatorSelector
from day_trading_o3 import DataLoader


class S3DatabaseStrategy:
    """
    Production S3 strategy using real database data with transformer-based
    indicator selection for optimal trading performance.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,
                 top_k_indicators: int = 15,
                 selection_interval: int = 100,
                 data_days_lookback: int = 30):
        """
        Initialize S3 database strategy.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
            capital: Starting capital
            adaptive_indicators: Use transformer-based selection
            top_k_indicators: Max indicators to select
            selection_interval: Bars between re-selection
            data_days_lookback: Days of data to load for analysis
        """
        
        self.symbol = symbol
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.data_days_lookback = data_days_lookback
        self.bars_since_selection = 0
        
        # Initialize database connection
        try:
            self.data_loader = DataLoader()
            print(f"✅ Database connection established")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")
        
        # Initialize transformer (if enabled)
        if adaptive_indicators:
            try:
                self.indicator_selector = AIIndicatorSelector(device='cpu')
                print(f"✅ AI indicator selector initialized")
            except Exception as e:
                print(f"⚠️ AI selector failed, falling back to basic indicators: {e}")
                self.adaptive_indicators = False
        
        # Strategy state
        self.current_indicators = {}
        self.last_market_regime = "Unknown"
        self.cached_data = None
        self.last_data_update = None
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        self.performance_history = []
        
        # Setup logging
        self.logger = logging.getLogger(f'S3DB_{symbol}')
        self.logger.info(f"🚀 S3 Database Strategy initialized for {symbol}")
        self.logger.info(f"   Capital: ${capital:,.2f}")
        self.logger.info(f"   Adaptive indicators: {adaptive_indicators}")
        self.logger.info(f"   Data lookback: {data_days_lookback} days")
    
    def load_historical_data(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           force_refresh: bool = False) -> pd.DataFrame:
        """
        Load historical data from database with caching.
        
        Args:
            start_date: Start date (YYYY-MM-DD) or None for auto
            end_date: End date (YYYY-MM-DD) or None for auto  
            force_refresh: Force reload even if cached
            
        Returns:
            DataFrame with OHLCV data
        """
        
        # Use cached data if available and recent
        if (not force_refresh and 
            self.cached_data is not None and 
            self.last_data_update is not None and
            (datetime.now() - self.last_data_update).seconds < 3600):  # 1 hour cache
            
            self.logger.info(f"📦 Using cached data ({len(self.cached_data)} bars)")
            return self.cached_data
        
        # Calculate date range if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=self.data_days_lookback)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        self.logger.info(f"📊 Loading {self.symbol} data from {start_date} to {end_date}...")
        
        try:
            # Load data from database
            data = self.data_loader.get_price_data(
                symbol=self.symbol,
                start=start_date,
                end=end_date
            )
            
            if data.empty:
                raise ValueError(f"No data available for {self.symbol} in date range")
            
            # Validate data quality
            self._validate_data_quality(data)
            
            # Cache the data
            self.cached_data = data
            self.last_data_update = datetime.now()
            
            self.logger.info(f"✅ Loaded {len(data):,} bars of data")
            self.logger.info(f"   Date range: {data.index[0]} to {data.index[-1]}")
            self.logger.info(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def _validate_data_quality(self, data: pd.DataFrame):
        """Validate data quality and completeness."""
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for NaN values
        nan_counts = data[required_cols].isnull().sum()
        if nan_counts.any():
            self.logger.warning(f"⚠️ Found NaN values: {dict(nan_counts[nan_counts > 0])}")
            # Forward fill NaN values
            data[required_cols] = data[required_cols].fillna(method='ffill')
        
        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            invalid_prices = (data[col] <= 0).sum()
            if invalid_prices > 0:
                self.logger.warning(f"⚠️ Found {invalid_prices} invalid prices in {col}")
        
        # Check data completeness (market hours coverage)
        total_bars = len(data)
        expected_bars = self.data_days_lookback * 390 * 0.7  # ~70% market days
        coverage = total_bars / expected_bars if expected_bars > 0 else 0
        
        self.logger.info(f"📊 Data quality:")
        self.logger.info(f"   Total bars: {total_bars:,}")
        self.logger.info(f"   Coverage estimate: {coverage:.1%}")
        self.logger.info(f"   Avg daily volume: {data['volume'].mean():,.0f}")
        
        if coverage < 0.5:
            self.logger.warning(f"⚠️ Low data coverage ({coverage:.1%})")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features using real market data."""
        
        start_time = time.time()
        
        # Always include basic market features
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_price_trend'] = features['volume_ratio'] * features['returns']
        
        # Volatility features
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        
        # Add transformer-selected indicators if enabled
        if self.adaptive_indicators:
            # Re-select indicators periodically
            if (self.bars_since_selection >= self.selection_interval or 
                len(self.current_indicators) == 0):
                self._select_indicators_from_real_data(data)
                self.bars_since_selection = 0
            else:
                self.bars_since_selection += 1
            
            # Add selected indicators
            selected_features = self._compute_selected_indicators(data)
            for col in selected_features.columns:
                features[col] = selected_features[col]
        else:
            # Add basic technical indicators
            features['sma_20'] = data['close'].rolling(20).mean()
            features['sma_50'] = data['close'].rolling(50).mean()
            features['ema_12'] = data['close'].ewm(span=12).mean()
            features['rsi_14'] = self._compute_rsi(data['close'], 14)
            features['macd'] = self._compute_macd(data['close'])
        
        # Clean features
        features = features.fillna(method='ffill').fillna(0)
        
        # Replace infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        prep_time = time.time() - start_time
        self.logger.info(f"📊 Prepared {features.shape[1]} features in {prep_time:.2f}s")
        
        return features
    
    def _select_indicators_from_real_data(self, data: pd.DataFrame):
        """Select indicators using transformer on real market data."""
        
        try:
            selection_start = time.time()
            
            # Use recent data for selection (last 1000 bars or all if less)
            recent_data = data.iloc[-min(1000, len(data)):].copy()
            
            self.logger.info(f"🎯 Selecting indicators using {len(recent_data)} recent bars...")
            
            # Get transformer selection
            selection = self.indicator_selector.select_indicators(
                data=recent_data,
                top_k=self.top_k_indicators
            )
            
            # Update state
            self.current_indicators = selection['selected_indicators']
            self.last_market_regime = selection['regime_detection']
            
            selection_time = time.time() - selection_start
            
            self.logger.info(f"✅ Indicator selection completed in {selection_time:.2f}s")
            self.logger.info(f"   Market regime: {self.last_market_regime}")
            self.logger.info(f"   Selected {len(self.current_indicators)} indicators")
            
            # Log top indicators
            if self.current_indicators:
                top_indicators = sorted(
                    self.current_indicators.items(),
                    key=lambda x: x[1]['weight'],
                    reverse=True
                )[:5]
                
                self.logger.info("   Top indicators:")
                for name, info in top_indicators:
                    self.logger.info(f"     {name}: weight={info['weight']:.3f}, "
                                   f"confidence={info['selection_prob']:.3f}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Indicator selection failed: {e}")
            # Fall back to empty selection
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
    
    def _compute_selected_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute transformer-selected indicators efficiently."""
        
        features = pd.DataFrame(index=data.index)
        
        # Sort indicators by weight (compute most important first)
        sorted_indicators = sorted(
            self.current_indicators.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )
        
        for indicator_name, info in sorted_indicators:
            try:
                weight = info['weight']
                
                # Compute based on indicator type
                if indicator_name.startswith('SMA_'):
                    period = int(indicator_name.split('_')[1])
                    values = data['close'].rolling(period).mean()
                    features[f'sma_{period}_w'] = values * weight
                    
                elif indicator_name.startswith('EMA_'):
                    period = int(indicator_name.split('_')[1])
                    values = data['close'].ewm(span=period).mean()
                    features[f'ema_{period}_w'] = values * weight
                    
                elif indicator_name == 'RSI_14':
                    values = self._compute_rsi(data['close'], 14)
                    features['rsi_14_w'] = values * weight
                    
                elif indicator_name == 'MACD':
                    values = self._compute_macd(data['close'])
                    features['macd_w'] = values * weight
                    
                elif indicator_name == 'ATR_14':
                    values = self._compute_atr(data, 14)
                    features['atr_14_w'] = values * weight
                    
                elif indicator_name == 'BB_20':
                    values = self._compute_bollinger_width(data['close'], 20)
                    features['bb_width_w'] = values * weight
                    
                elif indicator_name == 'OBV':
                    values = self._compute_obv(data)
                    features['obv_w'] = values * weight
                    
                elif indicator_name == 'VWAP':
                    values = self._compute_vwap(data)
                    features['vwap_ratio_w'] = (data['close'] / values) * weight
                    
                # Add high-confidence indicator interactions
                if info.get('selection_prob', 0) > 0.7:
                    feature_name = list(features.columns)[-1]  # Last added feature
                    if feature_name in features.columns:
                        features[f'{feature_name}_lag1'] = features[feature_name].shift(1)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to compute {indicator_name}: {e}")
                continue
        
        return features
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _compute_macd(self, prices: pd.Series) -> pd.Series:
        """Compute MACD line."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26
    
    def _compute_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def _compute_bollinger_width(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Compute Bollinger Band width."""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        return (std_dev * 2) / sma
    
    def _compute_obv(self, data: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume."""
        return (data['volume'] * (~data['close'].diff().le(0) * 2 - 1)).cumsum()
    
    def _compute_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Compute Volume Weighted Average Price."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
    def generate_signals(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on features and market regime.
        
        Args:
            features: Prepared feature DataFrame
            data: Original OHLCV data for context
            
        Returns:
            Series with signals (1=buy, -1=sell, 0=hold)
        """
        
        signals = pd.Series(0, index=features.index)
        
        # Basic momentum signals with regime adjustment
        if 'returns' in features.columns:
            # Multi-timeframe momentum
            momentum_short = features['returns'].rolling(5).mean()
            momentum_long = features['returns'].rolling(20).mean()
            
            # Regime-adjusted thresholds
            if "Strong Uptrend" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.0002, -0.0005
                momentum_weight = 1.2  # More aggressive in trends
            elif "Strong Downtrend" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.0005, -0.0002
                momentum_weight = 1.2
            elif "High Volatility" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.001, -0.001
                momentum_weight = 0.8  # More conservative in volatility
            elif "Ranging" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.0003, -0.0003
                momentum_weight = 1.0
            else:
                buy_thresh, sell_thresh = 0.0005, -0.0005
                momentum_weight = 1.0
            
            # Combined momentum signal
            momentum_signal = (momentum_short * 0.7 + momentum_long * 0.3) * momentum_weight
            
            signals[momentum_signal > buy_thresh] = 1
            signals[momentum_signal < sell_thresh] = -1
        
        # Volume confirmation
        if 'volume_ratio' in features.columns:
            volume_threshold = 1.5 if "High Volatility" in self.last_market_regime else 1.2
            high_volume = features['volume_ratio'] > volume_threshold
            
            # Only trade with volume confirmation
            signals = signals * high_volume.astype(int)
        
        # Volatility filter
        if 'volatility' in features.columns:
            # Don't trade in extremely high volatility unless it's a volatility regime
            if "High Volatility" not in self.last_market_regime:
                vol_percentile = features['volatility'].rolling(100).quantile(0.9)
                extreme_vol = features['volatility'] > vol_percentile
                signals[extreme_vol] = 0
        
        # RSI filter (if available)
        if any('rsi' in col for col in features.columns):
            rsi_col = [col for col in features.columns if 'rsi' in col][0]
            rsi_values = features[rsi_col]
            
            # Standard RSI overbought/oversold
            overbought = rsi_values > 70
            oversold = rsi_values < 30
            
            # Don't buy when overbought, don't sell when oversold
            signals[(signals == 1) & overbought] = 0
            signals[(signals == -1) & oversold] = 0
            
            # Add contrarian signals in ranging markets
            if "Ranging" in self.last_market_regime:
                signals[(rsi_values > 75) & (signals == 0)] = -1  # Sell extreme overbought
                signals[(rsi_values < 25) & (signals == 0)] = 1   # Buy extreme oversold
        
        # Price position filter
        if 'price_position' in features.columns:
            # Don't buy at daily highs, don't sell at daily lows
            at_high = features['price_position'] > 0.95
            at_low = features['price_position'] < 0.05
            
            signals[(signals == 1) & at_high] = 0
            signals[(signals == -1) & at_low] = 0
        
        return signals
    
    def calculate_position_size(self, 
                              signal: int, 
                              features: pd.Series, 
                              current_price: float) -> int:
        """
        Calculate position size with regime and confidence adjustments.
        
        Args:
            signal: Trading signal (1, -1, or 0)
            features: Feature values for current bar
            current_price: Current market price
            
        Returns:
            Number of shares to trade (positive for long, negative for short)
        """
        
        if signal == 0:
            return 0
        
        # Base position sizing (2% risk)
        risk_amount = self.capital * 0.02
        
        # Estimate stop loss distance
        if 'volatility' in features.index:
            volatility = features['volatility']
            stop_distance = max(current_price * 0.01, volatility * current_price * 2)
        else:
            stop_distance = current_price * 0.02  # 2% default
        
        base_shares = int(risk_amount / stop_distance)
        
        # Confidence adjustment based on indicators
        confidence_multiplier = 1.0
        if self.adaptive_indicators and self.current_indicators:
            avg_confidence = np.mean([
                info['selection_prob'] for info in self.current_indicators.values()
            ])
            confidence_multiplier = 0.6 + (avg_confidence * 0.8)  # 0.6 to 1.4 range
        
        # Regime-based adjustment
        regime_multipliers = {
            "Strong Uptrend": 1.2,
            "Strong Downtrend": 1.1,
            "High Volatility": 0.7,
            "Ranging/Sideways": 0.9,
            "Transitional": 0.8,
            "Unknown": 1.0
        }
        regime_multiplier = regime_multipliers.get(self.last_market_regime, 1.0)
        
        # Volume confirmation bonus
        volume_multiplier = 1.0
        if 'volume_ratio' in features.index and features['volume_ratio'] > 2.0:
            volume_multiplier = 1.1  # 10% bonus for high volume
        
        # Calculate final position size
        final_shares = int(base_shares * confidence_multiplier * regime_multiplier * volume_multiplier)
        
        # Apply maximum position limits (10% of capital)
        max_shares = int(self.capital * 0.1 / current_price)
        final_shares = min(abs(final_shares), max_shares)
        
        return final_shares * signal  # Apply direction
    
    def backtest(self, 
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive backtest using real database data.
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD) 
            verbose: Print progress updates
            
        Returns:
            Dictionary with detailed backtest results
        """
        
        if verbose:
            print(f"\n🚀 Starting backtest for {self.symbol}")
            print("=" * 50)
        
        # Load data
        data = self.load_historical_data(start_date, end_date)
        
        if len(data) < 100:
            raise ValueError(f"Insufficient data: {len(data)} bars (need at least 100)")
        
        # Prepare features
        if verbose:
            print("📊 Preparing features...")
        features = self.prepare_features(data)
        
        # Generate signals  
        if verbose:
            print("🎯 Generating signals...")
        signals = self.generate_signals(features, data)
        
        active_signals = (signals != 0).sum()
        if verbose:
            print(f"   Generated {active_signals:,} active signals")
        
        # Run simulation
        if verbose:
            print("💼 Running trading simulation...")
        
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        capital = self.capital
        max_capital = capital
        min_capital = capital
        
        # Skip first 100 bars for feature warmup
        for i in range(100, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]
            feature_row = features.iloc[i]
            
            # Close existing position
            if position != 0 and (signal != np.sign(position) or signal == 0):
                exit_price = current_price
                pnl = (exit_price - entry_price) * position
                
                trade = {
                    'symbol': self.symbol,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'pnl': pnl,
                    'return_pct': (exit_price - entry_price) / entry_price * (1 if position > 0 else -1),
                    'regime': self.last_market_regime,
                    'hold_bars': i - trades[-1]['entry_bar'] if trades else 0,
                    'indicators_used': list(self.current_indicators.keys()) if self.current_indicators else []
                }
                trades.append(trade)
                capital += pnl
                max_capital = max(max_capital, capital)
                min_capital = min(min_capital, capital)
                position = 0
            
            # Open new position
            if signal != 0 and position == 0:
                position_size = self.calculate_position_size(signal, feature_row, current_price)
                if abs(position_size) > 0:
                    position = position_size
                    entry_price = current_price
                    entry_time = current_time
                    if trades:
                        trades[-1]['entry_bar'] = i
        
        # Close final position if open
        if position != 0:
            exit_price = data['close'].iloc[-1]
            pnl = (exit_price - entry_price) * position
            trades.append({
                'symbol': self.symbol,
                'entry_time': entry_time,
                'exit_time': data.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'pnl': pnl,
                'return_pct': (exit_price - entry_price) / entry_price * (1 if position > 0 else -1),
                'regime': self.last_market_regime,
                'hold_bars': len(data) - 1 - (trades[-1]['entry_bar'] if trades else 100),
                'indicators_used': list(self.current_indicators.keys()) if self.current_indicators else []
            })
            capital += pnl
        
        # Calculate performance metrics
        if trades:
            total_pnl = sum(t['pnl'] for t in trades)
            returns = [t['return_pct'] for t in trades]
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            max_drawdown = (max_capital - min_capital) / max_capital
            total_return = (capital - self.capital) / self.capital
            
            # Sharpe ratio (annualized)
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Average holding period
            avg_hold_bars = np.mean([t['hold_bars'] for t in trades if 'hold_bars' in t])
            
        else:
            total_pnl = win_rate = avg_win = avg_loss = profit_factor = 0
            max_drawdown = total_return = sharpe_ratio = avg_hold_bars = 0
        
        # Compile results
        results = {
            'symbol': self.symbol,
            'start_date': data.index[0].strftime('%Y-%m-%d'),
            'end_date': data.index[-1].strftime('%Y-%m-%d'),
            'total_bars': len(data),
            'total_trades': len(trades),
            'winning_trades': len(winning_trades) if trades else 0,
            'losing_trades': len(losing_trades) if trades else 0,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_bars': avg_hold_bars,
            'final_capital': capital,
            'max_capital': max_capital,
            'min_capital': min_capital,
            'market_regime': self.last_market_regime,
            'features_used': features.shape[1],
            'signals_generated': active_signals,
            'indicators_selected': len(self.current_indicators) if self.adaptive_indicators else 0,
            'trades': trades,
            'data_quality': {
                'total_bars': len(data),
                'price_range': f"${data['close'].min():.2f} - ${data['close'].max():.2f}",
                'avg_volume': data['volume'].mean()
            }
        }
        
        # Print summary
        if verbose:
            print(f"\n📈 Backtest Results for {self.symbol}:")
            print(f"   Period: {results['start_date']} to {results['end_date']}")
            print(f"   Total trades: {results['total_trades']}")
            print(f"   Win rate: {results['win_rate']:.2%}")
            print(f"   Total return: {results['total_return']:.2%}")
            print(f"   Total PnL: ${results['total_pnl']:.2f}")
            print(f"   Profit factor: {results['profit_factor']:.2f}")
            print(f"   Sharpe ratio: {results['sharpe_ratio']:.2f}")
            print(f"   Max drawdown: {results['max_drawdown']:.2%}")
            print(f"   Market regime: {results['market_regime']}")
            print(f"   Features used: {results['features_used']}")
            if self.adaptive_indicators:
                print(f"   AI indicators: {results['indicators_selected']}")
        
        return results
    
    def get_latest_signals(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Get latest trading signals for recent data.
        
        Args:
            lookback_hours: Hours of recent data to analyze
            
        Returns:
            Dictionary with latest signals and market analysis
        """
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        # Load recent data
        data = self.load_historical_data(
            start_date=start_time.strftime('%Y-%m-%d'),
            end_date=end_time.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            return {'error': 'No recent data available'}
        
        # Prepare features and signals
        features = self.prepare_features(data)
        signals = self.generate_signals(features, data)
        
        # Get latest values
        latest_idx = -1
        latest_signal = signals.iloc[latest_idx]
        latest_price = data['close'].iloc[latest_idx]
        latest_features = features.iloc[latest_idx]
        
        # Calculate position size for latest signal
        position_size = self.calculate_position_size(
            latest_signal, 
            latest_features, 
            latest_price
        )
        
        return {
            'timestamp': data.index[latest_idx],
            'symbol': self.symbol,
            'price': latest_price,
            'signal': int(latest_signal),
            'signal_strength': abs(latest_signal),
            'position_size': position_size,
            'market_regime': self.last_market_regime,
            'recent_signals': signals.iloc[-10:].tolist(),
            'selected_indicators': list(self.current_indicators.keys()) if self.current_indicators else [],
            'data_bars': len(data)
        }


def main():
    """Example usage of S3 Database Strategy."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    print("🚀 S3 Database Strategy - Real Market Data")
    print("=" * 60)
    
    # Test with different symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        try:
            print(f"\n📊 Testing {symbol}...")
            
            # Create strategy instance
            strategy = S3DatabaseStrategy(
                symbol=symbol,
                capital=100000,
                adaptive_indicators=True,
                top_k_indicators=12,
                data_days_lookback=14  # 2 weeks of data
            )
            
            # Run backtest
            results = strategy.backtest(verbose=True)
            
            # Get latest signals
            latest = strategy.get_latest_signals(lookback_hours=48)
            if 'error' not in latest:
                print(f"\n🎯 Latest Signal for {symbol}:")
                print(f"   Price: ${latest['price']:.2f}")
                print(f"   Signal: {latest['signal']} ({'BUY' if latest['signal'] > 0 else 'SELL' if latest['signal'] < 0 else 'HOLD'})")
                print(f"   Position size: {latest['position_size']} shares")
                print(f"   Market regime: {latest['market_regime']}")
            
            # Save results
            results_file = f"backtest_results_{symbol}_{datetime.now().strftime('%Y%m%d')}.json"
            with open(results_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                json_results = results.copy()
                json_results['trades'] = []  # Remove trades for file size
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {results_file}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            continue
    
    print(f"\n🎉 S3 Database Strategy testing completed!")


if __name__ == "__main__":
    main()