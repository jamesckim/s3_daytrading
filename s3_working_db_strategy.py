#!/usr/bin/env python3
"""
S3 Working Strategy - Real Database Version
==========================================
Optimized version that uses real PostgreSQL data without hanging issues.
Based on s3_working_strategy but with database integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import time
warnings.filterwarnings('ignore')

# Import only what we need
from indicator_transformer import AIIndicatorSelector
from day_trading_o3 import DataLoader


class S3WorkingDatabaseStrategy:
    """
    Working S3 strategy with real database data and optimized transformer selection.
    Designed to avoid hanging issues while using production data.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,
                 top_k_indicators: int = 10,  # Reduced for speed
                 selection_interval: int = 100,
                 data_days_lookback: int = 7,  # 1 week default
                 db_timeout: int = 30):  # Database timeout
        """Initialize S3 working database strategy."""
        
        self.symbol = symbol
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.data_days_lookback = data_days_lookback
        self.db_timeout = db_timeout
        self.bars_since_selection = 0
        
        # Initialize database connection with timeout
        self.data_loader = None
        self._init_database_connection()
        
        # Initialize transformer (lightweight)
        if adaptive_indicators and self.data_loader is not None:
            try:
                self.indicator_selector = AIIndicatorSelector(device='cpu')
                self.current_indicators = {}
                self.last_market_regime = "Unknown"
                print(f"✅ AI indicator selector initialized")
            except Exception as e:
                print(f"⚠️ AI selector failed, using basic indicators: {e}")
                self.adaptive_indicators = False
                self.current_indicators = {}
                self.last_market_regime = "Unknown"
        else:
            self.adaptive_indicators = False
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        self.cached_data = None
        self.last_data_update = None
        
        # Setup logging
        self.logger = logging.getLogger(f'S3WorkingDB_{symbol}')
        
        print(f"🚀 S3 Working Database Strategy initialized for {symbol}")
        print(f"   Database connected: {self.data_loader is not None}")
        print(f"   Adaptive indicators: {self.adaptive_indicators}")
        print(f"   Data lookback: {data_days_lookback} days")
    
    def _init_database_connection(self):
        """Initialize database connection with timeout handling."""
        try:
            print(f"🗄️ Connecting to database...")
            start_time = time.time()
            
            # Simple connection test first
            self.data_loader = DataLoader()
            
            # Test with a quick query
            test_end = datetime.now().strftime('%Y-%m-%d')
            test_start = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Try to get a small amount of data to test connection
            test_data = self.data_loader.get_price_data(
                symbol=self.symbol,
                start=test_start,
                end=test_end
            )
            
            connection_time = time.time() - start_time
            
            if test_data.empty:
                print(f"⚠️ No data available for {self.symbol} in test range")
                print(f"   Connection successful but no recent data")
            else:
                print(f"✅ Database connection successful ({connection_time:.2f}s)")
                print(f"   Test data: {len(test_data)} bars available")
                
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print(f"   Will use synthetic data as fallback")
            self.data_loader = None
    
    def load_historical_data(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           force_refresh: bool = False) -> pd.DataFrame:
        """
        Load historical data from database with caching and fallback.
        """
        
        # Check cache first
        if (not force_refresh and 
            self.cached_data is not None and 
            self.last_data_update is not None and
            (datetime.now() - self.last_data_update).seconds < 1800):  # 30 min cache
            
            print(f"📦 Using cached data ({len(self.cached_data)} bars)")
            return self.cached_data
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=self.data_days_lookback)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        # Try to load from database
        if self.data_loader is not None:
            try:
                print(f"📊 Loading {self.symbol} data from database ({start_date} to {end_date})...")
                start_time = time.time()
                
                data = self.data_loader.get_price_data(
                    symbol=self.symbol,
                    start=start_date,
                    end=end_date
                )
                
                load_time = time.time() - start_time
                
                if not data.empty:
                    # Validate and cache data
                    self._validate_data_quality(data)
                    self.cached_data = data
                    self.last_data_update = datetime.now()
                    
                    print(f"✅ Loaded {len(data):,} bars from database ({load_time:.2f}s)")
                    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
                    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
                    
                    return data
                else:
                    print(f"⚠️ No database data available for {self.symbol}")
                    
            except Exception as e:
                print(f"❌ Database query failed: {e}")
        
        # Fallback to synthetic data
        print(f"🧪 Generating synthetic data as fallback...")
        return self._generate_synthetic_data(start_date, end_date)
    
    def _generate_synthetic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic market data for testing."""
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days
        
        # Generate minute-level data (assume 6.5 hours * 60 minutes = 390 minutes per day)
        total_minutes = days * 390
        dates = pd.date_range(start=start_dt, periods=total_minutes, freq='1min')
        
        np.random.seed(hash(self.symbol) % 2**32)  # Consistent seed per symbol
        
        # More realistic price simulation
        base_price = 100
        drift = 0.00001  # Small positive drift
        volatility = 0.002  # 0.2% per minute volatility
        
        returns = np.random.normal(drift, volatility, len(dates))
        
        # Add some regime changes
        regime_length = len(dates) // 3
        returns[:regime_length] += 0.0001  # Slight uptrend
        returns[regime_length:2*regime_length] -= 0.0001  # Slight downtrend
        
        # Calculate prices
        log_prices = np.log(base_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Generate OHLC data
        price_noise = np.random.normal(0, 0.0005, len(dates))
        
        data = pd.DataFrame({
            'open': prices * (1 + price_noise),
            'high': prices * (1 + np.abs(price_noise) + 0.001),
            'low': prices * (1 - np.abs(price_noise) - 0.001),
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Cache synthetic data
        self.cached_data = data
        self.last_data_update = datetime.now()
        
        print(f"✅ Generated {len(data):,} bars of synthetic data")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        return data
    
    def _validate_data_quality(self, data: pd.DataFrame):
        """Quick data quality validation."""
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Check for obvious data issues
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (data[col] <= 0).any():
                print(f"⚠️ Found invalid prices in {col}")
        
        # Fill any NaN values
        if data[required_cols].isnull().any().any():
            print(f"⚠️ Found NaN values, forward filling...")
            data[required_cols] = data[required_cols].fillna(method='ffill')
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features efficiently with real or synthetic data."""
        
        start_time = time.time()
        
        # Always include basic features
        features = pd.DataFrame(index=data.index)
        
        # Core price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['price_momentum'] = data['close'].pct_change(5)
        features['volatility'] = features['returns'].rolling(10).std()
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        
        # Add transformer-selected indicators if enabled
        if self.adaptive_indicators:
            # Only re-select periodically to avoid overhead
            if (self.bars_since_selection >= self.selection_interval or 
                len(self.current_indicators) == 0):
                self._select_indicators_fast(data)
                self.bars_since_selection = 0
            else:
                self.bars_since_selection += 1
            
            # Add selected indicators efficiently
            self._add_selected_features(data, features)
        else:
            # Add basic technical indicators
            features['sma_10'] = data['close'].rolling(10).mean()
            features['sma_20'] = data['close'].rolling(20).mean()
            features['ema_12'] = data['close'].ewm(span=12).mean()
            features['rsi'] = self._compute_rsi(data['close'], 14)
            features['macd'] = self._compute_macd(data['close'])
        
        # Clean features
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        prep_time = time.time() - start_time
        print(f"📊 Prepared {features.shape[1]} features in {prep_time:.2f}s")
        
        return features
    
    def _select_indicators_fast(self, data: pd.DataFrame):
        """Fast indicator selection with reduced complexity."""
        try:
            selection_start = time.time()
            
            # Use smaller sample for speed (last 300 bars or all if less)
            sample_data = data.iloc[-min(300, len(data)):].copy()
            
            print(f"🎯 Selecting indicators using {len(sample_data)} recent bars...")
            
            # Quick selection with timeout
            selection = self.indicator_selector.select_indicators(
                sample_data, 
                top_k=self.top_k_indicators
            )
            
            self.current_indicators = selection['selected_indicators']
            self.last_market_regime = selection['regime_detection']
            
            selection_time = time.time() - selection_start
            
            print(f"✅ Indicator selection completed in {selection_time:.2f}s")
            print(f"   Market regime: {self.last_market_regime}")
            print(f"   Selected {len(self.current_indicators)} indicators")
            
        except Exception as e:
            print(f"⚠️ Indicator selection failed: {e}")
            # Fall back to basic indicators
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
    
    def _add_selected_features(self, data: pd.DataFrame, features: pd.DataFrame):
        """Add transformer-selected indicators efficiently."""
        
        # Add top 5 indicators only to avoid complexity
        top_indicators = sorted(
            self.current_indicators.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )[:5]
        
        for indicator_name, info in top_indicators:
            try:
                weight = info['weight']
                
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
                    features['rsi_w'] = values * weight
                    
                elif indicator_name == 'MACD':
                    values = self._compute_macd(data['close'])
                    features['macd_w'] = values * weight
                    
                elif indicator_name == 'ATR_14':
                    values = self._compute_atr(data)
                    features['atr_w'] = values * weight
                
            except Exception as e:
                print(f"⚠️ Failed to compute {indicator_name}: {e}")
                continue
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI efficiently."""
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
        """Compute ATR."""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    def generate_signals(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on regime and features."""
        signals = pd.Series(0, index=features.index)
        
        # Basic momentum signals with regime adjustment
        if 'returns' in features.columns:
            momentum_short = features['returns'].rolling(5).mean()
            momentum_long = features['returns'].rolling(20).mean()
            
            # Regime-adjusted thresholds
            if "Uptrend" in self.last_market_regime or "Strong" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.0003, -0.0005
                momentum_weight = 1.1
            elif "Downtrend" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.0005, -0.0003
                momentum_weight = 1.1
            elif "Volatility" in self.last_market_regime or "High" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.001, -0.001
                momentum_weight = 0.8
            else:
                buy_thresh, sell_thresh = 0.0005, -0.0005
                momentum_weight = 1.0
            
            # Combined momentum signal
            momentum_signal = (momentum_short * 0.7 + momentum_long * 0.3) * momentum_weight
            
            signals[momentum_signal > buy_thresh] = 1
            signals[momentum_signal < sell_thresh] = -1
        
        # Volume confirmation
        if 'volume_ratio' in features.columns:
            volume_threshold = 1.5 if "Volatility" in self.last_market_regime else 1.2
            high_volume = features['volume_ratio'] > volume_threshold
            
            # Only trade with volume confirmation
            signals = signals * high_volume.astype(int)
        
        # RSI filter if available
        rsi_cols = [col for col in features.columns if 'rsi' in col.lower()]
        if rsi_cols:
            rsi_values = features[rsi_cols[0]]
            
            # Standard RSI overbought/oversold
            overbought = rsi_values > 70
            oversold = rsi_values < 30
            
            # Don't buy when overbought, don't sell when oversold
            signals[(signals == 1) & overbought] = 0
            signals[(signals == -1) & oversold] = 0
        
        return signals
    
    def backtest(self, 
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """Run comprehensive backtest using real or synthetic data."""
        
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
        
        # Skip first 50 bars for feature warmup
        for i in range(50, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Close existing position
            if position != 0 and (signal != np.sign(position) or signal == 0):
                exit_price = current_price
                pnl = (exit_price - entry_price) * abs(position)
                if position < 0:  # Short position
                    pnl = -pnl
                
                trade = {
                    'symbol': self.symbol,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'pnl': pnl,
                    'return_pct': (exit_price - entry_price) / entry_price * (1 if position > 0 else -1),
                    'regime': self.last_market_regime
                }
                trades.append(trade)
                capital += pnl
                position = 0
            
            # Open new position
            if signal != 0 and position == 0:
                # Simple position sizing (10% of capital)
                position_value = capital * 0.1
                shares = int(position_value / current_price)
                if shares > 0:
                    position = shares * signal
                    entry_price = current_price
                    entry_time = current_time
        
        # Close final position if open
        if position != 0:
            exit_price = data['close'].iloc[-1]
            pnl = (exit_price - entry_price) * abs(position)
            if position < 0:
                pnl = -pnl
            trades.append({
                'symbol': self.symbol,
                'entry_time': entry_time,
                'exit_time': data.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'pnl': pnl,
                'return_pct': (exit_price - entry_price) / entry_price * (1 if position > 0 else -1),
                'regime': self.last_market_regime
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
            
            total_return = (capital - self.capital) / self.capital
            
            # Sharpe ratio (annualized approximation)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            total_pnl = win_rate = avg_win = avg_loss = profit_factor = 0
            total_return = sharpe_ratio = 0
        
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
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_capital': capital,
            'market_regime': self.last_market_regime,
            'features_used': features.shape[1],
            'signals_generated': active_signals,
            'indicators_selected': len(self.current_indicators) if self.adaptive_indicators else 0,
            'data_source': 'database' if self.data_loader is not None else 'synthetic',
            'trades': trades
        }
        
        # Print summary
        if verbose:
            print(f"\n📈 Backtest Results for {self.symbol}:")
            print(f"   Data source: {results['data_source']}")
            print(f"   Period: {results['start_date']} to {results['end_date']}")
            print(f"   Total trades: {results['total_trades']}")
            print(f"   Win rate: {results['win_rate']:.2%}")
            print(f"   Total return: {results['total_return']:.2%}")
            print(f"   Total PnL: ${results['total_pnl']:.2f}")
            print(f"   Profit factor: {results['profit_factor']:.2f}")
            print(f"   Sharpe ratio: {results['sharpe_ratio']:.2f}")
            print(f"   Market regime: {results['market_regime']}")
            print(f"   Features used: {results['features_used']}")
            if self.adaptive_indicators:
                print(f"   AI indicators: {results['indicators_selected']}")
        
        return results
    
    def get_latest_signals(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get latest trading signals for recent data."""
        
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
        
        return {
            'timestamp': data.index[latest_idx],
            'symbol': self.symbol,
            'price': latest_price,
            'signal': int(latest_signal),
            'signal_strength': abs(latest_signal),
            'market_regime': self.last_market_regime,
            'recent_signals': signals.iloc[-10:].tolist(),
            'selected_indicators': list(self.current_indicators.keys()) if self.current_indicators else [],
            'data_bars': len(data),
            'data_source': 'database' if self.data_loader is not None else 'synthetic'
        }


def main():
    """Example usage of S3 Working Database Strategy."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    print("🚀 S3 Working Database Strategy - Real Market Data")
    print("=" * 60)
    
    # Test with a few symbols
    symbols = ['AAPL', 'MSFT']
    
    for symbol in symbols:
        try:
            print(f"\n📊 Testing {symbol}...")
            
            # Create strategy instance
            strategy = S3WorkingDatabaseStrategy(
                symbol=symbol,
                capital=100000,
                adaptive_indicators=True,
                top_k_indicators=8,  # Reduced for speed
                data_days_lookback=5  # 5 days for quick test
            )
            
            # Run backtest
            results = strategy.backtest(verbose=True)
            
            # Get latest signals
            latest = strategy.get_latest_signals(lookback_hours=24)
            if 'error' not in latest:
                print(f"\n🎯 Latest Signal for {symbol}:")
                print(f"   Price: ${latest['price']:.2f}")
                signal_text = 'BUY' if latest['signal'] > 0 else 'SELL' if latest['signal'] < 0 else 'HOLD'
                print(f"   Signal: {signal_text}")
                print(f"   Market regime: {latest['market_regime']}")
                print(f"   Data source: {latest['data_source']}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            continue
    
    print(f"\n🎉 S3 Working Database Strategy testing completed!")


if __name__ == "__main__":
    main()