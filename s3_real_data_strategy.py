#!/usr/bin/env python3
"""
S3 Real Data Strategy - Database version with robust connection handling
======================================================================
Production-ready S3 strategy that uses real PostgreSQL data with proper
timeout handling and connection management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
import psycopg2
from psycopg2 import pool
import time
from contextlib import contextmanager
warnings.filterwarnings('ignore')

# Import only the transformer if needed
try:
    from indicator_transformer import AIIndicatorSelector
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI indicator selector not available")


class DatabaseConnection:
    """Robust database connection manager with pooling and timeouts."""
    
    def __init__(self, max_connections: int = 5, connection_timeout: int = 10):
        """Initialize database connection pool."""
        self.pool = None
        self.connection_timeout = connection_timeout
        self.logger = logging.getLogger('DatabaseConnection')
        
        # Get connection string from environment
        self.connection_string = os.getenv('POSTGRES_URI')
        if not self.connection_string:
            # Try alternative env var
            self.connection_string = os.getenv('DATABASE_URL')
        
        if self.connection_string:
            self._init_pool(max_connections)
        else:
            self.logger.warning("No database connection string found in environment")
    
    def _init_pool(self, max_connections: int):
        """Initialize connection pool with timeout."""
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                1, max_connections,
                self.connection_string,
                connect_timeout=self.connection_timeout
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            self.pool = None
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with automatic cleanup."""
        if not self.pool:
            yield None
            return
        
        conn = None
        try:
            conn = self.pool.getconn()
            conn.set_session(autocommit=True, readonly=True)
            yield conn
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            yield None
        finally:
            if conn and self.pool:
                self.pool.putconn(conn)
    
    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            self.logger.info("Connection pool closed")


class S3RealDataStrategy:
    """
    S3 strategy using real database data with robust error handling.
    Falls back to synthetic data if database is unavailable.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,
                 top_k_indicators: int = 10,
                 selection_interval: int = 100,
                 data_days_lookback: int = 14,
                 db_timeout: int = 5):
        """Initialize S3 real data strategy."""
        
        self.symbol = symbol
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators and AI_AVAILABLE
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.data_days_lookback = data_days_lookback
        self.bars_since_selection = 0
        
        # Initialize database connection
        self.db_conn = DatabaseConnection(connection_timeout=db_timeout)
        
        # Initialize transformer if available and requested
        if self.adaptive_indicators:
            try:
                self.indicator_selector = AIIndicatorSelector(device='cpu')
                self.current_indicators = {}
                self.last_market_regime = "Unknown"
                print(f"✅ AI indicator selector initialized")
            except Exception as e:
                print(f"⚠️ AI selector initialization failed: {e}")
                self.adaptive_indicators = False
        
        if not self.adaptive_indicators:
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
        
        # Performance tracking
        self.trades = []
        self.cached_data = None
        self.last_data_update = None
        
        # Setup logging
        self.logger = logging.getLogger(f'S3Real_{symbol}')
        
        print(f"🚀 S3 Real Data Strategy initialized for {symbol}")
        print(f"   Database available: {self.db_conn.pool is not None}")
        print(f"   Adaptive indicators: {self.adaptive_indicators}")
        print(f"   Data lookback: {data_days_lookback} days")
    
    def load_historical_data(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           force_refresh: bool = False) -> pd.DataFrame:
        """Load historical data from database with fallback to synthetic."""
        
        # Check cache first
        if (not force_refresh and 
            self.cached_data is not None and 
            self.last_data_update is not None and
            (datetime.now() - self.last_data_update).seconds < 3600):
            print(f"📦 Using cached data ({len(self.cached_data)} bars)")
            return self.cached_data
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=self.data_days_lookback)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        # Try database first
        data = self._load_from_database(start_date, end_date)
        
        if data is None or data.empty:
            print(f"🧪 Using synthetic data as fallback...")
            data = self._generate_synthetic_data(start_date, end_date)
        
        # Cache the data
        self.cached_data = data
        self.last_data_update = datetime.now()
        
        return data
    
    def _load_from_database(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data from PostgreSQL database with timeout handling."""
        
        with self.db_conn.get_connection() as conn:
            if conn is None:
                return None
            
            try:
                print(f"📊 Loading {self.symbol} from database ({start_date} to {end_date})...")
                
                query = """
                    SELECT 
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM minute_prices
                    WHERE symbol = %s
                        AND DATE(timestamp) >= %s
                        AND DATE(timestamp) <= %s
                    ORDER BY timestamp
                """
                
                # Use pandas read_sql with the connection
                data = pd.read_sql(
                    query,
                    conn,
                    params=(self.symbol, start_date, end_date),
                    index_col='timestamp',
                    parse_dates=['timestamp']
                )
                
                if not data.empty:
                    print(f"✅ Loaded {len(data):,} bars from database")
                    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
                    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
                    
                    # Validate data quality
                    self._validate_data(data)
                    
                    return data
                else:
                    print(f"⚠️ No data found for {self.symbol} in database")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Database query failed: {e}")
                return None
    
    def _validate_data(self, data: pd.DataFrame):
        """Validate and clean data."""
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle NaN values
        nan_counts = data[required_cols].isnull().sum()
        if nan_counts.any():
            print(f"⚠️ Found NaN values, forward filling...")
            data[required_cols] = data[required_cols].fillna(method='ffill')
        
        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (data[col] <= 0).any():
                print(f"⚠️ Found invalid prices in {col}, cleaning...")
                data.loc[data[col] <= 0, col] = np.nan
                data[col] = data[col].fillna(method='ffill')
    
    def _generate_synthetic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic data for testing."""
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Create minute-level timestamps for market hours only
        timestamps = []
        current_date = start_dt
        
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Monday-Friday
                # Market hours: 9:30 AM - 4:00 PM ET
                market_open = current_date.replace(hour=9, minute=30, second=0)
                market_close = current_date.replace(hour=16, minute=0, second=0)
                
                # Generate minute bars for this day
                day_timestamps = pd.date_range(
                    start=market_open,
                    end=market_close,
                    freq='1min'
                )[:-1]  # Exclude 4:00 PM
                
                timestamps.extend(day_timestamps)
            
            current_date += timedelta(days=1)
        
        if not timestamps:
            # If no market days, just create some data
            timestamps = pd.date_range(start=start_dt, periods=390, freq='1min')
        
        # Generate realistic price movements
        np.random.seed(hash(self.symbol) % 2**32)
        
        # Base parameters
        base_price = 150 if self.symbol == 'AAPL' else 100
        drift = 0.00002
        volatility = 0.0015
        
        # Generate returns with some patterns
        returns = np.random.normal(drift, volatility, len(timestamps))
        
        # Add some momentum
        for i in range(1, len(returns)):
            returns[i] += returns[i-1] * 0.1  # 10% momentum
        
        # Calculate prices
        log_prices = np.log(base_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Generate OHLC data
        data = pd.DataFrame(index=timestamps)
        data['close'] = prices
        
        # Generate realistic OHLC bars
        bar_volatility = np.random.uniform(0.0005, 0.002, len(timestamps))
        data['open'] = data['close'] * (1 + np.random.normal(0, bar_volatility))
        data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, bar_volatility)))
        data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, bar_volatility)))
        
        # Generate volume with intraday pattern
        base_volume = 1000000
        time_of_day = (data.index.hour - 9.5) / 6.5  # Normalize to 0-1
        volume_pattern = 1 + 0.5 * np.sin(np.pi * time_of_day)  # U-shaped pattern
        data['volume'] = (base_volume * volume_pattern * 
                         np.random.uniform(0.5, 1.5, len(timestamps))).astype(int)
        
        print(f"✅ Generated {len(data):,} bars of synthetic data")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from market data."""
        
        start_time = time.time()
        features = pd.DataFrame(index=data.index)
        
        # Core features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        features['high_low_spread'] = (data['high'] - data['low']) / data['close']
        
        # Momentum features
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['close'].pct_change(period)
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        
        # Basic technical indicators
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        features['rsi_14'] = self._compute_rsi(data['close'], 14)
        
        # Add AI-selected indicators if enabled
        if self.adaptive_indicators:
            if (self.bars_since_selection >= self.selection_interval or 
                len(self.current_indicators) == 0):
                self._select_indicators(data)
                self.bars_since_selection = 0
            else:
                self.bars_since_selection += 1
            
            # Add selected indicators
            if self.current_indicators:
                self._add_selected_indicators(data, features)
        
        # Clean features
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        prep_time = time.time() - start_time
        self.logger.info(f"Prepared {features.shape[1]} features in {prep_time:.2f}s")
        
        return features
    
    def _select_indicators(self, data: pd.DataFrame):
        """Select indicators using transformer."""
        try:
            # Use recent data for selection
            recent_data = data.iloc[-min(500, len(data)):].copy()
            
            selection = self.indicator_selector.select_indicators(
                recent_data,
                top_k=self.top_k_indicators
            )
            
            self.current_indicators = selection['selected_indicators']
            self.last_market_regime = selection['regime_detection']
            
            print(f"🎯 Selected {len(self.current_indicators)} indicators, regime: {self.last_market_regime}")
            
        except Exception as e:
            self.logger.warning(f"Indicator selection failed: {e}")
            self.current_indicators = {}
    
    def _add_selected_indicators(self, data: pd.DataFrame, features: pd.DataFrame):
        """Add AI-selected indicators to features."""
        
        # Add top weighted indicators
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
                    features[f'ai_sma_{period}'] = data['close'].rolling(period).mean() * weight
                    
                elif indicator_name.startswith('EMA_'):
                    period = int(indicator_name.split('_')[1])
                    features[f'ai_ema_{period}'] = data['close'].ewm(span=period).mean() * weight
                    
                elif indicator_name == 'RSI_14':
                    features['ai_rsi'] = self._compute_rsi(data['close'], 14) * weight
                    
                elif indicator_name == 'MACD':
                    features['ai_macd'] = self._compute_macd(data['close']) * weight
                    
            except Exception as e:
                self.logger.warning(f"Failed to add {indicator_name}: {e}")
    
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
    
    def generate_signals(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        
        signals = pd.Series(0, index=features.index)
        
        # Momentum-based signals
        if 'momentum_5' in features.columns and 'momentum_20' in features.columns:
            # Combined momentum
            momentum = features['momentum_5'] * 0.7 + features['momentum_20'] * 0.3
            
            # Regime-based thresholds
            if "Uptrend" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.0003, -0.0005
            elif "Downtrend" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.0005, -0.0003
            elif "Volatility" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.001, -0.001
            else:
                buy_thresh, sell_thresh = 0.0005, -0.0005
            
            signals[momentum > buy_thresh] = 1
            signals[momentum < sell_thresh] = -1
        
        # Volume confirmation
        if 'volume_ratio' in features.columns:
            high_volume = features['volume_ratio'] > 1.2
            signals = signals * high_volume.astype(int)
        
        # RSI filter
        if 'rsi_14' in features.columns:
            overbought = features['rsi_14'] > 70
            oversold = features['rsi_14'] < 30
            
            signals[(signals == 1) & overbought] = 0
            signals[(signals == -1) & oversold] = 0
        
        # Volatility filter
        if 'volatility_ratio' in features.columns:
            extreme_vol = features['volatility_ratio'] > 2.0
            if "Volatility" not in self.last_market_regime:
                signals[extreme_vol] = 0
        
        return signals
    
    def backtest(self, 
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """Run backtest on historical data."""
        
        if verbose:
            print(f"\n🚀 Starting backtest for {self.symbol}")
            print("=" * 50)
        
        # Load data
        data = self.load_historical_data(start_date, end_date)
        
        if len(data) < 100:
            raise ValueError(f"Insufficient data: {len(data)} bars")
        
        # Prepare features and signals
        features = self.prepare_features(data)
        signals = self.generate_signals(features, data)
        
        active_signals = (signals != 0).sum()
        if verbose:
            print(f"📊 Generated {active_signals:,} active signals")
        
        # Run simulation
        trades = []
        position = 0
        entry_price = 0
        capital = self.capital
        
        for i in range(100, len(data)):  # Skip warmup period
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Close position if needed
            if position != 0 and signal != np.sign(position):
                pnl = (current_price - entry_price) * position
                trades.append({
                    'entry': entry_price,
                    'exit': current_price,
                    'pnl': pnl,
                    'position': position
                })
                capital += pnl
                position = 0
            
            # Open new position
            if signal != 0 and position == 0:
                shares = int(capital * 0.1 / current_price)
                if shares > 0:
                    position = shares * signal
                    entry_price = current_price
        
        # Calculate metrics
        if trades:
            total_pnl = sum(t['pnl'] for t in trades)
            wins = [t for t in trades if t['pnl'] > 0]
            win_rate = len(wins) / len(trades)
            
            results = {
                'symbol': self.symbol,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'final_capital': capital,
                'total_return': (capital - self.capital) / self.capital,
                'market_regime': self.last_market_regime,
                'data_source': 'database' if self.db_conn.pool else 'synthetic',
                'data_bars': len(data),
                'signals_generated': active_signals
            }
        else:
            results = {
                'symbol': self.symbol,
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'final_capital': self.capital,
                'total_return': 0,
                'market_regime': self.last_market_regime,
                'data_source': 'database' if self.db_conn.pool else 'synthetic',
                'data_bars': len(data),
                'signals_generated': active_signals
            }
        
        if verbose:
            print(f"\n📈 Backtest Results:")
            print(f"   Data source: {results['data_source']}")
            print(f"   Total trades: {results['total_trades']}")
            print(f"   Win rate: {results['win_rate']:.1%}")
            print(f"   Total PnL: ${results['total_pnl']:.2f}")
            print(f"   Total return: {results['total_return']:.1%}")
        
        return results
    
    def __del__(self):
        """Cleanup database connections."""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()


def main():
    """Test the S3 real data strategy."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    print("🚀 S3 Real Data Strategy Test")
    print("=" * 60)
    
    # Test with AAPL
    strategy = S3RealDataStrategy(
        symbol='AAPL',
        capital=100000,
        adaptive_indicators=True,
        data_days_lookback=7
    )
    
    # Run backtest
    results = strategy.backtest(verbose=True)
    
    print("\n🎉 Test completed!")


if __name__ == "__main__":
    main()