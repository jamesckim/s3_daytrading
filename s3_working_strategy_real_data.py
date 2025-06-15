#!/usr/bin/env python3
"""
S3 Working Strategy - Real Database Data Version
================================================
Enhanced version of s3_working_strategy that uses real market data from PostgreSQL
instead of synthetic data for production-ready backtesting and analysis.

Key improvements over original:
- Real data from PostgreSQL database via DataLoader
- Enhanced data validation and quality checks
- Robust error handling and fallback mechanisms
- Production-ready caching and performance optimization
- Comprehensive backtesting with real market conditions
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

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from indicator_transformer import AIIndicatorSelector
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

# Required imports
from day_trading_o3 import DataLoader


class S3WorkingStrategyRealData:
    """
    Production S3 working strategy using real database data with transformer-based 
    indicator selection. Optimized for reliability and performance.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,
                 top_k_indicators: int = 15,
                 selection_interval: int = 50,
                 data_days_lookback: int = 7,
                 db_timeout: int = 30,
                 enable_caching: bool = True):
        """
        Initialize S3 working strategy with real data.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
            capital: Starting capital for backtesting
            adaptive_indicators: Use transformer-based indicator selection
            top_k_indicators: Maximum indicators to select
            selection_interval: Bars between indicator re-selection
            data_days_lookback: Days of historical data to load
            db_timeout: Database connection timeout in seconds
            enable_caching: Enable data caching for performance
        """
        
        self.symbol = symbol.upper()
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.data_days_lookback = data_days_lookback
        self.db_timeout = db_timeout
        self.enable_caching = enable_caching
        self.bars_since_selection = 0
        
        # Initialize database connection
        self.data_loader = None
        self._init_database_connection()
        
        # Initialize transformer (lightweight)
        if adaptive_indicators and self.data_loader is not None and TRANSFORMER_AVAILABLE:
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
            if adaptive_indicators and not TRANSFORMER_AVAILABLE:
                print(f"⚠️ Transformer not available, using basic indicators")
            self.adaptive_indicators = False
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        self.cached_data = None
        self.last_data_update = None
        self.performance_history = []
        
        # Setup logging
        self.logger = logging.getLogger(f'S3RealData_{symbol}')
        self.logger.setLevel(logging.INFO)
        
        print(f"🚀 S3 Working Strategy (Real Data) initialized for {symbol}")
        print(f"   Database connected: {self.data_loader is not None}")
        print(f"   Adaptive indicators: {self.adaptive_indicators}")
        print(f"   Data lookback: {data_days_lookback} days")
        print(f"   Capital: ${capital:,.2f}")
    
    def _init_database_connection(self):
        """Initialize database connection with timeout handling."""
        try:
            print(f"🗄️ Connecting to database for {self.symbol}...")
            start_time = time.time()
            
            # Initialize DataLoader
            self.data_loader = DataLoader()
            
            # Test connection with small query
            test_end = datetime.now().strftime('%Y-%m-%d')
            test_start = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Quick connectivity test
            test_data = self.data_loader.get_price_data(
                symbol=self.symbol,
                start=test_start,
                end=test_end
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Database connection successful ({elapsed:.1f}s)")
            
            if not test_data.empty:
                print(f"📊 Test query returned {len(test_data)} bars")
            else:
                print(f"⚠️ No recent data found for {self.symbol}")
                
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print(f"   Strategy will use fallback synthetic data")
            self.data_loader = None
    
    def load_real_data(self, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      force_refresh: bool = False) -> pd.DataFrame:
        """
        Load real market data from database with intelligent caching.
        
        Args:
            start_date: Start date (YYYY-MM-DD) or None for auto
            end_date: End date (YYYY-MM-DD) or None for auto  
            force_refresh: Force reload even if cached
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        
        # Check cache first
        if (self.enable_caching and not force_refresh and 
            self.cached_data is not None and 
            self.last_data_update is not None and
            (datetime.now() - self.last_data_update).seconds < 1800):  # 30 min cache
            
            print(f"📦 Using cached data ({len(self.cached_data)} bars)")
            return self.cached_data
        
        # Calculate date range if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=self.data_days_lookback)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        print(f"📊 Loading real data for {self.symbol} from {start_date} to {end_date}...")
        
        # Fallback to synthetic data if no database connection
        if self.data_loader is None:
            print(f"⚠️ No database connection, generating synthetic data")
            return self._generate_fallback_data(start_date, end_date)
        
        try:
            # Load data from database
            data = self.data_loader.get_price_data(
                symbol=self.symbol,
                start=start_date,
                end=end_date
            )
            
            if data.empty:
                print(f"⚠️ No data available for {self.symbol}, using synthetic fallback")
                return self._generate_fallback_data(start_date, end_date)
            
            # Validate and clean data
            data = self._validate_and_clean_data(data)
            
            # Cache the data
            if self.enable_caching:
                self.cached_data = data
                self.last_data_update = datetime.now()
            
            print(f"✅ Loaded {len(data):,} bars of real data")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            print(f"   Average volume: {data['volume'].mean():,.0f}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            print(f"   Falling back to synthetic data")
            return self._generate_fallback_data(start_date, end_date)
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean real market data."""
        
        original_len = len(data)
        
        # Remove rows with missing critical data
        data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Remove rows with invalid prices (negative or zero)
        data = data[(data['open'] > 0) & (data['high'] > 0) & 
                   (data['low'] > 0) & (data['close'] > 0)]
        
        # Remove rows with invalid OHLC relationships
        data = data[(data['high'] >= data['low']) & 
                   (data['high'] >= data['open']) & (data['high'] >= data['close']) &
                   (data['low'] <= data['open']) & (data['low'] <= data['close'])]
        
        # Remove extreme outliers (likely data errors)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            q1 = data[col].quantile(0.01)
            q99 = data[col].quantile(0.99)
            data = data[(data[col] >= q1) & (data[col] <= q99)]
        
        # Sort by timestamp
        data = data.sort_index()
        
        cleaned_len = len(data)
        if cleaned_len < original_len:
            print(f"🧹 Cleaned data: {original_len} → {cleaned_len} bars "
                  f"({original_len - cleaned_len} removed)")
        
        return data
    
    def _generate_fallback_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic data as fallback when real data unavailable."""
        
        print(f"🔄 Generating synthetic fallback data for {self.symbol}")
        
        # Calculate number of trading minutes
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate minute-level timestamps (trading hours only: 9:30-16:00 ET)
        dates = []
        current_date = start_dt
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Monday-Friday only
                for hour in range(9, 16):
                    for minute in range(60):
                        if hour == 9 and minute < 30:  # Market opens at 9:30
                            continue
                        timestamp = current_date.replace(hour=hour, minute=minute)
                        dates.append(timestamp)
            current_date += timedelta(days=1)
        
        if not dates:
            # Fallback to simple range
            dates = pd.date_range(start_date, end_date, freq='1min')
        
        dates = dates[:min(len(dates), 10000)]  # Limit to reasonable size
        
        # Generate realistic price movement
        np.random.seed(hash(self.symbol) % 2**31)  # Deterministic per symbol
        
        # Base price varies by symbol
        symbol_seeds = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'TSLA': 200}
        base_price = symbol_seeds.get(self.symbol, 100)
        
        # Generate returns with some trending behavior
        returns = np.random.normal(0, 0.001, len(dates))
        
        # Add some trending periods
        trend_length = len(dates) // 4
        if trend_length > 0:
            trend_start = len(dates) // 3
            returns[trend_start:trend_start + trend_length] += 0.0002
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        opens = prices
        closes = prices * (1 + np.random.normal(0, 0.0005, len(dates)))
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.0003, len(dates))))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.0003, len(dates))))
        volumes = np.random.lognormal(10, 1, len(dates))  # Realistic volume distribution
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes.astype(int)
        }, index=pd.DatetimeIndex(dates, name='timestamp'))
        
        print(f"📈 Generated {len(data)} bars of synthetic data")
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features efficiently with real market data considerations."""
        
        # Always include basic features
        features = pd.DataFrame(index=data.index)
        features['returns'] = data['close'].pct_change()
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20, min_periods=1).mean()
        features['price_momentum'] = data['close'].pct_change(5)
        features['volatility'] = features['returns'].rolling(10, min_periods=1).std()
        
        # VWAP calculation (important for real trading)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        features['vwap'] = (typical_price * data['volume']).rolling(20, min_periods=1).sum() / \
                          data['volume'].rolling(20, min_periods=1).sum()
        features['vwap_deviation'] = (data['close'] - features['vwap']) / features['vwap']
        
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
            features['sma_20'] = data['close'].rolling(20, min_periods=1).mean()
            features['sma_50'] = data['close'].rolling(50, min_periods=1).mean()
            features['rsi'] = self._compute_rsi(data['close'], 14)
            features['bollinger_upper'], features['bollinger_lower'] = self._compute_bollinger_bands(data['close'])
        
        # Clean features with forward fill then zero fill
        features = features.fillna(method='ffill').fillna(0)
        
        # Remove any infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def _select_indicators_fast(self, data: pd.DataFrame):
        """Fast indicator selection with reduced complexity for real data."""
        try:
            if not hasattr(self, 'indicator_selector'):
                return
            
            # Use smaller sample for speed
            sample_size = min(200, len(data))
            if sample_size < 50:
                return
            
            sample_data = data.tail(sample_size)
            
            # Get returns for transformer
            returns = sample_data['close'].pct_change().fillna(0)
            
            # Generate basic features for selection
            basic_features = pd.DataFrame(index=sample_data.index)
            basic_features['returns'] = returns
            basic_features['volume_change'] = sample_data['volume'].pct_change().fillna(0)
            basic_features['price_change'] = sample_data['close'].pct_change(5).fillna(0)
            
            # Get indicator selection
            selected = self.indicator_selector.select_indicators(
                basic_features.fillna(0), 
                top_k=self.top_k_indicators
            )
            
            self.current_indicators = selected
            self.last_market_regime = selected.get('regime', 'Unknown')
            
            print(f"🔄 Selected {len(selected)} indicators for regime: {self.last_market_regime}")
            
        except Exception as e:
            print(f"⚠️ Indicator selection failed: {e}")
            # Use default indicators as fallback
            self.current_indicators = {
                'sma_crossover': True,
                'rsi_signal': True,
                'volume_surge': True
            }
    
    def _add_selected_features(self, data: pd.DataFrame, features: pd.DataFrame):
        """Add transformer-selected features to the feature set."""
        
        for indicator_name, selected in self.current_indicators.items():
            if not selected:
                continue
                
            try:
                if 'sma' in indicator_name:
                    period = 20 if '20' in indicator_name else 50
                    features[f'sma_{period}'] = data['close'].rolling(period, min_periods=1).mean()
                elif 'rsi' in indicator_name:
                    features['rsi'] = self._compute_rsi(data['close'], 14)
                elif 'volume' in indicator_name:
                    features['volume_surge'] = (data['volume'] > 
                                               data['volume'].rolling(20, min_periods=1).mean() * 1.5).astype(int)
                elif 'bollinger' in indicator_name:
                    upper, lower = self._compute_bollinger_bands(data['close'])
                    features['bollinger_upper'] = upper
                    features['bollinger_lower'] = lower
                elif 'macd' in indicator_name:
                    features['macd'] = self._compute_macd(data['close'])
                    
            except Exception as e:
                print(f"⚠️ Failed to compute {indicator_name}: {e}")
                continue
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def _compute_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Compute Bollinger Bands."""
        sma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Compute MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on features."""
        
        signals = pd.Series(0, index=features.index)
        
        if len(features) < 20:
            return signals
        
        # Momentum-based signals (enhanced for real market conditions)
        momentum_threshold = 0.002  # 0.2% threshold for real markets
        volume_threshold = 1.2  # Volume surge threshold
        
        # Buy signals (long)
        buy_conditions = (
            (features['price_momentum'] > momentum_threshold) &
            (features['volume_ratio'] > volume_threshold) &
            (features['vwap_deviation'] > -0.001)  # Not too far below VWAP
        )
        
        # Sell signals (short) 
        sell_conditions = (
            (features['price_momentum'] < -momentum_threshold) &
            (features['volume_ratio'] > volume_threshold) &
            (features['vwap_deviation'] < 0.001)  # Not too far above VWAP
        )
        
        signals[buy_conditions] = 1
        signals[sell_conditions] = -1
        
        # Add regime-based adjustments
        if self.last_market_regime in ['Bull', 'Trending_Up']:
            # More aggressive buying in bull markets
            signals[(features['price_momentum'] > momentum_threshold * 0.7) & 
                   (features['volume_ratio'] > 1.1)] = 1
        elif self.last_market_regime in ['Bear', 'Trending_Down']:
            # More conservative approach in bear markets
            signals[(features['price_momentum'] > momentum_threshold * 1.3) & 
                   (features['volume_ratio'] > 1.5)] = 1
        
        # RSI-based signal filtering (if available)
        if 'rsi' in features.columns:
            overbought = features['rsi'] > 70
            oversold = features['rsi'] < 30
            signals[(signals == 1) & overbought] = 0  # Don't buy when overbought
            signals[(signals == -1) & oversold] = 0   # Don't sell when oversold
        
        return signals
    
    def backtest(self, 
                data: Optional[pd.DataFrame] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive backtest with real market data.
        
        Args:
            data: Pre-loaded data (optional)
            start_date: Start date for backtest
            end_date: End date for backtest
            verbose: Print detailed progress
            
        Returns:
            Dictionary with backtest results and metrics
        """
        
        if verbose:
            print(f"🚀 Running real data backtest for {self.symbol}...")
        
        # Load data if not provided
        if data is None:
            data = self.load_real_data(start_date, end_date)
        
        if data.empty:
            return {'error': 'No data available for backtesting'}
        
        if verbose:
            print(f"📊 Backtesting on {len(data)} bars ({data.index[0]} to {data.index[-1]})")
        
        # Prepare features
        features = self.prepare_features(data)
        if verbose:
            print(f"🔧 Prepared {features.shape[1]} features")
        
        # Generate signals
        signals = self.generate_signals(features)
        active_signals = (signals != 0).sum()
        if verbose:
            print(f"🎯 Generated {active_signals} active signals")
        
        # Enhanced backtest simulation with real market considerations
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        capital = self.capital
        
        # Risk management parameters
        max_position_size = 0.1  # 10% of capital per trade
        stop_loss_pct = 0.02     # 2% stop loss
        take_profit_pct = 0.04   # 4% take profit
        max_hold_time = 240      # 4 hours max hold time (minutes)
        
        for i in range(50, len(data)):  # Skip first 50 for warmup
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Check for position exit conditions
            if position != 0:
                held_time = (current_time - entry_time).total_seconds() / 60  # minutes
                price_change = (current_price - entry_price) / entry_price
                
                # Stop loss or take profit
                if position > 0:  # Long position
                    if price_change <= -stop_loss_pct or price_change >= take_profit_pct:
                        signal = -1  # Force exit
                else:  # Short position
                    if price_change >= stop_loss_pct or price_change <= -take_profit_pct:
                        signal = 1  # Force exit
                
                # Max hold time exit
                if held_time >= max_hold_time:
                    signal = -position  # Force exit
            
            # Close position
            if position != 0 and signal != position:
                pnl = (current_price - entry_price) * abs(position)
                if position < 0:  # Short position
                    pnl = -pnl
                
                # Apply realistic trading costs
                trading_cost = abs(position) * current_price * 0.001  # 0.1% trading cost
                pnl -= trading_cost
                
                trade_duration = (current_time - entry_time).total_seconds() / 60
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position_size': abs(position),
                    'direction': 'Long' if position > 0 else 'Short',
                    'pnl': pnl,
                    'duration_minutes': trade_duration,
                    'regime': self.last_market_regime,
                    'return_pct': pnl / (entry_price * abs(position))
                })
                
                capital += pnl
                position = 0
                entry_price = 0
                entry_time = None
            
            # Open new position
            if signal != 0 and position == 0:
                position_value = capital * max_position_size
                shares = int(position_value / current_price)
                
                if shares > 0:
                    position = shares * signal
                    entry_price = current_price
                    entry_time = current_time
        
        # Calculate comprehensive metrics
        if trades:
            trade_returns = [t['return_pct'] for t in trades]
            total_pnl = sum(t['pnl'] for t in trades)
            wins = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(wins) / len(trades)
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            avg_duration = np.mean([t['duration_minutes'] for t in trades])
            max_drawdown = self._calculate_max_drawdown([t['pnl'] for t in trades])
            
            # Sharpe ratio calculation
            if len(trade_returns) > 1:
                sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
                
        else:
            total_pnl = win_rate = avg_win = avg_loss = profit_factor = 0
            avg_duration = max_drawdown = sharpe_ratio = 0
        
        # Compile results
        results = {
            'symbol': self.symbol,
            'data_source': 'Real Database' if self.data_loader else 'Synthetic',
            'backtest_period': f"{data.index[0]} to {data.index[-1]}",
            'total_bars': len(data),
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': (capital - self.capital) / self.capital,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_duration_minutes': avg_duration,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': capital,
            'market_regime': self.last_market_regime,
            'features_used': features.shape[1],
            'signals_generated': active_signals,
            'indicators_selected': len(self.current_indicators) if self.adaptive_indicators else 'N/A',
            'trades': trades  # Include individual trade details
        }
        
        if verbose:
            print(f"\n📈 Backtest Results for {self.symbol}:")
            print(f"   Total Trades: {len(trades)}")
            print(f"   Win Rate: {win_rate:.1%}")
            print(f"   Total PnL: ${total_pnl:.2f}")
            print(f"   Total Return: {results['total_return_pct']:.1%}")
            print(f"   Profit Factor: {profit_factor:.2f}")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"   Average Duration: {avg_duration:.1f} minutes")
            print(f"   Max Drawdown: ${max_drawdown:.2f}")
            print(f"   Market Regime: {self.last_market_regime}")
        
        return results
    
    def _calculate_max_drawdown(self, pnl_series: List[float]) -> float:
        """Calculate maximum drawdown from PnL series."""
        if not pnl_series:
            return 0
        
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return np.max(drawdown)


def test_s3_real_data():
    """Test the S3 strategy with real data from multiple symbols."""
    print("🧪 Testing S3 Working Strategy with Real Data")
    print("=" * 50)
    
    # Test with multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        print(f"\n🔍 Testing {symbol}...")
        print("-" * 30)
        
        try:
            # Initialize strategy
            strategy = S3WorkingStrategyRealData(
                symbol=symbol,
                capital=100000,
                adaptive_indicators=True,
                data_days_lookback=5  # 5 days for quick test
            )
            
            # Run backtest
            results = strategy.backtest(verbose=True)
            
            # Save results
            output_file = f's3_real_data_results_{symbol}.json'
            with open(output_file, 'w') as f:
                # Remove trades from saved results for cleaner file
                save_results = results.copy()
                save_results.pop('trades', None)
                json.dump(save_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            continue
    
    print(f"\n✅ Testing complete!")


if __name__ == "__main__":
    test_s3_real_data()