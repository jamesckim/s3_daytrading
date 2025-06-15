#!/usr/bin/env python3
"""
S3 Database Final - Production-ready strategy with real PostgreSQL data
=====================================================================
Uses real market data from the PostgreSQL database with proper .env loading
and connection management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path
import psycopg2
import time
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Import components
try:
    from indicator_transformer import AIIndicatorSelector
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI indicator selector not available")


class S3DatabaseFinal:
    """
    Production S3 strategy using real PostgreSQL data.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,
                 top_k_indicators: int = 10,
                 selection_interval: int = 100,
                 data_days_lookback: int = 14):
        """Initialize S3 database strategy."""
        
        self.symbol = symbol
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators and AI_AVAILABLE
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.data_days_lookback = data_days_lookback
        self.bars_since_selection = 0
        
        # Database connection
        self.db_conn = None
        self.connection_string = os.getenv('POSTGRES_URI')
        
        if not self.connection_string:
            raise ValueError("POSTGRES_URI not found in environment variables")
        
        # Test database connection
        self._test_connection()
        
        # Initialize AI if enabled
        if self.adaptive_indicators:
            try:
                self.indicator_selector = AIIndicatorSelector(device='cpu')
                self.current_indicators = {}
                self.last_market_regime = "Unknown"
                print(f"✅ AI indicator selector initialized")
            except Exception as e:
                print(f"⚠️ AI initialization failed: {e}")
                self.adaptive_indicators = False
        
        if not self.adaptive_indicators:
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
        
        # Caching
        self.cached_data = None
        self.last_data_update = None
        
        # Logging
        self.logger = logging.getLogger(f'S3Final_{symbol}')
        
        print(f"🚀 S3 Database Strategy initialized for {symbol}")
        print(f"   Adaptive indicators: {self.adaptive_indicators}")
        print(f"   Data lookback: {data_days_lookback} days")
    
    def _test_connection(self):
        """Test database connection."""
        try:
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            print("✅ Database connection successful")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
    
    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(self.connection_string)
    
    def load_historical_data(self, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           force_refresh: bool = False) -> pd.DataFrame:
        """Load historical data from PostgreSQL."""
        
        # Check cache
        if (not force_refresh and 
            self.cached_data is not None and 
            self.last_data_update is not None and
            (datetime.now() - self.last_data_update).seconds < 3600):
            print(f"📦 Using cached data ({len(self.cached_data)} bars)")
            return self.cached_data
        
        # Calculate dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=self.data_days_lookback)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        print(f"📊 Loading {self.symbol} data from {start_date} to {end_date}...")
        
        try:
            conn = self._get_connection()
            
            query = """
                SELECT 
                    mp.datetime AT TIME ZONE 'UTC' AS timestamp,
                    mp.open,
                    mp.high,
                    mp.low,
                    mp.close,
                    mp.volume
                FROM minute_prices mp
                JOIN tickers t ON mp.ticker_id = t.id
                WHERE t.symbol = %s
                    AND DATE(mp.datetime) >= %s
                    AND DATE(mp.datetime) <= %s
                ORDER BY timestamp
            """
            
            data = pd.read_sql(
                query,
                conn,
                params=(self.symbol, start_date, end_date),
                index_col='timestamp',
                parse_dates=['timestamp']
            )
            
            conn.close()
            
            if data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Clean data
            data = self._clean_data(data)
            
            # Cache
            self.cached_data = data
            self.last_data_update = datetime.now()
            
            print(f"✅ Loaded {len(data):,} bars")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Fill NaN values
        data = data.fillna(method='ffill')
        
        # Fix any zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            data.loc[data[col] <= 0, col] = np.nan
        data[price_cols] = data[price_cols].fillna(method='ffill')
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare trading features."""
        
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_range'] = (data['high'] - data['low']) / data['close']
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['dollar_volume'] = data['close'] * data['volume']
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['close'].pct_change(period)
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Technical indicators
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        features['rsi_14'] = self._compute_rsi(data['close'])
        
        # Add AI indicators if enabled
        if self.adaptive_indicators:
            if (self.bars_since_selection >= self.selection_interval or 
                not self.current_indicators):
                self._select_indicators(data)
                self.bars_since_selection = 0
            else:
                self.bars_since_selection += 1
            
            # Add selected indicators
            self._add_ai_indicators(data, features)
        
        # Clean features
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def _select_indicators(self, data: pd.DataFrame):
        """Select indicators using AI."""
        try:
            # Use recent data
            recent_data = data.iloc[-min(500, len(data)):]
            
            selection = self.indicator_selector.select_indicators(
                recent_data,
                top_k=self.top_k_indicators
            )
            
            self.current_indicators = selection['selected_indicators']
            self.last_market_regime = selection['regime_detection']
            
            print(f"🎯 Selected {len(self.current_indicators)} indicators")
            print(f"   Market regime: {self.last_market_regime}")
            
        except Exception as e:
            self.logger.warning(f"Indicator selection failed: {e}")
    
    def _add_ai_indicators(self, data: pd.DataFrame, features: pd.DataFrame):
        """Add AI-selected indicators."""
        
        if not self.current_indicators:
            return
        
        # Get top indicators by weight
        top_indicators = sorted(
            self.current_indicators.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )[:5]
        
        for name, info in top_indicators:
            try:
                if name.startswith('SMA_'):
                    period = int(name.split('_')[1])
                    features[f'ai_sma_{period}'] = data['close'].rolling(period).mean()
                elif name.startswith('EMA_'):
                    period = int(name.split('_')[1])
                    features[f'ai_ema_{period}'] = data['close'].ewm(span=period).mean()
                elif name == 'RSI_14':
                    features['ai_rsi'] = self._compute_rsi(data['close'])
            except Exception as e:
                self.logger.warning(f"Failed to add {name}: {e}")
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        
        signals = pd.Series(0, index=features.index)
        
        # Momentum signals
        momentum = features['momentum_5'] * 0.6 + features['momentum_10'] * 0.3 + features['momentum_20'] * 0.1
        
        # Adjust thresholds by regime
        if "Uptrend" in self.last_market_regime:
            buy_thresh, sell_thresh = 0.0003, -0.0005
        elif "Downtrend" in self.last_market_regime:
            buy_thresh, sell_thresh = 0.0005, -0.0003
        elif "High Volatility" in self.last_market_regime:
            buy_thresh, sell_thresh = 0.001, -0.001
        else:
            buy_thresh, sell_thresh = 0.0005, -0.0005
        
        # Generate base signals
        signals[momentum > buy_thresh] = 1
        signals[momentum < sell_thresh] = -1
        
        # Volume filter
        high_volume = features['volume_ratio'] > 1.2
        signals = signals * high_volume
        
        # RSI filter
        if 'rsi_14' in features.columns:
            signals[(features['rsi_14'] > 70) & (signals == 1)] = 0
            signals[(features['rsi_14'] < 30) & (signals == -1)] = 0
        
        return signals
    
    def backtest(self, 
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """Run backtest."""
        
        if verbose:
            print(f"\n🚀 Running backtest for {self.symbol}")
        
        # Load data
        data = self.load_historical_data(start_date, end_date)
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Generate signals
        signals = self.generate_signals(features)
        active_signals = (signals != 0).sum()
        
        if verbose:
            print(f"📊 Generated {active_signals:,} trading signals")
        
        # Simulate trading
        trades = []
        position = 0
        entry_price = 0
        capital = self.capital
        
        # Skip warmup period
        for i in range(100, len(data)):
            price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Close position
            if position != 0 and signal != np.sign(position):
                pnl = (price - entry_price) * position
                trades.append({
                    'entry': entry_price,
                    'exit': price,
                    'pnl': pnl,
                    'position': position
                })
                capital += pnl
                position = 0
            
            # Open position
            if signal != 0 and position == 0:
                shares = int(capital * 0.1 / price)
                if shares > 0:
                    position = shares * signal
                    entry_price = price
        
        # Calculate results
        if trades:
            total_pnl = sum(t['pnl'] for t in trades)
            wins = [t for t in trades if t['pnl'] > 0]
            win_rate = len(wins) / len(trades)
            
            results = {
                'symbol': self.symbol,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return': (capital - self.capital) / self.capital,
                'final_capital': capital,
                'market_regime': self.last_market_regime,
                'data_bars': len(data),
                'signals_generated': active_signals
            }
        else:
            results = {
                'symbol': self.symbol,
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'final_capital': self.capital,
                'market_regime': self.last_market_regime,
                'data_bars': len(data),
                'signals_generated': active_signals
            }
        
        if verbose:
            print(f"\n📈 Results:")
            print(f"   Trades: {results['total_trades']}")
            print(f"   Win rate: {results['win_rate']:.1%}")
            print(f"   Total PnL: ${results['total_pnl']:.2f}")
            print(f"   Return: {results['total_return']:.1%}")
            print(f"   Regime: {results['market_regime']}")
        
        return results
    
    def get_latest_signals(self, hours: int = 24) -> Dict[str, Any]:
        """Get latest trading signals."""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)
        
        data = self.load_historical_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            return {'error': 'No recent data'}
        
        features = self.prepare_features(data)
        signals = self.generate_signals(features)
        
        latest_signal = signals.iloc[-1]
        latest_price = data['close'].iloc[-1]
        
        return {
            'timestamp': data.index[-1],
            'symbol': self.symbol,
            'price': latest_price,
            'signal': int(latest_signal),
            'signal_text': 'BUY' if latest_signal > 0 else 'SELL' if latest_signal < 0 else 'HOLD',
            'market_regime': self.last_market_regime,
            'recent_signals': signals.iloc[-10:].tolist()
        }


def main():
    """Test the strategy."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 S3 Database Final Strategy")
    print("=" * 60)
    
    try:
        # Test with AAPL
        strategy = S3DatabaseFinal(
            symbol='AAPL',
            capital=100000,
            adaptive_indicators=True,
            data_days_lookback=7
        )
        
        # Run backtest
        results = strategy.backtest(verbose=True)
        
        # Get latest signals
        latest = strategy.get_latest_signals(hours=24)
        
        print(f"\n🎯 Latest Signal:")
        print(f"   Price: ${latest['price']:.2f}")
        print(f"   Signal: {latest['signal_text']}")
        print(f"   Regime: {latest['market_regime']}")
        
        print("\n✅ Strategy test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()