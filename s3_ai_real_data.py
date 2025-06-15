#!/usr/bin/env python3
"""
S3 AI Real Data Strategy
========================
AI-enhanced S3 trading strategy using REAL market data from PostgreSQL database.
This version replaces synthetic data with actual historical minute-level OHLCV data.

Features:
- Real PostgreSQL data integration
- Transformer-based AI indicator selection  
- Market regime detection
- Comprehensive backtesting with real market conditions
- Multi-symbol analysis across sectors
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import warnings
import time
warnings.filterwarnings('ignore')

# Import AI components and real data loader
from s3_ai_wrapper import S3AIWrapper
from day_trading_o3 import DataLoader


class S3AIRealDataStrategy:
    """
    S3 AI strategy using real market data from PostgreSQL database.
    
    This strategy combines:
    - Real historical minute-level OHLCV data
    - AI-powered indicator selection via transformer models
    - Market regime detection and adaptation
    - Comprehensive risk management
    """
    
    def __init__(self, 
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,
                 top_k_indicators: int = 8,
                 selection_interval: int = 30,
                 risk_per_trade: float = 0.02):
        
        self.symbol = symbol
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.risk_per_trade = risk_per_trade
        
        # Initialize real data loader
        self.data_loader = DataLoader()
        
        # AI components
        self.indicator_selector = None
        self.current_indicators = {}
        self.bars_since_selection = 0
        self.last_market_regime = "Unknown"
        self.regime_confidence = 0.0
        self.ai_selection_history = []
        
        if adaptive_indicators:
            self._init_ai_components()
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        
        print(f"🤖 S3 AI Real Data Strategy initialized for {symbol}")
        print(f"   AI Status: {'✅ Active' if self.indicator_selector else '❌ Disabled'}")
        print(f"   Data Source: 📊 PostgreSQL Database")
        print(f"   Top-K Indicators: {top_k_indicators}")
        print(f"   Selection Interval: {selection_interval} bars")
    
    def _init_ai_components(self):
        """Initialize AI components safely."""
        try:
            print(f"🧠 Initializing AI components...")
            
            self.indicator_selector = S3AIWrapper(
                confidence_threshold=0.1,  # Much lower than original 0.3
                min_indicators=3,         # Ensure we always get some indicators
                max_indicators=8          # Match our top_k setting
            )
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
            self.regime_confidence = 0.0
            
            print(f"✅ AI Indicator Selector ready")
            
        except Exception as e:
            print(f"❌ AI initialization failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            
            # Don't disable - fix the actual issue
            raise e
    
    def get_real_market_data(self, 
                           start_date: str,
                           end_date: str) -> pd.DataFrame:
        """Get real market data from PostgreSQL database."""
        
        print(f"📊 Fetching real market data for {self.symbol}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Source: PostgreSQL minute_prices table")
        
        try:
            # Use a timeout and simplified query approach
            print(f"   Connecting to database...")
            
            # Try with a shorter query first
            data = self.data_loader.get_price_data(
                symbol=self.symbol,
                start=start_date,
                end=end_date,
                timeframe="1min"
            )
            
            if data.empty:
                raise ValueError(f"No data found for {self.symbol} in the specified date range")
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert timezone to market hours (ET) and filter trading hours
            data = data.tz_convert('US/Eastern')
            
            # Filter to trading hours (9:30 AM - 4:00 PM ET)
            data = data.between_time('09:30', '16:00')
            
            # Remove any rows with missing data
            data = data.dropna()
            
            # Calculate basic statistics
            total_bars = len(data)
            date_range = f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
            price_range = f"${data['close'].min():.2f} - ${data['close'].max():.2f}"
            avg_volume = data['volume'].mean()
            total_return = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
            
            print(f"✅ Real market data loaded successfully:")
            print(f"   Total bars: {total_bars:,}")
            print(f"   Date range: {date_range}")
            print(f"   Price range: {price_range}")
            print(f"   Total return: {total_return:.2f}%")
            print(f"   Average volume: {avg_volume:,.0f}")
            
            return data
            
        except Exception as e:
            print(f"❌ Failed to load real market data: {e}")
            print(f"   Error type: {type(e).__name__}")
            raise e
    
    def prepare_ai_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for AI analysis."""
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['returns_2'] = data['close'].pct_change(2)
        features['returns_5'] = data['close'].pct_change(5)
        features['returns_10'] = data['close'].pct_change(10)
        
        # Volume features
        features['volume'] = data['volume']
        features['volume_ma'] = data['volume'].rolling(20, min_periods=1).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma']
        features['volume_change'] = data['volume'].pct_change()
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window, min_periods=1).std()
        features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # VWAP and positioning
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        for window in [10, 20, 50]:
            vwap_num = (typical_price * data['volume']).rolling(window, min_periods=1).sum()
            vwap_den = data['volume'].rolling(window, min_periods=1).sum()
            features[f'vwap_{window}'] = vwap_num / vwap_den
            features[f'vwap_dev_{window}'] = (data['close'] - features[f'vwap_{window}']) / features[f'vwap_{window}']
        
        # AI indicator selection
        if self.adaptive_indicators and self.indicator_selector is not None:
            if (self.bars_since_selection >= self.selection_interval or 
                len(self.current_indicators) == 0):
                self._ai_select_indicators(data, features)
                self.bars_since_selection = 0
            else:
                self.bars_since_selection += 1
            
            self._add_ai_selected_features(data, features)
        else:
            self._add_standard_features(data, features)
        
        # Time-based features
        features['hour'] = data.index.hour
        features['minute'] = data.index.minute
        features['day_of_week'] = data.index.dayofweek
        
        # Market session features
        features['is_opening'] = (data.index.hour == 9) & (data.index.minute < 45)
        features['is_lunch'] = (data.index.hour >= 12) & (data.index.hour < 14)
        features['is_closing'] = (data.index.hour >= 15) & (data.index.minute >= 30)
        
        # Fill any remaining NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _ai_select_indicators(self, data: pd.DataFrame, features: pd.DataFrame):
        """Use AI to select optimal indicators."""
        
        try:
            print(f"🧠 AI selecting optimal indicators...")
            
            sample_size = min(300, len(data))
            if sample_size < 50:
                print(f"⚠️ Insufficient data for AI selection")
                return
            
            sample_data = data.tail(sample_size).copy()
            
            # Ensure data has required columns for AI
            if not all(col in sample_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                print(f"⚠️ Missing required OHLCV columns for AI analysis")
                return
            
            # Get AI selection using raw OHLCV data
            selection_result = self.indicator_selector.select_indicators(
                sample_data, 
                top_k=self.top_k_indicators
            )
            
            # Debug: print what we actually get
            print(f"🔍 AI selection result type: {type(selection_result)}")
            print(f"🔍 AI selection keys: {list(selection_result.keys()) if isinstance(selection_result, dict) else 'Not a dict'}")
            
            # Extract regime information from the AI response
            regime_info = selection_result.get('regime_detection', {})
            print(f"🔍 Regime info type: {type(regime_info)}")
            print(f"🔍 Regime info value: {regime_info}")
            
            # Handle both string and dict cases
            if isinstance(regime_info, str):
                self.last_market_regime = regime_info
                self.regime_confidence = 0.0
            elif isinstance(regime_info, dict):
                self.last_market_regime = regime_info.get('primary_regime', 'Unknown')
                self.regime_confidence = regime_info.get('confidence', 0.0)
            else:
                self.last_market_regime = "Unknown"
                self.regime_confidence = 0.0
            
            # Convert AI selection to strategy format
            selected_indicators = selection_result.get('selected_indicators', {})
            print(f"🔍 Selected indicators count: {len(selected_indicators)}")
            print(f"🔍 Selected indicators: {list(selected_indicators.keys())}")
            
            self.current_indicators = {}
            
            # Map AI indicators to strategy indicators
            for ai_indicator, info in selected_indicators.items():
                # Map specific AI indicators to strategy indicators
                if 'rsi' in ai_indicator.lower():
                    self.current_indicators['rsi_momentum'] = True
                elif 'sma' in ai_indicator.lower() or 'ema' in ai_indicator.lower():
                    self.current_indicators['sma_trend'] = True
                elif 'macd' in ai_indicator.lower():
                    self.current_indicators['macd_momentum'] = True
                elif 'bb' in ai_indicator.lower() or 'bollinger' in ai_indicator.lower():
                    self.current_indicators['bollinger_bands'] = True
                elif 'volume' in ai_indicator.lower():
                    self.current_indicators['volume_breakout'] = True
                elif 'volatility' in ai_indicator.lower() or 'atr' in ai_indicator.lower():
                    self.current_indicators['volatility_regime'] = True
                elif 'vwap' in ai_indicator.lower():
                    self.current_indicators['vwap_deviation'] = True
            
            # Ensure at least some indicators are selected
            if len(self.current_indicators) == 0:
                self.current_indicators = {
                    'sma_trend': True,
                    'rsi_momentum': True,
                    'volume_breakout': True
                }
            
            # Track selection history
            self.ai_selection_history.append({
                'timestamp': datetime.now(),
                'regime': self.last_market_regime,
                'confidence': self.regime_confidence,
                'indicators_selected': len([k for k, v in selection_result.items() 
                                          if isinstance(v, bool) and v]),
                'sample_size': sample_size
            })
            
            print(f"✅ AI Selection Complete:")
            print(f"   Regime: {self.last_market_regime}")
            print(f"   Confidence: {self.regime_confidence:.2f}")
            indicators_count = len([k for k, v in selection_result.items() if isinstance(v, bool) and v])
            print(f"   Indicators Selected: {indicators_count}")
            
        except Exception as e:
            print(f"❌ AI selection failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            
            # Don't fallback - fix the actual issue
            raise e
    
    def _add_ai_selected_features(self, data: pd.DataFrame, features: pd.DataFrame):
        """Add AI-selected indicators."""
        
        for indicator_name, selected in self.current_indicators.items():
            if not isinstance(selected, bool) or not selected:
                continue
                
            try:
                if 'sma_trend' in indicator_name:
                    for period in [10, 20, 50]:
                        features[f'sma_{period}'] = data['close'].rolling(period, min_periods=1).mean()
                        features[f'sma_{period}_signal'] = (data['close'] > features[f'sma_{period}']).astype(int)
                        
                elif 'rsi_momentum' in indicator_name:
                    features['rsi'] = self._compute_rsi(data['close'])
                    features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
                    features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
                    
                elif 'volume_breakout' in indicator_name:
                    vol_ma = data['volume'].rolling(20, min_periods=1).mean()
                    features['volume_breakout'] = (data['volume'] > vol_ma * 1.5).astype(int)
                    
                elif 'volatility_regime' in indicator_name:
                    features['atr'] = self._compute_atr(data)
                    atr_ma = features['atr'].rolling(20, min_periods=1).mean()
                    features['high_volatility'] = (features['atr'] > atr_ma * 1.2).astype(int)
                    
                elif 'bollinger_bands' in indicator_name:
                    bb_upper, bb_lower = self._compute_bollinger_bands(data['close'])
                    features['bb_upper'] = bb_upper
                    features['bb_lower'] = bb_lower
                    features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
                    
                elif 'macd' in indicator_name.lower():
                    features['macd'] = self._compute_macd(data['close'])
                    features['macd_signal'] = features['macd'].ewm(span=9).mean()
                    features['macd_histogram'] = features['macd'] - features['macd_signal']
                    
            except Exception as e:
                print(f"⚠️ Failed to compute AI indicator {indicator_name}: {e}")
    
    def _add_standard_features(self, data: pd.DataFrame, features: pd.DataFrame):
        """Add standard technical indicators when AI is disabled."""
        
        # Moving averages
        for period in [10, 20, 50]:
            features[f'sma_{period}'] = data['close'].rolling(period, min_periods=1).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        
        # Technical indicators
        features['rsi'] = self._compute_rsi(data['close'])
        features['macd'] = self._compute_macd(data['close'])
        
        # Bollinger Bands
        bb_upper, bb_lower = self._compute_bollinger_bands(data['close'])
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
    
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
        return sma + (std * std_dev), sma - (std * std_dev)
    
    def _compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26):
        """Compute MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _compute_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period, min_periods=1).mean()
    
    def generate_ai_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate trading signals with AI enhancement."""
        
        signals = pd.Series(0, index=features.index)
        
        # AI-enhanced signal generation based on selected indicators
        for i in range(len(features)):
            if i < 20:  # Skip first 20 bars for indicator stability
                continue
                
            current_features = features.iloc[i]
            
            # Mean reversion signals (enhanced by AI regime detection)
            if self.last_market_regime in ['High Volatility', 'Mean Reversion']:
                # VWAP mean reversion
                if 'vwap_dev_20' in current_features:
                    vwap_dev = current_features['vwap_dev_20']
                    if vwap_dev < -0.002:  # Price below VWAP
                        signals.iloc[i] = 1  # Buy signal
                    elif vwap_dev > 0.002:  # Price above VWAP
                        signals.iloc[i] = -1  # Sell signal
                
                # RSI mean reversion (if selected by AI)
                if 'rsi_oversold' in current_features and current_features['rsi_oversold']:
                    signals.iloc[i] = max(signals.iloc[i], 1)
                elif 'rsi_overbought' in current_features and current_features['rsi_overbought']:
                    signals.iloc[i] = min(signals.iloc[i], -1)
            
            # Momentum signals (enhanced by AI regime detection)
            elif self.last_market_regime in ['Trending', 'Breakout']:
                # Volume breakout (if selected by AI)
                if 'volume_breakout' in current_features and current_features['volume_breakout']:
                    # Direction based on price momentum
                    if 'returns_5' in current_features and current_features['returns_5'] > 0.001:
                        signals.iloc[i] = 1
                    elif 'returns_5' in current_features and current_features['returns_5'] < -0.001:
                        signals.iloc[i] = -1
                
                # Bollinger band breakout (if selected by AI)
                if 'bb_position' in current_features:
                    bb_pos = current_features['bb_position']
                    if bb_pos > 1.0:  # Above upper band
                        signals.iloc[i] = 1
                    elif bb_pos < 0.0:  # Below lower band
                        signals.iloc[i] = -1
            
            # Default mixed signals for unknown regimes
            else:
                # Conservative signals using multiple confirmations
                signal_count = 0
                
                if 'sma_20_signal' in current_features and current_features['sma_20_signal']:
                    signal_count += 1
                if 'macd_histogram' in current_features and current_features['macd_histogram'] > 0:
                    signal_count += 1
                if 'volume_ratio' in current_features and current_features['volume_ratio'] > 1.2:
                    signal_count += 1
                
                if signal_count >= 2:
                    signals.iloc[i] = 1
                elif signal_count <= -2:
                    signals.iloc[i] = -1
        
        return signals
    
    def backtest_real_data_strategy(self, 
                                  start_date: str,
                                  end_date: str,
                                  verbose: bool = True) -> Dict[str, Any]:
        """Run comprehensive backtest on real market data."""
        
        # Get real market data
        data = self.get_real_market_data(start_date, end_date)
        
        if len(data) < 100:
            raise ValueError(f"Insufficient data: only {len(data)} bars available")
        
        # Prepare AI features
        features = self.prepare_ai_features(data)
        
        # Generate AI-enhanced signals
        signals = self.generate_ai_signals(features)
        
        # Execute backtest
        capital = self.capital
        position = 0
        trades = []
        equity_curve = []
        
        # Track AI metrics
        ai_selections = len(self.ai_selection_history)
        regime_changes = len(set([h['regime'] for h in self.ai_selection_history]))
        avg_confidence = np.mean([h['confidence'] for h in self.ai_selection_history]) if self.ai_selection_history else 0
        
        active_signals = (signals != 0).sum()
        
        commission = 1.0  # $1 per trade
        slippage = 0.0005  # 0.05% slippage
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            current_signal = signals.iloc[i]
            
            # Exit existing position
            if position != 0:
                exit_price = current_price * (1 - slippage if position > 0 else 1 + slippage)
                pnl = position * (exit_price - entry_price) - commission
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': data.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'pnl': pnl,
                    'regime': self.last_market_regime
                })
                
                capital += pnl
                position = 0
            
            # Enter new position
            if current_signal != 0:
                position_size = min(1000, capital * self.risk_per_trade / (current_price * 0.02))
                position = position_size if current_signal > 0 else -position_size
                entry_price = current_price * (1 + slippage if position > 0 else 1 - slippage)
                entry_time = data.index[i]
                capital -= commission
            
            equity_curve.append(capital)
        
        # Calculate performance metrics
        if trades:
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades
            total_pnl = sum([t['pnl'] for t in trades])
            total_return_pct = total_pnl / self.capital
            
            # Calculate profit factor
            gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
            gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate Sharpe ratio (simplified)
            returns = pd.Series([t['pnl'] for t in trades])
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Calculate max drawdown
            equity_series = pd.Series(equity_curve)
            peak = equity_series.cummax()
            drawdown = (equity_series - peak) / peak
            max_drawdown_pct = drawdown.min()
        else:
            total_trades = 0
            win_rate = 0
            total_pnl = 0
            total_return_pct = 0
            profit_factor = 0
            sharpe_ratio = 0
            max_drawdown_pct = 0
        
        # Compile results
        results = {
            'symbol': self.symbol,
            'ai_enabled': self.adaptive_indicators,
            'data_source': 'PostgreSQL Database',
            'backtest_period': f"{data.index[0]} to {data.index[-1]}",
            'total_bars': len(data),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'final_capital': capital,
            'ai_regime': self.last_market_regime,
            'ai_confidence': self.regime_confidence,
            'features_count': features.shape[1],
            'signals_generated': active_signals,
            'ai_selections': ai_selections,
            'regime_changes': regime_changes,
            'avg_ai_confidence': avg_confidence,
            'trades_sample': trades[:5] if trades else []
        }
        
        if verbose:
            self._print_real_data_results(results)
        
        return results
    
    def _print_real_data_results(self, results: Dict[str, Any]):
        """Print real data backtest results."""
        print(f"\n🤖 AI Real Data Results for {results['symbol']}:")
        print("=" * 50)
        print(f"Data Source: {results['data_source']}")
        print(f"Period: {results['backtest_period']}")
        print(f"AI Status: {'✅ Active' if results['ai_enabled'] else '❌ Disabled'}")
        print()
        print("📊 Trading Performance:")
        print(f"  Total Bars: {results['total_bars']:,}")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Total Return: {results['total_return_pct']:.1%}")
        print(f"  Total PnL: ${results['total_pnl']:.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.1%}")
        print()
        print("🧠 AI Intelligence:")
        print(f"  Final Regime: {results['ai_regime']}")
        print(f"  Current Confidence: {results['ai_confidence']:.2f}")
        print(f"  Features Used: {results['features_count']}")
        print(f"  AI Selections: {results['ai_selections']}")
        print(f"  Regime Changes: {results['regime_changes']}")
        print(f"  Avg Confidence: {results['avg_ai_confidence']:.2f}")


def test_real_data_strategy():
    """Test the AI strategy with real market data."""
    print("🤖 Testing S3 AI Strategy with REAL Market Data")
    print("=" * 55)
    
    # Test with just one symbol first to check database connection
    symbols = ['AAPL']
    
    # Define test period (much shorter to avoid database timeout)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    
    results_summary = []
    
    for symbol in symbols:
        print(f"\n🔍 Testing real data strategy with {symbol}...")
        print("-" * 40)
        
        try:
            # Create AI strategy with real data
            ai_strategy = S3AIRealDataStrategy(
                symbol=symbol,
                capital=100000,
                adaptive_indicators=True,
                top_k_indicators=8,
                selection_interval=30,
                risk_per_trade=0.02
            )
            
            # Run backtest with real data
            results = ai_strategy.backtest_real_data_strategy(
                start_date=start_date,
                end_date=end_date,
                verbose=True
            )
            
            results_summary.append({
                'symbol': symbol,
                'ai_enabled': results.get('ai_enabled', False),
                'data_source': results.get('data_source', 'Unknown'),
                'trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0),
                'return_pct': results.get('total_return_pct', 0),
                'sharpe': results.get('sharpe_ratio', 0),
                'regime': results.get('ai_regime', 'Unknown'),
                'bars': results.get('total_bars', 0)
            })
            
            # Save results
            output_file = f's3_real_data_results_{symbol}.json'
            with open(output_file, 'w') as f:
                save_results = results.copy()
                save_results.pop('trades_sample', None)
                json.dump(save_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            results_summary.append({
                'symbol': symbol,
                'ai_enabled': False,
                'data_source': 'Failed',
                'trades': 0,
                'win_rate': 0,
                'return_pct': 0,
                'sharpe': 0,
                'regime': 'Error',
                'bars': 0
            })
    
    # Print summary
    print(f"\n🤖 Real Data AI Strategy Summary:")
    print("=" * 80)
    print(f"{'Symbol':<8} {'Source':<12} {'Bars':<8} {'Trades':<8} {'Win%':<8} {'Return%':<8} {'Sharpe':<8} {'Regime':<12}")
    print("-" * 80)
    
    for result in results_summary:
        print(f"{result['symbol']:<8} {result['data_source']:<12} {result['bars']:<8} "
              f"{result['trades']:<8} {result['win_rate']:<8.1%} {result['return_pct']:<8.1%} "
              f"{result['sharpe']:<8.2f} {result['regime']:<12}")
    
    # Calculate aggregate statistics
    successful_results = [r for r in results_summary if r['data_source'] != 'Failed']
    if successful_results:
        total_bars = sum(r['bars'] for r in successful_results)
        total_trades = sum(r['trades'] for r in successful_results)
        avg_return = sum(r['return_pct'] for r in successful_results) / len(successful_results)
        avg_sharpe = sum(r['sharpe'] for r in successful_results) / len(successful_results)
        
        print(f"\n📊 Aggregate Real Data Performance:")
        print(f"  Symbols Tested: {len(successful_results)}/{len(symbols)}")
        print(f"  Total Bars: {total_bars:,}")
        print(f"  Total Trades: {total_trades}")
        print(f"  Average Return: {avg_return:.1%}")
        print(f"  Average Sharpe: {avg_sharpe:.2f}")
    
    print(f"\n✅ Real market data testing complete!")


if __name__ == "__main__":
    test_real_data_strategy()