#!/usr/bin/env python3
"""
S3 AI-Enabled Strategy - Real Data with Transformer Intelligence
===============================================================
Enhanced version of S3 strategy with full AI/transformer integration for
dynamic indicator selection and market regime adaptation.

Key AI Features:
- Transformer-based indicator selection
- Dynamic market regime detection
- Adaptive parameter optimization
- Performance-based learning
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

# Import AI components
from indicator_transformer import AIIndicatorSelector
from day_trading_o3 import DataLoader


class S3AIEnabledStrategy:
    """
    S3 strategy with full AI integration using transformer-based indicator selection
    and market regime adaptation for optimal trading performance.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,  # AI ENABLED by default
                 top_k_indicators: int = 12,
                 selection_interval: int = 25,  # More frequent AI updates
                 data_days_lookback: int = 7,
                 ai_learning_rate: float = 0.001,
                 ai_device: str = 'cpu',
                 enable_caching: bool = True,
                 enable_regime_detection: bool = True):
        """
        Initialize AI-enabled S3 strategy.
        
        Args:
            symbol: Trading symbol
            capital: Starting capital
            adaptive_indicators: Enable AI indicator selection
            top_k_indicators: Number of top indicators to select
            selection_interval: Bars between AI re-selection (lower = more adaptive)
            data_days_lookback: Days of historical data
            ai_learning_rate: Learning rate for AI adaptation
            ai_device: Device for AI computation ('cpu' or 'cuda')
            enable_caching: Cache data for performance
            enable_regime_detection: Enable market regime detection
        """
        
        self.symbol = symbol.upper()
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.data_days_lookback = data_days_lookback
        self.ai_learning_rate = ai_learning_rate
        self.ai_device = ai_device
        self.enable_caching = enable_caching
        self.enable_regime_detection = enable_regime_detection
        self.bars_since_selection = 0
        
        # Initialize database connection
        self.data_loader = None
        self._init_database_connection()
        
        # Initialize AI components
        self.indicator_selector = None
        self.current_indicators = {}
        self.last_market_regime = "Unknown"
        self.ai_performance_history = []
        self.regime_confidence = 0.0
        
        if adaptive_indicators:
            self._init_ai_components()
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        self.cached_data = None
        self.last_data_update = None
        self.ai_selection_history = []
        
        # Setup logging
        self.logger = logging.getLogger(f'S3AI_{symbol}')
        self.logger.setLevel(logging.INFO)
        
        print(f"🤖 S3 AI-Enabled Strategy initialized for {symbol}")
        print(f"   AI Components: {'✅ Active' if self.indicator_selector else '❌ Disabled'}")
        print(f"   Database: {'✅ Connected' if self.data_loader else '❌ Disconnected'}")
        print(f"   Regime Detection: {'✅ Enabled' if enable_regime_detection else '❌ Disabled'}")
        print(f"   Top-K Indicators: {top_k_indicators}")
        print(f"   Selection Interval: {selection_interval} bars")
        print(f"   AI Device: {ai_device}")
    
    def _init_database_connection(self):
        """Initialize database connection with timeout handling."""
        try:
            print(f"🗄️ Connecting to database for {self.symbol}...")
            start_time = time.time()
            
            self.data_loader = DataLoader()
            
            # Quick connectivity test
            test_end = datetime.now().strftime('%Y-%m-%d')
            test_start = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            test_data = self.data_loader.get_price_data(
                symbol=self.symbol,
                start=test_start,
                end=test_end
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Database connection successful ({elapsed:.1f}s)")
            
            if not test_data.empty:
                print(f"📊 Test query: {len(test_data)} bars available")
            else:
                print(f"⚠️ No recent data for {self.symbol}")
                
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print(f"   Will use synthetic data fallback")
            self.data_loader = None
    
    def _init_ai_components(self):
        """Initialize AI/transformer components."""
        try:
            print(f"🧠 Initializing AI components...")
            
            # Initialize the transformer-based indicator selector
            self.indicator_selector = AIIndicatorSelector(
                device=self.ai_device,
                learning_rate=self.ai_learning_rate
            )
            
            # Set up initial state
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
            self.regime_confidence = 0.0
            
            print(f"✅ AI Indicator Selector initialized")
            print(f"   Device: {self.ai_device}")
            print(f"   Learning Rate: {self.ai_learning_rate}")
            
        except Exception as e:
            print(f"❌ AI initialization failed: {e}")
            print(f"   Falling back to basic indicators")
            self.adaptive_indicators = False
            self.indicator_selector = None
    
    def load_data_with_ai_preprocessing(self, 
                                      start_date: Optional[str] = None,
                                      end_date: Optional[str] = None,
                                      force_refresh: bool = False) -> pd.DataFrame:
        """
        Load data with AI-specific preprocessing and validation.
        """
        
        # Use cached data if available and recent
        if (self.enable_caching and not force_refresh and 
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
        
        print(f"📊 Loading data for AI analysis: {self.symbol} ({start_date} to {end_date})")
        
        # Load real data or fallback to synthetic
        if self.data_loader is not None:
            try:
                data = self.data_loader.get_price_data(
                    symbol=self.symbol,
                    start=start_date,
                    end=end_date
                )
                
                if data.empty:
                    print(f"⚠️ No real data available, generating synthetic data")
                    data = self._generate_ai_optimized_synthetic_data(start_date, end_date)
                else:
                    print(f"✅ Loaded {len(data):,} bars of real data")
                    data = self._validate_and_clean_data(data)
                    
            except Exception as e:
                print(f"❌ Data loading error: {e}")
                data = self._generate_ai_optimized_synthetic_data(start_date, end_date)
        else:
            data = self._generate_ai_optimized_synthetic_data(start_date, end_date)
        
        # AI-specific data preprocessing
        if self.adaptive_indicators and self.indicator_selector is not None:
            data = self._ai_preprocess_data(data)
        
        # Cache the data
        if self.enable_caching:
            self.cached_data = data
            self.last_data_update = datetime.now()
        
        return data
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data validation for AI processing."""
        
        original_len = len(data)
        
        # Remove invalid data
        data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        data = data[(data['open'] > 0) & (data['high'] > 0) & 
                   (data['low'] > 0) & (data['close'] > 0)]
        data = data[(data['high'] >= data['low']) & 
                   (data['high'] >= data['open']) & (data['high'] >= data['close']) &
                   (data['low'] <= data['open']) & (data['low'] <= data['close'])]
        
        # Remove extreme outliers (important for AI stability)
        for col in ['open', 'high', 'low', 'close']:
            q1 = data[col].quantile(0.005)  # More conservative for AI
            q99 = data[col].quantile(0.995)
            data = data[(data[col] >= q1) & (data[col] <= q99)]
        
        # Ensure sufficient data for AI
        if len(data) < 100:
            print(f"⚠️ Insufficient data for AI analysis ({len(data)} bars)")
        
        data = data.sort_index()
        
        if len(data) < original_len:
            print(f"🧹 Data cleaned: {original_len} → {len(data)} bars")
        
        return data
    
    def _ai_preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """AI-specific data preprocessing."""
        
        # Add AI-relevant features
        data = data.copy()
        
        # Normalize prices for AI stability
        data['price_normalized'] = (data['close'] - data['close'].rolling(50).mean()) / data['close'].rolling(50).std()
        
        # Volume normalization
        data['volume_normalized'] = (data['volume'] - data['volume'].rolling(50).mean()) / data['volume'].rolling(50).std()
        
        # Remove AI preprocessing columns (keep original OHLCV)
        ai_columns = ['price_normalized', 'volume_normalized']
        data = data.drop(columns=[col for col in ai_columns if col in data.columns], errors='ignore')
        
        return data
    
    def _generate_ai_optimized_synthetic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic data optimized for AI training."""
        
        print(f"🔄 Generating AI-optimized synthetic data for {self.symbol}")
        
        # Calculate trading timestamps
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        timestamps = []
        current_date = start_dt
        
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Weekdays only
                for hour in range(9, 16):
                    for minute in range(60):
                        if hour == 9 and minute < 30:
                            continue
                        timestamp = current_date.replace(hour=hour, minute=minute)
                        timestamps.append(timestamp)
            current_date += timedelta(days=1)
        
        timestamps = timestamps[:min(len(timestamps), 3000)]  # AI can handle more data
        
        # Generate realistic market data with AI-friendly patterns
        np.random.seed(hash(self.symbol) % 2**31)
        
        # Symbol-specific parameters
        symbol_params = {
            'AAPL': {'base_price': 180, 'volatility': 0.0008, 'trend_strength': 0.0001},
            'MSFT': {'base_price': 350, 'volatility': 0.0007, 'trend_strength': 0.00008},
            'GOOGL': {'base_price': 140, 'volatility': 0.0009, 'trend_strength': 0.00012},
            'TSLA': {'base_price': 200, 'volatility': 0.0015, 'trend_strength': 0.0002}
        }
        
        params = symbol_params.get(self.symbol, {
            'base_price': 100, 'volatility': 0.001, 'trend_strength': 0.0001
        })
        
        # Generate returns with regime changes (good for AI learning)
        returns = np.random.normal(0, params['volatility'], len(timestamps))
        
        # Add regime-based patterns
        regime_length = len(returns) // 4
        regimes = ['Bull', 'Bear', 'Sideways', 'Volatile']
        
        for i, regime in enumerate(regimes):
            start_idx = i * regime_length
            end_idx = min((i + 1) * regime_length, len(returns))
            
            if regime == 'Bull':
                returns[start_idx:end_idx] += params['trend_strength']
            elif regime == 'Bear':
                returns[start_idx:end_idx] -= params['trend_strength']
            elif regime == 'Volatile':
                returns[start_idx:end_idx] *= 1.5
            # Sideways remains unchanged
        
        # Add intraday patterns
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            
            # Opening volatility
            if hour == 9:
                returns[i] += np.random.normal(0, params['volatility'] * 0.5)
            # Lunch lull
            elif 12 <= hour < 14:
                returns[i] *= 0.8
            # Closing activity
            elif hour >= 15:
                returns[i] += np.random.normal(0, params['volatility'] * 0.3)
        
        # Calculate prices
        prices = params['base_price'] * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        closes = prices
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        
        # Realistic intrabar movement
        noise = np.random.normal(0, params['volatility'] * 0.5, len(closes))
        highs = np.maximum(opens, closes) * (1 + np.abs(noise))
        lows = np.minimum(opens, closes) * (1 - np.abs(noise))
        
        # Volume with regime correlation
        base_volume = 150000 if self.symbol in ['AAPL', 'MSFT', 'TSLA'] else 75000
        volume_multipliers = []
        
        for i, ts in enumerate(timestamps):
            # Time-based volume
            hour = ts.hour
            if hour == 9:
                vol_mult = 2.5
            elif hour == 15:
                vol_mult = 2.0
            else:
                vol_mult = 1.0
            
            # Price-volume correlation
            if i > 0:
                price_change = abs(returns[i])
                vol_mult *= (1 + price_change * 500)
            
            volume = int(base_volume * vol_mult * np.random.lognormal(0, 0.2))
            volume_multipliers.append(max(volume, 1000))
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volume_multipliers
        }, index=pd.DatetimeIndex(timestamps, name='timestamp'))
        
        data.index = data.index.tz_localize('UTC')
        
        print(f"✅ Generated {len(data)} bars of AI-optimized synthetic data")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        return data
    
    def prepare_ai_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features with AI enhancement."""
        
        features = pd.DataFrame(index=data.index)
        
        # Core price features
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
        
        # VWAP and price positioning
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        for window in [10, 20, 50]:
            vwap_num = (typical_price * data['volume']).rolling(window, min_periods=1).sum()
            vwap_den = data['volume'].rolling(window, min_periods=1).sum()
            features[f'vwap_{window}'] = vwap_num / vwap_den
            features[f'vwap_dev_{window}'] = (data['close'] - features[f'vwap_{window}']) / features[f'vwap_{window}']
        
        # AI-enhanced indicator selection
        if self.adaptive_indicators and self.indicator_selector is not None:
            # Check if we need to re-select indicators
            if (self.bars_since_selection >= self.selection_interval or 
                len(self.current_indicators) == 0):
                self._ai_select_indicators(data, features)
                self.bars_since_selection = 0
            else:
                self.bars_since_selection += 1
            
            # Add AI-selected indicators
            self._add_ai_selected_features(data, features)
        else:
            # Fallback to comprehensive manual indicators
            self._add_comprehensive_indicators(data, features)
        
        # Time-based features
        features['hour'] = data.index.hour
        features['minute'] = data.index.minute
        features['is_opening'] = ((features['hour'] == 9) & (features['minute'] >= 30) | 
                                 (features['hour'] == 10)).astype(int)
        features['is_closing'] = (features['hour'] >= 15).astype(int)
        features['is_midday'] = ((features['hour'] >= 12) & (features['hour'] < 14)).astype(int)
        
        # Clean and normalize features
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def _ai_select_indicators(self, data: pd.DataFrame, features: pd.DataFrame):
        """Use AI to dynamically select optimal indicators."""
        
        try:
            print(f"🧠 AI selecting optimal indicators...")
            
            # Prepare data for AI analysis
            sample_size = min(500, len(data))  # Use more data for better AI decisions
            if sample_size < 50:
                print(f"⚠️ Insufficient data for AI selection")
                return
            
            sample_data = data.tail(sample_size).copy()
            sample_features = features.tail(sample_size).copy()
            
            # Calculate returns for AI training
            returns = sample_data['close'].pct_change().fillna(0)
            
            # Create feature matrix for AI
            ai_features = pd.DataFrame(index=sample_data.index)
            ai_features['returns'] = returns
            ai_features['volume_change'] = sample_data['volume'].pct_change().fillna(0)
            ai_features['volatility'] = returns.rolling(10).std().fillna(0)
            ai_features['momentum'] = sample_data['close'].pct_change(5).fillna(0)
            
            # Add some basic technical indicators for AI to evaluate
            ai_features['rsi'] = self._compute_rsi(sample_data['close'])
            ai_features['sma_ratio'] = (sample_data['close'] / sample_data['close'].rolling(20).mean()).fillna(1)
            
            # Use AI to select indicators and detect regime
            selection_result = self.indicator_selector.select_indicators(
                ai_features.fillna(0), 
                top_k=self.top_k_indicators
            )
            
            # Update strategy state
            self.current_indicators = selection_result
            self.last_market_regime = selection_result.get('regime', 'Unknown')
            self.regime_confidence = selection_result.get('confidence', 0.0)
            
            # Track AI performance
            self.ai_selection_history.append({
                'timestamp': datetime.now(),
                'regime': self.last_market_regime,
                'confidence': self.regime_confidence,
                'indicators_count': len([k for k, v in selection_result.items() 
                                       if isinstance(v, bool) and v]),
                'data_points': sample_size
            })
            
            print(f"✅ AI selection complete:")
            print(f"   Regime: {self.last_market_regime} (confidence: {self.regime_confidence:.2f})")
            print(f"   Selected indicators: {len([k for k, v in selection_result.items() if isinstance(v, bool) and v])}")
            
        except Exception as e:
            print(f"❌ AI indicator selection failed: {e}")
            # Use default indicators as fallback
            self.current_indicators = {
                'sma_crossover': True,
                'rsi_momentum': True,
                'volume_breakout': True,
                'volatility_filter': True
            }
            self.last_market_regime = "Unknown"
            self.regime_confidence = 0.0
    
    def _add_ai_selected_features(self, data: pd.DataFrame, features: pd.DataFrame):
        """Add AI-selected indicators to feature set."""
        
        for indicator_name, selected in self.current_indicators.items():
            if not isinstance(selected, bool) or not selected:
                continue
                
            try:
                # Map AI selections to actual indicators
                if 'sma' in indicator_name.lower():
                    periods = [10, 20, 50]
                    for period in periods:
                        features[f'sma_{period}'] = data['close'].rolling(period, min_periods=1).mean()
                        features[f'price_vs_sma_{period}'] = (data['close'] - features[f'sma_{period}']) / features[f'sma_{period}']
                
                elif 'rsi' in indicator_name.lower():
                    features['rsi'] = self._compute_rsi(data['close'])
                    features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
                    features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
                
                elif 'volume' in indicator_name.lower():
                    features['volume_surge'] = (features['volume_ratio'] > 1.5).astype(int)
                    features['volume_expansion'] = (features['volume_ratio'] > features['volume_ratio'].rolling(10).mean()).astype(int)
                
                elif 'volatility' in indicator_name.lower():
                    features['vol_expansion'] = (features['volatility_5'] > features['volatility_20']).astype(int)
                    features['vol_contraction'] = (features['volatility_5'] < features['volatility_20'] * 0.8).astype(int)
                
                elif 'bollinger' in indicator_name.lower():
                    bb_upper, bb_lower = self._compute_bollinger_bands(data['close'])
                    features['bb_upper'] = bb_upper
                    features['bb_lower'] = bb_lower
                    features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
                
                elif 'macd' in indicator_name.lower():
                    features['macd'] = self._compute_macd(data['close'])
                    features['macd_signal'] = features['macd'].ewm(span=9).mean()
                    features['macd_histogram'] = features['macd'] - features['macd_signal']
                    
            except Exception as e:
                print(f"⚠️ Failed to compute AI-selected indicator {indicator_name}: {e}")
                continue
    
    def _add_comprehensive_indicators(self, data: pd.DataFrame, features: pd.DataFrame):
        """Add comprehensive technical indicators when AI is disabled."""
        
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
        
        # Additional indicators
        features['atr'] = self._compute_atr(data)
        features['stoch_k'] = self._compute_stochastic(data)
    
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
    
    def _compute_stochastic(self, data: pd.DataFrame, k_period: int = 14) -> pd.Series:
        """Compute Stochastic Oscillator."""
        lowest_low = data['low'].rolling(window=k_period, min_periods=1).min()
        highest_high = data['high'].rolling(window=k_period, min_periods=1).max()
        return 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    
    def generate_ai_enhanced_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate trading signals enhanced by AI regime detection."""
        
        signals = pd.Series(0, index=features.index)
        
        if len(features) < 50:
            return signals
        
        # Base signal thresholds (adjusted by AI regime detection)
        base_momentum_threshold = 0.0015
        base_volume_threshold = 1.3
        
        # AI regime-based adjustments
        if self.last_market_regime == "Bull" and self.regime_confidence > 0.7:
            momentum_threshold = base_momentum_threshold * 0.8  # More sensitive in bull markets
            volume_threshold = base_volume_threshold * 0.9
            print(f"🐂 Bull regime detected - using aggressive thresholds")
            
        elif self.last_market_regime == "Bear" and self.regime_confidence > 0.7:
            momentum_threshold = base_momentum_threshold * 1.3  # More conservative in bear markets
            volume_threshold = base_volume_threshold * 1.2
            print(f"🐻 Bear regime detected - using conservative thresholds")
            
        elif self.last_market_regime == "Volatile" and self.regime_confidence > 0.6:
            momentum_threshold = base_momentum_threshold * 0.7  # Capitalize on volatility
            volume_threshold = base_volume_threshold * 1.4  # Require higher volume confirmation
            print(f"⚡ Volatile regime detected - using volatility-adapted thresholds")
            
        else:
            momentum_threshold = base_momentum_threshold
            volume_threshold = base_volume_threshold
            print(f"📊 Standard regime - using base thresholds")
        
        # Generate signals based on available features
        if 'returns_5' in features.columns and 'volume_ratio' in features.columns:
            # Momentum signals
            momentum_buy = (
                (features['returns_5'] > momentum_threshold) &
                (features['volume_ratio'] > volume_threshold)
            )
            
            momentum_sell = (
                (features['returns_5'] < -momentum_threshold) &
                (features['volume_ratio'] > volume_threshold)
            )
            
            signals[momentum_buy] = 1
            signals[momentum_sell] = -1
        
        # Add AI-enhanced filtering
        if 'rsi' in features.columns:
            # RSI filters adjusted by regime
            if self.last_market_regime == "Bull":
                rsi_overbought = 75  # Allow higher RSI in bull markets
                rsi_oversold = 25
            else:
                rsi_overbought = 70
                rsi_oversold = 30
            
            overbought = features['rsi'] > rsi_overbought
            oversold = features['rsi'] < rsi_oversold
            signals[(signals == 1) & overbought] = 0
            signals[(signals == -1) & oversold] = 0
        
        # VWAP-based filtering
        if 'vwap_dev_20' in features.columns:
            vwap_threshold = 0.002
            too_far_above = features['vwap_dev_20'] > vwap_threshold
            too_far_below = features['vwap_dev_20'] < -vwap_threshold
            
            signals[(signals == 1) & too_far_above] = 0  # Don't buy too far above VWAP
            signals[(signals == -1) & too_far_below] = 0  # Don't sell too far below VWAP
        
        # Time-based filters enhanced by regime
        if 'is_opening' in features.columns:
            # More aggressive during openings in trending regimes
            if self.last_market_regime in ["Bull", "Bear"] and self.regime_confidence > 0.6:
                opening_signals = (
                    (features['is_opening'] == 1) &
                    (features.get('volume_ratio', 1) > volume_threshold * 0.8)
                )
                if self.last_market_regime == "Bull":
                    signals[opening_signals & (features.get('returns_5', 0) > momentum_threshold * 0.7)] = 1
                else:  # Bear
                    signals[opening_signals & (features.get('returns_5', 0) < -momentum_threshold * 0.7)] = -1
        
        return signals
    
    def backtest_with_ai(self, 
                        data: Optional[pd.DataFrame] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        verbose: bool = True) -> Dict[str, Any]:
        """Run AI-enhanced backtest with detailed analytics."""
        
        if verbose:
            print(f"🤖 Running AI-enhanced backtest for {self.symbol}...")
        
        # Load data with AI preprocessing
        if data is None:
            data = self.load_data_with_ai_preprocessing(start_date, end_date)
        
        if data.empty:
            return {'error': 'No data available for AI backtesting'}
        
        if verbose:
            print(f"📊 AI Backtesting: {len(data)} bars ({data.index[0]} to {data.index[-1]})")
        
        # Prepare AI-enhanced features
        features = self.prepare_ai_enhanced_features(data)
        if verbose:
            print(f"🔧 AI Features: {features.shape[1]} indicators prepared")
            if self.adaptive_indicators:
                print(f"🧠 AI Regime: {self.last_market_regime} (confidence: {self.regime_confidence:.2f})")
        
        # Generate AI-enhanced signals
        signals = self.generate_ai_enhanced_signals(features)
        active_signals = (signals != 0).sum()
        if verbose:
            print(f"🎯 AI Signals: {active_signals} signals generated")
        
        # Enhanced backtest simulation
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        capital = self.capital
        equity_curve = []
        ai_decisions = []
        
        # AI-enhanced risk management
        max_position_pct = 0.10  # Slightly higher for AI
        stop_loss_pct = 0.018    # Tighter stops with AI
        take_profit_pct = 0.035  # Higher targets with AI
        max_hold_minutes = 150   # Shorter holds with AI
        
        for i in range(100, len(data)):  # Longer warmup for AI
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Track AI decision context
            if signal != 0:
                ai_decisions.append({
                    'timestamp': current_time,
                    'signal': signal,
                    'regime': self.last_market_regime,
                    'confidence': self.regime_confidence,
                    'price': current_price
                })
            
            # Track equity curve
            if position != 0:
                unrealized_pnl = (current_price - entry_price) * abs(position)
                if position < 0:
                    unrealized_pnl = -unrealized_pnl
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            equity_curve.append(current_equity)
            
            # AI-enhanced position management
            if position != 0:
                held_minutes = (current_time - entry_time).total_seconds() / 60
                price_change = (current_price - entry_price) / entry_price
                
                # AI regime-adjusted exit conditions
                should_exit = False
                exit_reason = ""
                
                # Regime-adjusted stops and targets
                current_stop = stop_loss_pct
                current_target = take_profit_pct
                
                if self.last_market_regime == "Volatile" and self.regime_confidence > 0.6:
                    current_stop *= 1.2  # Wider stops in volatile markets
                    current_target *= 0.8  # Quicker profits in volatile markets
                elif self.last_market_regime in ["Bull", "Bear"] and self.regime_confidence > 0.7:
                    current_target *= 1.2  # Higher targets in trending markets
                
                if position > 0:  # Long position
                    if price_change <= -current_stop:
                        should_exit = True
                        exit_reason = "AI Stop Loss"
                    elif price_change >= current_target:
                        should_exit = True
                        exit_reason = "AI Take Profit"
                else:  # Short position
                    if price_change >= current_stop:
                        should_exit = True
                        exit_reason = "AI Stop Loss"
                    elif price_change <= -current_target:
                        should_exit = True
                        exit_reason = "AI Take Profit"
                
                if held_minutes >= max_hold_minutes:
                    should_exit = True
                    exit_reason = "AI Max Hold Time"
                
                if signal != 0 and signal != position:
                    should_exit = True
                    exit_reason = "AI Signal Reversal"
                
                # Execute exit
                if should_exit:
                    shares = abs(position)
                    gross_pnl = (current_price - entry_price) * shares
                    if position < 0:
                        gross_pnl = -gross_pnl
                    
                    # Trading costs
                    commission = shares * 0.005
                    slippage = shares * current_price * 0.0002  # Lower slippage with AI
                    net_pnl = gross_pnl - commission - slippage
                    
                    trade_return = net_pnl / (entry_price * shares)
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'direction': 'Long' if position > 0 else 'Short',
                        'gross_pnl': gross_pnl,
                        'net_pnl': net_pnl,
                        'return_pct': trade_return,
                        'duration_minutes': held_minutes,
                        'exit_reason': exit_reason,
                        'entry_regime': getattr(self, '_entry_regime', 'Unknown'),
                        'exit_regime': self.last_market_regime,
                        'regime_confidence': self.regime_confidence
                    })
                    
                    capital += net_pnl
                    position = 0
                    entry_price = 0
                    entry_time = None
            
            # AI-enhanced entry logic
            if signal != 0 and position == 0:
                # Regime-adjusted position sizing
                base_position_pct = max_position_pct
                if self.last_market_regime == "Volatile" and self.regime_confidence > 0.6:
                    base_position_pct *= 0.8  # Smaller positions in volatile markets
                elif self.last_market_regime in ["Bull", "Bear"] and self.regime_confidence > 0.8:
                    base_position_pct *= 1.1  # Larger positions in strong trends
                
                position_value = capital * base_position_pct
                shares = int(position_value / current_price)
                
                if shares > 0:
                    position = shares * signal
                    entry_price = current_price
                    entry_time = current_time
                    self._entry_regime = self.last_market_regime
        
        # Calculate AI-enhanced metrics
        if trades and len(equity_curve) > 0:
            total_trades = len(trades)
            wins = [t for t in trades if t['net_pnl'] > 0]
            losses = [t for t in trades if t['net_pnl'] <= 0]
            
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            total_pnl = sum(t['net_pnl'] for t in trades)
            total_return = (capital - self.capital) / self.capital
            
            avg_win = np.mean([t['net_pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['net_pnl'] for t in losses]) if losses else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # AI-specific metrics
            regime_performance = {}
            for trade in trades:
                regime = trade.get('entry_regime', 'Unknown')
                if regime not in regime_performance:
                    regime_performance[regime] = {'trades': 0, 'pnl': 0, 'wins': 0}
                regime_performance[regime]['trades'] += 1
                regime_performance[regime]['pnl'] += trade['net_pnl']
                if trade['net_pnl'] > 0:
                    regime_performance[regime]['wins'] += 1
            
            # Calculate regime win rates
            for regime in regime_performance:
                regime_performance[regime]['win_rate'] = (
                    regime_performance[regime]['wins'] / regime_performance[regime]['trades']
                    if regime_performance[regime]['trades'] > 0 else 0
                )
            
            # Advanced metrics
            returns = [t['return_pct'] for t in trades]
            sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 and np.std(returns) > 0 else 0
            
            # Drawdown
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown_pct = drawdown.min()
            
        else:
            total_trades = win_rate = total_pnl = total_return = 0
            profit_factor = avg_win = avg_loss = sharpe_ratio = 0
            max_drawdown_pct = 0
            regime_performance = {}
        
        # Compile AI-enhanced results
        results = {
            'symbol': self.symbol,
            'ai_enabled': self.adaptive_indicators,
            'backtest_period': f"{data.index[0]} to {data.index[-1]}",
            'total_bars': len(data),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'final_capital': capital,
            'ai_regime': self.last_market_regime,
            'ai_confidence': self.regime_confidence,
            'features_count': features.shape[1],
            'signals_generated': active_signals,
            'ai_decisions_count': len(ai_decisions),
            'regime_performance': regime_performance,
            'ai_selection_history': self.ai_selection_history,
            'trades_sample': trades[:5] if trades else []  # First 5 trades
        }
        
        if verbose:
            self._print_ai_results(results)
        
        return results
    
    def _print_ai_results(self, results: Dict[str, Any]):
        """Print AI-enhanced backtest results."""
        print(f"\n🤖 AI-Enhanced Backtest Results for {results['symbol']}:")
        print("=" * 50)
        print(f"Period: {results['backtest_period']}")
        print(f"AI Enabled: {'✅ Yes' if results['ai_enabled'] else '❌ No'}")
        print(f"Total Bars: {results['total_bars']:,}")
        print()
        print("📊 Trading Performance:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Total Return: {results['total_return_pct']:.1%}")
        print(f"  Total PnL: ${results['total_pnl']:.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.1%}")
        print()
        print("🧠 AI Intelligence:")
        print(f"  Current Regime: {results['ai_regime']}")
        print(f"  Confidence: {results['ai_confidence']:.2f}")
        print(f"  Features Used: {results['features_count']}")
        print(f"  AI Decisions: {results['ai_decisions_count']}")
        print(f"  Regime Changes: {len(results['ai_selection_history'])}")
        
        if results['regime_performance']:
            print("\n📈 Performance by Regime:")
            for regime, perf in results['regime_performance'].items():
                print(f"  {regime}: {perf['trades']} trades, "
                      f"{perf['win_rate']:.1%} win rate, "
                      f"${perf['pnl']:.2f} PnL")


def test_ai_enabled_s3():
    """Test the AI-enabled S3 strategy."""
    print("🤖 Testing S3 AI-Enabled Strategy")
    print("=" * 40)
    
    symbols = ['AAPL', 'MSFT']
    results_summary = []
    
    for symbol in symbols:
        print(f"\n🔍 Testing AI-enabled strategy with {symbol}...")
        print("-" * 40)
        
        try:
            # Initialize AI-enabled strategy
            strategy = S3AIEnabledStrategy(
                symbol=symbol,
                capital=100000,
                adaptive_indicators=True,  # AI ENABLED
                top_k_indicators=10,
                selection_interval=30,
                data_days_lookback=5,
                ai_learning_rate=0.001,
                enable_regime_detection=True
            )
            
            # Run AI-enhanced backtest
            results = strategy.backtest_with_ai(verbose=True)
            
            # Store summary
            results_summary.append({
                'symbol': symbol,
                'ai_enabled': results.get('ai_enabled', False),
                'trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0),
                'return_pct': results.get('total_return_pct', 0),
                'sharpe': results.get('sharpe_ratio', 0),
                'regime': results.get('ai_regime', 'Unknown'),
                'confidence': results.get('ai_confidence', 0)
            })
            
            # Save results
            output_file = f's3_ai_results_{symbol}.json'
            with open(output_file, 'w') as f:
                save_results = results.copy()
                save_results.pop('trades_sample', None)
                json.dump(save_results, f, indent=2, default=str)
            
            print(f"💾 AI results saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error testing AI strategy for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary comparison
    if results_summary:
        print(f"\n🤖 AI Strategy Summary:")
        print("=" * 80)
        print(f"{'Symbol':<8} {'AI':<5} {'Trades':<8} {'Win%':<8} {'Return%':<8} {'Sharpe':<8} {'Regime':<12} {'Conf':<6}")
        print("-" * 80)
        for result in results_summary:
            ai_status = "✅" if result['ai_enabled'] else "❌"
            print(f"{result['symbol']:<8} {ai_status:<5} {result['trades']:<8} "
                  f"{result['win_rate']:<8.1%} {result['return_pct']:<8.1%} "
                  f"{result['sharpe']:<8.2f} {result['regime']:<12} {result['confidence']:<6.2f}")
    
    print(f"\n✅ AI testing complete!")


if __name__ == "__main__":
    test_ai_enabled_s3()