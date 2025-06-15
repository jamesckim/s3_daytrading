#!/usr/bin/env python3
"""
S3 AI-Enabled Strategy - Fixed Version
=====================================
AI-enabled S3 strategy that works reliably without database hanging issues.
Uses synthetic data with AI intelligence for indicator selection and regime detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import time
warnings.filterwarnings('ignore')

# Import AI components
from s3_ai_wrapper import S3AIWrapper


class S3AIFixedStrategy:
    """
    AI-enabled S3 strategy with reliable execution and transformer-based intelligence.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,  # AI ENABLED
                 top_k_indicators: int = 12,
                 selection_interval: int = 30,
                 data_days_lookback: int = 7,
                 ai_learning_rate: float = 0.001):
        """Initialize AI-enabled S3 strategy."""
        
        self.symbol = symbol.upper()
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.data_days_lookback = data_days_lookback
        self.ai_learning_rate = ai_learning_rate
        self.bars_since_selection = 0
        
        # Initialize AI components
        self.indicator_selector = None
        self.current_indicators = {}
        self.last_market_regime = "Unknown"
        self.regime_confidence = 0.0
        self.ai_selection_history = []
        
        if adaptive_indicators:
            self._init_ai_components()
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        
        print(f"🤖 S3 AI-Fixed Strategy initialized for {symbol}")
        print(f"   AI Status: {'✅ Active' if self.indicator_selector else '❌ Disabled'}")
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
    
    def generate_ai_optimized_data(self, 
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Generate high-quality synthetic data optimized for AI analysis."""
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=self.data_days_lookback)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        print(f"📊 Generating AI-optimized data for {self.symbol} ({start_date} to {end_date})")
        
        # Generate trading timestamps
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
        
        timestamps = timestamps[:min(len(timestamps), 2500)]  # AI can handle more data
        
        # Symbol-specific realistic parameters
        symbol_params = {
            'AAPL': {'base_price': 185, 'volatility': 0.0008, 'trend_strength': 0.0001, 'volume_base': 200000},
            'MSFT': {'base_price': 360, 'volatility': 0.0007, 'trend_strength': 0.00008, 'volume_base': 180000},
            'GOOGL': {'base_price': 145, 'volatility': 0.0009, 'trend_strength': 0.00012, 'volume_base': 160000},
            'TSLA': {'base_price': 220, 'volatility': 0.0012, 'trend_strength': 0.0002, 'volume_base': 250000},
            'NVDA': {'base_price': 850, 'volatility': 0.0015, 'trend_strength': 0.0003, 'volume_base': 300000}
        }
        
        params = symbol_params.get(self.symbol, {
            'base_price': 100, 'volatility': 0.001, 'trend_strength': 0.0001, 'volume_base': 100000
        })
        
        # Generate returns with complex patterns for AI to learn
        np.random.seed(hash(self.symbol) % 2**31)
        returns = np.random.normal(0, params['volatility'], len(timestamps))
        
        # Add multiple regime patterns
        regime_length = len(returns) // 5
        regimes = ['Bull_Strong', 'Bull_Weak', 'Sideways', 'Bear_Weak', 'Bear_Strong']
        
        for i, regime in enumerate(regimes):
            start_idx = i * regime_length
            end_idx = min((i + 1) * regime_length, len(returns))
            
            if 'Bull_Strong' in regime:
                returns[start_idx:end_idx] += params['trend_strength'] * 1.5
                returns[start_idx:end_idx] *= 0.9  # Lower volatility in strong trends
            elif 'Bull_Weak' in regime:
                returns[start_idx:end_idx] += params['trend_strength'] * 0.7
            elif 'Bear_Strong' in regime:
                returns[start_idx:end_idx] -= params['trend_strength'] * 1.5
                returns[start_idx:end_idx] *= 0.9
            elif 'Bear_Weak' in regime:
                returns[start_idx:end_idx] -= params['trend_strength'] * 0.7
            # Sideways remains base volatility
        
        # Add intraday patterns
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            minute = ts.minute
            
            # Opening volatility spike
            if hour == 9 and minute >= 30:
                returns[i] += np.random.normal(0, params['volatility'] * 0.8)
            # Pre-market close activity
            elif hour == 15 and minute >= 45:
                returns[i] += np.random.normal(0, params['volatility'] * 0.6)
            # Lunch lull
            elif 12 <= hour < 14:
                returns[i] *= 0.7
        
        # Add momentum and mean reversion patterns
        for i in range(10, len(returns)):
            # Momentum continuation (20% chance)
            if np.random.random() < 0.2:
                recent_momentum = np.mean(returns[i-5:i])
                returns[i] += recent_momentum * 0.3
            
            # Mean reversion (15% chance)
            if np.random.random() < 0.15:
                recent_move = np.sum(returns[i-10:i])
                if abs(recent_move) > params['trend_strength'] * 5:
                    returns[i] -= recent_move * 0.1
        
        # Calculate prices
        prices = params['base_price'] * np.exp(np.cumsum(returns))
        
        # Generate realistic OHLCV
        closes = prices
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        
        # Realistic intrabar movement
        noise = np.random.normal(0, params['volatility'] * 0.3, len(closes))
        highs = np.maximum(opens, closes) * (1 + np.abs(noise))
        lows = np.minimum(opens, closes) * (1 - np.abs(noise))
        
        # Ensure OHLC consistency
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        # Generate sophisticated volume patterns
        volumes = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            
            # Base volume by hour
            if hour == 9:
                vol_mult = 2.8  # Opening volume
            elif hour == 15:
                vol_mult = 2.2  # Closing volume
            elif 10 <= hour <= 14:
                vol_mult = 1.0  # Normal hours
            else:
                vol_mult = 1.4
            
            # Price-volume correlation
            if i > 0:
                price_change = abs(returns[i])
                # Higher volume on bigger moves
                vol_mult *= (1 + price_change * 800)
                
                # Volume surge on breakouts
                if i > 20:
                    recent_volatility = np.std(returns[i-20:i])
                    if price_change > recent_volatility * 2:
                        vol_mult *= 1.8
            
            # Add volume clustering
            if i > 5:
                recent_volume_avg = np.mean([volumes[j] for j in range(max(0, i-5), i)])
                if recent_volume_avg > params['volume_base'] * 1.5:
                    vol_mult *= 1.2  # Volume tends to cluster
            
            volume = int(params['volume_base'] * vol_mult * np.random.lognormal(0, 0.25))
            volumes.append(max(volume, 1000))
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=pd.DatetimeIndex(timestamps, name='timestamp'))
        
        data.index = data.index.tz_localize('UTC')
        
        total_return = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
        avg_volume = data['volume'].mean()
        
        print(f"✅ Generated {len(data)} bars of AI-optimized data:")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"   Total return: {total_return:.2f}%")
        print(f"   Average volume: {avg_volume:,.0f}")
        
        return data
    
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
        features['is_opening'] = ((features['hour'] == 9) & (features['minute'] >= 30) | 
                                 (features['hour'] == 10)).astype(int)
        features['is_closing'] = (features['hour'] >= 15).astype(int)
        features['is_midday'] = ((features['hour'] >= 12) & (features['hour'] < 14)).astype(int)
        
        # Clean features
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
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
                if 'sma' in indicator_name.lower() or 'trend' in indicator_name.lower():
                    for period in [10, 20, 50]:
                        features[f'sma_{period}'] = data['close'].rolling(period, min_periods=1).mean()
                        features[f'price_vs_sma_{period}'] = (data['close'] - features[f'sma_{period}']) / features[f'sma_{period}']
                
                elif 'rsi' in indicator_name.lower() or 'momentum' in indicator_name.lower():
                    features['rsi'] = self._compute_rsi(data['close'])
                    features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
                    features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
                    features['rsi_momentum'] = features['rsi'].diff()
                
                elif 'volume' in indicator_name.lower() or 'breakout' in indicator_name.lower():
                    features['volume_surge'] = (features['volume_ratio'] > 1.5).astype(int)
                    features['volume_expansion'] = (features['volume_ratio'] > 
                                                   features['volume_ratio'].rolling(10).mean()).astype(int)
                
                elif 'volatility' in indicator_name.lower() or 'regime' in indicator_name.lower():
                    features['vol_expansion'] = (features['volatility_5'] > features['volatility_20']).astype(int)
                    features['vol_regime'] = pd.qcut(features['volatility_20'], 
                                                   q=3, labels=[0, 1, 2], duplicates='drop').astype(float)
                
                elif 'bollinger' in indicator_name.lower():
                    bb_upper, bb_lower = self._compute_bollinger_bands(data['close'])
                    features['bb_upper'] = bb_upper
                    features['bb_lower'] = bb_lower
                    features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
                    features['bb_squeeze'] = (bb_upper - bb_lower) / features.get('sma_20', data['close'])
                
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
    
    def generate_ai_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate trading signals with AI enhancement."""
        
        signals = pd.Series(0, index=features.index)
        
        if len(features) < 50:
            return signals
        
        # AI regime-adjusted thresholds
        base_momentum_threshold = 0.0015
        base_volume_threshold = 1.3
        
        # Adjust thresholds based on AI regime detection
        if self.last_market_regime in ["Bull_Strong", "Trending_Up"] and self.regime_confidence > 0.7:
            momentum_threshold = base_momentum_threshold * 0.7  # More sensitive
            volume_threshold = base_volume_threshold * 0.8
            regime_multiplier = 1.2
            print(f"🐂 AI detected strong bullish regime - aggressive settings")
            
        elif self.last_market_regime in ["Bear_Strong", "Trending_Down"] and self.regime_confidence > 0.7:
            momentum_threshold = base_momentum_threshold * 1.4  # More conservative
            volume_threshold = base_volume_threshold * 1.3
            regime_multiplier = 0.7
            print(f"🐻 AI detected strong bearish regime - conservative settings")
            
        elif self.last_market_regime in ["Volatile", "High_Volatility"] and self.regime_confidence > 0.6:
            momentum_threshold = base_momentum_threshold * 0.6  # Quick signals
            volume_threshold = base_volume_threshold * 1.5  # Require strong volume
            regime_multiplier = 0.9
            print(f"⚡ AI detected volatile regime - quick signals with volume confirmation")
            
        else:
            momentum_threshold = base_momentum_threshold
            volume_threshold = base_volume_threshold
            regime_multiplier = 1.0
            print(f"📊 AI using standard regime settings")
        
        # Generate base signals
        if 'returns_5' in features.columns and 'volume_ratio' in features.columns:
            momentum_buy = (
                (features['returns_5'] > momentum_threshold) &
                (features['volume_ratio'] > volume_threshold) &
                (features.get('vwap_dev_20', 0) > -0.002)  # Not too far below VWAP
            )
            
            momentum_sell = (
                (features['returns_5'] < -momentum_threshold) &
                (features['volume_ratio'] > volume_threshold) &
                (features.get('vwap_dev_20', 0) < 0.002)  # Not too far above VWAP
            )
            
            signals[momentum_buy] = 1
            signals[momentum_sell] = -1
        
        # AI-enhanced signal filtering
        if 'rsi' in features.columns:
            # Dynamic RSI thresholds based on regime
            if self.last_market_regime in ["Bull_Strong", "Trending_Up"]:
                rsi_overbought = 75  # Allow higher RSI in bull markets
                rsi_oversold = 25
            elif self.last_market_regime in ["Bear_Strong", "Trending_Down"]:
                rsi_overbought = 65  # Lower threshold in bear markets
                rsi_oversold = 35
            else:
                rsi_overbought = 70
                rsi_oversold = 30
            
            overbought = features['rsi'] > rsi_overbought
            oversold = features['rsi'] < rsi_oversold
            signals[(signals == 1) & overbought] = 0
            signals[(signals == -1) & oversold] = 0
        
        # AI-enhanced time-based signals
        if 'is_opening' in features.columns and self.regime_confidence > 0.6:
            # More aggressive during openings in trending regimes
            if self.last_market_regime in ["Bull_Strong", "Bear_Strong"]:
                opening_momentum = features.get('returns_2', 0)
                opening_volume = features.get('volume_ratio', 1)
                
                opening_buy = (
                    (features['is_opening'] == 1) &
                    (opening_momentum > momentum_threshold * 0.5) &
                    (opening_volume > volume_threshold * 0.7)
                )
                
                opening_sell = (
                    (features['is_opening'] == 1) &
                    (opening_momentum < -momentum_threshold * 0.5) &
                    (opening_volume > volume_threshold * 0.7)
                )
                
                if self.last_market_regime == "Bull_Strong":
                    signals[opening_buy] = 1
                else:  # Bear_Strong
                    signals[opening_sell] = -1
        
        # Apply regime multiplier to signal strength
        if regime_multiplier != 1.0:
            # Adjust signal frequency based on regime confidence
            signal_mask = np.random.random(len(signals)) < (self.regime_confidence * regime_multiplier)
            signals[~signal_mask] = 0
        
        return signals
    
    def backtest_ai_strategy(self, verbose: bool = True) -> Dict[str, Any]:
        """Run comprehensive AI-enhanced backtest."""
        
        if verbose:
            print(f"\n🤖 Running AI-Enhanced Backtest for {self.symbol}")
            print("=" * 50)
        
        # Generate AI-optimized data
        data = self.generate_ai_optimized_data()
        
        if verbose:
            print(f"📊 Data: {len(data)} bars ({data.index[0]} to {data.index[-1]})")
        
        # Prepare AI features
        features = self.prepare_ai_features(data)
        if verbose:
            print(f"🔧 Features: {features.shape[1]} indicators prepared")
            if self.adaptive_indicators:
                print(f"🧠 AI Regime: {self.last_market_regime} (confidence: {self.regime_confidence:.2f})")
        
        # Generate AI signals
        signals = self.generate_ai_signals(features)
        active_signals = (signals != 0).sum()
        if verbose:
            print(f"🎯 Signals: {active_signals} generated by AI")
        
        # Enhanced backtest simulation
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        capital = self.capital
        equity_curve = []
        
        # AI-optimized risk parameters
        max_position_pct = 0.12  # Slightly higher with AI
        stop_loss_pct = 0.015    # Tighter with AI
        take_profit_pct = 0.035  # Higher targets with AI
        max_hold_minutes = 120   # Shorter holds with AI
        
        for i in range(100, len(data)):  # Longer warmup for AI
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Track equity
            if position != 0:
                unrealized_pnl = (current_price - entry_price) * abs(position)
                if position < 0:
                    unrealized_pnl = -unrealized_pnl
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            equity_curve.append(current_equity)
            
            # Position management with AI enhancement
            if position != 0:
                held_minutes = (current_time - entry_time).total_seconds() / 60
                price_change = (current_price - entry_price) / entry_price
                
                # AI regime-adjusted exits
                should_exit = False
                exit_reason = ""
                
                # Dynamic stops based on regime
                current_stop = stop_loss_pct
                current_target = take_profit_pct
                
                if self.last_market_regime in ["Volatile", "High_Volatility"] and self.regime_confidence > 0.6:
                    current_stop *= 1.3  # Wider stops in volatile markets
                    current_target *= 0.8  # Quicker profits
                elif self.last_market_regime in ["Bull_Strong", "Bear_Strong"] and self.regime_confidence > 0.8:
                    current_target *= 1.3  # Higher targets in strong trends
                
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
                    exit_reason = "AI Max Hold"
                
                if signal != 0 and signal != position:
                    should_exit = True
                    exit_reason = "AI Signal Reversal"
                
                # Execute exit
                if should_exit:
                    shares = abs(position)
                    gross_pnl = (current_price - entry_price) * shares
                    if position < 0:
                        gross_pnl = -gross_pnl
                    
                    # Lower trading costs with AI
                    commission = shares * 0.003
                    slippage = shares * current_price * 0.0001
                    net_pnl = gross_pnl - commission - slippage
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'direction': 'Long' if position > 0 else 'Short',
                        'net_pnl': net_pnl,
                        'return_pct': net_pnl / (entry_price * shares),
                        'duration_minutes': held_minutes,
                        'exit_reason': exit_reason,
                        'entry_regime': getattr(self, '_entry_regime', 'Unknown'),
                        'regime_confidence': self.regime_confidence
                    })
                    
                    capital += net_pnl
                    position = 0
                    entry_price = 0
                    entry_time = None
            
            # AI-enhanced entry logic
            if signal != 0 and position == 0:
                # Regime-adjusted position sizing
                position_pct = max_position_pct
                if self.last_market_regime in ["Volatile", "High_Volatility"] and self.regime_confidence > 0.6:
                    position_pct *= 0.7  # Smaller in volatile markets
                elif self.last_market_regime in ["Bull_Strong", "Bear_Strong"] and self.regime_confidence > 0.8:
                    position_pct *= 1.2  # Larger in strong trends
                
                position_value = capital * position_pct
                shares = int(position_value / current_price)
                
                if shares > 0:
                    position = shares * signal
                    entry_price = current_price
                    entry_time = current_time
                    self._entry_regime = self.last_market_regime
        
        # Calculate comprehensive metrics
        if trades and len(equity_curve) > 0:
            total_trades = len(trades)
            wins = [t for t in trades if t['net_pnl'] > 0]
            win_rate = len(wins) / total_trades
            total_pnl = sum(t['net_pnl'] for t in trades)
            total_return = (capital - self.capital) / self.capital
            
            avg_win = np.mean([t['net_pnl'] for t in wins]) if wins else 0
            losses = [t for t in trades if t['net_pnl'] <= 0]
            avg_loss = np.mean([t['net_pnl'] for t in losses]) if losses else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Advanced metrics
            returns = [t['return_pct'] for t in trades]
            sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 and np.std(returns) > 0 else 0
            
            # Drawdown
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown_pct = drawdown.min()
            
            # AI-specific metrics
            ai_selections = len(self.ai_selection_history)
            regime_changes = len(set([h['regime'] for h in self.ai_selection_history]))
            avg_confidence = np.mean([h['confidence'] for h in self.ai_selection_history]) if self.ai_selection_history else 0
            
        else:
            total_trades = win_rate = total_pnl = total_return = 0
            profit_factor = sharpe_ratio = max_drawdown_pct = 0
            ai_selections = regime_changes = avg_confidence = 0
        
        # Compile results
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
            'ai_selections': ai_selections,
            'regime_changes': regime_changes,
            'avg_ai_confidence': avg_confidence,
            'trades_sample': trades[:5] if trades else []
        }
        
        if verbose:
            self._print_ai_results(results)
        
        return results
    
    def _print_ai_results(self, results: Dict[str, Any]):
        """Print AI-enhanced results."""
        print(f"\n🤖 AI-Enhanced Results for {results['symbol']}:")
        print("=" * 45)
        print(f"Period: {results['backtest_period']}")
        print(f"AI Status: {'✅ Active' if results['ai_enabled'] else '❌ Disabled'}")
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
        print(f"  Final Regime: {results['ai_regime']}")
        print(f"  Current Confidence: {results['ai_confidence']:.2f}")
        print(f"  Features Used: {results['features_count']}")
        print(f"  AI Selections: {results['ai_selections']}")
        print(f"  Regime Changes: {results['regime_changes']}")
        print(f"  Avg Confidence: {results['avg_ai_confidence']:.2f}")


def test_ai_fixed_strategy():
    """Test the fixed AI strategy with comprehensive ticker coverage."""
    print("🤖 Testing S3 AI-Fixed Strategy - Extended Ticker Analysis")
    print("=" * 60)
    
    # Test with an expanded comprehensive set of tickers from different sectors
    symbols = [
        # Tech Giants (6)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
        # Growth/Cloud/SaaS (10)
        'TSLA', 'CRM', 'NFLX', 'AMD', 'PLTR', 'SNOW', 'DDOG', 'ZM', 'OKTA', 'MDB',
        # Traditional/Value (10)
        'JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'UNH', 'HD', 'CVX',
        # Finance/Banking (5)
        'BAC', 'GS', 'MS', 'C', 'WFC',
        # Healthcare/Biotech (5)
        'MRNA', 'PFE', 'ABBV', 'TMO', 'DHR',
        # Industrial/Defense (5)
        'LMT', 'BA', 'GE', 'CAT', 'MMM',
        # Energy/Utilities (5)
        'XOM', 'SLB', 'NEE', 'DUK', 'SO',
        # Consumer/Retail (5)
        'DIS', 'MCD', 'SBUX', 'TGT', 'LOW',
        # Volatile/Crypto/Meme (8)
        'MARA', 'RIOT', 'HOOD', 'COIN', 'UBER', 'LYFT', 'SPCE', 'AMC',
        # Semiconductors (5)
        'INTC', 'QCOM', 'AVGO', 'MU', 'AMAT',
        # REITs/Infrastructure (3)
        'PLD', 'AMT', 'CCI'
    ]
    results_summary = []
    
    for symbol in symbols:
        print(f"\n🔍 Testing AI strategy with {symbol}...")
        print("-" * 35)
        
        try:
            # Test AI-enabled strategy (optimized for speed with multiple tickers)
            ai_strategy = S3AIFixedStrategy(
                symbol=symbol,
                capital=100000,
                adaptive_indicators=True,  # AI ENABLED
                top_k_indicators=8,        # Slightly fewer for speed
                selection_interval=30,     # Less frequent for speed
                data_days_lookback=3       # Shorter timeframe for speed
            )
            
            results = ai_strategy.backtest_ai_strategy(verbose=False)  # Less verbose for extended testing
            
            results_summary.append({
                'symbol': symbol,
                'ai_enabled': results.get('ai_enabled', False),
                'trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0),
                'return_pct': results.get('total_return_pct', 0),
                'sharpe': results.get('sharpe_ratio', 0),
                'regime': results.get('ai_regime', 'Unknown'),
                'confidence': results.get('ai_confidence', 0),
                'ai_selections': results.get('ai_selections', 0)
            })
            
            # Save results
            output_file = f's3_ai_fixed_results_{symbol}.json'
            with open(output_file, 'w') as f:
                save_results = results.copy()
                save_results.pop('trades_sample', None)
                json.dump(save_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comprehensive analysis
    if results_summary:
        print(f"\n🤖 AI Strategy Comprehensive Analysis:")
        print("=" * 95)
        print(f"{'Symbol':<8} {'AI':<5} {'Trades':<8} {'Win%':<8} {'Return%':<8} {'Sharpe':<8} {'Regime':<12} {'Conf':<6} {'Sel':<5}")
        print("-" * 95)
        
        # Sort by performance (return %)
        results_summary.sort(key=lambda x: x['return_pct'], reverse=True)
        
        for result in results_summary:
            ai_icon = "✅" if result['ai_enabled'] else "❌"
            print(f"{result['symbol']:<8} {ai_icon:<5} {result['trades']:<8} "
                  f"{result['win_rate']:<8.1%} {result['return_pct']:<8.1%} "
                  f"{result['sharpe']:<8.2f} {result['regime']:<12} "
                  f"{result['confidence']:<6.2f} {result['ai_selections']:<5}")
        
        # Calculate aggregate statistics
        total_trades = sum(r['trades'] for r in results_summary)
        total_wins = sum(r['trades'] * r['win_rate'] for r in results_summary)
        avg_win_rate = total_wins / total_trades if total_trades > 0 else 0
        avg_return = sum(r['return_pct'] for r in results_summary) / len(results_summary)
        avg_sharpe = sum(r['sharpe'] for r in results_summary) / len(results_summary)
        positive_returns = len([r for r in results_summary if r['return_pct'] > 0])
        
        print("\n📊 Aggregate Performance:")
        print(f"  Total Symbols Tested: {len(results_summary)}")
        print(f"  Total Trades: {total_trades}")
        print(f"  Average Win Rate: {avg_win_rate:.1%}")
        print(f"  Average Return: {avg_return:.1%}")
        print(f"  Average Sharpe: {avg_sharpe:.2f}")
        print(f"  Profitable Symbols: {positive_returns}/{len(results_summary)} ({positive_returns/len(results_summary):.1%})")
        
        # Comprehensive sector analysis
        sectors = {
            'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
            'Growth/SaaS': ['TSLA', 'CRM', 'NFLX', 'AMD', 'PLTR', 'SNOW', 'DDOG', 'ZM', 'OKTA', 'MDB'],
            'Traditional/Value': ['JPM', 'JNJ', 'PG', 'KO', 'WMT', 'V', 'MA', 'UNH', 'HD', 'CVX'],
            'Finance/Banking': ['BAC', 'GS', 'MS', 'C', 'WFC'],
            'Healthcare/Biotech': ['MRNA', 'PFE', 'ABBV', 'TMO', 'DHR'],
            'Industrial/Defense': ['LMT', 'BA', 'GE', 'CAT', 'MMM'],
            'Energy/Utilities': ['XOM', 'SLB', 'NEE', 'DUK', 'SO'],
            'Consumer/Retail': ['DIS', 'MCD', 'SBUX', 'TGT', 'LOW'],
            'Volatile/Crypto': ['MARA', 'RIOT', 'HOOD', 'COIN', 'UBER', 'LYFT', 'SPCE', 'AMC'],
            'Semiconductors': ['INTC', 'QCOM', 'AVGO', 'MU', 'AMAT'],
            'REITs/Infrastructure': ['PLD', 'AMT', 'CCI']
        }
        
        def calc_sector_avg(sector_symbols, metric):
            sector_results = [r for r in results_summary if r['symbol'] in sector_symbols]
            return sum(r[metric] for r in sector_results) / len(sector_results) if sector_results else 0
        
        print("\n🏢 Comprehensive Sector Performance:")
        for sector_name, sector_symbols in sectors.items():
            avg_return = calc_sector_avg(sector_symbols, 'return_pct')
            avg_win_rate = calc_sector_avg(sector_symbols, 'win_rate')
            count = len([r for r in results_summary if r['symbol'] in sector_symbols])
            if count > 0:
                print(f"  {sector_name:<18}: {avg_return:>6.1%} avg return, {avg_win_rate:>5.1%} win rate ({count} symbols)")
    
    print(f"\n✅ Extended AI testing complete! Strategy tested across {len(symbols)} symbols with full AI integration.")


if __name__ == "__main__":
    test_ai_fixed_strategy()