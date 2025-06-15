#!/usr/bin/env python3
"""
ML-Enhanced Signal Strength Strategy
===================================
Uses machine learning to discover additional signals and optimize signal strength scoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import sys
import os
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_strength_position_sizing import SignalStrengthPositionSizing
from day_trading_o3 import DataLoader

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class MLEnhancedSignalStrength(SignalStrengthPositionSizing):
    """
    ML-Enhanced signal strength strategy that uses machine learning to:
    1. Discover additional technical indicators as signals
    2. Optimize signal weights automatically
    3. Predict optimal position sizes
    4. Adapt to market conditions dynamically
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 # Base strategy parameters
                 volume_multiplier: float = 0.3,
                 breakout_threshold: float = 0.0002,
                 profit_target_pct: float = 0.018,
                 stop_loss_pct: float = 0.012,
                 max_holding_minutes: int = 150,
                 min_prev_day_range_pct: float = 0.005,
                 volume_lookback: int = 20,
                 
                 # Regime awareness parameters
                 enable_regime_awareness: bool = True,
                 volatility_lookback: int = 20,
                 trend_lookback: int = 10,
                 vix_threshold_high: float = 25.0,
                 vix_threshold_low: float = 15.0,
                 trend_threshold: float = 0.02,
                 
                 # ML-enhanced parameters
                 enable_ml_signals: bool = True,
                 ml_lookback_window: int = 1000,  # Minimum data for ML training
                 feature_selection_k: int = 15,   # Top K features to select
                 ml_retrain_frequency: int = 500, # Retrain every N bars
                 
                 # Signal strength parameters
                 enable_signal_strength: bool = True,
                 strong_signal_threshold: float = 0.70,  # Lowered from 0.75
                 weak_signal_threshold: float = 0.35,   # Lowered from 0.40
                 min_position_size: float = 0.25,
                 max_position_size: float = 1.00,
                 
                 # Standard improvements
                 avoid_first_hour: bool = True,
                 avoid_lunch_lull: bool = True,
                 require_strong_candle: bool = True,
                 use_simple_momentum: bool = True,
                 use_better_exits: bool = True):
        
        # ML-specific settings (set before parent init)
        self.enable_ml_signals = enable_ml_signals
        self.ml_lookback_window = ml_lookback_window
        self.feature_selection_k = feature_selection_k
        self.ml_retrain_frequency = ml_retrain_frequency
        
        # Initialize parent
        super().__init__(df, volume_multiplier, breakout_threshold, profit_target_pct,
                        stop_loss_pct, max_holding_minutes, min_prev_day_range_pct, volume_lookback,
                        enable_regime_awareness, volatility_lookback, trend_lookback,
                        vix_threshold_high, vix_threshold_low, trend_threshold,
                        enable_signal_strength, strong_signal_threshold, weak_signal_threshold,
                        min_position_size, max_position_size,
                        avoid_first_hour, avoid_lunch_lull, require_strong_candle,
                        use_simple_momentum, use_better_exits)
        
        # ML models and data
        self.ml_signal_model = None
        self.ml_position_model = None
        self.feature_scaler = RobustScaler()
        self.selected_features = None
        self.ml_feature_importance = {}
        
        # Enhanced feature engineering
        if self.enable_ml_signals:
            try:
                self._add_ml_features()
                print("✅ ML features added")
            except Exception as e:
                print(f"⚠️ ML feature addition failed: {e}")
                self.enable_ml_signals = False
    
    def _add_ml_features(self):
        """Add comprehensive technical indicators for ML feature discovery."""
        print("🤖 Adding ML features...")
        
        # 1. PRICE-BASED FEATURES
        self._add_price_features()
        
        # 2. VOLUME-BASED FEATURES  
        self._add_volume_features()
        
        # 3. VOLATILITY FEATURES
        self._add_volatility_features()
        
        # 4. MOMENTUM & OSCILLATORS
        self._add_momentum_features()
        
        # 5. PATTERN RECOGNITION
        self._add_pattern_features()
        
        # 6. MARKET MICROSTRUCTURE
        self._add_microstructure_features()
        
        # 7. TIME-BASED FEATURES
        self._add_time_features()
        
        print(f"🎯 Added {self._count_ml_features()} ML features for analysis")
    
    def _add_price_features(self):
        """Add price-based technical indicators."""
        print("  📈 Adding price features...")
        
        # Moving averages (multiple timeframes)
        for period in [5, 10, 20, 50, 100, 200]:
            self.df[f'sma_{period}'] = self.df['close'].rolling(period).mean()
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period).mean()
            
            # Price relative to moving averages
            self.df[f'price_vs_sma_{period}'] = (self.df['close'] - self.df[f'sma_{period}']) / self.df[f'sma_{period}']
            self.df[f'price_vs_ema_{period}'] = (self.df['close'] - self.df[f'ema_{period}']) / self.df[f'ema_{period}']
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_std = self.df['close'].rolling(period).std()
            bb_mid = self.df['close'].rolling(period).mean()
            self.df[f'bb_upper_{period}'] = bb_mid + (2 * bb_std)
            self.df[f'bb_lower_{period}'] = bb_mid - (2 * bb_std)
            self.df[f'bb_position_{period}'] = (self.df['close'] - bb_mid) / (2 * bb_std)
            self.df[f'bb_width_{period}'] = (self.df[f'bb_upper_{period}'] - self.df[f'bb_lower_{period}']) / bb_mid
        
        # Support/Resistance levels
        for period in [10, 20, 50]:
            self.df[f'high_max_{period}'] = self.df['high'].rolling(period).max()
            self.df[f'low_min_{period}'] = self.df['low'].rolling(period).min()
            self.df[f'range_{period}'] = self.df[f'high_max_{period}'] - self.df[f'low_min_{period}']
            self.df[f'price_position_{period}'] = (self.df['close'] - self.df[f'low_min_{period}']) / self.df[f'range_{period}']
    
    def _add_volume_features(self):
        """Add volume-based indicators."""
        print("  📊 Adding volume features...")
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            self.df[f'vol_sma_{period}'] = self.df['volume'].rolling(period).mean()
            self.df[f'vol_ratio_{period}'] = self.df['volume'] / self.df[f'vol_sma_{period}']
        
        # Volume Price Trend (VPT)
        self.df['vpt'] = (self.df['volume'] * self.df['close'].pct_change()).cumsum()
        
        # On-Balance Volume (OBV)
        self.df['obv'] = (self.df['volume'] * np.where(self.df['close'] > self.df['close'].shift(1), 1, -1)).cumsum()
        
        # Accumulation/Distribution Line
        clv = ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / (self.df['high'] - self.df['low'])
        self.df['ad_line'] = (clv * self.df['volume']).cumsum()
        
        # Volume oscillators
        for short, long in [(5, 20), (10, 30)]:
            vol_short = self.df['volume'].rolling(short).mean()
            vol_long = self.df['volume'].rolling(long).mean()
            self.df[f'vol_osc_{short}_{long}'] = (vol_short - vol_long) / vol_long
    
    def _add_volatility_features(self):
        """Add volatility indicators."""
        print("  ⚡ Adding volatility features...")
        
        # True Range and ATR
        self.df['true_range'] = np.maximum(
            self.df['high'] - self.df['low'],
            np.maximum(
                abs(self.df['high'] - self.df['close'].shift(1)),
                abs(self.df['low'] - self.df['close'].shift(1))
            )
        )
        
        for period in [14, 20, 50]:
            self.df[f'atr_{period}'] = self.df['true_range'].rolling(period).mean()
            self.df[f'atr_pct_{period}'] = self.df[f'atr_{period}'] / self.df['close']
        
        # Volatility indicators
        for period in [10, 20, 50]:
            returns = self.df['close'].pct_change()
            self.df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252 * 390)
            self.df[f'vol_rank_{period}'] = self.df[f'volatility_{period}'].rolling(period * 2).rank(pct=True)
        
        # Parkinson estimator (high-low volatility)
        self.df['parkinson_vol'] = np.sqrt(0.361 * (np.log(self.df['high'] / self.df['low'])) ** 2)
    
    def _add_momentum_features(self):
        """Add momentum and oscillator indicators."""
        print("  🚀 Adding momentum features...")
        
        # Rate of Change (ROC)
        for period in [1, 5, 10, 20]:
            self.df[f'roc_{period}'] = self.df['close'].pct_change(period)
        
        # Relative Strength Index (RSI)
        for period in [14, 21, 50]:
            delta = self.df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = self.df['low'].rolling(period).min()
            high_max = self.df['high'].rolling(period).max()
            self.df[f'stoch_k_{period}'] = 100 * (self.df['close'] - low_min) / (high_max - low_min)
            self.df[f'stoch_d_{period}'] = self.df[f'stoch_k_{period}'].rolling(3).mean()
        
        # MACD
        for fast, slow in [(12, 26), (5, 15)]:
            ema_fast = self.df['close'].ewm(span=fast).mean()
            ema_slow = self.df['close'].ewm(span=slow).mean()
            self.df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
            self.df[f'macd_signal_{fast}_{slow}'] = self.df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
            self.df[f'macd_histogram_{fast}_{slow}'] = self.df[f'macd_{fast}_{slow}'] - self.df[f'macd_signal_{fast}_{slow}']
        
        # Williams %R
        for period in [14, 21]:
            high_max = self.df['high'].rolling(period).max()
            low_min = self.df['low'].rolling(period).min()
            self.df[f'williams_r_{period}'] = -100 * (high_max - self.df['close']) / (high_max - low_min)
    
    def _add_pattern_features(self):
        """Add pattern recognition features."""
        print("  🎯 Adding pattern features...")
        
        # Candle patterns
        self.df['doji'] = abs(self.df['close'] - self.df['open']) < (0.1 * (self.df['high'] - self.df['low']))
        self.df['hammer'] = ((self.df['high'] - self.df['low']) > 3 * abs(self.df['close'] - self.df['open'])) & \
                           ((self.df['close'] - self.df['low']) > 2 * (self.df['high'] - self.df['close']))
        self.df['shooting_star'] = ((self.df['high'] - self.df['low']) > 3 * abs(self.df['close'] - self.df['open'])) & \
                                   ((self.df['high'] - self.df['close']) > 2 * (self.df['close'] - self.df['low']))
        
        # Gap analysis
        self.df['gap_up'] = self.df['low'] > self.df['high'].shift(1)
        self.df['gap_down'] = self.df['high'] < self.df['low'].shift(1)
        self.df['gap_size'] = np.where(self.df['gap_up'], 
                                      self.df['low'] - self.df['high'].shift(1),
                                      np.where(self.df['gap_down'],
                                              self.df['low'].shift(1) - self.df['high'], 0))
        
        # Price action patterns
        for period in [3, 5, 10]:
            self.df[f'higher_highs_{period}'] = (self.df['high'] > self.df['high'].shift(1)).rolling(period).sum()
            self.df[f'lower_lows_{period}'] = (self.df['low'] < self.df['low'].shift(1)).rolling(period).sum()
            self.df[f'inside_bars_{period}'] = ((self.df['high'] < self.df['high'].shift(1)) & 
                                               (self.df['low'] > self.df['low'].shift(1))).rolling(period).sum()
    
    def _add_microstructure_features(self):
        """Add market microstructure features."""
        print("  🔬 Adding microstructure features...")
        
        # Bid-ask spread proxies
        self.df['spread_proxy'] = self.df['high'] - self.df['low']
        self.df['spread_pct'] = self.df['spread_proxy'] / self.df['close']
        
        # Tick direction approximation
        self.df['tick_direction'] = np.where(self.df['close'] > self.df['close'].shift(1), 1,
                                           np.where(self.df['close'] < self.df['close'].shift(1), -1, 0))
        
        # Realized spreads
        for period in [5, 10, 20]:
            self.df[f'avg_spread_{period}'] = self.df['spread_pct'].rolling(period).mean()
            self.df[f'tick_imbalance_{period}'] = self.df['tick_direction'].rolling(period).sum()
        
        # Volume-weighted features
        self.df['vwap'] = (self.df['volume'] * self.df['close']).cumsum() / self.df['volume'].cumsum()
        self.df['price_vs_vwap'] = (self.df['close'] - self.df['vwap']) / self.df['vwap']
        
        # Intraday momentum
        self.df['intraday_return'] = (self.df['close'] - self.df['open']) / self.df['open']
        self.df['overnight_return'] = (self.df['open'] - self.df['close'].shift(1)) / self.df['close'].shift(1)
    
    def _add_time_features(self):
        """Add time-based features."""
        print("  ⏰ Adding time features...")
        
        # Hour and minute features
        if hasattr(self.df.index, 'hour'):
            self.df['hour'] = self.df.index.hour
            self.df['minute'] = self.df.index.minute
            self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
            self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
            self.df['minute_sin'] = np.sin(2 * np.pi * self.df['minute'] / 60)
            self.df['minute_cos'] = np.cos(2 * np.pi * self.df['minute'] / 60)
        
        # Time since market open
        if 'session_minute' in self.df.columns:
            self.df['time_normalized'] = self.df['session_minute'] / 390  # Normalize to [0,1]
            self.df['time_squared'] = self.df['time_normalized'] ** 2
        
        # Day of week effects (if available)
        if hasattr(self.df.index, 'dayofweek'):
            self.df['day_of_week'] = self.df.index.dayofweek
            for day in range(5):  # Monday=0 to Friday=4
                self.df[f'is_day_{day}'] = (self.df['day_of_week'] == day).astype(int)
    
    def _count_ml_features(self) -> int:
        """Count the number of ML features added."""
        ml_feature_prefixes = [
            'sma_', 'ema_', 'price_vs_', 'bb_', 'high_max_', 'low_min_', 'range_', 'price_position_',
            'vol_sma_', 'vol_ratio_', 'vpt', 'obv', 'ad_line', 'vol_osc_',
            'atr_', 'volatility_', 'vol_rank_', 'parkinson_vol',
            'roc_', 'rsi_', 'stoch_', 'macd_', 'williams_r_',
            'doji', 'hammer', 'shooting_star', 'gap_', 'higher_highs_', 'lower_lows_', 'inside_bars_',
            'spread_', 'tick_', 'vwap', 'price_vs_vwap', 'intraday_return', 'overnight_return',
            'hour', 'minute', 'time_', 'day_of_week', 'is_day_'
        ]
        
        count = 0
        for col in self.df.columns:
            if any(col.startswith(prefix) for prefix in ml_feature_prefixes):
                count += 1
        return count
    
    def _prepare_ml_features(self, start_idx: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and targets for ML training."""
        if start_idx is None:
            start_idx = self.ml_lookback_window
        
        # Get all ML features
        ml_feature_prefixes = [
            'sma_', 'ema_', 'price_vs_', 'bb_', 'high_max_', 'low_min_', 'range_', 'price_position_',
            'vol_sma_', 'vol_ratio_', 'vol_osc_', 'atr_', 'volatility_', 'vol_rank_',
            'roc_', 'rsi_', 'stoch_', 'macd_', 'williams_r_',
            'spread_', 'tick_', 'price_vs_vwap', 'intraday_return', 'overnight_return',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'time_normalized', 'time_squared'
        ]
        
        # Also include original signal strength components
        original_features = [
            'volume_strength', 'magnitude_strength', 'momentum_strength', 
            'regime_favorability', 'candle_strength', 'confluence_score', 'time_favorability'
        ]
        
        # Select feature columns
        feature_cols = []
        for col in self.df.columns:
            if (any(col.startswith(prefix) for prefix in ml_feature_prefixes) or 
                col in original_features or
                col in ['doji', 'hammer', 'shooting_star', 'gap_up', 'gap_down', 'vpt', 'obv', 'ad_line', 'parkinson_vol']):
                feature_cols.append(col)
        
        # Prepare feature matrix
        features_df = self.df[feature_cols].iloc[start_idx:].copy()
        
        # Remove infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Create target: future return (next 5 bars average)
        future_returns = []
        for i in range(start_idx, len(self.df)):
            if i + 5 < len(self.df):
                future_ret = (self.df['close'].iloc[i+1:i+6].mean() - self.df['close'].iloc[i]) / self.df['close'].iloc[i]
                future_returns.append(future_ret)
            else:
                future_returns.append(0)
        
        target = pd.Series(future_returns, index=features_df.index)
        
        return features_df, target
    
    def _train_ml_models(self, start_idx: int = None):
        """Train ML models for signal enhancement and position sizing."""
        print("🤖 Training ML models...")
        
        # Prepare data
        features, target = self._prepare_ml_features(start_idx)
        
        if len(features) < 100:  # Need minimum data
            print("⚠️ Insufficient data for ML training")
            return
        
        # Feature selection
        print("  🎯 Selecting best features...")
        selector = SelectKBest(score_func=f_regression, k=min(self.feature_selection_k, len(features.columns)))
        features_selected = selector.fit_transform(features, target)
        selected_feature_names = features.columns[selector.get_support()].tolist()
        self.selected_features = selected_feature_names
        
        print(f"  ✅ Selected {len(selected_feature_names)} features")
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features_selected)
        
        # Train signal strength model
        print("  🎯 Training signal strength model...")
        self.ml_signal_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.ml_signal_model, features_scaled, target, cv=tscv, scoring='r2')
        
        self.ml_signal_model.fit(features_scaled, target)
        
        print(f"  ✅ Signal model trained (CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
        
        # Get feature importance
        if len(selected_feature_names) == len(self.ml_signal_model.feature_importances_):
            self.ml_feature_importance = dict(zip(selected_feature_names, self.ml_signal_model.feature_importances_))
            
            # Print top features
            top_features = sorted(self.ml_feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print("  🏆 Top 10 ML features:")
            for feature, importance in top_features:
                print(f"    {feature}: {importance:.3f}")
    
    def _get_ml_signal_strength(self, idx: int) -> float:
        """Get ML-enhanced signal strength for a specific index."""
        if (self.ml_signal_model is None or self.selected_features is None or 
            idx < self.ml_lookback_window):
            return 0.5  # Default neutral score
        
        try:
            # Get features for current bar
            feature_row = self.df[self.selected_features].iloc[idx:idx+1]
            feature_row = feature_row.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Scale features
            feature_scaled = self.feature_scaler.transform(feature_row)
            
            # Predict signal strength
            ml_prediction = self.ml_signal_model.predict(feature_scaled)[0]
            
            # Convert prediction to 0-1 range (sigmoid-like transformation)
            ml_signal_strength = 1 / (1 + np.exp(-ml_prediction * 10))
            
            return np.clip(ml_signal_strength, 0, 1)
            
        except Exception as e:
            print(f"⚠️ ML prediction error at index {idx}: {e}")
            return 0.5
    
    def _calculate_enhanced_signal_strength_score(self):
        """Calculate enhanced signal strength score combining original + ML signals."""
        if not self.enable_ml_signals:
            return super()._calculate_signal_strength_score()
        
        print("  🤖 Calculating ML-enhanced signal strength...")
        
        # First calculate original signal strength components
        super()._calculate_signal_strength_score()
        
        # Train ML models on sufficient data
        if len(self.df) >= self.ml_lookback_window:
            self._train_ml_models()
        
        # Enhance signal strength with ML
        ml_enhanced_scores = []
        
        for i in range(len(self.df)):
            if i >= self.ml_lookback_window and self.ml_signal_model is not None:
                # Get original signal strength
                original_strength = self.df['signal_strength'].iloc[i]
                
                # Get ML signal strength
                ml_strength = self._get_ml_signal_strength(i)
                
                # Combine original and ML signals (weighted average)
                combined_strength = 0.6 * original_strength + 0.4 * ml_strength
                
                ml_enhanced_scores.append(combined_strength)
            else:
                # Use original signal strength for early bars
                ml_enhanced_scores.append(self.df['signal_strength'].iloc[i])
        
        # Update signal strength with ML enhancement
        self.df['signal_strength'] = ml_enhanced_scores
        self.df['ml_signal_component'] = [self._get_ml_signal_strength(i) if i >= self.ml_lookback_window 
                                         else 0.5 for i in range(len(self.df))]
        
        # Recalculate signal levels and position sizes
        self.df['signal_level'] = 'Weak'
        strong_mask = self.df['signal_strength'] >= self.strong_signal_threshold
        medium_mask = (self.df['signal_strength'] >= 0.6) & (self.df['signal_strength'] < self.strong_signal_threshold)
        
        self.df.loc[strong_mask, 'signal_level'] = 'Strong'
        self.df.loc[medium_mask, 'signal_level'] = 'Medium'
        
        # Recalculate dynamic position sizes
        self._calculate_position_sizes()
    
    def _calculate_signal_strength_score(self):
        """Override to use ML-enhanced scoring."""
        if self.enable_ml_signals:
            self._calculate_enhanced_signal_strength_score()
        else:
            super()._calculate_signal_strength_score()
    
    def get_ml_feature_summary(self) -> Dict:
        """Get summary of ML feature importance and model performance."""
        if not self.enable_ml_signals or self.ml_signal_model is None:
            return {"status": "ML not enabled or trained"}
        
        summary = {
            "ml_enabled": self.enable_ml_signals,
            "model_trained": self.ml_signal_model is not None,
            "selected_features_count": len(self.selected_features) if self.selected_features else 0,
            "feature_importance": self.ml_feature_importance,
            "top_features": sorted(self.ml_feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)[:10] if self.ml_feature_importance else []
        }
        
        return summary
    
    def backtest_strategy(self, initial_capital=100000):
        """Override to use ML-enhanced strategy."""
        print("🤖 Running ML-Enhanced Signal Strength Strategy backtest...")
        
        if self.enable_ml_signals:
            # Use ML-enhanced signal strength detection
            entries, exits = self.build_signal_strength_signals()
            
            # Run backtest with ML enhancement tracking
            trades_df = self._backtest_with_ml_tracking(initial_capital, entries, exits)
            
            if len(trades_df) > 0:
                self.trades_df = trades_df
                self._calculate_ml_enhanced_performance_metrics(initial_capital)
            else:
                print("⚠️ No trades executed!")
                self.trades_df = pd.DataFrame()
                
            return trades_df
        else:
            return super().backtest_strategy(initial_capital)
    
    def _backtest_with_ml_tracking(self, initial_capital: float, entries: pd.Series, exits: pd.Series) -> pd.DataFrame:
        """Run backtest with ML enhancement tracking."""
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        entry_regime = None
        entry_signal_strength = 0
        entry_ml_component = 0
        entry_position_size = 0
        capital = initial_capital
        
        for i in range(len(self.df)):
            current_bar = self.df.iloc[i]
            current_time = self.df.index[i]
            current_price = current_bar['close']
            current_regime = current_bar.get('market_regime', 'normal')
            
            # Enter position
            if entries.iloc[i] and position == 0:
                position = 1
                entry_price = current_price
                entry_time = current_time
                entry_regime = current_regime
                entry_signal_strength = current_bar['signal_strength']
                entry_ml_component = current_bar.get('ml_signal_component', 0.5)
                entry_position_size = current_bar['position_size']
                
            # Exit position
            elif exits.iloc[i] and position == 1:
                exit_price = current_price
                return_pct = (exit_price - entry_price) / entry_price
                return_dollars = capital * return_pct * entry_position_size
                
                minutes_held = (current_time - entry_time).total_seconds() / 60
                adjusted_params = self.get_regime_adjusted_parameters(current_regime)
                
                # Enhanced exit reason detection
                exit_reason = 'Unknown'
                if return_pct >= adjusted_params['profit_target_pct']:
                    exit_reason = 'Profit Target'
                elif return_pct <= -adjusted_params['stop_loss_pct']:
                    exit_reason = 'Stop Loss'
                elif minutes_held >= self.max_holding_minutes:
                    exit_reason = 'Time Exit'
                elif entry_regime != current_regime:
                    exit_reason = 'Regime Change'
                else:
                    exit_reason = 'Adaptive Exit'
                
                # Enhanced trade record with ML tracking
                trade = {
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': return_pct,
                    'return_dollars': return_dollars,
                    'holding_minutes': minutes_held,
                    'exit_reason': exit_reason,
                    'entry_regime': entry_regime,
                    'exit_regime': current_regime,
                    'signal_strength': entry_signal_strength,
                    'ml_component': entry_ml_component,
                    'position_size': entry_position_size,
                    'signal_level': 'Strong' if entry_signal_strength >= self.strong_signal_threshold else 'Medium' if entry_signal_strength >= 0.6 else 'Weak',
                    'ml_enhanced': entry_ml_component != 0.5  # Track if ML was used
                }
                trades.append(trade)
                
                capital += return_dollars
                position = 0
                entry_price = 0
                entry_time = None
                entry_regime = None
                entry_signal_strength = 0
                entry_ml_component = 0
                entry_position_size = 0
        
        return pd.DataFrame(trades)
    
    def _calculate_ml_enhanced_performance_metrics(self, initial_capital):
        """Calculate performance metrics with ML enhancement analysis."""
        if self.trades_df.empty:
            return
        
        # Call parent method first
        super()._calculate_signal_strength_performance_metrics(initial_capital)
        
        # Add ML-specific analysis
        trades = self.trades_df
        
        # ML enhancement analysis
        ml_enhanced_trades = trades[trades['ml_enhanced'] == True]
        non_ml_trades = trades[trades['ml_enhanced'] == False]
        
        print("\n" + "="*70)
        print("🤖 ML ENHANCEMENT ANALYSIS")
        print("="*70)
        
        if len(ml_enhanced_trades) > 0:
            print(f"ML-Enhanced Trades: {len(ml_enhanced_trades)}")
            print(f"  Win Rate: {(ml_enhanced_trades['return_pct'] > 0).mean() * 100:.1f}%")
            print(f"  Avg Return: {ml_enhanced_trades['return_pct'].mean() * 100:.2f}%")
            print(f"  Avg ML Component: {ml_enhanced_trades['ml_component'].mean():.3f}")
        
        if len(non_ml_trades) > 0:
            print(f"\nNon-ML Trades: {len(non_ml_trades)}")
            print(f"  Win Rate: {(non_ml_trades['return_pct'] > 0).mean() * 100:.1f}%")
            print(f"  Avg Return: {non_ml_trades['return_pct'].mean() * 100:.2f}%")
        
        # Feature importance summary
        if self.ml_feature_importance:
            print(f"\n🏆 Top 5 ML Features:")
            top_features = sorted(self.ml_feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in top_features:
                print(f"  {feature}: {importance:.3f}")
        
        print("="*70)

def test_ml_enhanced_strategy():
    """Test ML-enhanced signal strength strategy."""
    from day_trading_o3 import DataLoader
    from signal_strength_position_sizing import SignalStrengthPositionSizing
    from datetime import datetime, timedelta
    
    print("🤖 Testing ML-ENHANCED SIGNAL STRENGTH Strategy")
    print("=" * 80)
    
    # Get more data for ML training
    data_loader = DataLoader()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')  # More data for ML
    
    df = data_loader.get_price_data('AAPL', start_date, end_date)
    
    if df.empty:
        print("❌ No data available")
        return
    
    print(f"📊 Testing with {len(df)} bars of AAPL data ({60} days)")
    
    # Test baseline signal strength strategy
    print("\n💪 BASELINE SIGNAL STRENGTH STRATEGY:")
    baseline_strategy = SignalStrengthPositionSizing(
        df.copy(),
        enable_signal_strength=True,
        enable_regime_awareness=True,
        strong_signal_threshold=0.70,
        weak_signal_threshold=0.35
    )
    baseline_trades = baseline_strategy.backtest_strategy(100000)
    
    # Test ML-enhanced strategy
    print("\n🤖 ML-ENHANCED SIGNAL STRENGTH STRATEGY:")
    ml_strategy = MLEnhancedSignalStrength(
        df.copy(),
        enable_ml_signals=True,
        enable_signal_strength=True,
        enable_regime_awareness=True,
        strong_signal_threshold=0.70,
        weak_signal_threshold=0.35,
        ml_lookback_window=1000,
        feature_selection_k=15
    )
    ml_trades = ml_strategy.backtest_strategy(100000)
    
    # Compare results
    print("\n" + "="*80)
    print("🤖 ML ENHANCEMENT COMPARISON")
    print("="*80)
    
    if not baseline_trades.empty or not ml_trades.empty:
        baseline_metrics = baseline_strategy.performance_metrics if not baseline_trades.empty else {}
        ml_metrics = ml_strategy.performance_metrics if not ml_trades.empty else {}
        
        print(f"{'Metric':<25} {'Baseline':<15} {'ML-Enhanced':<15} {'Improvement':<15}")
        print("-" * 80)
        
        metrics = [
            ('Total Trades', 'total_trades'),
            ('Win Rate (%)', 'win_rate'),
            ('Total Return (%)', 'total_return_pct'),
            ('Avg Return (%)', 'avg_return_pct'),
            ('Profit Factor', 'profit_factor'),
            ('Max Gain (%)', 'max_gain'),
            ('Max Loss (%)', 'max_loss'),
            ('Avg Hold (min)', 'avg_holding_minutes')
        ]
        
        for label, key in metrics:
            baseline_val = baseline_metrics.get(key, 0)
            ml_val = ml_metrics.get(key, 0)
            
            if key == 'total_trades':
                improvement = f"+{ml_val - baseline_val}" if ml_val > baseline_val else f"{ml_val - baseline_val}"
            else:
                improvement = f"+{ml_val - baseline_val:.2f}" if ml_val > baseline_val else f"{ml_val - baseline_val:.2f}"
            
            print(f"{label:<25} {baseline_val:<15.2f} {ml_val:<15.2f} {improvement:<15}")
        
        print("="*80)
        
        # Show ML insights
        ml_summary = ml_strategy.get_ml_feature_summary()
        if ml_summary.get('model_trained'):
            print(f"🤖 ML Model trained with {ml_summary['selected_features_count']} features")
            print("🏆 Top ML Features:")
            for feature, importance in ml_summary['top_features'][:5]:
                print(f"   {feature}: {importance:.3f}")
    
    return ml_strategy

if __name__ == "__main__":
    strategy = test_ml_enhanced_strategy()