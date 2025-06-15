#!/usr/bin/env python3
"""
Simple ML Enhancement for Signal Strength
=========================================
Focused ML enhancement that adds key technical indicators and optimizes signal weights.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

class SimpleMlEnhancement(SignalStrengthPositionSizing):
    """
    Simple ML enhancement that adds proven technical indicators and optimizes weights.
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 # All base parameters
                 **kwargs):
        
        # Initialize parent first
        super().__init__(df, **kwargs)
        
        # Add ML features and retrain signal scoring
        self._add_key_ml_features()
        self._optimize_signal_weights()
        
        print("✅ Simple ML enhancement complete")
    
    def _add_key_ml_features(self):
        """Add key proven technical indicators."""
        print("🤖 Adding key ML features...")
        
        # 1. RSI (Relative Strength Index)
        self._add_rsi()
        
        # 2. MACD
        self._add_macd()
        
        # 3. Bollinger Bands
        self._add_bollinger_bands()
        
        # 4. Additional Moving Averages
        self._add_additional_mas()
        
        # 5. Volume Indicators
        self._add_volume_indicators()
        
        print("✅ Added key ML features")
    
    def _add_rsi(self):
        """Add RSI indicator."""
        for period in [14, 21]:
            delta = self.df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # RSI signals
            self.df[f'rsi_{period}_oversold'] = self.df[f'rsi_{period}'] < 30
            self.df[f'rsi_{period}_overbought'] = self.df[f'rsi_{period}'] > 70
    
    def _add_macd(self):
        """Add MACD indicator."""
        # Standard MACD (12, 26, 9)
        ema_12 = self.df['close'].ewm(span=12).mean()
        ema_26 = self.df['close'].ewm(span=26).mean()
        
        self.df['macd'] = ema_12 - ema_26
        self.df['macd_signal'] = self.df['macd'].ewm(span=9).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        
        # MACD signals
        self.df['macd_bullish'] = (self.df['macd'] > self.df['macd_signal']) & \
                                 (self.df['macd'].shift(1) <= self.df['macd_signal'].shift(1))
        self.df['macd_bearish'] = (self.df['macd'] < self.df['macd_signal']) & \
                                 (self.df['macd'].shift(1) >= self.df['macd_signal'].shift(1))
    
    def _add_bollinger_bands(self):
        """Add Bollinger Bands."""
        for period in [20, 50]:
            bb_middle = self.df['close'].rolling(period).mean()
            bb_std = self.df['close'].rolling(period).std()
            
            self.df[f'bb_upper_{period}'] = bb_middle + (2 * bb_std)
            self.df[f'bb_lower_{period}'] = bb_middle - (2 * bb_std)
            self.df[f'bb_width_{period}'] = (self.df[f'bb_upper_{period}'] - self.df[f'bb_lower_{period}']) / bb_middle
            
            # BB position (0 = lower band, 1 = upper band)
            self.df[f'bb_position_{period}'] = (self.df['close'] - self.df[f'bb_lower_{period}']) / \
                                              (self.df[f'bb_upper_{period}'] - self.df[f'bb_lower_{period}'])
            
            # BB signals
            self.df[f'bb_squeeze_{period}'] = self.df[f'bb_width_{period}'] < self.df[f'bb_width_{period}'].rolling(50).quantile(0.2)
    
    def _add_additional_mas(self):
        """Add additional moving averages."""
        for period in [10, 50, 200]:
            self.df[f'sma_{period}'] = self.df['close'].rolling(period).mean()
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period).mean()
            
            # Price relative to MA
            self.df[f'price_vs_sma_{period}'] = (self.df['close'] - self.df[f'sma_{period}']) / self.df[f'sma_{period}']
            self.df[f'price_vs_ema_{period}'] = (self.df['close'] - self.df[f'ema_{period}']) / self.df[f'ema_{period}']
            
            # MA slope
            self.df[f'sma_{period}_slope'] = self.df[f'sma_{period}'].diff(5) / self.df[f'sma_{period}']
    
    def _add_volume_indicators(self):
        """Add volume-based indicators."""
        # On-Balance Volume (OBV)
        self.df['obv'] = (self.df['volume'] * np.where(self.df['close'] > self.df['close'].shift(1), 1, -1)).cumsum()
        
        # Volume Rate of Change
        self.df['volume_roc'] = self.df['volume'].pct_change(10)
        
        # Volume moving averages
        for period in [10, 20]:
            self.df[f'volume_sma_{period}'] = self.df['volume'].rolling(period).mean()
            self.df[f'volume_ratio_{period}'] = self.df['volume'] / self.df[f'volume_sma_{period}']
    
    def _optimize_signal_weights(self):
        """Use ML to optimize signal component weights."""
        print("🎯 Optimizing signal weights with ML...")
        
        # Prepare features for ML optimization
        features = self._prepare_ml_features()
        targets = self._prepare_targets()
        
        if len(features) < 500:  # Need sufficient data
            print("⚠️ Insufficient data for ML optimization")
            return
        
        # Train Random Forest to predict future returns
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        
        # Use time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Get feature importance
        rf.fit(features, targets)
        feature_importance = dict(zip(features.columns, rf.feature_importances_))
        
        # Update signal weights based on feature importance
        self._update_signal_weights(feature_importance)
        
        # Recalculate signal strength with optimized weights
        self._recalculate_optimized_signal_strength()
        
        print("✅ Signal weights optimized")
    
    def _prepare_ml_features(self) -> pd.DataFrame:
        """Prepare features for ML optimization."""
        feature_columns = [
            # Original signal components
            'volume_strength', 'magnitude_strength', 'momentum_strength',
            'regime_favorability', 'candle_strength', 'confluence_score', 'time_favorability',
            
            # New ML features
            'rsi_14', 'rsi_21', 'macd', 'macd_histogram',
            'bb_position_20', 'bb_width_20', 'bb_position_50',
            'price_vs_sma_10', 'price_vs_sma_50', 'price_vs_ema_10', 'price_vs_ema_50',
            'sma_10_slope', 'sma_50_slope', 'volume_ratio_10', 'volume_ratio_20'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        # Prepare feature matrix
        features_df = self.df[available_features].copy()
        features_df = features_df.fillna(method='ffill').fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Remove early rows with insufficient data
        return features_df.iloc[200:]
    
    def _prepare_targets(self) -> pd.Series:
        """Prepare target variable (future returns)."""
        # Future return over next 5 bars
        future_returns = []
        
        for i in range(200, len(self.df)):
            if i + 5 < len(self.df):
                future_ret = (self.df['close'].iloc[i+5] - self.df['close'].iloc[i]) / self.df['close'].iloc[i]
                future_returns.append(future_ret)
            else:
                future_returns.append(0)
        
        return pd.Series(future_returns)
    
    def _update_signal_weights(self, feature_importance: Dict):
        """Update signal weights based on ML feature importance."""
        
        # Map features to original signal components
        component_importance = {
            'volume_strength': feature_importance.get('volume_strength', 0) + \
                              feature_importance.get('volume_ratio_10', 0) + \
                              feature_importance.get('volume_ratio_20', 0),
            
            'magnitude_strength': feature_importance.get('magnitude_strength', 0),
            
            'momentum_strength': feature_importance.get('momentum_strength', 0) + \
                               feature_importance.get('macd', 0) + \
                               feature_importance.get('macd_histogram', 0) + \
                               feature_importance.get('sma_10_slope', 0),
            
            'regime_favorability': feature_importance.get('regime_favorability', 0),
            
            'candle_strength': feature_importance.get('candle_strength', 0),
            
            'confluence_score': feature_importance.get('confluence_score', 0) + \
                              feature_importance.get('rsi_14', 0) + \
                              feature_importance.get('bb_position_20', 0),
            
            'time_favorability': feature_importance.get('time_favorability', 0)
        }
        
        # Normalize to sum to 1.0
        total_importance = sum(component_importance.values())
        if total_importance > 0:
            self.optimized_weights = {k: v/total_importance for k, v in component_importance.items()}
        else:
            # Fallback to original weights
            self.optimized_weights = {
                'volume_strength': 0.25,
                'magnitude_strength': 0.20,
                'momentum_strength': 0.15,
                'regime_favorability': 0.15,
                'candle_strength': 0.10,
                'confluence_score': 0.10,
                'time_favorability': 0.05
            }
        
        print(f"🎯 Optimized weights:")
        for component, weight in self.optimized_weights.items():
            print(f"  {component}: {weight:.3f}")
    
    def _recalculate_optimized_signal_strength(self):
        """Recalculate signal strength with optimized weights."""
        print("  🔄 Recalculating signal strength with optimized weights...")
        
        # Calculate new signal strength with optimized weights
        optimized_signal_strength = (
            self.optimized_weights['volume_strength'] * self.df['volume_strength'].fillna(0.5) +
            self.optimized_weights['magnitude_strength'] * self.df['magnitude_strength'].fillna(0.5) +
            self.optimized_weights['momentum_strength'] * self.df['momentum_strength'].fillna(0.5) +
            self.optimized_weights['regime_favorability'] * self.df['regime_favorability'].fillna(0.5) +
            self.optimized_weights['candle_strength'] * self.df['candle_strength'].fillna(0.5) +
            self.optimized_weights['confluence_score'] * self.df['confluence_score'].fillna(0.5) +
            self.optimized_weights['time_favorability'] * self.df['time_favorability'].fillna(0.5)
        )
        
        # Add ML feature contributions
        if 'rsi_14' in self.df.columns:
            # RSI contribution (oversold/overbought signals)
            rsi_contribution = np.where(
                (self.df['rsi_14'] < 30) | (self.df['rsi_14'] > 70),
                0.1, 0  # Boost signal strength for extreme RSI
            )
            optimized_signal_strength += rsi_contribution
        
        if 'macd_bullish' in self.df.columns:
            # MACD contribution
            macd_contribution = np.where(
                self.df['macd_bullish'] | self.df['macd_bearish'],
                0.05, 0  # Boost for MACD signals
            )
            optimized_signal_strength += macd_contribution
        
        if 'bb_squeeze_20' in self.df.columns:
            # Bollinger Band squeeze contribution
            bb_contribution = np.where(
                self.df['bb_squeeze_20'],
                0.05, 0  # Boost for BB squeeze (volatility expansion expected)
            )
            optimized_signal_strength += bb_contribution
        
        # Ensure 0-1 range
        self.df['signal_strength'] = np.clip(optimized_signal_strength, 0, 1)
        
        # Update signal levels and position sizes
        self.df['signal_level'] = 'Weak'
        strong_mask = self.df['signal_strength'] >= self.strong_signal_threshold
        medium_mask = (self.df['signal_strength'] >= 0.6) & (self.df['signal_strength'] < self.strong_signal_threshold)
        
        self.df.loc[strong_mask, 'signal_level'] = 'Strong'
        self.df.loc[medium_mask, 'signal_level'] = 'Medium'
        
        # Recalculate position sizes
        self._calculate_position_sizes()
    
    def get_ml_summary(self) -> Dict:
        """Get summary of ML enhancements."""
        summary = {
            'ml_features_added': True,
            'optimized_weights': getattr(self, 'optimized_weights', {}),
            'new_features': [
                'rsi_14', 'rsi_21', 'macd', 'macd_histogram',
                'bb_position_20', 'bb_width_20', 'volume_ratio_10'
            ],
            'weight_optimization': 'RandomForest feature importance'
        }
        
        return summary

def test_simple_ml_enhancement():
    """Test simple ML enhancement strategy."""
    from signal_strength_position_sizing import SignalStrengthPositionSizing
    from datetime import datetime, timedelta
    
    print("🤖 Testing SIMPLE ML ENHANCEMENT Strategy")
    print("=" * 70)
    
    # Get data
    data_loader = DataLoader()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    
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
        strong_signal_threshold=0.70,
        weak_signal_threshold=0.35
    )
    baseline_trades = baseline_strategy.backtest_strategy(100000)
    
    # Test ML-enhanced strategy  
    print("\n🤖 SIMPLE ML ENHANCED STRATEGY:")
    ml_strategy = SimpleMlEnhancement(
        df.copy(),
        enable_signal_strength=True,
        strong_signal_threshold=0.70,
        weak_signal_threshold=0.35
    )
    ml_trades = ml_strategy.backtest_strategy(100000)
    
    # Compare results
    print("\n" + "="*70)
    print("🤖 SIMPLE ML ENHANCEMENT COMPARISON")
    print("="*70)
    
    if not baseline_trades.empty or not ml_trades.empty:
        baseline_metrics = baseline_strategy.performance_metrics if not baseline_trades.empty else {}
        ml_metrics = ml_strategy.performance_metrics if not ml_trades.empty else {}
        
        print(f"{'Metric':<25} {'Baseline':<15} {'ML Enhanced':<15} {'Improvement':<15}")
        print("-" * 70)
        
        metrics = [
            ('Total Trades', 'total_trades'),
            ('Win Rate (%)', 'win_rate'),
            ('Total Return (%)', 'total_return_pct'),
            ('Avg Return (%)', 'avg_return_pct'),
            ('Profit Factor', 'profit_factor'),
            ('Max Gain (%)', 'max_gain'),
            ('Max Loss (%)', 'max_loss')
        ]
        
        for label, key in metrics:
            baseline_val = baseline_metrics.get(key, 0)
            ml_val = ml_metrics.get(key, 0)
            
            if key == 'total_trades':
                improvement = f"+{ml_val - baseline_val}" if ml_val > baseline_val else f"{ml_val - baseline_val}"
            else:
                improvement = f"+{ml_val - baseline_val:.2f}" if ml_val > baseline_val else f"{ml_val - baseline_val:.2f}"
            
            print(f"{label:<25} {baseline_val:<15.2f} {ml_val:<15.2f} {improvement:<15}")
        
        print("="*70)
        
        # Show ML insights
        ml_summary = ml_strategy.get_ml_summary()
        if ml_summary.get('optimized_weights'):
            print("🎯 ML Optimized Weights:")
            for component, weight in ml_summary['optimized_weights'].items():
                print(f"   {component}: {weight:.3f}")
    
    return ml_strategy

def compare_multiple_ml_strategies():
    """Compare multiple ML enhancement approaches."""
    print("🔬 MULTIPLE ML STRATEGIES COMPARISON")
    print("=" * 80)
    
    symbols = ['AAPL', 'MSFT', 'TSLA']
    results = {}
    
    data_loader = DataLoader()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')
    
    for symbol in symbols:
        print(f"\n📈 Testing {symbol}...")
        
        df = data_loader.get_price_data(symbol, start_date, end_date)
        
        if df.empty or len(df) < 5000:
            print(f"❌ Insufficient data for {symbol}")
            continue
        
        # Baseline
        baseline = SignalStrengthPositionSizing(df.copy(), enable_signal_strength=True)
        baseline_trades = baseline.backtest_strategy(100000)
        baseline_return = baseline.performance_metrics.get('total_return_pct', 0) if not baseline_trades.empty else 0
        
        # ML Enhanced
        ml_enhanced = SimpleMlEnhancement(df.copy(), enable_signal_strength=True)
        ml_trades = ml_enhanced.backtest_strategy(100000)
        ml_return = ml_enhanced.performance_metrics.get('total_return_pct', 0) if not ml_trades.empty else 0
        
        results[symbol] = {
            'baseline_return': baseline_return,
            'ml_return': ml_return,
            'improvement': ml_return - baseline_return,
            'baseline_trades': baseline.performance_metrics.get('total_trades', 0) if not baseline_trades.empty else 0,
            'ml_trades': ml_enhanced.performance_metrics.get('total_trades', 0) if not ml_trades.empty else 0
        }
        
        print(f"  Baseline: {baseline_return:.2f}%, ML: {ml_return:.2f}%, Improvement: {ml_return - baseline_return:+.2f}%")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    improvements = [r['improvement'] for r in results.values()]
    positive_improvements = sum(1 for imp in improvements if imp > 0)
    
    print(f"Symbols with positive improvement: {positive_improvements}/{len(results)}")
    print(f"Average improvement: {np.mean(improvements):+.2f}%")
    print(f"Total improvement range: {min(improvements):+.2f}% to {max(improvements):+.2f}%")
    
    return results

if __name__ == "__main__":
    print("Choose test:")
    print("1. Single symbol ML enhancement test")
    print("2. Multiple symbols comparison")
    
    # Run single symbol test
    print("\n1️⃣ Running single symbol test...")
    strategy = test_simple_ml_enhancement()
    
    print("\n2️⃣ Running multiple symbols comparison...")
    comparison = compare_multiple_ml_strategies()