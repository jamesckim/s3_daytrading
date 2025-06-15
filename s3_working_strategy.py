#!/usr/bin/env python3
"""
S3 Working Strategy - Optimized version that doesn't hang
========================================================
Streamlined implementation focusing on the core transformer functionality
without the complex inheritance chain that causes issues.
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
warnings.filterwarnings('ignore')

# Import only what we need
from indicator_transformer import AIIndicatorSelector
from day_trading_o3 import DataLoader


class S3WorkingStrategy:
    """
    Working S3 strategy with transformer-based indicator selection.
    Optimized to avoid hanging issues.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,
                 top_k_indicators: int = 15,
                 selection_interval: int = 50):  # Reduced from 100
        """Initialize S3 working strategy."""
        
        self.symbol = symbol
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.bars_since_selection = 0
        
        # Initialize transformer (lightweight)
        if adaptive_indicators:
            self.indicator_selector = AIIndicatorSelector(device='cpu')
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
        else:
            self.current_indicators = {}
            self.last_market_regime = "Unknown"
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        
        # Setup logging
        self.logger = logging.getLogger(f'S3Working_{symbol}')
        
        print(f"🚀 S3 Working Strategy initialized for {symbol}")
        print(f"   Adaptive indicators: {adaptive_indicators}")
        print(f"   Top K indicators: {top_k_indicators}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features efficiently."""
        
        # Always include basic features
        features = pd.DataFrame(index=data.index)
        features['returns'] = data['close'].pct_change()
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['price_momentum'] = data['close'].pct_change(5)
        features['volatility'] = features['returns'].rolling(10).std()
        
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
            features['sma_20'] = data['close'].rolling(20).mean()
            features['rsi'] = self._compute_rsi(data['close'], 14)
        
        # Clean features
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _select_indicators_fast(self, data: pd.DataFrame):
        """Fast indicator selection with reduced complexity."""
        try:
            # Use smaller data sample for speed
            sample_data = data.iloc[-min(500, len(data)):].copy()
            
            # Quick selection
            selection = self.indicator_selector.select_indicators(
                sample_data, 
                top_k=self.top_k_indicators
            )
            
            self.current_indicators = selection['selected_indicators']
            self.last_market_regime = selection['regime_detection']
            
            print(f"🎯 Selected {len(self.current_indicators)} indicators, "
                  f"regime: {self.last_market_regime}")
            
        except Exception as e:
            print(f"⚠️ Indicator selection failed: {e}")
            # Fall back to basic indicators
            self.current_indicators = {}
    
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
                if indicator_name.startswith('SMA'):
                    period = int(indicator_name.split('_')[1])
                    features[f'sma_{period}'] = data['close'].rolling(period).mean()
                elif indicator_name.startswith('EMA'):
                    period = int(indicator_name.split('_')[1])
                    features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
                elif indicator_name == 'RSI_14':
                    features['rsi'] = self._compute_rsi(data['close'], 14)
                elif indicator_name == 'MACD':
                    features['macd'] = self._compute_macd(data['close'])
                elif indicator_name == 'ATR_14':
                    features['atr'] = self._compute_atr(data)
                # Add more as needed
                
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
    
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on regime and features."""
        signals = pd.Series(0, index=features.index)
        
        # Basic momentum signals
        if 'returns' in features.columns:
            momentum = features['returns'].rolling(5).mean()
            
            # Regime-adjusted thresholds
            if "Trend" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.0003, -0.0003
            elif "Volatility" in self.last_market_regime:
                buy_thresh, sell_thresh = 0.001, -0.001
            else:
                buy_thresh, sell_thresh = 0.0005, -0.0005
            
            # Generate signals
            signals[momentum > buy_thresh] = 1
            signals[momentum < sell_thresh] = -1
        
        # Volume filter
        if 'volume_ratio' in features.columns:
            high_volume = features['volume_ratio'] > 1.2
            signals = signals * high_volume.astype(int)
        
        # RSI filter if available
        if 'rsi' in features.columns:
            overbought = features['rsi'] > 70
            oversold = features['rsi'] < 30
            signals[(signals == 1) & overbought] = 0  # Don't buy when overbought
            signals[(signals == -1) & oversold] = 0   # Don't sell when oversold
        
        return signals
    
    def backtest(self, data: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """Run efficient backtest."""
        
        if verbose:
            print(f"🚀 Running backtest on {len(data)} bars...")
        
        # Prepare features
        features = self.prepare_features(data)
        if verbose:
            print(f"📊 Prepared {features.shape[1]} features")
        
        # Generate signals
        signals = self.generate_signals(features)
        active_signals = (signals != 0).sum()
        if verbose:
            print(f"🎯 Generated {active_signals} active signals")
        
        # Simple backtest simulation
        trades = []
        position = 0
        entry_price = 0
        capital = self.capital
        
        for i in range(50, len(data)):  # Skip first 50 for warmup
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Close position
            if position != 0 and signal != position:
                pnl = (current_price - entry_price) * abs(position)
                if position < 0:  # Short
                    pnl = -pnl
                
                trades.append({
                    'entry': entry_price,
                    'exit': current_price,
                    'pnl': pnl,
                    'regime': self.last_market_regime
                })
                capital += pnl
                position = 0
            
            # Open position
            if signal != 0 and position == 0:
                shares = int(capital * 0.1 / current_price)  # 10% of capital
                if shares > 0:
                    position = shares * signal
                    entry_price = current_price
        
        # Calculate metrics
        if trades:
            total_pnl = sum(t['pnl'] for t in trades)
            wins = [t for t in trades if t['pnl'] > 0]
            win_rate = len(wins) / len(trades)
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            losses = [t for t in trades if t['pnl'] <= 0]
            avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            total_pnl = win_rate = avg_win = avg_loss = profit_factor = 0
        
        results = {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'final_capital': capital,
            'regime': self.last_market_regime,
            'features_used': features.shape[1],
            'signals_generated': active_signals,
            'indicators_selected': len(self.current_indicators) if self.adaptive_indicators else 0
        }
        
        if verbose:
            print(f"📈 Results: {len(trades)} trades, {win_rate:.1%} win rate, "
                  f"${total_pnl:.2f} PnL")
        
        return results


def test_s3_working():
    """Test the working S3 strategy."""
    print("🧪 Testing S3 Working Strategy")
    print("=" * 40)
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    # Realistic price movement
    returns = np.random.normal(0, 0.001, len(dates))
    returns[300:600] += 0.0001  # Trend period
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(1000, 5000, len(dates))
    }, index=dates)
    
    print(f"📊 Created {len(data)} bars of test data")
    
    # Test with transformer
    print("\n🤖 Testing with transformer (adaptive_indicators=True)...")
    strategy_ai = S3WorkingStrategy(
        symbol='TEST',
        capital=100000,
        adaptive_indicators=True,
        top_k_indicators=10
    )
    
    results_ai = strategy_ai.backtest(data, verbose=True)
    
    # Test without transformer
    print("\n📈 Testing without transformer (adaptive_indicators=False)...")
    strategy_basic = S3WorkingStrategy(
        symbol='TEST',
        capital=100000,
        adaptive_indicators=False
    )
    
    results_basic = strategy_basic.backtest(data, verbose=True)
    
    # Compare results
    print(f"\n📊 Comparison:")
    print(f"   AI Strategy:    {results_ai['total_trades']} trades, "
          f"{results_ai['win_rate']:.1%} win rate, ${results_ai['total_pnl']:.2f} PnL")
    print(f"   Basic Strategy: {results_basic['total_trades']} trades, "
          f"{results_basic['win_rate']:.1%} win rate, ${results_basic['total_pnl']:.2f} PnL")
    
    if results_ai['total_pnl'] > results_basic['total_pnl']:
        print("🏆 AI strategy outperformed!")
    else:
        print("📊 Strategies performed similarly")
    
    return strategy_ai, results_ai


def test_with_real_data():
    """Test with real database data if available."""
    print("\n" + "="*50)
    print("🗄️ Testing with real database data...")
    
    try:
        data_loader = DataLoader()
        print("✅ Database connected")
        
        # Load recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        data = data_loader.get_price_data(
            'AAPL',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("❌ No data available")
            return
        
        print(f"✅ Loaded {len(data)} bars of AAPL data")
        
        # Quick test
        strategy = S3WorkingStrategy(
            symbol='AAPL',
            adaptive_indicators=True,
            top_k_indicators=8
        )
        
        results = strategy.backtest(data)
        print(f"🎯 Real data results: {results['total_trades']} trades, "
              f"{results['win_rate']:.1%} win rate")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    
    # Test with synthetic data
    test_s3_working()
    
    # Test with real data if available
    test_with_real_data()