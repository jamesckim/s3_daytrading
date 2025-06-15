#!/usr/bin/env python3
"""
S3 Optimized Strategy - Enhanced for Higher Win Rates
====================================================
Implements win rate optimization techniques based on backtesting analysis.
Target: 50-55% win rate (up from 43.9%)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging
from s3_database_final import S3DatabaseFinal

warnings.filterwarnings('ignore')


class S3OptimizedStrategy(S3DatabaseFinal):
    """
    Optimized S3 strategy with enhanced filters for higher win rates.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,
                 top_k_indicators: int = 15,  # Increased from 10
                 selection_interval: int = 50,  # Reduced from 100
                 data_days_lookback: int = 30,
                 # New optimization parameters
                 min_win_rate_threshold: float = 0.45,
                 use_time_filters: bool = True,
                 use_microstructure: bool = True,
                 daily_loss_limit_pct: float = 0.02):
        """
        Initialize optimized S3 strategy with enhanced parameters.
        """
        
        # Initialize parent class
        super().__init__(
            symbol=symbol,
            capital=capital,
            adaptive_indicators=adaptive_indicators,
            top_k_indicators=top_k_indicators,
            selection_interval=selection_interval,
            data_days_lookback=data_days_lookback
        )
        
        # Optimization parameters
        self.min_win_rate_threshold = min_win_rate_threshold
        self.use_time_filters = use_time_filters
        self.use_microstructure = use_microstructure
        self.daily_loss_limit_pct = daily_loss_limit_pct
        
        # Performance tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.recent_trades_window = []
        self.rolling_win_rate = 0.5
        
        # Enhanced thresholds based on analysis
        self.optimized_thresholds = {
            'High Volatility': {
                'momentum': 0.001,  # Doubled from 0.0005
                'volume_ratio': 1.5,  # Increased from 1.2
                'max_rsi': 65,  # Tightened from 70
                'min_rsi': 35,  # Tightened from 30
                'price_position_buy': 0.9,  # Don't buy above 90% of daily range
                'price_position_sell': 0.1  # Don't sell below 10% of daily range
            },
            'Uptrend': {
                'momentum': 0.0007,
                'volume_ratio': 1.3,
                'max_rsi': 70,
                'min_rsi': 30,
                'price_position_buy': 0.95,
                'price_position_sell': 0.05
            },
            'Ranging': {
                'momentum': 0.0008,
                'volume_ratio': 1.4,
                'max_rsi': 60,
                'min_rsi': 40,
                'price_position_buy': 0.85,
                'price_position_sell': 0.15
            }
        }
        
        # Known good/bad performers from analysis
        self.high_win_rate_stocks = {
            'GS', 'AAPL', 'CRM', 'QCOM', 'NKE', 'LLY', 'TMO', 
            'IBM', 'SCHW', 'DIS', 'HOOD', 'GOOGL', 'SMCI'
        }
        
        self.low_win_rate_stocks = {
            'F', 'HL', 'BBAI', 'RIG', 'CIFR', 'WULF', 'UUUU', 
            'UEC', 'INTC', 'PFE', 'T', 'GM'
        }
        
        self.logger.info(f"🚀 S3 Optimized Strategy initialized for {symbol}")
        self.logger.info(f"   Win rate optimization: Enabled")
        self.logger.info(f"   Time filters: {use_time_filters}")
        self.logger.info(f"   Microstructure: {use_microstructure}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features including microstructure."""
        
        # Store data for use in generate_signals
        self.current_data = data
        
        # Get base features from parent
        features = super().prepare_features(data)
        
        # Add microstructure features if enabled
        if self.use_microstructure:
            # Bid-ask spread proxy
            features['high_low_spread'] = (self.current_data['high'] - self.current_data['low']) / self.current_data['close']
            
            # Order flow imbalance
            features['volume_delta'] = self.current_data['volume'] * np.sign(self.current_data['close'] - self.current_data['open'])
            features['order_flow'] = features['volume_delta'].rolling(10).sum()
            
            # Price acceleration
            features['price_acceleration'] = self.current_data['close'].diff().diff()
            
            # Relative volume
            features['relative_volume'] = self.current_data['volume'] / self.current_data['volume'].rolling(50).mean()
            
            # Intraday momentum
            features['intraday_momentum'] = (self.current_data['close'] - self.current_data['open']) / self.current_data['open']
        
        # Add time-based features
        if self.use_time_filters:
            # Extract hour and minute
            features['hour'] = self.current_data.index.hour
            features['minute'] = self.current_data.index.minute
            features['time_of_day'] = features['hour'] + features['minute'] / 60
            
            # Market session indicators
            features['opening_session'] = ((features['hour'] == 9) & (features['minute'] >= 30)) | \
                                        ((features['hour'] == 10) & (features['minute'] < 30))
            features['closing_session'] = (features['hour'] == 15) & (features['minute'] >= 30)
            features['prime_hours'] = (features['hour'] >= 10) & (features['hour'] < 15)
        
        # Multi-timeframe features
        # 5-min aggregation
        if len(self.current_data) > 5:
            features['momentum_5min'] = self.current_data['close'].pct_change(5)
            features['volume_5min'] = self.current_data['volume'].rolling(5).sum()
        
        # 15-min aggregation  
        if len(self.current_data) > 15:
            features['momentum_15min'] = self.current_data['close'].pct_change(15)
            features['high_15min'] = self.current_data['high'].rolling(15).max()
            features['low_15min'] = self.current_data['low'].rolling(15).min()
            features['range_15min'] = (features['high_15min'] - features['low_15min']) / self.current_data['close']
        
        # 30-min aggregation
        if len(self.current_data) > 30:
            features['momentum_30min'] = self.current_data['close'].pct_change(30)
            features['vwap_30min'] = (self.current_data['close'] * self.current_data['volume']).rolling(30).sum() / \
                                     self.current_data['volume'].rolling(30).sum()
            features['vwap_deviation'] = (self.current_data['close'] - features['vwap_30min']) / features['vwap_30min']
        
        # Quality score for current conditions
        features['signal_quality'] = 0
        
        # Increase quality for good microstructure
        if self.use_microstructure:
            features.loc[features['high_low_spread'] < features['high_low_spread'].quantile(0.3), 'signal_quality'] += 1
            features.loc[features['order_flow'] > 0, 'signal_quality'] += 1
            features.loc[features['relative_volume'] > 1.5, 'signal_quality'] += 1
        
        # Increase quality for good timing
        if self.use_time_filters:
            features.loc[features['prime_hours'], 'signal_quality'] += 1
            features.loc[~features['opening_session'] & ~features['closing_session'], 'signal_quality'] += 1
        
        # Symbol-specific adjustments
        if self.symbol in self.high_win_rate_stocks:
            features['signal_quality'] += 1
        elif self.symbol in self.low_win_rate_stocks:
            features['signal_quality'] -= 1
        
        return features
    
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate optimized trading signals with enhanced filters."""
        
        signals = pd.Series(0, index=features.index)
        
        # Get regime-specific thresholds
        regime_key = 'High Volatility'  # Default
        for key in self.optimized_thresholds:
            if key.lower() in self.last_market_regime.lower():
                regime_key = key
                break
        
        thresholds = self.optimized_thresholds[regime_key]
        
        # Calculate combined momentum
        momentum_weights = {'momentum_5': 0.5, 'momentum_10': 0.3, 'momentum_20': 0.2}
        combined_momentum = 0
        
        for period, weight in momentum_weights.items():
            if period in features.columns:
                combined_momentum += features[period] * weight
        
        # Base conditions
        strong_bullish = combined_momentum > thresholds['momentum']
        strong_bearish = combined_momentum < -thresholds['momentum']
        
        # Volume confirmation
        if 'volume_ratio' in features.columns:
            volume_confirmed = features['volume_ratio'] > thresholds['volume_ratio']
        else:
            volume_confirmed = True
        
        # RSI boundaries
        if 'rsi_14' in features.columns:
            rsi_not_overbought = features['rsi_14'] < thresholds['max_rsi']
            rsi_not_oversold = features['rsi_14'] > thresholds['min_rsi']
        else:
            rsi_not_overbought = True
            rsi_not_oversold = True
        
        # Price position filter
        if 'price_position' in features.columns:
            not_at_high = features['price_position'] < thresholds['price_position_buy']
            not_at_low = features['price_position'] > thresholds['price_position_sell']
        else:
            not_at_high = True
            not_at_low = True
        
        # Time filters
        if self.use_time_filters and 'prime_hours' in features.columns:
            good_time = features['prime_hours'] | \
                       ((features['hour'] == 10) & (features['minute'] >= 30)) | \
                       ((features['hour'] == 15) & (features['minute'] <= 0))
        else:
            good_time = True
        
        # Microstructure filters
        if self.use_microstructure and 'signal_quality' in features.columns:
            quality_signal = features['signal_quality'] >= 3  # At least 3 quality points
        else:
            quality_signal = True
        
        # Multi-timeframe confirmation
        if 'momentum_15min' in features.columns and 'momentum_30min' in features.columns:
            trend_aligned = (
                ((features['momentum_5min'] > 0) & (features['momentum_15min'] > 0)) |
                ((features['momentum_5min'] < 0) & (features['momentum_15min'] < 0))
            )
        else:
            trend_aligned = True
        
        # Generate BUY signals with all confirmations
        buy_conditions = (
            strong_bullish &
            volume_confirmed &
            rsi_not_overbought &
            not_at_high &
            good_time &
            quality_signal &
            trend_aligned
        )
        
        # Generate SELL signals with all confirmations
        sell_conditions = (
            strong_bearish &
            volume_confirmed &
            rsi_not_oversold &
            not_at_low &
            good_time &
            quality_signal &
            trend_aligned
        )
        
        signals[buy_conditions] = 1
        signals[sell_conditions] = -1
        
        # Additional safety filters
        
        # 1. Check daily loss limit
        if hasattr(self, 'daily_pnl') and self.daily_pnl < -(self.capital * self.daily_loss_limit_pct):
            signals.iloc[-1] = 0  # No new trades if daily loss limit hit
        
        # 2. Reduce signals after consecutive losses
        if hasattr(self, 'consecutive_losses') and self.consecutive_losses >= 3:
            # Only take very high quality signals
            if 'signal_quality' in features.columns:
                signals[features['signal_quality'] < 4] = 0
        
        # 3. Volatility filter
        if 'volatility' in features.columns:
            extreme_volatility = features['volatility'] > features['volatility'].quantile(0.95)
            signals[extreme_volatility] = 0
        
        # 4. Spread filter for microstructure
        if self.use_microstructure and 'high_low_spread' in features.columns:
            wide_spread = features['high_low_spread'] > features['high_low_spread'].quantile(0.8)
            signals[wide_spread] = 0
        
        return signals
    
    def calculate_position_size(self, 
                              signal: int, 
                              features: pd.Series, 
                              current_price: float) -> int:
        """Calculate position size with win rate and quality adjustments."""
        
        if signal == 0:
            return 0
        
        # Base position sizing (1.5% risk for optimized strategy)
        risk_amount = self.capital * 0.015
        
        # Quality-based adjustment
        signal_quality = features.get('signal_quality', 3)
        quality_multiplier = 0.5 + (signal_quality / 10)  # 0.5 to 1.0 range
        
        # Win rate adjustment
        if hasattr(self, 'rolling_win_rate'):
            if self.rolling_win_rate > 0.55:
                win_rate_multiplier = 1.2
            elif self.rolling_win_rate > 0.50:
                win_rate_multiplier = 1.1
            elif self.rolling_win_rate < 0.40:
                win_rate_multiplier = 0.7
            else:
                win_rate_multiplier = 1.0
        else:
            win_rate_multiplier = 1.0
        
        # Symbol-specific adjustment
        if self.symbol in self.high_win_rate_stocks:
            symbol_multiplier = 1.1
        elif self.symbol in self.low_win_rate_stocks:
            symbol_multiplier = 0.8
        else:
            symbol_multiplier = 1.0
        
        # Volatility adjustment
        if 'volatility' in features.index:
            current_vol = features['volatility']
            avg_vol = 0.015  # Assume 1.5% average volatility
            if current_vol > 0:
                vol_multiplier = min(1.5, max(0.5, avg_vol / current_vol))
            else:
                vol_multiplier = 1.0
        else:
            vol_multiplier = 1.0
        
        # Calculate stop distance
        if 'atr_14' in features.index:
            stop_distance = features['atr_14'] * 2
        else:
            stop_distance = current_price * 0.015  # 1.5% default
        
        # Calculate final position size
        base_shares = int(risk_amount / stop_distance)
        
        final_multiplier = (quality_multiplier * win_rate_multiplier * 
                           symbol_multiplier * vol_multiplier)
        
        final_shares = int(base_shares * final_multiplier)
        
        # Apply maximum position limits (8% of capital for high conviction)
        max_shares = int(self.capital * 0.08 / current_price)
        final_shares = min(final_shares, max_shares)
        
        return final_shares * signal
    
    def update_performance_tracking(self, trade_result: dict):
        """Update rolling performance metrics after each trade."""
        
        # Update daily PnL
        if 'pnl' in trade_result:
            self.daily_pnl += trade_result['pnl']
        
        # Track consecutive losses
        if trade_result.get('pnl', 0) < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update rolling window
        self.recent_trades_window.append(trade_result)
        if len(self.recent_trades_window) > 20:  # Keep last 20 trades
            self.recent_trades_window.pop(0)
        
        # Calculate rolling win rate
        if len(self.recent_trades_window) >= 5:
            wins = sum(1 for t in self.recent_trades_window if t.get('pnl', 0) > 0)
            self.rolling_win_rate = wins / len(self.recent_trades_window)
    
    def get_optimization_stats(self) -> dict:
        """Get current optimization statistics."""
        
        return {
            'rolling_win_rate': self.rolling_win_rate,
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': self.daily_pnl,
            'recent_trades': len(self.recent_trades_window),
            'signal_quality_threshold': 3,
            'current_regime': self.last_market_regime,
            'is_high_win_rate_stock': self.symbol in self.high_win_rate_stocks
        }


def main():
    """Test the optimized strategy."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 S3 Optimized Strategy Test")
    print("=" * 60)
    
    # Test with high win rate stocks
    test_symbols = ['AAPL', 'GS', 'CRM', 'NKE']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        try:
            strategy = S3OptimizedStrategy(
                symbol=symbol,
                capital=100000,
                adaptive_indicators=True,
                use_time_filters=True,
                use_microstructure=True
            )
            
            # Run backtest
            results = strategy.backtest(verbose=False)
            
            print(f"✅ Results for {symbol}:")
            print(f"   Win rate: {results['win_rate']:.1%}")
            print(f"   Total trades: {results['total_trades']}")
            print(f"   Total PnL: ${results['total_pnl']:.2f}")
            print(f"   Return: {results['total_return']:.2%}")
            
            # Get optimization stats
            stats = strategy.get_optimization_stats()
            print(f"   Optimization stats: {stats}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
    
    print("\n✅ Optimization test completed!")


if __name__ == "__main__":
    main()