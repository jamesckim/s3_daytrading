#!/usr/bin/env python3
"""
S3 Minimal Strategy - Database-only version without yfinance dependencies
========================================================================
Simplified version of S3 strategy that avoids the complex S2 inheritance chain
and focuses purely on transformer-based indicator selection.
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
from indicator_transformer import AIIndicatorSelector, IndicatorLibrary
from day_trading_o3 import DataLoader


class S3MinimalStrategy:
    """
    Minimal S3 strategy focusing on transformer-based indicator selection
    without the complex S2 inheritance chain.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 risk_per_trade: float = 0.02,
                 transformer_model_path: Optional[Path] = None,
                 adaptive_indicators: bool = True,
                 indicator_selection_interval: int = 100,
                 top_k_indicators: int = 25,
                 use_gpu: bool = False):
        """
        Initialize minimal S3 strategy.
        
        Args:
            symbol: Trading symbol
            capital: Starting capital
            risk_per_trade: Risk per trade (2% default)
            transformer_model_path: Path to pre-trained transformer model
            adaptive_indicators: Whether to use dynamic indicator selection
            indicator_selection_interval: Bars between indicator re-selection
            top_k_indicators: Maximum indicators to use at once
            use_gpu: Use GPU for transformer if available
        """
        self.symbol = symbol
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        
        # S3-specific configuration
        self.adaptive_indicators = adaptive_indicators
        self.indicator_selection_interval = indicator_selection_interval
        self.top_k_indicators = top_k_indicators
        self.bars_since_selection = 0
        
        # Initialize transformer components
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.indicator_selector = AIIndicatorSelector(
            model_path=transformer_model_path,
            device=device
        )
        
        # Enhanced indicator library
        self.indicator_library = IndicatorLibrary()
        
        # Current indicator selection
        self.current_indicators = {}
        self.indicator_weights = {}
        self.last_market_regime = "Unknown"
        
        # Performance tracking
        self.indicator_performance = {}
        self.regime_performance = {}
        self.trades = []
        
        # Setup logging
        self.logger = logging.getLogger(f'S3Minimal_{symbol}')
        self.logger.info(f"🚀 S3 Minimal Strategy initialized for {symbol}")
        self.logger.info(f"   Adaptive indicators: {adaptive_indicators}")
        self.logger.info(f"   Top K indicators: {top_k_indicators}")
        self.logger.info(f"   Device: {device}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features using transformer-selected indicators.
        """
        if not self.adaptive_indicators:
            # Fall back to basic features
            return self._prepare_basic_features(data)
        
        # Check if we need to re-select indicators
        if (self.bars_since_selection >= self.indicator_selection_interval or 
            len(self.current_indicators) == 0):
            self._select_indicators(data)
            self.bars_since_selection = 0
        else:
            self.bars_since_selection += 1
        
        # Compute selected indicators
        features_df = pd.DataFrame(index=data.index)
        
        # Always include basic price features
        features_df['returns'] = data['close'].pct_change()
        features_df['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features_df['high_low_ratio'] = data['high'] / data['low']
        features_df['close_open_ratio'] = data['close'] / data['open']
        features_df['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Add transformer-selected indicators
        for indicator_name, indicator_info in self.current_indicators.items():
            try:
                # Compute indicator
                indicator_values = self.indicator_library.compute_indicator(data, indicator_name)
                
                # Apply weight if specified
                weight = indicator_info.get('weight', 1.0)
                feature_name = f"{indicator_name}_weighted"
                features_df[feature_name] = indicator_values * weight
                
                # Add indicator lag features if important
                if indicator_info.get('selection_prob', 0) > 0.7:
                    features_df[f"{feature_name}_lag1"] = features_df[feature_name].shift(1)
                    features_df[f"{feature_name}_diff"] = features_df[feature_name].diff()
                
            except Exception as e:
                self.logger.warning(f"Failed to compute {indicator_name}: {e}")
                continue
        
        # Add market regime features
        regime_features = self._create_regime_features(data)
        for col in regime_features.columns:
            features_df[col] = regime_features[col]
        
        # Forward fill and handle NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        self.logger.info(f"Prepared {len(features_df.columns)} features using {len(self.current_indicators)} selected indicators")
        
        return features_df
    
    def _prepare_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare basic features without transformer selection."""
        features_df = pd.DataFrame(index=data.index)
        
        # Basic price features
        features_df['returns'] = data['close'].pct_change()
        features_df['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features_df['high_low_ratio'] = data['high'] / data['low']
        features_df['close_open_ratio'] = data['close'] / data['open']
        features_df['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Simple technical indicators
        features_df['sma_20'] = data['close'].rolling(20).mean()
        features_df['sma_50'] = data['close'].rolling(50).mean()
        features_df['rsi_14'] = self._compute_rsi(data['close'], 14)
        features_df['bb_width'] = self._compute_bollinger_width(data['close'], 20)
        
        return features_df.fillna(method='ffill').fillna(0)
    
    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_bollinger_width(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute Bollinger Band width."""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        bb_width = (std_dev * 2) / sma
        return bb_width
    
    def _select_indicators(self, data: pd.DataFrame):
        """
        Use transformer to select optimal indicators for current market.
        """
        self.logger.info("🎯 Selecting new indicators based on market conditions...")
        
        # Get indicator selection from transformer
        selection_result = self.indicator_selector.select_indicators(
            data=data,
            top_k=self.top_k_indicators
        )
        
        # Update current indicators
        self.current_indicators = selection_result['selected_indicators']
        self.last_market_regime = selection_result['regime_detection']
        
        # Log selection
        self.logger.info(f"📊 Market Regime: {self.last_market_regime}")
        self.logger.info(f"✅ Selected {len(self.current_indicators)} indicators:")
        
        # Log top 5 indicators by weight
        sorted_indicators = sorted(
            self.current_indicators.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )[:5]
        
        for indicator_name, info in sorted_indicators:
            self.logger.info(f"   {indicator_name}: weight={info['weight']:.3f}, prob={info['selection_prob']:.3f}")
    
    def _create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on detected market regime.
        """
        regime_features = pd.DataFrame(index=data.index)
        
        # Encode regime as features
        regime_encoding = {
            "Strong Uptrend": [1, 0, 0, 0, 0],
            "Strong Downtrend": [-1, 0, 0, 0, 0],
            "High Volatility": [0, 1, 0, 0, 0],
            "Ranging/Sideways": [0, 0, 1, 0, 0],
            "Transitional": [0, 0, 0, 1, 0],
            "Unknown": [0, 0, 0, 0, 1]
        }
        
        encoding = regime_encoding.get(self.last_market_regime, regime_encoding["Unknown"])
        for i, value in enumerate(encoding):
            regime_features[f'regime_{i}'] = value
        
        return regime_features
    
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on features and market regime.
        """
        signals = pd.Series(0, index=features.index)
        
        # Simple momentum-based signals
        if 'returns' in features.columns:
            momentum = features['returns'].rolling(5).mean()
            
            # Adjust thresholds based on regime
            if "Trend" in self.last_market_regime:
                buy_threshold = 0.0005  # Lower threshold for trends
                sell_threshold = -0.0005
            elif "Volatility" in self.last_market_regime:
                buy_threshold = 0.001   # Higher threshold for volatility
                sell_threshold = -0.001
            else:
                buy_threshold = 0.0008  # Standard thresholds
                sell_threshold = -0.0008
            
            signals[momentum > buy_threshold] = 1
            signals[momentum < sell_threshold] = -1
        
        # Add volume confirmation if available
        if 'volume_ratio' in features.columns:
            high_volume = features['volume_ratio'] > 1.2
            signals = signals * high_volume.astype(int)  # Only trade on high volume
        
        return signals
    
    def calculate_position_size(self, signal: int, features: pd.Series, current_price: float) -> int:
        """
        Calculate position size with indicator confidence adjustment.
        """
        if signal == 0:
            return 0
        
        # Base position size calculation
        risk_amount = self.capital * self.risk_per_trade
        
        # Estimate stop loss (2% for now)
        stop_loss_pct = 0.02
        stop_loss_amount = current_price * stop_loss_pct
        
        base_shares = int(risk_amount / stop_loss_amount)
        
        if not self.adaptive_indicators or not self.current_indicators:
            return base_shares
        
        # Adjust based on indicator selection confidence
        avg_confidence = np.mean([ind['selection_prob'] for ind in self.current_indicators.values()])
        
        # Scale position size by confidence (0.5 to 1.0 multiplier)
        confidence_multiplier = 0.5 + (avg_confidence * 0.5)
        
        # Additional adjustment based on regime
        regime_multipliers = {
            "Strong Uptrend": 1.1,
            "Strong Downtrend": 0.9,
            "High Volatility": 0.7,
            "Ranging/Sideways": 0.8,
            "Transitional": 0.9,
            "Unknown": 1.0
        }
        
        regime_multiplier = regime_multipliers.get(self.last_market_regime, 1.0)
        
        # Calculate final position size
        adjusted_shares = int(base_shares * confidence_multiplier * regime_multiplier)
        
        self.logger.debug(f"Position sizing: base={base_shares}, confidence={confidence_multiplier:.2f}, "
                         f"regime={regime_multiplier:.2f}, final={adjusted_shares}")
        
        return adjusted_shares
    
    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a simple backtest of the strategy.
        """
        self.logger.info(f"🚀 Starting backtest with {len(data)} bars")
        
        # Prepare features
        features = self.prepare_features(data)
        
        # Generate signals
        signals = self.generate_signals(features)
        
        # Simulate trades
        trades = []
        position = 0
        entry_price = 0
        capital = self.capital
        
        for i in range(len(data)):
            if i < 50:  # Skip first 50 bars for feature warmup
                continue
            
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]
            feature_row = features.iloc[i]
            
            # Close existing position
            if position != 0 and signal != position:
                exit_price = current_price
                pnl = (exit_price - entry_price) * abs(position)
                if position < 0:  # Short position
                    pnl = -pnl
                
                trade = {
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'pnl': pnl,
                    'regime': self.last_market_regime,
                    'indicators_used': list(self.current_indicators.keys())
                }
                trades.append(trade)
                capital += pnl
                position = 0
            
            # Open new position
            if signal != 0 and position == 0:
                position_size = self.calculate_position_size(signal, feature_row, current_price)
                if position_size > 0:
                    position = position_size * signal  # Positive for long, negative for short
                    entry_price = current_price
        
        # Calculate performance metrics
        if trades:
            total_pnl = sum(trade['pnl'] for trade in trades)
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            total_pnl = win_rate = avg_win = avg_loss = profit_factor = 0
        
        results = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades) if trades else 0,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_capital': capital,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'trades': trades,
            'signals_generated': signals.sum(),
            'features_used': features.shape[1],
            'indicators_selected': len(self.current_indicators),
            'market_regime': self.last_market_regime
        }
        
        self.logger.info(f"📈 Backtest complete: {len(trades)} trades, {win_rate:.2%} win rate, ${total_pnl:.2f} PnL")
        
        return results
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """
        Update performance tracking for indicators and regimes.
        """
        if not self.adaptive_indicators:
            return
        
        # Update transformer with trade result
        pnl = trade_result.get('pnl', 0)
        self.indicator_selector.update_performance(
            self.current_indicators,
            pnl
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report.
        """
        # Get transformer metrics
        transformer_report = self.indicator_selector.get_performance_report()
        
        report = {
            'transformer_metrics': transformer_report,
            'current_indicators': self.current_indicators,
            'market_regime': self.last_market_regime,
            'total_trades': len(self.trades),
            'capital': self.capital
        }
        
        return report


def main():
    """Example usage of S3 Minimal Strategy."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    print("🚀 S3 Minimal Strategy - Database Only")
    print("=" * 50)
    
    # Initialize data loader
    try:
        data_loader = DataLoader()
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    # Load historical data
    print("\n📊 Loading historical data from database...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Just 7 days for testing
    
    try:
        data = data_loader.get_price_data(
            'AAPL',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("❌ No data available")
            return
        
        print(f"✅ Loaded {len(data)} bars of AAPL data")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Create strategy instance
    print("\n🎯 Creating S3 Minimal Strategy...")
    strategy = S3MinimalStrategy(
        symbol='AAPL',
        capital=100000,
        adaptive_indicators=True,
        top_k_indicators=15,
        indicator_selection_interval=100
    )
    
    # Run backtest
    print("\n📈 Running backtest...")
    results = strategy.backtest(data)
    
    print(f"\n🎯 Results:")
    print(f"   Total trades: {results['total_trades']}")
    print(f"   Win rate: {results['win_rate']:.2%}")
    print(f"   Total PnL: ${results['total_pnl']:.2f}")
    print(f"   Profit factor: {results['profit_factor']:.2f}")
    print(f"   Market regime: {results['market_regime']}")
    print(f"   Indicators used: {results['indicators_selected']}")


if __name__ == "__main__":
    main()