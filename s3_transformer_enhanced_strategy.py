#!/usr/bin/env python3
"""
S3 Transformer-Enhanced Strategy
================================
Next-generation trading strategy that combines S2's XGBoost approach with
transformer-based dynamic indicator selection for superior performance.

Features:
- AI-driven indicator selection based on market conditions
- Transformer attention mechanism for indicator relationships
- Adaptive feature engineering
- Enhanced signal generation with context awareness
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

# Import base components
from s2_enhanced_xgboost_strategy import EnhancedXGBoostTADiscovery
from indicator_transformer import AIIndicatorSelector, IndicatorLibrary
from day_trading_o3 import DataLoader


class S3TransformerEnhancedStrategy(EnhancedXGBoostTADiscovery):
    """
    Advanced trading strategy using transformer-based indicator selection
    combined with XGBoost for signal generation.
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 symbol: str,
                 capital: float = 100000,
                 risk_per_trade: float = 0.02,
                 transformer_model_path: Optional[Path] = None,
                 adaptive_indicators: bool = True,
                 indicator_selection_interval: int = 100,
                 top_k_indicators: int = 25,
                 use_gpu: bool = False,
                 **kwargs):
        """
        Initialize S3 strategy with transformer enhancement.
        
        Args:
            df: Historical price data DataFrame
            symbol: Trading symbol
            capital: Starting capital
            risk_per_trade: Risk per trade (2% default)
            transformer_model_path: Path to pre-trained transformer model
            adaptive_indicators: Whether to use dynamic indicator selection
            indicator_selection_interval: Bars between indicator re-selection
            top_k_indicators: Maximum indicators to use at once
            use_gpu: Use GPU for transformer if available
        """
        # Store symbol for logging
        self.symbol = symbol
        
        # Initialize base strategy with DataFrame
        super().__init__(df, **kwargs)
        
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
        
        # Setup enhanced logging
        self.logger = logging.getLogger(f'S3Strategy_{symbol}')
        self.logger.info(f"🚀 S3 Transformer-Enhanced Strategy initialized for {symbol}")
        self.logger.info(f"   Adaptive indicators: {adaptive_indicators}")
        self.logger.info(f"   Top K indicators: {top_k_indicators}")
        self.logger.info(f"   Device: {device}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Override feature preparation to use transformer-selected indicators.
        """
        if not self.adaptive_indicators:
            # Fall back to standard S2 features
            return super().prepare_features(data)
        
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
        
        # Add interaction features for strongly connected indicators
        if hasattr(self, 'last_attention_patterns'):
            interaction_features = self._create_interaction_features(features_df)
            for col in interaction_features.columns:
                features_df[col] = interaction_features[col]
        
        # Forward fill and handle NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        self.logger.info(f"Prepared {len(features_df.columns)} features using {len(self.current_indicators)} selected indicators")
        
        return features_df
    
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
        self.last_attention_patterns = selection_result['attention_patterns']
        
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
        
        # Track regime changes
        if self.last_market_regime not in self.regime_performance:
            self.regime_performance[self.last_market_regime] = {
                'trades': 0,
                'wins': 0,
                'total_pnl': 0
            }
    
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
        
        # Add regime-specific derived features
        if "Trend" in self.last_market_regime:
            # Trend continuation features
            regime_features['trend_strength'] = data['close'].rolling(20).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0]
            )
        elif "Volatility" in self.last_market_regime:
            # Volatility features
            regime_features['volatility_ratio'] = data['close'].rolling(10).std() / \
                                                 data['close'].rolling(50).std()
        elif "Ranging" in self.last_market_regime:
            # Mean reversion features
            regime_features['distance_from_mean'] = (data['close'] - data['close'].rolling(20).mean()) / \
                                                   data['close'].rolling(20).std()
        
        return regime_features
    
    def _create_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features based on attention patterns.
        """
        interaction_features = pd.DataFrame(index=features_df.index)
        
        if 'strong_connections' not in self.last_attention_patterns:
            return interaction_features
        
        # Create features for top indicator interactions
        for conn in self.last_attention_patterns['strong_connections'][:5]:
            ind1 = f"{conn['indicator1']}_weighted"
            ind2 = f"{conn['indicator2']}_weighted"
            
            if ind1 in features_df.columns and ind2 in features_df.columns:
                # Multiplication interaction
                interaction_name = f"interact_{conn['indicator1']}_{conn['indicator2']}"
                interaction_features[interaction_name] = features_df[ind1] * features_df[ind2]
                
                # Difference interaction
                diff_name = f"diff_{conn['indicator1']}_{conn['indicator2']}"
                interaction_features[diff_name] = features_df[ind1] - features_df[ind2]
        
        return interaction_features
    
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate enhanced signals using regime-aware XGBoost models.
        """
        # Use different model ensemble based on regime
        if self.adaptive_indicators and self.ensemble_models:
            if "Trend" in self.last_market_regime:
                # Use trend-following focused model
                model_weights = {'conservative': 0.2, 'aggressive': 0.6, 'balanced': 0.2}
            elif "Volatility" in self.last_market_regime:
                # Use conservative model for high volatility
                model_weights = {'conservative': 0.6, 'aggressive': 0.1, 'balanced': 0.3}
            elif "Ranging" in self.last_market_regime:
                # Use balanced approach for ranging markets
                model_weights = {'conservative': 0.2, 'aggressive': 0.2, 'balanced': 0.6}
            else:
                # Default equal weights
                model_weights = {'conservative': 0.33, 'aggressive': 0.34, 'balanced': 0.33}
            
            # Generate weighted ensemble predictions
            ensemble_predictions = np.zeros(len(features))
            
            for model_name, model in self.ensemble_models.items():
                weight = model_weights.get(model_name, 0.33)
                predictions = model.predict_proba(features)[:, 1]
                ensemble_predictions += predictions * weight
            
            # Convert to signals with regime-adjusted thresholds
            if "Trend" in self.last_market_regime:
                buy_threshold = 0.55  # Lower threshold for trends
                sell_threshold = 0.45
            elif "Volatility" in self.last_market_regime:
                buy_threshold = 0.65  # Higher threshold for volatility
                sell_threshold = 0.35
            else:
                buy_threshold = 0.6   # Standard thresholds
                sell_threshold = 0.4
            
            signals = pd.Series(0, index=features.index)
            signals[ensemble_predictions > buy_threshold] = 1
            signals[ensemble_predictions < sell_threshold] = -1
            
            return signals
        else:
            # Simple fallback signal generation based on features
            signals = pd.Series(0, index=features.index)
            
            # Use simple momentum-based signals if models aren't available
            if 'returns' in features.columns:
                momentum = features['returns'].rolling(5).mean()
                signals[momentum > 0.001] = 1
                signals[momentum < -0.001] = -1
            
            return signals
    
    def calculate_position_size(self, signal: int, features: pd.Series, current_price: float) -> int:
        """
        Calculate position size with indicator confidence adjustment.
        """
        # Get base position size
        base_size = super().calculate_position_size(signal, features, current_price)
        
        if not self.adaptive_indicators or base_size == 0:
            return base_size
        
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
        adjusted_size = int(base_size * confidence_multiplier * regime_multiplier)
        
        self.logger.debug(f"Position sizing: base={base_size}, confidence={confidence_multiplier:.2f}, "
                         f"regime={regime_multiplier:.2f}, final={adjusted_size}")
        
        return adjusted_size
    
    def update_performance_tracking(self, trade_result: Dict[str, Any]):
        """
        Update performance tracking for indicators and regimes.
        """
        # Update base strategy tracking
        super().update_performance_tracking(trade_result)
        
        if not self.adaptive_indicators:
            return
        
        # Update transformer with trade result
        pnl = trade_result.get('pnl', 0)
        self.indicator_selector.update_performance(
            self.current_indicators,
            pnl
        )
        
        # Update regime performance
        if self.last_market_regime in self.regime_performance:
            regime_stats = self.regime_performance[self.last_market_regime]
            regime_stats['trades'] += 1
            regime_stats['total_pnl'] += pnl
            if pnl > 0:
                regime_stats['wins'] += 1
        
        # Update indicator-specific performance
        for indicator_name in self.current_indicators:
            if indicator_name not in self.indicator_performance:
                self.indicator_performance[indicator_name] = {
                    'trades': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }
            
            ind_stats = self.indicator_performance[indicator_name]
            ind_stats['trades'] += 1
            ind_stats['total_pnl'] += pnl
            ind_stats['avg_pnl'] = ind_stats['total_pnl'] / ind_stats['trades']
    
    def get_strategy_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive strategy performance report.
        """
        # Get base report
        report = super().get_strategy_report()
        
        if not self.adaptive_indicators:
            return report
        
        # Add transformer-specific metrics
        transformer_report = self.indicator_selector.get_performance_report()
        report['transformer_metrics'] = transformer_report
        
        # Add regime performance
        regime_summary = {}
        for regime, stats in self.regime_performance.items():
            if stats['trades'] > 0:
                regime_summary[regime] = {
                    'trades': stats['trades'],
                    'win_rate': stats['wins'] / stats['trades'],
                    'avg_pnl': stats['total_pnl'] / stats['trades'],
                    'total_pnl': stats['total_pnl']
                }
        
        report['regime_performance'] = regime_summary
        
        # Add top performing indicators
        top_indicators = sorted(
            self.indicator_performance.items(),
            key=lambda x: x[1]['avg_pnl'],
            reverse=True
        )[:10]
        
        report['top_performing_indicators'] = [
            {
                'name': name,
                'trades': stats['trades'],
                'avg_pnl': stats['avg_pnl'],
                'total_pnl': stats['total_pnl']
            }
            for name, stats in top_indicators
        ]
        
        return report
    
    def save_strategy_state(self, path: Path):
        """
        Save complete strategy state including transformer model.
        """
        # Save base strategy state
        super().save_models(path / 'base_models')
        
        # Save transformer model
        transformer_path = path / 'transformer_model.pt'
        self.indicator_selector.save_model(transformer_path)
        
        # Save S3-specific state
        s3_state = {
            'current_indicators': self.current_indicators,
            'last_market_regime': self.last_market_regime,
            'regime_performance': self.regime_performance,
            'indicator_performance': self.indicator_performance,
            'configuration': {
                'adaptive_indicators': self.adaptive_indicators,
                'indicator_selection_interval': self.indicator_selection_interval,
                'top_k_indicators': self.top_k_indicators
            }
        }
        
        with open(path / 's3_state.json', 'w') as f:
            json.dump(s3_state, f, indent=2, default=str)
        
        self.logger.info(f"S3 strategy state saved to {path}")
    
    def load_strategy_state(self, path: Path):
        """
        Load complete strategy state including transformer model.
        """
        # Load base models
        base_path = path / 'base_models'
        if base_path.exists():
            self.load_models(base_path)
        
        # Load transformer model
        transformer_path = path / 'transformer_model.pt'
        if transformer_path.exists():
            self.indicator_selector.load_model(transformer_path)
        
        # Load S3-specific state
        s3_state_path = path / 's3_state.json'
        if s3_state_path.exists():
            with open(s3_state_path, 'r') as f:
                s3_state = json.load(f)
            
            self.current_indicators = s3_state.get('current_indicators', {})
            self.last_market_regime = s3_state.get('last_market_regime', 'Unknown')
            self.regime_performance = s3_state.get('regime_performance', {})
            self.indicator_performance = s3_state.get('indicator_performance', {})
        
        self.logger.info(f"S3 strategy state loaded from {path}")


def main():
    """Example usage of S3 Transformer-Enhanced Strategy."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    print("🚀 S3 Transformer-Enhanced Trading Strategy")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load historical data
    print("\n📊 Loading historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = data_loader.get_price_data(
        'AAPL',
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if data.empty:
        print("❌ No data available")
        return
    
    print(f"✅ Loaded {len(data)} bars of data")
    
    # Create strategy instance
    strategy = S3TransformerEnhancedStrategy(
        df=data,
        symbol='AAPL',
        capital=100000,
        adaptive_indicators=True,
        top_k_indicators=20,
        indicator_selection_interval=100
    )
    
    # Run strategy - prepare features and generate signals
    print("\n🎯 Running strategy...")
    
    # Prepare features using transformer
    features = strategy.prepare_features(data)
    print(f"✅ Prepared {features.shape[1]} features")
    
    # Generate signals
    valid_features = features.dropna()
    if len(valid_features) > 0:
        signals = strategy.generate_signals(valid_features)
        print(f"✅ Generated {len(signals)} signals")
    else:
        print("❌ No valid features for signal generation")
        return
    
    # Get performance report
    report = strategy.get_strategy_report()
    
    print("\n📈 Strategy Performance Report:")
    print(f"   Total trades: {report.get('total_trades', 0)}")
    print(f"   Win rate: {report.get('win_rate', 0):.2%}")
    print(f"   Profit factor: {report.get('profit_factor', 0):.2f}")
    print(f"   Sharpe ratio: {report.get('sharpe_ratio', 0):.2f}")
    
    if 'regime_performance' in report:
        print("\n🌐 Performance by Market Regime:")
        for regime, stats in report['regime_performance'].items():
            print(f"   {regime}: {stats['win_rate']:.2%} win rate, "
                  f"${stats['avg_pnl']:.2f} avg PnL")
    
    if 'top_performing_indicators' in report:
        print("\n🏆 Top Performing Indicators:")
        for ind in report['top_performing_indicators'][:5]:
            print(f"   {ind['name']}: ${ind['avg_pnl']:.2f} avg PnL "
                  f"({ind['trades']} trades)")
    
    # Save strategy state
    save_path = Path('models') / 's3_strategy' / 'AAPL'
    save_path.mkdir(parents=True, exist_ok=True)
    strategy.save_strategy_state(save_path)
    print(f"\n💾 Strategy state saved to {save_path}")


if __name__ == "__main__":
    main()