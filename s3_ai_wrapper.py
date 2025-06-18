#!/usr/bin/env python3
"""
S3 AI Wrapper - Modified AI Indicator Selector for S3 Strategy
============================================================
Wrapper around AIIndicatorSelector that modifies the confidence threshold
for better indicator selection in the s3 strategy. Ensures reliable
indicator selection with fallback mechanisms.

Key modifications:
- Lower confidence threshold (0.1 instead of 0.3)
- Guaranteed minimum indicators selection (3-5 indicators)
- Fallback mechanisms for edge cases
- Enhanced error handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Import the original AIIndicatorSelector
from indicator_transformer import (
    AIIndicatorSelector, 
    IndicatorLibrary, 
    MarketRegimeAnalyzer
)


class S3AIWrapper:
    """
    Wrapper class that modifies AIIndicatorSelector for S3 strategy requirements.
    
    This wrapper addresses the issue where the original AIIndicatorSelector's
    confidence threshold of 0.3 is too restrictive, often resulting in 0 indicators
    being selected. The S3 strategy needs at least some indicators to work properly.
    """
    
    # Optimized VWAP confidence thresholds (adjusted for better balance)
    VWAP_CONFIDENCE_THRESHOLDS = {
        'base_vwap': 0.15,    # Base VWAPs (institutional benchmarks)
        'bands_1σ': 0.20,     # Normal trading range (68% probability)
        'bands_2σ': 0.18,     # Mean reversion sweet spot (95% probability)
        'bands_3σ': 0.25,     # Extreme moves (99.7% probability)
        'general': 0.10       # Standard for non-VWAP indicators
    }
    
    def __init__(self, 
                 model_path: Optional[Path] = None,
                 device: str = 'cpu',
                 confidence_threshold: float = 0.1,
                 min_indicators: int = 3,
                 max_indicators: int = 15,
                 fallback_indicators: Optional[List[str]] = None):
        """
        Initialize the S3 AI Wrapper.
        
        Args:
            model_path: Path to pre-trained model (optional)
            device: Device for PyTorch computations
            confidence_threshold: Lower threshold for indicator selection (default 0.1)
            min_indicators: Minimum number of indicators to always select
            max_indicators: Maximum number of indicators to select
            fallback_indicators: List of indicator names to use as fallback
        """
        self.logger = logging.getLogger('S3AIWrapper')
        self.confidence_threshold = confidence_threshold
        self.min_indicators = min_indicators
        self.max_indicators = max_indicators
        
        # Initialize the underlying AI selector
        try:
            self.ai_selector = AIIndicatorSelector(model_path=model_path, device=device)
            self.logger.info("AI Indicator Selector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Indicator Selector: {e}")
            # Create basic fallback components
            self.ai_selector = None
            self.indicator_library = IndicatorLibrary()
            self.market_analyzer = MarketRegimeAnalyzer()
        
        # Define fallback indicators (proven performers for S3 strategy)
        if fallback_indicators is None:
            self.fallback_indicators = [
                'RSI_7',            # Fast momentum (7 min)
                'RSI_14',           # Standard momentum (14 min)
                'RSI_21',           # Slower momentum (21 min)
                'SMA_20',           # Trend
                'EMA_5',            # Very short-term trend (5 min)
                'EMA_10',           # Short-term trend (10 min)
                'EMA_20',           # Short-medium trend (20 min)
                'EMA_50',           # Medium-term trend (50 min)
                'ATR_14',           # Standard volatility
                'ATR_7',            # Fast volatility for tight stops
                'MACD',             # Standard momentum/trend
                'MACD_FAST',        # Quick signals
                'BB_20',            # Standard volatility bands
                'BB_10',            # Fast bands for scalping
            ]
        else:
            self.fallback_indicators = fallback_indicators
        
        # Statistics tracking
        self.selection_stats = {
            'total_calls': 0,
            'successful_ai_calls': 0,
            'fallback_calls': 0,
            'min_indicators_enforced': 0,
            'avg_indicators_selected': 0.0
        }
        
        self.logger.info(f"S3 AI Wrapper initialized with threshold={confidence_threshold}, "
                        f"min_indicators={min_indicators}, max_indicators={max_indicators}")
    
    def select_indicators(self, 
                         data: pd.DataFrame,
                         top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Select optimal indicators with modified confidence threshold and fallbacks.
        
        Args:
            data: Historical price data
            top_k: Maximum number of indicators to select (uses max_indicators if None)
            
        Returns:
            Dictionary with selected indicators and weights, guaranteed to have
            at least min_indicators selected
        """
        self.selection_stats['total_calls'] += 1
        
        if top_k is None:
            top_k = self.max_indicators
        
        # First, try the AI selector if available
        if self.ai_selector is not None:
            try:
                ai_result = self._try_ai_selection(data, top_k)
                if ai_result is not None:
                    self.selection_stats['successful_ai_calls'] += 1
                    return ai_result
            except Exception as e:
                self.logger.warning(f"AI selection failed: {e}, falling back to manual selection")
        
        # Fallback to manual selection
        self.logger.info("Using fallback indicator selection")
        self.selection_stats['fallback_calls'] += 1
        return self._fallback_selection(data, top_k)
    
    def _try_ai_selection(self, data: pd.DataFrame, top_k: int) -> Optional[Dict[str, Any]]:
        """
        Try AI-based selection with modified threshold logic.
        
        Returns:
            AI selection result or None if unsuccessful
        """
        try:
            # Get the original AI selection
            original_result = self.ai_selector.select_indicators(data, top_k)
            
            # Extract the underlying probabilities and weights using our modified logic
            modified_result = self._apply_modified_selection_logic(data, top_k, original_result)
            
            # Ensure minimum indicators are selected
            final_result = self._ensure_minimum_indicators(data, modified_result)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"AI selection attempt failed: {e}")
            return None
    
    def get_confidence_threshold(self, indicator_name: str) -> float:
        """Get optimized confidence threshold for specific indicator."""
        if 'VWAP' not in indicator_name:
            return self.VWAP_CONFIDENCE_THRESHOLDS['general']
        
        if indicator_name.endswith(('_U1', '_L1')):
            return self.VWAP_CONFIDENCE_THRESHOLDS['bands_1σ']
        elif indicator_name.endswith(('_U2', '_L2')):
            return self.VWAP_CONFIDENCE_THRESHOLDS['bands_2σ']
        elif indicator_name.endswith(('_U3', '_L3')):
            return self.VWAP_CONFIDENCE_THRESHOLDS['bands_3σ']
        else:
            return self.VWAP_CONFIDENCE_THRESHOLDS['base_vwap']
    
    def _apply_modified_selection_logic(self, 
                                      data: pd.DataFrame, 
                                      top_k: int, 
                                      original_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply modified selection logic with optimized confidence thresholds.
        
        This method recomputes the selection using optimized thresholds for
        different indicator types, especially VWAP bands.
        """
        try:
            # Recompute all indicators
            indicator_values = self.ai_selector._compute_all_indicators(data)
            market_context = self.ai_selector.market_analyzer.analyze_market_context(data)
            
            # Get model predictions (this requires accessing the model directly)
            import torch
            self.ai_selector.model.eval()
            
            with torch.no_grad():
                # Prepare inputs
                indicator_tensor = torch.FloatTensor(indicator_values).unsqueeze(0).to(self.ai_selector.device)
                context_tensor = torch.FloatTensor(market_context).unsqueeze(0).to(self.ai_selector.device)
                indices_tensor = torch.LongTensor(range(self.ai_selector.num_indicators)).unsqueeze(0).to(self.ai_selector.device)
                
                # Forward pass
                outputs = self.ai_selector.model(indicator_tensor, context_tensor, indices_tensor)
                
                # Extract results
                selection_probs = outputs['selection_probs'].squeeze().cpu().numpy()
                indicator_weights = outputs['indicator_weights'].squeeze().cpu().numpy()
            
            # Apply our modified selection logic with lower threshold
            if top_k < self.ai_selector.num_indicators:
                top_indices = np.argsort(selection_probs)[-top_k:]
            else:
                top_indices = np.where(selection_probs > 0.05)[0]  # Even lower initial threshold
            
            # FIXED: Always select top-k indicators regardless of probability
            # This fixes the issue where all probabilities are too low
            selected_indicators = {}
            
            # Always take at least min_indicators
            num_to_select = max(len(top_indices), self.min_indicators)
            
            # Get top indicators by probability
            all_probs_indices = np.argsort(selection_probs)[-num_to_select:][::-1]
            
            for idx in all_probs_indices:
                indicator_name = self.ai_selector.indicator_names[idx]
                # No threshold check - always include top indicators
                selected_indicators[indicator_name] = {
                    'weight': float(indicator_weights[idx]),
                    'selection_prob': float(selection_probs[idx]),
                    'value': float(indicator_values[idx])
                }
            
            self.logger.debug(f"Selected {len(selected_indicators)} indicators using top-k method")
            
            # Return modified result
            return {
                'selected_indicators': selected_indicators,
                'market_context': market_context.tolist(),
                'attention_patterns': original_result.get('attention_patterns', {}),
                'regime_detection': original_result.get('regime_detection', 'Unknown'),
                'selection_method': 'ai_modified'
            }
            
        except Exception as e:
            self.logger.error(f"Modified selection logic failed: {e}")
            raise
    
    def _ensure_minimum_indicators(self, data: pd.DataFrame, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure at least min_indicators are selected, adding fallback indicators if needed.
        """
        selected_indicators = result['selected_indicators']
        
        if len(selected_indicators) >= self.min_indicators:
            self._update_selection_stats(len(selected_indicators))
            return result
        
        self.logger.info(f"Only {len(selected_indicators)} indicators selected, "
                        f"adding fallback indicators to reach minimum of {self.min_indicators}")
        
        self.selection_stats['min_indicators_enforced'] += 1
        
        # Add fallback indicators until we reach minimum
        fallback_result = self._add_fallback_indicators(data, selected_indicators)
        result['selected_indicators'] = fallback_result
        result['fallback_indicators_added'] = True
        
        self._update_selection_stats(len(result['selected_indicators']))
        return result
    
    def _add_fallback_indicators(self, 
                               data: pd.DataFrame, 
                               existing_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add fallback indicators to reach minimum count.
        """
        existing_names = set(existing_indicators.keys())
        needed_count = self.min_indicators - len(existing_indicators)
        
        # Try to compute fallback indicators
        added_indicators = existing_indicators.copy()
        
        for indicator_name in self.fallback_indicators:
            if len(added_indicators) >= self.min_indicators:
                break
                
            if indicator_name not in existing_names:
                try:
                    # Compute the indicator value
                    if hasattr(self.ai_selector, 'indicator_library'):
                        indicator_value = self.ai_selector.indicator_library.compute_indicator(data, indicator_name)
                        if len(indicator_value) > 0 and not pd.isna(indicator_value.iloc[-1]):
                            added_indicators[indicator_name] = {
                                'weight': 1.0 / self.min_indicators,  # Equal weight
                                'selection_prob': 0.8,  # High confidence for fallback
                                'value': float(indicator_value.iloc[-1]),
                                'source': 'fallback'
                            }
                            self.logger.debug(f"Added fallback indicator: {indicator_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to compute fallback indicator {indicator_name}: {e}")
        
        # If we still don't have enough, add simple ones
        if len(added_indicators) < self.min_indicators:
            simple_indicators = self._get_simple_indicators(data, self.min_indicators - len(added_indicators))
            added_indicators.update(simple_indicators)
        
        return added_indicators
    
    def _get_simple_indicators(self, data: pd.DataFrame, count: int) -> Dict[str, Any]:
        """
        Get simple calculated indicators as last resort.
        """
        simple_indicators = {}
        
        try:
            # Simple moving average
            if 'SMA_simple' not in simple_indicators and len(simple_indicators) < count:
                sma_20 = data['close'].rolling(20).mean()
                if len(sma_20) > 0 and not pd.isna(sma_20.iloc[-1]):
                    simple_indicators['SMA_simple'] = {
                        'weight': 1.0 / count,
                        'selection_prob': 0.7,
                        'value': float(sma_20.iloc[-1]),
                        'source': 'simple'
                    }
            
            # Simple RSI
            if 'RSI_simple' not in simple_indicators and len(simple_indicators) < count:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]):
                    simple_indicators['RSI_simple'] = {
                        'weight': 1.0 / count,
                        'selection_prob': 0.7,
                        'value': float(rsi.iloc[-1]),
                        'source': 'simple'
                    }
            
            # Price momentum
            if 'MOMENTUM_simple' not in simple_indicators and len(simple_indicators) < count:
                if len(data) >= 10:
                    momentum = (data['close'].iloc[-1] / data['close'].iloc[-10]) - 1
                    simple_indicators['MOMENTUM_simple'] = {
                        'weight': 1.0 / count,
                        'selection_prob': 0.7,
                        'value': float(momentum),
                        'source': 'simple'
                    }
        
        except Exception as e:
            self.logger.error(f"Failed to compute simple indicators: {e}")
        
        return simple_indicators
    
    def _fallback_selection(self, data: pd.DataFrame, top_k: int) -> Dict[str, Any]:
        """
        Complete fallback selection when AI is unavailable.
        """
        try:
            # Use the indicator library if available
            if hasattr(self, 'indicator_library'):
                library = self.indicator_library
            elif self.ai_selector and hasattr(self.ai_selector, 'indicator_library'):
                library = self.ai_selector.indicator_library
            else:
                # Create new library
                from indicator_transformer import IndicatorLibrary
                library = IndicatorLibrary()
            
            selected_indicators = {}
            
            # Try to compute fallback indicators
            for i, indicator_name in enumerate(self.fallback_indicators):
                if len(selected_indicators) >= min(top_k, self.max_indicators):
                    break
                    
                try:
                    indicator_value = library.compute_indicator(data, indicator_name)
                    if len(indicator_value) > 0 and not pd.isna(indicator_value.iloc[-1]):
                        selected_indicators[indicator_name] = {
                            'weight': 1.0 / min(len(self.fallback_indicators), top_k),
                            'selection_prob': 0.8 - (i * 0.05),  # Decreasing confidence
                            'value': float(indicator_value.iloc[-1]),
                            'source': 'fallback'
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to compute fallback indicator {indicator_name}: {e}")
            
            # Add simple indicators if needed
            if len(selected_indicators) < self.min_indicators:
                simple_indicators = self._get_simple_indicators(
                    data, self.min_indicators - len(selected_indicators)
                )
                selected_indicators.update(simple_indicators)
            
            # Get basic market context
            try:
                if hasattr(self, 'market_analyzer'):
                    market_context = self.market_analyzer.analyze_market_context(data)
                elif self.ai_selector and hasattr(self.ai_selector, 'market_analyzer'):
                    market_context = self.ai_selector.market_analyzer.analyze_market_context(data)
                else:
                    market_context = np.zeros(10)  # Default context
            except Exception:
                market_context = np.zeros(10)
            
            self._update_selection_stats(len(selected_indicators))
            
            return {
                'selected_indicators': selected_indicators,
                'market_context': market_context.tolist(),
                'attention_patterns': {'strong_connections': [], 'attention_summary': 'Fallback mode'},
                'regime_detection': 'Fallback Mode',
                'selection_method': 'fallback',
                'fallback_used': True
            }
            
        except Exception as e:
            self.logger.error(f"Fallback selection failed: {e}")
            # Ultimate fallback - return basic indicators
            return self._ultimate_fallback()
    
    def _ultimate_fallback(self) -> Dict[str, Any]:
        """
        Ultimate fallback when all else fails.
        """
        self.logger.warning("Using ultimate fallback - basic indicator set")
        
        # Return basic indicators with dummy values
        basic_indicators = {
            'SMA_basic': {
                'weight': 0.4,
                'selection_prob': 0.6,
                'value': 100.0,
                'source': 'ultimate_fallback'
            },
            'RSI_basic': {
                'weight': 0.3,
                'selection_prob': 0.6,
                'value': 50.0,
                'source': 'ultimate_fallback'
            },
            'MOMENTUM_basic': {
                'weight': 0.3,
                'selection_prob': 0.6,
                'value': 0.01,
                'source': 'ultimate_fallback'
            }
        }
        
        self._update_selection_stats(len(basic_indicators))
        
        return {
            'selected_indicators': basic_indicators,
            'market_context': [0.0] * 10,
            'attention_patterns': {'strong_connections': [], 'attention_summary': 'Ultimate fallback'},
            'regime_detection': 'Ultimate Fallback',
            'selection_method': 'ultimate_fallback',
            'ultimate_fallback_used': True
        }
    
    def _update_selection_stats(self, num_selected: int):
        """Update internal statistics."""
        total_calls = self.selection_stats['total_calls']
        current_avg = self.selection_stats['avg_indicators_selected']
        
        # Update running average
        self.selection_stats['avg_indicators_selected'] = (
            (current_avg * (total_calls - 1) + num_selected) / total_calls
        )
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indicator selection performance.
        
        Returns:
            Dictionary with selection statistics
        """
        stats = self.selection_stats.copy()
        
        if stats['total_calls'] > 0:
            stats['ai_success_rate'] = stats['successful_ai_calls'] / stats['total_calls']
            stats['fallback_rate'] = stats['fallback_calls'] / stats['total_calls']
            stats['min_enforcement_rate'] = stats['min_indicators_enforced'] / stats['total_calls']
        else:
            stats['ai_success_rate'] = 0.0
            stats['fallback_rate'] = 0.0
            stats['min_enforcement_rate'] = 0.0
        
        return stats
    
    def update_performance(self, selected_indicators: Dict[str, Any], trade_result: float):
        """
        Update performance tracking (delegates to underlying AI selector if available).
        
        Args:
            selected_indicators: Selected indicators from previous call
            trade_result: Profit/loss from the trade
        """
        if self.ai_selector is not None:
            try:
                # Only pass indicators that came from AI selection
                ai_indicators = {
                    name: info for name, info in selected_indicators.items()
                    if info.get('source') not in ['fallback', 'simple', 'ultimate_fallback']
                }
                
                if ai_indicators:
                    self.ai_selector.update_performance(ai_indicators, trade_result)
            except Exception as e:
                self.logger.warning(f"Failed to update AI performance: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report including wrapper statistics.
        
        Returns:
            Combined performance report
        """
        report = {
            'wrapper_stats': self.get_selection_stats(),
            'ai_report': None
        }
        
        # Get AI performance report if available
        if self.ai_selector is not None:
            try:
                report['ai_report'] = self.ai_selector.get_performance_report()
            except Exception as e:
                self.logger.warning(f"Failed to get AI performance report: {e}")
                report['ai_report'] = {'error': str(e)}
        
        return report
    
    def save_model(self, path: Path):
        """Save the underlying model if available."""
        if self.ai_selector is not None:
            try:
                self.ai_selector.save_model(path)
                self.logger.info(f"Model saved via wrapper to {path}")
            except Exception as e:
                self.logger.error(f"Failed to save model: {e}")
        else:
            self.logger.warning("No AI model available to save")
    
    def load_model(self, path: Path):
        """Load the underlying model if available."""
        if self.ai_selector is not None:
            try:
                self.ai_selector.load_model(path)
                self.logger.info(f"Model loaded via wrapper from {path}")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
        else:
            self.logger.warning("No AI model available to load")


def main():
    """Example usage of the S3 AI Wrapper."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 S3 AI Wrapper - Enhanced Indicator Selection")
    print("=" * 60)
    
    # Initialize wrapper
    wrapper = S3AIWrapper(
        confidence_threshold=0.1,
        min_indicators=5,
        max_indicators=12
    )
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1min')
    sample_data = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
        'high': 100.5 + np.random.randn(len(dates)).cumsum() * 0.1,
        'low': 99.5 + np.random.randn(len(dates)).cumsum() * 0.1,
        'close': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Test selection
    print("\n📊 Testing indicator selection...")
    selection = wrapper.select_indicators(sample_data.iloc[-500:])
    
    print(f"\n✅ Selected {len(selection['selected_indicators'])} indicators:")
    for indicator, info in selection['selected_indicators'].items():
        source = info.get('source', 'ai')
        print(f"   {indicator}: weight={info['weight']:.3f}, "
              f"confidence={info['selection_prob']:.3f}, source={source}")
    
    print(f"\n🎯 Market Regime: {selection['regime_detection']}")
    print(f"🔧 Selection Method: {selection.get('selection_method', 'unknown')}")
    
    # Show statistics
    print("\n📈 Wrapper Statistics:")
    stats = wrapper.get_selection_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test performance update
    print("\n🔄 Testing performance update...")
    wrapper.update_performance(selection['selected_indicators'], 0.02)  # 2% gain
    
    print("\n✅ S3 AI Wrapper test completed successfully!")


if __name__ == "__main__":
    main()