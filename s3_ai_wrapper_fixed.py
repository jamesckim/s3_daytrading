#!/usr/bin/env python3
"""
S3 AI Wrapper - Fixed Version
=============================
Fixed version that actually selects indicators instead of always
falling back to defaults. Uses top-k selection to ensure indicators
are selected regardless of low probabilities.
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

class S3AIWrapperFixed:
    """
    Fixed wrapper that ensures indicators are actually selected.
    Uses top-k selection instead of probability thresholds.
    """
    
    def __init__(self, 
                 model_path: Optional[Path] = None,
                 device: str = 'cpu',
                 min_indicators: int = 5,  # Increased from 3
                 max_indicators: int = 15,
                 fallback_indicators: Optional[List[str]] = None):
        """
        Initialize the fixed S3 AI wrapper.
        
        Args:
            model_path: Path to the trained model
            device: Device to run model on ('cpu' or 'cuda')
            min_indicators: Minimum indicators to select (default: 5)
            max_indicators: Maximum indicators to select (default: 15)
            fallback_indicators: List of fallback indicators
        """
        self.min_indicators = min_indicators
        self.max_indicators = max_indicators
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Try to initialize AI selector
        try:
            self.ai_selector = AIIndicatorSelector(model_path=model_path, device=device)
            self.logger.info("AI Indicator Selector loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load AI selector: {e}")
            self.ai_selector = None
        
        # Default fallback indicators
        if fallback_indicators is None:
            self.fallback_indicators = [
                'RSI_14', 'RSI_30', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER',
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'ATR_14',
                'SUPPORT_RESISTANCE', 'VWAP', 'VOLUME_RATIO'
            ]
        else:
            self.fallback_indicators = fallback_indicators
        
        self.logger.info(f"S3 AI Wrapper Fixed initialized with min={min_indicators}, max={max_indicators}")
    
    def select_indicators(self, 
                         symbol: str,
                         data: pd.DataFrame,
                         top_k: Optional[int] = None) -> Tuple[List[str], Dict[str, float], float]:
        """
        Select optimal indicators using top-k method.
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            top_k: Number of indicators to select
            
        Returns:
            Tuple of (selected_indicators, weights, confidence)
        """
        if top_k is None:
            top_k = min(self.max_indicators, 10)  # Default to 10
        
        # Ensure we select at least min_indicators
        top_k = max(top_k, self.min_indicators)
        
        if self.ai_selector is not None:
            try:
                # Get AI predictions
                selected, weights, confidence = self._ai_select_top_k(data, top_k)
                
                if len(selected) >= self.min_indicators:
                    self.logger.info(f"Selected {len(selected)} indicators for {symbol} using AI (top-k)")
                    return selected, weights, confidence
                else:
                    self.logger.warning(f"AI selected only {len(selected)} indicators, adding fallbacks")
                    
            except Exception as e:
                self.logger.error(f"AI selection failed: {e}")
        
        # Fallback selection
        return self._fallback_selection(data, top_k)
    
    def _ai_select_top_k(self, data: pd.DataFrame, top_k: int) -> Tuple[List[str], Dict[str, float], float]:
        """
        Select top-k indicators by AI model probability.
        """
        # Compute all indicators
        indicator_values = self.ai_selector._compute_all_indicators(data)
        market_context = self.ai_selector.market_analyzer.analyze_market_context(data)
        
        # Get model predictions
        import torch
        self.ai_selector.model.eval()
        
        with torch.no_grad():
            indicator_tensor = torch.FloatTensor(indicator_values).unsqueeze(0).to(self.device)
            context_tensor = torch.FloatTensor(market_context).unsqueeze(0).to(self.device)
            indices_tensor = torch.LongTensor(range(self.ai_selector.num_indicators)).unsqueeze(0).to(self.device)
            
            outputs = self.ai_selector.model(indicator_tensor, context_tensor, indices_tensor)
            selection_probs = outputs['selection_probs'].squeeze().cpu().numpy()
            indicator_weights = outputs['indicator_weights'].squeeze().cpu().numpy()
        
        # Select top-k indicators by probability (no threshold)
        top_indices = np.argsort(selection_probs)[-top_k:][::-1]
        
        selected_indicators = []
        weights_dict = {}
        
        for idx in top_indices:
            indicator_name = self.ai_selector.indicator_names[idx]
            selected_indicators.append(indicator_name)
            
            # Normalize weights to sum to 1
            weight = max(0.05, float(indicator_weights[idx]))  # Minimum weight of 0.05
            weights_dict[indicator_name] = weight
        
        # Normalize weights
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            weights_dict = {k: v/total_weight for k, v in weights_dict.items()}
        
        # Average confidence of selected indicators
        confidence = float(np.mean(selection_probs[top_indices]))
        
        return selected_indicators, weights_dict, confidence
    
    def _fallback_selection(self, data: pd.DataFrame, top_k: int) -> Tuple[List[str], Dict[str, float], float]:
        """
        Fallback indicator selection when AI is not available.
        """
        # Select diverse indicators
        selected = []
        weights = {}
        
        # Priority groups
        priority_groups = [
            ['RSI_14', 'RSI_30'],           # Momentum
            ['MACD_SIGNAL', 'MACD'],        # Trend
            ['BB_UPPER', 'BB_LOWER'],       # Volatility
            ['SMA_20', 'SMA_50', 'EMA_12'], # Moving averages
            ['ATR_14', 'ATR_30'],           # Volatility
            ['VWAP', 'VOLUME_RATIO'],       # Volume
            ['SUPPORT_RESISTANCE']          # S/R
        ]
        
        # Select from each group
        for group in priority_groups:
            if len(selected) >= top_k:
                break
            for indicator in group:
                if len(selected) < top_k and indicator not in selected:
                    selected.append(indicator)
                    weights[indicator] = 1.0 / top_k
        
        # Fill remaining with any fallbacks
        for indicator in self.fallback_indicators:
            if len(selected) >= top_k:
                break
            if indicator not in selected:
                selected.append(indicator)
                weights[indicator] = 1.0 / top_k
        
        return selected, weights, 0.5  # Default confidence
    
    def analyze_intraday(self, symbol: str, data: pd.DataFrame, 
                        current_time: pd.Timestamp) -> Tuple[str, float, Dict]:
        """
        Analyze intraday data for trading signals.
        
        Returns:
            Tuple of (action, signal_strength, info)
        """
        # Get recent data window
        window_start = current_time - pd.Timedelta(minutes=60)
        recent_data = data[window_start:current_time]
        
        if len(recent_data) < 20:
            return "HOLD", 0.0, {"reason": "Insufficient data"}
        
        # Select indicators
        indicators, weights, confidence = self.select_indicators(symbol, recent_data)
        
        # Compute indicator values
        indicator_values = {}
        indicator_library = IndicatorLibrary()
        
        for indicator in indicators:
            try:
                value = indicator_library.compute_indicator(recent_data, indicator)
                if len(value) > 0 and not pd.isna(value.iloc[-1]):
                    indicator_values[indicator] = float(value.iloc[-1])
            except:
                continue
        
        # Simple signal aggregation
        signal_strength = 0.0
        
        for indicator, value in indicator_values.items():
            weight = weights.get(indicator, 0.1)
            
            # Simple rules (can be enhanced)
            if 'RSI' in indicator:
                if value < 30:
                    signal_strength += weight * 0.5
                elif value > 70:
                    signal_strength -= weight * 0.5
            elif 'MACD' in indicator:
                if value > 0:
                    signal_strength += weight * 0.3
                else:
                    signal_strength -= weight * 0.3
            # Add more rules...
        
        # Determine action
        if signal_strength > 0.3:
            action = "BUY"
        elif signal_strength < -0.3:
            action = "SELL"
        else:
            action = "HOLD"
        
        info = {
            'indicators': indicator_values,
            'selected_indicators': indicators,
            'weights': weights,
            'confidence': confidence,
            'signal_components': {ind: weights.get(ind, 0) for ind in indicators}
        }
        
        return action, signal_strength, info