#!/usr/bin/env python3
"""
Fix AI Indicator Selection Issues
=================================
Diagnose and fix the issue where AI selection probabilities are too low.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_wrapper import S3AIWrapper
from indicator_transformer import AIIndicatorSelector
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider

def diagnose_ai_selection():
    """Diagnose why AI selection is failing."""
    
    print("🔍 Diagnosing AI Indicator Selection Issues...\n")
    
    # Initialize components
    wrapper = S3AIWrapper()
    
    # Get some sample data
    db = DatabaseDataProvider()
    data = db.get_minute_data('AAPL', 
                             pd.Timestamp.now() - pd.Timedelta(days=30),
                             pd.Timestamp.now())
    
    if data.empty:
        print("❌ No data available for testing")
        return
    
    # Sample last 1000 minutes
    test_data = data.tail(1000)
    
    print(f"📊 Test data: {len(test_data)} minutes of AAPL\n")
    
    # Direct access to AI selector
    ai_selector = wrapper.ai_selector
    
    # Compute indicators
    print("1️⃣ Computing all indicators...")
    indicator_values = ai_selector._compute_all_indicators(test_data)
    print(f"   ✓ Computed {len(indicator_values)} indicator values")
    
    # Analyze market context
    print("\n2️⃣ Analyzing market context...")
    market_context = ai_selector.market_analyzer.analyze_market_context(test_data)
    print(f"   ✓ Market context shape: {market_context.shape}")
    
    # Get model predictions
    print("\n3️⃣ Getting AI model predictions...")
    ai_selector.model.eval()
    
    with torch.no_grad():
        # Prepare inputs
        indicator_tensor = torch.FloatTensor(indicator_values).unsqueeze(0).to(ai_selector.device)
        context_tensor = torch.FloatTensor(market_context).unsqueeze(0).to(ai_selector.device)
        indices_tensor = torch.LongTensor(range(ai_selector.num_indicators)).unsqueeze(0).to(ai_selector.device)
        
        # Forward pass
        outputs = ai_selector.model(indicator_tensor, context_tensor, indices_tensor)
        
        # Extract results
        selection_probs = outputs['selection_probs'].squeeze().cpu().numpy()
        indicator_weights = outputs['indicator_weights'].squeeze().cpu().numpy()
    
    print(f"   ✓ Got predictions for {len(selection_probs)} indicators")
    
    # Analyze probabilities
    print("\n4️⃣ Analyzing selection probabilities:")
    print(f"   - Min probability: {selection_probs.min():.6f}")
    print(f"   - Max probability: {selection_probs.max():.6f}")
    print(f"   - Mean probability: {selection_probs.mean():.6f}")
    print(f"   - Std deviation: {selection_probs.std():.6f}")
    
    # Check against thresholds
    print("\n5️⃣ Checking against thresholds:")
    thresholds = [0.25, 0.20, 0.15, 0.10, 0.05, 0.01]
    for threshold in thresholds:
        count = np.sum(selection_probs > threshold)
        print(f"   - Indicators above {threshold:.2f}: {count}")
    
    # Show top indicators by probability
    print("\n6️⃣ Top 10 indicators by selection probability:")
    top_indices = np.argsort(selection_probs)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        indicator_name = ai_selector.indicator_names[idx]
        prob = selection_probs[idx]
        weight = indicator_weights[idx]
        print(f"   {i+1:2}. {indicator_name:25} prob={prob:.6f} weight={weight:.4f}")
    
    # Test wrapper selection
    print("\n7️⃣ Testing S3AIWrapper selection:")
    result = wrapper.select_indicators(test_data)
    selected = result.get('selected_indicators', {})
    print(f"   - Selected {len(selected)} indicators")
    if selected:
        for name, info in selected.items():
            print(f"     • {name}: weight={info.get('weight', 0):.4f}")
    
    # Recommendations
    print("\n💡 DIAGNOSIS RESULTS:")
    max_prob = selection_probs.max()
    if max_prob < 0.01:
        print("   ❌ CRITICAL: All probabilities are extremely low (<0.01)")
        print("   → The AI model may not be properly trained or loaded")
        print("   → Consider retraining or using a pre-trained model")
    elif max_prob < 0.10:
        print("   ⚠️ WARNING: All probabilities are below minimum threshold (0.10)")
        print("   → Lower the confidence thresholds significantly")
        print("   → Or use top-k selection instead of threshold-based")
    else:
        print("   ✅ Some indicators have reasonable probabilities")
        print("   → Fine-tune thresholds for better selection")
    
    return selection_probs, indicator_values

def create_fixed_wrapper():
    """Create a fixed version of S3AIWrapper with better selection logic."""
    
    class FixedS3AIWrapper(S3AIWrapper):
        """Fixed wrapper that uses top-k selection instead of thresholds."""
        
        def _apply_modified_selection_logic(self, data, top_k, original_result):
            """Use top-k selection regardless of probabilities."""
            try:
                # Get all the computed values
                indicator_values = self.ai_selector._compute_all_indicators(data)
                market_context = self.ai_selector.market_analyzer.analyze_market_context(data)
                
                # Get model predictions
                import torch
                self.ai_selector.model.eval()
                
                with torch.no_grad():
                    indicator_tensor = torch.FloatTensor(indicator_values).unsqueeze(0).to(self.ai_selector.device)
                    context_tensor = torch.FloatTensor(market_context).unsqueeze(0).to(self.ai_selector.device)
                    indices_tensor = torch.LongTensor(range(self.ai_selector.num_indicators)).unsqueeze(0).to(self.ai_selector.device)
                    
                    outputs = self.ai_selector.model(indicator_tensor, context_tensor, indices_tensor)
                    selection_probs = outputs['selection_probs'].squeeze().cpu().numpy()
                    indicator_weights = outputs['indicator_weights'].squeeze().cpu().numpy()
                
                # FIXED: Always select top-k indicators by probability
                # Don't filter by threshold
                top_indices = np.argsort(selection_probs)[-min(top_k, 15):][::-1]
                
                selected_indicators = {}
                for idx in top_indices:
                    indicator_name = self.ai_selector.indicator_names[idx]
                    selected_indicators[indicator_name] = {
                        'weight': float(indicator_weights[idx]),
                        'selection_prob': float(selection_probs[idx]),
                        'value': float(indicator_values[idx])
                    }
                
                self.logger.info(f"Selected {len(selected_indicators)} indicators using top-k method")
                
                return {
                    'selected_indicators': selected_indicators,
                    'market_context': market_context.tolist(),
                    'attention_patterns': original_result.get('attention_patterns', {}),
                    'regime_detection': original_result.get('regime_detection', 'Unknown'),
                    'selection_method': 'top_k_fixed'
                }
                
            except Exception as e:
                self.logger.error(f"Fixed selection logic failed: {e}")
                raise
    
    return FixedS3AIWrapper

if __name__ == "__main__":
    # Run diagnosis
    probs, values = diagnose_ai_selection()
    
    print("\n" + "="*60)
    print("RECOMMENDED FIXES:")
    print("="*60)
    print("1. Modify s3_ai_wrapper.py to use top-k selection instead of thresholds")
    print("2. Or drastically lower thresholds (e.g., 0.001 instead of 0.10)")
    print("3. Check if the AI model file exists and is properly loaded")
    print("4. Consider retraining the model with better data")
    print("\nThe issue is that the AI model outputs very low probabilities,")
    print("causing all indicators to be filtered out by the thresholds.")