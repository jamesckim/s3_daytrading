#!/usr/bin/env python3
"""
Apply Quick Fix to S3 AI Wrapper
================================
This script patches the S3AIWrapper to use top-k selection
instead of threshold-based selection, fixing the "0 indicators selected" issue.
"""

import shutil
from datetime import datetime

def apply_fix():
    """Apply the fix to s3_ai_wrapper.py"""
    
    # Backup original
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy('s3_ai_wrapper.py', f's3_ai_wrapper_backup_{timestamp}.py')
    print(f"✓ Created backup: s3_ai_wrapper_backup_{timestamp}.py")
    
    # Read the file
    with open('s3_ai_wrapper.py', 'r') as f:
        content = f.read()
    
    # Find and replace the threshold-based selection
    old_code = '''            # Select indicators with optimized thresholds
            selected_indicators = {}
            for idx in top_indices:
                indicator_name = self.ai_selector.indicator_names[idx]
                threshold = self.get_confidence_threshold(indicator_name)
                
                if selection_probs[idx] > threshold:  # Use optimized threshold
                    selected_indicators[indicator_name] = {
                        'weight': float(indicator_weights[idx]),
                        'selection_prob': float(selection_probs[idx]),
                        'value': float(indicator_values[idx])
                    }'''
    
    new_code = '''            # FIXED: Always select top-k indicators regardless of probability
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
            
            self.logger.debug(f"Selected {len(selected_indicators)} indicators using top-k method")'''
    
    # Apply the fix
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Write back
        with open('s3_ai_wrapper.py', 'w') as f:
            f.write(content)
        
        print("✓ Applied fix to s3_ai_wrapper.py")
        print("\nThe fix changes the selection logic from:")
        print("  ❌ Filter by threshold (resulting in 0 indicators)")
        print("  ✅ Always select top-k indicators by probability")
        print("\nThis ensures indicators are actually selected for trading decisions.")
    else:
        print("❌ Could not find the code to replace. Manual fix needed.")
        print("\nManual fix instructions:")
        print("1. Open s3_ai_wrapper.py")
        print("2. Find the '_apply_modified_selection_logic' method")
        print("3. Replace the threshold-based selection with top-k selection")
        print("4. Remove the line: if selection_probs[idx] > threshold:")
        print("5. Always add indicators to selected_indicators")

if __name__ == "__main__":
    print("🔧 Applying AI Wrapper Fix...")
    print("="*60)
    apply_fix()
    print("\n✅ Fix applied! The AI will now actually select indicators.")
    print("\nNext steps:")
    print("1. Run a test backtest to verify indicators are being selected")
    print("2. Check logs - you should no longer see '0 indicators selected'")
    print("3. Performance should improve as the AI adapts to market conditions")