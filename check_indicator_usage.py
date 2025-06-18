#!/usr/bin/env python3
"""
Check Indicator Usage
====================
Check which indicators are being selected by the AI.
"""

import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_wrapper import S3AIWrapper
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider

def main():
    """Check indicator usage patterns."""
    print("Checking Indicator Usage Patterns")
    print("=" * 60)
    
    # Initialize components
    wrapper = S3AIWrapper()
    db_provider = DatabaseDataProvider()
    
    # Test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Test symbols
    test_symbols = ["AAPL", "TSLA", "SPY", "AAOI", "META", "NVDA"]
    
    # Track indicator usage
    indicator_usage = defaultdict(int)
    indicator_confidence = defaultdict(list)
    new_ema_usage = defaultdict(int)
    
    print("\nTesting indicator selection for each symbol...")
    print("-" * 60)
    
    for symbol in test_symbols:
        print(f"\n{symbol}:")
        
        try:
            # Get minute data
            data = db_provider.get_minute_data(symbol, start_date, end_date)
            
            if not data.empty and len(data) > 100:
                # Test multiple time windows
                for i in range(3):
                    # Get different time slices
                    if i == 0:
                        test_data = data.head(100)
                    elif i == 1:
                        test_data = data[len(data)//2:len(data)//2+100]
                    else:
                        test_data = data.tail(100)
                    
                    # Get indicator selection
                    selection = wrapper.select_indicators(test_data)
                    
                    # Track usage
                    for ind_name, info in selection['selected_indicators'].items():
                        indicator_usage[ind_name] += 1
                        indicator_confidence[ind_name].append(info['selection_prob'])
                        
                        # Check if it's a new EMA
                        if ind_name.startswith('EMA_'):
                            period = int(ind_name.split('_')[1])
                            if period in [1, 2, 3, 5, 7, 100, 150, 300, 500, 1000]:
                                new_ema_usage[ind_name] += 1
                
                # Show selected indicators for this symbol
                last_selection = selection['selected_indicators']
                ema_selected = [name for name in last_selection.keys() if name.startswith('EMA_')]
                if ema_selected:
                    print(f"  EMAs selected: {', '.join(ema_selected)}")
                print(f"  Total indicators: {len(last_selection)}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("INDICATOR USAGE SUMMARY")
    print("=" * 60)
    
    # Sort by usage
    sorted_usage = sorted(indicator_usage.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Most Used Indicators:")
    for i, (ind_name, count) in enumerate(sorted_usage[:20]):
        avg_conf = sum(indicator_confidence[ind_name]) / len(indicator_confidence[ind_name])
        print(f"{i+1:2}. {ind_name:15} - Used {count:3} times, Avg confidence: {avg_conf:.3f}")
    
    # EMA analysis
    print("\n" + "-" * 60)
    print("EMA USAGE ANALYSIS:")
    
    # Original EMAs
    original_emas = ['EMA_10', 'EMA_20', 'EMA_50', 'EMA_200']
    new_short_emas = ['EMA_1', 'EMA_2', 'EMA_3', 'EMA_5', 'EMA_7']
    new_long_emas = ['EMA_100', 'EMA_150', 'EMA_300', 'EMA_500', 'EMA_1000']
    
    orig_usage = sum(indicator_usage.get(ema, 0) for ema in original_emas)
    short_usage = sum(indicator_usage.get(ema, 0) for ema in new_short_emas)
    long_usage = sum(indicator_usage.get(ema, 0) for ema in new_long_emas)
    
    print(f"\nOriginal EMAs (10,20,50,200): {orig_usage} uses")
    for ema in original_emas:
        if ema in indicator_usage:
            print(f"  {ema}: {indicator_usage[ema]} times")
    
    print(f"\nNew Short EMAs (1,2,3,5,7): {short_usage} uses")
    for ema in new_short_emas:
        if ema in indicator_usage:
            print(f"  {ema}: {indicator_usage[ema]} times")
    
    print(f"\nNew Long EMAs (100,150,300,500,1000): {long_usage} uses")
    for ema in new_long_emas:
        if ema in indicator_usage:
            print(f"  {ema}: {indicator_usage[ema]} times")
    
    # Usage ratio
    total_ema_usage = orig_usage + short_usage + long_usage
    if total_ema_usage > 0:
        print(f"\nEMA Usage Distribution:")
        print(f"  Original: {orig_usage/total_ema_usage*100:.1f}%")
        print(f"  New Short: {short_usage/total_ema_usage*100:.1f}%")
        print(f"  New Long: {long_usage/total_ema_usage*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    main()