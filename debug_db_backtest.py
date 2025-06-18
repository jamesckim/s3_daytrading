#!/usr/bin/env python3
"""
Debug Database Backtest
======================
Debug why trades aren't being generated.
"""

import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from s3_ai_db_backtest_all_tickers import S3AIDBBacktester, DatabaseDataProvider, S3AIMinuteStrategy
from s3_ai_wrapper import S3AIWrapper

def main():
    # Initialize components
    provider = DatabaseDataProvider()
    ai_wrapper = S3AIWrapper()
    strategy = S3AIMinuteStrategy(ai_wrapper)
    
    # Get data for AAPL
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    print(f"Fetching data for {symbol}...")
    data = provider.get_minute_data(symbol, start_date, end_date)
    print(f"Data shape: {data.shape}")
    
    if not data.empty:
        # Test strategy at different points
        test_times = data.index[100:110]  # Test 10 different times
        
        for test_time in test_times:
            print(f"\n--- Testing at {test_time} ---")
            
            # Get action and signal
            action, signal_strength, info = strategy.analyze_intraday(symbol, data, test_time)
            
            print(f"Action: {action}")
            print(f"Signal Strength: {signal_strength:.4f}")
            print(f"Regime: {info.get('regime', 'Unknown')}")
            print(f"Time of Day: {info.get('time_of_day', 'Unknown')}")
            
            # Check what indicators were selected
            indicators = info.get('indicators', {})
            if indicators:
                print("Selected Indicators:")
                for ind_name, ind_info in list(indicators.items())[:3]:
                    print(f"  {ind_name}: value={ind_info.get('value', 'N/A')}, weight={ind_info.get('weight', 0):.3f}")
            
            # If signal strength is close to threshold, show why
            if 0.3 < signal_strength < 0.5:
                print(f"Signal close to threshold (0.4) - might trade with aggressive settings")

if __name__ == "__main__":
    main()