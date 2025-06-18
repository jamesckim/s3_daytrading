#!/usr/bin/env python3
"""
Run AI Backtest for All Tickers
===============================
Main script to run comprehensive AI backtests with optimized settings.
"""

import sys
import os
import json
from datetime import datetime

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main execution function."""
    # Check if running full or test mode
    import argparse
    parser = argparse.ArgumentParser(description='Run S3 AI Backtest')
    parser.add_argument('--test', action='store_true', help='Run test with limited tickers')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of tickers')
    args = parser.parse_args()
    
    # Load tickers
    tickers_file = "../tickers/all_tickers.json"
    with open(tickers_file, 'r') as f:
        all_tickers = json.load(f)
    
    # Filter tickers based on mode
    if args.test:
        # Test with high-volume stocks likely to have good signals
        test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
                       "META", "TSLA", "AMD", "SPY", "QQQ"]
        tickers = [t for t in test_tickers if t in all_tickers]
        print(f"🧪 Test mode: Running backtest for {len(tickers)} popular tickers")
    elif args.limit:
        tickers = all_tickers[:args.limit]
        print(f"🔧 Limited mode: Running backtest for first {len(tickers)} tickers")
    else:
        tickers = all_tickers
        print(f"📊 Full mode: Running backtest for all {len(tickers)} tickers")
    
    # Run backtest
    if args.parallel:
        from s3_ai_parallel_backtest import ParallelBacktester
        print("🚀 Using parallel processing")
        
        # Use more workers for full runs
        num_workers = 8 if not args.test else 4
        backtester = ParallelBacktester(num_workers=num_workers)
        backtester.run_parallel_backtest(tickers, batch_size=10)
        backtester.print_summary()
    else:
        from s3_ai_backtest_all_tickers import S3AIBacktester
        print("🔧 Using single-threaded processing")
        
        # Modify the backtester to use lower signal threshold
        class AggressiveBacktester(S3AIBacktester):
            def __init__(self, initial_capital: float = 100000):
                super().__init__(initial_capital)
                # More aggressive settings
                self.position_size = 20000  # Larger positions
                self.max_positions = 5      # Fewer, larger positions
                
            def backtest_symbol(self, symbol: str, data):
                """Override to use lower signal threshold."""
                # Temporarily store original method
                original_analyze = self.strategy.analyze_symbol
                
                # Create wrapper that adjusts signal strength
                def aggressive_analyze(sym, price):
                    action, strength, info = original_analyze(sym, price)
                    # Boost signal strength to encourage trading
                    if action != "HOLD":
                        strength = min(1.0, strength * 2.0)
                    return action, strength, info
                
                # Replace method temporarily
                self.strategy.analyze_symbol = aggressive_analyze
                
                # Run backtest
                result = super().backtest_symbol(symbol, data)
                
                # Restore original method
                self.strategy.analyze_symbol = original_analyze
                
                return result
        
        backtester = AggressiveBacktester(initial_capital=100000)
        backtester.run_backtest(tickers, save_results=True)
        backtester.print_summary()
    
    print(f"\n✅ Backtest completed at {datetime.now().strftime('%Y-%m-%d %I:%M %p PT')}")


if __name__ == "__main__":
    main()