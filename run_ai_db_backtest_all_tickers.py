#!/usr/bin/env python3
"""
Run AI Database Backtest for All Tickers
========================================
Main script to run comprehensive AI backtests using PostgreSQL data.
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_db_backtest_all_tickers import S3AIDBBacktester, DatabaseDataProvider

def main():
    """Main execution function."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run S3 AI Database Backtest')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of tickers')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive trading parameters')
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = S3AIDBBacktester(initial_capital=100000)
    
    # Get available symbols from database
    db_provider = DatabaseDataProvider()
    db_symbols = db_provider.get_available_symbols()
    
    # Load all tickers and filter by what's in database
    tickers_file = "../tickers/all_tickers.json"
    with open(tickers_file, 'r') as f:
        all_tickers = json.load(f)
    
    # Filter to available symbols
    available_tickers = [t for t in all_tickers if t in db_symbols]
    print(f"📊 Found {len(available_tickers)} tickers with data in database")
    
    # Apply limit if specified
    if args.limit:
        available_tickers = available_tickers[:args.limit]
        print(f"🔧 Limited to {len(available_tickers)} tickers")
    
    # Use aggressive settings if specified
    if args.aggressive:
        print("🚀 Using aggressive trading parameters")
        backtester.position_size_pct = 0.05  # 5% positions
        backtester.max_hold_minutes = 60  # 1 hour max hold
        backtester.stop_loss = 0.015  # 1.5% stop loss
        backtester.take_profit = 0.01  # 1% take profit
        
        # Modify strategy thresholds
        class AggressiveStrategy:
            def __init__(self, original_strategy):
                self.original = original_strategy
                self.ai_wrapper = original_strategy.ai_wrapper
                self.regime_cache = original_strategy.regime_cache
                self.indicator_cache = original_strategy.indicator_cache
                self.eastern_tz = original_strategy.eastern_tz
                
            def analyze_intraday(self, symbol, data, current_time):
                action, strength, info = self.original.analyze_intraday(symbol, data, current_time)
                # Boost signal strength to encourage more trades
                if action != "HOLD" and strength > 0.2:
                    strength = min(1.0, strength * 2.5)
                return action, strength, info
                
            def prepare_data_for_ai(self, data):
                return self.original.prepare_data_for_ai(data)
        
        backtester.strategy = AggressiveStrategy(backtester.strategy)
    
    # Set backtest period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    print(f"📅 Backtest period: {start_date.date()} to {end_date.date()}")
    
    # Run backtest
    backtester.run_backtest(available_tickers, start_date, end_date, save_results=True)
    
    # Print summary
    backtester.print_summary()
    
    # Print additional stats
    if hasattr(backtester, 'results'):
        total_trades = sum(r.total_trades for r in backtester.results.values() if not r.error)
        print(f"\n📊 Total trades across all symbols: {total_trades}")
        
        # Find most active symbols
        active_symbols = sorted(
            [(s, r.total_trades) for s, r in backtester.results.items() if r.total_trades > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        if active_symbols:
            print("\n🔥 Most active symbols:")
            for symbol, trades in active_symbols[:10]:
                result = backtester.results[symbol]
                print(f"  {symbol}: {trades} trades, {result.total_return_pct:.2f}% return")
    
    print(f"\n✅ Backtest completed at {datetime.now().strftime('%Y-%m-%d %I:%M %p PT')}")


if __name__ == "__main__":
    main()