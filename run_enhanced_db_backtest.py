#!/usr/bin/env python3
"""
Run Enhanced Database Backtest
==============================
Uses the improved minute-level strategy for better trade generation.
"""

import sys
import os
import json
from datetime import datetime, timedelta

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_minute_strategy_v2 import EnhancedS3AIDBBacktester
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider
import pandas as pd
from tabulate import tabulate

def main():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser(description='Run Enhanced S3 AI Database Backtest')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of tickers')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = EnhancedS3AIDBBacktester(initial_capital=args.capital)
    
    # Get available symbols
    db_provider = DatabaseDataProvider()
    db_symbols = db_provider.get_available_symbols()
    
    # Load all tickers
    tickers_file = "../tickers/all_tickers.json"
    with open(tickers_file, 'r') as f:
        all_tickers = json.load(f)
    
    # Filter to available symbols
    available_tickers = [t for t in all_tickers if t in db_symbols]
    print(f"📊 Found {len(available_tickers)} tickers with data in database")
    
    # Apply limit
    if args.limit:
        # Select diverse tickers including high-volume ones
        priority_tickers = ["AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "META", "GOOGL", 
                           "SPY", "QQQ", "AMD", "NFLX", "BRK-B", "JPM", "BAC", "XOM"]
        selected = []
        
        # Add priority tickers first
        for ticker in priority_tickers:
            if ticker in available_tickers and len(selected) < args.limit:
                selected.append(ticker)
        
        # Fill remaining with other tickers
        for ticker in available_tickers:
            if ticker not in selected and len(selected) < args.limit:
                selected.append(ticker)
        
        available_tickers = selected
        print(f"🔧 Testing {len(available_tickers)} tickers: {', '.join(available_tickers[:10])}...")
    
    # Set backtest period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    print(f"📅 Backtest period: {start_date.date()} to {end_date.date()}")
    print(f"💰 Initial capital: ${args.capital:,.0f}")
    print("-" * 80)
    
    # Run backtests
    results = []
    total_trades = 0
    
    for i, symbol in enumerate(available_tickers):
        if i % 5 == 0 and i > 0:
            print(f"\n📊 Progress: {i}/{len(available_tickers)} ({i/len(available_tickers)*100:.1f}%)")
        
        result = backtester.backtest_symbol(symbol, start_date, end_date)
        
        if not result.error and result.total_trades > 0:
            results.append({
                'Symbol': symbol,
                'Return %': result.total_return_pct,
                'Trades': result.total_trades,
                'Win Rate %': result.win_rate,
                'Avg Hold (min)': result.average_hold_time,
                'Sharpe': result.sharpe_ratio,
                'Max DD %': result.max_drawdown_pct
            })
            total_trades += result.total_trades
    
    # Sort by return
    results = sorted(results, key=lambda x: x['Return %'], reverse=True)
    
    # Print results
    print("\n" + "=" * 80)
    print("ENHANCED MINUTE STRATEGY BACKTEST RESULTS")
    print("=" * 80)
    
    if results:
        # Top performers
        print("\n🏆 TOP 10 PERFORMERS:")
        top_10 = pd.DataFrame(results[:10])
        print(tabulate(top_10, headers='keys', floatfmt='.2f', tablefmt='simple'))
        
        # Bottom performers
        if len(results) > 10:
            print("\n📉 BOTTOM 10 PERFORMERS:")
            bottom_10 = pd.DataFrame(results[-10:])
            print(tabulate(bottom_10, headers='keys', floatfmt='.2f', tablefmt='simple'))
        
        # Summary statistics
        df = pd.DataFrame(results)
        print("\n📊 SUMMARY STATISTICS:")
        print(f"Total Symbols Tested: {len(available_tickers)}")
        print(f"Symbols with Trades: {len(results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Average Return: {df['Return %'].mean():.2f}%")
        print(f"Median Return: {df['Return %'].median():.2f}%")
        print(f"Best Return: {df['Return %'].max():.2f}% ({results[0]['Symbol']})")
        print(f"Worst Return: {df['Return %'].min():.2f}% ({results[-1]['Symbol']})")
        print(f"Profitable Symbols: {len(df[df['Return %'] > 0])} ({len(df[df['Return %'] > 0])/len(df)*100:.1f}%)")
        print(f"Average Win Rate: {df['Win Rate %'].mean():.1f}%")
        print(f"Average Trades per Symbol: {df['Trades'].mean():.1f}")
        print(f"Average Hold Time: {df['Avg Hold (min)'].mean():.1f} minutes")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_db_backtest_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to {results_file}")
    else:
        print("No results to display")
    
    print(f"\n✅ Backtest completed at {datetime.now().strftime('%Y-%m-%d %I:%M %p PT')}")


if __name__ == "__main__":
    main()