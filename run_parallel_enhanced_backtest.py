#!/usr/bin/env python3
"""
Run Parallel Enhanced Database Backtest
=======================================
Parallel version for faster execution.
"""

import sys
import os
import json
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import pandas as pd
from tabulate import tabulate

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_minute_strategy_v2 import EnhancedS3AIDBBacktester
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider


def process_symbol(args):
    """Process a single symbol."""
    symbol, start_date, end_date, initial_capital = args
    
    try:
        # Create new backtester instance for this process
        backtester = EnhancedS3AIDBBacktester(initial_capital=initial_capital)
        result = backtester.backtest_symbol(symbol, start_date, end_date)
        
        if not result.error and result.total_trades > 0:
            return {
                'Symbol': symbol,
                'Return %': result.total_return_pct,
                'Trades': result.total_trades,
                'Win Rate %': result.win_rate,
                'Avg Hold (min)': result.average_hold_time,
                'Sharpe': result.sharpe_ratio,
                'Max DD %': result.max_drawdown_pct,
                'Final Capital': result.final_capital
            }
        return None
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None


def main():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser(description='Run Parallel Enhanced Backtest')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of tickers')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--workers', type=int, default=cpu_count()-1, help='Number of parallel workers')
    args = parser.parse_args()
    
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
        # Select diverse tickers
        priority_tickers = ["AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "META", "GOOGL", 
                           "SPY", "QQQ", "AMD", "NFLX", "BRK-B", "JPM", "BAC", "XOM",
                           "AAOI", "ABNB", "ABR", "ABT", "ACAD"]  # Include AAOI since it performed well
        selected = []
        
        for ticker in priority_tickers:
            if ticker in available_tickers and len(selected) < args.limit:
                selected.append(ticker)
        
        for ticker in available_tickers:
            if ticker not in selected and len(selected) < args.limit:
                selected.append(ticker)
        
        available_tickers = selected
        print(f"🔧 Testing {len(available_tickers)} tickers")
    
    # Set backtest period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    print(f"📅 Backtest period: {start_date.date()} to {end_date.date()}")
    print(f"💰 Initial capital: ${args.capital:,.0f}")
    print(f"🔧 Using {args.workers} parallel workers")
    print("-" * 80)
    
    # Prepare arguments for parallel processing
    process_args = [(symbol, start_date, end_date, args.capital) 
                    for symbol in available_tickers]
    
    # Run parallel backtest
    start_time = datetime.now()
    print(f"🚀 Starting parallel backtest at {start_time.strftime('%I:%M %p PT')}")
    
    with Pool(processes=args.workers) as pool:
        results = pool.map(process_symbol, process_args)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Sort by return
    results = sorted(results, key=lambda x: x['Return %'], reverse=True)
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Print results
    print("\n" + "=" * 80)
    print("PARALLEL ENHANCED BACKTEST RESULTS")
    print("=" * 80)
    
    if results:
        # Top performers
        print("\n🏆 TOP 20 PERFORMERS:")
        top_20 = pd.DataFrame(results[:20])
        print(tabulate(top_20, headers='keys', floatfmt='.2f', tablefmt='simple'))
        
        # Summary statistics
        df = pd.DataFrame(results)
        total_trades = df['Trades'].sum()
        
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
        
        # Portfolio simulation
        total_capital = len(results) * args.capital
        final_capital = df['Final Capital'].sum()
        portfolio_return = (final_capital - total_capital) / total_capital * 100
        print(f"\n💼 PORTFOLIO SIMULATION:")
        print(f"If trading all {len(results)} symbols equally:")
        print(f"Total Initial Capital: ${total_capital:,.0f}")
        print(f"Total Final Capital: ${final_capital:,.0f}")
        print(f"Portfolio Return: {portfolio_return:.2f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"parallel_enhanced_backtest_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to {results_file}")
    else:
        print("No results to display")
    
    print(f"\n⏱️ Execution time: {execution_time:.1f} seconds")
    print(f"✅ Backtest completed at {datetime.now().strftime('%Y-%m-%d %I:%M %p PT')}")


if __name__ == "__main__":
    main()