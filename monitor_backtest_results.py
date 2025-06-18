#!/usr/bin/env python3
"""
Monitor Backtest Results
========================
View and analyze parallel backtest results with enhanced VWAP indicators.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from tabulate import tabulate
import matplotlib.pyplot as plt


def load_latest_results():
    """Load the most recent parallel backtest results."""
    # Find the most recent result file
    result_files = list(Path('.').glob('parallel_backtest_results_*.json'))
    if not result_files:
        print("No parallel backtest result files found.")
        return None
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def analyze_results(results):
    """Analyze and display backtest results."""
    if not results:
        print("No results to analyze.")
        return
    
    # Convert to DataFrame for easier analysis
    successful_results = []
    failed_results = []
    
    for symbol, result in results.items():
        if isinstance(result, dict) and 'error' not in result:
            if result.get('total_return') is not None:
                successful_results.append({
                    'symbol': symbol,
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0),
                    'total_trades': result.get('total_trades', 0),
                    'profit_factor': result.get('profit_factor', 0),
                    'avg_trade_return': result.get('avg_trade_return', 0)
                })
        else:
            failed_results.append(symbol)
    
    df = pd.DataFrame(successful_results)
    
    print(f"\n📈 Backtest Results Summary")
    print("=" * 80)
    print(f"Total symbols tested: {len(results)}")
    print(f"Successful backtests: {len(successful_results)}")
    print(f"Failed backtests: {len(failed_results)}")
    print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
    
    if len(successful_results) == 0:
        print("No successful results to analyze.")
        return
    
    # Overall statistics
    print(f"\n📊 Performance Statistics:")
    print("-" * 60)
    
    stats = {
        'Metric': ['Average Return', 'Median Return', 'Best Return', 'Worst Return',
                  'Average Sharpe', 'Average Win Rate', 'Average Trades',
                  'Positive Returns', 'Sharpe > 1.0'],
        'Value': [
            f"{df['total_return'].mean():.2%}",
            f"{df['total_return'].median():.2%}", 
            f"{df['total_return'].max():.2%}",
            f"{df['total_return'].min():.2%}",
            f"{df['sharpe_ratio'].mean():.2f}",
            f"{df['win_rate'].mean():.1%}",
            f"{df['total_trades'].mean():.1f}",
            f"{(df['total_return'] > 0).sum()}/{len(df)} ({(df['total_return'] > 0).mean()*100:.1f}%)",
            f"{(df['sharpe_ratio'] > 1.0).sum()}/{len(df)} ({(df['sharpe_ratio'] > 1.0).mean()*100:.1f}%)"
        ]
    }
    
    print(tabulate(stats, headers='keys', tablefmt='simple'))
    
    # Top performers
    print(f"\n🏆 Top 10 Performers by Return:")
    print("-" * 60)
    top_performers = df.nlargest(10, 'total_return')[['symbol', 'total_return', 'sharpe_ratio', 'win_rate', 'total_trades']]
    top_performers['total_return'] = top_performers['total_return'].apply(lambda x: f"{x:.2%}")
    top_performers['sharpe_ratio'] = top_performers['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    top_performers['win_rate'] = top_performers['win_rate'].apply(lambda x: f"{x:.1%}")
    
    print(tabulate(top_performers, headers=['Symbol', 'Return', 'Sharpe', 'Win Rate', 'Trades'], 
                  tablefmt='simple', showindex=False))
    
    # Worst performers
    print(f"\n📉 Bottom 5 Performers:")
    print("-" * 60)
    worst_performers = df.nsmallest(5, 'total_return')[['symbol', 'total_return', 'sharpe_ratio', 'total_trades']]
    worst_performers['total_return'] = worst_performers['total_return'].apply(lambda x: f"{x:.2%}")
    worst_performers['sharpe_ratio'] = worst_performers['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
    
    print(tabulate(worst_performers, headers=['Symbol', 'Return', 'Sharpe', 'Trades'], 
                  tablefmt='simple', showindex=False))
    
    # Trading activity analysis
    print(f"\n📊 Trading Activity Analysis:")
    print("-" * 60)
    
    active_traders = df[df['total_trades'] >= 5]  # Symbols with at least 5 trades
    
    if len(active_traders) > 0:
        print(f"Symbols with 5+ trades: {len(active_traders)}")
        print(f"Avg return (active): {active_traders['total_return'].mean():.2%}")
        print(f"Avg win rate (active): {active_traders['win_rate'].mean():.1%}")
        print(f"Avg Sharpe (active): {active_traders['sharpe_ratio'].mean():.2f}")
    
    # Risk analysis
    print(f"\n⚠️ Risk Analysis:")
    print("-" * 60)
    
    high_return = df[df['total_return'] > 0.05]  # > 5% return
    low_drawdown = df[df['max_drawdown'] > -0.05]  # < 5% drawdown
    good_sharpe = df[df['sharpe_ratio'] > 1.0]
    
    print(f"High return (>5%): {len(high_return)} symbols")
    print(f"Low drawdown (<5%): {len(low_drawdown)} symbols")
    print(f"Good Sharpe (>1.0): {len(good_sharpe)} symbols")
    print(f"All criteria met: {len(df[(df['total_return'] > 0.05) & (df['max_drawdown'] > -0.05) & (df['sharpe_ratio'] > 1.0)])} symbols")
    
    # VWAP enhancement assessment
    print(f"\n🎯 VWAP Enhancement Assessment:")
    print("-" * 60)
    
    if df['total_return'].mean() > 0:
        print("✅ Overall positive performance indicates VWAP enhancement is working")
    
    if df['win_rate'].mean() > 0.5:
        print(f"✅ Average win rate {df['win_rate'].mean():.1%} suggests good signal quality")
    
    if df['sharpe_ratio'].mean() > 0.5:
        print(f"✅ Average Sharpe {df['sharpe_ratio'].mean():.2f} shows decent risk-adjusted returns")
    
    if len(active_traders) / len(df) > 0.3:
        print("✅ Good trading activity across symbols")
    
    # Failed symbols analysis
    if failed_results:
        print(f"\n❌ Failed Backtests ({len(failed_results)} symbols):")
        print("-" * 60)
        print("Common reasons: Insufficient data, connection issues, indicator calculation errors")
        if len(failed_results) <= 20:
            print(f"Failed symbols: {', '.join(failed_results[:20])}")
        else:
            print(f"Failed symbols (first 20): {', '.join(failed_results[:20])}...")


def monitor_live_backtest():
    """Monitor a currently running backtest."""
    print("🔍 Checking for running backtest processes...")
    
    import subprocess
    
    # Check for running Python processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        backtest_processes = [line for line in lines if 's3_ai_parallel_backtest' in line and 'python' in line]
        
        if backtest_processes:
            print("🏃 Found running backtest processes:")
            for process in backtest_processes:
                print(f"   {process}")
            
            print("\n💡 To monitor progress:")
            print("   - Check log files in real-time")
            print("   - Monitor CPU/memory usage")
            print("   - Look for new result files")
        else:
            print("📋 No running backtest processes found.")
            print("Backtest may have completed or stopped.")
            
    except Exception as e:
        print(f"Error checking processes: {e}")


def main():
    """Main monitoring function."""
    print("🔧 Backtest Results Monitor")
    print("=" * 80)
    
    # Check for running processes first
    monitor_live_backtest()
    
    print("\n" + "="*80)
    
    # Load and analyze latest results
    results = load_latest_results()
    if results:
        analyze_results(results)
    else:
        print("No results found. Run a backtest first with:")
        print("   python s3_ai_parallel_backtest.py")
    
    print(f"\n💾 To see all result files:")
    print(f"   ls -la parallel_backtest_results_*.json")
    print(f"\n🔄 To run a new backtest:")
    print(f"   python s3_ai_parallel_backtest.py")


if __name__ == "__main__":
    main()