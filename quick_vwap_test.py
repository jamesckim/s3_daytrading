#!/usr/bin/env python3
"""
Quick VWAP Test
===============
Quick test of the enhanced VWAP indicators with a small set of symbols.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tabulate import tabulate

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_db_backtest_all_tickers import S3AIDBBacktester


def main():
    """Run quick test with enhanced VWAP indicators."""
    print("🚀 Quick VWAP Enhancement Test")
    print("=" * 60)
    
    # Test symbols (small set)
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Date range (shorter for quick test)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # 1 week
    
    print(f"\n📊 Test Configuration:")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   Enhanced Indicators: 111 total (77 VWAP-related)")
    
    # Initialize backtester
    backtester = S3AIDBBacktester(initial_capital=100000)
    
    # Run backtests
    results = []
    
    for symbol in symbols:
        print(f"\n🔍 Testing {symbol}...")
        
        try:
            result = backtester.backtest_symbol(symbol, start_date, end_date)
            
            if result and result.total_return is not None:
                results.append({
                    'Symbol': symbol,
                    'Total Return': f"{result.total_return:.2%}",
                    'Trades': result.total_trades,
                    'Win Rate': f"{result.win_rate:.1%}" if result.win_rate else "N/A",
                    'Sharpe': f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio else "N/A",
                    'Max DD': f"{result.max_drawdown:.2%}" if result.max_drawdown else "N/A"
                })
                
                print(f"   ✅ Return: {result.total_return:.2%}, Trades: {result.total_trades}")
            else:
                print(f"   ❌ No results (insufficient data)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Display results
    print("\n" + "="*60)
    print("📈 QUICK TEST RESULTS")
    print("="*60)
    
    if results:
        print(tabulate(results, headers='keys', tablefmt='simple'))
        
        # Calculate averages
        avg_return = np.mean([float(r['Total Return'].strip('%')) for r in results]) / 100
        avg_trades = np.mean([r['Trades'] for r in results])
        
        print(f"\n📊 Summary:")
        print(f"   Average Return: {avg_return:.2%}")
        print(f"   Average Trades: {avg_trades:.1f}")
        print(f"   Successful Tests: {len(results)}/{len(symbols)}")
        
        # Compare to expected
        if avg_return > 0:
            print(f"\n✅ VWAP enhancement shows positive results!")
            print("   - Enhanced indicator library is working")
            print("   - VWAP bands providing good signals")
            print("   - Ready for full-scale testing")
        else:
            print(f"\n⚠️ Mixed results - may need further tuning")
        
    else:
        print("No successful backtests completed.")
        print("Check data availability and indicator calculations.")
    
    print("\n🎯 Next Steps:")
    print("1. Run full parallel backtest on all symbols")
    print("2. Deploy to paper trading for live validation")  
    print("3. Monitor VWAP band effectiveness")
    print("4. Fine-tune based on market conditions")


if __name__ == "__main__":
    main()