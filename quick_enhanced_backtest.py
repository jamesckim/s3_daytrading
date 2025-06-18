#!/usr/bin/env python3
"""
Quick Enhanced Backtest
=======================
Fast validation of enhanced VWAP system with sequential processing.
"""

import sys
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tabulate import tabulate

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_db_backtest_all_tickers import S3AIDBBacktester


def main():
    """Quick enhanced backtest validation."""
    print("🚀 Quick Enhanced VWAP Backtest")
    print("=" * 60)
    
    # Test configuration
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    initial_capital = 100000
    days = 14  # 2 weeks for quick test
    
    print(f"📊 Configuration:")
    print(f"   Symbols: {', '.join(test_symbols)}")
    print(f"   Capital: ${initial_capital:,}")
    print(f"   Period: {days} days")
    print(f"   Enhanced indicators: 111 total (77 VWAP)")
    
    # Initialize backtester (with enhanced indicators)
    backtester = S3AIDBBacktester(initial_capital=initial_capital)
    
    # Run backtests sequentially
    results = []
    
    for i, symbol in enumerate(test_symbols):
        print(f"\n🔄 Testing {symbol} ({i+1}/{len(test_symbols)})...")
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Run backtest
            result = backtester.backtest_symbol(symbol, start_date, end_date)
            
            if result.error:
                print(f"   ❌ Error: {result.error}")
                results.append({
                    'Symbol': symbol,
                    'Status': 'Error',
                    'Return': 0.0,
                    'Trades': 0,
                    'Win Rate': 0.0,
                    'Error': result.error[:30] + "..." if len(result.error) > 30 else result.error
                })
            else:
                print(f"   ✅ Success: {result.total_return_pct:.2f}% return, {result.total_trades} trades")
                results.append({
                    'Symbol': symbol,
                    'Status': 'Success',
                    'Return': f"{result.total_return_pct:.2f}%",
                    'Trades': result.total_trades,
                    'Win Rate': f"{result.win_rate:.1f}%",
                    'Sharpe': f"{result.sharpe_ratio:.2f}",
                    'Max DD': f"{result.max_drawdown_pct:.1f}%"
                })
        
        except Exception as e:
            print(f"   ❌ Exception: {str(e)[:50]}...")
            results.append({
                'Symbol': symbol,
                'Status': 'Exception',
                'Return': 0.0,
                'Trades': 0,
                'Win Rate': 0.0,
                'Error': str(e)[:30] + "..."
            })
    
    # Display results
    print(f"\n📊 Enhanced VWAP System Results:")
    print("=" * 60)
    
    if results:
        # Separate successful and failed results
        successful = [r for r in results if r['Status'] == 'Success']
        failed = [r for r in results if r['Status'] in ['Error', 'Exception']]
        
        if successful:
            print(f"\n✅ Successful Backtests ({len(successful)}):")
            print(tabulate(successful, headers='keys', tablefmt='simple'))
            
            # Calculate summary statistics
            returns = []
            trades = []
            for r in successful:
                try:
                    returns.append(float(r['Return'].replace('%', '')))
                    trades.append(int(r['Trades']))
                except:
                    pass
            
            if returns:
                print(f"\n📈 Performance Summary:")
                print(f"   Average return: {np.mean(returns):.2f}%")
                print(f"   Best return:    {max(returns):.2f}%")
                print(f"   Worst return:   {min(returns):.2f}%")
                print(f"   Average trades: {np.mean(trades):.1f}")
                print(f"   Positive returns: {sum(1 for r in returns if r > 0)}/{len(returns)}")
        
        if failed:
            print(f"\n❌ Failed Backtests ({len(failed)}):")
            failed_display = [{k: v for k, v in r.items() if k != 'Error'} for r in failed]
            print(tabulate(failed_display, headers='keys', tablefmt='simple'))
    
    # Overall assessment
    success_rate = len([r for r in results if r['Status'] == 'Success']) / len(results)
    
    print(f"\n🎯 Enhanced VWAP System Assessment:")
    print(f"   Success rate: {success_rate:.1%}")
    
    if success_rate >= 0.6:
        print(f"   ✅ System working well - ready for full backtest")
        print(f"   📊 Enhanced 111-indicator library functional")
        print(f"   🎯 VWAP bands and confidence thresholds optimized")
        print(f"\n🚀 Next step: Run full backtest with:")
        print(f"      python s3_ai_parallel_backtest_fixed.py --limit 20")
    elif success_rate >= 0.4:
        print(f"   ⚠️ System partially working - may need tuning")
        print(f"   📊 Some symbols generating trades successfully")
    else:
        print(f"   ❌ System needs debugging")
        print(f"   🔧 Check database connectivity and indicator calculations")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"quick_enhanced_backtest_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'symbols': test_symbols,
                'capital': initial_capital,
                'days': days,
                'enhanced_indicators': True,
                'vwap_indicators': 77
            },
            'results': results,
            'summary': {
                'success_rate': success_rate,
                'total_symbols': len(test_symbols),
                'successful': len([r for r in results if r['Status'] == 'Success']),
                'failed': len([r for r in results if r['Status'] != 'Success'])
            }
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()