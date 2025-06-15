#!/usr/bin/env python3
"""
Test S3 Optimized Strategy on Advanced Quality Stocks
====================================================
Test the win-rate optimized S3 strategy on stocks identified by the
advanced quality stock finder to validate performance improvements.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import glob
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from s3_optimized_strategy import S3OptimizedStrategy


def load_quality_stock_recommendations():
    """Load the most recent quality stock recommendations."""
    
    # Find most recent S3 recommendations file
    files = glob.glob("s3_quality_recommendations_*.json")
    if not files:
        print("❌ No quality stock recommendations found")
        return None
    
    latest_file = max(files)
    print(f"📄 Loading recommendations from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return data


def test_single_stock(symbol: str, capital: float = 100000, 
                     days_back: int = 30) -> dict:
    """Test a single stock with optimized S3 strategy."""
    
    try:
        # Initialize optimized strategy
        strategy = S3OptimizedStrategy(
            symbol=symbol,
            capital=capital,
            adaptive_indicators=True,
            top_k_indicators=15,
            selection_interval=50,
            data_days_lookback=days_back,
            # Optimization parameters
            min_win_rate_threshold=0.45,
            use_time_filters=True,
            use_microstructure=True,
            daily_loss_limit_pct=0.02
        )
        
        # Run backtest
        results = strategy.backtest(verbose=False)
        
        # Get optimization stats
        opt_stats = strategy.get_optimization_stats()
        
        return {
            'symbol': symbol,
            'success': True,
            'total_trades': results['total_trades'],
            'winning_trades': int(results['total_trades'] * results['win_rate']) if results['total_trades'] > 0 else 0,
            'losing_trades': int(results['total_trades'] * (1 - results['win_rate'])) if results['total_trades'] > 0 else 0,
            'win_rate': results['win_rate'],
            'total_pnl': results['total_pnl'],
            'total_return': results['total_return'],
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown': results.get('max_drawdown', 0),
            'regime': strategy.last_market_regime,
            'optimization_stats': opt_stats,
            'error': None
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }


def test_quality_stocks_parallel(stock_lists: dict, 
                               max_workers: int = 8,
                               days_back: int = 30):
    """Test multiple quality stock categories in parallel."""
    
    all_results = {}
    
    for category, symbols in stock_lists.items():
        if not symbols:
            continue
            
        print(f"\n🔍 Testing {category} ({len(symbols)} stocks)...")
        print("=" * 80)
        
        category_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(test_single_stock, symbol, 100000, days_back): symbol
                for symbol in symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                result = future.result()
                category_results.append(result)
                
                if result['success']:
                    print(f"✅ {result['symbol']}: Win rate: {result['win_rate']:.1%}, "
                          f"PnL: ${result['total_pnl']:.2f}")
                else:
                    print(f"❌ {result['symbol']}: {result['error']}")
        
        all_results[category] = category_results
    
    return all_results


def analyze_optimized_results(results: dict):
    """Analyze results from optimized strategy testing."""
    
    print("\n" + "="*100)
    print("📊 S3 OPTIMIZED STRATEGY RESULTS ON QUALITY STOCKS")
    print("="*100)
    
    for category, category_results in results.items():
        successful = [r for r in category_results if r['success']]
        
        if not successful:
            print(f"\n❌ {category}: No successful tests")
            continue
        
        print(f"\n📈 {category} ({len(successful)} stocks tested)")
        print("-" * 80)
        
        # Calculate aggregate metrics
        total_trades = sum(r['total_trades'] for r in successful)
        total_pnl = sum(r['total_pnl'] for r in successful)
        avg_win_rate = np.mean([r['win_rate'] for r in successful])
        avg_return = np.mean([r['total_return'] for r in successful])
        
        print(f"   Total trades: {total_trades:,}")
        print(f"   Total PnL: ${total_pnl:,.2f}")
        print(f"   Average win rate: {avg_win_rate:.1%}")
        print(f"   Average return: {avg_return:.2%}")
        
        # Win rate distribution
        win_rates = [r['win_rate'] for r in successful]
        print(f"\n   Win rate distribution:")
        print(f"   - Above 55%: {sum(1 for wr in win_rates if wr > 0.55)} stocks")
        print(f"   - 50-55%: {sum(1 for wr in win_rates if 0.50 <= wr <= 0.55)} stocks")
        print(f"   - 45-50%: {sum(1 for wr in win_rates if 0.45 <= wr < 0.50)} stocks")
        print(f"   - Below 45%: {sum(1 for wr in win_rates if wr < 0.45)} stocks")
        
        # Top performers by win rate
        sorted_by_wr = sorted(successful, key=lambda x: x['win_rate'], reverse=True)
        print(f"\n   Top 5 by win rate:")
        for r in sorted_by_wr[:5]:
            opt_stats = r.get('optimization_stats', {})
            print(f"   - {r['symbol']}: {r['win_rate']:.1%} "
                  f"(PnL: ${r['total_pnl']:.2f}, Trades: {r['total_trades']})")
        
        # Top performers by PnL
        sorted_by_pnl = sorted(successful, key=lambda x: x['total_pnl'], reverse=True)
        print(f"\n   Top 5 by PnL:")
        for r in sorted_by_pnl[:5]:
            print(f"   - {r['symbol']}: ${r['total_pnl']:.2f} "
                  f"(Win rate: {r['win_rate']:.1%}, Trades: {r['total_trades']})")
    
    # Overall comparison
    print("\n" + "="*100)
    print("📊 OPTIMIZATION IMPACT ANALYSIS")
    print("="*100)
    
    # Compare with previous results if available
    print("\n📈 Win Rate Improvements:")
    print("   Previous average: 43.9% (quality stocks)")
    
    all_successful = []
    for category_results in results.values():
        all_successful.extend([r for r in category_results if r['success']])
    
    if all_successful:
        overall_avg_wr = np.mean([r['win_rate'] for r in all_successful])
        print(f"   Optimized average: {overall_avg_wr:.1%}")
        print(f"   Improvement: +{(overall_avg_wr - 0.439) * 100:.1f} percentage points")
        
        # Count stocks meeting target
        target_met = sum(1 for r in all_successful if r['win_rate'] >= 0.50)
        print(f"\n   Stocks meeting 50%+ target: {target_met}/{len(all_successful)} "
              f"({target_met/len(all_successful)*100:.1%})")
        
        # Best performing categories
        print("\n📊 Category Performance:")
        for category, category_results in results.items():
            successful = [r for r in category_results if r['success']]
            if successful:
                cat_avg_wr = np.mean([r['win_rate'] for r in successful])
                print(f"   {category}: {cat_avg_wr:.1%} average win rate")


def save_optimized_results(results: dict):
    """Save optimized test results for future reference."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare summary data
    summary = {
        'timestamp': timestamp,
        'strategy': 'S3_Optimized_Quality_Test',
        'categories': {}
    }
    
    for category, category_results in results.items():
        successful = [r for r in category_results if r['success']]
        
        summary['categories'][category] = {
            'stocks_tested': len(category_results),
            'successful': len(successful),
            'average_win_rate': np.mean([r['win_rate'] for r in successful]) if successful else 0,
            'total_pnl': sum(r['total_pnl'] for r in successful) if successful else 0,
            'stocks_above_50pct': sum(1 for r in successful if r['win_rate'] >= 0.50),
            'top_performers': [
                {
                    'symbol': r['symbol'],
                    'win_rate': r['win_rate'],
                    'pnl': r['total_pnl'],
                    'trades': r['total_trades']
                }
                for r in sorted(successful, key=lambda x: x['win_rate'], reverse=True)[:5]
            ] if successful else []
        }
    
    # Save summary
    filename = f"s3_optimized_test_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Create recommended portfolio
    all_successful = []
    for category_results in results.values():
        all_successful.extend([r for r in category_results if r['success']])
    
    if all_successful:
        # Filter for 50%+ win rate
        high_wr = [r for r in all_successful if r['win_rate'] >= 0.50]
        high_wr_sorted = sorted(high_wr, key=lambda x: (x['win_rate'], x['total_pnl']), reverse=True)
        
        portfolio = {
            'timestamp': timestamp,
            'strategy': 'S3_Optimized_50Plus_Portfolio',
            'selection_criteria': {
                'min_win_rate': 0.50,
                'optimization_enabled': True,
                'features': ['time_filters', 'microstructure', 'quality_scoring']
            },
            'recommended_stocks': [
                {
                    'symbol': r['symbol'],
                    'win_rate': r['win_rate'],
                    'expected_pnl': r['total_pnl'],
                    'allocation': '8%' if i < 5 else '5%' if i < 10 else '2.5%'
                }
                for i, r in enumerate(high_wr_sorted[:20])
            ]
        }
        
        portfolio_file = f"s3_optimized_portfolio_{timestamp}.json"
        with open(portfolio_file, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        print(f"📋 Portfolio saved to: {portfolio_file}")


def main():
    """Main execution function."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 S3 Optimized Strategy Test on Quality Stocks")
    print("=" * 100)
    print("Target: Achieve 50-55% win rate (up from 43.9%)")
    print("=" * 100)
    
    # Load quality stock recommendations
    recommendations = load_quality_stock_recommendations()
    
    if not recommendations:
        # Fallback to known quality stocks
        print("\n⚠️ Using fallback quality stock list")
        stock_lists = {
            'high_confidence': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS'
            ],
            'medium_confidence': [
                'GS', 'NKE', 'CRM', 'QCOM', 'LLY', 'TMO', 'IBM', 'SCHW',
                'BA', 'CSCO', 'NFLX', 'ADBE', 'PFE', 'WMT', 'CVX'
            ]
        }
    else:
        # Use recommendations
        stock_lists = {
            'high_confidence': recommendations.get('high_confidence', [])[:20],
            'medium_confidence': recommendations.get('medium_confidence', [])[:15]
        }
    
    # Run tests
    results = test_quality_stocks_parallel(stock_lists, max_workers=8, days_back=30)
    
    # Analyze results
    analyze_optimized_results(results)
    
    # Save results
    save_optimized_results(results)
    
    print("\n✅ S3 Optimized Strategy testing completed!")


if __name__ == "__main__":
    main()