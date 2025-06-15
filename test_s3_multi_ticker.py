#!/usr/bin/env python3
"""
Test S3 Database Strategy with Multiple Tickers
==============================================
Comprehensive testing across multiple symbols using real PostgreSQL data.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import our strategy
from s3_database_final import S3DatabaseFinal


def load_james_tickers():
    """Load the validated ticker list from james_tickers.json."""
    try:
        with open('../james_tickers.json', 'r') as f:
            data = json.load(f)
            return data.get('symbols', [])
    except FileNotFoundError:
        print("⚠️ james_tickers.json not found, using default list")
        # Default high-liquidity tickers
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA',
            'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX', 'PFE', 'CSCO',
            'PEP', 'TMO', 'ABT', 'INTC', 'VZ', 'NKE', 'ORCL',
            'QCOM', 'TXN', 'HON', 'COST'
        ]


def test_single_ticker(symbol: str, 
                      capital: float = 100000,
                      days_back: int = 14,
                      adaptive: bool = True) -> dict:
    """Test a single ticker and return results."""
    
    print(f"\n{'='*60}")
    print(f"📊 Testing {symbol}")
    print(f"{'='*60}")
    
    results = {
        'symbol': symbol,
        'success': False,
        'error': None,
        'data_available': False,
        'performance': {}
    }
    
    try:
        # Create strategy instance
        strategy = S3DatabaseFinal(
            symbol=symbol,
            capital=capital,
            adaptive_indicators=adaptive,
            top_k_indicators=8,  # Reduced for speed
            selection_interval=100,
            data_days_lookback=days_back
        )
        
        # Run backtest
        backtest_results = strategy.backtest(verbose=False)
        
        # Get latest signals
        latest_signals = strategy.get_latest_signals(hours=24)
        
        # Store results
        results['success'] = True
        results['data_available'] = True
        results['performance'] = {
            'total_trades': backtest_results['total_trades'],
            'win_rate': backtest_results['win_rate'],
            'total_pnl': backtest_results['total_pnl'],
            'total_return': backtest_results['total_return'],
            'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
            'market_regime': backtest_results['market_regime'],
            'signals_generated': backtest_results['signals_generated'],
            'data_bars': backtest_results['data_bars'],
            'latest_signal': latest_signals.get('signal_text', 'UNKNOWN'),
            'latest_price': latest_signals.get('price', 0)
        }
        
        # Print summary
        print(f"✅ {symbol} Results:")
        print(f"   Data bars: {backtest_results['data_bars']:,}")
        print(f"   Trades: {backtest_results['total_trades']}")
        print(f"   Win rate: {backtest_results['win_rate']:.1%}")
        print(f"   Total PnL: ${backtest_results['total_pnl']:.2f}")
        print(f"   Return: {backtest_results['total_return']:.2%}")
        print(f"   Regime: {backtest_results['market_regime']}")
        print(f"   Latest: {latest_signals.get('signal_text', 'UNKNOWN')} @ ${latest_signals.get('price', 0):.2f}")
        
    except ValueError as e:
        if "No data found" in str(e):
            results['error'] = "No data available"
            results['data_available'] = False
            print(f"❌ {symbol}: No data available in database")
        else:
            results['error'] = str(e)
            print(f"❌ {symbol}: {e}")
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ {symbol}: Unexpected error - {e}")
    
    return results


def test_multiple_tickers_parallel(tickers: list, 
                                 capital: float = 100000,
                                 days_back: int = 14,
                                 max_workers: int = 4) -> pd.DataFrame:
    """Test multiple tickers in parallel for efficiency."""
    
    print(f"\n🚀 Testing {len(tickers)} tickers in parallel (max {max_workers} workers)")
    print(f"   Capital: ${capital:,.0f}")
    print(f"   Days back: {days_back}")
    print(f"   Adaptive indicators: Enabled")
    
    all_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(test_single_ticker, ticker, capital, days_back, True): ticker
            for ticker in tickers
        }
        
        # Process completed tasks
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"❌ {ticker}: Failed to process - {e}")
                all_results.append({
                    'symbol': ticker,
                    'success': False,
                    'error': str(e),
                    'data_available': False,
                    'performance': {}
                })
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Completed testing in {elapsed_time:.1f} seconds")
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(all_results)
    
    return df_results


def analyze_results(df_results: pd.DataFrame):
    """Analyze and display comprehensive results."""
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    # Overall statistics
    total_tickers = len(df_results)
    successful_tickers = df_results['success'].sum()
    data_available = df_results['data_available'].sum()
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total tickers tested: {total_tickers}")
    print(f"   Successful backtests: {successful_tickers}")
    print(f"   Tickers with data: {data_available}")
    print(f"   Success rate: {successful_tickers/total_tickers:.1%}")
    
    # Filter successful results
    successful_df = df_results[df_results['success']].copy()
    
    if len(successful_df) > 0:
        # Extract performance metrics
        for metric in ['total_trades', 'win_rate', 'total_pnl', 'total_return', 'signals_generated']:
            successful_df[metric] = successful_df['performance'].apply(lambda x: x.get(metric, 0))
        
        # Top performers by PnL
        print(f"\n💰 Top 10 by Total PnL:")
        top_pnl = successful_df.nlargest(10, 'total_pnl')[['symbol', 'total_pnl', 'win_rate', 'total_trades']]
        for _, row in top_pnl.iterrows():
            print(f"   {row['symbol']:6} ${row['total_pnl']:8.2f} | Win: {row['win_rate']:5.1%} | Trades: {row['total_trades']:3}")
        
        # Top performers by win rate (minimum 10 trades)
        active_traders = successful_df[successful_df['total_trades'] >= 10]
        if len(active_traders) > 0:
            print(f"\n🎯 Top 10 by Win Rate (min 10 trades):")
            top_winrate = active_traders.nlargest(10, 'win_rate')[['symbol', 'win_rate', 'total_trades', 'total_pnl']]
            for _, row in top_winrate.iterrows():
                print(f"   {row['symbol']:6} {row['win_rate']:5.1%} | Trades: {row['total_trades']:3} | PnL: ${row['total_pnl']:8.2f}")
        
        # Most active traders
        print(f"\n📊 Most Active Traders:")
        most_active = successful_df.nlargest(10, 'total_trades')[['symbol', 'total_trades', 'win_rate', 'total_pnl']]
        for _, row in most_active.iterrows():
            print(f"   {row['symbol']:6} {row['total_trades']:3} trades | Win: {row['win_rate']:5.1%} | PnL: ${row['total_pnl']:8.2f}")
        
        # Market regime distribution
        print(f"\n🌐 Market Regime Distribution:")
        regime_counts = successful_df['performance'].apply(lambda x: x.get('market_regime', 'Unknown')).value_counts()
        for regime, count in regime_counts.items():
            print(f"   {regime}: {count} tickers ({count/len(successful_df):.1%})")
        
        # Signal distribution
        print(f"\n🎯 Latest Signal Distribution:")
        signal_counts = successful_df['performance'].apply(lambda x: x.get('latest_signal', 'UNKNOWN')).value_counts()
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} tickers")
        
        # Summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"   Average trades per ticker: {successful_df['total_trades'].mean():.1f}")
        print(f"   Average win rate: {successful_df['win_rate'].mean():.1%}")
        print(f"   Total PnL across all: ${successful_df['total_pnl'].sum():.2f}")
        print(f"   Average return: {successful_df['total_return'].mean():.2%}")
        
        # Profitable vs unprofitable
        profitable = (successful_df['total_pnl'] > 0).sum()
        print(f"\n💵 Profitability:")
        print(f"   Profitable strategies: {profitable} ({profitable/len(successful_df):.1%})")
        print(f"   Unprofitable strategies: {len(successful_df) - profitable} ({(len(successful_df) - profitable)/len(successful_df):.1%})")
    
    # Failed tickers
    failed_df = df_results[~df_results['success']]
    if len(failed_df) > 0:
        print(f"\n❌ Failed Tickers ({len(failed_df)}):")
        for _, row in failed_df.iterrows():
            print(f"   {row['symbol']}: {row['error']}")
    
    return successful_df


def save_results(df_results: pd.DataFrame, successful_df: pd.DataFrame):
    """Save results to files for later analysis."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full results
    results_file = f"s3_multi_ticker_results_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n💾 Full results saved to: {results_file}")
    
    # Save summary report
    report_file = f"s3_multi_ticker_report_{timestamp}.json"
    
    summary = {
        'test_timestamp': timestamp,
        'total_tickers': len(df_results),
        'successful_tests': len(successful_df),
        'total_pnl': float(successful_df['total_pnl'].sum()) if len(successful_df) > 0 else 0,
        'average_win_rate': float(successful_df['win_rate'].mean()) if len(successful_df) > 0 else 0,
        'top_performers': [],
        'regime_distribution': {},
        'signal_distribution': {}
    }
    
    if len(successful_df) > 0:
        # Add top performers
        top_5 = successful_df.nlargest(5, 'total_pnl')
        for _, row in top_5.iterrows():
            summary['top_performers'].append({
                'symbol': row['symbol'],
                'total_pnl': float(row['total_pnl']),
                'win_rate': float(row['win_rate']),
                'total_trades': int(row['total_trades'])
            })
        
        # Add distributions
        regime_counts = successful_df['performance'].apply(lambda x: x.get('market_regime', 'Unknown')).value_counts()
        summary['regime_distribution'] = regime_counts.to_dict()
        
        signal_counts = successful_df['performance'].apply(lambda x: x.get('latest_signal', 'UNKNOWN')).value_counts()
        summary['signal_distribution'] = signal_counts.to_dict()
    
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Summary report saved to: {report_file}")


def main():
    """Main execution function."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    print("🚀 S3 Multi-Ticker Database Strategy Test")
    print("=" * 80)
    
    # Load tickers
    tickers = load_james_tickers()
    print(f"📋 Loaded {len(tickers)} tickers for testing")
    
    # Test parameters
    capital = 100000
    days_back = 14  # 2 weeks of data
    max_workers = 4  # Parallel workers
    
    # Option to test subset first
    test_subset = input("\nTest full list or subset? (full/subset): ").lower()
    
    if test_subset == 'subset':
        num_tickers = int(input("How many tickers to test? "))
        tickers = tickers[:num_tickers]
        print(f"📌 Testing subset of {len(tickers)} tickers")
    
    # Run tests
    df_results = test_multiple_tickers_parallel(
        tickers=tickers,
        capital=capital,
        days_back=days_back,
        max_workers=max_workers
    )
    
    # Analyze results
    successful_df = analyze_results(df_results)
    
    # Save results
    save_results(df_results, successful_df)
    
    print("\n✅ Multi-ticker testing completed!")
    
    # Option to test with different parameters
    retest = input("\nTest with different parameters? (y/n): ").lower()
    if retest == 'y':
        days_back = int(input("Days back (default 14): ") or "14")
        adaptive = input("Use adaptive indicators? (y/n): ").lower() == 'y'
        
        print(f"\n🔄 Retesting with days_back={days_back}, adaptive={adaptive}")
        
        # Rerun with new parameters
        df_results = test_multiple_tickers_parallel(
            tickers=tickers[:10],  # Test subset for speed
            capital=capital,
            days_back=days_back,
            max_workers=max_workers
        )
        
        analyze_results(df_results)


if __name__ == "__main__":
    main()