#!/usr/bin/env python3
"""
Test S3 Strategy on Top 50 Highest Volume Stocks
================================================
Find the 50 stocks with highest average daily volume over the last 10 trading days
and test the S3 strategy on them.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from dotenv import load_dotenv
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from s3_database_final import S3DatabaseFinal

# Load environment variables
load_dotenv()


def get_top_volume_stocks(limit: int = 50, days_back: int = 10) -> list:
    """Get the top stocks by average daily volume from the database."""
    
    print(f"🔍 Finding top {limit} stocks by average volume (last {days_back} trading days)")
    
    try:
        # Connect to database
        conn_string = os.getenv('POSTGRES_URI')
        if not conn_string:
            raise ValueError("POSTGRES_URI not found in environment")
        
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # Calculate date range (10 trading days = ~14 calendar days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back + 4)  # Add buffer for weekends
        
        # Query to find top volume stocks
        query = """
            WITH daily_volumes AS (
                SELECT 
                    t.symbol,
                    DATE(mp.datetime) as trading_date,
                    SUM(mp.volume) as daily_volume,
                    AVG(mp.close * mp.volume) as daily_dollar_volume
                FROM minute_prices mp
                JOIN tickers t ON mp.ticker_id = t.id
                WHERE mp.datetime >= %s
                    AND mp.datetime <= %s
                GROUP BY t.symbol, DATE(mp.datetime)
            ),
            avg_volumes AS (
                SELECT 
                    symbol,
                    AVG(daily_volume) as avg_daily_volume,
                    AVG(daily_dollar_volume) as avg_dollar_volume,
                    COUNT(DISTINCT trading_date) as trading_days,
                    MAX(trading_date) as last_trading_date
                FROM daily_volumes
                GROUP BY symbol
                HAVING COUNT(DISTINCT trading_date) >= %s  -- Minimum trading days
            )
            SELECT 
                symbol,
                avg_daily_volume,
                avg_dollar_volume,
                trading_days,
                last_trading_date
            FROM avg_volumes
            WHERE last_trading_date >= %s  -- Must have recent data
            ORDER BY avg_daily_volume DESC
            LIMIT %s
        """
        
        # Execute query
        min_trading_days = max(5, days_back // 2)  # At least half the days
        min_date = end_date - timedelta(days=3)  # Must have data within 3 days
        
        cursor.execute(query, (start_date, end_date, min_trading_days, min_date, limit))
        results = cursor.fetchall()
        
        # Process results
        high_volume_stocks = []
        print(f"\n📊 Top {len(results)} Stocks by Average Daily Volume:")
        print("=" * 80)
        print(f"{'Rank':<5} {'Symbol':<8} {'Avg Daily Volume':>20} {'Avg $ Volume':>20} {'Days':<6}")
        print("-" * 80)
        
        for i, (symbol, avg_volume, dollar_volume, days, last_date) in enumerate(results):
            high_volume_stocks.append(symbol)
            print(f"{i+1:<5} {symbol:<8} {avg_volume:>20,.0f} ${dollar_volume:>19,.0f} {days:<6}")
        
        cursor.close()
        conn.close()
        
        if len(high_volume_stocks) < limit:
            print(f"\n⚠️ Only found {len(high_volume_stocks)} stocks with sufficient data")
        
        return high_volume_stocks
        
    except Exception as e:
        print(f"❌ Error getting high volume stocks: {e}")
        # Return default list if database query fails
        default_stocks = [
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'BRK.B',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'BAC',
            'ADBE', 'CRM', 'NFLX', 'CSCO', 'PEP', 'TMO', 'ABT', 'ORCL', 'ACN',
            'COST', 'CVX', 'INTC', 'AMD', 'VZ', 'NKE', 'QCOM', 'TXN', 'PM',
            'HON', 'UPS', 'MS', 'GS', 'BLK', 'SCHW', 'C', 'AXP', 'SPGI',
            'PYPL', 'SQ', 'UBER', 'ABNB', 'COIN', 'HOOD'
        ]
        return default_stocks[:limit]


def test_single_stock(symbol: str, days_back: int = 14) -> dict:
    """Test S3 strategy on a single stock."""
    
    result = {
        'symbol': symbol,
        'success': False,
        'error': None,
        'trades': 0,
        'win_rate': 0.0,
        'total_pnl': 0.0,
        'total_return': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'market_regime': 'Unknown',
        'signals': 0,
        'data_bars': 0,
        'latest_signal': 'UNKNOWN',
        'latest_price': 0.0,
        'volume_rank': 0
    }
    
    try:
        # Create strategy
        strategy = S3DatabaseFinal(
            symbol=symbol,
            capital=100000,
            adaptive_indicators=True,
            top_k_indicators=8,
            selection_interval=100,
            data_days_lookback=days_back
        )
        
        # Run backtest
        backtest_results = strategy.backtest(verbose=False)
        
        # Get latest signals
        latest = strategy.get_latest_signals(hours=24)
        
        # Update results
        result['success'] = True
        result['trades'] = backtest_results['total_trades']
        result['win_rate'] = backtest_results['win_rate']
        result['total_pnl'] = backtest_results['total_pnl']
        result['total_return'] = backtest_results['total_return']
        result['sharpe_ratio'] = backtest_results.get('sharpe_ratio', 0)
        result['max_drawdown'] = backtest_results.get('max_drawdown', 0)
        result['market_regime'] = backtest_results['market_regime']
        result['signals'] = backtest_results['signals_generated']
        result['data_bars'] = backtest_results['data_bars']
        result['latest_signal'] = latest.get('signal_text', 'UNKNOWN')
        result['latest_price'] = latest.get('price', 0)
        
        print(f"✅ {symbol}: {result['trades']} trades, {result['win_rate']:.1%} win rate, ${result['total_pnl']:.2f} PnL")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ {symbol}: {str(e)[:50]}...")
    
    return result


def test_high_volume_stocks(max_workers: int = 6):
    """Test S3 strategy on high volume stocks."""
    
    print("🚀 S3 Strategy Test - High Volume Stocks")
    print("=" * 80)
    
    # Get high volume stocks
    high_volume_stocks = get_top_volume_stocks(limit=50, days_back=10)
    
    if not high_volume_stocks:
        print("❌ No stocks found")
        return
    
    print(f"\n📊 Testing {len(high_volume_stocks)} high volume stocks...")
    print("=" * 80)
    
    # Test all stocks in parallel
    all_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(test_single_stock, symbol, 14): (i+1, symbol)
            for i, symbol in enumerate(high_volume_stocks)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_symbol):
            rank, symbol = future_to_symbol[future]
            try:
                result = future.result()
                result['volume_rank'] = rank
                all_results.append(result)
            except Exception as e:
                print(f"❌ {symbol}: Failed to process - {e}")
                all_results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': str(e),
                    'volume_rank': rank
                })
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Completed testing in {elapsed_time:.1f} seconds")
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(all_results)
    
    # Separate successful and failed
    successful = df_results[df_results['success']]
    failed = df_results[~df_results['success']]
    
    # Display results
    print("\n" + "="*100)
    print("📊 HIGH VOLUME STOCKS - S3 STRATEGY RESULTS")
    print("="*100)
    
    if len(successful) > 0:
        # Summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"   Successful tests: {len(successful)}/{len(high_volume_stocks)}")
        print(f"   Total PnL: ${successful['total_pnl'].sum():.2f}")
        print(f"   Average win rate: {successful['win_rate'].mean():.1%}")
        print(f"   Average return: {successful['total_return'].mean():.2%}")
        print(f"   Total trades: {successful['trades'].sum():,}")
        
        # Top performers by PnL
        print(f"\n💰 Top 10 by Total PnL:")
        print(f"{'Rank':<6} {'Symbol':<8} {'Vol Rank':<10} {'Trades':<8} {'Win%':<8} {'PnL':>12} {'Return':>10} {'Regime':<15}")
        print("-" * 95)
        
        top_pnl = successful.nlargest(10, 'total_pnl')
        for _, row in top_pnl.iterrows():
            print(f"{len(top_pnl) - _:<6} {row['symbol']:<8} {row['volume_rank']:<10} "
                  f"{row['trades']:<8} {row['win_rate']:>6.1%}  ${row['total_pnl']:>11.2f} "
                  f"{row['total_return']:>9.2%} {row['market_regime']:<15}")
        
        # Top performers by win rate (min 20 trades)
        active_traders = successful[successful['trades'] >= 20]
        if len(active_traders) > 0:
            print(f"\n🎯 Top 10 by Win Rate (min 20 trades):")
            print(f"{'Rank':<6} {'Symbol':<8} {'Vol Rank':<10} {'Trades':<8} {'Win%':<8} {'PnL':>12} {'Return':>10}")
            print("-" * 80)
            
            top_winrate = active_traders.nlargest(10, 'win_rate')
            for _, row in top_winrate.iterrows():
                print(f"{len(top_winrate) - _:<6} {row['symbol']:<8} {row['volume_rank']:<10} "
                      f"{row['trades']:<8} {row['win_rate']:>6.1%}  ${row['total_pnl']:>11.2f} "
                      f"{row['total_return']:>9.2%}")
        
        # Most active traders
        print(f"\n📊 Most Active Traders:")
        print(f"{'Rank':<6} {'Symbol':<8} {'Vol Rank':<10} {'Trades':<8} {'Win%':<8} {'PnL':>12} {'Signals':<10}")
        print("-" * 80)
        
        most_active = successful.nlargest(10, 'trades')
        for _, row in most_active.iterrows():
            print(f"{len(most_active) - _:<6} {row['symbol']:<8} {row['volume_rank']:<10} "
                  f"{row['trades']:<8} {row['win_rate']:>6.1%}  ${row['total_pnl']:>11.2f} "
                  f"{row['signals']:<10}")
        
        # Market regime distribution
        print(f"\n🌐 Market Regime Distribution:")
        regime_counts = successful['market_regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"   {regime}: {count} stocks ({count/len(successful):.1%})")
        
        # Signal distribution
        print(f"\n🎯 Latest Signal Distribution:")
        signal_counts = successful['latest_signal'].value_counts()
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} stocks")
        
        # Profitability analysis
        profitable = successful[successful['total_pnl'] > 0]
        print(f"\n💵 Profitability Analysis:")
        print(f"   Profitable strategies: {len(profitable)}/{len(successful)} ({len(profitable)/len(successful):.1%})")
        print(f"   Average profitable PnL: ${profitable['total_pnl'].mean():.2f}" if len(profitable) > 0 else "   No profitable strategies")
        
        # Volume rank correlation
        print(f"\n📈 Performance by Volume Rank:")
        for i in range(0, 50, 10):
            rank_range = successful[(successful['volume_rank'] > i) & (successful['volume_rank'] <= i+10)]
            if len(rank_range) > 0:
                avg_pnl = rank_range['total_pnl'].mean()
                avg_wr = rank_range['win_rate'].mean()
                print(f"   Ranks {i+1}-{i+10}: Avg PnL ${avg_pnl:.2f}, Avg Win Rate {avg_wr:.1%}")
    
    # Failed stocks
    if len(failed) > 0:
        print(f"\n❌ Failed Tests ({len(failed)} stocks):")
        for _, row in failed.head(10).iterrows():
            error_msg = row['error'][:60] + '...' if len(row['error']) > 60 else row['error']
            print(f"   {row['symbol']} (Vol Rank {row['volume_rank']}): {error_msg}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = f"s3_high_volume_results_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Save summary report
    if len(successful) > 0:
        summary = {
            'test_timestamp': timestamp,
            'total_stocks_tested': len(high_volume_stocks),
            'successful_tests': len(successful),
            'failed_tests': len(failed),
            'total_pnl': float(successful['total_pnl'].sum()),
            'average_win_rate': float(successful['win_rate'].mean()),
            'average_return': float(successful['total_return'].mean()),
            'total_trades': int(successful['trades'].sum()),
            'profitable_strategies': len(successful[successful['total_pnl'] > 0]),
            'top_5_by_pnl': successful.nlargest(5, 'total_pnl')[['symbol', 'total_pnl', 'win_rate', 'trades']].to_dict('records'),
            'top_5_by_volume': successful.nsmallest(5, 'volume_rank')[['symbol', 'volume_rank', 'total_pnl', 'win_rate']].to_dict('records'),
            'regime_distribution': successful['market_regime'].value_counts().to_dict(),
            'signal_distribution': successful['latest_signal'].value_counts().to_dict()
        }
        
        report_file = f"s3_high_volume_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Summary report saved to: {report_file}")
    
    return df_results


if __name__ == "__main__":
    # Run the high volume stock test
    results = test_high_volume_stocks(max_workers=6)
    
    print("\n✅ High volume stock testing completed!")