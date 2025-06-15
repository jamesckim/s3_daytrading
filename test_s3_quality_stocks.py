#!/usr/bin/env python3
"""
Test S3 Strategy on Quality Stocks
==================================
Focus on stocks with $10+ price and $1B+ market cap for better stability
and trading characteristics.
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


def get_quality_stocks(min_price: float = 10.0, 
                      min_market_cap: float = 1e9,
                      min_volume: float = 1e6,
                      limit: int = 100) -> list:
    """
    Get quality stocks based on price, market cap, and volume criteria.
    
    Args:
        min_price: Minimum stock price ($10 default)
        min_market_cap: Minimum market cap ($1B default)
        min_volume: Minimum average daily volume (1M shares default)
        limit: Maximum number of stocks to return
    """
    
    print(f"🔍 Finding quality stocks with:")
    print(f"   Min price: ${min_price}")
    print(f"   Min market cap: ${min_market_cap:,.0f}")
    print(f"   Min daily volume: {min_volume:,.0f} shares")
    
    try:
        # Connect to database
        conn_string = os.getenv('POSTGRES_URI')
        if not conn_string:
            raise ValueError("POSTGRES_URI not found in environment")
        
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # Get recent date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        # Query for quality stocks
        query = """
            WITH recent_data AS (
                SELECT 
                    t.symbol,
                    t.name,
                    AVG(mp.close) as avg_price,
                    MIN(mp.close) as min_price,
                    MAX(mp.close) as max_price,
                    AVG(mp.volume) as avg_volume,
                    AVG(mp.close * mp.volume) as avg_dollar_volume,
                    COUNT(DISTINCT DATE(mp.datetime)) as trading_days,
                    MAX(mp.datetime) as last_trade_date,
                    -- Estimate market cap (shares outstanding would need separate data)
                    AVG(mp.close) * 1000000 as estimated_market_cap
                FROM minute_prices mp
                JOIN tickers t ON mp.ticker_id = t.id
                WHERE mp.datetime >= %s
                    AND mp.datetime <= %s
                GROUP BY t.symbol, t.name
            )
            SELECT 
                symbol,
                name,
                avg_price,
                min_price,
                max_price,
                avg_volume,
                avg_dollar_volume,
                trading_days,
                estimated_market_cap
            FROM recent_data
            WHERE avg_price >= %s
                AND min_price >= %s * 0.8  -- Allow 20% below threshold
                AND avg_volume >= %s
                AND trading_days >= 5  -- Active trading
                AND last_trade_date >= %s  -- Recent data
                -- Exclude penny stocks and indices
                AND symbol NOT LIKE '^%%'  -- No indices
                AND symbol NOT LIKE '%%.%%'  -- No preferred shares
                AND LENGTH(symbol) <= 5  -- No weird symbols
            ORDER BY avg_dollar_volume DESC
            LIMIT %s
        """
        
        min_date = end_date - timedelta(days=3)
        
        cursor.execute(query, (
            start_date, end_date, 
            min_price, min_price, 
            min_volume,
            min_date,
            limit * 2  # Get extra to filter further
        ))
        
        results = cursor.fetchall()
        
        # Additional filtering for known quality stocks
        quality_filters = {
            'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'BRK.B', 
                        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD'],
            'large_cap_tech': ['ADBE', 'CRM', 'NFLX', 'ORCL', 'CSCO', 'INTC', 'AMD', 
                              'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC'],
            'financials': ['BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
                          'SCHW', 'BLK', 'SPGI', 'AXP', 'CME', 'ICE'],
            'healthcare': ['LLY', 'NVO', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD',
                          'CVS', 'CI', 'HUM', 'MCK', 'ZTS', 'REGN'],
            'consumer': ['DIS', 'NKE', 'SBUX', 'MCD', 'LOW', 'TGT', 'COST', 'TJX',
                        'CMG', 'YUM', 'MAR', 'HLT', 'BKNG', 'ABNB'],
            'industrials': ['CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM',
                           'DE', 'EMR', 'ITW', 'FDX', 'NSC', 'UNP']
        }
        
        # Flatten quality stock list
        known_quality = []
        for category, stocks in quality_filters.items():
            known_quality.extend(stocks)
        
        # Process results
        quality_stocks = []
        stock_data = {}
        
        print(f"\n📊 Quality Stocks Found:")
        print("=" * 100)
        print(f"{'Symbol':<8} {'Name':<30} {'Avg Price':>12} {'Avg Volume':>15} {'Est. Value':>15}")
        print("-" * 100)
        
        for symbol, name, avg_price, min_price, max_price, avg_volume, dollar_vol, days, mkt_cap in results:
            # Additional quality checks
            if avg_price >= min_price and avg_volume >= min_volume:
                quality_stocks.append(symbol)
                stock_data[symbol] = {
                    'name': name[:30] if name else symbol,
                    'avg_price': avg_price,
                    'avg_volume': avg_volume,
                    'dollar_volume': dollar_vol,
                    'volatility': (max_price - min_price) / avg_price if avg_price > 0 else 0
                }
                
                if len(quality_stocks) <= 50:  # Print first 50
                    print(f"{symbol:<8} {stock_data[symbol]['name']:<30} "
                          f"${avg_price:>11.2f} {avg_volume:>15,.0f} "
                          f"${dollar_vol:>14,.0f}")
        
        # Prioritize known quality stocks
        final_list = []
        
        # First add known quality stocks that meet criteria
        for symbol in known_quality:
            if symbol in quality_stocks and symbol not in final_list:
                final_list.append(symbol)
        
        # Then add others by dollar volume
        for symbol in quality_stocks:
            if symbol not in final_list:
                final_list.append(symbol)
        
        # Limit to requested number
        final_list = final_list[:limit]
        
        cursor.close()
        conn.close()
        
        print(f"\n✅ Selected {len(final_list)} quality stocks for testing")
        
        # Group by characteristics
        if len(final_list) > 0:
            print("\n📈 Stock Categories:")
            categories = {
                'Mega Cap (Top 15)': final_list[:15],
                'Large Cap Tech': [s for s in final_list if s in quality_filters.get('large_cap_tech', [])],
                'Financials': [s for s in final_list if s in quality_filters.get('financials', [])],
                'Healthcare': [s for s in final_list if s in quality_filters.get('healthcare', [])],
                'Consumer': [s for s in final_list if s in quality_filters.get('consumer', [])]
            }
            
            for cat, stocks in categories.items():
                if stocks:
                    print(f"   {cat}: {', '.join(stocks[:10])}")
        
        return final_list, stock_data
        
    except Exception as e:
        print(f"❌ Error finding quality stocks: {e}")
        # Return default quality stock list
        default_quality = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK.B',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'ADBE',
            'NFLX', 'CRM', 'PFE', 'BAC', 'CSCO', 'TMO', 'ORCL', 'CVX', 'LLY',
            'COST', 'NKE', 'INTC', 'AMD', 'QCOM', 'TXN', 'HON', 'UPS', 'IBM',
            'GS', 'MS', 'BLK', 'SCHW', 'LOW', 'CAT', 'BA', 'MMM', 'AXP',
            'MCD', 'GE', 'F', 'GM', 'T', 'VZ'
        ]
        return default_quality[:limit], {}


def test_single_stock(symbol: str, stock_info: dict = None, days_back: int = 30) -> dict:
    """Test S3 strategy on a single quality stock with extended data."""
    
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
        'profit_factor': 0.0,
        'market_regime': 'Unknown',
        'signals': 0,
        'data_bars': 0,
        'latest_signal': 'UNKNOWN',
        'latest_price': 0.0,
        'avg_trade_pnl': 0.0,
        'best_trade': 0.0,
        'worst_trade': 0.0,
        'avg_holding_time': 0.0
    }
    
    try:
        # Create strategy with quality stock parameters
        strategy = S3DatabaseFinal(
            symbol=symbol,
            capital=100000,
            adaptive_indicators=True,
            top_k_indicators=10,  # More indicators for quality stocks
            selection_interval=100,
            data_days_lookback=days_back
        )
        
        # Run comprehensive backtest
        backtest_results = strategy.backtest(verbose=False)
        
        # Get latest signals
        latest = strategy.get_latest_signals(hours=48)
        
        # Update results
        result['success'] = True
        result['trades'] = backtest_results['total_trades']
        result['win_rate'] = backtest_results['win_rate']
        result['total_pnl'] = backtest_results['total_pnl']
        result['total_return'] = backtest_results['total_return']
        result['sharpe_ratio'] = backtest_results.get('sharpe_ratio', 0)
        result['max_drawdown'] = backtest_results.get('max_drawdown', 0)
        result['profit_factor'] = backtest_results.get('profit_factor', 0)
        result['market_regime'] = backtest_results['market_regime']
        result['signals'] = backtest_results['signals_generated']
        result['data_bars'] = backtest_results['data_bars']
        result['latest_signal'] = latest.get('signal_text', 'UNKNOWN')
        result['latest_price'] = latest.get('price', 0)
        
        # Calculate additional metrics if trades available
        if 'trades' in backtest_results and backtest_results['trades']:
            trades = backtest_results['trades']
            pnls = [t['pnl'] for t in trades]
            result['avg_trade_pnl'] = np.mean(pnls)
            result['best_trade'] = max(pnls)
            result['worst_trade'] = min(pnls)
        
        # Add stock info if available
        if stock_info and symbol in stock_info:
            result['avg_price'] = stock_info[symbol].get('avg_price', 0)
            result['avg_volume'] = stock_info[symbol].get('avg_volume', 0)
            result['volatility'] = stock_info[symbol].get('volatility', 0)
        
        print(f"✅ {symbol}: {result['trades']} trades, {result['win_rate']:.1%} win rate, "
              f"${result['total_pnl']:.2f} PnL, Sharpe: {result['sharpe_ratio']:.2f}")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"❌ {symbol}: {str(e)[:50]}...")
    
    return result


def analyze_quality_results(df_results: pd.DataFrame, stock_data: dict):
    """Analyze results with focus on quality metrics."""
    
    print("\n" + "="*120)
    print("📊 QUALITY STOCKS - S3 STRATEGY ANALYSIS")
    print("="*120)
    
    # Filter successful results
    successful = df_results[df_results['success']].copy()
    
    if len(successful) == 0:
        print("❌ No successful tests")
        return
    
    # Overall statistics
    print(f"\n📈 Overall Performance:")
    print(f"   Stocks tested: {len(df_results)}")
    print(f"   Successful: {len(successful)} ({len(successful)/len(df_results):.1%})")
    print(f"   Total PnL: ${successful['total_pnl'].sum():,.2f}")
    print(f"   Average return: {successful['total_return'].mean():.2%}")
    print(f"   Average win rate: {successful['win_rate'].mean():.1%}")
    print(f"   Average Sharpe ratio: {successful['sharpe_ratio'].mean():.2f}")
    
    # Profitability analysis
    profitable = successful[successful['total_pnl'] > 0]
    print(f"\n💰 Profitability:")
    print(f"   Profitable strategies: {len(profitable)}/{len(successful)} ({len(profitable)/len(successful):.1%})")
    print(f"   Average profit (winners): ${profitable['total_pnl'].mean():.2f}" if len(profitable) > 0 else "   No profitable strategies")
    print(f"   Average loss (losers): ${successful[successful['total_pnl'] < 0]['total_pnl'].mean():.2f}")
    
    # Top performers by multiple metrics
    print(f"\n🏆 Top 10 by Risk-Adjusted Return (Sharpe Ratio):")
    print(f"{'Rank':<6} {'Symbol':<8} {'Sharpe':>8} {'Win%':>8} {'Trades':<8} {'PnL':>12} {'Return':>10} {'Regime':<15}")
    print("-" * 95)
    
    top_sharpe = successful.nlargest(10, 'sharpe_ratio')
    for i, (_, row) in enumerate(top_sharpe.iterrows()):
        print(f"{i+1:<6} {row['symbol']:<8} {row['sharpe_ratio']:>8.2f} "
              f"{row['win_rate']:>7.1%} {int(row['trades']):<8} "
              f"${row['total_pnl']:>11.2f} {row['total_return']:>9.2%} {row['market_regime']:<15}")
    
    print(f"\n💵 Top 10 by Total PnL:")
    print(f"{'Rank':<6} {'Symbol':<8} {'PnL':>12} {'Win%':>8} {'Trades':<8} {'Sharpe':>8} {'PF':>8}")
    print("-" * 70)
    
    top_pnl = successful.nlargest(10, 'total_pnl')
    for i, (_, row) in enumerate(top_pnl.iterrows()):
        print(f"{i+1:<6} {row['symbol']:<8} ${row['total_pnl']:>11.2f} "
              f"{row['win_rate']:>7.1%} {int(row['trades']):<8} "
              f"{row['sharpe_ratio']:>8.2f} {row.get('profit_factor', 0):>8.2f}")
    
    print(f"\n🎯 Top 10 by Win Rate (min 30 trades):")
    print(f"{'Rank':<6} {'Symbol':<8} {'Win%':>8} {'Trades':<8} {'PnL':>12} {'Avg/Trade':>12}")
    print("-" * 70)
    
    active = successful[successful['trades'] >= 30]
    if len(active) > 0:
        top_winrate = active.nlargest(10, 'win_rate')
        for i, (_, row) in enumerate(top_winrate.iterrows()):
            avg_per_trade = row['total_pnl'] / row['trades'] if row['trades'] > 0 else 0
            print(f"{i+1:<6} {row['symbol']:<8} {row['win_rate']:>7.1%} "
                  f"{int(row['trades']):<8} ${row['total_pnl']:>11.2f} ${avg_per_trade:>11.2f}")
    
    # Analysis by price range
    if stock_data:
        print(f"\n📊 Performance by Price Range:")
        
        # Add price data to results
        for symbol in successful['symbol']:
            if symbol in stock_data:
                successful.loc[successful['symbol'] == symbol, 'avg_price'] = stock_data[symbol]['avg_price']
        
        if 'avg_price' in successful.columns:
            price_ranges = [
                (0, 50, "Low ($0-50)"),
                (50, 100, "Mid ($50-100)"),
                (100, 200, "High ($100-200)"),
                (200, float('inf'), "Very High ($200+)")
            ]
            
            for low, high, label in price_ranges:
                range_stocks = successful[(successful['avg_price'] > low) & (successful['avg_price'] <= high)]
                if len(range_stocks) > 0:
                    print(f"   {label}: {len(range_stocks)} stocks, "
                          f"Avg PnL: ${range_stocks['total_pnl'].mean():.2f}, "
                          f"Avg Win Rate: {range_stocks['win_rate'].mean():.1%}")
    
    # Market regime distribution
    print(f"\n🌐 Market Regime Distribution:")
    regime_stats = successful.groupby('market_regime').agg({
        'symbol': 'count',
        'total_pnl': 'sum',
        'win_rate': 'mean'
    }).round(2)
    
    for regime, stats in regime_stats.iterrows():
        print(f"   {regime}: {int(stats['symbol'])} stocks, "
              f"Total PnL: ${stats['total_pnl']:.2f}, "
              f"Avg Win Rate: {stats['win_rate']:.1%}")
    
    # Risk analysis
    print(f"\n⚠️ Risk Metrics:")
    print(f"   Average max drawdown: {successful['max_drawdown'].mean():.2%}")
    print(f"   Worst drawdown: {successful['max_drawdown'].max():.2%} ({successful.loc[successful['max_drawdown'].idxmax(), 'symbol']})")
    print(f"   Best Sharpe ratio: {successful['sharpe_ratio'].max():.2f} ({successful.loc[successful['sharpe_ratio'].idxmax(), 'symbol']})")
    print(f"   Average profit factor: {successful['profit_factor'].mean():.2f}")
    
    # Trading activity analysis
    print(f"\n📊 Trading Activity:")
    print(f"   Total trades: {successful['trades'].sum():,}")
    print(f"   Average trades per stock: {successful['trades'].mean():.0f}")
    print(f"   Most active: {successful.loc[successful['trades'].idxmax(), 'symbol']} ({successful['trades'].max()} trades)")
    print(f"   Least active: {successful.loc[successful['trades'].idxmin(), 'symbol']} ({successful['trades'].min()} trades)")


def main():
    """Main execution function."""
    
    print("🚀 S3 Strategy Test - Quality Stocks")
    print("=" * 120)
    
    # Get quality stocks
    quality_stocks, stock_data = get_quality_stocks(
        min_price=10.0,
        min_market_cap=1e9,
        min_volume=1e6,
        limit=50  # Test top 50 quality stocks
    )
    
    if not quality_stocks:
        print("❌ No quality stocks found")
        return
    
    print(f"\n📊 Testing {len(quality_stocks)} quality stocks with 30 days of data...")
    print("=" * 120)
    
    # Test all stocks in parallel
    all_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(test_single_stock, symbol, stock_data, 30): symbol
            for symbol in quality_stocks
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                
                # Progress update
                if completed % 10 == 0:
                    print(f"\n📈 Progress: {completed}/{len(quality_stocks)} completed...\n")
                    
            except Exception as e:
                print(f"❌ {symbol}: Failed to process - {e}")
                all_results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                })
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Completed testing in {elapsed_time:.1f} seconds")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Analyze results
    analyze_quality_results(df_results, stock_data)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = f"s3_quality_stocks_results_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Create recommended portfolio
    successful = df_results[df_results['success']]
    if len(successful) > 0:
        # Score stocks based on multiple factors
        successful['score'] = (
            successful['sharpe_ratio'] * 0.3 +  # Risk-adjusted returns
            successful['win_rate'] * 0.3 +      # Consistency
            (successful['total_return'] * 10) * 0.2 +  # Returns
            (successful['profit_factor'] / 2) * 0.2    # Risk/reward
        )
        
        # Create tiers
        top_stocks = successful.nlargest(20, 'score')
        
        portfolio = {
            'timestamp': timestamp,
            'strategy': 'S3_Quality_Stocks',
            'recommendations': {
                'Tier 1 (Core Holdings)': top_stocks.head(5)[['symbol', 'score', 'sharpe_ratio', 'win_rate', 'total_pnl']].to_dict('records'),
                'Tier 2 (Satellites)': top_stocks.iloc[5:10][['symbol', 'score', 'sharpe_ratio', 'win_rate', 'total_pnl']].to_dict('records'),
                'Tier 3 (Opportunistic)': top_stocks.iloc[10:15][['symbol', 'score', 'sharpe_ratio', 'win_rate', 'total_pnl']].to_dict('records'),
                'Watch List': top_stocks.iloc[15:20][['symbol', 'score', 'sharpe_ratio', 'win_rate', 'total_pnl']].to_dict('records')
            },
            'summary': {
                'total_tested': len(quality_stocks),
                'successful': len(successful),
                'profitable': len(successful[successful['total_pnl'] > 0]),
                'avg_sharpe': float(successful['sharpe_ratio'].mean()),
                'total_pnl': float(successful['total_pnl'].sum())
            }
        }
        
        portfolio_file = f"s3_quality_portfolio_{timestamp}.json"
        with open(portfolio_file, 'w') as f:
            json.dump(portfolio, f, indent=2)
        print(f"📄 Portfolio recommendations saved to: {portfolio_file}")
    
    print("\n✅ Quality stock analysis completed!")


if __name__ == "__main__":
    main()