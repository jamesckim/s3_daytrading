#!/usr/bin/env python3
"""
Automated S3 Multi-Ticker Test Runner
====================================
Run comprehensive tests across multiple tickers without user interaction.
"""

import json
import pandas as pd
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from s3_database_final import S3DatabaseFinal


def test_ticker(symbol: str, capital: float = 100000, days_back: int = 14) -> dict:
    """Test a single ticker and return results."""
    
    print(f"Testing {symbol}...", end='', flush=True)
    
    results = {
        'symbol': symbol,
        'success': False,
        'error': None,
        'total_trades': 0,
        'win_rate': 0.0,
        'total_pnl': 0.0,
        'total_return': 0.0,
        'market_regime': 'Unknown',
        'signals_generated': 0,
        'data_bars': 0,
        'latest_signal': 'UNKNOWN',
        'latest_price': 0.0
    }
    
    try:
        # Create and test strategy
        strategy = S3DatabaseFinal(
            symbol=symbol,
            capital=capital,
            adaptive_indicators=True,
            top_k_indicators=8,
            selection_interval=100,
            data_days_lookback=days_back
        )
        
        # Run backtest
        backtest_results = strategy.backtest(verbose=False)
        latest_signals = strategy.get_latest_signals(hours=24)
        
        # Extract results
        results['success'] = True
        results['total_trades'] = backtest_results['total_trades']
        results['win_rate'] = backtest_results['win_rate']
        results['total_pnl'] = backtest_results['total_pnl']
        results['total_return'] = backtest_results['total_return']
        results['market_regime'] = backtest_results['market_regime']
        results['signals_generated'] = backtest_results['signals_generated']
        results['data_bars'] = backtest_results['data_bars']
        results['latest_signal'] = latest_signals.get('signal_text', 'UNKNOWN')
        results['latest_price'] = latest_signals.get('price', 0)
        
        print(f" ✅ {results['total_trades']} trades, {results['win_rate']:.1%} win rate")
        
    except Exception as e:
        results['error'] = str(e)
        print(f" ❌ {str(e)[:50]}...")
    
    return results


def main():
    """Run multi-ticker test."""
    
    print("🚀 S3 Multi-Ticker Strategy Test")
    print("=" * 60)
    
    # Default ticker list
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA',
        'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX', 'PFE', 'CSCO',
        'PEP', 'TMO', 'ABT', 'INTC', 'VZ', 'NKE', 'ORCL',
        'QCOM', 'TXN', 'HON', 'COST'
    ]
    
    # Try to load james_tickers.json
    try:
        with open('../james_tickers.json', 'r') as f:
            data = json.load(f)
            tickers = data.get('symbols', tickers)
            print(f"📋 Loaded {len(tickers)} tickers from james_tickers.json")
    except:
        print(f"📋 Using default list of {len(tickers)} tickers")
    
    # Test first 20 tickers for speed
    test_tickers = tickers[:20]
    print(f"🧪 Testing {len(test_tickers)} tickers with 14 days of data\n")
    
    # Run tests in parallel
    start_time = time.time()
    all_results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(test_ticker, ticker): ticker for ticker in test_tickers}
        
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
    
    elapsed = time.time() - start_time
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    successful = df[df['success']]
    
    print(f"\n⏱️ Completed in {elapsed:.1f} seconds")
    print(f"\n📊 Results Summary:")
    print(f"   Successful: {len(successful)}/{len(test_tickers)}")
    
    if len(successful) > 0:
        print(f"   Total PnL: ${successful['total_pnl'].sum():.2f}")
        print(f"   Avg Win Rate: {successful['win_rate'].mean():.1%}")
        print(f"   Total Trades: {successful['total_trades'].sum()}")
        
        print(f"\n💰 Top 5 by PnL:")
        top_pnl = successful.nlargest(5, 'total_pnl')[['symbol', 'total_pnl', 'win_rate', 'total_trades']]
        for _, row in top_pnl.iterrows():
            print(f"   {row['symbol']:6} ${row['total_pnl']:8.2f} | Win: {row['win_rate']:5.1%} | Trades: {row['total_trades']:3}")
        
        print(f"\n🎯 Top 5 by Win Rate (min 10 trades):")
        active = successful[successful['total_trades'] >= 10]
        if len(active) > 0:
            top_wr = active.nlargest(5, 'win_rate')[['symbol', 'win_rate', 'total_trades', 'total_pnl']]
            for _, row in top_wr.iterrows():
                print(f"   {row['symbol']:6} {row['win_rate']:5.1%} | Trades: {row['total_trades']:3} | PnL: ${row['total_pnl']:7.2f}")
        
        print(f"\n🌐 Market Regimes:")
        regime_counts = successful['market_regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"   {regime}: {count} tickers")
        
        print(f"\n📈 Latest Signals:")
        signal_counts = successful['latest_signal'].value_counts()
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} tickers")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"s3_test_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
    
    # Show failed tickers
    failed = df[~df['success']]
    if len(failed) > 0:
        print(f"\n❌ Failed Tickers:")
        for _, row in failed.head(10).iterrows():
            error = row['error'][:50] + '...' if len(row['error']) > 50 else row['error']
            print(f"   {row['symbol']}: {error}")


if __name__ == "__main__":
    main()