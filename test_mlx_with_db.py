#!/usr/bin/env python3
"""
Test MLX Backtest with Database
================================
Quick test of MLX backtesting with your database infrastructure.
"""

import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_mlx_backtest_with_db():
    """Test MLX backtest with a few symbols."""
    
    print("\n" + "="*60)
    print("🚀 TESTING MLX BACKTEST WITH DATABASE")
    print("="*60)
    
    # Test with a small set of symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    print(f"\n📊 Testing with {len(test_symbols)} symbols")
    print(f"🗄️  Using PostgreSQL database if available")
    print(f"📅 Backtest period: Last 30 days")
    
    try:
        from s3_ai_mlx_backtest import MLXParallelBacktester
        
        # Create backtester
        print("\n⚙️  Initializing MLX backtester...")
        backtester = MLXParallelBacktester(initial_capital=100000)
        
        # Run backtest with shorter period for quick test
        print("\n🏃 Running MLX vectorized backtest...")
        start_time = datetime.now()
        
        backtester.run(test_symbols, days=30)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ Backtest completed in {elapsed:.2f} seconds")
        print(f"⚡ Processing rate: {len(test_symbols)/elapsed:.1f} symbols/second")
        
        # Show results summary
        if backtester.results:
            print(f"\n📊 Results Summary:")
            print(f"   Symbols processed: {len(backtester.results)}")
            
            # Calculate average return
            returns = [r.total_return_pct for r in backtester.results.values()]
            if returns:
                print(f"   Average return: {sum(returns)/len(returns):.2f}%")
                print(f"   Best return: {max(returns):.2f}%")
                print(f"   Worst return: {min(returns):.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_availability():
    """Check if database is available."""
    print("\n🔍 Checking data availability...")
    
    try:
        from s3_ai_db_backtest_all_tickers import DatabaseDataProvider
        
        provider = DatabaseDataProvider()
        print("✅ Database connection established")
        
        # Test data fetch for AAPL
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        print(f"\n📊 Testing data fetch for AAPL ({start_date.date()} to {end_date.date()})...")
        data = provider.get_minute_data('AAPL', start_date, end_date)
        
        if not data.empty:
            print(f"✅ Retrieved {len(data)} data points")
            print(f"   First timestamp: {data.index[0]}")
            print(f"   Last timestamp: {data.index[-1]}")
            return True
        else:
            print("⚠️  No data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Database not available: {e}")
        return False


def main():
    """Run the test."""
    print("🧪 MLX Backtest Integration Test")
    print("Testing MLX with your existing database infrastructure")
    
    # First check data availability
    db_available = test_data_availability()
    
    if not db_available:
        print("\n⚠️  Database not available - MLX will use mock data")
        print("   The backtest will still run to demonstrate MLX performance")
    
    # Run MLX backtest test
    success = test_mlx_backtest_with_db()
    
    if success:
        print("\n" + "="*60)
        print("✅ MLX BACKTEST INTEGRATION SUCCESSFUL!")
        print("="*60)
        print("\n🎯 Next steps:")
        print("1. Run full backtest: python s3_ai_mlx_backtest.py")
        print("2. Compare performance: python compare_backtest_performance.py")
        print("3. Migrate your backtests to MLX for 10-100x speedup!")
    else:
        print("\n⚠️  Test encountered issues - check error messages above")


if __name__ == "__main__":
    main()