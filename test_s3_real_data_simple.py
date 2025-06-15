#!/usr/bin/env python3
"""
Simple test for S3 Real Data Strategy
===================================
Simplified test to check database connectivity and basic functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import the strategy
try:
    from s3_working_strategy_real_data import S3WorkingStrategyRealData
    print("✅ Successfully imported S3WorkingStrategyRealData")
except ImportError as e:
    print(f"❌ Failed to import strategy: {e}")
    exit(1)


def test_basic_functionality():
    """Test basic functionality without heavy processing."""
    print("🧪 Testing S3 Real Data Strategy - Basic Functionality")
    print("=" * 55)
    
    symbol = 'AAPL'
    print(f"\n🔍 Testing with {symbol}...")
    
    try:
        # Initialize strategy with conservative settings
        print("1. Initializing strategy...")
        strategy = S3WorkingStrategyRealData(
            symbol=symbol,
            capital=10000,  # Smaller capital for testing
            adaptive_indicators=False,  # Disable AI for simplicity
            data_days_lookback=2,  # Just 2 days
            db_timeout=10  # Short timeout
        )
        
        print("2. Testing data loading...")
        # Try to load a small amount of data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        data = strategy.load_real_data(start_date, end_date)
        
        if data.empty:
            print("⚠️ No data loaded, but no error occurred")
            return False
        
        print(f"✅ Loaded {len(data)} bars of data")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Columns: {list(data.columns)}")
        
        print("3. Testing feature preparation...")
        features = strategy.prepare_features(data)
        print(f"✅ Created {features.shape[1]} features from {features.shape[0]} bars")
        
        print("4. Testing signal generation...")
        signals = strategy.generate_signals(features)
        active_signals = (signals != 0).sum()
        print(f"✅ Generated {active_signals} active signals")
        
        print("5. Testing simple backtest...")
        if len(data) > 50:  # Only if we have enough data
            results = strategy.backtest(data, verbose=False)
            print(f"✅ Backtest completed:")
            print(f"   Total trades: {results.get('total_trades', 0)}")
            print(f"   PnL: ${results.get('total_pnl', 0):.2f}")
            print(f"   Data source: {results.get('data_source', 'Unknown')}")
        else:
            print("⚠️ Not enough data for backtest")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_connection():
    """Test just the database connection."""
    print("\n🗄️ Testing Database Connection...")
    print("-" * 30)
    
    try:
        from day_trading_o3 import DataLoader
        
        print("1. Creating DataLoader...")
        loader = DataLoader()
        print("✅ DataLoader created successfully")
        
        print("2. Testing query...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        data = loader.get_price_data('AAPL', start_date, end_date)
        
        if data.empty:
            print("⚠️ Query successful but no data returned")
        else:
            print(f"✅ Query successful: {len(data)} bars returned")
            
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 S3 Real Data Strategy - Simple Test Suite")
    print("=" * 50)
    
    # Test database first
    db_success = test_database_connection()
    
    if db_success:
        print("\n📊 Database connection successful, testing strategy...")
        strategy_success = test_basic_functionality()
        
        if strategy_success:
            print("\n✅ All tests passed! Strategy is working with real data.")
        else:
            print("\n⚠️ Strategy test failed, but database connection works.")
    else:
        print("\n❌ Database connection failed. Strategy will use synthetic data.")
        
        # Test with synthetic data fallback
        print("\n🔄 Testing with synthetic data fallback...")
        try:
            strategy = S3WorkingStrategyRealData(
                symbol='AAPL',
                capital=10000,
                adaptive_indicators=False,
                data_days_lookback=2
            )
            
            # This should trigger synthetic data generation
            data = strategy.load_real_data()
            print(f"✅ Synthetic data generated: {len(data)} bars")
            
        except Exception as e:
            print(f"❌ Even synthetic data failed: {e}")


if __name__ == "__main__":
    main()