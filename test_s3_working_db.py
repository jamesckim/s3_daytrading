#!/usr/bin/env python3
"""
Test S3 Working Database Strategy - Quick validation
"""

import sys
import signal
from datetime import datetime

def timeout_handler(signum, frame):
    print(f"\n❌ Test timed out")
    sys.exit(1)

def test_s3_working_db():
    """Test S3 working database strategy."""
    
    print("🗄️ Testing S3 Working Database Strategy")
    print("=" * 45)
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(90)  # 90 second timeout
    
    try:
        # Test import
        print("📦 Testing imports...")
        from s3_working_db_strategy import S3WorkingDatabaseStrategy
        print("✅ Import successful")
        
        # Test basic initialization (no AI to avoid hanging)
        print("\n🚀 Testing basic initialization...")
        strategy = S3WorkingDatabaseStrategy(
            symbol='AAPL',
            capital=50000,
            adaptive_indicators=False,  # Start without AI
            data_days_lookback=2  # Just 2 days for speed
        )
        print("✅ Strategy initialized")
        
        # Test data loading
        print("\n📊 Testing data loading...")
        data = strategy.load_historical_data()
        print(f"✅ Data loaded: {len(data):,} bars")
        print(f"   Data source: {'database' if strategy.data_loader else 'synthetic'}")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Test feature preparation
        print("\n📈 Testing feature preparation...")
        features = strategy.prepare_features(data)
        print(f"✅ Features prepared: {features.shape[1]} columns")
        
        # Test signal generation
        print("\n🎯 Testing signal generation...")
        signals = strategy.generate_signals(features, data)
        active_signals = (signals != 0).sum()
        print(f"✅ Signals generated: {active_signals} active signals")
        
        # Test quick backtest
        print("\n💼 Running quick backtest...")
        results = strategy.backtest(verbose=False)
        
        print(f"✅ Backtest completed:")
        print(f"   Total trades: {results['total_trades']}")
        print(f"   Win rate: {results['win_rate']:.1%}")
        print(f"   Total PnL: ${results['total_pnl']:.2f}")
        print(f"   Data source: {results['data_source']}")
        
        # Test latest signals
        print("\n🔍 Testing latest signal analysis...")
        latest = strategy.get_latest_signals(lookback_hours=12)
        
        if 'error' not in latest:
            signal_text = 'BUY' if latest['signal'] > 0 else 'SELL' if latest['signal'] < 0 else 'HOLD'
            print(f"✅ Latest signal analysis:")
            print(f"   Current price: ${latest['price']:.2f}")
            print(f"   Signal: {signal_text}")
            print(f"   Data source: {latest['data_source']}")
        
        print(f"\n🎉 S3 Working Database Strategy test completed successfully!")
        
        return strategy, results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        signal.alarm(0)  # Cancel timeout


def test_with_ai():
    """Test with AI if basic version works."""
    
    print("\n" + "="*50)
    print("🤖 Testing with AI Indicator Selection")
    print("="*50)
    
    try:
        from s3_working_db_strategy import S3WorkingDatabaseStrategy
        
        # Create AI-enabled strategy
        strategy = S3WorkingDatabaseStrategy(
            symbol='AAPL',
            capital=50000,
            adaptive_indicators=True,  # Enable AI
            top_k_indicators=5,  # Small number for speed
            data_days_lookback=3
        )
        
        print("📊 Running AI-enhanced test...")
        results = strategy.backtest(verbose=False)
        
        print(f"✅ AI test results:")
        print(f"   Trades: {results['total_trades']}")
        print(f"   Win rate: {results['win_rate']:.1%}")
        print(f"   PnL: ${results['total_pnl']:.2f}")
        print(f"   Data source: {results['data_source']}")
        print(f"   Regime: {results['market_regime']}")
        print(f"   Indicators selected: {results['indicators_selected']}")
        
        return results
        
    except Exception as e:
        print(f"❌ AI test failed: {e}")
        return None


if __name__ == "__main__":
    # Test basic functionality first
    strategy, results = test_s3_working_db()
    
    # Test AI functionality if basic works
    if strategy is not None:
        ai_results = test_with_ai()
        
        if results and ai_results:
            print(f"\n📊 Comparison:")
            print(f"   Basic Strategy: {results['total_trades']} trades, "
                  f"{results['win_rate']:.1%} win rate, ${results['total_pnl']:.2f} PnL")
            print(f"   AI Strategy:    {ai_results['total_trades']} trades, "
                  f"{ai_results['win_rate']:.1%} win rate, ${ai_results['total_pnl']:.2f} PnL")
    else:
        print("\n💡 Setup notes:")
        print("   - Database strategy will use synthetic data if PostgreSQL unavailable")
        print("   - AI features will be disabled if transformer initialization fails")
        print("   - Check database connection for real data usage")