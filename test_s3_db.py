#!/usr/bin/env python3
"""
Test S3 Database Strategy - Quick validation with real data
"""

import sys
import signal
from datetime import datetime, timedelta

def timeout_handler(signum, frame):
    print(f"\n❌ Database test timed out")
    sys.exit(1)

def test_s3_database():
    """Test S3 database strategy with real data."""
    
    print("🗄️ Testing S3 Database Strategy")
    print("=" * 45)
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)  # 2 minute timeout
    
    try:
        # Test 1: Import and database connection
        print("📦 Testing imports and database connection...")
        from s3_db_strategy import S3DatabaseStrategy
        
        # Test basic connection first
        strategy = S3DatabaseStrategy(
            symbol='AAPL',
            capital=50000,
            adaptive_indicators=False,  # Start without AI for speed
            data_days_lookback=3  # Just 3 days for quick test
        )
        print("✅ Database connection successful")
        
        # Test 2: Data loading
        print("\n📊 Testing data loading...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        data = strategy.load_historical_data(start_date, end_date)
        print(f"✅ Loaded {len(data):,} bars of real AAPL data")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Test 3: Feature preparation
        print("\n📈 Testing feature preparation...")
        features = strategy.prepare_features(data)
        print(f"✅ Prepared {features.shape[1]} features")
        
        # Test 4: Signal generation
        print("\n🎯 Testing signal generation...")
        signals = strategy.generate_signals(features, data)
        active_signals = (signals != 0).sum()
        print(f"✅ Generated {active_signals} active signals out of {len(signals)}")
        
        # Test 5: Quick backtest
        print("\n💼 Running quick backtest...")
        results = strategy.backtest(start_date, end_date, verbose=False)
        
        print(f"✅ Backtest completed:")
        print(f"   Total trades: {results['total_trades']}")
        print(f"   Win rate: {results['win_rate']:.1%}")
        print(f"   Total PnL: ${results['total_pnl']:.2f}")
        print(f"   Market regime: {results['market_regime']}")
        
        # Test 6: Latest signals
        print("\n🔍 Testing latest signal analysis...")
        latest = strategy.get_latest_signals(lookback_hours=24)
        
        if 'error' not in latest:
            signal_text = 'BUY' if latest['signal'] > 0 else 'SELL' if latest['signal'] < 0 else 'HOLD'
            print(f"✅ Latest signal analysis:")
            print(f"   Current price: ${latest['price']:.2f}")
            print(f"   Signal: {signal_text}")
            print(f"   Regime: {latest['market_regime']}")
        
        print(f"\n🎉 Database strategy test completed successfully!")
        
        return strategy, results
        
    except ConnectionError as e:
        print(f"❌ Database connection failed: {e}")
        print("   Make sure PostgreSQL is running and .env is configured")
        return None, None
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        signal.alarm(0)  # Cancel timeout


def test_with_ai():
    """Test with AI indicator selection enabled."""
    
    print("\n" + "="*50)
    print("🤖 Testing with AI Indicator Selection")
    print("="*50)
    
    try:
        # Create AI-enabled strategy
        strategy = S3DatabaseStrategy(
            symbol='AAPL',
            capital=50000,
            adaptive_indicators=True,  # Enable AI
            top_k_indicators=10,
            data_days_lookback=5
        )
        
        # Quick test
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        print("📊 Loading data for AI analysis...")
        data = strategy.load_historical_data(start_date, end_date)
        
        print("🎯 Running AI indicator selection...")
        features = strategy.prepare_features(data)
        
        print(f"✅ AI strategy results:")
        print(f"   Features prepared: {features.shape[1]}")
        print(f"   Market regime detected: {strategy.last_market_regime}")
        print(f"   AI indicators selected: {len(strategy.current_indicators)}")
        
        if strategy.current_indicators:
            print("   Top indicators:")
            top_indicators = sorted(
                strategy.current_indicators.items(),
                key=lambda x: x[1]['weight'],
                reverse=True
            )[:3]
            
            for name, info in top_indicators:
                print(f"     {name}: weight={info['weight']:.3f}")
        
        # Quick backtest with AI
        print("\n💼 Running AI-enhanced backtest...")
        results = strategy.backtest(start_date, end_date, verbose=False)
        
        print(f"✅ AI backtest results:")
        print(f"   Trades: {results['total_trades']}")
        print(f"   Win rate: {results['win_rate']:.1%}")
        print(f"   PnL: ${results['total_pnl']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"❌ AI test failed: {e}")
        return None


if __name__ == "__main__":
    # Test basic functionality first
    strategy, results = test_s3_database()
    
    # Test AI functionality if basic works
    if strategy is not None:
        ai_results = test_with_ai()
        
        if results and ai_results:
            print(f"\n📊 Comparison:")
            print(f"   Basic Strategy: {results['total_trades']} trades, "
                  f"{results['win_rate']:.1%} win rate, ${results['total_pnl']:.2f} PnL")
            print(f"   AI Strategy:    {ai_results['total_trades']} trades, "
                  f"{ai_results['win_rate']:.1%} win rate, ${ai_results['total_pnl']:.2f} PnL")
            
            if ai_results['total_pnl'] > results['total_pnl']:
                improvement = ai_results['total_pnl'] - results['total_pnl']
                print(f"🏆 AI strategy outperformed by ${improvement:.2f}!")
            else:
                print("📊 Similar performance between strategies")
    else:
        print("\n💡 Database setup required:")
        print("   1. Ensure PostgreSQL is running")
        print("   2. Check .env file has POSTGRES_URI")
        print("   3. Load historical data into database")
        print("   4. Run: python s3_db_strategy.py")