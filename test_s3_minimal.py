#!/usr/bin/env python3
"""
Test S3 Minimal Strategy - Without database dependencies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def test_s3_minimal():
    """Test S3 minimal strategy with synthetic data."""
    
    print("🧪 Testing S3 Minimal Strategy (Database-Free)")
    print("=" * 55)
    
    # Test 1: Import
    try:
        from s3_minimal_strategy import S3MinimalStrategy
        print("✅ S3 minimal strategy import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 2: Create synthetic data
    print("\n📊 Creating synthetic test data...")
    dates = pd.date_range(start='2024-01-01', periods=2000, freq='1min')
    np.random.seed(42)
    
    # Generate realistic price data with some trends
    returns = np.random.normal(0, 0.001, len(dates))
    # Add some trending periods
    returns[500:800] += 0.0002  # Uptrend
    returns[1200:1500] -= 0.0002  # Downtrend
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(dates)))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    print(f"✅ Created {len(test_data)} bars of synthetic data")
    print(f"   Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    
    # Test 3: Initialize strategy
    print("\n🚀 Initializing S3 minimal strategy...")
    try:
        strategy = S3MinimalStrategy(
            symbol='TEST',
            capital=100000,
            adaptive_indicators=True,
            top_k_indicators=15,
            indicator_selection_interval=100
        )
        print("✅ Strategy initialized successfully")
    except Exception as e:
        print(f"❌ Strategy initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Prepare features
    print("\n📈 Testing feature preparation...")
    try:
        features = strategy.prepare_features(test_data)
        print(f"✅ Features prepared: {features.shape[1]} features, {features.shape[0]} rows")
        print(f"   Selected indicators: {len(strategy.current_indicators)}")
        print(f"   Market regime: {strategy.last_market_regime}")
        
        # Show feature names
        print(f"   Feature names: {list(features.columns[:10])}...")
        
    except Exception as e:
        print(f"❌ Feature preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Generate signals
    print("\n🎯 Testing signal generation...")
    try:
        signals = strategy.generate_signals(features)
        print(f"✅ Signals generated: {len(signals)} signals")
        
        # Count signal distribution
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        neutral_signals = (signals == 0).sum()
        
        print(f"   Long: {long_signals}, Short: {short_signals}, Neutral: {neutral_signals}")
        
        if long_signals + short_signals > 0:
            print(f"   Signal rate: {(long_signals + short_signals) / len(signals):.2%}")
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 6: Run backtest
    print("\n📊 Testing backtest functionality...")
    try:
        results = strategy.backtest(test_data)
        
        print("✅ Backtest completed successfully")
        print(f"   Total trades: {results['total_trades']}")
        print(f"   Win rate: {results['win_rate']:.2%}")
        print(f"   Total PnL: ${results['total_pnl']:.2f}")
        print(f"   Final capital: ${results['final_capital']:.2f}")
        print(f"   Profit factor: {results['profit_factor']:.2f}")
        
        if results['total_trades'] > 0:
            print(f"   Average win: ${results['avg_win']:.2f}")
            print(f"   Average loss: ${results['avg_loss']:.2f}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 7: Indicator selection details
    print("\n🔍 Indicator Selection Details:")
    if strategy.current_indicators:
        sorted_indicators = sorted(
            strategy.current_indicators.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )[:8]
        
        for indicator_name, info in sorted_indicators:
            print(f"   {indicator_name}: weight={info['weight']:.3f}, "
                  f"confidence={info['selection_prob']:.3f}")
    else:
        print("   No indicators selected")
    
    # Test 8: Performance report
    print("\n📈 Testing performance report...")
    try:
        report = strategy.get_performance_report()
        print("✅ Performance report generated")
        
        if 'transformer_metrics' in report:
            tm = report['transformer_metrics']
            print(f"   Transformer: {tm.get('unique_indicators_used', 0)} unique indicators used")
            print(f"   Selection efficiency: {tm.get('selection_efficiency', 0):.2%}")
        
    except Exception as e:
        print(f"❌ Performance report failed: {e}")
    
    print("\n🎉 S3 Minimal Strategy test completed successfully!")
    print(f"📊 Summary: {results['total_trades']} trades, "
          f"{results['win_rate']:.1%} win rate, "
          f"${results['total_pnl']:.2f} PnL")
    
    return strategy, test_data, results


def test_with_database():
    """Test with actual database if available."""
    print("\n" + "="*60)
    print("🗄️ Testing with actual database...")
    
    try:
        from s3_minimal_strategy import S3MinimalStrategy
        from day_trading_o3 import DataLoader
        
        # Try to connect to database
        data_loader = DataLoader()
        print("✅ Database connection successful")
        
        # Load real data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)  # Just 3 days
        
        data = data_loader.get_price_data(
            'AAPL',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("❌ No AAPL data available")
            return
        
        print(f"✅ Loaded {len(data)} bars of real AAPL data")
        
        # Create strategy and test
        strategy = S3MinimalStrategy(
            symbol='AAPL',
            capital=100000,
            adaptive_indicators=True,
            top_k_indicators=12
        )
        
        # Quick test
        features = strategy.prepare_features(data)
        signals = strategy.generate_signals(features)
        
        print(f"✅ Real data test: {features.shape[1]} features, "
              f"{(signals != 0).sum()} signals generated")
        print(f"   Market regime detected: {strategy.last_market_regime}")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        print("   (This is normal if database isn't set up)")


if __name__ == "__main__":
    # Setup minimal logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Test with synthetic data (always works)
    test_s3_minimal()
    
    # Test with database (if available)
    test_with_database()