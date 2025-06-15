#!/usr/bin/env python3
"""
Test S3 Transformer Strategy - Quick validation without database
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

def test_s3_strategy():
    """Quick test of S3 strategy without database dependencies."""
    
    print("🧪 Testing S3 Transformer-Enhanced Strategy")
    print("=" * 50)
    
    # Test 1: Import
    try:
        from s3_transformer_enhanced_strategy import S3TransformerEnhancedStrategy
        print("✅ S3 strategy import successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test 2: Create synthetic data
    print("\n📊 Creating synthetic test data...")
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.001, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.0001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(dates)))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    print(f"✅ Created {len(test_data)} bars of synthetic data")
    
    # Test 3: Initialize strategy
    print("\n🚀 Initializing S3 strategy...")
    try:
        strategy = S3TransformerEnhancedStrategy(
            df=test_data,
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
    except Exception as e:
        print(f"❌ Feature preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Generate signals
    print("\n🎯 Testing signal generation...")
    try:
        # Use only valid features (no NaN)
        valid_features = features.dropna()
        if len(valid_features) > 0:
            signals = strategy.generate_signals(valid_features)
            print(f"✅ Signals generated: {len(signals)} signals")
            
            # Count signal distribution
            long_signals = (signals == 1).sum()
            short_signals = (signals == -1).sum()
            neutral_signals = (signals == 0).sum()
            
            print(f"   Long: {long_signals}, Short: {short_signals}, Neutral: {neutral_signals}")
        else:
            print("⚠️ No valid features available for signal generation")
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 6: Indicator selection details
    print("\n🔍 Indicator Selection Details:")
    if strategy.current_indicators:
        sorted_indicators = sorted(
            strategy.current_indicators.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )[:5]
        
        for indicator_name, info in sorted_indicators:
            print(f"   {indicator_name}: weight={info['weight']:.3f}, "
                  f"confidence={info['selection_prob']:.3f}")
    else:
        print("   No indicators selected")
    
    # Test 7: Performance report
    print("\n📊 Testing performance report...")
    try:
        report = strategy.get_strategy_report()
        print("✅ Performance report generated")
        
        if 'transformer_metrics' in report:
            tm = report['transformer_metrics']
            print(f"   Transformer metrics: {tm.get('unique_indicators_used', 0)} indicators used")
        
    except Exception as e:
        print(f"❌ Performance report failed: {e}")
    
    print("\n🎉 S3 Strategy test completed successfully!")
    
    return strategy, test_data, features


if __name__ == "__main__":
    # Setup minimal logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    test_s3_strategy()