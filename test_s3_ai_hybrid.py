#!/usr/bin/env python3
"""
Test S3 AI Hybrid Paper Trading System
======================================
Test script to validate the S3 AI hybrid paper trading system components.
"""

import sys
import os
from datetime import datetime
import pandas as pd

def test_data_engine():
    """Test the S3 AI Data Engine."""
    print("🧪 Testing S3 AI Data Engine...")
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIDataEngine
        
        # Initialize engine
        engine = S3AIDataEngine()
        print("✅ Data engine initialized successfully")
        
        # Test data fetching
        test_symbol = "AAPL"
        print(f"📊 Testing data fetch for {test_symbol}...")
        
        data = engine.get_recent_data(test_symbol, bars=100)
        if not data.empty:
            print(f"✅ Data fetch successful: {len(data)} bars")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        else:
            print(f"❌ No data returned for {test_symbol}")
            return False
        
        # Test AI analysis
        print(f"🧠 Testing AI analysis for {test_symbol}...")
        ai_analysis = engine.get_ai_analysis(test_symbol)
        
        regime = ai_analysis.get("regime", "Unknown")
        confidence = ai_analysis.get("confidence", 0.0)
        indicators = ai_analysis.get("indicators", {})
        
        print(f"✅ AI analysis successful:")
        print(f"   Regime: {regime}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Indicators selected: {len(indicators)}")
        print(f"   Active indicators: {list(indicators.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trading_strategy():
    """Test the S3 AI Trading Strategy."""
    print("\n🧪 Testing S3 AI Trading Strategy...")
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIDataEngine, S3AITradingStrategy
        
        # Initialize components
        engine = S3AIDataEngine()
        strategy = S3AITradingStrategy(engine)
        print("✅ Trading strategy initialized successfully")
        
        # Test signal generation
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in test_symbols:
            print(f"🔍 Testing signal generation for {symbol}...")
            
            # Get current price (use recent data)
            data = engine.get_recent_data(symbol, bars=1)
            if data.empty:
                print(f"❌ No data for {symbol}")
                continue
            
            current_price = float(data['close'].iloc[-1])
            
            # Generate signal
            action, strength, signal_info = strategy.generate_signal(symbol, current_price)
            
            regime = signal_info.get("regime", "Unknown")
            confidence = signal_info.get("confidence", 0.0)
            indicators_active = signal_info.get("indicators_active", 0)
            
            print(f"✅ Signal generated for {symbol}:")
            print(f"   Action: {action}")
            print(f"   Strength: {strength:.2f}")
            print(f"   Regime: {regime}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Active indicators: {indicators_active}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_manager():
    """Test the Portfolio Manager."""
    print("\n🧪 Testing S3 AI Portfolio Manager...")
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIPortfolioManager
        
        # Initialize portfolio
        portfolio = S3AIPortfolioManager(initial_capital=100000, max_positions=10)
        print("✅ Portfolio manager initialized successfully")
        print(f"   Initial capital: ${portfolio.initial_capital:,.0f}")
        print(f"   Max positions: {portfolio.max_positions}")
        
        # Test position opening
        test_symbol = "AAPL"
        test_price = 150.0
        test_strength = 0.8
        test_regime = "High Volatility"
        
        print(f"📈 Testing position opening for {test_symbol}...")
        success = portfolio.open_position(test_symbol, "BUY", test_price, test_strength, test_regime)
        
        if success:
            print(f"✅ Position opened successfully")
            
            # Check position
            if test_symbol in portfolio.positions:
                pos = portfolio.positions[test_symbol]
                print(f"   Position: {pos.quantity} shares @ ${pos.entry_price:.2f}")
                print(f"   Regime: {pos.regime}")
        else:
            print(f"❌ Failed to open position")
            return False
        
        # Test portfolio summary
        summary = portfolio.get_portfolio_summary()
        print(f"💼 Portfolio summary:")
        print(f"   Total value: ${summary['total_value']:,.2f}")
        print(f"   Positions: {summary['positions_count']}")
        print(f"   Total trades: {summary['total_trades']}")
        
        # Test position closing
        print(f"📉 Testing position closing...")
        close_price = 155.0  # 5 profit
        success = portfolio.close_position(test_symbol, close_price, "Test Close")
        
        if success:
            print(f"✅ Position closed successfully")
            
            # Check final summary
            final_summary = portfolio.get_portfolio_summary()
            print(f"   Final total value: ${final_summary['total_value']:,.2f}")
            print(f"   Total trades: {final_summary['total_trades']}")
            print(f"   Win rate: {final_summary['win_rate']:.1f}%")
        else:
            print(f"❌ Failed to close position")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration Loading...")
    
    try:
        config_file = "s3_ai_trading_config.json"
        
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print(f"✅ Configuration loaded successfully")
            print(f"   Symbols: {len(config.get('symbols', []))}")
            print(f"   Initial capital: ${config.get('initial_capital', 0):,.0f}")
            print(f"   Max positions: {config.get('max_positions', 0)}")
            print(f"   Scan interval: {config.get('scan_interval', 0)}s")
            
            return True
        else:
            print(f"❌ Configuration file not found: {config_file}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_full_system():
    """Test the complete hybrid system initialization."""
    print("\n🧪 Testing Complete S3 AI Hybrid System...")
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIHybridPaperTrader
        
        # Initialize trader (but don't run)
        trader = S3AIHybridPaperTrader()
        print("✅ S3 AI Hybrid Paper Trader initialized successfully")
        
        # Test market hours check
        is_open = trader.is_market_open()
        print(f"📅 Market status: {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
        
        # Test price fetching
        print("💰 Testing price fetching...")
        prices = trader.get_current_prices()
        
        if prices:
            print(f"✅ Price fetch successful: {len(prices)} symbols")
            for symbol, price in list(prices.items())[:5]:  # Show first 5
                print(f"   {symbol}: ${price:.2f}")
        else:
            print("❌ No prices fetched")
            return False
        
        # Test portfolio status
        print("📊 Testing portfolio status...")
        trader.print_portfolio_status()
        
        return True
        
    except Exception as e:
        print(f"❌ Full system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 S3 AI Hybrid Paper Trading System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Data Engine", test_data_engine),
        ("Trading Strategy", test_trading_strategy),
        ("Portfolio Manager", test_portfolio_manager),
        ("Configuration", test_configuration),
        ("Full System", test_full_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            print(f"💥 {test_name} test CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("🧪 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-"*60)
    print(f"TOTAL: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)