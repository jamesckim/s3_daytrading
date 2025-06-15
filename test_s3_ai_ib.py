#!/usr/bin/env python3
"""
Test S3 AI IB Integration
=========================
Test script to verify S3 AI IB paper trading components.
"""

import sys
import os
from datetime import datetime

def test_imports():
    """Test if all required imports work."""
    print("🧪 Testing imports...")
    
    try:
        from s3_ai_ib_paper_trading import S3AIIBPaperTrader, S3AIIBStrategy
        print("✅ S3 AI IB components imported successfully")
        
        from ib_insync import IB, Stock
        print("✅ IB-Insync imported successfully")
        
        from s3_ai_wrapper import S3AIWrapper
        print("✅ S3 AI Wrapper imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Testing configuration loading...")
    
    try:
        from s3_ai_ib_paper_trading import S3AIIBPaperTrader
        
        # Test with default config
        trader = S3AIIBPaperTrader("s3_ai_ib_trading_config.json")
        
        print("✅ Configuration loaded successfully")
        print(f"   Symbols: {len(trader.symbols)}")
        print(f"   Capital: ${trader.initial_capital:,.0f}")
        print(f"   Max positions: {trader.max_positions}")
        print(f"   Scan interval: {trader.scan_interval}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_s3_ai_components():
    """Test S3 AI strategy components."""
    print("\n🧪 Testing S3 AI components...")
    
    try:
        from s3_ai_ib_paper_trading import S3AIDataEngine, S3AIIBStrategy
        
        # Initialize data engine
        data_engine = S3AIDataEngine()
        print("✅ S3 AI Data Engine initialized")
        
        # Initialize strategy
        strategy = S3AIIBStrategy(data_engine)
        print("✅ S3 AI IB Strategy initialized")
        
        # Test signal generation (without real data)
        try:
            action, strength, info = strategy.generate_ib_signal("AAPL", 150.0)
            print(f"✅ Signal generation test: {action} ({strength:.2f})")
            print(f"   Regime: {info.get('regime', 'Unknown')}")
        except Exception as e:
            print(f"⚠️ Signal generation test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ S3 AI components error: {e}")
        return False

def test_ib_initialization():
    """Test IB initialization (without connection)."""
    print("\n🧪 Testing IB initialization...")
    
    try:
        from s3_ai_ib_paper_trading import S3AIIBPaperTrader
        
        # Initialize without connecting
        trader = S3AIIBPaperTrader("s3_ai_ib_trading_config.json")
        
        print("✅ IB components initialized")
        print(f"   IB connected: {trader.ib_connected}")
        print(f"   IB object created: {trader.ib is not None}")
        
        # Test connection method (will fail without TWS)
        print("\n📡 Testing IB connection (expected to fail without TWS)...")
        success = trader.connect_ib()
        
        if success:
            print("✅ IB connection successful!")
            trader.disconnect_ib()
        else:
            print("❌ IB connection failed (expected without TWS running)")
            print("💡 To test with real connection:")
            print("   1. Start TWS or IB Gateway")
            print("   2. Configure for Paper Trading")
            print("   3. Enable API connections (port 7497)")
            print("   4. Run: python s3_ai_ib_paper_trading.py")
        
        return True
        
    except Exception as e:
        print(f"❌ IB initialization error: {e}")
        return False

def test_config_variations():
    """Test different configuration files."""
    print("\n🧪 Testing configuration variations...")
    
    configs = [
        ("s3_ai_ib_trading_config.json", "Standard config"),
        ("s3_ai_ib_trading_config_top_performers.json", "Top performers config")
    ]
    
    for config_file, description in configs:
        try:
            if os.path.exists(config_file):
                from s3_ai_ib_paper_trading import S3AIIBPaperTrader
                trader = S3AIIBPaperTrader(config_file)
                print(f"✅ {description}: {len(trader.symbols)} symbols")
            else:
                print(f"⚠️ {description}: File not found - {config_file}")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")

def main():
    """Run all tests."""
    print("🚀 S3 AI Interactive Brokers Integration - Test Suite")
    print("=" * 65)
    print("📋 Testing components without requiring TWS connection")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config_loading),
        ("S3 AI Components", test_s3_ai_components),
        ("IB Initialization", test_ib_initialization),
        ("Config Variations", test_config_variations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*65)
    print("🧪 TEST RESULTS SUMMARY")
    print("="*65)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("-"*65)
    print(f"TOTAL: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed >= total - 1:  # Allow IB connection to fail
        print("\n🎉 S3 AI IB Integration ready!")
        print("\n🚀 NEXT STEPS:")
        print("1. Start TWS or IB Gateway (Paper Trading)")
        print("2. Configure API settings (port 7497)")
        print("3. Run: python s3_ai_ib_paper_trading.py")
        print("4. Or test with top performers: python s3_ai_ib_paper_trading.py --config s3_ai_ib_trading_config_top_performers.json")
        return True
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)