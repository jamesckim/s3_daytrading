#!/usr/bin/env python3
"""
Test S3 AI FMP Integration
==========================
Test script to verify S3 AI FMP hybrid trading components.
"""

import sys
import os
from datetime import datetime

def test_fmp_api_key():
    """Test if FMP API key is available."""
    print("🧪 Testing FMP API access...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("FMP_API_KEY")
        if api_key:
            print(f"✅ FMP API key found: ...{api_key[-8:]}")
            return True
        else:
            print("❌ FMP_API_KEY not found in environment")
            print("💡 Make sure FMP_API_KEY is set in .env file")
            return False
        
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def test_fmp_imports():
    """Test if all FMP components import correctly."""
    print("\n🧪 Testing FMP imports...")
    
    try:
        from s3_ai_fmp_hybrid_trading import (
            FMPDataProvider, S3AIFMPStrategy, S3AIFMPHybridTrader, FMPQuote
        )
        print("✅ FMP hybrid components imported successfully")
        
        import requests
        print("✅ Requests library available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_fmp_data_provider():
    """Test FMP data provider initialization."""
    print("\n🧪 Testing FMP Data Provider...")
    
    try:
        from s3_ai_fmp_hybrid_trading import FMPDataProvider
        
        provider = FMPDataProvider()
        print("✅ FMP Data Provider initialized")
        print(f"   Rate limit: {provider.request_limit} requests/minute")
        print(f"   Cache duration: {provider.cache_duration} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ FMP Data Provider error: {e}")
        return False

def test_fmp_real_time_data():
    """Test FMP real-time data fetching."""
    print("\n🧪 Testing FMP real-time data...")
    
    try:
        from s3_ai_fmp_hybrid_trading import FMPDataProvider
        
        provider = FMPDataProvider()
        
        # Test single quote
        test_symbols = ["AAPL"]
        print(f"📡 Fetching real-time quote for {test_symbols[0]}...")
        
        quotes = provider.get_real_time_quotes(test_symbols)
        
        if quotes and test_symbols[0] in quotes:
            quote = quotes[test_symbols[0]]
            print(f"✅ Real-time data received:")
            print(f"   Symbol: {quote.symbol}")
            print(f"   Price: ${quote.price:.2f}")
            print(f"   Bid: ${quote.bid:.2f}")
            print(f"   Ask: ${quote.ask:.2f}")
            print(f"   Volume: {quote.volume:,}")
            print(f"   Change: {quote.change_percent:+.2f}%")
            return True
        else:
            print("❌ No real-time data received")
            print("💡 Check FMP API key and rate limits")
            return False
        
    except Exception as e:
        print(f"❌ FMP real-time data error: {e}")
        return False

def test_fmp_market_status():
    """Test FMP market status check."""
    print("\n🧪 Testing FMP market status...")
    
    try:
        from s3_ai_fmp_hybrid_trading import FMPDataProvider
        
        provider = FMPDataProvider()
        
        print("📅 Checking market status...")
        is_open = provider.is_market_open()
        
        status = "🟢 OPEN" if is_open else "🔴 CLOSED"
        print(f"✅ Market status check successful: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market status error: {e}")
        return False

def test_s3_ai_fmp_strategy():
    """Test S3 AI FMP strategy integration."""
    print("\n🧪 Testing S3 AI FMP Strategy...")
    
    try:
        from s3_ai_fmp_hybrid_trading import (
            S3AIDataEngine, FMPDataProvider, S3AIFMPStrategy, FMPQuote
        )
        
        # Initialize components
        data_engine = S3AIDataEngine()
        fmp_provider = FMPDataProvider()
        strategy = S3AIFMPStrategy(data_engine, fmp_provider)
        
        print("✅ S3 AI FMP Strategy initialized")
        
        # Create mock quote for testing
        mock_quote = FMPQuote(
            symbol="AAPL",
            price=225.50,
            bid=225.45,
            ask=225.55,
            volume=1000000,
            timestamp=datetime.now(),
            change=2.50,
            change_percent=1.12
        )
        
        # Test signal generation
        print("🧠 Testing signal generation with mock data...")
        action, strength, info = strategy.generate_fmp_signal("AAPL", mock_quote)
        
        print(f"✅ Signal generated:")
        print(f"   Action: {action}")
        print(f"   Strength: {strength:.2f}")
        print(f"   Regime: {info.get('regime', 'Unknown')}")
        print(f"   FMP optimized: {info.get('fmp_optimized', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ S3 AI FMP Strategy error: {e}")
        return False

def test_fmp_hybrid_system():
    """Test complete FMP hybrid system."""
    print("\n🧪 Testing FMP Hybrid System...")
    
    try:
        from s3_ai_fmp_hybrid_trading import S3AIFMPHybridTrader
        
        # Initialize with standard config
        trader = S3AIFMPHybridTrader("s3_ai_fmp_trading_config.json")
        
        print("✅ FMP Hybrid System initialized")
        print(f"   Symbols: {len(trader.symbols)}")
        print(f"   Capital: ${trader.initial_capital:,.0f}")
        print(f"   Max positions: {trader.max_positions}")
        print(f"   Scan interval: {trader.scan_interval}s")
        
        # Test market status
        is_open = trader.is_market_open()
        status = "🟢 OPEN" if is_open else "🔴 CLOSED"
        print(f"   Market status: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ FMP Hybrid System error: {e}")
        return False

def test_config_variations():
    """Test different configuration files."""
    print("\n🧪 Testing FMP configuration variations...")
    
    configs = [
        ("s3_ai_fmp_trading_config.json", "Standard config"),
        ("s3_ai_fmp_trading_config_top_performers.json", "Top performers config")
    ]
    
    for config_file, description in configs:
        try:
            if os.path.exists(config_file):
                from s3_ai_fmp_hybrid_trading import S3AIFMPHybridTrader
                trader = S3AIFMPHybridTrader(config_file)
                print(f"✅ {description}: {len(trader.symbols)} symbols")
            else:
                print(f"⚠️ {description}: File not found - {config_file}")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")

def main():
    """Run all FMP tests."""
    print("🚀 S3 AI FMP Hybrid Trading System - Test Suite")
    print("=" * 60)
    print("📡 Testing FMP real-time data integration")
    print()
    
    tests = [
        ("FMP API Key", test_fmp_api_key),
        ("FMP Imports", test_fmp_imports),
        ("FMP Data Provider", test_fmp_data_provider),
        ("FMP Real-time Data", test_fmp_real_time_data),
        ("FMP Market Status", test_fmp_market_status),
        ("S3 AI FMP Strategy", test_s3_ai_fmp_strategy),
        ("FMP Hybrid System", test_fmp_hybrid_system),
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
    print("\n" + "="*60)
    print("🧪 FMP TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-"*60)
    print(f"TOTAL: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed >= total - 1:  # Allow one test to fail (real-time data might fail)
        print("\n🎉 S3 AI FMP Hybrid System ready!")
        print("\n🚀 NEXT STEPS:")
        print("1. Ensure FMP_API_KEY is set in .env file")
        print("2. Run: python s3_ai_fmp_hybrid_trading.py")
        print("3. Or with top performers: modify config in the script")
        print("4. Monitor real-time trading with FMP data")
        return True
    else:
        print("⚠️ Some tests failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)