#!/usr/bin/env python3
"""
Quick S3 AI System Test
========================
Simplified test script to verify core functionality.
"""

import os
from dotenv import load_dotenv
load_dotenv()

def test_data_engine():
    """Test data engine with single symbol."""
    print("🧪 Testing S3 AI Data Engine...")
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIDataEngine
        engine = S3AIDataEngine()
        
        # Test with single symbol
        data = engine.get_recent_data("AAPL", bars=10)
        if not data.empty:
            print(f"✅ Data fetch successful: {len(data)} bars")
            return True
        else:
            print("❌ No data returned")
            return False
    except Exception as e:
        print(f"❌ Data engine test failed: {e}")
        return False

def test_ai_analysis():
    """Test AI analysis component."""
    print("\n🧪 Testing AI Analysis...")
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIDataEngine
        engine = S3AIDataEngine()
        
        # Test AI analysis
        ai_analysis = engine.get_ai_analysis("AAPL")
        regime = ai_analysis.get("regime", "Unknown")
        confidence = ai_analysis.get("confidence", 0.0)
        indicators = ai_analysis.get("indicators", {})
        
        print(f"✅ AI analysis successful:")
        print(f"   Regime: {regime}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Indicators: {len(indicators)}")
        return True
        
    except Exception as e:
        print(f"❌ AI analysis failed: {e}")
        return False

def test_signal_generation():
    """Test signal generation."""
    print("\n🧪 Testing Signal Generation...")
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIDataEngine, S3AITradingStrategy
        
        engine = S3AIDataEngine()
        strategy = S3AITradingStrategy(engine)
        
        # Generate signal for AAPL
        action, strength, signal_info = strategy.generate_signal("AAPL", 225.0)
        
        print(f"✅ Signal generated:")
        print(f"   Action: {action}")
        print(f"   Strength: {strength:.2f}")
        print(f"   Regime: {signal_info.get('regime', 'Unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False

def test_portfolio():
    """Test portfolio management."""
    print("\n🧪 Testing Portfolio Management...")
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIPortfolioManager
        
        portfolio = S3AIPortfolioManager(initial_capital=100000, max_positions=5)
        
        # Test position management
        open_success = portfolio.open_position("AAPL", "BUY", 225.0, 0.8, "High Volatility")
        
        if open_success:
            summary = portfolio.get_portfolio_summary()
            print(f"✅ Position opened successfully")
            print(f"   Portfolio value: ${summary['total_value']:,.2f}")
            print(f"   Positions: {summary['positions_count']}")
            
            # Close position
            close_success = portfolio.close_position("AAPL", 228.0, "Test")
            if close_success:
                final_summary = portfolio.get_portfolio_summary()
                print(f"✅ Position closed successfully")
                print(f"   Final value: ${final_summary['total_value']:,.2f}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Portfolio test failed: {e}")
        return False

def main():
    """Run all quick tests."""
    print("🚀 S3 AI Hybrid System - Quick Test Suite")
    print("=" * 50)
    
    tests = [
        ("Data Engine", test_data_engine),
        ("AI Analysis", test_ai_analysis),
        ("Signal Generation", test_signal_generation),
        ("Portfolio Management", test_portfolio)
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
    print("\n" + "="*50)
    print("🧪 QUICK TEST RESULTS")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-"*50)
    print(f"TOTAL: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! S3 AI system is ready!")
        return True
    else:
        print("⚠️ Some tests failed. Please review.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)