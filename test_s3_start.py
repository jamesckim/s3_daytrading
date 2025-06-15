#!/usr/bin/env python3
"""
Test S3 AI Start Functionality
===============================
Quick test to verify the start command works correctly.
"""

import time
import signal
import sys
from datetime import datetime

def signal_handler(sig, frame):
    print('\n🛑 Test interrupted by user')
    sys.exit(0)

# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)

def test_start():
    """Test the start functionality with timeout."""
    print("🧪 Testing S3 AI Start Command...")
    print("⏰ Starting system for 30 seconds...")
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIHybridPaperTrader
        
        # Initialize with quick config
        trader = S3AIHybridPaperTrader("s3_ai_quick_config.json")
        print("✅ System initialized successfully")
        
        # Show initial status
        trader.print_portfolio_status()
        
        # Check market status
        is_open = trader.is_market_open()
        print(f"\n📅 Market status: {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
        
        # Run one scan cycle
        print("\n🔍 Running one scan cycle...")
        trader.scan_and_trade()
        
        # Show final status
        print("\n📊 Final Status:")
        trader.print_portfolio_status()
        
        print("\n✅ Start test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Start test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 S3 AI Start Test")
    print("=" * 40)
    
    success = test_start()
    
    if success:
        print("\n🎉 Start functionality is working!")
        print("💡 Use: python run_s3_ai_trading.py --start --config s3_ai_quick_config.json")
    else:
        print("\n⚠️ Start test failed. Check errors above.")
    
    sys.exit(0 if success else 1)