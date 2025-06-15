#!/usr/bin/env python3
"""
S3 AI Trading System Demo
=========================
Interactive demo showing the S3 AI system in action.
"""

import time
import signal
import sys
from datetime import datetime

def signal_handler(sig, frame):
    print('\n🛑 Demo stopped by user')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    """Run S3 AI demo."""
    print("🤖 S3 AI Hybrid Paper Trading System - LIVE DEMO")
    print("=" * 60)
    print("⏰ Demo will run for 3 scan cycles then stop")
    print("📋 Press Ctrl+C to stop early")
    print()
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIHybridPaperTrader
        
        # Initialize with quick config
        print("🔧 Initializing S3 AI system...")
        trader = S3AIHybridPaperTrader("s3_ai_quick_config.json")
        
        print("✅ System initialized successfully!")
        print()
        
        # Show initial portfolio status
        trader.print_portfolio_status()
        
        # Check market status
        is_open = trader.is_market_open()
        print(f"\n📅 Market Status: {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
        print()
        
        # Run 3 demo scan cycles
        for cycle in range(1, 4):
            print(f"🔍 DEMO SCAN CYCLE #{cycle}")
            print("-" * 40)
            
            # Get current prices
            print("📊 Fetching current market prices...")
            prices = trader.get_current_prices()
            
            if prices:
                print(f"✅ Retrieved prices for {len(prices)} symbols:")
                for symbol, price in prices.items():
                    print(f"   {symbol}: ${price:.2f}")
                
                print("\n🧠 Generating AI signals...")
                
                # Generate signals for each symbol
                signals = {}
                for symbol, price in prices.items():
                    action, strength, signal_info = trader.strategy.generate_signal(symbol, price)
                    signals[symbol] = {
                        'action': action,
                        'strength': strength,
                        'regime': signal_info.get('regime', 'Unknown'),
                        'confidence': signal_info.get('confidence', 0.0)
                    }
                
                print("📈 AI TRADING SIGNALS:")
                for symbol, signal in signals.items():
                    status_icon = '🟢' if signal['action'] == 'BUY' else ('🔴' if signal['action'] == 'SELL' else '⚪')
                    print(f"   {status_icon} {symbol}: {signal['action']} | "
                          f"Strength: {signal['strength']:.2f} | "
                          f"Regime: {signal['regime']} | "
                          f"Confidence: {signal['confidence']:.2f}")
                
                # Simulate trading execution
                print("\n🎯 Executing trades...")
                strong_signals = [s for s in signals.values() if s['strength'] > 0.6 and s['action'] != 'HOLD']
                
                if strong_signals:
                    print(f"   Found {len(strong_signals)} strong signals for execution")
                    for symbol, signal in signals.items():
                        if signal['strength'] > 0.6 and signal['action'] != 'HOLD':
                            price = prices[symbol]
                            if signal['action'] == 'BUY' and symbol not in trader.portfolio.positions:
                                success = trader.portfolio.open_position(
                                    symbol, signal['action'], price, signal['strength'], signal['regime']
                                )
                                if success:
                                    print(f"   ✅ Opened {signal['action']} position in {symbol}")
                            elif signal['action'] == 'SELL' and symbol in trader.portfolio.positions:
                                success = trader.portfolio.close_position(symbol, price, "Demo Exit")
                                if success:
                                    print(f"   ✅ Closed position in {symbol}")
                else:
                    print("   📭 No strong signals for execution")
                
                # Update portfolio
                trader.portfolio.update_positions(prices)
                
                # Show portfolio status
                print("\n💼 PORTFOLIO UPDATE:")
                summary = trader.portfolio.get_portfolio_summary()
                print(f"   Portfolio Value: ${summary['total_value']:,.2f}")
                print(f"   Active Positions: {summary['positions_count']}")
                print(f"   Total Trades: {summary['total_trades']}")
                if summary['total_trades'] > 0:
                    print(f"   Win Rate: {summary['win_rate']:.1f}%")
                
            else:
                print("❌ No market data available")
            
            if cycle < 3:
                print(f"\n💤 Waiting 10 seconds before next cycle...")
                time.sleep(10)
            
            print()
        
        # Final status
        print("🏁 DEMO COMPLETED")
        print("=" * 60)
        trader.print_portfolio_status()
        
        # Save demo results
        trader.save_results()
        
        print("\n🎉 S3 AI Demo completed successfully!")
        print("💡 Use: python run_s3_ai_trading.py --start --config s3_ai_quick_config.json")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()