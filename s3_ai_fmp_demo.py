#!/usr/bin/env python3
"""
S3 AI FMP Hybrid Demo
====================
Live demonstration of S3 AI trading with FMP real-time data.
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
    """Run S3 AI FMP hybrid demo."""
    print("🤖 S3 AI FMP Hybrid Trading System - LIVE DEMO")
    print("=" * 60)
    print("📡 Real-time data powered by Financial Modeling Prep")
    print("🔒 Simulated paper trading with realistic execution")
    print("⏰ Demo will run for 3 cycles then stop")
    print("📋 Press Ctrl+C to stop early")
    print()
    
    try:
        from s3_ai_fmp_hybrid_trading import S3AIFMPHybridTrader
        
        # Initialize with standard config
        print("🔧 Initializing S3 AI FMP Hybrid System...")
        trader = S3AIFMPHybridTrader("s3_ai_fmp_trading_config.json")
        
        print("✅ System initialized with FMP real-time data!")
        print(f"📊 Monitoring {len(trader.symbols)} symbols")
        print(f"💰 Initial capital: ${trader.initial_capital:,.0f}")
        print()
        
        # Show portfolio status
        trader.print_status()
        
        # Check FMP API status
        print("\n📡 Testing FMP API connection...")
        test_quotes = trader.fmp_provider.get_real_time_quotes(["AAPL", "MSFT"])
        
        if test_quotes:
            print(f"✅ FMP API working - received {len(test_quotes)} quotes")
            for symbol, quote in test_quotes.items():
                print(f"   📊 {symbol}: ${quote.price:.2f} (Vol: {quote.volume:,}, "
                      f"Change: {quote.change_percent:+.1f}%)")
        else:
            print("❌ FMP API connection issues")
        
        print("\n🔍 STARTING LIVE FMP TRADING DEMO")
        print("-" * 50)
        
        # Run 3 demo cycles
        for cycle in range(1, 4):
            print(f"\n🔄 FMP DEMO CYCLE #{cycle}")
            print("-" * 30)
            
            # Get real-time FMP data for first 5 symbols
            demo_symbols = trader.symbols[:5]
            print(f"📡 Fetching FMP real-time data for: {', '.join(demo_symbols)}")
            
            quotes = trader.fmp_provider.get_real_time_quotes(demo_symbols)
            
            if quotes:
                print(f"✅ Retrieved FMP quotes for {len(quotes)} symbols:")
                
                for symbol, quote in quotes.items():
                    # Generate S3 AI signal with FMP data
                    action, strength, signal_info = trader.strategy.generate_fmp_signal(symbol, quote)
                    
                    regime = signal_info.get("regime", "Unknown")
                    confidence = signal_info.get("confidence", 0.0)
                    
                    # Status icon
                    status_icon = '🟢' if action == 'BUY' else ('🔴' if action == 'SELL' else '⚪')
                    
                    print(f"   {status_icon} {symbol}: ${quote.price:.2f} | {action} ({strength:.2f}) | "
                          f"{regime} | Vol: {quote.volume:,} | Chg: {quote.change_percent:+.1f}%")
                
                print("\n🎯 Simulating trade execution...")
                
                # Check for strong signals
                strong_signals = []
                for symbol, quote in quotes.items():
                    action, strength, _ = trader.strategy.generate_fmp_signal(symbol, quote)
                    if strength >= trader.min_signal_strength and action != "HOLD":
                        strong_signals.append((symbol, action, strength, quote))
                
                if strong_signals:
                    print(f"   🏆 Found {len(strong_signals)} strong signals!")
                    for symbol, action, strength, quote in strong_signals:
                        # Simulate position management
                        if action == "BUY" and symbol not in trader.positions:
                            quantity = trader.calculate_position_size(symbol, quote.price, strength)
                            success, fill_price = trader.simulate_order_fill(symbol, "BUY", quantity, quote)
                            if success:
                                trader.open_position(symbol, "BUY", quantity, fill_price, "FMP Signal")
                                print(f"   ✅ Simulated BUY: {quantity} {symbol} @ ${fill_price:.2f}")
                        
                        elif action == "SELL" and symbol in trader.positions:
                            position = trader.positions[symbol]
                            quantity = int(abs(position.quantity))
                            success, fill_price = trader.simulate_order_fill(symbol, "SELL", quantity, quote)
                            if success:
                                trader.close_position(symbol, fill_price, "FMP Exit")
                                print(f"   ✅ Simulated SELL: {quantity} {symbol} @ ${fill_price:.2f}")
                else:
                    print("   📭 No strong signals for execution")
                
                # Update positions with current FMP data
                trader.update_positions_with_fmp_data(quotes)
                
                # Show updated portfolio
                print(f"\n💼 PORTFOLIO UPDATE:")
                total_value = trader.current_capital + sum(pos.unrealized_pnl for pos in trader.positions.values())
                return_pct = ((total_value / trader.initial_capital) - 1) * 100
                
                print(f"   Portfolio Value: ${total_value:,.2f}")
                print(f"   Return: {return_pct:+.2f}%")
                print(f"   Active Positions: {len(trader.positions)}")
                print(f"   Cash Available: ${trader.current_capital:,.2f}")
                
                if trader.positions:
                    print("   📋 Positions:")
                    for symbol, pos in trader.positions.items():
                        pnl_pct = (pos.unrealized_pnl / abs(pos.quantity * pos.entry_price)) * 100
                        side = "LONG" if pos.quantity > 0 else "SHORT"
                        print(f"      {symbol}: {side} {abs(pos.quantity):.0f} @ ${pos.entry_price:.2f} | "
                              f"P&L: ${pos.unrealized_pnl:+.2f} ({pnl_pct:+.1f}%)")
            
            else:
                print("❌ No FMP data received this cycle")
            
            if cycle < 3:
                print(f"\n💤 Waiting 15 seconds before next cycle...")
                time.sleep(15)
        
        # Final status
        print("\n🏁 FMP DEMO COMPLETED")
        print("=" * 60)
        trader.print_status()
        
        # Save demo results
        trader.save_results()
        
        print("\n🎉 S3 AI FMP Demo completed successfully!")
        print("📡 System successfully integrated with FMP real-time data")
        print("🧠 S3 AI strategy working with live market conditions")
        print("💼 Realistic position management and execution simulation")
        print("\n🚀 To run live system:")
        print("   python s3_ai_fmp_hybrid_trading.py")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()