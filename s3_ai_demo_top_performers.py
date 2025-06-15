#!/usr/bin/env python3
"""
S3 AI Top Performers Demo
=========================
Demo of S3 AI system using the top performing stocks from backtesting.
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
    """Run S3 AI demo with top performers."""
    print("🏆 S3 AI Hybrid Paper Trading System - TOP PERFORMERS DEMO")
    print("=" * 70)
    print("📊 Using data-driven symbol selection from comprehensive backtesting")
    print("🎯 48 symbols with >50% returns, >35% win rates, >1000 trades")
    print("⏰ Demo will run for 3 scan cycles then stop")
    print("📋 Press Ctrl+C to stop early")
    print()
    
    try:
        from s3_ai_hybrid_paper_trading import S3AIHybridPaperTrader
        
        # Initialize with top performers config
        print("🔧 Initializing S3 AI system with TOP PERFORMERS...")
        trader = S3AIHybridPaperTrader("s3_ai_trading_config_top_performers.json")
        
        print("✅ System initialized with top performing symbols!")
        print(f"📈 Expected performance: 15,000%+ average returns")
        print(f"🎯 Expected win rate: 45%+ average")
        print()
        
        # Show initial portfolio status
        trader.print_portfolio_status()
        
        # Check market status
        is_open = trader.is_market_open()
        print(f"\n📅 Market Status: {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
        print()
        
        # Show top symbols being monitored
        top_symbols = trader.symbols[:10]  # First 10 symbols
        print("🏆 TOP 10 SYMBOLS BEING MONITORED:")
        symbol_performance = {
            'A': '967,770% return, 36.9% win rate',
            'AAOI': '319,060% return, 39.6% win rate', 
            'ABNB': '33,400% return, 36.9% win rate',
            'ACAD': '194,448% return, 38.8% win rate',
            'ACGL': '8,104% return, 49.2% win rate',
            'ACVA': '903,480% return, 35.5% win rate',
            'ADMA': '9,450% return, 37.3% win rate',
            'ADP': '8,438% return, 41.2% win rate',
            'ADSK': '8,647% return, 40.6% win rate',
            'AEE': '942,580% return, 35.3% win rate'
        }
        
        for symbol in top_symbols:
            perf = symbol_performance.get(symbol, 'High performer')
            print(f"   🏅 {symbol}: {perf}")
        print()
        
        # Run 3 demo scan cycles with top performers
        for cycle in range(1, 4):
            print(f"🔍 TOP PERFORMERS SCAN CYCLE #{cycle}")
            print("-" * 50)
            
            # Get current prices for first 5 symbols (faster demo)
            demo_symbols = trader.symbols[:5]
            print(f"📊 Fetching prices for top 5 performers: {', '.join(demo_symbols)}")
            
            prices = trader.get_current_prices(demo_symbols)
            
            if prices:
                print(f"✅ Retrieved prices for {len(prices)} top performing symbols:")
                for symbol, price in prices.items():
                    perf = symbol_performance.get(symbol, 'High performer')
                    print(f"   💎 {symbol}: ${price:.2f} | Historical: {perf}")
                
                print("\n🧠 Generating AI signals for top performers...")
                
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
                
                print("📈 AI SIGNALS FOR TOP PERFORMERS:")
                for symbol, signal in signals.items():
                    status_icon = '🟢' if signal['action'] == 'BUY' else ('🔴' if signal['action'] == 'SELL' else '⚪')
                    print(f"   {status_icon} {symbol}: {signal['action']} | "
                          f"Strength: {signal['strength']:.2f} | "
                          f"Regime: {signal['regime']} | "
                          f"Confidence: {signal['confidence']:.2f}")
                
                # Simulate trading execution with top performers
                print("\n🎯 Executing trades with top performers...")
                strong_signals = [s for s in signals.values() if s['strength'] > 0.6 and s['action'] != 'HOLD']
                
                if strong_signals:
                    print(f"   🏆 Found {len(strong_signals)} strong signals from top performers!")
                    for symbol, signal in signals.items():
                        if signal['strength'] > 0.6 and signal['action'] != 'HOLD':
                            price = prices[symbol]
                            if signal['action'] == 'BUY' and symbol not in trader.portfolio.positions:
                                success = trader.portfolio.open_position(
                                    symbol, signal['action'], price, signal['strength'], signal['regime']
                                )
                                if success:
                                    print(f"   ✅ Opened {signal['action']} position in top performer {symbol}")
                            elif signal['action'] == 'SELL' and symbol in trader.portfolio.positions:
                                success = trader.portfolio.close_position(symbol, price, "Top Performer Exit")
                                if success:
                                    print(f"   ✅ Closed position in top performer {symbol}")
                else:
                    print("   📭 No strong signals from top performers this cycle")
                
                # Update portfolio with top performer positions
                trader.portfolio.update_positions(prices)
                
                # Show portfolio status
                print("\n💼 TOP PERFORMERS PORTFOLIO UPDATE:")
                summary = trader.portfolio.get_portfolio_summary()
                print(f"   Portfolio Value: ${summary['total_value']:,.2f}")
                print(f"   Active Positions: {summary['positions_count']}")
                print(f"   Total Trades: {summary['total_trades']}")
                if summary['total_trades'] > 0:
                    print(f"   Win Rate: {summary['win_rate']:.1f}%")
                    print(f"   Return: {summary['total_return_pct']:.2f}%")
                
            else:
                print("❌ No market data available for top performers")
            
            if cycle < 3:
                print(f"\n💤 Waiting 10 seconds before next cycle...")
                time.sleep(10)
            
            print()
        
        # Final status
        print("🏁 TOP PERFORMERS DEMO COMPLETED")
        print("=" * 70)
        trader.print_portfolio_status()
        
        # Save demo results
        trader.save_results()
        
        print("\n🎉 Top Performers S3 AI Demo completed successfully!")
        print("💎 Symbols tested represent the highest performing from 662 symbol backtest")
        print("📊 Each symbol showed >50% returns and >35% win rates historically")
        print("🚀 Use: python run_s3_ai_trading.py --start --config s3_ai_trading_config_top_performers.json")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()