#!/usr/bin/env python3
"""
S3 AI Trading Monitor
====================
Real-time monitoring dashboard for S3 AI Hybrid Paper Trading system.
Provides live performance metrics, position tracking, and AI analysis.
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


class S3AITradingMonitor:
    """Real-time monitoring for S3 AI trading system."""
    
    def __init__(self, results_pattern: str = "s3_ai_trading_results_*.json"):
        self.results_pattern = results_pattern
        self.last_update = None
        
    def find_latest_results(self) -> str:
        """Find the most recent results file."""
        import glob
        
        files = glob.glob(self.results_pattern)
        if not files:
            return None
        
        # Sort by modification time, return newest
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load trading results from file."""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return {}
    
    def display_performance_summary(self, results: Dict[str, Any]):
        """Display performance summary."""
        summary = results.get("session_summary", {})
        
        print("\n" + "="*70)
        print("🤖 S3 AI HYBRID PAPER TRADING - LIVE PERFORMANCE")
        print("="*70)
        
        # Portfolio metrics
        total_value = summary.get("total_value", 0)
        initial_capital = summary.get("initial_capital", 1000000)
        return_pct = summary.get("total_return_pct", 0)
        unrealized_pnl = summary.get("unrealized_pnl", 0)
        
        print(f"💰 Portfolio Value: ${total_value:,.2f}")
        print(f"📈 Total Return: {return_pct:+.2f}%")
        print(f"💵 Available Cash: ${summary.get('current_capital', 0):,.2f}")
        print(f"📊 Unrealized P&L: ${unrealized_pnl:+,.2f}")
        
        # Trading metrics
        positions_count = summary.get("positions_count", 0)
        total_trades = summary.get("total_trades", 0)
        win_rate = summary.get("win_rate", 0)
        
        print(f"📋 Active Positions: {positions_count}")
        print(f"🔄 Total Trades: {total_trades}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        
    def display_active_positions(self, results: Dict[str, Any]):
        """Display active positions."""
        positions = results.get("positions", [])
        
        if not positions:
            print("\n📭 No active positions")
            return
        
        print(f"\n📋 ACTIVE POSITIONS ({len(positions)}):")
        print("-" * 70)
        print(f"{'Symbol':<8} {'Side':<6} {'Qty':<8} {'Entry':<10} {'Current':<10} {'P&L':<12} {'Regime':<15}")
        print("-" * 70)
        
        total_unrealized = 0
        for pos in positions:
            symbol = pos.get("symbol", "")
            quantity = pos.get("quantity", 0)
            side = "LONG" if quantity > 0 else "SHORT"
            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", 0)
            unrealized_pnl = pos.get("unrealized_pnl", 0)
            regime = pos.get("regime", "Unknown")
            
            total_unrealized += unrealized_pnl
            
            print(f"{symbol:<8} {side:<6} {abs(quantity):<8.0f} ${entry_price:<9.2f} "
                  f"${current_price:<9.2f} ${unrealized_pnl:<11.2f} {regime:<15}")
        
        print("-" * 70)
        print(f"{'TOTAL':<40} ${total_unrealized:+.2f}")
    
    def display_recent_trades(self, results: Dict[str, Any], limit: int = 10):
        """Display recent trades."""
        trades = results.get("trades", [])
        
        if not trades:
            print("\n📭 No trades executed")
            return
        
        # Sort by timestamp (most recent first)
        trades = sorted(trades, key=lambda x: x.get("timestamp", ""), reverse=True)
        recent_trades = trades[:limit]
        
        print(f"\n🔄 RECENT TRADES (Last {len(recent_trades)}):")
        print("-" * 80)
        print(f"{'Time':<8} {'Symbol':<8} {'Side':<6} {'Qty':<8} {'Price':<10} {'P&L':<12} {'Regime':<15}")
        print("-" * 80)
        
        for trade in recent_trades:
            timestamp = trade.get("timestamp", "")
            if timestamp:
                time_str = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%H:%M")
            else:
                time_str = "--:--"
            
            symbol = trade.get("symbol", "")
            side = trade.get("side", "")
            quantity = trade.get("quantity", 0)
            price = trade.get("price", 0)
            pnl = trade.get("pnl", 0)
            regime = trade.get("regime", "Unknown")
            
            print(f"{time_str:<8} {symbol:<8} {side:<6} {quantity:<8.0f} ${price:<9.2f} "
                  f"${pnl:<11.2f} {regime:<15}")
    
    def display_ai_analysis_summary(self, results: Dict[str, Any]):
        """Display AI analysis summary."""
        trades = results.get("trades", [])
        
        if not trades:
            return
        
        # Analyze regime distribution
        regimes = {}
        total_pnl_by_regime = {}
        
        for trade in trades:
            regime = trade.get("regime", "Unknown")
            pnl = trade.get("pnl", 0)
            
            if regime not in regimes:
                regimes[regime] = 0
                total_pnl_by_regime[regime] = 0
            
            regimes[regime] += 1
            total_pnl_by_regime[regime] += pnl
        
        print(f"\n🧠 AI REGIME ANALYSIS:")
        print("-" * 50)
        print(f"{'Regime':<20} {'Trades':<8} {'Total P&L':<15}")
        print("-" * 50)
        
        for regime, count in regimes.items():
            pnl = total_pnl_by_regime[regime]
            print(f"{regime:<20} {count:<8} ${pnl:<14.2f}")
    
    def display_performance_metrics(self, results: Dict[str, Any]):
        """Display detailed performance metrics."""
        trades = results.get("trades", [])
        
        if not trades:
            return
        
        # Calculate metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get("pnl", 0) > 0])
        losing_trades = total_trades - winning_trades
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            
            pnls = [t.get("pnl", 0) for t in trades]
            total_pnl = sum(pnls)
            avg_win = sum([p for p in pnls if p > 0]) / max(1, winning_trades)
            avg_loss = sum([p for p in pnls if p < 0]) / max(1, losing_trades)
            
            # Profit factor
            gross_profit = sum([p for p in pnls if p > 0])
            gross_loss = abs(sum([p for p in pnls if p < 0]))
            profit_factor = gross_profit / max(1, gross_loss)
            
            print(f"\n📊 DETAILED PERFORMANCE METRICS:")
            print("-" * 40)
            print(f"Total Trades: {total_trades}")
            print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
            print(f"Losing Trades: {losing_trades}")
            print(f"Average Win: ${avg_win:.2f}")
            print(f"Average Loss: ${avg_loss:.2f}")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Total P&L: ${total_pnl:.2f}")
    
    def monitor_live(self, refresh_interval: int = 30):
        """Start live monitoring with auto-refresh."""
        
        print("🤖 Starting S3 AI Trading Live Monitor")
        print(f"🔄 Auto-refresh every {refresh_interval} seconds")
        print("📋 Press Ctrl+C to stop")
        
        try:
            while True:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Find and load latest results
                latest_file = self.find_latest_results()
                if latest_file:
                    results = self.load_results(latest_file)
                    
                    if results:
                        print(f"📁 Loading: {latest_file}")
                        print(f"⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Display all sections
                        self.display_performance_summary(results)
                        self.display_active_positions(results)
                        self.display_recent_trades(results, limit=8)
                        self.display_ai_analysis_summary(results)
                        self.display_performance_metrics(results)
                        
                        print(f"\n🔄 Next refresh in {refresh_interval} seconds...")
                    else:
                        print("❌ No valid results data found")
                else:
                    print("❌ No results files found")
                    print(f"   Looking for: {self.results_pattern}")
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
    
    def generate_report(self, output_file: str = None):
        """Generate a comprehensive trading report."""
        
        latest_file = self.find_latest_results()
        if not latest_file:
            print("❌ No results files found for report generation")
            return
        
        results = self.load_results(latest_file)
        if not results:
            print("❌ Could not load results data")
            return
        
        # Generate report content
        report_lines = []
        report_lines.append("S3 AI HYBRID PAPER TRADING - COMPREHENSIVE REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Data source: {latest_file}")
        report_lines.append("")
        
        # Session info
        session_start = results.get("session_start", "")
        session_end = results.get("session_end", "")
        report_lines.append(f"Session: {session_start} to {session_end}")
        report_lines.append("")
        
        # Performance summary
        summary = results.get("session_summary", {})
        report_lines.append("PERFORMANCE SUMMARY:")
        report_lines.append("-" * 30)
        report_lines.append(f"Initial Capital: ${summary.get('initial_capital', 0):,.2f}")
        report_lines.append(f"Final Value: ${summary.get('total_value', 0):,.2f}")
        report_lines.append(f"Total Return: {summary.get('total_return_pct', 0):+.2f}%")
        report_lines.append(f"Unrealized P&L: ${summary.get('unrealized_pnl', 0):+,.2f}")
        report_lines.append(f"Active Positions: {summary.get('positions_count', 0)}")
        report_lines.append(f"Total Trades: {summary.get('total_trades', 0)}")
        report_lines.append(f"Win Rate: {summary.get('win_rate', 0):.1f}%")
        report_lines.append("")
        
        # All trades
        trades = results.get("trades", [])
        if trades:
            report_lines.append("ALL TRADES:")
            report_lines.append("-" * 80)
            report_lines.append(f"{'Time':<12} {'Symbol':<8} {'Side':<6} {'Qty':<8} {'Price':<10} {'P&L':<12} {'Regime':<15}")
            report_lines.append("-" * 80)
            
            for trade in trades:
                timestamp = trade.get("timestamp", "")
                if timestamp:
                    time_str = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%H:%M:%S")
                else:
                    time_str = "--:--:--"
                
                symbol = trade.get("symbol", "")
                side = trade.get("side", "")
                quantity = trade.get("quantity", 0)
                price = trade.get("price", 0)
                pnl = trade.get("pnl", 0)
                regime = trade.get("regime", "Unknown")
                
                report_lines.append(f"{time_str:<12} {symbol:<8} {side:<6} {quantity:<8.0f} "
                                  f"${price:<9.2f} ${pnl:<11.2f} {regime:<15}")
        
        # Save report
        if output_file is None:
            output_file = f"s3_ai_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(output_file, 'w') as f:
                f.write('\n'.join(report_lines))
            
            print(f"📄 Report saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to save report: {e}")


def main():
    """Main entry point for monitoring."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="S3 AI Trading Monitor")
    parser.add_argument("--live", action="store_true", help="Start live monitoring")
    parser.add_argument("--report", action="store_true", help="Generate trading report")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval for live monitoring (seconds)")
    parser.add_argument("--output", type=str, help="Output file for report")
    
    args = parser.parse_args()
    
    monitor = S3AITradingMonitor()
    
    if args.live:
        monitor.monitor_live(refresh_interval=args.interval)
    elif args.report:
        monitor.generate_report(output_file=args.output)
    else:
        # Show current status once
        latest_file = monitor.find_latest_results()
        if latest_file:
            results = monitor.load_results(latest_file)
            if results:
                monitor.display_performance_summary(results)
                monitor.display_active_positions(results)
                monitor.display_recent_trades(results)
                monitor.display_ai_analysis_summary(results)
                monitor.display_performance_metrics(results)
            else:
                print("❌ No valid results data found")
        else:
            print("❌ No results files found")
            print("💡 Make sure the S3 AI trading system is running or has been run")


if __name__ == "__main__":
    main()