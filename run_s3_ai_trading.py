#!/usr/bin/env python3
"""
S3 AI Trading System Launcher
=============================
Main launcher script for the S3 AI Hybrid Paper Trading System.
Provides easy startup with various modes and options.
"""

import sys
import os
import argparse
import subprocess
import time
from datetime import datetime


def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking system dependencies...")
    
    required_modules = [
        'pandas', 'numpy', 'sqlalchemy', 'dotenv',
        'psycopg2', 'warnings', 'datetime', 'threading',
        'queue', 'dataclasses', 'json', 'logging'
    ]
    
    missing = []
    for module in required_modules:
        try:
            if module == 'psycopg2':
                __import__('psycopg2')
            else:
                __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("💡 Install with: uv add " + " ".join(missing))
        return False
    
    print("✅ All dependencies available")
    return True

def check_environment():
    """Check if required environment variables are set."""
    print("🔍 Checking environment configuration...")
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['POSTGRES_URI']
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("💡 Make sure .env file contains all required variables")
        return False
    
    print("✅ Environment configuration OK")
    return True

def check_ai_components():
    """Check if AI components are available."""
    print("🔍 Checking AI components...")
    
    try:
        from s3_ai_wrapper import S3AIWrapper
        print("✅ S3 AI Wrapper available")
        return True
    except ImportError as e:
        print(f"❌ S3 AI Wrapper not found: {e}")
        print("💡 Make sure s3_ai_wrapper.py is in the current directory")
        return False

def run_tests():
    """Run the test suite."""
    print("🧪 Running S3 AI system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_s3_ai_hybrid.py"], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    except FileNotFoundError:
        print("❌ Test script not found: test_s3_ai_hybrid.py")
        return False

def start_trading_system(config_file: str = None):
    """Start the main trading system."""
    print("🚀 Starting S3 AI Hybrid Paper Trading System...")
    import sys
    sys.stdout.flush()
    
    try:
        # Import and initialize
        from s3_ai_hybrid_paper_trading import S3AIHybridPaperTrader
        
        print("🔧 Initializing trading components...")
        sys.stdout.flush()
        
        if config_file:
            trader = S3AIHybridPaperTrader(config_file)
        else:
            trader = S3AIHybridPaperTrader()
        
        print("✅ System initialized successfully")
        sys.stdout.flush()
        
        # Show initial status
        trader.print_portfolio_status()
        
        # Start trading
        print("\n🎯 Starting trading operations...")
        print("📋 Press Ctrl+C to stop trading")
        sys.stdout.flush()
        
        trader.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Trading system stopped by user")
    except Exception as e:
        print(f"❌ Trading system error: {e}")
        import traceback
        traceback.print_exc()

def start_monitor(live: bool = False, interval: int = 30):
    """Start the monitoring system."""
    if live:
        print(f"📊 Starting live monitor (refresh every {interval}s)...")
        try:
            subprocess.run([sys.executable, "s3_ai_monitor.py", "--live", "--interval", str(interval)])
        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
    else:
        print("📊 Showing current status...")
        subprocess.run([sys.executable, "s3_ai_monitor.py"])

def generate_report():
    """Generate a trading report."""
    print("📄 Generating trading report...")
    subprocess.run([sys.executable, "s3_ai_monitor.py", "--report"])

def show_status():
    """Show current system status."""
    print("📊 S3 AI Trading System Status")
    print("=" * 40)
    
    # Check if trading system is running
    try:
        import glob
        results_files = glob.glob("s3_ai_trading_results_*.json")
        if results_files:
            latest_file = max(results_files, key=os.path.getmtime)
            mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
            print(f"📁 Latest results: {latest_file}")
            print(f"⏰ Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show quick summary
            subprocess.run([sys.executable, "s3_ai_monitor.py"])
        else:
            print("📭 No results files found")
            print("💡 System may not be running or hasn't started yet")
    except Exception as e:
        print(f"❌ Error checking status: {e}")

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="S3 AI Hybrid Paper Trading System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_s3_ai_trading.py --start          # Start trading system
  python run_s3_ai_trading.py --test           # Run test suite
  python run_s3_ai_trading.py --monitor        # Show current status
  python run_s3_ai_trading.py --monitor --live # Start live monitor
  python run_s3_ai_trading.py --report         # Generate report
  python run_s3_ai_trading.py --status         # Show system status
        """
    )
    
    # Main actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--start", action="store_true", 
                             help="Start the trading system")
    action_group.add_argument("--test", action="store_true", 
                             help="Run system tests")
    action_group.add_argument("--monitor", action="store_true", 
                             help="Start monitoring dashboard")
    action_group.add_argument("--report", action="store_true", 
                             help="Generate trading report")
    action_group.add_argument("--status", action="store_true", 
                             help="Show current system status")
    
    # Options
    parser.add_argument("--config", type=str, 
                       help="Configuration file path")
    parser.add_argument("--live", action="store_true", 
                       help="Live monitoring mode (use with --monitor)")
    parser.add_argument("--interval", type=int, default=30, 
                       help="Monitor refresh interval in seconds")
    parser.add_argument("--no-checks", action="store_true", 
                       help="Skip dependency and environment checks")
    
    args = parser.parse_args()
    
    # Print header
    print("🤖 S3 AI Hybrid Paper Trading System")
    print("=" * 50)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run pre-checks (unless skipped)
    if not args.no_checks:
        checks_passed = True
        
        if not check_dependencies():
            checks_passed = False
        
        if not check_environment():
            checks_passed = False
        
        if not check_ai_components():
            checks_passed = False
        
        if not checks_passed:
            print("\n❌ Pre-checks failed. Please fix issues before proceeding.")
            print("💡 Use --no-checks to skip these checks")
            sys.exit(1)
        
        print("✅ All pre-checks passed\n")
    
    # Execute requested action
    try:
        if args.test:
            success = run_tests()
            sys.exit(0 if success else 1)
        
        elif args.start:
            start_trading_system(args.config)
        
        elif args.monitor:
            start_monitor(live=args.live, interval=args.interval)
        
        elif args.report:
            generate_report()
        
        elif args.status:
            show_status()
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()