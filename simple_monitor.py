#!/usr/bin/env python3
"""
Simple Backtest Monitor
=======================
Quick status check for parallel backtests.
"""

import json
import glob
import os
from datetime import datetime


def main():
    """Simple monitoring display."""
    print("📊 Quick Backtest Status")
    print("=" * 50)
    
    # Check for running processes
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 's3_ai_parallel_backtest'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("🏃 Status: RUNNING")
            print(f"   PIDs: {result.stdout.strip()}")
        else:
            print("📋 Status: NOT RUNNING")
    except:
        print("❓ Status: UNKNOWN")
    
    # Check result files
    result_files = glob.glob('parallel_backtest_results_*.json')
    result_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"\n📁 Result Files: {len(result_files)} found")
    
    if result_files:
        latest = result_files[0]
        mod_time = datetime.fromtimestamp(os.path.getmtime(latest))
        print(f"   Latest: {latest}")
        print(f"   Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Quick check of latest results
        try:
            with open(latest) as f:
                data = json.load(f)
            
            successful = sum(1 for v in data.values() 
                           if isinstance(v, dict) and 'total_return' in v and 'error' not in v)
            
            print(f"   Symbols: {len(data)} tested, {successful} successful")
            
            if successful > 0:
                print("   ✅ Enhanced VWAP system is working!")
            else:
                print("   ⚠️  Check for errors in backtest")
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"\n🔧 Monitor Options:")
    print(f"   Full analysis: python monitor_backtest_results.py")
    print(f"   Run new test:  python quick_vwap_test.py") 
    print(f"   Live monitor:  watch -n 5 python simple_monitor.py")


if __name__ == "__main__":
    main()