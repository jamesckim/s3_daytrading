#!/usr/bin/env python3
"""
Monitor Training Progress
=========================
Real-time monitoring of training progress and cache performance.
"""

import time
import json
from pathlib import Path
from datetime import datetime
import os
from indicator_cache import IndicatorCache

def monitor_training():
    """Monitor training progress in real-time."""
    print("📊 Training Monitor")
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring\n")
    
    cache = IndicatorCache()
    
    # Look for latest checkpoint
    model_dir = Path("models")
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"📊 Training Monitor - {datetime.now().strftime('%I:%M:%S %p')}")
            print("=" * 80)
            
            # Cache statistics
            cache_stats = cache.get_stats()
            print("\n📦 Cache Performance:")
            print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
            print(f"  Total Hits: {cache_stats['hits']:,}")
            print(f"  Total Misses: {cache_stats['misses']:,}")
            print(f"  New Computations: {cache_stats['saves']:,}")
            print(f"  Cache Size: {cache_stats['total_size_mb']:.1f} MB")
            print(f"  Files: {cache_stats['num_files']:,}")
            
            # Check for model files
            print("\n📁 Model Files:")
            if model_dir.exists():
                model_files = list(model_dir.glob("*.pth"))
                if model_files:
                    for mf in sorted(model_files, key=lambda x: x.stat().st_mtime)[-3:]:
                        size_mb = mf.stat().st_size / 1024 / 1024
                        mod_time = datetime.fromtimestamp(mf.stat().st_mtime)
                        age = (datetime.now() - mod_time).total_seconds()
                        
                        if age < 300:  # Modified in last 5 minutes
                            status = "🟢 ACTIVE"
                        elif age < 3600:  # Modified in last hour
                            status = "🟡 RECENT"
                        else:
                            status = "⚪ OLD"
                            
                        print(f"  {status} {mf.name} ({size_mb:.1f}MB, {mod_time.strftime('%I:%M %p')})")
                else:
                    print("  No model files found yet")
            
            # Memory usage and device info
            try:
                import psutil
                import torch
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"\n💾 Memory Usage: {memory_mb:.0f} MB")
                
                # Show device being used
                if torch.backends.mps.is_available():
                    print("🎯 Device: Apple Silicon GPU (MPS)")
                elif torch.cuda.is_available():
                    print("🎯 Device: NVIDIA GPU (CUDA)")
                else:
                    print("🎯 Device: CPU")
            except:
                pass
            
            # Cache breakdown by symbol
            cache_info = cache.get_cache_info()
            if cache_info['entries_by_symbol']:
                print("\n📊 Cache by Symbol:")
                for symbol, count in sorted(cache_info['entries_by_symbol'].items(), 
                                          key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {symbol}: {count} entries")
            
            # Cache breakdown by indicator
            if cache_info['entries_by_indicator']:
                print("\n📈 Cache by Indicator Type:")
                indicator_groups = {}
                for indicator, count in cache_info['entries_by_indicator'].items():
                    # Group by indicator type
                    ind_type = indicator.split('_')[0]
                    if ind_type not in indicator_groups:
                        indicator_groups[ind_type] = 0
                    indicator_groups[ind_type] += count
                
                for ind_type, count in sorted(indicator_groups.items(), 
                                            key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {ind_type}: {count} computations")
            
            print("\n[Refreshing every 5 seconds...]")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        
        # Final summary
        final_stats = cache.get_stats()
        print("\n📊 Final Summary:")
        print(f"  Total cache operations: {final_stats['hits'] + final_stats['misses']:,}")
        print(f"  Cache efficiency: {final_stats['hit_rate']:.1%}")
        print(f"  Computational savings: ~{final_stats['hits'] * 0.1:.0f} seconds")


if __name__ == "__main__":
    monitor_training()