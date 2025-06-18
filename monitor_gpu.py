#!/usr/bin/env python3
"""
Monitor GPU Usage During Training
=================================
Shows real-time GPU utilization.
"""

import subprocess
import time
import os

def monitor_gpu():
    """Monitor GPU usage."""
    print("🔥 GPU Monitor - Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        while True:
            # Clear screen
            os.system('clear')
            
            print("🔥 MLX GPU Usage Monitor")
            print("=" * 60)
            print(f"Time: {time.strftime('%H:%M:%S')}")
            print()
            
            # Try different commands based on system
            try:
                # macOS with powermetrics (requires sudo)
                result = subprocess.run(
                    ['sudo', 'powermetrics', '--samplers', 'gpu', '-i', '1000', '-n', '1'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    output = result.stdout
                    # Extract GPU usage
                    for line in output.split('\n'):
                        if 'GPU' in line and '%' in line:
                            print(line.strip())
                else:
                    print("Note: Run with sudo for detailed GPU stats")
                    
            except:
                # Fallback - just show that training is running
                try:
                    # Check if MLX training is running
                    result = subprocess.run(
                        ['pgrep', '-f', 'train_mlx'],
                        capture_output=True
                    )
                    if result.returncode == 0:
                        print("✅ MLX Training is running")
                        print("   (Install mactop for detailed GPU monitoring)")
                    else:
                        print("❌ No MLX training detected")
                except:
                    pass
            
            print("\nTips for better GPU usage:")
            print("• Use larger batch sizes (128+)")
            print("• Process multiple symbols in parallel")
            print("• Keep data on GPU between epochs")
            print("• Use GPU-optimized version: python train_mlx.py")
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_gpu()