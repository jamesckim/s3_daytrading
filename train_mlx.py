#!/usr/bin/env python3
"""
MLX Training - Single Entry Point
=================================
Always uses the latest improvements automatically.
"""

import subprocess
import sys
import os
import json
import argparse

# This is the latest and best implementation
LATEST_TRAINER = "train_mlx_gpu_optimized.py"  # GPU-optimized version
# LATEST_TRAINER = "train_mlx_gpu_ultrafast.py"  # Ultra-fast Apple Silicon version (experimental)
# LATEST_TRAINER = "train_mlx_practical.py"  # Practical balanced version
CONFIG_FILE = "mlx_config.json"

def load_defaults():
    """Load default configuration."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get('defaults', {})
    return {}

def main():
    """Smart MLX training with defaults."""
    
    # Load defaults
    defaults = load_defaults()
    
    # Create parser just to show help
    parser = argparse.ArgumentParser(
        description='Train MLX model (always uses latest improvements)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with your preferred settings
  python train_mlx.py
  
  # Custom settings
  python train_mlx.py --days 30 --epochs 20
  
  # Use all tickers
  python train_mlx.py --tickers-file ../tickers/all_tickers.json
  
  # Test with fewer epochs
  python train_mlx.py --epochs 5 --sample-interval 30

Current defaults (from mlx_config.json):
""" + json.dumps(defaults, indent=2)
    )
    
    # Add arguments (just for help display)
    parser.add_argument('--tickers-file', help='Ticker list JSON file')
    parser.add_argument('--days', type=int, help='Days of minute data')
    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--sample-interval', type=int, help='Minutes between samples')
    parser.add_argument('--min-samples', type=int, help='Minimum samples required')
    
    # Parse args
    args, unknown = parser.parse_known_args()
    
    # Build command with defaults
    cmd = [sys.executable, LATEST_TRAINER]
    
    # Add defaults first
    for key, value in defaults.items():
        arg_name = '--' + key.replace('_', '-')
        # Only add if not already specified by user
        if arg_name not in sys.argv:
            cmd.extend([arg_name, str(value)])
    
    # Add all user arguments (these override defaults)
    cmd.extend(sys.argv[1:])
    
    # Show what we're running
    print(f"🚀 MLX Training")
    print(f"━" * 60)
    print(f"Using: {LATEST_TRAINER}")
    print(f"Config: {CONFIG_FILE}")
    
    # Show effective settings
    print(f"\nEffective settings:")
    effective_args = {}
    i = 1
    while i < len(cmd):
        if cmd[i].startswith('--'):
            key = cmd[i][2:].replace('-', '_')
            if i + 1 < len(cmd) and not cmd[i + 1].startswith('--'):
                effective_args[key] = cmd[i + 1]
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    for key, value in effective_args.items():
        print(f"  {key}: {value}")
    
    print(f"━" * 60 + "\n")
    
    # Run it
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()