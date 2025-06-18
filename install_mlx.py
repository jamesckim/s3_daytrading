#!/usr/bin/env python3
"""
Install MLX for Apple Silicon
==============================
Installs Apple's MLX framework for machine learning.
"""

import subprocess
import sys
import platform

print("🍎 MLX Installation for Apple Silicon")
print("="*50)

# Check if running on macOS
if platform.system() != "Darwin":
    print("❌ MLX is only available for macOS")
    sys.exit(1)

# Check if Apple Silicon
if platform.processor() != 'arm':
    print("⚠️  Warning: MLX is optimized for Apple Silicon (M1/M2/M3)")
    print("   It may work on Intel Macs but with limited performance")

print("\nMLX is Apple's machine learning framework that offers:")
print("  ✓ Unified memory (no CPU-GPU transfers)")
print("  ✓ Lazy evaluation")
print("  ✓ Automatic differentiation")
print("  ✓ Native Metal acceleration")

response = input("\nInstall MLX? (y/n): ")

if response.lower() == 'y':
    print("\nInstalling MLX...")
    
    # Use uv if available, otherwise pip
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        cmd = ["uv", "pip", "install", "mlx"]
        print("Using uv to install...")
    except:
        cmd = [sys.executable, "-m", "pip", "install", "mlx"]
        print("Using pip to install...")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ MLX installed successfully!")
        
        # Test installation
        print("\nTesting MLX installation...")
        test_cmd = [sys.executable, "-c", 
                   "import mlx; import mlx.core as mx; "
                   "print('MLX imported successfully'); "
                   "print(f'Simple computation: {mx.array([1, 2, 3]).sum()}')"]
        subprocess.run(test_cmd, check=True)
        
        print("\n✅ MLX is working correctly!")
        print("\nYou can now run the benchmark:")
        print("  python benchmark_pytorch_vs_mlx.py")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("\nTry installing manually:")
        print("  pip install mlx")
else:
    print("\nSkipping MLX installation.")
    print("You can install it later with: pip install mlx")