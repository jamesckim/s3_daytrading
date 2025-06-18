#!/usr/bin/env python3
"""
Upgrade to PyTorch Nightly for Apple Silicon
=============================================
Installs the latest PyTorch nightly build with improved MPS support.
Based on recommendations from pytorch_m3_o3.md
"""

import subprocess
import sys
import torch

print("🔧 PyTorch Nightly Upgrade for Maximum MPS Performance")
print("=" * 60)

# Check current PyTorch version
print(f"\nCurrent PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

print("\nThe PyTorch nightly build includes:")
print("  ✓ 2-3x faster kernels for GEMM/conv operations")
print("  ✓ Fewer CPU fallbacks")
print("  ✓ BFloat16 autocast support")
print("  ✓ Metal kernel fusion improvements")
print("  ✓ Bug fixes for MPS backend")

response = input("\nUpgrade to PyTorch nightly? (y/n): ")

if response.lower() == 'y':
    print("\nUpgrading PyTorch to nightly build...")
    
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--pre", "torch", "torchvision", "torchaudio",
        "--extra-index-url", "https://download.pytorch.org/whl/nightly/cpu",
        "--upgrade"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ PyTorch nightly installed successfully!")
        
        # Verify installation
        # Need to restart Python to load new version
        print("\nPlease restart Python/Jupyter to use the new version.")
        print("Then verify with: python -c 'import torch; print(torch.__version__)'")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        print("\nTry running manually:")
        print("uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu")
else:
    print("\nSkipping upgrade. Current version will be used.")