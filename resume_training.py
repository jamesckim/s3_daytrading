#!/usr/bin/env python3
"""
Resume Training from Checkpoint
================================
Resume training from an interrupted checkpoint or continue from best model.
"""

import torch
import argparse
import os
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(model_dir="models"):
    """Find the latest checkpoint file."""
    model_path = Path(model_dir)
    
    # Look for interrupted checkpoints first
    interrupted_checkpoints = list(model_path.glob("checkpoint_epoch_*_interrupted.pth"))
    if interrupted_checkpoints:
        # Sort by modification time
        latest = max(interrupted_checkpoints, key=lambda p: p.stat().st_mtime)
        return latest, "interrupted"
    
    # Look for regular checkpoints
    regular_checkpoints = list(model_path.glob("indicator_transformer_checkpoint_*.pth"))
    if regular_checkpoints:
        # Extract epoch numbers and find the latest
        latest = max(regular_checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        return latest, "regular"
    
    # Look for best model
    best_model = model_path / "indicator_transformer_best.pth"
    if best_model.exists():
        return best_model, "best"
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument('--checkpoint', help='Specific checkpoint file to resume from')
    parser.add_argument('--epochs', type=int, help='Additional epochs to train')
    parser.add_argument('--list', action='store_true', help='List available checkpoints')
    
    args = parser.parse_args()
    
    if args.list:
        # List all available checkpoints
        model_dir = Path("models")
        if model_dir.exists():
            checkpoints = []
            
            # Interrupted checkpoints
            for ckpt in model_dir.glob("checkpoint_epoch_*_interrupted.pth"):
                checkpoints.append((ckpt, "interrupted", ckpt.stat().st_mtime))
            
            # Regular checkpoints
            for ckpt in model_dir.glob("indicator_transformer_checkpoint_*.pth"):
                checkpoints.append((ckpt, "regular", ckpt.stat().st_mtime))
            
            # Best model
            best = model_dir / "indicator_transformer_best.pth"
            if best.exists():
                checkpoints.append((best, "best", best.stat().st_mtime))
            
            if checkpoints:
                print("\nAvailable checkpoints:")
                print("-" * 80)
                for ckpt, ckpt_type, mtime in sorted(checkpoints, key=lambda x: x[2], reverse=True):
                    mod_time = datetime.fromtimestamp(mtime)
                    size_mb = ckpt.stat().st_size / 1024 / 1024
                    
                    # Load checkpoint to get epoch info
                    try:
                        checkpoint = torch.load(ckpt, map_location='cpu')
                        epoch = checkpoint.get('epoch', 'unknown')
                        val_loss = checkpoint.get('val_loss', 'N/A')
                        print(f"{ckpt.name:<50} | Epoch: {epoch:<6} | "
                              f"Val Loss: {val_loss:<10.4f} | "
                              f"Type: {ckpt_type:<12} | "
                              f"Size: {size_mb:.1f}MB | "
                              f"Modified: {mod_time:%Y-%m-%d %I:%M %p}")
                    except:
                        print(f"{ckpt.name:<50} | Type: {ckpt_type:<12} | "
                              f"Size: {size_mb:.1f}MB | "
                              f"Modified: {mod_time:%Y-%m-%d %I:%M %p}")
            else:
                print("No checkpoints found.")
        return
    
    # Find checkpoint to resume from
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
        checkpoint_type = "specified"
    else:
        checkpoint_path, checkpoint_type = find_latest_checkpoint()
        if checkpoint_path is None:
            logger.error("No checkpoint found to resume from!")
            logger.info("Train a new model with: python train_indicator_transformer.py")
            return
    
    logger.info(f"\n📂 Found checkpoint: {checkpoint_path}")
    logger.info(f"   Type: {checkpoint_type}")
    
    # Load checkpoint info
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0) + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"   Epoch: {start_epoch - 1}")
        logger.info(f"   Train losses recorded: {len(train_losses)}")
        logger.info(f"   Val losses recorded: {len(val_losses)}")
        
        if train_losses:
            logger.info(f"   Last train loss: {train_losses[-1]:.4f}")
        if val_losses:
            logger.info(f"   Last val loss: {val_losses[-1]:.4f}")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return
    
    # Determine total epochs
    if args.epochs:
        total_epochs = start_epoch + args.epochs - 1
    else:
        # Default: train for 20 more epochs
        total_epochs = start_epoch + 19
    
    logger.info(f"\n🚀 Resuming training:")
    logger.info(f"   Starting from epoch: {start_epoch}")
    logger.info(f"   Training until epoch: {total_epochs}")
    logger.info(f"   Additional epochs: {total_epochs - start_epoch + 1}")
    
    # Build command to resume training
    import subprocess
    import sys
    
    cmd = [
        sys.executable,
        "train_indicator_transformer.py",
        "--resume", str(checkpoint_path),
        "--epochs", str(total_epochs)
    ]
    
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted")


if __name__ == "__main__":
    main()