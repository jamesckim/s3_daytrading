#!/usr/bin/env python3
"""
Fixed MLX Trainer
=================
Corrected MLX trainer implementation.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm
import logging
import os

logger = logging.getLogger(__name__)


class MLXTrainerFixed:
    """Fixed MLX trainer for the indicator transformer."""
    
    def __init__(self, model, learning_rate: float = 2e-4):
        self.model = model
        self.optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-5)
        
        # History
        self.train_losses = []
        self.val_losses = []
        
        logger.info("🚀 MLX Trainer initialized")
        logger.info("   Using native Apple Silicon optimization")
    
    def loss_fn(self, model, batch):
        """Calculate loss for a batch."""
        # Forward pass
        outputs = model(
            batch['indicator_values'],
            batch['market_context'],
            batch['indicator_indices']
        )
        
        selection_probs = outputs['selection_probs']
        indicator_weights = outputs['indicator_weights']
        indicator_usefulness = batch['indicator_usefulness']
        
        # Selection loss (BCE)
        selection_loss = mx.mean(
            -indicator_usefulness * mx.log(selection_probs + 1e-8) -
            (1 - indicator_usefulness) * mx.log(1 - selection_probs + 1e-8)
        )
        
        # Weight loss (MSE)
        weight_targets = indicator_usefulness * selection_probs
        weight_targets = weight_targets / (mx.sum(weight_targets, axis=1, keepdims=True) + 1e-8)
        weight_loss = mx.mean((indicator_weights - weight_targets) ** 2)
        
        # Sparsity penalty
        sparsity_loss = mx.mean(selection_probs) * 0.1
        
        # Total loss (return single value for MLX)
        total_loss = selection_loss + 0.5 * weight_loss + sparsity_loss
        
        return total_loss
    
    def train_epoch(self, dataset, batch_size: int = 32):
        """Train for one epoch."""
        num_samples = len(dataset)
        indices = np.random.permutation(num_samples)
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(range(0, num_samples, batch_size), desc="Training")
        
        for i in pbar:
            # Get batch
            batch_indices = indices[i:i + batch_size].tolist()
            batch = dataset.get_batch(batch_indices)
            
            # Define loss function with fixed signature
            def loss_and_grad(model):
                return self.loss_fn(model, batch)
            
            # Compute loss and gradients
            loss_and_grad_fn = mx.value_and_grad(loss_and_grad)
            loss, grads = loss_and_grad_fn(self.model)
            
            # Update weights
            self.optimizer.update(self.model, grads)
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss/num_batches:.4f}'
            })
            
            # Force evaluation periodically
            if num_batches % 10 == 0:
                mx.eval(self.model.parameters())
        
        return total_loss / num_batches
    
    def validate(self, dataset, batch_size: int = 32):
        """Validate the model."""
        num_samples = len(dataset)
        total_loss = 0.0
        num_batches = 0
        
        metrics = {
            'avg_selected': 0,
            'selection_accuracy': 0,
        }
        
        for i in range(0, num_samples, batch_size):
            # Get batch
            batch_indices = list(range(i, min(i + batch_size, num_samples)))
            batch = dataset.get_batch(batch_indices)
            
            # Forward pass only
            outputs = self.model(
                batch['indicator_values'],
                batch['market_context'],
                batch['indicator_indices']
            )
            
            # Calculate loss
            loss = self.loss_fn(self.model, batch)
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate metrics
            selection_probs = outputs['selection_probs']
            indicator_usefulness = batch['indicator_usefulness']
            
            selected = (selection_probs > 0.3).astype(mx.float32)
            metrics['avg_selected'] += mx.mean(mx.sum(selected, axis=1)).item()
            
            # Selection accuracy
            useful = (indicator_usefulness > 0.5).astype(mx.float32)
            correct = mx.sum(selected * useful, axis=1)
            total_useful = mx.sum(useful, axis=1) + 1e-8
            metrics['selection_accuracy'] += mx.mean(correct / total_useful).item()
        
        # Average metrics
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return total_loss / num_batches, metrics
    
    def train(self, train_dataset, val_dataset, epochs: int = 50, 
              batch_size: int = 32, save_path: str = 'models'):
        """Train the model."""
        os.makedirs(save_path, exist_ok=True)
        best_val_loss = float('inf')
        
        logger.info(f"\n🎯 Starting MLX training")
        logger.info(f"   Epochs: {epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Training samples: {len(train_dataset)}")
        logger.info(f"   Validation samples: {len(val_dataset)}")
        
        import time
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_dataset, batch_size)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_dataset, batch_size)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Avg Selected: {metrics['avg_selected']:.1f}")
            logger.info(f"  Selection Accuracy: {metrics['selection_accuracy']:.2%}")
            logger.info(f"  Epoch Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(save_path, 'best', epoch, train_loss, val_loss, metrics)
                logger.info("  ✓ Saved best model")
        
        total_time = time.time() - start_time
        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Total time: {total_time/60:.1f} min")
        logger.info(f"   Best val loss: {best_val_loss:.4f}")
    
    def _save_model(self, save_path, suffix, epoch, train_loss, val_loss, metrics):
        """Save model weights."""
        model_path = f"{save_path}/indicator_transformer_mlx_{suffix}.npz"
        
        # Get model weights
        weights = self.model.parameters()
        
        # Flatten weights dictionary
        flat_weights = {}
        
        def flatten_dict(d, parent_key=''):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, new_key)
                else:
                    flat_weights[new_key] = np.array(v)
        
        flatten_dict(weights)
        
        # Save
        np.savez(
            model_path,
            **flat_weights,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            **{f"metric_{k}": v for k, v in metrics.items()}
        )
        
        logger.info(f"Model saved to {model_path}")