#!/usr/bin/env python3
"""
Train Indicator Transformer Model
=================================
Trains the AI model to select optimal indicators based on market conditions.
Uses self-supervised learning with historical price data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import time
import signal
import atexit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicator_transformer import (
    IndicatorTransformer, 
    IndicatorLibrary, 
    MarketRegimeAnalyzer,
    AIIndicatorSelector
)
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider
from indicator_cache import IndicatorCache, CachedIndicatorComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalHandler:
    """Handle graceful shutdown on SIGINT (Ctrl-C)."""
    
    def __init__(self):
        self.shutdown_requested = False
        self.executor = None
        self.active_futures = []
        self.original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        atexit.register(self.cleanup)
        
    def _handle_signal(self, signum, frame):
        """Handle SIGINT signal."""
        logger.info("\n\n🛑 Shutdown requested (Ctrl-C detected)...")
        self.shutdown_requested = True
        
        # Cancel futures
        if self.active_futures:
            logger.info(f"Cancelling {len(self.active_futures)} active tasks...")
            for future in self.active_futures:
                future.cancel()
        
        # Shutdown executor
        if self.executor:
            logger.info("Shutting down thread pool...")
            self.executor.shutdown(wait=False)
        
        # Save any pending cache data
        logger.info("Saving cache manifest...")
        try:
            from indicator_cache import IndicatorCache
            cache = IndicatorCache()
            cache._save_manifest()
        except:
            pass
        
        logger.info("Cleanup complete. Exiting...")
        sys.exit(0)
    
    def register_executor(self, executor):
        """Register ThreadPoolExecutor for cleanup."""
        self.executor = executor
    
    def register_future(self, future):
        """Register a future for cancellation."""
        self.active_futures.append(future)
    
    def unregister_future(self, future):
        """Remove completed future."""
        if future in self.active_futures:
            self.active_futures.remove(future)
    
    def cleanup(self):
        """Cleanup on exit."""
        # Restore original signal handler
        signal.signal(signal.SIGINT, self.original_sigint)
    
    @property
    def should_stop(self):
        """Check if shutdown was requested."""
        return self.shutdown_requested


# Global signal handler
signal_handler = SignalHandler()


class IndicatorSelectionDataset(Dataset):
    """Dataset for training indicator selection model."""
    
    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime, 
                 window_size: int = 100, future_window: int = 20, num_workers: int = None):
        """
        Initialize dataset.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            window_size: Size of historical window for indicators
            future_window: Future window for calculating returns
        """
        self.symbols = symbols
        self.window_size = window_size
        self.future_window = future_window
        self.num_workers = num_workers
        
        # Initialize components
        self.db_provider = DatabaseDataProvider()
        self.indicator_library = IndicatorLibrary()
        self.market_analyzer = MarketRegimeAnalyzer()
        
        # Initialize cache
        self.cache = IndicatorCache()
        self.cached_computer = CachedIndicatorComputer(self.indicator_library, self.cache)
        
        # Load and prepare data
        logger.info(f"Loading data for {len(symbols)} symbols...")
        logger.info(f"Cache stats: {self.cache.get_stats()}")
        self.samples = []
        self._prepare_samples(start_date, end_date)
        logger.info(f"Prepared {len(self.samples)} training samples")
        
        # Final cache stats
        final_stats = self.cache.get_stats()
        logger.info(f"Cache performance: {final_stats['hit_rate']:.1%} hit rate, {final_stats['saves']} new computations")
        
    def _prepare_samples(self, start_date: datetime, end_date: datetime):
        """Prepare training samples from historical data with enhanced progress tracking."""
        # Track overall progress
        total_symbols = len(self.symbols)
        start_time = time.time()
        samples_per_symbol = {}
        
        # Memory monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process symbols with ThreadPoolExecutor
        # Use more workers but leave some CPU headroom for other processes
        if self.num_workers is None:
            # Default: use all cores minus 2 (leave headroom), but not more than number of symbols
            max_workers = max(1, min(os.cpu_count() - 2, len(self.symbols)))
        else:
            # User-specified number of workers
            max_workers = max(1, min(self.num_workers, len(self.symbols)))
        
        logger.info(f"Using {max_workers} worker threads for parallel processing (out of {os.cpu_count()} CPU cores)")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Register executor with signal handler
            signal_handler.register_executor(executor)
            
            # Submit all tasks
            future_to_symbol = {}
            for symbol in self.symbols:
                if signal_handler.should_stop:
                    break
                future = executor.submit(self._process_symbol, symbol, start_date, end_date)
                signal_handler.register_future(future)
                future_to_symbol[future] = symbol
            
            # Progress bar for symbol processing
            with tqdm(total=total_symbols, desc="Processing symbols") as pbar:
                for future in as_completed(future_to_symbol):
                    # Check for shutdown
                    if signal_handler.should_stop:
                        logger.info("Stopping data preparation due to shutdown request...")
                        break
                    
                    symbol = future_to_symbol[future]
                    signal_handler.unregister_future(future)
                    
                    try:
                        symbol_samples = future.result()
                        self.samples.extend(symbol_samples)
                        samples_per_symbol[symbol] = len(symbol_samples)
                        
                        # Update progress
                        pbar.update(1)
                        elapsed = time.time() - start_time
                        symbols_done = len(samples_per_symbol)
                        eta = (elapsed / symbols_done) * (total_symbols - symbols_done) if symbols_done > 0 else 0
                        
                        # Memory check
                        current_memory = process.memory_info().rss / 1024 / 1024
                        memory_increase = current_memory - initial_memory
                        
                        pbar.set_postfix({
                            'samples': len(self.samples),
                            'ETA': f"{eta/60:.1f}m",
                            'mem': f"+{memory_increase:.0f}MB",
                            'cache_hits': self.cache.get_stats()['hit_rate']
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing {symbol}: {e}")
                        samples_per_symbol[symbol] = 0
        
        # Summary
        total_time = time.time() - start_time
        logger.info(f"\nData preparation complete:")
        logger.info(f"  Total time: {total_time/60:.1f} minutes")
        logger.info(f"  Total samples: {len(self.samples):,}")
        logger.info(f"  Avg samples/symbol: {len(self.samples)/len(self.symbols):.0f}")
        logger.info(f"  Memory used: +{(current_memory - initial_memory):.0f}MB")
        
        # Show top/bottom symbols by sample count
        sorted_symbols = sorted(samples_per_symbol.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"  Most samples: {sorted_symbols[0][0]} ({sorted_symbols[0][1]} samples)")
        logger.info(f"  Least samples: {sorted_symbols[-1][0]} ({sorted_symbols[-1][1]} samples)")
    
    def _process_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Process a single symbol and return its samples."""
        symbol_samples = []
        
        try:
            # Get data with extra buffer for indicators
            buffer_start = start_date - timedelta(days=30)
            data = self.db_provider.get_minute_data(symbol, buffer_start, end_date)
            
            if len(data) < self.window_size + self.future_window:
                return symbol_samples
            
            # Progress for this symbol
            num_windows = (len(data) - self.window_size - self.future_window) // 60
            
            # Create samples at regular intervals
            for i in range(self.window_size, len(data) - self.future_window, 60):  # Every hour
                # Historical window
                hist_data = data.iloc[i-self.window_size:i]
                
                # Future data for labels
                future_data = data.iloc[i:i+self.future_window]
                
                # Compute features
                try:
                    indicator_values = self._compute_indicators_cached(hist_data, symbol)
                    market_context = self.market_analyzer.analyze_market_context(hist_data)
                    
                    # Calculate future return as label
                    future_return = (future_data['close'].iloc[-1] / hist_data['close'].iloc[-1]) - 1
                    
                    # Calculate which indicators would have been useful
                    indicator_usefulness = self._calculate_indicator_usefulness(
                        hist_data, future_data, indicator_values
                    )
                    
                    symbol_samples.append({
                        'symbol': symbol,
                        'timestamp': hist_data.index[-1],
                        'indicator_values': indicator_values,
                        'market_context': market_context,
                        'future_return': future_return,
                        'indicator_usefulness': indicator_usefulness
                    })
                    
                except Exception as e:
                    logger.debug(f"Error processing sample for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error loading data for {symbol}: {e}")
            
        return symbol_samples
                
    def _compute_indicators(self, data: pd.DataFrame) -> np.ndarray:
        """Compute all indicator values."""
        values = []
        for indicator_name in self.indicator_library.indicators:
            try:
                indicator_series = self.indicator_library.compute_indicator(data, indicator_name)
                if len(indicator_series) > 0 and not pd.isna(indicator_series.iloc[-1]):
                    values.append(float(indicator_series.iloc[-1]))
                else:
                    values.append(0.0)
            except:
                values.append(0.0)
        return np.array(values, dtype=np.float32)
    
    def _compute_indicators_cached(self, data: pd.DataFrame, symbol: str) -> np.ndarray:
        """Compute all indicator values with caching."""
        values = []
        for indicator_name in self.indicator_library.indicators:
            try:
                indicator_series = self.cached_computer.compute_indicator(data, indicator_name, symbol)
                if len(indicator_series) > 0 and not pd.isna(indicator_series.iloc[-1]):
                    values.append(float(indicator_series.iloc[-1]))
                else:
                    values.append(0.0)
            except:
                values.append(0.0)
        return np.array(values, dtype=np.float32)
    
    def _calculate_indicator_usefulness(self, hist_data: pd.DataFrame, 
                                      future_data: pd.DataFrame, 
                                      indicator_values: np.ndarray) -> np.ndarray:
        """
        Calculate how useful each indicator would have been for predicting future price movement.
        This creates our training labels.
        """
        usefulness = np.zeros(len(indicator_values), dtype=np.float32)
        
        # Future price movement
        future_return = (future_data['close'].iloc[-1] / hist_data['close'].iloc[-1]) - 1
        future_volatility = future_data['close'].pct_change().std()
        
        # Analyze each indicator's predictive value
        for i, (indicator_name, indicator_func) in enumerate(self.indicator_library.indicators.items()):
            try:
                # Get indicator time series
                indicator_series = self.indicator_library.compute_indicator(hist_data, indicator_name)
                
                if len(indicator_series) < 10:
                    continue
                
                # Calculate indicator signal
                current_value = indicator_series.iloc[-1]
                mean_value = indicator_series.mean()
                std_value = indicator_series.std() + 1e-8
                
                # Normalize indicator
                z_score = (current_value - mean_value) / std_value
                
                # Different scoring for different indicator types
                if 'RSI' in indicator_name:
                    # RSI useful at extremes
                    if (current_value < 30 and future_return > 0) or \
                       (current_value > 70 and future_return < 0):
                        usefulness[i] = abs(future_return) * 10
                        
                elif 'MACD' in indicator_name:
                    # MACD useful for trend following
                    if np.sign(current_value) == np.sign(future_return):
                        usefulness[i] = abs(future_return) * 5
                        
                elif 'BB' in indicator_name:
                    # Bollinger bands useful in ranging markets
                    if future_volatility < hist_data['close'].pct_change().std():
                        usefulness[i] = 0.5
                        
                elif 'ATR' in indicator_name:
                    # ATR always somewhat useful for risk management
                    usefulness[i] = 0.3 + min(future_volatility * 10, 0.7)
                    
                elif 'VWAP' in indicator_name:
                    # VWAP useful for mean reversion
                    price_to_vwap = hist_data['close'].iloc[-1] / current_value - 1
                    if abs(price_to_vwap) > 0.02 and np.sign(price_to_vwap) != np.sign(future_return):
                        usefulness[i] = abs(price_to_vwap) * 10
                        
                else:
                    # Generic scoring based on correlation with future returns
                    if abs(z_score) > 1.5 and np.sign(z_score) == np.sign(future_return):
                        usefulness[i] = abs(z_score) * abs(future_return) * 3
                        
            except Exception as e:
                logger.debug(f"Error calculating usefulness for {indicator_name}: {e}")
                
        # Normalize usefulness scores
        max_usefulness = usefulness.max() + 1e-8
        usefulness = usefulness / max_usefulness
        
        # Ensure at least some indicators are marked as useful
        if usefulness.max() < 0.1:
            # Mark momentum indicators as slightly useful
            for i, name in enumerate(self.indicator_library.indicators.keys()):
                if any(x in name for x in ['RSI', 'MACD', 'EMA']):
                    usefulness[i] = 0.3
                    
        return usefulness
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'indicator_values': torch.FloatTensor(sample['indicator_values']),
            'market_context': torch.FloatTensor(sample['market_context']),
            'indicator_usefulness': torch.FloatTensor(sample['indicator_usefulness']),
            'future_return': torch.FloatTensor([sample['future_return']])
        }


class IndicatorTransformerTrainer:
    """Trainer for the indicator selection model."""
    
    def __init__(self, model: IndicatorTransformer, device: str = 'cuda', use_compile: bool = True):
        self.device = torch.device(device)
        
        # Move model to device with optimal memory format
        if device == 'mps':
            # Note: channels_last is for CNNs, but we'll keep contiguous for transformer
            self.model = model.to(device)
        else:
            self.model = model.to(device)
        
        # Compile model for better performance (Metal kernel fusion on MPS)
        if use_compile and device in ['cuda', 'mps']:
            try:
                logger.info("Compiling model with torch.compile for optimized performance...")
                self.model = torch.compile(self.model, backend="inductor", mode="max-autotune")
                self.compiled = True
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Falling back to eager mode.")
                self.compiled = False
        else:
            self.compiled = False
        
        # Adjust learning rate for MPS (slightly higher works better)
        lr = 2e-4 if device == 'mps' else 1e-4
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Enable mixed precision for MPS/CUDA
        self.use_amp = device in ['cuda', 'mps']
        if self.use_amp:
            # Use bfloat16 for MPS (better than fp16)
            self.amp_dtype = torch.bfloat16 if device == 'mps' else torch.float16
            logger.info(f"Mixed precision training enabled with {self.amp_dtype}")
        
        # Loss functions
        self.selection_criterion = nn.BCELoss()
        self.weight_criterion = nn.MSELoss()
        
        # History
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch with detailed progress and mixed precision."""
        self.model.train()
        total_loss = 0.0
        batch_losses = []
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Check for shutdown request
            if signal_handler.should_stop:
                logger.info("Training interrupted by user...")
                return total_loss / max(1, batch_idx)
            # Move to device
            indicator_values = batch['indicator_values'].to(self.device)
            market_context = batch['market_context'].to(self.device)
            indicator_usefulness = batch['indicator_usefulness'].to(self.device)
            
            batch_size = indicator_values.size(0)
            num_indicators = indicator_values.size(1)
            
            # Create indicator indices
            indicator_indices = torch.arange(num_indicators).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            
            # Use automatic mixed precision if available
            if self.use_amp:
                with torch.autocast(device_type=str(self.device), dtype=self.amp_dtype):
                    # Forward pass
                    outputs = self.model(indicator_values, market_context, indicator_indices)
                    
                    # Calculate losses
                    selection_probs = outputs['selection_probs']
                    indicator_weights = outputs['indicator_weights']
                    
                    # Selection loss: encourage selecting useful indicators
                    selection_loss = self.selection_criterion(selection_probs, indicator_usefulness)
                    
                    # Weight loss: weights should be proportional to usefulness
                    weight_targets = indicator_usefulness * selection_probs.detach()
                    weight_targets = weight_targets / (weight_targets.sum(dim=1, keepdim=True) + 1e-8)
                    weight_loss = self.weight_criterion(indicator_weights, weight_targets)
                    
                    # Sparsity penalty: encourage selecting fewer indicators
                    sparsity_loss = selection_probs.mean() * 0.1
                    
                    # Total loss
                    loss = selection_loss + 0.5 * weight_loss + sparsity_loss
            else:
                # Non-AMP path (same as before)
                outputs = self.model(indicator_values, market_context, indicator_indices)
                selection_probs = outputs['selection_probs']
                indicator_weights = outputs['indicator_weights']
                selection_loss = self.selection_criterion(selection_probs, indicator_usefulness)
                weight_targets = indicator_usefulness * selection_probs.detach()
                weight_targets = weight_targets / (weight_targets.sum(dim=1, keepdim=True) + 1e-8)
                weight_loss = self.weight_criterion(indicator_weights, weight_targets)
                sparsity_loss = selection_probs.mean() * 0.1
                loss = selection_loss + 0.5 * weight_loss + sparsity_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Update progress bar with running stats
            postfix_dict = {
                'loss': f'{batch_loss:.4f}',
                'avg': f'{np.mean(batch_losses[-50:]):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            }
            
            # Add device utilization for MPS
            if str(self.device) == 'mps':
                # MPS doesn't have direct utilization API, but we can track timing
                if not hasattr(self, 'batch_times'):
                    self.batch_times = []
                if batch_idx > 0:
                    self.batch_times.append(time.time() - self.last_batch_time)
                    avg_batch_time = np.mean(self.batch_times[-50:])
                    postfix_dict['ms/batch'] = f'{avg_batch_time*1000:.0f}'
                self.last_batch_time = time.time()
            
            pbar.set_postfix(postfix_dict)
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        metrics = {
            'avg_selected': 0,
            'selection_accuracy': 0,
            'weight_correlation': 0
        }
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                indicator_values = batch['indicator_values'].to(self.device)
                market_context = batch['market_context'].to(self.device)
                indicator_usefulness = batch['indicator_usefulness'].to(self.device)
                
                batch_size = indicator_values.size(0)
                num_indicators = indicator_values.size(1)
                
                # Create indicator indices
                indicator_indices = torch.arange(num_indicators).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                
                # Use mixed precision for validation too
                if self.use_amp:
                    with torch.autocast(device_type=str(self.device), dtype=self.amp_dtype):
                        # Forward pass
                        outputs = self.model(indicator_values, market_context, indicator_indices)
                        
                        # Calculate losses
                        selection_probs = outputs['selection_probs']
                        indicator_weights = outputs['indicator_weights']
                        
                        selection_loss = self.selection_criterion(selection_probs, indicator_usefulness)
                        
                        weight_targets = indicator_usefulness * selection_probs
                        weight_targets = weight_targets / (weight_targets.sum(dim=1, keepdim=True) + 1e-8)
                        weight_loss = self.weight_criterion(indicator_weights, weight_targets)
                        
                        loss = selection_loss + 0.5 * weight_loss
                else:
                    outputs = self.model(indicator_values, market_context, indicator_indices)
                    selection_probs = outputs['selection_probs']
                    indicator_weights = outputs['indicator_weights']
                    selection_loss = self.selection_criterion(selection_probs, indicator_usefulness)
                    weight_targets = indicator_usefulness * selection_probs
                    weight_targets = weight_targets / (weight_targets.sum(dim=1, keepdim=True) + 1e-8)
                    weight_loss = self.weight_criterion(indicator_weights, weight_targets)
                    loss = selection_loss + 0.5 * weight_loss
                
                total_loss += loss.item()
                
                # Calculate metrics
                selected = (selection_probs > 0.3).float()
                metrics['avg_selected'] += selected.sum(dim=1).mean().item()
                
                # Selection accuracy: how well we identify useful indicators
                useful = (indicator_usefulness > 0.5).float()
                correct = (selected * useful).sum(dim=1)
                total_useful = useful.sum(dim=1) + 1e-8
                metrics['selection_accuracy'] += (correct / total_useful).mean().item()
                
        # Average metrics
        num_batches = len(dataloader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return total_loss / num_batches, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, save_path: str = 'models', start_epoch: int = 0):
        """Train the model with comprehensive progress tracking."""
        os.makedirs(save_path, exist_ok=True)
        best_val_loss = float('inf')
        self.train_start_time = time.time()
        
        for epoch in range(start_epoch, start_epoch + epochs):
            # Check for shutdown
            if signal_handler.should_stop:
                logger.info("\nTraining stopped by user request.")
                break
                
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Check again after training epoch
            if signal_handler.should_stop:
                logger.info("\nSaving checkpoint before exit...")
                self._save_checkpoint(epoch, save_path, train_loss, 0.0, {})
                break
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate schedule
            self.scheduler.step()
            
            # Detailed logging with progress tracking
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Avg Selected: {metrics['avg_selected']:.1f}")
            logger.info(f"  Selection Accuracy: {metrics['selection_accuracy']:.2%}")
            
            # Show optimization status
            if hasattr(self, 'use_amp') and self.use_amp:
                logger.info(f"  Mixed Precision: {self.amp_dtype}")
            if hasattr(self, 'compiled') and self.compiled:
                logger.info(f"  Model Compiled: ✓")
            
            # Estimate time remaining
            elapsed_epochs = epoch - start_epoch + 1
            elapsed_time = time.time() - self.train_start_time
            avg_epoch_time = elapsed_time / elapsed_epochs
            remaining_epochs = (start_epoch + epochs) - epoch - 1
            eta_seconds = avg_epoch_time * remaining_epochs
            
            logger.info(f"  Time per epoch: {avg_epoch_time/60:.1f} min")
            logger.info(f"  ETA: {eta_seconds/60:.1f} min ({eta_seconds/3600:.1f} hours)")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': metrics
                }, f"{save_path}/indicator_transformer_best.pth")
                logger.info("  ✓ Saved best model")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, f"{save_path}/indicator_transformer_checkpoint_{epoch+1}.pth")
        
        # Save final model
        final_epoch = start_epoch + epochs - 1
        torch.save({
            'epoch': final_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, f"{save_path}/indicator_transformer_final.pth")
        
        # Plot training history
        self._plot_training_history(save_path)
    
    def _save_checkpoint(self, epoch, save_path, train_loss, val_loss, metrics):
        """Save checkpoint during training."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_path = f"{save_path}/checkpoint_epoch_{epoch+1}_interrupted.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
    def _plot_training_history(self, save_path: str):
        """Plot and save training history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Indicator Transformer Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/training_history.png")
        plt.close()


def get_optimal_device():
    """Get optimal device for training with Apple Silicon support."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU
        return 'mps'
    else:
        return 'cpu'


def main():
    """Main training function with enhanced monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Indicator Transformer Model')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on')
    parser.add_argument('--days', type=int, default=180, help='Days of historical data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default=get_optimal_device(), help='Device (cuda/mps/cpu)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--clear-cache', action='store_true', help='Clear indicator cache before training')
    parser.add_argument('--num-workers', type=int, help='Number of parallel workers (default: auto)')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile optimization')
    parser.add_argument('--resume', help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Default symbols if none provided
    if not args.symbols:
        args.symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
            'JPM', 'BAC', 'GS', 'WMT', 'HD', 'DIS', 'NFLX', 'AMD'
        ]
    
    logger.info("🚀 Starting Indicator Transformer Training")
    logger.info(f"Training on {len(args.symbols)} symbols for {args.epochs} epochs")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Optimize settings for MPS
    if args.device == 'mps':
        # Set optimal matmul precision for mixed precision training
        torch.set_float32_matmul_precision('medium')
        logger.info("  Set matmul precision to 'medium' for better MPS performance")
        
        # Adjust batch size if needed
        if args.batch_size == 32:
            args.batch_size = 16
            logger.info(f"  Adjusted batch size to {args.batch_size} for optimal MPS performance")
        
        # Set OMP threads for CPU operations
        import subprocess
        try:
            physical_cpus = int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).decode().strip())
            os.environ['OMP_NUM_THREADS'] = str(physical_cpus)
            logger.info(f"  Set OMP_NUM_THREADS={physical_cpus} for CPU operations")
        except:
            pass
    
    # System info
    logger.info(f"\nSystem Information:")
    logger.info(f"  CPU cores: {os.cpu_count()}")
    logger.info(f"  Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    
    if args.device == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")
    elif args.device == 'mps':
        logger.info(f"  Apple Silicon GPU: Metal Performance Shaders enabled")
        logger.info(f"  Unified memory architecture - no CPU-GPU transfers needed")
        # Set environment variable for better MPS performance
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Clear cache if requested
    if args.clear_cache:
        logger.info("\nClearing indicator cache...")
        cache = IndicatorCache()
        cache.clear()
    
    # Create dataset with timing
    logger.info("\nLoading training data...")
    data_start_time = time.time()
    dataset = IndicatorSelectionDataset(args.symbols, start_date, end_date, num_workers=args.num_workers)
    data_load_time = time.time() - data_start_time
    logger.info(f"Data loading completed in {data_load_time/60:.1f} minutes")
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders with device-specific settings
    # Optimize for feeding the GPU fast enough
    if args.device == 'mps':
        # MPS can use workers, but keep it reasonable
        dataloader_workers = min(4, os.cpu_count() // 2)
        persistent_workers = True if dataloader_workers > 0 else False
    else:
        dataloader_workers = 0  # Keep simple for now
        persistent_workers = False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=dataloader_workers,
        persistent_workers=persistent_workers,
        pin_memory=(args.device == 'cuda'),  # Only pin memory for CUDA
        prefetch_factor=2 if dataloader_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=dataloader_workers,
        persistent_workers=persistent_workers,
        pin_memory=(args.device == 'cuda'),
        prefetch_factor=2 if dataloader_workers > 0 else None
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    indicator_library = IndicatorLibrary()
    num_indicators = len(indicator_library.indicators)
    
    model = IndicatorTransformer(
        num_indicators=num_indicators,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"\n📂 Resuming from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"   Resumed from epoch {start_epoch}")
            
            # Adjust epochs if needed
            if args.epochs <= start_epoch:
                args.epochs = start_epoch + 20  # Train 20 more epochs by default
                logger.info(f"   Adjusted total epochs to {args.epochs}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh training instead...")
            start_epoch = 0
    
    # Train
    use_compile = not args.no_compile
    trainer = IndicatorTransformerTrainer(model, device=args.device, use_compile=use_compile)
    
    # Load optimizer state if resuming
    if args.resume and start_epoch > 0:
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.train_losses = checkpoint.get('train_losses', [])
            trainer.val_losses = checkpoint.get('val_losses', [])
            logger.info("   Restored optimizer and loss history")
        except:
            logger.warning("   Could not restore optimizer state")
    
    # Adjust training epochs
    remaining_epochs = args.epochs - start_epoch
    if remaining_epochs > 0:
        logger.info(f"\n🎯 Training for {remaining_epochs} more epochs (from {start_epoch} to {args.epochs})")
        trainer.train(train_loader, val_loader, epochs=remaining_epochs, start_epoch=start_epoch)
    else:
        logger.info("No additional epochs to train.")
    
    total_training_time = time.time() - data_start_time
    
    if signal_handler.should_stop:
        logger.info("\n⚠️ Training interrupted by user")
        logger.info("Partial results saved. Resume training from checkpoint if needed.")
    else:
        logger.info(f"\n✅ Training complete!")
        logger.info(f"Total time: {total_training_time/60:.1f} minutes ({total_training_time/3600:.1f} hours)")
        logger.info("Model saved to models/indicator_transformer_best.pth")
    
    # Final cache stats
    cache = IndicatorCache()
    cache_stats = cache.get_stats()
    logger.info(f"\nCache Statistics:")
    logger.info(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    logger.info(f"  Total saves: {cache_stats['saves']:,}")
    logger.info(f"  Cache size: {cache_stats['total_size_mb']:.1f} MB")
    
    # Test the trained model
    logger.info("\n🧪 Testing trained model...")
    test_model()


def test_model():
    """Test the trained model to ensure it's working."""
    # Load the trained model
    model_path = Path("models/indicator_transformer_best.pth")
    if not model_path.exists():
        logger.error("No trained model found!")
        return
    
    # Initialize selector with trained model
    selector = AIIndicatorSelector(model_path=model_path)
    
    # Create test data
    dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
    test_data = pd.DataFrame({
        'open': 100 + np.random.randn(500).cumsum() * 0.1,
        'high': 100.5 + np.random.randn(500).cumsum() * 0.1,
        'low': 99.5 + np.random.randn(500).cumsum() * 0.1,
        'close': 100 + np.random.randn(500).cumsum() * 0.1,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    # Test selection
    result = selector.select_indicators(test_data)
    selected = result['selected_indicators']
    
    logger.info(f"\nSelected {len(selected)} indicators:")
    for name, info in list(selected.items())[:10]:
        logger.info(f"  {name}: prob={info['selection_prob']:.3f}, weight={info['weight']:.3f}")
    
    # Check probabilities
    probs = [info['selection_prob'] for info in selected.values()]
    logger.info(f"\nProbability stats:")
    logger.info(f"  Min: {min(probs):.6f}")
    logger.info(f"  Max: {max(probs):.6f}")
    logger.info(f"  Mean: {np.mean(probs):.6f}")
    
    if max(probs) > 0.01:
        logger.info("\n✅ Model is producing reasonable probabilities!")
    else:
        logger.warning("\n⚠️ Model still producing low probabilities, may need more training")


if __name__ == "__main__":
    main()