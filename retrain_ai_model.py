#!/usr/bin/env python3
"""
Retrain AI Model with Optimized Indicators
==========================================
Retrain the transformer model after removing noisy EMA indicators.
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicator_transformer import IndicatorTransformer, IndicatorLibrary, MarketRegimeAnalyzer
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingDataset(Dataset):
    """Dataset for training the indicator selection model."""
    
    def __init__(self, 
                 symbols: List[str],
                 start_date: datetime,
                 end_date: datetime,
                 lookback_window: int = 100,
                 db_provider: Optional[DatabaseDataProvider] = None):
        """Initialize trading dataset."""
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_window = lookback_window
        
        # Initialize components
        self.db_provider = db_provider or DatabaseDataProvider()
        self.indicator_library = IndicatorLibrary()
        self.market_analyzer = MarketRegimeAnalyzer()
        
        # Get indicator names
        self.indicator_names = list(self.indicator_library.indicators.keys())
        self.num_indicators = len(self.indicator_names)
        
        # Prepare training data
        self.training_samples = []
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare training data from historical price data."""
        logger.info(f"Preparing training data for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                # Get minute data
                data = self.db_provider.get_minute_data(symbol, self.start_date, self.end_date)
                
                if data.empty or len(data) < self.lookback_window * 2:
                    continue
                
                # Generate samples with sliding window
                for i in range(self.lookback_window, len(data) - 1):
                    # Historical window
                    hist_data = data.iloc[i-self.lookback_window:i]
                    
                    # Future return for label (next 30 minutes)
                    current_price = data['close'].iloc[i]
                    future_price = data['close'].iloc[min(i+30, len(data)-1)]
                    future_return = (future_price - current_price) / current_price
                    
                    # Compute all indicators
                    indicator_values = self._compute_indicators(hist_data)
                    
                    # Get market context
                    market_context = self.market_analyzer.analyze_market_context(hist_data)
                    
                    # Create training sample
                    sample = {
                        'symbol': symbol,
                        'timestamp': data.index[i],
                        'indicator_values': indicator_values,
                        'market_context': market_context,
                        'future_return': future_return,
                        'current_price': current_price
                    }
                    
                    self.training_samples.append(sample)
                
                logger.info(f"  {symbol}: Generated {len(self.training_samples)} samples")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        logger.info(f"Total training samples: {len(self.training_samples)}")
    
    def _compute_indicators(self, data: pd.DataFrame) -> np.ndarray:
        """Compute all indicators for the given data."""
        indicator_values = []
        
        for indicator_name in self.indicator_names:
            try:
                value = self.indicator_library.compute_indicator(data, indicator_name)
                if len(value) > 0 and not pd.isna(value.iloc[-1]):
                    indicator_values.append(value.iloc[-1])
                else:
                    indicator_values.append(0.0)
            except:
                indicator_values.append(0.0)
        
        return np.array(indicator_values, dtype=np.float32)
    
    def __len__(self):
        return len(self.training_samples)
    
    def __getitem__(self, idx):
        sample = self.training_samples[idx]
        
        return {
            'indicator_values': torch.FloatTensor(sample['indicator_values']),
            'market_context': torch.FloatTensor(sample['market_context']),
            'indicator_indices': torch.LongTensor(range(self.num_indicators)),
            'future_return': torch.FloatTensor([sample['future_return']]),
            'symbol': sample['symbol']
        }


class ModelTrainer:
    """Trainer for the indicator selection model."""
    
    def __init__(self, model: IndicatorTransformer, device: str = 'cpu'):
        """Initialize trainer."""
        self.model = model.to(device)
        self.device = torch.device(device)
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss functions
        self.selection_criterion = nn.BCELoss()  # For indicator selection
        self.return_criterion = nn.MSELoss()     # For return prediction
        
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'selection_accuracy': [],
            'return_correlation': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            # Move to device
            indicator_values = batch['indicator_values'].to(self.device)
            market_context = batch['market_context'].to(self.device)
            indicator_indices = batch['indicator_indices'].to(self.device)
            future_returns = batch['future_return'].to(self.device)
            
            # Forward pass
            outputs = self.model(indicator_values, market_context, indicator_indices)
            
            # Create optimal selection labels based on future returns
            # Indicators that correlate with positive returns should be selected
            selection_labels = self._create_selection_labels(
                indicator_values, future_returns
            )
            
            # Calculate losses
            selection_loss = self.selection_criterion(
                outputs['selection_probs'], selection_labels
            )
            
            # Predict returns using weighted indicators
            predicted_returns = self._predict_returns(
                indicator_values, outputs['indicator_weights']
            )
            return_loss = self.return_criterion(predicted_returns, future_returns)
            
            # Combined loss
            loss = selection_loss + 0.5 * return_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _create_selection_labels(self, 
                                indicator_values: torch.Tensor, 
                                future_returns: torch.Tensor) -> torch.Tensor:
        """Create optimal selection labels based on indicator-return correlation."""
        batch_size, num_indicators = indicator_values.shape
        
        # Simple heuristic: select indicators that move in direction of returns
        # This is a simplified approach - in practice, you'd use more sophisticated logic
        selection_labels = torch.zeros_like(indicator_values)
        
        for i in range(batch_size):
            # Normalize indicators
            indicators = indicator_values[i]
            indicators_norm = (indicators - indicators.mean()) / (indicators.std() + 1e-8)
            
            # Select indicators with strong signal
            if future_returns[i] > 0:
                # For positive returns, select indicators with positive values
                selection_labels[i] = torch.sigmoid(indicators_norm * 2)
            else:
                # For negative returns, select indicators with negative values
                selection_labels[i] = torch.sigmoid(-indicators_norm * 2)
        
        return selection_labels
    
    def _predict_returns(self, 
                        indicator_values: torch.Tensor,
                        indicator_weights: torch.Tensor) -> torch.Tensor:
        """Predict returns using weighted indicators."""
        # Simple linear combination of weighted indicators
        # In practice, you'd use a more sophisticated prediction model
        weighted_sum = (indicator_values * indicator_weights).sum(dim=1, keepdim=True)
        
        # Scale to return range
        predicted_returns = torch.tanh(weighted_sum * 0.1)
        
        return predicted_returns
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct_selections = 0
        total_selections = 0
        predicted_returns = []
        actual_returns = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                indicator_values = batch['indicator_values'].to(self.device)
                market_context = batch['market_context'].to(self.device)
                indicator_indices = batch['indicator_indices'].to(self.device)
                future_returns = batch['future_return'].to(self.device)
                
                # Forward pass
                outputs = self.model(indicator_values, market_context, indicator_indices)
                
                # Create labels
                selection_labels = self._create_selection_labels(
                    indicator_values, future_returns
                )
                
                # Calculate losses
                selection_loss = self.selection_criterion(
                    outputs['selection_probs'], selection_labels
                )
                
                pred_returns = self._predict_returns(
                    indicator_values, outputs['indicator_weights']
                )
                return_loss = self.return_criterion(pred_returns, future_returns)
                
                loss = selection_loss + 0.5 * return_loss
                total_loss += loss.item()
                
                # Track metrics
                selected = outputs['selection_probs'] > 0.5
                correct = (selected == (selection_labels > 0.5)).float().sum()
                correct_selections += correct.item()
                total_selections += selected.numel()
                
                predicted_returns.extend(pred_returns.cpu().numpy())
                actual_returns.extend(future_returns.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        selection_accuracy = correct_selections / total_selections
        
        # Return correlation
        if len(predicted_returns) > 1:
            correlation = np.corrcoef(predicted_returns, actual_returns)[0, 1]
        else:
            correlation = 0.0
        
        return avg_loss, selection_accuracy, correlation
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 50,
              save_path: Path = None):
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, selection_acc, return_corr = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Selection Accuracy: {selection_acc:.2%}")
            logger.info(f"  Return Correlation: {return_corr:.3f}")
            
            # Save training history
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['selection_accuracy'].append(selection_acc)
            self.training_history['return_correlation'].append(return_corr)
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'training_history': self.training_history
                }, save_path)
                logger.info(f"  Saved best model with val_loss: {val_loss:.4f}")
        
        logger.info("Training completed!")
        return self.training_history


def main():
    """Main training script."""
    # Training configuration
    config = {
        'symbols': ["AAPL", "TSLA", "SPY", "NVDA", "AAOI", "META", "XOM", "AMD", 
                   "MSFT", "GOOGL", "AMZN", "JPM", "BAC", "NFLX", "QQQ"],
        'train_days': 180,  # 6 months of training data
        'val_days': 30,     # 1 month of validation data
        'batch_size': 32,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info("=" * 60)
    logger.info("AI Model Retraining with Optimized Indicators")
    logger.info("=" * 60)
    logger.info(f"Device: {config['device']}")
    
    # Data preparation
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    val_start = end_date - timedelta(days=config['val_days'])
    train_start = val_start - timedelta(days=config['train_days'])
    
    logger.info(f"Training period: {train_start.date()} to {val_start.date()}")
    logger.info(f"Validation period: {val_start.date()} to {end_date.date()}")
    
    # Create datasets
    logger.info("\nPreparing datasets...")
    train_dataset = TradingDataset(
        symbols=config['symbols'],
        start_date=train_start,
        end_date=val_start
    )
    
    val_dataset = TradingDataset(
        symbols=config['symbols'],
        start_date=val_start,
        end_date=end_date
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    logger.info("\nInitializing model...")
    model = IndicatorTransformer(
        num_indicators=train_dataset.num_indicators,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Create trainer
    trainer = ModelTrainer(model, device=config['device'])
    
    # Train model
    save_path = Path("models/retrained_indicator_transformer_optimized.pt")
    save_path.parent.mkdir(exist_ok=True)
    
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        save_path=save_path
    )
    
    # Save training history
    history_path = save_path.parent / "training_history_optimized.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining history saved to {history_path}")
    logger.info(f"Model saved to {save_path}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"Final selection accuracy: {history['selection_accuracy'][-1]:.2%}")
    logger.info(f"Final return correlation: {history['return_correlation'][-1]:.3f}")
    
    # Plot training curves if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train')
        axes[0, 0].plot(history['epoch'], history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        
        # Selection accuracy
        axes[0, 1].plot(history['epoch'], history['selection_accuracy'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Indicator Selection Accuracy')
        
        # Return correlation
        axes[1, 0].plot(history['epoch'], history['return_correlation'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_title('Return Prediction Correlation')
        
        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, 'Model Retrained\nwith Optimized Indicators', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plot_path = save_path.parent / "training_curves_optimized.png"
        plt.savefig(plot_path)
        logger.info(f"Training curves saved to {plot_path}")
        
    except ImportError:
        logger.info("Matplotlib not available - skipping plots")
    
    logger.info("\nRetraining completed successfully!")


if __name__ == "__main__":
    main()