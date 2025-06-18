#!/usr/bin/env python3
"""
Hybrid PyTorch/MLX Inference System
===================================
Production-ready inference that leverages MLX for speed while maintaining PyTorch compatibility.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import mlx.core as mx

from mlx_production_inference import MLXProductionInference, TradingSignal
from train_indicator_transformer import IndicatorTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for hybrid inference system."""
    use_mlx: bool = True
    mlx_threshold: int = 10  # Use MLX for batches larger than this
    fallback_to_pytorch: bool = True
    cache_models: bool = True
    benchmark_mode: bool = False
    log_performance: bool = True
    mlx_model_path: Optional[str] = None
    pytorch_model_path: Optional[str] = None


@dataclass 
class InferenceMetrics:
    """Track inference performance metrics."""
    total_inferences: int = 0
    mlx_inferences: int = 0
    pytorch_inferences: int = 0
    total_time_ms: float = 0.0
    mlx_time_ms: float = 0.0
    pytorch_time_ms: float = 0.0
    symbols_processed: int = 0
    cache_hits: int = 0
    errors: int = 0
    
    def add_inference(self, is_mlx: bool, time_ms: float, num_symbols: int):
        """Record an inference."""
        self.total_inferences += 1
        self.total_time_ms += time_ms
        self.symbols_processed += num_symbols
        
        if is_mlx:
            self.mlx_inferences += 1
            self.mlx_time_ms += time_ms
        else:
            self.pytorch_inferences += 1
            self.pytorch_time_ms += time_ms
    
    def get_summary(self) -> Dict:
        """Get performance summary."""
        return {
            "total_inferences": self.total_inferences,
            "mlx_percentage": (self.mlx_inferences / max(1, self.total_inferences)) * 100,
            "avg_time_ms": self.total_time_ms / max(1, self.total_inferences),
            "mlx_avg_ms": self.mlx_time_ms / max(1, self.mlx_inferences),
            "pytorch_avg_ms": self.pytorch_time_ms / max(1, self.pytorch_inferences),
            "throughput_symbols_per_sec": self.symbols_processed / max(0.001, self.total_time_ms / 1000),
            "mlx_speedup": (self.pytorch_time_ms / max(1, self.pytorch_inferences)) / 
                          (self.mlx_time_ms / max(1, self.mlx_inferences)) if self.mlx_inferences > 0 else 0
        }


class HybridInferenceSystem:
    """Hybrid inference system that intelligently uses PyTorch or MLX."""
    
    def __init__(self, config: InferenceConfig):
        """Initialize hybrid system."""
        self.config = config
        self.metrics = InferenceMetrics()
        
        # Model cache
        self._pytorch_model = None
        self._mlx_inference = None
        self._model_cache = {}
        
        # Performance tracking
        self.performance_log = []
        
        # Initialize models if paths provided
        if config.pytorch_model_path:
            self._load_pytorch_model(config.pytorch_model_path)
        if config.mlx_model_path:
            self._load_mlx_model(config.mlx_model_path)
    
    def _load_pytorch_model(self, checkpoint_path: str):
        """Load PyTorch model."""
        try:
            logger.info(f"Loading PyTorch model from {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract config
            config = checkpoint.get('config', {})
            self._pytorch_model = IndicatorTransformer(
                num_indicators=config.get('num_indicators', 111),
                d_model=config.get('d_model', 256),
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 6)
            )
            
            self._pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            self._pytorch_model.eval()
            
            # Move to appropriate device
            if torch.cuda.is_available():
                self._pytorch_model = self._pytorch_model.cuda()
            elif torch.backends.mps.is_available():
                self._pytorch_model = self._pytorch_model.to('mps')
                
            logger.info("✅ PyTorch model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            self._pytorch_model = None
    
    def _load_mlx_model(self, model_path: str):
        """Load MLX model."""
        try:
            logger.info(f"Loading MLX model from {model_path}")
            self._mlx_inference = MLXProductionInference(model_path)
            logger.info("✅ MLX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            self._mlx_inference = None
    
    def should_use_mlx(self, num_symbols: int) -> bool:
        """Determine whether to use MLX based on context."""
        if not self.config.use_mlx or self._mlx_inference is None:
            return False
        
        if self._pytorch_model is None:
            return True  # Only MLX available
        
        # Use MLX for larger batches
        if num_symbols >= self.config.mlx_threshold:
            return True
        
        # Use MLX if it's been performing better
        if self.metrics.mlx_inferences > 10 and self.metrics.pytorch_inferences > 10:
            mlx_avg = self.metrics.mlx_time_ms / self.metrics.mlx_inferences
            pytorch_avg = self.metrics.pytorch_time_ms / self.metrics.pytorch_inferences
            if mlx_avg < pytorch_avg * 0.8:  # MLX is 20% faster
                return True
        
        return False
    
    def process_symbols(self, symbols: Union[str, List[str]], 
                       data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> List[TradingSignal]:
        """Process one or more symbols with intelligent framework selection."""
        # Normalize inputs
        if isinstance(symbols, str):
            symbols = [symbols]
            data = {symbols[0]: data}
        elif isinstance(data, pd.DataFrame):
            data = {symbols[0]: data}
        
        num_symbols = len(symbols)
        use_mlx = self.should_use_mlx(num_symbols)
        
        start_time = time.time()
        signals = []
        
        try:
            if use_mlx and self._mlx_inference is not None:
                # Use MLX for inference
                if num_symbols == 1:
                    signals = [self._mlx_inference.process_symbol(symbols[0], data[symbols[0]])]
                else:
                    signals = self._mlx_inference.process_batch(symbols, data)
                framework = "MLX"
                
            elif self._pytorch_model is not None:
                # Fallback to PyTorch
                signals = self._process_with_pytorch(symbols, data)
                framework = "PyTorch"
                
            else:
                raise RuntimeError("No model available for inference")
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            self.metrics.errors += 1
            
            # Try fallback if enabled
            if self.config.fallback_to_pytorch and use_mlx and self._pytorch_model is not None:
                logger.info("Falling back to PyTorch...")
                try:
                    signals = self._process_with_pytorch(symbols, data)
                    framework = "PyTorch (fallback)"
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    return []
            else:
                return []
        
        # Record metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self.metrics.add_inference(use_mlx, elapsed_ms, num_symbols)
        
        if self.config.log_performance:
            self._log_performance(framework, num_symbols, elapsed_ms)
        
        return signals
    
    def _process_with_pytorch(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Process symbols using PyTorch model."""
        signals = []
        
        for symbol in symbols:
            if symbol not in data_dict:
                continue
                
            df = data_dict[symbol]
            
            # Prepare data for PyTorch model
            # This is a simplified version - adapt to your actual preprocessing
            indicator_values = self._compute_indicators_pytorch(df)
            market_context = self._compute_market_context_pytorch(df)
            
            # Model inference
            with torch.no_grad():
                indicator_tensor = torch.FloatTensor(indicator_values).unsqueeze(0)
                context_tensor = torch.FloatTensor(market_context).unsqueeze(0)
                indices_tensor = torch.arange(len(indicator_values)).unsqueeze(0)
                
                if self._pytorch_model.device.type != 'cpu':
                    indicator_tensor = indicator_tensor.to(self._pytorch_model.device)
                    context_tensor = context_tensor.to(self._pytorch_model.device)
                    indices_tensor = indices_tensor.to(self._pytorch_model.device)
                
                outputs = self._pytorch_model(indicator_tensor, context_tensor, indices_tensor)
                
            # Convert to trading signal
            signal = self._pytorch_output_to_signal(symbol, outputs, df)
            signals.append(signal)
        
        return signals
    
    def _compute_indicators_pytorch(self, df: pd.DataFrame) -> np.ndarray:
        """Compute indicators for PyTorch model."""
        # Simplified - implement your actual indicator computation
        indicators = []
        
        # RSI
        if 'close' in df:
            close = df['close'].values
            rsi = self._calculate_rsi(close, 14)
            indicators.append(rsi[-1] if len(rsi) > 0 else 50.0)
        
        # Add more indicators as needed
        # MACD, BB, ATR, etc.
        
        # Pad to expected size
        while len(indicators) < 10:
            indicators.append(0.0)
        
        return np.array(indicators)
    
    def _compute_market_context_pytorch(self, df: pd.DataFrame) -> np.ndarray:
        """Compute market context for PyTorch model."""
        context = []
        
        if 'close' in df and len(df) > 20:
            close = df['close'].values[-20:]
            returns = np.diff(close) / close[:-1]
            
            context.extend([
                np.mean(returns),
                np.std(returns),
                np.min(returns),
                np.max(returns)
            ])
        
        # Pad to expected size
        while len(context) < 10:
            context.append(0.0)
        
        return np.array(context)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return np.array([50.0])
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _pytorch_output_to_signal(self, symbol: str, outputs: Dict, df: pd.DataFrame) -> TradingSignal:
        """Convert PyTorch output to trading signal."""
        # Extract selection probabilities
        selection_probs = outputs['selection_probs'][0].cpu().numpy()
        confidence = float(np.mean(selection_probs))
        
        # Simple signal generation
        rsi = self._calculate_rsi(df['close'].values)[-1] if len(df) > 14 else 50.0
        
        if rsi < 30 and confidence > 0.6:
            action = 'BUY'
        elif rsi > 70 and confidence > 0.6:
            action = 'SELL'  
        else:
            action = 'HOLD'
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            indicators={'RSI': rsi},
            timestamp=datetime.now(),
            processing_time_ms=0  # Set by caller
        )
    
    def _log_performance(self, framework: str, num_symbols: int, elapsed_ms: float):
        """Log performance metrics."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'framework': framework,
            'num_symbols': num_symbols,
            'elapsed_ms': elapsed_ms,
            'throughput': num_symbols / (elapsed_ms / 1000)
        }
        self.performance_log.append(entry)
        
        if len(self.performance_log) % 100 == 0:
            self._save_performance_log()
    
    def _save_performance_log(self):
        """Save performance log to file."""
        log_path = Path("logs") / f"hybrid_inference_{datetime.now().strftime('%Y%m%d')}.json"
        log_path.parent.mkdir(exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
    
    def benchmark_frameworks(self, test_symbols: List[str], test_data: Dict[str, pd.DataFrame]):
        """Benchmark both frameworks on the same data."""
        logger.info("\n" + "="*60)
        logger.info("🏃 Benchmarking Hybrid Inference System")
        logger.info("="*60)
        
        results = {}
        
        # Test PyTorch
        if self._pytorch_model is not None:
            logger.info("\nBenchmarking PyTorch...")
            start = time.time()
            pytorch_signals = self._process_with_pytorch(test_symbols, test_data)
            pytorch_time = (time.time() - start) * 1000
            results['pytorch'] = {
                'time_ms': pytorch_time,
                'signals': len(pytorch_signals),
                'throughput': len(test_symbols) / (pytorch_time / 1000)
            }
        
        # Test MLX
        if self._mlx_inference is not None:
            logger.info("\nBenchmarking MLX...")
            start = time.time()
            mlx_signals = self._mlx_inference.process_batch(test_symbols, test_data)
            mlx_time = (time.time() - start) * 1000
            results['mlx'] = {
                'time_ms': mlx_time,
                'signals': len(mlx_signals),
                'throughput': len(test_symbols) / (mlx_time / 1000)
            }
        
        # Compare results
        logger.info("\n📊 Benchmark Results:")
        logger.info(f"Test symbols: {len(test_symbols)}")
        
        for framework, metrics in results.items():
            logger.info(f"\n{framework.upper()}:")
            logger.info(f"  Time: {metrics['time_ms']:.1f}ms")
            logger.info(f"  Throughput: {metrics['throughput']:.1f} symbols/sec")
        
        if 'pytorch' in results and 'mlx' in results:
            speedup = results['pytorch']['time_ms'] / results['mlx']['time_ms']
            logger.info(f"\n🚀 MLX Speedup: {speedup:.2f}x")
        
        return results
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        summary = self.metrics.get_summary()
        
        # Add framework availability
        summary['frameworks_available'] = {
            'pytorch': self._pytorch_model is not None,
            'mlx': self._mlx_inference is not None
        }
        
        # Add recent performance
        if len(self.performance_log) > 0:
            recent = self.performance_log[-100:]
            summary['recent_performance'] = {
                'avg_throughput': np.mean([e['throughput'] for e in recent]),
                'framework_distribution': pd.Series([e['framework'] for e in recent]).value_counts().to_dict()
            }
        
        return summary


def demo_hybrid_system():
    """Demonstrate the hybrid inference system."""
    logger.info("🚀 Hybrid Inference System Demo")
    logger.info("="*60)
    
    # Configuration
    config = InferenceConfig(
        use_mlx=True,
        mlx_threshold=5,  # Use MLX for 5+ symbols
        fallback_to_pytorch=True,
        log_performance=True
    )
    
    # Initialize system
    system = HybridInferenceSystem(config)
    
    # Generate test data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC', 'GS']
    test_data = {}
    
    for symbol in symbols:
        dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
        test_data[symbol] = pd.DataFrame({
            'open': 100 + np.random.randn(500).cumsum() * 0.1,
            'high': 100.5 + np.random.randn(500).cumsum() * 0.1,
            'low': 99.5 + np.random.randn(500).cumsum() * 0.1,
            'close': 100 + np.random.randn(500).cumsum() * 0.1,
            'volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
    
    # Test single symbol (should use PyTorch)
    logger.info("\n📈 Processing single symbol...")
    signal = system.process_symbols('AAPL', test_data['AAPL'])
    
    # Test batch (should use MLX)
    logger.info("\n📊 Processing batch of symbols...")
    signals = system.process_symbols(symbols, test_data)
    
    # Benchmark
    if system._pytorch_model is not None and system._mlx_inference is not None:
        system.benchmark_frameworks(symbols, test_data)
    
    # Show performance summary
    summary = system.get_performance_summary()
    logger.info("\n📈 Performance Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            logger.info(f"\n{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    demo_hybrid_system()