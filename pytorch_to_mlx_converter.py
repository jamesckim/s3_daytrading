#!/usr/bin/env python3
"""
PyTorch to MLX Model Converter
==============================
Converts trained PyTorch indicator transformer models to MLX format.
"""

import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    num_indicators: int = 111
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 1024
    market_context_size: int = 10
    dropout: float = 0.1
    max_indicators: int = 50


class WeightConverter:
    """Convert PyTorch weights to MLX format with proper shape handling."""
    
    @staticmethod
    def convert_linear(pytorch_weight: torch.Tensor, pytorch_bias: Optional[torch.Tensor] = None) -> Tuple[mx.array, Optional[mx.array]]:
        """Convert linear layer weights."""
        # PyTorch linear: [out_features, in_features]
        # MLX linear: [in_features, out_features]
        weight = mx.array(pytorch_weight.detach().cpu().numpy().T)
        bias = mx.array(pytorch_bias.detach().cpu().numpy()) if pytorch_bias is not None else None
        return weight, bias
    
    @staticmethod
    def convert_layer_norm(pytorch_weight: torch.Tensor, pytorch_bias: torch.Tensor) -> Tuple[mx.array, mx.array]:
        """Convert layer norm parameters."""
        weight = mx.array(pytorch_weight.detach().cpu().numpy())
        bias = mx.array(pytorch_bias.detach().cpu().numpy())
        return weight, bias
    
    @staticmethod
    def convert_embedding(pytorch_weight: torch.Tensor) -> mx.array:
        """Convert embedding weights."""
        return mx.array(pytorch_weight.detach().cpu().numpy())
    
    @staticmethod
    def convert_multihead_attention(state_dict: Dict, prefix: str) -> Dict[str, mx.array]:
        """Convert multi-head attention weights."""
        converted = {}
        
        # Convert Q, K, V projections
        for proj in ['q_proj', 'k_proj', 'v_proj']:
            weight_key = f"{prefix}.{proj}.weight"
            bias_key = f"{prefix}.{proj}.bias"
            
            if weight_key in state_dict:
                weight, bias = WeightConverter.convert_linear(
                    state_dict[weight_key],
                    state_dict.get(bias_key)
                )
                converted[f"{proj}.weight"] = weight
                if bias is not None:
                    converted[f"{proj}.bias"] = bias
        
        # Convert output projection
        out_weight_key = f"{prefix}.o_proj.weight"
        out_bias_key = f"{prefix}.o_proj.bias"
        
        if out_weight_key in state_dict:
            weight, bias = WeightConverter.convert_linear(
                state_dict[out_weight_key],
                state_dict.get(out_bias_key)
            )
            converted["o_proj.weight"] = weight
            if bias is not None:
                converted["o_proj.bias"] = bias
        
        return converted


class PyTorchToMLXConverter:
    """Convert PyTorch indicator transformer to MLX format."""
    
    def __init__(self, config: ModelConfig):
        """Initialize converter with model configuration."""
        self.config = config
        self.conversion_map = {}
        
    def convert_checkpoint(self, checkpoint_path: str, output_path: str):
        """Convert entire PyTorch checkpoint to MLX format."""
        logger.info(f"Loading PyTorch checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        pytorch_state_dict = checkpoint['model_state_dict']
        
        # Convert weights
        mlx_weights = self.convert_state_dict(pytorch_state_dict)
        
        # Save MLX weights
        mlx_checkpoint = {
            'model_state_dict': mlx_weights,
            'config': self.config.__dict__,
            'pytorch_checkpoint': checkpoint_path,
            'conversion_timestamp': str(Path(checkpoint_path).stat().st_mtime)
        }
        
        # Save as npz for MLX
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert mx.array to numpy for saving
        np_weights = {}
        for key, value in mlx_weights.items():
            if isinstance(value, mx.array):
                np_weights[key] = np.array(value)
            else:
                np_weights[key] = value
        
        np.savez(output_path, **np_weights)
        
        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'pytorch_checkpoint': checkpoint_path,
                'weight_shapes': {k: list(v.shape) for k, v in mlx_weights.items()}
            }, f, indent=2)
        
        logger.info(f"Saved MLX model to {output_path}")
        logger.info(f"Saved metadata to {metadata_path}")
        
        return mlx_weights
    
    def convert_state_dict(self, pytorch_state_dict: Dict) -> Dict[str, mx.array]:
        """Convert PyTorch state dict to MLX format."""
        mlx_weights = {}
        
        # Convert all weights generically
        for key, tensor in pytorch_state_dict.items():
            try:
                # Handle different layer types
                if 'weight' in key and 'norm' in key:
                    # Layer normalization
                    weight = mx.array(tensor.detach().cpu().numpy())
                    mlx_weights[key] = weight
                elif 'bias' in key and 'norm' in key:
                    # Layer norm bias
                    bias = mx.array(tensor.detach().cpu().numpy())
                    mlx_weights[key] = bias
                elif 'embedding' in key:
                    # Embeddings
                    mlx_weights[key] = WeightConverter.convert_embedding(tensor)
                elif 'weight' in key:
                    # Linear layers - need to transpose
                    if len(tensor.shape) == 2:
                        weight, _ = WeightConverter.convert_linear(tensor, None)
                        mlx_weights[key] = weight
                    else:
                        # Other weights (conv, etc)
                        mlx_weights[key] = mx.array(tensor.detach().cpu().numpy())
                elif 'bias' in key:
                    # Biases
                    mlx_weights[key] = mx.array(tensor.detach().cpu().numpy())
                else:
                    # Other parameters
                    mlx_weights[key] = mx.array(tensor.detach().cpu().numpy())
                    
            except Exception as e:
                logger.warning(f"Failed to convert {key}: {e}")
                # Try direct conversion as fallback
                mlx_weights[key] = mx.array(tensor.detach().cpu().numpy())
        
        logger.info(f"Converted {len(mlx_weights)} weight tensors")
        
        # Log converted keys for debugging
        logger.debug("Converted keys:")
        for i, key in enumerate(sorted(mlx_weights.keys())):
            if i < 10:
                logger.debug(f"  - {key}: shape {mlx_weights[key].shape}")
            elif i == 10:
                logger.debug(f"  ... and {len(mlx_weights) - 10} more")
                break
        
        return mlx_weights
    
    def validate_conversion(self, pytorch_model, mlx_model, test_batch_size: int = 4):
        """Validate that MLX model produces similar outputs to PyTorch model."""
        logger.info("\nValidating model conversion...")
        
        # Generate test inputs
        indicator_values = torch.randn(test_batch_size, 10)
        market_context = torch.randn(test_batch_size, self.config.market_context_size)
        indicator_indices = torch.randint(0, self.config.num_indicators, (test_batch_size, 10))
        
        # PyTorch forward pass
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_outputs = pytorch_model(indicator_values, market_context, indicator_indices)
        
        # MLX forward pass
        mlx_indicator_values = mx.array(indicator_values.numpy())
        mlx_market_context = mx.array(market_context.numpy())
        mlx_indicator_indices = mx.array(indicator_indices.numpy())
        
        mlx_outputs = mlx_model(mlx_indicator_values, mlx_market_context, mlx_indicator_indices)
        
        # Compare outputs
        pytorch_probs = pytorch_outputs['selection_probs'].numpy()
        mlx_probs = np.array(mlx_outputs['selection_probs'])
        
        max_diff = np.max(np.abs(pytorch_probs - mlx_probs))
        mean_diff = np.mean(np.abs(pytorch_probs - mlx_probs))
        
        logger.info(f"Max output difference: {max_diff:.6f}")
        logger.info(f"Mean output difference: {mean_diff:.6f}")
        
        if max_diff < 0.01:
            logger.info("✅ Conversion validated successfully!")
            return True
        else:
            logger.warning("⚠️  Large differences detected, conversion may have issues")
            return False


def convert_model(checkpoint_path: str, output_dir: str = "mlx_models"):
    """High-level function to convert a PyTorch checkpoint to MLX."""
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config (adjust based on your checkpoint structure)
    config = ModelConfig()
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create converter
    converter = PyTorchToMLXConverter(config)
    
    # Convert checkpoint
    output_path = Path(output_dir) / Path(checkpoint_path).stem / "model.npz"
    mlx_weights = converter.convert_checkpoint(checkpoint_path, str(output_path))
    
    return output_path, config


def main():
    """Demonstrate model conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch model to MLX")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint")
    parser.add_argument("--output-dir", default="mlx_models", help="Output directory")
    parser.add_argument("--validate", action="store_true", help="Validate conversion")
    
    args = parser.parse_args()
    
    logger.info("🔄 PyTorch to MLX Model Converter")
    logger.info("="*60)
    
    # Convert model
    output_path, config = convert_model(args.checkpoint, args.output_dir)
    
    logger.info("\n✅ Conversion complete!")
    logger.info(f"MLX model saved to: {output_path}")
    
    if args.validate:
        # Import both model classes
        from train_indicator_transformer import IndicatorTransformer
        from indicator_transformer_mlx import IndicatorTransformerMLX
        
        # Load models
        pytorch_model = IndicatorTransformer(
            num_indicators=config.num_indicators,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        
        mlx_model = IndicatorTransformerMLX(
            num_indicators=config.num_indicators,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers
        )
        # Load converted weights into MLX model
        mlx_weights = np.load(output_path)
        # Note: You'll need to implement load_weights in your MLX model
        
        converter = PyTorchToMLXConverter(config)
        converter.validate_conversion(pytorch_model, mlx_model)


if __name__ == "__main__":
    main()