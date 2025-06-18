#!/usr/bin/env python3
"""
Diagnose AI Model Training Issues
=================================
Comprehensive diagnostic tool to determine if the AI indicator selection
model was trained properly or if it's producing invalid outputs.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicator_transformer import AIIndicatorSelector
from s3_ai_wrapper import S3AIWrapper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModelDiagnostics:
    """Comprehensive diagnostics for AI model training issues."""
    
    def __init__(self):
        self.results = {}
        self.issues = []
        
    def run_full_diagnostics(self):
        """Run comprehensive model diagnostics."""
        print("🔍 AI MODEL TRAINING DIAGNOSTICS")
        print("="*80)
        
        # 1. Check model file existence and properties
        print("\n1️⃣ Checking model files...")
        self.check_model_files()
        
        # 2. Load and inspect model architecture
        print("\n2️⃣ Loading model and checking architecture...")
        self.check_model_architecture()
        
        # 3. Analyze model weights and parameters
        print("\n3️⃣ Analyzing model parameters...")
        self.analyze_model_parameters()
        
        # 4. Test model outputs
        print("\n4️⃣ Testing model outputs...")
        self.test_model_outputs()
        
        # 5. Check for training artifacts
        print("\n5️⃣ Looking for training artifacts...")
        self.check_training_artifacts()
        
        # 6. Behavioral analysis
        print("\n6️⃣ Analyzing model behavior...")
        self.analyze_model_behavior()
        
        # 7. Summary and recommendations
        print("\n7️⃣ DIAGNOSTIC SUMMARY")
        self.print_diagnostic_summary()
        
    def check_model_files(self):
        """Check for model files and their properties."""
        model_paths = [
            'models/indicator_transformer.pth',
            'models/indicator_transformer_best.pth',
            'indicator_transformer_model.pth',
            '../models/indicator_transformer.pth'
        ]
        
        found_models = []
        for path in model_paths:
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)  # MB
                modified = datetime.fromtimestamp(os.path.getmtime(path))
                found_models.append({
                    'path': path,
                    'size_mb': size,
                    'modified': modified
                })
                print(f"   ✓ Found: {path} ({size:.2f}MB, modified: {modified})")
        
        if not found_models:
            print("   ❌ No model files found!")
            self.issues.append("NO_MODEL_FILE: No trained model file exists")
        else:
            # Check model size
            for model in found_models:
                if model['size_mb'] < 0.1:  # Less than 100KB
                    self.issues.append(f"TINY_MODEL: Model file {model['path']} is suspiciously small ({model['size_mb']:.2f}MB)")
                elif model['size_mb'] > 1000:  # More than 1GB
                    self.issues.append(f"HUGE_MODEL: Model file {model['path']} is suspiciously large ({model['size_mb']:.2f}MB)")
        
        self.results['model_files'] = found_models
        
    def check_model_architecture(self):
        """Load model and inspect architecture."""
        try:
            # Try to load the model
            selector = AIIndicatorSelector()
            model = selector.model
            
            # Check model structure
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   ✓ Model loaded successfully")
            print(f"   • Total parameters: {total_params:,}")
            print(f"   • Trainable parameters: {trainable_params:,}")
            print(f"   • Model type: {type(model).__name__}")
            
            # Check if model is in training or eval mode
            print(f"   • Training mode: {model.training}")
            
            # Check for frozen parameters
            frozen_params = total_params - trainable_params
            if frozen_params == total_params:
                self.issues.append("FROZEN_MODEL: All parameters are frozen (requires_grad=False)")
            
            self.results['architecture'] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_type': type(model).__name__
            }
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            self.issues.append(f"LOAD_ERROR: Cannot load model - {str(e)}")
            
    def analyze_model_parameters(self):
        """Analyze model weights and biases."""
        try:
            selector = AIIndicatorSelector()
            model = selector.model
            
            weight_stats = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    stats = {
                        'layer': name,
                        'shape': list(param.shape),
                        'mean': float(param.data.mean()),
                        'std': float(param.data.std()),
                        'min': float(param.data.min()),
                        'max': float(param.data.max()),
                        'zeros': int((param.data == 0).sum()),
                        'inf_nan': int(torch.isnan(param.data).sum() + torch.isinf(param.data).sum())
                    }
                    weight_stats.append(stats)
                    
                    # Check for issues
                    if stats['inf_nan'] > 0:
                        self.issues.append(f"NAN_INF_WEIGHTS: Layer {name} contains NaN or Inf values")
                    
                    if stats['std'] < 1e-6:
                        self.issues.append(f"DEAD_LAYER: Layer {name} has near-zero variance (std={stats['std']:.6f})")
                    
                    if abs(stats['mean']) > 10:
                        self.issues.append(f"EXTREME_WEIGHTS: Layer {name} has extreme mean ({stats['mean']:.2f})")
            
            # Summary statistics
            all_stds = [s['std'] for s in weight_stats]
            print(f"   • Average weight std: {np.mean(all_stds):.4f}")
            print(f"   • Min layer std: {min(all_stds):.6f}")
            print(f"   • Max layer std: {max(all_stds):.4f}")
            
            # Check for uniformity (sign of no training)
            if max(all_stds) < 0.01:
                self.issues.append("NO_TRAINING: All weights have extremely low variance - model likely untrained")
            
            self.results['weight_stats'] = weight_stats
            
        except Exception as e:
            print(f"   ❌ Failed to analyze parameters: {e}")
            
    def test_model_outputs(self):
        """Test model outputs with various inputs."""
        try:
            selector = AIIndicatorSelector()
            model = selector.model
            model.eval()
            
            # Create test inputs
            batch_size = 10
            num_indicators = selector.num_indicators
            context_size = 64  # Typical context size
            
            test_cases = {
                'random': torch.randn(batch_size, num_indicators),
                'zeros': torch.zeros(batch_size, num_indicators),
                'ones': torch.ones(batch_size, num_indicators),
                'extreme_high': torch.ones(batch_size, num_indicators) * 100,
                'extreme_low': torch.ones(batch_size, num_indicators) * -100
            }
            
            output_stats = {}
            
            for test_name, indicator_tensor in test_cases.items():
                context_tensor = torch.randn(batch_size, context_size)
                indices_tensor = torch.arange(num_indicators).repeat(batch_size, 1)
                
                with torch.no_grad():
                    outputs = model(
                        indicator_tensor.to(selector.device),
                        context_tensor.to(selector.device),
                        indices_tensor.to(selector.device)
                    )
                
                probs = outputs['selection_probs'].cpu().numpy()
                weights = outputs['indicator_weights'].cpu().numpy()
                
                stats = {
                    'prob_mean': float(probs.mean()),
                    'prob_std': float(probs.std()),
                    'prob_min': float(probs.min()),
                    'prob_max': float(probs.max()),
                    'weight_mean': float(weights.mean()),
                    'weight_std': float(weights.std()),
                    'all_same': bool(probs.std() < 1e-6)
                }
                
                output_stats[test_name] = stats
                
                print(f"\n   Test: {test_name}")
                print(f"   • Prob range: [{stats['prob_min']:.6f}, {stats['prob_max']:.6f}]")
                print(f"   • Prob mean±std: {stats['prob_mean']:.6f} ± {stats['prob_std']:.6f}")
                
                # Check for issues
                if stats['all_same']:
                    self.issues.append(f"UNIFORM_OUTPUT_{test_name}: Model produces identical outputs for {test_name} input")
                
                if stats['prob_max'] < 0.01:
                    self.issues.append(f"LOW_PROBS_{test_name}: All probabilities < 0.01 for {test_name} input")
                
                if stats['prob_std'] < 1e-4:
                    self.issues.append(f"NO_DISCRIMINATION_{test_name}: Model doesn't discriminate between indicators")
            
            # Check if outputs vary with different inputs
            prob_means = [stats['prob_mean'] for stats in output_stats.values()]
            if np.std(prob_means) < 1e-4:
                self.issues.append("STATIC_MODEL: Model produces same outputs regardless of input")
            
            self.results['output_tests'] = output_stats
            
        except Exception as e:
            print(f"   ❌ Failed to test outputs: {e}")
            self.issues.append(f"OUTPUT_TEST_ERROR: {str(e)}")
            
    def check_training_artifacts(self):
        """Look for training logs and checkpoints."""
        artifacts = {
            'training_logs': [],
            'checkpoints': [],
            'config_files': []
        }
        
        # Common locations for training artifacts
        search_dirs = ['.', 'logs', 'models', 'checkpoints', 'runs', '../']
        
        for dir_path in search_dirs:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if 'train' in file.lower() and file.endswith(('.log', '.txt')):
                        artifacts['training_logs'].append(os.path.join(dir_path, file))
                    elif file.endswith('.pth') and 'checkpoint' in file.lower():
                        artifacts['checkpoints'].append(os.path.join(dir_path, file))
                    elif file.endswith('.json') and any(x in file.lower() for x in ['config', 'params', 'hyperparams']):
                        artifacts['config_files'].append(os.path.join(dir_path, file))
        
        # Report findings
        print(f"   • Training logs found: {len(artifacts['training_logs'])}")
        print(f"   • Checkpoints found: {len(artifacts['checkpoints'])}")
        print(f"   • Config files found: {len(artifacts['config_files'])}")
        
        if not artifacts['training_logs']:
            self.issues.append("NO_TRAINING_LOGS: No training logs found - model may be untrained")
        
        # Try to read training metrics
        for log_file in artifacts['training_logs'][:1]:  # Check first log
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if 'loss' in content.lower():
                        print(f"   ✓ Found training metrics in {log_file}")
                    else:
                        self.issues.append("NO_LOSS_INFO: Training log contains no loss information")
            except:
                pass
        
        self.results['artifacts'] = artifacts
        
    def analyze_model_behavior(self):
        """Analyze how model behaves with real market data."""
        try:
            wrapper = S3AIWrapper()
            
            # Create realistic market scenarios
            scenarios = {
                'bull_trend': self._create_bull_market_data(),
                'bear_trend': self._create_bear_market_data(),
                'sideways': self._create_sideways_data(),
                'volatile': self._create_volatile_data()
            }
            
            behavior_analysis = {}
            
            for scenario_name, data in scenarios.items():
                result = wrapper.select_indicators(data)
                selected = result.get('selected_indicators', {})
                
                analysis = {
                    'num_selected': len(selected),
                    'indicators': list(selected.keys()),
                    'avg_confidence': np.mean([v['selection_prob'] for v in selected.values()]) if selected else 0,
                    'method': result.get('selection_method', 'unknown')
                }
                
                behavior_analysis[scenario_name] = analysis
                
                print(f"\n   Scenario: {scenario_name}")
                print(f"   • Indicators selected: {analysis['num_selected']}")
                print(f"   • Avg confidence: {analysis['avg_confidence']:.6f}")
                print(f"   • Method: {analysis['method']}")
            
            # Check if model adapts to different scenarios
            all_indicators = [set(a['indicators']) for a in behavior_analysis.values()]
            if len(all_indicators) > 1:
                common_indicators = set.intersection(*all_indicators)
                if len(common_indicators) == len(all_indicators[0]):
                    self.issues.append("NO_ADAPTATION: Model selects same indicators for all market conditions")
            
            self.results['behavior'] = behavior_analysis
            
        except Exception as e:
            print(f"   ❌ Failed behavior analysis: {e}")
            
    def _create_bull_market_data(self):
        """Create bullish market data."""
        dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
        trend = np.linspace(100, 110, 500) + np.random.randn(500) * 0.1
        return pd.DataFrame({
            'open': trend - 0.1,
            'high': trend + 0.2,
            'low': trend - 0.2,
            'close': trend,
            'volume': np.random.randint(1000, 5000, 500)
        }, index=dates)
        
    def _create_bear_market_data(self):
        """Create bearish market data."""
        dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
        trend = np.linspace(110, 100, 500) + np.random.randn(500) * 0.1
        return pd.DataFrame({
            'open': trend + 0.1,
            'high': trend + 0.2,
            'low': trend - 0.2,
            'close': trend,
            'volume': np.random.randint(1000, 5000, 500)
        }, index=dates)
        
    def _create_sideways_data(self):
        """Create sideways market data."""
        dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
        price = 105 + np.sin(np.linspace(0, 10*np.pi, 500)) * 2 + np.random.randn(500) * 0.1
        return pd.DataFrame({
            'open': price - 0.05,
            'high': price + 0.1,
            'low': price - 0.1,
            'close': price,
            'volume': np.random.randint(1000, 5000, 500)
        }, index=dates)
        
    def _create_volatile_data(self):
        """Create volatile market data."""
        dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
        price = 105 + np.random.randn(500).cumsum() * 0.5
        return pd.DataFrame({
            'open': price,
            'high': price + abs(np.random.randn(500)) * 0.5,
            'low': price - abs(np.random.randn(500)) * 0.5,
            'close': price + np.random.randn(500) * 0.2,
            'volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
        
    def print_diagnostic_summary(self):
        """Print comprehensive diagnostic summary."""
        print("="*80)
        
        if not self.issues:
            print("✅ NO CRITICAL ISSUES FOUND")
            print("The model appears to be properly trained.")
        else:
            print(f"❌ FOUND {len(self.issues)} ISSUES:")
            
            # Categorize issues
            critical_issues = [i for i in self.issues if any(x in i for x in ['NO_MODEL', 'LOAD_ERROR', 'NO_TRAINING', 'NAN_INF'])]
            major_issues = [i for i in self.issues if any(x in i for x in ['LOW_PROBS', 'STATIC_MODEL', 'NO_ADAPTATION', 'UNIFORM_OUTPUT'])]
            minor_issues = [i for i in self.issues if i not in critical_issues + major_issues]
            
            if critical_issues:
                print("\n🔴 CRITICAL ISSUES (Model unusable):")
                for issue in critical_issues:
                    print(f"   • {issue}")
                    
            if major_issues:
                print("\n🟠 MAJOR ISSUES (Model poorly trained):")
                for issue in major_issues:
                    print(f"   • {issue}")
                    
            if minor_issues:
                print("\n🟡 MINOR ISSUES:")
                for issue in minor_issues:
                    print(f"   • {issue}")
        
        # Diagnosis
        print("\n📊 DIAGNOSIS:")
        if 'NO_MODEL_FILE' in str(self.issues) or 'LOAD_ERROR' in str(self.issues):
            print("   ❌ Model file is missing or corrupted")
            print("   → Need to train a new model or obtain a pre-trained one")
        elif 'NO_TRAINING' in str(self.issues) or 'LOW_PROBS' in str(self.issues):
            print("   ❌ Model exists but was never properly trained")
            print("   → Model outputs near-zero probabilities for all indicators")
            print("   → This is why you see '0 indicators selected' constantly")
        elif 'STATIC_MODEL' in str(self.issues):
            print("   ❌ Model is not responding to different inputs")
            print("   → Likely trained with insufficient data or poor hyperparameters")
        elif 'NO_ADAPTATION' in str(self.issues):
            print("   ⚠️ Model is trained but doesn't adapt to market conditions")
            print("   → May need retraining with more diverse data")
        else:
            print("   ✅ Model appears to be properly trained")
            print("   → Issues may be with integration or configuration")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if any('NO_MODEL' in str(i) or 'NO_TRAINING' in str(i) for i in self.issues):
            print("   1. Train a new model using train_indicator_transformer.py")
            print("   2. Or download a pre-trained model")
            print("   3. Ensure training runs for sufficient epochs (>50)")
            print("   4. Monitor training loss - should decrease over time")
        elif 'LOW_PROBS' in str(self.issues):
            print("   1. The current fix (removing threshold) is appropriate")
            print("   2. Consider retraining with output probability calibration")
            print("   3. Use temperature scaling in the model")
        
        # Save full report
        report_file = f'model_diagnostic_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump({
                'issues': self.issues,
                'results': self.results,
                'timestamp': str(datetime.now())
            }, f, indent=2, default=str)
        print(f"\n💾 Full report saved to: {report_file}")


def main():
    """Run comprehensive model diagnostics."""
    diagnostics = AIModelDiagnostics()
    diagnostics.run_full_diagnostics()


if __name__ == "__main__":
    main()