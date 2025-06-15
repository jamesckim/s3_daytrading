#!/usr/bin/env python3
"""
ML Signal Discovery and Optimization
===================================
Advanced machine learning for discovering optimal trading signals and weights.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import sys
import os
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_enhanced_signal_strength import MLEnhancedSignalStrength
from day_trading_o3 import DataLoader

# Advanced ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import optuna
import joblib

class MLSignalDiscovery:
    """
    Advanced ML signal discovery system that:
    1. Tests multiple ML algorithms for signal prediction
    2. Discovers optimal feature combinations
    3. Optimizes hyperparameters automatically
    4. Creates ensemble models for robust predictions
    5. Provides interpretable feature importance analysis
    """
    
    def __init__(self, 
                 symbols: List[str] = None,
                 days: int = 90,
                 min_data_points: int = 2000):
        """
        Initialize ML signal discovery system.
        
        Args:
            symbols: List of symbols to analyze
            days: Number of days of historical data
            min_data_points: Minimum data points required for analysis
        """
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        self.days = days
        self.min_data_points = min_data_points
        self.data_loader = DataLoader()
        
        # Date range
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Storage for results
        self.results = {}
        self.best_models = {}
        self.feature_rankings = {}
        self.ensemble_weights = {}
        
        print(f"🔬 ML Signal Discovery Initialized")
        print(f"📅 Period: {self.start_date} to {self.end_date}")
        print(f"🎯 Symbols: {', '.join(self.symbols)}")
    
    def discover_optimal_signals(self, symbol: str) -> Dict:
        """
        Discover optimal signals for a single symbol using comprehensive ML analysis.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Dictionary with discovery results
        """
        print(f"\n🔍 Discovering optimal signals for {symbol}...")
        
        try:
            # Load and prepare data
            df = self.data_loader.get_price_data(symbol, self.start_date, self.end_date)
            
            if df.empty or len(df) < self.min_data_points:
                print(f"❌ Insufficient data for {symbol}")
                return {'symbol': symbol, 'status': 'insufficient_data'}
            
            # Create ML-enhanced strategy to get features
            strategy = MLEnhancedSignalStrength(
                df.copy(),
                enable_ml_signals=True,
                ml_lookback_window=500
            )
            
            # Prepare ML features and targets
            features, targets = self._prepare_comprehensive_features(strategy)
            
            if len(features) < 1000:
                print(f"❌ Insufficient features for {symbol}")
                return {'symbol': symbol, 'status': 'insufficient_features'}
            
            print(f"📊 Prepared {len(features)} samples with {len(features.columns)} features")
            
            # 1. Feature Selection Analysis
            feature_analysis = self._analyze_feature_importance(features, targets)
            
            # 2. Model Selection and Optimization
            model_results = self._optimize_multiple_models(features, targets)
            
            # 3. Ensemble Model Creation
            ensemble_model = self._create_ensemble_model(features, targets, model_results)
            
            # 4. Performance Validation
            validation_results = self._validate_predictions(features, targets, ensemble_model)
            
            # 5. Signal Strength Calibration
            calibration_results = self._calibrate_signal_strength(features, targets, ensemble_model)
            
            result = {
                'symbol': symbol,
                'status': 'success',
                'data_points': len(features),
                'feature_count': len(features.columns),
                'feature_analysis': feature_analysis,
                'model_results': model_results,
                'ensemble_performance': validation_results,
                'signal_calibration': calibration_results,
                'best_features': feature_analysis['top_features'][:20],
                'optimal_model': ensemble_model
            }
            
            self.results[symbol] = result
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}
    
    def _prepare_comprehensive_features(self, strategy: MLEnhancedSignalStrength) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare comprehensive feature set and multiple target variables."""
        
        # Get base features from strategy
        ml_feature_prefixes = [
            'sma_', 'ema_', 'price_vs_', 'bb_', 'high_max_', 'low_min_', 'range_', 'price_position_',
            'vol_sma_', 'vol_ratio_', 'vol_osc_', 'atr_', 'volatility_', 'vol_rank_',
            'roc_', 'rsi_', 'stoch_', 'macd_', 'williams_r_',
            'spread_', 'tick_', 'price_vs_vwap', 'intraday_return', 'overnight_return',
            'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'time_normalized'
        ]
        
        # Additional engineered features
        original_features = [
            'volume_strength', 'magnitude_strength', 'momentum_strength', 
            'regime_favorability', 'candle_strength', 'confluence_score', 'time_favorability'
        ]
        
        # Boolean features
        boolean_features = ['doji', 'hammer', 'shooting_star', 'gap_up', 'gap_down']
        
        # Cumulative features
        cumulative_features = ['vpt', 'obv', 'ad_line']
        
        # Select all relevant features
        feature_cols = []
        for col in strategy.df.columns:
            if (any(col.startswith(prefix) for prefix in ml_feature_prefixes) or 
                col in original_features or col in boolean_features or col in cumulative_features or
                col == 'parkinson_vol'):
                feature_cols.append(col)
        
        # Add interaction features
        strategy.df = self._add_interaction_features(strategy.df, feature_cols[:20])  # Top 20 for interactions
        
        # Update feature columns with interactions
        interaction_cols = [col for col in strategy.df.columns if '_x_' in col]
        feature_cols.extend(interaction_cols)
        
        # Prepare feature matrix
        start_idx = 1000  # Need sufficient lookback
        features_df = strategy.df[feature_cols].iloc[start_idx:].copy()
        
        # Clean features
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Create multiple targets for comprehensive analysis
        targets = self._create_multiple_targets(strategy.df, start_idx)
        
        # Use primary target (future returns)
        primary_target = targets['future_return_5']
        
        return features_df, primary_target
    
    def _add_interaction_features(self, df: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
        """Add interaction features between top indicators."""
        print("  🔗 Adding interaction features...")
        
        # Select top features for interactions (to avoid explosion)
        important_features = base_features[:10]  # Top 10 most important
        
        interaction_count = 0
        for i, feat1 in enumerate(important_features):
            for feat2 in important_features[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    # Multiplicative interaction
                    df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                    interaction_count += 1
                    
                    # Stop if too many interactions
                    if interaction_count >= 20:
                        break
            if interaction_count >= 20:
                break
        
        print(f"    ✅ Added {interaction_count} interaction features")
        return df
    
    def _create_multiple_targets(self, df: pd.DataFrame, start_idx: int) -> Dict[str, pd.Series]:
        """Create multiple target variables for different prediction horizons."""
        targets = {}
        
        # Future returns at different horizons
        for horizon in [1, 3, 5, 10, 20]:
            future_returns = []
            for i in range(start_idx, len(df)):
                if i + horizon < len(df):
                    future_ret = (df['close'].iloc[i + horizon] - df['close'].iloc[i]) / df['close'].iloc[i]
                    future_returns.append(future_ret)
                else:
                    future_returns.append(0)
            
            targets[f'future_return_{horizon}'] = pd.Series(future_returns, 
                                                            index=df.index[start_idx:start_idx+len(future_returns)])
        
        # Volatility prediction
        vol_target = []
        for i in range(start_idx, len(df)):
            if i + 10 < len(df):
                future_vol = df['close'].iloc[i:i+10].pct_change().std()
                vol_target.append(future_vol)
            else:
                vol_target.append(0)
        
        targets['future_volatility'] = pd.Series(vol_target, 
                                                 index=df.index[start_idx:start_idx+len(vol_target)])
        
        return targets
    
    def _analyze_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Comprehensive feature importance analysis using multiple methods."""
        print("  🎯 Analyzing feature importance...")
        
        # Remove any remaining NaN values
        clean_features = features.fillna(0)
        clean_target = target.fillna(0)
        
        importance_results = {}
        
        # 1. Random Forest Feature Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(clean_features, clean_target)
        rf_importance = dict(zip(features.columns, rf.feature_importances_))
        importance_results['random_forest'] = rf_importance
        
        # 2. Gradient Boosting Feature Importance
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(clean_features, clean_target)
        gb_importance = dict(zip(features.columns, gb.feature_importances_))
        importance_results['gradient_boosting'] = gb_importance
        
        # 3. Univariate F-test
        f_scores, _ = f_regression(clean_features, clean_target)
        f_importance = dict(zip(features.columns, f_scores))
        importance_results['f_test'] = f_importance
        
        # 4. Mutual Information
        mi_scores = mutual_info_regression(clean_features, clean_target, random_state=42)
        mi_importance = dict(zip(features.columns, mi_scores))
        importance_results['mutual_info'] = mi_importance
        
        # 5. L1 Regularization (Lasso)
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(StandardScaler().fit_transform(clean_features), clean_target)
        lasso_importance = dict(zip(features.columns, np.abs(lasso.coef_)))
        importance_results['lasso'] = lasso_importance
        
        # Combine importance scores (ensemble ranking)
        combined_scores = {}
        for feature in features.columns:
            # Normalize each score to 0-1 range
            rf_norm = rf_importance[feature] / max(rf_importance.values()) if max(rf_importance.values()) > 0 else 0
            gb_norm = gb_importance[feature] / max(gb_importance.values()) if max(gb_importance.values()) > 0 else 0
            f_norm = f_importance[feature] / max(f_importance.values()) if max(f_importance.values()) > 0 else 0
            mi_norm = mi_importance[feature] / max(mi_importance.values()) if max(mi_importance.values()) > 0 else 0
            lasso_norm = lasso_importance[feature] / max(lasso_importance.values()) if max(lasso_importance.values()) > 0 else 0
            
            # Combined score (weighted average)
            combined_scores[feature] = (0.3 * rf_norm + 0.3 * gb_norm + 0.15 * f_norm + 
                                      0.15 * mi_norm + 0.1 * lasso_norm)
        
        # Get top features
        top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"    ✅ Top 10 features identified:")
        for feature, score in top_features[:10]:
            print(f"      {feature}: {score:.3f}")
        
        return {
            'individual_methods': importance_results,
            'combined_scores': combined_scores,
            'top_features': [feat for feat, score in top_features]
        }
    
    def _optimize_multiple_models(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """Optimize multiple ML models and compare performance."""
        print("  🔧 Optimizing multiple ML models...")
        
        # Prepare data
        clean_features = features.fillna(0)
        clean_target = target.fillna(0)
        
        # Scale features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(clean_features)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        models_to_test = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [5, 10, 20],
                    'min_samples_leaf': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [5, 10, 20]
                }
            },
            'elastic_net': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            }
        }
        
        model_results = {}
        
        for model_name, model_config in models_to_test.items():
            print(f"    🔧 Optimizing {model_name}...")
            
            try:
                # Randomized search for efficiency
                search = RandomizedSearchCV(
                    model_config['model'],
                    model_config['params'],
                    n_iter=20,
                    cv=tscv,
                    scoring='r2',
                    random_state=42,
                    n_jobs=-1
                )
                
                # Fit on scaled features for linear models, original for tree models
                if model_name in ['elastic_net', 'ridge', 'lasso']:
                    search.fit(scaled_features, clean_target)
                else:
                    search.fit(clean_features, clean_target)
                
                # Store results
                model_results[model_name] = {
                    'best_model': search.best_estimator_,
                    'best_score': search.best_score_,
                    'best_params': search.best_params_,
                    'cv_scores': search.cv_results_['mean_test_score']
                }
                
                print(f"      ✅ {model_name}: R² = {search.best_score_:.3f}")
                
            except Exception as e:
                print(f"      ❌ {model_name} failed: {e}")
                model_results[model_name] = {'error': str(e)}
        
        return model_results
    
    def _create_ensemble_model(self, features: pd.DataFrame, target: pd.Series, model_results: Dict) -> Dict:
        """Create ensemble model from best performing individual models."""
        print("  🎭 Creating ensemble model...")
        
        # Get successful models
        successful_models = {name: results for name, results in model_results.items() 
                           if 'best_model' in results and results['best_score'] > 0}
        
        if len(successful_models) < 2:
            print("    ⚠️ Not enough successful models for ensemble")
            return None
        
        # Calculate ensemble weights based on performance
        total_score = sum(results['best_score'] for results in successful_models.values())
        ensemble_weights = {name: results['best_score'] / total_score 
                          for name, results in successful_models.items()}
        
        print(f"    ✅ Ensemble weights:")
        for name, weight in ensemble_weights.items():
            print(f"      {name}: {weight:.3f}")
        
        return {
            'models': {name: results['best_model'] for name, results in successful_models.items()},
            'weights': ensemble_weights,
            'individual_scores': {name: results['best_score'] for name, results in successful_models.items()}
        }
    
    def _validate_predictions(self, features: pd.DataFrame, target: pd.Series, ensemble_model: Dict) -> Dict:
        """Validate ensemble model predictions using time series cross-validation."""
        if ensemble_model is None:
            return {'status': 'no_ensemble'}
        
        print("  ✅ Validating ensemble predictions...")
        
        clean_features = features.fillna(0)
        clean_target = target.fillna(0)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        all_predictions = []
        all_actuals = []
        
        scaler = RobustScaler()
        
        for train_idx, test_idx in tscv.split(clean_features):
            X_train, X_test = clean_features.iloc[train_idx], clean_features.iloc[test_idx]
            y_train, y_test = clean_target.iloc[train_idx], clean_target.iloc[test_idx]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Make ensemble predictions
            ensemble_pred = np.zeros(len(X_test))
            
            for model_name, model in ensemble_model['models'].items():
                weight = ensemble_model['weights'][model_name]
                
                # Use scaled features for linear models
                if model_name in ['elastic_net', 'ridge', 'lasso']:
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                
                ensemble_pred += weight * pred
            
            all_predictions.extend(ensemble_pred)
            all_actuals.extend(y_test)
        
        # Calculate validation metrics
        r2 = r2_score(all_actuals, all_predictions)
        mse = mean_squared_error(all_actuals, all_predictions)
        mae = mean_absolute_error(all_actuals, all_predictions)
        
        # Directional accuracy
        actual_direction = np.sign(all_actuals)
        pred_direction = np.sign(all_predictions)
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        validation_results = {
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'predictions_std': np.std(all_predictions),
            'actuals_std': np.std(all_actuals)
        }
        
        print(f"    ✅ Validation results:")
        print(f"      R² Score: {r2:.3f}")
        print(f"      Directional Accuracy: {directional_accuracy:.3f}")
        print(f"      MAE: {mae:.6f}")
        
        return validation_results
    
    def _calibrate_signal_strength(self, features: pd.DataFrame, target: pd.Series, ensemble_model: Dict) -> Dict:
        """Calibrate ensemble predictions to signal strength scale (0-1)."""
        if ensemble_model is None:
            return {'status': 'no_ensemble'}
        
        print("  🎯 Calibrating signal strength...")
        
        # Make predictions on full dataset
        clean_features = features.fillna(0)
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(clean_features)
        
        ensemble_predictions = np.zeros(len(clean_features))
        
        for model_name, model in ensemble_model['models'].items():
            weight = ensemble_model['weights'][model_name]
            
            if model_name in ['elastic_net', 'ridge', 'lasso']:
                pred = model.predict(scaled_features)
            else:
                pred = model.predict(clean_features)
            
            ensemble_predictions += weight * pred
        
        # Calibration: convert predictions to 0-1 signal strength
        # Use percentile-based calibration
        prediction_percentiles = pd.Series(ensemble_predictions).rank(pct=True)
        
        # Apply sigmoid transformation for smooth calibration
        calibrated_signals = 1 / (1 + np.exp(-5 * (prediction_percentiles - 0.5)))
        
        # Validate calibration quality
        correlation_with_target = np.corrcoef(calibrated_signals, target.fillna(0))[0, 1]
        
        calibration_results = {
            'raw_predictions_range': [float(ensemble_predictions.min()), float(ensemble_predictions.max())],
            'calibrated_signals_range': [float(calibrated_signals.min()), float(calibrated_signals.max())],
            'correlation_with_target': float(correlation_with_target),
            'signal_distribution': {
                'mean': float(calibrated_signals.mean()),
                'std': float(calibrated_signals.std()),
                'percentiles': {
                    '25%': float(np.percentile(calibrated_signals, 25)),
                    '50%': float(np.percentile(calibrated_signals, 50)),
                    '75%': float(np.percentile(calibrated_signals, 75)),
                    '90%': float(np.percentile(calibrated_signals, 90))
                }
            }
        }
        
        print(f"    ✅ Signal correlation with target: {correlation_with_target:.3f}")
        
        return calibration_results
    
    def run_comprehensive_discovery(self) -> Dict:
        """Run comprehensive ML signal discovery across all symbols."""
        print("🚀 Running Comprehensive ML Signal Discovery")
        print("=" * 80)
        
        all_results = {}
        successful_analyses = 0
        
        for i, symbol in enumerate(self.symbols, 1):
            print(f"\n[{i}/{len(self.symbols)}] Analyzing {symbol}...")
            
            result = self.discover_optimal_signals(symbol)
            all_results[symbol] = result
            
            if result.get('status') == 'success':
                successful_analyses += 1
        
        # Generate cross-symbol analysis
        cross_analysis = self._analyze_cross_symbol_patterns(all_results)
        
        comprehensive_results = {
            'individual_results': all_results,
            'successful_analyses': successful_analyses,
            'cross_symbol_analysis': cross_analysis,
            'summary': self._generate_discovery_summary(all_results)
        }
        
        return comprehensive_results
    
    def _analyze_cross_symbol_patterns(self, all_results: Dict) -> Dict:
        """Analyze patterns across multiple symbols."""
        print("\n🔗 Analyzing cross-symbol patterns...")
        
        successful_results = {symbol: result for symbol, result in all_results.items() 
                            if result.get('status') == 'success'}
        
        if len(successful_results) < 2:
            return {'status': 'insufficient_data'}
        
        # Aggregate feature importance across symbols
        aggregated_importance = {}
        for symbol, result in successful_results.items():
            combined_scores = result['feature_analysis']['combined_scores']
            for feature, score in combined_scores.items():
                if feature not in aggregated_importance:
                    aggregated_importance[feature] = []
                aggregated_importance[feature].append(score)
        
        # Calculate mean importance and consistency
        feature_summary = {}
        for feature, scores in aggregated_importance.items():
            feature_summary[feature] = {
                'mean_importance': np.mean(scores),
                'std_importance': np.std(scores),
                'consistency': 1 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0,
                'symbols_count': len(scores)
            }
        
        # Get most consistent and important features
        consistent_features = sorted(feature_summary.items(), 
                                   key=lambda x: x[1]['consistency'] * x[1]['mean_importance'], 
                                   reverse=True)
        
        print(f"✅ Top 10 consistent features across symbols:")
        for feature, stats in consistent_features[:10]:
            print(f"  {feature}: importance={stats['mean_importance']:.3f}, consistency={stats['consistency']:.3f}")
        
        return {
            'feature_summary': feature_summary,
            'top_consistent_features': [feat for feat, stats in consistent_features[:20]],
            'cross_symbol_correlation': self._calculate_cross_symbol_correlations(successful_results)
        }
    
    def _calculate_cross_symbol_correlations(self, successful_results: Dict) -> Dict:
        """Calculate correlations between symbols' model performance."""
        
        # Extract performance metrics
        performance_data = {}
        for symbol, result in successful_results.items():
            if 'ensemble_performance' in result and result['ensemble_performance']:
                perf = result['ensemble_performance']
                performance_data[symbol] = {
                    'r2_score': perf.get('r2_score', 0),
                    'directional_accuracy': perf.get('directional_accuracy', 0),
                    'mae': perf.get('mae', 0)
                }
        
        return {
            'performance_correlations': 'calculated',
            'symbols_analyzed': list(performance_data.keys()),
            'avg_r2': np.mean([data['r2_score'] for data in performance_data.values()]),
            'avg_directional_accuracy': np.mean([data['directional_accuracy'] for data in performance_data.values()])
        }
    
    def _generate_discovery_summary(self, all_results: Dict) -> Dict:
        """Generate summary of ML signal discovery results."""
        
        successful_results = {symbol: result for symbol, result in all_results.items() 
                            if result.get('status') == 'success'}
        
        if not successful_results:
            return {'status': 'no_successful_analyses'}
        
        # Aggregate statistics
        total_features = []
        r2_scores = []
        directional_accuracies = []
        
        for result in successful_results.values():
            total_features.append(result.get('feature_count', 0))
            
            if 'ensemble_performance' in result and result['ensemble_performance']:
                perf = result['ensemble_performance']
                r2_scores.append(perf.get('r2_score', 0))
                directional_accuracies.append(perf.get('directional_accuracy', 0))
        
        summary = {
            'successful_symbols': len(successful_results),
            'total_symbols': len(all_results),
            'success_rate': len(successful_results) / len(all_results) * 100,
            'avg_features_per_symbol': np.mean(total_features) if total_features else 0,
            'avg_r2_score': np.mean(r2_scores) if r2_scores else 0,
            'avg_directional_accuracy': np.mean(directional_accuracies) if directional_accuracies else 0,
            'performance_consistency': {
                'r2_std': np.std(r2_scores) if r2_scores else 0,
                'directional_std': np.std(directional_accuracies) if directional_accuracies else 0
            }
        }
        
        return summary

def run_ml_signal_discovery():
    """Run comprehensive ML signal discovery analysis."""
    print("🔬 COMPREHENSIVE ML SIGNAL DISCOVERY")
    print("=" * 80)
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    discovery = MLSignalDiscovery(symbols=test_symbols, days=90)
    results = discovery.run_comprehensive_discovery()
    
    # Print summary
    print("\n" + "="*80)
    print("🎊 ML SIGNAL DISCOVERY COMPLETE")
    print("="*80)
    
    summary = results['summary']
    print(f"Successful Analyses: {summary['successful_symbols']}/{summary['total_symbols']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Average R² Score: {summary['avg_r2_score']:.3f}")
    print(f"Average Directional Accuracy: {summary['avg_directional_accuracy']:.3f}")
    
    # Cross-symbol insights
    if 'cross_symbol_analysis' in results:
        cross_analysis = results['cross_symbol_analysis']
        if 'top_consistent_features' in cross_analysis:
            print(f"\n🏆 Top Consistent Features Across Symbols:")
            for feature in cross_analysis['top_consistent_features'][:10]:
                print(f"  • {feature}")
    
    return discovery, results

if __name__ == "__main__":
    discovery, results = run_ml_signal_discovery()