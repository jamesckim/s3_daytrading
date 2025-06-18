#!/usr/bin/env python3
"""
Compare Indicator Performance
=============================
Compare backtest performance with original vs enhanced indicators.
"""

import sys
import os
import json
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_backtest_with_indicators(indicator_set, test_symbols, days=30):
    """Run backtest with specified indicator set."""
    from s3_ai_minute_strategy_v2 import EnhancedS3AIDBBacktester
    
    results = []
    backtester = EnhancedS3AIDBBacktester(initial_capital=100000)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for symbol in test_symbols:
        result = backtester.backtest_symbol(symbol, start_date, end_date)
        if not result.error:
            results.append({
                'Symbol': symbol,
                'Return %': result.total_return_pct,
                'Trades': result.total_trades,
                'Win Rate %': result.win_rate,
                'Sharpe': result.sharpe_ratio,
                'Max DD %': result.max_drawdown_pct
            })
    
    return results

def modify_indicator_library(use_enhanced=True):
    """Temporarily modify the indicator library."""
    import indicator_transformer
    
    # Save original build function
    original_build = indicator_transformer.IndicatorLibrary._build_indicator_library
    
    if not use_enhanced:
        # Create a modified version with only original indicators
        def limited_build(self):
            indicators = {}
            
            # Original periods only
            for period in [10, 20, 50, 200]:
                indicators[f'SMA_{period}'] = indicator_transformer.IndicatorConfig(
                    name=f'SMA_{period}',
                    category='trend',
                    params={'period': period},
                    compute_function='sma',
                    lookback_required=period,
                    normalization='none'
                )
                indicators[f'EMA_{period}'] = indicator_transformer.IndicatorConfig(
                    name=f'EMA_{period}',
                    category='trend',
                    params={'period': period},
                    compute_function='ema',
                    lookback_required=period * 2,
                    normalization='none'
                )
            
            # Add other indicators (RSI, MACD, etc.)
            indicators['RSI_14'] = indicator_transformer.IndicatorConfig(
                name='RSI_14',
                category='momentum',
                params={'period': 14},
                compute_function='rsi',
                lookback_required=28,
                normalization='none'
            )
            
            indicators['MACD'] = indicator_transformer.IndicatorConfig(
                name='MACD',
                category='momentum',
                params={'fast': 12, 'slow': 26, 'signal': 9},
                compute_function='macd',
                lookback_required=52,
                normalization='standard'
            )
            
            indicators['ATR_14'] = indicator_transformer.IndicatorConfig(
                name='ATR_14',
                category='volatility',
                params={'period': 14},
                compute_function='atr',
                lookback_required=28,
                normalization='standard'
            )
            
            # Add remaining standard indicators
            for period in [20]:
                indicators[f'BB_{period}'] = indicator_transformer.IndicatorConfig(
                    name=f'BB_{period}',
                    category='volatility',
                    params={'period': period, 'std_dev': 2},
                    compute_function='bollinger_bands',
                    lookback_required=period * 2,
                    normalization='none'
                )
            
            indicators['OBV'] = indicator_transformer.IndicatorConfig(
                name='OBV',
                category='volume',
                params={},
                compute_function='obv',
                lookback_required=50,
                normalization='standard'
            )
            
            indicators['VWAP'] = indicator_transformer.IndicatorConfig(
                name='VWAP',
                category='volume',
                params={},
                compute_function='vwap',
                lookback_required=1,
                normalization='none'
            )
            
            indicators['MFI_14'] = indicator_transformer.IndicatorConfig(
                name='MFI_14',
                category='volume',
                params={'period': 14},
                compute_function='mfi',
                lookback_required=28,
                normalization='none'
            )
            
            for period in [14]:
                indicators[f'STOCH_{period}'] = indicator_transformer.IndicatorConfig(
                    name=f'STOCH_{period}',
                    category='momentum',
                    params={'period': period, 'smooth_k': 3, 'smooth_d': 3},
                    compute_function='stochastic',
                    lookback_required=period * 2,
                    normalization='none'
                )
                
                indicators[f'CCI_{period}'] = indicator_transformer.IndicatorConfig(
                    name=f'CCI_{period}',
                    category='momentum',
                    params={'period': period},
                    compute_function='cci',
                    lookback_required=period * 2,
                    normalization='standard'
                )
                
                indicators[f'ROC_{period}'] = indicator_transformer.IndicatorConfig(
                    name=f'ROC_{period}',
                    category='momentum',
                    params={'period': period},
                    compute_function='roc',
                    lookback_required=period + 1,
                    normalization='standard'
                )
                
                indicators[f'CMF_{period}'] = indicator_transformer.IndicatorConfig(
                    name=f'CMF_{period}',
                    category='volume',
                    params={'period': period},
                    compute_function='cmf',
                    lookback_required=period * 2,
                    normalization='none'
                )
            
            indicators['ADX_14'] = indicator_transformer.IndicatorConfig(
                name='ADX_14',
                category='trend',
                params={'period': 14},
                compute_function='adx',
                lookback_required=28,
                normalization='none'
            )
            
            indicators['PSAR'] = indicator_transformer.IndicatorConfig(
                name='PSAR',
                category='trend',
                params={'acceleration': 0.02, 'maximum': 0.2},
                compute_function='psar',
                lookback_required=50,
                normalization='none'
            )
            
            indicators['KC_20'] = indicator_transformer.IndicatorConfig(
                name='KC_20',
                category='volatility',
                params={'period': 20, 'multiplier': 1.5},
                compute_function='keltner_channel',
                lookback_required=40,
                normalization='none'
            )
            
            indicators['DC_20'] = indicator_transformer.IndicatorConfig(
                name='DC_20',
                category='volatility',
                params={'period': 20},
                compute_function='donchian_channel',
                lookback_required=20,
                normalization='none'
            )
            
            indicators['PIVOT'] = indicator_transformer.IndicatorConfig(
                name='PIVOT',
                category='market_structure',
                params={},
                compute_function='pivot_points',
                lookback_required=1,
                normalization='none'
            )
            
            indicators['SUPPORT'] = indicator_transformer.IndicatorConfig(
                name='SUPPORT',
                category='market_structure',
                params={'lookback': 50},
                compute_function='support_resistance',
                lookback_required=50,
                normalization='none'
            )
            
            indicators['RESISTANCE'] = indicator_transformer.IndicatorConfig(
                name='RESISTANCE',
                category='market_structure',
                params={'lookback': 50},
                compute_function='support_resistance',
                lookback_required=50,
                normalization='none'
            )
            
            return indicators
        
        # Replace the build function
        indicator_transformer.IndicatorLibrary._build_indicator_library = limited_build
    
    return original_build

def main():
    """Compare performance with and without enhanced indicators."""
    print("Comparing Indicator Performance")
    print("=" * 60)
    
    # Test symbols - mix of different types
    test_symbols = ["AAPL", "TSLA", "SPY", "AAOI", "META", "XOM", "NVDA", "AMD"]
    
    # First, run with original indicators only
    print("\n1. Testing with ORIGINAL indicators (4 EMA periods)...")
    print("-" * 60)
    
    original_build = modify_indicator_library(use_enhanced=False)
    original_results = run_backtest_with_indicators("original", test_symbols)
    
    # Restore and run with enhanced indicators
    import indicator_transformer
    indicator_transformer.IndicatorLibrary._build_indicator_library = original_build
    
    print("\n2. Testing with ENHANCED indicators (14 EMA periods)...")
    print("-" * 60)
    
    enhanced_results = run_backtest_with_indicators("enhanced", test_symbols)
    
    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Create comparison dataframe
    comparison_data = []
    
    for i, symbol in enumerate(test_symbols):
        orig = next((r for r in original_results if r['Symbol'] == symbol), None)
        enh = next((r for r in enhanced_results if r['Symbol'] == symbol), None)
        
        if orig and enh:
            comparison_data.append({
                'Symbol': symbol,
                'Original Return %': orig['Return %'],
                'Enhanced Return %': enh['Return %'],
                'Improvement %': enh['Return %'] - orig['Return %'],
                'Original Trades': orig['Trades'],
                'Enhanced Trades': enh['Trades'],
                'Trade Diff': enh['Trades'] - orig['Trades'],
                'Original Win %': orig['Win Rate %'],
                'Enhanced Win %': enh['Win Rate %'],
                'Win % Diff': enh['Win Rate %'] - orig['Win Rate %']
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\nDetailed Comparison:")
        print(tabulate(df, headers='keys', floatfmt='.2f', tablefmt='simple'))
        
        # Summary statistics
        print("\n" + "-" * 60)
        print("SUMMARY STATISTICS:")
        print(f"Average Return Improvement: {df['Improvement %'].mean():.2f}%")
        print(f"Symbols with Better Returns: {len(df[df['Improvement %'] > 0])} / {len(df)}")
        print(f"Average Trade Count Change: {df['Trade Diff'].mean():.1f}")
        print(f"Average Win Rate Change: {df['Win % Diff'].mean():.2f}%")
        
        # Best improvements
        best_improvement = df.loc[df['Improvement %'].idxmax()]
        worst_decline = df.loc[df['Improvement %'].idxmin()]
        
        print(f"\nBest Improvement: {best_improvement['Symbol']} "
              f"({best_improvement['Improvement %']:.2f}% better)")
        print(f"Worst Decline: {worst_decline['Symbol']} "
              f"({worst_decline['Improvement %']:.2f}% worse)")
        
        # Overall portfolio comparison
        orig_portfolio_return = df['Original Return %'].mean()
        enh_portfolio_return = df['Enhanced Return %'].mean()
        
        print(f"\nPortfolio Returns (equal weight):")
        print(f"Original Indicators: {orig_portfolio_return:.2f}%")
        print(f"Enhanced Indicators: {enh_portfolio_return:.2f}%")
        print(f"Improvement: {enh_portfolio_return - orig_portfolio_return:.2f}%")
        
    print("\n" + "=" * 60)
    print("Comparison complete!")


if __name__ == "__main__":
    main()