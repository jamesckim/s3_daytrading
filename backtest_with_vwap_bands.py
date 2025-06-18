#!/usr/bin/env python3
"""
Comprehensive Backtest with VWAP Bands
======================================
Run backtests comparing performance with and without VWAP bands to measure
the impact of the enhanced indicator set.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_db_backtest_all_tickers import run_backtest
from s3_ai_wrapper import S3AIWrapper
from indicator_transformer import IndicatorLibrary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VWAPBacktestComparison:
    """Compare backtest results with and without VWAP bands."""
    
    def __init__(self):
        self.library = IndicatorLibrary()
        self.results_dir = Path('backtest_results')
        self.results_dir.mkdir(exist_ok=True)
        
    def get_test_symbols(self) -> List[str]:
        """Get symbols for testing."""
        # Try to load from james_tickers.json
        try:
            with open('../james_tickers.json', 'r') as f:
                tickers_data = json.load(f)
            if isinstance(tickers_data, dict):
                symbols = list(tickers_data.keys())[:5]  # Top 5 for testing
            else:
                symbols = tickers_data[:5]
        except:
            # Fallback symbols
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        return symbols
    
    def disable_vwap_bands(self):
        """Temporarily disable VWAP band indicators."""
        disabled = []
        for name in list(self.library.indicators.keys()):
            if 'VWAP' in name and name.endswith(('_U1', '_U2', '_U3', '_L1', '_L2', '_L3')):
                del self.library.indicators[name]
                disabled.append(name)
        return disabled
    
    def enable_vwap_bands(self, disabled_list):
        """Re-enable VWAP band indicators."""
        # Reinitialize library to restore all indicators
        self.library = IndicatorLibrary()
    
    def run_single_backtest(self, symbol: str, start_date: datetime, 
                          end_date: datetime, with_vwap: bool) -> Dict:
        """Run backtest for a single symbol."""
        try:
            logger.info(f"Running backtest for {symbol} (VWAP bands: {with_vwap})")
            
            # Run backtest
            results = run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000,
                position_size=0.1,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                max_positions=3,
                commission=0.001
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            return None
    
    def compare_results(self, results_without: Dict, results_with: Dict) -> Dict:
        """Compare backtest results."""
        if not results_without or not results_with:
            return None
        
        comparison = {
            'total_return_without': results_without.get('total_return', 0),
            'total_return_with': results_with.get('total_return', 0),
            'return_improvement': results_with.get('total_return', 0) - results_without.get('total_return', 0),
            
            'sharpe_without': results_without.get('sharpe_ratio', 0),
            'sharpe_with': results_with.get('sharpe_ratio', 0),
            'sharpe_improvement': results_with.get('sharpe_ratio', 0) - results_without.get('sharpe_ratio', 0),
            
            'win_rate_without': results_without.get('win_rate', 0),
            'win_rate_with': results_with.get('win_rate', 0),
            'win_rate_improvement': results_with.get('win_rate', 0) - results_without.get('win_rate', 0),
            
            'max_drawdown_without': results_without.get('max_drawdown', 0),
            'max_drawdown_with': results_with.get('max_drawdown', 0),
            'drawdown_improvement': results_with.get('max_drawdown', 0) - results_without.get('max_drawdown', 0),
            
            'trades_without': results_without.get('total_trades', 0),
            'trades_with': results_with.get('total_trades', 0),
        }
        
        return comparison
    
    def analyze_vwap_indicator_usage(self, symbol: str, start_date: datetime, 
                                   end_date: datetime) -> Dict:
        """Analyze which VWAP indicators are selected most frequently."""
        logger.info(f"Analyzing VWAP indicator usage for {symbol}")
        
        # Initialize wrapper with all indicators
        wrapper = S3AIWrapper()
        
        # Track indicator selections
        vwap_selections = {}
        total_selections = 0
        
        try:
            # Simulate multiple trading days
            from s3_ai_db_backtest_all_tickers import DatabaseDataProvider
            db_provider = DatabaseDataProvider()
            
            data = db_provider.get_minute_data(symbol, start_date, end_date)
            
            if len(data) > 500:
                # Sample every 30 minutes
                for i in range(100, len(data), 30):
                    if i + 100 <= len(data):
                        window_data = data.iloc[i-100:i]
                        
                        # Get AI selection
                        selection = wrapper.select_indicators(window_data)
                        
                        # Track VWAP selections
                        for ind_name in selection['selected_indicators']:
                            if 'VWAP' in ind_name:
                                vwap_selections[ind_name] = vwap_selections.get(ind_name, 0) + 1
                            total_selections += 1
                
        except Exception as e:
            logger.error(f"Error analyzing indicator usage: {e}")
        
        # Calculate selection frequencies
        vwap_frequencies = {
            ind: count / max(total_selections, 1) 
            for ind, count in vwap_selections.items()
        }
        
        return vwap_frequencies
    
    def run_comprehensive_comparison(self):
        """Run comprehensive backtest comparison."""
        print("\n🔄 Comprehensive Backtest: With vs Without VWAP Bands")
        print("=" * 80)
        
        # Test parameters
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 1 month backtest
        symbols = self.get_test_symbols()
        
        print(f"\n📊 Test Configuration:")
        print(f"   Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Initial Capital: $100,000")
        
        # Results storage
        all_comparisons = []
        vwap_usage_stats = {}
        
        # Run backtests
        for symbol in symbols:
            print(f"\n{'='*60}")
            print(f"Testing {symbol}...")
            
            # 1. Backtest WITHOUT VWAP bands
            print("   Running without VWAP bands...")
            disabled = self.disable_vwap_bands()
            results_without = self.run_single_backtest(symbol, start_date, end_date, False)
            
            # 2. Backtest WITH VWAP bands
            print("   Running with VWAP bands...")
            self.enable_vwap_bands(disabled)
            results_with = self.run_single_backtest(symbol, start_date, end_date, True)
            
            # 3. Compare results
            if results_without and results_with:
                comparison = self.compare_results(results_without, results_with)
                comparison['symbol'] = symbol
                all_comparisons.append(comparison)
                
                # Print immediate results
                print(f"\n   Results for {symbol}:")
                print(f"   Return: {comparison['total_return_without']:.2%} → {comparison['total_return_with']:.2%} "
                      f"({comparison['return_improvement']:+.2%})")
                print(f"   Sharpe: {comparison['sharpe_without']:.2f} → {comparison['sharpe_with']:.2f} "
                      f"({comparison['sharpe_improvement']:+.2f})")
                print(f"   Win Rate: {comparison['win_rate_without']:.2%} → {comparison['win_rate_with']:.2%} "
                      f"({comparison['win_rate_improvement']:+.2%})")
            
            # 4. Analyze VWAP indicator usage
            vwap_usage = self.analyze_vwap_indicator_usage(symbol, start_date, end_date)
            vwap_usage_stats[symbol] = vwap_usage
        
        # Generate summary report
        self.generate_summary_report(all_comparisons, vwap_usage_stats)
        
        return all_comparisons, vwap_usage_stats
    
    def generate_summary_report(self, comparisons: List[Dict], vwap_usage: Dict):
        """Generate comprehensive summary report."""
        print("\n" + "=" * 80)
        print("📈 BACKTEST SUMMARY REPORT")
        print("=" * 80)
        
        if not comparisons:
            print("No valid comparisons available.")
            return
        
        # Average improvements
        avg_return_imp = np.mean([c['return_improvement'] for c in comparisons])
        avg_sharpe_imp = np.mean([c['sharpe_improvement'] for c in comparisons])
        avg_winrate_imp = np.mean([c['win_rate_improvement'] for c in comparisons])
        avg_dd_imp = np.mean([c['drawdown_improvement'] for c in comparisons])
        
        print("\n1. AVERAGE PERFORMANCE IMPROVEMENTS:")
        print(f"   • Return: {avg_return_imp:+.2%}")
        print(f"   • Sharpe Ratio: {avg_sharpe_imp:+.2f}")
        print(f"   • Win Rate: {avg_winrate_imp:+.2%}")
        print(f"   • Max Drawdown: {avg_dd_imp:+.2%}")
        
        # Per-symbol results
        print("\n2. PER-SYMBOL RESULTS:")
        summary_data = []
        for comp in comparisons:
            summary_data.append({
                'Symbol': comp['symbol'],
                'Return Δ': f"{comp['return_improvement']:+.2%}",
                'Sharpe Δ': f"{comp['sharpe_improvement']:+.2f}",
                'Win Rate Δ': f"{comp['win_rate_improvement']:+.2%}",
                'Trades': f"{comp['trades_without']} → {comp['trades_with']}"
            })
        
        print(tabulate(summary_data, headers='keys', tablefmt='simple'))
        
        # Most selected VWAP indicators
        print("\n3. MOST FREQUENTLY SELECTED VWAP INDICATORS:")
        all_vwap_counts = {}
        for symbol_usage in vwap_usage.values():
            for ind, freq in symbol_usage.items():
                all_vwap_counts[ind] = all_vwap_counts.get(ind, 0) + freq
        
        # Sort by frequency
        top_vwaps = sorted(all_vwap_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\nTop 10 VWAP Indicators by Selection Frequency:")
        for i, (indicator, freq) in enumerate(top_vwaps, 1):
            print(f"   {i}. {indicator}: {freq:.1%}")
        
        # Save detailed results
        results_file = self.results_dir / f'vwap_backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'comparisons': comparisons,
                'vwap_usage': vwap_usage,
                'summary': {
                    'avg_return_improvement': avg_return_imp,
                    'avg_sharpe_improvement': avg_sharpe_imp,
                    'avg_winrate_improvement': avg_winrate_imp,
                    'avg_drawdown_improvement': avg_dd_imp
                }
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")


def main():
    """Run comprehensive backtest comparison."""
    comparison = VWAPBacktestComparison()
    
    try:
        comparisons, vwap_usage = comparison.run_comprehensive_comparison()
        
        # Final recommendations
        print("\n" + "=" * 80)
        print("🎯 RECOMMENDATIONS")
        print("=" * 80)
        
        if comparisons:
            avg_improvement = np.mean([c['return_improvement'] for c in comparisons])
            
            if avg_improvement > 0:
                print("\n✅ VWAP bands show positive impact on performance!")
                print("\nRecommended Actions:")
                print("1. Deploy enhanced indicator set to production")
                print("2. Monitor VWAP band signals closely")
                print("3. Adjust position sizing based on VWAP band distance")
                print("4. Use 2σ bands for high-confidence reversals")
            else:
                print("\n⚠️ Mixed results - further optimization needed")
                print("\nRecommended Actions:")
                print("1. Fine-tune VWAP band parameters")
                print("2. Test different timeframes")
                print("3. Combine with other indicators")
        
        print("\nNext step: Fine-tune confidence thresholds for VWAP selection")
        
    except Exception as e:
        logger.error(f"Backtest comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()