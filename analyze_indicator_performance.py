#!/usr/bin/env python3
"""
Analyze Indicator Performance from Backtest Results
=================================================
Analyzes which indicators were most and least successful
based on backtest trade data.
"""

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import pandas as pd
from datetime import datetime
import glob

def load_backtest_results(pattern: str = "*backtest*.json") -> Dict[str, Any]:
    """Load all backtest result files matching pattern."""
    results = {}
    
    # Search in current directory and subdirectories
    for filepath in glob.glob(pattern, recursive=True):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results[filepath] = data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    # Also search in backtest_results directory
    backtest_dir = "backtest_results"
    if os.path.exists(backtest_dir):
        for filepath in glob.glob(f"{backtest_dir}/*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[filepath] = data
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return results

def extract_trades_with_indicators(results: Dict[str, Any]) -> List[Dict]:
    """Extract all trades that have indicator information."""
    all_trades = []
    
    for filepath, data in results.items():
        # Handle different result formats
        
        # Format 1: Direct symbol -> result mapping
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            for symbol, result in data.items():
                if isinstance(result, dict) and 'trades' in result:
                    trades = result.get('trades', [])
                    if isinstance(trades, list):
                        for trade in trades:
                            if isinstance(trade, dict) and 'indicators' in trade:
                                trade['source_file'] = filepath
                                trade['symbol'] = symbol
                                all_trades.append(trade)
        
        # Format 2: VWAP analysis format
        elif 'trades' in data and 'selections' in data:
            # Map indicator usage to trades
            indicator_usage = {}
            for selection in data.get('selections', []):
                if 'symbol' in selection and 'top_indicators' in selection:
                    indicator_usage[selection['symbol']] = selection
            
            for trade_summary in data.get('trades', []):
                if 'symbol' in trade_summary:
                    symbol = trade_summary['symbol']
                    if symbol in indicator_usage:
                        # Create synthetic trades based on summary
                        for i in range(trade_summary.get('total_trades', 0)):
                            trade = {
                                'symbol': symbol,
                                'source_file': filepath,
                                'win_rate': trade_summary.get('win_rate', 0),
                                'avg_return': trade_summary.get('avg_return', 0),
                                'sharpe_ratio': trade_summary.get('sharpe_ratio', 0),
                                'indicator_usage': indicator_usage[symbol],
                                'is_winning': i < trade_summary.get('winning_trades', 0)
                            }
                            all_trades.append(trade)
    
    return all_trades

def analyze_indicator_performance(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze which indicators led to successful vs unsuccessful trades."""
    
    # Track indicator performance
    indicator_stats = defaultdict(lambda: {
        'total_uses': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_return': 0.0,
        'symbols': set(),
        'win_rate': 0.0
    })
    
    # Track combinations
    winning_combinations = Counter()
    losing_combinations = Counter()
    
    for trade in trades:
        # Extract indicators from trade
        indicators = []
        
        if 'indicators' in trade:
            indicators = list(trade['indicators'].keys())
        elif 'indicator_usage' in trade:
            # Extract from VWAP format
            usage = trade['indicator_usage']
            if 'top_indicators' in usage:
                indicators = [ind[0] for ind in usage['top_indicators']]
        
        if not indicators:
            continue
        
        # Determine if winning trade
        is_winning = False
        if 'pnl' in trade:
            is_winning = trade['pnl'] > 0
        elif 'is_winning' in trade:
            is_winning = trade['is_winning']
        elif 'pnl_pct' in trade:
            is_winning = trade['pnl_pct'] > 0
        
        # Get return
        trade_return = 0.0
        if 'pnl_pct' in trade:
            trade_return = trade['pnl_pct']
        elif 'avg_return' in trade:
            trade_return = trade['avg_return']
        
        # Update stats for each indicator
        for indicator in indicators:
            stats = indicator_stats[indicator]
            stats['total_uses'] += 1
            stats['symbols'].add(trade.get('symbol', 'UNKNOWN'))
            
            if is_winning:
                stats['winning_trades'] += 1
            else:
                stats['losing_trades'] += 1
            
            stats['total_return'] += trade_return
        
        # Track combinations
        if len(indicators) > 1:
            combo = tuple(sorted(indicators[:5]))  # Top 5 indicators
            if is_winning:
                winning_combinations[combo] += 1
            else:
                losing_combinations[combo] += 1
    
    # Calculate win rates
    for indicator, stats in indicator_stats.items():
        total = stats['winning_trades'] + stats['losing_trades']
        if total > 0:
            stats['win_rate'] = stats['winning_trades'] / total
            stats['avg_return'] = stats['total_return'] / total
    
    return {
        'indicator_stats': dict(indicator_stats),
        'winning_combinations': dict(winning_combinations.most_common(10)),
        'losing_combinations': dict(losing_combinations.most_common(10))
    }

def print_analysis_report(analysis: Dict[str, Any]):
    """Print a formatted analysis report."""
    print("="*80)
    print("S3 AI INDICATOR PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Convert to DataFrame for easier analysis
    indicator_df = pd.DataFrame.from_dict(
        analysis['indicator_stats'], 
        orient='index'
    )
    
    if indicator_df.empty:
        print("No indicator data found in backtest results.")
        return
    
    # Convert sets to counts for display
    indicator_df['symbol_count'] = indicator_df['symbols'].apply(len)
    indicator_df['symbols'] = indicator_df['symbols'].apply(lambda x: ', '.join(sorted(x)[:3]) + ('...' if len(x) > 3 else ''))
    
    # Sort by different metrics
    print("\n📊 TOP 20 INDICATORS BY USAGE FREQUENCY:")
    print("-"*80)
    by_usage = indicator_df.nlargest(20, 'total_uses')[['total_uses', 'win_rate', 'avg_return', 'symbol_count']]
    for idx, row in by_usage.iterrows():
        print(f"{idx:25} | Uses: {row['total_uses']:4} | Win Rate: {row['win_rate']:5.1%} | Avg Return: {row['avg_return']:7.3%}")
    
    print("\n🏆 TOP 20 INDICATORS BY WIN RATE (min 5 uses):")
    print("-"*80)
    qualified = indicator_df[indicator_df['total_uses'] >= 5]
    if not qualified.empty:
        by_win_rate = qualified.nlargest(20, 'win_rate')[['total_uses', 'win_rate', 'winning_trades', 'losing_trades', 'avg_return']]
        for idx, row in by_win_rate.iterrows():
            print(f"{idx:25} | Win Rate: {row['win_rate']:5.1%} | W/L: {row['winning_trades']}/{row['losing_trades']} | Avg Return: {row['avg_return']:7.3%}")
    
    print("\n💰 TOP 20 INDICATORS BY AVERAGE RETURN (min 5 uses):")
    print("-"*80)
    if not qualified.empty:
        by_return = qualified.nlargest(20, 'avg_return')[['total_uses', 'win_rate', 'avg_return', 'total_return']]
        for idx, row in by_return.iterrows():
            print(f"{idx:25} | Avg Return: {row['avg_return']:7.3%} | Total Return: {row['total_return']:7.3%} | Win Rate: {row['win_rate']:5.1%}")
    
    print("\n❌ WORST 20 INDICATORS BY WIN RATE (min 5 uses):")
    print("-"*80)
    if not qualified.empty:
        by_worst = qualified.nsmallest(20, 'win_rate')[['total_uses', 'win_rate', 'winning_trades', 'losing_trades', 'avg_return']]
        for idx, row in by_worst.iterrows():
            print(f"{idx:25} | Win Rate: {row['win_rate']:5.1%} | W/L: {row['winning_trades']}/{row['losing_trades']} | Avg Return: {row['avg_return']:7.3%}")
    
    # Indicator categories analysis
    print("\n📈 INDICATOR CATEGORY ANALYSIS:")
    print("-"*80)
    categories = {
        'ATR': ['ATR_'],
        'VWAP': ['VWAP', 'AVWAP'],
        'Moving Averages': ['SMA_', 'EMA_'],
        'RSI': ['RSI_'],
        'Bollinger': ['BB_'],
        'MACD': ['MACD'],
        'Support/Resistance': ['SUPPORT_RESISTANCE'],
        'Stochastic': ['STOCH_']
    }
    
    category_stats = {}
    for cat_name, patterns in categories.items():
        cat_indicators = [ind for ind in indicator_df.index if any(p in ind for p in patterns)]
        if cat_indicators:
            cat_df = indicator_df.loc[cat_indicators]
            total_uses = cat_df['total_uses'].sum()
            avg_win_rate = (cat_df['winning_trades'].sum() / (cat_df['winning_trades'].sum() + cat_df['losing_trades'].sum())) if (cat_df['winning_trades'].sum() + cat_df['losing_trades'].sum()) > 0 else 0
            avg_return = cat_df['total_return'].sum() / total_uses if total_uses > 0 else 0
            
            category_stats[cat_name] = {
                'indicators': len(cat_indicators),
                'total_uses': total_uses,
                'avg_win_rate': avg_win_rate,
                'avg_return': avg_return
            }
    
    cat_df = pd.DataFrame.from_dict(category_stats, orient='index')
    if not cat_df.empty:
        cat_df = cat_df.sort_values('total_uses', ascending=False)
        for idx, row in cat_df.iterrows():
            print(f"{idx:20} | Indicators: {row['indicators']:2} | Uses: {row['total_uses']:4} | Avg Win Rate: {row['avg_win_rate']:5.1%} | Avg Return: {row['avg_return']:7.3%}")
    
    # Winning combinations
    if analysis['winning_combinations']:
        print("\n✅ TOP WINNING INDICATOR COMBINATIONS:")
        print("-"*80)
        for combo, count in list(analysis['winning_combinations'].items())[:5]:
            indicators = ', '.join(combo[:3]) + ('...' if len(combo) > 3 else '')
            print(f"Count: {count:3} | {indicators}")
    
    print("\n" + "="*80)

def main():
    """Main analysis function."""
    print("🔍 Loading backtest results...")
    
    # Load all backtest results
    results = load_backtest_results()
    print(f"📁 Found {len(results)} result files")
    
    # Extract trades with indicators
    trades = extract_trades_with_indicators(results)
    print(f"📊 Found {len(trades)} trades with indicator data")
    
    if not trades:
        print("\n⚠️ No trades with indicator data found!")
        print("Make sure to run backtests that save indicator information in trades.")
        return
    
    # Analyze indicator performance
    analysis = analyze_indicator_performance(trades)
    
    # Print report
    print_analysis_report(analysis)
    
    # Save analysis to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"indicator_performance_analysis_{timestamp}.json"
    
    # Prepare data for JSON serialization
    json_safe_analysis = {
        'indicator_stats': analysis['indicator_stats'],
        'winning_combinations': {str(k): v for k, v in analysis['winning_combinations'].items()},
        'losing_combinations': {str(k): v for k, v in analysis['losing_combinations'].items()}
    }
    
    # Convert sets to lists
    for ind, stats in json_safe_analysis['indicator_stats'].items():
        stats['symbols'] = list(stats['symbols'])
    
    with open(output_file, 'w') as f:
        json.dump(json_safe_analysis, f, indent=2)
    print(f"\n💾 Analysis saved to {output_file}")

if __name__ == "__main__":
    main()