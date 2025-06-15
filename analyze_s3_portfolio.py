#!/usr/bin/env python3
"""
S3 Portfolio Analysis and Ranking
=================================
Analyze S3 strategy performance across multiple tickers and create
tiered portfolio recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple

from s3_database_final import S3DatabaseFinal


def calculate_risk_metrics(trades: List[Dict]) -> Dict[str, float]:
    """Calculate advanced risk metrics from trade history."""
    
    if not trades or len(trades) == 0:
        return {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'profit_factor': 0.0,
            'recovery_factor': 0.0
        }
    
    # Extract returns
    returns = [t.get('return_pct', 0) for t in trades if 'return_pct' in t]
    if not returns:
        returns = [(t['pnl'] / 100000) for t in trades]  # Estimate returns
    
    returns_series = pd.Series(returns)
    
    # Sharpe Ratio (annualized)
    if returns_series.std() > 0:
        sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    # Sortino Ratio (downside deviation)
    negative_returns = returns_series[returns_series < 0]
    if len(negative_returns) > 0 and negative_returns.std() > 0:
        sortino_ratio = (returns_series.mean() / negative_returns.std()) * np.sqrt(252)
    else:
        sortino_ratio = sharpe_ratio * 1.5  # Estimate if no negative returns
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns_series).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # Calmar Ratio
    annual_return = returns_series.mean() * 252
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    # Profit Factor
    wins = returns_series[returns_series > 0].sum()
    losses = abs(returns_series[returns_series < 0].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')
    
    # Recovery Factor
    total_profit = returns_series.sum()
    recovery_factor = total_profit / max_drawdown if max_drawdown > 0 else 0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'profit_factor': profit_factor,
        'recovery_factor': recovery_factor
    }


def score_strategy(results: Dict, risk_metrics: Dict) -> float:
    """Calculate comprehensive strategy score (0-100)."""
    
    score = 0.0
    
    # Win rate component (20 points)
    win_rate = results.get('win_rate', 0)
    score += min(20, win_rate * 40)  # Max at 50% win rate
    
    # Profitability component (20 points)
    total_return = results.get('total_return', 0)
    score += min(20, max(0, total_return * 200))  # Max at 10% return
    
    # Risk-adjusted returns (30 points)
    sharpe = risk_metrics.get('sharpe_ratio', 0)
    score += min(15, max(0, sharpe * 10))  # Max at 1.5 Sharpe
    
    sortino = risk_metrics.get('sortino_ratio', 0)
    score += min(15, max(0, sortino * 7.5))  # Max at 2.0 Sortino
    
    # Consistency component (15 points)
    profit_factor = risk_metrics.get('profit_factor', 0)
    score += min(15, max(0, (profit_factor - 1) * 15))  # Max at PF 2.0
    
    # Activity component (15 points)
    trades = results.get('total_trades', 0)
    if trades >= 50:
        score += 15
    elif trades >= 20:
        score += 10
    elif trades >= 10:
        score += 5
    
    return min(100, score)


def analyze_all_tickers(tickers: List[str], days_back: int = 30) -> pd.DataFrame:
    """Analyze all tickers and return comprehensive results."""
    
    print(f"🔬 Analyzing {len(tickers)} tickers with {days_back} days of data")
    print("=" * 60)
    
    all_results = []
    
    for i, symbol in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] Analyzing {symbol}...", end='', flush=True)
        
        try:
            # Create strategy
            strategy = S3DatabaseFinal(
                symbol=symbol,
                capital=100000,
                adaptive_indicators=True,
                top_k_indicators=10,
                data_days_lookback=days_back
            )
            
            # Run comprehensive backtest
            results = strategy.backtest(verbose=False)
            
            # Get additional details
            latest = strategy.get_latest_signals(hours=48)
            
            # Calculate risk metrics if we have trade details
            risk_metrics = {}
            if 'trades' in results and results['trades']:
                risk_metrics = calculate_risk_metrics(results['trades'])
            else:
                # Estimate risk metrics from basic results
                risk_metrics = {
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'sortino_ratio': results.get('sharpe_ratio', 0) * 1.2,
                    'max_drawdown': results.get('max_drawdown', 0.1),
                    'calmar_ratio': 0,
                    'profit_factor': results.get('profit_factor', 1.0),
                    'recovery_factor': 0
                }
            
            # Calculate comprehensive score
            score = score_strategy(results, risk_metrics)
            
            # Compile all metrics
            ticker_results = {
                'symbol': symbol,
                'score': score,
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'total_pnl': results['total_pnl'],
                'total_return': results['total_return'],
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                'sortino_ratio': risk_metrics.get('sortino_ratio', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0),
                'profit_factor': risk_metrics.get('profit_factor', 0),
                'market_regime': results['market_regime'],
                'signals_generated': results['signals_generated'],
                'latest_signal': latest.get('signal_text', 'UNKNOWN'),
                'latest_price': latest.get('price', 0),
                'data_bars': results['data_bars'],
                'success': True
            }
            
            all_results.append(ticker_results)
            print(f" ✅ Score: {score:.1f}, PnL: ${results['total_pnl']:.2f}")
            
        except Exception as e:
            all_results.append({
                'symbol': symbol,
                'score': 0,
                'success': False,
                'error': str(e)
            })
            print(f" ❌ {str(e)[:40]}...")
    
    return pd.DataFrame(all_results)


def create_tiered_portfolio(df_results: pd.DataFrame) -> Dict:
    """Create tiered portfolio recommendations based on scores."""
    
    # Filter successful strategies
    successful = df_results[df_results['success']].copy()
    
    if len(successful) == 0:
        return {'error': 'No successful strategies found'}
    
    # Sort by score
    successful = successful.sort_values('score', ascending=False)
    
    # Define tiers
    tier_1_cutoff = successful['score'].quantile(0.8)  # Top 20%
    tier_2_cutoff = successful['score'].quantile(0.6)  # Next 20%
    tier_3_cutoff = successful['score'].quantile(0.4)  # Next 20%
    tier_4_cutoff = successful['score'].quantile(0.2)  # Next 20%
    
    # Assign tiers
    tiers = {
        'Tier 1 (8% allocation)': successful[successful['score'] >= tier_1_cutoff],
        'Tier 2 (5% allocation)': successful[(successful['score'] >= tier_2_cutoff) & 
                                            (successful['score'] < tier_1_cutoff)],
        'Tier 3 (2.5% allocation)': successful[(successful['score'] >= tier_3_cutoff) & 
                                              (successful['score'] < tier_2_cutoff)],
        'Tier 4 (1% allocation)': successful[(successful['score'] >= tier_4_cutoff) & 
                                            (successful['score'] < tier_3_cutoff)],
        'Not Recommended': successful[successful['score'] < tier_4_cutoff]
    }
    
    # Create portfolio configuration
    portfolio_config = {
        'created_at': datetime.now().isoformat(),
        'strategy': 'S3_Database_AI',
        'total_capital': 500000,
        'tiers': {}
    }
    
    for tier_name, tier_df in tiers.items():
        if 'Not Recommended' in tier_name:
            continue
            
        # Extract allocation percentage
        if 'Tier 1' in tier_name:
            allocation_pct = 0.08
        elif 'Tier 2' in tier_name:
            allocation_pct = 0.05
        elif 'Tier 3' in tier_name:
            allocation_pct = 0.025
        else:
            allocation_pct = 0.01
        
        tier_symbols = []
        for _, row in tier_df.iterrows():
            tier_symbols.append({
                'symbol': row['symbol'],
                'score': round(row['score'], 1),
                'win_rate': round(row['win_rate'], 3),
                'sharpe_ratio': round(row['sharpe_ratio'], 2),
                'total_pnl': round(row['total_pnl'], 2),
                'latest_signal': row['latest_signal']
            })
        
        portfolio_config['tiers'][tier_name] = {
            'allocation_per_symbol': allocation_pct,
            'symbols': tier_symbols
        }
    
    return portfolio_config, tiers


def main():
    """Main analysis function."""
    
    print("🚀 S3 Portfolio Analysis and Ranking")
    print("=" * 60)
    
    # Load tickers
    try:
        with open('../james_tickers.json', 'r') as f:
            data = json.load(f)
            tickers = data.get('symbols', [])
            print(f"📋 Loaded {len(tickers)} tickers from james_tickers.json")
    except:
        # Default list
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA'
        ]
        print(f"📋 Using default list of {len(tickers)} tickers")
    
    # Analyze all tickers
    df_results = analyze_all_tickers(tickers, days_back=30)
    
    # Create tiered portfolio
    portfolio_config, tiers = create_tiered_portfolio(df_results)
    
    if 'error' in portfolio_config:
        print(f"\n❌ {portfolio_config['error']}")
        return
    
    # Display results
    print("\n" + "="*80)
    print("📊 TIERED PORTFOLIO RECOMMENDATIONS")
    print("="*80)
    
    for tier_name, tier_df in tiers.items():
        print(f"\n{tier_name} ({len(tier_df)} symbols)")
        print("-" * 60)
        
        if len(tier_df) > 0:
            display_df = tier_df[['symbol', 'score', 'win_rate', 'total_pnl', 
                                 'sharpe_ratio', 'total_trades', 'latest_signal']].round(2)
            
            for _, row in display_df.iterrows():
                print(f"{row['symbol']:6} | Score: {row['score']:5.1f} | "
                      f"Win: {row['win_rate']:5.1%} | PnL: ${row['total_pnl']:8.2f} | "
                      f"Sharpe: {row['sharpe_ratio']:5.2f} | "
                      f"Trades: {int(row['total_trades']):3} | {row['latest_signal']}")
    
    # Save configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    results_file = f"s3_portfolio_analysis_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Save portfolio configuration
    config_file = f"s3_portfolio_config_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump(portfolio_config, f, indent=2)
    print(f"📄 Portfolio configuration saved to: {config_file}")
    
    # Summary statistics
    successful = df_results[df_results['success']]
    if len(successful) > 0:
        print(f"\n📊 Summary Statistics:")
        print(f"   Average Score: {successful['score'].mean():.1f}")
        print(f"   Average Win Rate: {successful['win_rate'].mean():.1%}")
        print(f"   Total PnL: ${successful['total_pnl'].sum():.2f}")
        print(f"   Average Sharpe: {successful['sharpe_ratio'].mean():.2f}")
        print(f"   Profitable Strategies: {(successful['total_pnl'] > 0).sum()}/{len(successful)}")


if __name__ == "__main__":
    main()