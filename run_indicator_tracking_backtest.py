#!/usr/bin/env python3
"""
Run Backtests with Enhanced Indicator Tracking
==============================================
This script runs backtests on selected symbols and ensures
all indicator data is saved for analysis.
"""

import json
import sys
import os
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_db_backtest_all_tickers import S3AIDBBacktester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_indicator_tracking_backtest(symbols=None, days=30):
    """Run backtest with enhanced indicator tracking."""
    
    if symbols is None:
        # Default to highly liquid symbols that generate many trades
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS',
            'BAC', 'ADBE', 'NFLX', 'CRM', 'PFE', 'ABBV', 'NKE',
            'TMO', 'COST', 'WFC', 'DHR', 'UNP', 'QCOM', 'INTC',
            'AMD', 'GS', 'SBUX', 'MDLZ', 'GILD', 'AMGN', 'AXP'
        ]
    
    # Initialize backtester
    backtester = S3AIDBBacktester(initial_capital=100000)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    results = {}
    trades_with_indicators = []
    
    logger.info(f"Starting indicator tracking backtest for {len(symbols)} symbols")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    for i, symbol in enumerate(symbols):
        logger.info(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")
        
        try:
            # Run backtest
            result = backtester.backtest_symbol(symbol, start_date, end_date)
            
            # Store result
            results[symbol] = {
                'total_return_pct': result.total_return_pct,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'sharpe_ratio': result.sharpe_ratio,
                'trades': len(result.trades)
            }
            
            # Extract trades with indicators
            for trade in result.trades:
                trade_data = {
                    'symbol': symbol,
                    'timestamp': str(trade.timestamp),
                    'action': trade.action,
                    'price': trade.price,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'hold_time_minutes': trade.hold_time_minutes,
                    'signal_strength': trade.signal_strength,
                    'regime': trade.regime,
                    'indicators': trade.indicators,
                    'is_winning': trade.pnl > 0
                }
                trades_with_indicators.append(trade_data)
            
            logger.info(f"✅ {symbol}: {result.total_trades} trades, "
                       f"{result.win_rate:.1%} win rate, "
                       f"{result.total_return_pct:.2%} return")
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            results[symbol] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary_file = f'indicator_tracking_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'symbols': len(symbols),
            'total_trades': len(trades_with_indicators),
            'date_range': {
                'start': str(start_date),
                'end': str(end_date),
                'days': days
            },
            'results': results
        }, f, indent=2)
    
    # Save detailed trades
    trades_file = f'indicator_tracking_trades_{timestamp}.json'
    with open(trades_file, 'w') as f:
        json.dump(trades_with_indicators, f, indent=2)
    
    logger.info(f"\n📊 Summary: {len(trades_with_indicators)} total trades collected")
    logger.info(f"💾 Saved to {summary_file} and {trades_file}")
    
    # Print indicator frequency
    if trades_with_indicators:
        from collections import Counter
        all_indicators = []
        for trade in trades_with_indicators:
            if trade.get('indicators'):
                all_indicators.extend(trade['indicators'].keys())
        
        indicator_counts = Counter(all_indicators)
        logger.info("\n📈 Top 20 Most Used Indicators:")
        for indicator, count in indicator_counts.most_common(20):
            logger.info(f"  {indicator}: {count} uses")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run indicator tracking backtest')
    parser.add_argument('--symbols', nargs='+', help='Symbols to backtest')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--from-file', help='Load symbols from JSON file')
    
    args = parser.parse_args()
    
    symbols = args.symbols
    if args.from_file:
        with open(args.from_file, 'r') as f:
            symbols = json.load(f)
    
    run_indicator_tracking_backtest(symbols=symbols, days=args.days)

if __name__ == "__main__":
    main()