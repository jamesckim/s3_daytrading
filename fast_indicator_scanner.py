#!/usr/bin/env python3
"""
Fast Indicator Scanner
=====================
Quickly scans multiple symbols to collect indicator usage data
without running full backtests.
"""

import json
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict, Counter
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_wrapper import S3AIWrapper
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scan_indicators_for_symbol(symbol, days=30):
    """Scan indicators for a single symbol without full backtest."""
    try:
        # Initialize components
        db_provider = DatabaseDataProvider()
        ai_wrapper = S3AIWrapper()
        
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = db_provider.get_minute_data(symbol, start_date, end_date)
        if data.empty:
            return None
        
        # Sample every 30 minutes during market hours
        samples = []
        indicators_used = Counter()
        
        for date, day_data in data.groupby(data.index.date):
            # Sample at specific times
            sample_times = ['09:30', '10:00', '10:30', '11:00', '11:30', 
                          '12:00', '12:30', '13:00', '13:30', '14:00', 
                          '14:30', '15:00', '15:30']
            
            for time_str in sample_times:
                try:
                    # Get 5-minute window around sample time
                    hour, minute = map(int, time_str.split(':'))
                    sample_data = day_data.between_time(
                        f"{hour:02d}:{max(0, minute-5):02d}",
                        f"{hour:02d}:{min(59, minute+5):02d}"
                    )
                    
                    if len(sample_data) > 0:
                        # Get AI indicator selection
                        selected_indicators, weights, confidence = ai_wrapper.select_indicators(
                            symbol, sample_data
                        )
                        
                        if selected_indicators:
                            for ind in selected_indicators:
                                indicators_used[ind] += 1
                            
                            samples.append({
                                'timestamp': str(sample_data.index[-1]),
                                'indicators': selected_indicators,
                                'confidence': confidence
                            })
                
                except Exception as e:
                    continue
        
        return {
            'symbol': symbol,
            'samples': len(samples),
            'indicators_used': dict(indicators_used),
            'top_indicators': indicators_used.most_common(10)
        }
        
    except Exception as e:
        logger.error(f"Error scanning {symbol}: {e}")
        return None

def main():
    # High-volume symbols likely to have good data
    symbols = [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'META',
        'GOOGL', 'AMZN', 'NFLX', 'BAC', 'JPM', 'XOM', 'CVX', 'V',
        'MA', 'COST', 'WMT', 'HD', 'DIS', 'ADBE', 'CRM', 'INTC'
    ]
    
    results = []
    indicator_summary = Counter()
    
    logger.info(f"Scanning {len(symbols)} symbols for indicator usage...")
    
    for i, symbol in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] Scanning {symbol}...")
        result = scan_indicators_for_symbol(symbol, days=30)
        
        if result:
            results.append(result)
            # Aggregate indicators
            for ind, count in result['indicators_used'].items():
                indicator_summary[ind] += count
            logger.info(f"✅ {symbol}: {result['samples']} samples, "
                       f"{len(result['indicators_used'])} unique indicators")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'indicator_scan_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'scan_date': str(datetime.now()),
            'symbols_scanned': len(symbols),
            'total_samples': sum(r['samples'] for r in results),
            'indicator_summary': dict(indicator_summary.most_common()),
            'symbol_results': results
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("INDICATOR USAGE SUMMARY")
    print("="*60)
    print(f"Symbols scanned: {len(results)}")
    print(f"Total samples: {sum(r['samples'] for r in results)}")
    print("\nTop 20 Most Used Indicators:")
    for ind, count in indicator_summary.most_common(20):
        print(f"  {ind:30} {count:5} uses")
    
    print(f"\n💾 Results saved to {output_file}")

if __name__ == "__main__":
    main()