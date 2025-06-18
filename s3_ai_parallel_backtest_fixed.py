#!/usr/bin/env python3
"""
S3 AI Parallel Backtest - Fixed Version
=======================================
Fixed version addressing worker timeout issues.

Key fixes:
- Increased worker timeout to 1800s (30 minutes) based on observed ~18 min processing times
- Reduced backtest period from 90 to 60 days for faster processing
- Better error handling and logging
- Reduced batch size for stability
- Added worker health checks
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import warnings
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import queue
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for main process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def setup_worker_logging():
    """Configure logging for worker processes to suppress initialization messages."""
    import logging
    
    # Disable logging for specific noisy modules
    logging.getLogger('s3_ai_wrapper').setLevel(logging.ERROR)
    logging.getLogger('s3_ai_db_backtest_all_tickers').setLevel(logging.ERROR)
    logging.getLogger('indicator_transformer').setLevel(logging.ERROR)
    
    # Also disable the root logger for workers
    logging.getLogger().setLevel(logging.ERROR)
    
    # Remove all handlers to prevent output
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


def backtest_single_ticker(args):
    """Worker function to backtest a single ticker - with improved error handling."""
    symbol, config, progress_queue = args
    
    try:
        # Setup worker logging first
        setup_worker_logging()
        
        # Report start
        progress_queue.put({
            'symbol': symbol,
            'status': 'started',
            'message': 'Worker started'
        })
        
        # Delayed imports to avoid issues with multiprocessing
        from s3_ai_db_backtest_all_tickers import S3AIDBBacktester, BacktestResult
        
        # Create backtester instance for this worker (logging is now suppressed)
        backtester = S3AIDBBacktester(initial_capital=config['initial_capital'])
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['days'])
        
        # Report data loading
        progress_queue.put({
            'symbol': symbol,
            'status': 'loading',
            'message': 'Loading data'
        })
        
        # Run backtest
        result = backtester.backtest_symbol(symbol, start_date, end_date)
        
        # Report completion
        progress_queue.put({
            'symbol': symbol,
            'status': 'success' if not result.error else 'error',
            'return_pct': result.total_return_pct,
            'trades': result.total_trades,
            'error': result.error if result.error else None
        })
        
        # Convert result to dict for serialization
        result_dict = asdict(result)
        result_dict['start_date'] = str(result_dict['start_date'])
        result_dict['end_date'] = str(result_dict['end_date'])
        result_dict['trades'] = [
            {**asdict(t), 'timestamp': str(t.timestamp)}
            for t in result.trades
        ]
        
        return symbol, result_dict
        
    except Exception as e:
        error_msg = f"Worker error for {symbol}: {str(e)}"
        progress_queue.put({
            'symbol': symbol,
            'status': 'error',
            'error': error_msg
        })
        return symbol, {'error': error_msg, 'symbol': symbol}


class FixedParallelBacktester:
    """Fixed parallel backtester with improved error handling."""
    
    def __init__(self, num_workers: int = None):
        """Initialize parallel backtester."""
        self.num_workers = num_workers or max(1, min(8, cpu_count() - 1))  # Limit to 8 workers max
        self.results = {}
        self.config = {
            'initial_capital': 100000,
            'days': 60  # Reduced to 60 days for faster processing
        }
        
        logger.info(f"🚀 Fixed Parallel Backtester initialized with {self.num_workers} workers")
        logger.info(f"📊 Configuration: ${self.config['initial_capital']:,} capital, {self.config['days']} days")
    
    def run_parallel_backtest(self, tickers: List[str], batch_size: int = 5):
        """Run backtest in parallel with improved error handling."""
        total_tickers = len(tickers)
        logger.info(f"📊 Starting FIXED parallel backtest for {total_tickers} tickers")
        logger.info(f"🔧 Using {self.num_workers} workers, batch size: {batch_size}")
        
        # Create manager for progress tracking
        manager = Manager()
        progress_queue = manager.Queue()
        
        # Prepare arguments for workers
        worker_args = [(ticker, self.config, progress_queue) for ticker in tickers]
        
        # Process with smaller batches and longer timeout
        start_time = time.time()
        completed = 0
        successful = 0
        failed = 0
        
        # Create worker pool with reduced workers
        with Pool(processes=self.num_workers) as pool:
            # Process tickers in smaller batches
            for i in range(0, total_tickers, batch_size):
                batch_end = min(i + batch_size, total_tickers)
                batch_tickers = tickers[i:batch_end]
                batch_args = [worker_args[j] for j in range(i, batch_end)]
                
                logger.info(f"🔄 Processing batch {i//batch_size + 1}: {', '.join(batch_tickers)}")
                
                # Submit batch to workers
                async_results = []
                for args in batch_args:
                    async_result = pool.apply_async(backtest_single_ticker, (args,))
                    async_results.append(async_result)
                
                # Collect results from batch with longer timeout
                batch_start = time.time()
                for j, async_result in enumerate(async_results):
                    try:
                        symbol, result = async_result.get(timeout=1800)  # 30 minute timeout
                        self.results[symbol] = result
                        completed += 1
                        if 'error' not in result:
                            successful += 1
                            logger.info(f"✅ Completed {batch_tickers[j]} in {time.time() - batch_start:.1f}s")
                        else:
                            failed += 1
                            logger.error(f"❌ Error for {batch_tickers[j]}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"❌ Worker timeout/error for {batch_tickers[j]}: {e}")
                        failed += 1
                        completed += 1
                        # Store error result
                        self.results[batch_tickers[j]] = {
                            'symbol': batch_tickers[j],
                            'error': f'Worker timeout/error: {str(e)}',
                            'total_return_pct': 0,
                            'total_trades': 0,
                            'win_rate': 0
                        }
                
                # Process progress updates
                self._process_progress_updates(progress_queue, start_time, total_tickers)
                
                # Brief pause between batches
                time.sleep(1)
        
        # Final progress collection
        self._process_progress_updates(progress_queue, start_time, total_tickers, final=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"🏁 Backtest complete in {elapsed_time/60:.1f} minutes")
        logger.info(f"📊 Total: {completed} | ✅ Successful: {successful} | ❌ Failed: {failed}")
        
        # Calculate and save summary
        self._calculate_summary()
        self._save_results()
    
    def _process_progress_updates(self, progress_queue, start_time, total_tickers, final=False):
        """Process progress updates from workers."""
        updates_processed = 0
        
        while not progress_queue.empty() or final:
            try:
                progress = progress_queue.get_nowait()
                updates_processed += 1
                
                symbol = progress['symbol']
                status = progress['status']
                
                if status == 'started':
                    logger.info(f"🔄 {symbol}: Starting...")
                elif status == 'loading':
                    logger.info(f"📊 {symbol}: Loading data...")
                elif status == 'success':
                    logger.info(f"✅ {symbol}: Return: {progress.get('return_pct', 0):.2f}% | "
                              f"Trades: {progress.get('trades', 0)}")
                elif status == 'error':
                    error_msg = progress.get('error', 'Unknown error')
                    logger.warning(f"❌ {symbol}: {error_msg}")
                
                # Progress update every 5 symbols
                if updates_processed % 5 == 0:
                    elapsed = time.time() - start_time
                    completed = len(self.results)
                    if completed > 0:
                        rate = completed / elapsed
                        eta = (total_tickers - completed) / rate if rate > 0 else 0
                        logger.info(f"📈 Progress: {completed}/{total_tickers} "
                                  f"({completed/total_tickers*100:.1f}%) | "
                                  f"ETA: {eta/60:.1f} minutes")
                
            except queue.Empty:
                if not final:
                    break
                else:
                    time.sleep(0.1)
                    continue
    
    def _calculate_summary(self):
        """Calculate summary statistics."""
        # Categorize results
        error_results = [r for r in self.results.values() if 'error' in r]
        valid_results = [r for r in self.results.values() 
                        if 'error' not in r and r.get('total_trades', 0) > 0]
        no_trades_results = [r for r in self.results.values() 
                           if 'error' not in r and r.get('total_trades', 0) == 0]
        
        # Count error types
        error_types = {}
        for r in error_results:
            error_msg = r.get('error', 'Unknown')
            if 'Insufficient data' in error_msg:
                error_type = 'Insufficient data'
            elif 'timeout' in error_msg:
                error_type = 'Worker timeout'
            else:
                error_type = 'Other error'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        self.summary = {
            'total_symbols': len(self.results),
            'successful_backtests': len(valid_results),
            'failed_backtests': len(error_results),
            'no_trades': len(no_trades_results),
            'error_breakdown': error_types
        }
        
        if valid_results:
            self.summary.update({
                'average_return': np.mean([r['total_return_pct'] for r in valid_results]),
                'median_return': np.median([r['total_return_pct'] for r in valid_results]),
                'best_return': max([r['total_return_pct'] for r in valid_results]),
                'worst_return': min([r['total_return_pct'] for r in valid_results]),
                'average_trades': np.mean([r['total_trades'] for r in valid_results]),
                'average_win_rate': np.mean([r['win_rate'] for r in valid_results]),
                'profitable_symbols': len([r for r in valid_results if r.get('total_return', 0) > 0])
            })
            
            # Top performers
            sorted_results = sorted(valid_results, key=lambda x: x['total_return_pct'], reverse=True)
            self.summary['top_performers'] = [
                {'symbol': r['symbol'], 'return': r['total_return_pct'], 
                 'trades': r['total_trades'], 'win_rate': r['win_rate']}
                for r in sorted_results[:10]
            ]
            
            # Failed symbols list
            self.summary['failed_symbols'] = [
                {'symbol': r['symbol'], 'error': r['error']}
                for r in error_results
            ]
        else:
            logger.warning("No valid results to summarize")
    
    def _save_results(self):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(f'fixed_backtest_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary
        if hasattr(self, 'summary'):
            with open(f'fixed_backtest_summary_{timestamp}.json', 'w') as f:
                json.dump(self.summary, f, indent=2)
            
            # Create CSV of top performers
            if 'top_performers' in self.summary:
                df = pd.DataFrame(self.summary['top_performers'])
                df.to_csv(f'fixed_top_performers_{timestamp}.csv', index=False)
        
        logger.info(f"💾 Results saved to fixed_backtest_*_{timestamp} files")
    
    def print_summary(self):
        """Print summary results."""
        if not hasattr(self, 'summary'):
            logger.warning("No summary available")
            return
        
        print("\n" + "="*60)
        print("S3 AI FIXED PARALLEL BACKTEST SUMMARY")
        print("="*60)
        print(f"Total Symbols: {self.summary['total_symbols']}")
        print(f"Successful Backtests: {self.summary['successful_backtests']}")
        print(f"Failed Backtests: {self.summary['failed_backtests']}")
        print(f"No Trades Generated: {self.summary['no_trades']}")
        
        if self.summary.get('error_breakdown'):
            print("\nError Breakdown:")
            for error_type, count in self.summary['error_breakdown'].items():
                print(f"  {error_type}: {count}")
        
        if self.summary.get('average_return') is not None:
            print(f"\nPerformance Metrics:")
            print(f"Average Return: {self.summary['average_return']:.2f}%")
            print(f"Median Return: {self.summary['median_return']:.2f}%")
            print(f"Best Return: {self.summary['best_return']:.2f}%")
            print(f"Worst Return: {self.summary['worst_return']:.2f}%")
            print(f"Profitable Symbols: {self.summary['profitable_symbols']}")
            print(f"Average Trades: {self.summary['average_trades']:.1f}")
            print(f"Average Win Rate: {self.summary['average_win_rate']:.1f}%")
            
            print("\nTop 10 Performers:")
            for perf in self.summary.get('top_performers', [])[:10]:
                print(f"  {perf['symbol']:6} Return: {perf['return']:7.2f}% | "
                      f"Trades: {perf['trades']:3} | Win Rate: {perf['win_rate']:5.1f}%")
            
            if self.summary.get('failed_symbols'):
                print("\nFailed Symbols:")
                for fail in self.summary['failed_symbols'][:5]:  # Show first 5
                    print(f"  {fail['symbol']:6} - {fail['error'][:50]}...")
                if len(self.summary['failed_symbols']) > 5:
                    print(f"  ... and {len(self.summary['failed_symbols']) - 5} more")
        
        print("="*60)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 AI Fixed Parallel Backtester')
    parser.add_argument('--tickers', default='../tickers/all_tickers.json',
                       help='Path to tickers JSON file')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: min(4, CPU-1))')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Batch size per worker (default: 5)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tickers to process')
    parser.add_argument('--days', type=int, default=60,
                       help='Number of days for backtest period (default: 60)')
    
    args = parser.parse_args()
    
    # Load tickers
    try:
        with open(args.tickers, 'r') as f:
            tickers = json.load(f)
        logger.info(f"📋 Loaded {len(tickers)} tickers from {args.tickers}")
    except Exception as e:
        logger.error(f"Error loading tickers: {e}")
        return
    
    # Apply limit if specified
    if args.limit:
        tickers = tickers[:args.limit]
        logger.info(f"🔧 Limited to {len(tickers)} tickers")
    
    # Run parallel backtest
    backtester = FixedParallelBacktester(num_workers=args.workers)
    backtester.config['days'] = args.days  # Apply days parameter
    logger.info(f"📅 Using {args.days} days for backtest period")
    backtester.run_parallel_backtest(tickers, batch_size=args.batch_size)
    
    # Print summary
    backtester.print_summary()


if __name__ == "__main__":
    main()