#!/usr/bin/env python3
"""
S3 AI Parallel Backtester
========================
High-performance parallel backtesting for all tickers using multiprocessing.
Designed to handle 600+ tickers efficiently.

Features:
- Parallel processing with worker pool
- Batch processing to respect API limits
- Progress tracking with ETA
- Memory-efficient processing
- Automatic retry on failures
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
import signal
import atexit
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

# Global variable to track worker pool for cleanup
global_pool = None
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl-C) to gracefully shutdown worker processes."""
    global global_pool, shutdown_requested
    
    logger.info("🛑 Interrupt received (Ctrl-C). Shutting down gracefully...")
    shutdown_requested = True
    
    if global_pool is not None:
        logger.info("🔄 Terminating worker processes...")
        global_pool.terminate()
        global_pool.join(timeout=5)
        logger.info("✅ Worker processes terminated")
    
    logger.info("👋 Shutdown complete")
    sys.exit(0)


def cleanup_on_exit():
    """Cleanup function called on normal exit."""
    global global_pool
    
    if global_pool is not None:
        logger.info("🧹 Cleaning up worker processes...")
        global_pool.close()
        global_pool.join()


def setup_worker_logging():
    """Configure logging for worker processes to suppress initialization messages."""
    import logging
    
    # Disable logging for specific noisy modules
    logging.getLogger('s3_ai_wrapper').setLevel(logging.WARNING)
    logging.getLogger('s3_ai_backtest_all_tickers').setLevel(logging.WARNING)
    logging.getLogger('indicator_transformer').setLevel(logging.WARNING)
    logging.getLogger('s3_ai_db_backtest_all_tickers').setLevel(logging.WARNING)
    
    # Also disable the root logger for workers
    logging.getLogger().setLevel(logging.WARNING)
    
    # Remove all handlers to prevent output
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


def backtest_single_ticker(args):
    """Worker function to backtest a single ticker."""
    symbol, config, progress_queue = args
    
    try:
        # Setup worker logging first
        setup_worker_logging()
        
        # Delayed imports to avoid issues with multiprocessing
        from s3_ai_db_backtest_all_tickers import S3AIDBBacktester, BacktestResult
        
        # Create backtester instance for this worker (logging is now suppressed)
        backtester = S3AIDBBacktester(initial_capital=config['initial_capital'])
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['days'])
        
        # Run backtest
        result = backtester.backtest_symbol(symbol, start_date, end_date)
        
        # Report progress
        progress_queue.put({
            'symbol': symbol,
            'status': 'success' if not result.error else 'error',
            'return_pct': result.total_return_pct,
            'trades': result.total_trades
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
        progress_queue.put({
            'symbol': symbol,
            'status': 'error',
            'error': str(e)
        })
        return symbol, {'error': str(e), 'symbol': symbol}


class ParallelBacktester:
    """Manages parallel backtesting operations."""
    
    def __init__(self, num_workers: int = None):
        """Initialize parallel backtester."""
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.results = {}
        self.config = {
            'initial_capital': 100000,
            'days': 252  # 1 year of data
        }
        
        logger.info(f"🚀 Parallel Backtester initialized with {self.num_workers} workers")
    
    def run_parallel_backtest(self, tickers: List[str], batch_size: int = 10):
        """Run backtest in parallel with batching."""
        global global_pool, shutdown_requested
        
        total_tickers = len(tickers)
        logger.info(f"📊 Starting parallel backtest for {total_tickers} tickers")
        logger.info(f"🔧 Using {self.num_workers} workers, batch size: {batch_size}")
        
        # Create manager for progress tracking
        manager = Manager()
        progress_queue = manager.Queue()
        
        # Prepare arguments for workers
        worker_args = [(ticker, self.config, progress_queue) for ticker in tickers]
        
        # Process in batches to avoid overwhelming the system
        start_time = time.time()
        completed = 0
        successful = 0
        failed = 0
        
        # Create worker pool
        with Pool(processes=self.num_workers) as pool:
            global_pool = pool  # Store reference for signal handler
            # Process tickers in batches
            for i in range(0, total_tickers, batch_size * self.num_workers):
                # Check for shutdown signal
                if shutdown_requested:
                    logger.info("🛑 Shutdown requested, stopping batch processing")
                    break
                    
                batch_end = min(i + batch_size * self.num_workers, total_tickers)
                batch_args = worker_args[i:batch_end]
                
                # Submit batch to workers
                async_results = []
                for args in batch_args:
                    if shutdown_requested:
                        break
                    async_result = pool.apply_async(backtest_single_ticker, (args,))
                    async_results.append(async_result)
                
                # Collect results from batch
                for async_result in async_results:
                    try:
                        symbol, result = async_result.get(timeout=60)  # 60 second timeout
                        self.results[symbol] = result
                    except Exception as e:
                        logger.error(f"Worker error: {e}")
                
                # Process progress updates
                while not progress_queue.empty():
                    try:
                        progress = progress_queue.get_nowait()
                        completed += 1
                        
                        if progress['status'] == 'success':
                            successful += 1
                            logger.info(f"✅ {progress['symbol']}: "
                                      f"Return: {progress.get('return_pct', 0):.2f}% | "
                                      f"Trades: {progress.get('trades', 0)}")
                        else:
                            failed += 1
                            logger.warning(f"❌ {progress['symbol']}: {progress.get('error', 'Unknown error')}")
                        
                        # Progress update
                        if completed % 10 == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed
                            eta = (total_tickers - completed) / rate if rate > 0 else 0
                            logger.info(f"📈 Progress: {completed}/{total_tickers} "
                                      f"({completed/total_tickers*100:.1f}%) | "
                                      f"ETA: {eta/60:.1f} minutes")
                        
                    except queue.Empty:
                        break
                
                # Brief pause between batches (unless shutting down)
                if not shutdown_requested:
                    time.sleep(0.5)
        
        # Final progress collection
        while not progress_queue.empty():
            try:
                progress = progress_queue.get_nowait()
                completed += 1
                if progress['status'] == 'success':
                    successful += 1
                else:
                    failed += 1
            except queue.Empty:
                break
        
        # Clear global pool reference
        global_pool = None
        
        elapsed_time = time.time() - start_time
        if shutdown_requested:
            logger.info(f"🛑 Backtest interrupted after {elapsed_time/60:.1f} minutes")
        else:
            logger.info(f"🏁 Backtest complete in {elapsed_time/60:.1f} minutes")
        logger.info(f"✅ Successful: {successful} | ❌ Failed: {failed}")
        
        # Calculate and save summary (unless shutdown was requested)
        if not shutdown_requested:
            self._calculate_summary()
            self._save_results()
        else:
            logger.info("⏭️ Skipping summary calculation due to early shutdown")
    
    def _calculate_summary(self):
        """Calculate summary statistics."""
        valid_results = [r for r in self.results.values() 
                        if 'error' not in r and r.get('total_trades', 0) > 0]
        
        if not valid_results:
            logger.warning("No valid results to summarize")
            return
        
        self.summary = {
            'total_symbols': len(self.results),
            'successful_backtests': len(valid_results),
            'average_return': np.mean([r['total_return_pct'] for r in valid_results]),
            'median_return': np.median([r['total_return_pct'] for r in valid_results]),
            'best_return': max([r['total_return_pct'] for r in valid_results]),
            'worst_return': min([r['total_return_pct'] for r in valid_results]),
            'average_trades': np.mean([r['total_trades'] for r in valid_results]),
            'average_win_rate': np.mean([r['win_rate'] for r in valid_results]),
            'profitable_symbols': len([r for r in valid_results if r['total_return'] > 0])
        }
        
        # Top performers
        sorted_results = sorted(valid_results, key=lambda x: x['total_return_pct'], reverse=True)
        self.summary['top_performers'] = [
            {'symbol': r['symbol'], 'return': r['total_return_pct'], 
             'trades': r['total_trades'], 'win_rate': r['win_rate']}
            for r in sorted_results[:20]
        ]
    
    def _save_results(self):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(f'parallel_backtest_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary
        if hasattr(self, 'summary'):
            with open(f'parallel_backtest_summary_{timestamp}.json', 'w') as f:
                json.dump(self.summary, f, indent=2)
            
            # Create CSV of top performers
            if 'top_performers' in self.summary:
                df = pd.DataFrame(self.summary['top_performers'])
                df.to_csv(f'top_performers_{timestamp}.csv', index=False)
        
        logger.info(f"💾 Results saved to parallel_backtest_*_{timestamp} files")
    
    def print_summary(self):
        """Print summary results."""
        if not hasattr(self, 'summary'):
            logger.warning("No summary available")
            return
        
        print("\n" + "="*60)
        print("S3 AI PARALLEL BACKTEST SUMMARY")
        print("="*60)
        print(f"Total Symbols: {self.summary['total_symbols']}")
        print(f"Successful Backtests: {self.summary['successful_backtests']}")
        print(f"Average Return: {self.summary['average_return']:.2f}%")
        print(f"Median Return: {self.summary['median_return']:.2f}%")
        print(f"Best Return: {self.summary['best_return']:.2f}%")
        print(f"Worst Return: {self.summary['worst_return']:.2f}%")
        print(f"Profitable Symbols: {self.summary['profitable_symbols']}")
        
        print("\nTop 20 Performers:")
        for perf in self.summary.get('top_performers', [])[:20]:
            print(f"  {perf['symbol']:6} Return: {perf['return']:7.2f}% | "
                  f"Trades: {perf['trades']:3} | Win Rate: {perf['win_rate']:5.1f}%")
        print("="*60)


def main():
    """Main execution function."""
    import argparse
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_on_exit)
    
    parser = argparse.ArgumentParser(description='S3 AI Parallel Backtester')
    parser.add_argument('--tickers', default='../tickers/all_tickers.json',
                       help='Path to tickers JSON file')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size per worker (default: 10)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tickers to process (for testing)')
    
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
        logger.info(f"🔧 Limited to {len(tickers)} tickers for testing")
    
    # Run parallel backtest
    backtester = ParallelBacktester(num_workers=args.workers)
    backtester.run_parallel_backtest(tickers, batch_size=args.batch_size)
    
    # Print summary
    backtester.print_summary()


if __name__ == "__main__":
    main()