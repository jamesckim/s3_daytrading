#!/usr/bin/env python3
"""
S3 AI Parallel Backtest - Debug Version
=======================================
Enhanced version with detailed progress tracking to identify bottlenecks.

Key enhancements:
- Detailed timing for each processing step
- Progress tracking within worker processes
- Memory usage monitoring
- Database query performance tracking
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
from multiprocessing import Pool, cpu_count, Manager, current_process
from functools import partial
import queue
import psutil
import traceback
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


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def setup_worker_logging():
    """Configure logging for worker processes."""
    import logging
    
    # Create a worker-specific logger
    worker_logger = logging.getLogger(f'worker_{current_process().name}')
    worker_logger.setLevel(logging.DEBUG)
    
    # Create handler that writes to a worker-specific log file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    handler = logging.FileHandler(
        os.path.join(log_dir, f'worker_{current_process().name}_{os.getpid()}.log')
    )
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    worker_logger.addHandler(handler)
    
    # Disable other loggers
    logging.getLogger('s3_ai_wrapper').setLevel(logging.ERROR)
    logging.getLogger('s3_ai_db_backtest_all_tickers').setLevel(logging.ERROR)
    logging.getLogger('indicator_transformer').setLevel(logging.ERROR)
    
    return worker_logger


def backtest_single_ticker_debug(args):
    """Worker function with detailed progress tracking."""
    symbol, config, progress_queue = args
    
    # Setup worker logging
    worker_logger = setup_worker_logging()
    worker_logger.info(f"Starting backtest for {symbol}")
    
    start_time = time.time()
    step_times = {}
    
    try:
        # Report start with memory usage
        progress_queue.put({
            'symbol': symbol,
            'status': 'started',
            'message': f'Worker started (Memory: {get_memory_usage():.1f}MB)',
            'timestamp': time.time()
        })
        
        # Step 1: Import modules
        import_start = time.time()
        worker_logger.info(f"{symbol}: Importing modules...")
        from s3_ai_db_backtest_all_tickers import S3AIDBBacktester, BacktestResult
        step_times['import'] = time.time() - import_start
        worker_logger.info(f"{symbol}: Import completed in {step_times['import']:.2f}s")
        
        # Step 2: Create backtester instance
        init_start = time.time()
        worker_logger.info(f"{symbol}: Creating backtester instance...")
        progress_queue.put({
            'symbol': symbol,
            'status': 'initializing',
            'message': 'Creating backtester',
            'timestamp': time.time()
        })
        
        backtester = S3AIDBBacktester(initial_capital=config['initial_capital'])
        step_times['init'] = time.time() - init_start
        worker_logger.info(f"{symbol}: Backtester created in {step_times['init']:.2f}s")
        
        # Step 3: Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config['days'])
        
        # Step 4: Database query
        db_start = time.time()
        worker_logger.info(f"{symbol}: Starting database query for {start_date} to {end_date}")
        progress_queue.put({
            'symbol': symbol,
            'status': 'loading',
            'message': f'Loading data ({config["days"]} days)',
            'timestamp': time.time(),
            'memory_mb': get_memory_usage()
        })
        
        # Add progress tracking to the backtest
        # We'll need to modify the backtest method to report progress
        # For now, we'll run it as is but track the total time
        
        # Run backtest with timing
        backtest_start = time.time()
        worker_logger.info(f"{symbol}: Starting backtest computation...")
        
        # Monkey-patch the backtest method to add progress reporting
        original_backtest = backtester.backtest_symbol
        
        def backtest_with_progress(symbol, start_date, end_date):
            # Report data loading phase
            progress_queue.put({
                'symbol': symbol,
                'status': 'processing',
                'message': 'Running strategy analysis',
                'timestamp': time.time(),
                'phase': 'strategy'
            })
            
            result = original_backtest(symbol, start_date, end_date)
            
            # Report completion of strategy
            progress_queue.put({
                'symbol': symbol,
                'status': 'processing',
                'message': 'Calculating metrics',
                'timestamp': time.time(),
                'phase': 'metrics'
            })
            
            return result
        
        backtester.backtest_symbol = backtest_with_progress
        
        # Run the actual backtest
        result = backtester.backtest_symbol(symbol, start_date, end_date)
        
        step_times['backtest'] = time.time() - backtest_start
        worker_logger.info(f"{symbol}: Backtest completed in {step_times['backtest']:.2f}s")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Report completion with detailed timing
        progress_queue.put({
            'symbol': symbol,
            'status': 'success' if not result.error else 'error',
            'return_pct': result.total_return_pct,
            'trades': result.total_trades,
            'error': result.error if result.error else None,
            'timestamp': time.time(),
            'timing': {
                'total': total_time,
                'import': step_times.get('import', 0),
                'init': step_times.get('init', 0),
                'backtest': step_times.get('backtest', 0)
            },
            'memory_mb': get_memory_usage()
        })
        
        worker_logger.info(f"{symbol}: Total processing time: {total_time:.2f}s")
        worker_logger.info(f"{symbol}: Step breakdown: {step_times}")
        
        # Convert result to dict for serialization
        result_dict = asdict(result)
        result_dict['start_date'] = str(result_dict['start_date'])
        result_dict['end_date'] = str(result_dict['end_date'])
        result_dict['trades'] = [
            {**asdict(t), 'timestamp': str(t.timestamp)}
            for t in result.trades
        ]
        result_dict['timing'] = step_times
        result_dict['total_time'] = total_time
        
        return symbol, result_dict
        
    except Exception as e:
        error_msg = f"Worker error for {symbol}: {str(e)}"
        worker_logger.error(f"{symbol}: {error_msg}")
        worker_logger.error(f"{symbol}: Traceback:\n{traceback.format_exc()}")
        
        progress_queue.put({
            'symbol': symbol,
            'status': 'error',
            'error': error_msg,
            'timestamp': time.time(),
            'timing': step_times,
            'traceback': traceback.format_exc()
        })
        
        return symbol, {
            'error': error_msg,
            'symbol': symbol,
            'timing': step_times,
            'traceback': traceback.format_exc()
        }


class DebugParallelBacktester:
    """Parallel backtester with enhanced debugging and progress tracking."""
    
    def __init__(self, num_workers: int = None):
        """Initialize parallel backtester."""
        self.num_workers = num_workers or max(1, min(4, cpu_count() - 1))
        self.results = {}
        self.timing_stats = {}
        self.config = {
            'initial_capital': 100000,
            'days': 90
        }
        
        logger.info(f"🚀 Debug Parallel Backtester initialized with {self.num_workers} workers")
        logger.info(f"📊 Configuration: ${self.config['initial_capital']:,} capital, {self.config['days']} days")
        logger.info(f"💾 System Memory: {psutil.virtual_memory().percent:.1f}% used")
    
    def run_parallel_backtest(self, tickers: List[str], batch_size: int = 3):
        """Run backtest in parallel with detailed progress tracking."""
        total_tickers = len(tickers)
        logger.info(f"📊 Starting DEBUG parallel backtest for {total_tickers} tickers")
        logger.info(f"🔧 Using {self.num_workers} workers, batch size: {batch_size}")
        
        # Create log directory
        os.makedirs("logs", exist_ok=True)
        
        # Create manager for progress tracking
        manager = Manager()
        progress_queue = manager.Queue()
        
        # Prepare arguments for workers
        worker_args = [(ticker, self.config, progress_queue) for ticker in tickers]
        
        # Process with timing
        start_time = time.time()
        completed = 0
        successful = 0
        failed = 0
        
        # Track timing by phase
        phase_times = {
            'import': [],
            'init': [],
            'backtest': [],
            'total': []
        }
        
        # Create worker pool
        with Pool(processes=self.num_workers) as pool:
            # Process tickers in batches
            for i in range(0, total_tickers, batch_size):
                batch_end = min(i + batch_size, total_tickers)
                batch_tickers = tickers[i:batch_end]
                batch_args = [worker_args[j] for j in range(i, batch_end)]
                
                logger.info(f"\n🔄 Processing batch {i//batch_size + 1}/{(total_tickers + batch_size - 1)//batch_size}")
                logger.info(f"📦 Batch tickers: {', '.join(batch_tickers)}")
                
                # Submit batch to workers
                async_results = []
                batch_submit_time = time.time()
                for args in batch_args:
                    async_result = pool.apply_async(backtest_single_ticker_debug, (args,))
                    async_results.append(async_result)
                
                # Collect results from batch
                batch_start = time.time()
                for j, async_result in enumerate(async_results):
                    ticker = batch_tickers[j]
                    try:
                        # Longer timeout for debugging
                        symbol, result = async_result.get(timeout=600)  # 10 minute timeout
                        self.results[symbol] = result
                        completed += 1
                        
                        if 'error' not in result:
                            successful += 1
                            timing = result.get('timing', {})
                            logger.info(f"✅ {ticker}: Completed in {result.get('total_time', 0):.1f}s " +
                                      f"(Import: {timing.get('import', 0):.1f}s, " +
                                      f"Init: {timing.get('init', 0):.1f}s, " +
                                      f"Backtest: {timing.get('backtest', 0):.1f}s)")
                            
                            # Track timing
                            for phase, phase_time in timing.items():
                                if phase in phase_times:
                                    phase_times[phase].append(phase_time)
                            phase_times['total'].append(result.get('total_time', 0))
                        else:
                            failed += 1
                            logger.error(f"❌ {ticker}: {result.get('error', 'Unknown error')}")
                            if 'traceback' in result:
                                logger.debug(f"Traceback for {ticker}:\n{result['traceback']}")
                                
                    except Exception as e:
                        logger.error(f"❌ {ticker}: Worker timeout/error after {time.time() - batch_start:.1f}s: {e}")
                        failed += 1
                        completed += 1
                        self.results[ticker] = {
                            'symbol': ticker,
                            'error': f'Worker timeout/error: {str(e)}',
                            'timeout_seconds': time.time() - batch_start
                        }
                
                # Process progress updates
                self._process_progress_updates(progress_queue, start_time, total_tickers)
                
                # Memory check
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}%")
                
                # Brief pause between batches
                time.sleep(0.5)
        
        # Final progress collection
        self._process_progress_updates(progress_queue, start_time, total_tickers, final=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n🏁 Backtest complete in {elapsed_time/60:.1f} minutes")
        logger.info(f"📊 Total: {completed} | ✅ Successful: {successful} | ❌ Failed: {failed}")
        
        # Print timing statistics
        if phase_times['total']:
            logger.info("\n⏱️ Average Timing by Phase:")
            for phase, times in phase_times.items():
                if times:
                    avg_time = np.mean(times)
                    logger.info(f"  {phase.capitalize()}: {avg_time:.2f}s (min: {min(times):.2f}s, max: {max(times):.2f}s)")
        
        # Calculate and save summary
        self._calculate_summary()
        self._save_results()
    
    def _process_progress_updates(self, progress_queue, start_time, total_tickers, final=False):
        """Process progress updates from workers with detailed logging."""
        updates_processed = 0
        
        while not progress_queue.empty() or final:
            try:
                progress = progress_queue.get_nowait()
                updates_processed += 1
                
                symbol = progress['symbol']
                status = progress['status']
                timestamp = progress.get('timestamp', time.time())
                
                if status == 'started':
                    logger.debug(f"🔄 {symbol}: Worker started - {progress.get('message', '')}")
                elif status == 'initializing':
                    logger.debug(f"🔧 {symbol}: Initializing backtester")
                elif status == 'loading':
                    logger.debug(f"📊 {symbol}: Loading data - {progress.get('message', '')} " +
                               f"(Memory: {progress.get('memory_mb', 0):.1f}MB)")
                elif status == 'processing':
                    phase = progress.get('phase', 'unknown')
                    logger.debug(f"⚙️ {symbol}: Processing - {progress.get('message', '')} (Phase: {phase})")
                elif status == 'success':
                    timing = progress.get('timing', {})
                    logger.debug(f"✅ {symbol}: Success - Return: {progress.get('return_pct', 0):.2f}% | " +
                               f"Trades: {progress.get('trades', 0)} | " +
                               f"Memory: {progress.get('memory_mb', 0):.1f}MB")
                elif status == 'error':
                    logger.debug(f"❌ {symbol}: Error - {progress.get('error', 'Unknown error')}")
                
            except queue.Empty:
                if not final:
                    break
                else:
                    time.sleep(0.1)
                    continue
    
    def _calculate_summary(self):
        """Calculate summary statistics with timing analysis."""
        # Standard summary calculation
        error_results = [r for r in self.results.values() if 'error' in r]
        valid_results = [r for r in self.results.values() 
                        if 'error' not in r and r.get('total_trades', 0) > 0]
        no_trades_results = [r for r in self.results.values() 
                           if 'error' not in r and r.get('total_trades', 0) == 0]
        
        # Analyze errors
        error_types = {}
        timeout_errors = []
        for r in error_results:
            error_msg = r.get('error', 'Unknown')
            if 'timeout' in error_msg.lower():
                error_type = 'Worker timeout'
                timeout_errors.append({
                    'symbol': r['symbol'],
                    'timeout_seconds': r.get('timeout_seconds', 0)
                })
            elif 'Insufficient data' in error_msg:
                error_type = 'Insufficient data'
            else:
                error_type = 'Other error'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        self.summary = {
            'total_symbols': len(self.results),
            'successful_backtests': len(valid_results),
            'failed_backtests': len(error_results),
            'no_trades': len(no_trades_results),
            'error_breakdown': error_types,
            'timeout_errors': timeout_errors
        }
        
        # Add timing analysis
        if valid_results:
            timing_data = []
            for r in valid_results:
                if 'timing' in r:
                    timing_data.append({
                        'symbol': r['symbol'],
                        'total': r.get('total_time', 0),
                        **r.get('timing', {})
                    })
            
            if timing_data:
                self.summary['timing_analysis'] = {
                    'slowest_symbols': sorted(timing_data, key=lambda x: x['total'], reverse=True)[:5],
                    'average_times': {
                        'total': np.mean([t['total'] for t in timing_data if 'total' in t]),
                        'import': np.mean([t['import'] for t in timing_data if 'import' in t]),
                        'init': np.mean([t['init'] for t in timing_data if 'init' in t]),
                        'backtest': np.mean([t['backtest'] for t in timing_data if 'backtest' in t])
                    }
                }
        
        # Standard performance metrics
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
    
    def _save_results(self):
        """Save results with debug information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(f'debug_backtest_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary
        if hasattr(self, 'summary'):
            with open(f'debug_backtest_summary_{timestamp}.json', 'w') as f:
                json.dump(self.summary, f, indent=2)
            
            # Create timing analysis CSV
            if 'timing_analysis' in self.summary and 'slowest_symbols' in self.summary['timing_analysis']:
                df = pd.DataFrame(self.summary['timing_analysis']['slowest_symbols'])
                df.to_csv(f'debug_timing_analysis_{timestamp}.csv', index=False)
        
        # Save worker logs
        logger.info(f"💾 Results saved to debug_backtest_*_{timestamp} files")
        logger.info(f"📝 Worker logs saved in logs/ directory")
    
    def print_summary(self):
        """Print enhanced summary with timing analysis."""
        if not hasattr(self, 'summary'):
            logger.warning("No summary available")
            return
        
        print("\n" + "="*80)
        print("S3 AI DEBUG PARALLEL BACKTEST SUMMARY")
        print("="*80)
        print(f"Total Symbols: {self.summary['total_symbols']}")
        print(f"Successful Backtests: {self.summary['successful_backtests']}")
        print(f"Failed Backtests: {self.summary['failed_backtests']}")
        print(f"No Trades Generated: {self.summary['no_trades']}")
        
        if self.summary.get('error_breakdown'):
            print("\nError Breakdown:")
            for error_type, count in self.summary['error_breakdown'].items():
                print(f"  {error_type}: {count}")
        
        if self.summary.get('timeout_errors'):
            print("\nTimeout Errors:")
            for timeout in self.summary['timeout_errors'][:5]:
                print(f"  {timeout['symbol']}: {timeout['timeout_seconds']:.1f}s")
        
        if 'timing_analysis' in self.summary:
            print("\n⏱️ Timing Analysis:")
            avg_times = self.summary['timing_analysis']['average_times']
            print(f"Average Total Time: {avg_times['total']:.2f}s")
            print(f"  - Import: {avg_times.get('import', 0):.2f}s")
            print(f"  - Init: {avg_times.get('init', 0):.2f}s")
            print(f"  - Backtest: {avg_times.get('backtest', 0):.2f}s")
            
            print("\nSlowest Symbols:")
            for symbol_timing in self.summary['timing_analysis']['slowest_symbols'][:5]:
                print(f"  {symbol_timing['symbol']:6} Total: {symbol_timing['total']:6.1f}s | " +
                      f"Import: {symbol_timing.get('import', 0):5.1f}s | " +
                      f"Init: {symbol_timing.get('init', 0):5.1f}s | " +
                      f"Backtest: {symbol_timing.get('backtest', 0):6.1f}s")
        
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
                print(f"  {perf['symbol']:6} Return: {perf['return']:7.2f}% | " +
                      f"Trades: {perf['trades']:3} | Win Rate: {perf['win_rate']:5.1f}%")
        
        print("="*80)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 AI Debug Parallel Backtester')
    parser.add_argument('--tickers', default='../james_tickers.json',
                       help='Path to tickers JSON file')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: min(4, CPU-1))')
    parser.add_argument('--batch-size', type=int, default=3,
                       help='Batch size per worker (default: 3)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tickers to process')
    parser.add_argument('--days', type=int, default=90,
                       help='Number of days for backtest period (default: 90)')
    
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
    backtester = DebugParallelBacktester(num_workers=args.workers)
    backtester.config['days'] = args.days
    logger.info(f"📅 Using {args.days} days for backtest period")
    backtester.run_parallel_backtest(tickers, batch_size=args.batch_size)
    
    # Print summary
    backtester.print_summary()


if __name__ == "__main__":
    main()