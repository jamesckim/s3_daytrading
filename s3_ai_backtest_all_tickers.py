#!/usr/bin/env python3
"""
S3 AI Backtester for All Tickers
================================
Runs comprehensive backtests on all tickers using the S3 AI strategy
with FMP historical data and AI-selected indicators.

Features:
- Processes 600+ tickers from all_tickers.json
- Uses FMP Ultimate for historical data
- AI-driven indicator selection and regime detection
- Comprehensive performance metrics
- Progress tracking and error handling
- Results saved to CSV and JSON formats
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# S3 AI components
from s3_ai_wrapper import S3AIWrapper
from s3_ai_fmp_ib_hybrid import FMPDataEngine, S3AIStrategy

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f's3_ai_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Trade record for backtesting."""
    timestamp: datetime
    symbol: str
    action: str
    price: float
    shares: int
    value: float
    signal_strength: float
    regime: str
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Backtest results for a symbol."""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    trades: List[BacktestTrade]
    equity_curve: List[float]
    error: str = None


class S3AIBacktester:
    """Backtester for S3 AI strategy across all tickers."""
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize backtester."""
        # Get FMP API key
        self.fmp_api_key = os.getenv("FMP_API_KEY")
        if not self.fmp_api_key:
            raise ValueError("FMP_API_KEY environment variable required")
        
        # Initialize components
        self.data_engine = FMPDataEngine(self.fmp_api_key)
        self.ai_wrapper = S3AIWrapper()
        self.strategy = S3AIStrategy(self.ai_wrapper, self.data_engine)
        
        # Backtest parameters
        self.initial_capital = initial_capital
        self.position_size = 10000  # Fixed position size
        self.max_positions = 10
        self.commission = 1.0  # Per trade commission
        self.slippage = 0.001  # 0.1% slippage
        
        # Results storage
        self.results = {}
        self.summary_stats = {}
        
        logger.info(f"🚀 S3 AI Backtester initialized with ${initial_capital:,.0f}")
    
    def load_tickers(self, tickers_file: str) -> List[str]:
        """Load tickers from JSON file."""
        try:
            with open(tickers_file, 'r') as f:
                tickers = json.load(f)
            logger.info(f"📋 Loaded {len(tickers)} tickers from {tickers_file}")
            return tickers
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            return []
    
    def get_historical_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Get historical daily data from FMP."""
        try:
            # Use daily data for backtesting
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
            params = {
                "apikey": self.fmp_api_key,
                "from": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "to": datetime.now().strftime("%Y-%m-%d")
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                if "historical" in data:
                    df = pd.DataFrame(data["historical"])
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                        df.set_index('date', inplace=True)
                        return df[['open', 'high', 'low', 'close', 'volume']]
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def backtest_symbol(self, symbol: str, data: pd.DataFrame) -> BacktestResult:
        """Run backtest for a single symbol."""
        try:
            if data.empty or len(data) < 50:
                return BacktestResult(
                    symbol=symbol,
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                    initial_capital=self.initial_capital,
                    final_capital=self.initial_capital,
                    total_return=0,
                    total_return_pct=0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0,
                    average_win=0,
                    average_loss=0,
                    profit_factor=0,
                    sharpe_ratio=0,
                    max_drawdown=0,
                    max_drawdown_pct=0,
                    trades=[],
                    equity_curve=[],
                    error="Insufficient data"
                )
            
            # Initialize backtest state
            capital = self.initial_capital
            position = None
            trades = []
            equity_curve = [capital]
            
            # Run through each day
            for i in range(50, len(data)):
                current_date = data.index[i]
                current_price = data['close'].iloc[i]
                
                # Get historical data up to current point
                hist_data = data.iloc[:i].tail(100)
                
                # Get AI signal
                action, signal_strength, info = self.strategy.analyze_symbol(symbol, current_price)
                
                # Execute trades based on signal
                if action == "BUY" and position is None and signal_strength >= 0.5:
                    # Enter position
                    shares = int(self.position_size / current_price)
                    if shares > 0:
                        entry_price = current_price * (1 + self.slippage)
                        cost = shares * entry_price + self.commission
                        
                        if cost <= capital:
                            position = {
                                'entry_date': current_date,
                                'entry_price': entry_price,
                                'shares': shares,
                                'signal_strength': signal_strength,
                                'regime': info.get('regime', 'Unknown')
                            }
                            capital -= cost
                
                elif action == "SELL" and position is not None:
                    # Exit position
                    exit_price = current_price * (1 - self.slippage)
                    proceeds = position['shares'] * exit_price - self.commission
                    
                    # Calculate P&L
                    pnl = proceeds - (position['shares'] * position['entry_price'])
                    pnl_pct = pnl / (position['shares'] * position['entry_price']) * 100
                    
                    # Record trade
                    trade = BacktestTrade(
                        timestamp=current_date,
                        symbol=symbol,
                        action="SELL",
                        price=exit_price,
                        shares=position['shares'],
                        value=proceeds,
                        signal_strength=signal_strength,
                        regime=info.get('regime', 'Unknown'),
                        pnl=pnl,
                        pnl_pct=pnl_pct
                    )
                    trades.append(trade)
                    
                    capital += proceeds
                    position = None
                
                # Update equity curve
                current_equity = capital
                if position:
                    current_equity += position['shares'] * current_price
                equity_curve.append(current_equity)
            
            # Close any open position at end
            if position:
                exit_price = data['close'].iloc[-1]
                proceeds = position['shares'] * exit_price - self.commission
                pnl = proceeds - (position['shares'] * position['entry_price'])
                pnl_pct = pnl / (position['shares'] * position['entry_price']) * 100
                
                trade = BacktestTrade(
                    timestamp=data.index[-1],
                    symbol=symbol,
                    action="SELL",
                    price=exit_price,
                    shares=position['shares'],
                    value=proceeds,
                    signal_strength=1.0,
                    regime="End of backtest",
                    pnl=pnl,
                    pnl_pct=pnl_pct
                )
                trades.append(trade)
                capital += proceeds
            
            # Calculate metrics
            final_capital = capital
            total_return = final_capital - self.initial_capital
            total_return_pct = total_return / self.initial_capital * 100
            
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
            average_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            average_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else 0
            
            # Calculate Sharpe ratio
            if len(equity_curve) > 1:
                returns = np.diff(equity_curve) / equity_curve[:-1]
                sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            peak = equity_curve[0]
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            
            max_drawdown_pct = max_dd * 100
            max_drawdown = max_dd * peak
            
            return BacktestResult(
                symbol=symbol,
                start_date=data.index[0],
                end_date=data.index[-1],
                initial_capital=self.initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                total_return_pct=total_return_pct,
                total_trades=len(trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                average_win=average_win,
                average_loss=average_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                trades=trades,
                equity_curve=equity_curve
            )
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            return BacktestResult(
                symbol=symbol,
                start_date=datetime.now(),
                end_date=datetime.now(),
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return=0,
                total_return_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                average_win=0,
                average_loss=0,
                profit_factor=0,
                sharpe_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                trades=[],
                equity_curve=[],
                error=str(e)
            )
    
    def run_backtest(self, tickers: List[str], save_results: bool = True):
        """Run backtest for all tickers."""
        logger.info(f"🏁 Starting backtest for {len(tickers)} tickers")
        
        start_time = time.time()
        successful_backtests = 0
        failed_backtests = 0
        
        for i, symbol in enumerate(tickers):
            try:
                # Progress update
                if i % 10 == 0:
                    logger.info(f"📊 Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")
                
                # Get historical data
                data = self.get_historical_data(symbol)
                
                if data.empty:
                    logger.warning(f"⚠️ No data for {symbol}")
                    failed_backtests += 1
                    continue
                
                # Run backtest
                result = self.backtest_symbol(symbol, data)
                self.results[symbol] = result
                
                if result.error:
                    failed_backtests += 1
                else:
                    successful_backtests += 1
                    logger.info(f"✅ {symbol}: Return: {result.total_return_pct:.2f}% | "
                              f"Trades: {result.total_trades} | Win Rate: {result.win_rate:.1f}%")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Failed to backtest {symbol}: {e}")
                failed_backtests += 1
        
        # Calculate summary statistics
        self._calculate_summary_stats()
        
        # Save results
        if save_results:
            self._save_results()
        
        elapsed_time = time.time() - start_time
        logger.info(f"🏁 Backtest complete in {elapsed_time:.1f} seconds")
        logger.info(f"✅ Successful: {successful_backtests} | ❌ Failed: {failed_backtests}")
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics across all backtests."""
        valid_results = [r for r in self.results.values() if not r.error and r.total_trades > 0]
        
        if not valid_results:
            logger.warning("No valid backtest results to summarize")
            return
        
        self.summary_stats = {
            'total_symbols': len(self.results),
            'successful_backtests': len(valid_results),
            'average_return': np.mean([r.total_return_pct for r in valid_results]),
            'median_return': np.median([r.total_return_pct for r in valid_results]),
            'best_return': max([r.total_return_pct for r in valid_results]),
            'worst_return': min([r.total_return_pct for r in valid_results]),
            'average_trades': np.mean([r.total_trades for r in valid_results]),
            'average_win_rate': np.mean([r.win_rate for r in valid_results]),
            'average_sharpe': np.mean([r.sharpe_ratio for r in valid_results]),
            'profitable_symbols': len([r for r in valid_results if r.total_return > 0]),
            'unprofitable_symbols': len([r for r in valid_results if r.total_return <= 0])
        }
        
        # Find top performers
        sorted_results = sorted(valid_results, key=lambda x: x.total_return_pct, reverse=True)
        self.summary_stats['top_10_symbols'] = [(r.symbol, r.total_return_pct) for r in sorted_results[:10]]
        self.summary_stats['bottom_10_symbols'] = [(r.symbol, r.total_return_pct) for r in sorted_results[-10:]]
    
    def _save_results(self):
        """Save backtest results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to JSON
        results_data = {}
        for symbol, result in self.results.items():
            result_dict = asdict(result)
            # Convert datetime objects to strings
            result_dict['start_date'] = str(result_dict['start_date'])
            result_dict['end_date'] = str(result_dict['end_date'])
            result_dict['trades'] = [
                {**asdict(t), 'timestamp': str(t.timestamp)}
                for t in result.trades
            ]
            results_data[symbol] = result_dict
        
        with open(f'backtest_results_{timestamp}.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary to CSV
        summary_df = pd.DataFrame([
            {
                'symbol': symbol,
                'return_pct': result.total_return_pct,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown_pct': result.max_drawdown_pct,
                'profit_factor': result.profit_factor
            }
            for symbol, result in self.results.items()
            if not result.error
        ])
        
        if not summary_df.empty:
            summary_df.to_csv(f'backtest_summary_{timestamp}.csv', index=False)
        
        # Save summary statistics
        with open(f'backtest_stats_{timestamp}.json', 'w') as f:
            json.dump(self.summary_stats, f, indent=2)
        
        logger.info(f"💾 Results saved to backtest_*_{timestamp} files")
    
    def print_summary(self):
        """Print summary of backtest results."""
        if not self.summary_stats:
            logger.warning("No summary statistics available")
            return
        
        print("\n" + "="*60)
        print("S3 AI BACKTEST SUMMARY")
        print("="*60)
        print(f"Total Symbols: {self.summary_stats['total_symbols']}")
        print(f"Successful Backtests: {self.summary_stats['successful_backtests']}")
        print(f"Average Return: {self.summary_stats['average_return']:.2f}%")
        print(f"Median Return: {self.summary_stats['median_return']:.2f}%")
        print(f"Best Return: {self.summary_stats['best_return']:.2f}%")
        print(f"Worst Return: {self.summary_stats['worst_return']:.2f}%")
        print(f"Average Win Rate: {self.summary_stats['average_win_rate']:.1f}%")
        print(f"Average Sharpe Ratio: {self.summary_stats['average_sharpe']:.2f}")
        print(f"Profitable Symbols: {self.summary_stats['profitable_symbols']}")
        print(f"Unprofitable Symbols: {self.summary_stats['unprofitable_symbols']}")
        
        print("\nTop 10 Performers:")
        for symbol, return_pct in self.summary_stats['top_10_symbols']:
            print(f"  {symbol}: {return_pct:.2f}%")
        
        print("\nBottom 10 Performers:")
        for symbol, return_pct in self.summary_stats['bottom_10_symbols']:
            print(f"  {symbol}: {return_pct:.2f}%")
        print("="*60)


def main():
    """Main execution function."""
    # Load tickers
    tickers_file = "../tickers/all_tickers.json"
    
    # Initialize backtester
    backtester = S3AIBacktester(initial_capital=100000)
    
    # Load tickers
    tickers = backtester.load_tickers(tickers_file)
    
    if not tickers:
        logger.error("No tickers loaded. Exiting.")
        return
    
    # For testing, you can limit the number of tickers
    # tickers = tickers[:10]  # Uncomment to test with first 10 tickers
    
    # Run backtest
    backtester.run_backtest(tickers, save_results=True)
    
    # Print summary
    backtester.print_summary()


if __name__ == "__main__":
    main()