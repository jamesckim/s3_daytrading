#!/usr/bin/env python3
"""
S3 AI Minute Strategy V2
========================
Enhanced version that generates more trades with minute-level data.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Any, List, Optional, Tuple
import logging
import pytz
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_ai_wrapper import S3AIWrapper
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider, BacktestTrade, BacktestResult, BacktestTrade
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class S3AIMinuteStrategyV2:
    """Enhanced S3 AI Strategy for minute-level trading."""
    
    def __init__(self, ai_wrapper: S3AIWrapper):
        self.ai_wrapper = ai_wrapper
        self.eastern_tz = pytz.timezone('US/Eastern')
        
    def analyze_intraday(self, symbol: str, minute_data: pd.DataFrame, 
                        current_time: datetime) -> Tuple[str, float, Dict]:
        """Analyze symbol using minute-level data with enhanced signals."""
        try:
            # Get recent data
            recent_data = minute_data[minute_data.index <= current_time].tail(100)
            
            if len(recent_data) < 20:
                return "HOLD", 0.0, {"reason": "insufficient_data"}
            
            # Calculate minute-level indicators
            features = self._calculate_minute_features(recent_data)
            
            # Determine action based on minute patterns
            action, signal_strength = self._minute_signal_logic(features)
            
            # Build info
            info = {
                "features": features,
                "signal_strength": signal_strength,
                "current_price": recent_data['close'].iloc[-1]
            }
            
            return action, abs(signal_strength), info
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return "HOLD", 0.0, {"error": str(e)}
    
    def _calculate_minute_features(self, data: pd.DataFrame) -> Dict:
        """Calculate minute-level trading features."""
        features = {}
        
        # Price momentum (1, 5, 10 minute returns)
        current_price = data['close'].iloc[-1]
        features['return_1min'] = (current_price / data['close'].iloc[-2] - 1) if len(data) >= 2 else 0
        features['return_5min'] = (current_price / data['close'].iloc[-6] - 1) if len(data) >= 6 else 0
        features['return_10min'] = (current_price / data['close'].iloc[-11] - 1) if len(data) >= 11 else 0
        
        # Volume surge detection
        recent_vol = data['volume'].iloc[-5:].mean()
        avg_vol = data['volume'].iloc[-20:].mean()
        features['volume_surge'] = recent_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Price breakout detection
        high_10min = data['high'].iloc[-10:].max()
        low_10min = data['low'].iloc[-10:].min()
        features['near_high'] = (current_price - high_10min) / high_10min
        features['near_low'] = (current_price - low_10min) / low_10min
        
        # Simple moving averages
        features['sma_5'] = data['close'].rolling(5).mean().iloc[-1]
        features['sma_10'] = data['close'].rolling(10).mean().iloc[-1]
        features['price_vs_sma5'] = (current_price - features['sma_5']) / features['sma_5']
        
        # Volatility
        features['volatility'] = data['close'].pct_change().rolling(10).std().iloc[-1]
        
        # RSI-like indicator (simplified)
        changes = data['close'].diff()
        gains = changes.where(changes > 0, 0).rolling(14).mean()
        losses = -changes.where(changes < 0, 0).rolling(14).mean()
        rs = gains.iloc[-1] / losses.iloc[-1] if losses.iloc[-1] > 0 else 100
        features['rsi'] = 100 - (100 / (1 + rs))
        
        return features
    
    def _minute_signal_logic(self, features: Dict) -> Tuple[str, float]:
        """Generate trading signals from minute-level features."""
        signal_strength = 0.0
        
        # Momentum signals
        if features['return_1min'] > 0.001 and features['return_5min'] > 0.002:
            # Upward momentum
            signal_strength += 0.3
            if features['volume_surge'] > 1.5:
                signal_strength += 0.2
        elif features['return_1min'] < -0.001 and features['return_5min'] < -0.002:
            # Downward momentum
            signal_strength -= 0.3
            if features['volume_surge'] > 1.5:
                signal_strength -= 0.2
        
        # Breakout signals
        if features['near_high'] > -0.001:  # Within 0.1% of 10-min high
            signal_strength += 0.2
        elif features['near_low'] < 0.001:  # Within 0.1% of 10-min low
            signal_strength -= 0.2
        
        # Mean reversion signals (counter-trend)
        if features['price_vs_sma5'] > 0.003:  # 0.3% above SMA
            signal_strength -= 0.15
        elif features['price_vs_sma5'] < -0.003:  # 0.3% below SMA
            signal_strength += 0.15
        
        # RSI signals
        if features.get('rsi', 50) < 30:
            signal_strength += 0.15
        elif features.get('rsi', 50) > 70:
            signal_strength -= 0.15
        
        # Volatility adjustment
        if features.get('volatility', 0) > 0.002:  # High volatility
            signal_strength *= 1.2  # Amplify signals
        
        # Determine action
        signal_strength = max(-1.0, min(1.0, signal_strength))
        
        if signal_strength > 0.25:
            return "BUY", signal_strength
        elif signal_strength < -0.25:
            return "SELL", signal_strength
        else:
            return "HOLD", signal_strength


class EnhancedS3AIDBBacktester:
    """Enhanced backtester with the new minute strategy."""
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize enhanced backtester."""
        self.db_provider = DatabaseDataProvider()
        self.ai_wrapper = S3AIWrapper()
        self.strategy = S3AIMinuteStrategyV2(self.ai_wrapper)
        
        # Backtest parameters
        self.initial_capital = initial_capital
        self.position_size_pct = 0.02  # 2% positions (more conservative)
        self.max_positions = 3  # Limit concurrent positions
        self.commission = 1.0
        self.slippage = 0.0005  # Higher slippage for minute bars
        
        # Intraday parameters
        self.max_hold_minutes = 30  # Shorter hold time
        self.stop_loss = 0.005  # 0.5% stop loss
        self.take_profit = 0.01  # 1% take profit
        
        # Market hours
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(16, 0)
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        self.results = {}
        
    def backtest_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Run backtest for a single symbol."""
        try:
            # Get minute data
            data = self.db_provider.get_minute_data(symbol, start_date, end_date)
            
            if data.empty or len(data) < 100:
                return self._create_empty_result(symbol, start_date, end_date, "Insufficient data")
            
            # Initialize state
            capital = self.initial_capital
            positions = []
            trades = []
            daily_capital = {start_date.date(): capital}
            total_position_value = 0  # Track total value in positions
            
            # Process each day
            for date, day_data in data.groupby(data.index.date):
                if len(day_data) < 50:
                    continue
                
                # Process each minute
                for i, timestamp in enumerate(day_data.index):
                    # Convert to Eastern time
                    et_time = timestamp.tz_convert(self.eastern_tz)
                    
                    # Skip outside market hours
                    if not (self.market_open <= et_time.time() <= self.market_close):
                        continue
                    
                    # Skip first 10 minutes of day
                    if i < 10:
                        continue
                    
                    # Check and close positions
                    positions, closed_trades = self._check_close_positions(
                        positions, day_data, timestamp
                    )
                    trades.extend(closed_trades)
                    
                    # DON'T update capital during trading - use fixed position sizing
                    # This prevents unrealistic compounding
                    
                    # Check for new signals every minute (more frequent)
                    if len(positions) < self.max_positions:
                        # Calculate current position value
                        total_position_value = sum(pos['shares'] * pos['entry_price'] for pos in positions)
                        
                        # Don't open new positions if we're already heavily invested
                        if total_position_value < self.initial_capital * 0.5:  # Max 50% invested
                            action, signal_strength, info = self.strategy.analyze_intraday(
                                symbol, day_data, timestamp
                            )
                            
                            if action == "BUY" and signal_strength >= 0.25:
                                # Use initial capital for position sizing to avoid compounding
                                position = self._open_position(
                                    symbol, day_data, timestamp, self.initial_capital, 
                                    signal_strength, info
                                )
                                if position:
                                    positions.append(position)
                
                # Close all positions at end of day
                for pos in positions:
                    close_price = day_data['close'].iloc[-1]
                    trade = self._close_position(pos, close_price, day_data.index[-1], "EOD")
                    trades.append(trade)
                
                positions = []
                # Calculate capital at end of day
                capital = self.initial_capital + sum(t.pnl for t in trades)
                daily_capital[date] = capital
            
            # Calculate results
            return self._calculate_results(symbol, start_date, end_date, trades, daily_capital)
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            return self._create_empty_result(symbol, start_date, end_date, str(e))
    
    def _check_close_positions(self, positions: List[Dict], data: pd.DataFrame, 
                             current_time: datetime) -> Tuple[List[Dict], List[BacktestTrade]]:
        """Check if positions should be closed."""
        remaining_positions = []
        closed_trades = []
        
        for position in positions:
            current_price = data.loc[current_time, 'close']
            hold_time = (current_time - position['entry_time']).seconds / 60
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            
            should_close = False
            close_reason = ""
            
            # Exit conditions
            if pnl_pct <= -self.stop_loss:
                should_close = True
                close_reason = "StopLoss"
            elif pnl_pct >= self.take_profit:
                should_close = True
                close_reason = "TakeProfit"
            elif hold_time >= self.max_hold_minutes:
                should_close = True
                close_reason = "TimeLimit"
            
            if should_close:
                trade = self._close_position(position, current_price, current_time, close_reason)
                closed_trades.append(trade)
            else:
                remaining_positions.append(position)
        
        return remaining_positions, closed_trades
    
    def _open_position(self, symbol: str, data: pd.DataFrame, timestamp: datetime,
                      capital: float, signal_strength: float, info: Dict) -> Optional[Dict]:
        """Open a new position."""
        try:
            current_price = data.loc[timestamp, 'close']
            
            # Calculate position size - always use initial capital to prevent compounding
            position_value = capital * self.position_size_pct
            shares = int(position_value / current_price)
            
            if shares < 1:
                return None
            
            # Apply slippage
            entry_price = current_price * (1 + self.slippage)
            cost = shares * entry_price + self.commission
            
            # Sanity check - position shouldn't exceed reasonable limits
            if cost > self.initial_capital * 0.10:  # Max 10% per position
                shares = int((self.initial_capital * 0.10 - self.commission) / entry_price)
                if shares < 1:
                    return None
                cost = shares * entry_price + self.commission
            
            return {
                'symbol': symbol,
                'entry_time': timestamp,
                'entry_price': entry_price,
                'shares': shares,
                'signal_strength': signal_strength,
                'features': info.get('features', {})
            }
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    def _close_position(self, position: Dict, close_price: float, 
                       close_time: datetime, reason: str) -> BacktestTrade:
        """Close a position."""
        exit_price = close_price * (1 - self.slippage)
        
        gross_pnl = (exit_price - position['entry_price']) * position['shares']
        net_pnl = gross_pnl - self.commission * 2
        pnl_pct = net_pnl / (position['entry_price'] * position['shares']) * 100
        
        hold_time = int((close_time - position['entry_time']).seconds / 60)
        
        return BacktestTrade(
            timestamp=close_time,
            symbol=position['symbol'],
            action=f"CLOSE_{reason}",
            price=exit_price,
            shares=position['shares'],
            value=exit_price * position['shares'],
            signal_strength=position['signal_strength'],
            regime="Minute",
            indicators=position.get('features', {}),
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            hold_time_minutes=hold_time
        )
    
    def _calculate_results(self, symbol: str, start_date: datetime, end_date: datetime,
                          trades: List[BacktestTrade], daily_capital: Dict) -> BacktestResult:
        """Calculate backtest results."""
        if not trades:
            return self._create_empty_result(symbol, start_date, end_date, "No trades")
        
        final_capital = self.initial_capital + sum(t.pnl for t in trades)
        total_return = final_capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital * 100
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        average_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        average_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        average_hold_time = np.mean([t.hold_time_minutes for t in trades])
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else 0
        
        # Calculate daily returns
        dates = sorted(daily_capital.keys())
        if len(dates) > 1:
            daily_values = [daily_capital[d] for d in dates]
            daily_returns = np.diff(daily_values) / daily_values[:-1]
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
        else:
            daily_returns = []
            sharpe_ratio = 0
        
        # Calculate max drawdown
        equity_curve = list(daily_capital.values())
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        best_trade = max(trades, key=lambda t: t.pnl_pct) if trades else None
        worst_trade = min(trades, key=lambda t: t.pnl_pct) if trades else None
        
        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
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
            average_hold_time=average_hold_time,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_dd * peak,
            max_drawdown_pct=max_dd * 100,
            best_trade=asdict(best_trade) if best_trade else {},
            worst_trade=asdict(worst_trade) if worst_trade else {},
            trades=trades,
            daily_returns=daily_returns.tolist() if len(daily_returns) > 0 else []
        )
    
    def _create_empty_result(self, symbol: str, start_date: datetime, 
                           end_date: datetime, error: str) -> BacktestResult:
        """Create empty result."""
        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
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
            average_hold_time=0,
            profit_factor=0,
            sharpe_ratio=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            best_trade={},
            worst_trade={},
            trades=[],
            daily_returns=[],
            error=error
        )


def main():
    """Test the enhanced strategy."""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test with a few symbols
    backtester = EnhancedS3AIDBBacktester(initial_capital=100000)
    
    # Test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    test_symbols = ["AAPL", "TSLA", "NVDA", "AMZN", "MSFT"]
    
    print(f"\n🚀 Testing Enhanced Minute Strategy")
    print(f"📅 Period: {start_date.date()} to {end_date.date()}")
    print(f"📊 Symbols: {', '.join(test_symbols)}")
    print("-" * 60)
    
    total_trades = 0
    profitable_symbols = 0
    
    for symbol in test_symbols:
        print(f"\n📈 Testing {symbol}...")
        result = backtester.backtest_symbol(symbol, start_date, end_date)
        
        if result.error:
            print(f"❌ Error: {result.error}")
        else:
            print(f"✅ Return: {result.total_return_pct:.2f}%")
            print(f"   Trades: {result.total_trades}")
            print(f"   Win Rate: {result.win_rate:.1f}%")
            print(f"   Avg Hold: {result.average_hold_time:.1f} minutes")
            
            total_trades += result.total_trades
            if result.total_return > 0:
                profitable_symbols += 1
            
            # Show sample trades
            if result.trades:
                print(f"   First Trade: {result.trades[0].timestamp.strftime('%H:%M')} - "
                      f"P&L: ${result.trades[0].pnl:.2f} ({result.trades[0].pnl_pct:.2f}%)")
                if len(result.trades) > 1:
                    print(f"   Last Trade: {result.trades[-1].timestamp.strftime('%H:%M')} - "
                          f"P&L: ${result.trades[-1].pnl:.2f} ({result.trades[-1].pnl_pct:.2f}%)")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_trades} total trades across {len(test_symbols)} symbols")
    print(f"Profitable symbols: {profitable_symbols}/{len(test_symbols)}")
    print("=" * 60)


if __name__ == "__main__":
    main()