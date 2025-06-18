#!/usr/bin/env python3
"""
S3 AI Database Backtester for All Tickers
=========================================
Runs comprehensive backtests on all tickers using the S3 AI strategy
with PostgreSQL minute-level historical data.

Features:
- Uses stocks_net_db PostgreSQL database for minute-level data
- AI-driven indicator selection and regime detection
- Intraday trading simulation with realistic fills
- Comprehensive performance metrics
- Memory-efficient streaming from database
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import warnings
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import pytz
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# S3 AI components
from s3_ai_wrapper import S3AIWrapper
from s3_ai_cache_manager import MinuteDataCache, BacktestResultCache

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f's3_ai_db_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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
    indicators: Dict[str, float]
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_time_minutes: int = 0


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
    average_hold_time: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    best_trade: Dict
    worst_trade: Dict
    trades: List[BacktestTrade]
    daily_returns: List[float]
    error: str = None


class DatabaseDataProvider:
    """Provides historical data from PostgreSQL database."""
    
    def __init__(self):
        """Initialize database connection."""
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            # Try constructing from individual components
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5432")
            db_name = os.getenv("DB_NAME", "stocks_net")
            db_user = os.getenv("DB_USER", "postgres")
            db_pass = os.getenv("DB_PASSWORD", "")
            
            self.db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        
        try:
            # Create engine with optimized connection pooling
            self.engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_pre_ping=True,
                pool_size=20,              # Increase from default 5
                max_overflow=40,           # Allow more overflow connections
                pool_timeout=30,           # Timeout for getting connection
                pool_recycle=3600,         # Recycle connections after 1 hour
                connect_args={
                    "options": "-c statement_timeout=300000"  # 5 min query timeout
                }
            )
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("✅ Database connection established with connection pooling")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
        
        # Initialize cache
        self.cache = MinuteDataCache(cache_dir="data_cache/minute_data")
    
    def get_minute_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get minute-level data from cache or database."""
        # Try cache first
        cached_data = self.cache.load_from_cache(symbol, start_date, end_date)
        if cached_data is not None:
            return cached_data
        
        # Load from database
        query = """
        SELECT 
            mp.datetime AT TIME ZONE 'UTC' AS timestamp,
            mp.open,
            mp.high,
            mp.low,
            mp.close,
            mp.volume
        FROM minute_prices mp
        JOIN tickers t ON mp.ticker_id = t.id
        WHERE t.symbol = %(symbol)s
          AND mp.datetime >= %(start_date)s
          AND mp.datetime <= %(end_date)s
        ORDER BY mp.datetime
        """
        
        try:
            df = pd.read_sql(
                query,
                self.engine,
                params={
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                },
                parse_dates=['timestamp']
            )
            
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                # Ensure timezone aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                # Save to cache for future use
                self.cache.save_to_cache(symbol, start_date, end_date, df)
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_minute_data_batch(self, symbols: List[str], start_date: datetime, 
                             end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Get minute-level data for multiple symbols in a single query.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result_dict = {}
        
        # Check cache first for each symbol
        symbols_to_query = []
        for symbol in symbols:
            cached_data = self.cache.load_from_cache(symbol, start_date, end_date)
            if cached_data is not None:
                result_dict[symbol] = cached_data
            else:
                symbols_to_query.append(symbol)
        
        if not symbols_to_query:
            return result_dict
        
        # Batch query for remaining symbols
        query = """
        SELECT 
            t.symbol,
            mp.datetime AT TIME ZONE 'UTC' AS timestamp,
            mp.open,
            mp.high,
            mp.low,
            mp.close,
            mp.volume
        FROM minute_prices mp
        JOIN tickers t ON mp.ticker_id = t.id
        WHERE t.symbol = ANY(%(symbols)s)
          AND mp.datetime >= %(start_date)s
          AND mp.datetime <= %(end_date)s
        ORDER BY t.symbol, mp.datetime
        """
        
        try:
            logger.info(f"📊 Batch loading {len(symbols_to_query)} symbols from database")
            
            df = pd.read_sql(
                query,
                self.engine,
                params={
                    'symbols': symbols_to_query,
                    'start_date': start_date,
                    'end_date': end_date
                },
                parse_dates=['timestamp']
            )
            
            if not df.empty:
                # Split by symbol and process
                for symbol, group in df.groupby('symbol'):
                    symbol_df = group.drop('symbol', axis=1)
                    symbol_df.set_index('timestamp', inplace=True)
                    
                    # Ensure timezone aware
                    if symbol_df.index.tz is None:
                        symbol_df.index = symbol_df.index.tz_localize('UTC')
                    
                    # Save to cache
                    self.cache.save_to_cache(symbol, start_date, end_date, symbol_df)
                    result_dict[symbol] = symbol_df
            
            logger.info(f"✅ Loaded data for {len(result_dict)} symbols")
            return result_dict
            
        except Exception as e:
            logger.error(f"Error in batch data fetch: {e}")
            # Fall back to individual queries
            for symbol in symbols_to_query:
                try:
                    data = self.get_minute_data(symbol, start_date, end_date)
                    if not data.empty:
                        result_dict[symbol] = data
                except Exception as sym_e:
                    logger.error(f"Error fetching {symbol}: {sym_e}")
            
            return result_dict
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with data in the database."""
        query = """
        SELECT DISTINCT t.symbol
        FROM tickers t
        JOIN minute_prices mp ON t.id = mp.ticker_id
        WHERE mp.datetime > NOW() - INTERVAL '30 days'
        ORDER BY t.symbol
        """
        
        try:
            result = pd.read_sql(query, self.engine)
            return result['symbol'].tolist()
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []


class S3AIMinuteStrategy:
    """S3 AI Strategy adapted for minute-level data."""
    
    def __init__(self, ai_wrapper: S3AIWrapper):
        self.ai_wrapper = ai_wrapper
        self.regime_cache = {}
        self.indicator_cache = {}
        self.eastern_tz = pytz.timezone('US/Eastern')
        
    def prepare_data_for_ai(self, minute_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare minute data for AI analysis by resampling."""
        # Resample to 5-minute bars for AI analysis
        resampled = minute_data.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def analyze_intraday(self, symbol: str, minute_data: pd.DataFrame, 
                        current_time: datetime) -> Tuple[str, float, Dict]:
        """Analyze symbol using minute-level data."""
        try:
            # Get recent data up to current time
            recent_data = minute_data[minute_data.index <= current_time].tail(500)
            
            if len(recent_data) < 100:
                return "HOLD", 0.0, {"reason": "insufficient_data"}
            
            # Prepare data for AI
            ai_data = self.prepare_data_for_ai(recent_data)
            
            if len(ai_data) < 20:
                return "HOLD", 0.0, {"reason": "insufficient_5min_bars"}
            
            # Get AI-selected indicators
            ai_result = self.ai_wrapper.select_indicators(ai_data, top_k=10)
            selected_indicators = ai_result.get('selected_indicators', {})
            regime = ai_result.get('regime_detection', 'Unknown')
            
            # Get current price
            current_price = recent_data['close'].iloc[-1]
            
            # Calculate intraday features
            features = self._calculate_intraday_features(recent_data, current_price)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(
                features, selected_indicators, regime, current_time
            )
            
            # Determine action
            action = self._determine_action(signal_strength, regime, features)
            
            # Build info dict
            info = {
                "regime": regime,
                "indicators": selected_indicators,
                "signal_strength": signal_strength,
                "current_price": current_price,
                "volume_ratio": features.get('volume_ratio', 1.0),
                "intraday_return": features.get('intraday_return', 0.0),
                "time_of_day": features.get('time_of_day', 'mid')
            }
            
            return action, abs(signal_strength), info
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return "HOLD", 0.0, {"error": str(e)}
    
    def _calculate_intraday_features(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Calculate intraday trading features."""
        features = {}
        
        # Price momentum
        features['return_5min'] = (current_price / data['close'].iloc[-6] - 1) if len(data) >= 6 else 0
        features['return_15min'] = (current_price / data['close'].iloc[-16] - 1) if len(data) >= 16 else 0
        features['return_30min'] = (current_price / data['close'].iloc[-31] - 1) if len(data) >= 31 else 0
        
        # Volume analysis
        recent_volume = data['volume'].iloc[-5:].mean()
        avg_volume = data['volume'].iloc[-60:].mean() if len(data) >= 60 else recent_volume
        features['volume_ratio'] = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Intraday metrics
        today_data = data[data.index.date == data.index[-1].date]
        if len(today_data) > 0:
            features['intraday_high'] = today_data['high'].max()
            features['intraday_low'] = today_data['low'].min()
            features['intraday_return'] = (current_price / today_data['open'].iloc[0] - 1)
            features['high_distance'] = (features['intraday_high'] - current_price) / current_price
            features['low_distance'] = (current_price - features['intraday_low']) / current_price
        
        # Time of day
        current_hour = data.index[-1].hour
        current_minute = data.index[-1].minute
        
        if current_hour < 10 or (current_hour == 10 and current_minute < 30):
            features['time_of_day'] = 'open'
        elif current_hour >= 15:
            features['time_of_day'] = 'close'
        else:
            features['time_of_day'] = 'mid'
        
        # Technical indicators on minute data
        if len(data) >= 20:
            features['sma_20'] = data['close'].rolling(20).mean().iloc[-1]
            features['price_vs_sma20'] = (current_price - features['sma_20']) / features['sma_20']
        
        return features
    
    def _calculate_signal_strength(self, features: Dict, indicators: Dict, 
                                 regime: str, current_time: datetime) -> float:
        """Calculate trading signal strength for intraday trading."""
        signal_strength = 0.0
        weights_sum = 0.0
        
        # Time-based adjustments
        hour = current_time.hour
        minute = current_time.minute
        
        # Avoid first and last 30 minutes
        if (hour == 9 and minute < 60) or (hour >= 15 and minute >= 30):
            return 0.0
        
        # Regime-based strategies
        if regime in ['Trending', 'Breakout']:
            # Momentum strategy
            if features.get('volume_ratio', 1.0) > 1.5:
                if features.get('return_5min', 0) > 0.002:  # 0.2% in 5 min
                    signal_strength += 0.4
                elif features.get('return_5min', 0) < -0.002:
                    signal_strength -= 0.4
                weights_sum += 1.0
        
        elif regime in ['Mean Reversion', 'High Volatility']:
            # Mean reversion strategy
            price_vs_sma = features.get('price_vs_sma20', 0)
            if abs(price_vs_sma) > 0.005:  # 0.5% deviation
                signal_strength -= price_vs_sma * 2  # Bet on reversion
                weights_sum += 1.0
        
        # Use AI indicators
        for indicator_name, indicator_info in indicators.items():
            weight = indicator_info.get('weight', 0.1)
            value = indicator_info.get('value', 0)
            
            # Apply indicator logic
            if 'RSI' in indicator_name:
                if value < 30:
                    signal_strength += weight * 0.3
                elif value > 70:
                    signal_strength -= weight * 0.3
            elif 'MACD' in indicator_name:
                if value > 0:
                    signal_strength += weight * 0.2
                else:
                    signal_strength -= weight * 0.2
            
            weights_sum += weight
        
        # Normalize
        if weights_sum > 0:
            signal_strength = signal_strength / weights_sum
        
        return max(-1.0, min(1.0, signal_strength))
    
    def _determine_action(self, signal_strength: float, regime: str, features: Dict) -> str:
        """Determine trading action for intraday trading."""
        # Higher threshold for intraday trading
        threshold = 0.4
        
        # Adjust threshold based on time of day
        if features.get('time_of_day') == 'close':
            # More conservative near close
            threshold = 0.6
        
        if abs(signal_strength) < threshold:
            return "HOLD"
        elif signal_strength > threshold:
            return "BUY"
        else:
            return "SELL"


class S3AIDBBacktester:
    """Backtester using PostgreSQL minute-level data."""
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize database backtester."""
        # Initialize components
        self.db_provider = DatabaseDataProvider()
        self.ai_wrapper = S3AIWrapper()
        self.strategy = S3AIMinuteStrategy(self.ai_wrapper)
        
        # Backtest parameters
        self.initial_capital = initial_capital
        self.position_size_pct = 0.02  # 2% of capital per position
        self.max_positions = 10
        self.commission = 1.0  # Per trade commission
        self.slippage = 0.0001  # 0.01% slippage
        
        # Intraday parameters
        self.max_hold_minutes = 120  # Maximum 2 hours hold time
        self.stop_loss = 0.01  # 1% stop loss
        self.take_profit = 0.02  # 2% take profit
        
        # Market hours (Eastern Time)
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(16, 0)
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        # Results storage
        self.results = {}
        self.summary_stats = {}
        
        logger.info(f"🚀 S3 AI Database Backtester initialized with ${initial_capital:,.0f}")
    
    def backtest_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Run backtest for a single symbol using minute data."""
        try:
            # Get minute data from database
            data = self.db_provider.get_minute_data(symbol, start_date, end_date)
            
            if data.empty or len(data) < 1000:  # Need at least 1000 minutes
                return self._create_empty_result(symbol, start_date, end_date, "Insufficient data")
            
            # Initialize backtest state
            capital = self.initial_capital
            positions = []  # Can have multiple positions
            trades = []
            daily_capital = {start_date.date(): capital}
            
            # Group by trading day
            for date, day_data in data.groupby(data.index.date):
                # Skip if not enough data for the day
                if len(day_data) < 100:
                    continue
                
                # Process each minute
                for timestamp in day_data.index:
                    # Convert to Eastern time for market hours check
                    et_time = timestamp.tz_convert(self.eastern_tz)
                    
                    # Check market hours
                    if not (self.market_open <= et_time.time() <= self.market_close):
                        continue
                    
                    # Check and close positions
                    positions = self._check_close_positions(
                        positions, day_data, timestamp, capital, trades
                    )
                    
                    # Update capital after closes
                    capital = self.initial_capital + sum(t.pnl for t in trades)
                    
                    # Check for new signals (limit positions)
                    if len(positions) < self.max_positions:
                        # Only check every 5 minutes to avoid overtrading
                        if timestamp.minute % 5 == 0:
                            action, signal_strength, info = self.strategy.analyze_intraday(
                                symbol, day_data, timestamp
                            )
                            
                            if action == "BUY" and signal_strength >= 0.5:
                                position = self._open_position(
                                    symbol, day_data, timestamp, capital, 
                                    signal_strength, info
                                )
                                if position:
                                    positions.append(position)
                
                # Force close all positions at end of day
                for pos in positions:
                    close_price = day_data['close'].iloc[-1]
                    trade = self._close_position(pos, close_price, day_data.index[-1], "EOD")
                    trades.append(trade)
                    capital += trade.pnl
                
                positions = []  # Clear all positions
                
                # Record end of day capital
                daily_capital[date] = capital
            
            # Calculate metrics
            return self._calculate_results(symbol, start_date, end_date, trades, daily_capital)
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            return self._create_empty_result(symbol, start_date, end_date, str(e))
    
    def _check_close_positions(self, positions: List[Dict], data: pd.DataFrame, 
                             current_time: datetime, capital: float, trades: List) -> List[Dict]:
        """Check if any positions should be closed."""
        remaining_positions = []
        
        for position in positions:
            current_price = data.loc[current_time, 'close']
            hold_time = (current_time - position['entry_time']).seconds / 60
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            
            should_close = False
            close_reason = ""
            
            # Check exit conditions
            if pnl_pct <= -self.stop_loss:
                should_close = True
                close_reason = "StopLoss"
            elif pnl_pct >= self.take_profit:
                should_close = True
                close_reason = "TakeProfit"
            elif hold_time >= self.max_hold_minutes:
                should_close = True
                close_reason = "TimeLimit"
            else:
                # Check for sell signal
                action, strength, info = self.strategy.analyze_intraday(
                    position['symbol'], data, current_time
                )
                if action == "SELL" and strength >= 0.3:
                    should_close = True
                    close_reason = "SellSignal"
            
            if should_close:
                trade = self._close_position(position, current_price, current_time, close_reason)
                trades.append(trade)
            else:
                remaining_positions.append(position)
        
        return remaining_positions
    
    def _open_position(self, symbol: str, data: pd.DataFrame, timestamp: datetime,
                      capital: float, signal_strength: float, info: Dict) -> Optional[Dict]:
        """Open a new position."""
        try:
            current_price = data.loc[timestamp, 'close']
            
            # Calculate position size
            position_value = capital * self.position_size_pct
            shares = int(position_value / current_price)
            
            if shares < 1:
                return None
            
            # Apply slippage
            entry_price = current_price * (1 + self.slippage)
            cost = shares * entry_price + self.commission
            
            if cost > capital * 0.1:  # Don't use more than 10% of capital
                return None
            
            return {
                'symbol': symbol,
                'entry_time': timestamp,
                'entry_price': entry_price,
                'shares': shares,
                'signal_strength': signal_strength,
                'regime': info.get('regime', 'Unknown'),
                'indicators': info.get('indicators', {})
            }
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    def _close_position(self, position: Dict, close_price: float, 
                       close_time: datetime, reason: str) -> BacktestTrade:
        """Close a position and create trade record."""
        # Apply slippage
        exit_price = close_price * (1 - self.slippage)
        
        # Calculate P&L
        gross_pnl = (exit_price - position['entry_price']) * position['shares']
        net_pnl = gross_pnl - self.commission * 2  # Entry and exit commissions
        pnl_pct = net_pnl / (position['entry_price'] * position['shares']) * 100
        
        # Calculate hold time
        hold_time = int((close_time - position['entry_time']).seconds / 60)
        
        return BacktestTrade(
            timestamp=close_time,
            symbol=position['symbol'],
            action=f"CLOSE_{reason}",
            price=exit_price,
            shares=position['shares'],
            value=exit_price * position['shares'],
            signal_strength=position['signal_strength'],
            regime=position['regime'],
            indicators=position['indicators'],
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            hold_time_minutes=hold_time
        )
    
    def _calculate_results(self, symbol: str, start_date: datetime, end_date: datetime,
                          trades: List[BacktestTrade], daily_capital: Dict) -> BacktestResult:
        """Calculate backtest results from trades."""
        if not trades:
            return self._create_empty_result(symbol, start_date, end_date, "No trades")
        
        # Calculate metrics
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
        
        # Calculate daily returns for Sharpe ratio
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
        
        max_drawdown_pct = max_dd * 100
        max_drawdown = max_dd * peak
        
        # Best and worst trades
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
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            best_trade=asdict(best_trade) if best_trade else {},
            worst_trade=asdict(worst_trade) if worst_trade else {},
            trades=trades,
            daily_returns=daily_returns.tolist() if len(daily_returns) > 0 else []
        )
    
    def _create_empty_result(self, symbol: str, start_date: datetime, 
                           end_date: datetime, error: str) -> BacktestResult:
        """Create empty result for failed backtest."""
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
    
    def run_backtest(self, tickers: List[str], start_date: datetime, 
                    end_date: datetime, save_results: bool = True):
        """Run backtest for all tickers."""
        logger.info(f"🏁 Starting database backtest for {len(tickers)} tickers")
        logger.info(f"📅 Period: {start_date.date()} to {end_date.date()}")
        
        start_time = time.time()
        successful_backtests = 0
        failed_backtests = 0
        
        for i, symbol in enumerate(tickers):
            try:
                # Progress update
                if i % 10 == 0:
                    logger.info(f"📊 Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")
                
                # Run backtest
                result = self.backtest_symbol(symbol, start_date, end_date)
                self.results[symbol] = result
                
                if result.error:
                    failed_backtests += 1
                    logger.warning(f"⚠️ {symbol}: {result.error}")
                else:
                    successful_backtests += 1
                    logger.info(f"✅ {symbol}: Return: {result.total_return_pct:.2f}% | "
                              f"Trades: {result.total_trades} | Win Rate: {result.win_rate:.1f}% | "
                              f"Avg Hold: {result.average_hold_time:.0f}min")
                
            except Exception as e:
                logger.error(f"❌ Failed to backtest {symbol}: {e}")
                failed_backtests += 1
        
        # Calculate summary statistics
        self._calculate_summary_stats()
        
        # Save results
        if save_results:
            self._save_results()
        
        elapsed_time = time.time() - start_time
        logger.info(f"🏁 Backtest complete in {elapsed_time/60:.1f} minutes")
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
            'average_hold_time': np.mean([r.average_hold_time for r in valid_results]),
            'average_sharpe': np.mean([r.sharpe_ratio for r in valid_results]),
            'profitable_symbols': len([r for r in valid_results if r.total_return > 0]),
            'unprofitable_symbols': len([r for r in valid_results if r.total_return <= 0])
        }
        
        # Find top performers
        sorted_results = sorted(valid_results, key=lambda x: x.total_return_pct, reverse=True)
        self.summary_stats['top_10_symbols'] = [
            {
                'symbol': r.symbol,
                'return_pct': r.total_return_pct,
                'trades': r.total_trades,
                'win_rate': r.win_rate,
                'avg_hold_time': r.average_hold_time
            }
            for r in sorted_results[:10]
        ]
        self.summary_stats['bottom_10_symbols'] = [
            {
                'symbol': r.symbol,
                'return_pct': r.total_return_pct,
                'trades': r.total_trades,
                'win_rate': r.win_rate,
                'avg_hold_time': r.average_hold_time
            }
            for r in sorted_results[-10:]
        ]
    
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
        
        with open(f'db_backtest_results_{timestamp}.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary to CSV
        summary_rows = []
        for symbol, result in self.results.items():
            if not result.error:
                summary_rows.append({
                    'symbol': symbol,
                    'return_pct': result.total_return_pct,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'avg_hold_time': result.average_hold_time,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown_pct': result.max_drawdown_pct,
                    'profit_factor': result.profit_factor
                })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(f'db_backtest_summary_{timestamp}.csv', index=False)
        
        # Save summary statistics
        if hasattr(self, 'summary_stats'):
            with open(f'db_backtest_stats_{timestamp}.json', 'w') as f:
                json.dump(self.summary_stats, f, indent=2)
        
        logger.info(f"💾 Results saved to db_backtest_*_{timestamp} files")
    
    def print_summary(self):
        """Print summary of backtest results."""
        if not hasattr(self, 'summary_stats') or not self.summary_stats:
            logger.warning("No summary statistics available")
            return
        
        print("\n" + "="*60)
        print("S3 AI DATABASE BACKTEST SUMMARY")
        print("="*60)
        print(f"Total Symbols: {self.summary_stats['total_symbols']}")
        print(f"Successful Backtests: {self.summary_stats['successful_backtests']}")
        print(f"Average Return: {self.summary_stats['average_return']:.2f}%")
        print(f"Median Return: {self.summary_stats['median_return']:.2f}%")
        print(f"Best Return: {self.summary_stats['best_return']:.2f}%")
        print(f"Worst Return: {self.summary_stats['worst_return']:.2f}%")
        print(f"Average Win Rate: {self.summary_stats['average_win_rate']:.1f}%")
        print(f"Average Hold Time: {self.summary_stats['average_hold_time']:.1f} minutes")
        print(f"Average Sharpe Ratio: {self.summary_stats['average_sharpe']:.2f}")
        print(f"Profitable Symbols: {self.summary_stats['profitable_symbols']}")
        print(f"Unprofitable Symbols: {self.summary_stats['unprofitable_symbols']}")
        
        print("\nTop 10 Performers:")
        for perf in self.summary_stats.get('top_10_symbols', []):
            print(f"  {perf['symbol']:6} Return: {perf['return_pct']:7.2f}% | "
                  f"Trades: {perf['trades']:3} | Win Rate: {perf['win_rate']:5.1f}% | "
                  f"Hold: {perf['avg_hold_time']:3.0f}min")
        
        print("\nBottom 10 Performers:")
        for perf in self.summary_stats.get('bottom_10_symbols', []):
            print(f"  {perf['symbol']:6} Return: {perf['return_pct']:7.2f}% | "
                  f"Trades: {perf['trades']:3} | Win Rate: {perf['win_rate']:5.1f}% | "
                  f"Hold: {perf['avg_hold_time']:3.0f}min")
        print("="*60)


def main():
    """Main execution function."""
    # Load tickers
    tickers_file = "../tickers/all_tickers.json"
    with open(tickers_file, 'r') as f:
        all_tickers = json.load(f)
    
    # Get available symbols from database
    backtester = S3AIDBBacktester(initial_capital=100000)
    db_symbols = backtester.db_provider.get_available_symbols()
    
    # Filter tickers to only those available in database
    available_tickers = [t for t in all_tickers if t in db_symbols]
    logger.info(f"📊 Found {len(available_tickers)} tickers with data in database")
    
    # Set backtest period (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # For testing, limit to first 10 tickers
    # Remove this line for full backtest
    available_tickers = available_tickers[:10]
    
    # Run backtest
    backtester.run_backtest(available_tickers, start_date, end_date, save_results=True)
    
    # Print summary
    backtester.print_summary()


if __name__ == "__main__":
    main()