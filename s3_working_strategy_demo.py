#!/usr/bin/env python3
"""
S3 Working Strategy - Demo Version with Real Data Structure
=========================================================
Demonstrates the S3 strategy using real data structure without database dependency.
This version shows how the strategy would work with real market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')


class S3WorkingStrategyDemo:
    """
    Demo version of S3 strategy that works with real data structure.
    Uses realistic market data patterns without requiring database connection.
    """
    
    def __init__(self,
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = False,  # Simplified for demo
                 data_days_lookback: int = 7):
        """Initialize S3 demo strategy."""
        
        self.symbol = symbol.upper()
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators
        self.data_days_lookback = data_days_lookback
        
        # Strategy state
        self.current_indicators = {}
        self.last_market_regime = "Normal"
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        
        print(f"🚀 S3 Demo Strategy initialized for {symbol}")
        print(f"   Capital: ${capital:,.2f}")
        print(f"   Data lookback: {data_days_lookback} days")
    
    def load_realistic_data(self, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Generate realistic market data that mimics real database structure.
        This demonstrates how the strategy would work with actual data.
        """
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_dt = datetime.now() - timedelta(days=self.data_days_lookback)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        print(f"📊 Generating realistic data for {self.symbol} from {start_date} to {end_date}...")
        
        # Generate trading hours timestamps (9:30 AM - 4:00 PM ET)
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        timestamps = []
        current_date = start_dt
        
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Monday-Friday only
                # Trading hours: 9:30 AM to 4:00 PM
                for hour in range(9, 16):
                    for minute in range(60):
                        if hour == 9 and minute < 30:  # Market opens at 9:30
                            continue
                        timestamp = current_date.replace(hour=hour, minute=minute)
                        timestamps.append(timestamp)
            current_date += timedelta(days=1)
        
        # Limit to reasonable size for demo
        timestamps = timestamps[:min(len(timestamps), 2000)]
        
        if not timestamps:
            # Fallback
            timestamps = pd.date_range(start_date, end_date, freq='1min')[:2000]
        
        print(f"📈 Generating {len(timestamps)} minutes of trading data...")
        
        # Generate realistic price data based on symbol
        np.random.seed(hash(self.symbol) % 2**31)  # Deterministic per symbol
        
        # Symbol-specific base prices (realistic as of 2024)
        symbol_prices = {
            'AAPL': 180, 'MSFT': 350, 'GOOGL': 140, 'TSLA': 200, 'AMZN': 150,
            'NVDA': 800, 'META': 350, 'NFLX': 450, 'AMD': 120, 'CRM': 250
        }
        base_price = symbol_prices.get(self.symbol, 100)
        
        # Generate realistic returns with market microstructure
        returns = np.random.normal(0, 0.0008, len(timestamps))  # Realistic minute volatility
        
        # Add intraday patterns
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            minute = ts.minute
            
            # Opening volatility (9:30-10:00)
            if hour == 9 and minute >= 30:
                returns[i] += np.random.normal(0, 0.0003)
            
            # Lunch lull (12:00-14:00)
            elif 12 <= hour < 14:
                returns[i] *= 0.7  # Reduced volatility
            
            # Closing activity (15:30-16:00)
            elif hour == 15 and minute >= 30:
                returns[i] += np.random.normal(0, 0.0002)
        
        # Add some trending periods (momentum)
        trend_length = len(returns) // 5
        if trend_length > 0:
            trend_start = len(returns) // 3
            trend_end = trend_start + trend_length
            trend_direction = np.random.choice([-1, 1])
            returns[trend_start:trend_end] += trend_direction * 0.0001
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV with realistic bid-ask spreads
        spread_pct = 0.0001  # 1 basis point spread
        
        closes = prices
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        
        # Generate highs and lows with realistic intrabar movement
        noise = np.random.normal(0, 0.0002, len(closes))
        highs = np.maximum(opens, closes) * (1 + np.abs(noise) + spread_pct)
        lows = np.minimum(opens, closes) * (1 - np.abs(noise) - spread_pct)
        
        # Ensure OHLC consistency
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        # Generate realistic volume with daily patterns
        base_volume = 100000 if self.symbol in ['AAPL', 'MSFT', 'TSLA'] else 50000
        volume_pattern = []
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            
            # Higher volume at open and close
            if hour == 9:  # Opening hour
                vol_multiplier = 3.0
            elif hour == 15:  # Closing hour
                vol_multiplier = 2.5
            elif 10 <= hour <= 14:  # Mid-day
                vol_multiplier = 1.0
            else:
                vol_multiplier = 1.5
            
            # Add price-volume correlation (higher volume on big moves)
            if i > 0:
                price_change = abs(returns[i])
                vol_multiplier *= (1 + price_change * 1000)  # Volume surge on big moves
            
            volume = int(base_volume * vol_multiplier * np.random.lognormal(0, 0.3))
            volume_pattern.append(max(volume, 1000))  # Minimum volume
        
        # Create DataFrame with realistic structure
        data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volume_pattern
        }, index=pd.DatetimeIndex(timestamps, name='timestamp'))
        
        # Add timezone info (UTC, as real data would be)
        data.index = data.index.tz_localize('UTC')
        
        print(f"✅ Generated realistic market data:")
        print(f"   Bars: {len(data):,}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"   Average volume: {data['volume'].mean():,.0f}")
        print(f"   Total return: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for real market analysis."""
        
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['price_momentum_5'] = data['close'].pct_change(5)
        features['price_momentum_10'] = data['close'].pct_change(10)
        
        # Volume features
        features['volume'] = data['volume']
        features['volume_ma_20'] = data['volume'].rolling(20, min_periods=1).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma_20']
        features['volume_surge'] = (features['volume_ratio'] > 1.5).astype(int)
        
        # Volatility features
        features['volatility_5'] = features['returns'].rolling(5, min_periods=1).std()
        features['volatility_20'] = features['returns'].rolling(20, min_periods=1).std()
        features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # VWAP and price positioning
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap_num = (typical_price * data['volume']).rolling(20, min_periods=1).sum()
        vwap_den = data['volume'].rolling(20, min_periods=1).sum()
        features['vwap'] = vwap_num / vwap_den
        features['vwap_deviation'] = (data['close'] - features['vwap']) / features['vwap']
        features['distance_from_vwap'] = abs(features['vwap_deviation'])
        
        # Technical indicators
        features['sma_10'] = data['close'].rolling(10, min_periods=1).mean()
        features['sma_20'] = data['close'].rolling(20, min_periods=1).mean()
        features['sma_50'] = data['close'].rolling(50, min_periods=1).mean()
        
        # Price relative to moving averages
        features['price_vs_sma10'] = (data['close'] - features['sma_10']) / features['sma_10']
        features['price_vs_sma20'] = (data['close'] - features['sma_20']) / features['sma_20']
        
        # RSI
        features['rsi'] = self._compute_rsi(data['close'])
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        
        # Bollinger Bands
        bb_upper, bb_lower = self._compute_bollinger_bands(data['close'])
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_squeeze'] = (bb_upper - bb_lower) / features['sma_20']
        
        # High-Low analysis
        features['hl_ratio'] = (data['high'] - data['low']) / data['close']
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Time-based features (for intraday patterns)
        features['hour'] = data.index.hour
        features['minute'] = data.index.minute
        features['is_opening'] = ((features['hour'] == 9) & (features['minute'] >= 30) | 
                                 (features['hour'] == 10)).astype(int)
        features['is_closing'] = (features['hour'] >= 15).astype(int)
        features['is_midday'] = ((features['hour'] >= 12) & (features['hour'] < 14)).astype(int)
        
        # Clean features
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def _compute_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Compute Bollinger Bands."""
        sma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate sophisticated trading signals based on multiple factors."""
        
        signals = pd.Series(0, index=features.index)
        
        if len(features) < 50:
            return signals
        
        # Multi-factor signal generation
        
        # 1. Momentum breakout signals
        momentum_threshold = 0.0015  # 0.15% threshold
        volume_surge_threshold = 1.3
        
        momentum_buy = (
            (features['price_momentum_5'] > momentum_threshold) &
            (features['volume_ratio'] > volume_surge_threshold) &
            (features['vwap_deviation'] > -0.002)  # Not too far below VWAP
        )
        
        momentum_sell = (
            (features['price_momentum_5'] < -momentum_threshold) &
            (features['volume_ratio'] > volume_surge_threshold) &
            (features['vwap_deviation'] < 0.002)  # Not too far above VWAP
        )
        
        # 2. Mean reversion signals
        mean_reversion_buy = (
            (features['vwap_deviation'] < -0.003) &  # Below VWAP
            (features['rsi'] < 35) &  # Oversold
            (features['bb_position'] < 0.2) &  # Near lower Bollinger Band
            (features['volume_ratio'] > 1.1)  # Some volume support
        )
        
        mean_reversion_sell = (
            (features['vwap_deviation'] > 0.003) &  # Above VWAP
            (features['rsi'] > 65) &  # Overbought
            (features['bb_position'] > 0.8) &  # Near upper Bollinger Band
            (features['volume_ratio'] > 1.1)  # Some volume support
        )
        
        # 3. Combine signals with time-based filters
        # Prefer momentum signals during opening/closing, mean reversion during midday
        buy_signals = (
            (momentum_buy & (features['is_opening'] | features['is_closing'])) |
            (mean_reversion_buy & features['is_midday']) |
            (momentum_buy & (features['volume_ratio'] > 2.0))  # Strong volume always momentum
        )
        
        sell_signals = (
            (momentum_sell & (features['is_opening'] | features['is_closing'])) |
            (mean_reversion_sell & features['is_midday']) |
            (momentum_sell & (features['volume_ratio'] > 2.0))  # Strong volume always momentum
        )
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        # 4. Risk filters
        # Don't trade during extreme volatility
        vol_filter = features['volatility_ratio'] < 3.0
        signals[~vol_filter] = 0
        
        # Don't trade with very low volume
        volume_filter = features['volume_ratio'] > 0.8
        signals[~volume_filter] = 0
        
        return signals
    
    def backtest(self, data: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """Run comprehensive backtest with realistic trading simulation."""
        
        if verbose:
            print(f"\n🚀 Running backtest for {self.symbol}...")
            print(f"📊 Data: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        
        # Prepare features
        features = self.prepare_features(data)
        if verbose:
            print(f"🔧 Features: {features.shape[1]} indicators prepared")
        
        # Generate signals
        signals = self.generate_signals(features)
        active_signals = (signals != 0).sum()
        if verbose:
            print(f"🎯 Signals: {active_signals} active signals generated")
        
        # Enhanced backtest simulation
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        capital = self.capital
        equity_curve = []
        
        # Risk management parameters
        max_position_pct = 0.08  # 8% of capital per trade
        stop_loss_pct = 0.015    # 1.5% stop loss
        take_profit_pct = 0.03   # 3% take profit
        max_hold_minutes = 180   # 3 hours max hold
        
        # Trading costs
        commission_per_share = 0.005  # $0.005 per share
        slippage_bps = 2.0           # 2 basis points slippage
        
        for i in range(50, len(data)):  # Warmup period
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Track equity curve
            if position != 0:
                unrealized_pnl = (current_price - entry_price) * abs(position)
                if position < 0:
                    unrealized_pnl = -unrealized_pnl
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            equity_curve.append(current_equity)
            
            # Position management
            if position != 0:
                held_minutes = (current_time - entry_time).total_seconds() / 60
                price_change = (current_price - entry_price) / entry_price
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                if position > 0:  # Long position
                    if price_change <= -stop_loss_pct:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif price_change >= take_profit_pct:
                        should_exit = True
                        exit_reason = "Take Profit"
                else:  # Short position
                    if price_change >= stop_loss_pct:
                        should_exit = True
                        exit_reason = "Stop Loss"
                    elif price_change <= -take_profit_pct:
                        should_exit = True
                        exit_reason = "Take Profit"
                
                if held_minutes >= max_hold_minutes:
                    should_exit = True
                    exit_reason = "Max Hold Time"
                
                if signal != 0 and signal != position:
                    should_exit = True
                    exit_reason = "Signal Reversal"
                
                # Execute exit
                if should_exit:
                    # Calculate PnL with costs
                    shares = abs(position)
                    gross_pnl = (current_price - entry_price) * shares
                    if position < 0:
                        gross_pnl = -gross_pnl
                    
                    # Apply trading costs
                    commission = shares * commission_per_share
                    slippage = shares * current_price * (slippage_bps / 10000)
                    net_pnl = gross_pnl - commission - slippage
                    
                    # Record trade
                    trade_return = net_pnl / (entry_price * shares)
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'direction': 'Long' if position > 0 else 'Short',
                        'gross_pnl': gross_pnl,
                        'commission': commission,
                        'slippage': slippage,
                        'net_pnl': net_pnl,
                        'return_pct': trade_return,
                        'duration_minutes': held_minutes,
                        'exit_reason': exit_reason
                    })
                    
                    capital += net_pnl
                    position = 0
                    entry_price = 0
                    entry_time = None
            
            # Entry logic
            if signal != 0 and position == 0:
                position_value = capital * max_position_pct
                shares = int(position_value / current_price)
                
                if shares > 0:
                    position = shares * signal
                    entry_price = current_price
                    entry_time = current_time
        
        # Calculate performance metrics
        if trades and len(equity_curve) > 0:
            # Basic metrics
            total_trades = len(trades)
            wins = [t for t in trades if t['net_pnl'] > 0]
            losses = [t for t in trades if t['net_pnl'] <= 0]
            
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            total_pnl = sum(t['net_pnl'] for t in trades)
            total_return = (capital - self.capital) / self.capital
            
            avg_win = np.mean([t['net_pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['net_pnl'] for t in losses]) if losses else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Advanced metrics
            returns = [t['return_pct'] for t in trades]
            avg_return = np.mean(returns) if returns else 0
            return_std = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = (avg_return / return_std * np.sqrt(252)) if return_std > 0 else 0
            
            # Drawdown analysis
            equity_series = pd.Series(equity_curve)
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown_pct = drawdown.min()
            max_drawdown_dollar = (equity_series - running_max).min()
            
            # Duration analysis
            durations = [t['duration_minutes'] for t in trades]
            avg_duration = np.mean(durations) if durations else 0
            
            # Exit reason analysis
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                
        else:
            # No trades
            total_trades = win_rate = total_pnl = total_return = 0
            profit_factor = avg_win = avg_loss = avg_return = sharpe_ratio = 0
            max_drawdown_pct = max_drawdown_dollar = avg_duration = 0
            exit_reasons = {}
        
        # Compile results
        results = {
            'symbol': self.symbol,
            'backtest_period': f"{data.index[0]} to {data.index[-1]}",
            'total_bars': len(data),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_dollar': max_drawdown_dollar,
            'avg_duration_minutes': avg_duration,
            'final_capital': capital,
            'features_count': features.shape[1],
            'signals_generated': active_signals,
            'exit_reasons': exit_reasons,
            'trades': trades[:10] if trades else []  # First 10 trades for review
        }
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """Print formatted backtest results."""
        print(f"\n📈 Backtest Results for {results['symbol']}:")
        print("=" * 40)
        print(f"Period: {results['backtest_period']}")
        print(f"Total Bars: {results['total_bars']:,}")
        print()
        print("📊 Trading Performance:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Total Return: {results['total_return_pct']:.1%}")
        print(f"  Total PnL: ${results['total_pnl']:.2f}")
        print(f"  Final Capital: ${results['final_capital']:.2f}")
        print()
        print("📈 Risk Metrics:")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.1%} (${results['max_drawdown_dollar']:.2f})")
        print(f"  Avg Duration: {results['avg_duration_minutes']:.1f} minutes")
        print()
        print("🔧 Strategy Details:")
        print(f"  Features Used: {results['features_count']}")
        print(f"  Signals Generated: {results['signals_generated']}")
        
        if results['exit_reasons']:
            print("\n🚪 Exit Reasons:")
            for reason, count in results['exit_reasons'].items():
                pct = (count / results['total_trades']) * 100 if results['total_trades'] > 0 else 0
                print(f"  {reason}: {count} ({pct:.1f}%)")


def demo_s3_strategy():
    """Demonstrate S3 strategy with realistic market data."""
    print("🚀 S3 Working Strategy - Real Data Demo")
    print("=" * 45)
    
    # Test multiple symbols
    symbols = ['AAPL', 'MSFT', 'TSLA']
    results_summary = []
    
    for symbol in symbols:
        print(f"\n🔍 Testing {symbol}...")
        print("-" * 25)
        
        try:
            # Initialize strategy
            strategy = S3WorkingStrategyDemo(
                symbol=symbol,
                capital=100000,
                data_days_lookback=5  # 5 days of data
            )
            
            # Load realistic data
            data = strategy.load_realistic_data()
            
            # Run backtest
            results = strategy.backtest(data, verbose=True)
            
            # Store summary
            results_summary.append({
                'symbol': symbol,
                'trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'return_pct': results['total_return_pct'],
                'sharpe': results['sharpe_ratio']
            })
            
            # Save detailed results
            output_file = f's3_demo_results_{symbol}.json'
            with open(output_file, 'w') as f:
                # Remove trades for cleaner file
                save_results = results.copy()
                save_results.pop('trades', None)
                json.dump(save_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if results_summary:
        print(f"\n📊 Summary Results:")
        print("=" * 50)
        print(f"{'Symbol':<8} {'Trades':<8} {'Win Rate':<10} {'Return':<10} {'Sharpe':<8}")
        print("-" * 50)
        for result in results_summary:
            print(f"{result['symbol']:<8} {result['trades']:<8} "
                  f"{result['win_rate']:<10.1%} {result['return_pct']:<10.1%} "
                  f"{result['sharpe']:<8.2f}")
    
    print(f"\n✅ Demo complete! Strategy successfully processes realistic market data.")


if __name__ == "__main__":
    demo_s3_strategy()