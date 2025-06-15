#!/usr/bin/env python3
"""
S3 AI Hybrid Paper Trading System
=================================
Production-ready hybrid paper trading system combining:
- S3 AI strategy with PostgreSQL real data
- Real-time market data integration
- Portfolio management and risk controls
- Performance tracking and monitoring
- Multi-symbol execution engine

Features:
- Real-time AI indicator selection
- Market regime detection and adaptation
- Position sizing and risk management
- End-of-day position closure
- Comprehensive logging and reporting
"""

import pandas as pd
import numpy as np
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import threading
import queue
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Import AI components
from s3_ai_wrapper import S3AIWrapper

# Load environment variables
load_dotenv()


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    regime: str = "Unknown"


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    pnl: float = 0.0
    commission: float = 1.0
    regime: str = "Unknown"


@dataclass
class MarketData:
    """Real-time market data structure."""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime


class S3AIDataEngine:
    """Handles real-time data fetching and AI analysis for S3 strategy."""
    
    def __init__(self):
        """Initialize data engine with PostgreSQL connection."""
        self.postgres_uri = os.getenv("POSTGRES_URI")
        if not self.postgres_uri:
            raise ValueError("POSTGRES_URI environment variable required")
        
        self.engine = create_engine(self.postgres_uri, pool_pre_ping=True)
        self.ai_cache = {}  # Cache AI selections
        self.data_cache = {}  # Cache recent price data
        
        print("📊 S3 AI Data Engine initialized")
    
    def get_recent_data(self, symbol: str, bars: int = 200) -> pd.DataFrame:
        """Get recent historical data for AI analysis."""
        try:
            query = """
            SELECT mp.datetime AT TIME ZONE 'UTC' AS ts,
                   mp.open, mp.high, mp.low, mp.close, mp.volume
              FROM minute_prices mp
              JOIN tickers t ON mp.ticker_id = t.id
             WHERE t.symbol = %(symbol)s
             ORDER BY mp.datetime DESC
             LIMIT %(bars)s
            """
            
            df = pd.read_sql(
                query,
                self.engine,
                params={"symbol": symbol, "bars": bars},
                parse_dates=["ts"]
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Sort chronologically and set index
            df = df.sort_values('ts').set_index('ts')
            
            # Convert timezone
            if df.index.tz is None:
                df = df.tz_localize("UTC")
            df = df.tz_convert("US/Eastern")
            
            # Filter trading hours
            df = df.between_time('09:30', '16:00')
            
            return df.dropna()
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_ai_analysis(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get AI indicator selection and regime detection for symbol."""
        
        # Check cache first (refresh every 30 minutes)
        cache_key = symbol
        if not force_refresh and cache_key in self.ai_cache:
            cache_time, analysis = self.ai_cache[cache_key]
            if (datetime.now() - cache_time).seconds < 1800:  # 30 minutes
                return analysis
        
        try:
            # Get recent data
            data = self.get_recent_data(symbol, bars=300)
            if data.empty or len(data) < 50:
                return {"regime": "Unknown", "indicators": {}, "confidence": 0.0}
            
            # Initialize AI selector
            ai_selector = S3AIWrapper(
                confidence_threshold=0.1,
                min_indicators=3,
                max_indicators=8
            )
            
            # Get AI selection
            selection_result = ai_selector.select_indicators(data, top_k=8)
            
            # Extract regime information
            regime_info = selection_result.get('regime_detection', {})
            if isinstance(regime_info, str):
                regime = regime_info
                confidence = 0.0
            elif isinstance(regime_info, dict):
                regime = regime_info.get('primary_regime', 'Unknown')
                confidence = regime_info.get('confidence', 0.0)
            else:
                regime = "Unknown"
                confidence = 0.0
            
            # Map AI indicators to strategy indicators
            selected_indicators = selection_result.get('selected_indicators', {})
            strategy_indicators = {}
            
            for ai_indicator, info in selected_indicators.items():
                if 'rsi' in ai_indicator.lower():
                    strategy_indicators['rsi_momentum'] = True
                elif 'sma' in ai_indicator.lower() or 'ema' in ai_indicator.lower():
                    strategy_indicators['sma_trend'] = True
                elif 'macd' in ai_indicator.lower():
                    strategy_indicators['macd_momentum'] = True
                elif 'bb' in ai_indicator.lower() or 'bollinger' in ai_indicator.lower():
                    strategy_indicators['bollinger_bands'] = True
                elif 'volume' in ai_indicator.lower():
                    strategy_indicators['volume_breakout'] = True
                elif 'volatility' in ai_indicator.lower() or 'atr' in ai_indicator.lower():
                    strategy_indicators['volatility_regime'] = True
                elif 'vwap' in ai_indicator.lower():
                    strategy_indicators['vwap_deviation'] = True
            
            # Ensure minimum indicators
            if len(strategy_indicators) == 0:
                strategy_indicators = {
                    'sma_trend': True,
                    'rsi_momentum': True,
                    'vwap_deviation': True
                }
            
            analysis = {
                "regime": regime,
                "confidence": confidence,
                "indicators": strategy_indicators,
                "raw_selection": selected_indicators,
                "data_points": len(data)
            }
            
            # Cache result
            self.ai_cache[cache_key] = (datetime.now(), analysis)
            
            return analysis
            
        except Exception as e:
            print(f"❌ AI analysis failed for {symbol}: {e}")
            return {"regime": "Unknown", "indicators": {}, "confidence": 0.0}


class S3AITradingStrategy:
    """S3 AI trading strategy implementation for real-time execution."""
    
    def __init__(self, data_engine: S3AIDataEngine):
        self.data_engine = data_engine
        self.signal_cache = {}
        
    def generate_signal(self, symbol: str, current_price: float) -> Tuple[str, float, Dict]:
        """Generate trading signal for symbol at current price."""
        
        try:
            # Get recent data and AI analysis
            data = self.data_engine.get_recent_data(symbol, bars=100)
            if data.empty or len(data) < 20:
                return "HOLD", 0.0, {}
            
            ai_analysis = self.data_engine.get_ai_analysis(symbol)
            regime = ai_analysis.get("regime", "Unknown")
            indicators = ai_analysis.get("indicators", {})
            
            # Calculate features
            features = self._calculate_features(data, current_price)
            
            # Generate signal based on regime and indicators
            signal_strength = 0.0
            
            if regime in ['High Volatility', 'Mean Reversion']:
                # VWAP mean reversion strategy
                vwap_dev = features.get('vwap_deviation', 0.0)
                if vwap_dev < -0.002:  # Price below VWAP
                    signal_strength += 0.4
                elif vwap_dev > 0.002:  # Price above VWAP
                    signal_strength -= 0.4
                
                # RSI signals (if selected by AI)
                if indicators.get('rsi_momentum', False):
                    rsi = features.get('rsi', 50)
                    if rsi < 30:  # Oversold
                        signal_strength += 0.3
                    elif rsi > 70:  # Overbought
                        signal_strength -= 0.3
            
            elif regime in ['Trending', 'Breakout']:
                # Momentum strategy
                if indicators.get('volume_breakout', False):
                    volume_ratio = features.get('volume_ratio', 1.0)
                    price_momentum = features.get('returns_5', 0.0)
                    
                    if volume_ratio > 1.5 and price_momentum > 0.001:
                        signal_strength += 0.5
                    elif volume_ratio > 1.5 and price_momentum < -0.001:
                        signal_strength -= 0.5
                
                # Bollinger band breakout
                if indicators.get('bollinger_bands', False):
                    bb_position = features.get('bb_position', 0.5)
                    if bb_position > 1.0:  # Above upper band
                        signal_strength += 0.3
                    elif bb_position < 0.0:  # Below lower band
                        signal_strength -= 0.3
            
            else:
                # Conservative mixed signals for unknown regimes
                if indicators.get('sma_trend', False):
                    sma_signal = features.get('sma_signal', 0)
                    signal_strength += sma_signal * 0.2
                
                if indicators.get('macd_momentum', False):
                    macd_hist = features.get('macd_histogram', 0.0)
                    if macd_hist > 0:
                        signal_strength += 0.2
                    elif macd_hist < 0:
                        signal_strength -= 0.2
            
            # Determine action
            if signal_strength > 0.5:
                action = "BUY"
            elif signal_strength < -0.5:
                action = "SELL"
            else:
                action = "HOLD"
            
            signal_info = {
                "strength": signal_strength,
                "regime": regime,
                "confidence": ai_analysis.get("confidence", 0.0),
                "indicators_active": len([k for k, v in indicators.items() if v]),
                "features": features
            }
            
            return action, abs(signal_strength), signal_info
            
        except Exception as e:
            print(f"❌ Signal generation failed for {symbol}: {e}")
            return "HOLD", 0.0, {}
    
    def _calculate_features(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Calculate technical features for signal generation."""
        
        if len(data) < 20:
            return {}
        
        features = {}
        
        try:
            # Price features
            close_prices = data['close']
            features['returns_5'] = (current_price / close_prices.iloc[-5] - 1) if len(close_prices) >= 5 else 0.0
            
            # VWAP calculation
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap_num = (typical_price * data['volume']).rolling(20, min_periods=1).sum()
            vwap_den = data['volume'].rolling(20, min_periods=1).sum()
            vwap = (vwap_num / vwap_den).iloc[-1]
            features['vwap_deviation'] = (current_price - vwap) / vwap if vwap > 0 else 0.0
            
            # Volume ratio
            volume_ma = data['volume'].rolling(20, min_periods=1).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            features['volume_ratio'] = current_volume / volume_ma if volume_ma > 0 else 1.0
            
            # RSI calculation
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            features['rsi'] = rsi if not np.isnan(rsi) else 50
            
            # SMA signal
            sma_20 = close_prices.rolling(20, min_periods=1).mean().iloc[-1]
            features['sma_signal'] = 1 if current_price > sma_20 else -1
            
            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            features['macd_histogram'] = (macd - macd_signal).iloc[-1]
            
            # Bollinger Bands position
            bb_sma = close_prices.rolling(20, min_periods=1).mean()
            bb_std = close_prices.rolling(20, min_periods=1).std()
            bb_upper = bb_sma + (bb_std * 2)
            bb_lower = bb_sma - (bb_std * 2)
            bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            if bb_range > 0:
                features['bb_position'] = (current_price - bb_lower.iloc[-1]) / bb_range
            else:
                features['bb_position'] = 0.5
            
        except Exception as e:
            print(f"⚠️ Feature calculation error: {e}")
        
        return features


class S3AIPortfolioManager:
    """Portfolio management for S3 AI hybrid paper trading."""
    
    def __init__(self, initial_capital: float = 1000000, max_positions: int = 20):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_pnl = []
        
        # Risk management
        self.max_position_size = 0.05  # 5% per position
        self.max_portfolio_risk = 0.20  # 20% total risk
        self.commission = 1.0  # $1 per trade
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        
        print(f"💼 S3 AI Portfolio Manager initialized: ${initial_capital:,.0f} capital")
    
    def calculate_position_size(self, symbol: str, price: float, signal_strength: float) -> int:
        """Calculate position size based on risk management rules."""
        
        try:
            # Base position size (percentage of capital)
            base_size = self.current_capital * self.max_position_size
            
            # Adjust by signal strength (0.5 to 1.0 multiplier)
            strength_multiplier = 0.5 + (signal_strength * 0.5)
            adjusted_size = base_size * strength_multiplier
            
            # Calculate shares
            shares = int(adjusted_size / price)
            
            # Ensure minimum viable position
            min_shares = max(1, int(1000 / price))
            shares = max(shares, min_shares)
            
            # Respect maximum position limits
            max_shares = int((self.current_capital * self.max_position_size) / price)
            shares = min(shares, max_shares)
            
            return shares
            
        except Exception as e:
            print(f"⚠️ Position sizing error for {symbol}: {e}")
            return 0
    
    def open_position(self, symbol: str, action: str, price: float, 
                     signal_strength: float, regime: str) -> bool:
        """Open a new position."""
        
        try:
            # Check if we already have a position
            if symbol in self.positions:
                return False
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                return False
            
            # Calculate position size
            quantity = self.calculate_position_size(symbol, price, signal_strength)
            if quantity <= 0:
                return False
            
            # Calculate cost including commission
            cost = (quantity * price) + self.commission
            
            # Check if we have enough capital
            if cost > self.current_capital:
                return False
            
            # Create position
            position = Position(
                symbol=symbol,
                quantity=quantity if action == "BUY" else -quantity,
                entry_price=price,
                entry_time=datetime.now(),
                current_price=price,
                regime=regime
            )
            
            # Update capital and positions
            self.current_capital -= cost
            self.positions[symbol] = position
            
            # Record trade
            trade = Trade(
                symbol=symbol,
                side=action,
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                commission=self.commission,
                regime=regime
            )
            self.trades.append(trade)
            self.total_trades += 1
            
            print(f"🟢 {action} {quantity} {symbol} @ ${price:.2f} | Regime: {regime}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to open position for {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, price: float, reason: str = "Signal") -> bool:
        """Close an existing position."""
        
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            
            # Calculate PnL
            if position.quantity > 0:  # Long position
                pnl = (price - position.entry_price) * position.quantity - self.commission
                action = "SELL"
            else:  # Short position
                pnl = (position.entry_price - price) * abs(position.quantity) - self.commission
                action = "COVER"
            
            # Update capital
            proceeds = abs(position.quantity) * price - self.commission
            self.current_capital += proceeds + pnl
            
            # Track performance
            if pnl > 0:
                self.winning_trades += 1
            
            # Record closing trade
            trade = Trade(
                symbol=symbol,
                side=action,
                quantity=abs(position.quantity),
                price=price,
                timestamp=datetime.now(),
                pnl=pnl,
                commission=self.commission,
                regime=position.regime
            )
            self.trades.append(trade)
            self.total_trades += 1
            
            # Remove position
            del self.positions[symbol]
            
            print(f"🔴 {action} {abs(position.quantity)} {symbol} @ ${price:.2f} | "
                  f"PnL: ${pnl:.2f} | Reason: {reason}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to close position for {symbol}: {e}")
            return False
    
    def update_positions(self, market_data: Dict[str, float]):
        """Update all positions with current market prices."""
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.current_price = market_data[symbol]
                
                # Calculate unrealized PnL
                if position.quantity > 0:  # Long
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                else:  # Short
                    position.unrealized_pnl = (position.entry_price - position.current_price) * abs(position.quantity)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(trade.pnl for trade in self.trades)
        total_value = self.current_capital + total_unrealized_pnl
        
        win_rate = (self.winning_trades / max(1, len([t for t in self.trades if t.pnl != 0]))) * 100
        
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "total_value": total_value,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": total_realized_pnl,
            "total_return_pct": ((total_value / self.initial_capital) - 1) * 100,
            "positions_count": len(self.positions),
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "active_symbols": list(self.positions.keys())
        }


class S3AIHybridPaperTrader:
    """Main S3 AI hybrid paper trading system."""
    
    def __init__(self, config_file: str = "s3_ai_trading_config.json"):
        """Initialize the hybrid paper trading system."""
        
        self.config = self._load_config(config_file)
        self.running = False
        
        # Initialize components
        self.data_engine = S3AIDataEngine()
        self.strategy = S3AITradingStrategy(self.data_engine)
        self.portfolio = S3AIPortfolioManager(
            initial_capital=self.config.get("initial_capital", 1000000),
            max_positions=self.config.get("max_positions", 20)
        )
        
        # Trading parameters
        self.symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"])
        self.scan_interval = self.config.get("scan_interval", 60)  # seconds
        self.market_hours = self.config.get("market_hours", {"start": "09:30", "end": "15:55"})
        
        # Logging setup
        self._setup_logging()
        
        print(f"🤖 S3 AI Hybrid Paper Trading System initialized")
        print(f"📊 Symbols: {len(self.symbols)} | Capital: ${self.portfolio.initial_capital:,.0f}")
        print(f"⏱️  Scan interval: {self.scan_interval}s | Max positions: {self.portfolio.max_positions}")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load trading configuration."""
        default_config = {
            "initial_capital": 1000000,
            "max_positions": 20,
            "scan_interval": 60,
            "market_hours": {"start": "09:30", "end": "15:55"},
            "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                return {**default_config, **config}
            else:
                # Create default config file
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                print(f"📝 Created default config: {config_file}")
                return default_config
        except Exception as e:
            print(f"⚠️ Config error, using defaults: {e}")
            return default_config
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f's3_ai_trading_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check market hours (ET)
        current_time = now.strftime("%H:%M")
        return self.market_hours["start"] <= current_time <= self.market_hours["end"]
    
    def get_current_prices(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get current prices for symbols (optimized batch query)."""
        if symbols is None:
            symbols = self.symbols
        
        prices = {}
        
        try:
            # Batch query for efficiency
            symbols_str = "', '".join(symbols)
            query = f"""
            SELECT t.symbol, mp.close
            FROM minute_prices mp
            JOIN tickers t ON mp.ticker_id = t.id
            WHERE t.symbol IN ('{symbols_str}')
            AND mp.datetime = (
                SELECT MAX(mp2.datetime)
                FROM minute_prices mp2
                JOIN tickers t2 ON mp2.ticker_id = t2.id
                WHERE t2.symbol = t.symbol
            )
            """
            
            import pandas as pd
            df = pd.read_sql(query, self.data_engine.engine)
            
            for _, row in df.iterrows():
                prices[row['symbol']] = float(row['close'])
            
            # Fill missing symbols individually if needed
            missing_symbols = set(symbols) - set(prices.keys())
            for symbol in missing_symbols:
                try:
                    recent_data = self.data_engine.get_recent_data(symbol, bars=1)
                    if not recent_data.empty:
                        prices[symbol] = float(recent_data['close'].iloc[-1])
                    else:
                        print(f"⚠️ No price data available for {symbol}")
                except Exception as e:
                    print(f"❌ Price fetch error for {symbol}: {e}")
        
        except Exception as e:
            print(f"❌ Batch price fetch failed, falling back to individual queries: {e}")
            # Fallback to individual queries
            for symbol in symbols:
                try:
                    recent_data = self.data_engine.get_recent_data(symbol, bars=1)
                    if not recent_data.empty:
                        prices[symbol] = float(recent_data['close'].iloc[-1])
                except Exception as e:
                    print(f"❌ Price fetch error for {symbol}: {e}")
        
        return prices
    
    def scan_and_trade(self):
        """Main trading loop - scan symbols and execute trades."""
        
        try:
            # Get current market prices
            prices = self.get_current_prices()
            if not prices:
                return
            
            # Update existing positions
            self.portfolio.update_positions(prices)
            
            # Scan for new opportunities (limit to first few symbols for efficiency)
            active_symbols = list(prices.keys())[:5]  # Process max 5 symbols per scan
            
            for symbol in active_symbols:
                current_price = prices[symbol]
                
                # Generate signal
                action, strength, signal_info = self.strategy.generate_signal(symbol, current_price)
                
                # Log signal
                regime = signal_info.get("regime", "Unknown")
                confidence = signal_info.get("confidence", 0.0)
                
                if action != "HOLD":
                    print(f"🔍 {symbol}: {action} signal | Strength: {strength:.2f} | "
                          f"Regime: {regime} | Confidence: {confidence:.2f}")
                
                # Execute trades
                if action == "BUY" and symbol not in self.portfolio.positions:
                    if strength > 0.6:  # Only take strong signals
                        self.portfolio.open_position(symbol, action, current_price, strength, regime)
                
                elif action == "SELL" and symbol in self.portfolio.positions:
                    if strength > 0.6:  # Only close on strong signals
                        self.portfolio.close_position(symbol, current_price, "Exit Signal")
            
            # Risk management - close positions near market close
            current_time = datetime.now().strftime("%H:%M")
            if current_time >= "15:50":  # Close all positions before market close
                for symbol in list(self.portfolio.positions.keys()):
                    if symbol in prices:
                        self.portfolio.close_position(symbol, prices[symbol], "EOD Close")
            
            # Log portfolio status
            summary = self.portfolio.get_portfolio_summary()
            if summary["positions_count"] > 0 or len(self.portfolio.trades) > 0:
                self.logger.info(f"Portfolio: ${summary['total_value']:,.0f} | "
                                f"Return: {summary['total_return_pct']:.1f}% | "
                                f"Positions: {summary['positions_count']} | "
                                f"Trades: {summary['total_trades']}")
            
        except Exception as e:
            self.logger.error(f"Trading loop error: {e}")
    
    def print_portfolio_status(self):
        """Print detailed portfolio status."""
        summary = self.portfolio.get_portfolio_summary()
        
        print("\n" + "="*80)
        print("🤖 S3 AI HYBRID PAPER TRADING PORTFOLIO STATUS")
        print("="*80)
        print(f"💰 Portfolio Value: ${summary['total_value']:,.2f}")
        print(f"📈 Total Return: {summary['total_return_pct']:+.2f}%")
        print(f"💵 Available Cash: ${summary['current_capital']:,.2f}")
        print(f"📊 Unrealized P&L: ${summary['unrealized_pnl']:+,.2f}")
        print(f"💼 Active Positions: {summary['positions_count']}")
        print(f"🔄 Total Trades: {summary['total_trades']}")
        print(f"🎯 Win Rate: {summary['win_rate']:.1f}%")
        
        if self.portfolio.positions:
            print(f"\n📋 ACTIVE POSITIONS:")
            for symbol, pos in self.portfolio.positions.items():
                pnl_pct = (pos.unrealized_pnl / (abs(pos.quantity) * pos.entry_price)) * 100
                side = "LONG" if pos.quantity > 0 else "SHORT"
                print(f"   {symbol}: {side} {abs(pos.quantity)} @ ${pos.entry_price:.2f} | "
                      f"Current: ${pos.current_price:.2f} | "
                      f"P&L: ${pos.unrealized_pnl:+.2f} ({pnl_pct:+.1f}%) | "
                      f"Regime: {pos.regime}")
        
        print("="*80)
    
    def run(self):
        """Start the hybrid paper trading system."""
        
        print(f"\n🚀 Starting S3 AI Hybrid Paper Trading System...")
        print(f"📊 Monitoring {len(self.symbols)} symbols")
        print(f"⏰ Market hours: {self.market_hours['start']} - {self.market_hours['end']} ET")
        print(f"🔄 Scan interval: {self.scan_interval} seconds")
        print(f"💰 Initial capital: ${self.portfolio.initial_capital:,.0f}")
        
        self.running = True
        
        try:
            scan_count = 0
            max_scans = 100  # Prevent infinite running in demo
            
            while self.running and scan_count < max_scans:
                if self.is_market_open():
                    print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Scanning markets... (#{scan_count + 1})")
                    self.scan_and_trade()
                    scan_count += 1
                    
                    # Brief pause between scans
                    print(f"💤 Waiting {self.scan_interval} seconds until next scan...")
                    time.sleep(self.scan_interval)
                else:
                    print(f"🏪 Market closed, waiting... {datetime.now().strftime('%H:%M:%S')}")
                    # Print status when market is closed
                    self.print_portfolio_status()
                    time.sleep(60)  # 1 minute when closed
                    scan_count += 1
            
            if scan_count >= max_scans:
                print(f"\n⏰ Reached maximum scan limit ({max_scans}), stopping demo...")
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping S3 AI Hybrid Paper Trading System...")
            self.running = False
            
            # Final portfolio status
            self.print_portfolio_status()
            
            # Save final results
            self.save_results()
            
            print(f"✅ S3 AI Hybrid Paper Trading System stopped")
    
    def save_results(self):
        """Save trading results to file."""
        try:
            summary = self.portfolio.get_portfolio_summary()
            
            results = {
                "session_summary": summary,
                "positions": [asdict(pos) for pos in self.portfolio.positions.values()],
                "trades": [asdict(trade) for trade in self.portfolio.trades],
                "config": self.config,
                "session_start": self.portfolio.start_time.isoformat(),
                "session_end": datetime.now().isoformat()
            }
            
            filename = f"s3_ai_trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """Main entry point for S3 AI Hybrid Paper Trading."""
    
    print("🤖 S3 AI Hybrid Paper Trading System")
    print("=" * 50)
    
    try:
        # Initialize trader
        trader = S3AIHybridPaperTrader()
        
        # Show initial status
        trader.print_portfolio_status()
        
        # Start trading
        trader.run()
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()