#!/usr/bin/env python3
"""
S3 AI FMP Hybrid Trading System
===============================
Combines S3 AI strategy with FMP real-time price data for live trading simulation.

Features:
- S3 AI indicator selection and regime detection
- Real-time FMP price data and market monitoring
- PostgreSQL for historical analysis and backtesting
- Simulated order execution with realistic fills
- Risk management and position controls
- Performance tracking and monitoring

Uses FMP Ultimate API for real-time data access.
"""

import sys
import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# S3 AI components
from s3_ai_wrapper import S3AIWrapper
from s3_ai_hybrid_paper_trading import S3AIDataEngine, Position, Trade

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f's3_ai_fmp_trading_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FMPQuote:
    """Real-time quote from FMP."""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime
    change: float = 0.0
    change_percent: float = 0.0


class FMPDataProvider:
    """Real-time data provider using FMP API."""
    
    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY environment variable required")
        
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.quote_cache = {}
        self.cache_duration = 15  # Cache quotes for 15 seconds
        self.request_count = 0
        self.request_limit = 3000  # FMP Ultimate limit per minute
        self.request_window_start = datetime.now()
        
        logger.info("📊 FMP Data Provider initialized")
    
    def _check_rate_limit(self):
        """Check and enforce FMP rate limits."""
        now = datetime.now()
        
        # Reset counter every minute
        if (now - self.request_window_start).seconds >= 60:
            self.request_count = 0
            self.request_window_start = now
        
        if self.request_count >= self.request_limit:
            wait_time = 60 - (now - self.request_window_start).seconds
            if wait_time > 0:
                logger.warning(f"⏳ Rate limit reached, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                self.request_count = 0
                self.request_window_start = datetime.now()
    
    def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, FMPQuote]:
        """Get real-time quotes for multiple symbols."""
        
        # Check cache first
        current_time = datetime.now()
        cached_quotes = {}
        fresh_symbols = []
        
        for symbol in symbols:
            if symbol in self.quote_cache:
                quote, cache_time = self.quote_cache[symbol]
                if (current_time - cache_time).seconds < self.cache_duration:
                    cached_quotes[symbol] = quote
                else:
                    fresh_symbols.append(symbol)
            else:
                fresh_symbols.append(symbol)
        
        # Fetch fresh data for uncached symbols
        fresh_quotes = {}
        if fresh_symbols:
            fresh_quotes = self._fetch_quotes_from_fmp(fresh_symbols)
        
        # Combine cached and fresh quotes
        all_quotes = {**cached_quotes, **fresh_quotes}
        return all_quotes
    
    def _fetch_quotes_from_fmp(self, symbols: List[str]) -> Dict[str, FMPQuote]:
        """Fetch quotes from FMP API."""
        
        if not symbols:
            return {}
        
        try:
            self._check_rate_limit()
            
            # Use batch quote endpoint for efficiency
            symbols_str = ",".join(symbols)
            url = f"{self.base_url}/quote/{symbols_str}"
            
            params = {
                "apikey": self.api_key
            }
            
            logger.debug(f"📡 Fetching FMP quotes for {len(symbols)} symbols...")
            response = requests.get(url, params=params, timeout=10)
            self.request_count += 1
            
            if response.status_code == 200:
                data = response.json()
                quotes = {}
                current_time = datetime.now()
                
                for item in data:
                    symbol = item.get('symbol', '')
                    if symbol in symbols:
                        quote = FMPQuote(
                            symbol=symbol,
                            price=float(item.get('price', 0) or 0),
                            bid=float(item.get('bid', 0) or 0),
                            ask=float(item.get('ask', 0) or 0),
                            volume=int(item.get('volume', 0) or 0),
                            change=float(item.get('change', 0) or 0),
                            change_percent=float(item.get('changesPercentage', 0) or 0),
                            timestamp=current_time
                        )
                        quotes[symbol] = quote
                        
                        # Cache the quote
                        self.quote_cache[symbol] = (quote, current_time)
                
                logger.debug(f"✅ Retrieved {len(quotes)} FMP quotes")
                return quotes
            
            else:
                logger.error(f"❌ FMP API error: {response.status_code} - {response.text}")
                return {}
                
        except requests.exceptions.Timeout:
            logger.error("❌ FMP API timeout")
            return {}
        except Exception as e:
            logger.error(f"❌ FMP API error: {e}")
            return {}
    
    def get_single_quote(self, symbol: str) -> Optional[FMPQuote]:
        """Get real-time quote for single symbol."""
        quotes = self.get_real_time_quotes([symbol])
        return quotes.get(symbol)
    
    def is_market_open(self) -> bool:
        """Check if US market is open using FMP."""
        try:
            self._check_rate_limit()
            
            url = f"{self.base_url}/is-the-market-open"
            params = {"apikey": self.api_key}
            
            response = requests.get(url, params=params, timeout=5)
            self.request_count += 1
            
            if response.status_code == 200:
                data = response.json()
                return data.get('isTheStockMarketOpen', False)
            else:
                # Fallback to time-based check
                return self._time_based_market_check()
                
        except Exception as e:
            logger.warning(f"⚠️ Market status check failed: {e}, using time-based fallback")
            return self._time_based_market_check()
    
    def _time_based_market_check(self) -> bool:
        """Fallback market hours check based on time."""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Market hours check (9:30 AM - 4:00 PM ET)
        current_time = now.time()
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        
        return market_open <= current_time <= market_close


class S3AIFMPStrategy:
    """S3 AI strategy optimized for FMP real-time data."""
    
    def __init__(self, data_engine: S3AIDataEngine, fmp_provider: FMPDataProvider):
        self.data_engine = data_engine
        self.fmp_provider = fmp_provider
        self.signal_cache = {}
        self.last_analysis_time = {}
        
        logger.info("🧠 S3 AI FMP Strategy initialized")
    
    def generate_fmp_signal(self, symbol: str, current_quote: FMPQuote, 
                           force_refresh: bool = False) -> Tuple[str, float, Dict]:
        """Generate trading signal using FMP real-time data."""
        
        # Check cache first (refresh every 3 minutes for FMP)
        cache_key = symbol
        now = datetime.now()
        if (not force_refresh and cache_key in self.signal_cache and 
            cache_key in self.last_analysis_time):
            last_time = self.last_analysis_time[cache_key]
            if (now - last_time).seconds < 180:  # 3 minutes
                cached_signal = self.signal_cache[cache_key]
                return cached_signal['action'], cached_signal['strength'], cached_signal['info']
        
        try:
            # Get recent historical data for analysis
            data = self.data_engine.get_recent_data(symbol, bars=100)
            if data.empty or len(data) < 20:
                return "HOLD", 0.0, {"error": "Insufficient historical data"}
            
            # Get AI analysis
            ai_analysis = self.data_engine.get_ai_analysis(symbol, force_refresh=force_refresh)
            regime = ai_analysis.get("regime", "Unknown")
            indicators = ai_analysis.get("indicators", {})
            confidence = ai_analysis.get("confidence", 0.0)
            
            # Calculate FMP-enhanced features
            features = self._calculate_fmp_features(data, current_quote)
            
            # Generate signal based on regime and FMP real-time data
            signal_strength = 0.0
            
            # FMP-specific signal logic with real-time data
            if regime in ['High Volatility', 'Mean Reversion']:
                # VWAP mean reversion with real-time price
                vwap_dev = features.get('vwap_deviation', 0.0)
                if vwap_dev < -0.002:  # Price below VWAP
                    signal_strength += 0.4
                elif vwap_dev > 0.002:  # Price above VWAP
                    signal_strength -= 0.4
                
                # Volume confirmation from FMP
                volume_ratio = features.get('volume_ratio', 1.0)
                if volume_ratio > 1.5:  # Higher than average volume
                    signal_strength *= 1.2  # Amplify signal
                
                # RSI confirmation
                if indicators.get('rsi_momentum', False):
                    rsi = features.get('rsi', 50)
                    if rsi < 30:  # Oversold
                        signal_strength += 0.3
                    elif rsi > 70:  # Overbought
                        signal_strength -= 0.3
            
            elif regime in ['Trending', 'Breakout']:
                # Momentum strategy with FMP real-time momentum
                price_momentum = features.get('real_time_momentum', 0.0)
                volume_ratio = features.get('volume_ratio', 1.0)
                
                if volume_ratio > 2.0 and price_momentum > 0.001:
                    signal_strength += 0.5
                elif volume_ratio > 2.0 and price_momentum < -0.001:
                    signal_strength -= 0.5
                
                # Bid-ask spread check (tighter spreads = better liquidity)
                spread_ratio = features.get('spread_ratio', 0.0)
                if spread_ratio < 0.001:  # Tight spread
                    signal_strength *= 1.1  # Slight boost for good liquidity
            
            # Conservative mixed signals for unknown regimes
            else:
                if indicators.get('sma_trend', False):
                    sma_signal = features.get('sma_signal', 0)
                    signal_strength += sma_signal * 0.2
                
                # Real-time price vs historical average
                if features.get('price_vs_avg', 0) > 0.01:
                    signal_strength += 0.15
                elif features.get('price_vs_avg', 0) < -0.01:
                    signal_strength -= 0.15
            
            # FMP-specific signal strength adjustment
            signal_strength = max(-1.0, min(1.0, signal_strength))  # Clamp to [-1, 1]
            
            # Determine action with appropriate thresholds
            if signal_strength > 0.6:
                action = "BUY"
            elif signal_strength < -0.6:
                action = "SELL"
            else:
                action = "HOLD"
            
            signal_info = {
                "strength": signal_strength,
                "regime": regime,
                "confidence": confidence,
                "indicators_active": len([k for k, v in indicators.items() if v]),
                "features": features,
                "fmp_optimized": True,
                "real_time_price": current_quote.price,
                "volume": current_quote.volume,
                "change_percent": current_quote.change_percent
            }
            
            # Cache result
            self.signal_cache[cache_key] = {
                'action': action,
                'strength': abs(signal_strength),
                'info': signal_info
            }
            self.last_analysis_time[cache_key] = now
            
            return action, abs(signal_strength), signal_info
            
        except Exception as e:
            logger.error(f"❌ FMP signal generation failed for {symbol}: {e}")
            return "HOLD", 0.0, {"error": str(e)}
    
    def _calculate_fmp_features(self, historical_data: pd.DataFrame, 
                               current_quote: FMPQuote) -> Dict:
        """Calculate features using historical + real-time FMP data."""
        
        if len(historical_data) < 20:
            return {}
        
        features = {}
        current_price = current_quote.price
        
        try:
            # Historical price features
            close_prices = historical_data['close']
            features['returns_5'] = (current_price / close_prices.iloc[-5] - 1) if len(close_prices) >= 5 else 0.0
            
            # VWAP calculation with real-time price
            typical_price = (historical_data['high'] + historical_data['low'] + historical_data['close']) / 3
            vwap_num = (typical_price * historical_data['volume']).rolling(20, min_periods=1).sum()
            vwap_den = historical_data['volume'].rolling(20, min_periods=1).sum()
            vwap = (vwap_num / vwap_den).iloc[-1]
            features['vwap_deviation'] = (current_price - vwap) / vwap if vwap > 0 else 0.0
            
            # Volume analysis with real-time volume
            volume_ma = historical_data['volume'].rolling(20, min_periods=1).mean().iloc[-1]
            features['volume_ratio'] = current_quote.volume / volume_ma if volume_ma > 0 else 1.0
            
            # Real-time momentum (change from previous close)
            last_close = close_prices.iloc[-1]
            features['real_time_momentum'] = (current_price - last_close) / last_close
            
            # Bid-ask spread analysis
            if current_quote.bid > 0 and current_quote.ask > 0:
                spread = current_quote.ask - current_quote.bid
                features['spread_ratio'] = spread / current_price
            else:
                features['spread_ratio'] = 0.001  # Default small spread
            
            # RSI calculation
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            features['rsi'] = rsi if not np.isnan(rsi) else 50
            
            # SMA trend with real-time price
            sma_20 = close_prices.rolling(20, min_periods=1).mean().iloc[-1]
            features['sma_signal'] = 1 if current_price > sma_20 else -1
            
            # Price vs recent average
            recent_avg = close_prices.tail(5).mean()
            features['price_vs_avg'] = (current_price - recent_avg) / recent_avg
            
            # Real-time volatility indicator
            features['daily_change_pct'] = current_quote.change_percent / 100.0
            
        except Exception as e:
            logger.warning(f"⚠️ FMP feature calculation error: {e}")
        
        return features


class S3AIFMPHybridTrader:
    """Main S3 AI FMP Hybrid Trading System."""
    
    def __init__(self, config_file: str = "s3_ai_fmp_trading_config_top_performers.json"):
        """Initialize S3 AI FMP hybrid trader."""
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.running = False
        
        # Initialize components
        self.data_engine = S3AIDataEngine()
        self.fmp_provider = FMPDataProvider()
        self.strategy = S3AIFMPStrategy(self.data_engine, self.fmp_provider)
        
        # Trading parameters
        self.symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL"])
        self.scan_interval = self.config.get("scan_interval", 60)
        self.max_positions = self.config.get("max_positions", 10)
        self.initial_capital = self.config.get("initial_capital", 100000)
        self.current_capital = self.initial_capital
        
        # Risk management
        risk_config = self.config.get("risk_management", {})
        self.max_position_size = risk_config.get("max_position_size", 0.05)
        self.commission = risk_config.get("commission", 1.0)
        self.min_signal_strength = risk_config.get("min_signal_strength", 0.6)
        
        # Market hours
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(15, 55)
        
        # State tracking
        self.positions = {}  # symbol -> Position
        self.trades = []
        self.last_scan_time = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info("🤖 S3 AI FMP Hybrid Trader initialized")
        logger.info(f"📊 Symbols: {len(self.symbols)} | Capital: ${self.initial_capital:,.0f}")
        logger.info(f"⏱️  Scan interval: {self.scan_interval}s | Max positions: {self.max_positions}")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load trading configuration."""
        default_config = {
            "initial_capital": 100000,
            "max_positions": 10,
            "scan_interval": 60,
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "risk_management": {
                "max_position_size": 0.05,
                "commission": 1.0,
                "min_signal_strength": 0.6
            },
            "fmp_settings": {
                "cache_duration": 15,
                "rate_limit": 3000
            }
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"📝 Loaded config from {config_file}")
                return {**default_config, **config}
            else:
                # Create default config
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"📝 Created default config: {config_file}")
                return default_config
        except Exception as e:
            logger.warning(f"⚠️ Config error: {e}, using defaults")
            return default_config
    
    def _setup_logging(self):
        """Setup additional logging handlers."""
        self.logger = logger
    
    def calculate_position_size(self, symbol: str, price: float, signal_strength: float) -> int:
        """Calculate position size based on risk management."""
        try:
            # Base position size
            base_size = self.current_capital * self.max_position_size
            
            # Adjust by signal strength
            strength_multiplier = 0.5 + (signal_strength * 0.5)
            adjusted_size = base_size * strength_multiplier
            
            # Calculate shares
            shares = int(adjusted_size / price)
            
            # Minimum viable position
            min_shares = max(1, int(1000 / price))
            shares = max(shares, min_shares)
            
            # Maximum position limit
            max_shares = int((self.current_capital * self.max_position_size) / price)
            shares = min(shares, max_shares)
            
            return shares
            
        except Exception as e:
            logger.error(f"⚠️ Position sizing error for {symbol}: {e}")
            return 0
    
    def simulate_order_fill(self, symbol: str, action: str, quantity: int, 
                           quote: FMPQuote) -> Tuple[bool, float]:
        """Simulate realistic order execution using FMP quote data."""
        try:
            # Use bid/ask for realistic fills
            if action == "BUY":
                # Buy at ask price (realistic market order)
                fill_price = quote.ask if quote.ask > 0 else quote.price
                # Add small slippage for realism
                fill_price *= 1.001  # 0.1% slippage
            else:  # SELL
                # Sell at bid price (realistic market order)
                fill_price = quote.bid if quote.bid > 0 else quote.price
                # Add slippage
                fill_price *= 0.999  # 0.1% slippage
            
            # Simulate fill success (99% success rate for liquid stocks)
            fill_success = np.random.random() < 0.99
            
            if fill_success:
                logger.info(f"✅ Simulated fill: {action} {quantity} {symbol} @ ${fill_price:.2f}")
                return True, fill_price
            else:
                logger.warning(f"❌ Simulated fill failed: {symbol}")
                return False, 0.0
                
        except Exception as e:
            logger.error(f"❌ Order simulation error for {symbol}: {e}")
            return False, 0.0
    
    def is_market_open(self) -> bool:
        """Check if market is open using FMP or fallback."""
        return self.fmp_provider.is_market_open()
    
    def scan_and_trade(self):
        """Main trading loop using FMP real-time data."""
        try:
            logger.info(f"🔍 Scanning {len(self.symbols)} symbols with FMP data...")
            
            # Get real-time quotes from FMP
            quotes = self.fmp_provider.get_real_time_quotes(self.symbols)
            
            if not quotes:
                logger.warning("⚠️ No FMP quotes received")
                return
            
            logger.debug(f"📊 Retrieved {len(quotes)} FMP quotes")
            
            # Update existing positions with current prices
            self.update_positions_with_fmp_data(quotes)
            
            # Scan for new opportunities
            for symbol in self.symbols:
                try:
                    if symbol not in quotes:
                        logger.warning(f"⚠️ No FMP data for {symbol}")
                        continue
                    
                    quote = quotes[symbol]
                    
                    # Skip if no valid price
                    if quote.price <= 0:
                        continue
                    
                    # Generate S3 AI signal with FMP data
                    action, strength, signal_info = self.strategy.generate_fmp_signal(symbol, quote)
                    
                    regime = signal_info.get("regime", "Unknown")
                    confidence = signal_info.get("confidence", 0.0)
                    
                    if action != "HOLD" and strength >= self.min_signal_strength:
                        logger.info(f"🔍 {symbol}: {action} signal | Strength: {strength:.2f} | "
                                  f"Price: ${quote.price:.2f} | Regime: {regime} | "
                                  f"Change: {quote.change_percent:+.1f}%")
                        
                        # Execute trade
                        if action == "BUY" and symbol not in self.positions:
                            if len(self.positions) < self.max_positions:
                                quantity = self.calculate_position_size(symbol, quote.price, strength)
                                if quantity > 0:
                                    success, fill_price = self.simulate_order_fill(symbol, "BUY", quantity, quote)
                                    if success:
                                        self.open_position(symbol, "BUY", quantity, fill_price, regime)
                        
                        elif action == "SELL" and symbol in self.positions:
                            position = self.positions[symbol]
                            quantity = int(abs(position.quantity))
                            success, fill_price = self.simulate_order_fill(symbol, "SELL", quantity, quote)
                            if success:
                                self.close_position(symbol, fill_price, "Signal Exit")
                
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol}: {e}")
                    continue
            
            # Close positions near market close
            current_time = datetime.now().time()
            if current_time >= dt_time(15, 50):
                self.close_all_positions(quotes)
            
            self.last_scan_time = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Scan and trade error: {e}")
    
    def open_position(self, symbol: str, action: str, quantity: int, 
                     price: float, regime: str):
        """Open new position."""
        try:
            cost = (quantity * price) + self.commission
            
            if cost > self.current_capital:
                logger.warning(f"⚠️ Insufficient capital for {symbol}")
                return
            
            position = Position(
                symbol=symbol,
                quantity=quantity if action == "BUY" else -quantity,
                entry_price=price,
                entry_time=datetime.now(),
                current_price=price,
                regime=regime
            )
            
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
            
            logger.info(f"🟢 Opened {action} position: {quantity} {symbol} @ ${price:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to open position {symbol}: {e}")
    
    def close_position(self, symbol: str, price: float, reason: str = "Signal"):
        """Close existing position."""
        try:
            if symbol not in self.positions:
                return
            
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
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"🔴 Closed position: {action} {abs(position.quantity)} {symbol} @ ${price:.2f} | "
                       f"PnL: ${pnl:+.2f} | Reason: {reason}")
            
        except Exception as e:
            logger.error(f"❌ Failed to close position {symbol}: {e}")
    
    def update_positions_with_fmp_data(self, quotes: Dict[str, FMPQuote]):
        """Update positions with FMP real-time data."""
        for symbol, position in self.positions.items():
            if symbol in quotes:
                quote = quotes[symbol]
                position.current_price = quote.price
                
                # Calculate unrealized PnL
                if position.quantity > 0:  # Long
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                else:  # Short
                    position.unrealized_pnl = (position.entry_price - position.current_price) * abs(position.quantity)
    
    def close_all_positions(self, quotes: Dict[str, FMPQuote]):
        """Close all positions (end of day)."""
        if not self.positions:
            return
        
        logger.info("🕐 Market close approaching, closing all positions...")
        
        for symbol in list(self.positions.keys()):
            if symbol in quotes:
                quote = quotes[symbol]
                success, fill_price = self.simulate_order_fill(
                    symbol, "SELL", int(abs(self.positions[symbol].quantity)), quote
                )
                if success:
                    self.close_position(symbol, fill_price, "EOD Close")
    
    def print_status(self):
        """Print current trading status."""
        print("\n" + "="*80)
        print("🤖 S3 AI FMP HYBRID TRADING STATUS")
        print("="*80)
        
        # Market status
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        print(f"📅 Market Status: {market_status}")
        
        # Portfolio summary
        total_value = self.current_capital
        unrealized_pnl = 0
        
        if self.positions:
            print(f"\n📋 ACTIVE POSITIONS ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                side = "LONG" if pos.quantity > 0 else "SHORT"
                pnl_pct = (pos.unrealized_pnl / abs(pos.quantity * pos.entry_price)) * 100
                print(f"   {symbol}: {side} {abs(pos.quantity):.0f} @ ${pos.entry_price:.2f} | "
                      f"Current: ${pos.current_price:.2f} | "
                      f"P&L: ${pos.unrealized_pnl:+.2f} ({pnl_pct:+.1f}%)")
                total_value += pos.unrealized_pnl
                unrealized_pnl += pos.unrealized_pnl
        else:
            print("\n📭 No active positions")
        
        # Trading summary
        total_trades = len(self.trades)
        if total_trades > 0:
            realized_pnl = sum(t.pnl for t in self.trades if hasattr(t, 'pnl'))
            winning_trades = len([t for t in self.trades if hasattr(t, 'pnl') and t.pnl > 0])
            win_rate = (winning_trades / len([t for t in self.trades if hasattr(t, 'pnl')])) * 100 if any(hasattr(t, 'pnl') for t in self.trades) else 0
            
            print(f"\n🔄 TRADING SUMMARY:")
            print(f"   Total Trades: {total_trades}")
            print(f"   Realized P&L: ${realized_pnl:+.2f}")
            if win_rate > 0:
                print(f"   Win Rate: {win_rate:.1f}%")
        
        total_return = ((total_value / self.initial_capital) - 1) * 100
        
        print(f"\n💰 Portfolio Value: ${total_value:,.2f}")
        print(f"📊 Unrealized P&L: ${unrealized_pnl:+,.2f}")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"💵 Available Cash: ${self.current_capital:,.2f}")
        
        if self.last_scan_time:
            print(f"⏰ Last Scan: {self.last_scan_time.strftime('%H:%M:%S')}")
        
        print("="*80)
    
    def run(self):
        """Start the S3 AI FMP hybrid trading system."""
        
        logger.info("🚀 Starting S3 AI FMP Hybrid Trading System...")
        logger.info("📡 Using FMP for real-time market data")
        
        self.running = True
        
        try:
            # Main trading loop
            while self.running:
                if self.is_market_open():
                    logger.info(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Market scan...")
                    self.scan_and_trade()
                    self.print_status()
                    time.sleep(self.scan_interval)
                else:
                    logger.info(f"🏪 Market closed - {datetime.now().strftime('%H:%M:%S')}")
                    self.print_status()
                    time.sleep(300)  # 5 minutes when closed
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping S3 AI FMP Hybrid Trading...")
            self.running = False
        
        finally:
            # Cleanup and save results
            self.save_results()
            logger.info("✅ S3 AI FMP Hybrid Trading stopped")
    
    def save_results(self):
        """Save trading results."""
        try:
            # Get final portfolio summary
            total_value = self.current_capital + sum(pos.unrealized_pnl for pos in self.positions.values())
            
            results = {
                "session_start": datetime.now().isoformat(),
                "session_summary": {
                    "initial_capital": self.initial_capital,
                    "final_value": total_value,
                    "total_return_pct": ((total_value / self.initial_capital) - 1) * 100,
                    "positions_count": len(self.positions),
                    "total_trades": len(self.trades)
                },
                "positions": [asdict(pos) for pos in self.positions.values()],
                "trades": [asdict(trade) for trade in self.trades],
                "config": self.config
            }
            
            filename = f"s3_ai_fmp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")


def main():
    """Main entry point with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 AI FMP Hybrid Trading System')
    parser.add_argument('--config', 
                       default='s3_ai_fmp_trading_config_top_performers.json',
                       help='Configuration file (default: top performers config)')
    parser.add_argument('--start', action='store_true', default=True,
                       help='Start trading system (default action)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode instead of live trading')
    parser.add_argument('--test', action='store_true', 
                       help='Run test suite')
    parser.add_argument('--status', action='store_true',
                       help='Show system status only')
    
    # If no arguments provided, default to --start
    if len(sys.argv) == 1:
        args = parser.parse_args(['--start'])
    else:
        args = parser.parse_args()
        # Override default if other actions specified
        if args.demo or args.test or args.status:
            args.start = False
    
    print("🤖 S3 AI FMP Hybrid Trading System")
    print("=" * 50)
    print("📡 Real-time data powered by Financial Modeling Prep")
    print("🔒 Simulated paper trading with realistic execution")
    print(f"📊 Using config: {args.config}")
    print()
    
    try:
        if args.test:
            print("🧪 Running test suite...")
            from test_s3_ai_fmp import main as test_main
            test_main()
            return
        
        if args.demo:
            print("🎮 Running demo mode...")
            from s3_ai_fmp_demo import main as demo_main
            demo_main()
            return
        
        # Initialize trader with specified config
        trader = S3AIFMPHybridTrader(args.config)
        
        if args.status:
            print("📊 System status:")
            trader.print_status()
            return
        
        if args.start:
            print("🚀 Starting live trading system...")
            trader.run()
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Handle graceful shutdown
    import signal
    def signal_handler(sig, frame):
        print('\n🛑 Shutdown signal received')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    main()