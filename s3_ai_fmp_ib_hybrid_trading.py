#!/usr/bin/env python3
"""
S3 AI FMP-IB Hybrid Trading System
==================================
Ultimate hybrid combining:
- FMP real-time data for trading decisions and market monitoring
- PostgreSQL historical data for AI analysis and backtesting  
- Interactive Brokers execution for real paper trading
- S3 AI strategy with regime detection and indicator selection

This system provides the best of all worlds:
- Real-time FMP data (bandwidth efficient)
- Historical PostgreSQL analysis (comprehensive)
- Real IB execution (actual trading)
"""

import sys
import os
import signal
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

# IB components
from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract
from ib_insync.util import df

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
        logging.FileHandler(f's3_ai_fmp_ib_hybrid_{datetime.now().strftime("%Y%m%d")}.log'),
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
    """FMP real-time data provider for trading decisions."""
    
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
        
        logger.info("📊 FMP Data Provider initialized for real-time trading data")
    
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
                logger.warning(f"⏳ FMP rate limit reached, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                self.request_count = 0
                self.request_window_start = datetime.now()
    
    def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, FMPQuote]:
        """Get real-time quotes for multiple symbols from FMP."""
        
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
            
            params = {"apikey": self.api_key}
            
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
            logger.warning(f"⚠️ FMP market status check failed: {e}, using time-based fallback")
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


class IBExecutionManager:
    """Interactive Brokers execution manager for real trading."""
    
    def __init__(self, config: Dict):
        self.ib = IB()
        self.config = config
        self.connected = False
        self.ib_settings = config.get("ib_settings", {})
        
        # IB connection parameters
        self.host = self.ib_settings.get("host", "127.0.0.1")
        self.port = self.ib_settings.get("port", 7497)  # Paper trading port
        self.client_id = self.ib_settings.get("client_id", 1)
        
        logger.info("🏦 IB Execution Manager initialized")
    
    def connect(self) -> bool:
        """Connect to Interactive Brokers."""
        try:
            if self.connected:
                return True
            
            logger.info(f"🔌 Connecting to IB at {self.host}:{self.port}...")
            
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=15
            )
            
            self.connected = True
            account = self.ib.managedAccounts()[0] if self.ib.managedAccounts() else "Unknown"
            
            logger.info(f"✅ Connected to IB successfully")
            logger.info(f"📊 Account: {account}")
            logger.info(f"🔗 Connection: {self.host}:{self.port}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ IB connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("🔌 Disconnected from IB")
        except Exception as e:
            logger.warning(f"⚠️ IB disconnect warning: {e}")
    
    def create_contract(self, symbol: str) -> Contract:
        """Create IB contract for symbol."""
        return Stock(symbol, 'SMART', 'USD')
    
    def execute_market_order(self, symbol: str, action: str, quantity: int) -> Tuple[bool, float, str]:
        """Execute market order through IB."""
        if not self.connected:
            logger.error("❌ IB not connected")
            return False, 0.0, "Not connected"
        
        try:
            contract = self.create_contract(symbol)
            
            # Create market order
            order = MarketOrder(action.upper(), quantity)
            
            logger.info(f"📋 Placing IB order: {action} {quantity} {symbol}")
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for fill (with timeout)
            timeout = 30  # 30 seconds
            start_time = time.time()
            
            while not trade.isDone() and (time.time() - start_time) < timeout:
                self.ib.sleep(1)
            
            if trade.isDone():
                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice
                    logger.info(f"✅ IB order filled: {action} {quantity} {symbol} @ ${fill_price:.2f}")
                    return True, fill_price, "Filled"
                else:
                    status = trade.orderStatus.status
                    logger.warning(f"⚠️ IB order not filled: {status}")
                    return False, 0.0, status
            else:
                logger.warning(f"⏰ IB order timeout for {symbol}")
                self.ib.cancelOrder(order)
                return False, 0.0, "Timeout"
                
        except Exception as e:
            logger.error(f"❌ IB order execution failed for {symbol}: {e}")
            return False, 0.0, str(e)
    
    def get_account_summary(self) -> Dict:
        """Get IB account summary."""
        if not self.connected:
            return {}
        
        try:
            summary = self.ib.accountSummary()
            account_data = {}
            
            for item in summary:
                if item.tag in ['TotalCashValue', 'NetLiquidation', 'AvailableFunds']:
                    account_data[item.tag] = float(item.value)
            
            return account_data
            
        except Exception as e:
            logger.error(f"❌ IB account summary failed: {e}")
            return {}
    
    def get_positions(self) -> List:
        """Get current IB positions."""
        if not self.connected:
            return []
        
        try:
            return self.ib.positions()
        except Exception as e:
            logger.error(f"❌ IB positions failed: {e}")
            return []


class S3AIFMPIBStrategy:
    """S3 AI strategy using FMP real-time data and PostgreSQL historical analysis."""
    
    def __init__(self, data_engine: S3AIDataEngine, fmp_provider: FMPDataProvider):
        self.data_engine = data_engine
        self.fmp_provider = fmp_provider
        self.signal_cache = {}
        self.last_analysis_time = {}
        
        logger.info("🧠 S3 AI FMP-IB Strategy initialized")
        logger.info("📊 Data Sources: PostgreSQL (historical) + FMP (real-time)")
    
    def generate_trading_signal(self, symbol: str, current_quote: FMPQuote, 
                               force_refresh: bool = False) -> Tuple[str, float, Dict]:
        """Generate trading signal using PostgreSQL AI analysis + FMP real-time data."""
        
        # Check cache first (refresh every 3 minutes)
        cache_key = symbol
        now = datetime.now()
        if (not force_refresh and cache_key in self.signal_cache and 
            cache_key in self.last_analysis_time):
            last_time = self.last_analysis_time[cache_key]
            if (now - last_time).seconds < 180:  # 3 minutes
                cached_signal = self.signal_cache[cache_key]
                return cached_signal['action'], cached_signal['strength'], cached_signal['info']
        
        try:
            # Get historical data from PostgreSQL for AI analysis
            historical_data = self.data_engine.get_recent_data(symbol, bars=100)
            if historical_data.empty or len(historical_data) < 20:
                return "HOLD", 0.0, {"error": "Insufficient historical data"}
            
            # Get AI analysis from PostgreSQL data
            ai_analysis = self.data_engine.get_ai_analysis(symbol, force_refresh=force_refresh)
            regime = ai_analysis.get("regime", "Unknown")
            indicators = ai_analysis.get("indicators", {})
            confidence = ai_analysis.get("confidence", 0.0)
            
            # Calculate features combining historical + real-time FMP data
            features = self._calculate_hybrid_features(historical_data, current_quote)
            
            # Generate signal based on regime and hybrid data
            signal_strength = 0.0
            
            # FMP-enhanced signal logic with PostgreSQL AI regime detection
            if regime in ['High Volatility', 'Mean Reversion']:
                # VWAP mean reversion with FMP real-time price
                vwap_dev = features.get('vwap_deviation', 0.0)
                if vwap_dev < -0.002:  # Price below VWAP
                    signal_strength += 0.4
                elif vwap_dev > 0.002:  # Price above VWAP
                    signal_strength -= 0.4
                
                # Volume confirmation from FMP
                volume_ratio = features.get('volume_ratio', 1.0)
                if volume_ratio > 1.5:  # Higher than average volume
                    signal_strength *= 1.2  # Amplify signal
                
                # RSI confirmation from PostgreSQL
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
                
                # Bid-ask spread check (from FMP)
                spread_ratio = features.get('spread_ratio', 0.0)
                if spread_ratio < 0.001:  # Tight spread
                    signal_strength *= 1.1  # Boost for good liquidity
            
            # Conservative signals for unknown regimes
            else:
                if indicators.get('sma_trend', False):
                    sma_signal = features.get('sma_signal', 0)
                    signal_strength += sma_signal * 0.2
                
                # Real-time price vs historical average
                if features.get('price_vs_avg', 0) > 0.01:
                    signal_strength += 0.15
                elif features.get('price_vs_avg', 0) < -0.01:
                    signal_strength -= 0.15
            
            # Clamp signal strength
            signal_strength = max(-1.0, min(1.0, signal_strength))
            
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
                "hybrid_optimized": True,
                "data_sources": "PostgreSQL (AI) + FMP (real-time)",
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
            logger.error(f"❌ Hybrid signal generation failed for {symbol}: {e}")
            return "HOLD", 0.0, {"error": str(e)}
    
    def _calculate_hybrid_features(self, historical_data: pd.DataFrame, 
                                  current_quote: FMPQuote) -> Dict:
        """Calculate features using PostgreSQL historical + FMP real-time data."""
        
        if len(historical_data) < 20:
            return {}
        
        features = {}
        current_price = current_quote.price
        
        try:
            # Historical price features from PostgreSQL
            close_prices = historical_data['close']
            features['returns_5'] = (current_price / close_prices.iloc[-5] - 1) if len(close_prices) >= 5 else 0.0
            
            # VWAP calculation with FMP real-time price
            typical_price = (historical_data['high'] + historical_data['low'] + historical_data['close']) / 3
            vwap_num = (typical_price * historical_data['volume']).rolling(20, min_periods=1).sum()
            vwap_den = historical_data['volume'].rolling(20, min_periods=1).sum()
            vwap = (vwap_num / vwap_den).iloc[-1]
            features['vwap_deviation'] = (current_price - vwap) / vwap if vwap > 0 else 0.0
            
            # Volume analysis: PostgreSQL average vs FMP real-time
            volume_ma = historical_data['volume'].rolling(20, min_periods=1).mean().iloc[-1]
            features['volume_ratio'] = current_quote.volume / volume_ma if volume_ma > 0 else 1.0
            
            # Real-time momentum (FMP current vs PostgreSQL last close)
            last_close = close_prices.iloc[-1]
            features['real_time_momentum'] = (current_price - last_close) / last_close
            
            # Bid-ask spread analysis (FMP only)
            if current_quote.bid > 0 and current_quote.ask > 0:
                spread = current_quote.ask - current_quote.bid
                features['spread_ratio'] = spread / current_price
            else:
                features['spread_ratio'] = 0.001  # Default small spread
            
            # RSI calculation from PostgreSQL historical data
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            features['rsi'] = rsi if not np.isnan(rsi) else 50
            
            # SMA trend with FMP real-time price vs PostgreSQL average
            sma_20 = close_prices.rolling(20, min_periods=1).mean().iloc[-1]
            features['sma_signal'] = 1 if current_price > sma_20 else -1
            
            # Price vs recent PostgreSQL average
            recent_avg = close_prices.tail(5).mean()
            features['price_vs_avg'] = (current_price - recent_avg) / recent_avg
            
            # Real-time volatility from FMP
            features['daily_change_pct'] = current_quote.change_percent / 100.0
            
        except Exception as e:
            logger.warning(f"⚠️ Hybrid feature calculation error: {e}")
        
        return features


class S3AIFMPIBHybridTrader:
    """Main S3 AI FMP-IB Hybrid Trading System."""
    
    def __init__(self, config_file: str = "s3_ai_fmp_trading_config_top_performers.json"):
        """Initialize S3 AI FMP-IB hybrid trader."""
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.running = False
        
        # Initialize components
        self.data_engine = S3AIDataEngine()  # PostgreSQL historical data
        self.fmp_provider = FMPDataProvider()  # FMP real-time data
        self.strategy = S3AIFMPIBStrategy(self.data_engine, self.fmp_provider)
        self.ib_manager = IBExecutionManager(self.config)  # IB execution
        
        # Trading parameters
        self.symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL"])
        self.scan_interval = self.config.get("scan_interval", 60)
        self.max_positions = self.config.get("max_positions", 10)
        self.initial_capital = self.config.get("initial_capital", 100000)
        
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
        
        logger.info("🤖 S3 AI FMP-IB Hybrid Trader initialized")
        logger.info(f"📊 Symbols: {len(self.symbols)} | Capital: ${self.initial_capital:,.0f}")
        logger.info(f"🏦 IB Execution: {self.config.get('ib_settings', {}).get('host', '127.0.0.1')}:{self.config.get('ib_settings', {}).get('port', 7497)}")
        logger.info(f"📡 FMP Real-time Data + 🗄️ PostgreSQL Historical Analysis")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load trading configuration."""
        default_config = {
            "initial_capital": 100000,
            "max_positions": 20,
            "scan_interval": 30,
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "risk_management": {
                "max_position_size": 0.04,
                "commission": 1.0,
                "min_signal_strength": 0.6
            },
            "ib_settings": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1
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
                
                # Add IB settings if not present
                if 'ib_settings' not in config:
                    config['ib_settings'] = default_config['ib_settings']
                
                logger.info(f"📝 Loaded config from {config_file}")
                return {**default_config, **config}
            else:
                # Create default config with IB settings
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
    
    def connect_to_ib(self) -> bool:
        """Connect to Interactive Brokers."""
        return self.ib_manager.connect()
    
    def calculate_position_size(self, symbol: str, price: float, signal_strength: float) -> int:
        """Calculate position size based on risk management."""
        try:
            # Get current IB account value
            account_data = self.ib_manager.get_account_summary()
            current_capital = account_data.get('NetLiquidation', self.initial_capital)
            
            # Base position size
            base_size = current_capital * self.max_position_size
            
            # Adjust by signal strength
            strength_multiplier = 0.5 + (signal_strength * 0.5)
            adjusted_size = base_size * strength_multiplier
            
            # Calculate shares
            shares = int(adjusted_size / price)
            
            # Minimum viable position
            min_shares = max(1, int(1000 / price))
            shares = max(shares, min_shares)
            
            # Maximum position limit
            max_shares = int((current_capital * self.max_position_size) / price)
            shares = min(shares, max_shares)
            
            return shares
            
        except Exception as e:
            logger.error(f"⚠️ Position sizing error for {symbol}: {e}")
            return 0
    
    def is_market_open(self) -> bool:
        """Check if market is open using FMP."""
        return self.fmp_provider.is_market_open()
    
    def scan_and_trade(self):
        """Main trading loop using FMP real-time data + IB execution."""
        try:
            logger.info(f"🔍 Scanning {len(self.symbols)} symbols with hybrid FMP-IB system...")
            
            # Get real-time quotes from FMP
            quotes = self.fmp_provider.get_real_time_quotes(self.symbols)
            
            if not quotes:
                logger.warning("⚠️ No FMP quotes received")
                return
            
            logger.debug(f"📊 Retrieved {len(quotes)} FMP quotes")
            
            # Update position tracking
            self.update_positions_from_ib()
            
            # Scan for trading opportunities
            for symbol in self.symbols:
                try:
                    if symbol not in quotes:
                        logger.warning(f"⚠️ No FMP data for {symbol}")
                        continue
                    
                    quote = quotes[symbol]
                    
                    # Skip if no valid price
                    if quote.price <= 0:
                        continue
                    
                    # Generate hybrid signal (PostgreSQL AI + FMP real-time)
                    action, strength, signal_info = self.strategy.generate_trading_signal(symbol, quote)
                    
                    regime = signal_info.get("regime", "Unknown")
                    confidence = signal_info.get("confidence", 0.0)
                    
                    if action != "HOLD" and strength >= self.min_signal_strength:
                        logger.info(f"🔍 {symbol}: {action} signal | Strength: {strength:.2f} | "
                                  f"Price: ${quote.price:.2f} | Regime: {regime} | "
                                  f"Change: {quote.change_percent:+.1f}%")
                        
                        # Execute trade through IB
                        if action == "BUY" and symbol not in self.positions:
                            if len(self.positions) < self.max_positions:
                                quantity = self.calculate_position_size(symbol, quote.price, strength)
                                if quantity > 0:
                                    success, fill_price, status = self.ib_manager.execute_market_order(
                                        symbol, "BUY", quantity
                                    )
                                    if success:
                                        self.record_trade(symbol, "BUY", quantity, fill_price, regime)
                        
                        elif action == "SELL" and symbol in self.positions:
                            position = self.positions[symbol]
                            quantity = int(abs(position.quantity))
                            success, fill_price, status = self.ib_manager.execute_market_order(
                                symbol, "SELL", quantity
                            )
                            if success:
                                self.record_trade(symbol, "SELL", quantity, fill_price, "Signal Exit")
                
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol}: {e}")
                    continue
            
            # Close positions near market close
            current_time = datetime.now().time()
            if current_time >= dt_time(15, 50):
                self.close_all_positions()
            
            self.last_scan_time = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Scan and trade error: {e}")
    
    def record_trade(self, symbol: str, action: str, quantity: int, price: float, regime: str):
        """Record executed trade."""
        try:
            # Create trade record
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
            
            # Update position tracking
            if action == "BUY":
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price,
                    regime=regime
                )
                logger.info(f"🟢 IB BUY executed: {quantity} {symbol} @ ${price:.2f}")
            
            elif action == "SELL" and symbol in self.positions:
                position = self.positions[symbol]
                pnl = (price - position.entry_price) * quantity - self.commission
                trade.pnl = pnl
                del self.positions[symbol]
                logger.info(f"🔴 IB SELL executed: {quantity} {symbol} @ ${price:.2f} | PnL: ${pnl:+.2f}")
            
        except Exception as e:
            logger.error(f"❌ Trade recording error: {e}")
    
    def update_positions_from_ib(self):
        """Update positions from IB account."""
        try:
            ib_positions = self.ib_manager.get_positions()
            
            # Update tracked positions with IB data
            current_symbols = set()
            for pos in ib_positions:
                if hasattr(pos, 'contract') and hasattr(pos.contract, 'symbol'):
                    symbol = pos.contract.symbol
                    current_symbols.add(symbol)
                    
                    if symbol in self.positions:
                        # Update existing position
                        self.positions[symbol].quantity = pos.position
                        self.positions[symbol].current_price = getattr(pos, 'marketPrice', 0)
                    elif pos.position != 0:
                        # New position from IB
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            quantity=pos.position,
                            entry_price=getattr(pos, 'avgCost', 0),
                            entry_time=datetime.now(),
                            current_price=getattr(pos, 'marketPrice', 0),
                            regime="Unknown"
                        )
            
            # Remove positions that no longer exist in IB
            to_remove = []
            for symbol in self.positions:
                if symbol not in current_symbols:
                    to_remove.append(symbol)
            
            for symbol in to_remove:
                del self.positions[symbol]
                
        except Exception as e:
            logger.warning(f"⚠️ IB position update warning: {e}")
    
    def close_all_positions(self):
        """Close all positions at end of day."""
        if not self.positions:
            return
        
        logger.info("🕐 Market close approaching, closing all IB positions...")
        
        for symbol in list(self.positions.keys()):
            try:
                position = self.positions[symbol]
                quantity = int(abs(position.quantity))
                
                if quantity > 0:
                    success, fill_price, status = self.ib_manager.execute_market_order(
                        symbol, "SELL", quantity
                    )
                    if success:
                        self.record_trade(symbol, "SELL", quantity, fill_price, "EOD Close")
            
            except Exception as e:
                logger.error(f"❌ EOD close error for {symbol}: {e}")
    
    def print_status(self):
        """Print current trading status."""
        print("\\n" + "="*80)
        print("🤖 S3 AI FMP-IB HYBRID TRADING STATUS")
        print("="*80)
        
        # Connection status
        ib_status = "🟢 CONNECTED" if self.ib_manager.connected else "🔴 DISCONNECTED"
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        print(f"🏦 IB Status: {ib_status}")
        print(f"📅 Market Status: {market_status}")
        
        # Account summary
        account_data = self.ib_manager.get_account_summary()
        if account_data:
            print(f"💰 Account Value: ${account_data.get('NetLiquidation', 0):,.2f}")
            print(f"💵 Available Funds: ${account_data.get('AvailableFunds', 0):,.2f}")
        
        # Positions
        if self.positions:
            print(f"\\n📋 ACTIVE POSITIONS ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                side = "LONG" if pos.quantity > 0 else "SHORT"
                print(f"   {symbol}: {side} {abs(pos.quantity):.0f} @ ${pos.entry_price:.2f} | "
                      f"Current: ${pos.current_price:.2f}")
        else:
            print("\\n📭 No active positions")
        
        # Trading summary
        total_trades = len(self.trades)
        if total_trades > 0:
            realized_pnl = sum(t.pnl for t in self.trades if hasattr(t, 'pnl') and t.pnl)
            print(f"\\n🔄 TRADING SUMMARY:")
            print(f"   Total Trades: {total_trades}")
            print(f"   Realized P&L: ${realized_pnl:+.2f}")
        
        if self.last_scan_time:
            print(f"⏰ Last Scan: {self.last_scan_time.strftime('%H:%M:%S')}")
        
        print("="*80)
    
    def run(self):
        """Start the S3 AI FMP-IB hybrid trading system."""
        
        logger.info("🚀 Starting S3 AI FMP-IB Hybrid Trading System...")
        logger.info("📡 Real-time data: FMP | 🗄️ Historical analysis: PostgreSQL | 🏦 Execution: Interactive Brokers")
        
        # Connect to IB first
        if not self.connect_to_ib():
            logger.error("❌ Failed to connect to IB. Exiting.")
            return
        
        self.running = True
        
        try:
            # Main trading loop
            while self.running:
                if self.is_market_open():
                    logger.info(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Hybrid market scan...")
                    self.scan_and_trade()
                    self.print_status()
                    time.sleep(self.scan_interval)
                else:
                    logger.info(f"🏪 Market closed - {datetime.now().strftime('%H:%M:%S')}")
                    self.print_status()
                    time.sleep(300)  # 5 minutes when closed
        
        except KeyboardInterrupt:
            logger.info("\\n🛑 Stopping S3 AI FMP-IB Hybrid Trading...")
            self.running = False
        
        finally:
            # Cleanup and save results
            self.close_all_positions()
            self.ib_manager.disconnect()
            self.save_results()
            logger.info("✅ S3 AI FMP-IB Hybrid Trading stopped")
    
    def save_results(self):
        """Save trading results."""
        try:
            # Get final account summary
            account_data = self.ib_manager.get_account_summary()
            final_value = account_data.get('NetLiquidation', self.initial_capital)
            
            results = {
                "session_start": datetime.now().isoformat(),
                "session_summary": {
                    "initial_capital": self.initial_capital,
                    "final_value": final_value,
                    "total_return_pct": ((final_value / self.initial_capital) - 1) * 100,
                    "positions_count": len(self.positions),
                    "total_trades": len(self.trades)
                },
                "positions": [asdict(pos) for pos in self.positions.values()],
                "trades": [asdict(trade) for trade in self.trades],
                "config": self.config,
                "data_sources": {
                    "real_time": "Financial Modeling Prep (FMP)",
                    "historical": "PostgreSQL Database",
                    "execution": "Interactive Brokers"
                }
            }
            
            filename = f"s3_ai_fmp_ib_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")


def main():
    """Main entry point with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='S3 AI FMP-IB Hybrid Trading System')
    parser.add_argument('--config', 
                       default='s3_ai_fmp_trading_config_top_performers.json',
                       help='Configuration file (default: top performers config)')
    parser.add_argument('--start', action='store_true', default=True,
                       help='Start trading system (default action)')
    parser.add_argument('--test-connection', action='store_true',
                       help='Test IB connection only')
    parser.add_argument('--status', action='store_true',
                       help='Show system status only')
    
    # If no arguments provided, default to --start
    if len(sys.argv) == 1:
        args = parser.parse_args(['--start'])
    else:
        args = parser.parse_args()
        # Override default if other actions specified
        if args.test_connection or args.status:
            args.start = False
    
    print("🤖 S3 AI FMP-IB Hybrid Trading System")
    print("=" * 50)
    print("📡 Real-time data: Financial Modeling Prep")
    print("🗄️ Historical analysis: PostgreSQL Database")
    print("🏦 Execution: Interactive Brokers")
    print(f"📊 Using config: {args.config}")
    print()
    
    try:
        if args.test_connection:
            print("🔌 Testing IB connection...")
            trader = S3AIFMPIBHybridTrader(args.config)
            if trader.connect_to_ib():
                print("✅ IB connection successful!")
                trader.ib_manager.disconnect()
            else:
                print("❌ IB connection failed!")
            return
        
        # Initialize trader with specified config
        trader = S3AIFMPIBHybridTrader(args.config)
        
        if args.status:
            print("📊 System status:")
            trader.connect_to_ib()
            trader.print_status()
            trader.ib_manager.disconnect()
            return
        
        if args.start:
            print("🚀 Starting hybrid live trading system...")
            trader.run()
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print('\\n🛑 Shutdown signal received')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    main()