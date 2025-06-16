#!/usr/bin/env python3
"""
S3 AI FMP-IB Hybrid Trading System
==================================
Uses FMP for market data and Interactive Brokers for execution.

Features:
- Financial Modeling Prep API for real-time market data
- Interactive Brokers for real paper/live trading execution
- Best of both worlds: FMP data quality + IB execution
- S3 AI strategy with regime detection
- Risk management and position controls

Requires:
- FMP API key (Ultimate tier)
- TWS/IB Gateway running on localhost:7497
"""

import sys
import os
import signal
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import warnings
import pytz
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# IB components
from ib_insync import IB, Stock, MarketOrder, LimitOrder, Contract
from ib_insync.util import df

# S3 AI components
from s3_ai_wrapper import S3AIWrapper
from s3_ai_hybrid_paper_trading import S3AIDataEngine, Position, Trade
from s3_ai_fmp_hybrid_trading import FMPDataProvider, FMPQuote

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
class HybridPosition:
    """Position tracking for FMP-IB hybrid system."""
    symbol: str
    quantity: float
    avg_cost: float
    ib_order_id: int
    entry_time: datetime
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    regime: str = "Unknown"
    signal_strength: float = 0.0


class S3AIFMPIBHybrid:
    """S3 AI system using FMP data with IB execution."""
    
    def __init__(self, config_file: str = "s3_ai_fmp_ib_config.json"):
        """Initialize FMP-IB hybrid trader."""
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.running = False
        
        # Initialize S3 AI components
        self.data_engine = S3AIDataEngine()
        self.s3_wrapper = S3AIWrapper(self.data_engine)
        
        # Initialize data provider (FMP)
        self.fmp_provider = FMPDataProvider()
        
        # Initialize IB connection
        self.ib = IB()
        self.ib_connected = False
        
        # Trading parameters
        self.symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL"])
        self.scan_interval = self.config.get("scan_interval", 300)
        self.max_positions = self.config.get("max_positions", 10)
        self.initial_capital = self.config.get("initial_capital", 100000)
        
        # Risk management
        risk_config = self.config.get("risk_management", {})
        self.position_size = risk_config.get("position_size", 10000)
        self.max_position_size = risk_config.get("max_position_size", 0.05)
        self.stop_loss = risk_config.get("stop_loss", 0.02)
        self.take_profit = risk_config.get("take_profit", 0.03)
        self.min_signal_strength = risk_config.get("min_signal_strength", 0.7)
        
        # Market hours (Eastern Time)
        self.market_open = dt_time(9, 30)  # 9:30 AM ET
        self.market_close = dt_time(15, 55)  # 3:55 PM ET
        self.eastern_tz = pytz.timezone('America/New_York')
        self.local_tz = pytz.timezone('America/Los_Angeles')  # Pacific Time
        
        # State tracking
        self.positions: Dict[str, HybridPosition] = {}
        self.trades: List[Trade] = []
        self.pending_orders = {}
        self.last_scan_time = None
        
        # Performance tracking
        self.starting_capital = self.initial_capital
        self.current_capital = self.initial_capital
        
        logger.info("🚀 S3 AI FMP-IB Hybrid System initialized")
        logger.info(f"📊 Data: FMP API | Execution: Interactive Brokers")
        logger.info(f"💰 Capital: ${self.initial_capital:,.0f} | Symbols: {len(self.symbols)}")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load trading configuration."""
        default_config = {
            "initial_capital": 100000,
            "max_positions": 10,
            "scan_interval": 300,
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "ib_settings": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1
            },
            "risk_management": {
                "position_size": 10000,
                "max_position_size": 0.05,
                "stop_loss": 0.02,
                "take_profit": 0.03,
                "min_signal_strength": 0.7
            }
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"📝 Loaded config from {config_file}")
                return {**default_config, **config}
        except Exception as e:
            logger.warning(f"⚠️ Config load error: {e}, using defaults")
        
        return default_config
    
    def connect_ib(self) -> bool:
        """Connect to Interactive Brokers."""
        try:
            ib_settings = self.config.get("ib_settings", {})
            host = ib_settings.get("host", "127.0.0.1")
            port = ib_settings.get("port", 7497)
            client_id = ib_settings.get("client_id", 1)
            
            logger.info(f"🔌 Connecting to IB at {host}:{port}...")
            self.ib.connect(host, port, clientId=client_id)
            
            # Wait for connection
            time.sleep(2)
            
            if self.ib.isConnected():
                self.ib_connected = True
                logger.info("✅ Connected to Interactive Brokers")
                
                # Request account info
                self._update_account_info()
                return True
            else:
                logger.error("❌ Failed to connect to IB")
                return False
                
        except Exception as e:
            logger.error(f"❌ IB connection error: {e}")
            return False
    
    def _update_account_info(self):
        """Update account information from IB."""
        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    self.current_capital = float(av.value)
                    logger.info(f"💵 Account value: ${self.current_capital:,.2f}")
                    break
        except Exception as e:
            logger.warning(f"⚠️ Could not get account info: {e}")
    
    def _get_fmp_quote(self, symbol: str) -> Optional[FMPQuote]:
        """Get real-time quote from FMP."""
        return self.fmp_provider.get_quote(symbol)
    
    def _create_ib_order(self, symbol: str, quantity: int, order_type: str = "MKT") -> Any:
        """Create IB order."""
        contract = Stock(symbol, 'SMART', 'USD')
        
        if order_type == "MKT":
            order = MarketOrder('BUY' if quantity > 0 else 'SELL', abs(quantity))
        else:
            # Could add limit order logic here
            order = MarketOrder('BUY' if quantity > 0 else 'SELL', abs(quantity))
        
        return self.ib.placeOrder(contract, order)
    
    def analyze_and_trade(self, symbol: str) -> Optional[Dict]:
        """Analyze symbol using FMP data and execute via IB."""
        try:
            # Get FMP quote
            quote = self._get_fmp_quote(symbol)
            if not quote:
                logger.warning(f"⚠️ No FMP quote for {symbol}")
                return None
            
            current_price = quote.price
            logger.info(f"📊 {symbol}: ${current_price:.2f} ({quote.change_percent:+.2f}%)")
            
            # Get S3 AI signal using FMP price
            action, signal_strength, info = self.s3_wrapper.get_trading_signal(
                symbol, current_price
            )
            
            # Log signal
            regime = info.get('regime', 'Unknown')
            logger.info(f"🎯 {symbol}: {action} | Strength: {signal_strength:.2f} | Regime: {regime}")
            
            # Check if we should trade
            if action == "HOLD" or signal_strength < self.min_signal_strength:
                return None
            
            # Check position limits
            if action == "BUY" and len(self.positions) >= self.max_positions:
                logger.warning(f"⚠️ Max positions reached ({self.max_positions})")
                return None
            
            # Execute trade via IB
            if action == "BUY" and symbol not in self.positions:
                return self._execute_buy(symbol, quote, signal_strength, info)
            elif action == "SELL" and symbol in self.positions:
                return self._execute_sell(symbol, quote, signal_strength, info)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    def _execute_buy(self, symbol: str, quote: FMPQuote, signal_strength: float, info: Dict) -> Optional[Dict]:
        """Execute buy order via IB."""
        try:
            # Calculate position size
            position_value = min(self.position_size, self.current_capital * self.max_position_size)
            shares = int(position_value / quote.price)
            
            if shares < 1:
                logger.warning(f"⚠️ Insufficient capital for {symbol}")
                return None
            
            # Place IB order
            logger.info(f"🛒 Buying {shares} shares of {symbol} at ${quote.price:.2f}")
            trade = self._create_ib_order(symbol, shares, "MKT")
            
            # Wait for fill
            time.sleep(2)
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                
                # Create position record
                position = HybridPosition(
                    symbol=symbol,
                    quantity=shares,
                    avg_cost=fill_price,
                    ib_order_id=trade.order.orderId,
                    entry_time=datetime.now(),
                    entry_price=fill_price,
                    current_price=quote.price,
                    regime=info.get('regime', 'Unknown'),
                    signal_strength=signal_strength
                )
                
                self.positions[symbol] = position
                
                # Record trade
                trade_record = Trade(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action="BUY",
                    quantity=shares,
                    price=fill_price,
                    value=shares * fill_price,
                    signal_strength=signal_strength,
                    regime=info.get('regime', 'Unknown')
                )
                self.trades.append(trade_record)
                
                logger.info(f"✅ Bought {shares} {symbol} @ ${fill_price:.2f} (IB Order: {trade.order.orderId})")
                
                return {
                    "action": "BUY",
                    "symbol": symbol,
                    "shares": shares,
                    "price": fill_price,
                    "value": shares * fill_price
                }
            else:
                logger.warning(f"⚠️ Order not filled: {trade.orderStatus.status}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Buy execution error: {e}")
            return None
    
    def _execute_sell(self, symbol: str, quote: FMPQuote, signal_strength: float, info: Dict) -> Optional[Dict]:
        """Execute sell order via IB."""
        try:
            position = self.positions.get(symbol)
            if not position:
                return None
            
            shares = position.quantity
            
            # Place IB order
            logger.info(f"💰 Selling {shares} shares of {symbol} at ${quote.price:.2f}")
            trade = self._create_ib_order(symbol, -shares, "MKT")
            
            # Wait for fill
            time.sleep(2)
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                
                # Calculate P&L
                pnl = (fill_price - position.avg_cost) * shares
                pnl_pct = (fill_price - position.avg_cost) / position.avg_cost * 100
                
                # Record trade
                trade_record = Trade(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action="SELL",
                    quantity=shares,
                    price=fill_price,
                    value=shares * fill_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    signal_strength=signal_strength,
                    regime=info.get('regime', 'Unknown')
                )
                self.trades.append(trade_record)
                
                # Remove position
                del self.positions[symbol]
                
                # Update capital
                self.current_capital += pnl
                
                logger.info(f"✅ Sold {shares} {symbol} @ ${fill_price:.2f} | P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                
                return {
                    "action": "SELL",
                    "symbol": symbol,
                    "shares": shares,
                    "price": fill_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct
                }
            else:
                logger.warning(f"⚠️ Sell order not filled: {trade.orderStatus.status}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Sell execution error: {e}")
            return None
    
    def update_positions(self):
        """Update position values with FMP prices."""
        for symbol, position in self.positions.items():
            quote = self._get_fmp_quote(symbol)
            if quote:
                position.current_price = quote.price
                position.unrealized_pnl = (quote.price - position.avg_cost) * position.quantity
    
    def check_stop_loss(self):
        """Check positions for stop loss using FMP prices."""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            quote = self._get_fmp_quote(symbol)
            if not quote:
                continue
            
            # Calculate return
            returns = (quote.price - position.entry_price) / position.entry_price
            
            # Check stop loss
            if returns <= -self.stop_loss:
                logger.warning(f"🛑 Stop loss triggered for {symbol} at {returns:.2%}")
                positions_to_close.append(symbol)
            
            # Check take profit
            elif returns >= self.take_profit:
                logger.info(f"🎯 Take profit triggered for {symbol} at {returns:.2%}")
                positions_to_close.append(symbol)
        
        # Close positions
        for symbol in positions_to_close:
            quote = self._get_fmp_quote(symbol)
            if quote:
                self._execute_sell(symbol, quote, 1.0, {"reason": "stop_loss/take_profit"})
    
    def scan_market(self):
        """Scan all symbols for trading opportunities."""
        logger.info("🔍 Starting market scan...")
        
        # Update positions first
        self.update_positions()
        
        # Check stop losses
        self.check_stop_loss()
        
        # Scan for new opportunities
        for symbol in self.symbols:
            if symbol not in self.positions:
                self.analyze_and_trade(symbol)
        
        # Show summary
        self._show_portfolio_summary()
    
    def _show_portfolio_summary(self):
        """Display current portfolio status."""
        if not self.positions:
            logger.info("📊 No open positions")
            return
        
        total_value = 0
        total_pnl = 0
        
        logger.info("\n📊 PORTFOLIO SUMMARY")
        logger.info("-" * 60)
        
        for symbol, pos in self.positions.items():
            value = pos.quantity * pos.current_price
            pnl = pos.unrealized_pnl
            pnl_pct = (pnl / (pos.quantity * pos.avg_cost)) * 100
            
            total_value += value
            total_pnl += pnl
            
            logger.info(f"{symbol:6} | {pos.quantity:5} @ ${pos.avg_cost:7.2f} | "
                       f"Now: ${pos.current_price:7.2f} | "
                       f"P&L: ${pnl:8.2f} ({pnl_pct:+6.2f}%)")
        
        logger.info("-" * 60)
        logger.info(f"TOTAL  | Positions: {len(self.positions)} | "
                   f"Value: ${total_value:,.2f} | P&L: ${total_pnl:+,.2f}")
        
        # Account summary
        total_return = ((self.current_capital - self.starting_capital) / self.starting_capital) * 100
        logger.info(f"\n💼 Account: ${self.current_capital:,.2f} ({total_return:+.2f}% return)")
    
    def run(self):
        """Run the trading system."""
        logger.info("🚀 Starting S3 AI FMP-IB Hybrid Trading System")
        
        # Connect to IB
        if not self.connect_ib():
            logger.error("❌ Failed to connect to IB. Exiting.")
            return
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = True
        
        try:
            while self.running:
                # Get current time in Eastern timezone
                now_et = datetime.now(self.eastern_tz)
                current_time_et = now_et.time()
                
                # Check market hours (in Eastern Time)
                if current_time_et < self.market_open or current_time_et > self.market_close:
                    logger.info(f"🌙 Market closed. ET: {current_time_et.strftime('%I:%M %p')} | PT: {datetime.now().strftime('%I:%M %p')}")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                logger.info(f"📈 Market open. ET: {current_time_et.strftime('%I:%M %p')} | PT: {datetime.now().strftime('%I:%M %p')}")
                
                # Scan market
                self.scan_market()
                
                # Wait for next scan
                logger.info(f"⏱️  Next scan in {self.scan_interval} seconds...")
                time.sleep(self.scan_interval)
                
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
        finally:
            self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("📴 Shutdown signal received")
        self.running = False
    
    def shutdown(self):
        """Clean shutdown."""
        logger.info("🔚 Shutting down...")
        
        # Show final summary
        self._show_portfolio_summary()
        
        # Save trades
        if self.trades:
            self._save_trades()
        
        # Disconnect from IB
        if self.ib_connected:
            self.ib.disconnect()
            logger.info("🔌 Disconnected from IB")
        
        logger.info("👋 Shutdown complete")
    
    def _save_trades(self):
        """Save trade history."""
        try:
            trades_df = pd.DataFrame([asdict(t) for t in self.trades])
            filename = f"s3_fmp_ib_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trades_df.to_csv(filename, index=False)
            logger.info(f"💾 Saved {len(self.trades)} trades to {filename}")
        except Exception as e:
            logger.error(f"❌ Error saving trades: {e}")


if __name__ == "__main__":
    # Check for config file argument
    config_file = "s3_ai_fmp_ib_config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # Create and run trader
    trader = S3AIFMPIBHybrid(config_file)
    trader.run()