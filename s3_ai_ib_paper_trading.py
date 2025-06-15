#!/usr/bin/env python3
"""
S3 AI Interactive Brokers Paper Trading System
==============================================
Combines S3 AI strategy with Interactive Brokers TWS for real paper trading.

Features:
- S3 AI indicator selection and regime detection
- Real-time IB price data and order execution
- PostgreSQL for historical analysis and backtesting
- Risk management and position controls
- Performance tracking and monitoring

Requires: TWS/IB Gateway running on localhost:7497
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
        logging.FileHandler(f's3_ai_ib_trading_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class IBPosition:
    """IB position with S3 AI metadata."""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    entry_time: datetime
    regime: str = "Unknown"
    signal_strength: float = 0.0


class S3AIIBStrategy:
    """S3 AI strategy optimized for IB execution."""
    
    def __init__(self, data_engine: S3AIDataEngine):
        self.data_engine = data_engine
        self.signal_cache = {}
        self.last_analysis_time = {}
        
        logger.info("🧠 S3 AI IB Strategy initialized")
    
    def generate_ib_signal(self, symbol: str, current_price: float, 
                          force_refresh: bool = False) -> Tuple[str, float, Dict]:
        """Generate trading signal optimized for IB execution."""
        
        # Check cache first (refresh every 5 minutes for IB)
        cache_key = symbol
        now = datetime.now()
        if (not force_refresh and cache_key in self.signal_cache and 
            cache_key in self.last_analysis_time):
            last_time = self.last_analysis_time[cache_key]
            if (now - last_time).seconds < 300:  # 5 minutes
                cached_signal = self.signal_cache[cache_key]
                return cached_signal['action'], cached_signal['strength'], cached_signal['info']
        
        try:
            # Get recent data for analysis
            data = self.data_engine.get_recent_data(symbol, bars=100)
            if data.empty or len(data) < 20:
                return "HOLD", 0.0, {"error": "Insufficient data"}
            
            # Get AI analysis
            ai_analysis = self.data_engine.get_ai_analysis(symbol, force_refresh=force_refresh)
            regime = ai_analysis.get("regime", "Unknown")
            indicators = ai_analysis.get("indicators", {})
            confidence = ai_analysis.get("confidence", 0.0)
            
            # Calculate IB-optimized features
            features = self._calculate_ib_features(data, current_price)
            
            # Generate signal based on regime and IB constraints
            signal_strength = 0.0
            
            # IB-specific signal logic (more conservative)
            if regime in ['High Volatility', 'Mean Reversion']:
                # VWAP mean reversion for volatile markets
                vwap_dev = features.get('vwap_deviation', 0.0)
                if vwap_dev < -0.003:  # Stronger signal needed for IB
                    signal_strength += 0.5
                elif vwap_dev > 0.003:
                    signal_strength -= 0.5
                
                # RSI confirmation
                if indicators.get('rsi_momentum', False):
                    rsi = features.get('rsi', 50)
                    if rsi < 25:  # More extreme levels for IB
                        signal_strength += 0.3
                    elif rsi > 75:
                        signal_strength -= 0.3
            
            elif regime in ['Trending', 'Breakout']:
                # Momentum strategy with volume confirmation
                if indicators.get('volume_breakout', False):
                    volume_ratio = features.get('volume_ratio', 1.0)
                    price_momentum = features.get('returns_5', 0.0)
                    
                    if volume_ratio > 2.0 and price_momentum > 0.002:  # Higher thresholds
                        signal_strength += 0.6
                    elif volume_ratio > 2.0 and price_momentum < -0.002:
                        signal_strength -= 0.6
            
            # Conservative mixed signals for unknown regimes
            else:
                if indicators.get('sma_trend', False):
                    sma_signal = features.get('sma_signal', 0)
                    signal_strength += sma_signal * 0.15  # Reduced for conservatism
            
            # IB-specific signal strength adjustment
            signal_strength = max(-1.0, min(1.0, signal_strength))  # Clamp to [-1, 1]
            
            # Determine action with IB-appropriate thresholds
            if signal_strength > 0.7:  # Higher threshold for IB
                action = "BUY"
            elif signal_strength < -0.7:
                action = "SELL"
            else:
                action = "HOLD"
            
            signal_info = {
                "strength": signal_strength,
                "regime": regime,
                "confidence": confidence,
                "indicators_active": len([k for k, v in indicators.items() if v]),
                "features": features,
                "ib_optimized": True
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
            logger.error(f"❌ IB signal generation failed for {symbol}: {e}")
            return "HOLD", 0.0, {"error": str(e)}
    
    def _calculate_ib_features(self, data: pd.DataFrame, current_price: float) -> Dict:
        """Calculate features optimized for IB trading."""
        
        if len(data) < 20:
            return {}
        
        features = {}
        
        try:
            # Price features
            close_prices = data['close']
            features['returns_5'] = (current_price / close_prices.iloc[-5] - 1) if len(close_prices) >= 5 else 0.0
            
            # VWAP calculation (20-period)
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap_num = (typical_price * data['volume']).rolling(20, min_periods=1).sum()
            vwap_den = data['volume'].rolling(20, min_periods=1).sum()
            vwap = (vwap_num / vwap_den).iloc[-1]
            features['vwap_deviation'] = (current_price - vwap) / vwap if vwap > 0 else 0.0
            
            # Volume analysis
            volume_ma = data['volume'].rolling(20, min_periods=1).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            features['volume_ratio'] = current_volume / volume_ma if volume_ma > 0 else 1.0
            
            # RSI (14-period)
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            features['rsi'] = rsi if not np.isnan(rsi) else 50
            
            # SMA trend (20-period)
            sma_20 = close_prices.rolling(20, min_periods=1).mean().iloc[-1]
            features['sma_signal'] = 1 if current_price > sma_20 else -1
            
            # Volatility (for position sizing)
            returns = close_prices.pct_change().dropna()
            features['volatility'] = returns.rolling(20, min_periods=1).std().iloc[-1] * np.sqrt(252)
            
        except Exception as e:
            logger.warning(f"⚠️ IB feature calculation error: {e}")
        
        return features


class S3AIIBPaperTrader:
    """Main S3 AI Interactive Brokers paper trading system."""
    
    def __init__(self, config_file: str = "s3_ai_ib_trading_config.json"):
        """Initialize S3 AI IB paper trader."""
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.running = False
        
        # Initialize S3 AI components
        self.data_engine = S3AIDataEngine()
        self.strategy = S3AIIBStrategy(self.data_engine)
        
        # Initialize IB components
        self.ib = IB()
        self.ib_connected = False
        
        # Trading parameters
        self.symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL"])
        self.scan_interval = self.config.get("scan_interval", 60)
        self.max_positions = self.config.get("max_positions", 10)
        self.initial_capital = self.config.get("initial_capital", 100000)
        
        # Risk management
        risk_config = self.config.get("risk_management", {})
        self.max_position_size = risk_config.get("max_position_size", 0.05)
        self.commission = risk_config.get("commission", 1.0)
        self.min_signal_strength = risk_config.get("min_signal_strength", 0.7)
        
        # Market hours
        market_hours = self.config.get("market_hours", {})
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(15, 55)
        
        # State tracking
        self.positions = {}  # symbol -> IBPosition
        self.trades = []
        self.last_scan_time = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info("🤖 S3 AI IB Paper Trader initialized")
        logger.info(f"📊 Symbols: {len(self.symbols)} | Capital: ${self.initial_capital:,.0f}")
        logger.info(f"⏱️  Scan interval: {self.scan_interval}s | Max positions: {self.max_positions}")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load trading configuration."""
        default_config = {
            "initial_capital": 100000,
            "max_positions": 10,
            "scan_interval": 60,
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "ib_settings": {
                "host": "127.0.0.1",
                "port": 7497,
                "client_id": 1
            },
            "risk_management": {
                "max_position_size": 0.05,
                "commission": 1.0,
                "min_signal_strength": 0.7
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
    
    def connect_ib(self) -> bool:
        """Connect to Interactive Brokers TWS."""
        try:
            ib_settings = self.config.get('ib_settings', {})
            
            logger.info("🔌 Connecting to Interactive Brokers TWS...")
            self.ib.connect(
                host=ib_settings.get('host', '127.0.0.1'),
                port=ib_settings.get('port', 7497),
                clientId=ib_settings.get('client_id', 1),
                timeout=15
            )
            
            if self.ib.isConnected():
                logger.info("✅ IB connected successfully!")
                self.ib_connected = True
                
                # Get account info
                try:
                    account_summary = self.ib.accountSummary()
                    for item in account_summary:
                        if item.tag == 'NetLiquidation':
                            logger.info(f"💰 IB Account Value: ${float(item.value):,.2f}")
                            break
                except Exception as e:
                    logger.warning(f"⚠️ Could not get IB account info: {e}")
                
                return True
            else:
                logger.error("❌ IB connection failed!")
                return False
                
        except Exception as e:
            logger.error(f"❌ IB connection error: {e}")
            logger.info("💡 Make sure TWS or IB Gateway is running on localhost:7497")
            return False
    
    def disconnect_ib(self):
        """Disconnect from IB."""
        if self.ib_connected and self.ib.isConnected():
            self.ib.disconnect()
            self.ib_connected = False
            logger.info("🔌 Disconnected from IB")
    
    def get_ib_price(self, symbol: str) -> Optional[float]:
        """Get current market price from IB."""
        try:
            if not self.ib_connected:
                return None
            
            contract = Stock(symbol, 'SMART', 'USD')
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(2)  # Wait for price data
            
            if ticker.marketPrice() and ticker.marketPrice() > 0:
                price = ticker.marketPrice()
                self.ib.cancelMktData(contract)
                return price
            else:
                # Try last price
                if ticker.last > 0:
                    price = ticker.last
                    self.ib.cancelMktData(contract)
                    return price
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting IB price for {symbol}: {e}")
        
        return None
    
    def calculate_position_size(self, symbol: str, price: float, signal_strength: float) -> int:
        """Calculate position size based on risk management."""
        try:
            # Get account value
            account_value = self.initial_capital  # Fallback
            if self.ib_connected:
                try:
                    account_summary = self.ib.accountSummary()
                    for item in account_summary:
                        if item.tag == 'NetLiquidation':
                            account_value = float(item.value)
                            break
                except:
                    pass
            
            # Base position size
            base_size = account_value * self.max_position_size
            
            # Adjust by signal strength
            strength_multiplier = 0.5 + (signal_strength * 0.5)
            adjusted_size = base_size * strength_multiplier
            
            # Calculate shares
            shares = int(adjusted_size / price)
            
            # Minimum viable position
            min_shares = max(1, int(1000 / price))
            shares = max(shares, min_shares)
            
            # Maximum position limit
            max_shares = int((account_value * self.max_position_size) / price)
            shares = min(shares, max_shares)
            
            return shares
            
        except Exception as e:
            logger.error(f"⚠️ Position sizing error for {symbol}: {e}")
            return 0
    
    def place_ib_order(self, symbol: str, action: str, quantity: int, price: float) -> bool:
        """Place order through IB."""
        try:
            if not self.ib_connected:
                logger.error("❌ Cannot place order: IB not connected")
                return False
            
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Use market order for simplicity (could add limit orders later)
            order = MarketOrder(action, quantity)
            
            trade = self.ib.placeOrder(contract, order)
            
            logger.info(f"📤 IB Order placed: {action} {quantity} {symbol} @ Market")
            
            # Wait for fill (up to 30 seconds)
            for _ in range(30):
                self.ib.sleep(1)
                if trade.isDone():
                    fill_price = trade.orderStatus.avgFillPrice
                    logger.info(f"✅ Order filled: {symbol} @ ${fill_price:.2f}")
                    return True
            
            logger.warning(f"⚠️ Order not filled within timeout: {symbol}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Order placement error for {symbol}: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """Check if market is open."""
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Market hours check
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close
    
    def scan_and_trade(self):
        """Main trading loop."""
        try:
            if not self.ib_connected:
                logger.error("❌ Cannot scan: IB not connected")
                return
            
            logger.info(f"🔍 Scanning {len(self.symbols)} symbols...")
            
            # Update existing positions
            self.update_ib_positions()
            
            # Scan for new opportunities
            for symbol in self.symbols:
                try:
                    # Get current price from IB
                    current_price = self.get_ib_price(symbol)
                    if not current_price:
                        logger.warning(f"⚠️ No price data for {symbol}")
                        continue
                    
                    # Generate S3 AI signal
                    action, strength, signal_info = self.strategy.generate_ib_signal(symbol, current_price)
                    
                    regime = signal_info.get("regime", "Unknown")
                    confidence = signal_info.get("confidence", 0.0)
                    
                    if action != "HOLD" and strength >= self.min_signal_strength:
                        logger.info(f"🔍 {symbol}: {action} signal | Strength: {strength:.2f} | "
                                  f"Regime: {regime} | Confidence: {confidence:.2f}")
                        
                        # Execute trade
                        if action == "BUY" and symbol not in self.positions:
                            if len(self.positions) < self.max_positions:
                                quantity = self.calculate_position_size(symbol, current_price, strength)
                                if quantity > 0:
                                    success = self.place_ib_order(symbol, "BUY", quantity, current_price)
                                    if success:
                                        self.record_trade(symbol, "BUY", quantity, current_price, regime)
                        
                        elif action == "SELL" and symbol in self.positions:
                            position = self.positions[symbol]
                            quantity = int(abs(position.quantity))
                            success = self.place_ib_order(symbol, "SELL", quantity, current_price)
                            if success:
                                self.record_trade(symbol, "SELL", quantity, current_price, regime)
                
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
    
    def update_ib_positions(self):
        """Update positions from IB."""
        try:
            if not self.ib_connected:
                return
            
            ib_positions = self.ib.positions()
            current_positions = {}
            
            for pos in ib_positions:
                if pos.contract.secType == 'STK' and pos.position != 0:
                    symbol = pos.contract.symbol
                    
                    # Get current market value
                    ticker = self.ib.reqMktData(pos.contract, '', False, False)
                    self.ib.sleep(1)
                    market_price = ticker.marketPrice() or ticker.last
                    self.ib.cancelMktData(pos.contract)
                    
                    if market_price:
                        market_value = pos.position * market_price
                        unrealized_pnl = market_value - (pos.position * pos.avgCost)
                        
                        current_positions[symbol] = IBPosition(
                            symbol=symbol,
                            quantity=pos.position,
                            avg_cost=pos.avgCost,
                            market_value=market_value,
                            unrealized_pnl=unrealized_pnl,
                            entry_time=self.positions.get(symbol, {}).get('entry_time', datetime.now()),
                            regime=getattr(self.positions.get(symbol), 'regime', 'Unknown')
                        )
            
            self.positions = current_positions
            
        except Exception as e:
            logger.error(f"❌ Error updating IB positions: {e}")
    
    def record_trade(self, symbol: str, side: str, quantity: int, price: float, regime: str):
        """Record completed trade."""
        trade = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "commission": self.commission
        }
        self.trades.append(trade)
        logger.info(f"📝 Trade recorded: {side} {quantity} {symbol} @ ${price:.2f}")
    
    def close_all_positions(self):
        """Close all open positions (end of day)."""
        if not self.positions:
            return
        
        logger.info("🕐 Market close approaching, closing all positions...")
        
        for symbol, position in list(self.positions.items()):
            try:
                quantity = int(abs(position.quantity))
                action = "SELL" if position.quantity > 0 else "BUY"
                
                current_price = self.get_ib_price(symbol)
                if current_price:
                    success = self.place_ib_order(symbol, action, quantity, current_price)
                    if success:
                        self.record_trade(symbol, action, quantity, current_price, "EOD Close")
                        
            except Exception as e:
                logger.error(f"❌ Error closing position {symbol}: {e}")
    
    def print_status(self):
        """Print current trading status."""
        print("\n" + "="*80)
        print("🤖 S3 AI INTERACTIVE BROKERS PAPER TRADING STATUS")
        print("="*80)
        
        # IB connection status
        ib_status = "🟢 CONNECTED" if self.ib_connected else "🔴 DISCONNECTED"
        print(f"🔌 IB Connection: {ib_status}")
        
        # Market status
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        print(f"📅 Market Status: {market_status}")
        
        # Portfolio summary
        total_value = 0
        unrealized_pnl = 0
        
        if self.positions:
            print(f"\n📋 ACTIVE POSITIONS ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                side = "LONG" if pos.quantity > 0 else "SHORT"
                pnl_pct = (pos.unrealized_pnl / abs(pos.quantity * pos.avg_cost)) * 100
                print(f"   {symbol}: {side} {abs(pos.quantity):.0f} @ ${pos.avg_cost:.2f} | "
                      f"Current: ${pos.market_value/pos.quantity:.2f} | "
                      f"P&L: ${pos.unrealized_pnl:+.2f} ({pnl_pct:+.1f}%)")
                total_value += pos.market_value
                unrealized_pnl += pos.unrealized_pnl
        else:
            print("\n📭 No active positions")
        
        # Trading summary
        total_trades = len(self.trades)
        if total_trades > 0:
            winning_trades = len([t for t in self.trades if 
                                self._calculate_trade_pnl(t) > 0])
            win_rate = (winning_trades / total_trades) * 100
            print(f"\n🔄 TRADING SUMMARY:")
            print(f"   Total Trades: {total_trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
        
        print(f"\n💰 Portfolio Value: ${total_value:,.2f}")
        print(f"📊 Unrealized P&L: ${unrealized_pnl:+,.2f}")
        
        if self.last_scan_time:
            print(f"⏰ Last Scan: {self.last_scan_time.strftime('%H:%M:%S')}")
        
        print("="*80)
    
    def _calculate_trade_pnl(self, trade: Dict) -> float:
        """Calculate P&L for a completed trade pair."""
        # Simplified P&L calculation
        return 0.0  # Would need trade pairing logic
    
    def run(self):
        """Start the S3 AI IB paper trading system."""
        
        logger.info("🚀 Starting S3 AI IB Paper Trading System...")
        
        # Connect to IB
        if not self.connect_ib():
            logger.error("❌ Failed to connect to IB. Exiting.")
            return
        
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
                    time.sleep(300)  # 5 minutes when closed
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping S3 AI IB Paper Trading...")
            self.running = False
        
        finally:
            # Cleanup
            self.disconnect_ib()
            self.save_results()
            logger.info("✅ S3 AI IB Paper Trading stopped")
    
    def save_results(self):
        """Save trading results."""
        try:
            results = {
                "session_start": datetime.now().isoformat(),
                "ib_connected": self.ib_connected,
                "positions": [asdict(pos) for pos in self.positions.values()],
                "trades": self.trades,
                "config": self.config
            }
            
            filename = f"s3_ai_ib_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")


def main():
    """Main entry point."""
    
    print("🤖 S3 AI Interactive Brokers Paper Trading System")
    print("=" * 60)
    print("📋 Make sure TWS/IB Gateway is running on localhost:7497")
    print("🔒 Use PAPER TRADING account only!")
    print()
    
    try:
        # Initialize trader
        trader = S3AIIBPaperTrader()
        
        # Start trading
        trader.run()
        
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print('\n🛑 Shutdown signal received')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    main()