#!/usr/bin/env python3
"""
MLX Real-Time Market Scanner
============================
Scan thousands of symbols in real-time using MLX's vectorized operations.
"""

import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
import mlx.core as mx
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests

from mlx_vectorized_indicators import MLXIndicatorEngine
from indicator_transformer_mlx import IndicatorTransformerMLX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScanCriteria:
    """Criteria for market scanning."""
    # Technical indicators
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # Volume
    volume_spike_threshold: float = 2.0  # 2x average volume
    
    # Price action
    price_breakout_pct: float = 2.0  # 2% above resistance
    price_breakdown_pct: float = -2.0  # 2% below support
    
    # Momentum
    macd_bullish_cross: bool = True
    macd_bearish_cross: bool = True
    
    # Volatility
    atr_expansion_threshold: float = 1.5  # 1.5x average ATR
    
    # Custom
    custom_filters: Dict = field(default_factory=dict)


@dataclass
class ScanResult:
    """Result from market scan."""
    symbol: str
    signals: List[str]
    scores: Dict[str, float]
    indicators: Dict[str, float]
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class MLXRealtimeScanner:
    """High-performance market scanner using MLX."""
    
    def __init__(self, 
                 symbols: Optional[List[str]] = None,
                 update_interval: int = 60,
                 batch_size: int = 100):
        """Initialize scanner."""
        self.symbols = symbols or []
        self.update_interval = update_interval
        self.batch_size = batch_size
        
        # MLX components
        self.indicator_engine = MLXIndicatorEngine()
        self.model = None  # Initialize if using ML model
        
        # Scanning state
        self.last_scan_time = {}
        self.scan_results = {}
        self.opportunity_queue = []
        
        # Performance tracking
        self.scan_times = []
        self.symbols_per_second = []
        
        # Data cache
        self.data_cache = {}
        self.cache_expiry = timedelta(minutes=5)
        
    def add_symbols(self, symbols: List[str]):
        """Add symbols to scan list."""
        self.symbols.extend(symbols)
        self.symbols = list(set(self.symbols))  # Remove duplicates
        logger.info(f"Scanner tracking {len(self.symbols)} symbols")
    
    def remove_symbols(self, symbols: List[str]):
        """Remove symbols from scan list."""
        for symbol in symbols:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
    
    async def scan_market_async(self, criteria: ScanCriteria) -> List[ScanResult]:
        """Asynchronously scan entire market."""
        start_time = time.time()
        logger.info(f"🔍 Starting market scan of {len(self.symbols)} symbols...")
        
        # Fetch data for all symbols
        all_data = await self._fetch_all_data_async()
        
        # Process in batches for memory efficiency
        all_results = []
        
        for i in range(0, len(self.symbols), self.batch_size):
            batch_symbols = self.symbols[i:i + self.batch_size]
            batch_data = {s: all_data[s] for s in batch_symbols if s in all_data}
            
            if batch_data:
                results = self._scan_batch(batch_data, criteria)
                all_results.extend(results)
        
        # Sort by opportunity score
        all_results.sort(key=lambda x: sum(x.scores.values()), reverse=True)
        
        # Performance metrics
        elapsed = time.time() - start_time
        self.scan_times.append(elapsed)
        self.symbols_per_second.append(len(self.symbols) / elapsed)
        
        logger.info(f"✅ Scan complete: {len(all_results)} opportunities in {elapsed:.2f}s")
        logger.info(f"📊 Throughput: {len(self.symbols) / elapsed:.1f} symbols/second")
        
        # Cache results
        self.scan_results = {r.symbol: r for r in all_results}
        self.opportunity_queue = all_results[:50]  # Top 50 opportunities
        
        return all_results
    
    def scan_market_sync(self, criteria: ScanCriteria) -> List[ScanResult]:
        """Synchronous version of market scan."""
        return asyncio.run(self.scan_market_async(criteria))
    
    def _scan_batch(self, batch_data: Dict[str, pd.DataFrame], criteria: ScanCriteria) -> List[ScanResult]:
        """Scan a batch of symbols using MLX vectorized operations."""
        # Convert data to MLX arrays
        symbols = list(batch_data.keys())
        data_arrays = []
        valid_symbols = []
        
        for symbol in symbols:
            df = batch_data[symbol]
            if len(df) < 100:  # Need minimum data
                continue
                
            # Extract OHLCV
            ohlcv = np.array([
                df['open'].values[-200:],
                df['high'].values[-200:],
                df['low'].values[-200:],
                df['close'].values[-200:],
                df['volume'].values[-200:]
            ]).T
            
            data_arrays.append(ohlcv)
            valid_symbols.append(symbol)
        
        if not data_arrays:
            return []
        
        # Stack into single array [num_symbols, time_steps, features]
        market_data = mx.array(data_arrays)
        
        # Compute all indicators in parallel
        indicators = self.indicator_engine.compute_all(market_data, valid_symbols)
        
        # Scan for opportunities
        results = []
        
        for i, symbol in enumerate(valid_symbols):
            signals = []
            scores = {}
            symbol_indicators = {}
            
            # Extract latest indicator values
            for name, values in indicators.items():
                if values.shape[0] > i:
                    latest_value = float(values[i, -1])
                    symbol_indicators[name] = latest_value
            
            # Check RSI conditions
            if 'RSI_14' in symbol_indicators:
                rsi = symbol_indicators['RSI_14']
                if rsi < criteria.rsi_oversold:
                    signals.append('RSI_OVERSOLD')
                    scores['rsi_oversold'] = (criteria.rsi_oversold - rsi) / criteria.rsi_oversold
                elif rsi > criteria.rsi_overbought:
                    signals.append('RSI_OVERBOUGHT')
                    scores['rsi_overbought'] = (rsi - criteria.rsi_overbought) / (100 - criteria.rsi_overbought)
            
            # Check MACD crossover
            if all(k in symbol_indicators for k in ['MACD', 'MACD_signal']):
                macd = symbol_indicators['MACD']
                signal = symbol_indicators['MACD_signal']
                
                # Get previous values for crossover detection
                if 'MACD' in indicators and indicators['MACD'].shape[1] > 1:
                    prev_macd = float(indicators['MACD'][i, -2])
                    prev_signal = float(indicators['MACD_signal'][i, -2])
                    
                    # Bullish crossover
                    if criteria.macd_bullish_cross and prev_macd < prev_signal and macd > signal:
                        signals.append('MACD_BULLISH_CROSS')
                        scores['macd_bullish'] = abs(macd - signal)
                    
                    # Bearish crossover
                    elif criteria.macd_bearish_cross and prev_macd > prev_signal and macd < signal:
                        signals.append('MACD_BEARISH_CROSS')
                        scores['macd_bearish'] = abs(macd - signal)
            
            # Check volume spike
            if 'OBV' in indicators and indicators['OBV'].shape[1] > 20:
                recent_volume = mx.mean(market_data[i, -5:, 4])
                avg_volume = mx.mean(market_data[i, -20:, 4])
                volume_ratio = float(recent_volume / (avg_volume + 1e-10))
                
                if volume_ratio > criteria.volume_spike_threshold:
                    signals.append('VOLUME_SPIKE')
                    scores['volume'] = volume_ratio / criteria.volume_spike_threshold
            
            # Check ATR expansion
            if 'ATR' in symbol_indicators and 'ATR' in indicators:
                current_atr = symbol_indicators['ATR']
                if indicators['ATR'].shape[1] > 20:
                    avg_atr = float(mx.mean(indicators['ATR'][i, -20:]))
                    atr_ratio = current_atr / (avg_atr + 1e-10)
                    
                    if atr_ratio > criteria.atr_expansion_threshold:
                        signals.append('VOLATILITY_EXPANSION')
                        scores['volatility'] = atr_ratio / criteria.atr_expansion_threshold
            
            # Check Bollinger Band squeeze/expansion
            if all(k in symbol_indicators for k in ['BB_upper', 'BB_lower']):
                bb_width = symbol_indicators['BB_upper'] - symbol_indicators['BB_lower']
                price = float(market_data[i, -1, 3])  # Latest close
                
                # Price near bands
                if price > symbol_indicators['BB_upper'] * 0.98:
                    signals.append('NEAR_BB_UPPER')
                    scores['bb_upper'] = (price - symbol_indicators['BB_upper']) / symbol_indicators['BB_upper']
                elif price < symbol_indicators['BB_lower'] * 1.02:
                    signals.append('NEAR_BB_LOWER')
                    scores['bb_lower'] = (symbol_indicators['BB_lower'] - price) / symbol_indicators['BB_lower']
            
            # Create result if any signals found
            if signals:
                result = ScanResult(
                    symbol=symbol,
                    signals=signals,
                    scores=scores,
                    indicators=symbol_indicators,
                    timestamp=datetime.now(),
                    metadata={
                        'data_points': market_data.shape[1],
                        'latest_price': float(market_data[i, -1, 3]),
                        'price_change_pct': float((market_data[i, -1, 3] - market_data[i, -2, 3]) / market_data[i, -2, 3] * 100)
                    }
                )
                results.append(result)
        
        return results
    
    async def _fetch_all_data_async(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols asynchronously."""
        # This is a mock implementation - replace with your actual data fetching
        # In production, this would connect to your data source (FMP API, database, etc.)
        
        async def fetch_symbol_data(symbol: str) -> Tuple[str, pd.DataFrame]:
            # Check cache first
            if symbol in self.data_cache:
                cache_time, data = self.data_cache[symbol]
                if datetime.now() - cache_time < self.cache_expiry:
                    return symbol, data
            
            # Mock data generation
            await asyncio.sleep(0.001)  # Simulate network delay
            
            dates = pd.date_range(end=datetime.now(), periods=500, freq='1min')
            df = pd.DataFrame({
                'open': 100 + np.random.randn(500).cumsum() * 0.1,
                'high': 100.5 + np.random.randn(500).cumsum() * 0.1,
                'low': 99.5 + np.random.randn(500).cumsum() * 0.1,
                'close': 100 + np.random.randn(500).cumsum() * 0.1,
                'volume': np.random.randint(1000, 10000, 500)
            }, index=dates)
            
            # Update cache
            self.data_cache[symbol] = (datetime.now(), df)
            
            return symbol, df
        
        # Fetch all symbols concurrently
        tasks = [fetch_symbol_data(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    def get_top_opportunities(self, n: int = 10) -> List[ScanResult]:
        """Get top N opportunities from last scan."""
        return self.opportunity_queue[:n]
    
    def get_signals_for_symbol(self, symbol: str) -> Optional[ScanResult]:
        """Get scan results for specific symbol."""
        return self.scan_results.get(symbol)
    
    def run_continuous_scan(self, criteria: ScanCriteria, callback=None):
        """Run continuous market scanning."""
        logger.info("🚀 Starting continuous market scanner...")
        
        while True:
            try:
                # Run scan
                results = self.scan_market_sync(criteria)
                
                # Callback for processing results
                if callback:
                    callback(results)
                
                # Show top opportunities
                top_opps = self.get_top_opportunities(5)
                if top_opps:
                    logger.info("\n📊 Top Opportunities:")
                    for i, opp in enumerate(top_opps, 1):
                        total_score = sum(opp.scores.values())
                        logger.info(f"{i}. {opp.symbol}: {', '.join(opp.signals)} (score: {total_score:.2f})")
                
                # Wait for next scan
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping scanner...")
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")
                time.sleep(5)
    
    def export_scan_results(self, filepath: str):
        """Export scan results to JSON."""
        results_data = []
        
        for result in self.scan_results.values():
            results_data.append({
                'symbol': result.symbol,
                'signals': result.signals,
                'scores': result.scores,
                'indicators': result.indicators,
                'timestamp': result.timestamp.isoformat(),
                'metadata': result.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Exported {len(results_data)} scan results to {filepath}")


class MLXAlertSystem:
    """Alert system for scan results."""
    
    def __init__(self):
        self.alert_history = []
        self.alert_callbacks = []
    
    def add_callback(self, callback):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def process_scan_results(self, results: List[ScanResult]):
        """Process scan results and generate alerts."""
        high_priority_alerts = []
        
        for result in results:
            # High priority: Multiple signals with high scores
            if len(result.signals) >= 3 and sum(result.scores.values()) > 2.0:
                alert = {
                    'type': 'HIGH_PRIORITY',
                    'symbol': result.symbol,
                    'signals': result.signals,
                    'score': sum(result.scores.values()),
                    'timestamp': datetime.now()
                }
                high_priority_alerts.append(alert)
            
            # RSI extremes
            elif 'RSI_OVERSOLD' in result.signals and result.indicators.get('RSI_14', 50) < 20:
                alert = {
                    'type': 'EXTREME_RSI',
                    'symbol': result.symbol,
                    'rsi': result.indicators['RSI_14'],
                    'timestamp': datetime.now()
                }
                high_priority_alerts.append(alert)
        
        # Send alerts
        for alert in high_priority_alerts:
            self.alert_history.append(alert)
            for callback in self.alert_callbacks:
                callback(alert)
        
        return high_priority_alerts


def demo_realtime_scanner():
    """Demonstrate real-time market scanning."""
    logger.info("🔍 MLX Real-Time Market Scanner Demo")
    logger.info("="*60)
    
    # Initialize scanner with S&P 500 symbols (mock)
    symbols = [f"STOCK_{i}" for i in range(500)]  # Mock 500 symbols
    scanner = MLXRealtimeScanner(symbols=symbols, update_interval=30)
    
    # Define scan criteria
    criteria = ScanCriteria(
        rsi_oversold=30,
        rsi_overbought=70,
        volume_spike_threshold=2.0,
        macd_bullish_cross=True,
        atr_expansion_threshold=1.5
    )
    
    # Initialize alert system
    alert_system = MLXAlertSystem()
    
    def alert_callback(alert):
        logger.info(f"\n🚨 ALERT: {alert['type']} for {alert.get('symbol', 'N/A')}")
    
    alert_system.add_callback(alert_callback)
    
    # Run single scan
    logger.info("\nRunning market scan...")
    results = scanner.scan_market_sync(criteria)
    
    # Process alerts
    alerts = alert_system.process_scan_results(results)
    
    # Show performance
    logger.info(f"\n📊 Scan Performance:")
    logger.info(f"  Symbols scanned: {len(symbols)}")
    logger.info(f"  Opportunities found: {len(results)}")
    logger.info(f"  High priority alerts: {len(alerts)}")
    if scanner.scan_times:
        logger.info(f"  Scan time: {scanner.scan_times[-1]:.2f}s")
        logger.info(f"  Throughput: {scanner.symbols_per_second[-1]:.1f} symbols/sec")
    
    # Export results
    scanner.export_scan_results("scan_results.json")
    
    # Demonstrate continuous scanning
    logger.info("\n💡 Continuous scanning available with:")
    logger.info("  scanner.run_continuous_scan(criteria, callback=alert_system.process_scan_results)")


if __name__ == "__main__":
    demo_realtime_scanner()