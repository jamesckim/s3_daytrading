#!/usr/bin/env python3
"""
S3 AI Simple Real Data Strategy
===============================
Simplified version using direct PostgreSQL connection without the complex DataLoader.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import warnings
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Import AI components
from s3_ai_wrapper import S3AIWrapper


class S3AISimpleRealDataStrategy:
    """
    Simplified S3 AI strategy using direct PostgreSQL connection.
    """
    
    def __init__(self, 
                 symbol: str,
                 capital: float = 100000,
                 adaptive_indicators: bool = True,
                 top_k_indicators: int = 8,
                 selection_interval: int = 30,
                 risk_per_trade: float = 0.02):
        
        self.symbol = symbol
        self.capital = capital
        self.adaptive_indicators = adaptive_indicators
        self.top_k_indicators = top_k_indicators
        self.selection_interval = selection_interval
        self.risk_per_trade = risk_per_trade
        
        # Direct database connection
        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI environment variable not found")
        
        self.engine = create_engine(postgres_uri, pool_pre_ping=True)
        
        # AI components
        self.indicator_selector = None
        self.current_indicators = {}
        self.bars_since_selection = 0
        self.last_market_regime = "Unknown"
        self.regime_confidence = 0.0
        self.ai_selection_history = []
        
        if adaptive_indicators:
            self._init_ai_components()
        
        # Performance tracking
        self.trades = []
        self.total_pnl = 0
        
        print(f"🤖 S3 AI Simple Real Data Strategy initialized for {symbol}")
        print(f"   AI Status: {'✅ Active' if self.indicator_selector else '❌ Disabled'}")
        print(f"   Data Source: 📊 PostgreSQL Direct Connection")
    
    def _init_ai_components(self):
        """Initialize AI components safely."""
        try:
            print(f"🧠 Initializing AI components...")
            
            self.indicator_selector = S3AIWrapper(
                confidence_threshold=0.1,
                min_indicators=3,
                max_indicators=8
            )
            
            print(f"✅ AI Indicator Selector ready")
            
        except Exception as e:
            print(f"❌ AI initialization failed: {e}")
            raise e
    
    def get_simple_real_data(self, 
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
        """Get real market data using direct SQL query."""
        
        print(f"📊 Fetching real market data for {self.symbol}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Source: Direct PostgreSQL query")
        
        try:
            # Direct SQL query with timeout
            query = """
            SELECT mp.datetime AT TIME ZONE 'UTC' AS ts,
                   mp.open, mp.high, mp.low, mp.close, mp.volume
              FROM minute_prices mp
              JOIN tickers t ON mp.ticker_id = t.id
             WHERE t.symbol = %(symbol)s
               AND mp.datetime >= %(start_date)s
               AND mp.datetime <= %(end_date)s
             ORDER BY ts
             LIMIT 150000
            """
            
            print(f"   Executing SQL query...")
            
            df = pd.read_sql(
                query,
                self.engine,
                params={
                    "symbol": self.symbol,
                    "start_date": start_date,
                    "end_date": end_date
                },
                parse_dates=["ts"]
            )
            
            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Set index and timezone
            df = df.set_index("ts")
            if df.index.tz is None:
                df = df.tz_localize("UTC")
            df = df.tz_convert("US/Eastern")
            
            # Filter trading hours
            df = df.between_time('09:30', '16:00')
            df = df.dropna()
            
            total_bars = len(df)
            print(f"✅ Real market data loaded: {total_bars} bars")
            
            return df
            
        except Exception as e:
            print(f"❌ Failed to load real market data: {e}")
            raise e
    
    def prepare_ai_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare basic features for AI analysis."""
        
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['returns'] = data['close'].pct_change()
        features['volume'] = data['volume']
        features['volume_ma'] = data['volume'].rolling(20, min_periods=1).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma']
        
        # VWAP
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap_num = (typical_price * data['volume']).rolling(20, min_periods=1).sum()
        vwap_den = data['volume'].rolling(20, min_periods=1).sum()
        features['vwap_20'] = vwap_num / vwap_den
        features['vwap_dev_20'] = (data['close'] - features['vwap_20']) / features['vwap_20']
        
        # AI indicator selection
        if self.adaptive_indicators and self.indicator_selector is not None:
            if (self.bars_since_selection >= self.selection_interval or 
                len(self.current_indicators) == 0):
                self._ai_select_indicators(data, features)
                self.bars_since_selection = 0
            else:
                self.bars_since_selection += 1
            
            self._add_ai_selected_features(data, features)
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _ai_select_indicators(self, data: pd.DataFrame, features: pd.DataFrame):
        """Use AI to select optimal indicators."""
        
        try:
            print(f"🧠 AI selecting indicators...")
            
            sample_size = min(200, len(data))
            sample_data = data.tail(sample_size).copy()
            
            # Get AI selection
            selection_result = self.indicator_selector.select_indicators(
                sample_data, 
                top_k=self.top_k_indicators
            )
            
            # Extract regime
            regime_info = selection_result.get('regime_detection', {})
            if isinstance(regime_info, str):
                self.last_market_regime = regime_info
                self.regime_confidence = 0.0
            elif isinstance(regime_info, dict):
                self.last_market_regime = regime_info.get('primary_regime', 'Unknown')
                self.regime_confidence = regime_info.get('confidence', 0.0)
            
            # Map indicators
            selected_indicators = selection_result.get('selected_indicators', {})
            self.current_indicators = {}
            
            for ai_indicator, info in selected_indicators.items():
                if 'rsi' in ai_indicator.lower():
                    self.current_indicators['rsi_momentum'] = True
                elif 'sma' in ai_indicator.lower() or 'ema' in ai_indicator.lower():
                    self.current_indicators['sma_trend'] = True
                elif 'volume' in ai_indicator.lower():
                    self.current_indicators['volume_breakout'] = True
            
            # Ensure at least some indicators
            if len(self.current_indicators) == 0:
                self.current_indicators = {
                    'sma_trend': True,
                    'rsi_momentum': True,
                    'volume_breakout': True
                }
            
            print(f"✅ AI Selection: {self.last_market_regime}, {len(selected_indicators)} indicators")
            
        except Exception as e:
            print(f"❌ AI selection failed: {e}")
            self.current_indicators = {
                'sma_trend': True,
                'rsi_momentum': True
            }
    
    def _add_ai_selected_features(self, data: pd.DataFrame, features: pd.DataFrame):
        """Add AI-selected indicators."""
        
        for indicator_name, selected in self.current_indicators.items():
            if not selected:
                continue
                
            try:
                if 'sma_trend' in indicator_name:
                    features['sma_20'] = data['close'].rolling(20, min_periods=1).mean()
                    features['sma_20_signal'] = (data['close'] > features['sma_20']).astype(int)
                    
                elif 'rsi_momentum' in indicator_name:
                    features['rsi'] = self._compute_rsi(data['close'])
                    features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
                    features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
                    
                elif 'volume_breakout' in indicator_name:
                    vol_ma = data['volume'].rolling(20, min_periods=1).mean()
                    features['volume_breakout'] = (data['volume'] > vol_ma * 1.5).astype(int)
                    
            except Exception as e:
                print(f"⚠️ Failed to compute {indicator_name}: {e}")
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def generate_ai_signals(self, features: pd.DataFrame) -> pd.Series:
        """Generate simple trading signals."""
        
        signals = pd.Series(0, index=features.index)
        
        for i in range(len(features)):
            if i < 20:
                continue
                
            current_features = features.iloc[i]
            
            # VWAP mean reversion
            if 'vwap_dev_20' in current_features:
                vwap_dev = current_features['vwap_dev_20']
                if vwap_dev < -0.002:  # Price below VWAP
                    signals.iloc[i] = 1
                elif vwap_dev > 0.002:  # Price above VWAP
                    signals.iloc[i] = -1
        
        return signals
    
    def backtest_simple_strategy(self, 
                                start_date: str,
                                end_date: str) -> Dict[str, Any]:
        """Run simple backtest."""
        
        # Get real data
        data = self.get_simple_real_data(start_date, end_date)
        
        if len(data) < 50:
            raise ValueError(f"Insufficient data: only {len(data)} bars")
        
        # Prepare features
        features = self.prepare_ai_features(data)
        
        # Generate signals
        signals = self.generate_ai_signals(features)
        
        # Simple backtest
        capital = self.capital
        position = 0
        trades = []
        
        active_signals = (signals != 0).sum()
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            current_signal = signals.iloc[i]
            
            # Exit position
            if position != 0:
                pnl = position * (current_price - entry_price) - 1.0  # $1 commission
                trades.append({
                    'pnl': pnl,
                    'regime': self.last_market_regime
                })
                capital += pnl
                position = 0
            
            # Enter position
            if current_signal != 0:
                position_size = min(1000, capital * 0.02 / current_price)
                position = position_size if current_signal > 0 else -position_size
                entry_price = current_price
                capital -= 1.0  # Commission
        
        # Calculate results
        if trades:
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = winning_trades / total_trades
            total_pnl = sum([t['pnl'] for t in trades])
            total_return_pct = total_pnl / self.capital
        else:
            total_trades = 0
            win_rate = 0
            total_pnl = 0
            total_return_pct = 0
        
        results = {
            'symbol': self.symbol,
            'ai_enabled': self.adaptive_indicators,
            'data_source': 'PostgreSQL Direct',
            'total_bars': len(data),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'ai_regime': self.last_market_regime,
            'signals_generated': active_signals,
            'features_count': features.shape[1]
        }
        
        print(f"\n🤖 Simple Real Data Results for {self.symbol}:")
        print(f"   Data Source: {results['data_source']}")
        print(f"   Total Bars: {results['total_bars']}")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   Total Return: {results['total_return_pct']:.1%}")
        print(f"   AI Regime: {results['ai_regime']}")
        
        return results


def test_simple_real_data_strategy():
    """Test the simplified AI strategy with real PostgreSQL data across ALL TICKERS for FULL YEAR."""
    print("🚀 ULTIMATE SCALE S3 AI Strategy Testing - ALL 664 TICKERS, FULL YEAR Real Data")
    print("=" * 85)
    
    # Load ALL tickers from the comprehensive universe
    import json
    tickers_file = '/Users/jkim/Desktop/code/trading/tickers/all_tickers.json'
    
    try:
        with open(tickers_file, 'r') as f:
            symbols = json.load(f)
        print(f"📂 Loaded {len(symbols)} tickers from all_tickers.json")
    except FileNotFoundError:
        print(f"❌ Could not find {tickers_file}")
        # Fallback to a smaller set
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        print(f"📂 Using fallback set of {len(symbols)} symbols")
    
    # Use FULL YEAR for maximum historical testing
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # FULL YEAR
    
    results_summary = []
    
    print(f"\n📈 Starting ULTIMATE test of {len(symbols)} symbols over FULL YEAR...")
    print(f"📊 Estimated data volume: ~{len(symbols) * 100000:,} bars of real market data")
    print(f"⚡ Expected: 25M+ minute bars, 5M+ trades, 16+ sectors analyzed")
    print(f"⏱️  Estimated completion time: {len(symbols) * 3:.0f} minutes")
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n🔍 [{i:3d}/{len(symbols)}] Testing PostgreSQL strategy with {symbol}...")
        print("-" * 55)
        
        try:
            ai_strategy = S3AISimpleRealDataStrategy(
                symbol=symbol,
                capital=100000,
                adaptive_indicators=True,
                top_k_indicators=8,
                selection_interval=30,
                risk_per_trade=0.02
            )
            
            results = ai_strategy.backtest_simple_strategy(
                start_date=start_date,
                end_date=end_date
            )
            
            results_summary.append({
                'symbol': symbol,
                'ai_enabled': results.get('ai_enabled', False),
                'data_source': results.get('data_source', 'Unknown'),
                'bars': results.get('total_bars', 0),
                'trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0),
                'return_pct': results.get('total_return_pct', 0),
                'regime': results.get('ai_regime', 'Unknown'),
                'features': results.get('features_count', 0),
                'signals': results.get('signals_generated', 0)
            })
            
            # Save individual results
            output_file = f's3_postgres_results_{symbol}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            results_summary.append({
                'symbol': symbol,
                'ai_enabled': False,
                'data_source': 'Failed',
                'bars': 0,
                'trades': 0,
                'win_rate': 0,
                'return_pct': 0,
                'regime': 'Error',
                'features': 0,
                'signals': 0
            })
    
    # Print comprehensive summary
    print(f"\n🤖 PostgreSQL Real Data AI Strategy Summary:")
    print("=" * 95)
    print(f"{'Symbol':<8} {'Source':<12} {'Bars':<6} {'Trades':<7} {'Win%':<6} {'Return%':<8} {'Signals':<8} {'Regime':<15}")
    print("-" * 95)
    
    for result in results_summary:
        print(f"{result['symbol']:<8} {result['data_source']:<12} {result['bars']:<6} "
              f"{result['trades']:<7} {result['win_rate']:<6.1%} {result['return_pct']:<8.1%} "
              f"{result['signals']:<8} {result['regime']:<15}")
    
    # Calculate aggregate statistics
    successful_results = [r for r in results_summary if r['data_source'] != 'Failed']
    failed_results = [r for r in results_summary if r['data_source'] == 'Failed']
    
    if successful_results:
        total_bars = sum(r['bars'] for r in successful_results)
        total_trades = sum(r['trades'] for r in successful_results)
        total_signals = sum(r['signals'] for r in successful_results)
        avg_return = sum(r['return_pct'] for r in successful_results) / len(successful_results)
        avg_win_rate = sum(r['win_rate'] for r in successful_results) / len(successful_results)
        
        # Performance tiers
        high_performers = [r for r in successful_results if r['win_rate'] > 0.25]
        medium_performers = [r for r in successful_results if 0.15 <= r['win_rate'] <= 0.25]
        low_performers = [r for r in successful_results if r['win_rate'] < 0.15]
        
        print(f"\n📊 Aggregate PostgreSQL Real Data Performance:")
        print(f"  Symbols Tested: {len(successful_results)}/{len(symbols)}")
        print(f"  Failed Symbols: {len(failed_results)}")
        print(f"  Total Bars: {total_bars:,}")
        print(f"  Total Trades: {total_trades:,}")
        print(f"  Total Signals: {total_signals:,}")
        print(f"  Average Return: {avg_return:.2%}")
        print(f"  Average Win Rate: {avg_win_rate:.2%}")
        
        print(f"\n🎯 Performance Tiers:")
        print(f"  High Performers (>25% win rate): {len(high_performers)} symbols")
        if high_performers:
            high_symbols = [r['symbol'] for r in high_performers]
            print(f"    Symbols: {', '.join(high_symbols)}")
        
        print(f"  Medium Performers (15-25% win rate): {len(medium_performers)} symbols")
        if medium_performers:
            medium_symbols = [r['symbol'] for r in medium_performers]
            print(f"    Symbols: {', '.join(medium_symbols)}")
        
        print(f"  Low Performers (<15% win rate): {len(low_performers)} symbols")
        if low_performers:
            low_symbols = [r['symbol'] for r in low_performers]
            print(f"    Symbols: {', '.join(low_symbols)}")
        
        # Regime distribution
        regimes = [r['regime'] for r in successful_results]
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        print(f"\n🧠 AI Regime Detection Summary:")
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} symbols ({count/len(successful_results):.1%})")
        
        # Comprehensive sector analysis
        sector_map = {
            'Mega Tech': ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA'],
            'Large Tech': ['NFLX', 'CRM', 'ORCL', 'ADBE', 'INTC', 'AMD', 'QCOM', 'CSCO', 'IBM', 'AVGO',
                          'NOW', 'INTU', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'PANW', 'CRWD',
                          'ZM', 'DOCU', 'OKTA', 'SNOW', 'PLTR', 'SHOP', 'SQ', 'PYPL', 'COIN'],
            'Auto/EV': ['F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FORD'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
                       'AXP', 'BLK', 'SCHW', 'SPGI', 'ICE', 'CME', 'MCO', 'V', 'MA', 'FIS'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'MRNA', 'BNTX', 'GILD', 'BIIB', 'REGN',
                          'TMO', 'DHR', 'ABT', 'BMY', 'LLY', 'MDT', 'ISRG', 'VRTX', 'ILMN', 'AMGN'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'KMI', 'OKE', 'WMB', 'MPC', 'VLO',
                      'PSX', 'HAL', 'BKR', 'DVN', 'FANG', 'MRO', 'APA', 'OXY'],
            'Consumer Disc': ['WMT', 'HD', 'COST', 'TGT', 'LOW', 'SBUX', 'MCD', 'NKE', 'LULU', 'TJX',
                             'BKNG', 'ABNB', 'UBER', 'LYFT', 'DIS', 'CMCSA', 'PARA', 'WBD'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'CL', 'KMB', 'GIS', 'K', 'CPB',
                                'CAG', 'TSN', 'HRL', 'SJM', 'MKC', 'CLX', 'CHD'],
            'Airlines/Travel': ['DAL', 'UAL', 'LUV', 'JBLU', 'ALK', 'SAVE', 'HA', 'SKYW',
                               'MAR', 'HLT', 'H', 'IHG', 'WYNN', 'LVS', 'MGM', 'CZR'],
            'Communication': ['VZ', 'T', 'TMUS', 'CHTR', 'DISH', 'SIRI', 'TWTR', 'SNAP', 'PINS', 'MTCH'],
            'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
                          'GD', 'DE', 'EMR', 'ETN', 'PH', 'ITW', 'ROK', 'DOV', 'XYL', 'FTV'],
            'REITs': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'UDR', 'ESS',
                     'MAA', 'CPT', 'AIV', 'BXP', 'VTR', 'WELL', 'PEAK', 'O', 'STAG'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'PCG', 'ED',
                         'ES', 'PPL', 'FE', 'ETR', 'CNP', 'NI', 'LNT', 'CMS', 'DTE', 'ATO'],
            'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'GOLD', 'AA', 'X', 'CLF',
                         'NUE', 'STLD', 'RS', 'VMC', 'MLM', 'DOW', 'DD', 'LYB', 'EMN', 'FMC'],
            'Semiconductors': ['TSM', 'ASML', 'MCHP', 'SWKS', 'QRVO', 'MPWR', 'ENPH', 'SEDG'],
            'ETFs/Funds': ['ARKK', 'ARKQ', 'ARKG', 'ARKW', 'ARKF', 'SOXL', 'TQQQ', 'SPXL', 'UPRO',
                          'QQQ', 'SPY', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO', 'BND', 'TLT', 'GLD']
        }
        
        print(f"\n🏭 Sector Performance Analysis:")
        for sector, sector_symbols in sector_map.items():
            sector_results = [r for r in successful_results if r['symbol'] in sector_symbols]
            if sector_results:
                sector_avg_win = sum(r['win_rate'] for r in sector_results) / len(sector_results)
                sector_avg_return = sum(r['return_pct'] for r in sector_results) / len(sector_results)
                sector_total_trades = sum(r['trades'] for r in sector_results)
                print(f"  {sector:<12}: {len(sector_results):2d} symbols, {sector_avg_win:5.1%} avg win rate, "
                      f"{sector_avg_return:6.1%} avg return, {sector_total_trades:4d} total trades")
        
        if failed_results:
            failed_symbols = [r['symbol'] for r in failed_results]
            print(f"\n❌ Failed Symbols ({len(failed_results)}):")
            print(f"  {', '.join(failed_symbols)}")
    
    print(f"\n✅ Multi-stock PostgreSQL real data testing complete!")
    return results_summary


if __name__ == "__main__":
    test_simple_real_data_strategy()