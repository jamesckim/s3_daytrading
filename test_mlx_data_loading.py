#!/usr/bin/env python3
"""
Test MLX Data Loading
=====================
Diagnose data loading issues for MLX training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import logging
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider
from indicator_transformer import IndicatorLibrary, MarketRegimeAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database_connection():
    """Test database connection."""
    logger.info("Testing database connection...")
    try:
        provider = DatabaseDataProvider()
        logger.info("✅ Database connection successful")
        return provider
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return None


def test_data_availability(provider, symbols):
    """Test data availability for symbols."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"\nTesting data availability for {symbols}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    for symbol in symbols:
        try:
            data = provider.get_minute_data(symbol, start_date, end_date)
            if not data.empty:
                logger.info(f"✅ {symbol}: {len(data)} data points")
            else:
                logger.warning(f"⚠️  {symbol}: No data available")
        except Exception as e:
            logger.error(f"❌ {symbol}: Error - {e}")


def test_indicator_computation():
    """Test indicator computation."""
    logger.info("\nTesting indicator computation...")
    
    # Create sample data
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(end=datetime.now(), periods=200, freq='1min')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(200).cumsum() * 0.1,
        'high': 100.5 + np.random.randn(200).cumsum() * 0.1,
        'low': 99.5 + np.random.randn(200).cumsum() * 0.1,
        'close': 100 + np.random.randn(200).cumsum() * 0.1,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    library = IndicatorLibrary()
    analyzer = MarketRegimeAnalyzer()
    
    try:
        # Compute a few indicators
        rsi = library.compute_indicator(data, 'RSI_14')
        logger.info(f"✅ RSI computed: last value = {rsi.iloc[-1]:.2f}")
        
        sma = library.compute_indicator(data, 'SMA_20')
        logger.info(f"✅ SMA computed: last value = {sma.iloc[-1]:.2f}")
        
        # Market context
        context = analyzer.analyze_market_context(data)
        logger.info(f"✅ Market context computed: {len(context)} features")
        
    except Exception as e:
        logger.error(f"❌ Indicator computation failed: {e}")


def test_minimal_training():
    """Test minimal MLX training setup."""
    logger.info("\nTesting minimal MLX training...")
    
    try:
        import mlx.core as mx
        import mlx.nn as nn
        
        # Create tiny model
        model = nn.Sequential(
            nn.Linear(29, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Test forward pass
        x = mx.random.normal([4, 29])
        y = model(x)
        
        logger.info(f"✅ MLX model forward pass successful: output shape = {y.shape}")
        
    except Exception as e:
        logger.error(f"❌ MLX test failed: {e}")


def main():
    """Run diagnostics."""
    logger.info("🔍 MLX Training Diagnostics")
    logger.info("="*50)
    
    # Test database
    provider = test_database_connection()
    
    if provider:
        # Test data for the symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        test_data_availability(provider, symbols)
    
    # Test indicators
    test_indicator_computation()
    
    # Test MLX
    test_minimal_training()
    
    logger.info("\n" + "="*50)
    logger.info("Diagnostics complete!")
    
    if provider:
        logger.info("\n💡 If data loading is slow:")
        logger.info("   1. Check if minute data exists in the database")
        logger.info("   2. Consider using daily data for testing")
        logger.info("   3. Reduce the date range (use --days 7)")
        logger.info("   4. Check database performance")
        
        # Quick check for data volume
        try:
            from sqlalchemy import text
            with provider.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT COUNT(*) as count FROM minute_data WHERE symbol IN ('AAPL', 'MSFT', 'GOOGL')"
                )).fetchone()
                logger.info(f"\n📊 Total minute records for test symbols: {result[0]:,}")
        except:
            pass


if __name__ == "__main__":
    main()