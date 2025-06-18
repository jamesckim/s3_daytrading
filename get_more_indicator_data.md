# How to Get More Indicator Performance Data

## 1. **Run the Optimized Backtest with Indicator Tracking**

First, ensure the backtest saves indicator data with each trade. The current backtest needs modification to save the `info` dictionary that contains indicators.

### Quick Solution: Run a targeted backtest on high-volume symbols

```bash
# Run backtest on the most liquid symbols that are likely to generate trades
python s3_ai_parallel_backtest_optimized.py --limit 50

# Or run on specific known active symbols
python run_backtest_with_indicators.py --symbols AAPL MSFT GOOGL AMZN TSLA SPY QQQ NVDA META
```

## 2. **Modify the Backtest to Save Full Indicator Data**

The issue is that the current `BacktestTrade` dataclass doesn't include the full indicator data. Here's what needs to be modified:

### In `s3_ai_db_backtest_all_tickers.py`:

1. Update the `BacktestTrade` dataclass to include all indicator values:
```python
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
    indicators: Dict[str, float]  # This already exists
    indicator_weights: Dict[str, float] = None  # Add this
    selected_indicators: List[str] = None  # Add this
    # ... rest of fields
```

2. In `_open_position` method, ensure indicators are captured:
```python
def _open_position(self, symbol, data, timestamp, capital, signal_strength, info):
    # ... existing code ...
    
    # Make sure to store the indicators from info
    position['indicators'] = info.get('indicators', {})
    position['selected_indicators'] = info.get('selected_indicators', [])
    position['indicator_weights'] = info.get('weights', {})
```

## 3. **Run Historical Analysis Scripts**

Use existing analysis scripts that might have more comprehensive data:

```bash
# Look for existing comprehensive analysis
python backtest_with_vwap_bands.py
python compare_vwap_performance.py
python s3_ai_analysis.py  # If it exists
```

## 4. **Create a Dedicated Indicator Analysis Backtest**

Create a new script that specifically focuses on capturing indicator performance:

```bash
# This script would run backtests with enhanced logging
python s3_ai_indicator_analysis_backtest.py
```

## 5. **Use Existing Database Data**

Query the PostgreSQL database directly for historical trades if they're stored:

```python
import pandas as pd
from sqlalchemy import create_engine

# Connect to database
engine = create_engine(os.getenv("DATABASE_URL"))

# Query for trades (if table exists)
query = """
SELECT * FROM backtest_trades 
WHERE created_at >= NOW() - INTERVAL '30 days'
"""
trades_df = pd.read_sql(query, engine)
```

## 6. **Run Shorter Timeframe Backtests for More Trades**

More trades = more data points:

```bash
# Use shorter timeframes to generate more trades
python s3_ai_parallel_backtest_optimized.py --days 30  # Instead of 60

# Run on high-volatility periods
python s3_ai_parallel_backtest_optimized.py --start-date 2024-10-01 --end-date 2024-12-31
```

## 7. **Enable Debug Mode for Full Indicator Logging**

Some scripts may have debug modes that log all indicator values:

```bash
# Set environment variable
export S3_AI_DEBUG=1
export LOG_INDICATORS=1

python s3_ai_parallel_backtest_optimized.py
```

## 8. **Use the Paper Trading Results**

Paper trading results often have more detailed data:

```bash
# Check paper trading logs
cat logs/s3_ai_paper_trading_*.log | grep -i indicator

# Or analyze paper trading results
python analyze_paper_trading_results.py
```

## Next Steps:

1. **Immediate**: Run the optimized backtest on known active symbols
2. **Short-term**: Modify the backtest to save complete indicator data
3. **Analysis**: Create specialized scripts to analyze indicator performance across different market conditions

The key is ensuring that every trade saves:
- All indicators considered
- The specific indicators that triggered the trade
- The weights/importance of each indicator
- The market regime during the trade
- The outcome (profit/loss)