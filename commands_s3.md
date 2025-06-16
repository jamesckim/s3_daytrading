cd /Users/jkim/Desktop/code/trading/s3_daytrading

  python s3_ai_fmp_ib_hybrid.py

  1. FMP for all market data - Consistent data source for analysis and
   signals
  2. IB for execution only - Real fills, real positions, real P&L
  tracking
  3. Best of both worlds - FMP's data quality with IB's execution
  realism
  4. Production-ready - Can easily switch to live trading

uses s3_ai_fmp_trading_config_top_performers.json

  # Monitor performance  
    python s3_ai_fmp_hybrid_trading.py --demo    # Run demo
  python s3_ai_fmp_hybrid_trading.py --test    # Run tests  
  python s3_ai_fmp_hybrid_trading.py --status  # Show status only
  

  # Generate reports
  python run_s3_ai_trading.py --report

python s3_ai_fixed.py -- uses synthetic data

python s3_ai_simple_real_data.py

[ ] add market cap data
from launch_items
  python add_market_cap_table.py
  python collect_market_cap_data.py

    # Run test suite
  python quick_s3_test.py


  # backfill treasury data
[ ] write treasury_rates backfill script and add to scheduler.py
