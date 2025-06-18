cd /Users/jkim/Desktop/code/trading/s3_daytrading

# Trade
python s3_ai_fmp_ib_hybrid_trading.py

  1. FMP for all market data - Consistent data source for analysis and
   signals
  2. IB for execution only - Real fills, real positions, real P&L
  tracking
  3. Best of both worlds - FMP's data quality with IB's execution
  realism
  4. Production-ready - Can easily switch to live trading

# Train
Train your full model with MLX: 
```
python train_mlx.py
mlx_config.json
```
MLX_QUICK_REFERENCE.md

Examples:
  # Your typical training run
  python train_mlx.py

  # Quick 5-epoch test
python train_mlx.py --epochs 5

  # Train on all tickers
  python train_mlx.py --tickers-file ../tickers/all_tickers.json

  # See what settings will be used
  python train_mlx.py --help


# Backtest
s3_ai_mlx_backtest.py

ai_parallel_backtest_optimized.py              # Run all 664 tickers
