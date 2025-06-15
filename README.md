# S3 Day Trading Strategy

This project contains the S3 trading strategy implementation extracted from the day_trading_o3 project. The S3 strategy is an advanced mean reversion trading system that uses machine learning and AI-enhanced signals.

## Overview

The S3 strategy family includes:
- **S3 AI**: AI-enhanced trading with transformer models
- **S3 Database**: PostgreSQL-backed data management
- **S3 Minimal**: Lightweight implementation
- **S3 Optimized**: Performance-optimized version
- **S3 Working**: Production-ready implementations

## Key Components

### Core Strategies
- `s3_ai_enabled_strategy.py` - Main AI-enhanced strategy
- `s3_optimized_strategy.py` - Optimized performance version
- `s3_minimal_strategy.py` - Minimal implementation
- `s3_database_final.py` - Database integration
- `s3_transformer_enhanced_strategy.py` - Transformer model integration

### Trading Systems
- `s3_ai_hybrid_paper_trading.py` - Hybrid paper trading
- `s3_ai_ib_paper_trading.py` - Interactive Brokers integration
- `s3_ai_fmp_hybrid_trading.py` - FMP data integration

### Analysis Tools
- `analyze_s3_portfolio.py` - Portfolio analysis
- `test_s3_quality_stocks.py` - Quality stock screening
- `test_s3_high_volume.py` - High volume stock analysis

## Setup

1. Install dependencies:
```bash
uv pip install -e .
```

2. Set up environment variables in `.env`:
```
FMP_API_KEY=your_api_key_here
DATABASE_URL=postgresql://user:pass@localhost/dbname
```

3. Run a strategy:
```bash
python s3_ai_demo.py
```

## Configuration

Trading configurations are stored in JSON files:
- `s3_ai_trading_config.json` - Main trading configuration
- `s3_ai_ib_trading_config.json` - IB-specific settings
- `s3_ai_fmp_trading_config.json` - FMP data settings

## Documentation

- [S3 AI Setup Guide](S3_AI_IB_SETUP_GUIDE.md)
- [Database Strategy Summary](S3_DATABASE_STRATEGY_SUMMARY.md)
- [High Volume Results](S3_HIGH_VOLUME_RESULTS.md)
- [Multi-Ticker Results](S3_MULTI_TICKER_RESULTS.md)
- [Quality Stocks Analysis](S3_QUALITY_STOCKS_ANALYSIS.md)
- [Win Rate Optimization](S3_WIN_RATE_OPTIMIZATION.md)