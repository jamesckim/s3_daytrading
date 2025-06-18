#!/usr/bin/env python3
"""
Check Database Data
==================
Check what data is available in the database.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def main():
    # Connect to database
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "stocks_net")
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASSWORD", "")
        db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    engine = create_engine(db_url)
    
    # Check data availability
    query = """
    SELECT 
        t.symbol,
        COUNT(*) as data_points,
        MIN(mp.datetime) as earliest_date,
        MAX(mp.datetime) as latest_date,
        MAX(mp.datetime) - MIN(mp.datetime) as date_range
    FROM minute_prices mp
    JOIN tickers t ON mp.ticker_id = t.id
    WHERE t.symbol IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA')
    GROUP BY t.symbol
    ORDER BY t.symbol
    """
    
    print("Checking data availability for key stocks...")
    df = pd.read_sql(query, engine)
    print(df)
    
    # Check recent data for AAPL
    print("\nChecking recent AAPL data...")
    recent_query = """
    SELECT 
        DATE(mp.datetime) as date,
        COUNT(*) as minute_bars,
        MIN(mp.datetime) as first_bar,
        MAX(mp.datetime) as last_bar
    FROM minute_prices mp
    JOIN tickers t ON mp.ticker_id = t.id
    WHERE t.symbol = 'AAPL'
      AND mp.datetime >= NOW() - INTERVAL '7 days'
    GROUP BY DATE(mp.datetime)
    ORDER BY date DESC
    """
    
    recent_df = pd.read_sql(recent_query, engine)
    print(recent_df)
    
    # Check if we have enough historical data
    print("\nChecking historical data coverage...")
    hist_query = """
    SELECT 
        COUNT(DISTINCT DATE(mp.datetime)) as trading_days,
        COUNT(*) as total_minute_bars,
        MIN(DATE(mp.datetime)) as first_date,
        MAX(DATE(mp.datetime)) as last_date
    FROM minute_prices mp
    JOIN tickers t ON mp.ticker_id = t.id
    WHERE t.symbol = 'AAPL'
      AND mp.datetime >= NOW() - INTERVAL '30 days'
    """
    
    hist_df = pd.read_sql(hist_query, engine)
    print(hist_df)

if __name__ == "__main__":
    main()