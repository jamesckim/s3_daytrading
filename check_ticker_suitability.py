#!/usr/bin/env python3
"""
Check Ticker Suitability for S3 Minute-Level Trading
====================================================

This program checks if tickers have sufficient data for the S3 AI trading strategy.
The S3 strategy requires at least 1,000 minute records within the analysis period.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from tabulate import tabulate
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stocks_net_db.models import DATABASE_URL


def check_ticker_data(tickers, days=30, min_records=1000):
    """
    Check if tickers have sufficient minute data for S3 trading.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days to check
        min_records: Minimum records required (S3 uses 1000)
    """
    engine = create_engine(DATABASE_URL)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"\nChecking ticker suitability for S3 trading strategy")
    print(f"Period: {start_date.date()} to {end_date.date()} ({days} days)")
    print(f"Minimum records required: {min_records}")
    print("="*80)
    
    results = []
    
    query = text("""
        SELECT 
            t.symbol,
            COUNT(mp.*) as total_records,
            COUNT(DISTINCT DATE(mp.datetime)) as trading_days,
            MIN(mp.datetime) as first_record,
            MAX(mp.datetime) as last_record
        FROM tickers t
        LEFT JOIN minute_prices mp ON t.id = mp.ticker_id 
            AND mp.datetime >= :start_date 
            AND mp.datetime <= :end_date
        WHERE t.symbol = :symbol
        GROUP BY t.symbol
    """)
    
    with engine.connect() as conn:
        for ticker in tickers:
            result = conn.execute(query, {
                'symbol': ticker,
                'start_date': start_date,
                'end_date': end_date
            }).fetchone()
            
            if result:
                suitable = result.total_records >= min_records
                avg_per_day = result.total_records / result.trading_days if result.trading_days > 0 else 0
                
                results.append({
                    'Symbol': ticker,
                    'Total Records': result.total_records,
                    'Trading Days': result.trading_days,
                    'Avg/Day': round(avg_per_day, 1),
                    'Suitable': '✓' if suitable else '✗',
                    'Status': 'OK' if suitable else f'Need {min_records - result.total_records} more'
                })
            else:
                results.append({
                    'Symbol': ticker,
                    'Total Records': 0,
                    'Trading Days': 0,
                    'Avg/Day': 0,
                    'Suitable': '✗',
                    'Status': 'No data'
                })
    
    # Display results
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='simple', showindex=False))
    
    # Summary
    suitable_count = len([r for r in results if r['Suitable'] == '✓'])
    print(f"\nSummary: {suitable_count}/{len(tickers)} tickers suitable for S3 trading")
    
    # Save results if requested
    if len(tickers) > 20:  # Auto-save for large lists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f's3_suitability_check_{timestamp}.csv', index=False)
        
        # Save suitable and unsuitable ticker lists
        suitable_tickers = [r['Symbol'] for r in results if r['Suitable'] == '✓']
        unsuitable_tickers = [r['Symbol'] for r in results if r['Suitable'] == '✗']
        
        with open(f's3_suitable_tickers_{timestamp}.json', 'w') as f:
            json.dump(suitable_tickers, f, indent=2)
        
        with open(f's3_unsuitable_tickers_{timestamp}.json', 'w') as f:
            json.dump(unsuitable_tickers, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  - s3_suitability_check_{timestamp}.csv")
        print(f"  - s3_suitable_tickers_{timestamp}.json ({len(suitable_tickers)} tickers)")
        print(f"  - s3_unsuitable_tickers_{timestamp}.json ({len(unsuitable_tickers)} tickers)")
    
    # Recommendations for unsuitable tickers
    unsuitable = [r for r in results if r['Suitable'] == '✗']
    if unsuitable:
        print("\nRecommendations for unsuitable tickers:")
        for ticker in unsuitable:
            if ticker['Total Records'] == 0:
                print(f"  {ticker['Symbol']}: No data available - needs data collection")
            elif ticker['Total Records'] < min_records:
                days_needed = int((min_records - ticker['Total Records']) / (ticker['Avg/Day'] if ticker['Avg/Day'] > 0 else 1))
                print(f"  {ticker['Symbol']}: Increase lookback period to ~{days + days_needed} days")
    
    return df


def check_s3_compatibility(days_list=[30, 60, 90, 120]):
    """Check how many days of data are needed for each ticker."""
    import json
    
    # Load james_tickers.json
    with open('../james_tickers.json', 'r') as f:
        tickers = json.load(f)
    
    print("\nS3 Strategy Data Requirements Analysis")
    print("="*80)
    
    engine = create_engine(DATABASE_URL)
    
    for ticker in tickers:
        print(f"\n{ticker}:")
        
        query = text("""
            SELECT 
                COUNT(*) as record_count,
                MIN(datetime) as oldest_record,
                MAX(datetime) as newest_record
            FROM minute_prices mp
            JOIN tickers t ON mp.ticker_id = t.id
            WHERE t.symbol = :symbol
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {'symbol': ticker}).fetchone()
            
            if result and result.record_count > 0:
                print(f"  Total records: {result.record_count:,}")
                print(f"  Date range: {result.oldest_record} to {result.newest_record}")
                
                # Check different day ranges
                print("  Records by period:")
                for days in days_list:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    count_query = text("""
                        SELECT COUNT(*) as count
                        FROM minute_prices mp
                        JOIN tickers t ON mp.ticker_id = t.id
                        WHERE t.symbol = :symbol AND mp.datetime >= :start_date
                    """)
                    
                    count_result = conn.execute(count_query, {
                        'symbol': ticker,
                        'start_date': start_date
                    }).fetchone()
                    
                    suitable = "✓" if count_result.count >= 1000 else "✗"
                    print(f"    {days:3d} days: {count_result.count:5d} records {suitable}")
            else:
                print("  No data available")


def main():
    parser = argparse.ArgumentParser(
        description='Check ticker suitability for S3 minute-level trading'
    )
    parser.add_argument('--tickers', nargs='+', 
                       help='Specific tickers to check')
    parser.add_argument('--file', default='../james_tickers.json',
                       help='JSON file with tickers (default: ../james_tickers.json)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to analyze (default: 30)')
    parser.add_argument('--min-records', type=int, default=1000,
                       help='Minimum records required (default: 1000)')
    parser.add_argument('--analyze', action='store_true',
                       help='Run detailed analysis for different time periods')
    
    args = parser.parse_args()
    
    if args.analyze:
        check_s3_compatibility()
    else:
        # Load tickers
        if args.tickers:
            tickers = args.tickers
        else:
            with open(args.file, 'r') as f:
                tickers = json.load(f)
        
        check_ticker_data(tickers, args.days, args.min_records)


if __name__ == "__main__":
    main()