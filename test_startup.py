#!/usr/bin/env python3
"""Test startup sequence to diagnose where the script is stuck."""

import sys
import os

print("1. Starting test...")

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("2. Loaded .env")
    
    # Check environment variables
    fmp_key = os.getenv("FMP_API_KEY")
    postgres_uri = os.getenv("POSTGRES_URI")
    
    print(f"3. FMP_API_KEY: {'Set' if fmp_key and fmp_key != 'your_fmp_api_key_here' else 'Not set'}")
    print(f"4. POSTGRES_URI: {'Set' if postgres_uri and 'user:password' not in postgres_uri else 'Not set'}")
    
    # Try importing S3 components
    print("5. Importing S3AIDataEngine...")
    from s3_ai_hybrid_paper_trading import S3AIDataEngine
    print("6. Creating S3AIDataEngine...")
    
    # This will fail if POSTGRES_URI is not set properly
    try:
        data_engine = S3AIDataEngine()
        print("7. S3AIDataEngine created successfully")
    except ValueError as e:
        print(f"7. ERROR: {e}")
        print("\nThe script is stuck because POSTGRES_URI is not properly configured.")
        print("Please update .env with your actual database credentials.")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nTo fix this, update .env with real values:")
print("- FMP_API_KEY: Your actual FMP API key")
print("- POSTGRES_URI: postgresql://username:password@localhost:5432/your_db")