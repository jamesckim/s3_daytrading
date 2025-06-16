#!/usr/bin/env python3
"""Test FMP and IB connections separately."""

import os
import sys
from dotenv import load_dotenv

print("🔧 Testing S3 AI FMP-IB Hybrid connections...\n")

# Load environment
load_dotenv()

# Test 1: Check environment variables
print("1. Environment Variables:")
fmp_key = os.getenv("FMP_API_KEY")
print(f"   FMP_API_KEY: {'✅ Set' if fmp_key and 'your_' not in fmp_key else '❌ Not set (using placeholder)'}")

postgres_uri = os.getenv("POSTGRES_URI")
if not postgres_uri:
    postgres_uri = os.getenv("DATABASE_URL")
print(f"   POSTGRES_URI: {'✅ Set' if postgres_uri and 'user:password' not in str(postgres_uri) else '❌ Not set (using placeholder)'}")

# Test 2: FMP Connection
print("\n2. Testing FMP Connection:")
try:
    from s3_ai_fmp_hybrid_trading import FMPDataProvider
    if fmp_key and 'your_' not in fmp_key:
        provider = FMPDataProvider()
        quote = provider.get_quote("AAPL")
        if quote:
            print(f"   ✅ FMP working: AAPL = ${quote.price}")
        else:
            print("   ❌ FMP connection failed")
    else:
        print("   ⚠️  Skipping - need real FMP_API_KEY")
except Exception as e:
    print(f"   ❌ FMP error: {e}")

# Test 3: PostgreSQL Connection
print("\n3. Testing PostgreSQL Connection:")
try:
    from s3_ai_hybrid_paper_trading import S3AIDataEngine
    engine = S3AIDataEngine()
    print("   ✅ PostgreSQL connection successful")
except Exception as e:
    print(f"   ❌ PostgreSQL error: {e}")

# Test 4: IB Connection
print("\n4. Testing Interactive Brokers:")
print("   Checking if TWS/IB Gateway is running...")
try:
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 7497))
    sock.close()
    
    if result == 0:
        print("   ✅ Port 7497 is open (TWS/Gateway likely running)")
        
        # Try actual connection
        from ib_insync import IB
        ib = IB()
        try:
            print("   🔌 Attempting IB connection...")
            ib.connect('127.0.0.1', 7497, clientId=999, timeout=5)
            print("   ✅ IB connection successful!")
            ib.disconnect()
        except Exception as e:
            print(f"   ❌ IB connection failed: {e}")
            print("   💡 Make sure TWS is configured to accept API connections")
    else:
        print("   ❌ Port 7497 is closed - TWS/IB Gateway not running")
        print("   💡 Please start TWS or IB Gateway first")
except Exception as e:
    print(f"   ❌ IB test error: {e}")

print("\n" + "="*50)
print("Summary:")
print("- Fix any ❌ items above before running s3_ai_fmp_ib_hybrid.py")
print("- Make sure TWS is running and configured for API access")
print("- Update .env with real API keys and database credentials")