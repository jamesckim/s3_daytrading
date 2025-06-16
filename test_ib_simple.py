#!/usr/bin/env python3
"""Simple IB connection test."""

from ib_insync import IB
import time

print("Testing IB connection...")

ib = IB()

try:
    # Try different client IDs
    for client_id in [1, 2, 3, 999]:
        print(f"\nTrying clientId {client_id}...")
        try:
            ib.connect('127.0.0.1', 7497, clientId=client_id, timeout=10)
            print(f"✅ Connected with clientId {client_id}")
            
            # Test account info
            account = ib.managedAccounts()
            print(f"   Accounts: {account}")
            
            # Get account values
            values = ib.accountValues()
            for v in values[:5]:  # Show first 5 values
                print(f"   {v.tag}: {v.value}")
            
            ib.disconnect()
            print(f"✅ Disconnected successfully")
            break
            
        except Exception as e:
            print(f"❌ Failed with clientId {client_id}: {e}")
            continue
            
except Exception as e:
    print(f"\n❌ Fatal error: {e}")
    
print("\n" + "="*50)
print("If all client IDs fail, check TWS/IB Gateway:")
print("1. Go to File → Global Configuration → API → Settings")
print("2. Enable 'Enable ActiveX and Socket Clients'")
print("3. Add 127.0.0.1 to 'Trusted IP Addresses'")
print("4. Uncheck 'Read-Only API' if you want to place orders")
print("5. Set 'Socket port' to 7497 (TWS) or 4001 (Gateway)")
print("6. Restart TWS/Gateway after changes")