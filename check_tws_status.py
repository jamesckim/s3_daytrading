#!/usr/bin/env python3
"""Check TWS API configuration status."""

import socket
import subprocess
import time
from datetime import datetime

print("🔍 TWS API Configuration Checker")
print("=" * 50)
print(f"Time: {datetime.now().strftime('%I:%M:%S %p PT')}\n")

# Check if TWS process is running
print("1. Checking if TWS is running...")
try:
    result = subprocess.run(['pgrep', '-f', 'Trader Workstation'], 
                          capture_output=True, text=True)
    if result.stdout.strip():
        print("   ✅ TWS process found (PID: {})".format(result.stdout.strip()))
    else:
        print("   ❌ TWS is not running")
        print("   💡 Please start Trader Workstation first")
except Exception as e:
    print(f"   ⚠️  Could not check process: {e}")

# Check if port 7497 is open
print("\n2. Checking if port 7497 is accessible...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
result = sock.connect_ex(('127.0.0.1', 7497))
sock.close()

if result == 0:
    print("   ✅ Port 7497 is OPEN - API likely enabled")
else:
    print("   ❌ Port 7497 is CLOSED - API not accessible")
    print("   💡 Check these TWS settings:")
    print("      - File → Global Configuration → API → Settings")
    print("      - Enable 'Enable ActiveX and Socket Clients'")
    print("      - Restart TWS after changing settings")

# Try a quick connection
print("\n3. Testing API connection...")
try:
    from ib_insync import IB
    ib = IB()
    
    # Try to connect with a short timeout
    ib.connect('127.0.0.1', 7497, clientId=999, timeout=5)
    
    print("   ✅ API connection successful!")
    
    # Get some basic info
    accounts = ib.managedAccounts()
    if accounts:
        print(f"   📊 Account(s): {', '.join(accounts)}")
    
    # Check if it's paper trading
    account_values = ib.accountValues()
    for av in account_values:
        if av.tag == 'AccountType':
            print(f"   📝 Account Type: {av.value}")
            break
    
    ib.disconnect()
    
except Exception as e:
    error_msg = str(e)
    print(f"   ❌ Connection failed: {error_msg}")
    
    if "TimeoutError" in error_msg or not error_msg:
        print("\n   🔧 TWS Configuration Steps:")
        print("   1. In TWS: File → Global Configuration")
        print("   2. Navigate to API → Settings")
        print("   3. Check 'Enable ActiveX and Socket Clients'")
        print("   4. Add '127.0.0.1' to Trusted IP Addresses")
        print("   5. Set Socket port to 7497")
        print("   6. Uncheck 'Read-Only API' for trading")
        print("   7. Click Apply & OK")
        print("   8. RESTART TWS (File → Exit)")
    elif "port 7497" in error_msg:
        print("   💡 Port 7497 is blocked or in use")
    elif "client id" in error_msg.lower():
        print("   💡 Client ID 999 may be in use, try another ID")

print("\n" + "=" * 50)
print("Summary:")
if result == 0:
    print("✅ Port is open but API connection is failing")
    print("🔧 TWS needs configuration - see guide above")
else:
    print("❌ TWS API is not accessible")
    print("🔧 Enable API in TWS settings and restart")

print("\n📖 See TWS_API_SETUP_GUIDE.md for detailed instructions")