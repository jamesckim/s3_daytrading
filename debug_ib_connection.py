#!/usr/bin/env python3
"""Debug IB API connection with detailed error information."""

import socket
import sys
import time
from datetime import datetime

print("🔍 IB API Connection Debugger")
print("=" * 50)
print(f"Time: {datetime.now().strftime('%I:%M:%S %p PT')}")

# First check if port is open
print("\n1. Testing raw socket connection to 127.0.0.1:7497...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)

try:
    result = sock.connect_ex(('127.0.0.1', 7497))
    if result == 0:
        print("   ✅ Socket connection successful - port is open")
        
        # Try to send API handshake
        print("\n2. Attempting API handshake...")
        try:
            # IB API v9.72+ handshake
            handshake = b'API\0\0\0\0\x09v100..142'
            sock.send(handshake)
            
            # Wait for response
            sock.settimeout(5)
            response = sock.recv(1024)
            
            if response:
                print(f"   ✅ Received response: {response[:50]}...")
                print("   API is responding!")
            else:
                print("   ❌ No response from API")
                
        except socket.timeout:
            print("   ❌ Timeout waiting for API response")
            print("   This suggests API is not enabled in TWS")
        except Exception as e:
            print(f"   ❌ Handshake error: {e}")
    else:
        print(f"   ❌ Socket connection failed (error code: {result})")
        print("   Port 7497 is not accessible")
        
except Exception as e:
    print(f"   ❌ Connection error: {e}")
finally:
    sock.close()

# Now try with ib_insync
print("\n3. Testing with ib_insync library...")
try:
    from ib_insync import IB, util
    
    # Enable debug logging
    util.logToConsole(level='DEBUG')
    
    ib = IB()
    print("   Attempting connection with debug logging enabled...")
    
    try:
        ib.connect('127.0.0.1', 7497, clientId=998, timeout=10)
        print("   ✅ Connection successful!")
        ib.disconnect()
    except Exception as e:
        print(f"   ❌ ib_insync error: {type(e).__name__}: {e}")
        
except ImportError:
    print("   ❌ ib_insync not installed")

print("\n" + "=" * 50)
print("To view TWS API logs:")
print("1. In TWS: Help → Troubleshooting → API Logs")
print("2. Or check: View → API → Socket Log")
print("3. Make sure 'Log Level' is set to 'Detail' or 'Information'")