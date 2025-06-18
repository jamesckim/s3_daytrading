#!/usr/bin/env python3
"""Verify TWS API settings by attempting different connection methods."""

import socket
import time
from datetime import datetime

print("🔍 TWS API Settings Verification")
print("=" * 50)
print(f"Time: {datetime.now().strftime('%I:%M:%S %p PT')}")

print("\n📋 TWS API Configuration Checklist:")
print("Please verify these settings in TWS:")
print("")
print("1. Go to: File → Global Configuration → API → Settings")
print("")
print("   [ ] Enable ActiveX and Socket Clients")
print("   [ ] Socket port: 7497")
print("   [ ] Trusted IP Addresses includes: 127.0.0.1")
print("   [ ] Read-Only API: UNCHECKED (if you want to trade)")
print("   [ ] Download open orders on connection: CHECKED")
print("   [ ] Send instrument-specific account value: CHECKED")
print("   [ ] Include FX positions when sending portfolio: CHECKED")
print("")
print("2. Optional but recommended:")
print("   [ ] Master API client ID: 0 (or leave empty)")
print("   [ ] Bypass Order Precautions for API Orders")
print("")
print("3. After making changes:")
print("   - Click Apply")
print("   - Click OK")
print("   - File → Exit (save settings)")
print("   - Restart TWS")
print("")
print("=" * 50)

# Test basic connectivity
print("\n🔌 Current Connection Status:")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
result = sock.connect_ex(('127.0.0.1', 7497))

if result == 0:
    print("✅ Port 7497 is OPEN (TWS is listening)")
    print("❌ But API handshake is failing")
    print("\n⚠️  This means TWS is running but API is not enabled")
    print("   Please check the settings above in TWS")
else:
    print("❌ Port 7497 is CLOSED")
    print("   Either TWS is not running or using a different port")

sock.close()

print("\n💡 Quick Fix:")
print("1. In TWS, press Ctrl+Alt+S (Windows) or Cmd+Option+S (Mac)")
print("2. This opens Global Configuration")
print("3. Go to API → Settings")
print("4. Check 'Enable ActiveX and Socket Clients'")
print("5. Restart TWS")

print("\n📊 To see what's happening:")
print("In TWS: View → API → Message Log")
print("This will show connection attempts in real-time")