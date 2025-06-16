# TWS API Configuration Guide for S3 AI FMP-IB Hybrid

## Step-by-Step TWS Configuration

### 1. Open TWS Global Configuration
- In TWS, go to **File → Global Configuration**
- Or use keyboard shortcut: **Ctrl+Alt+U** (Windows) or **Cmd+Option+U** (Mac)

### 2. Navigate to API Settings
- In the left sidebar, click on **API**
- Then click on **Settings**

### 3. Configure API Settings
You need to enable these settings:

✅ **Enable ActiveX and Socket Clients**
- This MUST be checked for API access to work

✅ **Socket port**: 7497
- This is the default for TWS (IB Gateway uses 4001)

✅ **Trusted IP Addresses**: 127.0.0.1
- Click "Create" and add 127.0.0.1
- This allows local connections

✅ **Master API client ID**: (leave blank or set to 0)
- This allows any client ID to connect

❌ **Read-Only API**: UNCHECKED
- Uncheck this to allow order placement

✅ **Bypass Order Precautions for API Orders**: (Optional)
- Check this to avoid popup confirmations

### 4. Additional Settings (Optional but Recommended)

#### Under "Precautions":
- Consider unchecking "Bypass Order Precautions for API Orders" if you want to review orders

#### Under "Lock and Exit":
- Set appropriate timeout values

### 5. Apply and Restart
1. Click **Apply** then **OK**
2. **IMPORTANT**: You MUST restart TWS for changes to take effect
3. File → Exit → Yes to save settings

### 6. After Restart
When TWS restarts:
1. Log in normally
2. You should see "API" in green in the status bar
3. If you see a popup about accepting connections, click "Accept"

## Testing the Connection

After restarting TWS, test with this command:
```bash
cd /Users/jkim/Desktop/code/trading/s3_daytrading
python test_ib_simple.py
```

You should see:
```
✅ Connected with clientId 1
   Accounts: ['DU1234567']
```

## Common Issues and Solutions

### Issue: "API connection failed: TimeoutError()"
**Solution**: TWS is not accepting API connections
- Verify "Enable ActiveX and Socket Clients" is checked
- Make sure you restarted TWS after configuration
- Check that port 7497 is not blocked by firewall

### Issue: "Client ID already in use"
**Solution**: Another application is using that client ID
- Try a different client ID (1-999)
- Or close other API applications

### Issue: "No security definition found"
**Solution**: Normal for paper trading accounts
- This is expected behavior for some symbols

### Issue: Popup windows when placing orders
**Solution**: Configure order precautions
- In Global Configuration → API → Precautions
- Check "Bypass Order Precautions for API Orders"

## Running the FMP-IB Hybrid System

Once TWS is properly configured:

```bash
# Kill any running instances
pkill -f "s3_ai"

# Run the FMP-IB hybrid system
python s3_ai_fmp_ib_hybrid.py

# Or with custom config
python s3_ai_fmp_ib_hybrid.py my_config.json
```

## Verification Checklist

Before running the hybrid system, verify:
- [ ] TWS is running (not IB Gateway)
- [ ] You see "API" in green in the TWS status bar
- [ ] Port 7497 is open (test with `telnet localhost 7497`)
- [ ] You've restarted TWS after configuration changes
- [ ] The test script connects successfully

## Need Help?

If you still have issues:
1. Check TWS logs: Help → Troubleshooting → API Logs
2. Try IB Gateway instead of TWS (uses port 4001)
3. Make sure no firewall is blocking port 7497
4. Try running TWS as administrator (Windows) or with proper permissions (Mac)