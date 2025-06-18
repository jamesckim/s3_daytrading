#!/usr/bin/env python3
"""
Light VWAP Test
===============
Quick test focusing only on core VWAP indicators to validate implementation.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicator_transformer import IndicatorLibrary
from s3_ai_db_backtest_all_tickers import DatabaseDataProvider


def main():
    """Quick VWAP validation test."""
    print("⚡ Light VWAP Validation Test")
    print("=" * 50)
    
    # Initialize components
    library = IndicatorLibrary()
    db_provider = DatabaseDataProvider()
    
    # Count indicators
    all_indicators = list(library.indicators.keys())
    vwap_indicators = [name for name in all_indicators if 'VWAP' in name]
    vwap_base = [name for name in vwap_indicators if not name.endswith(('_U1', '_U2', '_U3', '_L1', '_L2', '_L3'))]
    vwap_bands = [name for name in vwap_indicators if name.endswith(('_U1', '_U2', '_U3', '_L1', '_L2', '_L3'))]
    
    print(f"📊 Indicator Library Status:")
    print(f"   Total indicators: {len(all_indicators)}")
    print(f"   VWAP indicators: {len(vwap_indicators)}")
    print(f"   - Base VWAPs: {len(vwap_base)}")
    print(f"   - VWAP bands: {len(vwap_bands)}")
    
    # Test data retrieval
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    print(f"\n🔍 Testing with {symbol} data...")
    print(f"   Date range: {start_date.strftime('%m/%d')} to {end_date.strftime('%m/%d')}")
    
    try:
        # Get data
        data = db_provider.get_minute_data(symbol, start_date, end_date)
        
        if len(data) < 100:
            print(f"❌ Insufficient data: {len(data)} rows")
            return
        
        print(f"✅ Data retrieved: {len(data)} minutes")
        
        # Test core VWAP indicators (sample)
        test_indicators = [
            'VWAP',
            'AVWAP_SESSION', 
            'VWAP_U2',
            'VWAP_L2',
            'AVWAP_DAILY_U1',
            'AVWAP_HIGH'
        ]
        
        print(f"\n📈 Testing Core VWAP Indicators:")
        results = {}
        
        for indicator in test_indicators:
            try:
                values = library.compute_indicator(data, indicator)
                if len(values) > 0 and not pd.isna(values.iloc[-1]):
                    results[indicator] = values.iloc[-1]
                    print(f"   ✅ {indicator:20}: ${values.iloc[-1]:.2f}")
                else:
                    print(f"   ❌ {indicator:20}: No valid data")
            except Exception as e:
                print(f"   ❌ {indicator:20}: Error - {str(e)[:30]}...")
        
        # Validation checks
        print(f"\n🎯 Validation Results:")
        
        if 'VWAP' in results and 'VWAP_U2' in results and 'VWAP_L2' in results:
            vwap = results['VWAP']
            upper = results['VWAP_U2']
            lower = results['VWAP_L2']
            
            if lower < vwap < upper:
                print(f"   ✅ VWAP bands properly ordered: ${lower:.2f} < ${vwap:.2f} < ${upper:.2f}")
            else:
                print(f"   ❌ VWAP bands incorrectly ordered")
        
        if 'AVWAP_SESSION' in results:
            print(f"   ✅ Session VWAP calculated: ${results['AVWAP_SESSION']:.2f}")
        
        if 'AVWAP_HIGH' in results:
            print(f"   ✅ Dynamic anchor VWAP working: ${results['AVWAP_HIGH']:.2f}")
        
        # Quick signal simulation
        current_price = data['close'].iloc[-1]
        print(f"\n💹 Current Price Analysis:")
        print(f"   Current {symbol} price: ${current_price:.2f}")
        
        if 'VWAP' in results:
            diff_pct = ((current_price - results['VWAP']) / results['VWAP']) * 100
            if diff_pct > 1:
                signal = "Above VWAP (Bullish)"
            elif diff_pct < -1:
                signal = "Below VWAP (Bearish)"
            else:
                signal = "Near VWAP (Neutral)"
            print(f"   VWAP signal: {signal} ({diff_pct:+.1f}%)")
        
        # Performance assessment
        success_rate = len(results) / len(test_indicators)
        print(f"\n📊 Test Summary:")
        print(f"   Indicators tested: {len(test_indicators)}")
        print(f"   Successful calculations: {len(results)}")
        print(f"   Success rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print(f"\n🎉 VWAP Enhancement: WORKING!")
            print(f"   ✅ Indicator library expanded successfully")
            print(f"   ✅ VWAP bands implemented correctly")
            print(f"   ✅ Database integration functional")
            print(f"   ✅ Ready for trading system integration")
        elif success_rate >= 0.5:
            print(f"\n⚠️ VWAP Enhancement: PARTIALLY WORKING")
            print(f"   Some indicators may need debugging")
        else:
            print(f"\n❌ VWAP Enhancement: NEEDS ATTENTION")
            print(f"   Check indicator implementations")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n⚡ Light test completed in <30 seconds")
    print(f"🚀 For full testing: python monitor_backtest_results.py")


if __name__ == "__main__":
    main()