#!/usr/bin/env python3
"""
Summary of VWAP Indicators with Bands
=====================================
Complete summary of all VWAP indicators and their standard deviation bands.
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicator_transformer import IndicatorLibrary
from tabulate import tabulate


def main():
    """Summarize all VWAP indicators with bands."""
    print("VWAP Indicators with Standard Deviation Bands")
    print("=" * 80)
    
    # Initialize library
    library = IndicatorLibrary()
    all_indicators = list(library.indicators.keys())
    
    # Categorize VWAP indicators
    base_vwaps = [name for name in all_indicators if 'VWAP' in name 
                  and not name.endswith(('_U1', '_U2', '_U3', '_L1', '_L2', '_L3'))]
    
    print(f"\nTotal Base VWAP Indicators: {len(base_vwaps)}")
    print("-" * 80)
    
    # Standard VWAP
    print("\n1. STANDARD VWAP")
    print("   • VWAP: Volume Weighted Average Price (cumulative)")
    print("   • Bands:")
    for std in [1, 2, 3]:
        print(f"     - VWAP_U{std}: Upper {std}σ band")
        print(f"     - VWAP_L{std}: Lower {std}σ band")
    
    # Anchored VWAPs
    print("\n2. TIME-ANCHORED VWAPs")
    time_anchored = ['AVWAP_SESSION', 'AVWAP_DAILY', 'AVWAP_WEEKLY', 'AVWAP_MONTHLY']
    for i, vwap in enumerate(time_anchored, 1):
        config = library.indicators[vwap]
        anchor = config.params['anchor']
        
        print(f"\n   {i}. {vwap}: Anchored at {anchor}")
        print("      • Bands:")
        for std in [1, 2, 3]:
            print(f"        - {vwap}_U{std}: Upper {std}σ band")
            print(f"        - {vwap}_L{std}: Lower {std}σ band")
    
    # Dynamic VWAPs
    print("\n3. DYNAMICALLY-ANCHORED VWAPs (Price/Volume Events)")
    dynamic_vwaps = [
        ('AVWAP_HIGH', 'Intraday high price'),
        ('AVWAP_LOW', 'Intraday low price'),
        ('AVWAP_HVOL', 'Highest volume minute'),
        ('AVWAP_PREV_HIGH', 'Previous day high'),
        ('AVWAP_PREV_LOW', 'Previous day low'),
        ('AVWAP_PREV_HVOL', 'Previous day high volume')
    ]
    
    for i, (vwap, desc) in enumerate(dynamic_vwaps, 1):
        print(f"\n   {i}. {vwap}: Anchored at {desc}")
        print("      • Bands:")
        for std in [1, 2, 3]:
            print(f"        - {vwap}_U{std}: Upper {std}σ band")
            print(f"        - {vwap}_L{std}: Lower {std}σ band")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    total_base = len(base_vwaps)
    bands_per_vwap = 6  # 3 upper + 3 lower
    total_bands = total_base * bands_per_vwap
    total_indicators = total_base + total_bands
    
    summary_data = [
        ['Base VWAP Indicators', total_base],
        ['Standard VWAP', 1],
        ['Time-Anchored VWAPs', 4],
        ['Dynamic-Anchored VWAPs', 6],
        ['Bands per VWAP', bands_per_vwap],
        ['Total VWAP Bands', total_bands],
        ['Total VWAP-related Indicators', total_indicators]
    ]
    
    print(tabulate(summary_data, headers=['Metric', 'Count'], tablefmt='simple'))
    
    # Trading applications
    print("\n" + "=" * 80)
    print("TRADING APPLICATIONS")
    print("=" * 80)
    
    print("\n📊 Standard Deviation Bands:")
    print("   • 1σ (68%): Normal trading range")
    print("   • 2σ (95%): Extended range, potential reversal")
    print("   • 3σ (99.7%): Extreme range, strong reversal signal")
    
    print("\n📈 Band Trading Strategies:")
    print("   1. Mean Reversion: Buy at lower bands, sell at upper bands")
    print("   2. Breakout: Trade continuation when price breaks 2σ or 3σ")
    print("   3. Squeeze: Low volatility (narrow bands) precedes big moves")
    print("   4. Trend Following: Stay long above VWAP, short below")
    
    print("\n🎯 Multiple VWAP Confluence:")
    print("   • Strong support: Multiple VWAPs clustered below price")
    print("   • Strong resistance: Multiple VWAPs clustered above price")
    print("   • Breakout confirmation: Price above all VWAPs and bands")
    print("   • Trend exhaustion: Price at 3σ bands of multiple VWAPs")
    
    print("\n⏱️ Timeframe Selection:")
    print("   • Scalping: VWAP, AVWAP_SESSION with 1σ bands")
    print("   • Day Trading: AVWAP_DAILY, dynamic VWAPs")
    print("   • Swing Trading: AVWAP_WEEKLY with 2σ bands")
    print("   • Position Trading: AVWAP_MONTHLY with 3σ bands")
    
    print("\n🔍 Dynamic Anchor Benefits:")
    print("   • AVWAP_HIGH/LOW: Natural support/resistance levels")
    print("   • AVWAP_HVOL: Institutional interest points")
    print("   • AVWAP_PREV_*: Yesterday's levels for gap analysis")


if __name__ == "__main__":
    main()