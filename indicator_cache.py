#!/usr/bin/env python3
"""
Indicator Cache System
======================
Disk-based caching for computed indicator values to speed up training.
Uses Parquet format for efficient storage and retrieval.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import threading
import os
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)


class IndicatorCache:
    """Thread-safe disk-based cache for indicator computations."""
    
    def __init__(self, cache_dir: str = "data_cache/indicators"):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'errors': 0
        }
        self.manifest_file = self.cache_dir / "cache_manifest.json"
        self._load_manifest()
        
    def _load_manifest(self):
        """Load cache manifest containing metadata."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    self.manifest = json.load(f)
            except:
                self.manifest = {}
        else:
            self.manifest = {}
    
    def _save_manifest(self):
        """Save cache manifest."""
        with self._lock:
            try:
                # Create a copy to avoid "dictionary changed size during iteration" error
                manifest_copy = self.manifest.copy()
                with open(self.manifest_file, 'w') as f:
                    json.dump(manifest_copy, f, indent=2)
            except RuntimeError as e:
                logger.warning(f"Failed to save cache metadata: {e}")
                # Try again with a deep copy
                try:
                    import copy
                    manifest_copy = copy.deepcopy(self.manifest)
                    with open(self.manifest_file, 'w') as f:
                        json.dump(manifest_copy, f, indent=2)
                except Exception as e2:
                    logger.error(f"Failed to save cache metadata after retry: {e2}")
    
    def _get_cache_key(self, symbol: str, start_date: datetime, end_date: datetime, 
                      indicator_name: str, params: Dict[str, Any]) -> str:
        """Generate unique cache key for indicator computation."""
        key_data = {
            'symbol': symbol,
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'indicator': indicator_name,
            'params': params
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, symbol: str, start_date: datetime, end_date: datetime,
            indicator_name: str, params: Dict[str, Any] = None) -> Optional[pd.Series]:
        """
        Retrieve cached indicator values.
        
        Returns:
            Cached indicator series or None if not found
        """
        if params is None:
            params = {}
            
        cache_key = self._get_cache_key(symbol, start_date, end_date, indicator_name, params)
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if cache_file.exists():
            try:
                # Check file size first - parquet files need at least 8 bytes for footer
                file_size = cache_file.stat().st_size
                if file_size < 8:
                    logger.warning(f"Cache file {cache_file} is too small ({file_size} bytes), removing...")
                    cache_file.unlink()
                    # Remove from manifest
                    with self._lock:
                        if cache_key in self.manifest:
                            del self.manifest[cache_key]
                    return None
                
                # Read from parquet
                df = pd.read_parquet(cache_file)
                
                # Validate data
                if df.empty or 'value' not in df.columns:
                    logger.warning(f"Invalid cache data in {cache_file}, removing...")
                    cache_file.unlink()
                    with self._lock:
                        if cache_key in self.manifest:
                            del self.manifest[cache_key]
                    return None
                
                # Convert index back to datetime
                df.index = pd.to_datetime(df.index)
                
                with self._lock:
                    self._stats['hits'] += 1
                    
                logger.debug(f"Cache hit for {symbol} {indicator_name}")
                return df['value']
                
            except Exception as e:
                logger.error(f"Error reading cache {cache_file}: {e}")
                with self._lock:
                    self._stats['errors'] += 1
                # Remove corrupted cache file
                try:
                    cache_file.unlink()
                    # Remove from manifest
                    if cache_key in self.manifest:
                        del self.manifest[cache_key]
                except Exception as e2:
                    logger.error(f"Error removing corrupted cache file: {e2}")
        
        with self._lock:
            self._stats['misses'] += 1
        return None
    
    def put(self, symbol: str, start_date: datetime, end_date: datetime,
            indicator_name: str, values: pd.Series, params: Dict[str, Any] = None):
        """
        Store indicator values in cache.
        
        Args:
            symbol: Stock symbol
            start_date: Start date of data
            end_date: End date of data
            indicator_name: Name of indicator
            values: Computed indicator values
            params: Indicator parameters
        """
        if params is None:
            params = {}
            
        cache_key = self._get_cache_key(symbol, start_date, end_date, indicator_name, params)
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        try:
            # Create DataFrame for storage
            df = pd.DataFrame({'value': values})
            
            # Use atomic write - write to temp file then rename
            temp_file = cache_file.with_suffix('.tmp')
            
            # Save to temp file first
            df.to_parquet(temp_file, engine='pyarrow', compression='snappy')
            
            # Atomic rename (this prevents partial writes)
            temp_file.replace(cache_file)
            
            # Update manifest
            saves_count = 0
            with self._lock:
                self.manifest[cache_key] = {
                    'symbol': symbol,
                    'indicator': indicator_name,
                    'params': params,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'created': datetime.now().isoformat(),
                    'size_bytes': cache_file.stat().st_size,
                    'num_values': len(values)
                }
                self._stats['saves'] += 1
                saves_count = self._stats['saves']
            
            # Save manifest periodically outside the main lock to avoid blocking
            if saves_count % 100 == 0:
                self._save_manifest()
                
            logger.debug(f"Cached {symbol} {indicator_name}")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            with self._lock:
                self._stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.copy()
            
        # Calculate cache size
        total_size = 0
        num_files = 0
        for file in self.cache_dir.glob("*.parquet"):
            total_size += file.stat().st_size
            num_files += 1
        
        stats['total_size_mb'] = total_size / (1024 * 1024)
        stats['num_files'] = num_files
        stats['hit_rate'] = stats['hits'] / (stats['hits'] + stats['misses']) if (stats['hits'] + stats['misses']) > 0 else 0
        
        return stats
    
    def clear(self):
        """Clear all cache files."""
        logger.info("Clearing indicator cache...")
        
        # Remove all parquet files
        for file in self.cache_dir.glob("*.parquet"):
            try:
                file.unlink()
            except Exception as e:
                logger.error(f"Error removing {file}: {e}")
        
        # Clear manifest
        self.manifest = {}
        self._save_manifest()
        
        # Reset stats
        with self._lock:
            self._stats = {
                'hits': 0,
                'misses': 0,
                'saves': 0,
                'errors': 0
            }
        
        logger.info("Cache cleared")
    
    def cleanup_old(self, days: int = 7):
        """Remove cache entries older than specified days."""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        removed = 0
        
        # Create a list of keys to remove to avoid modifying dict during iteration
        keys_to_remove = []
        
        with self._lock:
            for cache_key, info in self.manifest.items():
                try:
                    created = datetime.fromisoformat(info['created']).timestamp()
                    if created < cutoff_date:
                        keys_to_remove.append(cache_key)
                except Exception as e:
                    logger.error(f"Error checking {cache_key}: {e}")
        
        # Now remove the old entries
        for cache_key in keys_to_remove:
            try:
                # Remove file
                cache_file = self.cache_dir / f"{cache_key}.parquet"
                if cache_file.exists():
                    cache_file.unlink()
                
                # Remove from manifest
                with self._lock:
                    if cache_key in self.manifest:
                        del self.manifest[cache_key]
                        removed += 1
            except Exception as e:
                logger.error(f"Error cleaning up {cache_key}: {e}")
        
        if removed > 0:
            self._save_manifest()
            logger.info(f"Removed {removed} old cache entries")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        info = {
            'cache_dir': str(self.cache_dir),
            'stats': self.get_stats(),
            'entries_by_symbol': {},
            'entries_by_indicator': {},
            'oldest_entry': None,
            'newest_entry': None
        }
        
        # Analyze manifest (create a copy to avoid concurrent modification)
        with self._lock:
            manifest_copy = self.manifest.copy()
        
        for cache_key, entry in manifest_copy.items():
            symbol = entry['symbol']
            indicator = entry['indicator']
            created = entry['created']
            
            # Count by symbol
            if symbol not in info['entries_by_symbol']:
                info['entries_by_symbol'][symbol] = 0
            info['entries_by_symbol'][symbol] += 1
            
            # Count by indicator
            if indicator not in info['entries_by_indicator']:
                info['entries_by_indicator'][indicator] = 0
            info['entries_by_indicator'][indicator] += 1
            
            # Track oldest/newest
            if info['oldest_entry'] is None or created < info['oldest_entry']:
                info['oldest_entry'] = created
            if info['newest_entry'] is None or created > info['newest_entry']:
                info['newest_entry'] = created
        
        return info


class CachedIndicatorComputer:
    """Wrapper for indicator computation with caching."""
    
    def __init__(self, indicator_library, cache: Optional[IndicatorCache] = None):
        """
        Initialize cached indicator computer.
        
        Args:
            indicator_library: IndicatorLibrary instance
            cache: IndicatorCache instance (creates new if None)
        """
        self.indicator_library = indicator_library
        self.cache = cache or IndicatorCache()
        
    def compute_indicator(self, data: pd.DataFrame, indicator_name: str, 
                         symbol: str = "UNKNOWN") -> pd.Series:
        """
        Compute indicator with caching.
        
        Args:
            data: OHLCV data
            indicator_name: Name of indicator
            symbol: Stock symbol (for cache key)
            
        Returns:
            Computed indicator series
        """
        # Extract date range
        start_date = data.index[0]
        end_date = data.index[-1]
        
        # Get indicator config
        indicator_config = self.indicator_library.indicators.get(indicator_name)
        params = indicator_config.params if indicator_config else {}
        
        # Check cache
        cached_values = self.cache.get(symbol, start_date, end_date, indicator_name, params)
        if cached_values is not None:
            # Ensure index alignment
            return cached_values.reindex(data.index, method='ffill')
        
        # Compute indicator
        try:
            values = self.indicator_library.compute_indicator(data, indicator_name)
            
            # Cache the result
            if len(values) > 0:
                self.cache.put(symbol, start_date, end_date, indicator_name, values, params)
            
            return values
            
        except Exception as e:
            logger.error(f"Error computing {indicator_name}: {e}")
            return pd.Series(index=data.index, dtype=float)


def test_cache():
    """Test the caching system."""
    print("Testing Indicator Cache System...")
    
    # Create cache
    cache = IndicatorCache("test_cache")
    
    # Create test data
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='1min')
    test_series = pd.Series(np.random.randn(len(dates)), index=dates)
    
    # Test put/get
    cache.put('AAPL', dates[0], dates[-1], 'RSI_14', test_series, {'period': 14})
    
    # Retrieve
    retrieved = cache.get('AAPL', dates[0], dates[-1], 'RSI_14', {'period': 14})
    
    if retrieved is not None:
        print("✓ Cache store/retrieve working")
        print(f"  Original length: {len(test_series)}")
        print(f"  Retrieved length: {len(retrieved)}")
    else:
        print("✗ Cache retrieve failed")
    
    # Test stats
    stats = cache.get_stats()
    print(f"\nCache Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    cache.clear()
    print("\n✓ Cache cleared")


if __name__ == "__main__":
    test_cache()