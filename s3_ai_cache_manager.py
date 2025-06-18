#!/usr/bin/env python3
"""
S3 AI Cache Manager
==================
Provides caching functionality for minute data to speed up repeated backtests.
Uses Parquet format for efficient storage and fast loading.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class MinuteDataCache:
    """Cache manager for minute-level stock data."""
    
    def __init__(self, cache_dir: str = "data_cache"):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cached data files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        self.minute_data_dir = self.cache_dir / "minute_data"
        self.minute_data_dir.mkdir(exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"📦 Cache manager initialized at {self.cache_dir}")
    
    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def get_cache_key(self, symbol: str, start_date: datetime, end_date: datetime) -> str:
        """Generate cache key for data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Cache key string
        """
        key_string = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        # Use hash for shorter filenames
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cache_path(self, symbol: str, start_date: datetime, end_date: datetime) -> Path:
        """Get cache file path.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Path to cache file
        """
        cache_key = self.get_cache_key(symbol, start_date, end_date)
        return self.minute_data_dir / f"{symbol}_{cache_key}.parquet"
    
    def save_to_cache(self, symbol: str, start_date: datetime, end_date: datetime, 
                      data: pd.DataFrame) -> bool:
        """Save data to cache.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            data: DataFrame to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_path = self.get_cache_path(symbol, start_date, end_date)
            
            # Save to parquet with compression
            data.to_parquet(cache_path, compression='snappy', index=True)
            
            # Update metadata
            cache_key = self.get_cache_key(symbol, start_date, end_date)
            self.metadata[cache_key] = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'cached_at': datetime.now(),
                'row_count': len(data),
                'file_size': cache_path.stat().st_size
            }
            self._save_metadata()
            
            logger.debug(f"💾 Cached {symbol} data: {len(data)} rows, "
                        f"{cache_path.stat().st_size / 1024 / 1024:.1f}MB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache {symbol} data: {e}")
            return False
    
    def load_from_cache(self, symbol: str, start_date: datetime, 
                       end_date: datetime) -> pd.DataFrame:
        """Load data from cache if available.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame if cached, None otherwise
        """
        try:
            cache_path = self.get_cache_path(symbol, start_date, end_date)
            
            if cache_path.exists():
                # Check if cache is recent (optional: add age check)
                data = pd.read_parquet(cache_path)
                logger.debug(f"📂 Loaded {symbol} from cache: {len(data)} rows")
                return data
                
        except Exception as e:
            logger.warning(f"Failed to load {symbol} from cache: {e}")
        
        return None
    
    def clear_cache(self, older_than_days: int = None):
        """Clear cache files.
        
        Args:
            older_than_days: Only clear files older than this many days
        """
        cleared_count = 0
        cleared_size = 0
        
        for cache_file in self.minute_data_dir.glob("*.parquet"):
            try:
                if older_than_days:
                    file_age = (datetime.now() - datetime.fromtimestamp(
                        cache_file.stat().st_mtime)).days
                    if file_age < older_than_days:
                        continue
                
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                cleared_count += 1
                cleared_size += file_size
                
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        # Clear metadata
        if not older_than_days:
            self.metadata = {}
            self._save_metadata()
        
        logger.info(f"🧹 Cleared {cleared_count} cache files, "
                   f"{cleared_size / 1024 / 1024:.1f}MB freed")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_files = 0
        total_size = 0
        
        for cache_file in self.minute_data_dir.glob("*.parquet"):
            total_files += 1
            total_size += cache_file.stat().st_size
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size / 1024 / 1024,
            'cache_dir': str(self.cache_dir),
            'metadata_entries': len(self.metadata)
        }


class BacktestResultCache:
    """Cache manager for backtest results."""
    
    def __init__(self, cache_dir: str = "data_cache"):
        """Initialize result cache manager.
        
        Args:
            cache_dir: Directory to store cached results
        """
        self.cache_dir = Path(cache_dir)
        self.results_dir = self.cache_dir / "backtest_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_result(self, symbol: str, result: dict, config: dict) -> bool:
        """Save backtest result to cache.
        
        Args:
            symbol: Stock symbol
            result: Backtest result dictionary
            config: Backtest configuration
            
        Returns:
            True if successful
        """
        try:
            # Create filename with config hash
            config_str = json.dumps(config, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            
            result_file = self.results_dir / f"{symbol}_{config_hash}.json"
            
            # Add metadata
            result_with_meta = {
                'symbol': symbol,
                'config': config,
                'cached_at': datetime.now().isoformat(),
                'result': result
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_with_meta, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache result for {symbol}: {e}")
            return False
    
    def load_result(self, symbol: str, config: dict) -> dict:
        """Load cached backtest result if available.
        
        Args:
            symbol: Stock symbol
            config: Backtest configuration
            
        Returns:
            Result dictionary if cached, None otherwise
        """
        try:
            config_str = json.dumps(config, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            
            result_file = self.results_dir / f"{symbol}_{config_hash}.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                # Check if config matches exactly
                if data['config'] == config:
                    logger.debug(f"📂 Loaded cached result for {symbol}")
                    return data['result']
                    
        except Exception as e:
            logger.warning(f"Failed to load cached result for {symbol}: {e}")
        
        return None