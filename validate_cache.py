#!/usr/bin/env python3
"""
Validate and Clean Indicator Cache
==================================
Checks for corrupted cache files and removes them.
"""

import os
from pathlib import Path
import logging
from indicator_cache import IndicatorCache
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_cache(cache_dir: str = "data_cache/indicators"):
    """Validate all cache files and remove corrupted ones."""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        logger.info("Cache directory doesn't exist yet")
        return
    
    logger.info(f"Validating cache in {cache_path}")
    
    # Statistics
    total_files = 0
    valid_files = 0
    corrupted_files = 0
    small_files = 0
    removed_size = 0
    
    # Check all parquet files
    for file in cache_path.glob("*.parquet"):
        total_files += 1
        file_size = file.stat().st_size
        
        # Check file size
        if file_size < 8:
            logger.warning(f"File too small ({file_size} bytes): {file.name}")
            small_files += 1
            removed_size += file_size
            try:
                file.unlink()
                logger.info(f"  Removed: {file.name}")
            except Exception as e:
                logger.error(f"  Failed to remove: {e}")
            continue
        
        # Try to read the file
        try:
            # Quick validation - just read metadata
            pq.read_metadata(file)
            valid_files += 1
        except Exception as e:
            logger.warning(f"Corrupted file: {file.name} - {e}")
            corrupted_files += 1
            removed_size += file_size
            try:
                file.unlink()
                logger.info(f"  Removed: {file.name}")
            except Exception as e2:
                logger.error(f"  Failed to remove: {e2}")
    
    # Clean up manifest
    cache = IndicatorCache(cache_dir)
    
    # Remove manifest entries for non-existent files
    removed_entries = 0
    for cache_key in list(cache.manifest.keys()):
        cache_file = cache_path / f"{cache_key}.parquet"
        if not cache_file.exists():
            del cache.manifest[cache_key]
            removed_entries += 1
    
    if removed_entries > 0:
        cache._save_manifest()
        logger.info(f"Removed {removed_entries} orphaned manifest entries")
    
    # Summary
    logger.info("\nCache Validation Summary:")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Valid files: {valid_files}")
    logger.info(f"  Small files removed: {small_files}")
    logger.info(f"  Corrupted files removed: {corrupted_files}")
    logger.info(f"  Space recovered: {removed_size / 1024:.1f} KB")
    
    # Cache stats
    stats = cache.get_stats()
    logger.info(f"\nCache Statistics:")
    logger.info(f"  Total size: {stats['total_size_mb']:.1f} MB")
    logger.info(f"  Hit rate: {stats['hit_rate']:.1%}")
    logger.info(f"  Total operations: {stats['hits'] + stats['misses']}")


def main():
    """Run cache validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and clean indicator cache')
    parser.add_argument('--cache-dir', default='data_cache/indicators', 
                       help='Cache directory path')
    parser.add_argument('--clear-all', action='store_true',
                       help='Clear entire cache')
    
    args = parser.parse_args()
    
    if args.clear_all:
        response = input("Are you sure you want to clear the entire cache? (y/n): ")
        if response.lower() == 'y':
            cache = IndicatorCache(args.cache_dir)
            cache.clear()
            logger.info("Cache cleared successfully")
        else:
            logger.info("Clear operation cancelled")
    else:
        validate_cache(args.cache_dir)


if __name__ == "__main__":
    main()