#!/usr/bin/env python3
"""
Swift AVWAP Python Wrapper
==========================
High-performance AVWAP calculations using native Swift + Accelerate.
"""

import ctypes
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
import logging

logger = logging.getLogger(__name__)


class SwiftAVWAPCalculator:
    """Python wrapper for Swift AVWAP calculations."""
    
    def __init__(self):
        self.lib = None
        self._load_swift_library()
    
    def _load_swift_library(self):
        """Load the compiled Swift library."""
        # Look for the compiled library
        possible_paths = [
            "./swift_avwap/.build/release/libSwiftAVWAP.dylib",
            "./swift_avwap/.build/debug/libSwiftAVWAP.dylib",
            "./libSwiftAVWAP.dylib",
            "/usr/local/lib/libSwiftAVWAP.dylib"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.lib = ctypes.CDLL(path)
                    self._setup_function_signatures()
                    logger.info(f"✅ Loaded Swift AVWAP library from {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        
        # If no library found, compile it
        logger.info("🔨 Swift library not found, compiling...")
        self._compile_swift_library()
    
    def _compile_swift_library(self):
        """Compile the Swift library."""
        import subprocess
        
        try:
            # Change to swift directory and build
            os.chdir("swift_avwap")
            
            # Build in release mode for maximum performance
            result = subprocess.run([
                "swift", "build", "-c", "release", "--product", "SwiftAVWAP"
            ], capture_output=True, text=True, check=True)
            
            # Load the compiled library
            lib_path = ".build/release/libSwiftAVWAP.dylib"
            if os.path.exists(lib_path):
                self.lib = ctypes.CDLL(lib_path)
                self._setup_function_signatures()
                logger.info("✅ Swift AVWAP library compiled and loaded")
            else:
                raise FileNotFoundError("Compiled library not found")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Swift compilation failed: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Failed to compile Swift library: {e}")
            raise
        finally:
            os.chdir("..")
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for Swift functions."""
        if not self.lib:
            return
        
        # swift_calculate_avwap
        self.lib.swift_calculate_avwap.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # prices
            ctypes.POINTER(ctypes.c_double),  # volumes
            ctypes.POINTER(ctypes.c_int64),   # timestamps
            ctypes.c_int,                     # count
            ctypes.POINTER(ctypes.c_int),     # session_starts
            ctypes.c_int,                     # session_count
            ctypes.POINTER(ctypes.c_double)   # result
        ]
        self.lib.swift_calculate_avwap.restype = None
        
        # swift_calculate_rolling_avwap
        self.lib.swift_calculate_rolling_avwap.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # prices
            ctypes.POINTER(ctypes.c_double),  # volumes
            ctypes.c_int,                     # count
            ctypes.c_int,                     # window
            ctypes.POINTER(ctypes.c_double)   # result
        ]
        self.lib.swift_calculate_rolling_avwap.restype = None
        
        # swift_calculate_avwap_bands
        self.lib.swift_calculate_avwap_bands.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # prices
            ctypes.POINTER(ctypes.c_double),  # avwap
            ctypes.c_int,                     # count
            ctypes.c_int,                     # window
            ctypes.c_double,                  # std_multiplier
            ctypes.POINTER(ctypes.c_double),  # upper_result
            ctypes.POINTER(ctypes.c_double)   # lower_result
        ]
        self.lib.swift_calculate_avwap_bands.restype = None
        
        # swift_calculate_all_avwap_indicators
        self.lib.swift_calculate_all_avwap_indicators.argtypes = [
            ctypes.POINTER(ctypes.c_double),              # prices
            ctypes.POINTER(ctypes.c_double),              # volumes
            ctypes.POINTER(ctypes.c_int64),               # timestamps
            ctypes.c_int,                                 # count
            ctypes.POINTER(ctypes.c_int),                 # session_starts
            ctypes.c_int,                                 # session_count
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), # results
            ctypes.c_int                                  # result_count
        ]
        self.lib.swift_calculate_all_avwap_indicators.restype = ctypes.c_int
    
    def calculate_session_avwap(self, data: pd.DataFrame, session_starts: List[int] = None) -> np.ndarray:
        """Calculate session AVWAP using Swift."""
        if not self.lib:
            raise RuntimeError("Swift library not loaded")
        
        # Prepare data
        prices = data['close'].values.astype(np.float64)
        volumes = data['volume'].values.astype(np.float64)
        timestamps = np.arange(len(data), dtype=np.int64)
        
        # Default session starts (assume 390-minute sessions)
        if session_starts is None:
            session_starts = list(range(0, len(data), 390))
        
        session_starts_array = np.array(session_starts, dtype=np.int32)
        result = np.full(len(data), np.nan, dtype=np.float64)
        
        # Convert to ctypes
        prices_ptr = prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        volumes_ptr = volumes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        timestamps_ptr = timestamps.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        session_starts_ptr = session_starts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call Swift function
        self.lib.swift_calculate_avwap(
            prices_ptr,
            volumes_ptr,
            timestamps_ptr,
            len(data),
            session_starts_ptr,
            len(session_starts),
            result_ptr
        )
        
        return result
    
    def calculate_rolling_avwap(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Calculate rolling AVWAP using Swift."""
        if not self.lib:
            raise RuntimeError("Swift library not loaded")
        
        prices = data['close'].values.astype(np.float64)
        volumes = data['volume'].values.astype(np.float64)
        result = np.full(len(data), np.nan, dtype=np.float64)
        
        # Convert to ctypes
        prices_ptr = prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        volumes_ptr = volumes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call Swift function
        self.lib.swift_calculate_rolling_avwap(
            prices_ptr,
            volumes_ptr,
            len(data),
            window,
            result_ptr
        )
        
        return result
    
    def calculate_avwap_bands(self, data: pd.DataFrame, avwap: np.ndarray, 
                             window: int = 20, std_multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate AVWAP bands using Swift."""
        if not self.lib:
            raise RuntimeError("Swift library not loaded")
        
        prices = data['close'].values.astype(np.float64)
        avwap_array = avwap.astype(np.float64)
        upper_result = np.full(len(data), np.nan, dtype=np.float64)
        lower_result = np.full(len(data), np.nan, dtype=np.float64)
        
        # Convert to ctypes
        prices_ptr = prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        avwap_ptr = avwap_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        upper_ptr = upper_result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lower_ptr = lower_result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call Swift function
        self.lib.swift_calculate_avwap_bands(
            prices_ptr,
            avwap_ptr,
            len(data),
            window,
            std_multiplier,
            upper_ptr,
            lower_ptr
        )
        
        return upper_result, lower_result
    
    def calculate_all_avwap_indicators(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate all AVWAP indicators in one Swift call."""
        if not self.lib:
            raise RuntimeError("Swift library not loaded")
        
        # Prepare data
        prices = data['close'].values.astype(np.float64)
        volumes = data['volume'].values.astype(np.float64)
        timestamps = np.arange(len(data), dtype=np.int64)
        
        # Session starts (390-minute sessions)
        session_starts = list(range(0, len(data), 390))
        session_starts_array = np.array(session_starts, dtype=np.int32)
        
        # Prepare result arrays
        num_indicators = 13
        results = []
        result_ptrs = []
        
        for _ in range(num_indicators):
            result_array = np.full(len(data), np.nan, dtype=np.float64)
            results.append(result_array)
            result_ptrs.append(result_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        
        # Create array of pointers
        results_array = (ctypes.POINTER(ctypes.c_double) * num_indicators)(*result_ptrs)
        
        # Convert to ctypes
        prices_ptr = prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        volumes_ptr = volumes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        timestamps_ptr = timestamps.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        session_starts_ptr = session_starts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        
        # Call Swift function
        indicators_calculated = self.lib.swift_calculate_all_avwap_indicators(
            prices_ptr,
            volumes_ptr,
            timestamps_ptr,
            len(data),
            session_starts_ptr,
            len(session_starts),
            results_array,
            num_indicators
        )
        
        if indicators_calculated != num_indicators:
            logger.warning(f"Expected {num_indicators} indicators, got {indicators_calculated}")
        
        # Return as dictionary with proper names
        indicator_names = [
            'AVWAP_SESSION',
            'AVWAP_ROLLING_50',
            'AVWAP_ROLLING_100', 
            'AVWAP_ROLLING_200',
            'AVWAP_SESSION_U1',
            'AVWAP_SESSION_L1',
            'AVWAP_SESSION_U2',
            'AVWAP_SESSION_L2',
            'AVWAP_SESSION_U3',
            'AVWAP_SESSION_L3',
            'AVWAP_DAILY',
            'AVWAP_WEEKLY',
            'AVWAP_MONTHLY'
        ]
        
        return {name: results[i] for i, name in enumerate(indicator_names)}


def create_swift_avwap_calculator():
    """Factory function to create Swift AVWAP calculator."""
    return SwiftAVWAPCalculator()


# Benchmark function
def benchmark_swift_vs_python():
    """Compare Swift vs Python AVWAP performance."""
    import time
    
    # Create sample data
    n_bars = 5000
    data = pd.DataFrame({
        'close': np.random.randn(n_bars).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, n_bars)
    })
    
    # Test Swift version
    swift_calc = SwiftAVWAPCalculator()
    
    start = time.time()
    swift_results = swift_calc.calculate_all_avwap_indicators(data)
    swift_time = time.time() - start
    
    print(f"\n🚀 Swift AVWAP Performance:")
    print(f"   • Time: {swift_time:.3f}s")
    print(f"   • Bars processed: {n_bars}")
    print(f"   • Indicators: {len(swift_results)}")
    print(f"   • Throughput: {n_bars / swift_time:.0f} bars/sec")
    
    # Test Python version (simplified)
    start = time.time()
    
    # Simple Python VWAP for comparison
    pv = data['close'] * data['volume']
    cumulative_pv = pv.cumsum()
    cumulative_volume = data['volume'].cumsum()
    python_vwap = cumulative_pv / cumulative_volume
    
    python_time = time.time() - start
    
    print(f"\n🐍 Python VWAP Performance:")
    print(f"   • Time: {python_time:.3f}s")
    print(f"   • Speedup: {python_time / swift_time:.1f}x faster with Swift")


if __name__ == "__main__":
    benchmark_swift_vs_python()