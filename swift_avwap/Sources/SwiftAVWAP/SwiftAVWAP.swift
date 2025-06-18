import Foundation
import Accelerate

@_cdecl("swift_calculate_avwap")
public func swift_calculate_avwap(
    prices: UnsafePointer<Double>,
    volumes: UnsafePointer<Double>,
    timestamps: UnsafePointer<Int64>,
    count: Int,
    session_starts: UnsafePointer<Int>,
    session_count: Int,
    result: UnsafeMutablePointer<Double>
) {
    // Use Accelerate framework for vectorized operations
    calculateSessionAVWAP(
        prices: prices,
        volumes: volumes,
        timestamps: timestamps,
        count: count,
        sessionStarts: session_starts,
        sessionCount: session_count,
        result: result
    )
}

@_cdecl("swift_calculate_rolling_avwap")
public func swift_calculate_rolling_avwap(
    prices: UnsafePointer<Double>,
    volumes: UnsafePointer<Double>,
    count: Int,
    window: Int,
    result: UnsafeMutablePointer<Double>
) {
    calculateRollingAVWAP(
        prices: prices,
        volumes: volumes,
        count: count,
        window: window,
        result: result
    )
}

@_cdecl("swift_calculate_avwap_bands")
public func swift_calculate_avwap_bands(
    prices: UnsafePointer<Double>,
    avwap: UnsafePointer<Double>,
    count: Int,
    window: Int,
    std_multiplier: Double,
    upper_result: UnsafeMutablePointer<Double>,
    lower_result: UnsafeMutablePointer<Double>
) {
    calculateAVWAPBands(
        prices: prices,
        avwap: avwap,
        count: count,
        window: window,
        stdMultiplier: std_multiplier,
        upperResult: upper_result,
        lowerResult: lower_result
    )
}

// MARK: - Implementation Functions

private func calculateSessionAVWAP(
    prices: UnsafePointer<Double>,
    volumes: UnsafePointer<Double>,
    timestamps: UnsafePointer<Int64>,
    count: Int,
    sessionStarts: UnsafePointer<Int>,
    sessionCount: Int,
    result: UnsafeMutablePointer<Double>
) {
    // Initialize result array with NaN
    for i in 0..<count {
        result[i] = Double.nan
    }
    
    // Process each session
    for sessionIdx in 0..<sessionCount {
        let startIdx = Int(sessionStarts[sessionIdx])
        let endIdx = sessionIdx < sessionCount - 1 ? Int(sessionStarts[sessionIdx + 1]) : count
        
        guard startIdx < count && endIdx <= count && startIdx < endIdx else { continue }
        
        // Calculate cumulative PV and volume for this session
        var cumulativePV = 0.0
        var cumulativeVolume = 0.0
        
        for i in startIdx..<endIdx {
            let pv = prices[i] * volumes[i]
            cumulativePV += pv
            cumulativeVolume += volumes[i]
            
            // AVWAP = cumulative PV / cumulative volume
            if cumulativeVolume > 0 {
                result[i] = cumulativePV / cumulativeVolume
            }
        }
    }
}

private func calculateRollingAVWAP(
    prices: UnsafePointer<Double>,
    volumes: UnsafePointer<Double>,
    count: Int,
    window: Int,
    result: UnsafeMutablePointer<Double>
) {
    // Initialize with NaN
    for i in 0..<min(window, count) {
        result[i] = Double.nan
    }
    
    // Use Accelerate for vectorized operations
    var pvArray = [Double](repeating: 0.0, count: count)
    
    // Calculate price * volume array
    vDSP_vmulD(prices, 1, volumes, 1, &pvArray, 1, vDSP_Length(count))
    
    // Rolling window calculation
    for i in window..<count {
        let startIdx = i - window + 1
        
        // Sum PV and volume for window
        var windowPV = 0.0
        var windowVolume = 0.0
        
        pvArray.withUnsafeBufferPointer { buffer in
            vDSP_sveD(buffer.baseAddress! + startIdx, 1, &windowPV, vDSP_Length(window))
        }
        vDSP_sveD(volumes + startIdx, 1, &windowVolume, vDSP_Length(window))
        
        if windowVolume > 0 {
            result[i] = windowPV / windowVolume
        } else {
            result[i] = Double.nan
        }
    }
}

private func calculateAVWAPBands(
    prices: UnsafePointer<Double>,
    avwap: UnsafePointer<Double>,
    count: Int,
    window: Int,
    stdMultiplier: Double,
    upperResult: UnsafeMutablePointer<Double>,
    lowerResult: UnsafeMutablePointer<Double>
) {
    // Initialize with NaN
    for i in 0..<min(window, count) {
        upperResult[i] = Double.nan
        lowerResult[i] = Double.nan
    }
    
    // Calculate rolling standard deviation of (price - avwap)
    var diffArray = [Double](repeating: 0.0, count: window)
    var diffSquaredArray = [Double](repeating: 0.0, count: window)
    
    for i in window..<count {
        let startIdx = i - window + 1
        
        // Calculate differences for this window
        for j in 0..<window {
            let idx = startIdx + j
            if !avwap[idx].isNaN {
                diffArray[j] = prices[idx] - avwap[idx]
                diffSquaredArray[j] = diffArray[j] * diffArray[j]
            } else {
                diffArray[j] = 0.0
                diffSquaredArray[j] = 0.0
            }
        }
        
        // Calculate standard deviation using Accelerate
        var mean = 0.0
        var meanSquared = 0.0
        
        vDSP_meanvD(diffArray, 1, &mean, vDSP_Length(window))
        vDSP_meanvD(diffSquaredArray, 1, &meanSquared, vDSP_Length(window))
        
        let variance = meanSquared - (mean * mean)
        let std = sqrt(max(variance, 0.0))
        
        if !avwap[i].isNaN {
            upperResult[i] = avwap[i] + (stdMultiplier * std)
            lowerResult[i] = avwap[i] - (stdMultiplier * std)
        } else {
            upperResult[i] = Double.nan
            lowerResult[i] = Double.nan
        }
    }
}

// MARK: - Batch Processing Functions

@_cdecl("swift_calculate_all_avwap_indicators")
public func swift_calculate_all_avwap_indicators(
    prices: UnsafePointer<Double>,
    volumes: UnsafePointer<Double>,
    timestamps: UnsafePointer<Int64>,
    count: Int,
    session_starts: UnsafePointer<Int>,
    session_count: Int,
    results: UnsafeMutablePointer<UnsafeMutablePointer<Double>>,
    result_count: Int
) -> Int {
    guard result_count >= 13 else { return -1 } // Need at least 13 result arrays
    
    // Session AVWAP
    swift_calculate_avwap(
        prices: prices,
        volumes: volumes,
        timestamps: timestamps,
        count: count,
        session_starts: session_starts,
        session_count: session_count,
        result: results[0]
    )
    
    // Rolling AVWAP variants
    let rollingPeriods = [50, 100, 200]
    for (idx, period) in rollingPeriods.enumerated() {
        swift_calculate_rolling_avwap(
            prices: prices,
            volumes: volumes,
            count: count,
            window: period,
            result: results[1 + idx]
        )
    }
    
    // AVWAP bands (1, 2, 3 standard deviations)
    let stdMultipliers = [1.0, 2.0, 3.0]
    for (idx, multiplier) in stdMultipliers.enumerated() {
        // Upper bands
        swift_calculate_avwap_bands(
            prices: prices,
            avwap: results[0], // Session AVWAP
            count: count,
            window: 20,
            std_multiplier: multiplier,
            upper_result: results[4 + idx * 2],
            lower_result: results[5 + idx * 2]
        )
    }
    
    // Daily AVWAP (simplified - assume 390 minute sessions)
    let sessionSize = 390
    var dailyStarts = [Int]()
    for i in stride(from: 0, to: count, by: sessionSize) {
        dailyStarts.append(i)
    }
    
    dailyStarts.withUnsafeBufferPointer { buffer in
        swift_calculate_avwap(
            prices: prices,
            volumes: volumes,
            timestamps: timestamps,
            count: count,
            session_starts: buffer.baseAddress!,
            session_count: dailyStarts.count,
            result: results[10]
        )
    }
    
    // Weekly AVWAP (5 days = 5 * 390 minutes)
    let weeklySessionSize = sessionSize * 5
    var weeklyStarts = [Int]()
    for i in stride(from: 0, to: count, by: weeklySessionSize) {
        weeklyStarts.append(i)
    }
    
    weeklyStarts.withUnsafeBufferPointer { buffer in
        swift_calculate_avwap(
            prices: prices,
            volumes: volumes,
            timestamps: timestamps,
            count: count,
            session_starts: buffer.baseAddress!,
            session_count: weeklyStarts.count,
            result: results[11]
        )
    }
    
    // Monthly AVWAP (20 days = 20 * 390 minutes)
    let monthlySessionSize = sessionSize * 20
    var monthlyStarts = [Int]()
    for i in stride(from: 0, to: count, by: monthlySessionSize) {
        monthlyStarts.append(i)
    }
    
    monthlyStarts.withUnsafeBufferPointer { buffer in
        swift_calculate_avwap(
            prices: prices,
            volumes: volumes,
            timestamps: timestamps,
            count: count,
            session_starts: buffer.baseAddress!,
            session_count: monthlyStarts.count,
            result: results[12]
        )
    }
    
    return 13 // Number of indicators calculated
}