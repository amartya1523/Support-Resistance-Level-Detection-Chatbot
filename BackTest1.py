import pandas as pd
import numpy as np
import requests
from scipy.signal import find_peaks
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

def get_klines(symbol: str, interval: str, limit: int = 1000):
    """
    Fetch historical klines (candlestick data) for a given symbol and interval from Binance API
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        return df.tail(limit)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Binance API: {e}")
        return None

def get_klines_historical(symbol: str, interval: str, start_time: str, end_time: str):
    """
    Fetch historical klines for a specific time period from Binance API
    """
    url = "https://api.binance.com/api/v3/klines"
    
    # Convert string dates to timestamps
    start_timestamp = int(datetime.strptime(start_time, "%Y-%m-%d").timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_time, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    current_start = start_timestamp
    
    while current_start < end_timestamp:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start,
            "endTime": min(current_start + (1000 * interval_to_milliseconds(interval) * 1000), end_timestamp), # Adjusted for 1000 candles
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            current_start = data[-1][0] + interval_to_milliseconds(interval)
            
            # Rate limiting
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    return df

def interval_to_milliseconds(interval: str) -> int:
    """Convert interval string to milliseconds"""
    unit = interval[-1]
    value = int(interval[:-1])
    
    if unit == 'm':
        return value * 60 * 1000
    elif unit == 'h':
        return value * 60 * 60 * 1000
    elif unit == 'd':
        return value * 24 * 60 * 60 * 1000
    elif unit == 'w':
        return value * 7 * 24 * 60 * 60 * 1000
    elif unit == 'M':
        return value * 30 * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Unsupported interval: {interval}")

def interval_to_timedelta(interval_str: str) -> timedelta:
    """
    Converts a Binance interval string to a timedelta object.
    """
    unit = interval_str[-1]
    value = int(interval_str[:-1])
    if unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    elif unit == 'w':
        return timedelta(weeks=value)
    elif unit == 'M': # Binance '1M' is 1 month, approx 30 days
        return timedelta(days=value * 30) # Approximation for month
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")

def count_touches(df: pd.DataFrame, level: float, price_type: str, tolerance_percent: float = 0.005):
    """
    Counts how many times the price 'touches' a given level and calculates the span in days.
    A 'touch' is defined as the price (high for resistance, low for support)
    coming within a certain percentage tolerance of the level.
    """
    touches = 0
    tolerance = level * tolerance_percent
    touch_timestamps = []

    if price_type == 'resistance':
        for index, high_price in df['high'].items():
            if level - tolerance <= high_price <= level + tolerance:
                touches += 1
                touch_timestamps.append(index)
    elif price_type == 'support':
        for index, low_price in df['low'].items():
            if level - tolerance <= low_price <= level + tolerance:
                touches += 1
                touch_timestamps.append(index)

    days_span = 0
    if len(touch_timestamps) > 1:
        time_diff = touch_timestamps[-1] - touch_timestamps[0]
        days_span = time_diff.days

    return touches, days_span

def calculate_levels(df: pd.DataFrame, level_type: str, num_candles: int, min_deviation_percent: float = 0.005, timeframe_type: str = 'general'):
    """
    Calculates support or resistance levels for a given DataFrame slice.
    Returns a list of dictionaries with 'level', 'touches', 'days_span', and 'type'.
    Applies deduplication based on min_deviation_percent.
    """
    if df.empty:
        return []

    # Dynamic parameters based on num_candles and timeframe_type
    strong_peak_distance = max(1, num_candles // 10)
    general_peak_distance = max(1, num_candles // 20)
    
    price_range = df['high'].max() - df['low'].min()

    if timeframe_type == 'short':
        strong_peak_prominence = 0.005 * price_range # More sensitive for short-term
        peak_rank_width = 0.002 * price_range # More sensitive for short-term
        min_pivot_rank = 1 # More lenient for short-term
    else: # mid and long
        strong_peak_prominence = 0.01 * price_range
        peak_rank_width = 0.005 * price_range
        min_pivot_rank = 2

    levels = []

    if level_type == 'resistance':
        price_data = df['high']
        # Find strong peaks
        strong_indices, _ = find_peaks(price_data, distance=strong_peak_distance, prominence=strong_peak_prominence)
        for idx in strong_indices:
            level = price_data.iloc[idx]
            touches, days_span = count_touches(df, level, 'resistance')
            levels.append({'level': level, 'touches': touches, 'days_span': days_span, 'type': 'STRONG 💪'})

        # Find general peaks
        general_indices, _ = find_peaks(price_data, distance=general_peak_distance)
        peak_to_rank = {idx: 0 for idx in general_indices}
        for i, current_peak_idx in enumerate(general_indices):
            current_level = price_data.iloc[current_peak_idx]
            for previous_peak_idx in general_indices[:i]:
                if abs(current_level - price_data.iloc[previous_peak_idx]) <= peak_rank_width:
                    peak_to_rank[current_peak_idx] += 1

        for peak_idx, rank in peak_to_rank.items():
            if rank >= min_pivot_rank:
                level = price_data.iloc[peak_idx]
                touches, days_span = count_touches(df, level, 'resistance')
                levels.append({'level': level, 'touches': touches, 'days_span': days_span, 'type': 'GENERAL'})
        
        # Sort and remove duplicates, prioritizing more touches for same level
        unique_levels = {}
        for l in levels:
            if l['level'] not in unique_levels or l['touches'] > unique_levels[l['level']]['touches']:
                unique_levels[l['level']] = l
        levels = sorted(list(unique_levels.values()), key=lambda x: x['level'], reverse=True)

        # Apply deduplication based on min_deviation_percent
        filtered_levels = []
        if levels:
            filtered_levels.append(levels[0])
            for i in range(1, len(levels)):
                if abs((levels[i]['level'] - filtered_levels[-1]['level']) / filtered_levels[-1]['level']) >= min_deviation_percent:
                    filtered_levels.append(levels[i])
        levels = filtered_levels
        
    elif level_type == 'support':
        price_data = -df['low'] # Invert for finding peaks (troughs)
        # Find strong troughs
        strong_indices, _ = find_peaks(price_data, distance=strong_peak_distance, prominence=strong_peak_prominence)
        for idx in strong_indices:
            level = -price_data.iloc[idx] # Invert back to original price
            touches, days_span = count_touches(df, level, 'support')
            levels.append({'level': level, 'touches': touches, 'days_span': days_span, 'type': 'STRONG 💪'})

        # Find general troughs
        general_indices, _ = find_peaks(price_data, distance=general_peak_distance)
        trough_to_rank = {idx: 0 for idx in general_indices}
        for i, current_trough_idx in enumerate(general_indices):
            current_level = price_data.iloc[current_trough_idx]
            for previous_trough_idx in general_indices[:i]:
                if abs(current_level - price_data.iloc[previous_trough_idx]) <= peak_rank_width:
                    trough_to_rank[current_trough_idx] += 1

        for trough_idx, rank in trough_to_rank.items():
            if rank >= min_pivot_rank:
                level = -price_data.iloc[trough_idx] # Invert back to original price
                touches, days_span = count_touches(df, level, 'support')
                levels.append({'level': level, 'touches': touches, 'days_span': days_span, 'type': 'GENERAL'})
        
        # Sort and remove duplicates, prioritizing more touches for same level
        unique_levels = {}
        for l in levels:
            if l['level'] not in unique_levels or l['touches'] > unique_levels[l['level']]['touches']:
                unique_levels[l['level']] = l
        levels = sorted(list(unique_levels.values()), key=lambda x: x['level'], reverse=True) # Changed to reverse=True

        # Apply deduplication based on min_deviation_percent
        filtered_levels = []
        if levels:
            filtered_levels.append(levels[0])
            for i in range(1, len(levels)):
                if abs((levels[i]['level'] - filtered_levels[-1]['level']) / filtered_levels[-1]['level']) >= min_deviation_percent:
                    filtered_levels.append(levels[i])
        levels = filtered_levels
        
    return levels

def get_sr_levels_for_period(df: pd.DataFrame, interval: str, analysis_window: int = 100):
    """Get support and resistance levels for a given period"""
    if len(df) < analysis_window:
        analysis_window = len(df)
    
    analysis_df = df.tail(analysis_window)
    
    # Calculate candle counts for different timeframes
    candle_duration = interval_to_timedelta(interval)
    short_term_candles = int(7 * (timedelta(days=1) / candle_duration))
    mid_term_candles = int(30 * (timedelta(days=1) / candle_duration))
    long_term_candles = int(90 * (timedelta(days=1) / candle_duration))
    
    short_term_candles = min(short_term_candles, len(analysis_df))
    mid_term_candles = min(mid_term_candles, len(analysis_df))
    long_term_candles = min(long_term_candles, len(analysis_df))
    
    # Get levels for each timeframe
    short_term_df = analysis_df.tail(short_term_candles)
    mid_term_df = analysis_df.tail(mid_term_candles)
    long_term_df = analysis_df.tail(long_term_candles)
    
    results = {
        'short_term': {
            'supports': calculate_levels(short_term_df, 'support', len(short_term_df), timeframe_type='short'),
            'resistances': calculate_levels(short_term_df, 'resistance', len(short_term_df), timeframe_type='short')
        },
        'mid_term': {
            'supports': calculate_levels(mid_term_df, 'support', len(mid_term_df), timeframe_type='mid'),
            'resistances': calculate_levels(mid_term_df, 'resistance', len(mid_term_df), timeframe_type='mid')
        },
        'long_term': {
            'supports': calculate_levels(long_term_df, 'support', len(long_term_df), timeframe_type='long'),
            'resistances': calculate_levels(long_term_df, 'resistance', len(long_term_df), timeframe_type='long')
        }
    }
    
    return results

def test_level_effectiveness(df: pd.DataFrame, level: float, level_type: str, tolerance_percent: float = 0.005) -> Dict:
    """Test how effective a support/resistance level is over a test period"""
    tolerance = level * tolerance_percent
    
    results = {
        'bounces': 0,
        'breaks': 0,
        'touches': 0,
        'effectiveness_ratio': 0.0,
        'max_deviation': 0.0,
        'avg_deviation': 0.0
    }
    
    deviations = []
    
    for i, (timestamp, row) in enumerate(df.iterrows()):
        if level_type == 'support':
            # Check if price touched the support level
            if level - tolerance <= row['low'] <= level + tolerance:
                results['touches'] += 1
                
                # Check if it bounced (next few candles show upward movement)
                if i < len(df) - 3:
                    future_closes = df.iloc[i+1:i+4]['close'].values
                    if any(close > row['close'] * 1.01 for close in future_closes):  # 1% bounce threshold
                        results['bounces'] += 1
                    else:
                        results['breaks'] += 1
                        
            # Track deviation from support level
            if row['low'] < level:
                deviation = abs(row['low'] - level) / level
                deviations.append(deviation)
                
        elif level_type == 'resistance':
            # Check if price touched the resistance level
            if level - tolerance <= row['high'] <= level + tolerance:
                results['touches'] += 1
                
                # Check if it bounced (next few candles show downward movement)
                if i < len(df) - 3:
                    future_closes = df.iloc[i+1:i+4]['close'].values
                    if any(close < row['close'] * 0.99 for close in future_closes):  # 1% bounce threshold
                        results['bounces'] += 1
                    else:
                        results['breaks'] += 1
                        
            # Track deviation from resistance level
            if row['high'] > level:
                deviation = abs(row['high'] - level) / level
                deviations.append(deviation)
    
    if results['touches'] > 0:
        results['effectiveness_ratio'] = results['bounces'] / results['touches']
    
    if deviations:
        results['max_deviation'] = max(deviations)
        results['avg_deviation'] = np.mean(deviations)
    
    return results

def run_backtest(symbol: str, interval: str, start_date: str, end_date: str, analysis_window: int = 100, test_window_days: int = 7) -> Dict:
    """Run comprehensive backtest on support/resistance levels"""
    print(f"Starting backtest for {symbol} on {interval} timeframe")
    print(f"Period: {start_date} to {end_date}")
    print(f"Analysis window: {analysis_window} candles, Test window: {test_window_days} days")
    
    # Fetch historical data
    df = get_klines_historical(symbol, interval, start_date, end_date)
    if df is None or df.empty:
        print("Failed to fetch historical data")
        return {}
    
    print(f"Fetched {len(df)} candles")
    
    # Calculate test_candles based on test_window_days and interval
    interval_td = interval_to_timedelta(interval)
    test_candles_per_day = timedelta(days=1) / interval_td
    test_candles = int(test_window_days * test_candles_per_day)

    results = {
        'symbol': symbol,
        'interval': interval,
        'start_date': start_date,
        'end_date': end_date,
        'total_candles': len(df),
        'analysis_window': analysis_window,
        'test_window': test_window_days,
        'test_points': [], # Store test points with S/R levels
        'overall_stats': {}
    }
    
    # Run tests for different time periods
    test_points = []
    step_size = max(1, analysis_window // 4)  # Test every quarter of analysis window
    
    for i in range(analysis_window, len(df) - test_candles, step_size):
        analysis_data = df.iloc[:i]
        test_data = df.iloc[i:i+test_candles]
        
        if len(test_data) < test_candles // 2:  # Skip if not enough test data
            continue
            
        # Get S/R levels using the existing accurate logic
        sr_levels = get_sr_levels_for_period(analysis_data, interval, analysis_window)
        
        test_point = {
            'timestamp': df.index[i],
            'price': df.iloc[i]['close'],
            'timeframes': sr_levels # Store the calculated S/R levels directly
        }
        
        # Test each timeframe
        for timeframe in ['short_term', 'mid_term', 'long_term']:
            # Test support levels
            for support in sr_levels[timeframe]['supports']:
                # Only test levels below current price for support
                if support['level'] < test_point['price']:
                    effectiveness = test_level_effectiveness(test_data, support['level'], 'support')
                    support['effectiveness'] = effectiveness # Add effectiveness to the level dict
            
            # Test resistance levels
            for resistance in sr_levels[timeframe]['resistances']:
                # Only test levels above current price for resistance
                if resistance['level'] > test_point['price']:
                    effectiveness = test_level_effectiveness(test_data, resistance['level'], 'resistance')
                    resistance['effectiveness'] = effectiveness # Add effectiveness to the level dict
            
        test_points.append(test_point)
        
        if len(test_points) % 10 == 0:
            print(f"Processed {len(test_points)} test points...")
    
    results['test_points'] = test_points
    
    # Calculate overall statistics
    overall_stats = calculate_overall_stats(test_points)
    results['overall_stats'] = overall_stats
    
    print(f"Backtest completed. Processed {len(test_points)} test points.")
    return results

def calculate_overall_stats(test_points: List[Dict]) -> Dict:
    """Calculate overall statistics from test points"""
    stats = {
        'total_tests': 0,
        'timeframe_stats': {}
    }
    
    for timeframe in ['short_term', 'mid_term', 'long_term']:
        timeframe_stats = {
            'support_stats': {'total_levels': 0, 'total_touches': 0, 'total_bounces': 0, 'avg_effectiveness': 0.0},
            'resistance_stats': {'total_levels': 0, 'total_touches': 0, 'total_bounces': 0, 'avg_effectiveness': 0.0}
        }
        
        support_effectiveness = []
        resistance_effectiveness = []
        
        for test_point in test_points:
            if timeframe in test_point['timeframes']:
                # Support stats
                for support in test_point['timeframes'][timeframe]['supports']:
                    if 'effectiveness' in support: # Only count if tested
                        timeframe_stats['support_stats']['total_levels'] += 1
                        timeframe_stats['support_stats']['total_touches'] += support['effectiveness']['touches']
                        timeframe_stats['support_stats']['total_bounces'] += support['effectiveness']['bounces']
                        if support['effectiveness']['effectiveness_ratio'] > 0:
                            support_effectiveness.append(support['effectiveness']['effectiveness_ratio'])
                
                # Resistance stats
                for resistance in test_point['timeframes'][timeframe]['resistances']:
                    if 'effectiveness' in resistance: # Only count if tested
                        timeframe_stats['resistance_stats']['total_levels'] += 1
                        timeframe_stats['resistance_stats']['total_touches'] += resistance['effectiveness']['touches']
                        timeframe_stats['resistance_stats']['total_bounces'] += resistance['effectiveness']['bounces']
                        if resistance['effectiveness']['effectiveness_ratio'] > 0:
                            resistance_effectiveness.append(resistance['effectiveness']['effectiveness_ratio'])
        
        # Calculate averages
        if support_effectiveness:
            timeframe_stats['support_stats']['avg_effectiveness'] = np.mean(support_effectiveness)
        if resistance_effectiveness:
            timeframe_stats['resistance_stats']['avg_effectiveness'] = np.mean(resistance_effectiveness)
        
        stats['timeframe_stats'][timeframe] = timeframe_stats
    
    stats['total_tests'] = len(test_points)
    return stats

def generate_backtest_report(results: Dict, save_path: str = None):
    """Generate a comprehensive backtest report"""
    report = f"""
=== SUPPORT & RESISTANCE BACKTEST REPORT ===

Symbol: {results['symbol']}
Interval: {results['interval']}
Period: {results['start_date']} to {results['end_date']}
Total Candles: {results['total_candles']}
Analysis Window: {results['analysis_window']} candles
Test Window: {results['test_window']} days
Total Test Points: {results['overall_stats']['total_tests']}

=== OVERALL PERFORMANCE ===
"""
    
    for timeframe, stats in results['overall_stats']['timeframe_stats'].items():
        report += f"\n{timeframe.upper().replace('_', ' ')} ANALYSIS:\n"
        report += f"  Support Levels:\n"
        report += f"    Total Levels Tested: {stats['support_stats']['total_levels']}\n"
        report += f"    Total Touches: {stats['support_stats']['total_touches']}\n"
        report += f"    Total Bounces: {stats['support_stats']['total_bounces']}\n"
        report += f"    Average Effectiveness: {stats['support_stats']['avg_effectiveness']:.2%}\n"
        
        report += f"  Resistance Levels:\n"
        report += f"    Total Levels Tested: {stats['resistance_stats']['total_levels']}\n"
        report += f"    Total Touches: {stats['resistance_stats']['total_touches']}\n"
        report += f"    Total Bounces: {stats['resistance_stats']['total_bounces']}\n"
        report += f"    Average Effectiveness: {stats['resistance_stats']['avg_effectiveness']:.2%}\n"

    report += "\n=== DETAILED TEST POINTS ===\n"
    for i, tp in enumerate(results['test_points']):
        report += f"\n--- Test Point {i+1} ---\n"
        report += f"Timestamp: {tp['timestamp']}\n"
        report += f"Price: {tp['price']:.8f}\n"
        for timeframe, sr_levels in tp['timeframes'].items():
            report += f"  {timeframe.upper().replace('_', ' ')}:\n"
            report += f"    Resistances:\n"
            if sr_levels['resistances']:
                for r in sr_levels['resistances']:
                    report += f"      - Level: {r['level']:.8f}"
                    if 'effectiveness' in r:
                        report += f" (Touches: {r['effectiveness']['touches']}, Bounces: {r['effectiveness']['bounces']}, Effectiveness: {r['effectiveness']['effectiveness_ratio']:.2%})\n"
                    else:
                        report += "\n"
            else:
                report += "      No resistances found.\n"
            
            report += f"    Supports:\n"
            if sr_levels['supports']:
                for s in sr_levels['supports']:
                    report += f"      - Level: {s['level']:.8f}"
                    if 'effectiveness' in s:
                        report += f" (Touches: {s['effectiveness']['touches']}, Bounces: {s['effectiveness']['bounces']}, Effectiveness: {s['effectiveness']['effectiveness_ratio']:.2%})\n"
                    else:
                        report += "\n"
            else:
                report += "      No supports found.\n"
    
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report

def create_backtest_visualizations(results: Dict):
    """Create visualizations for backtest results"""
    try:
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Backtest Results: {results['symbol']} ({results['interval']})", fontsize=16)
        
        timeframes = ['short_term', 'mid_term', 'long_term']
        colors = ['blue', 'orange', 'green']
        
        # Plot 1: Support Effectiveness by Timeframe
        support_effectiveness = []
        for timeframe in timeframes:
            stats = results['overall_stats']['timeframe_stats'][timeframe]['support_stats']
            support_effectiveness.append(stats['avg_effectiveness'])
        
        axes[0, 0].bar(timeframes, support_effectiveness, color=colors, alpha=0.7)
        axes[0, 0].set_title('Support Level Effectiveness by Timeframe')
        axes[0, 0].set_ylabel('Average Effectiveness Ratio')
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Resistance Effectiveness by Timeframe
        resistance_effectiveness = []
        for timeframe in timeframes:
            stats = results['overall_stats']['timeframe_stats'][timeframe]['resistance_stats']
            resistance_effectiveness.append(stats['avg_effectiveness'])
        
        axes[0, 1].bar(timeframes, resistance_effectiveness, color=colors, alpha=0.7)
        axes[0, 1].set_title('Resistance Level Effectiveness by Timeframe')
        axes[0, 1].set_ylabel('Average Effectiveness Ratio')
        axes[0, 1].set_ylim(0, 1)
        
        # Plot 3: Total Touches by Timeframe
        support_touches = []
        resistance_touches = []
        for timeframe in timeframes:
            support_stats = results['overall_stats']['timeframe_stats'][timeframe]['support_stats']
            resistance_stats = results['overall_stats']['timeframe_stats'][timeframe]['resistance_stats']
            support_touches.append(support_stats['total_touches'])
            resistance_touches.append(resistance_stats['total_touches'])
        
        x = np.arange(len(timeframes))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, support_touches, width, label='Support', alpha=0.7)
        axes[1, 0].bar(x + width/2, resistance_touches, width, label='Resistance', alpha=0.7)
        axes[1, 0].set_title('Total Touches by Timeframe')
        axes[1, 0].set_ylabel('Number of Touches')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(timeframes)
        axes[1, 0].legend()
        
        # Plot 4: Bounce Rate by Timeframe
        support_bounce_rates = []
        resistance_bounce_rates = []
        for timeframe in timeframes:
            support_stats = results['overall_stats']['timeframe_stats'][timeframe]['support_stats']
            resistance_stats = results['overall_stats']['timeframe_stats'][timeframe]['resistance_stats']
            
            support_rate = support_stats['total_bounces'] / max(support_stats['total_touches'], 1)
            resistance_rate = resistance_stats['total_bounces'] / max(resistance_stats['total_touches'], 1)
            
            support_bounce_rates.append(support_rate)
            resistance_bounce_rates.append(resistance_rate)
        
        axes[1, 1].bar(x - width/2, support_bounce_rates, width, label='Support', alpha=0.7)
        axes[1, 1].bar(x + width/2, resistance_bounce_rates, width, label='Resistance', alpha=0.7)
        axes[1, 1].set_title('Bounce Rate by Timeframe')
        axes[1, 1].set_ylabel('Bounce Rate')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(timeframes)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"backtest_visualization_{results['symbol']}_{results['interval']}_{results['start_date']}_{results['end_date']}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {plot_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    print("\n===== Crypto Support & Resistance Bot ====")
    print("1. Run S/R Analysis")
    print("2. Run Backtest")
    print("========================================")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        while True:
            currency_pair = input("Enter cryptocurrency pair (e.g., XRPUSDT, BTCUSDT) or 'exit' to quit: ").upper()
            if currency_pair == 'EXIT':
                break

            interval = input("Enter interval ['30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']: ")
            num_candles_input = int(input("How many candles of data? (minimum 100, max 1000): "))

            if num_candles_input < 100 or num_candles_input > 1000:
                print("Please enter a number of candles between 100 and 1000.")
                continue

            print(f"Fetching {num_candles_input} candles for {currency_pair} on {interval} interval...")
            start_time = time.time()
            df = get_klines(currency_pair, interval, num_candles_input)
            end_time = time.time()

            if df is None or df.empty:
                print("Could not fetch data or data is empty. Please check the currency pair and interval.")
                continue

            print(f"Successfully fetched data for {currency_pair} from Binance API.")
            print(f"Analysis completed in {end_time - start_time:.2f} seconds.")

            print(f"\n:: Support/Resistance Analysis for {currency_pair} on {interval} timeframe ::")
            current_price = df['close'].iloc[-1]
            print(f"📈 Current Price: ${current_price:.8f}\n")

            # Calculate candle counts based on timeframes and interval
            candle_duration = interval_to_timedelta(interval)
            
            short_term_days = 7
            mid_term_days = 30
            long_term_days = 90

            short_term_candles_count = int(short_term_days * (timedelta(days=1) / candle_duration))
            mid_term_candles_count = int(mid_term_days * (timedelta(days=1) / candle_duration))
            long_term_candles_count = int(long_term_days * (timedelta(days=1) / candle_duration))

            # Ensure calculated candle counts do not exceed the fetched data
            short_term_candles_count = min(short_term_candles_count, len(df))
            mid_term_candles_count = min(mid_term_candles_count, len(df))
            long_term_candles_count = min(long_term_candles_count, len(df))

            # Ensure we have enough data for long-term analysis
            if len(df) < long_term_candles_count:
                print(f"Warning: Not enough data for full long-term analysis. Fetched {len(df)} candles, but {long_term_candles_count} are needed for 90 days.")
                print("Please consider increasing the 'How many candles of data?' input.")
                long_term_candles_count = len(df) # Use all available data for long-term

            # Slice the DataFrame for each timeframe
            short_term_df = df.tail(short_term_candles_count)
            mid_term_df = df.tail(mid_term_candles_count)
            long_term_df = df.tail(long_term_candles_count)

            # Calculate levels for each timeframe, passing timeframe_type
            short_term_resistances = calculate_levels(short_term_df, 'resistance', len(short_term_df), timeframe_type='short')
            mid_term_resistances = calculate_levels(mid_term_df, 'resistance', len(mid_term_df), timeframe_type='mid')
            long_term_resistances = calculate_levels(long_term_df, 'resistance', len(long_term_df), timeframe_type='long')

            short_term_supports = calculate_levels(short_term_df, 'support', len(short_term_df), timeframe_type='short')
            mid_term_supports = calculate_levels(mid_term_df, 'support', len(mid_term_df), timeframe_type='mid')
            long_term_supports = calculate_levels(long_term_df, 'support', len(long_term_df), timeframe_type='long')

            print("🔴 RESISTANCE LEVELS:")
            
            def print_resistance_levels(title, levels, current_price):
                print(f"   📊 {title}:")
                displayed_count = 0
                for level_info in levels:
                    if level_info['level'] > current_price and displayed_count < 3:
                        percentage_change = ((level_info['level'] - current_price) / current_price) * 100
                        print(f"      R{displayed_count+1}: ${level_info['level']:.8f} ({percentage_change:+.2f}%) | {level_info['type']}")
                        print(f"          📊 {level_info['touches']} touches | ⏱️ {level_info['days_span']} days span")
                        displayed_count += 1
                if displayed_count == 0:
                    print("      No resistance levels found above current price.")
                print()

            print_resistance_levels("Short-term", short_term_resistances, current_price)
            print_resistance_levels("Mid-term", mid_term_resistances, current_price)
            print_resistance_levels("Long-term", long_term_resistances, current_price)

            print("🟢 SUPPORT LEVELS:")

            def print_support_levels(title, levels, current_price):
                print(f"   📊 {title}:")
                displayed_count = 0
                for level_info in levels:
                    if level_info['level'] < current_price and displayed_count < 3:
                        percentage_change = ((level_info['level'] - current_price) / current_price) * 100
                        print(f"      S{displayed_count+1}: ${level_info['level']:.8f} ({percentage_change:+.2f}%) | {level_info['type']}")
                        print(f"          📊 {level_info['touches']} touches | ⏱️ {level_info['days_span']} days span")
                        displayed_count += 1
                if displayed_count == 0:
                    print("      No support levels found below current price.")
                print()

            print_support_levels("Short-term", short_term_supports, current_price)
            print_support_levels("Mid-term", mid_term_supports, current_price)
            print_support_levels("Long-term", long_term_supports, current_price)

            # Summary calculation based on displayed levels and total found levels
            short_term_R_displayed = 0
            for level_info in short_term_resistances:
                if level_info['level'] > current_price and short_term_R_displayed < 3:
                    short_term_R_displayed += 1
            short_term_S_displayed = 0
            for level_info in short_term_supports:
                if level_info['level'] < current_price and short_term_S_displayed < 3:
                    short_term_S_displayed += 1

            mid_term_R_displayed = 0
            for level_info in mid_term_resistances:
                if level_info['level'] > current_price and mid_term_R_displayed < 3:
                    mid_term_R_displayed += 1
            mid_term_S_displayed = 0
            for level_info in mid_term_supports:
                if level_info['level'] < current_price and mid_term_S_displayed < 3:
                    mid_term_S_displayed += 1

            long_term_R_displayed = 0
            for level_info in long_term_resistances:
                if level_info['level'] > current_price and long_term_R_displayed < 3:
                    long_term_R_displayed += 1
            long_term_S_displayed = 0
            for level_info in long_term_supports:
                if level_info['level'] < current_price and long_term_S_displayed < 3:
                    long_term_S_displayed += 1

            # Calculate overall levels for summary (using the full dataframe)
            overall_resistances = calculate_levels(df, 'resistance', len(df))
            overall_supports = calculate_levels(df, 'support', len(df))

            print("💡 Summary:")
            print(f"   • Total Resistance Levels: {len(overall_resistances)}")
            print(f"   • Total Support Levels: {len(overall_supports)}")
            print(f"   • Short-term: {short_term_R_displayed}R + {short_term_S_displayed}S")
            print(f"   • Mid-term: {mid_term_R_displayed}R + {mid_term_S_displayed}S")
            print(f"   • Long-term: {long_term_R_displayed}R + {long_term_S_displayed}S")
            print("\nWould you like to analyze another cryptocurrency pair?\n")
            print("======================================================================")

    elif choice == '2':
        print("\n=== AUTOMATED SUPPORT & RESISTANCE BACKTESTING ===")
        
        symbol = input("Enter cryptocurrency pair (e.g., XRPUSDT, BTCUSDT): ").upper()
        interval = input("Enter interval ['30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']: ")
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
        
        analysis_window = int(input("Enter analysis window (number of candles for S/R calculation, default 100): ") or 100)
        test_window = int(input("Enter test window (days to test effectiveness, default 7): ") or 7)
        
        # Run backtest
        results = run_backtest(symbol, interval, start_date, end_date, analysis_window, test_window)
        
        if results:
            # Generate report
            report_filename = f"backtest_report_{symbol}_{interval}_{start_date}_{end_date}.txt"
            generate_backtest_report(results, report_filename)
            
            # Create visualizations
            create_backtest_visualizations(results)
        else:
            print("Backtest failed. Please check your inputs and try again.")
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()

