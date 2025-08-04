import pandas as pd
import numpy as np
import requests
from scipy.signal import find_peaks
import time
from datetime import timedelta

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
        print(f"Successfully fetched data for {symbol} from Binance API.")
        
        return df.tail(limit)
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Binance API: {e}")
        return None

def count_touches(df: pd.DataFrame, level: float, price_type: str, tolerance_percent: float = 0.001):
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

def calculate_levels(df: pd.DataFrame, level_type: str, num_candles: int, min_deviation_percent: float = 0.005):
    """
    Calculates support or resistance levels for a given DataFrame slice.
    Returns a list of dictionaries with 'level', 'touches', 'days_span', and 'type'.
    Applies deduplication based on min_deviation_percent.
    """
    if df.empty:
        return []

    # Dynamic parameters based on num_candles
    strong_peak_distance = max(1, num_candles // 10)
    general_peak_distance = max(1, num_candles // 20)
    
    price_range = df['high'].max() - df['low'].min()
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
        levels = sorted(list(unique_levels.values()), key=lambda x: x['level'])

        # Apply deduplication based on min_deviation_percent
        filtered_levels = []
        if levels:
            filtered_levels.append(levels[0])
            for i in range(1, len(levels)):
                if abs((levels[i]['level'] - filtered_levels[-1]['level']) / filtered_levels[-1]['level']) >= min_deviation_percent:
                    filtered_levels.append(levels[i])
        levels = filtered_levels
        
    return levels

def main():
    print("\n===== Enhanced Crypto Support & Resistance Bot ====")
    print("📊 Timeframe Analysis (Dynamic based on interval):")
    print("   • Short-term, Mid-term, and Long-term candle ranges adjust based on selected interval.")
    print("🎯 Shows top 3 closest levels for all timeframes")
    print("======================================================================")

    while True:
        currency_pair = input("Enter cryptocurrency pair (e.g., XRPUSDT, BTCUSDT) or 'exit' to quit: ").upper()
        if currency_pair == 'EXIT':
            break

        interval = input("Enter interval ['30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']: ")
        num_candles = int(input("How many candles of data? (minimum 100, max 1000): "))

        if num_candles < 100 or num_candles > 1000:
            print("Please enter a number of candles between 100 and 1000.")
            continue

        print(f"Fetching {num_candles} candles for {currency_pair} on {interval} interval...")
        start_time = time.time()
        df = get_klines(currency_pair, interval, num_candles)
        end_time = time.time()

        if df is None or df.empty:
            print("Could not fetch data or data is empty. Please check the currency pair and interval.")
            continue

        print(f"Analysis completed in {end_time - start_time:.2f} seconds.")

        print(f"\n:: Support/Resistance Analysis for {currency_pair} on {interval} timeframe ::")
        current_price = df['close'].iloc[-1]
        print(f"📈 Current Price: ${current_price:.8f}\n")

        # Define fixed candle ranges for different timeframes as per user's request
        short_term_candles_count = 10
        mid_term_candles_count = 50
        long_term_candles_count = 100

        # Ensure we have enough data for each term by slicing the main df
        short_term_df = df.tail(min(short_term_candles_count, len(df)))
        mid_term_df = df.tail(min(mid_term_candles_count, len(df)))
        long_term_df = df.tail(min(long_term_candles_count, len(df)))

        # Calculate levels for each timeframe
        short_term_resistances = calculate_levels(short_term_df, 'resistance', len(short_term_df))
        mid_term_resistances = calculate_levels(mid_term_df, 'resistance', len(mid_term_df))
        long_term_resistances = calculate_levels(long_term_df, 'resistance', len(long_term_df))

        short_term_supports = calculate_levels(short_term_df, 'support', len(short_term_df))
        mid_term_supports = calculate_levels(mid_term_df, 'support', len(mid_term_df))
        long_term_supports = calculate_levels(long_term_df, 'support', len(long_term_df))

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

if __name__ == "__main__":
    main()


