
import pandas as pd
from binance.client import Client
import numpy as np
from datetime import datetime, timedelta
import math

# Binance API (replace with your actual API key and secret)
api_key = ''
api_secret = ''
client = Client(api_key, api_secret)

def get_historical_klines(symbol, interval, start_str, end_str=None):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def calculate_sr_levels(df, reference_timestamp, interval):
    ref_dt = pd.to_datetime(reference_timestamp)
    df_ref = df[df['open_time'] <= ref_dt].copy()

    if df_ref.empty:
        print("Error: No data available up to the reference timestamp.")
        return {}, {}, 0

    current_price_at_ref = df_ref.iloc[-1]['close']

    def find_levels(data, is_resistance):
        levels = []
        for i in range(1, len(data) - 1):
            if is_resistance:
                # Resistance: high is higher than previous and next candle's high
                if data['high'].iloc[i] > data['high'].iloc[i-1] and data['high'].iloc[i] > data['high'].iloc[i+1]:
                    levels.append(data['high'].iloc[i])
            else:
                # Support: low is lower than previous and next candle's low
                if data['low'].iloc[i] < data['low'].iloc[i-1] and data['low'].iloc[i] < data['low'].iloc[i+1]:
                    levels.append(data['low'].iloc[i])
        return sorted(list(set(levels)), reverse=is_resistance)

    def get_timeframe_data(df, interval, lookback_days):
        if interval == '1d':
            start_date = ref_dt - timedelta(days=lookback_days)
        elif interval == '4h':
            start_date = ref_dt - timedelta(hours=lookback_days * 24)
        elif interval == '1h':
            start_date = ref_dt - timedelta(hours=lookback_days * 24)
        elif interval == '30m':
            start_date = ref_dt - timedelta(minutes=lookback_days * 24 * 60)
        else:
            # Default to 1 day for other intervals if not specified
            start_date = ref_dt - timedelta(days=lookback_days)
        return df[df['open_time'] >= start_date]

    # Define lookback periods based on interval
    if interval == '1d':
        short_term_lookback = 7
        mid_term_lookback = 30
        long_term_lookback = 90
    elif interval == '4h':
        short_term_lookback = 7 * (1/6) # Approx 7 days in 4h candles
        mid_term_lookback = 30 * (1/6)
        long_term_lookback = 90 * (1/6)
    elif interval == '1h':
        short_term_lookback = 7 * (1/24)
        mid_term_lookback = 30 * (1/24)
        long_term_lookback = 90 * (1/24)
    elif interval == '30m':
        short_term_lookback = 7 * (1/48)
        mid_term_lookback = 30 * (1/48)
        long_term_lookback = 90 * (1/48)
    else:
        # Default lookbacks if interval is not explicitly handled
        short_term_lookback = 7
        mid_term_lookback = 30
        long_term_lookback = 90

    # Short-term levels
    df_short = get_timeframe_data(df_ref, interval, short_term_lookback)
    res_short = find_levels(df_short, True)
    sup_short = find_levels(df_short, False)

    # Mid-term levels
    df_mid = get_timeframe_data(df_ref, interval, mid_term_lookback)
    res_mid = find_levels(df_mid, True)
    sup_mid = find_levels(df_mid, False)

    # Long-term levels
    df_long = get_timeframe_data(df_ref, interval, long_term_lookback)
    res_long = find_levels(df_long, True)
    sup_long = find_levels(df_long, False)

    def get_strength_info(level, data, is_resistance):
        touches = 0
        first_touch_time = None
        last_touch_time = None
        
        for _, row in data.iterrows():
            if is_resistance:
                if row['low'] <= level <= row['high']:
                    touches += 1
                    if first_touch_time is None: 
                        first_touch_time = row['open_time']
                    last_touch_time = row['open_time']
            else:
                if row['low'] <= level <= row['high']:
                    touches += 1
                    if first_touch_time is None: 
                        first_touch_time = row['open_time']
                    last_touch_time = row['open_time']
        
        span_days = 0
        if first_touch_time and last_touch_time:
            span_days = (last_touch_time - first_touch_time).days

        strength = "WEAK" if touches < 3 else "STRONG 💪"
        return f"   📊 {touches} touches | ⏱️ {span_days} days span", strength

    def filter_and_format_levels(levels, current_price, data_frame, is_resistance, num_levels=3):
        formatted_levels = []
        if is_resistance:
            # Only resistances above current price
            levels = [l for l in levels if l > current_price]
        else:
            # Only supports below current price
            levels = [l for l in levels if l < current_price]
        
        # Sort again to ensure closest levels are first
        levels.sort(key=lambda x: abs(x - current_price))

        for level in levels[:num_levels]:
            percentage_diff = ((level - current_price) / current_price) * 100
            strength_info, strength_text = get_strength_info(level, data_frame, is_resistance)
            formatted_levels.append(f"{level:.8f} ({percentage_diff:+.2f}%) | {strength_text}\n{strength_info}")
        return formatted_levels

    resistance_levels = {
        'short': filter_and_format_levels(res_short, current_price_at_ref, df_short, True),
        'mid': filter_and_format_levels(res_mid, current_price_at_ref, df_mid, True),
        'long': filter_and_format_levels(res_long, current_price_at_ref, df_long, True)
    }

    support_levels = {
        'short': filter_and_format_levels(sup_short, current_price_at_ref, df_short, False),
        'mid': filter_and_format_levels(sup_mid, current_price_at_ref, df_mid, False),
        'long': filter_and_format_levels(sup_long, current_price_at_ref, df_long, False)
    }

    return resistance_levels, support_levels, current_price_at_ref

def run_backtest(df, reference_timestamp, num_candles, resistance_levels_dict, support_levels_dict, interval):
    ref_dt = pd.to_datetime(reference_timestamp)
    
    # Get all levels in a flat list for easier checking, and map them to their terms and types
    all_levels_info = []
    for term, levels_list in resistance_levels_dict.items():
        for l_str in levels_list:
            level_val = float(l_str.split(' ')[0])
            all_levels_info.append({'level': level_val, 'type': 'Resistance', 'term': term.capitalize(), 'raw_str': l_str})

    for term, levels_list in support_levels_dict.items():
        for l_str in levels_list:
            level_val = float(l_str.split(' ')[0])
            all_levels_info.append({'level': level_val, 'type': 'Support', 'term': term.capitalize(), 'raw_str': l_str})

    # Sort by level value to easily find closest levels
    all_levels_info.sort(key=lambda x: x['level'])

    backtest_results = []
    
    # Find the index of the reference candle
    ref_index = df[df['open_time'] == ref_dt].index
    if ref_index.empty:
        print("Reference timestamp not found in data for backtesting.")
        return []
    ref_index = ref_index[0]

    for i in range(1, num_candles + 1):
        if ref_index + i < len(df):
            candle = df.iloc[ref_index + i]
            candle_data = {
                'candle_num': i,
                'timestamp': candle['open_time'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'interactions': []
            }

            for level_info in all_levels_info:
                level = level_info['level']
                level_type = level_info['type']
                level_term = level_info['term']
                raw_str = level_info['raw_str']

                # Check for interaction
                if candle['low'] <= level <= candle['high']:
                    deviation = 0
                    hit_type = ""
                    action = ""
                    confidence = ""

                    if level_type == 'Resistance':
                        # Deviation based on how close the high was to the resistance
                        deviation = ((candle['high'] - level) / level) * 100
                        if abs(deviation) < 0.05: # Tighter range for EXACT
                            hit_type = "EXACT HIT"
                        elif abs(deviation) < 0.2: # Tighter range for NEAR
                            hit_type = "NEAR HIT"
                        else:
                            hit_type = "TOUCH"
                        
                        if candle['close'] < level: # Close below resistance
                            action = "Rejected (Bounce 🔄)"
                        else: # Close above resistance
                            action = "Breakout (Break 🔓)"

                    elif level_type == 'Support':
                        # Deviation based on how close the low was to the support
                        deviation = ((candle['low'] - level) / level) * 100
                        if abs(deviation) < 0.05: # Tighter range for EXACT
                            hit_type = "EXACT HIT"
                        elif abs(deviation) < 0.2: # Tighter range for NEAR
                            hit_type = "NEAR HIT"
                        else:
                            hit_type = "TOUCH"
                        
                        if candle['close'] > level: # Close above support
                            action = "Rejected (Bounce 🔄)"
                        else: # Close below support
                            action = "Breakout (Break 🔓)"
                    
                    # Confidence logic (can be refined further)
                    if hit_type == "EXACT HIT":
                        confidence = "Very Strong"
                    elif hit_type == "NEAR HIT":
                        confidence = "Strong"
                    else:
                        confidence = "Medium"

                    interaction = {
                        'level_type': level_type,
                        'level_term': level_term,
                        'level_value': level,
                        'raw_level_str': raw_str,
                        'deviation': deviation,
                        'hit_type': hit_type,
                        'action': action,
                        'confidence': confidence
                    }
                    candle_data['interactions'].append(interaction)
            backtest_results.append(candle_data)

    return backtest_results

def main():
    print("==== S/R Backtesting Script ====")
    print("This script fetches historical data, calculates S/R levels based on a reference point,")
    print("and then analyzes subsequent candles for interactions with those levels.")
    print("======================================================================")

    symbol = input("Enter cryptocurrency pair (e.g., XRPUSDT, BTCUSDT): ").upper()
    interval = input("Enter interval (e.g., '30m', '1h', '4h', '1d'): ").lower()

    # Calculate start and end times for data fetching
    end_time = datetime.now()
    # Fetch enough data for long-term lookback (e.g., 90 days for 1d, adjust for other intervals)
    if interval == '1d':
        start_time = end_time - timedelta(days=120) # 90 days for long-term + buffer
    elif interval == '4h':
        start_time = end_time - timedelta(days=120) # 90 days in 4h candles
    elif interval == '1h':
        start_time = end_time - timedelta(days=120) # 90 days in 1h candles
    elif interval == '30m':
        start_time = end_time - timedelta(days=120) # 90 days in 30m candles
    else:
        start_time = end_time - timedelta(days=120) # Default for others

    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Fetching data from {start_str} to {end_str}")
    df = get_historical_klines(symbol, interval, start_str, end_str)

    if df.empty:
        print("Could not fetch data. Please check symbol and interval.")
        return

    reference_timestamp = input("Enter reference timestamp (YYYY-MM-DD HH:MM:SS, e.g., 2025-06-16 08:00:00): ")

    print(f"\n--- S/R Levels calculated based on data up to {reference_timestamp} ---")
    resistance_levels, support_levels, current_price = calculate_sr_levels(df, reference_timestamp, interval)
    print(f"Current Price at Reference: ${current_price:.8f}")

    print("\n🔴 RESISTANCE LEVELS:")
    if resistance_levels['short']:
        print("   📊 Short-term:")
        for i, level in enumerate(resistance_levels['short']):
            print(f"      R{i+1}: {level}")
    else:
        print("   📊 Short-term:\n      No R levels found above current price.")

    if resistance_levels['mid']:
        print("\n   📊 Mid-term:")
        for i, level in enumerate(resistance_levels['mid']):
            print(f"      R{i+1}: {level}")

    if resistance_levels['long']:
        print("\n   📊 Long-term:")
        for i, level in enumerate(resistance_levels['long']):
            print(f"      R{i+1}: {level}")

    print("\n🟢 SUPPORT LEVELS:")
    if support_levels['short']:
        print("   📊 Short-term:")
        for i, level in enumerate(support_levels['short']):
            print(f"      S{i+1}: {level}")
    else:
        print("   📊 Short-term:\n      No S levels found below current price.")

    if support_levels['mid']:
        print("\n   📊 Mid-term:")
        for i, level in enumerate(support_levels['mid']):
            print(f"      S{i+1}: {level}")

    if support_levels['long']:
        print("\n   📊 Long-term:")
        for i, level in enumerate(support_levels['long']):
            print(f"      S{i+1}: {level}")

    print("----------------------------------------------------------------------")

    num_candles = int(input("Enter number of candles to analyze after reference: "))

    print("\n===== Backtest Summary =====")
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Reference Timestamp: {reference_timestamp}")
    print(f"Current Price at Reference: ${current_price:.4f}")
    print(f"Candles Analyzed After Reference: {num_candles}")
    print("------------------------------")
    print("🟢 Support/Resistance Hits After Reference")

    total_touches = 0
    support_hits = 0
    resistance_hits = 0
    exact_hits = 0
    near_hits = 0
    bounces = 0
    breakouts = 0
    
    # To track levels most respected
    level_hit_counts = {}

    backtest_results = run_backtest(df, reference_timestamp, num_candles, resistance_levels, support_levels, interval)

    if backtest_results:
        for candle_data in backtest_results:
            print(f"\n📅 Candle {candle_data['candle_num']} — {candle_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"High: ${candle_data['high']:.4f} | Low: ${candle_data['low']:.4f} | Close: ${candle_data['close']:.4f}")

            if candle_data['interactions']:
                for interaction in candle_data['interactions']:
                    total_touches += 1
                    if interaction['level_type'] == 'Support':
                        support_hits += 1
                    else:
                        resistance_hits += 1
                    
                    if interaction['hit_type'] == 'EXACT HIT':
                        exact_hits += 1
                    elif interaction['hit_type'] == 'NEAR HIT':
                        near_hits += 1

                    if 'Bounce' in interaction['action']:
                        bounces += 1
                    elif 'Breakout' in interaction['action']:
                        breakouts += 1
                    
                    # Using level value directly for tracking, as raw_level_str might contain formatting
                    level_key = f"{interaction['level_type'][0]}{interaction['level_value']:.4f}"
                    level_hit_counts[level_key] = level_hit_counts.get(level_key, 0) + 1

                    print(f"\n✅ HIT: {interaction['level_term']}-term {interaction['level_type']} ({interaction['level_type'][0]}{interaction['level_value']:.4f})")
                    print(f"→ Deviation: {interaction['deviation']:.2f}%")
                    print(f"→ Type: {interaction['hit_type']}")
                    print(f"→ Action: {interaction['action']}")
                    print(f"→ Confidence: {interaction['confidence']}")
            else:
                print("No significant interaction with S/R levels.")
    else:
        print("No interactions found in the backtest period.")

    print("\n===== Interaction Summary =====")
    print(f"Total Candles Analyzed: {num_candles}")
    print(f"Total S/R Touches: {total_touches}")
    print(f"\n- Support Hits: {support_hits}")
    print(f"- Resistance Hits: {resistance_hits}")
    print(f"- Exact Hits: {exact_hits}")
    print(f"- Near Hits: {near_hits}")
    print(f"- Bounce: {bounces}")
    print(f"- Breakouts: {breakouts}")

    print("\nLevels Most Respected:")
    if level_hit_counts:
        sorted_levels = sorted(level_hit_counts.items(), key=lambda item: item[1], reverse=True)
        for level_key, count in sorted_levels:
            print(f"✅ {level_key}: {count} hits")
    else:
        print("No levels were touched.")

    print("\nBacktest complete.")

if __name__ == "__main__":
    main()


