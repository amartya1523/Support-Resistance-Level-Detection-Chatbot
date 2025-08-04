import pandas as pd
import requests
import numpy as np
from datetime import datetime
import time

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

def detect_levels(df, round_precision=4, tolerance_pct=0.005, interval='1d'):
    """
    Detect support and resistance levels from historical price data
    """
    all_prices = []
    for _, row in df.iterrows():
        all_prices.append(row['high'])
        all_prices.append(row['low'])
    
    # Sort unique prices to facilitate clustering
    unique_prices = sorted(list(set([round(p, round_precision) for p in all_prices])))

    clustered_levels = []  # Stores (average_level_value, [prices_in_cluster])

    for price in unique_prices:
        found_cluster = False
        for i, (avg_cl_value, cl_prices) in enumerate(clustered_levels):
            # Check if the current price is within tolerance of the cluster's average value
            if abs(price - avg_cl_value) / avg_cl_value < tolerance_pct:
                cl_prices.append(price)
                clustered_levels[i] = (np.mean(cl_prices), cl_prices)  # Update average
                found_cluster = True
                break
        if not found_cluster:
            clustered_levels.append((price, [price]))

    final_levels_with_touches = []  # Stores (level_type, level_value, touches, term)

    # Define lookback periods for short, mid, long term based on interval
    # These are in terms of number of candles
    data_length = len(df)
    short_term_candles = max(7, data_length // 4)
    mid_term_candles = max(20, data_length // 2)
    long_term_candles = data_length

    for avg_cl_value, _ in clustered_levels:
        total_touches = 0
        short_term_touches = 0
        mid_term_touches = 0
        long_term_touches = 0

        for j in range(len(df)):
            is_touched = False
            if (abs(df['high'].iloc[j] - avg_cl_value) / avg_cl_value < tolerance_pct) or \
               (abs(df['low'].iloc[j] - avg_cl_value) / avg_cl_value < tolerance_pct):
                is_touched = True
                total_touches += 1
            
            if is_touched:
                # Check if the touch falls within the lookback periods
                if j >= len(df) - short_term_candles:
                    short_term_touches += 1
                if j >= len(df) - mid_term_candles:
                    mid_term_touches += 1
                if j >= len(df) - long_term_candles:
                    long_term_touches += 1
        
        if total_touches > 0:  # Only consider levels that have been touched
            term = ""
            if short_term_touches > 0: term = "Short-term"
            if mid_term_touches > 0 and mid_term_touches > short_term_touches: term = "Mid-term"
            if long_term_touches > 0 and long_term_touches > mid_term_touches: term = "Long-term"
            
            # Determine level type (resistance or support) based on current price
            level_type = 'resistance' if avg_cl_value > df['close'].iloc[-1] else 'support'
            
            final_levels_with_touches.append((level_type, round(avg_cl_value, round_precision), total_touches, term))

    # Separate into resistance and support lists
    final_resistances = []
    final_supports = []
    for level_type, level_value, touches, term in final_levels_with_touches:
        if level_type == 'resistance':
            final_resistances.append((level_value, touches, term))
        else:
            final_supports.append((level_value, touches, term))

    return final_supports, final_resistances

def format_levels(levels, current_price, is_support=True, num_levels=5):
    """
    Format and filter levels based on current price
    """
    filtered = []
    for lvl, touches, term in levels:
        diff_pct = ((lvl - current_price) / current_price) * 100
        if (is_support and lvl < current_price) or (not is_support and lvl > current_price):
            filtered.append((lvl, diff_pct, touches, term))

    # Sort by number of touches (strength) first, then by proximity to current price
    sorted_levels = sorted(filtered, key=lambda x: (-x[2], abs(x[1])))
    
    # Return requested number of levels
    return sorted_levels[:num_levels]

def classify_strength(touches):
    """
    Classify the strength of a level based on number of touches
    """
    if touches >= 5:
        return "STRONG"
    elif touches >= 3:
        return "MEDIUM"
    else:
        return "LIGHT"

def display_levels(supports, resistances):
    """
    Display formatted support and resistance levels
    """
    print("\n🔴 RESISTANCE LEVELS (Above Current Price):")
    for i, (lvl, pct, touches, term) in enumerate(resistances, 1):
        strength = classify_strength(touches)
        strength_indicator = "💪" if strength == "STRONG" else "👍" if strength == "MEDIUM" else "👌"
        print(f"   R{i}: ${lvl:.8f} (+{pct:.2f}%) [{strength} {strength_indicator} - {touches} touches] ({term})")

    print("\n🟢 SUPPORT LEVELS (Below Current Price):")
    for i, (lvl, pct, touches, term) in enumerate(supports, 1):
        strength = classify_strength(touches)
        strength_indicator = "💪" if strength == "STRONG" else "👍" if strength == "MEDIUM" else "👌"
        print(f"   S{i}: ${lvl:.8f} ({pct:.2f}%) [{strength} {strength_indicator} - {touches} touches] ({term})")

def interactive_bot():
    """
    Interactive chatbot for support/resistance analysis
    """
    print("\n===== Crypto Support & Resistance Bot =====")
    print("This bot calculates accurate support and resistance levels for cryptocurrency pairs.")
    
    while True:
        print("\n" + "="*50)
        # Get cryptocurrency pair
        symbol = input("Enter cryptocurrency pair (e.g., XRPUSDT, BTCUSDT) or 'exit' to quit: ").strip().upper()
        if symbol.lower() == 'exit':
            print("Thank you for using the Support & Resistance Bot. Goodbye!")
            break
            
        # Get timeframe
        valid_timeframes = ['30m', '1h', '4h', '1d', '1w', '1M']
        timeframe = input(f"Enter timeframe {valid_timeframes}: ").strip()
        if timeframe not in valid_timeframes:
            print(f"Invalid timeframe. Using default: 1d")
            timeframe = '1d'
        
        # Get number of candles directly
        try:
            num_candles = int(input("How many candles of data do you want? (10-1000): "))
            num_candles = max(10, min(1000, num_candles))  # Limit between 10 and 1000
        except ValueError:
            print("Invalid input. Using default: 100 candles")
            num_candles = 100
            
        # Get number of levels
        try:
            num_levels = int(input("How many support/resistance levels do you want? (1-10): "))
            num_levels = max(1, min(10, num_levels))  # Limit between 1 and 10
        except ValueError:
            print("Invalid input. Using default: 5 levels")
            num_levels = 5
            
        print(f"\nFetching {num_candles} candles for {symbol} on {timeframe} timeframe...")
        
        # Get data from API
        df = get_klines(symbol, timeframe, limit=num_candles)
            
        if df is None or df.empty:
            print(f"Failed to retrieve data for {symbol}. Please check the symbol name and try again.")
            continue
            
        current_price = df['close'].iloc[-1]
        
        print(f"Analyzing {len(df)} candles for {symbol}...")
        print(f"Current Price: ${current_price:.4f}")
        
        # Calculate support and resistance levels
        start_time = time.time()
        supports, resistances = detect_levels(df, round_precision=4, tolerance_pct=0.005, interval=timeframe)
        
        # Format and display levels
        supports = format_levels(supports, current_price, is_support=True, num_levels=num_levels)
        resistances = format_levels(resistances, current_price, is_support=False, num_levels=num_levels)
        
        end_time = time.time()
        print(f"Analysis completed in {end_time - start_time:.2f} seconds.")
        
        print(f"\n:: {symbol} Support/Resistance Analysis on {timeframe.upper()} timeframe with {len(df)} candles ::")
        display_levels(supports, resistances)
        
        print("\nWould you like to analyze another cryptocurrency pair?")

if __name__ == "__main__":
    interactive_bot()