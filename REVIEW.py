import pandas as pd
import requests
import numpy as np
from datetime import datetime
import time

def get_klines(symbol: str, interval: str, limit: int = 1000, csv_path: str = None):
    """
    Fetch historical klines (candlestick data) for a given symbol and interval
    Prioritizes Binance API and uses CSV as fallback
    """
    # Try to fetch from Binance API first
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
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        print(f"Successfully fetched data for {symbol} from Binance API.")
        
        # Return exactly 1780 hours of data for all timeframes
        if interval == '1d':
            return df.tail(76)  # 1780 hours ÷ 24 hours/day = 74.17 ≈ 76 days
        elif interval == '30m':
            return df.tail(3560)  # 1780 hours × 2 candles/hour = 3560 candles
        elif interval == '1h':
            return df.tail(1780)  # 1780 hours × 1 candle/hour = 1780 candles
        elif interval == '4h':
            return df.tail(445)  # 1780 hours ÷ 4 hours/candle = 445 candles
        elif interval == '1w':
            return df.tail(11)   # 1780 hours ÷ 168 hours/week = 10.6 ≈ 11 weeks
        elif interval == '1M':
            return df.tail(3)    # 1780 hours ÷ 730 hours/month = 2.4 ≈ 3 months
        else:
            return df.tail(limit)  # Default fallback
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Binance API: {e}")
        print("Attempting to use backup CSV data...")
        
        # Fallback to CSV if API fails
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
                df.columns = [
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ]
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
                print(f"Successfully loaded data from backup CSV: {csv_path}")
                
                # Return exactly 1780 hours of data for all timeframes
                if interval == '1d':
                    return df.tail(76)  # 1780 hours ÷ 24 hours/day = 74.17 ≈ 76 days
                elif interval == '30m':
                    return df.tail(3560)  # 1780 hours × 2 candles/hour = 3560 candles
                elif interval == '1h':
                    return df.tail(1780)  # 1780 hours × 1 candle/hour = 1780 candles
                elif interval == '4h':
                    return df.tail(445)  # 1780 hours ÷ 4 hours/candle = 445 candles
                elif interval == '1w':
                    return df.tail(11)   # 1780 hours ÷ 168 hours/week = 10.6 ≈ 11 weeks
                elif interval == '1M':
                    return df.tail(3)    # 1780 hours ÷ 730 hours/month = 2.4 ≈ 3 months
                else:
                    return df.tail(limit)
                    
            except Exception as e:
                print(f"Error loading backup CSV: {e}")
                return None
        else:
            print("No backup CSV provided and API request failed.")
            return None

def detect_levels(df, round_precision=4, tolerance_pct=0.002):
    """
    Detect support and resistance levels from historical price data
    Using improved algorithm for better accuracy
    """
    # First, identify swing highs and lows
    window = 5  # Window size for identifying swing points
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df) - window):
        # Check for swing high
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
            swing_highs.append(df['high'].iloc[i])
        # Check for swing low
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
            swing_lows.append(df['low'].iloc[i])
    
    # Combine all potential levels
    all_levels = swing_highs + swing_lows
    all_levels = [round(level, round_precision) for level in all_levels]
    
    # Cluster similar price levels
    clustered_levels = []  # Stores (level_value, [prices_in_cluster])
    for price in sorted(all_levels):
        found_cluster = False
        for i, (cl_value, cl_prices) in enumerate(clustered_levels):
            if abs(price - cl_value) / cl_value < tolerance_pct:
                cl_prices.append(price)
                clustered_levels[i] = (np.mean(cl_prices), cl_prices)  # Update average
                found_cluster = True
                break
        if not found_cluster:
            clustered_levels.append((price, [price]))

    # Count touches for each clustered level
    final_levels_with_touches = []  # Stores (level_value, touches)
    for cl_value, _ in clustered_levels:
        touches = 0
        # Iterate through all candles to count touches for this clustered level
        for _, row in df.iterrows():
            # A touch is when high or low is within tolerance of the level
            if (abs(row['high'] - cl_value) / cl_value < tolerance_pct) or \
               (abs(row['low'] - cl_value) / cl_value < tolerance_pct):
                touches += 1
        
        if touches > 0:  # Only consider levels that have been touched
            final_levels_with_touches.append((round(cl_value, round_precision), touches))

    # Add additional key price levels that might have been missed
    # Check for round numbers that often act as psychological levels
    current_price = df['close'].iloc[-1]
    price_magnitude = 10 ** int(np.log10(current_price))
    
    # Generate round numbers at different scales
    round_numbers = []
    for multiplier in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        base_round = price_magnitude * multiplier
        # Add some round numbers around the current price range
        for offset in [-2, -1, 0, 1, 2]:
            round_num = base_round + (offset * base_round * 0.1)
            if round_num > 0:
                round_numbers.append(round_num)
    
    for round_num in round_numbers:
        # Skip if this round number is already in our levels
        already_exists = False
        for level, _ in final_levels_with_touches:
            if round_num > 0 and abs(round_num - level) / level < tolerance_pct:
                already_exists = True
                break
                
        if not already_exists and round_num > 0:
            # Count touches for this round number
            touches = 0
            for _, row in df.iterrows():
                if (abs(row['high'] - round_num) / round_num < tolerance_pct) or \
                   (abs(row['low'] - round_num) / round_num < tolerance_pct):
                    touches += 1
            
            if touches > 0:  # Only add if it has been touched
                final_levels_with_touches.append((round(round_num, round_precision), touches))

    return final_levels_with_touches

def format_levels(levels, current_price, is_support=True, num_levels=5):
    """
    Format and filter levels based on current price
    """
    filtered = []
    for lvl, touches in levels:
        diff_pct = ((lvl - current_price) / current_price) * 100
        if (is_support and lvl < current_price) or (not is_support and lvl > current_price):
            filtered.append((lvl, diff_pct, touches))

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
    if resistances:
        for i, (lvl, pct, touches) in enumerate(resistances, 1):
            strength = classify_strength(touches)
            strength_indicator = "💪" if strength == "STRONG" else "👍" if strength == "MEDIUM" else "👌"
            print(f"   R{i}: ${lvl:.8f} (+{pct:.2f}%) [{strength} {strength_indicator} - {touches} touches]")
    else:
        print("   No significant resistance levels found above current price.")

    print("\n🟢 SUPPORT LEVELS (Below Current Price):")
    if supports:
        for i, (lvl, pct, touches) in enumerate(supports, 1):
            strength = classify_strength(touches)
            strength_indicator = "💪" if strength == "STRONG" else "👍" if strength == "MEDIUM" else "👌"
            print(f"   S{i}: ${lvl:.8f} ({pct:.2f}%) [{strength} {strength_indicator} - {touches} touches]")
    else:
        print("   No significant support levels found below current price.")

def interactive_bot():
    """
    Interactive chatbot for support/resistance analysis
    """
    print("\n===== Crypto Support & Resistance Bot =====")
    print("This bot calculates accurate support and resistance levels for cryptocurrency pairs.")
    
    # Default CSV path for fallback mode
    backup_csv_path = '/home/ubuntu/upload/XRPUSDT_1d_500.csv'
    
    while True:
        print("\n" + "="*50)
        # Get cryptocurrency pair
        symbol = input("Enter cryptocurrency pair (e.g., XRPUSDT, BTCUSDT) or 'exit' to quit: ").strip().upper()
        if symbol.lower() == 'exit':
            print("Thank you for using the Support & Resistance Bot. Goodbye!")
            break
            
        # Get timeframe - Updated to include 1M (1 month)
        valid_timeframes = ['30m', '1h', '4h', '1d', '1w', '1M']
        print(f"Available timeframes: {', '.join(valid_timeframes)}")
        print("Note: 1M = 1 month, 1w = 1 week")
        timeframe = input(f"Enter timeframe: ").strip()
        
        if timeframe not in valid_timeframes:
            print(f"Invalid timeframe. Available options: {', '.join(valid_timeframes)}")
            print("Using default: 1d")
            timeframe = '1d'
            
        # Get number of levels
        try:
            num_levels = int(input("How many support/resistance levels do you want? (1-10): "))
            num_levels = max(1, min(10, num_levels))  # Limit between 1 and 10
        except ValueError:
            print("Invalid input. Using default: 5 levels")
            num_levels = 5
            
        print(f"\nFetching data for {symbol} on {timeframe} timeframe...")
        
        # Try to get data from API first, with CSV as fallback
        df = None
        try:
            # Increase limit to ensure we have enough data for 30m intervals
            api_limit = 3560 if timeframe == '30m' else 1000
            # Only use the backup CSV if the API fails and the symbol is XRPUSDT
            df = get_klines(symbol, timeframe, limit=api_limit, 
                           csv_path=backup_csv_path if symbol == 'XRPUSDT' else None)
        except Exception as e:
            print(f"Error: {e}")
            
        if df is None or df.empty:
            print(f"Failed to retrieve data for {symbol}. Please try again.")
            continue
            
        current_price = df['close'].iloc[-1]
        
        print(f"Analyzing {len(df)} candles for {symbol}...")
        print(f"Current Price: ${current_price:.8f}")
        print(f"Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Calculate support and resistance levels
        start_time = time.time()
        all_detected_levels = detect_levels(df, round_precision=8, tolerance_pct=0.002)
        
        if not all_detected_levels:
            print("No significant support/resistance levels detected. Try a different timeframe or symbol.")
            continue
        
        # Separate into support and resistance
        supports = []
        resistances = []
        for lvl, touches in all_detected_levels:
            if lvl < current_price:
                supports.append((lvl, touches))
            else:
                resistances.append((lvl, touches))
                
        # Format and display levels
        supports = format_levels(supports, current_price, is_support=True, num_levels=num_levels)
        resistances = format_levels(resistances, current_price, is_support=False, num_levels=num_levels)
        
        end_time = time.time()
        print(f"Analysis completed in {end_time - start_time:.2f} seconds.")
        
        print(f"\n:: {symbol} Support/Resistance Analysis on {timeframe.upper()} timeframe ::")
        display_levels(supports, resistances)
        
        print("\nWould you like to analyze another cryptocurrency pair?")

if __name__ == "__main__":
    interactive_bot()