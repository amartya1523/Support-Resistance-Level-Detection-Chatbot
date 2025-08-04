import pandas as pd
import numpy as np
import requests
from scipy.signal import find_peaks
from datetime import timedelta

def get_klines(symbol: str, interval: str, limit: int = 1000):
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

def interval_to_timedelta(interval_str: str) -> timedelta:
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
    elif unit == 'M':
        return timedelta(days=value * 30)
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")

def count_touches(df: pd.DataFrame, level: float, price_type: str, tolerance_percent: float = 0.005):
    touches = 0
    tolerance = level * tolerance_percent
    touch_timestamps = []
    deviations = []

    if price_type == 'resistance':
        for index, high_price in df['high'].items():
            if level - tolerance <= high_price <= level + tolerance:
                touches += 1
                touch_timestamps.append(index)
                deviations.append(abs((high_price - level) / level))
    elif price_type == 'support':
        for index, low_price in df['low'].items():
            if level - tolerance <= low_price <= level + tolerance:
                touches += 1
                touch_timestamps.append(index)
                deviations.append(abs((low_price - level) / level))

    days_span = 0
    if len(touch_timestamps) > 1:
        time_diff = touch_timestamps[-1] - touch_timestamps[0]
        days_span = time_diff.days

    avg_deviation = np.mean(deviations) if deviations else 0
    return touches, days_span, avg_deviation

def calculate_levels(df: pd.DataFrame, level_type: str, num_candles: int, min_deviation_percent: float = 0.005, timeframe_type: str = 'general'):
    if df.empty:
        return []

    strong_peak_distance = max(1, num_candles // 10)
    general_peak_distance = max(1, num_candles // 20)
    
    price_range = df['high'].max() - df['low'].min()

    if timeframe_type == 'short':
        strong_peak_prominence = 0.005 * price_range
        peak_rank_width = 0.002 * price_range
        min_pivot_rank = 1
    else:
        strong_peak_prominence = 0.01 * price_range
        peak_rank_width = 0.005 * price_range
        min_pivot_rank = 2

    levels = []

    if level_type == 'resistance':
        price_data = df['high']
        strong_indices, _ = find_peaks(price_data, distance=strong_peak_distance, prominence=strong_peak_prominence)
        for idx in strong_indices:
            level = price_data.iloc[idx]
            touches, days_span, _ = count_touches(df, level, 'resistance')
            levels.append({'level': level, 'touches': touches, 'days_span': days_span, 'type': 'STRONG 💪'})

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
                touches, days_span, _ = count_touches(df, level, 'resistance')
                levels.append({'level': level, 'touches': touches, 'days_span': days_span, 'type': 'GENERAL'})
        
        unique_levels = {}
        for l in levels:
            if l['level'] not in unique_levels or l['touches'] > unique_levels[l['level']]['touches']:
                unique_levels[l['level']] = l
        levels = sorted(list(unique_levels.values()), key=lambda x: x['level'], reverse=True)

        filtered_levels = []
        if levels:
            filtered_levels.append(levels[0])
            for i in range(1, len(levels)):
                if abs((levels[i]['level'] - filtered_levels[-1]['level']) / filtered_levels[-1]['level']) >= min_deviation_percent:
                    filtered_levels.append(levels[i])
        levels = filtered_levels
        
    elif level_type == 'support':
        price_data = -df['low']
        strong_indices, _ = find_peaks(price_data, distance=strong_peak_distance, prominence=strong_peak_prominence)
        for idx in strong_indices:
            level = -price_data.iloc[idx]
            touches, days_span, _ = count_touches(df, level, 'support')
            levels.append({'level': level, 'touches': touches, 'days_span': days_span, 'type': 'STRONG 💪'})

        general_indices, _ = find_peaks(price_data, distance=general_peak_distance)
        trough_to_rank = {idx: 0 for idx in general_indices}
        for i, current_trough_idx in enumerate(general_indices):
            current_level = price_data.iloc[current_trough_idx]
            for previous_trough_idx in general_indices[:i]:
                if abs(current_level - price_data.iloc[previous_trough_idx]) <= peak_rank_width:
                    trough_to_rank[current_trough_idx] += 1

        for trough_idx, rank in trough_to_rank.items():
            if rank >= min_pivot_rank:
                level = -price_data.iloc[trough_idx]
                touches, days_span, _ = count_touches(df, level, 'support')
                levels.append({'level': level, 'touches': touches, 'days_span': days_span, 'type': 'GENERAL'})
        
        unique_levels = {}
        for l in levels:
            if l['level'] not in unique_levels or l['touches'] > unique_levels[l['level']]['touches']:
                unique_levels[l['level']] = l
        levels = sorted(list(unique_levels.values()), key=lambda x: x['level'], reverse=True)

        filtered_levels = []
        if levels:
            filtered_levels.append(levels[0])
            for i in range(1, len(levels)):
                if abs((levels[i]['level'] - filtered_levels[-1]['level']) / filtered_levels[-1]['level']) >= min_deviation_percent:
                    filtered_levels.append(levels[i])
        levels = filtered_levels
        
    return levels

def evaluate_level(df_test: pd.DataFrame, level: float, level_type: str, tolerance_percent: float = 0.005, deviation_threshold: float = 0.01):
    touches, days_span, avg_deviation = count_touches(df_test, level, level_type, tolerance_percent)
    
    deviations_above_threshold = 0
    tolerance = level * tolerance_percent
    if level_type == 'resistance':
        for high_price in df_test['high']:
            if high_price > level * (1 + deviation_threshold):
                deviations_above_threshold += 1
    elif level_type == 'support':
        for low_price in df_test['low']:
            if low_price < level * (1 - deviation_threshold):
                deviations_above_threshold += 1
                
    return {
        'level': level,
        'touches': touches,
        'days_span': days_span,
        'avg_deviation_percent': avg_deviation * 100,
        'deviations_above_threshold': deviations_above_threshold
    }

def run_backtest(symbol: str, interval: str, num_candles: int, 
                 train_ratio: float = 0.7, 
                 tolerance_percent: float = 0.005,
                 entry_buffer_percent: float = 0.001, 
                 stop_loss_percent: float = 0.01, 
                 take_profit_percent: float = 0.02,
                 deviation_threshold: float = 0.01
                ):
    
    df = get_klines(symbol, interval, num_candles)
    if df is None or df.empty:
        return {"error": "Could not fetch data or data is empty."}

    train_size = int(len(df) * train_ratio)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    # Calculate S/R levels on training data
    resistances_train = calculate_levels(df_train, 'resistance', len(df_train), tolerance_percent)
    supports_train = calculate_levels(df_train, 'support', len(df_train), tolerance_percent)

    trades = []
    in_position = False
    position_entry_price = 0
    position_type = ""

    # Backtest on testing data
    for i in range(len(df_test)):
        current_candle = df_test.iloc[i]
        current_price = current_candle['close']

        if not in_position:
            # Check for potential long entry (bounce off support)
            for support_level_info in supports_train:
                support_level = support_level_info['level']
                if current_price > support_level and current_price <= support_level * (1 + entry_buffer_percent):
                    if current_candle['low'] <= support_level * (1 + tolerance_percent) and current_candle['low'] >= support_level * (1 - tolerance_percent):
                        in_position = True
                        position_entry_price = current_price
                        position_type = "long"
                        trades.append({
                            'entry_time': current_candle.name,
                            'entry_price': position_entry_price,
                            'type': position_type,
                            'stop_loss': position_entry_price * (1 - stop_loss_percent),
                            'take_profit': position_entry_price * (1 + take_profit_percent)
                        })
                        break
            
            # Check for potential short entry (rejection from resistance)
            if not in_position:
                for resistance_level_info in resistances_train:
                    resistance_level = resistance_level_info['level']
                    if current_price < resistance_level and current_price >= resistance_level * (1 - entry_buffer_percent):
                        if current_candle['high'] >= resistance_level * (1 - tolerance_percent) and current_candle['high'] <= resistance_level * (1 + tolerance_percent):
                            in_position = True
                            position_entry_price = current_price
                            position_type = "short"
                            trades.append({
                                'entry_time': current_candle.name,
                                'entry_price': position_entry_price,
                                'type': position_type,
                                'stop_loss': position_entry_price * (1 + stop_loss_percent),
                                'take_profit': position_entry_price * (1 - take_profit_percent)
                            })
                            break

        else:
            if position_type == "long":
                if current_candle['low'] <= trades[-1]['stop_loss']:
                    trades[-1]['exit_time'] = current_candle.name
                    trades[-1]['exit_price'] = trades[-1]['stop_loss']
                    trades[-1]['profit_loss_percent'] = ((trades[-1]['exit_price'] - trades[-1]['entry_price']) / trades[-1]['entry_price']) * 100
                    in_position = False
                elif current_candle['high'] >= trades[-1]['take_profit']:
                    trades[-1]['exit_time'] = current_candle.name
                    trades[-1]['exit_price'] = trades[-1]['take_profit']
                    trades[-1]['profit_loss_percent'] = ((trades[-1]['exit_price'] - trades[-1]['entry_price']) / trades[-1]['entry_price']) * 100
                    in_position = False
            
            elif position_type == "short":
                if current_candle['high'] >= trades[-1]['stop_loss']:
                    trades[-1]['exit_time'] = current_candle.name
                    trades[-1]['exit_price'] = trades[-1]['stop_loss']
                    trades[-1]['profit_loss_percent'] = ((trades[-1]['entry_price'] - trades[-1]['exit_price']) / trades[-1]['entry_price']) * 100
                    in_position = False
                elif current_candle['low'] <= trades[-1]['take_profit']:
                    trades[-1]['exit_time'] = current_candle.name
                    trades[-1]['exit_price'] = trades[-1]['take_profit']
                    trades[-1]['profit_loss_percent'] = ((trades[-1]['entry_price'] - trades[-1]['exit_price']) / trades[-1]['entry_price']) * 100
                    in_position = False

    # Evaluate S/R levels on testing data
    evaluated_resistances = [evaluate_level(df_test, r['level'], 'resistance', tolerance_percent, deviation_threshold) for r in resistances_train]
    evaluated_supports = [evaluate_level(df_test, s['level'], 'support', tolerance_percent, deviation_threshold) for s in supports_train]

    total_profit_loss_percent = sum([t['profit_loss_percent'] for t in trades if 'profit_loss_percent' in t])
    num_trades = len(trades)
    winning_trades = [t for t in trades if 'profit_loss_percent' in t and t['profit_loss_percent'] > 0]
    num_winning_trades = len(winning_trades)
    win_rate = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0

    results = {
        "total_profit_loss_percent": total_profit_loss_percent,
        "num_trades": num_trades,
        "num_winning_trades": num_winning_trades,
        "win_rate": win_rate,
        "trades": trades,
        "evaluated_resistances": evaluated_resistances,
        "evaluated_supports": evaluated_supports
    }
    return results

if __name__ == "__main__":
    print("\n===== Support & Resistance Backtester ====")
    print("Please provide the following inputs for backtesting:")

    while True:
        currency_pair = input("Enter cryptocurrency pair (e.g., XRPUSDT, BTCUSDT): ").upper()
        if currency_pair:
            break
        else:
            print("Currency pair cannot be empty. Please try again.")

    while True:
        interval = input("Enter interval [30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M]: ")
        if interval in ['30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']:
            break
        else:
            print("Invalid interval. Please choose from the provided options.")

    while True:
        try:
            num_candles_input = int(input("How many candles of data? (minimum 100, max 1000): "))
            if 100 <= num_candles_input <= 1000:
                break
            else:
                print("Please enter a number of candles between 100 and 1000.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            train_ratio_input = float(input("Enter training data ratio (e.g., 0.7 for 70% training, 30% testing): "))
            if 0.1 <= train_ratio_input <= 0.9:
                break
            else:
                print("Please enter a training ratio between 0.1 and 0.9.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            deviation_threshold_input = float(input("Enter deviation threshold for evaluation (e.g., 0.01 for 1%): "))
            if 0 < deviation_threshold_input < 1:
                break
            else:
                print("Please enter a deviation threshold between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print(f"\nRunning backtest for {currency_pair} on {interval} interval with {num_candles_input} candles...")
    backtest_results = run_backtest(currency_pair, interval, num_candles_input, 
                                    train_ratio=train_ratio_input,
                                    deviation_threshold=deviation_threshold_input)

    if "error" in backtest_results:
        print(backtest_results["error"])
    else:
        print("\n--- Backtest Results ---")
        print(f"Total Profit/Loss: {backtest_results['total_profit_loss_percent']:.2f}%")
        print(f"Number of Trades: {backtest_results['num_trades']}")
        print(f"Winning Trades: {backtest_results['num_winning_trades']}")
        print(f"Win Rate: {backtest_results['win_rate']:.2f}%")
        
        print("\n--- Evaluated Resistance Levels (on Test Data) ---")
        if backtest_results['evaluated_resistances']:
            for r_eval in backtest_results['evaluated_resistances']:
                print(f"  Level: {r_eval['level']:.8f}, Touches: {r_eval['touches']}, Avg Dev: {r_eval['avg_deviation_percent']:.2f}%, Dev > Threshold: {r_eval['deviations_above_threshold']}")
        else:
            print("No resistance levels evaluated.")

        print("\n--- Evaluated Support Levels (on Test Data) ---")
        if backtest_results['evaluated_supports']:
            for s_eval in backtest_results['evaluated_supports']:
                print(f"  Level: {s_eval['level']:.8f}, Touches: {s_eval['touches']}, Avg Dev: {s_eval['avg_deviation_percent']:.2f}%, Dev > Threshold: {s_eval['deviations_above_threshold']}")
        else:
            print("No support levels evaluated.")


