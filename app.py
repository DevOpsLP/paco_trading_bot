import pandas as pd
import numpy as np
import json
import os
import requests
from typing import Dict, List

def fetch_binance_futures_candles(symbols, intervals, total_candles=5000, chunk_size=1000):
    base_url = 'https://fapi.binance.com/fapi/v1/klines'
    
    for interval in intervals:
        all_candles = {}
        
        for symbol in symbols:
            all_symbol_candles = []
            end_time = None
            
            while len(all_symbol_candles) < total_candles:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': min(chunk_size, total_candles - len(all_symbol_candles))
                }
                if end_time:
                    params['endTime'] = end_time
                
                response = requests.get(base_url, params=params)
                data = response.json()
                
                if not data or 'code' in data:
                    # Handle error if data is empty or an error code is returned
                    print(f"Error fetching data for symbol {symbol} at interval {interval}: {data}")
                    break
                
                all_symbol_candles.extend(data)
                end_time = data[-1][6]  # Set the end time for the next iteration to the close time of the last candle in this batch
                
                if len(data) < chunk_size:
                    # If we received less data than requested, it means we have reached the end
                    break
            
            all_candles[symbol] = all_symbol_candles
        
        # Save to a file named with the interval
        filename = f'candles_{interval}.json'
        with open(filename, 'w') as f:
            json.dump(all_candles, f, indent=4)
        
        print(f"Data for interval {interval} saved to {filename}")

# Example usage:
symbols = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'MATICUSDT', 'LTCUSDT',
    'LINKUSDT', 'BCHUSDT', 'XLMUSDT', 'VETUSDT', 'FILUSDT',
    'TRXUSDT', 'EOSUSDT', 'ATOMUSDT', 'XTZUSDT', 'AXSUSDT'
]



intervals = ['5m', '30m', '1h', '4h', '1d', '1w']

def read_candles(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_dmi_adx(df, period=14):
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    
    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                         np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                         np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    
    df['TR_smooth'] = df['TR'].rolling(window=period, min_periods=1).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=period, min_periods=1).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period, min_periods=1).sum()
    
    for i in range(period, len(df)):
        df.iloc[i, df.columns.get_loc('TR_smooth')] = df.iloc[i-1, df.columns.get_loc('TR_smooth')] - (df.iloc[i-1, df.columns.get_loc('TR_smooth')] / period) + df.iloc[i, df.columns.get_loc('TR')]
        df.iloc[i, df.columns.get_loc('+DM_smooth')] = df.iloc[i-1, df.columns.get_loc('+DM_smooth')] - (df.iloc[i-1, df.columns.get_loc('+DM_smooth')] / period) + df.iloc[i, df.columns.get_loc('+DM')]
        df.iloc[i, df.columns.get_loc('-DM_smooth')] = df.iloc[i-1, df.columns.get_loc('-DM_smooth')] - (df.iloc[i-1, df.columns.get_loc('-DM_smooth')] / period) + df.iloc[i, df.columns.get_loc('-DM')]
    
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    
    df['ADX'] = df['DX'].rolling(window=period, min_periods=1).mean()
    
    for i in range(period * 2, len(df)):
        df.iloc[i, df.columns.get_loc('ADX')] = (df.iloc[i-1, df.columns.get_loc('ADX')] * (period - 1) + df.iloc[i, df.columns.get_loc('DX')]) / period
    
    return df[['+DI', '-DI', 'DX', 'ADX']].add_suffix('_5m')

def calculate_dmi_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])

    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )

    df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                         np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                         np.maximum(df['low'].shift(1) - df['low'], 0), 0)

    df['TR_smooth'] = df['TR'].rolling(window=period, min_periods=1).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=period, min_periods=1).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period, min_periods=1).sum()

    for i in range(period, len(df)):
        df.iloc[i, df.columns.get_loc('TR_smooth')] = df.iloc[i-1, df.columns.get_loc('TR_smooth')] - (df.iloc[i-1, df.columns.get_loc('TR_smooth')] / period) + df.iloc[i, df.columns.get_loc('TR')]
        df.iloc[i, df.columns.get_loc('+DM_smooth')] = df.iloc[i-1, df.columns.get_loc('+DM_smooth')] - (df.iloc[i-1, df.columns.get_loc('+DM_smooth')] / period) + df.iloc[i, df.columns.get_loc('+DM')]
        df.iloc[i, df.columns.get_loc('-DM_smooth')] = df.iloc[i-1, df.columns.get_loc('-DM_smooth')] - (df.iloc[i-1, df.columns.get_loc('-DM_smooth')] / period) + df.iloc[i, df.columns.get_loc('-DM')]

    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].rolling(window=period, min_periods=1).mean()

    for i in range(period * 2, len(df)):
        df.iloc[i, df.columns.get_loc('ADX')] = (df.iloc[i-1, df.columns.get_loc('ADX')] * (period - 1) + df.iloc[i, df.columns.get_loc('DX')]) / period

    return df[['+DI', '-DI', 'DX', 'ADX']].add_suffix('_5m')

def get_levels(data: List[List[float]], interval: str) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def process_strategy(data: Dict[str, Dict[str, List[List[float]]]], symbol: str):
    print(f"Processing strategy for {symbol}")
    
    # Parameters
    dmi_filter_enabled = False
    dmi_plus_level = 5
    adx_level = 10
    margin_profit = 0.00001  # 0.001%
    margin_stop_loss = 0.0015  # 0.15%
    filter_gap_percentage = 0.02  # 2%
    minim_profit = 0.003  # 0.3%
    candle_bounce_filter_bars = 4

    # Get data for different intervals
    intervals = ['5m', '30m', '1h', '4h', '1d', '1w']
    dfs = {}
    for interval in intervals:
        if interval in data and symbol in data[interval]:
            dfs[interval] = get_levels(data[interval][symbol], interval)
        else:
            print(f"Warning: {interval} data not found for {symbol}")

    if '5m' not in dfs:
        print(f"Error: 5m data is required but not available for {symbol}")
        return []

    # Calculate DMI and ADX for 5m interval
    dmi_adx = calculate_dmi_adx(dfs['5m'])
    dfs['5m'] = pd.concat([dfs['5m'], dmi_adx], axis=1)

    # Initialize variables
    last_short_price = None
    last_objective_price = None
    fixed_stop_loss_price = None
    fixed_low_4hours = None
    fixed_low_day = None
    prev_high_30min = None
    prev_low_30min = None
    prev_high_1hour = None
    prev_low_1hour = None

    signals = []

    for i in range(len(dfs['5m'])):
        current_5m = dfs['5m'].iloc[i]
        current_30m = dfs['30m'].loc[dfs['30m'].index <= current_5m.name].iloc[-1]
        current_1h = dfs['1h'].loc[dfs['1h'].index <= current_5m.name].iloc[-1]
        current_4h = dfs['4h'].loc[dfs['4h'].index <= current_5m.name].iloc[-1]
        current_1d = dfs['1d'].loc[dfs['1d'].index <= current_5m.name].iloc[-1]
        current_1w = dfs['1w'].loc[dfs['1w'].index <= current_5m.name].iloc[-1]

        # Convert string values to float
        current_5m = current_5m.astype(float)
        current_30m = current_30m.astype(float)
        current_1h = current_1h.astype(float)
        current_4h = current_4h.astype(float)
        current_1d = current_1d.astype(float)
        current_1w = current_1w.astype(float)

        # Calculate percentage changes
        change_high_30min = 0 if prev_high_30min is None else abs((current_30m['high'] - prev_high_30min) / prev_high_30min)
        change_low_30min = 0 if prev_low_30min is None else abs((current_30m['low'] - prev_low_30min) / prev_low_30min)
        change_high_1hour = 0 if prev_high_1hour is None else abs((current_1h['high'] - prev_high_1hour) / prev_high_1hour)
        change_low_1hour = 0 if prev_low_1hour is None else abs((current_1h['low'] - prev_low_1hour) / prev_low_1hour)

        # Update previous values
        prev_high_30min = current_30m['high']
        prev_low_30min = current_30m['low']
        prev_high_1hour = current_1h['high']
        prev_low_1hour = current_1h['low']

        # Check conditions
        condition_crossunder = current_30m['high'] < current_1h['high'] and dfs['30m'].iloc[i-1]['high'] >= dfs['1h'].iloc[i-1]['high']
        condition_dmi_filter = not dmi_filter_enabled or (current_5m['+DI_5m'] > dmi_plus_level and current_5m['ADX_5m'] > adx_level)
        condition_change_high_30min = change_high_30min < filter_gap_percentage
        condition_change_low_30min = change_low_30min < filter_gap_percentage
        condition_change_high_1hour = change_high_1hour < filter_gap_percentage
        condition_change_low_1hour = change_low_1hour < filter_gap_percentage

        short_condition = (condition_crossunder and condition_dmi_filter and
                           condition_change_high_30min and condition_change_low_30min and
                           condition_change_high_1hour and condition_change_low_1hour)

        if short_condition:
            last_short_price = float(current_5m['close'])
            stop_loss_price = float(current_1h['high'])

            if stop_loss_price < last_short_price:
                stop_loss_price = float(current_4h['high'])

            stop_loss_price *= (1 + margin_stop_loss)

            fixed_stop_loss_price = stop_loss_price
            fixed_low_4hours = float(current_4h['low'])
            fixed_low_day = float(current_1d['low'])

            if current_5m['close'] > current_30m['low']:
                last_objective_price = float(current_30m['low'])
            else:
                last_objective_price = float(current_1h['low'])

            adjusted_objective_price = last_objective_price * (1 - margin_profit)

            potential_profit_percent = (last_short_price - adjusted_objective_price) / last_short_price

            if potential_profit_percent >= minim_profit:
                candle_bounce_condition = True
                for j in range(candle_bounce_filter_bars):
                    if float(dfs['5m'].iloc[i-j]['low']) <= adjusted_objective_price:
                        candle_bounce_condition = False
                        break

                if candle_bounce_condition:
                    signals.append({
                        'timestamp': current_5m.name,
                        'action': 'SHORT',
                        'price': last_short_price,
                        'stop_loss': fixed_stop_loss_price,
                        'take_profit': adjusted_objective_price
                    })

        # Check for "ROTURA MINIMO SEMANAL"
        if (current_5m['close'] < current_30m['low'] and
            current_5m['close'] < current_1h['low'] and
            current_5m['close'] < current_4h['low'] and
            current_5m['close'] < current_1d['low'] and
            current_5m['close'] < current_1w['low']):
            signals.append({
                'timestamp': current_5m.name,
                'action': 'WEEKLY LOW BREAK',
                'price': float(current_5m['close'])
            })

    return signals

# Main execution
if __name__ == "__main__":
    folder_path = 'candles'  # Replace with your actual folder path
    intervals = ['5m', '30m', '1h', '4h', '1d', '1w']

    all_data = {}
    for interval in intervals:
        file_path = os.path.join(folder_path, f'candles_{interval}.json')
        try:
            interval_data = read_candles(file_path)
            all_data[interval] = interval_data
            print(f"Successfully loaded data for {interval}")
            print(f"Symbols in {interval} data:", list(interval_data.keys()))
        except FileNotFoundError:
            print(f"Warning: File not found for interval {interval}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file for interval {interval}")

    # Process strategy for BTCUSDT
    symbol = 'BTCUSDT'
    results = process_strategy(all_data, symbol)
    
    print(f"\nSignals for {symbol}:")
    for signal in results:
        print(signal)