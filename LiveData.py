import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

# Create output folder for combined data
combined_folder = "LiveData"
os.makedirs(combined_folder, exist_ok=True)

def unix_to_human(timestamp):
    """Convert Unix timestamp to human-readable format with timezone-aware UTC."""
    return datetime.fromtimestamp(int(timestamp) / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

def fetch_okx_open_interest(symbol, interval, start_time, end_time):
    """Fetch OKX Open Interest data."""
    url = "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history"
    all_data = []
    current_date = start_time

    while current_date <= end_time:
        next_date = current_date + timedelta(hours=24)
        start_ts = int(current_date.timestamp() * 1000)
        end_ts = int(next_date.timestamp() * 1000)

        params = {
            "instId": symbol,
            "period": interval,
            "begin": str(start_ts),
            "end": str(end_ts),
            "limit": "24",
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'data' in data and data['data']:
                all_data.extend(data['data'])
            else:
                print(f"No OKX data for {symbol} ({interval}) from {current_date} to {next_date}.")
        except Exception as e:
            print(f"Error fetching OKX data for {symbol} ({interval}): {e}")
        
        current_date = next_date
        time.sleep(0.2)

    if all_data:
        all_data.sort(key=lambda x: x[0])
    return all_data

def fetch_binance_futures_data(symbol, interval, start_time, end_time):
    """Fetch Binance Futures data."""
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/klines"
    all_data = []
    current_time = start_time

    while current_time < end_time:
        next_time = min(current_time + timedelta(days=15), end_time)
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(current_time.timestamp() * 1000),
            "endTime": int(next_time.timestamp() * 1000),
            "limit": 1000,
        }
        try:
            response = requests.get(base_url + endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            if data:
                all_data.extend(data)
            else:
                print(f"No Binance data for {symbol} ({interval}) from {current_time} to {next_time}.")
        except Exception as e:
            print(f"Error fetching Binance data for {symbol} ({interval}): {e}")
        current_time = next_time
        time.sleep(0.1)

    return all_data

def synchronize_and_save(symbol, interval, binance_data, okx_data):
    """Synchronize Binance and OKX data by common dates/hours and save to a combined CSV."""
    file_name = f"{symbol.replace('USDT', '')}_{interval}.csv"
    file_path = os.path.join(combined_folder, file_name)

    # Check if the file exists
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        existing_df["Date"] = pd.to_datetime(existing_df["Date"])
        last_date = existing_df["Date"].max()
        print(f"Existing file found for {symbol} ({interval}). Last Date: {last_date}")
    else:
        existing_df = pd.DataFrame()
        last_date = None

    # Process Binance Data
    if binance_data:
        binance_df = pd.DataFrame(binance_data, columns=[
            "Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time", "Quote Asset Volume", "_4", "_5", "_6", "_7"
        ])
        binance_df["Open Time"] = pd.to_datetime(binance_df["Open Time"], unit='ms')
        binance_df["Date"] = binance_df["Open Time"].dt.date if interval == "1d" else binance_df["Open Time"].dt.floor(interval)
        # Keep only the desired "Quote Asset Volume" and rename it to "Volume"
        binance_df.rename(columns={"Quote Asset Volume": "Volume"}, inplace=True)
        binance_df = binance_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        binance_df.sort_values("Date", inplace=True)
        binance_df["Date"] = pd.to_datetime(binance_df["Date"])
    else:
        print(f"No Binance data for {symbol} ({interval}).")
        return

    # Process OKX Data
    if okx_data:
        okx_df = pd.DataFrame(okx_data, columns=["Time", "_1", "_2", "Open Interest (USD)"])
        okx_df["Open Time"] = pd.to_datetime(okx_df["Time"].astype(int), unit='ms')
        okx_df["Date"] = okx_df["Open Time"].dt.date if interval == "1D" else okx_df["Open Time"].dt.floor(interval.lower())
        okx_df = okx_df[["Date", "Open Interest (USD)"]]
        okx_df.sort_values("Date", inplace=True)
        okx_df["Date"] = pd.to_datetime(okx_df["Date"])
    else:
        print(f"No OKX data for {symbol} ({interval}).")
        return

    # Filter for Common Dates
    common_dates = set(binance_df["Date"]).intersection(set(okx_df["Date"]))
    binance_df = binance_df[binance_df["Date"].isin(common_dates)]
    okx_df = okx_df[okx_df["Date"].isin(common_dates)]

    # Merge on 'Date'
    combined_df = pd.merge(binance_df, okx_df, on="Date", how="inner")

    if last_date:
        combined_df = combined_df[combined_df["Date"] > last_date]

    if combined_df.empty:
        print(f"No new data for {symbol} ({interval}).")
        return

    if not existing_df.empty:
        combined_df = pd.concat([existing_df, combined_df]).drop_duplicates().sort_values("Date")

    # Save to CSV without the first "Volume"
    combined_df.to_csv(file_path, index=False)
    print(f"Data for {symbol} ({interval}) saved to {file_path}")


def main():
    # Define date range
    end_date = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start_date = end_date - timedelta(hours=1440)

    print(f"Fetching data from {start_date} to {end_date}...")

    # Define symbols and intervals
    binance_symbols = ["BTCUSDT"]
    okx_symbols = ["BTC-USDT-SWAP"]
    binance_intervals = ["1d", ]
    okx_intervals = ["1D"]

    for binance_symbol, okx_symbol in zip(binance_symbols, okx_symbols):
        for binance_interval, okx_interval in zip(binance_intervals, okx_intervals):
            print(f"Processing {binance_symbol} ({binance_interval})...")
            binance_data = fetch_binance_futures_data(binance_symbol, binance_interval, start_date, end_date)
            okx_data = fetch_okx_open_interest(okx_symbol, okx_interval, start_date, end_date)
            synchronize_and_save(binance_symbol, binance_interval, binance_data, okx_data)

if __name__ == "__main__":
    main()
