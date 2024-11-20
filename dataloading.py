import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from typing import Tuple


def convert_date(df: pd.DataFrame) -> pd.DataFrame:
    # Rename the first column to 'Date'
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    return df


def load_data(
    factors:str="5_factor",
    symbol:str="AAPL",
    pred_len:int=0) -> pd.DataFrame:

    # Import Factor Data
    factor_path = os.path.join("data/fama_french", f"{factors}.csv")
    factor_data = convert_date(pd.read_csv(factor_path, skiprows=3))

    print(factor_data.head())

    # Import S&P 500 data
    sp = pd.read_csv("data/sp500/sp500_stocks.csv")

    # Select AAPL stock
    stock_data = sp[sp['Symbol'] == symbol].copy()

    # Calculate daily returns using Adj Close
    stock_data['Daily Returns'] = stock_data['Adj Close'].pct_change()
    stock_data.dropna(inplace=True) # Get rid of first row (NaN)
    stock_data['Date'] = pd.to_datetime(stock_data['Date']) # Ensure date is in datetime format

    # Align factor data with stock data
    start_date = stock_data['Date'].iloc[0]
    end_date = min(
        stock_data['Date'].iloc[-1],
        factor_data['Date'].iloc[-1]
    )

    # Filter dataframes to only include dates between start_date and end_date
    stock_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
    factor_data = factor_data[(factor_data['Date'] >= start_date) & (factor_data['Date'] <= end_date)]

    # Join the two dataframes on 'Date'
    full_data = pd.merge(stock_data, factor_data, on='Date', how='inner')

    return full_data


if __name__ == "__main__":
    factors = "5_factor"
    symbol = "AAPL"
    print(load_data(factors, symbol).head())
