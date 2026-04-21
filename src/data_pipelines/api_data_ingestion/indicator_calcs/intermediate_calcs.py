"""
File: intermediate_calcs.py
Author: Drew Hill
This file is used for intermediate calculations used for technical indicator calculations.
"""
import pandas as pd

def daily_returns(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the day over day return for each ticker using the closing price series.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the daily return for each row grouped by ticker.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: (s / s.shift(1)) - 1)

def monday_open(df: pd.DataFrame) -> pd.Series:
    """
    Extracts the Monday open for each ticker week and propagates it across all rows in that week.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the Monday open repeated across each ticker-week.
    """
    monday_vals = df["Open"].where(df["Date"].dt.weekday == 0)

    return monday_vals.groupby([df["Ticker"], df["Date"].dt.to_period("W-FRI")]).transform("first")

def _friday_vals(df: pd.DataFrame, val_str: str) -> pd.Series:
    """
    Extracts Friday values from the specified column while leaving non-Friday rows as missing.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing Friday values for the requested column and missing values elsewhere.
    """
    return df[val_str].where(df["Date"].dt.weekday == 4)

def friday_open(df: pd.DataFrame) -> pd.Series:
    """
    Extracts the Friday open for each ticker week and propagates it across all rows in that week.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the Friday open repeated across each ticker-week.
    """
    return _friday_vals(df, "Open").groupby([df["Ticker"], df["Date"].dt.to_period("W-FRI")]).transform("first")

def friday_close(df: pd.DataFrame) -> pd.Series:
    """
    Extracts the Friday close for each ticker week and propagates it across all rows in that week.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the Friday close repeated across each ticker-week.
    """
    return _friday_vals(df, "Close").groupby([df["Ticker"], df["Date"].dt.to_period("W-FRI")]).transform("first")

def friday_volume(df: pd.DataFrame) -> pd.Series:
    """
    Extracts the Friday volume for each ticker week and propagates it across all rows in that week.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the Friday volume repeated across each ticker-week.
    """
    return _friday_vals(df, "Volume").groupby([df["Ticker"], df["Date"].dt.to_period("W-FRI")]).transform("first")

def weekly_high(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the maximum high price within each ticker-week and propagates it across that week.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the weekly high for each row's ticker-week.
    """
    return df["High"].groupby([df["Ticker"], df["Date"].dt.to_period("W-FRI")]).transform("max")

def weekly_low(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the minimum low price within each ticker-week and propagates it across that week.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the weekly low for each row's ticker-week.
    """
    return df["Low"].groupby([df["Ticker"], df["Date"].dt.to_period("W-FRI")]).transform("min")

def weekly_avg_volume(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the mean trading volume within each ticker week and propagates it across that week.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the weekly average volume for each row's ticker-week.
    """
    return df["Volume"].groupby([df["Ticker"], df["Date"].dt.to_period("W-FRI")]).transform("mean")

def prev_friday_close(df: pd.DataFrame) -> pd.Series:
    """
    Carries forward the most recent Friday close for each ticker and shifts it to represent the prior Friday close.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the previous Friday close for each ticker.
    """
    f_close = df["Close"].where(df["Date"].dt.weekday == 4)

    return f_close.groupby(df["Ticker"]).ffill().shift(1)

def price_change(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the absolute day over day change in closing price for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the day-over-day close price change for each ticker.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: s.diff())

def gain(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the positive portion of each ticker's day over day closing price change.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing positive close-price changes and zeros otherwise.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: s.diff().clip(lower=0))

def loss(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the absolute value of the negative portion of each ticker's day over day closing price change.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing absolute negative close-price changes and zeros otherwise.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: -s.diff().clip(upper=0))

def avg_gain(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates the rolling average gain over the specified window for each ticker.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the average gain.
    :returns: A pandas Series containing the rolling average gain for each ticker.
    """
    return gain(df).groupby(df["Ticker"]).transform(lambda s: s.rolling(window).mean())

def avg_loss(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates the rolling average loss over the specified window for each ticker.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the average loss.
    :returns: A pandas Series containing the rolling average loss for each ticker.
    """
    return loss(df).groupby(df["Ticker"]).transform(lambda s: s.rolling(window).mean())

def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index for each ticker using rolling average gains and losses.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the rsi.
    :returns: A pandas Series containing the RSI value for each row grouped by ticker.
    """
    rs = avg_gain(df, window) / avg_loss(df, window)
    return 100 - (100 / (1 + rs))

def ema_12(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the 12 period exponential moving average of close for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the 12-period EMA of close for each ticker.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())

def ema_26(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the 26 period exponential moving average of close for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the 26-period EMA of close for each ticker.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())

def ema_50(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the 50 period exponential moving average of close for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the 50-period EMA of close for each ticker.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: s.ewm(span=50, adjust=False).mean())

def sma_20(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the 20-period simple moving average of close for each ticker.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the 20-period SMA of close for each ticker.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(20).mean())

def rolling_std_n(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculates the rolling standard deviation of close over the specified window for each ticker.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the rolling standard deviation.
    :returns: A pandas Series containing the rolling standard deviation of close for each ticker.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(window).std())

def upper_band(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculates the upper Bollinger Band as the rolling mean plus two rolling standard deviations.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the upper band.
    :returns: A pandas Series containing the upper Bollinger Band for each ticker.
    """
    sma_n = sma_20(df) if window == 20 else df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(window).mean())
    std_n = rolling_std_n(df, window)

    return sma_n + 2 * std_n

def lower_band(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculates the lower Bollinger Band as the rolling mean minus two rolling standard deviations.
    :param df: The dataframe that contains all the price data.
    :param window: The window that will be used to calculate the lower band.
    :returns: A pandas Series containing the lower Bollinger Band for each ticker.
    """
    sma_n = sma_20(df) if window == 20 else df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(window).mean())
    std_n = rolling_std_n(df, window)

    return sma_n - 2 * std_n

def prev_close(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the previous day's close for each ticker by shifting the close series by one row.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the previous close for each ticker.
    """
    return df.groupby("Ticker")["Close"].transform(lambda s: s.shift(1))

def tr(df: pd.DataFrame) -> pd.Series:
    """
    Calculates true range for each row as the maximum of high - low, high - prev_close absolute change, and
    low - prev_close absolute change.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the true range for each row.
    """
    prev_c = prev_close(df)
    hl = df["High"] - df["Low"]
    hp = (df["High"] - prev_c).abs()
    lp = (df["Low"] - prev_c).abs()

    return pd.concat([hl, hp, lp], axis=1).max(axis=1)

def typical_price(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the typical price for each row as the average of high, low, and close.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing the typical price for each row.
    """
    return (df["High"] + df["Low"] + df["Close"]) / 3

def raw_money_flow(df: pd.DataFrame) -> pd.Series:
    """
    Calculates raw money flow as typical price multiplied by trading volume.
    :param df: The dataframe that contains all the price data.
    :returns: A pandas Series containing raw money flow for each row.
    """
    return typical_price(df) * df["Volume"]

def prev_quarter_revenue(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the previous quarter's revenue for each ticker.
    :param df: The dataframe that contains all the fundamental data.
    :returns: A pandas Series containing the previous quarter revenue for each ticker.
    """
    quarterly = (df[["Ticker", "Year", "Quarter", "Revenues"]].drop_duplicates(
            subset=["Ticker", "Year", "Quarter"]).sort_values(["Ticker", "Year", "Quarter"]))

    quarterly["PrevRevenue"] = quarterly.groupby("Ticker")["Revenues"].shift(1)

    return (df.merge(quarterly[["Ticker", "Year", "Quarter", "PrevRevenue"]],
                 on=["Ticker", "Year", "Quarter"], how="left")["PrevRevenue"].reset_index(drop=True))
