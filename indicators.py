import numpy as np
import pandas as pd


# RSI
def rsi(close, periods=14):
    close_delta = close.diff()
    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi


def BBANDS(dataframe, windows):
    for window in windows:
        MA = dataframe.Close.rolling(window).mean()
        SD = dataframe.Close.rolling(window).std()
        dataframe['MiddleBand_' + str(window)] = MA
        dataframe['UpperBand_' + str(window)] = MA + (2 * SD)
        dataframe['LowerBand_' + str(window)] = MA - (2 * SD)
    return dataframe


# Simple Moving Average
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe["SMA_" + str(window)] = dataframe["Close"].transform(
            lambda x: x.shift(1).rolling(window=window, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


# Money Flow Index - MFI
def gain(x):
    return ((x > 0) * x).sum()


def loss(x):
    return ((x < 0) * x).sum()


# Calculate money flow index
def mfi(high, low, close, volume, n=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign
    mf_avg_gain = signed_mf.rolling(n).apply(gain, raw=True)
    mf_avg_loss = signed_mf.rolling(n).apply(loss, raw=True)
    return (100 - (100 / (1 + (mf_avg_gain / abs(mf_avg_loss))))).to_numpy()


def MACD(df, n_fast, n_slow, n_smooth):
    data = df['Close']
    fastEMA = data.ewm(span=n_fast, min_periods=n_slow).mean()
    slowEMA = data.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = pd.Series(fastEMA - slowEMA, name='MACD')
    MACDsig = pd.Series(MACD.ewm(span=n_smooth, min_periods=n_smooth).mean(), name='MACDsig')
    MACDhist = pd.Series(MACD - MACDsig, name='MACDhist')
    df = df.join(MACD)
    df = df.join(MACDsig)
    df = df.join(MACDhist)

    return df

