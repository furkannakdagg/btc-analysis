import pandas as pd
import streamlit as st
import yfinance as yf
from indicators import *


df = pd.read_csv("data.csv")
btc = yf.download("BTC-USD", start="2014-09-17", end="2022-09-20")
# btc.to_csv("btc.csv", index=False)
# data.index = btc.index.map(str)
# btc.index = pd.to_datetime(btc.index)
# data["Close Time"] = pd.to_datetime(data["Close Time"])
pred = pd.read_csv("prediction.csv")
pred.columns = ["Precision"]
data = df.copy()
data.index = data["Close Time"].astype("datetime64[ns]")
data["Close Time"] = pd.to_datetime(data["Close Time"]).dt.date
data["Precision"] = pred["Precision"]


def intro():
    import streamlit as st

    st.write("# Welcome to Bitcoin Analysis! 👋")
    st.sidebar.success("Select an operation above.")

    st.markdown(
        """
        This project is built in order to analyze of BTC with its indicators and making prediction to BTC.

        **👈 Select an operation from the dropdown on the left** to reach the pages. 

        ### Want to learn more?

        - Check out the [history of BTC](https://en.wikipedia.org/wiki/Bitcoin)
        - Jump into [Binance](https://www.binance.com/) for trading.
        - In order to check out the situation of stocks market [Yahoo Finance](https://finance.yahoo.com) and [TradingView](https://tradingview.com/).
        
        ### Contributor
        
        - You can check my [Github](https://github.com/furkannakdagg), [Linkedin](https://www.linkedin.com/in/furkanakdag/), [Kaggle](https://www.kaggle.com/furkannakdagg) and [Medium](https://medium.com/@furkannakdagg) pages.
        - For Data Science courses, check [Veri Bilimi Okulu](https://www.veribilimiokulu.com) and [Miuul](https://www.miuul.com), the instructor [Vahit Keskin](https://www.linkedin.com/in/vahitkeskin/).
    """
    )


def indicators():

    st.sidebar.markdown("Intro")
    st.sidebar.title("Indicators📊")
    st.sidebar.markdown("Prediction")
    st.sidebar.markdown("Other Coins")
    st.title("BTC Indicators")

    options = ["MFI", "RSI", "BBANDS", "MA50", "MA100"]

    dropdown = st.multiselect("Pick your indicators", options)

    start = st.date_input("Start", value=pd.to_datetime("2014-09-17"))
    end = st.date_input("End", value=pd.to_datetime("2022-09-19"))

    # st.line_chart(data["Close"])
    interval = (data.loc[:, "Close Time"] < end) & (data.loc[:, "Close Time"] > start)
    st.line_chart(data[interval]["Close"])

    if len(dropdown) > 0:
        try:
            for i in dropdown:
                # st.line_chart(data.loc[:, i])
                st.line_chart(data[interval][i])

        except KeyError:
            if "BBANDS" in dropdown:
                # st.line_chart(data.loc[:, ["Close", "MiddleBand_30", "UpperBand_30", "LowerBand_30"]])
                st.line_chart(data[interval][["Close", "MiddleBand_30", "UpperBand_30", "LowerBand_30"]])
            if "MA50" in dropdown:
                # st.line_chart(data.loc[:, ["Close", "SMA_50"]])
                st.line_chart(data[interval][["Close", "SMA_50"]])
            if "MA100" in dropdown:
                # st.line_chart(data.loc[:, ["Close", "SMA_100"]])
                st.line_chart(data[interval][["Close", "SMA_100"]])

    # df = yf.download(dropdown, start, end)["Adj Close"]
    # st.line_chart(df)
    # st.line_chart(data.loc[(data.loc[:, "Close Time"] > start) & (data.loc[:, "Close Time"] < end), dropdown])
    # st.line_chart(data[(data.index > start) & (data.index < end), dropdown])


def precision():
    st.markdown("# Prediction of BTC")
    st.sidebar.markdown("Intro")
    st.sidebar.markdown("Indicators")
    st.sidebar.title("Prediction 📈")
    st.sidebar.markdown("Other Coins")
    st.write(
        """
        There are 4 different charts for this section.
        - 1. Original chart of BTC
        - 2. Result of TES(Triple Exponential Smoothing) Method
        - 3. Result of SARIMA Model
        - 4. Result of LGBM Model 
        """
    )
    st.line_chart(data["Close"])
    st.line_chart(pred["Precision"])


def other_coins():
    st.markdown("# Other Coins")
    st.sidebar.markdown("Intro")
    st.sidebar.markdown("Indicators")
    st.sidebar.markdown("Prediction")
    st.sidebar.title("Other Coins 💰")
    st.write(
        """
        You can see the other coins graphs on this page. Select the coins you want look from the menu.
        """
    )

    tickers = ["ETH-USD", "BNB-USD", "AVAX-USD", "TRY=X"]
    options = ["None", "MFI", "RSI", "BBANDS", "MA50", "MA100"]

    selection = st.selectbox("Pick the asset", tickers)
    dropdown = st.selectbox("Pick your indicators", options)


    start = st.date_input("Start", value=pd.to_datetime("2014-09-17"))
    end = st.date_input("End", value=pd.to_datetime("2022-09-19"))

    others = yf.download(selection, start=start, end=end)
    st.line_chart(others["Adj Close"])
    if dropdown == "RSI":
        others['RSI'] = rsi(others['Close'])
        st.line_chart(others['RSI'])

    elif dropdown == "MFI":
        others["MFI"] = mfi(others['High'], others['Low'], others['Close'], others['Volume'])
        st.line_chart(others["MFI"])

    elif dropdown == "BBANDS":
        others = BBANDS(others, [30])
        st.line_chart(others[["Close", "MiddleBand_30", "UpperBand_30", "LowerBand_30"]])

    elif dropdown == "MA50":
        others = roll_mean_features(others, [50])
        st.line_chart(others[["Close", "SMA_50"]])

    elif dropdown == "MA100":
        others = roll_mean_features(others, [100])
        st.line_chart(others[["Close", "SMA_100"]])


page_names_to_funcs = {
    "Intro": intro,
    "Indicators": indicators,
    "Prediction": precision,
    "Other Coins": other_coins


}

demo_name = st.sidebar.selectbox("Choose a section", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
