import pandas as pd
import streamlit as st
import yfinance as yf
from indicators import *
from PIL import Image

qr = Image.open('QR.png')
btc_logo = Image.open('btc.png')
df = pd.read_csv("data.csv")

test_one = pd.read_csv("test_one.csv", index_col=[0])
test_one.index = test_one.index.astype("datetime64[ns]")
test_one.rename(columns={"Close": "Prediction"}, inplace=True)

tes_prediction = pd.read_csv("tes_prediction_small.csv", index_col=[0])
tes_prediction.index = tes_prediction.index.astype("datetime64[ns]")
tes_prediction.rename(columns={"Close": "Prediction"}, inplace=True)
concat_tes_test = pd.concat([test_one, tes_prediction])  # for better plotting

sarima_prediction = pd.read_csv("sarima_prediction.csv", index_col=[0])
sarima_prediction.index = sarima_prediction.index.astype("datetime64[ns]")
sarima_prediction.rename(columns={"predicted_mean": "Prediction"}, inplace=True)
concat_sarima_test = pd.concat([test_one, sarima_prediction], ignore_index=True)  # for better plotting

pred = pd.read_csv("prediction.csv")
pred["Close Time"] = pred["Close Time"].astype("datetime64[ns]")
pred.set_index("Close Time", inplace=True)
data = df.copy()
data.index = data["Close Time"].astype("datetime64[ns]")
data["Close Time"] = pd.to_datetime(data["Close Time"]).dt.date
data["Prediction"] = pred["Prediction"]
data["MACD_"] = data["MACD"]
data.drop("MACD", axis=1, inplace=True)

st.set_page_config(
    page_title="BTC Prediction System",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
)


def intro():
    import streamlit as st

    col1, col2 = st.columns(2)
    with col1:
        st.image(qr, width=125)
    with col2:
        st.markdown(
            "[![Foo](https://img.icons8.com/material-outlined/96/000000/github.png)](https://github.com/furkannakdagg/btc-analysis)")
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
    st.markdown("# BTC Indicators")

    options = ["RSI", "MACD", "BBANDS", "MFI", "MA50", "MA100"]

    dropdown = st.multiselect("Pick your indicators", options)

    start = st.date_input("Start", value=pd.to_datetime("2014-09-17"))
    end = st.date_input("End", value=pd.to_datetime("2022-09-19"))

    interval = (data.loc[:, "Close Time"] < end) & (data.loc[:, "Close Time"] > start)
    st.line_chart(data[interval]["Close"])

    if len(dropdown) > 0:
        try:
            for i in dropdown:
                st.line_chart(data[interval][i])

        except KeyError:
            if "BBANDS" in dropdown:
                st.line_chart(data[interval][["Close", "MiddleBand_30", "UpperBand_30", "LowerBand_30"]])
            if "MA50" in dropdown:
                st.line_chart(data[interval][["Close", "SMA_50"]])
            if "MA100" in dropdown:
                st.line_chart(data[interval][["Close", "SMA_100"]])
            if "MACD" in dropdown:
                st.line_chart(data[interval][["MACD_", "MACDsig"]])


def prediction():
    st.markdown("# Prediction of BTC")
    st.write(
        """
        There are 4 different charts for this section.
        - 1. Original chart of BTC
        - 2. Result of TES(Triple Exponential Smoothing) Method
        - 3. Result of SARIMA Model
        - 4. Result of LGBM Model 
        """
    )

    inc = "Expectation: Increase 📈"
    dec = "Expectation: Decrease 📉"

    def result(df):
        exp_list = []
        last_price = test_one["Prediction"][len(test_one) - 1]
        for i in range(len(df)):
            if df["Prediction"][i:i + 1][0] > last_price:
                st.success(str(df.index[i].date()) + " " + inc)
                last_price = df["Prediction"][i:i + 1][0]
                exp_list.append(1)
            else:
                st.error(str(df.index[i].date()) + " " + dec)
                last_price = df["Prediction"][i:i + 1][0]
                exp_list.append(0)
        return exp_list

    st.markdown("### Original Chart")
    st.line_chart(data["2020-11-01":].loc[:, "Close"])

    st.markdown("### LGBM Prediction")
    st.line_chart(pred["2022-05-08":])
    exp_lgbm = result(pred["2022-09-19":])
    df_lgbm = pd.DataFrame(exp_lgbm, columns=["lgbm_exp"], index=pred["2022-09-19":].index)

    st.markdown("### TES Prediction")
    st.line_chart(concat_tes_test["Prediction"])
    exp_tes = result(tes_prediction["2022-09-20":"2022-09-26"])
    df_tes = pd.DataFrame(exp_tes, columns=["tes_exp"], index=tes_prediction["2022-09-20":"2022-09-26"].index)

    st.markdown("### SARIMA Prediction")
    st.line_chart(sarima_prediction["Prediction"])
    exp_sar = result(sarima_prediction["2022-09-20":])
    df_sar = pd.DataFrame(exp_sar, columns=["sar_exp"], index=sarima_prediction["2022-09-20":].index)

    st.markdown("")
    st.markdown("")
    st.markdown("## Final Result")
    st.write(
        """
        This section calculates the overall result of the prediction. The prediction is based on the algorithm weights.
        
        Weights are considered with the following percentages:
        - LGBM: 40%
        - TES: 30%
        - SARIMA: 30%
        """
    )
    fin_res = pd.concat([df_lgbm, df_tes, df_sar], axis=1)
    for i in range(len(fin_res)):
        if i == 0:
            calc = fin_res.iloc[i, :]["lgbm_exp"]
        elif i < 4:
            calc = fin_res.iloc[i, :]["lgbm_exp"] * 0.4 + fin_res.iloc[i, :]["tes_exp"] * 0.3 + fin_res.iloc[i, :][
                "sar_exp"] * 0.3
        else:
            calc = fin_res.iloc[i, :]["tes_exp"] * 0.49 + fin_res.iloc[i, :]["sar_exp"] * 0.51

        if calc > 0.5:
            st.success(str(fin_res.index[i].date()) + " " + inc)

        else:
            st.error(str(fin_res.index[i].date()) + " " + dec)


def other_coins():
    st.markdown("# Other Coins")
    st.write(
        """
        You can see the other coins graphs on this page. Select the coins you want look from the menu.
        """
    )

    tickers = ["ETH-USD", "BNB-USD", "AVAX-USD", "TRY=X"]
    options = ["None", "RSI", "MACD", "BBANDS", "MFI", "MA50", "MA100"]

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

    elif dropdown == "MACD":
        others = MACD(others, 12, 26, 9)
        st.line_chart(others[["MACD", "MACDsig"]])


page_names_to_funcs = {
    "Intro 👋": intro,
    "Indicators 📊": indicators,
    "Prediction 📈": prediction,
    "Other Coins 💰": other_coins

}

demo_name = st.sidebar.selectbox("Choose a section", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
