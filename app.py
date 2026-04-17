import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================
st.set_page_config(page_title="Monitor IA Hedge Fund", layout="wide")
st_autorefresh(interval=300 * 1000, key="refresh")

# ================= INDICADORES COMPLETOS =================
def add_indicators(df):

    # ===== MÉDIAS =====
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA9"] = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()

    # ===== RSI =====
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    # ===== MACD =====
    exp12 = df["Close"].ewm(span=12).mean()
    exp26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp12 - exp26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # ===== BOLLINGER =====
    df["BB_mid"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_up"] = df["BB_mid"] + 2 * std
    df["BB_low"] = df["BB_mid"] - 2 * std

    # ===== VWAP =====
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    # ===== OBV =====
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # ===== ATR =====
    tr = np.maximum(df["High"] - df["Low"],
                    np.maximum(abs(df["High"] - df["Close"].shift()),
                               abs(df["Low"] - df["Close"].shift())))
    df["ATR"] = tr.rolling(14).mean()

    # ===== ESTOCÁSTICO =====
    low_min = df["Low"].rolling(14).min()
    high_max = df["High"].rolling(14).max()
    df["STOCH"] = 100 * (df["Close"] - low_min) / (high_max - low_min)

    # ===== ICHIMOKU =====
    df["tenkan"] = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
    df["kijun"] = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2

    # ===== FIBONACCI =====
    high = df["High"].max()
    low = df["Low"].min()
    df["FIB_0"] = low
    df["FIB_38"] = low + (high - low) * 0.382
    df["FIB_61"] = low + (high - low) * 0.618
    df["FIB_100"] = high

    return df

# ================= DOWNLOAD =================
@st.cache_data(ttl=300)
def get_data(ticker):
    df = yf.download(ticker, period="2y")
    return df

# ================= GRÁFICO PROFISSIONAL =================
def plot_chart(df):

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )

    # ===== CANDLE =====
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    ), row=1, col=1)

    # MÉDIAS + VWAP
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"], name="EMA9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA21"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP"), row=1, col=1)

    # BOLLINGER
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB Up"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_low"], name="BB Low"), row=1, col=1)

    # ===== RSI =====
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)

    # ===== MACD =====
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"]), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"]), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"]), row=3, col=1)

    # ===== VOLUME =====
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"]), row=4, col=1)

    fig.update_layout(height=900, template="plotly_dark")
    return fig

# ================= APP =================
st.title("🔥 Monitor Trader - Nível Hedge Fund")

ticker = st.text_input("Ativo", "PETR4.SA")

if ticker:
    df = get_data(ticker)

    if not df.empty:
        df = add_indicators(df)

        fig = plot_chart(df)
        st.plotly_chart(fig, use_container_width=True)

        # ===== SINAIS =====
        last = df.iloc[-1]

        st.subheader("📊 Leitura Inteligente")

        if last["RSI"] < 30 and last["MACD"] > last["MACD_signal"]:
            st.success("COMPRA FORTE")
        elif last["RSI"] > 70:
            st.error("VENDA FORTE")
        else:
            st.warning("NEUTRO")

        col1, col2, col3 = st.columns(3)
        col1.metric("RSI", round(last["RSI"], 2))
        col2.metric("ATR", round(last["ATR"], 2))
        col3.metric("OBV", int(last["OBV"]))

    else:
        st.error("Erro ao carregar dados")
