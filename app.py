import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================
st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300000, key="refresh")

# ================= UTIL =================
def calcular_graham(lpa, vpa):
    return np.sqrt(22.5 * lpa * vpa) if lpa > 0 and vpa > 0 else 0.0

def calcular_rsi_series(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calcular_rsi(close):
    return float(calcular_rsi_series(close).iloc[-1])

def calcular_score_value(info):
    score = 0
    if 0 < info.get("trailingPE", 99) < 15: score += 1
    if 0 < info.get("priceToBook", 99) < 1.5: score += 1
    if (info.get("dividendYield", 0) or 0) * 100 > 5: score += 1
    if (info.get("operatingMargins", 0) or 0) > 0.1: score += 1
    return score

# ================= SIMULAÇÃO =================
def simular(hist):
    if len(hist) < 300:
        return 0, 0

    close = hist["Close"]
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(200).mean()

    retorno = close.shift(-15) / close - 1

    buy = (rsi < 35) & (close > sma200)
    sell = (rsi > 70)

    taxa_c = (retorno[buy] > 0).mean() * 100 if buy.any() else 0
    taxa_v = (retorno[sell] < 0).mean() * 100 if sell.any() else 0

    return taxa_c, taxa_v

# ================= DADOS =================
@st.cache_data(ttl=600)
def obter_dados(tickers):
    tickers_yf = [t + ".SA" if not t.endswith(".SA") else t for t in tickers]

    hist = yf.download(
        tickers_yf,
        period="5y",
        auto_adjust=True,
        group_by="ticker",
        progress=False
    )

    dados = {}

    for i, t in enumerate(tickers):
        t_yf = tickers_yf[i]

        try:
            ticker = yf.Ticker(t_yf)
            fast = ticker.fast_info

            info = {
                "lastPrice": fast.get("last_price"),
                "marketCap": fast.get("market_cap")
            }

            # fallback leve
            try:
                full = ticker.info
                info.update(full)
            except:
                pass

            if isinstance(hist.columns, pd.MultiIndex):
                df = hist[t_yf].copy()
            else:
                df = hist.copy()

            dados[t] = (info, df)

        except:
            dados[t] = ({}, pd.DataFrame())

    return dados

# ================= PROCESSAMENTO =================
def processar(ticker, info, hist):

    if hist.empty:
        return None

    hist = hist.copy()
    hist["SMA20"] = hist["Close"].rolling(20).mean()

    preco = float(hist["Close"].iloc[-1])
    rsi = calcular_rsi(hist["Close"])

    pl = info.get("trailingPE", 0) or 0
    dy = (info.get("dividendYield", 0) or 0) * 100

    lpa = info.get("trailingEps", 0) or 0
    vpa = info.get("bookValue", 0) or 0

    graham = calcular_graham(lpa, vpa)
    upside = ((graham / preco) - 1) * 100 if graham > 0 else 0

    taxa_c, taxa_v = simular(hist)

    return {
        "ticker": ticker,
        "preco": preco,
        "rsi": rsi,
        "dy": dy,
        "pl": pl,
        "graham": graham,
        "upside": upside,
        "hist": hist,
        "taxa_c": taxa_c,
        "taxa_v": taxa_v
    }

# ================= UI =================
st.title("🤖 Monitor IA Pro (Refatorado)")

tickers = ["PETR4", "VALE3", "ITUB4", "WEGE3"]

dados = obter_dados(tickers)

resultados = []

for t in tickers:
    info, hist = dados[t]
    r = processar(t, info, hist)
    if r:
        resultados.append(r)

# ================= TABELA =================
df = pd.DataFrame(resultados)[
    ["ticker", "preco", "dy", "upside", "rsi", "taxa_c"]
]

st.dataframe(df.sort_values("upside", ascending=False), use_container_width=True)

# ================= GRÁFICO =================
for r in resultados:
    st.divider()
    st.subheader(f"{r['ticker']}")

    hist = r["hist"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_width=[0.2, 0.7]
    )

    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"]
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["SMA20"],
        name="SMA20"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=hist.index,
        y=hist["Volume"],
        name="Volume"
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preço", f"R$ {r['preco']:.2f}")
    c2.metric("RSI", f"{r['rsi']:.1f}")
    c3.metric("DY", f"{r['dy']:.2f}%")
    c4.metric("Upside", f"{r['upside']:.1f}%")
