import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ===================== CONFIG =====================
st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    st.session_state.filtros_ativos = True

# ===================== FUNÇÕES =====================
def calcular_graham(lpa, vpa):
    return np.sqrt(22.5 * lpa * vpa) if lpa > 0 and vpa > 0 else 0.0

def calcular_rsi_series(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)

def calcular_rsi(data):
    return float(calcular_rsi_series(data).iloc[-1]) if len(data) > 14 else 50

def calcular_score_value(info):
    score = 0
    if 0 < info.get("trailingPE", 99) < 15: score += 1
    if 0 < info.get("priceToBook", 99) < 1.5: score += 1
    if (info.get("dividendYield", 0) or 0) * 100 > 5: score += 1
    if (info.get("operatingMargins", 0) or 0) > 0.1: score += 1
    return score

# ===================== SIMULAÇÃO =====================
def simular_performance_historica(hist):
    if len(hist) < 300:
        return {k:0 for k in ["taxa_compra","taxa_venda","retorno_medio_compra",
                             "retorno_medio_venda","expectancy_compra","expectancy_venda",
                             "qtd_compra","qtd_venda"]}

    close = hist["Close"]
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(200).mean()
    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    sinal = macd.ewm(span=9).mean()
    ret = close.shift(-15)/close - 1

    buy = (rsi < 35) & (close > sma200) & (macd > sinal)
    sell = (rsi > 70) & (close < sma200)

    def calc(mask, invert=False):
        if not mask.any(): return 0,0,0,0
        r = ret[mask].dropna()
        taxa = (r < 0).mean()*100 if invert else (r > 0).mean()*100
        avg_win = r[r>0].mean() if (r>0).any() else 0
        avg_loss = abs(r[r<0].mean()) if (r<0).any() else 0
        exp = ((taxa/100)*avg_win - (1-taxa/100)*avg_loss)*100
        return taxa, r.mean()*100, exp, len(r)

    tc, rmc, ec, qc = calc(buy)
    tv, rmv, ev, qv = calc(sell, True)

    return {
        "taxa_compra":tc,"taxa_venda":tv,
        "retorno_medio_compra":rmc,"retorno_medio_venda":rmv,
        "expectancy_compra":ec,"expectancy_venda":ev,
        "qtd_compra":qc,"qtd_venda":qv
    }

# ===================== DADOS =====================
@st.cache_data(ttl=600)
def obter_dados_batch(tickers, mercado):
    tickers_yf = [t+".SA" if mercado=="Brasil" and not t.endswith(".SA") else t for t in tickers]
    hist_multi = yf.download(tickers_yf, period="5y", group_by="ticker", auto_adjust=True)

    infos, hists = {}, {}
    for i, t in enumerate(tickers):
        t_yf = tickers_yf[i]
        tk = yf.Ticker(t_yf)

        try:
            fast = tk.fast_info
            info = tk.info
            info.update({"lastPrice": fast.get("last_price")})
        except:
            info = {}

        if isinstance(hist_multi.columns, pd.MultiIndex):
            hist = hist_multi[t_yf].copy() if t_yf in hist_multi else pd.DataFrame()
        else:
            hist = hist_multi.copy()

        infos[t], hists[t] = info, hist

    return infos, hists

# ===================== PROCESSAMENTO =====================
def processar_ativo(tkr, info, hist, estrategia):
    if hist.empty or not info: return None

    hist = hist.copy()
    hist["SMA20"] = hist["Close"].rolling(20).mean()

    pl = info.get("trailingPE",0)
    dy = (info.get("dividendYield",0) or 0)*100
    p = hist["Close"].iloc[-1]

    graham = calcular_graham(info.get("trailingEps",0), info.get("bookValue",0))
    upside = ((graham/p)-1)*100 if graham>0 else 0

    rsi = calcular_rsi(hist["Close"])
    sim = simular_performance_historica(hist)

    return {
        "Ticker":tkr,
        "Preço":p,
        "DY":dy,
        "Upside":upside,
        "RSI":rsi,
        "Hist":hist,
        **sim
    }

# ===================== UI =====================
mercado = st.sidebar.radio("Mercado", ["Brasil","EUA"])
busca = st.sidebar.text_input("Ticker").upper()

lista = ["PETR4","VALE3","ITUB4"] if mercado=="Brasil" else ["AAPL","MSFT","NVDA"]
tickers = [busca] if busca else lista

infos,hists = obter_dados_batch(tickers, mercado)

dados = []
for t in tickers:
    r = processar_ativo(t, infos[t], hists[t], "")
    if r: dados.append(r)

# ===================== OUTPUT =====================
for acao in dados:
    st.subheader(acao["Ticker"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_width=[0.2,0.7])

    hist = acao["Hist"]

    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"]
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["SMA20"],
        name="SMA20"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"]
    ), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=500)

    st.plotly_chart(fig, use_container_width=True)

    st.metric("Preço", f"{acao['Preço']:.2f}")
    st.metric("Upside", f"{acao['Upside']:.2f}%")
    st.metric("RSI", f"{acao['RSI']:.1f}")
    st.metric("Assert Compra", f"{acao['taxa_compra']:.1f}%")
