# ===================== IMPORTS =====================
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

# ===================== FUNÇÕES BASE =====================
def calcular_graham(lpa, vpa):
    if lpa > 0 and vpa > 0:
        return np.sqrt(22.5 * lpa * vpa)
    return 0.0

def calcular_rsi_series(close, window=14):
    if len(close) < window:
        return pd.Series([50.0]*len(close), index=close.index)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()

    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rsi = 100 - (100/(1+rs))

    return rsi.fillna(50)

def calcular_rsi(data):
    return float(calcular_rsi_series(data).iloc[-1])

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
        return dict.fromkeys([
            "taxa_compra","taxa_venda","retorno_medio_compra",
            "retorno_medio_venda","expectancy_compra",
            "expectancy_venda","qtd_compra","qtd_venda"
        ],0)

    close = hist["Close"]
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(200).mean()

    exp12 = close.ewm(span=12).mean()
    exp26 = close.ewm(span=26).mean()
    macd = exp12-exp26
    sinal = macd.ewm(span=9).mean()

    ret = close.shift(-15)/close -1

    buy = (rsi<35)&(close>sma200)&(macd>sinal)&ret.notna()
    sell = (rsi>70)&((close<sma200)|(macd<sinal))&ret.notna()

    def calc(mask, positive=True):
        if not mask.any():
            return 0,0,0,0
        r = ret[mask]
        taxa = (r>0).mean() if positive else (r<0).mean()
        avg_win = r[r>0].mean() if (r>0).any() else 0
        avg_loss = abs(r[r<0].mean()) if (r<0).any() else 0
        exp = (taxa*avg_win) - ((1-taxa)*avg_loss)
        return taxa*100, r.mean()*100, exp*100, int(mask.sum())

    tc, rm_c, exp_c, qc = calc(buy, True)
    tv, rm_v, exp_v, qv = calc(sell, False)

    return {
        "taxa_compra":tc,"taxa_venda":tv,
        "retorno_medio_compra":rm_c,
        "retorno_medio_venda":rm_v,
        "expectancy_compra":exp_c,
        "expectancy_venda":exp_v,
        "qtd_compra":qc,"qtd_venda":qv
    }

# ===================== DADOS =====================
@st.cache_data(ttl=600)
def obter_dados_batch(tickers, mercado):
    tickers_yf = [t+".SA" if mercado=="Brasil" and not t.endswith(".SA") else t for t in tickers]
    hist_multi = yf.download(tickers_yf, period="5y", group_by="ticker", threads=True)

    info_dict, hist_dict = {}, {}

    for i,t in enumerate(tickers):
        t_yf = tickers_yf[i]
        tk = yf.Ticker(t_yf)

        try:
            info = tk.fast_info
            info_dict[t] = {
                "lastPrice": info.get("last_price"),
                "marketCap": info.get("market_cap"),
                **tk.info
            }
        except:
            info_dict[t] = {}

        try:
            if isinstance(hist_multi.columns, pd.MultiIndex):
                hist_dict[t] = hist_multi[t_yf].copy()
            else:
                hist_dict[t] = hist_multi.copy()
        except:
            hist_dict[t] = pd.DataFrame()

    return info_dict, hist_dict

# ===================== PROCESSAMENTO =====================
def processar_ativo(tkr, info, hist, estrategia):
    if hist.empty or not info:
        return None

    hist = hist.copy()
    hist["SMA20"] = hist["Close"].rolling(20).mean()

    p = float(hist["Close"].iloc[-1])
    rsi = calcular_rsi(hist["Close"])
    score = calcular_score_value(info)
    sim = simular_performance_historica(hist)

    veredito = "NEUTRO ⚖️"
    cor = "warning"

    if estrategia == "Análise Técnica (Trader)":
        if rsi > 70:
            veredito, cor = "VENDA 🚨", "error"
        elif rsi < 30:
            veredito, cor = "COMPRA ✅", "success"

    return {
        "Ticker":tkr,
        "Preço":p,
        "RSI":rsi,
        "Hist":hist,
        "Veredito":veredito,
        "Cor":cor,
        "TaxaCompra":sim["taxa_compra"],
        "TaxaVenda":sim["taxa_venda"]
    }

# ===================== UI =====================
st.title("🤖 Monitor IA PRO")

mercado = st.sidebar.radio("Mercado",["Brasil","EUA"])
estrategia = st.sidebar.selectbox("Estratégia",[
    "Análise Técnica (Trader)",
    "Value Investing (Graham/Buffett)",
    "Growth Investing",
    "Buy and Hold",
    "Dividend Investing",
    "Position Trading"
])

busca = st.sidebar.text_input("Buscar").upper()

lista = ["PETR4","VALE3","ITUB4"] if mercado=="Brasil" else ["AAPL","MSFT","NVDA"]

tickers = [busca] if busca else lista

dados = []
infos, hists = obter_dados_batch(tickers, mercado)

for t in tickers:
    r = processar_ativo(t, infos.get(t,{}), hists.get(t,pd.DataFrame()), estrategia)
    if r:
        dados.append(r)

# ===================== EXIBIÇÃO =====================
for acao in dados:
    st.divider()

    st.subheader(f"{acao['Ticker']} - {acao['Veredito']}")

    hist = acao["Hist"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"]
    ),row=1,col=1)

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["SMA20"],
        name="SMA20"
    ),row=1,col=1)

    fig.add_trace(go.Bar(
        x=hist.index,
        y=hist["Volume"]
    ),row=2,col=1)

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Preço",f"{acao['Preço']:.2f}")
    c2.metric("RSI",f"{acao['RSI']:.1f}")
    c3.metric("Win Rate",f"{acao['TaxaCompra']:.1f}%")
