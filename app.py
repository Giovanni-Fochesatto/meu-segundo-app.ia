import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ==============================================================================
# ATUALIZAÇÃO PRO: CÓDIGO COMPLETO (SEM CORTES)
# ==============================================================================

# Configurações de Página
st.set_page_config(page_title="Monitor IA Pro", layout="wide", page_icon="📈")
st_autorefresh(interval=300 * 1000, key="data_refresh")

# Estado da Sessão
if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False

# Funções Matemáticas e Técnicas
def calcular_graham(lpa, vpa):
    if lpa > 0 and vpa > 0:
        return np.sqrt(22.5 * lpa * vpa)
    return 0.0

def calcular_rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    if len(close) < window:
        return pd.Series([50.0] * len(close), index=close.index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calcular_score_value(info):
    score = 0
    if 0 < info.get("trailingPE", 99) < 15: score += 1
    if 0 < info.get("priceToBook", 99) < 1.5: score += 1
    if (info.get("dividendYield", 0) or 0) * 100 > 5: score += 1
    return score

# Performance Histórica (Simulação)
def simular_performance_historica(hist):
    if len(hist) < 250:
        return {k: 0.0 for k in ["taxa_compra", "taxa_venda", "retorno_medio_compra", "retorno_medio_venda"]} | {"qtd_compra": 0, "qtd_venda": 0}
    
    close = hist["Close"].copy()
    rsi = calcular_rsi_series(close)
    retorno_15d = close.shift(-15) / close - 1
    
    buy_mask = (rsi < 30)
    sell_mask = (rsi > 70)
    
    res = {}
    for pfx, mask in [("compra", buy_mask), ("venda", sell_mask)]:
        if mask.any():
            ret = retorno_15d[mask].dropna()
            res[f"taxa_{pfx}"] = (ret > 0).mean() * 100 if not ret.empty else 0.0
            res[f"qtd_{pfx}"] = int(mask.sum())
        else:
            res[f"taxa_{pfx}"], res[f"qtd_{pfx}"] = 0.0, 0
    return res

# Cache de Dados
@st.cache_data(ttl=600)
def obter_indices():
    res = {}
    for nome, ticker in {"Ibovespa": "^BVSP", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}.items():
        try:
            d = yf.Ticker(ticker).history(period="2d")
            res[nome] = (d["Close"].iloc[-1], ((d["Close"].iloc[-1]/d["Close"].iloc[-2])-1)*100)
        except: res[nome] = (0, 0)
    return res

@st.cache_data(ttl=90)
def obter_cambio():
    res = {}
    for n, t in {"Dólar": "USDBRL=X", "Bitcoin": "BTC-BRL"}.items():
        try:
            d = yf.Ticker(t).history(period="2d")
            res[n] = (d["Close"].iloc[-1], ((d["Close"].iloc[-1]/d["Close"].iloc[-2])-1)*100)
        except: res[n] = (0, 0)
    return res

@st.cache_data(ttl=600)
def obter_dados_batch(tickers, mercado):
    if not tickers: return {}, {}
    tk_yf = [t + ".SA" if mercado == "Brasil" and not t.endswith(".SA") else t for t in tickers]
    try:
        data = yf.download(tk_yf, period="2y", group_by="ticker", auto_adjust=True, progress=False)
        info_dict = {t: yf.Ticker(tk_yf[i]).info for i, t in enumerate(tickers)}
        hist_dict = {t: data[tk_yf[i]] if len(tickers) > 1 else data for i, t in enumerate(tickers)}
        return info_dict, hist_dict
    except: return {}, {}

# Processamento de Ativos
def processar_ativo(tkr, info, hist, estrategia, filtros_on, f_pl, f_pvp, f_dy, f_div_e, mercado):
    if hist is None or hist.empty or not info: return None
    
    p_atual = float(hist["Close"].iloc[-1])
    pl = info.get("trailingPE", 0) or 0
    pvp = info.get("priceToBook", 0) or 0
    dy = (info.get("dividendYield", 0) or 0) * 100
    ebitda = info.get("ebitda", 1) or 1
    div_e = (info.get("totalDebt", 0) - info.get("totalCash", 0)) / ebitda

    # Filtros de Segurança
    if filtros_on:
        if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy): return None

    lpa, vpa = info.get("trailingEps", 0) or 0, info.get("bookValue", 0) or 0
    p_justo = calcular_graham(lpa, vpa)
    upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 else 0.0

    # Notícias
    links = []
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={tkr}&hl=pt-BR")
        links = [{"titulo": e.title, "link": e.link} for e in feed.entries[:3]]
    except: pass

    sim = simular_performance_historica(hist)
    
    # Veredito Logics
    veredito, cor = "NEUTRO ⚖️", "warning"
    if estrategia == "Value Investing (Graham/Buffett)" and upside > 15: veredito, cor = "COMPRA ✅", "success"
    elif pl > 30 or upside < -10: veredito, cor = "CARO 🚨", "error"

    return {
        "Ticker": tkr, "Empresa": info.get("shortName", tkr), "Preço": p_atual, 
        "P/L": pl, "DY %": dy, "Graham": p_justo, "Upside %": upside, 
        "Veredito": veredito, "Cor": cor, "Hist": hist, "Links": links,
        "TaxaCompra": sim.get("taxa_compra", 0)
    }

# --- INTERFACE ---
st.sidebar.title("🌎 Monitor IA Pro")
indices = obter_indices()
for n, (v, var) in indices.items(): st.sidebar.metric(n, f"{v:,.0f}", f"{var:.2f}%")

cambio = obter_cambio()
st.sidebar.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
st.sidebar.metric("BTC", f"R$ {cambio['Bitcoin'][0]:,.0f}", f"{cambio['Bitcoin'][1]:.2f}%")

mercado = st.sidebar.radio("Mercado:", ["Brasil", "EUA"])
estrategia = st.sidebar.selectbox("Estratégia:", ["Value Investing (Graham/Buffett)", "Dividend Investing", "Análise Técnica"])
busca = st.sidebar.text_input("🔍 Ticker:").upper().strip()

with st.sidebar.expander("📊 Filtros"):
    f_pl = st.slider("P/L Máx", 0, 100, 100)
    f_pvp = st.slider("P/VP Máx", 0.0, 10.0, 10.0)
    f_dy = st.slider("DY Mín %", 0, 20, 0)

st.title(f"🤖 Monitor IA - {mercado}")
st.caption(f"Blumenau/SC | {time.strftime('%H:%M:%S')}")

lista_padrao = ["PETR4", "VALE3", "ITUB4", "BBAS3"] if mercado == "Brasil" else ["AAPL", "MSFT", "NVDA"]
lista = [busca] if busca else lista_padrao

infos, hists = obter_dados_batch(lista, mercado)
dados_finais = []

for tkr in lista:
    res = processar_ativo(tkr, infos.get(tkr), hists.get(tkr), estrategia, bool(busca==""), f_pl, f_pvp, f_dy, 0, mercado)
    if res: dados_finais.append(res)

if dados_finais:
    for acao in dados_finais:
        with st.container():
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.header(f"{acao['Ticker']} - {acao['Empresa']}")
            if acao['Cor'] == "success": c2.success(acao['Veredito'])
            else: c2.warning(acao['Veredito'])
            c3.metric("Assert. Compra", f"{acao['TaxaCompra']:.1f}%")

            # Gráfico Candlestick
            h = acao["Hist"]
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='Preço'))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
else:
    st.warning("Ajuste os filtros ou verifique o Ticker informado.")
