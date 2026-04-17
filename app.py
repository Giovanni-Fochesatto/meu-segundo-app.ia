import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    st.session_state.filtros_ativos = True

# ===================== FUNÇÕES BÁSICAS =====================
def calcular_graham(lpa, vpa):
    if lpa > 0 and vpa > 0:
        return np.sqrt(22.5 * lpa * vpa)
    return 0.0

def calcular_rsi(data, window: int = 14):
    if len(data) < window:
        return 50.0
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss.where(loss != 0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def calcular_score_value(info):
    score = 0
    criteria = []
    if 0 < info.get("trailingPE", 99) < 15:
        score += 1
        criteria.append("P/L baixo")
    if 0 < info.get("priceToBook", 99) < 1.5:
        score += 1
        criteria.append("P/VP baixo")
    if (info.get("dividendYield", 0) or 0) * 100 > 5:
        score += 1
        criteria.append("DY alto")
    if (info.get("operatingMargins", 0) or 0) > 0.1:
        score += 1
        criteria.append("Margem boa")
    return score, criteria

# ===================== CACHE =====================
@st.cache_data(ttl=300)
def obter_indices():
    indices = {"Ibovespa": "^BVSP"}
    resultados = {}
    for nome, ticker in indices.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if not data.empty and len(data) >= 2:
                atual = data["Close"].iloc[-1]
                anterior = data["Close"].iloc[-2]
                variacao = ((atual / anterior) - 1) * 100
                resultados[nome] = (atual, variacao)
            else:
                resultados[nome] = (0.0, 0.0)
        except:
            resultados[nome] = (0.0, 0.0)
    return resultados

@st.cache_data(ttl=90)
def obter_cambio():
    moedas = {"Dólar": "USDBRL=X"}
    resultados = {}
    for nome, ticker in moedas.items():
        try:
            t = yf.Ticker(ticker)
            data = t.history(period="2d")
            if not data.empty and len(data) >= 2:
                atual = float(data["Close"].iloc[-1])
                anterior = float(data["Close"].iloc[-2])
                variacao = ((atual / anterior) - 1) * 100
            else:
                atual = float(t.fast_info.last_price)
                variacao = 0.0
            resultados[nome] = (atual, variacao)
        except:
            resultados[nome] = (0.0, 0.0)
    resultados["Bitcoin"] = (0.0, 0.0)  # simplificado
    return resultados

@st.cache_data(ttl=600)
def obter_dados_batch(tickers, mercado):
    if not tickers:
        return {}, {}
    tickers_yf = [t + ".SA" if mercado == "Brasil" and not t.endswith(".SA") else t for t in tickers]
    hist_multi = yf.download(tickers_yf, period="1y", group_by="ticker", auto_adjust=True, progress=False)
    info_dict = {}
    hist_dict = {}
    for i, t_orig in enumerate(tickers):
        t_yf = tickers_yf[i]
        try:
            info_dict[t_orig] = yf.Ticker(t_yf).info
            if len(tickers) == 1:
                hist_dict[t_orig] = hist_multi
            else:
                hist_dict[t_orig] = hist_multi[t_yf] if t_yf in hist_multi.columns.get_level_values(0) else pd.DataFrame()
        except Exception as e:
            st.sidebar.error(f"Erro ao baixar {t_orig}: {str(e)}")
            info_dict[t_orig] = {}
            hist_dict[t_orig] = pd.DataFrame()
    return info_dict, hist_dict

# ===================== PROCESSAMENTO =====================
def processar_ativo(tkr, info, hist, estrategia_ativa, mercado):
    if hist.empty:
        st.sidebar.warning(f"Histórico vazio para {tkr}")
        return None
    if not info:
        st.sidebar.warning(f"Info vazio para {tkr}")
        return None

    p_atual = float(hist["Close"].iloc[-1]) if not hist.empty else 0
    pl = info.get("trailingPE", 0) or 0
    dy = (info.get("dividendYield", 0) or 0) * 100
    score_value, criteria = calcular_score_value(info)

    veredito = "NEUTRO ⚖️"
    cor = "warning"
    motivo = "Processando..."

    return {
        "Ticker": tkr,
        "Empresa": info.get("shortName", tkr),
        "Preço": p_atual,
        "P/L": pl,
        "DY %": dy,
        "Veredito": veredito,
        "Motivo": motivo,
        "ValueScore": score_value,
        "ValueCriteria": criteria,
        "Links": []  # será preenchido depois
    }

# ===================== INTERFACE =====================
st.title("🤖 Monitor IA - Brasil")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Gráfico Técnico", "📉 Fundamentalista", "📜 Backtest"])

mercado_selecionado = "Brasil"
lista_base = ["PETR4", "VALE3", "ITUB4", "BBAS3", "B3SA3", "EGIE3", "WEGE3", "PRIO3"]

busca_direta = st.sidebar.text_input("🔍 Busca Rápida (ex: PETR4)").upper().strip()
tickers_para_processar = [busca_direta] if busca_direta else lista_base

dados_vencedoras = []
if tickers_para_processar:
    with st.spinner("Baixando dados..."):
        infos, hists = obter_dados_batch(tickers_para_processar, mercado_selecionado)
    for tkr in tickers_para_processar:
        info = infos.get(tkr, {})
        hist = hists.get(tkr, pd.DataFrame())
        resultado = processar_ativo(tkr, info, hist, "Value Investing (Graham/Buffett)", mercado_selecionado)
        if resultado:
            dados_vencedoras.append(resultado)

# ===================== TAB 1 - OVERVIEW =====================
with tab1:
    if dados_vencedoras:
        st.subheader("🏆 Ranking")
        df = pd.DataFrame(dados_vencedoras)
        st.dataframe(df[["Ticker", "Preço", "DY %", "Veredito"]], use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum ativo encontrado. Tente digitar PETR4 na busca direta.")

# ===================== TAB 3 - FUNDAMENTALISTA =====================
with tab3:
    st.subheader("📉 Análise Fundamentalista")
    if dados_vencedoras:
        for acao in dados_vencedoras:
            st.write(f"**{acao['Empresa']} ({acao['Ticker']})**")
            score = acao.get("ValueScore", 0)
            criteria = acao.get("ValueCriteria", [])
            st.markdown(f"**Value Score: {score}/4**")
            st.progress(score / 4)
            if criteria:
                st.caption("Critérios: " + " • ".join(criteria))
            st.divider()
    else:
        st.info("Nenhum ativo encontrado.")

st.info("💡 Digite um ticker (ex: PETR4) na busca direta para testar.")
