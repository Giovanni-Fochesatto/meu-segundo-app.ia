import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
from streamlit_autorefresh import st_autorefresh

# ===================== CONFIGURAÇÕES =====================
st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    st.session_state.filtros_ativos = True

# ===================== FUNÇÕES TÉCNICAS =====================
# (todas as funções calcular_graham, rsi, score_value, simular_performance_historica permanecem iguais às suas)

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
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_rsi(data, window: int = 14):
    if len(data) < window:
        return 50.0
    rsi_series = calcular_rsi_series(data, window)
    return float(rsi_series.iloc[-1])

def calcular_score_value(info):
    score = 0
    if 0 < info.get("trailingPE", 99) < 15:
        score += 1
    if 0 < info.get("priceToBook", 99) < 1.5:
        score += 1
    if (info.get("dividendYield", 0) or 0) * 100 > 5:
        score += 1
    if (info.get("operatingMargins", 0) or 0) > 0.1:
        score += 1
    return score

def simular_performance_historica(hist):
    if len(hist) < 300:
        return 0.0, 0.0, 0.0, 0, 0
    close = hist["Close"].copy()
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(window=200).mean()
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()
    retorno_15d = close.shift(-15) / close - 1

    buy_mask = (
        (rsi < 35) & (close > sma200) & (macd > sinal_macd) & retorno_15d.notna()
    )
    sell_mask = (
        (rsi > 70) & ((close < sma200) | (macd < sinal_macd)) & retorno_15d.notna()
    )

    if buy_mask.any():
        ret_buy = retorno_15d[buy_mask]
        taxa_compra = (ret_buy > 0).mean() * 100
        retorno_medio = ret_buy.mean() * 100
        total_c = int(buy_mask.sum())
    else:
        taxa_compra = 0.0
        retorno_medio = 0.0
        total_c = 0

    taxa_venda = (retorno_15d[sell_mask] < 0).mean() * 100 if sell_mask.any() else 0.0
    total_v = int(sell_mask.sum()) if sell_mask.any() else 0
    return taxa_compra, taxa_venda, retorno_medio, total_c, total_v

# ===================== CACHE DE MERCADO =====================
@st.cache_data(ttl=300, show_spinner=False)
def obter_indices():
    indices = {"Ibovespa": "^BVSP", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}
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


@st.cache_data(ttl=90, show_spinner=False)   # ttl menor para Bitcoin
def obter_cambio():
    moedas = {
        "Dólar": "USDBRL=X",
        "Euro": "EURBRL=X",
        "Libra": "GBPBRL=X",
    }
    resultados = {}

    # Moedas normais
    for nome, ticker in moedas.items():
        try:
            t = yf.Ticker(ticker)
            data = t.history(period="2d")
            if not data.empty and len(data) >= 2:
                atual = data["Close"].iloc[-1]
                anterior = data["Close"].iloc[-2]
                variacao = ((atual / anterior) - 1) * 100
            else:
                atual = t.fast_info.last_price
                variacao = 0.0
            resultados[nome] = (atual, variacao)
        except:
            resultados[nome] = (0.0, 0.0)

    # ===================== BITCOIN - FORÇADO EM REAL =====================
    btc_real = 0.0
    variacao_btc = 0.0

    # 1. Tentativa direta BTC-BRL
    try:
        t = yf.Ticker("BTC-BRL")
        data = t.history(period="2d")
        if not data.empty and len(data) >= 2:
            atual = float(data["Close"].iloc[-1])
            anterior = float(data["Close"].iloc[-2])
            variacao_btc = ((atual / anterior) - 1) * 100
        else:
            atual = float(t.fast_info.last_price)
        if atual > 100000:   # valor plausível em Real
            btc_real = atual
    except:
        pass

    # 2. Fallback: BTC-USD convertido pelo dólar atual (mais confiável)
    if btc_real < 100000:
        try:
            btc_usd = float(yf.Ticker("BTC-USD").fast_info.last_price)
            dolar_brl = float(resultados.get("Dólar", (5.0, 0))[0])
            
            if btc_usd > 50000 and dolar_brl > 4.5:   # verificação de sanidade
                btc_real = btc_usd * dolar_brl
                variacao_btc = 0.0
                # st.sidebar.info(f"Debug: BTC-USD {btc_usd:,.0f} × Dólar {dolar_brl:.2f} = R$ {btc_real:,.0f}")  # descomente para debug
        except Exception as e:
            pass

    resultados["Bitcoin"] = (btc_real, variacao_btc)
    return resultados


# ===================== (O resto do código permanece igual) =====================
# ... (obter_dados_batch, processar_ativo, sidebar, processamento principal e interface)

# ===================== SIDEBAR =====================
st.sidebar.title("🌎 Mercado e Estratégia")

st.sidebar.subheader("📈 Índices Mundiais")
indices_data = obter_indices()
for nome, (valor, var) in indices_data.items():
    st.sidebar.metric(nome, f"{valor:,.0f} pts", f"{var:.2f}%")

st.sidebar.divider()

st.sidebar.subheader("💱 Câmbio em Tempo Real")
cambio = obter_cambio()

# Primeira linha: Dólar | Euro
col1, col2 = st.sidebar.columns(2)
col1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
col2.metric("Euro",  f"R$ {cambio['Euro'][0]:.2f}",  f"{cambio['Euro'][1]:.2f}%")

# Segunda linha: Libra | Bitcoin
col3, col4 = st.sidebar.columns(2)
col3.metric("Libra", f"R$ {cambio['Libra'][0]:.2f}", f"{cambio['Libra'][1]:.2f}%")
col4.metric("Bitcoin", f"R$ {cambio['Bitcoin'][0]:,.0f}", f"{cambio['Bitcoin'][1]:.2f}%")

st.sidebar.divider()

# ===================== RESTANTE DO CÓDIGO (mercado, estratégia, processamento e interface) =====================
# Copie aqui o restante do seu código original (lista_base, processamento, interface, etc.)
# (Para não repetir tudo novamente, mantenha exatamente como estava nas versões anteriores)

mercado_selecionado = st.sidebar.radio(
    "Escolha o Mercado:", ["Brasil", "EUA"], on_change=ativar_filtros
)

estrategia_ativa = st.sidebar.selectbox(
    "Foco da Análise:", ["Análise Técnica + Notícias", "Value Investing (Graham/Buffett)"]
)

busca_direta = st.sidebar.text_input(f"🔍 Busca Rápida ({mercado_selecionado}):").upper().strip()

with st.sidebar.expander("📊 Filtros de Valuation", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0, step=0.5, on_change=ativar_filtros)
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0, step=0.1, on_change=ativar_filtros)
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0, step=0.5, on_change=ativar_filtros)
    f_div_ebitda = st.slider("Dív.Líq/EBITDA Máximo", 0.0, 15.0, 15.0, step=0.5, on_change=ativar_filtros)

if st.sidebar.button("Resetar Filtros"):
    st.session_state.filtros_ativos = False
    st.rerun()

# (Continue com o resto do seu código: lista_base, cabeçalho, processamento principal e interface)
