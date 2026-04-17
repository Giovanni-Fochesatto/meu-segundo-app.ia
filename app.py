import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ===================== CONFIGURAÇÕES =====================
st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    st.session_state.filtros_ativos = True

# ===================== FUNÇÕES TÉCNICAS =====================
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
    rs = gain / loss.where(loss != 0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calcular_rsi(data, window: int = 14):
    if len(data) < window:
        return 50.0
    return float(calcular_rsi_series(data, window).iloc[-1])

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

# ===================== SIMULAÇÃO (mantida) =====================
def simular_performance_historica(hist, min_volume=50000):
    if len(hist) < 300:
        return {"taxa_compra": 0.0, "expectancy_compra": 0.0, "sharpe_compra": 0.0, "max_drawdown": 0.0, "qtd_compra": 0}
    close = hist["Close"].copy()
    volume = hist.get("Volume", pd.Series(0, index=close.index))
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(window=200).mean()
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()
    retorno_15d = close.shift(-15) / close - 1
    liquid_mask = volume > min_volume
    buy_mask = (
        (rsi < 35) & (close > sma200) & (macd > sinal_macd) &
        retorno_15d.notna() & liquid_mask
    )
    if buy_mask.any():
        ret_buy = retorno_15d[buy_mask]
        qtd_c = int(buy_mask.sum())
        expectancy_c = ret_buy.mean() * 100
        sharpe_c = ret_buy.mean() / ret_buy.std() * np.sqrt(252) if ret_buy.std() != 0 else 0
    else:
        expectancy_c = sharpe_c = 0.0
        qtd_c = 0
    if len(close) > 10:
        cum_ret = close.pct_change().cumsum()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min() * 100
    else:
        max_dd = 0.0
    return {
        "taxa_compra": 0.0,
        "expectancy_compra": expectancy_c,
        "sharpe_compra": sharpe_c,
        "max_drawdown": max_dd,
        "qtd_compra": qtd_c
    }

# ===================== CACHE =====================
@st.cache_data(ttl=600, show_spinner=False)
def obter_dados_batch(tickers, mercado):
    if not tickers:
        return {}, {}
    tickers_yf = [t + ".SA" if mercado == "Brasil" and not t.endswith(".SA") else t for t in tickers]
    hist_multi = yf.download(tickers_yf, period="5y", group_by="ticker", auto_adjust=True, progress=False, threads=True)
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

# ===================== PROCESSAMENTO CENTRAL (tolerante) =====================
def processar_ativo(tkr, info, hist, estrategia_ativa, filtros_ativos, f_pl, f_pvp, f_dy, f_div_ebitda, busca_direta, mercado):
    if hist.empty:
        st.sidebar.warning(f"Histórico vazio para {tkr}")
        return None
    if not info:
        st.sidebar.warning(f"Info vazio para {tkr}")
        return None

    hist = hist.copy()
    hist['SMA20'] = hist['Close'].rolling(window=20).mean()

    pl = info.get("trailingPE", 0) or 0
    pvp = info.get("priceToBook", 0) or 0
    dy = (info.get("dividendYield", 0) or 0) * 100
    ebitda = info.get("ebitda", 1) or 1
    div_liq = info.get("totalDebt", 0) or 0
    cash = info.get("totalCash", 0) or 0
    div_e = (div_liq - cash) / ebitda if ebitda != 0 else 999.0

    lpa = info.get("trailingEps", 0) or 0
    vpa = info.get("bookValue", 0) or 0
    p_justo = calcular_graham(lpa, vpa)
    p_atual = float(hist["Close"].iloc[-1]) if not hist.empty else 0
    upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 and p_atual > 0 else 0.0

    # Filtro desativado temporariamente para debug
    # if not busca_direta and filtros_ativos:
    #     if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
    #         return None

    # Notícias
    noticias_texto = ""
    lista_links = []
    try:
        lang = "pt-BR" if mercado == "Brasil" else "en-US"
        url = f"https://news.google.com/rss/search?q={tkr}&hl={lang}"
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            titulo = entry.title.lower()
            noticias_texto += titulo + " "
            lista_links.append({"titulo": entry.title, "link": entry.link})
    except:
        pass

    score_p = sum(noticias_texto.count(w) for w in ["alta", "lucro", "compra", "subiu", "dividend", "profit", "buy"])
    score_n = sum(noticias_texto.count(w) for w in ["queda", "prejuízo", "venda", "caiu", "risk", "loss", "sell"])

    rsi_val = calcular_rsi(hist["Close"])
    score_value, criteria = calcular_score_value(info)
    sim = simular_performance_historica(hist)

    # Veredito simples para teste
    veredito = "NEUTRO ⚖️"
    cor = "warning"
    motivo_detalhe = "Processando..."

    return {
        "Ticker": tkr,
        "Empresa": info.get("shortName", tkr),
        "Preço": p_atual,
        "P/L": pl,
        "DY %": dy,
        "Dívida": div_e,
        "Graham": p_justo,
        "Upside %": upside,
        "Veredito": veredito,
        "Cor": cor,
        "Motivo": motivo_detalhe,
        "RSI": rsi_val,
        "Hist": hist,
        "Links": lista_links,
        "ValueScore": score_value,
        "ValueCriteria": criteria,
        "ExpectancyCompra": sim.get("expectancy_compra", 0),
        "SharpeCompra": sim.get("sharpe_compra", 0),
        "QtdCompra": sim.get("qtd_compra", 0)
    }

# ===================== SIDEBAR (mantida) =====================
st.sidebar.title("🌎 Monitor IA Pro")
st.sidebar.subheader("📈 Índices Mundiais")
indices_data = obter_indices()
for nome, (valor, var) in indices_data.items():
    st.sidebar.metric(nome, f"{valor:,.0f} pts", f"{var:.2f}%")
st.sidebar.divider()
st.sidebar.subheader("💱 Câmbio em Tempo Real")
cambio = obter_cambio()
col1, col2 = st.sidebar.columns(2)
col1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}")
col2.metric("Bitcoin", f"R$ {cambio['Bitcoin'][0]:,.0f}")
st.sidebar.divider()

mercado_selecionado = st.sidebar.radio("Mercado:", ["Brasil", "EUA"], on_change=ativar_filtros)
estrategia_ativa = st.sidebar.selectbox("Estratégia:", ["Value Investing (Graham/Buffett)"])
busca_direta = st.sidebar.text_input(f"🔍 Busca Rápida ({mercado_selecionado}):").upper().strip()

# ===================== LISTA DE ATIVOS =====================
if mercado_selecionado == "Brasil":
    lista_base = ["PETR4", "VALE3", "ITUB4", "BBAS3", "BBDC4", "B3SA3", "EGIE3", "TRPL4", "TAEE11", "WEGE3", "PRIO3", "JBSS3"]
    moeda_simbolo = "R$"
else:
    lista_base = ["AAPL", "MSFT"]
    moeda_simbolo = "US$"

st.title(f"🤖 Monitor IA - {mercado_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Gráfico Técnico", "📉 Fundamentalista", "📜 Backtest"])

# ===================== PROCESSAMENTO =====================
tickers_para_processar = [busca_direta] if busca_direta else lista_base
dados_vencedoras = []

if tickers_para_processar:
    with st.spinner("📡 Baixando dados..."):
        infos, hists = obter_dados_batch(tickers_para_processar, mercado_selecionado)
    for tkr in tickers_para_processar:
        info = infos.get(tkr, {})
        hist = hists.get(tkr, pd.DataFrame())
        resultado = processar_ativo(tkr, info, hist, estrategia_ativa, False, 25, 3, 4, 3, busca_direta, mercado_selecionado)
        if resultado:
            dados_vencedoras.append(resultado)

# ===================== TAB 1 - OVERVIEW =====================
with tab1:
    if dados_vencedoras:
        st.subheader(f"🏆 Ranking - Estratégia: {estrategia_ativa}")
        df = pd.DataFrame(dados_vencedoras)
        st.dataframe(df[["Ticker", "Preço", "DY %", "Upside %", "Veredito"]], use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum ativo encontrado. Tente fazer uma busca direta (ex: PETR4)")

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
            if acao.get("Links"):
                st.markdown("**Manchetes:**")
                for n in acao["Links"]:
                    st.markdown(f"• [{n['titulo']}]({n['link']})")
            st.divider()
    else:
        st.info("Nenhum ativo encontrado.")

st.info("💡 Use a busca direta (ex: PETR4) para testar.")
