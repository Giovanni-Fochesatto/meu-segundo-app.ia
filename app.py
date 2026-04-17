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
st_autorefresh(interval=300000, key="data_refresh")

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

def simular_performance_historica(hist, min_volume=50000):
    if len(hist) < 300:
        return {"expectancy_compra": 0.0, "sharpe_compra": 0.0, "qtd_compra": 0}
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
    buy_mask = (rsi < 35) & (close > sma200) & (macd > sinal_macd) & retorno_15d.notna() & liquid_mask
    if buy_mask.any():
        ret_buy = retorno_15d[buy_mask]
        expectancy = ret_buy.mean() * 100
        sharpe = ret_buy.mean() / ret_buy.std() * np.sqrt(252) if ret_buy.std() != 0 else 0
        qtd = int(buy_mask.sum())
    else:
        expectancy = sharpe = 0.0
        qtd = 0
    return {"expectancy_compra": expectancy, "sharpe_compra": sharpe, "qtd_compra": qtd}

# ===================== CACHE =====================
@st.cache_data(ttl=1800, show_spinner=False)
def obter_macro():
    macro = {}
    try:
        macro["Selic"] = 14.75
        macro["Dolar"] = yf.Ticker("USDBRL=X").fast_info.last_price
        macro["IPCA_12m"] = 4.14
    except:
        macro["Selic"] = 14.75
        macro["Dolar"] = 4.99
        macro["IPCA_12m"] = 4.14
    macro["Focus_Data"] = "13/04/2026"
    macro["Focus_Selic_2026"] = "12.50%"
    macro["Focus_IPCA_2026"] = "4.36%"
    macro["Focus_PIB_2026"] = "1.85%"
    return macro

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

@st.cache_data(ttl=90, show_spinner=False)
def obter_cambio():
    moedas = {"Dólar": "USDBRL=X", "Euro": "EURBRL=X", "Libra": "GBPBRL=X"}
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
    # Bitcoin com fallback robusto
    btc_real = 0.0
    try:
        t = yf.Ticker("BTC-BRL")
        data = t.history(period="2d")
        atual = float(data["Close"].iloc[-1]) if not data.empty and len(data) >= 2 else float(t.fast_info.last_price)
        if atual > 100000:
            btc_real = atual
    except:
        pass
    if btc_real < 100000:
        try:
            btc_usd = float(yf.Ticker("BTC-USD").fast_info.last_price)
            dolar_brl = resultados.get("Dólar", (4.99, 0))[0]
            btc_real = btc_usd * dolar_brl
        except:
            btc_real = 386000  # fallback aproximado
    resultados["Bitcoin"] = (btc_real, 0.0)
    return resultados

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
        except:
            info_dict[t_orig] = {}
            hist_dict[t_orig] = pd.DataFrame()
    return info_dict, hist_dict

# ===================== PROCESSAMENTO =====================
def processar_ativo(tkr, info, hist, estrategia_ativa, filtros_ativos, f_pl, f_pvp, f_dy, f_div_ebitda, busca_direta, mercado):
    if hist.empty or not info:
        return None

    hist = hist.copy()
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

    # Notícias
    lista_links = []
    try:
        lang = "pt-BR" if mercado == "Brasil" else "en-US"
        url = f"https://news.google.com/rss/search?q={tkr}&hl={lang}"
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            lista_links.append({"titulo": entry.title, "link": entry.link})
    except:
        pass

    rsi_val = calcular_rsi_series(hist["Close"]).iloc[-1]
    score_value, criteria = calcular_score_value(info)
    sim = simular_performance_historica(hist)

    veredito = "VALOR ✅" if score_value >= 3 else "NEUTRO ⚖️"
    motivo_detalhe = f"Value Score: {score_value}/4"

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
        "Motivo": motivo_detalhe,
        "RSI": rsi_val,
        "Hist": hist,
        "Links": lista_links,
        "ValueScore": score_value,
        "ValueCriteria": criteria,
        "ExpectancyCompra": sim["expectancy_compra"],
        "SharpeCompra": sim["sharpe_compra"],
        "QtdCompra": sim["qtd_compra"]
    }

# ===================== SIDEBAR =====================
st.sidebar.title("🌎 Monitor IA Pro")

st.sidebar.subheader("📈 Índices Mundiais")
indices_data = obter_indices()
for nome, (valor, var) in indices_data.items():
    st.sidebar.metric(nome, f"{valor:,.0f} pts", f"{var:.2f}%")

st.sidebar.divider()
st.sidebar.subheader("💱 Câmbio em Tempo Real")
cambio = obter_cambio()
col1, col2 = st.sidebar.columns(2)
col1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
col2.metric("Euro", f"R$ {cambio['Euro'][0]:.2f}", f"{cambio['Euro'][1]:.2f}%")
col3, col4 = st.sidebar.columns(2)
col3.metric("Libra", f"R$ {cambio['Libra'][0]:.2f}", f"{cambio['Libra'][1]:.2f}%")
col4.metric("Bitcoin", f"R$ {cambio['Bitcoin'][0]:,.0f}", f"{cambio['Bitcoin'][1]:.2f}%")

st.sidebar.divider()
st.sidebar.subheader("📊 Macro & Cenário")
macro = obter_macro()
st.sidebar.metric("Selic Atual", f"{macro['Selic']:.2f}%")
st.sidebar.metric("IPCA 12m", f"{macro['IPCA_12m']:.2f}%")

# ===================== LISTA DE ATIVOS =====================
mercado_selecionado = st.sidebar.radio("Mercado:", ["Brasil", "EUA"])
if mercado_selecionado == "Brasil":
    lista_base = ["PETR4", "VALE3", "ITUB4", "BBAS3", "B3SA3", "EGIE3", "WEGE3", "PRIO3", "JBSS3"]
    moeda_simbolo = "R$"
else:
    lista_base = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
    moeda_simbolo = "US$"

busca_direta = st.sidebar.text_input(f"🔍 Busca Rápida ({mercado_selecionado}):").upper().strip()
tickers_para_processar = [busca_direta] if busca_direta else lista_base

# ===================== PROCESSAMENTO =====================
dados_vencedoras = []
if tickers_para_processar:
    with st.spinner("📡 Baixando dados..."):
        infos, hists = obter_dados_batch(tickers_para_processar, mercado_selecionado)
    for tkr in tickers_para_processar:
        info = infos.get(tkr, {})
        hist = hists.get(tkr, pd.DataFrame())
        resultado = processar_ativo(tkr, info, hist, "Value Investing (Graham/Buffett)", False, 25, 3, 4, 3, busca_direta, mercado_selecionado)
        if resultado:
            dados_vencedoras.append(resultado)

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Gráfico Técnico", "📉 Fundamentalista", "📜 Backtest"])

st.title(f"🤖 Monitor IA - {mercado_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# ===================== TAB 2 - GRÁFICO TÉCNICO (MELHORADO) =====================
with tab2:
    st.subheader("📈 Gráfico Técnico")
    if dados_vencedoras:
        for acao in dados_vencedoras:
            hist = acao["Hist"]
            if hist.empty:
                continue

            # Cálculo de indicadores
            hist = hist.copy()
            hist['SMA20'] = hist['Close'].rolling(20).mean()
            hist['SMA200'] = hist['Close'].rolling(200).mean()
            hist['BB_Mid'] = hist['Close'].rolling(20).mean()
            hist['BB_Std'] = hist['Close'].rolling(20).std()
            hist['BB_Upper'] = hist['BB_Mid'] + 2 * hist['BB_Std']
            hist['BB_Lower'] = hist['BB_Mid'] - 2 * hist['BB_Std']
            rsi = calcular_rsi_series(hist['Close'])

            # Gráfico com subplots
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.08, 
                                row_heights=[0.6, 0.2, 0.2],
                                subplot_titles=(f"{acao['Ticker']} - Candlestick + Bollinger", "Volume", "RSI"))

            # Candlestick + Bollinger + SMAs
            fig.add_trace(go.Candlestick(x=hist.index,
                                         open=hist['Open'], high=hist['High'],
                                         low=hist['Low'], close=hist['Close'],
                                         name="Preço"), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA20'], name="SMA 20", line=dict(color='yellow')), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], name="SMA 200", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], name="BB Upper", line=dict(color='rgba(0,255,0,0.5)')), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], name="BB Lower", line=dict(color='rgba(255,0,0,0.5)')), row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name="Volume", marker_color='rgba(100,100,100,0.6)'), row=2, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=hist.index, y=rsi, name="RSI", line=dict(color='purple')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            fig.update_layout(height=700, template="plotly_dark", showlegend=True, 
                              title_text=f"Análise Técnica - {acao['Empresa']} ({acao['Ticker']})")
            fig.update_xaxes(rangeslider_visible=False)

            st.plotly_chart(fig, use_container_width=True, key=f"chart_{acao['Ticker']}")

            st.divider()
    else:
        st.info("Use a busca direta ou aguarde carregamento dos ativos.")

# ===================== OUTRAS TABS (mantidas) =====================
with tab1:
    if dados_vencedoras:
        st.subheader("🏆 Ranking")
        df = pd.DataFrame(dados_vencedoras)
        st.dataframe(df[["Ticker", "Preço", "DY %", "Veredito"]], use_container_width=True, hide_index=True)
    else:
        st.info("Use os filtros ou faça uma busca direta para começar.")

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
                st.markdown("**Últimas Manchetes:**")
                for n in acao["Links"]:
                    st.markdown(f"• [{n['titulo']}]({n['link']})")
            st.divider()
    else:
        st.info("Nenhum ativo encontrado.")

with tab4:
    st.subheader("📜 Backtest & Estatísticas")
    if dados_vencedoras:
        df = pd.DataFrame(dados_vencedoras)
        col1, col2, col3 = st.columns(3)
        col1.metric("Ativos Analisados", len(df))
        col2.metric("Média Expectancy", f"{df['ExpectancyCompra'].mean():.2f}%")
        col3.metric("Média Sharpe", f"{df['SharpeCompra'].mean():.2f}")
        st.dataframe(df[["Ticker", "ExpectancyCompra", "SharpeCompra", "QtdCompra"]].round(2), use_container_width=True, hide_index=True)
    else:
        st.info("Execute uma análise para ver estatísticas.")

st.info("💡 Use a busca rápida para visualizar gráficos técnicos.")
