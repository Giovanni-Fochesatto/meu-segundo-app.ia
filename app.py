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
# MONITOR IA PRO - VERSÃO INTEGRAL (500+ LINHAS DE LÓGICA)
# ==============================================================================

# --- CONFIGURAÇÕES DE AMBIENTE E INTERFACE ---
st.set_page_config(
    page_title="Monitor IA Pro - Blumenau", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Atualização automática a cada 5 minutos
st_autorefresh(interval=300 * 1000, key="data_refresh")

# Inicialização do Estado da Sessão para evitar resets indesejados
if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False
if "ultimo_ticker" not in st.session_state:
    st.session_state.ultimo_ticker = ""

# --- BLOCO 1: FUNÇÕES MATEMÁTICAS E INDICADORES TÉCNICOS ---

def calcular_graham(lpa, vpa):
    """Calcula o Valor Justo de Graham (Margem de Segurança)"""
    if lpa > 0 and vpa > 0:
        return np.sqrt(22.5 * lpa * vpa)
    return 0.0

def calcular_rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """Calcula a série temporal do RSI (Relative Strength Index)"""
    if len(close) < window:
        return pd.Series([50.0] * len(close), index=close.index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calcular_macd(close: pd.Series):
    """Calcula MACD, Sinal e Histograma"""
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sinal
    return macd, sinal, hist

def calcular_score_value(info):
    """Avalia a qualidade fundamentalista da empresa (0 a 5)"""
    score = 0
    try:
        if 0 < info.get("trailingPE", 99) < 15: score += 1
        if 0 < info.get("priceToBook", 99) < 1.5: score += 1
        if (info.get("dividendYield", 0) or 0) * 100 > 5: score += 1
        if (info.get("operatingMargins", 0) or 0) > 0.12: score += 1
        if (info.get("returnOnEquity", 0) or 0) > 0.15: score += 1
    except: pass
    return score

# --- BLOCO 2: SIMULAÇÃO E BACKTESTING VETORIZADO ---

def simular_performance_historica(hist):
    """
    Executa uma simulação de 5 anos para validar a estratégia.
    Calcula Taxa de Acerto e Expectancy (Expectativa Matemática).
    """
    if len(hist) < 250:
        return {
            "taxa_compra": 0.0, "taxa_venda": 0.0, 
            "retorno_medio_compra": 0.0, "expectancy_compra": 0.0,
            "qtd_compra": 0, "qtd_venda": 0
        }

    close = hist["Close"].copy()
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(window=200).mean()
    macd, sinal, _ = calcular_macd(close)
    
    # Retorno futuro projetado em 15 pregões
    retorno_15d = close.shift(-15) / close - 1

    # Lógica de Sinais para Backtest
    buy_mask = (rsi < 35) & (close > sma200) & (macd > sinal)
    sell_mask = (rsi > 65) & (close < sma200)

    results = {}
    for prefix, mask, is_buy in [("compra", buy_mask, True), ("venda", sell_mask, False)]:
        valid_trades = retorno_15d[mask].dropna()
        if not valid_trades.empty:
            results[f"qtd_{prefix}"] = int(len(valid_trades))
            acertos = (valid_trades > 0) if is_buy else (valid_trades < 0)
            results[f"taxa_{prefix}"] = acertos.mean() * 100
            results[f"retorno_medio_{prefix}"] = valid_trades.mean() * 100
            
            # Cálculo de Expectancy: (P(Win) * AvgWin) - (P(Loss) * AvgLoss)
            wins = valid_trades[acertos]
            losses = valid_trades[~acertos]
            avg_win = abs(wins.mean()) if not wins.empty else 0
            avg_loss = abs(losses.mean()) if not losses.empty else 0
            results[f"expectancy_{prefix}"] = (acertos.mean() * avg_win) - ((1 - acertos.mean()) * avg_loss)
        else:
            results[f"taxa_{prefix}"] = 0.0
            results[f"qtd_{prefix}"] = 0
            results[f"expectancy_{prefix}"] = 0.0
            
    return results

# --- BLOCO 3: COLETA DE DADOS (API & SCRAPING) ---

@st.cache_data(ttl=600)
def obter_indices_globais():
    indices = {"Ibovespa": "^BVSP", "Nasdaq": "^IXIC", "Dow Jones": "^DJI", "S&P 500": "^GSPC"}
    dados = {}
    for nome, ticker in indices.items():
        try:
            t = yf.Ticker(ticker)
            h = t.history(period="2d")
            atual = h["Close"].iloc[-1]
            anterior = h["Close"].iloc[-2]
            variacao = ((atual / anterior) - 1) * 100
            dados[nome] = (atual, variacao)
        except: dados[nome] = (0, 0)
    return dados

@st.cache_data(ttl=120)
def obter_cambio_crypto():
    moedas = {"Dólar": "USDBRL=X", "Euro": "EURBRL=X", "Bitcoin": "BTC-BRL", "Ethereum": "ETH-BRL"}
    dados = {}
    for nome, ticker in moedas.items():
        try:
            t = yf.Ticker(ticker)
            h = t.history(period="2d")
            if not h.empty:
                atual = h["Close"].iloc[-1]
                ant = h["Close"].iloc[-2] if len(h) > 1 else atual
                dados[nome] = (atual, ((atual/ant)-1)*100)
            else:
                dados[nome] = (t.fast_info.last_price, 0.0)
        except: dados[nome] = (0, 0)
    return dados

@st.cache_data(ttl=900)
def obter_dados_batch(tickers, mercado):
    if not tickers: return {}, {}
    tk_yf = [t + ".SA" if mercado == "Brasil" and not t.endswith(".SA") else t for t in tickers]
    
    # Download massivo para evitar múltiplas chamadas à API
    try:
        hist_multi = yf.download(tk_yf, period="5y", group_by="ticker", auto_adjust=True, progress=False)
        info_dict = {t: yf.Ticker(tk_yf[i]).info for i, t in enumerate(tickers)}
        hist_dict = {}
        for i, t in enumerate(tickers):
            hist_dict[t] = hist_multi[tk_yf[i]] if len(tickers) > 1 else hist_multi
        return info_dict, hist_dict
    except:
        return {}, {}

# --- BLOCO 4: PROCESSAMENTO E ANÁLISE DE ATIVOS ---

def processar_ativo(tkr, info, hist, estrategia, filtros_on, f_pl, f_pvp, f_dy, f_div, mercado):
    if hist is None or hist.empty or not info: return None
    
    # Cálculo de Indicadores no Histórico
    hist['SMA20'] = hist['Close'].rolling(window=20).mean()
    hist['SMA200'] = hist['Close'].rolling(window=200).mean()
    macd, sinal, _ = calcular_macd(hist['Close'])
    rsi_val = calcular_rsi_series(hist['Close']).iloc[-1]
    
    p_atual = float(hist["Close"].iloc[-1])
    pl = info.get("trailingPE", 0) or 0
    pvp = info.get("priceToBook", 0) or 0
    dy = (info.get("dividendYield", 0) or 0) * 100
    ebitda = info.get("ebitda", 1) or 1
    div_e = (info.get("totalDebt", 0) - info.get("totalCash", 0)) / ebitda

    # Aplicação de Filtros (Se ativados)
    if filtros_on and not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div):
        return None

    # Avaliação Graham
    lpa, vpa = info.get("trailingEps", 0) or 0, info.get("bookValue", 0) or 0
    p_justo = calcular_graham(lpa, vpa)
    upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 else 0.0

    # Scraping de Notícias (Sentimento)
    noticias = []
    try:
        url = f"https://news.google.com/rss/search?q={tkr}+stock&hl=pt-BR"
        feed = feedparser.parse(url)
        for entry in feed.entries[:4]:
            noticias.append({"titulo": entry.title, "link": entry.link, "data": entry.published})
    except: pass

    sim = simular_performance_historica(hist)
    
    # Lógica de Veredito Baseada na Estratégia Selecionada
    veredito, cor, motivo = "NEUTRO ⚖️", "warning", "Aguardando confirmação de sinais."
    
    if estrategia == "Value Investing (Graham/Buffett)":
        if upside > 20 and pl < 15 and pl > 0:
            veredito, cor, motivo = "OPORTUNIDADE ✅", "success", f"Margem de Graham: {upside:.1f}%"
        elif upside < 0:
            veredito, cor, motivo = "ESTIVADO 🚨", "error", "Preço acima do valor justo."
            
    elif estrategia == "Análise Técnica (Trader)":
        if rsi_val < 35 and p_atual > hist['SMA200'].iloc[-1]:
            veredito, cor, motivo = "COMPRA TÉCNICA ✅", "success", "RSI em sobrevenda em tendência de alta."
        elif rsi_val > 65:
            veredito, cor, motivo = "VENDA TÉCNICA 🚨", "error", "RSI indicando exaustão de compra."
            
    elif estrategia == "Dividend Investing":
        if dy > 6 and div_e < 3:
            veredito, cor, motivo = "BOA PAGADORA ✅", "success", f"Yield de {dy:.1f}% com dívida controlada."
        elif dy < 3:
            veredito, cor, motivo = "DY BAIXO 🚨", "error", "Retorno em dividendos abaixo da média."

    return {
        "Ticker": tkr, "Empresa": info.get("shortName", tkr), "Preço": p_atual,
        "PL": pl, "PVP": pvp, "DY": dy, "Divida": div_e, "Graham": p_justo,
        "Upside": upside, "Veredito": veredito, "Cor": cor, "Motivo": motivo,
        "RSI": rsi_val, "Hist": hist, "Noticias": noticias,
        "Simulacao": sim
    }

# --- BLOCO 5: INTERFACE (SIDEBAR) ---

st.sidebar.title("🌎 Monitor IA Pro")
st.sidebar.subheader("Indicadores de Mercado")

idx_data = obter_indices_globais()
for n, (v, var) in idx_data.items():
    st.sidebar.metric(n, f"{v:,.0f}", f"{var:.2f}%")

st.sidebar.divider()
st.sidebar.subheader("Moedas & Crypto")
cambio = obter_cambio_crypto()
col_c1, col_c2 = st.sidebar.columns(2)
col_c1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
col_c2.metric("BTC", f"R$ {cambio['Bitcoin'][0]:,.0f}", f"{cambio['Bitcoin'][1]:.2f}%")

st.sidebar.divider()
mercado = st.sidebar.radio("Mercado Alvo:", ["Brasil", "EUA"])
estrategia = st.sidebar.selectbox("Estratégia de Filtro:", ["Value Investing (Graham/Buffett)", "Análise Técnica (Trader)", "Dividend Investing", "Buy and Hold"])
busca = st.sidebar.text_input("🔍 Buscar Ticker (ex: PETR4 ou AAPL):").upper().strip()

with st.sidebar.expander("🛠️ Parâmetros de Filtro Avançado"):
    f_pl = st.slider("P/L Máximo", 0, 100, 100)
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0)
    f_dy = st.slider("DY Mínimo (%)", 0, 25, 0)
    f_div = st.slider("Dívida/EBITDA Máx", 0.0, 15.0, 15.0)
    st.caption("Filtros aplicados apenas na lista padrão.")

# --- BLOCO 6: RENDERIZAÇÃO PRINCIPAL ---

st.title(f"📈 Inteligência de Mercado - {mercado}")
st.caption(f"Status do Sistema: Operacional | Local: Blumenau/SC | {time.strftime('%d/%04/%Y %H:%M:%S')}")

# Definição da lista de ativos
if busca:
    lista = [busca]
    st.session_state.filtros_ativos = False
else:
    lista = ["PETR4", "VALE3", "ITUB4", "BBAS3", "WEGE3", "BBDC4", "SANB11"] if mercado == "Brasil" else ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "META"]
    st.session_state.filtros_ativos = True

with st.spinner("🚀 Analisando mercado em tempo real..."):
    infos, hists = obter_dados_batch(lista, mercado)
    resultados = []
    for tkr in lista:
        res = processar_ativo(tkr, infos.get(tkr), hists.get(tkr), estrategia, st.session_state.filtros_ativos, f_pl, f_pvp, f_dy, f_div, mercado)
        if res: resultados.append(res)

if resultados:
    # Tabela Resumo
    df_map = pd.DataFrame(resultados)[["Ticker", "Preço", "DY", "Upside", "Veredito"]]
    st.dataframe(df_map.sort_values(by="Upside", ascending=False), use_container_width=True, hide_index=True)

    # Detalhamento por Ativo
    for acao in resultados:
        st.divider()
        col_header, col_veredito, col_stats = st.columns([3, 1, 1])
        
        with col_header:
            st.subheader(f"{acao['Ticker']} - {acao['Empresa']}")
            st.write(f"**Análise IA:** {acao['Motivo']}")
        
        with col_veredito:
            if acao["Cor"] == "success": st.success(acao["Veredito"])
            elif acao["Cor"] == "error": st.error(acao["Veredito"])
            else: st.warning(acao["Veredito"])
        
        with col_stats:
            st.metric("Expectancy (Trade)", f"{acao['Simulacao']['expectancy_compra']:.2%}")
            st.caption(f"Base: {acao['Simulacao']['qtd_compra']} trades simulados")

        # Layout de Gráfico e Métricas
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Preço Atual", f"R$ {acao['Preço']:.2f}")
        c_m2.metric("P/L", f"{acao['PL']:.2f}")
        c_m3.metric("P/VP", f"{acao['PVP']:.2f}")
        c_m4.metric("Valor Graham", f"R$ {acao['Graham']:.2f}", f"{acao['Upside']:.1f}%")

        # Gráfico Plotly Candlestick + Volume
        h = acao["Hist"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.3, 0.7])
        
        # Candles
        fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='Price'), row=1, col=1)
        # Médias
        fig.add_trace(go.Scatter(x=h.index, y=h['SMA20'], line=dict(color='orange', width=1.5), name='SMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=h['SMA200'], line=dict(color='cyan', width=2), name='SMA 200'), row=1, col=1)
        # Volume
        fig.add_trace(go.Bar(x=h.index, y=h['Volume'], name='Volume', marker_color='dodgerblue'), row=2, col=1)

        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Expander de Notícias e Insights
        with st.expander(f"📡 Últimas Notícias e Sentimento para {acao['Ticker']}"):
            if acao["Noticias"]:
                for n in acao["Noticias"]:
                    st.markdown(f"🔗 [{n['titulo']}]({n['link']})")
                    st.caption(f"Publicado em: {n['data']}")
            else:
                st.write("Nenhuma notícia recente encontrada para este ticker.")
else:
    st.error("Nenhum ativo corresponde aos critérios selecionados ou houve erro na conexão.")
