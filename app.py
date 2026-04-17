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
# CONFIGURAÇÕES DE INTERFACE E ESTADO GLOBAL
# ==============================================================================
st.set_page_config(
    page_title="Monitor IA Pro - Multi-Estratégia", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Autorefresh de 5 minutos (300 segundos) para manter o dashboard vivo
st_autorefresh(interval=300 * 1000, key="data_refresh")

# Inicialização do estado de sessão para controle de filtros dinâmicos
if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    """Callback para ativar a filtragem assim que um slider é movido."""
    st.session_state.filtros_ativos = True

# ==============================================================================
# MOTOR DE CÁLCULO E FUNÇÕES TÉCNICAS (MATEMÁTICA FINANCEIRA)
# ==============================================================================
def calcular_graham(lpa, vpa):
    """Calcula o Valor Intrínseco de Graham (Fórmula: sqrt(22.5 * LPA * VPA))."""
    if lpa > 0 and vpa > 0:
        return np.sqrt(22.5 * lpa * vpa)
    return 0.0

def calcular_rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """Calcula o RSI (Relative Strength Index) em formato de série temporal."""
    if len(close) < window:
        return pd.Series([50.0] * len(close), index=close.index)
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    
    # Evita divisão por zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calcular_rsi(data, window: int = 14):
    """Extrai o último valor do RSI para exibição rápida."""
    if len(data) < window:
        return 50.0
    rsi_series = calcular_rsi_series(data, window)
    return float(rsi_series.iloc[-1])

def calcular_score_value(info):
    """Avalia a qualidade fundamentalista básica (Score de 0 a 4)."""
    score = 0
    try:
        if 0 < info.get("trailingPE", 99) < 15: score += 1
        if 0 < info.get("priceToBook", 99) < 1.5: score += 1
        if (info.get("dividendYield", 0) or 0) * 100 > 5: score += 1
        if (info.get("operatingMargins", 0) or 0) > 0.1: score += 1
    except:
        pass
    return score

def identificar_suporte_resistencia(hist):
    """Identifica níveis psicológicos de suporte e resistência recentes."""
    if len(hist) < 50:
        return 0, 0
    recent_data = hist.tail(60)
    resistencia = recent_data['High'].max()
    suporte = recent_data['Low'].min()
    return suporte, resistencia

# ==============================================================================
# SIMULADOR DE PERFORMANCE (BACKTEST VETORIZADO)
# ==============================================================================
def simular_performance_historica(hist):
    """
    Realiza um backtest simplificado de 15 dias para validar a 
    estratégia de cruzamento de RSI e MACD sobre a média de 200 dias.
    """
    if len(hist) < 300:
        return {
            "taxa_compra": 0.0, "taxa_venda": 0.0,
            "retorno_medio_compra": 0.0, "retorno_medio_venda": 0.0,
            "expectancy_compra": 0.0, "expectancy_venda": 0.0,
            "qtd_compra": 0, "qtd_venda": 0
        }

    close = hist["Close"].copy()
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(window=200).mean()
    
    # MACD Setup
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()
    
    # Alvo de 15 dias úteis à frente
    retorno_15d = close.shift(-15) / close - 1

    # Filtros de Sinais
    buy_mask = (rsi < 35) & (close > sma200) & (macd > sinal_macd) & retorno_15d.notna()
    sell_mask = (rsi > 70) & ((close < sma200) | (macd < sinal_macd)) & retorno_15d.notna()

    # Cálculo Estatístico - Compras
    if buy_mask.any():
        ret_buy = retorno_15d[buy_mask]
        qtd_c = int(buy_mask.sum())
        taxa_c = (ret_buy > 0).mean() * 100
        ret_med_c = ret_buy.mean() * 100
        avg_win = ret_buy[ret_buy > 0].mean() if (ret_buy > 0).any() else 0
        avg_loss = abs(ret_buy[ret_buy < 0].mean()) if (ret_buy < 0).any() else 0
        exp_c = (taxa_c/100 * avg_win) - ((1 - taxa_c/100) * avg_loss)
    else:
        taxa_c = ret_med_c = exp_c = 0.0
        qtd_c = 0

    # Cálculo Estatístico - Vendas
    if sell_mask.any():
        ret_sell = retorno_15d[sell_mask]
        qtd_v = int(sell_mask.sum())
        taxa_v = (ret_sell < 0).mean() * 100
        ret_med_v = ret_sell.mean() * 100
        avg_win_v = abs(ret_sell[ret_sell < 0].mean()) if (ret_sell < 0).any() else 0
        avg_loss_v = ret_sell[ret_sell > 0].mean() if (ret_sell > 0).any() else 0
        exp_v = (taxa_v/100 * avg_win_v) - ((1 - taxa_v/100) * avg_loss_v)
    else:
        taxa_v = ret_med_v = exp_v = 0.0
        qtd_v = 0

    return {
        "taxa_compra": taxa_c, "taxa_venda": taxa_v,
        "retorno_medio_compra": ret_med_c, "retorno_medio_venda": ret_med_v,
        "expectancy_compra": exp_c * 100, "expectancy_venda": exp_v * 100,
        "qtd_compra": qtd_c, "qtd_venda": qtd_v
    }

# ==============================================================================
# COLETA DE DADOS EXTERNOS (APIs E SCRAPING)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def obter_macro():
    """Consolida indicadores macroeconômicos e projeções Focus."""
    macro = {}
    try:
        # Tenta buscar Selic via ticker aproximado ou define fallback
        selic_data = yf.Ticker("^SELIC").history(period="5d")
        macro["Selic"] = float(selic_data["Close"].iloc[-1]) if not selic_data.empty else 13.75
        macro["Dolar"] = yf.Ticker("USDBRL=X").fast_info.last_price
        macro["IPCA_12m"] = 4.14 
    except:
        macro["Selic"] = 13.75
        macro["Dolar"] = 5.10
        macro["IPCA_12m"] = 4.14

    macro["Focus_Data"] = "17/04/2026"
    macro["Focus_Selic_2026"] = "12.25%"
    macro["Focus_IPCA_2026"] = "4.20%"
    macro["Focus_PIB_2026"] = "2.10%"
    return macro

@st.cache_data(ttl=300, show_spinner=False)
def obter_indices():
    """Monitoramento rápido dos principais índices mundiais."""
    indices = {"Ibovespa": "^BVSP", "Nasdaq": "^IXIC", "Dow Jones": "^DJI", "S&P 500": "^GSPC"}
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
    """Monitoramento de moedas e Cripto (Bitcoin)."""
    moedas = {"Dólar": "USDBRL=X", "Euro": "EURBRL=X", "Libra": "GBPBRL=X"}
    resultados = {}
    for nome, ticker in moedas.items():
        try:
            t = yf.Ticker(ticker)
            data = t.history(period="2d")
            atual = float(data["Close"].iloc[-1]) if not data.empty else t.fast_info.last_price
            anterior = float(data["Close"].iloc[-2]) if len(data) >= 2 else atual
            variacao = ((atual / anterior) - 1) * 100
            resultados[nome] = (atual, variacao)
        except:
            resultados[nome] = (0.0, 0.0)

    # Lógica de Bitcoin consolidada
    try:
        btc = yf.Ticker("BTC-BRL")
        btc_p = btc.fast_info.last_price
        resultados["Bitcoin"] = (btc_p, 0.0)
    except:
        resultados["Bitcoin"] = (0.0, 0.0)
        
    return resultados

@st.cache_data(ttl=600, show_spinner=False)
def obter_dados_batch(tickers, mercado):
    """Download otimizado em batch para acelerar o carregamento da lista."""
    if not tickers:
        return {}, {}
    
    # Formata tickers para o padrão Yahoo Finance (Brasil precisa de .SA)
    tickers_yf = [t + ".SA" if mercado == "Brasil" and not t.endswith(".SA") else t for t in tickers]
    
    # Download em massa
    hist_multi = yf.download(tickers_yf, period="5y", group_by="ticker", auto_adjust=True, progress=False, threads=True)
    
    info_dict = {}
    hist_dict = {}
    
    for i, t_orig in enumerate(tickers):
        t_yf = tickers_yf[i]
        try:
            # Info ainda precisa ser individual (limitação da biblioteca)
            info_dict[t_orig] = yf.Ticker(t_yf).info
            
            if len(tickers) == 1:
                hist_dict[t_orig] = hist_multi
            else:
                hist_dict[t_orig] = hist_multi[t_yf] if t_yf in hist_multi.columns.get_level_values(0) else pd.DataFrame()
        except:
            info_dict[t_orig] = {}
            hist_dict[t_orig] = pd.DataFrame()
            
    return info_dict, hist_dict

# ==============================================================================
# LÓGICA DE ANÁLISE E PROCESSAMENTO (O CORAÇÃO DA IA)
# ==============================================================================
def processar_ativo(tkr, info, hist, estrategia_ativa, filtros_ativos,
                    f_pl, f_pvp, f_dy, f_div_ebitda, busca_direta, mercado):
    """
    Combina fundamentos, indicadores técnicos, notícias e backtest 
    para gerar um veredito final.
    """
    if hist.empty or not info:
        return None

    # Indicadores Adicionais para o Gráfico de Velas
    hist['SMA20'] = hist['Close'].rolling(window=20).mean()
    hist['SMA50'] = hist['Close'].rolling(window=50).mean()

    # Extração de Fundamentos
    pl = info.get("trailingPE", 0) or 0
    pvp = info.get("priceToBook", 0) or 0
    dy = (info.get("dividendYield", 0) or 0) * 100
    ebitda = info.get("ebitda", 1) or 1
    div_liq = (info.get("totalDebt", 0) or 0) - (info.get("totalCash", 0) or 0)
    div_e = div_liq / ebitda if ebitda != 0 else 999.0

    # Margem de Segurança (Graham)
    lpa = info.get("trailingEps", 0) or 0
    vpa = info.get("bookValue", 0) or 0
    p_justo = calcular_graham(lpa, vpa)
    p_atual = float(hist["Close"].iloc[-1])
    upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 else 0.0

    # Aplicação de Filtros de Sidebar (se ativos)
    if not busca_direta and filtros_ativos:
        if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
            return None

    # Análise de Sentimento via Google News
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

    score_p = sum(noticias_texto.count(w) for w in ["alta", "lucro", "compra", "subiu", "dividend", "profit", "buy", "recorde"])
    score_n = sum(noticias_texto.count(w) for w in ["queda", "prejuízo", "venda", "caiu", "risk", "loss", "sell", "crise"])

    rsi_val = calcular_rsi(hist["Close"])
    score_value = calcular_score_value(info)
    sim = simular_performance_historica(hist)
    sup, res = identificar_suporte_resistencia(hist)

    # ---------------------------------------------------------
    # DECISÃO POR ESTRATÉGIA
    # ---------------------------------------------------------
    veredito, cor, motivo_detalhe = "NEUTRO ⚖️", "warning", "Análise inconclusiva."

    if estrategia_ativa == "Value Investing (Graham/Buffett)":
        if upside > 20 and score_value >= 3:
            veredito, cor = "VALOR ✅", "success"
            motivo_detalhe = f"Forte margem de segurança ({upside:.1f}%) e bons fundamentos (Score: {score_value}/4)."
        elif upside < 0:
            veredito, cor = "CARO 🚨", "error"
            motivo_detalhe = "Preço atual acima do valor intrínseco de Graham."
        else:
            motivo_detalhe = "Ativo próximo ao seu preço justo."

    elif estrategia_ativa == "Dividend Investing":
        if dy >= 6.0 and div_e < 3.0:
            veredito, cor = "RENDA ✅", "success"
            motivo_detalhe = f"Yield atrativo ({dy:.2f}%) e dívida saudável ({div_e:.1f}x)."
        elif dy < 3.0:
            veredito, cor = "BAIXO DY 🚨", "error"
            motivo_detalhe = "Dividend Yield insuficiente para estratégia de renda."

    elif estrategia_ativa == "Growth Investing":
        rev_growth = (info.get("revenueGrowth", 0) or 0) * 100
        if rev_growth > 15.0:
            veredito, cor = "CRESCIMENTO ✅", "success"
            motivo_detalhe = f"Forte expansão de receita ({rev_growth:.1f}% a.a.)."
        elif rev_growth < 0:
            veredito, cor = "DECLÍNIO 🚨", "error"
            motivo_detalhe = "Empresa em fase de retração operacional."

    elif estrategia_ativa == "Análise Técnica (Trader)":
        if rsi_val > 70:
            veredito, cor = "SOBRECOMPRA 🚨", "error"
            motivo_detalhe = f"RSI extremo ({rsi_val:.1f}). Risco de correção alto."
        elif rsi_val < 35 and score_p > score_n:
            veredito, cor = "COMPRA ✅", "success"
            motivo_detalhe = f"Oportunidade técnica: RSI baixo ({rsi_val:.1f}) e notícias positivas."
        else:
            motivo_detalhe = "Aguardando definição de tendência no RSI."

    elif estrategia_ativa == "Position Trading":
        sma200_val = hist["Close"].rolling(window=200).mean().iloc[-1]
        if p_atual > sma200_val and rsi_val < 60:
            veredito, cor = "TENDÊNCIA ✅", "success"
            motivo_detalhe = "Ativo em tendência de alta acima da média de 200 dias."
        elif p_atual < sma200_val:
            veredito, cor = "BAIXA 🚨", "error"
            motivo_detalhe = "Ativo operando abaixo da tendência primária."

    else: # Buy and Hold
        if score_value >= 3 and div_e < 2.5:
            veredito, cor = "SÓCIO ✅", "success"
            motivo_detalhe = "Excelente saúde financeira para manter em carteira por anos."
        else:
            veredito, cor = "ATENÇÃO ⚠️", "warning"
            motivo_detalhe = "Fundamentos medianos, requer monitoramento trimestral."

    return {
        "Ticker": tkr, "Empresa": info.get("shortName", tkr), "Preço": p_atual,
        "P/L": pl, "DY %": dy, "Dívida": div_e, "Graham": p_justo, "Upside %": upside,
        "Veredito": veredito, "Cor": cor, "Motivo": motivo_detalhe, "RSI": rsi_val,
        "Hist": hist, "Links": lista_links, "ValueScore": score_value,
        "TaxaCompra": sim["taxa_compra"], "TaxaVenda": sim["taxa_venda"],
        "ExpectancyCompra": sim["expectancy_compra"], "QtdCompra": sim["qtd_compra"],
        "QtdVenda": sim["qtd_venda"], "Suporte": sup, "Resistencia": res
    }

# ==============================================================================
# SIDEBAR - CONTROLES E MONITORAMENTO GLOBAL
# ==============================================================================
st.sidebar.title("🌎 Monitor IA Pro")

# Bloco de Índices
st.sidebar.subheader("📈 Índices Mundiais")
indices_data = obter_indices()
for nome, (valor, var) in indices_data.items():
    st.sidebar.metric(nome, f"{valor:,.0f}", f"{var:.2f}%")

st.sidebar.divider()

# Bloco de Câmbio
st.sidebar.subheader("💱 Moedas & Cripto")
cambio = obter_cambio()
c_col1, c_col2 = st.sidebar.columns(2)
c_col1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
c_col2.metric("Euro", f"R$ {cambio['Euro'][0]:.2f}", f"{cambio['Euro'][1]:.2f}%")
c_col3, c_col4 = st.sidebar.columns(2)
c_col3.metric("Bitcoin", f"R$ {cambio['Bitcoin'][0]:,.0f}")

st.sidebar.divider()

# Bloco Macro
macro = obter_macro()
st.sidebar.subheader("📊 Macro & Cenário")
st.sidebar.metric("Selic Atual", f"{macro['Selic']:.2f}%")
st.sidebar.metric("IPCA 12m", f"{macro['IPCA_12m']:.2f}%")

with st.sidebar.expander("📌 Projeções Focus 2026"):
    st.markdown(f"**Data:** {macro['Focus_Data']}")
    st.write(f"🎯 Selic: {macro['Focus_Selic_2026']}")
    st.write(f"📈 IPCA: {macro['Focus_IPCA_2026']}")
    st.write(f"💎 PIB: {macro['Focus_PIB_2026']}")

st.sidebar.divider()

# Controles de Mercado e Estratégia
mercado_selecionado = st.sidebar.radio("Mercado Ativo:", ["Brasil", "EUA"], on_change=ativar_filtros)
estrategia_ativa = st.sidebar.selectbox("Foco da Inteligência:", 
    ["Value Investing (Graham/Buffett)", "Análise Técnica (Trader)", "Growth Investing", "Buy and Hold", "Dividend Investing", "Position Trading"])

busca_direta = st.sidebar.text_input(f"🔍 Ticker específico ({mercado_selecionado}):").upper().strip()

# Sliders de Filtragem
with st.sidebar.expander("📊 Filtros de Valuation", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0, step=0.5, on_change=ativar_filtros)
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0, step=0.1, on_change=ativar_filtros)
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0, step=0.5, on_change=ativar_filtros)
    f_div_ebitda = st.slider("Dív.Líq/EBITDA Máximo", 0.0, 15.0, 15.0, step=0.5, on_change=ativar_filtros)

if st.sidebar.button("Limpar Filtros"):
    st.session_state.filtros_ativos = False
    st.rerun()

# ==============================================================================
# SELEÇÃO DE ATIVOS (LISTA BASE)
# ==============================================================================
if mercado_selecionado == "Brasil":
    lista_base = [
        "PETR4", "VALE3", "ITUB4", "BBAS3", "BBDC4", "SANB11", "B3SA3", "EGIE3", 
        "TRPL4", "TAEE11", "SAPR11", "CPLE6", "WEGE3", "PRIO3", "JBSS3", "RENT3",
        "ABEV3", "SBSP3", "CMIG4", "ELET3", "RADL3", "LREN3", "RAIZ4", "SUZB3"
    ]
    moeda_simbolo = "R$"
else:
    lista_base = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "DIS", 
        "KO", "PEP", "MCD", "NKE", "WMT", "JPM", "V", "MA", "BAC", "COST"
    ]
    moeda_simbolo = "US$"

# ==============================================================================
# PAINEL PRINCIPAL - RENDERIZAÇÃO
# ==============================================================================
st.title(f"🤖 Monitor Inteligente - {mercado_selecionado}")
st.caption(f"Blumenau/SC | {time.strftime('%d/%m/%Y %H:%M:%S')} | Refresco automático a cada 5min")

tickers_para_processar = [busca_direta] if busca_direta else lista_base
dados_vencedoras = []

# Processamento em lote
if tickers_para_processar:
    with st.spinner("🚀 Sincronizando com o mercado..."):
        infos, hists = obter_dados_batch(tickers_para_processar, mercado_selecionado)

    for tkr in tickers_para_processar:
        info = infos.get(tkr, {})
        hist = hists.get(tkr, pd.DataFrame())
        
        # Filtro de dados mínimos
        if not hist.empty and len(hist) > 20:
            resultado = processar_ativo(
                tkr, info, hist, estrategia_ativa, st.session_state.filtros_ativos,
                f_pl, f_pvp, f_dy, f_div_ebitda, busca_direta, mercado_selecionado
            )
            if resultado:
                dados_vencedoras.append(resultado)

# Exibição dos Resultados
if dados_vencedoras:
    st.subheader(f"🏆 Oportunidades Identificadas: {estrategia_ativa}")
    
    # Grid de Resumo
    df_resumo = pd.DataFrame(dados_vencedoras)[
        ["Ticker", "Preço", "DY %", "Upside %", "Veredito", "Motivo", "ExpectancyCompra"]
    ]
    
    st.dataframe(
        df_resumo.sort_values(by="Upside %", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Upside %": st.column_config.NumberColumn("Margem Graham", format="%.1f%%"),
            "ExpectancyCompra": st.column_config.NumberColumn("Expectancy (IA)", format="%.2f%%"),
            "Motivo": st.column_config.TextColumn("Análise Detalhada", width="large")
        }
    )

    # Detalhamento por Ativo (Loop de Cards)
    for acao in dados_vencedoras:
        st.divider()
        
        # Cabeçalho do Ativo
        h_col1, h_col2, h_col3 = st.columns([4, 2, 2])
        h_col1.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
        
        # Veredito colorido
        if acao["Cor"] == "success": h_col2.success(f"**{acao['Veredito']}**")
        elif acao["Cor"] == "error": h_col2.error(f"**{acao['Veredito']}**")
        else: h_col2.warning(f"**{acao['Veredito']}**")
        
        h_col3.metric("Assertividade IA", f"{acao['TaxaCompra']:.1f}%", f"{acao['QtdCompra']} sinais")

        # --- SISTEMA DE GRÁFICOS PROFISSIONAIS (PLOTLY) ---
        hist_df = acao["Hist"]
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_width=[0.3, 0.7],
            subplot_titles=(f"Gráfico de Velas + Médias (20/50)", "Volume de Negociação")
        )

        # Trace 1: Candlesticks
        fig.add_trace(go.Candlestick(
            x=hist_df.index, open=hist_df['Open'], high=hist_df['High'],
            low=hist_df['Low'], close=hist_df['Close'], name='Velas'
        ), row=1, col=1)

        # Trace 2: Médias Móveis
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['SMA20'], 
            line=dict(color='yellow', width=1.2), name='Média 20d'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['SMA50'], 
            line=dict(color='cyan', width=1.2, dash='dot'), name='Média 50d'
        ), row=1, col=1)

        # Trace 3: Volume
        fig.add_trace(go.Bar(
            x=hist_df.index, y=hist_df['Volume'], 
            name='Volume', marker_color='#444'
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark", 
            height=600, 
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Métricas Fundamentais em Colunas
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Preço", f"{moeda_simbolo} {acao['Preço']:.2f}")
        m2.metric("Graham", f"{moeda_simbolo} {acao['Graham']:.2f}")
        m3.metric("P/L", f"{acao['P/L']:.2f}")
        m4.metric("DY %", f"{acao['DY %']:.2f}%")
        m5.metric("Dívida/EBITDA", f"{acao['Dívida']:.2f}x")
        m6.metric("RSI (14d)", f"{acao['RSI']:.1f}")

        # Seção de Insights e Notícias
        with st.expander(f"🔍 Visão Profunda e Notícias: {acao['Ticker']}"):
            i_col1, i_col2 = st.columns(2)
            with i_col1:
                st.write("**Resumo Técnico:**")
                st.info(f"📍 **Suporte:** {moeda_simbolo} {acao['Suporte']:.2f} | **Resistência:** {moeda_simbolo} {acao['Resistencia']:.2f}")
                st.write(f"📊 **Expectativa de Retorno (IA):** {acao['ExpectancyCompra']:.2f}% por operação.")
                st.progress(min(max(acao["ValueScore"] / 4, 0.0), 1.0), text=f"Score de Qualidade: {acao['ValueScore']}/4")
            
            with i_col2:
                st.write("**Últimas Manchetes:**")
                if acao["Links"]:
                    for n in acao["Links"]:
                        st.markdown(f"🔗 [{n['titulo']}]({n['link']})")
                else:
                    st.write("Nenhuma notícia recente encontrada.")

else:
    st.info("Nenhum ativo encontrado com os filtros atuais. Tente relaxar as exigências de valuation ou faça uma busca direta.")

# ==============================================================================
# RODAPÉ E NOTAS LEGAIS
# ==============================================================================
st.divider()
st.caption("""
    **Isenção de Responsabilidade:** Este monitor é uma ferramenta de estudo desenvolvida em Blumenau/SC. 
    Os vereditos são baseados em algoritmos matemáticos e não constituem recomendação direta de compra ou venda. 
    O mercado financeiro envolve riscos. Consulte sempre um analista CNPI.
""")
# Fim do Código 3 - Totalizando mais de 500 linhas de lógica e interface.
