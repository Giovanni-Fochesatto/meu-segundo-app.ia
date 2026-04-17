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
# DIFERENÇAS E ATUALIZAÇÕES (CÓDIGO 2 - VERSÃO PRO)
# 1. Integração Plotly: Adicionado suporte a gráficos interativos de Candlestick.
# 2. Análise de Volume: Inclusão de histograma de volume nos subplots.
# 3. Médias Móveis: Implementação da SMA 20 diretamente no gráfico para suporte visual.
# 4. Estratégias Completas: Finalização das lógicas de Position, B&H e Trader.
# 5. Robustez de Cache: Otimização do tempo de vida (TTL) para evitar bloqueios de API.
# 6. Contexto Regional: Localização configurada para Blumenau/SC no rodapé.
# ==============================================================================

# ===================== CONFIGURAÇÕES =====================
st.set_page_config(page_title="Monitor IA Pro", layout="wide", page_icon="📈")
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
    rs = gain / (loss + 1e-9) # Evita divisão por zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_rsi(data, window: int = 14):
    if len(data) < window:
        return 50.0
    rsi_series = calcular_rsi_series(data, window)
    return float(rsi_series.iloc[-1])

def calcular_score_value(info):
    score = 0
    if 0 < info.get("trailingPE", 99) < 15: score += 1
    if 0 < info.get("priceToBook", 99) < 1.5: score += 1
    if (info.get("dividendYield", 0) or 0) * 100 > 5: score += 1
    if (info.get("operatingMargins", 0) or 0) > 0.1: score += 1
    return score

# ===================== SIMULAÇÃO VETORIZADA =====================
def simular_performance_historica(hist):
    if len(hist) < 300:
        return {k: 0.0 for k in ["taxa_compra", "taxa_venda", "retorno_medio_compra", "retorno_medio_venda", "expectancy_compra", "expectancy_venda"]} | {"qtd_compra": 0, "qtd_venda": 0}

    close = hist["Close"].copy()
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(window=200).mean()
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()
    retorno_15d = close.shift(-15) / close - 1

    buy_mask = (rsi < 35) & (close > sma200) & (macd > sinal_macd) & retorno_15d.notna()
    sell_mask = (rsi > 70) & ((close < sma200) | (macd < sinal_macd)) & retorno_15d.notna()

    results = {}
    for prefix, mask, is_buy in [("compra", buy_mask, True), ("venda", sell_mask, False)]:
        if mask.any():
            ret = retorno_15d[mask]
            results[f"qtd_{prefix}"] = int(mask.sum())
            success_mask = ret > 0 if is_buy else ret < 0
            results[f"taxa_{prefix}"] = success_mask.mean() * 100
            results[f"retorno_medio_{prefix}"] = ret.mean() * 100
            avg_win = abs(ret[success_mask].mean()) if success_mask.any() else 0
            avg_loss = abs(ret[~success_mask].mean()) if (~success_mask).any() else 0
            results[f"expectancy_{prefix}"] = (results[f"taxa_{prefix}"]/100 * avg_win) - ((1 - results[f"taxa_{prefix}"]/100) * avg_loss)
        else:
            results[f"taxa_{prefix}"] = results[f"retorno_medio_{prefix}"] = results[f"expectancy_{prefix}"] = 0.0
            results[f"qtd_{prefix}"] = 0
    return results

# ===================== FUNÇÕES DE DADOS =====================
@st.cache_data(ttl=1800)
def obter_macro():
    macro = {"Selic": 14.75, "Dolar": 4.99, "IPCA_12m": 4.14, "Focus_Data": "13/04/2026", "Focus_Selic_2026": "12.50%"}
    try:
        selic_data = yf.Ticker("^SELIC").history(period="5d")
        if not selic_data.empty: macro["Selic"] = float(selic_data["Close"].iloc[-1])
        macro["Dolar"] = yf.Ticker("USDBRL=X").fast_info.last_price
    except: pass
    return macro

@st.cache_data(ttl=300)
def obter_indices():
    indices = {"Ibovespa": "^BVSP", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}
    res = {}
    for nome, ticker in indices.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            atual, anterior = data["Close"].iloc[-1], data["Close"].iloc[-2]
            res[nome] = (atual, ((atual / anterior) - 1) * 100)
        except: res[nome] = (0.0, 0.0)
    return res

@st.cache_data(ttl=90)
def obter_cambio():
    moedas = {"Dólar": "USDBRL=X", "Euro": "EURBRL=X", "Bitcoin": "BTC-BRL"}
    res = {}
    for nome, ticker in moedas.items():
        try:
            t = yf.Ticker(ticker)
            data = t.history(period="2d")
            atual = data["Close"].iloc[-1] if not data.empty else t.fast_info.last_price
            ant = data["Close"].iloc[-2] if len(data) > 1 else atual
            res[nome] = (atual, ((atual/ant)-1)*100 if ant != 0 else 0.0)
        except: res[nome] = (0.0, 0.0)
    return res

@st.cache_data(ttl=600)
def obter_dados_batch(tickers, mercado):
    if not tickers: return {}, {}
    tk_yf = [t + ".SA" if mercado == "Brasil" and not t.endswith(".SA") else t for t in tickers]
    hist_multi = yf.download(tk_yf, period="5y", group_by="ticker", auto_adjust=True, progress=False)
    info_dict, hist_dict = {}, {}
    for i, t_orig in enumerate(tickers):
        t_yf = tk_yf[i]
        try:
            info_dict[t_orig] = yf.Ticker(t_yf).info
            hist_dict[t_orig] = hist_multi[t_yf] if len(tickers) > 1 else hist_multi
        except: 
            info_dict[t_orig], hist_dict[t_orig] = {}, pd.DataFrame()
    return info_dict, hist_dict

# ===================== PROCESSAMENTO CENTRAL =====================
def processar_ativo(tkr, info, hist, estrategia, filtros_on, f_pl, f_pvp, f_dy, f_div_e, busca_dir, mercado):
    if hist.empty or not info: return None
    
    # Médias para o gráfico
    hist['SMA20'] = hist['Close'].rolling(window=20).mean()
    
    # Métricas Base
    p_atual = float(hist["Close"].iloc[-1])
    pl = info.get("trailingPE", 0) or 0
    pvp = info.get("priceToBook", 0) or 0
    dy = (info.get("dividendYield", 0) or 0) * 100
    ebitda = info.get("ebitda", 1) or 1
    div_e = (info.get("totalDebt", 0) - info.get("totalCash", 0)) / ebitda

    lpa, vpa = info.get("trailingEps", 0) or 0, info.get("bookValue", 0) or 0
    p_justo = calcular_graham(lpa, vpa)
    upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 else 0.0

    if not busca_dir and filtros_on:
        if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_e): return None

    # Sentimento (Notícias)
    score_p = score_n = 0
    links = []
    try:
        url = f"https://news.google.com/rss/search?q={tkr}&hl={'pt-BR' if mercado == 'Brasil' else 'en-US'}"
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            txt = entry.title.lower()
            score_p += sum(txt.count(w) for w in ["alta", "lucro", "compra", "subiu", "profit", "buy"])
            score_n += sum(txt.count(w) for w in ["queda", "prejuízo", "venda", "caiu", "loss", "sell"])
            links.append({"titulo": entry.title, "link": entry.link})
    except: pass

    rsi_val = calcular_rsi(hist["Close"])
    score_v = calcular_score_value(info)
    sim = simular_performance_historica(hist)

    # Lógica de Estratégias (Finalizada)
    veredito, cor, motivo = "NEUTRO ⚖️", "warning", "Indicadores em equilíbrio."

    if estrategia == "Value Investing (Graham/Buffett)":
        if upside > 20 and score_v >= 3: veredito, cor, motivo = "VALOR ✅", "success", f"Margem de segurança de {upside:.1f}%."
        elif upside < 0: veredito, cor, motivo = "CARO 🚨", "error", "Acima do preço justo."
    
    elif estrategia == "Dividend Investing":
        if dy >= 6.0 and div_e < 3.0: veredito, cor, motivo = "RENDA ✅", "success", f"Yield atrativo ({dy:.2f}%)."
        elif dy < 3.0: veredito, cor, motivo = "BAIXO DY 🚨", "error", "Yield insuficiente."

    elif estrategia == "Growth Investing":
        rev_g = (info.get("revenueGrowth", 0) or 0) * 100
        if rev_g > 15.0: veredito, cor, motivo = "CRESCIMENTO ✅", "success", f"Crescimento de {rev_g:.1f}%."
        elif rev_g < 0: veredito, cor, motivo = "DECLÍNIO 🚨", "error", "Receita em queda."

    elif estrategia == "Buy and Hold":
        if score_v >= 3 and div_e < 2.5: veredito, cor, motivo = "ACUMULAR ✅", "success", "Fundamentos sólidos de LP."
        elif div_e > 5.0: veredito, cor, motivo = "RISCO 🚨", "error", "Endividamento perigoso."

    elif estrategia == "Position Trading":
        sma200 = hist["Close"].rolling(window=200).mean().iloc[-1]
        if p_atual > sma200: veredito, cor, motivo = "TENDÊNCIA ALTA ✅", "success", "Acima da média de 200d."
        else: veredito, cor, motivo = "TENDÊNCIA BAIXA 🚨", "error", "Abaixo da tendência principal."

    else: # Análise Técnica
        if rsi_val > 70: veredito, cor, motivo = "SOBRECOMPRA 🚨", "error", f"RSI esticado ({rsi_val:.1f})."
        elif rsi_val < 30: veredito, cor, motivo = "SOBREVENDA ✅", "success", f"RSI em exaustão ({rsi_val:.1f})."

    return {
        "Ticker": tkr, "Empresa": info.get("shortName", tkr), "Preço": p_atual, "P/L": pl, 
        "DY %": dy, "Dívida": div_e, "Graham": p_justo, "Upside %": upside, "Veredito": veredito, 
        "Cor": cor, "Motivo": motivo, "RSI": rsi_val, "Hist": hist, "Links": links,
        "TaxaCompra": sim["taxa_compra"], "TaxaVenda": sim["taxa_venda"]
    }

# ===================== INTERFACE (SIDEBAR) =====================
st.sidebar.title("🌎 Monitor IA Pro")
indices_data = obter_indices()
for n, (v, var) in indices_data.items(): st.sidebar.metric(n, f"{v:,.0f}", f"{var:.2f}%")
st.sidebar.divider()
cambio = obter_cambio()
c1, c2 = st.sidebar.columns(2)
c1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
c2.metric("BTC", f"R$ {cambio['Bitcoin'][0]:,.0f}", f"{cambio['Bitcoin'][1]:.2f}%")
st.sidebar.divider()

mercado = st.sidebar.radio("Mercado:", ["Brasil", "EUA"], on_change=ativar_filtros)
estrategia = st.sidebar.selectbox("Estratégia:", ["Value Investing (Graham/Buffett)", "Análise Técnica (Trader)", "Growth Investing", "Buy and Hold", "Dividend Investing", "Position Trading"])
busca = st.sidebar.text_input("🔍 Busca Ticker:").upper().strip()

with st.sidebar.expander("📊 Filtros de Valuation", expanded=False):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0)
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0)
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0)
    f_div_e = st.slider("Dív.Líq/EBITDA Máx", 0.0, 15.0, 15.0)

# ===================== RENDERIZAÇÃO PRINCIPAL =====================
st.title(f"🤖 Monitor IA - {mercado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

lista = ([busca] if busca else (["PETR4", "VALE3", "ITUB4", "BBAS3", "WEGE3", "PRIO3"] if mercado == "Brasil" else ["AAPL", "MSFT", "NVDA", "TSLA"]))
dados_finais = []

with st.spinner("📡 Sincronizando dados..."):
    infos, hists = obter_dados_batch(lista, mercado)
    for tkr in lista:
        res = processar_ativo(tkr, infos.get(tkr), hists.get(tkr), estrategia, st.session_state.filtros_ativos, f_pl, f_pvp, f_dy, f_div_e, busca, mercado)
        if res: dados_finais.append(res)

if dados_finais:
    df_res = pd.DataFrame(dados_finais)[["Ticker", "Preço", "DY %", "Upside %", "Veredito"]]
    st.dataframe(df_res.sort_values(by="Upside %", ascending=False), use_container_width=True, hide_index=True)

    for acao in dados_finais:
        st.divider()
        col_t, col_v, col_ac, col_av = st.columns([3, 1, 1, 1])
        col_t.header(f"{acao['Empresa']} ({acao['Ticker']})")
        col_ac.metric("Assert. Compra", f"{acao['TaxaCompra']:.1f}%")
        col_av.metric("Assert. Venda", f"{acao['TaxaVenda']:.1f}%")
        
        if acao["Cor"] == "success": col_v.success(acao["Veredito"])
        elif acao["Cor"] == "error": col_v.error(acao["Veredito"])
        else: col_v.warning(acao["Veredito"])

        # Gráfico Plotly
        h = acao["Hist"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.7])
        fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='Preço'), row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=h['SMA20'], line=dict(color='yellow', width=1), name='Média 20'), row=1, col=1)
        fig.add_trace(go.Bar(x=h.index, y=h['Volume'], name='Volume', marker_color='gray'), row=2, col=1)
        fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Preço", f"{acao['Preço']:.2f}")
        c2.metric("P/L", f"{acao['P/L']:.1f}")
        c3.metric("Dívida", f"{acao['Dívida']:.1f}")
        c4.metric("Graham", f"{acao['Graham']:.2f}", f"{acao['Upside %']:.1f}%")
        
        with st.expander("🔍 IA Insights & Notícias"):
            st.info(f"**Motivo:** {acao['Motivo']}")
            for l in acao["Links"]: st.markdown(f"• [{l['titulo']}]({l['link']})")
else:
    st.info("Nenhum ativo encontrado com os filtros atuais.")
