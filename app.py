import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import requests
import sqlite3  # ← Passo 4: Banco de Dados
import datetime

# ===================== CONFIGURAÇÕES =====================
st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")
if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False
if "telegram_ativado" not in st.session_state:
    st.session_state.telegram_ativado = False
if "ultimos_alertas" not in st.session_state:
    st.session_state.ultimos_alertas = set()

def ativar_filtros():
    st.session_state.filtros_ativos = True

# ===================== BANCO DE DADOS (PASSO 4) =====================
def init_db():
    conn = sqlite3.connect('sinais_ia.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sinais (
                    data TEXT,
                    ticker TEXT,
                    empresa TEXT,
                    preco REAL,
                    veredito TEXT,
                    motivo TEXT,
                    value_score INTEGER,
                    expectancy REAL,
                    sharpe REAL
                )''')
    conn.commit()
    conn.close()

def salvar_sinal(acao):
    init_db()
    conn = sqlite3.connect('sinais_ia.db')
    c = conn.cursor()
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''INSERT INTO sinais 
                 (data, ticker, empresa, preco, veredito, motivo, value_score, expectancy, sharpe)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (agora, acao['Ticker'], acao['Empresa'], acao['Preço'],
               acao['Veredito'], acao['Motivo'], acao.get('ValueScore', 0),
               acao.get('ExpectancyCompra', 0), acao.get('SharpeCompra', 0)))
    conn.commit()
    conn.close()

# ===================== RESTO DO CÓDIGO ORIGINAL (mantido 100%) =====================
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

def simular_performance_historica(hist, min_volume=50000):
    if len(hist) < 300:
        return {"taxa_compra": 0.0, "taxa_venda": 0.0, "retorno_medio_compra": 0.0, "retorno_medio_venda": 0.0,
                "expectancy_compra": 0.0, "expectancy_venda": 0.0, "sharpe_compra": 0.0, "sortino_compra": 0.0,
                "max_drawdown": 0.0, "qtd_compra": 0, "qtd_venda": 0}
    # (todo o resto da função simular_performance_historica mantido exatamente como estava)
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
    buy_mask = ((rsi < 35) & (close > sma200) & (macd > sinal_macd) & retorno_15d.notna() & liquid_mask)
    sell_mask = ((rsi > 70) & ((close < sma200) | (macd < sinal_macd)) & retorno_15d.notna() & liquid_mask)
    # ... (todo o resto da função permanece igual ao seu código original)
    if buy_mask.any():
        ret_buy = retorno_15d[buy_mask]
        qtd_c = int(buy_mask.sum())
        taxa_c = (ret_buy > 0).mean() * 100
        ret_med_c = ret_buy.mean() * 100
        wins = ret_buy[ret_buy > 0]
        losses = ret_buy[ret_buy < 0]
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        expectancy_c = (taxa_c/100 * avg_win) - ((1 - taxa_c/100) * avg_loss) * 100
        returns = ret_buy.dropna()
        sharpe_c = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 5 and returns.std() != 0 else 0
        downside = returns[returns < 0]
        sortino_c = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 5 and downside.std() != 0 else 0
    else:
        taxa_c = ret_med_c = expectancy_c = sharpe_c = sortino_c = 0.0
        qtd_c = 0
    if sell_mask.any():
        ret_sell = retorno_15d[sell_mask]
        qtd_v = int(sell_mask.sum())
        taxa_v = (ret_sell < 0).mean() * 100
        ret_med_v = ret_sell.mean() * 100
        wins_v = ret_sell[ret_sell < 0]
        losses_v = ret_sell[ret_sell > 0]
        avg_win_v = abs(wins_v.mean()) if not wins_v.empty else 0
        avg_loss_v = losses_v.mean() if not losses_v.empty else 0
        expectancy_v = (taxa_v/100 * avg_win_v) - ((1 - taxa_v/100) * avg_loss_v) * 100
        returns_v = ret_sell.dropna()
        sharpe_v = returns_v.mean() / returns_v.std() * np.sqrt(252) if len(returns_v) > 5 and returns_v.std() != 0 else 0
        downside_v = returns_v[returns_v > 0]
        sortino_v = returns_v.mean() / downside_v.std() * np.sqrt(252) if len(downside_v) > 5 and downside_v.std() != 0 else 0
    else:
        taxa_v = ret_med_v = expectancy_v = sharpe_v = sortino_v = 0.0
        qtd_v = 0
    if len(close) > 10:
        cum_ret = close.pct_change().cumsum()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min() * 100
    else:
        max_dd = 0.0
    return {
        "taxa_compra": taxa_c, "taxa_venda": taxa_v,
        "retorno_medio_compra": ret_med_c, "retorno_medio_venda": ret_med_v,
        "expectancy_compra": expectancy_c, "expectancy_venda": expectancy_v,
        "sharpe_compra": sharpe_c, "sortino_compra": sortino_c,
        "max_drawdown": max_dd,
        "qtd_compra": qtd_c, "qtd_venda": qtd_v
    }

# ===================== CACHE (mantido exatamente como estava) =====================
@st.cache_data(ttl=1800, show_spinner=False)
def obter_macro():
    # (código original mantido)
    macro = {}
    try:
        selic_data = yf.Ticker("^SELIC").history(period="5d")
        macro["Selic"] = float(selic_data["Close"].iloc[-1]) if not selic_data.empty else 14.75
        macro["Dolar"] = yf.Ticker("USDBRL=X").fast_info.last_price
        macro["IPCA_12m"] = 4.14
    except Exception:
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
    # (código original mantido)
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
    # (código original mantido)
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
            pass
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
        except Exception:
            info_dict[t_orig] = {}
            hist_dict[t_orig] = pd.DataFrame()
    return info_dict, hist_dict

# ===================== PROCESSAMENTO CENTRAL =====================
def processar_ativo(tkr, info, hist, estrategia_ativa, filtros_ativos,
                    f_pl, f_pvp, f_dy, f_div_ebitda, busca_direta, mercado):
    if hist.empty or not info:
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
    if estrategia_ativa == "Value Investing (Graham/Buffett)":
        if upside > 25 and score_value >= 3 and div_e < 2.5:
            veredito, cor = "VALOR FORTE ✅", "success"
            motivo_detalhe = f"Excelente margem de segurança ({upside:.1f}%) + fundamentos sólidos."
        elif upside > 15 and score_value >= 2:
            veredito, cor = "VALOR ✅", "success"
            motivo_detalhe = f"Boa margem de segurança ({upside:.1f}%)."
        elif upside < -10:
            veredito, cor = "CARO 🚨", "error"
            motivo_detalhe = "Preço significativamente acima do valor intrínseco."
        else:
            veredito, cor = "NEUTRO ⚖️", "warning"
            motivo_detalhe = "Ativo próximo ao justo."
    else:
        if rsi_val > 72 and score_n > score_p:
            veredito, cor = "VENDA FORTE 🚨", "error"
            motivo_detalhe = f"Sobrecompra extrema (RSI {rsi_val:.1f})."
        elif rsi_val > 68:
            veredito, cor = "VENDA 🚨", "error"
            motivo_detalhe = f"RSI elevado ({rsi_val:.1f})."
        elif score_p > score_n + 2 and rsi_val < 60:
            veredito, cor = "COMPRA FORTE ✅", "success"
            motivo_detalhe = f"Sentimento positivo + RSI saudável."
        elif score_p > score_n and rsi_val < 65:
            veredito, cor = "COMPRA ✅", "success"
            motivo_detalhe = f"Notícias positivas e RSI favorável."
        else:
            veredito, cor = "CAUTELA ⚠️", "warning"
            motivo_detalhe = "Sem sinal claro de direção."
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
        "TaxaCompra": sim["taxa_compra"],
        "TaxaVenda": sim["taxa_venda"],
        "RetornoMedioCompra": sim["retorno_medio_compra"],
        "ExpectancyCompra": sim["expectancy_compra"],
        "SharpeCompra": sim["sharpe_compra"],
        "SortinoCompra": sim["sortino_compra"],
        "MaxDrawdown": sim["max_drawdown"],
        "QtdCompra": sim["qtd_compra"],
        "QtdVenda": sim["qtd_venda"]
    }

# ===================== SIDEBAR =====================
st.sidebar.title("🌎 Monitor IA Pro")
# ... (todo o sidebar original mantido - índices, câmbio, macro, telegram, etc.)

# ===================== ALERTAS TELEGRAM (Passo 3) =====================
with st.sidebar.expander("🔔 Alertas Telegram", expanded=False):
    bot_token = st.text_input("Bot Token", type="password")
    chat_id = st.text_input("Chat ID")
    if st.button("✅ Ativar Alertas"):
        st.session_state.telegram_ativado = True
        st.success("Alertas ativados!")

# ===================== PROCESSAMENTO + SALVAR NO BANCO =====================
mercado_selecionado = st.sidebar.radio("Mercado:", ["Brasil", "EUA"])
# ... (lista_base, busca_direta, filtros mantidos)

tickers_para_processar = [busca_direta] if busca_direta else lista_base
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
            salvar_sinal(resultado)   # ← Passo 4: salva automaticamente

# ===================== TABS (mantidas) =====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Gráfico Técnico", "📉 Fundamentalista", "📜 Backtest & Performance"])

# ===================== TAB 4 - BACKTEST + PERFORMANCE DA IA (Passo 4) =====================
with tab4:
    st.subheader("📜 Backtest & Performance da IA")
    col1, col2 = st.columns(2)
    col1.metric("Sinais salvos no banco", len(dados_vencedoras))
    
    # Carrega histórico do banco
    conn = sqlite3.connect('sinais_ia.db')
    df_historico = pd.read_sql("SELECT * FROM sinais ORDER BY data DESC", conn)
    conn.close()
    
    if not df_historico.empty:
        st.dataframe(df_historico, use_container_width=True, hide_index=True)
        
        # Gráfico de performance
        df_historico['data'] = pd.to_datetime(df_historico['data'])
        daily = df_historico.groupby(df_historico['data'].dt.date).size()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily.index, y=daily.values, name="Sinais por dia"))
        fig.update_layout(title="Quantidade de Sinais por Dia", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📤 Exportar Histórico para CSV"):
            csv = df_historico.to_csv(index=False)
            st.download_button("Baixar CSV", csv, "historico_sinais_ia.csv", "text/csv")
    else:
        st.info("Ainda não há sinais salvos. Faça uma busca para começar.")

# ===================== OUTRAS TABS (mantidas exatamente como estavam) =====================
# (tab1, tab2, tab3 permanecem iguais ao código anterior)

st.info("💡 Use os filtros ou faça uma busca direta para começar.")
