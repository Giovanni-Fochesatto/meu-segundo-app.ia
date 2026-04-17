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

# ===================== SIMULAÇÃO VETORIZADA MELHORADA =====================
def simular_performance_historica(hist):
    """Simulação mais realista de assertividade com Expectancy"""
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

    # === Compras ===
    if buy_mask.any():
        ret_buy = retorno_15d[buy_mask]
        qtd_compra = int(buy_mask.sum())
        taxa_compra = (ret_buy > 0).mean() * 100
        retorno_medio_compra = ret_buy.mean() * 100

        avg_win = ret_buy[ret_buy > 0].mean() if (ret_buy > 0).any() else 0
        avg_loss = abs(ret_buy[ret_buy < 0].mean()) if (ret_buy < 0).any() else 0
        expectancy_compra = (taxa_compra/100 * avg_win) - ((1 - taxa_compra/100) * avg_loss) * 100
    else:
        taxa_compra = retorno_medio_compra = expectancy_compra = 0.0
        qtd_compra = 0

    # === Vendas ===
    if sell_mask.any():
        ret_sell = retorno_15d[sell_mask]
        qtd_venda = int(sell_mask.sum())
        taxa_venda = (ret_sell < 0).mean() * 100
        retorno_medio_venda = ret_sell.mean() * 100

        avg_win_sell = abs(ret_sell[ret_sell < 0].mean()) if (ret_sell < 0).any() else 0
        avg_loss_sell = ret_sell[ret_sell > 0].mean() if (ret_sell > 0).any() else 0
        expectancy_venda = (taxa_venda/100 * avg_win_sell) - ((1 - taxa_venda/100) * avg_loss_sell) * 100
    else:
        taxa_venda = retorno_medio_venda = expectancy_venda = 0.0
        qtd_venda = 0

    return {
        "taxa_compra": taxa_compra,
        "taxa_venda": taxa_venda,
        "retorno_medio_compra": retorno_medio_compra,
        "retorno_medio_venda": retorno_medio_venda,
        "expectancy_compra": expectancy_compra,
        "expectancy_venda": expectancy_venda,
        "qtd_compra": qtd_compra,
        "qtd_venda": qtd_venda
    }


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

    # Bitcoin em Real
    btc_real = 0.0
    try:
        t = yf.Ticker("BTC-BRL")
        data = t.history(period="2d")
        if not data.empty and len(data) >= 2:
            atual = float(data["Close"].iloc[-1])
        else:
            atual = float(t.fast_info.last_price)
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


# ===================== DOWNLOAD EM BATCH =====================
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
    p_atual = float(hist["Close"].iloc[-1])
    upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 else 0.0

    if not busca_direta and filtros_ativos:
        if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
            return None

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
    score_value = calcular_score_value(info)

    # Nova simulação
    sim = simular_performance_historica(hist)

    # Lógica de veredito
    if estrategia_ativa == "Value Investing (Graham/Buffett)":
        if upside > 20 and score_value >= 3:
            veredito, cor = "VALOR ✅", "success"
            motivo_detalhe = f"Forte margem de segurança ({upside:.1f}%) e bons fundamentos (Score: {score_value}/4)."
        elif upside < 0:
            veredito, cor = "CARO 🚨", "error"
            motivo_detalhe = "Preço acima do valor intrínseco de Graham."
        else:
            veredito, cor = "NEUTRO ⚖️", "warning"
            motivo_detalhe = "Ativo próximo ao preço justo."
    else:
        if rsi_val > 70 and score_n > score_p:
            veredito, cor = "VENDA 🚨", "error"
            motivo_detalhe = f"RSI alto ({rsi_val:.1f}) e notícias negativas."
        elif rsi_val > 75:
            veredito, cor = "VENDA 🚨", "error"
            motivo_detalhe = f"RSI em nível extremo ({rsi_val:.1f})."
        elif score_p > score_n and rsi_val < 65:
            veredito, cor = "COMPRA ✅", "success"
            motivo_detalhe = f"Notícias positivas e RSI saudável ({rsi_val:.1f})."
        elif score_n > score_p or rsi_val > 70:
            veredito, cor = "CAUTELA ⚠️", "error"
            lista_motivos = []
            if rsi_val > 70: lista_motivos.append(f"RSI alto ({rsi_val:.1f})")
            if score_n > score_p: lista_motivos.append("Sentimento negativo")
            motivo_detalhe = " | ".join(lista_motivos)
        else:
            veredito, cor = "NEUTRO ⚖️", "warning"
            motivo_detalhe = "Indicadores em equilíbrio."

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

        # Métricas melhoradas
        "TaxaCompra": sim["taxa_compra"],
        "TaxaVenda": sim["taxa_venda"],
        "RetornoMedioCompra": sim["retorno_medio_compra"],
        "RetornoMedioVenda": sim["retorno_medio_venda"],
        "ExpectancyCompra": sim["expectancy_compra"],
        "ExpectancyVenda": sim["expectancy_venda"],
        "QtdCompra": sim["qtd_compra"],
        "QtdVenda": sim["qtd_venda"]
    }


# ===================== SIDEBAR =====================
st.sidebar.title("🌎 Mercado e Estratégia")

st.sidebar.subheader("📈 Índices Mundiais")
indices_data = obter_indices()
for nome, (valor, var) in indices_data.items():
    st.sidebar.metric(nome, f"{valor:,.0f} pts", f"{var:.2f}%")

st.sidebar.divider()

st.sidebar.subheader("💱 Câmbio em Tempo Real")
cambio = obter_cambio()

col1, col2 = st.sidebar.columns(2)
col1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
col2.metric("Euro",  f"R$ {cambio['Euro'][0]:.2f}",  f"{cambio['Euro'][1]:.2f}%")

col3, col4 = st.sidebar.columns(2)
col3.metric("Libra", f"R$ {cambio['Libra'][0]:.2f}", f"{cambio['Libra'][1]:.2f}%")
col4.metric("Bitcoin", f"R$ {cambio['Bitcoin'][0]:,.0f}", f"{cambio['Bitcoin'][1]:.2f}%")

st.sidebar.divider()

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

# ===================== LISTA DE ATIVOS =====================
if mercado_selecionado == "Brasil":
    lista_base = ["PETR4", "VALE3", "ITUB4", "BBAS3", "BBDC4", "SANB11", "B3SA3", "EGIE3", "TRPL4", "TAEE11", "SAPR11", "CPLE6", "ELET3", "CMIG4", "SBSP3", "ABEV3", "WEGE3", "RADL3", "RENT3", "MGLU3", "LREN3", "RAIZ4", "VBBR3", "SUZB3", "KLBN11", "GOAU4", "CSNA3", "PRIO3", "JBSS3", "BRFS3", "GGBR4", "HAPV3", "RDOR3"]
    moeda_simbolo = "R$"
else:
    lista_base = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "DIS", "KO", "PEP", "MCD", "NKE", "WMT", "JPM", "V", "MA", "BAC", "PYPL", "PFE", "JNJ", "PG", "COST", "ORCL"]
    moeda_simbolo = "US$"

# ===================== CABEÇALHO =====================
st.title(f"🤖 Monitor IA - {mercado_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# ===================== PROCESSAMENTO PRINCIPAL =====================
tickers_para_processar = [busca_direta] if busca_direta else lista_base
dados_vencedoras = []

if tickers_para_processar:
    with st.spinner("📡 Baixando dados em batch..."):
        infos, hists = obter_dados_batch(tickers_para_processar, mercado_selecionado)

    for tkr in tickers_para_processar:
        info = infos.get(tkr, {})
        hist = hists.get(tkr, pd.DataFrame())
        resultado = processar_ativo(
            tkr, info, hist, estrategia_ativa, st.session_state.filtros_ativos,
            f_pl, f_pvp, f_dy, f_div_ebitda, busca_direta, mercado_selecionado
        )
        if resultado:
            dados_vencedoras.append(resultado)

# ===================== INTERFACE =====================
if dados_vencedoras:
    st.subheader(f"🏆 Ranking de Oportunidades - Estratégia: {estrategia_ativa}")
    
    with st.expander("📌 Legenda de Sinais e Vereditos"):
        st.markdown("""
        * **VALOR ✅**: Grande desconto + bons fundamentos  
        * **COMPRA ✅**: RSI saudável + notícias positivas  
        * **VENDA / CARO 🚨**: RSI extremo ou preço acima do justo  
        * **CAUTELA ⚠️**: Divergência entre preço, RSI e sentimento  
        * **Expectancy**: Ganho esperado por operação (quanto maior, melhor)
        """)

    df_resumo = pd.DataFrame(dados_vencedoras)[
        ["Ticker", "Preço", "DY %", "Upside %", "Veredito", "Motivo", 
         "TaxaCompra", "TaxaVenda", "ExpectancyCompra", "QtdCompra"]
    ]

    st.dataframe(
        df_resumo.sort_values(by="Upside %", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Veredito": st.column_config.TextColumn("Veredito"),
            "Motivo": st.column_config.TextColumn("Motivo da IA", width="medium"),
            "TaxaCompra": st.column_config.NumberColumn("Win Rate Compra", format="%.1f%%"),
            "TaxaVenda": st.column_config.NumberColumn("Win Rate Venda", format="%.1f%%"),
            "ExpectancyCompra": st.column_config.NumberColumn("Expectancy Compra", format="%.2f%%"),
            "QtdCompra": st.column_config.NumberColumn("Sinais"),
        },
    )

    for acao in dados_vencedoras:
        st.divider()
        col_tit, col_ver, col_c, col_v = st.columns([3, 1, 1, 1])
        col_tit.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
        
        col_c.metric("Assertividade Compra", f"{acao['TaxaCompra']:.1f}%", f"{acao['QtdCompra']} sinais")
        col_v.metric("Assertividade Venda", f"{acao['TaxaVenda']:.1f}%", f"{acao['QtdVenda']} sinais")

        if acao["Cor"] == "success":
            col_ver.success(f"**{acao['Veredito']}**")
        elif acao["Cor"] == "error":
            col_ver.error(f"**{acao['Veredito']}**")
        else:
            col_ver.warning(f"**{acao['Veredito']}**")

        st.line_chart(acao["Hist"]["Close"], use_container_width=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Preço Atual", f"{moeda_simbolo} {acao['Preço']:.2f}")
        c2.metric("P/L", round(acao["P/L"], 2))
        c3.metric("DY", f"{acao['DY %']:.2f}%")
        c4.metric("Dív.Líq/EBITDA", round(acao["Dívida"], 2))
        c5.metric("Graham", f"{moeda_simbolo} {acao['Graham']:.2f}", f"{acao['Upside %']:.1f}%")

        with st.expander(f"📊 Detalhes Fundamentalistas e Técnicos: {acao['Ticker']}"):
            col_inf1, col_inf2 = st.columns(2)
            with col_inf1:
                st.write(f"**Fundamentos (Value Score):** {acao['ValueScore']}/4")
                st.progress(acao["ValueScore"] / 4)
                st.write(f"📈 RSI: {acao['RSI']:.2f}")
                st.write(f"**Expectancy Compra:** {acao['ExpectancyCompra']:.2f}%")
            with col_inf2:
                st.write(f"📝 **Motivo IA:** {acao['Motivo']}")
            st.markdown("---")
            st.markdown("**Últimas Manchetes:**")
            for n in acao["Links"]:
                st.markdown(f"• [{n['titulo']}]({n['link']})")

else:
    st.info("💡 Use os filtros ou faça uma busca direta para começar.")
