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
    """Retorna RSI completo (usado na simulação vetorizada)"""
    if len(close) < window:
        return pd.Series([50.0] * len(close), index=close.index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_rsi(data, window: int = 14):
    """RSI atual (compatível com código antigo)"""
    if len(data) &lt; window:
        return 50.0
    rsi_series = calcular_rsi_series(data, window)
    return float(rsi_series.iloc[-1])

def calcular_score_value(info):
    score = 0
    if 0 &lt; info.get("trailingPE", 99) &lt; 15:
        score += 1
    if 0 &lt; info.get("priceToBook", 99) &lt; 1.5:
        score += 1
    if (info.get("dividendYield", 0) or 0) * 100 &gt; 5:
        score += 1
    if (info.get("operatingMargins", 0) or 0) &gt; 0.1:
        score += 1
    return score

# ===================== SIMULAÇÃO VETORIZADA (5 ANOS) =====================
def simular_performance_historica(hist):
    """Versão vetorizada - muito mais rápida que o loop original"""
    if len(hist) &lt; 300:
        return 0.0, 0.0, 0.0, 0, 0

    close = hist["Close"].copy()
    rsi = calcular_rsi_series(close)
    sma200 = close.rolling(window=200).mean()
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, adjust=False).mean()

    retorno_15d = close.shift(-15) / close - 1

    # Sinais com filtro de dados válidos (15 dias à frente)
    buy_mask = (
        (rsi &lt; 35)
        &amp; (close &gt; sma200)
        &amp; (macd &gt; sinal_macd)
        &amp; retorno_15d.notna()
    )
    sell_mask = (
        (rsi &gt; 70)
        &amp; ((close &lt; sma200) | (macd &lt; sinal_macd))
        &amp; retorno_15d.notna()
    )

    # Métricas de compra
    if buy_mask.any():
        ret_buy = retorno_15d[buy_mask]
        taxa_compra = (ret_buy &gt; 0).mean() * 100
        retorno_medio = ret_buy.mean() * 100
        total_c = int(buy_mask.sum())
    else:
        taxa_compra = 0.0
        retorno_medio = 0.0
        total_c = 0

    # Métricas de venda
    taxa_venda = (retorno_15d[sell_mask] &lt; 0).mean() * 100 if sell_mask.any() else 0.0
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
            if not data.empty and len(data) &gt;= 2:
                atual = data["Close"].iloc[-1]
                anterior = data["Close"].iloc[-2]
                variacao = ((atual / anterior) - 1) * 100
                resultados[nome] = (atual, variacao)
            else:
                resultados[nome] = (0.0, 0.0)
        except:
            resultados[nome] = (0.0, 0.0)
    return resultados

@st.cache_data(ttl=300, show_spinner=False)
def obter_cambio():
    moedas = {"Dólar": "USDBRL=X", "Euro": "EURBRL=X", "Bitcoin": "BTC-BRL"}
    resultados = {}
    for nome, ticker in moedas.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if not data.empty and len(data) &gt;= 2:
                atual = data["Close"].iloc[-1]
                anterior = data["Close"].iloc[-2]
                variacao = ((atual / anterior) - 1) * 100
                resultados[nome] = (atual, variacao)
            else:
                resultados[nome] = (yf.Ticker(ticker).fast_info.last_price, 0.0)
        except:
            resultados[nome] = (0.0, 0.0)
    return resultados

# ===================== DOWNLOAD EM BATCH (O MAIOR GANHO) =====================
@st.cache_data(ttl=600, show_spinner=False)
def obter_dados_batch(tickers, mercado):
    """Baixa histórico de TODOS os tickers de uma vez + info individual"""
    if not tickers:
        return {}, {}

    # Normaliza tickers brasileiros
    tickers_yf = [
        t + ".SA" if mercado == "Brasil" and not t.endswith(".SA") else t
        for t in tickers
    ]

    # Download em massa
    hist_multi = yf.download(
        tickers_yf,
        period="5y",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    info_dict = {}
    hist_dict = {}

    for i, t_orig in enumerate(tickers):
        t_yf = tickers_yf[i]
        try:
            info_dict[t_orig] = yf.Ticker(t_yf).info
            # Extrai histórico correto (multi ou single ticker)
            if len(tickers) == 1:
                hist_dict[t_orig] = hist_multi
            else:
                hist_dict[t_orig] = hist_multi[t_yf] if t_yf in hist_multi.columns.get_level_values(0) else pd.DataFrame()
        except Exception:
            info_dict[t_orig] = {}
            hist_dict[t_orig] = pd.DataFrame()

    return info_dict, hist_dict

# ===================== PROCESSAMENTO CENTRAL (LIMPO) =====================
def processar_ativo(tkr, info, hist, estrategia_ativa, filtros_ativos,
                    f_pl, f_pvp, f_dy, f_div_ebitda, busca_direta, mercado):
    if hist.empty or not info:
        return None

    # === Métricas fundamentais ===
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
    upside = ((p_justo / p_atual) - 1) * 100 if p_justo &gt; 0 else 0.0

    # === Filtro de valuation ===
    if not busca_direta and filtros_ativos:
        if not (pl &lt;= f_pl and pvp &lt;= f_pvp and dy &gt;= f_dy and div_e &lt;= f_div_ebitda):
            return None

    # === Notícias Google ===
    noticias_texto = ""
    lista_links = []
    try:
        lang = "pt-BR" if mercado == "Brasil" else "en-US"
        url = f"https://news.google.com/rss/search?q={tkr}&amp;hl={lang}"
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

    # === Assertividade 5 anos (vetorizada) ===
    taxa_compra, taxa_venda, retorno_medio, total_c, total_v = simular_performance_historica(hist)

    # === Lógica de veredito (igual à original) ===
    motivo_detalhe = ""
    if estrategia_ativa == "Value Investing (Graham/Buffett)":
        if upside &gt; 20 and score_value &gt;= 3:
            veredito, cor = "VALOR ✅", "success"
            motivo_detalhe = f"Forte margem de segurança ({upside:.1f}%) e bons fundamentos (Score: {score_value}/4)."
        elif upside &lt; 0:
            veredito, cor = "CARO 🚨", "error"
            motivo_detalhe = "Preço acima do valor intrínseco de Graham."
        else:
            veredito, cor = "NEUTRO ⚖️", "warning"
            motivo_detalhe = "Ativo próximo ao preço justo, sem margem de segurança clara."
    else:
        if rsi_val &gt; 70 and score_n &gt; score_p:
            veredito, cor = "VENDA 🚨", "error"
            motivo_detalhe = f"RSI alto ({rsi_val:.1f}) e notícias negativas. Possível topo."
        elif rsi_val &gt; 75:
            veredito, cor = "VENDA 🚨", "error"
            motivo_detalhe = f"RSI em nível extremo ({rsi_val:.1f}). Ativo sobrecomprado."
        elif score_p &gt; score_n and rsi_val &lt; 65:
            veredito, cor = "COMPRA ✅", "success"
            motivo_detalhe = f"Notícias positivas e RSI saudável ({rsi_val:.1f})."
        elif score_n &gt; score_p or rsi_val &gt; 70:
            veredito, cor = "CAUTELA ⚠️", "error"
            lista_motivos = []
            if rsi_val &gt; 70:
                lista_motivos.append(f"RSI alto ({rsi_val:.1f})")
            if score_n &gt; score_p:
                lista_motivos.append("Sentimento negativo")
            motivo_detalhe = " | ".join(lista_motivos)
        else:
            veredito, cor = "NEUTRO ⚖️", "warning"
            motivo_detalhe = "Indicadores técnicos e notícias em equilíbrio."

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
        "TaxaCompra": taxa_compra,
        "TaxaVenda": taxa_venda,
        "RetornoMedio": retorno_medio,
        "QtdCompra": total_c,
        "QtdVenda": total_v,
        "ValueScore": score_value,
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
col_c1, col_c2 = st.sidebar.columns(2)
col_c1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
col_c2.metric("Euro", f"R$ {cambio['Euro'][0]:.2f}", f"{cambio['Euro'][1]:.2f}%")
st.sidebar.metric("Bitcoin", f"R$ {cambio['Bitcoin'][0]:.0f}", f"{cambio['Bitcoin'][1]:.2f}%")
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

# Lista base (mantida igual)
if mercado_selecionado == "Brasil":
    lista_base = [
        "PETR4", "VALE3", "ITUB4", "BBAS3", "BBDC4", "SANB11", "B3SA3",
        "EGIE3", "TRPL4", "TAEE11", "SAPR11", "CPLE6", "ELET3", "CMIG4", "SBSP3",
        "ABEV3", "WEGE3", "RADL3", "RENT3", "MGLU3", "LREN3", "RAIZ4", "VBBR3",
        "SUZB3", "KLBN11", "GOAU4", "CSNA3", "PRIO3", "JBSS3", "BRFS3", "GGBR4",
        "HAPV3", "RDOR3",
    ]
    moeda_simbolo = "R$"
else:
    lista_base = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
        "NFLX", "DIS", "KO", "PEP", "MCD", "NKE", "WMT",
        "JPM", "V", "MA", "BAC", "PYPL",
        "PFE", "JNJ", "PG", "COST", "ORCL",
    ]
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
            tkr=tkr,
            info=info,
            hist=hist,
            estrategia_ativa=estrategia_ativa,
            filtros_ativos=st.session_state.filtros_ativos,
            f_pl=f_pl,
            f_pvp=f_pvp,
            f_dy=f_dy,
            f_div_ebitda=f_div_ebitda,
            busca_direta=busca_direta,
            mercado=mercado_selecionado,
        )
        if resultado:
            dados_vencedoras.append(resultado)

# ===================== INTERFACE =====================
if dados_vencedoras:
    st.subheader(f"🏆 Ranking de Oportunidades - Estratégia: {estrategia_ativa}")

    with st.expander("📌 Legenda de Sinais e Vereditos"):
        st.markdown("""
        * **VALOR ✅**: (Value) Grande desconto + bons fundamentos  
        * **COMPRA ✅**: (Técnica) RSI saudável + notícias positivas  
        * **VENDA / CARO 🚨**: RSI extremo ou preço acima do justo  
        * **CAUTELA ⚠️**: Divergência entre preço, RSI e sentimento  
        * **Graham**: Valor intrínseco. Upside = potencial de valorização
        """)

    df_resumo = pd.DataFrame(dados_vencedoras)[
        ["Ticker", "Preço", "DY %", "Upside %", "Veredito", "Motivo", "TaxaCompra", "TaxaVenda"]
    ]

    st.dataframe(
        df_resumo.sort_values(by="Upside %", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Veredito": st.column_config.TextColumn("Veredito"),
            "Motivo": st.column_config.TextColumn("Motivo da IA", width="medium"),
            "TaxaCompra": st.column_config.NumberColumn("Assert. Compra (5y)", format="%.1f%%"),
            "TaxaVenda": st.column_config.NumberColumn("Assert. Venda (5y)", format="%.1f%%"),
        },
    )

    for acao in dados_vencedoras:
        st.divider()
        col_tit, col_ver, col_acc_c, col_acc_v = st.columns([3, 1, 1, 1])
        col_tit.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")

        col_acc_c.metric("Assertividade Compra", f"{acao['TaxaCompra']:.1f}%", f"{acao['QtdCompra']} sinais")
        col_acc_v.metric("Assertividade Venda", f"{acao['TaxaVenda']:.1f}%", f"{acao['QtdVenda']} sinais")

        if acao["Cor"] == "success":
            col_ver.success(f"**{acao['Veredito']}**")
        elif acao["Cor"] == "error":
            col_ver.error(f"**{acao['Veredito']}**")
            if acao["Motivo"]:
                st.info(f"👉 **Atenção:** {acao['Motivo']}")
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
            with col_inf2:
                st.write(f"📝 **Motivo IA:** {acao['Motivo']}")
            st.markdown("---")
            st.markdown("**Últimas Manchetes:**")
            for n in acao["Links"]:
                st.markdown(f"• [{n['titulo']}]({n['link']})")
else:
    st.info("💡 Use os filtros ou faça uma busca direta para começar.")
