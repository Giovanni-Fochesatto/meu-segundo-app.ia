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
    if 0 < info.get("trailingPE", 99) < 15: score += 1
    if 0 < info.get("priceToBook", 99) < 1.5: score += 1
    if (info.get("dividendYield", 0) or 0) * 100 > 5: score += 1
    if (info.get("operatingMargins", 0) or 0) > 0.1: score += 1
    return score

# ===================== SIMULAÇÃO (mantida igual) =====================
def simular_performance_historica(hist, min_volume=50000):
    if len(hist) < 300:
        return {
            "taxa_compra": 0.0, "taxa_venda": 0.0,
            "retorno_medio_compra": 0.0, "retorno_medio_venda": 0.0,
            "expectancy_compra": 0.0, "expectancy_venda": 0.0,
            "sharpe_compra": 0.0, "sortino_compra": 0.0,
            "max_drawdown": 0.0,
            "qtd_compra": 0, "qtd_venda": 0
        }
    # ... (todo o resto da função permanece igual ao seu código original)
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
    sell_mask = (
        (rsi > 70) & ((close < sma200) | (macd < sinal_macd)) &
        retorno_15d.notna() & liquid_mask
    )
    # Compras
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
    # Vendas (mantida igual)
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
        "taxa_compra": taxa_c,
        "taxa_venda": taxa_v,
        "retorno_medio_compra": ret_med_c,
        "retorno_medio_venda": ret_med_v,
        "expectancy_compra": expectancy_c,
        "expectancy_venda": expectancy_v,
        "sharpe_compra": sharpe_c,
        "sortino_compra": sortino_c,
        "max_drawdown": max_dd,
        "qtd_compra": qtd_c,
        "qtd_venda": qtd_v
    }

# ===================== CACHE (mantido) =====================
@st.cache_data(ttl=1800, show_spinner=False)
def obter_macro():
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

# (obter_indices, obter_cambio, obter_dados_batch permanecem iguais ao seu código)

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
    p_atual = float(hist["Close"].iloc[-1])
    upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 else 0.0

    if not busca_direta and filtros_ativos:
        if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
            return None

    # Notícias (mantido)
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
    sim = simular_performance_historica(hist)

    # Lógica de veredito (mantida)
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

# ===================== SIDEBAR (mantida) =====================
# ... (todo o sidebar permanece igual ao seu código)

# ===================== LISTA DE ATIVOS (melhorada) =====================
if mercado_selecionado == "Brasil":
    # Lista maior, mas controlada (as mais líquidas + Ibovespa)
    lista_base = ["PETR4","VALE3","ITUB4","BBAS3","BBDC4","SANB11","B3SA3","EGIE3","TRPL4","TAEE11",
                  "WEGE3","PRIO3","JBSS3","ABEV3","BBSE3","ELET3","EQTL3","GGBR4","HAPV3","IRBR3",
                  "KLBN11","LREN3","MGLU3","NTCO3","PETR3","RADL3","RENT3","SUZB3","VBBR3"]
    moeda_simbolo = "R$"
else:
    lista_base = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
    moeda_simbolo = "US$"

# ===================== CABEÇALHO =====================
st.title(f"🤖 Monitor IA - {mercado_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Gráfico Técnico", "📉 Fundamentalista", "📜 Backtest"])

# ===================== PROCESSAMENTO =====================
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

# ===================== TAB 1 - OVERVIEW =====================
with tab1:
    if dados_vencedoras:
        st.subheader(f"🏆 Ranking - Estratégia: {estrategia_ativa}")
        df_resumo = pd.DataFrame(dados_vencedoras)[
            ["Ticker", "Preço", "DY %", "Upside %", "Veredito", "Motivo",
             "TaxaCompra", "ExpectancyCompra", "SharpeCompra", "QtdCompra"]
        ]
        # Filtro opcional para reduzir zeros
        df_resumo = df_resumo[df_resumo["Preço"] > 0]
        st.dataframe(
            df_resumo.sort_values(by="ExpectancyCompra", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Veredito": st.column_config.TextColumn("Veredito"),
                "Motivo": st.column_config.TextColumn("Motivo da IA", width="medium"),
                "TaxaCompra": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                "ExpectancyCompra": st.column_config.NumberColumn("Expectancy", format="%.2f%%"),
                "SharpeCompra": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "QtdCompra": st.column_config.NumberColumn("Sinais"),
            },
        )
    else:
        st.info("Nenhum ativo encontrado com os filtros atuais.")

# ===================== TAB 3 - FUNDAMENTALISTA (Value Score destacado) =====================
with tab3:
    st.subheader("📉 Análise Fundamentalista")
    if dados_vencedoras:
        for acao in dados_vencedoras:
            st.write(f"**{acao['Empresa']} ({acao['Ticker']})**")
            col1, col2, col3 = st.columns(3)
            col1.metric("P/L", round(acao["P/L"], 2))
            col2.metric("P/VP", round(acao.get("P/VP", 0), 2))
            col3.metric("DY", f"{acao['DY %']:.2f}%")
            st.metric("Dívida Líquida / EBITDA", round(acao["Dívida"], 2))
            
            # Value Score bem visível
            score = acao.get("ValueScore", 0)
            st.markdown(f"**Value Score: {score}/4**")
            st.progress(score / 4)
            st.divider()
    else:
        st.info("Nenhum ativo encontrado.")

# ===================== TAB 4 - BACKTEST OTIMIZADA =====================
with tab4:
    st.subheader("📜 Backtest & Estatísticas")
    if dados_vencedoras:
        df = pd.DataFrame(dados_vencedoras)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ativos Analisados", len(df))
        col2.metric("Média Expectancy", f"{df['ExpectancyCompra'].mean():.2f}%")
        col3.metric("Média Sharpe", f"{df['SharpeCompra'].mean():.2f}")
        col4.metric("Total Sinais Compra", int(df['QtdCompra'].sum()))

        st.dataframe(
            df[["Ticker", "ExpectancyCompra", "SharpeCompra", "MaxDrawdown", "QtdCompra"]].round(2),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Execute uma análise para ver estatísticas de backtest.")

# ===================== FIM =====================
st.info("💡 Use os filtros ou faça uma busca direta para começar.")
