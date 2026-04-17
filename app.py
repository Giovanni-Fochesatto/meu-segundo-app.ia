 pip install -U streamlit yfinance pandas numpy feedparser streamlit-autorefresh
    streamlit run monitor_ia_pro.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import feedparser
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

# ===================== CONFIGURAÇÕES =====================
st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

# Yahoo Finance retorna dividendYield em fração (0.07 = 7%) em versões
# antigas e em percentual (7.0) em versões novas. Deixe True se seu
# yfinance ainda devolve fração.
DY_AS_FRACTION = True

# Quantos tickers coletar em paralelo (mais que isso e o Yahoo passa a
# bloquear por rate limit).
MAX_WORKERS = 8

# Reexecutar a coleta de .info quantas vezes em caso de falha.
INFO_MAX_RETRIES = 2


if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False
if "debug" not in st.session_state:
    st.session_state.debug = False


def ativar_filtros() -> None:
    st.session_state.filtros_ativos = True


# ===================== FUNÇÕES TÉCNICAS =====================
def calcular_graham(lpa: float, vpa: float) -> float:
    if lpa > 0 and vpa > 0:
        return float(np.sqrt(22.5 * lpa * vpa))
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


def calcular_rsi(data: pd.Series, window: int = 14) -> float:
    if len(data) < window:
        return 50.0
    rsi_series = calcular_rsi_series(data, window)
    val = rsi_series.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _dy_pct(info: dict) -> float:
    """Normaliza dividendYield para percentual."""
    raw = info.get("dividendYield", 0) or 0
    return raw * 100 if DY_AS_FRACTION else raw


def calcular_score_value(info: dict) -> int:
    score = 0
    if 0 < (info.get("trailingPE") or 99) < 15:
        score += 1
    if 0 < (info.get("priceToBook") or 99) < 1.5:
        score += 1
    if _dy_pct(info) > 5:
        score += 1
    if (info.get("operatingMargins") or 0) > 0.1:
        score += 1
    return score


# ===================== SIMULAÇÃO VETORIZADA =====================
def simular_performance_historica(hist: pd.DataFrame) -> dict:
    vazio = {
        "taxa_compra": 0.0, "taxa_venda": 0.0,
        "retorno_medio_compra": 0.0, "retorno_medio_venda": 0.0,
        "expectancy_compra": 0.0, "expectancy_venda": 0.0,
        "qtd_compra": 0, "qtd_venda": 0,
    }
    if hist is None or hist.empty or "Close" not in hist or len(hist) < 300:
        return vazio

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

    # Compras
    if buy_mask.any():
        ret_buy = retorno_15d[buy_mask]
        qtd_c = int(buy_mask.sum())
        taxa_c_frac = (ret_buy > 0).mean()
        taxa_c = taxa_c_frac * 100
        ret_med_c = ret_buy.mean() * 100
        avg_win = ret_buy[ret_buy > 0].mean() if (ret_buy > 0).any() else 0.0
        avg_loss = abs(ret_buy[ret_buy < 0].mean()) if (ret_buy < 0).any() else 0.0
        exp_c = (taxa_c_frac * avg_win - (1 - taxa_c_frac) * avg_loss) * 100
    else:
        taxa_c = ret_med_c = exp_c = 0.0
        qtd_c = 0

    # Vendas
    if sell_mask.any():
        ret_sell = retorno_15d[sell_mask]
        qtd_v = int(sell_mask.sum())
        taxa_v_frac = (ret_sell < 0).mean()
        taxa_v = taxa_v_frac * 100
        ret_med_v = ret_sell.mean() * 100
        avg_win_v = abs(ret_sell[ret_sell < 0].mean()) if (ret_sell < 0).any() else 0.0
        avg_loss_v = ret_sell[ret_sell > 0].mean() if (ret_sell > 0).any() else 0.0
        exp_v = (taxa_v_frac * avg_win_v - (1 - taxa_v_frac) * avg_loss_v) * 100
    else:
        taxa_v = ret_med_v = exp_v = 0.0
        qtd_v = 0

    return {
        "taxa_compra": float(taxa_c),
        "taxa_venda": float(taxa_v),
        "retorno_medio_compra": float(ret_med_c),
        "retorno_medio_venda": float(ret_med_v),
        "expectancy_compra": float(exp_c),
        "expectancy_venda": float(exp_v),
        "qtd_compra": qtd_c,
        "qtd_venda": qtd_v,
    }


# ===================== MACROECONÔMICOS =====================
@st.cache_data(ttl=1800, show_spinner=False)
def obter_macro() -> dict:
    macro: dict[str, Any] = {}
    try:
        selic_data = yf.Ticker("^SELIC").history(period="5d")
        macro["Selic"] = float(selic_data["Close"].iloc[-1]) if not selic_data.empty else 14.75
    except Exception:
        macro["Selic"] = 14.75
    try:
        macro["Dolar"] = float(yf.Ticker("USDBRL=X").fast_info.last_price)
    except Exception:
        macro["Dolar"] = 4.99
    macro["IPCA_12m"] = 4.14
    macro["Focus_Data"] = "13/04/2026"
    macro["Focus_Selic_2026"] = "12.50%"
    macro["Focus_IPCA_2026"] = "4.36%"
    macro["Focus_PIB_2026"] = "1.85%"
    return macro


# ===================== CACHE DE MERCADO =====================
@st.cache_data(ttl=300, show_spinner=False)
def obter_indices() -> dict:
    indices = {"Ibovespa": "^BVSP", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}
    resultados: dict[str, tuple[float, float]] = {}
    for nome, ticker in indices.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if not data.empty and len(data) >= 2:
                atual = float(data["Close"].iloc[-1])
                anterior = float(data["Close"].iloc[-2])
                variacao = ((atual / anterior) - 1) * 100
                resultados[nome] = (atual, variacao)
            else:
                resultados[nome] = (0.0, 0.0)
        except Exception:
            resultados[nome] = (0.0, 0.0)
    return resultados


@st.cache_data(ttl=90, show_spinner=False)
def obter_cambio() -> dict:
    moedas = {"Dólar": "USDBRL=X", "Euro": "EURBRL=X", "Libra": "GBPBRL=X"}
    resultados: dict[str, tuple[float, float]] = {}
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
        except Exception:
            resultados[nome] = (0.0, 0.0)

    # Bitcoin em Real
    btc_real = 0.0
    try:
        t = yf.Ticker("BTC-BRL")
        data = t.history(period="2d")
        if not data.empty and len(data) >= 1:
            btc_real = float(data["Close"].iloc[-1])
        else:
            btc_real = float(t.fast_info.last_price)
    except Exception:
        pass
    if btc_real < 100000:
        try:
            btc_usd = float(yf.Ticker("BTC-USD").fast_info.last_price)
            dolar_brl = resultados.get("Dólar", (4.99, 0.0))[0]
            btc_real = btc_usd * dolar_brl
        except Exception:
            btc_real = 0.0
    resultados["Bitcoin"] = (btc_real, 0.0)
    return resultados


# ===================== DOWNLOAD EM BATCH (ROBUSTO) =====================
def _extrair_hist_do_batch(hist_multi: pd.DataFrame, t_yf: str, n_tickers: int) -> pd.DataFrame:
    """
    Extrai o histórico de UM ticker de um DataFrame retornado por
    yf.download(..., group_by='ticker'). Lida com três cenários:
      * DataFrame vazio (download falhou inteiro).
      * 1 ticker (colunas flat: Open/High/.../Close).
      * N tickers (MultiIndex por ticker).
    """
    if hist_multi is None or hist_multi.empty:
        return pd.DataFrame()

    # 1 ticker: colunas flat
    if n_tickers == 1 or not isinstance(hist_multi.columns, pd.MultiIndex):
        return hist_multi.copy()

    # N tickers: MultiIndex
    try:
        level0 = hist_multi.columns.get_level_values(0)
    except Exception:
        return pd.DataFrame()

    if t_yf in level0:
        try:
            sub = hist_multi[t_yf].copy()
            # Se vier com todas colunas NaN, é falha do Yahoo pra esse ticker.
            if sub.dropna(how="all").empty:
                return pd.DataFrame()
            return sub
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def _buscar_info(t_yf: str) -> dict:
    """Busca .info com retry + backoff e fallback para fast_info."""
    for tentativa in range(INFO_MAX_RETRIES + 1):
        try:
            info = yf.Ticker(t_yf).info or {}
            # yfinance às vezes devolve dict com 1-2 chaves quando falha;
            # considere como falha e tente de novo.
            if len(info) > 5:
                return info
        except Exception:
            pass
        time.sleep(0.4 * (tentativa + 1))

    # Último recurso: fast_info dá pelo menos preço e marketcap.
    try:
        fi = yf.Ticker(t_yf).fast_info
        out: dict[str, Any] = {}
        for chave_src, chave_dst in (
            ("last_price", "regularMarketPrice"),
            ("market_cap", "marketCap"),
            ("shares", "sharesOutstanding"),
            ("currency", "currency"),
        ):
            try:
                out[chave_dst] = getattr(fi, chave_src)
            except Exception:
                continue
        return out
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def obter_dados_batch(tickers: list[str], mercado: str) -> tuple[dict, dict, list]:
    """
    Retorna (info_dict, hist_dict, erros).
    `erros` é uma lista de strings com diagnósticos para a UI.
    """
    erros: list[str] = []
    if not tickers:
        return {}, {}, erros

    tickers_yf = [
        t + ".SA" if mercado == "Brasil" and not t.endswith(".SA") else t
        for t in tickers
    ]

    # --- 1) Histórico em batch ---
    hist_multi: pd.DataFrame = pd.DataFrame()
    try:
        hist_multi = yf.download(
            tickers_yf,
            period="5y",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        erros.append(f"yf.download falhou: {e!r}")

    if hist_multi is None or hist_multi.empty:
        erros.append(
            "yf.download retornou DataFrame vazio. Provável rate-limit do Yahoo "
            "ou rede bloqueada. Tente rodar novamente em alguns minutos."
        )

    hist_dict: dict[str, pd.DataFrame] = {}
    for t_orig, t_yf in zip(tickers, tickers_yf):
        hist_dict[t_orig] = _extrair_hist_do_batch(hist_multi, t_yf, len(tickers))

    # --- 2) Info em paralelo (gargalo real) ---
    info_dict: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futuros = {pool.submit(_buscar_info, t_yf): t_orig for t_orig, t_yf in zip(tickers, tickers_yf)}
        for fut in as_completed(futuros):
            t_orig = futuros[fut]
            try:
                info_dict[t_orig] = fut.result() or {}
            except Exception as e:
                info_dict[t_orig] = {}
                erros.append(f"info {t_orig}: {e!r}")

    # Diagnóstico agregado
    total = len(tickers)
    hist_ok = sum(1 for h in hist_dict.values() if isinstance(h, pd.DataFrame) and not h.empty)
    info_ok = sum(1 for i in info_dict.values() if len(i) > 5)
    erros.insert(
        0,
        f"Resumo: {hist_ok}/{total} tickers com histórico, {info_ok}/{total} tickers com .info rico.",
    )

    return info_dict, hist_dict, erros


# ===================== PROCESSAMENTO CENTRAL =====================
def processar_ativo(
    tkr: str,
    info: dict,
    hist: pd.DataFrame,
    estrategia_ativa: str,
    filtros_ativos: bool,
    f_pl: float,
    f_pvp: float,
    f_dy: float,
    f_div_ebitda: float,
    busca_direta: str,
    mercado: str,
) -> dict | None:
    """
    Retorna dict com os dados do ativo, ou None se o ativo deve ser
    excluído (por falha de dados ou por não passar nos filtros).

    Se `st.session_state.debug` estiver ligado, motivos do descarte são
    impressos em `st.warning` para facilitar diagnóstico.
    """
    debug = st.session_state.get("debug", False)

    # Preço: exige histórico OU um preço vindo do fast_info/info
    preco_info = info.get("regularMarketPrice") or info.get("currentPrice")
    if (hist is None or hist.empty) and not preco_info:
        if debug:
            st.warning(f"[{tkr}] descartado: sem histórico e sem preço em info.")
        return None

    if not info:
        if debug:
            st.warning(f"[{tkr}] descartado: info vazio.")
        return None

    pl = info.get("trailingPE") or 0
    pvp = info.get("priceToBook") or 0
    dy = _dy_pct(info)
    ebitda = info.get("ebitda") or 1
    div_liq = info.get("totalDebt") or 0
    cash = info.get("totalCash") or 0
    div_e = (div_liq - cash) / ebitda if ebitda else 999.0

    lpa = info.get("trailingEps") or 0
    vpa = info.get("bookValue") or 0
    p_justo = calcular_graham(lpa, vpa)

    if hist is not None and not hist.empty and "Close" in hist:
        p_atual = float(hist["Close"].iloc[-1])
    else:
        p_atual = float(preco_info or 0.0)

    upside = ((p_justo / p_atual) - 1) * 100 if p_justo > 0 and p_atual > 0 else 0.0

    if not busca_direta and filtros_ativos:
        if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
            if debug:
                st.info(
                    f"[{tkr}] filtrado: P/L={pl:.1f}, P/VP={pvp:.1f}, "
                    f"DY={dy:.1f}%, Dív/EBITDA={div_e:.1f}."
                )
            return None

    # Notícias
    noticias_texto = ""
    lista_links: list[dict] = []
    try:
        lang = "pt-BR" if mercado == "Brasil" else "en-US"
        url = f"https://news.google.com/rss/search?q={tkr}&hl={lang}"
        feed = feedparser.parse(url)
        for entry in feed.entries[:5]:
            noticias_texto += entry.title.lower() + " "
            lista_links.append({"titulo": entry.title, "link": entry.link})
    except Exception:
        pass

    score_p = sum(
        noticias_texto.count(w)
        for w in ["alta", "lucro", "compra", "subiu", "dividend", "profit", "buy"]
    )
    score_n = sum(
        noticias_texto.count(w)
        for w in ["queda", "prejuízo", "venda", "caiu", "risk", "loss", "sell"]
    )

    rsi_val = (
        calcular_rsi(hist["Close"])
        if hist is not None and not hist.empty and "Close" in hist
        else 50.0
    )
    score_value = calcular_score_value(info)
    sim = simular_performance_historica(hist)

    # ===================== LÓGICA DE ESTRATÉGIAS =====================
    if estrategia_ativa == "Value Investing (Graham/Buffett)":
        if upside > 20 and score_value >= 3:
            veredito, cor = "VALOR ✅", "success"
            motivo = f"Forte margem de segurança ({upside:.1f}%) e bons fundamentos (Score: {score_value}/4)."
        elif upside < 0 and p_justo > 0:
            veredito, cor = "CARO 🚨", "error"
            motivo = "Preço acima do valor intrínseco de Graham."
        else:
            veredito, cor = "NEUTRO ⚖️", "warning"
            motivo = "Ativo próximo ao preço justo."

    elif estrategia_ativa == "Dividend Investing":
        if dy >= 6.0 and div_e < 3.0:
            veredito, cor = "RENDA ✅", "success"
            motivo = f"Alto Dividend Yield ({dy:.2f}%) e dívida controlada ({div_e:.1f}x)."
        elif dy < 3.0:
            veredito, cor = "BAIXO DY 🚨", "error"
            motivo = f"Dividend Yield fraco para a estratégia ({dy:.2f}%)."
        else:
            veredito, cor = "NEUTRO ⚖️", "warning"
            motivo = f"DY mediano ({dy:.2f}%), monitorar a consistência."

    elif estrategia_ativa == "Growth Investing":
        rev_growth = (info.get("revenueGrowth") or 0) * 100
        if rev_growth > 15.0:
            veredito, cor = "CRESCIMENTO ✅", "success"
            motivo = f"Forte aceleração de receita ({rev_growth:.1f}%) ano a ano."
        elif rev_growth < 0:
            veredito, cor = "DECLÍNIO 🚨", "error"
            motivo = "Empresa apresentando retração nas receitas operacionais."
        else:
            veredito, cor = "NEUTRO ⚖️", "warning"
            motivo = f"Crescimento de receita brando ou estagnado ({rev_growth:.1f}%)."

    elif estrategia_ativa == "Buy and Hold":
        if score_value >= 3 and div_e < 2.5 and 0 < pl < 25:
            veredito, cor = "ACUMULAR ✅", "success"
            motivo = "Excelentes fundamentos de longo prazo e baixo risco de endividamento."
        elif div_e > 5.0 or pl > 50:
            veredito, cor = "RISCO 🚨", "error"
            motivo = "Múltiplos esticados ou endividamento excessivo para carrego longo."
        else:
            veredito, cor = "MANTER ⚖️", "warning"
            motivo = "Fundamentos dentro da média, sem sinais de alerta graves."

    elif estrategia_ativa == "Position Trading":
        sma200_atual = (
            hist["Close"].rolling(window=200).mean().iloc[-1]
            if hist is not None and len(hist) >= 200
            else 0
        )
        sma50_atual = (
            hist["Close"].rolling(window=50).mean().iloc[-1]
            if hist is not None and len(hist) >= 50
            else 0
        )
        sma200_atual = float(sma200_atual) if pd.notna(sma200_atual) else 0.0
        sma50_atual = float(sma50_atual) if pd.notna(sma50_atual) else 0.0

        if p_atual > sma50_atual > sma200_atual > 0:
            veredito, cor = "TENDÊNCIA ALTA ✅", "success"
            motivo = "Preço suportado acima das médias móveis de 50 e 200 dias (Uptrend)."
        elif 0 < sma200_atual and p_atual < sma50_atual < sma200_atual:
            veredito, cor = "TENDÊNCIA BAIXA 🚨", "error"
            motivo = "Preço abaixo das principais médias móveis (Downtrend)."
        else:
            veredito, cor = "LATERAL ⚖️", "warning"
            motivo = "Ativo cruzando médias, sem tendência direcional confirmada."

    else:  # Análise Técnica (Trader)
        if rsi_val > 70 and score_n > score_p:
            veredito, cor = "VENDA 🚨", "error"
            motivo = f"RSI alto ({rsi_val:.1f}) e notícias negativas."
        elif rsi_val > 75:
            veredito, cor = "VENDA 🚨", "error"
            motivo = f"RSI em nível extremo ({rsi_val:.1f})."
        elif score_p > score_n and rsi_val < 65:
            veredito, cor = "COMPRA ✅", "success"
            motivo = f"Notícias positivas e RSI saudável ({rsi_val:.1f})."
        elif score_n > score_p or rsi_val > 70:
            veredito, cor = "CAUTELA ⚠️", "error"
            partes = []
            if rsi_val > 70:
                partes.append(f"RSI alto ({rsi_val:.1f})")
            if score_n > score_p:
                partes.append("Sentimento negativo")
            motivo = " | ".join(partes) or "Sinais mistos."
        else:
            veredito, cor = "NEUTRO ⚖️", "warning"
            motivo = "Indicadores em equilíbrio no curto prazo."

    return {
        "Ticker": tkr,
        "Empresa": info.get("shortName") or info.get("longName") or tkr,
        "Preço": p_atual,
        "P/L": pl,
        "DY %": dy,
        "Dívida": div_e,
        "Graham": p_justo,
        "Upside %": upside,
        "Veredito": veredito,
        "Cor": cor,
        "Motivo": motivo,
        "RSI": rsi_val,
        "Hist": hist if hist is not None else pd.DataFrame(),
        "Links": lista_links,
        "ValueScore": score_value,
        "TaxaCompra": sim["taxa_compra"],
        "TaxaVenda": sim["taxa_venda"],
        "RetornoMedioCompra": sim["retorno_medio_compra"],
        "ExpectancyCompra": sim["expectancy_compra"],
        "QtdCompra": sim["qtd_compra"],
        "QtdVenda": sim["qtd_venda"],
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

# ===================== MACRO & CENÁRIO =====================
st.sidebar.subheader("📊 Macro & Cenário")
macro = obter_macro()

st.sidebar.metric("Selic Atual", f"{macro['Selic']:.2f}%")
st.sidebar.metric("IPCA 12m", f"{macro['IPCA_12m']:.2f}%")
st.sidebar.metric("Dólar", f"R$ {macro['Dolar']:.2f}")

with st.sidebar.expander("📌 Impacto no Mercado", expanded=True):
    st.markdown(
        """
    **Selic Alta** → Prejudica ações de crescimento e empresas alavancadas  
    **Inflação Controlada** → Melhora margens de lucro  
    **Dólar Alto** → Beneficia exportadoras | Prejudica importadoras  
    **PIB Crescente** → Favorece consumo, varejo e serviços
    """
    )
    st.markdown(f"**Último Focus ({macro['Focus_Data']})**")
    st.markdown(f"- Selic 2026: **{macro['Focus_Selic_2026']}**")
    st.markdown(f"- IPCA 2026: **{macro['Focus_IPCA_2026']}**")
    st.markdown(f"- PIB 2026: **{macro['Focus_PIB_2026']}**")

st.sidebar.divider()

# ===================== ANÁLISE FUNDAMENTALISTA =====================
st.sidebar.subheader("📉 Análise Fundamentalista")
with st.sidebar.expander("Indicadores Chave", expanded=False):
    st.markdown(
        """
    **Eficiência e Rentabilidade:**
    - **ROE** — Retorno sobre o Patrimônio
    - **Margem Líquida** — Lucro Líquido / Receita
    - **Margem EBITDA** — Eficiência operacional

    **Valuation:**
    - **P/L** — Preço sobre Lucro
    - **P/VP** — Preço sobre Valor Patrimonial

    **Endividamento:**
    - **Dívida Líquida / EBITDA** — Nível de alavancagem
    """
    )

st.sidebar.divider()

# ===================== GESTÃO DE RISCO =====================
st.sidebar.subheader("🛡️ Gestão de Risco e Portfólio")
with st.sidebar.expander("Estratégias Recomendadas", expanded=False):
    st.markdown(
        """
    - **Alocação de Ativos**: Rebalancear carteira conforme risco e valorização
    - **Análise Técnica**: Definir pontos de entrada e saída com suporte/resistência
    - **Gestão de Risco**: Stop-loss, diversificação e controle de exposição
    """
    )

st.sidebar.divider()

# ===================== RESTO DO SIDEBAR =====================
mercado_selecionado = st.sidebar.radio(
    "Escolha o Mercado:", ["Brasil", "EUA"], on_change=ativar_filtros
)

estrategia_ativa = st.sidebar.selectbox(
    "Foco da Análise:",
    [
        "Value Investing (Graham/Buffett)",
        "Análise Técnica (Trader)",
        "Growth Investing",
        "Buy and Hold",
        "Dividend Investing",
        "Position Trading",
    ],
)

busca_direta = (
    st.sidebar.text_input(f"🔍 Busca Rápida ({mercado_selecionado}):").upper().strip()
)

with st.sidebar.expander("📊 Filtros de Valuation", expanded=True):
    f_pl = st.slider("P/L Máximo", 0.0, 50.0, 50.0, step=0.5, on_change=ativar_filtros)
    f_pvp = st.slider("P/VP Máximo", 0.0, 10.0, 10.0, step=0.1, on_change=ativar_filtros)
    f_dy = st.slider("DY Mínimo (%)", 0.0, 20.0, 0.0, step=0.5, on_change=ativar_filtros)
    f_div_ebitda = st.slider(
        "Dív.Líq/EBITDA Máximo", 0.0, 15.0, 15.0, step=0.5, on_change=ativar_filtros
    )

if st.sidebar.button("Resetar Filtros"):
    st.session_state.filtros_ativos = False
    st.rerun()

st.session_state.debug = st.sidebar.toggle(
    "🐞 Modo debug",
    value=st.session_state.debug,
    help="Mostra motivos de descarte de cada ticker.",
)

# ===================== LISTA DE ATIVOS =====================
if mercado_selecionado == "Brasil":
    lista_base = [
        "PETR4", "VALE3", "ITUB4", "BBAS3", "BBDC4", "SANB11", "B3SA3", "EGIE3",
        "TRPL4", "TAEE11", "SAPR11", "CPLE6", "ELET3", "CMIG4", "SBSP3", "ABEV3",
        "WEGE3", "RADL3", "RENT3", "MGLU3", "LREN3", "RAIZ4", "VBBR3", "SUZB3",
        "KLBN11", "GOAU4", "CSNA3", "PRIO3", "JBSS3", "BRFS3", "GGBR4", "HAPV3",
        "RDOR3",
    ]
    moeda_simbolo = "R$"
else:
    lista_base = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "DIS",
        "KO", "PEP", "MCD", "NKE", "WMT", "JPM", "V", "MA", "BAC", "PYPL", "PFE",
        "JNJ", "PG", "COST", "ORCL",
    ]
    moeda_simbolo = "US$"

# ===================== CABEÇALHO =====================
st.title(f"🤖 Monitor IA - {mercado_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# ===================== PROCESSAMENTO PRINCIPAL =====================
tickers_para_processar = [busca_direta] if busca_direta else lista_base
dados_vencedoras: list[dict] = []
erros_batch: list[str] = []

if tickers_para_processar:
    with st.spinner("📡 Baixando dados em batch..."):
        infos, hists, erros_batch = obter_dados_batch(tickers_para_processar, mercado_selecionado)

    for tkr in tickers_para_processar:
        info = infos.get(tkr, {})
        hist = hists.get(tkr, pd.DataFrame())
        resultado = processar_ativo(
            tkr, info, hist, estrategia_ativa,
            st.session_state.filtros_ativos,
            f_pl, f_pvp, f_dy, f_div_ebitda,
            busca_direta, mercado_selecionado,
        )
        if resultado:
            dados_vencedoras.append(resultado)

# Diagnóstico sempre visível (expandível) — ajuda a entender quando
# nada aparece na tela.
with st.expander("🧪 Status do Batch (diagnóstico)", expanded=not dados_vencedoras):
    if erros_batch:
        for linha in erros_batch:
            st.write("•", linha)
    else:
        st.write("Nada a reportar.")
    st.caption(
        "Se o resumo mostra 0/N histórico ou 0/N info, o Yahoo está rate-limitando "
        "ou sua rede está bloqueando. Tente novamente em alguns minutos ou rode "
        "`pip install -U yfinance`."
    )

# ===================== INTERFACE =====================
if dados_vencedoras:
    st.subheader(f"🏆 Ranking de Oportunidades - Estratégia: {estrategia_ativa}")

    with st.expander("📌 Legenda de Sinais e Vereditos"):
        st.markdown(
            """
        * **VALOR ✅**: Grande desconto + bons fundamentos  
        * **COMPRA ✅**: RSI saudável + notícias positivas  
        * **VENDA / CARO 🚨**: RSI extremo ou preço acima do justo  
        * **CAUTELA ⚠️**: Divergência entre preço, RSI e sentimento  
        * **Expectancy**: Ganho esperado por operação (quanto maior, melhor)
        """
        )

    df_resumo = pd.DataFrame(dados_vencedoras)[
        [
            "Ticker", "Preço", "DY %", "Upside %", "Veredito", "Motivo",
            "TaxaCompra", "ExpectancyCompra", "QtdCompra",
        ]
    ]

    st.dataframe(
        df_resumo.sort_values(by="Upside %", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Veredito": st.column_config.TextColumn("Veredito"),
            "Motivo": st.column_config.TextColumn("Motivo da IA", width="medium"),
            "TaxaCompra": st.column_config.NumberColumn("Win Rate Compra", format="%.1f%%"),
            "ExpectancyCompra": st.column_config.NumberColumn("Expectancy Compra", format="%.2f%%"),
            "QtdCompra": st.column_config.NumberColumn("Sinais"),
        },
    )

    for acao in dados_vencedoras:
        st.divider()
        col_tit, col_ver, col_acc_c, col_acc_v = st.columns([3, 1, 1, 1])
        col_tit.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")

        col_acc_c.metric("Assert. Compra", f"{acao['TaxaCompra']:.1f}%", f"{acao['QtdCompra']} sinais")
        col_acc_v.metric("Assert. Venda", f"{acao['TaxaVenda']:.1f}%", f"{acao['QtdVenda']} sinais")

        if acao["Cor"] == "success":
            col_ver.success(f"**{acao['Veredito']}**")
        elif acao["Cor"] == "error":
            col_ver.error(f"**{acao['Veredito']}**")
        else:
            col_ver.warning(f"**{acao['Veredito']}**")

        if isinstance(acao["Hist"], pd.DataFrame) and not acao["Hist"].empty and "Close" in acao["Hist"]:
            st.line_chart(acao["Hist"]["Close"], use_container_width=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Preço Atual", f"{moeda_simbolo} {acao['Preço']:.2f}")
        c2.metric("P/L", round(acao["P/L"], 2))
        c3.metric("DY", f"{acao['DY %']:.2f}%")
        c4.metric("Dív.Líq/EBITDA", round(acao["Dívida"], 2))
        c5.metric("Graham", f"{moeda_simbolo} {acao['Graham']:.2f}", f"{acao['Upside %']:.1f}%")

        with st.expander(f"📊 Detalhes: {acao['Ticker']}"):
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

elif erros_batch and any("0/" in e for e in erros_batch):
    st.error(
        "❌ Nenhum dado retornou do Yahoo Finance. Veja o **Status do Batch** "
        "acima para o motivo. Isso normalmente é rate-limit — tente de novo em "
        "alguns minutos."
    )
else:
    st.info("💡 Use os filtros ou faça uma busca direta para começar.")
