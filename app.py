import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser

st.set_page_config(page_title="Monitor IA Pro", layout="wide")

if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    st.session_state.filtros_ativos = True

# ===================== FUNÇÕES =====================
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

# ===================== CACHE =====================
@st.cache_data(ttl=600)
def obter_dados_batch(tickers, mercado):
    tickers_yf = [t + ".SA" if mercado == "Brasil" else t for t in tickers]
    hist_multi = yf.download(tickers_yf, period="2y", group_by="ticker", auto_adjust=True, progress=False)
    info_dict = {}
    hist_dict = {}
    for i, t_orig in enumerate(tickers):
        t_yf = tickers_yf[i]
        try:
            info_dict[t_orig] = yf.Ticker(t_yf).info
            # Tratamento multi-level
            if isinstance(hist_multi.columns, pd.MultiIndex):
                if t_yf in hist_multi.columns.get_level_values(0):
                    hist_dict[t_orig] = hist_multi[t_yf]
                else:
                    hist_dict[t_orig] = pd.DataFrame()
            else:
                hist_dict[t_orig] = hist_multi
        except:
            info_dict[t_orig] = {}
            hist_dict[t_orig] = pd.DataFrame()
    return info_dict, hist_dict

# ===================== PROCESSAMENTO =====================
def processar_ativo(tkr, info, hist):
    if hist.empty or not info:
        return None

    # Preço atual seguro
    if isinstance(hist.columns, pd.MultiIndex):
        close_series = hist[('Close', tkr)] if ('Close', tkr) in hist.columns else hist.iloc[:, -1]
    else:
        close_series = hist["Close"] if "Close" in hist.columns else hist.iloc[:, -1]

    p_atual = float(close_series.iloc[-1]) if not close_series.empty else 0

    pl = info.get("trailingPE", 0) or 0
    dy = (info.get("dividendYield", 0) or 0) * 100
    score_value, criteria = calcular_score_value(info)

    veredito = "NEUTRO ⚖️"
    motivo = "Dados carregados"

    return {
        "Ticker": tkr,
        "Empresa": info.get("shortName", tkr),
        "Preço": p_atual,
        "P/L": pl,
        "DY %": dy,
        "Veredito": veredito,
        "Motivo": motivo,
        "ValueScore": score_value,
        "ValueCriteria": criteria,
        "Links": []  # será preenchido com notícias
    }

# ===================== INTERFACE =====================
st.title("🤖 Monitor IA - Brasil")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

tab1, tab3 = st.tabs(["📊 Overview", "📉 Fundamentalista"])

mercado_selecionado = "Brasil"
lista_base = ["PETR4", "VALE3", "ITUB4", "BBAS3", "B3SA3", "EGIE3", "WEGE3", "PRIO3", "JBSS3"]

busca_direta = st.sidebar.text_input("🔍 Busca Rápida (ex: PETR4)").upper().strip()
tickers_para_processar = [busca_direta] if busca_direta else lista_base

dados_vencedoras = []
if tickers_para_processar:
    with st.spinner("Baixando dados..."):
        infos, hists = obter_dados_batch(tickers_para_processar, mercado_selecionado)
    for tkr in tickers_para_processar:
        info = infos.get(tkr, {})
        hist = hists.get(tkr, pd.DataFrame())
        resultado = processar_ativo(tkr, info, hist)
        if resultado:
            dados_vencedoras.append(resultado)

# ===================== TAB 1 - OVERVIEW =====================
with tab1:
    if dados_vencedoras:
        st.subheader("🏆 Ranking")
        df = pd.DataFrame(dados_vencedoras)
        st.dataframe(
            df[["Ticker", "Preço", "DY %", "Veredito"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Nenhum ativo encontrado. Digite PETR4 na busca rápida.")

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
            st.divider()
    else:
        st.info("Nenhum ativo encontrado.")

st.info("💡 Digite PETR4 na busca rápida para testar.")
