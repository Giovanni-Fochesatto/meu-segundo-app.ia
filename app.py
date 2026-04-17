import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import feedparser
import logging
from streamlit_autorefresh import st_autorefresh
from typing import Dict, Tuple, Optional

# ===================== CONFIGURAÇÕES & LOGGING =====================
st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(interval=300 * 1000, key="data_refresh")

# Configurar logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

if "filtros_ativos" not in st.session_state:
    st.session_state.filtros_ativos = False

def ativar_filtros():
    st.session_state.filtros_ativos = True

# ===================== FUNÇÕES AUXILIARES =====================
def safe_divide(numerador: float, denominador: float, default: float = 0.0) -> float:
    """Divisão segura que evita ZeroDivisionError"""
    try:
        if pd.isna(numerador) or pd.isna(denominador) or denominador == 0:
            return default
        return float(numerador) / float(denominador)
    except (ValueError, TypeError):
        return default

def validar_serie(series: pd.Series, min_length: int = 14) -> bool:
    """Valida se uma série tem dados suficientes"""
    return isinstance(series, pd.Series) and len(series) >= min_length and not series.empty

# ===================== FUNÇÕES TÉCNICAS =====================
def calcular_graham(lpa: float, vpa: float) -> float:
    """
    Calcula o preço justo usando a fórmula de Graham.
    Retorna NaN se os dados forem inválidos.
    """
    try:
        if pd.isna(lpa) or pd.isna(vpa) or lpa <= 0 or vpa <= 0:
            return np.nan
        valor = np.sqrt(22.5 * lpa * vpa)
        return float(valor) if not np.isnan(valor) else np.nan
    except (ValueError, OverflowError) as e:
        logger.warning(f"Erro ao calcular Graham: {e}")
        return np.nan

def calcular_rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    """Calcula RSI corrigido com proteção contra divisão por zero"""
    try:
        if not validar_serie(close, window):
            return pd.Series([50.0] * len(close), index=close.index)
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        # Proteção contra divisão por zero
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Substituir infinitos e NaN por 50 (neutro)
        rsi = rsi.replace([np.inf, -np.inf], 50.0).fillna(50.0)
        
        return rsi
    except Exception as e:
        logger.warning(f"Erro ao calcular RSI series: {e}")
        return pd.Series([50.0] * len(close), index=close.index)

def calcular_rsi(data: pd.Series, window: int = 14) -> float:
    """Calcula RSI do último valor da série"""
    try:
        if not validar_serie(data, window):
            return 50.0
        
        rsi_series = calcular_rsi_series(data, window)
        ultimo_rsi = float(rsi_series.iloc[-1])
        
        # Validar se é um número válido
        if pd.isna(ultimo_rsi) or np.isinf(ultimo_rsi):
            return 50.0
        
        return min(100.0, max(0.0, ultimo_rsi))  # Limitar entre 0 e 100
    except Exception as e:
        logger.warning(f"Erro ao calcular RSI: {e}")
        return 50.0

def calcular_score_value(info: Dict) -> int:
    """Calcula score de value investing (0-4)"""
    score = 0
    try:
        pl = info.get("trailingPE", 99)
        if pd.notna(pl) and 0 < pl < 15:
            score += 1
        
        pvp = info.get("priceToBook", 99)
        if pd.notna(pvp) and 0 < pvp < 1.5:
            score += 1
        
        dy = (info.get("dividendYield", 0) or 0) * 100
        if pd.notna(dy) and dy > 5:
            score += 1
        
        margin = info.get("operatingMargins", 0) or 0
        if pd.notna(margin) and margin > 0.1:
            score += 1
    except Exception as e:
        logger.warning(f"Erro ao calcular score value: {e}")
    
    return score

# ===================== SIMULAÇÃO VETORIZADA MELHORADA =====================
def simular_performance_historica(hist: pd.DataFrame) -> Dict:
    """Simula performance histórica com proteção contra erros"""
    
    resultado_padrao = {
        "taxa_compra": 0.0, "taxa_venda": 0.0,
        "retorno_medio_compra": 0.0, "retorno_medio_venda": 0.0,
        "expectancy_compra": 0.0, "expectancy_venda": 0.0,
        "qtd_compra": 0, "qtd_venda": 0
    }
    
    try:
        if hist.empty or len(hist) < 300:
            return resultado_padrao
        
        close = hist["Close"].copy()
        
        # Calcular indicadores
        rsi = calcular_rsi_series(close)
        sma200 = close.rolling(window=200).mean()
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        sinal_macd = macd.ewm(span=9, adjust=False).mean()
        retorno_15d = close.shift(-15) / close - 1
        
        # Criar máscaras
        buy_mask = (rsi < 35) & (close > sma200) & (macd > sinal_macd) & retorno_15d.notna()
        sell_mask = (rsi > 70) & ((close < sma200) | (macd < sinal_macd)) & retorno_15d.notna()
        
        # ===================== COMPRAS =====================
        taxa_c = ret_med_c = exp_c = 0.0
        qtd_c = 0
        
        if buy_mask.any():
            ret_buy = retorno_15d[buy_mask]
            qtd_c = int(buy_mask.sum())
            taxa_c = (ret_buy > 0).mean() * 100
            ret_med_c = ret_buy.mean() * 100
            
            avg_win = ret_buy[ret_buy > 0].mean() if (ret_buy > 0).any() else 0.0
            avg_loss = abs(ret_buy[ret_buy < 0].mean()) if (ret_buy < 0).any() else 0.0
            
            # Corrigir a fórmula de expectancy
            exp_c = ((taxa_c / 100) * avg_win - (1 - taxa_c / 100) * avg_loss) * 100
        
        # ===================== VENDAS =====================
        taxa_v = ret_med_v = exp_v = 0.0
        qtd_v = 0
        
        if sell_mask.any():
            ret_sell = retorno_15d[sell_mask]
            qtd_v = int(sell_mask.sum())
            taxa_v = (ret_sell < 0).mean() * 100
            ret_med_v = ret_sell.mean() * 100
            
            avg_win_v = abs(ret_sell[ret_sell < 0].mean()) if (ret_sell < 0).any() else 0.0
            avg_loss_v = ret_sell[ret_sell > 0].mean() if (ret_sell > 0).any() else 0.0
            
            # Corrigir a fórmula de expectancy
            exp_v = ((taxa_v / 100) * avg_win_v - (1 - taxa_v / 100) * avg_loss_v) * 100
        
        return {
            "taxa_compra": taxa_c,
            "taxa_venda": taxa_v,
            "retorno_medio_compra": ret_med_c,
            "retorno_medio_venda": ret_med_v,
            "expectancy_compra": exp_c,
            "expectancy_venda": exp_v,
            "qtd_compra": qtd_c,
            "qtd_venda": qtd_v
        }
    
    except Exception as e:
        logger.error(f"Erro ao simular performance: {e}")
        return resultado_padrao

# ===================== MACROECONÔMICOS =====================
@st.cache_data(ttl=300, show_spinner=False)
def obter_macro() -> Dict:
    """Obtém dados macroeconômicos com fallback"""
    macro = {}
    
    try:
        selic_data = yf.Ticker("^SELIC").history(period="5d")
        if not selic_data.empty:
            macro["Selic"] = float(selic_data["Close"].iloc[-1])
        else:
            macro["Selic"] = 14.75
    except Exception as e:
        logger.warning(f"Erro ao buscar Selic: {e}")
        macro["Selic"] = 14.75
    
    try:
        macro["Dolar"] = float(yf.Ticker("USDBRL=X").fast_info.last_price)
    except Exception as e:
        logger.warning(f"Erro ao buscar Dólar: {e}")
        macro["Dolar"] = 4.99
    
    try:
        macro["IPCA_12m"] = 4.14
    except Exception as e:
        logger.warning(f"Erro ao buscar IPCA: {e}")
        macro["IPCA_12m"] = 4.14
    
    # Dados do Focus
    macro["Focus_Data"] = "17/04/2026"
    macro["Focus_Selic_2026"] = "12.50%"
    macro["Focus_IPCA_2026"] = "4.36%"
    macro["Focus_PIB_2026"] = "1.85%"
    
    return macro

# ===================== CACHE DE MERCADO - CORRIGIDO =====================
@st.cache_data(ttl=300, show_spinner=False)
def obter_indices() -> Dict[str, Tuple[float, float]]:
    """Obtém índices mundiais com tratamento de erro - VOLTADO PARA INDIVIDUAL"""
    indices = {"Ibovespa": "^BVSP", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}
    resultados = {}
    
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
        except Exception as e:
            logger.warning(f"Erro ao buscar {nome}: {e}")
            resultados[nome] = (0.0, 0.0)
    
    return resultados

@st.cache_data(ttl=90, show_spinner=False)
def obter_cambio() -> Dict[str, Tuple[float, float]]:
    """Obtém câmbio em tempo real com fallback - VOLTADO PARA INDIVIDUAL"""
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
        except Exception as e:
            logger.warning(f"Erro ao buscar {nome}: {e}")
            resultados[nome] = (0.0, 0.0)
    
    # ===================== BITCOIN EM REAL =====================
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
    except Exception as e:
        logger.warning(f"Erro ao buscar BTC-BRL: {e}")
    
    # Fallback: calcular via BTC-USD
    if btc_real < 100000:
        try:
            btc_usd = float(yf.Ticker("BTC-USD").fast_info.last_price)
            dolar_brl = resultados.get("Dólar", (4.99, 0))[0]
            if dolar_brl > 0:
                btc_real = btc_usd * dolar_brl
        except Exception as e:
            logger.warning(f"Erro ao calcular BTC em reais: {e}")
    
    resultados["Bitcoin"] = (btc_real, 0.0)
    return resultados

# ===================== DOWNLOAD EM BATCH =====================
@st.cache_data(ttl=600, show_spinner=False)
def obter_dados_batch(tickers: list, mercado: str) -> Tuple[Dict, Dict]:
    """Obtém dados de múltiplos tickers com tratamento de erro robusto"""
    info_dict = {}
    hist_dict = {}
    
    if not tickers or not isinstance(tickers, list) or len(tickers) == 0:
        return info_dict, hist_dict
    
    try:
        # Adicionar sufixo .SA para Brasil
        tickers_yf = [
            t + ".SA" if mercado == "Brasil" and not t.endswith(".SA") else t 
            for t in tickers
        ]
        
        # Download em batch
        hist_multi = yf.download(
            tickers_yf, 
            period="5y", 
            group_by="ticker", 
            auto_adjust=True, 
            progress=False, 
            threads=True
        )
        
        for i, t_orig in enumerate(tickers):
            t_yf = tickers_yf[i]
            
            try:
                # Obter info
                info_dict[t_orig] = yf.Ticker(t_yf).info
                
                # Obter histórico com tratamento de MultiIndex
                if len(tickers) == 1:
                    hist_dict[t_orig] = hist_multi
                else:
                    # Verificar se tem MultiIndex
                    try:
                        if t_yf in hist_multi.columns.get_level_values(0):
                            hist_dict[t_orig] = hist_multi[t_yf]
                        else:
                            hist_dict[t_orig] = pd.DataFrame()
                    except (KeyError, AttributeError):
                        # Sem MultiIndex, tenta acesso direto
                        if t_yf in hist_multi.columns:
                            hist_dict[t_orig] = hist_multi[[t_yf]]
                            hist_dict[t_orig].columns = ["Close"]
                        else:
                            hist_dict[t_orig] = pd.DataFrame()
            
            except Exception as e:
                logger.warning(f"Erro ao processar {t_yf}: {e}")
                info_dict[t_orig] = {}
                hist_dict[t_orig] = pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Erro geral ao fazer download batch: {e}")
    
    return info_dict, hist_dict

# ===================== PROCESSAMENTO CENTRAL =====================
def processar_ativo(
    tkr: str,
    info: Dict,
    hist: pd.DataFrame,
    estrategia_ativa: str,
    filtros_ativos: bool,
    f_pl: float,
    f_pvp: float,
    f_dy: float,
    f_div_ebitda: float,
    busca_direta: str,
    mercado: str
) -> Optional[Dict]:
    """Processa um ativo individual com validação robusta"""
    
    try:
        # Validações iniciais
        if hist.empty or not info or "Close" not in hist.columns:
            return None
        
        # Extrair valores com proteção
        pl = info.get("trailingPE") or 0
        if pd.isna(pl):
            pl = 0
        
        pvp = info.get("priceToBook") or 0
        if pd.isna(pvp):
            pvp = 0
        
        dy = ((info.get("dividendYield") or 0) * 100)
        if pd.isna(dy):
            dy = 0
        
        ebitda = info.get("ebitda") or 1
        if pd.isna(ebitda) or ebitda == 0:
            ebitda = 1
        
        div_liq = info.get("totalDebt") or 0
        if pd.isna(div_liq):
            div_liq = 0
        
        cash = info.get("totalCash") or 0
        if pd.isna(cash):
            cash = 0
        
        div_e = safe_divide(div_liq - cash, ebitda, default=999.0)
        
        lpa = info.get("trailingEps") or 0
        if pd.isna(lpa):
            lpa = 0
        
        vpa = info.get("bookValue") or 0
        if pd.isna(vpa):
            vpa = 0
        
        p_justo = calcular_graham(lpa, vpa)
        p_atual = float(hist["Close"].iloc[-1])
        
        if pd.isna(p_justo) or p_justo <= 0:
            upside = 0.0
        else:
            upside = ((p_justo / p_atual) - 1) * 100 if p_atual > 0 else 0.0
        
        # Aplicar filtros
        if not busca_direta and filtros_ativos:
            if not (pl <= f_pl and pvp <= f_pvp and dy >= f_dy and div_e <= f_div_ebitda):
                return None
        
        # ===================== NOTÍCIAS =====================
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
        except Exception as e:
            logger.warning(f"Erro ao buscar notícias de {tkr}: {e}")
        
        score_p = sum(noticias_texto.count(w) for w in ["alta", "lucro", "compra", "subiu", "dividend", "profit", "buy"])
        score_n = sum(noticias_texto.count(w) for w in ["queda", "prejuízo", "venda", "caiu", "risk", "loss", "sell"])
        
        # ===================== INDICADORES =====================
        rsi_val = calcular_rsi(hist["Close"])
        score_value = calcular_score_value(info)
        sim = simular_performance_historica(hist)
        
        # ===================== LÓGICA DE ESTRATÉGIAS =====================
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
        
        else:  # Análise Técnica + Notícias
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
                if rsi_val > 70:
                    lista_motivos.append(f"RSI alto ({rsi_val:.1f})")
                if score_n > score_p:
                    lista_motivos.append("Sentimento negativo")
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
            "TaxaCompra": sim["taxa_compra"],
            "TaxaVenda": sim["taxa_venda"],
            "RetornoMedioCompra": sim["retorno_medio_compra"],
            "ExpectancyCompra": sim["expectancy_compra"],
            "QtdCompra": sim["qtd_compra"],
            "QtdVenda": sim["qtd_venda"]
        }
    
    except Exception as e:
        logger.error(f"Erro ao processar ativo {tkr}: {e}")
        return None

# ===================== SIDEBAR =====================
st.sidebar.title("🌎 Monitor IA Pro")

st.sidebar.subheader("📈 Índices Mundiais")
indices_data = obter_indices()
for nome, (valor, var) in indices_data.items():
    if valor > 0:
        st.sidebar.metric(nome, f"{valor:,.0f} pts", f"{var:.2f}%")
    else:
        st.sidebar.metric(nome, "N/A", "N/A")

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
    st.markdown("""
    **Selic Alta** → Prejudica ações de crescimento e empresas alavancadas  
    **Inflação Controlada** → Melhora margens de lucro  
    **Dólar Alto** → Beneficia exportadoras | Prejudica importadoras  
    **PIB Crescente** → Favorece consumo, varejo e serviços
    """)
    st.markdown(f"**Último Focus ({macro['Focus_Data']})**")
    st.markdown(f"- Selic 2026: **{macro['Focus_Selic_2026']}**")
    st.markdown(f"- IPCA 2026: **{macro['Focus_IPCA_2026']}**")
    st.markdown(f"- PIB 2026: **{macro['Focus_PIB_2026']}**")

st.sidebar.divider()

# ===================== ANÁLISE FUNDAMENTALISTA =====================
st.sidebar.subheader("📉 Análise Fundamentalista")
with st.sidebar.expander("Indicadores Chave", expanded=True):
    st.markdown("""
    **Eficiência e Rentabilidade:**
    - **ROE** — Retorno sobre o Patrimônio
    - **Margem Líquida** — Lucro Líquido / Receita
    - **Margem EBITDA** — Eficiência operacional

    **Valuation:**
    - **P/L** — Preço sobre Lucro
    - **P/VP** — Preço sobre Valor Patrimonial

    **Endividamento:**
    - **Dívida Líquida / EBITDA** — Nível de alavancagem
    """)

st.sidebar.divider()

# ===================== GESTÃO DE RISCO =====================
st.sidebar.subheader("🛡️ Gestão de Risco e Portfólio")
with st.sidebar.expander("Estratégias Recomendadas", expanded=True):
    st.markdown("""
    - **Alocação de Ativos**: Rebalancear carteira conforme risco e valorização
    - **Análise Técnica**: Definir pontos de entrada e saída com suporte/resistência
    - **Gestão de Risco**: Stop-loss, diversificação e controle de exposição
    """)

st.sidebar.divider()

# ===================== RESTO DO SIDEBAR =====================
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
        * **VALOR ✅**: Grande desconto + bons fundamentos (Value Investing)
        * **COMPRA ✅**: RSI saudável + notícias positivas (Análise Técnica)
        * **VENDA / CARO 🚨**: RSI extremo ou preço acima do justo  
        * **CAUTELA ⚠️**: Divergência entre preço, RSI e sentimento  
        * **Win Rate**: Porcentagem de operações rentáveis
        * **Expectancy**: Ganho esperado por operação (quanto maior, melhor)
        """)
    
    df_resumo = pd.DataFrame(dados_vencedoras)[
        ["Ticker", "Preço", "DY %", "Upside %", "Veredito", "Motivo", 
         "TaxaCompra", "ExpectancyCompra", "QtdCompra"]
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
        
        # Gráfico corrigido
        st.line_chart(acao["Hist"]["Close"], use_container_width=True)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Preço Atual", f"{moeda_simbolo} {acao['Preço']:.2f}")
        c2.metric("P/L", round(acao["P/L"], 2) if acao["P/L"] > 0 else "N/A")
        c3.metric("DY", f"{acao['DY %']:.2f}%")
        c4.metric("Dív.Líq/EBITDA", round(acao["Dívida"], 2) if acao["Dívida"] < 999 else "N/A")
        c5.metric("Graham", f"{moeda_simbolo} {acao['Graham']:.2f}" if not pd.isna(acao['Graham']) else "N/A", 
                  f"{acao['Upside %']:.1f}%" if not pd.isna(acao['Graham']) else "N/A")
        
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
            if acao["Links"]:
                for n in acao["Links"]:
                    st.markdown(f"• [{n['titulo']}]({n['link']})")
            else:
                st.info("Nenhuma manchete encontrada")

else:
    if busca_direta:
        st.warning(f"⚠️ Nenhum resultado encontrado para '{busca_direta}'. Verifique o ticker.")
    elif st.session_state.filtros_ativos:
        st.info("💡 Nenhum ativo atende aos critérios de filtro. Ajuste os valores.")
    else:
        st.info("💡 Use os filtros ou faça uma busca direta para começar.")
