importar streamlit como st
importar yfinance como yf
importar pandas como pd
importar numpy como np
tempo de importação
importar feedparser
de streamlit_autorefresh importar st_autorefresh

# ===================== CONFIGURAÇÕES =====================
st.set_page_config(page_title="Monitor IA Pro", layout="wide")
st_autorefresh(intervalo=300 * 1000, chave="data_refresh")

se "filtros_ativos" não estiver em st.session_state:
    st.session_state.filtros_ativos = Falso

def ativar_filtros():
    st.session_state.filtros_ativos = Verdadeiro

# ===================== FUNÇÕES TÉCNICAS =====================
def calcular_graham(lpa, vpa):
    se lpa > 0 e vpa > 0:
        retornar np.sqrt(22,5 * lpa * vpa)
    retornar 0,0

def calcular_rsi_series(fechar: pd.Series, janela: int = 14) -> pd.Series:
    se len(fechar) < janela:
        retornar pd.Série([50,0] * len(fechar), índice=fechar.índice)
    delta = fechar.diff()
    ganho = delta.onde(delta > 0, 0).rolando(janela=janela).média()
    perda = -delta.onde(delta < 0, 0).rolando(janela=janela).média()
    rs = ganho / perda
    rsi = 100 - (100 / (1 + rs))
    retornar rsi

def calcular_rsi(dados, janela: int = 14):
    se len(dados) < janela:
        retornar 50,0
    rsi_series = calcular_rsi_series(dados, janela)
    retornar float(rsi_series.iloc[-1])

def calcular_score_value(info):
    pontuação = 0
    se 0 < info.get("trailingPE", 99) < 15: pontuação += 1
    se 0 < info.get("priceToBook", 99) < 1,5: pontuação += 1
    se (info.get("dividendYield", 0) ou 0) * 100 > 5: pontuação += 1
    se (info.get("operatingMargins", 0) ou 0) > 0,1:pontuação += 1
    pontuação de retorno

# ===================== SIMULAÇÃO VETORIZADA MELHORADA =====================
def simular_performance_historica(hist):
    se len(hist) < 300:
        retornar {
            "taxa_compra": 0,0, "taxa_venda": 0,0,
            "retorno_medio_compra": 0,0, "retorno_medio_venda": 0,0,
            "expectativa_compra": 0,0, "expectativa_venda": 0,0,
            "qtd_compra": 0, "qtd_venda": 0
        }

    fechar = hist["Fechar"].copiar()
    rsi = calcular_rsi_series(fechar)
    sma200 = close.rolling(janela=200).mean()
    exp12 = close.ewm(span=12, ajuste=Falso).mean()
    exp26 = close.ewm(span=26, ajuste=Falso).mean()
    macd = exp12 - exp26
    sinal_macd = macd.ewm(span=9, ajuste=Falso).mean()
    retorno_15d = fechar.deslocamento(-15) /fechar - 1

    buy_mask = (rsi < 35) & (fechar > sma200) & (macd > sinal_macd) & retorno_15d.notna()
    sell_mask = (rsi > 70) & ((fechar < sma200) | (macd < sinal_macd)) & retorno_15d.notna()

    # Compras
    se buy_mask.any():
        ret_buy = retorno_15d[comprar_máscara]
        qtd_c = int(comprar_máscara.soma())
        taxa_c = (ret_buy > 0).média() * 100
        ret_med_c = ret_buy.mean() * 100
        avg_win = ret_buy[ret_buy > 0].mean() se (ret_buy > 0).any() senão 0
        avg_loss = abs(ret_buy[ret_buy < 0].mean()) se (ret_buy < 0).any() senão 0
        exp_c = (taxa_c/100 * média_vitória) - ((1 - taxa_c/100) * média_perda) * 100
    outro:
        taxa_c = ret_med_c = exp_c = 0,0
        qtd_c = 0

    # Vendas
    se sell_mask.any():ret_sell = retorno_15d[vender_máscara]
        qtd_v = int(vender_máscara.soma())
        taxa_v = (ret_sell < 0).média() * 100
        ret_med_v = ret_sell.mean() * 100
        avg_win_v = abs(ret_sell[ret_sell < 0].mean()) se (ret_sell < 0).any() senão 0
        avg_loss_v = ret_sell[ret_sell > 0].mean() se (ret_sell > 0).any() senão 0
        exp_v = (taxa_v/100 * avg_win_v) - ((1 - taxa_v/100) * avg_loss_v) * 100
    outro:
        taxa_v = ret_med_v = exp_v = 0,0
        qtd_v = 0

    retornar {
        "taxa_compra": taxa_c,
        "taxa_venda": taxa_v,
        "retorno_medio_compra": ret_med_c,
        "retorno_medio_venda": ret_med_v,
        "expectativa_compra": exp_c,
        "expectativa_venda": exp_v,
        "qtd_compra": qtd_c,
        "qtd_venda":qtd_v
    }

# ===================== MACROECONÔMICOS =====================
@st.cache_data(ttl=1800, show_spinner=Falso)
def obter_macro():
    macro = {}
    tente:
        selic_data = yf.Ticker("^SELIC").history(período="5d")
        macro["Selic"] = float(selic_data["Fechar"].iloc[-1]) se não selic_data.empty senão 14,75
        macro["Dolar"] = yf.Ticker("USDBRL=X").fast_info.last_price
        macro["IPCA_12m"] = 4,14
    exceto:
        macro["Sélico"] = 14,75
        macro["Dolar"] = 4,99
        macro["IPCA_12m"] = 4,14

    macro["Focus_Data"] = "13/04/2026"
    macro["Focus_Selic_2026"] = "12,50%"
    macro["Focus_IPCA_2026"] = "4,36%"
    macro["Focus_PIB_2026"] = "1.85%"

    retornar macro

# ===================== CACHE DE MERCADO =====================
@st.cache_data(ttl=300, show_spinner=Falso)
def obter_indices():
    índices = {"Ibovespa": "^BVSP", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}
    resultados = {}
    para nome, ticker em indices.items():
        tente:
            dados = yf.Ticker(ticker).history(período="2d")
            se não data.empty e len(data) >= 2:
                atual = data["Fechar"].iloc[-1]
                anterior = dados["Fechar"].iloc[-2]
                variacao = ((atual / anterior) - 1) * 100
                resultados[nome] = (atual, variacao)
            outro:
                resultados[nome] = (0,0, 0,0)
        exceto:
            resultados[nome] = (0,0, 0.0)
    retornar resultados


@st.cache_data(ttl=90, show_spinner=Falso)
def obter_cambio():
    moedas = {"Dólar": "USDBRL=X", "Euro": "EURBRL=X", "Libra": "GBPBRL=X"}
    resultados = {}
    para nome, ticker em moedas.items():
        tente:
            t = yf.Ticker(ticker)
            dados = t.history(período="2d")
            se não data.empty e len(data) >= 2:
                atual = float(dados["Fechar"].iloc[-1])
                anterior = float(dados["Fechar"].iloc[-2])
                variacao = ((atual / anterior) - 1) * 100
            outro:
                atual = float(t.fast_info.last_price)
                variacao = 0,0
            resultados[nome] = (atual, variacao)
        exceto:
            resultados[nome] = (0,0, 0.0)

    # Bitcoin em Real
    btc_real = 0,0
    tente:
        t = yf.Ticker("BTC-BRL")
        dados = t.history(período="2d")
        atual = float(data["Fechar"].iloc[-1]) se não data.empty e len(data) >= 2 senão float(t.fast_info.last_price)
        se atual > 100000:
            btc_real = atual
    exceto:
        passar
    se btc_real < 100000:
        tente:
            btc_usd = float(yf.Ticker("BTC-USD").fast_info.last_price)
            dolar_brl = resultados.get("Dólar", (4,99, 0))[0]
            btc_real = btc_usd * dolar_brl
        exceto:
            passar
    resultados["Bitcoin"] = (btc_real, 0,0)
    retornar resultados


# ===================== BAIXE EM BATCH =====================
@st.cache_data(ttl=600, show_spinner=Falso)
def obter_dados_batch(tickers, mercado):se não forem tickers:
        retornar {}, {}
    tickers_yf = [t + ".SA" se mercado == "Brasil" e não t.endswith(".SA") caso contrário t para t em tickers]
    hist_multi = yf.download(tickers_yf, period="5y", group_by="ticker", auto_adjust=True, progress=False, threads=True)
    info_dict = {}
    hist_dict = {}
    para i, t_orig em enumerate(tickers):
        t_yf = tickers_yf[i]
        tente:
            info_dict[t_orig] = yf.Ticker(t_yf).info
            se len(tickers) == 1:
                hist_dict[t_orig] = hist_multi
            outro:
                hist_dict[t_orig] = hist_multi[t_yf] se t_yf em hist_multi.columns.get_level_values(0) senão pd.DataFrame()
        exceto Exceção:
            info_dict[t_orig] = {}
            hist_dict[t_orig] = pd.DataFrame()
    retornar info_dict,hist_dict


# ===================== PROCESSAMENTO CENTRAL =====================
def processar_ativo(tkr, info, hist, estratégia_ativa, filtros_ativos,
                    f_pl, f_pvp, f_dy, f_div_ebitda, busca_direta, mercado):
    se hist.info vazio ou não:
        retornar Nenhum

    pl = info.get("trailingPE", 0) ou 0
    pvp = info.get("priceToBook", 0) ou 0
    dy = (info.get("dividendYield", 0) ou 0) * 100
    ebitda = info.get("ebitda", 1) ou 1
    div_liq = info.get("dívida total", 0) ou 0
    dinheiro = info.get("totalCash", 0) ou 0
    div_e = (div_liq - dinheiro) / ebitda se ebitda!= 0 senão 999,0

    lpa = info.get("trailingEps", 0) ou 0
    vpa = info.get("bookValue", 0) ou 0
    p_justo = calcular_graham(lpa, vpa)
    p_atual = float(hist["Fechar"].iloc[-1])
    lado positivo = ((p_justo / p_atual) - 1) * 100 se p_justo > 0 senão 0,0

    se não busca_direta e filtros_ativos:
        caso contrário (pl <= f_pl e pvp <= f_pvp e dy >= f_dy e div_e <= f_div_ebitda):
            retornar Nenhum

    # Notícias
    noticias_texto = ""
    lista_links = []
    tente:
        lang = "pt-BR" se mercado == "Brasil" senão "en-US"
        url = f"https://news.google.com/rss/search?q={tkr}&hl={lang}"
        feed = feedparser.parse(url)
        para entrada em feed.entries[:5]:
            título = entrada.título.inferior()
            noticias_texto += título + " "
            lista_links.append({"titulo": entrada.título, "link": entrada.link})
    exceto:
        passar

    score_p = sum(noticias_texto.count(w) para w em ["alta", "lucro", "compra", "subiu", "dividendo", "lucro", "comprar"])
    score_n = sum(noticias_texto.count(w) para w em ["queda", "prejuízo", "venda", "caiu", "risco", "perda", "vender"])

    rsi_val = calcular_rsi(hist["Fechar"])
    valor_pontuação = valor_pontuação_calcular(informações)

    sim = simular_performance_historica(hist)

    # ===================== LÓGICA DE ESTRATÉGIAS MULTIPLAS =====================
    se estratégia_ativa == "Investimento em Valor (Graham/Buffett)":
        se upside > 20 e score_value >= 3:
            veredito, cor = "VALOR ✅", "sucesso"
            motivo_detalhe = f"Forte margem de segurança ({upside:.1f}%) e bons fundamentos (Pontuação: {score_value}/4)."
        elif superior < 0:
            veredito, cor = "CARO 🚨", "erro"
            motivo_detalhe = "Preço acima do valor intrínseco de Graham"
        outro:
            veredito, cor = "NEUTRO ⚖️", "aviso"
            motivo_detalhe = "Ativo próximo ao preço justo"

    elif estratégia_ativa == "Investimento em Dividendos":
        se dy >= 6,0 e div_e < 3,0:
            veredito, cor = "RENDA ✅", "sucesso"
            motivo_detalhe = f"Alto Dividend Yield ({dy:.2f}%) e dívida controlada ({div_e:.1f}x)."
        elif dy < 3,0:
            veredito, cor = "BAIXO DY 🚨", "erro"
            motivo_detalhe = f"Rendimento de dividendos fraco para a estratégia ({dy:.2f}%)."
        outro:
            veredito,cor = "NEUTRO ⚖️", "aviso"
            motivo_detalhe = f"DY mediano ({dy:.2f}%), monitorar a consistência."

    elif estratégia_ativa == "Investimento em Crescimento":
        rev_growth = (info.get("revenueGrowth", 0) ou 0) * 100
        se rev_growth > 15,0:
            veredito, cor = "CRESCIMENTO ✅", "sucesso"
            motivo_detalhe = f"Forte aceleração de recepção ({rev_growth:.1f}%) ano a ano."
        elif rev_growth < 0:
            veredito, cor = "DECLÍNIO 🚨", "erro"
            motivo_detalhe = "Empresa apresentando retração nas receitas operacionais"
        outro:
            veredito, cor = "NEUTRO ⚖️", "aviso"
            motivo_detalhe = f"Crescimento de receita brando ou estagnado ({rev_growth:.1f}%)."

    elif estratégia_ativa == "Comprar e Manter":
        se score_value >= 3 e div_e < 2,5 e pl < 25:
            veredito, cor = "ACUMULAR ✅", "sucesso"
            motivo_detalhe = "Excelentes fundamentos de longo prazo e baixo risco de endividamento"
        elif div_e > 5,0 ou pl > 50:
            veredito, cor = "RISCO 🚨", "erro"
            motivo_detalhe = "Múltiplos armazenados ou endividamento excessivo para carregar longo"
        outro:
            veredito, cor = "MANTER ⚖️", "aviso"
            motivo_detalhe = "Fundamentos dentro da mídia, sem sinais de alerta graves"

    elif estratégia_ativa == "Negociação de Posições":
        tente:sma200_atual = hist["Fechar"].rolling(janela=200).mean().iloc[-1] se len(hist) >= 200 senão 0
            sma50_atual = hist["Fechar"].rolling(janela=50).mean().iloc[-1] se len(hist) >= 50 senão 0
        exceto:
            sma200_igual, sma50_igual = 0, 0
            
        se p_atual > sma50_atual e sma50_atual > sma200_atual:
            veredito, cor = "TENDÊNCIA ALTA ✅", "sucesso"
            motivo_detalhe = "Preço suportado acima das médias móveis de 50 e 200 dias (Tendência ascendente)"
        elif p_atual < sma50_atual e sma50_atual < sma200_atual:
            veredito, cor = "TENDÊNCIA BAIXA 🚨", "erro"
            motivo_detalhe = "Preço baixo das principais médias móveis (Tendência de baixa)"
        outro:
            veredito, cor = "LATERAL ⚖️", "aviso"
            motivo_detalhe = "Ativo cruzando médias, sem tendência direcional confirmada"

    caso contrário: # Análise Técnica (Comerciante)
        se rsi_val > 70 e score_n > score_p:
            veredito, cor = "VENDA 🚨", "erro"
            motivo_detalhe = f"RSI alto ({rsi_val:.1f}) e notícias negativas."
        elif rsi_val > 75:
            veredito, cor = "VENDA 🚨", "erro"
            motivo_detalhe = f"RSI em nível extremo ({rsi_val:.1f})."
        elif pontuação_p > pontuação_n e rsi_val < 65:
            veredito, cor = "COMPRA ✅", "sucesso"
            motivo_detalhe = f"Notícias positivas e RSI saudita ({rsi_val:.1f})."
        elif score_n > score_p ou rsi_val > 70:
            veredito,cor = "CAUTELA ⚠️", "erro"
            lista_motivos = []
            se rsi_val > 70: lista_motivos.append(f"RSI alto ({rsi_val:.1f})")
            se pontuação_n > pontuação_p: lista_motivos.append("Sentimento negativo")
            motivo_detalhe = " | ".join(lista_motivos)
        outro:
            veredito, cor = "NEUTRO ⚖️", "aviso"
            motivo_detalhe = "Indicadores em equilíbrio no curto prazo"

    retornar {
        "Ticker": tkr,
        "Empresa": info.get("shortName", tkr),
        "Preço": p_atual,
        "P/L": pl,
        "DY %": dy,
        "Dívia": div_e,
        "Graham": p_justo,
        "% de alta": alta,
        "Veredito": veredito,"Cor": cor,
        "Motivo": motivo_detalhe,
        "RSI": rsi_val,
        "Hist": hist,
        "Links": lista_links,
        "ValueScore": pontuação_valor,

        "TaxaCompra": sim["taxa_compra"],
        "TaxaVenda": sim["taxa_venda"],
        "RetornoMedioCompra": sim["retorno_medio_compra"],
        "ExpectancyCompra": sim["expectancy_compra"],
        "QtdCompra": sim["qtd_compra"],
        "QtdVenda": sim["qtd_venda"]
    }


# ===================== BARRA LATERAL =====================
st.sidebar.title("🌎 Monitor IA Pro")

st.sidebar.subheader("📈 Índices Mundiais")
índices_dados = obter_índices()
para nome, (valor, var) em indices_data.itens():
    st.sidebar.metric(nome, f"{valor:,.0f} pts", f"{var:.2f}%")

barra lateral St.divider()

st.sidebar.subheader("💱 Câmbio em Tempo Real")
cambio = obter_cambio()
col1, col2 = st.sidebar.columns(2)
col1.metric("Dólar", f"R$ {cambio['Dólar'][0]:.2f}", f"{cambio['Dólar'][1]:.2f}%")
col2.metric("Euro", f"R$ {cambio['Euro'][0]:.2f}", f"{cambio['Euro'][1]:.2f}%")
col3, col4 = st.sidebar.columns(2)
col3.metric("Libra", f"R$ {cambio['Libra'][0]:.2f}", f"{cambio['Libra'][1]:.2f}%")
col4.metric("Bitcoin", f"R$ {cambio['Bitcoin'][0]:,.0f}", f"{cambio['Bitcoin'][1]:.2f}%")

barra lateral St.divider()

# ===================== MACRO &CENÁRIO ====================
st.sidebar.subheader("📊 Macro & Cenário")
macro = obter_macro()

st.sidebar.metric("Selic Atual", f"{macro['Selic']:.2f}%")
st.sidebar.metric("IPCA 12m", f"{macro['IPCA_12m']:.2f}%")
st.sidebar.metric("Dólar", f"R$ {macro['Dolar']:.2f}")

com st.sidebar.expander("📌 Impacto no Mercado", expandido=Verdadeiro):
    st.markdown("""
    **Alta Sélica** → Prejudica ações de crescimento e empresas alavancadas  
    **Inflação Controlada** → Minha margens de lucro  
    **Dólar Alto** → Beneficia exportadoras | Prejudica importadoras  
    **PIB Crescente** → Favorece consumo, varejo e serviços
    """)
    st.markdown(f"**Último Focus ({macro['Focus_Data']})**")
    São.markdown(f"- Selic 2026: **{macro['Focus_Selic_2026']}**")
    st.markdown(f"- IPCA 2026: **{macro['Focus_IPCA_2026']}**")
    st.markdown(f"- PIB 2026: **{macro['Focus_PIB_2026']}**")

barra lateral St.divider()

# ===================== ANÁLISE FUNDAMENTALISTA =====================
st.sidebar.subheader("📉 Análise Fundamentalista")
com st.sidebar.expander("Indicadores Chave", expandido=Verdadeiro):
    st.markdown("""
    **Eficiência e Rentabilidade:**
    - **ROE** — Retorno sobre o Patrimônio
    - **Margem Líquida** — Lucro Líquido / Recebida
    - **Margem EBITDA** — Eficiência operacional

    **Avaliação:**
    - **P/L** — Preço sobre Lucro
    - **P/VP** — Preço sobre Valor Patrimonial

    **Endividamento:**
    - **Dívia Líquida / EBITDA** — Nível de alavancagem
    """)

barra lateral St.divider()

# ===================== GESTÃO DE RISCO =====================
st.sidebar.subheader("🛡️ Gestão de Risco e Portfólio")
com st.sidebar.expander("Estratégias Recomendadas", expandido=Verdadeiro):
    st.markdown("""
    - **Alocação de Ativos**: Rebalancear carta conforme risco e valorização
    - **Análise Técnica**: Definir pontos de entrada e saúde com suporte/resistência
    - **Gestão de Risco**: Stop-loss, diversificação e controle de exposição
    """)

barra lateral St.divider()

# ===================== RESTAURAR A BARRA LATERAL =====================
mercado_selecionado = st.sidebar.radio(
    "Escola o Mercado:", ["Brasil", "EUA"],on_change=ativar_filtros
)

# Adicionado como opções de Análise pedidas!
estratégia_ativa = st.sidebar.selectbox(
    "Foco da Análise:", [
        "Investimento de valor (Graham/Buffett)",
        "Análise Técnica (Comerciante)",
        "Investimento em Crescimento",
        "Comprar e Manter",
        "Investimento em dividendos",
        "Negociação de Posições"
    ]
)

busca_direta = st.sidebar.text_input(f"🔍 Busca Rápida ({mercado_selecionado}):").upper().strip()

com st.sidebar.expander("📊 Filtros de Valuation", expandido=Verdadeiro):
    f_pl = st.slider("P/L Máximo", 0,0, 50,0, 50,0, passo=0,5, on_change=ativar_filtros)
    f_pvp = st.slider("P/VP Máximo", 0,0, 10,0, 10,0, passo=0,1, on_change=ativar_filtros)
    f_dy = st.slider("DY Mínimo (%)", 0,0, 20,0, 0,0,passo=0,5, on_change=ativar_filtros)
    f_div_ebitda = st.slider("Dív.Líq/EBITDA Máximo", 0,0, 15,0, 15,0, passo=0,5, on_change=ativar_filtros)

se st.sidebar.button("Redefinir Filtros"):
    st.session_state.filtros_ativos = Falso
    st.rerun()

# ===================== LISTA DE ATIVOS =====================
se mercado_selecionado == "Brasil":
    lista_base = ["PETR4", "VALE3", "ITUB4", "BBAS3", "BBDC4", "SANB11", "B3SA3", "EGIE3", "TRPL4", "TAEE11", "SAPR11", "CPLE6", "ELET3", "CMIG4", "SBSP3", "ABEV3", "WEGE3", "RADL3", "RENT3", "MGLU3", "LREN3", "RAIZ4", "VBBR3", "SUZB3", "KLBN11", "GOAU4", "CSNA3", "PRIO3", "JBSS3", "BRFS3", "GGBR4", "HAPV3", "RDOR3"]
    moeda_simbolo = "R$"
outro:
    lista_base = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "DIS", "KO", "PEP", "MCD", "NKE", "WMT", "JPM", "V", "MA", "BAC", "PYPL", "PFE", "JNJ", "PG", "COST", "ORCL"]
    moeda_simbolo = "US$"

# ===================== CABEÇALHO =====================
st.title(f"🤖 Monitor IA - {mercado_selecionado}")
st.caption(f"Atualização: {time.strftime('%H:%M:%S')} | Local: Blumenau/SC")

# ===================== PROCESSAMENTO PRINCIPAL =====================
tickers_para_processar = [busca_direta] se busca_direta senão lista_base
dados_vencedoras = []

se tickers_para_processar:
    com st.spinner("📡 Baixando dados em lote..."):
        infos, hists = obter_dados_batch(tickers_para_processar, mercado_selecionado)

    para tkr em tickers_para_processar:
        info = infos.get(tkr, {})
        hist = hists.get(tkr, pd.DataFrame())
        resultado = processar_ativo(
            tkr, info, hist, estratégia_ativa, st.session_state.filtros_ativos,
            f_pl, f_pvp, f_dy, f_div_ebitda, busca_direta, mercado_selecionado
        )
        se resultado:
            dados_vencedoras.append(resultado)

# ===================== INTERFACE =====================
se dados_vencedoras:
    São.subheader(f"🏆 Ranking de Oportunidades - Estratégia: {estrategia_ativa}")
    
    com st.expander("📌 Legenda de Sinais e Vereditos"):
        st.markdown("""
        * **VALOR ✅**: Grande desconto + bons fundamentos  
        * **COMPRA ✅**: RSI saudita + notícias positivas  
        * **VENDA / CARO 🚨**: RSI extremo ou preto acima do justo  
        * **CAUTELA ⚠️**: Divergência entre preço, RSI e sentimento  
        * **Expectativa**: Ganho esperado por operação (quanto maior, melhor)
        """)

    df_resumo = pd.DataFrame(dados_vencedoras)[
        ["Ticker", "Preço", "DY %", "Upside %", "Veredito", "Motivo", 
         "TaxaCompra", "ExpectativaCompra", "QtdCompra"]
    ]

    São.quadro de dados(
        df_resumo.sort_values(by="% ascendente", ascendente=Falso),
        use_container_width=Verdadeiro,
        ocultar_índice=Verdadeiro,
        coluna_config={
            "Veredito": st.column_config.TextColumn("Veredito"),
            "Motivo": st.column_config.TextColumn("Motivo da IA", width="medium"),
            "TaxaCompra": st.column_config.NumberColumn("Compra de taxa de ganhos", format="%.1f%%"),
            "ExpectancyCompra": st.column_config.NumberColumn("Expectancy Compra", format="%.2f%%"),
            "QtdCompra": st.column_config.NumberColumn("Sinais"),
        },
    )

    para acao em dados_vencedoras:
        st.divider()
        col_tit, col_ver, col_acc_c, col_acc_v = colunas st([3, 1, 1,1])
        col_tit.header(f"🏢 {acao['Empresa']} ({acao['Ticker']})")
        
        col_acc_c.metric("Assert. Compra", f"{acao['TaxaCompra']:.1f}%", f"{acao['QtdCompra']} sinais")
        col_acc_v.metric("Afirmar. Venda", f"{acao['TaxaVenda']:.1f}%", f"{acao['QtdVenda']} sinais")

        se acao["Cor"] == "sucesso":
            col_ver.success(f"**{acao['Veredito']}**")
        elif acao["Cor"] == "erro":
            col_ver.error(f"**{acao['Veredito']}**")
        outro:
            col_ver.warning(f"**{acao['Veredito']}**")

        st.line_chart(acao["Hist"]["Fechar"], use_container_width=True)

        c1, c2,c3, c4, c5 = st.columns(5)
        c1.metric("Preço Atual", f"{moeda_simbolo} {acao['Preço']:.2f}")
        c2.metric("P/L", round(acao["P/L"], 2))
        c3.metric("DY", f"{acao['DY %']:.2f}%")
        c4.metric("Dív.Líq/EBITDA", round(acao["Dívida"], 2))
        c5.metric("Graham", f"{moeda_simbolo} {acao['Graham']:.2f}", f"{acao['Upside %']:.1f}%")

        com st.expander(f"📊 Detalhes: {acao['Ticker']}"):
            col_inf1, col_inf2 = st.columns(2)
            com col_inf1:
                st.write(f"**Fundamentos (Pontuação de Valor):** {acao['Pontuação de Valor']}/4")
                st.progress(acao["ValueScore"] / 4)
                st.write(f"📈 RSI:{acau['RSI']:.2f}")
                st.write(f"**Compra de Expectativa:** {acao['Compra de Expectativa']:.2f}%")
            com col_inf2:
                st.write(f"📝 **Motivo IA:** {acao['Motivo']}")
            st.markdown("---")
            st.markdown("**Últimas Manchetes:**")
            para n em acao["Links"]:
                st.markdown(f"• [{n['titulo']}]({n['link']})")

outro:
    st.info("💡 Use os filtros ou faça uma busca direta para comer.")

Quero que me fale os erros desse código, é possível melhorar ele sem perder qualidade?
Não quero que aposente funções ou dores de funcionalidade.)
            com col_inf2:
                st.write(f"📝 **Motivo IA:** {acao['Motivo']}")
            st.markdown("---")
            st.markdown("**Últimas Manchetes:**")
            para n em acao["Links"]:
                st.markdown(f"• [{n['titulo']}]({n['link']})")

outro:
    st.info("💡 Use os filtros ou faça uma busca direta para comer.")

Quero que me fale os erros desse código, é possível melhorar ele sem perder qualidade?
Não quero que aposente funções ou dores de funcionalidade.)
            com col_inf2:
                st.write(f"📝 **Motivo IA:** {acao['Motivo']}")
            st.markdown("---")
            st.markdown("**Últimas Manchetes:**")
            para n em acao["Links"]:
                st.markdown(f"• [{n['titulo']}]({n['link']})")

outro:
    st.info("💡 Use os filtros ou faça uma busca direta para comer.")

Quero que me fale os erros desse código, é possível melhorar ele sem perder qualidade?
Não quero que aposente funções ou dores de funcionalidade.• [{n['titulo']}]({n['link']})")

outro:
    st.info("💡 Use os filtros ou faça uma busca direta para comer.")
