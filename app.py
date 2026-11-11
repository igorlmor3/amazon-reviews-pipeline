"""Raio-X de Avaliações da Amazon com IA — app em Streamlit.

Rodar localmente:  streamlit run app.py
"""
import html
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
PROCESSADOS = RAIZ / "data" / "processed"

# No Streamlit Cloud a chave vem dos "Secrets"; localmente, do .env.
try:
    for chave in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_MODEL_RESERVA"):
        if chave in st.secrets:
            os.environ[chave] = st.secrets[chave]
except FileNotFoundError:
    pass

from busca import buscar, carregar_base, responder  # noqa: E402

LIMITE_PERGUNTAS = 10  # por sessão, para proteger a cota gratuita da API
ASPECTOS = ["qualidade", "sabor", "preco", "entrega", "embalagem", "outro"]
ROTULOS = {"preco": "preço"}
ORDEM_SENT = ["positivo", "neutro", "negativo"]
CORES = {"positivo": "#1ed760", "neutro": "#535353", "negativo": "#f15e6c"}

st.set_page_config(page_title="Raio-X de Avaliações", page_icon="🔎", layout="wide")
st.markdown(f"<style>{(RAIZ / 'assets' / 'estilo.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)


def limpar(texto: str) -> str:
    """Remove tags HTML do texto original e escapa para exibir com segurança."""
    return html.escape(re.sub(r"<[^>]+>", " ", str(texto)))


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def estrelas(n: int) -> str:
    return "★" * int(n) + "☆" * (5 - int(n))


def estilizar(fig):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#b3b3b3", size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=330, bargap=0.35,
        legend=dict(orientation="h", y=1.08, x=0, title=None),
        hoverlabel=dict(font_family="Inter, sans-serif", bgcolor="#000000", font_color="#ffffff", bordercolor="#1ed760"),
    )
    fig.update_xaxes(gridcolor="#2a2a2a", zeroline=False, title_font=dict(color="#727272"))
    fig.update_yaxes(gridcolor="#2a2a2a", zeroline=False, title_font=dict(color="#727272"))
    return fig


@st.cache_data
def carregar_classificadas() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSADOS / "avaliacoes_classificadas.parquet")
    df["aspectos"] = df["aspectos"].apply(list)
    return df


@st.cache_data
def carregar_exemplos() -> pd.DataFrame:
    arquivo = PROCESSADOS / "respostas_exemplo.parquet"
    return pd.read_parquet(arquivo) if arquivo.exists() else pd.DataFrame(columns=["pergunta"])


@st.cache_resource
def carregar_busca():
    return carregar_base()


@st.cache_data(show_spinner=False, max_entries=200)
def perguntar(pergunta: str, k: int, sentimento: str | None, aspecto: str | None):
    base, vetores = carregar_busca()
    encontradas = buscar(pergunta, base, vetores, k=k, sentimento=sentimento, aspecto=aspecto)
    return responder(pergunta, encontradas), encontradas


df = carregar_classificadas()

st.markdown(f"""
<div class="hero">
  <span class="eyebrow">✦ IA generativa aplicada a dados reais</span>
  <h1><span class="destaque">Raio-X</span> de Avaliações da Amazon</h1>
  <p>{fmt(len(df))} avaliações de alimentos classificadas por um LLM por sentimento e aspecto,
  com perguntas em linguagem natural respondidas a partir das próprias avaliações.</p>
  <div class="stack"><span>Gemini API</span><span>Embeddings</span><span>RAG</span>
  <span>Python · pandas</span><span>Streamlit · Plotly</span></div>
</div>""", unsafe_allow_html=True)

aba_painel, aba_perguntas, aba_sobre = st.tabs(["Painel", "Pergunte às avaliações", "Como funciona"])

with aba_painel:
    notas = st.segmented_control("Notas", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5],
                                 selection_mode="multi", format_func=lambda n: f"{n} ★",
                                 label_visibility="collapsed")
    dados = df[df["Score"].isin(notas or [1, 2, 3, 4, 5])]
    por_aspecto = dados.explode("aspectos").rename(columns={"aspectos": "aspecto"})
    por_aspecto["aspecto"] = por_aspecto["aspecto"].replace(ROTULOS)
    negativos = por_aspecto[por_aspecto["sentimento"] == "negativo"]["aspecto"].value_counts()
    divergentes = ((dados["Score"] >= 4) & (dados["sentimento"] == "negativo")) | \
                  ((dados["Score"] <= 2) & (dados["sentimento"] == "positivo"))

    st.markdown(f"""
    <div class="kpis">
      <div class="kpi"><div class="label">Avaliações analisadas</div>
        <div class="value">{fmt(len(dados))}</div><div class="hint">amostra estratificada por nota</div></div>
      <div class="kpi accent"><div class="label">Sentimento negativo</div>
        <div class="value">{(dados['sentimento'] == 'negativo').mean():.0%}</div><div class="hint">pelo texto, não pela nota</div></div>
      <div class="kpi"><div class="label">Aspecto mais criticado</div>
        <div class="value">{negativos.index[0].capitalize() if len(negativos) else '—'}</div>
        <div class="hint">{negativos.iloc[0] if len(negativos) else 0} menções negativas</div></div>
      <div class="kpi"><div class="label">Nota × texto divergentes</div>
        <div class="value">{divergentes.mean():.0%}</div><div class="hint">ex.: 5 estrelas com texto negativo</div></div>
    </div>""", unsafe_allow_html=True)

    esq, dir_ = st.columns(2, gap="medium")
    with esq, st.container(border=True):
        st.markdown('<p class="section-title">Sobre o que as pessoas falam</p>'
                    '<p class="section-sub">Menções por aspecto e sentimento</p>', unsafe_allow_html=True)
        contagem = por_aspecto.groupby(["aspecto", "sentimento"]).size().reset_index(name="avaliações")
        ordem = contagem.groupby("aspecto")["avaliações"].sum().sort_values().index.tolist()
        fig = px.bar(contagem, x="avaliações", y="aspecto", color="sentimento", orientation="h",
                     color_discrete_map=CORES,
                     category_orders={"sentimento": ORDEM_SENT, "aspecto": ordem[::-1]})
        st.plotly_chart(estilizar(fig).update_yaxes(title=None), width="stretch",
                        config={"displayModeBar": False})
    with dir_, st.container(border=True):
        st.markdown('<p class="section-title">O que o texto diz em cada nota</p>'
                    '<p class="section-sub">Sentimento identificado pelo LLM por número de estrelas</p>',
                    unsafe_allow_html=True)
        por_nota = dados.groupby(["Score", "sentimento"]).size().reset_index(name="avaliações")
        fig = px.bar(por_nota, x="Score", y="avaliações", color="sentimento", barmode="stack",
                     color_discrete_map=CORES, category_orders={"sentimento": ORDEM_SENT})
        st.plotly_chart(estilizar(fig).update_xaxes(title="estrelas", dtick=1), width="stretch",
                        config={"displayModeBar": False})

    with st.container(border=True):
        st.markdown('<p class="section-title">Principais motivos de insatisfação</p>'
                    '<p class="section-sub">Resumo do motivo gerado pelo LLM para cada avaliação negativa</p>',
                    unsafe_allow_html=True)
        escolha = st.segmented_control("Aspecto", ASPECTOS, default="sabor",
                                       format_func=lambda a: ROTULOS.get(a, a).capitalize(),
                                       label_visibility="collapsed") or "sabor"
        motivos = por_aspecto[(por_aspecto["aspecto"] == ROTULOS.get(escolha, escolha)) &
                              (por_aspecto["sentimento"] == "negativo")]
        st.dataframe(
            motivos[["Score", "Summary", "motivo"]].rename(
                columns={"Score": "Nota", "Summary": "Título original", "motivo": "Motivo (resumo do LLM)"}),
            hide_index=True, width="stretch", height=320,
            column_config={"Nota": st.column_config.NumberColumn(format="%d ★", width="small")},
        )

with aba_perguntas:
    if not (PROCESSADOS / "embeddings.npz").exists():
        st.info("Os embeddings ainda não foram gerados. Rode `python src/03_embeddings.py`.")
    else:
        exemplos = carregar_exemplos()
        with st.container(border=True):
            st.markdown('<p class="section-title">Pergunte às avaliações</p>'
                        '<p class="section-sub">A resposta usa só as avaliações mais relevantes e cita as fontes. '
                        'Clique num exemplo ou escreva a sua pergunta.</p>', unsafe_allow_html=True)
            exemplo = st.pills("Exemplos", exemplos["pergunta"].tolist(), label_visibility="collapsed",
                               key="exemplo") if len(exemplos) else None
            pergunta = st.text_input("Sua pergunta", placeholder="Ex.: O que as pessoas acham dos chás verdes?",
                                     label_visibility="collapsed")
            c1, c2, c3 = st.columns([1.2, 1, 1])
            k = c1.slider("Avaliações usadas como fonte", 5, 20, 10)
            sentimento = c2.selectbox("Sentimento", ["todos"] + ORDEM_SENT)
            aspecto = c3.selectbox("Aspecto", ["todos"] + ASPECTOS, format_func=lambda a: ROTULOS.get(a, a))
            clicou = st.button("Perguntar", type="primary", disabled=not pergunta.strip())

        resposta, fontes = None, None
        if clicou:
            if st.session_state.setdefault("perguntas_feitas", 0) >= LIMITE_PERGUNTAS:
                st.warning("Limite de perguntas desta sessão atingido (a demo usa uma cota gratuita).")
                st.stop()
            st.session_state["perguntas_feitas"] += 1
            with st.spinner("Buscando avaliações e gerando a resposta..."):
                try:
                    resposta, fontes = perguntar(pergunta.strip(), k,
                                                 None if sentimento == "todos" else sentimento,
                                                 None if aspecto == "todos" else aspecto)
                except Exception:
                    st.error("O Gemini está sobrecarregado agora e não respondeu. Tente de novo em alguns "
                             "instantes ou clique num dos exemplos, que já têm resposta pronta.")
                    st.stop()
        elif exemplo:
            # Resposta gerada antes (src/05_respostas_exemplo.py): aparece na hora, sem chamar a API.
            linha = exemplos[exemplos["pergunta"] == exemplo].iloc[0]
            pares = json.loads(linha["fontes"])
            base_busca, _ = carregar_busca()
            similaridade = dict(pares)
            fontes = base_busca[base_busca["Id"].isin(similaridade)].copy()
            fontes["similaridade"] = fontes["Id"].map(similaridade)
            fontes = fontes.sort_values("similaridade", ascending=False)
            resposta = linha["resposta"]

        if resposta is not None:
            st.markdown('<p class="section-title" style="margin-top:1rem">Resposta</p>', unsafe_allow_html=True)
            with st.container(border=True, key="resposta"):
                st.markdown(resposta)
            st.markdown(f'<p class="section-title" style="margin-top:1.25rem">Fontes '
                        f'<span class="badge id">{len(fontes)} avaliações</span></p>', unsafe_allow_html=True)
            cartoes = "".join(
                f'<div class="source"><div class="top"><span class="title">{limpar(r.Summary)}</span>'
                f'<span><span class="badge id">#{r.Id}</span> '
                f'<span class="badge {getattr(r, "sentimento", "neutro")}">{getattr(r, "sentimento", "")}</span></span></div>'
                f'<div class="stars">{estrelas(r.Score)} <span style="color:#727272;font-size:.75rem">'
                f'· similaridade {r.similaridade:.2f}</span></div>'
                f'<div class="text">{limpar(r.Text)[:420]}…</div></div>'
                for r in fontes.itertuples()
            )
            st.markdown(cartoes, unsafe_allow_html=True)

with aba_sobre, st.container(border=True):
    st.markdown("""
#### O problema
Milhares de avaliações em texto livre são difíceis de ler e resumir. Este projeto transforma esse
texto em análise consultável: **o que incomoda os clientes, em quais aspectos e com que frequência**.

#### Como funciona
1. **Amostra** — da base pública *Amazon Fine Food Reviews* (~568 mil avaliações), removi duplicatas
   e sorteei 3.000 avaliações, 600 por nota.
2. **Classificação com LLM** — o Gemini classifica cada avaliação por sentimento e aspectos
   (qualidade, sabor, preço, entrega, embalagem) e resume o motivo, com saída em JSON validada por schema.
3. **Busca semântica (RAG)** — cada avaliação vira um embedding com um modelo multilíngue local
   (perguntas em português encontram avaliações em inglês); a pergunta é comparada por
   similaridade de cosseno e o LLM responde usando só as avaliações encontradas, citando as fontes.

#### Stack
Python · pandas · NumPy · Gemini API · FastEmbed (embeddings locais) · Streamlit · Plotly
""")
