"""Busca semântica e resposta em linguagem natural sobre as avaliações (RAG).

1. A pergunta vira um embedding.
2. Encontramos as avaliações mais parecidas em significado (similaridade de cosseno).
3. O LLM responde usando só essas avaliações, citando de onde tirou cada ponto.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from google.genai import types

from gemini import MODELO, MODELO_EMBEDDING, cliente, com_tentativas

RAIZ = Path(__file__).resolve().parents[1]
PROCESSADOS = RAIZ / "data" / "processed"
DIMENSOES = 768

INSTRUCOES = """Você é um analista de experiência do cliente. Responda à pergunta em
português usando APENAS as avaliações fornecidas. Resuma os padrões (o que se repete),
cite exemplos concretos e indique a fonte de cada ponto no formato [#Id].
Se as avaliações não forem suficientes para responder, diga isso claramente."""


def carregar_base() -> tuple[pd.DataFrame, np.ndarray]:
    """Carrega a amostra (com as classificações, se já existirem) e os embeddings."""
    classificadas = PROCESSADOS / "avaliacoes_classificadas.parquet"
    base = pd.read_parquet(classificadas if classificadas.exists() else PROCESSADOS / "amostra.parquet")
    emb = np.load(PROCESSADOS / "embeddings.npz")
    posicao = {int(i): n for n, i in enumerate(emb["ids"])}
    base = base[base["Id"].isin(posicao)].reset_index(drop=True)
    vetores = emb["vetores"][[posicao[int(i)] for i in base["Id"]]].astype(np.float32)
    return base, vetores


def vetor_da_pergunta(pergunta: str) -> np.ndarray:
    resposta = com_tentativas(
        cliente().models.embed_content,
        model=MODELO_EMBEDDING,
        contents=pergunta,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=DIMENSOES),
    )
    v = np.array(resposta.embeddings[0].values, dtype=np.float32)
    return v / np.linalg.norm(v)


def buscar(pergunta: str, base: pd.DataFrame, vetores: np.ndarray, k: int = 10,
           sentimento: str | None = None, aspecto: str | None = None,
           vetor: np.ndarray | None = None) -> pd.DataFrame:
    """Devolve as k avaliações mais parecidas com a pergunta, com filtros opcionais."""
    mascara = np.ones(len(base), dtype=bool)
    if sentimento and "sentimento" in base:
        mascara &= (base["sentimento"] == sentimento).to_numpy()
    if aspecto and "aspectos" in base:
        mascara &= base["aspectos"].apply(lambda a: aspecto in list(a)).to_numpy()

    v = vetor if vetor is not None else vetor_da_pergunta(pergunta)
    similaridade = vetores @ v
    similaridade[~mascara] = -np.inf
    melhores = np.argsort(-similaridade)[:k]
    melhores = melhores[np.isfinite(similaridade[melhores])]
    return base.iloc[melhores].assign(similaridade=similaridade[melhores])


def responder(pergunta: str, encontradas: pd.DataFrame) -> str:
    if encontradas.empty:
        return "Nenhuma avaliação encontrada com esses filtros."
    contexto = "\n\n".join(
        f"[#{r.Id}] ({r.Score} estrelas) {r.Summary}: {str(r.Text)[:800]}"
        for r in encontradas.itertuples()
    )
    resposta = com_tentativas(
        cliente().models.generate_content,
        model=MODELO,
        contents=f"Pergunta: {pergunta}\n\nAvaliações:\n{contexto}",
        config=types.GenerateContentConfig(system_instruction=INSTRUCOES, temperature=0.2),
    )
    return resposta.text
