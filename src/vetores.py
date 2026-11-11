"""Embeddings gerados localmente, sem API e sem limite de uso.

Usa um modelo multilíngue pequeno (~220 MB, roda em CPU): perguntas em português
encontram avaliações em inglês com o mesmo significado. Na primeira execução o
modelo é baixado automaticamente e fica em cache.
"""
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

MODELO_LOCAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def modelo() -> TextEmbedding:
    return TextEmbedding(MODELO_LOCAL)


def normalizar(matriz: np.ndarray) -> np.ndarray:
    """Deixa cada vetor com tamanho 1, para o produto escalar ser a similaridade de cosseno."""
    return matriz / np.linalg.norm(matriz, axis=-1, keepdims=True)


def vetorizar(textos: list[str], lote: int = 64) -> np.ndarray:
    return normalizar(np.array(list(modelo().embed(textos, batch_size=lote)), dtype=np.float32))
