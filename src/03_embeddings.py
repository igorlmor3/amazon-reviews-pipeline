"""Gera os embeddings das avaliações da amostra para a busca semântica.

Cada avaliação vira um vetor de 384 números que representa o seu significado.
Textos com sentido parecido ficam próximos, mesmo com palavras diferentes ou em
outro idioma. Roda localmente (sem API), então não há limite de uso.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from vetores import MODELO_LOCAL, vetorizar

RAIZ = Path(__file__).resolve().parents[1]
AMOSTRA = RAIZ / "data" / "processed" / "amostra.parquet"
SAIDA = RAIZ / "data" / "processed" / "embeddings.npz"


def main() -> None:
    amostra = pd.read_parquet(AMOSTRA)
    # O modelo lê até ~128 tokens: título + começo do texto concentram o assunto.
    textos = (amostra["Summary"].fillna("") + ". " + amostra["Text"].fillna("")).str[:1000].tolist()

    print(f"Gerando {len(textos)} embeddings com {MODELO_LOCAL} (a 1ª vez baixa o modelo)...")
    matriz = vetorizar(textos)
    np.savez_compressed(SAIDA, ids=amostra["Id"].to_numpy(), vetores=matriz.astype(np.float16),
                        modelo=np.array(MODELO_LOCAL))
    print(f"{matriz.shape[0]} embeddings de {matriz.shape[1]} dimensões -> {SAIDA}")


if __name__ == "__main__":
    main()
