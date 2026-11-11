"""Gera os embeddings das avaliações da amostra para a busca semântica.

Cada avaliação vira um vetor de 768 números que representa o seu significado.
Perguntas parecidas em sentido ficam próximas desses vetores, mesmo sem
usar as mesmas palavras (e mesmo em outro idioma).
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google.genai import types
from tqdm import tqdm

from gemini import MODELO_EMBEDDING, LimiteDiarioAtingido, cliente, com_tentativas

RAIZ = Path(__file__).resolve().parents[1]
AMOSTRA = RAIZ / "data" / "processed" / "amostra.parquet"
PROGRESSO = RAIZ / "data" / "interim" / "embeddings_parcial.npz"
SAIDA = RAIZ / "data" / "processed" / "embeddings.npz"
DIMENSOES = 768
# O free tier conta cada texto do lote no limite por minuto, então usamos lotes
# pequenos com pausa entre eles. Ajuste no .env se continuar dando 429.
LOTE = int(os.getenv("EMBED_LOTE", "20"))
PAUSA = float(os.getenv("EMBED_PAUSA", "15"))
MAX_CARACTERES = 1500


def texto_da_avaliacao(r) -> str:
    return f"{r.Summary}. {r.Text}"[:MAX_CARACTERES]


def main() -> None:
    amostra = pd.read_parquet(AMOSTRA)
    textos = [texto_da_avaliacao(r) for r in amostra.itertuples()]
    api = cliente()

    # Retoma de onde parou, se uma execução anterior foi interrompida.
    vetores = list(np.load(PROGRESSO)["vetores"]) if PROGRESSO.exists() else []
    if vetores:
        print(f"Retomando: {len(vetores)} de {len(textos)} já prontos")
    PROGRESSO.parent.mkdir(parents=True, exist_ok=True)

    inicio = len(vetores)
    try:
        for i in tqdm(range(inicio, len(textos), LOTE), desc="Embeddings"):
            resposta = com_tentativas(
                api.models.embed_content,
                model=MODELO_EMBEDDING,
                contents=textos[i:i + LOTE],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT", output_dimensionality=DIMENSOES
                ),
            )
            vetores.extend(e.values for e in resposta.embeddings)
            np.savez_compressed(PROGRESSO, vetores=np.array(vetores, dtype=np.float32))
            time.sleep(PAUSA)
    except LimiteDiarioAtingido:
        print("Limite diário atingido. Rode de novo amanhã; o progresso está salvo.")
        return

    matriz = np.array(vetores, dtype=np.float32)
    # Normaliza para que o produto escalar seja a similaridade de cosseno.
    matriz /= np.linalg.norm(matriz, axis=1, keepdims=True)
    np.savez_compressed(SAIDA, ids=amostra["Id"].to_numpy(), vetores=matriz.astype(np.float16))
    print(f"{len(matriz)} embeddings de {DIMENSOES} dimensões -> {SAIDA}")


if __name__ == "__main__":
    main()
