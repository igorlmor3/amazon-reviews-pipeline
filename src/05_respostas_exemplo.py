"""Gera e salva as respostas das perguntas de exemplo do app.

Assim, quem abre a demo e clica num exemplo vê a resposta na hora, sem depender
da disponibilidade da API naquele momento. Rode de novo se mudar os exemplos
ou depois que a classificação terminar.
"""
import json
from pathlib import Path

import pandas as pd

from busca import buscar, carregar_base, responder

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "data" / "processed" / "respostas_exemplo.parquet"

EXEMPLOS = [
    {"pergunta": "Do que as pessoas mais reclamam nos cafés?", "sentimento": None, "aspecto": None},
    {"pergunta": "Quais problemas de embalagem aparecem com frequência?", "sentimento": "negativo", "aspecto": None},
    {"pergunta": "O que os donos de cachorro elogiam nas rações?", "sentimento": "positivo", "aspecto": None},
]
K = 10

base, vetores = carregar_base()
linhas = []
for ex in EXEMPLOS:
    print(f"\n> {ex['pergunta']}")
    fontes = buscar(ex["pergunta"], base, vetores, k=K, sentimento=ex["sentimento"], aspecto=ex["aspecto"])
    resposta = responder(ex["pergunta"], fontes)
    print(resposta[:300], "...")
    linhas.append({**ex, "resposta": resposta,
                   "fontes": json.dumps([[int(i), float(s)] for i, s in zip(fontes["Id"], fontes["similaridade"])])})

pd.DataFrame(linhas).to_parquet(SAIDA, index=False)
print(f"\n{len(linhas)} respostas salvas -> {SAIDA}")
