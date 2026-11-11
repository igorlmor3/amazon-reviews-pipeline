"""Gera uma amostra estratificada das avaliações para classificar com o LLM.

Classificar as ~568 mil avaliações no free tier levaria semanas, então
trabalhamos com uma amostra equilibrada por nota (1 a 5 estrelas).
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "Reviews.csv"
SAIDA = RAIZ / "data" / "processed" / "amostra.parquet"
POR_NOTA = 600  # 600 x 5 notas = 3.000 avaliações
SEMENTE = 42

colunas = ["Id", "ProductId", "UserId", "Score", "Time", "Summary", "Text"]
df = pd.read_csv(ENTRADA, usecols=colunas)
total = len(df)

# A base tem a mesma avaliação repetida em produtos diferentes (variações
# do mesmo item). Mantemos só uma ocorrência de cada texto por usuário.
df = df.drop_duplicates(subset=["UserId", "Time", "Text"])
df = df.dropna(subset=["Text"])
sem_duplicatas = len(df)

amostra = (
    df.groupby("Score", group_keys=False)
    .sample(n=POR_NOTA, random_state=SEMENTE)
    .assign(data=lambda d: pd.to_datetime(d["Time"], unit="s"))
    .drop(columns=["Time", "UserId"])
    .sort_values("Id")
    .reset_index(drop=True)
)

SAIDA.parent.mkdir(parents=True, exist_ok=True)
amostra.to_parquet(SAIDA, index=False)

print(f"Base original: {total:,} avaliações")
print(f"Sem duplicatas: {sem_duplicatas:,}")
print(f"Amostra: {len(amostra):,} avaliações ({POR_NOTA} por nota) -> {SAIDA}")
print(amostra["Score"].value_counts().sort_index().to_string())
