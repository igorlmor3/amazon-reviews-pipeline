"""Classifica cada avaliação da amostra por aspecto e sentimento com o Gemini.

- Envia as avaliações em lotes, para gastar menos requisições do free tier.
- Usa saída estruturada (JSON com schema), então a resposta já vem validada.
- Salva o progresso a cada lote: se cair no limite diário, é só rodar de novo
  no dia seguinte que ele continua de onde parou.
"""
import json
import os
import random
import time
from enum import Enum
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

RAIZ = Path(__file__).resolve().parents[1]
AMOSTRA = RAIZ / "data" / "processed" / "amostra.parquet"
PROGRESSO = RAIZ / "data" / "interim" / "classificacoes.jsonl"
SAIDA = RAIZ / "data" / "processed" / "avaliacoes_classificadas.parquet"

MODELO = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODELO_RESERVA = os.getenv("GEMINI_MODEL_RESERVA", "")
TENTATIVAS = 8
PAUSA = float(os.getenv("PAUSA_ENTRE_CHAMADAS", "5"))
TAMANHO_LOTE = 25
MAX_CARACTERES = 1200  # corta textos muito longos para economizar tokens


class Aspecto(str, Enum):
    qualidade = "qualidade"
    sabor = "sabor"
    preco = "preco"
    entrega = "entrega"
    embalagem = "embalagem"
    outro = "outro"


class Sentimento(str, Enum):
    positivo = "positivo"
    neutro = "neutro"
    negativo = "negativo"


class Classificacao(BaseModel):
    id: int
    sentimento: Sentimento
    aspectos: list[Aspecto]
    motivo: str


INSTRUCOES = """Você analisa avaliações de produtos alimentícios vendidos na Amazon.
Para cada avaliação, devolva:
- id: o mesmo id recebido.
- sentimento: positivo, neutro ou negativo, pelo que o texto diz (não pela nota).
- aspectos: os temas que a pessoa comenta, entre qualidade, sabor, preco,
  entrega, embalagem ou outro. Pode haver mais de um.
- motivo: uma frase curta, em português, com o principal motivo da opinião.
Classifique todas as avaliações recebidas, sem pular nenhuma."""


def montar_lote(linhas: pd.DataFrame) -> str:
    itens = [
        {"id": int(r.Id), "nota": int(r.Score), "titulo": r.Summary,
         "texto": str(r.Text)[:MAX_CARACTERES]}
        for r in linhas.itertuples()
    ]
    return json.dumps(itens, ensure_ascii=False)


def classificar(cliente: genai.Client, linhas: pd.DataFrame, modelo: str) -> list[dict]:
    resposta = cliente.models.generate_content(
        model=modelo,
        contents=montar_lote(linhas),
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCOES,
            response_mime_type="application/json",
            response_schema=list[Classificacao],
            temperature=0,
        ),
    )
    return [c.model_dump(mode="json") for c in resposta.parsed]


def ja_classificados() -> set[int]:
    if not PROGRESSO.exists():
        return set()
    with PROGRESSO.open(encoding="utf-8") as f:
        return {json.loads(linha)["id"] for linha in f if linha.strip()}


def main() -> None:
    cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    amostra = pd.read_parquet(AMOSTRA)
    feitos = ja_classificados()
    pendentes = amostra[~amostra["Id"].isin(feitos)]
    print(f"Já classificadas: {len(feitos)} | Pendentes: {len(pendentes)}")

    PROGRESSO.parent.mkdir(parents=True, exist_ok=True)
    lotes = [pendentes.iloc[i:i + TAMANHO_LOTE]
             for i in range(0, len(pendentes), TAMANHO_LOTE)]

    for lote in tqdm(lotes, desc="Lotes"):
        for tentativa in range(TENTATIVAS):
            # Depois de 3 falhas seguidas, tenta o modelo reserva (se houver).
            modelo = MODELO_RESERVA if MODELO_RESERVA and tentativa >= 3 else MODELO
            try:
                resultado = classificar(cliente, lote, modelo)
                break
            except errors.APIError as erro:
                if erro.code == 429 and "PerDay" in str(erro):
                    print("\nLimite diário atingido. Rode de novo amanhã para continuar.")
                    return finalizar(amostra)
                # 503 = servidor do Google sobrecarregado (temporário).
                # 429 = limite por minuto. Espera crescente com variação aleatória.
                espera = min(10 * 2 ** tentativa, 120) + random.uniform(0, 5)
                print(f"\nErro {erro.code} ({modelo}); nova tentativa em {espera:.0f}s")
                time.sleep(espera)
        else:
            print(f"\nLote falhou {TENTATIVAS} vezes; fica para a próxima execução.")
            continue

        ids_do_lote = set(lote["Id"])
        with PROGRESSO.open("a", encoding="utf-8") as f:
            for item in resultado:
                if item["id"] in ids_do_lote:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        time.sleep(PAUSA)

    finalizar(amostra)


def finalizar(amostra: pd.DataFrame) -> None:
    """Junta as classificações à amostra e salva o arquivo final."""
    if not PROGRESSO.exists():
        return
    classes = pd.read_json(PROGRESSO, lines=True).drop_duplicates("id", keep="last")
    final = amostra.merge(classes, left_on="Id", right_on="id", how="inner").drop(columns="id")
    final.to_parquet(SAIDA, index=False)
    print(f"\n{len(final)} de {len(amostra)} avaliações classificadas -> {SAIDA}")
    print(final["sentimento"].value_counts().to_string())


if __name__ == "__main__":
    main()
