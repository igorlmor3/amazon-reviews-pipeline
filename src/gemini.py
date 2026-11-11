"""Funções compartilhadas para chamar o Gemini com novas tentativas automáticas."""
import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

MODELO = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODELO_EMBEDDING = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
TENTATIVAS = 8


class LimiteDiarioAtingido(Exception):
    pass


def cliente() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def com_tentativas(funcao, *args, **kwargs):
    """Executa a chamada; em 503 (sobrecarga) ou 429 (limite por minuto), espera e tenta de novo."""
    for tentativa in range(TENTATIVAS):
        try:
            return funcao(*args, **kwargs)
        except errors.APIError as erro:
            if erro.code == 429 and "PerDay" in str(erro):
                raise LimiteDiarioAtingido from erro
            if erro.code not in (429, 500, 503) or tentativa == TENTATIVAS - 1:
                raise
            espera = min(10 * 2 ** tentativa, 120) + random.uniform(0, 5)
            print(f"Erro {erro.code}; nova tentativa em {espera:.0f}s")
            time.sleep(espera)
