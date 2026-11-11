"""Funções compartilhadas para chamar o Gemini com novas tentativas automáticas.

A API gratuita às vezes fica sobrecarregada (503) ou limita o ritmo de chamadas (429).
Esses erros são temporários e esperados: em vez de falhar, esperamos um tempo crescente
e tentamos de novo. Se o modelo principal continuar indisponível, usamos o modelo reserva.
"""
import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

MODELO = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODELO_RESERVA = os.getenv("GEMINI_MODEL_RESERVA", "")
TENTATIVAS = 8

MOTIVOS = {
    429: "limite de chamadas por minuto da API (429)",
    500: "erro temporário no servidor do Gemini (500)",
    503: "servidor do Gemini ocupado (503)",
}


class LimiteDiarioAtingido(Exception):
    pass


def cliente() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def espera_para(tentativa: int) -> float:
    """Espera crescente (10s, 20s, 40s... até 2 min) com variação aleatória."""
    return min(10 * 2 ** tentativa, 120) + random.uniform(0, 5)


def avisar(codigo: int, espera: float, tentativa: int, total: int, modelo: str = "") -> None:
    motivo = MOTIVOS.get(codigo, f"erro {codigo}")
    sufixo = f" [{modelo}]" if modelo else ""
    print(f"\n{motivo[0].upper() + motivo[1:]}: tentando de novo em {espera:.0f}s "
          f"(tentativa {tentativa + 2}/{total}){sufixo}")


def com_tentativas(funcao, *args, tentativas: int = TENTATIVAS, **kwargs):
    """Executa a chamada; em 429/500/503, espera e tenta de novo."""
    for tentativa in range(tentativas):
        try:
            return funcao(*args, **kwargs)
        except errors.APIError as erro:
            if erro.code == 429 and "PerDay" in str(erro):
                raise LimiteDiarioAtingido from erro
            if erro.code not in MOTIVOS or tentativa == tentativas - 1:
                raise
            espera = espera_para(tentativa)
            avisar(erro.code, espera, tentativa, tentativas)
            time.sleep(espera)


def gerar(**kwargs):
    """Gera texto com o modelo principal; se ele seguir indisponível, passa para o reserva."""
    modelos = [MODELO] + ([MODELO_RESERVA] if MODELO_RESERVA and MODELO_RESERVA != MODELO else [])
    api = cliente()
    for i, modelo in enumerate(modelos):
        ultimo = i == len(modelos) - 1
        try:
            return com_tentativas(api.models.generate_content, model=modelo,
                                  tentativas=TENTATIVAS if ultimo else 3, **kwargs)
        except errors.APIError as erro:
            if ultimo or erro.code not in MOTIVOS:
                raise
            print(f"\nModelo {modelo} indisponível; usando o modelo reserva {modelos[i + 1]}.")
