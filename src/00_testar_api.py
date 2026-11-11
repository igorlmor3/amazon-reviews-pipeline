"""Confere se a chave do Gemini funciona e lista os modelos disponíveis."""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Modelos que geram texto:")
for modelo in cliente.models.list():
    acoes = getattr(modelo, "supported_actions", None) or []
    if "generateContent" in acoes:
        print(" -", modelo.name.removeprefix("models/"))

print("\nModelos de embedding:")
for modelo in cliente.models.list():
    if "embedContent" in (getattr(modelo, "supported_actions", None) or []):
        print(" -", modelo.name.removeprefix("models/"))

modelo = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
resposta = cliente.models.generate_content(model=modelo, contents="Responda só: ok")
print(f"\nTeste com {modelo}: {resposta.text.strip()}")
