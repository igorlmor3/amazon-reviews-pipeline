"""Faz uma pergunta sobre as avaliações pelo terminal.

Exemplos:
  python src/04_perguntar.py "Do que as pessoas mais reclamam nos cafés?"
  python src/04_perguntar.py "Quais problemas de embalagem aparecem?" --sentimento negativo
  python src/04_perguntar.py "O que elogiam nos chás?" --aspecto sabor
"""
import argparse

from busca import buscar, carregar_base, responder

parser = argparse.ArgumentParser()
parser.add_argument("pergunta")
parser.add_argument("--k", type=int, default=10, help="quantas avaliações usar como fonte")
parser.add_argument("--sentimento", choices=["positivo", "neutro", "negativo"])
parser.add_argument("--aspecto", choices=["qualidade", "sabor", "preco", "entrega", "embalagem", "outro"])
args = parser.parse_args()

base, vetores = carregar_base()
encontradas = buscar(args.pergunta, base, vetores, k=args.k,
                     sentimento=args.sentimento, aspecto=args.aspecto)

print(responder(args.pergunta, encontradas))
print("\nFontes:")
for r in encontradas.itertuples():
    print(f"  #{r.Id} ({r.Score}/5, similaridade {r.similaridade:.2f}) {r.Summary}")
