# 🔎 Raio-X de Avaliações da Amazon com IA

Este projeto transforma o texto de ~568 mil avaliações de alimentos da Amazon em análise consultável.

A partir do texto e da nota, um LLM identifica **os motivos de insatisfação do cliente** e o que leva um
produto a ser bem ou mal avaliado, classificando cada avaliação por **sentimento** e **aspecto**
(qualidade, sabor, preço, entrega, embalagem). O app também permite **fazer perguntas em linguagem
natural** sobre as avaliações, com respostas que citam as avaliações usadas como fonte (RAG).

> **Status:** em desenvolvimento. Amostra e embeddings prontos; classificação com LLM em andamento
> (cerca de 800 de 3.000 avaliações). Os números abaixo são parciais.

## O que o app mostra

- **Painel:** volume de avaliações, % de sentimento negativo, aspecto mais criticado e avaliações em que
  a nota e o texto discordam (ex.: 5 estrelas com texto negativo).
- **Sobre o que as pessoas falam:** menções por aspecto e sentimento.
- **Motivos de insatisfação:** resumo, gerado pelo LLM, do motivo de cada avaliação negativa.
- **Pergunte às avaliações:** perguntas como *"Do que as pessoas mais reclamam nos cafés?"*, com
  filtros por sentimento e aspecto e as avaliações usadas como fonte.

**Primeiros resultados (parciais, ~800 avaliações):** 44% têm texto negativo, e **sabor** e
**qualidade** concentram a maior parte das críticas, bem à frente de preço, embalagem e entrega.

## Como funciona

```
Reviews.csv (568 mil) ──► 01 amostra ──► 3.000 avaliações (600 por nota)
                                              │
                    ┌─────────────────────────┴───────────────────────┐
                    ▼                                                 ▼
     02 classificar (Gemini, JSON com schema)         03 embeddings (modelo local multilíngue)
     sentimento · aspectos · motivo                    vetor de 384 dimensões por avaliação
                    │                                                 │
                    └────────────────────► app.py ◄───────────────────┘
                              painel (Plotly) + perguntas (busca semântica + Gemini)
```

1. **Amostra (`src/01_amostra.py`):** remove duplicatas (a base repete a mesma avaliação em variações do
   produto: 568.454 → 393.892 únicas) e sorteia 3.000 avaliações, 600 por nota, para equilibrar as notas.
2. **Classificação com LLM (`src/02_classificar.py`):** envia lotes de 25 avaliações ao Gemini com saída
   estruturada (JSON validado por schema Pydantic), salva o progresso a cada lote e retoma de onde parou.
3. **Embeddings (`src/03_embeddings.py`):** gera um vetor por avaliação com um modelo multilíngue que roda
   localmente (FastEmbed), então perguntas em português encontram avaliações em inglês.
4. **Busca e resposta (`src/busca.py`):** compara a pergunta com as avaliações por similaridade de cosseno,
   aplica os filtros e pede ao Gemini uma resposta baseada só nas avaliações encontradas, citando as fontes.

## Decisões técnicas

- **Amostra estratificada em vez da base inteira:** classificar 568 mil avaliações no plano gratuito da
  API levaria semanas. 3.000 avaliações equilibradas por nota já mostram os padrões com clareza.
- **Saída estruturada:** o LLM devolve JSON validado por schema, então não há texto para "interpretar"
  e categorias inválidas são rejeitadas.
- **Resiliência a limites da API:** novas tentativas com espera crescente para erros 503 (sobrecarga) e
  429 (limite por minuto), modelo reserva opcional e progresso salvo a cada lote.
- **Embeddings locais:** a primeira versão usava a API de embeddings do Gemini, mas o limite diário do plano
  gratuito tornava a geração lenta demais. O modelo local é gratuito, sem limite e multilíngue.
- **Proteção da cota na demo pública:** cada visitante pode fazer até 10 perguntas, e respostas iguais
  ficam em cache.

## Como rodar

Requer Python 3.11+ e uma chave gratuita da API do Gemini ([Google AI Studio](https://aistudio.google.com/apikey)).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e preencha `GEMINI_API_KEY`. O `.env` não vai para o repositório.

```bash
python src/00_testar_api.py       # confere a chave e lista os modelos disponíveis
python src/01_amostra.py          # requer o Reviews.csv do Kaggle na raiz do projeto
python src/02_classificar.py
python src/03_embeddings.py
streamlit run app.py
```

Os dados processados já estão em `data/processed/`, então para ver o app basta instalar as dependências,
criar o `.env` e rodar `streamlit run app.py`.

Também é possível perguntar pelo terminal:

```bash
python src/04_perguntar.py "Quais problemas de embalagem aparecem?" --sentimento negativo
```

## Estrutura

```
├── app.py                  # app Streamlit (painel + perguntas)
├── assets/estilo.css       # visual do app
├── .streamlit/config.toml  # tema
├── data/processed/         # amostra, classificações e embeddings (versionados)
└── src/
    ├── 00_testar_api.py    # teste da chave e lista de modelos
    ├── 01_amostra.py       # limpeza e amostragem
    ├── 02_classificar.py   # classificação com LLM
    ├── 03_embeddings.py    # embeddings locais
    ├── 04_perguntar.py     # perguntas pelo terminal
    ├── busca.py            # busca semântica + resposta (RAG)
    ├── gemini.py           # cliente da API com novas tentativas
    └── vetores.py          # modelo de embeddings
```

## Dados

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (Stanford SNAP,
via Kaggle): ~568 mil avaliações de alimentos da Amazon, de 1999 a 2012. A base não traz o nome dos produtos,
só um código, por isso as perguntas funcionam melhor por tipo de produto (café, chá, ração, salgadinho).

## Stack

Python · pandas · NumPy · Pydantic · Gemini API · FastEmbed · Streamlit · Plotly
