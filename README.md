# 🔎 Raio-X de Avaliações da Amazon com IA

Este projeto transforma o texto de ~568 mil avaliações de alimentos da Amazon em análise consultável.

A partir do texto e da nota, um LLM identifica **os motivos de insatisfação do cliente** e o que leva um
produto a ser bem ou mal avaliado, classificando cada avaliação por **sentimento** e **aspecto**
(qualidade, sabor, preço, entrega, embalagem). O app também permite **fazer perguntas em linguagem
natural** sobre as avaliações, com respostas que citam as avaliações usadas como fonte (RAG).

**🔗 Demo:** [raio-x-avaliacoes-amazon.streamlit.app](https://raio-x-avaliacoes-amazon.streamlit.app/)

## O que o app mostra

- **Painel:** volume de avaliações, % de sentimento negativo, aspecto mais criticado e avaliações em que
  a nota e o texto discordam (ex.: 5 estrelas com texto negativo).
- **Sobre o que as pessoas falam:** menções por aspecto e sentimento.
- **Motivos de insatisfação:** resumo, gerado pelo LLM, do motivo de cada avaliação negativa.
- **Pergunte às avaliações:** perguntas como *"Do que as pessoas mais reclamam nos cafés?"*, com
  filtros por sentimento e aspecto e as avaliações usadas como fonte.

## Principais resultados

Em 3.000 avaliações classificadas (600 por nota):

- **Qualidade e sabor concentram as críticas:** juntos somam 1.632 menções negativas, bem à frente de
  preço (258), embalagem (163) e entrega (117).
- **Embalagem é o aspecto com maior proporção de críticas:** 53% das menções a embalagem são negativas,
  contra 38% das menções a sabor. Ela aparece pouco, mas quando aparece costuma ser motivo de reclamação.
- **As avaliações de 3 estrelas são as mais ambíguas:** 66% têm texto neutro e 26% negativo, ou seja,
  a nota "média" esconde mais insatisfação do que elogio.
- **Nota e texto quase sempre concordam:** só 0,8% das avaliações têm nota alta com texto negativo
  (ou o contrário), o que indica que a classificação do LLM é consistente com a nota dada pelo cliente.

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
5. **Respostas de exemplo (`src/05_respostas_exemplo.py`):** gera antes as respostas das perguntas de exemplo
   do app, que aparecem na hora, sem chamar a API.

## Decisões técnicas

- **Amostra estratificada em vez da base inteira:** classificar 568 mil avaliações no plano gratuito da
  API levaria semanas. 3.000 avaliações equilibradas por nota já mostram os padrões com clareza.
- **Saída estruturada:** o LLM devolve JSON validado por schema, então não há texto para "interpretar"
  e categorias inválidas são rejeitadas.
- **Resiliência a limites da API:** a API gratuita do Gemini fica sobrecarregada com frequência (503) e
  limita o ritmo de chamadas (429). Esses erros são tratados como esperados: novas tentativas com espera
  crescente, troca automática para um modelo reserva quando o principal segue indisponível e progresso
  salvo a cada lote, para nunca perder trabalho.
- **Demo que não depende da API:** o painel lê apenas dados já processados, e as perguntas de exemplo têm
  respostas geradas antes. Só perguntas novas chamam o Gemini na hora.
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
python src/05_respostas_exemplo.py
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
    ├── 05_respostas_exemplo.py  # respostas prontas dos exemplos do app
    ├── busca.py            # busca semântica + resposta (RAG)
    ├── gemini.py           # cliente da API: novas tentativas e modelo reserva
    └── vetores.py          # modelo de embeddings
```

## Dados

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) (Stanford SNAP,
via Kaggle): ~568 mil avaliações de alimentos da Amazon, de 1999 a 2012. A base não traz o nome dos produtos,
só um código, por isso as perguntas funcionam melhor por tipo de produto (café, chá, ração, salgadinho).

## Stack

Python · pandas · NumPy · Pydantic · Gemini API · FastEmbed · Streamlit · Plotly
