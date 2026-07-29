# galaxy-rng-experiment
Experimental framework for evaluating the influence of pseudorandom number generators and seeds on galaxy morphology classification using machine learning.


O projeto integra o PCG64-CP A Lite 1, um gerador pseudoaleatório híbrido desenvolvido no contexto de uma pesquisa de Iniciação Científica, a um pipeline de classificação de imagens astronômicas. O objetivo é comparar seu comportamento com geradores tradicionais, como o PCG64 e o MT19937, analisando desempenho, estabilidade e reprodutibilidade experimental.

⸻

Objetivo

Investigar se a escolha do gerador pseudoaleatório e da semente utilizada em um pipeline de aprendizado de máquina influencia:

* o desempenho do modelo;
* a variabilidade entre execuções;
* a estabilidade das métricas;
* a reprodutibilidade dos resultados;
* a composição dos conjuntos de treinamento, validação e teste;
* a ordem de apresentação das amostras durante o treinamento.

O projeto também busca discutir se resultados obtidos com uma única semente e um único gerador são suficientes para representar adequadamente o comportamento de um modelo de classificação.

⸻

Pergunta de pesquisa

A escolha do gerador pseudoaleatório e da semente influencia o desempenho, a estabilidade e a reprodutibilidade de modelos utilizados na classificação morfológica de galáxias?

Como questão complementar:

Resultados obtidos com uma única semente e um único gerador são representativos da variabilidade experimental de um modelo de aprendizado de máquina?

⸻

Geradores avaliados

Inicialmente, o experimento deverá comparar:

* PCG64-CP A Lite 1
* PCG64 do NumPy
* MT19937 — Mersenne Twister

Outros geradores poderão ser adicionados posteriormente, desde que utilizem a mesma interface experimental.

---
# Arquitetura Prevista

galaxy-rng-experiment/
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
│
├── src/
│   ├── rng/
│   │   ├── core.py
│   │   ├── rng.py
│   │   ├── adapters.py
│   │   └── factory.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── splitting.py
│   │
│   ├── models/
│   │   ├── cnn.py
│   │   └── training.py
│   │
│   ├── experiments/
│   │   ├── run_experiment.py
│   │   ├── configurations.py
│   │   └── reproducibility.py
│   │
│   └── analysis/
│       ├── metrics.py
│       ├── statistics.py
│       └── plots.py
│
├── tests/
│   ├── test_rng.py
│   ├── test_split.py
│   └── test_reproducibility.py
│
├── configs/
│   ├── baseline.yaml
│   └── generators.yaml
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
└── results/
    ├── raw/
    ├── summaries/
    └── figures/

---
