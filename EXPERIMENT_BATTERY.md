# Bateria de experimentos

Este guia organiza a execucao manual dos cinco experimentos principais.

## Preparacao

No terminal, entre na pasta do repositorio:

```bash
cd /Users/alvaropereira/Documents/GitHub/galaxy-rng-experiment
```

Confira se o codigo esta saudavel:

```bash
python3 -m pytest
```

Liste os experimentos disponiveis:

```bash
python3 scripts/run_experiment_battery.py --list-experiments
```

## Teste pequeno

Antes de rodar a bateria completa, execute uma versao pequena com 2 seeds:

```bash
python3 scripts/run_experiment_battery.py --experiment 01_split_seed --seed-start 1000 --seed-count 2
```

Ao final, verifique os CSVs em:

```text
results/experiment_summaries/01_split_seed/
```

## Experimento 1

Pergunta: quanto a escolha do gerador e da seed do split altera as metricas
quando o Random Forest fica fixo em random_state 42?

```bash
python3 scripts/run_experiment_battery.py --experiment 01_split_seed --seed-start 1000 --seed-count 30
```

Saida bruta local:

```text
results/experiment_runs/01_split_seed/
```

CSVs versionaveis:

```text
results/experiment_summaries/01_split_seed/
```

## Experimento 2

Pergunta: mantendo o mesmo split, quanto a aleatoriedade interna do Random
Forest altera as metricas?

```bash
python3 scripts/run_experiment_battery.py --experiment 02_model_state --seed-start 1000 --seed-count 30
```

## Experimento 3

Pergunta: o que acontece quando a seed do split e o random_state do modelo
mudam juntos?

```bash
python3 scripts/run_experiment_battery.py --experiment 03_coupled_seed --seed-start 1000 --seed-count 30
```

## Experimento 4

Pergunta: a variabilidade causada por seed e gerador muda quando o tamanho do
dataset muda?

```bash
python3 scripts/run_experiment_battery.py --experiment 04_dataset_size --seed-start 1000 --seed-count 30
```

## Experimento 5

Pergunta: existe diferenca relevante de tempo na geracao dos splits entre os
geradores?

```bash
python3 scripts/run_experiment_battery.py --experiment 05_split_timing --seed-start 1000 --seed-count 30
```

## Tudo de uma vez

Use apenas depois de validar os passos acima:

```bash
python3 scripts/run_experiment_battery.py --experiment all --seed-start 1000 --seed-count 30
```

## Politica de arquivos

Resultados brutos ficam locais e nao devem ir para o GitHub:

```text
results/experiment_runs/
```

Tabelas e notas pequenas devem ser versionadas:

```text
results/experiment_summaries/
```
