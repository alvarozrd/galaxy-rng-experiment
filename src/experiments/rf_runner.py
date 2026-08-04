"""
Executor completo do experimento Random Forest com splits por manifesto.
"""

import json
from pathlib import Path
from typing import Dict, List

from src.data.galaxies import load_prepared_galaxy_dataset
from src.data.split_application import apply_split_manifest
from src.experiments.galaxy_runner import run_galaxy_split_experiment
from src.experiments.runner import DEFAULT_GENERATORS, DEFAULT_SEEDS
from src.models.random_forest import train_and_evaluate_random_forest


DEFAULT_MODEL_RANDOM_STATE = 42


def run_random_forest_experiment(
    dataset_path: str,
    generator_name: str,
    seed: int,
    output_directory: str,
    model_random_state: int = DEFAULT_MODEL_RANDOM_STATE,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict:
    """
    Executa o pipeline completo para uma combinação de gerador e seed.

    Fluxo:
        - carrega o dataset COSMOS;
        - gera o split estratificado com o PRNG escolhido;
        - aplica o manifesto nos dados;
        - treina Random Forest;
        - avalia validação e teste;
        - salva um JSON com manifesto e métricas.
    """
    if not isinstance(dataset_path, str):
        raise TypeError("dataset_path deve ser uma string.")

    if not isinstance(output_directory, str):
        raise TypeError("output_directory deve ser uma string.")

    if isinstance(model_random_state, bool) or not isinstance(
        model_random_state,
        int,
    ):
        raise TypeError("model_random_state deve ser um número inteiro.")

    features, target, _ = load_prepared_galaxy_dataset(dataset_path)

    split_record = run_galaxy_split_experiment(
        dataset_path=dataset_path,
        generator_name=generator_name,
        seed=seed,
        output_directory=output_directory,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    split_data = apply_split_manifest(
        features=features,
        target=target,
        manifest=split_record["manifest"],
    )

    model_results = train_and_evaluate_random_forest(
        split_data=split_data,
        random_state=model_random_state,
    )

    experiment_record = {
        "metadata": split_record["metadata"],
        "dataset": split_record["dataset"],
        "manifest": split_record["manifest"],
        "model": {
            "name": "RandomForestClassifier",
            "random_state": model_random_state,
            "results": model_results,
        },
    }

    output_path = Path(output_directory) / generator_name
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_path / f"seed_{seed}_random_forest.json"

    with output_file.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment_record,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return experiment_record


def run_batch_random_forest_experiments(
    dataset_path: str,
    output_directory: str,
    generators: List[str] = None,
    seeds: List[int] = None,
    model_random_state: int = DEFAULT_MODEL_RANDOM_STATE,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Dict[int, dict]]:
    """
    Executa o pipeline completo para múltiplos geradores e seeds.
    """
    if not isinstance(dataset_path, str):
        raise TypeError("dataset_path deve ser uma string.")

    if not isinstance(output_directory, str):
        raise TypeError("output_directory deve ser uma string.")

    selected_generators = (
        DEFAULT_GENERATORS
        if generators is None
        else generators
    )

    selected_seeds = (
        DEFAULT_SEEDS
        if seeds is None
        else seeds
    )

    if not isinstance(selected_generators, list):
        raise TypeError("generators deve ser uma lista ou None.")

    if not isinstance(selected_seeds, list):
        raise TypeError("seeds deve ser uma lista ou None.")

    results: Dict[str, Dict[int, dict]] = {}

    for generator_name in selected_generators:
        results[generator_name] = {}

        for seed in selected_seeds:
            experiment_record = run_random_forest_experiment(
                dataset_path=dataset_path,
                generator_name=generator_name,
                seed=seed,
                output_directory=output_directory,
                model_random_state=model_random_state,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
            )

            results[generator_name][seed] = experiment_record

    return results