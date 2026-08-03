"""
Executor de experimentos com múltiplos geradores e seeds.
"""

from pathlib import Path
from typing import Dict, List

from src.data.manifest import (
    create_split_manifest,
    save_split_manifest,
)
from src.data.splitting import split_stratified_indices
from src.experiments.metadata import create_execution_metadata


DEFAULT_GENERATORS = [
    "pcg64_cp_alite1",
    "numpy_pcg64",
    "numpy_mt19937",
]

DEFAULT_SEEDS = [
    123456789,
    987654321,
    20260429,
    3141592653,
    2718281828,
]


def run_split_experiment(
    labels: List[str],
    generator_name: str,
    seed: int,
    output_directory: str,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict:
    """
    Executa uma única combinação de gerador e seed.

    A função gera uma divisão estratificada, cria seu manifesto,
    registra metadados e salva o resultado em JSON.
    """
    if not isinstance(labels, list):
        raise TypeError("labels deve ser uma lista.")

    if not isinstance(output_directory, str):
        raise TypeError("output_directory deve ser uma string.")

    split_indices = split_stratified_indices(
        labels=labels,
        generator_name=generator_name,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    manifest = create_split_manifest(
        split_indices=split_indices,
        generator_name=generator_name,
        seed=seed,
        stratified=True,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        labels=labels,
    )

    metadata = create_execution_metadata(
        generator_name=generator_name,
        seed=seed,
    )

    experiment_record = {
        "metadata": metadata,
        "manifest": manifest,
    }

    output_path = Path(output_directory) / generator_name
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_path / f"seed_{seed}.json"

    save_split_manifest(
        manifest=experiment_record,
        output_path=str(output_file),
    )

    return experiment_record


def run_batch_split_experiments(
    labels: List[str],
    output_directory: str,
    generators: List[str] = None,
    seeds: List[int] = None,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Dict[int, dict]]:
    """
    Executa todas as combinações entre geradores e seeds.

    Retorna um dicionário organizado por gerador e seed.
    """
    if not isinstance(labels, list):
        raise TypeError("labels deve ser uma lista.")

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
            experiment_record = run_split_experiment(
                labels=labels,
                generator_name=generator_name,
                seed=seed,
                output_directory=output_directory,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
            )

            results[generator_name][seed] = experiment_record

    return results