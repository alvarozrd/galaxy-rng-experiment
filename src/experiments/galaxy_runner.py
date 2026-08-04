"""
Executor de divisões reprodutíveis para o dataset COSMOS de galáxias.
"""

from pathlib import Path
from typing import Dict, List

from src.data.galaxies import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMN,
    TARGET_COLUMN,
    load_prepared_galaxy_dataset,
)
from src.data.manifest import (
    create_split_manifest,
    save_split_manifest,
)
from src.data.splitting import split_stratified_indices
from src.experiments.metadata import create_execution_metadata
from src.experiments.runner import DEFAULT_GENERATORS, DEFAULT_SEEDS


DATASET_NAME = "COSMOS Zurich Structure and Morphology Catalog v1.0"

DATASET_FILTERS = [
    "type != 9",
    "junkflag == 0, quando a coluna existir",
    "acs_clean == 1, quando a coluna existir",
]


def create_galaxy_dataset_summary(
    dataset_path: str,
) -> dict:
    """
    Carrega o dataset de galáxias e resume sua estrutura experimental.
    """
    _, target, prepared = load_prepared_galaxy_dataset(dataset_path)

    class_distribution = (
        target.astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "dataset_name": DATASET_NAME,
        "source_path": dataset_path,
        "sample_count": len(prepared),
        "identifier_column": IDENTIFIER_COLUMN,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "filters": DATASET_FILTERS,
        "class_distribution": class_distribution,
    }


def run_galaxy_split_experiment(
    dataset_path: str,
    generator_name: str,
    seed: int,
    output_directory: str,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict:
    """
    Executa uma divisão estratificada do dataset COSMOS para um gerador.
    """
    if not isinstance(dataset_path, str):
        raise TypeError("dataset_path deve ser uma string.")

    if not isinstance(output_directory, str):
        raise TypeError("output_directory deve ser uma string.")

    _, target, prepared = load_prepared_galaxy_dataset(dataset_path)
    labels = target.astype(str).tolist()

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

    experiment_record = {
        "metadata": create_execution_metadata(
            generator_name=generator_name,
            seed=seed,
        ),
        "dataset": {
            "dataset_name": DATASET_NAME,
            "source_path": dataset_path,
            "sample_count": len(prepared),
            "identifier_column": IDENTIFIER_COLUMN,
            "target_column": TARGET_COLUMN,
            "feature_columns": FEATURE_COLUMNS,
            "filters": DATASET_FILTERS,
        },
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


def run_batch_galaxy_split_experiments(
    dataset_path: str,
    output_directory: str,
    generators: List[str] = None,
    seeds: List[int] = None,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Dict[int, dict]]:
    """
    Executa divisões do dataset COSMOS para múltiplos geradores e seeds.
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
            experiment_record = run_galaxy_split_experiment(
                dataset_path=dataset_path,
                generator_name=generator_name,
                seed=seed,
                output_directory=output_directory,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
            )

            results[generator_name][seed] = experiment_record

    return results
