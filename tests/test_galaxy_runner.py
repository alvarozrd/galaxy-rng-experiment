import json

import pandas as pd
import pytest

from src.data.galaxies import FEATURE_COLUMNS
from src.experiments.galaxy_runner import (
    DATASET_NAME,
    create_galaxy_dataset_summary,
    run_batch_galaxy_split_experiments,
    run_galaxy_split_experiment,
)


CANONICAL_SEED = 123456789


def create_raw_galaxy_dataframe() -> pd.DataFrame:
    """Cria amostras balanceadas e linhas inválidas para teste."""
    rows = []

    for index in range(1, 13):
        label = ((index - 1) % 3) + 1

        rows.append(
            {
                "SequentialID": index,
                "GG": 0.30 + index / 100,
                "M20": -2.0 + index / 50,
                "CC": 1.50 + index / 10,
                "AA": 0.05 + index / 100,
                "SERSIC_N_GIM2D": 1.0 + index / 10,
                "ELL_GIM2D": 0.10 + index / 100,
                "R_GIM2D": 0.80 + index / 20,
                "TYPE": label,
                "JUNKFLAG": 0,
                "ACS_CLEAN": 1,
            }
        )

    rows.append(
        {
            "SequentialID": 13,
            "GG": 0.50,
            "M20": -1.1,
            "CC": 2.0,
            "AA": 0.20,
            "SERSIC_N_GIM2D": 1.5,
            "ELL_GIM2D": 0.30,
            "R_GIM2D": 1.1,
            "TYPE": 9,
            "JUNKFLAG": 0,
            "ACS_CLEAN": 1,
        }
    )

    return pd.DataFrame(rows)


def save_dataset(tmp_path) -> str:
    """Salva o dataset mínimo em CSV."""
    csv_path = tmp_path / "galaxies.csv"
    create_raw_galaxy_dataframe().to_csv(csv_path, index=False)

    return str(csv_path)


def test_create_galaxy_dataset_summary(tmp_path) -> None:
    """O resumo deve documentar estrutura e distribuição do dataset."""
    dataset_path = save_dataset(tmp_path)

    summary = create_galaxy_dataset_summary(dataset_path)

    assert summary["dataset_name"] == DATASET_NAME
    assert summary["sample_count"] == 12
    assert summary["target_column"] == "type"
    assert summary["feature_columns"] == FEATURE_COLUMNS
    assert summary["class_distribution"] == {
        "1": 4,
        "2": 4,
        "3": 4,
    }


def test_run_galaxy_split_experiment_returns_record(tmp_path) -> None:
    """Uma execução deve retornar metadados, dataset e manifesto."""
    dataset_path = save_dataset(tmp_path)

    result = run_galaxy_split_experiment(
        dataset_path=dataset_path,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
        output_directory=str(tmp_path / "results"),
    )

    assert result["metadata"]["generator_name"] == "pcg64_cp_alite1"
    assert result["dataset"]["sample_count"] == 12
    assert result["manifest"]["dataset_size"] == 12
    assert result["manifest"]["class_distribution"]["train"] == {
        "1": 3,
        "2": 3,
        "3": 3,
    }


def test_run_galaxy_split_experiment_saves_json(tmp_path) -> None:
    """A execução deve salvar o manifesto completo em JSON."""
    dataset_path = save_dataset(tmp_path)
    output_directory = tmp_path / "results"

    run_galaxy_split_experiment(
        dataset_path=dataset_path,
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
        output_directory=str(output_directory),
    )

    output_file = (
        output_directory
        / "numpy_pcg64"
        / f"seed_{CANONICAL_SEED}.json"
    )

    with output_file.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        saved_data = json.load(file)

    assert saved_data["dataset"]["dataset_name"] == DATASET_NAME
    assert saved_data["manifest"]["seed"] == CANONICAL_SEED


def test_run_galaxy_split_experiment_is_reproducible(tmp_path) -> None:
    """A mesma configuração deve gerar o mesmo manifesto."""
    dataset_path = save_dataset(tmp_path)

    result_a = run_galaxy_split_experiment(
        dataset_path=dataset_path,
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
        output_directory=str(tmp_path / "run_a"),
    )

    result_b = run_galaxy_split_experiment(
        dataset_path=dataset_path,
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
        output_directory=str(tmp_path / "run_b"),
    )

    assert result_a["manifest"] == result_b["manifest"]


def test_run_batch_galaxy_split_experiments(tmp_path) -> None:
    """O lote deve executar todas as combinações pedidas."""
    dataset_path = save_dataset(tmp_path)
    generators = [
        "pcg64_cp_alite1",
        "numpy_pcg64",
    ]
    seeds = [
        123,
        456,
    ]

    results = run_batch_galaxy_split_experiments(
        dataset_path=dataset_path,
        output_directory=str(tmp_path / "results"),
        generators=generators,
        seeds=seeds,
    )

    assert set(results.keys()) == set(generators)

    for generator_name in generators:
        assert set(results[generator_name].keys()) == set(seeds)


def test_run_galaxy_split_experiment_rejects_invalid_paths(tmp_path) -> None:
    """Os caminhos de entrada devem ser strings."""
    with pytest.raises(TypeError):
        run_galaxy_split_experiment(
            dataset_path=tmp_path / "galaxies.csv",
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            output_directory=str(tmp_path),
        )

    with pytest.raises(TypeError):
        run_galaxy_split_experiment(
            dataset_path=str(tmp_path / "galaxies.csv"),
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            output_directory=tmp_path,
        )