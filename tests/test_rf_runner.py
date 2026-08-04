import json

import pandas as pd
import pytest

from src.experiments.rf_runner import (
    DEFAULT_MODEL_RANDOM_STATE,
    run_batch_random_forest_experiments,
    run_random_forest_experiment,
)


CANONICAL_SEED = 123456789


def create_raw_galaxy_dataframe() -> pd.DataFrame:
    """Cria um dataset simples, balanceado e separável."""
    rows = []

    class_patterns = {
        1: {
            "gg": 0.15,
            "m20": -1.90,
            "cc": 4.10,
            "aa": 0.05,
            "sersic_n_gim2d": 4.0,
            "ell_gim2d": 0.20,
            "r_gim2d": 0.80,
        },
        2: {
            "gg": 0.55,
            "m20": -1.10,
            "cc": 2.30,
            "aa": 0.20,
            "sersic_n_gim2d": 1.5,
            "ell_gim2d": 0.45,
            "r_gim2d": 1.40,
        },
        3: {
            "gg": 0.85,
            "m20": -0.55,
            "cc": 1.30,
            "aa": 0.55,
            "sersic_n_gim2d": 0.8,
            "ell_gim2d": 0.70,
            "r_gim2d": 1.00,
        },
    }

    sequential_id = 1

    for label, pattern in class_patterns.items():
        for offset in range(8):
            row = {
                "SequentialID": sequential_id,
                "TYPE": label,
                "JUNKFLAG": 0,
                "ACS_CLEAN": 1,
            }

            for column, value in pattern.items():
                row[column.upper()] = value + offset / 1000

            rows.append(row)
            sequential_id += 1

    rows.append(
        {
            "SequentialID": sequential_id,
            "GG": 0.50,
            "M20": -1.00,
            "CC": 2.00,
            "AA": 0.20,
            "SERSIC_N_GIM2D": 1.20,
            "ELL_GIM2D": 0.30,
            "R_GIM2D": 1.10,
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


def test_run_random_forest_experiment_returns_complete_record(
    tmp_path,
) -> None:
    """Uma execução deve retornar manifesto, dataset e métricas."""
    dataset_path = save_dataset(tmp_path)

    result = run_random_forest_experiment(
        dataset_path=dataset_path,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
        output_directory=str(tmp_path / "results"),
    )

    assert result["metadata"]["generator_name"] == "pcg64_cp_alite1"
    assert result["dataset"]["sample_count"] == 24
    assert result["manifest"]["dataset_size"] == 24
    assert result["model"]["name"] == "RandomForestClassifier"
    assert result["model"]["random_state"] == DEFAULT_MODEL_RANDOM_STATE
    assert "validation" in result["model"]["results"]
    assert "test" in result["model"]["results"]


def test_run_random_forest_experiment_saves_json(tmp_path) -> None:
    """Uma execução deve salvar o resultado completo em JSON."""
    dataset_path = save_dataset(tmp_path)
    output_directory = tmp_path / "results"

    run_random_forest_experiment(
        dataset_path=dataset_path,
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
        output_directory=str(output_directory),
    )

    output_file = (
        output_directory
        / "numpy_pcg64"
        / f"seed_{CANONICAL_SEED}_random_forest.json"
    )

    assert output_file.exists()

    with output_file.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        saved_data = json.load(file)

    assert saved_data["model"]["name"] == "RandomForestClassifier"
    assert saved_data["model"]["random_state"] == DEFAULT_MODEL_RANDOM_STATE
    assert "accuracy" in saved_data["model"]["results"]["test"]


def test_run_random_forest_experiment_uses_fixed_model_random_state(
    tmp_path,
) -> None:
    """A seed experimental não deve alterar o random_state do modelo."""
    dataset_path = save_dataset(tmp_path)

    result_a = run_random_forest_experiment(
        dataset_path=dataset_path,
        generator_name="numpy_mt19937",
        seed=111,
        output_directory=str(tmp_path / "run_a"),
    )

    result_b = run_random_forest_experiment(
        dataset_path=dataset_path,
        generator_name="numpy_mt19937",
        seed=222,
        output_directory=str(tmp_path / "run_b"),
    )

    assert result_a["model"]["random_state"] == DEFAULT_MODEL_RANDOM_STATE
    assert result_b["model"]["random_state"] == DEFAULT_MODEL_RANDOM_STATE


def test_run_batch_random_forest_experiments(tmp_path) -> None:
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

    results = run_batch_random_forest_experiments(
        dataset_path=dataset_path,
        output_directory=str(tmp_path / "results"),
        generators=generators,
        seeds=seeds,
    )

    assert set(results.keys()) == set(generators)

    for generator_name in generators:
        assert set(results[generator_name].keys()) == set(seeds)


def test_run_random_forest_experiment_rejects_invalid_paths(
    tmp_path,
) -> None:
    """Os caminhos de entrada devem ser strings."""
    with pytest.raises(TypeError):
        run_random_forest_experiment(
            dataset_path=tmp_path / "galaxies.csv",
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            output_directory=str(tmp_path),
        )

    with pytest.raises(TypeError):
        run_random_forest_experiment(
            dataset_path=str(tmp_path / "galaxies.csv"),
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            output_directory=tmp_path,
        )


def test_run_random_forest_experiment_rejects_invalid_model_random_state(
    tmp_path,
) -> None:
    """O random_state do modelo deve ser inteiro."""
    dataset_path = save_dataset(tmp_path)

    with pytest.raises(TypeError):
        run_random_forest_experiment(
            dataset_path=dataset_path,
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            output_directory=str(tmp_path),
            model_random_state="42",
        )
        