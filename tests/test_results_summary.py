import json

import pandas as pd
import pytest

from src.analysis.results_summary import (
    create_results_dataframe,
    extract_result_row,
    find_result_files,
    load_experiment_result,
    save_results_tables,
    summarize_by_generator,
)


def create_result(
    generator_name: str = "numpy_pcg64",
    seed: int = 123,
    accuracy: float = 0.90,
    f1_macro: float = 0.80,
) -> dict:
    """Cria resultado experimental mínimo para teste."""
    return {
        "metadata": {
            "generator_name": generator_name,
            "seed": seed,
        },
        "manifest": {
            "dataset_size": 100,
            "split_sizes": {
                "train": 70,
                "validation": 15,
                "test": 15,
            },
        },
        "model": {
            "name": "RandomForestClassifier",
            "random_state": 42,
            "results": {
                "validation": {
                    "accuracy": accuracy,
                    "precision_macro": 0.81,
                    "recall_macro": 0.82,
                    "f1_macro": f1_macro,
                    "f1_weighted": 0.88,
                },
                "test": {
                    "accuracy": accuracy - 0.01,
                    "precision_macro": 0.79,
                    "recall_macro": 0.80,
                    "f1_macro": f1_macro - 0.02,
                    "f1_weighted": 0.86,
                },
            },
        },
    }


def save_result(path, result: dict) -> None:
    """Salva um resultado JSON para teste."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4,
        )


def test_load_experiment_result(tmp_path) -> None:
    """A função deve carregar um resultado JSON."""
    result_path = tmp_path / "seed_123_random_forest.json"
    save_result(result_path, create_result())

    result = load_experiment_result(str(result_path))

    assert result["metadata"]["generator_name"] == "numpy_pcg64"


def test_load_experiment_result_rejects_missing_file(tmp_path) -> None:
    """Arquivo inexistente deve gerar erro claro."""
    with pytest.raises(FileNotFoundError):
        load_experiment_result(
            str(tmp_path / "missing.json"),
        )


def test_extract_result_row() -> None:
    """A extração deve gerar uma linha tabular."""
    row = extract_result_row(
        create_result(
            generator_name="numpy_mt19937",
            seed=456,
            accuracy=0.91,
            f1_macro=0.83,
        )
    )

    assert row["generator_name"] == "numpy_mt19937"
    assert row["seed"] == 456
    assert row["dataset_size"] == 100
    assert row["train_size"] == 70
    assert row["validation_accuracy"] == 0.91
    assert row["test_accuracy"] == 0.90

    #precisou substituir o pytest.approx para evitar erro de arredondamento
    assert row["test_f1_macro"] == pytest.approx(0.81)


def test_find_result_files(tmp_path) -> None:
    """A busca deve encontrar apenas resultados Random Forest."""
    save_result(
        tmp_path / "numpy_pcg64" / "seed_123_random_forest.json",
        create_result(),
    )

    save_result(
        tmp_path / "numpy_pcg64" / "seed_123.json",
        create_result(),
    )

    files = find_result_files(str(tmp_path))

    assert len(files) == 1
    assert files[0].name == "seed_123_random_forest.json"


def test_create_results_dataframe(tmp_path) -> None:
    """O DataFrame deve ter uma linha por execução."""
    save_result(
        tmp_path / "numpy_pcg64" / "seed_123_random_forest.json",
        create_result(
            generator_name="numpy_pcg64",
            seed=123,
        ),
    )

    save_result(
        tmp_path / "numpy_mt19937" / "seed_456_random_forest.json",
        create_result(
            generator_name="numpy_mt19937",
            seed=456,
            accuracy=0.92,
            f1_macro=0.84,
        ),
    )

    dataframe = create_results_dataframe(str(tmp_path))

    assert len(dataframe) == 2
    assert set(dataframe["generator_name"]) == {
        "numpy_pcg64",
        "numpy_mt19937",
    }


def test_summarize_by_generator() -> None:
    """O resumo deve agrupar métricas por gerador."""
    dataframe = pd.DataFrame(
        [
            extract_result_row(
                create_result(
                    generator_name="numpy_pcg64",
                    seed=1,
                    accuracy=0.90,
                    f1_macro=0.80,
                )
            ),
            extract_result_row(
                create_result(
                    generator_name="numpy_pcg64",
                    seed=2,
                    accuracy=0.92,
                    f1_macro=0.82,
                )
            ),
            extract_result_row(
                create_result(
                    generator_name="numpy_mt19937",
                    seed=1,
                    accuracy=0.88,
                    f1_macro=0.78,
                )
            ),
        ]
    )

    summary = summarize_by_generator(dataframe)

    assert len(summary) == 2

    pcg64_summary = summary[
        summary["generator_name"] == "numpy_pcg64"
    ].iloc[0]

    assert pcg64_summary["validation_accuracy_mean"] == pytest.approx(
        0.91,
    )

    assert pcg64_summary["validation_f1_macro_mean"] == pytest.approx(
        0.81,
    )


def test_save_results_tables(tmp_path) -> None:
    """As tabelas finais devem ser salvas em CSV."""
    dataframe = pd.DataFrame(
        [
            extract_result_row(create_result()),
        ]
    )
    summary = summarize_by_generator(dataframe)

    save_results_tables(
        results_dataframe=dataframe,
        generator_summary=summary,
        output_directory=str(tmp_path),
    )

    assert (tmp_path / "random_forest_runs.csv").exists()
    assert (
        tmp_path / "random_forest_summary_by_generator.csv"
    ).exists()


def test_summarize_by_generator_rejects_invalid_input() -> None:
    """O resumo exige DataFrame."""
    with pytest.raises(TypeError):
        summarize_by_generator([])
