import json

import pytest

from src.experiments.runner import (
    DEFAULT_GENERATORS,
    DEFAULT_SEEDS,
    run_batch_split_experiments,
    run_split_experiment,
)


CANONICAL_SEED = 123456789


#ambiente de teste para o arquivo que atomatiza e linca os testes de execução do split, garantindo que o código funcione corretamente com diferentes geradores e sementes.

def create_example_labels() -> list:
    """Cria rótulos fictícios para os testes."""
    return (
        ["espiral"] * 20
        + ["eliptica"] * 10
        + ["irregular"] * 10
    )


def test_single_experiment_returns_record(tmp_path) -> None:
    """Uma execução deve retornar metadados e manifesto."""
    result = run_split_experiment(
        labels=create_example_labels(),
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
        output_directory=str(tmp_path),
    )

    assert "metadata" in result
    assert "manifest" in result

    assert result["metadata"]["generator_name"] == (
        "pcg64_cp_alite1"
    )

    assert result["manifest"]["seed"] == CANONICAL_SEED


def test_single_experiment_saves_json(tmp_path) -> None:
    """Uma execução deve salvar um arquivo JSON."""
    run_split_experiment(
        labels=create_example_labels(),
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
        output_directory=str(tmp_path),
    )

    output_file = (
        tmp_path
        / "numpy_pcg64"
        / f"seed_{CANONICAL_SEED}.json"
    )

    assert output_file.exists()

    with output_file.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        saved_data = json.load(file)

    assert saved_data["metadata"]["seed"] == CANONICAL_SEED


def test_single_experiment_is_reproducible(tmp_path) -> None:
    """A mesma configuração deve produzir o mesmo manifesto."""
    result_a = run_split_experiment(
        labels=create_example_labels(),
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
        output_directory=str(tmp_path / "run_a"),
    )

    result_b = run_split_experiment(
        labels=create_example_labels(),
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
        output_directory=str(tmp_path / "run_b"),
    )

    assert result_a["manifest"] == result_b["manifest"]


def test_batch_executes_all_combinations(tmp_path) -> None:
    """O lote deve executar todos os geradores e seeds."""
    generators = [
        "pcg64_cp_alite1",
        "numpy_pcg64",
    ]

    seeds = [
        123,
        456,
        789,
    ]

    results = run_batch_split_experiments(
        labels=create_example_labels(),
        output_directory=str(tmp_path),
        generators=generators,
        seeds=seeds,
    )

    assert set(results.keys()) == set(generators)

    for generator_name in generators:
        assert set(results[generator_name].keys()) == set(seeds)


def test_batch_saves_all_files(tmp_path) -> None:
    """Cada combinação deve gerar um arquivo separado."""
    generators = [
        "pcg64_cp_alite1",
        "numpy_mt19937",
    ]

    seeds = [
        100,
        200,
    ]

    run_batch_split_experiments(
        labels=create_example_labels(),
        output_directory=str(tmp_path),
        generators=generators,
        seeds=seeds,
    )

    for generator_name in generators:
        for seed in seeds:
            output_file = (
                tmp_path
                / generator_name
                / f"seed_{seed}.json"
            )

            assert output_file.exists()


def test_default_configuration_contains_expected_values() -> None:
    """A configuração padrão deve conter três geradores e cinco seeds."""
    assert DEFAULT_GENERATORS == [
        "pcg64_cp_alite1",
        "numpy_pcg64",
        "numpy_mt19937",
    ]

    assert DEFAULT_SEEDS == [
        123456789,
        987654321,
        20260429,
        3141592653,
        2718281828,
    ]


def test_runner_rejects_invalid_labels(tmp_path) -> None:
    """labels deve ser uma lista."""
    with pytest.raises(TypeError):
        run_split_experiment(
            labels=("espiral", "eliptica"),
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            output_directory=str(tmp_path),
        )


def test_batch_rejects_invalid_configuration(tmp_path) -> None:
    """Geradores e seeds devem ser listas."""
    labels = create_example_labels()

    with pytest.raises(TypeError):
        run_batch_split_experiments(
            labels=labels,
            output_directory=str(tmp_path),
            generators="numpy_pcg64",
        )

    with pytest.raises(TypeError):
        run_batch_split_experiments(
            labels=labels,
            output_directory=str(tmp_path),
            seeds=123456789,
        )
