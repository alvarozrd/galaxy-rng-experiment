import json

import pytest

from src.data.manifest import (
    create_split_manifest,
    load_split_manifest,
    save_split_manifest,
)


CANONICAL_SEED = 123456789


def create_example_split() -> dict:
    """Cria uma divisão simples para os testes."""
    return {
        "train": [0, 1, 2, 3, 4, 5],
        "validation": [6, 7],
        "test": [8, 9],
    }


def test_create_manifest_records_configuration() -> None:
    """O manifesto deve registrar os parâmetros da divisão."""
    split_indices = create_example_split()

    manifest = create_split_manifest(
        split_indices=split_indices,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
        stratified=False,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
    )

    assert manifest["generator_name"] == "pcg64_cp_alite1"
    assert manifest["seed"] == CANONICAL_SEED
    assert manifest["stratified"] is False
    assert manifest["dataset_size"] == 10


def test_create_manifest_records_split_sizes() -> None:
    """O manifesto deve registrar os tamanhos dos conjuntos."""
    manifest = create_split_manifest(
        split_indices=create_example_split(),
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
        stratified=False,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
    )

    assert manifest["split_sizes"] == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }


def test_create_manifest_records_indices() -> None:
    """O manifesto deve preservar todos os índices."""
    split_indices = create_example_split()

    manifest = create_split_manifest(
        split_indices=split_indices,
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
        stratified=False,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
    )

    assert manifest["indices"] == split_indices


def test_create_manifest_records_class_distribution() -> None:
    """O manifesto deve contar a distribuição das classes."""
    labels = (
        ["espiral"] * 5
        + ["eliptica"] * 3
        + ["irregular"] * 2
    )

    manifest = create_split_manifest(
        split_indices=create_example_split(),
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
        stratified=True,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
        labels=labels,
    )

    assert manifest["class_distribution"]["train"] == {
        "espiral": 5,
        "eliptica": 1,
    }

    assert manifest["class_distribution"]["validation"] == {
        "eliptica": 2,
    }

    assert manifest["class_distribution"]["test"] == {
        "irregular": 2,
    }


def test_save_and_load_manifest(tmp_path) -> None:
    """Um manifesto salvo deve ser carregado sem alterações."""
    manifest = create_split_manifest(
        split_indices=create_example_split(),
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
        stratified=False,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
    )

    output_path = tmp_path / "split_manifest.json"

    save_split_manifest(
        manifest=manifest,
        output_path=str(output_path),
    )

    loaded_manifest = load_split_manifest(
        output_path=str(output_path),
    )

    assert loaded_manifest == manifest


def test_saved_manifest_is_valid_json(tmp_path) -> None:
    """O arquivo salvo deve possuir formato JSON válido."""
    manifest = create_split_manifest(
        split_indices=create_example_split(),
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
        stratified=False,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
    )

    output_path = tmp_path / "manifest.json"

    save_split_manifest(
        manifest=manifest,
        output_path=str(output_path),
    )

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        loaded_json = json.load(file)

    assert loaded_json == manifest


def test_manifest_rejects_overlapping_indices() -> None:
    """Um índice não pode aparecer em mais de um conjunto."""
    invalid_split = {
        "train": [0, 1, 2],
        "validation": [2, 3],
        "test": [4],
    }

    with pytest.raises(ValueError):
        create_split_manifest(
            split_indices=invalid_split,
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            stratified=False,
            train_ratio=0.60,
            validation_ratio=0.20,
            test_ratio=0.20,
        )


def test_manifest_rejects_missing_keys() -> None:
    """O dicionário deve conter os três conjuntos obrigatórios."""
    invalid_split = {
        "train": [0, 1],
        "test": [2],
    }

    with pytest.raises(ValueError):
        create_split_manifest(
            split_indices=invalid_split,
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            stratified=False,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
        )


def test_load_rejects_missing_file(tmp_path) -> None:
    """O carregamento deve rejeitar arquivos inexistentes."""
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_split_manifest(
            output_path=str(missing_path),
        )