import pandas as pd
import pytest

from src.data.split_application import apply_split_manifest


def create_features() -> pd.DataFrame:
    """Cria features tabulares simples para teste."""
    return pd.DataFrame(
        {
            "gg": [0.1, 0.2, 0.3, 0.4, 0.5],
            "m20": [-1.0, -1.1, -1.2, -1.3, -1.4],
        }
    )


def create_target() -> pd.Series:
    """Cria rótulos simples para teste."""
    return pd.Series(
        [1, 2, 1, 3, 2],
        name="type",
    )


def create_manifest() -> dict:
    """Cria um manifesto mínimo com índices de split."""
    return {
        "indices": {
            "train": [0, 2, 4],
            "validation": [1],
            "test": [3],
        }
    }


def test_apply_split_manifest_returns_expected_splits() -> None:
    """A função deve aplicar os índices nos conjuntos corretos."""
    split_data = apply_split_manifest(
        features=create_features(),
        target=create_target(),
        manifest=create_manifest(),
    )

    train_features, train_target = split_data["train"]
    validation_features, validation_target = split_data["validation"]
    test_features, test_target = split_data["test"]

    assert train_features["gg"].tolist() == [0.1, 0.3, 0.5]
    assert train_target.tolist() == [1, 1, 2]

    assert validation_features["gg"].tolist() == [0.2]
    assert validation_target.tolist() == [2]

    assert test_features["gg"].tolist() == [0.4]
    assert test_target.tolist() == [3]


def test_apply_split_manifest_resets_indices() -> None:
    """Os subconjuntos retornados devem ter índice reiniciado."""
    split_data = apply_split_manifest(
        features=create_features(),
        target=create_target(),
        manifest=create_manifest(),
    )

    train_features, train_target = split_data["train"]

    assert train_features.index.tolist() == [0, 1, 2]
    assert train_target.index.tolist() == [0, 1, 2]


def test_apply_split_manifest_rejects_invalid_features() -> None:
    """features deve ser um DataFrame."""
    with pytest.raises(TypeError):
        apply_split_manifest(
            features=[[0.1], [0.2]],
            target=create_target(),
            manifest=create_manifest(),
        )


def test_apply_split_manifest_rejects_invalid_target() -> None:
    """target deve ser uma Series."""
    with pytest.raises(TypeError):
        apply_split_manifest(
            features=create_features(),
            target=[1, 2, 3],
            manifest=create_manifest(),
        )


def test_apply_split_manifest_rejects_different_lengths() -> None:
    """features e target devem ter o mesmo tamanho."""
    with pytest.raises(ValueError):
        apply_split_manifest(
            features=create_features(),
            target=pd.Series([1, 2]),
            manifest=create_manifest(),
        )


def test_apply_split_manifest_rejects_missing_indices() -> None:
    """O manifesto deve possuir a chave indices."""
    with pytest.raises(ValueError):
        apply_split_manifest(
            features=create_features(),
            target=create_target(),
            manifest={},
        )


def test_apply_split_manifest_rejects_invalid_split_keys() -> None:
    """O manifesto deve conter train, validation e test."""
    manifest = {
        "indices": {
            "train": [0],
            "test": [1],
        }
    }

    with pytest.raises(ValueError):
        apply_split_manifest(
            features=create_features(),
            target=create_target(),
            manifest=manifest,
        )


def test_apply_split_manifest_rejects_non_list_indices() -> None:
    """Cada conjunto de índices deve ser uma lista."""
    manifest = {
        "indices": {
            "train": [0],
            "validation": (1,),
            "test": [2],
        }
    }

    with pytest.raises(TypeError):
        apply_split_manifest(
            features=create_features(),
            target=create_target(),
            manifest=manifest,
        )


def test_apply_split_manifest_rejects_non_integer_indices() -> None:
    """Todos os índices devem ser inteiros."""
    manifest = {
        "indices": {
            "train": [0, 1.2],
            "validation": [2],
            "test": [3],
        }
    }

    with pytest.raises(TypeError):
        apply_split_manifest(
            features=create_features(),
            target=create_target(),
            manifest=manifest,
        )


def test_apply_split_manifest_rejects_out_of_range_indices() -> None:
    """Índices fora do tamanho do dataset devem gerar erro."""
    manifest = {
        "indices": {
            "train": [0, 5],
            "validation": [1],
            "test": [2],
        }
    }

    with pytest.raises(ValueError):
        apply_split_manifest(
            features=create_features(),
            target=create_target(),
            manifest=manifest,
        )   