import re

import pytest

from src.experiments.metadata import (
    create_execution_metadata,
    get_git_commit,
)


CANONICAL_SEED = 123456789


def test_metadata_records_generator_and_seed() -> None:
    """Os metadados devem registrar gerador e seed."""
    metadata = create_execution_metadata(
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    assert metadata["generator_name"] == "pcg64_cp_alite1"
    assert metadata["seed"] == CANONICAL_SEED


def test_metadata_records_python_and_numpy_versions() -> None:
    """As versões do Python e NumPy devem ser registradas."""
    metadata = create_execution_metadata(
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
    )

    assert isinstance(metadata["python_version"], str)
    assert metadata["python_version"]

    assert isinstance(metadata["numpy_version"], str)
    assert metadata["numpy_version"]


def test_metadata_records_system_information() -> None:
    """Os dados básicos do sistema devem estar presentes."""
    metadata = create_execution_metadata(
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
    )

    assert isinstance(metadata["operating_system"], str)
    assert metadata["operating_system"]

    assert isinstance(metadata["machine"], str)
    assert metadata["machine"]


def test_metadata_records_timestamp() -> None:
    """O registro deve conter uma data e hora em formato ISO."""
    metadata = create_execution_metadata(
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    assert isinstance(metadata["timestamp_utc"], str)
    assert "T" in metadata["timestamp_utc"]


def test_git_commit_is_valid_or_none() -> None:
    """O hash do Git deve ser hexadecimal ou None."""
    commit = get_git_commit()

    if commit is not None:
        assert re.fullmatch(r"[0-9a-f]{40}", commit)


def test_metadata_rejects_invalid_generator_name() -> None:
    """O nome do gerador deve ser uma string."""
    with pytest.raises(TypeError):
        create_execution_metadata(
            generator_name=123,
            seed=CANONICAL_SEED,
        )


def test_metadata_rejects_invalid_seed() -> None:
    """A seed deve ser um número inteiro."""
    with pytest.raises(TypeError):
        create_execution_metadata(
            generator_name="pcg64_cp_alite1",
            seed=1.5,
        )

    with pytest.raises(TypeError):
        create_execution_metadata(
            generator_name="pcg64_cp_alite1",
            seed=True,
        )