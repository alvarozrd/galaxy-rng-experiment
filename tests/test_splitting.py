import pytest

from src.data.splitting import split_dataset_indices


CANONICAL_SEED = 123456789


def test_split_contains_all_indices() -> None:
    """A divisão deve preservar todos os índices do dataset."""
    result = split_dataset_indices(
        dataset_size=100,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    all_indices = (
        result["train"]
        + result["validation"]
        + result["test"]
    )

    assert len(all_indices) == 100
    assert sorted(all_indices) == list(range(100))


def test_split_has_no_overlapping_indices() -> None:
    """Um índice não pode aparecer em mais de um conjunto."""
    result = split_dataset_indices(
        dataset_size=100,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    train = set(result["train"])
    validation = set(result["validation"])
    test = set(result["test"])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_split_respects_default_sizes() -> None:
    """A divisão padrão deve usar 70%, 15% e 15%."""
    result = split_dataset_indices(
        dataset_size=100,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    assert len(result["train"]) == 70
    assert len(result["validation"]) == 15
    assert len(result["test"]) == 15


def test_split_is_reproducible() -> None:
    """A mesma configuração deve produzir a mesma divisão."""
    split_a = split_dataset_indices(
        dataset_size=100,
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
    )

    split_b = split_dataset_indices(
        dataset_size=100,
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
    )

    assert split_a == split_b


def test_split_changes_with_different_seeds() -> None:
    """Seeds diferentes devem normalmente alterar a divisão."""
    split_a = split_dataset_indices(
        dataset_size=100,
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
    )

    split_b = split_dataset_indices(
        dataset_size=100,
        generator_name="numpy_mt19937",
        seed=987654321,
    )

    assert split_a != split_b


def test_split_changes_with_different_generators() -> None:
    """Geradores diferentes devem produzir divisões distintas."""
    split_pcg64 = split_dataset_indices(
        dataset_size=100,
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
    )

    split_mt19937 = split_dataset_indices(
        dataset_size=100,
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
    )

    assert split_pcg64 != split_mt19937


def test_split_supports_custom_ratios() -> None:
    """A função deve aceitar proporções personalizadas."""
    result = split_dataset_indices(
        dataset_size=100,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
    )

    assert len(result["train"]) == 60
    assert len(result["validation"]) == 20
    assert len(result["test"]) == 20


def test_split_handles_empty_dataset() -> None:
    """Um dataset vazio deve gerar três conjuntos vazios."""
    result = split_dataset_indices(
        dataset_size=0,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    assert result == {
        "train": [],
        "validation": [],
        "test": [],
    }


def test_split_rejects_invalid_dataset_size() -> None:
    """O tamanho do dataset deve ser um inteiro não negativo."""
    with pytest.raises(TypeError):
        split_dataset_indices(
            dataset_size=100.0,
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
        )

    with pytest.raises(TypeError):
        split_dataset_indices(
            dataset_size=True,
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
        )

    with pytest.raises(ValueError):
        split_dataset_indices(
            dataset_size=-1,
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
        )


def test_split_rejects_invalid_ratios() -> None:
    """As proporções devem ser válidas e totalizar 1."""
    with pytest.raises(ValueError):
        split_dataset_indices(
            dataset_size=100,
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            train_ratio=0.80,
            validation_ratio=0.20,
            test_ratio=0.20,
        )

    with pytest.raises(ValueError):
        split_dataset_indices(
            dataset_size=100,
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            train_ratio=-0.10,
            validation_ratio=0.50,
            test_ratio=0.60,
        )

    with pytest.raises(TypeError):
        split_dataset_indices(
            dataset_size=100,
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
            train_ratio="0.70",
            validation_ratio=0.15,
            test_ratio=0.15,
        )