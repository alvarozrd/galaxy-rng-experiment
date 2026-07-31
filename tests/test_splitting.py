import pytest

from src.data.splitting import (
    split_dataset_indices,
    split_stratified_indices,
)


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

#testes para a função split_stratified_indices


def test_stratified_split_contains_all_indices() -> None:
    """A divisão estratificada deve preservar todas as amostras."""
    labels = (
        ["espiral"] * 60
        + ["eliptica"] * 30
        + ["irregular"] * 10
    )

    result = split_stratified_indices(
        labels=labels,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    all_indices = (
        result["train"]
        + result["validation"]
        + result["test"]
    )

    assert len(all_indices) == len(labels)
    assert sorted(all_indices) == list(range(len(labels)))


def test_stratified_split_has_no_overlap() -> None:
    """Uma amostra não pode aparecer em mais de um conjunto."""
    labels = (
        ["espiral"] * 60
        + ["eliptica"] * 30
        + ["irregular"] * 10
    )

    result = split_stratified_indices(
        labels=labels,
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
    )

    train = set(result["train"])
    validation = set(result["validation"])
    test = set(result["test"])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_stratified_split_preserves_class_distribution() -> None:
    """Cada conjunto deve receber amostras das diferentes classes."""
    labels = (
        ["espiral"] * 60
        + ["eliptica"] * 30
        + ["irregular"] * 10
    )

    result = split_stratified_indices(
        labels=labels,
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
    )

    def count_class(indices: list, class_name: str) -> int:
        return sum(
            labels[index] == class_name
            for index in indices
        )

    assert count_class(result["train"], "espiral") == 42
    assert count_class(result["train"], "eliptica") == 21
    assert count_class(result["train"], "irregular") == 7

    assert count_class(result["validation"], "espiral") == 9
    assert count_class(result["validation"], "eliptica") in (4, 5)
    assert count_class(result["validation"], "irregular") in (1, 2)


def test_stratified_split_is_reproducible() -> None:
    """A mesma configuração deve produzir a mesma divisão."""
    labels = ["espiral", "eliptica", "irregular"] * 40

    split_a = split_stratified_indices(
        labels=labels,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    split_b = split_stratified_indices(
        labels=labels,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    assert split_a == split_b


def test_stratified_split_changes_with_seed() -> None:
    """Seeds diferentes devem produzir divisões diferentes."""
    labels = ["espiral", "eliptica", "irregular"] * 40

    split_a = split_stratified_indices(
        labels=labels,
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
    )

    split_b = split_stratified_indices(
        labels=labels,
        generator_name="numpy_pcg64",
        seed=987654321,
    )

    assert split_a != split_b


def test_stratified_split_supports_small_classes() -> None:
    """Classes com poucas amostras não devem ser perdidas."""
    labels = [
        "espiral",
        "espiral",
        "eliptica",
        "irregular",
    ]

    result = split_stratified_indices(
        labels=labels,
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    all_indices = (
        result["train"]
        + result["validation"]
        + result["test"]
    )

    assert sorted(all_indices) == [0, 1, 2, 3]


def test_stratified_split_handles_empty_labels() -> None:
    """Uma lista vazia deve produzir três conjuntos vazios."""
    result = split_stratified_indices(
        labels=[],
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    assert result == {
        "train": [],
        "validation": [],
        "test": [],
    }


def test_stratified_split_rejects_invalid_labels() -> None:
    """labels deve ser uma lista com valores válidos."""
    with pytest.raises(TypeError):
        split_stratified_indices(
            labels=("espiral", "eliptica"),
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
        )

    with pytest.raises(TypeError):
        split_stratified_indices(
            labels=[["espiral"], ["eliptica"]],
            generator_name="pcg64_cp_alite1",
            seed=CANONICAL_SEED,
        )