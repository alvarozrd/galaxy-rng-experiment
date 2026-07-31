import pytest

from src.rng.adapters import NumPyPCG64RNG


CANONICAL_SEED = 123456789


def test_numpy_pcg64_random_returns_float() -> None:
    """O adaptador deve retornar floats no intervalo [0, 1)."""
    rng = NumPyPCG64RNG(seed=CANONICAL_SEED)

    for _ in range(1000):
        value = rng.random()

        assert isinstance(value, float)
        assert 0.0 <= value < 1.0


def test_numpy_pcg64_integers_respects_interval() -> None:
    """Os inteiros devem respeitar o intervalo solicitado."""
    rng = NumPyPCG64RNG(seed=CANONICAL_SEED)

    for _ in range(1000):
        value = rng.integers(5, 10)

        assert isinstance(value, int)
        assert 5 <= value < 10


def test_numpy_pcg64_is_reproducible() -> None:
    """A mesma seed deve produzir a mesma sequência."""
    rng_a = NumPyPCG64RNG(seed=CANONICAL_SEED)
    rng_b = NumPyPCG64RNG(seed=CANONICAL_SEED)

    sequence_a = [rng_a.random() for _ in range(100)]
    sequence_b = [rng_b.random() for _ in range(100)]

    assert sequence_a == sequence_b


def test_numpy_pcg64_shuffle_preserves_elements() -> None:
    """O embaralhamento deve preservar todos os elementos."""
    rng = NumPyPCG64RNG(seed=CANONICAL_SEED)
    values = list(range(100))
    original_values = values.copy()

    rng.shuffle(values)

    assert sorted(values) == sorted(original_values)


def test_numpy_pcg64_permutation_preserves_indices() -> None:
    """A permutação deve conter todos os índices uma única vez."""
    rng = NumPyPCG64RNG(seed=CANONICAL_SEED)

    result = rng.permutation(100)

    assert isinstance(result, list)
    assert sorted(result) == list(range(100))


def test_numpy_pcg64_choice_without_replacement() -> None:
    """A seleção sem reposição não deve repetir elementos."""
    rng = NumPyPCG64RNG(seed=CANONICAL_SEED)
    values = list(range(100))

    result = rng.choice(values, size=50, replace=False)

    assert isinstance(result, list)
    assert len(result) == 50
    assert len(set(result)) == 50


def test_numpy_pcg64_rejects_invalid_seed() -> None:
    """A seed deve ser um número inteiro."""
    with pytest.raises(TypeError):
        NumPyPCG64RNG(seed=1.5)

    with pytest.raises(TypeError):
        NumPyPCG64RNG(seed=True)


def test_numpy_pcg64_rejects_invalid_inputs() -> None:
    """O adaptador deve rejeitar entradas inválidas."""
    rng = NumPyPCG64RNG(seed=CANONICAL_SEED)

    with pytest.raises(ValueError):
        rng.integers(10, 10)

    with pytest.raises(TypeError):
        rng.shuffle((1, 2, 3))

    with pytest.raises(ValueError):
        rng.permutation(-1)

    with pytest.raises(ValueError):
        rng.choice([])