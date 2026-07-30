import pytest

from src.rng.rng import PCG64CPALite1RNG


CANONICAL_SEED = 123456789


def test_random_returns_float_in_unit_interval() -> None:
    """O método random deve retornar um float no intervalo [0, 1)."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    for _ in range(1000):
        value = rng.random()

        assert isinstance(value, float)
        assert 0.0 <= value < 1.0


def test_integers_returns_value_in_requested_interval() -> None:
    """O método integers deve respeitar o intervalo [low, high)."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    for _ in range(10_000):
        value = rng.integers(5, 10)

        assert isinstance(value, int)
        assert 5 <= value < 10


def test_integers_accepts_only_upper_bound() -> None:
    """Um único argumento deve ser interpretado como limite superior."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    for _ in range(1000):
        value = rng.integers(10)

        assert 0 <= value < 10


def test_integers_supports_negative_intervals() -> None:
    """O método deve funcionar com limites negativos."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    for _ in range(1000):
        value = rng.integers(-10, -2)

        assert -10 <= value < -2


def test_integers_is_reproducible() -> None:
    """A mesma seed deve produzir a mesma sequência de inteiros."""
    rng_a = PCG64CPALite1RNG(seed=CANONICAL_SEED)
    rng_b = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    sequence_a = [rng_a.integers(0, 100) for _ in range(100)]
    sequence_b = [rng_b.integers(0, 100) for _ in range(100)]

    assert sequence_a == sequence_b


def test_integers_rejects_empty_or_inverted_interval() -> None:
    """O limite superior deve ser maior que o inferior."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    with pytest.raises(ValueError):
        rng.integers(10, 10)

    with pytest.raises(ValueError):
        rng.integers(10, 5)

    with pytest.raises(ValueError):
        rng.integers(0)


def test_integers_rejects_non_integer_limits() -> None:
    """Os limites devem ser números inteiros."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    with pytest.raises(TypeError):
        rng.integers(0.0, 10)

    with pytest.raises(TypeError):
        rng.integers(0, 10.0)

    with pytest.raises(TypeError):
        rng.integers(True, 10)