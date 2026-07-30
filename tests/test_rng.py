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

#testes para a implementação da função shuffle

def test_shuffle_preserves_all_elements() -> None:
    """O embaralhamento deve preservar todos os elementos da lista."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)
    values = list(range(100))
    original_values = values.copy()

    rng.shuffle(values)

    assert sorted(values) == sorted(original_values)
    assert len(values) == len(original_values)


def test_shuffle_modifies_list_in_place() -> None:
    """O método shuffle deve modificar a própria lista."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)
    values = list(range(20))

    result = rng.shuffle(values)

    assert result is None


def test_shuffle_is_reproducible() -> None:
    """A mesma seed deve produzir o mesmo embaralhamento."""
    rng_a = PCG64CPALite1RNG(seed=CANONICAL_SEED)
    rng_b = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    values_a = list(range(100))
    values_b = list(range(100))

    rng_a.shuffle(values_a)
    rng_b.shuffle(values_b)

    assert values_a == values_b


def test_shuffle_with_different_seeds_changes_order() -> None:
    """Seeds diferentes devem normalmente produzir ordens diferentes."""
    rng_a = PCG64CPALite1RNG(seed=CANONICAL_SEED)
    rng_b = PCG64CPALite1RNG(seed=987654321)

    values_a = list(range(100))
    values_b = list(range(100))

    rng_a.shuffle(values_a)
    rng_b.shuffle(values_b)

    assert values_a != values_b


def test_shuffle_handles_empty_and_single_item_lists() -> None:
    """Listas vazias ou unitárias devem permanecer válidas."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    empty_values = []
    single_value = ["galaxy"]

    rng.shuffle(empty_values)
    rng.shuffle(single_value)

    assert empty_values == []
    assert single_value == ["galaxy"]


def test_shuffle_rejects_non_list_values() -> None:
    """O método deve rejeitar objetos que não sejam listas."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    with pytest.raises(TypeError):
        rng.shuffle((1, 2, 3))

    with pytest.raises(TypeError):
        rng.shuffle("galaxy")

#testes para a função de permutação 


def test_permutation_preserves_all_indices() -> None:
    """A permutação deve conter todos os índices exatamente uma vez."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    result = rng.permutation(100)

    assert isinstance(result, list)
    assert len(result) == 100
    assert sorted(result) == list(range(100))


def test_permutation_is_reproducible() -> None:
    """A mesma seed deve produzir a mesma permutação."""
    rng_a = PCG64CPALite1RNG(seed=CANONICAL_SEED)
    rng_b = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    permutation_a = rng_a.permutation(100)
    permutation_b = rng_b.permutation(100)

    assert permutation_a == permutation_b


def test_permutation_with_different_seeds_changes_order() -> None:
    """Seeds diferentes devem normalmente produzir permutações diferentes."""
    rng_a = PCG64CPALite1RNG(seed=CANONICAL_SEED)
    rng_b = PCG64CPALite1RNG(seed=987654321)

    permutation_a = rng_a.permutation(100)
    permutation_b = rng_b.permutation(100)

    assert permutation_a != permutation_b


def test_permutation_handles_zero_and_one() -> None:
    """Os casos com zero ou um elemento devem ser tratados corretamente."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    assert rng.permutation(0) == []
    assert rng.permutation(1) == [0]


def test_permutation_rejects_invalid_values() -> None:
    """O tamanho da permutação deve ser um inteiro não negativo."""
    rng = PCG64CPALite1RNG(seed=CANONICAL_SEED)

    with pytest.raises(TypeError):
        rng.permutation(5.0)

    with pytest.raises(TypeError):
        rng.permutation(True)

    with pytest.raises(ValueError):
        rng.permutation(-1)