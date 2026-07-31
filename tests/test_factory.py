import pytest

from src.rng import PCG64CPALite1RNG
from src.rng.adapters import NumPyMT19937RNG, NumPyPCG64RNG
from src.rng.factory import SUPPORTED_GENERATORS, create_rng


CANONICAL_SEED = 123456789


def test_factory_creates_pcg64_cp_alite1() -> None:
    """A fábrica deve criar o PCG64-CP A Lite 1."""
    rng = create_rng(
        generator_name="pcg64_cp_alite1",
        seed=CANONICAL_SEED,
    )

    assert isinstance(rng, PCG64CPALite1RNG)


def test_factory_creates_numpy_pcg64() -> None:
    """A fábrica deve criar o PCG64 do NumPy."""
    rng = create_rng(
        generator_name="numpy_pcg64",
        seed=CANONICAL_SEED,
    )

    assert isinstance(rng, NumPyPCG64RNG)


def test_factory_creates_numpy_mt19937() -> None:
    """A fábrica deve criar o MT19937 do NumPy."""
    rng = create_rng(
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
    )

    assert isinstance(rng, NumPyMT19937RNG)


def test_factory_normalizes_generator_name() -> None:
    """A fábrica deve ignorar espaços e diferenças entre maiúsculas."""
    rng = create_rng(
        generator_name="  NUMPY_PCG64  ",
        seed=CANONICAL_SEED,
    )

    assert isinstance(rng, NumPyPCG64RNG)


def test_factory_created_generators_are_reproducible() -> None:
    """A mesma configuração deve produzir a mesma sequência."""
    rng_a = create_rng(
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
    )
    rng_b = create_rng(
        generator_name="numpy_mt19937",
        seed=CANONICAL_SEED,
    )

    sequence_a = [rng_a.random() for _ in range(100)]
    sequence_b = [rng_b.random() for _ in range(100)]

    assert sequence_a == sequence_b


def test_factory_rejects_unknown_generator() -> None:
    """A fábrica deve rejeitar nomes de geradores desconhecidos."""
    with pytest.raises(ValueError):
        create_rng(
            generator_name="gerador_inexistente",
            seed=CANONICAL_SEED,
        )


def test_factory_rejects_non_string_name() -> None:
    """O nome do gerador deve ser uma string."""
    with pytest.raises(TypeError):
        create_rng(
            generator_name=123,
            seed=CANONICAL_SEED,
        )


def test_supported_generators_contains_expected_names() -> None:
    """A lista pública deve conter os três geradores do experimento."""
    assert SUPPORTED_GENERATORS == (
        "pcg64_cp_alite1",
        "numpy_pcg64",
        "numpy_mt19937",
    )