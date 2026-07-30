from src.rng.core import PCG64CPALite1


def test_same_seed_produces_same_sequence() -> None:
    """A mesma semente deve produzir exatamente a mesma sequência."""
    generator_a = PCG64CPALite1(seed=123456789)
    generator_b = PCG64CPALite1(seed=123456789)

    sequence_a = [generator_a.next_u64() for _ in range(10)]
    sequence_b = [generator_b.next_u64() for _ in range(10)]

    assert sequence_a == sequence_b


def test_reference_sequence() -> None:
    """A seed canônica deve reproduzir a sequência de referência conhecida."""
    generator = PCG64CPALite1(seed=123456789)

    expected_sequence = [
        7397393886516089083,
        15394037957713609850,
        13855128423709291985,
        11898936944525465787,
        4729619433350351713,
    ]

    generated_sequence = [generator.next_u64() for _ in range(5)]

    assert generated_sequence == expected_sequence


def test_different_seeds_produce_different_sequences() -> None:
    """Sementes diferentes devem produzir sequências iniciais diferentes."""
    generator_a = PCG64CPALite1(seed=123456789)
    generator_b = PCG64CPALite1(seed=987654321)

    sequence_a = [generator_a.next_u64() for _ in range(10)]
    sequence_b = [generator_b.next_u64() for _ in range(10)]

    assert sequence_a != sequence_b


def test_next_u64_returns_values_in_valid_range() -> None:
    """Os valores gerados devem permanecer no intervalo de um uint64."""
    generator = PCG64CPALite1(seed=123456789)

    for _ in range(1000):
        value = generator.next_u64()

        assert isinstance(value, int)
        assert 0 <= value <= (2**64 - 1)

def test_random_float_returns_values_in_unit_interval() -> None:
    """Os valores em ponto flutuante devem pertencer ao intervalo [0, 1)."""
    generator = PCG64CPALite1(seed=123456789)

    for _ in range(1000):
        value = generator.random_float()

        assert isinstance(value, float)
        assert 0.0 <= value < 1.0