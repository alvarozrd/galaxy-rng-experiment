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


if __name__ == "__main__":
    test_same_seed_produces_same_sequence()
    test_reference_sequence()
    test_different_seeds_produce_different_sequences()
    print("Três testes do núcleo concluídos com sucesso.")