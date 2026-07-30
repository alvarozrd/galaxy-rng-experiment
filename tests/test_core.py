from src.rng.core import PCG64CPALite1


CANONICAL_SEED = 123456789
ALTERNATIVE_SEED = 987654321
UINT64_MAX = (2**64) - 1


def test_same_seed_produces_same_sequence() -> None:
    """A mesma semente deve produzir exatamente a mesma sequência."""
    generator_a = PCG64CPALite1(seed=CANONICAL_SEED)
    generator_b = PCG64CPALite1(seed=CANONICAL_SEED)

    sequence_a = [generator_a.next_u64() for _ in range(100)]
    sequence_b = [generator_b.next_u64() for _ in range(100)]

    assert sequence_a == sequence_b


def test_reference_sequence() -> None:
    """A seed canônica deve reproduzir a sequência de referência conhecida."""
    generator = PCG64CPALite1(seed=CANONICAL_SEED)

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
    generator_a = PCG64CPALite1(seed=CANONICAL_SEED)
    generator_b = PCG64CPALite1(seed=ALTERNATIVE_SEED)

    sequence_a = [generator_a.next_u64() for _ in range(100)]
    sequence_b = [generator_b.next_u64() for _ in range(100)]

    assert sequence_a != sequence_b


def test_next_u64_returns_values_in_valid_range() -> None:
    """Os valores gerados devem ser inteiros válidos de 64 bits sem sinal."""
    generator = PCG64CPALite1(seed=CANONICAL_SEED)

    for _ in range(10_000):
        value = generator.next_u64()

        assert isinstance(value, int)
        assert 0 <= value <= UINT64_MAX


def test_random_float_returns_values_in_unit_interval() -> None:
    """Os floats gerados devem pertencer ao intervalo [0, 1)."""
    generator = PCG64CPALite1(seed=CANONICAL_SEED)

    for _ in range(10_000):
        value = generator.random_float()

        assert isinstance(value, float)
        assert 0.0 <= value < 1.0


def test_generate_u64_returns_requested_quantity() -> None:
    """A geração em lote deve retornar a quantidade solicitada de inteiros."""
    generator = PCG64CPALite1(seed=CANONICAL_SEED)

    values = generator.gerar_u64(25)

    assert isinstance(values, list)
    assert len(values) == 25
    assert all(isinstance(value, int) for value in values)
    assert all(0 <= value <= UINT64_MAX for value in values)


def test_generate_float_returns_requested_quantity() -> None:
    """A geração em lote deve retornar a quantidade solicitada de floats."""
    generator = PCG64CPALite1(seed=CANONICAL_SEED)

    values = generator.gerar_float(25)

    assert isinstance(values, list)
    assert len(values) == 25
    assert all(isinstance(value, float) for value in values)
    assert all(0.0 <= value < 1.0 for value in values)


def test_zero_quantity_returns_empty_lists() -> None:
    """Solicitar zero elementos deve retornar listas vazias."""
    generator = PCG64CPALite1(seed=CANONICAL_SEED)

    assert generator.gerar_u64(0) == []
    assert generator.gerar_float(0) == []


def test_batch_u64_matches_sequential_generation() -> None:
    """A geração em lote deve preservar a sequência de chamadas individuais."""
    batch_generator = PCG64CPALite1(seed=CANONICAL_SEED)
    sequential_generator = PCG64CPALite1(seed=CANONICAL_SEED)

    batch_values = batch_generator.gerar_u64(50)
    sequential_values = [sequential_generator.next_u64() for _ in range(50)]

    assert batch_values == sequential_values


def test_batch_float_matches_sequential_generation() -> None:
    """A geração em lote de floats deve preservar as chamadas individuais."""
    batch_generator = PCG64CPALite1(seed=CANONICAL_SEED)
    sequential_generator = PCG64CPALite1(seed=CANONICAL_SEED)

    batch_values = batch_generator.gerar_float(50)
    sequential_values = [sequential_generator.random_float() for _ in range(50)]

    assert batch_values == sequential_values


def test_generator_state_advances_after_each_call() -> None:
    """Chamadas consecutivas devem avançar o estado interno do gerador."""
    generator = PCG64CPALite1(seed=CANONICAL_SEED)

    first_value = generator.next_u64()
    second_value = generator.next_u64()

    assert first_value != second_value