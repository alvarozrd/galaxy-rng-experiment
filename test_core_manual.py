from src.rng.core import PCG64CPALite1


def test_same_seed_produces_same_sequence() -> None:
    """A mesma semente deve produzir exatamente a mesma sequência."""
    generator_a = PCG64CPALite1(seed=123456789)
    generator_b = PCG64CPALite1(seed=123456789)

    sequence_a = [generator_a.next_u64() for _ in range(10)]
    sequence_b = [generator_b.next_u64() for _ in range(10)]

    assert sequence_a == sequence_b


if __name__ == "__main__":
    test_same_seed_produces_same_sequence()
    print("Teste de reprodutibilidade concluído com sucesso.")