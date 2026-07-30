from src.rng.rng import PCG64CPALite1RNG


def test_random_returns_float_in_unit_interval() -> None:
    """O método random deve retornar um float no intervalo [0, 1)."""
    rng = PCG64CPALite1RNG(seed=123456789)

    for _ in range(1000):
        value = rng.random()

        assert isinstance(value, float)
        assert 0.0 <= value < 1.0