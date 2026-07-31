"""
Fábrica para criação dos geradores utilizados no experimento.
"""

from src.rng.adapters import NumPyMT19937RNG, NumPyPCG64RNG
from src.rng.rng import PCG64CPALite1RNG


SUPPORTED_GENERATORS = (
    "pcg64_cp_alite1",
    "numpy_pcg64",
    "numpy_mt19937",
)


def create_rng(generator_name: str, seed: int):
    """
    Cria e retorna um gerador pseudoaleatório pelo nome.

    Geradores disponíveis:
        pcg64_cp_alite1
        numpy_pcg64
        numpy_mt19937
    """
    if not isinstance(generator_name, str):
        raise TypeError("generator_name deve ser uma string.")

    normalized_name = generator_name.strip().lower()

    if normalized_name == "pcg64_cp_alite1":
        return PCG64CPALite1RNG(seed=seed)

    if normalized_name == "numpy_pcg64":
        return NumPyPCG64RNG(seed=seed)

    if normalized_name == "numpy_mt19937":
        return NumPyMT19937RNG(seed=seed)

    supported_names = ", ".join(SUPPORTED_GENERATORS)

    raise ValueError(
        f"Gerador desconhecido: {generator_name}. "
        f"Geradores disponíveis: {supported_names}."
    )