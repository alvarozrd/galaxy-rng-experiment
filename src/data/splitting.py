"""
Divisão reprodutível de datasets em treino, validação e teste.
"""

from typing import Dict, List

from src.rng.factory import create_rng


def split_dataset_indices(
    dataset_size: int,
    generator_name: str,
    seed: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, List[int]]:
    """
    Divide os índices de um dataset em treino, validação e teste.

    A ordem dos índices é definida pelo gerador pseudoaleatório escolhido.
    Com o mesmo gerador, seed e tamanho de dataset, a divisão é reproduzível.

    Args:
        dataset_size:
            Quantidade total de elementos do dataset.

        generator_name:
            Nome do gerador registrado na fábrica.

        seed:
            Semente utilizada pelo gerador.

        train_ratio:
            Proporção destinada ao conjunto de treino.

        validation_ratio:
            Proporção destinada ao conjunto de validação.

        test_ratio:
            Proporção destinada ao conjunto de teste.

    Returns:
        Dicionário contendo as listas de índices de treino,
        validação e teste.
    """
    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int):
        raise TypeError("dataset_size deve ser um número inteiro.")

    if dataset_size < 0:
        raise ValueError("dataset_size não pode ser negativo.")

    ratios = (
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    for ratio in ratios:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("As proporções devem ser números.")

        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                "Cada proporção deve estar no intervalo [0, 1]."
            )

    ratio_sum = train_ratio + validation_ratio + test_ratio

    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "A soma das proporções de treino, validação e teste "
            "deve ser igual a 1."
        )

    rng = create_rng(
        generator_name=generator_name,
        seed=seed,
    )

    shuffled_indices = rng.permutation(dataset_size)

    train_size = int(dataset_size * train_ratio)
    validation_size = int(dataset_size * validation_ratio)

    validation_start = train_size
    test_start = train_size + validation_size

    train_indices = shuffled_indices[:validation_start]

    validation_indices = shuffled_indices[
        validation_start:test_start
    ]

    test_indices = shuffled_indices[test_start:]

    return {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }