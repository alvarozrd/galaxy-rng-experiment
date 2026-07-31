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

# função p/ compatibilidade com versão extratificada de dados. Exemplo: se 50% das imagens forem espirais, cada subconjunto deverá manter aproximadamente 50% de espirais.

def _allocate_class_counts(
    class_size: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> List[int]:
    """
    Calcula quantos elementos de uma classe serão destinados a cada conjunto.

    O método utiliza as partes decimais para distribuir os elementos
    restantes, evitando que amostras sejam perdidas por arredondamento.
    """
    ratios = [
        train_ratio,
        validation_ratio,
        test_ratio,
    ]

    exact_counts = [
        class_size * ratio
        for ratio in ratios
    ]

    allocated_counts = [
        int(count)
        for count in exact_counts
    ]

    remaining = class_size - sum(allocated_counts)

    fractional_parts = [
        exact_counts[index] - allocated_counts[index]
        for index in range(3)
    ]

    priority_order = sorted(
        range(3),
        key=lambda index: (
            fractional_parts[index],
            ratios[index],
            -index,
        ),
        reverse=True,
    )

    for index in priority_order[:remaining]:
        allocated_counts[index] += 1

    return allocated_counts


def split_stratified_indices(
    labels: list,
    generator_name: str,
    seed: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, List[int]]:
    """
    Divide os índices de um dataset preservando a proporção das classes.

    Cada posição da lista labels representa a classe de uma amostra.
    Por exemplo:

        ["espiral", "eliptica", "espiral", "irregular"]

    O resultado contém os índices destinados aos conjuntos de treino,
    validação e teste.
    """
    if not isinstance(labels, list):
        raise TypeError("labels deve ser uma lista.")

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

    class_indices = {}

    for index, label in enumerate(labels):
        try:
            class_indices.setdefault(label, []).append(index)
        except TypeError as error:
            raise TypeError(
                "Cada label deve ser um valor utilizável como chave."
            ) from error

    rng = create_rng(
        generator_name=generator_name,
        seed=seed,
    )

    train_indices = []
    validation_indices = []
    test_indices = []

    for indices in class_indices.values():
        permutation = rng.permutation(len(indices))

        shuffled_class_indices = [
            indices[position]
            for position in permutation
        ]

        train_size, validation_size, test_size = _allocate_class_counts(
            class_size=len(indices),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )

        validation_start = train_size
        test_start = train_size + validation_size

        train_indices.extend(
            shuffled_class_indices[:validation_start]
        )

        validation_indices.extend(
            shuffled_class_indices[validation_start:test_start]
        )

        test_indices.extend(
            shuffled_class_indices[
                test_start:test_start + test_size
            ]
        )

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)

    return {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }

