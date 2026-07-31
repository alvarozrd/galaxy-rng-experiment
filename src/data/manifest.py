"""
Criação e armazenamento de manifestos de divisão do dataset.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def create_split_manifest(
    split_indices: Dict[str, List[int]],
    generator_name: str,
    seed: int,
    stratified: bool,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    labels: Optional[List[str]] = None,
) -> dict:
    """
    Cria um manifesto com os parâmetros e índices de uma divisão.

    O manifesto registra informações suficientes para documentar e
    reproduzir a separação do dataset.
    """
    if not isinstance(split_indices, dict):
        raise TypeError("split_indices deve ser um dicionário.")

    required_keys = {
        "train",
        "validation",
        "test",
    }

    if set(split_indices.keys()) != required_keys:
        raise ValueError(
            "split_indices deve conter exatamente as chaves "
            "'train', 'validation' e 'test'."
        )

    for key in required_keys:
        if not isinstance(split_indices[key], list):
            raise TypeError(
                f"O conjunto '{key}' deve ser representado por uma lista."
            )

        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in split_indices[key]
        ):
            raise TypeError(
                f"Todos os índices de '{key}' devem ser inteiros."
            )

    if not isinstance(generator_name, str):
        raise TypeError("generator_name deve ser uma string.")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um número inteiro.")

    if not isinstance(stratified, bool):
        raise TypeError("stratified deve ser um valor booleano.")

    if labels is not None and not isinstance(labels, list):
        raise TypeError("labels deve ser uma lista ou None.")

    all_indices = (
        split_indices["train"]
        + split_indices["validation"]
        + split_indices["test"]
    )

    if len(all_indices) != len(set(all_indices)):
        raise ValueError(
            "Os conjuntos de treino, validação e teste não podem "
            "possuir índices repetidos."
        )

    manifest = {
        "generator_name": generator_name,
        "seed": seed,
        "stratified": stratified,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "dataset_size": len(all_indices),
        "split_sizes": {
            "train": len(split_indices["train"]),
            "validation": len(split_indices["validation"]),
            "test": len(split_indices["test"]),
        },
        "indices": {
            "train": split_indices["train"],
            "validation": split_indices["validation"],
            "test": split_indices["test"],
        },
    }

    if labels is not None:
        manifest["class_distribution"] = {
            "train": _count_labels(
                split_indices["train"],
                labels,
            ),
            "validation": _count_labels(
                split_indices["validation"],
                labels,
            ),
            "test": _count_labels(
                split_indices["test"],
                labels,
            ),
        }

    return manifest


def _count_labels(
    indices: List[int],
    labels: List[str],
) -> Dict[str, int]:
    """Conta quantas amostras de cada classe existem em um conjunto."""
    distribution: Dict[str, int] = {}

    for index in indices:
        if index < 0 or index >= len(labels):
            raise ValueError(
                "Foi encontrado um índice fora do intervalo de labels."
            )

        label = labels[index]
        distribution[label] = distribution.get(label, 0) + 1

    return distribution


def save_split_manifest(
    manifest: dict,
    output_path: str,
) -> None:
    """Salva um manifesto em um arquivo JSON."""
    if not isinstance(manifest, dict):
        raise TypeError("manifest deve ser um dicionário.")

    if not isinstance(output_path, str):
        raise TypeError("output_path deve ser uma string.")

    path = Path(output_path)

    if path.parent != Path("."):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=4,
        )


def load_split_manifest(output_path: str) -> dict:
    """Carrega um manifesto salvo em JSON."""
    if not isinstance(output_path, str):
        raise TypeError("output_path deve ser uma string.")

    path = Path(output_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {output_path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)