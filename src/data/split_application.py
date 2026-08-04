"""
Aplicação de manifestos de divisão sobre datasets tabulares.
"""

from typing import Dict, Tuple

import pandas as pd


SplitDataset = Dict[str, Tuple[pd.DataFrame, pd.Series]]


def apply_split_manifest(
    features: pd.DataFrame,
    target: pd.Series,
    manifest: dict,
) -> SplitDataset:
    """
    Aplica os índices de um manifesto sobre X e y.

    Retorna um dicionário com os conjuntos:
        - train
        - validation
        - test
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features deve ser um pandas.DataFrame.")

    if not isinstance(target, pd.Series):
        raise TypeError("target deve ser um pandas.Series.")

    if not isinstance(manifest, dict):
        raise TypeError("manifest deve ser um dicionário.")

    if len(features) != len(target):
        raise ValueError(
            "features e target devem possuir o mesmo tamanho."
        )

    if "indices" not in manifest:
        raise ValueError("manifest deve possuir a chave 'indices'.")

    indices = manifest["indices"]

    required_keys = {
        "train",
        "validation",
        "test",
    }

    if set(indices.keys()) != required_keys:
        raise ValueError(
            "manifest['indices'] deve conter exatamente as chaves "
            "'train', 'validation' e 'test'."
        )

    split_data: SplitDataset = {}

    for split_name in ["train", "validation", "test"]:
        split_indices = indices[split_name]

        if not isinstance(split_indices, list):
            raise TypeError(
                f"Os índices de '{split_name}' devem estar em uma lista."
            )

        for index in split_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(
                    f"Todos os índices de '{split_name}' devem ser inteiros."
                )

            if index < 0 or index >= len(features):
                raise ValueError(
                    f"Índice fora do intervalo em '{split_name}': {index}."
                )

        split_features = features.iloc[split_indices].reset_index(
            drop=True,
        )

        split_target = target.iloc[split_indices].reset_index(
            drop=True,
        )

        split_data[split_name] = (
            split_features,
            split_target,
        )

    return split_data