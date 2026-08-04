"""
Carregamento e validação do dataset COSMOS Zurich de galáxias.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


TARGET_COLUMN = "type"
IDENTIFIER_COLUMN = "sequentialid"

FEATURE_COLUMNS = [
    "gg",
    "m20",
    "cc",
    "aa",
    "sersic_n_gim2d",
    "ell_gim2d",
    "r_gim2d",
]

FINAL_COLUMNS = [
    IDENTIFIER_COLUMN,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
]


def load_galaxy_dataframe(path: str) -> pd.DataFrame:
    """
    Carrega um arquivo CSV de galáxias.

    Os nomes das colunas são normalizados para letras minúsculas para
    compatibilidade com o catálogo original, que usa nomes em maiúsculas.
    """
    if not isinstance(path, str):
        raise TypeError("path deve ser uma string.")

    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    dataframe = pd.read_csv(csv_path)
    dataframe.columns = [
        column.strip().lower()
        for column in dataframe.columns
    ]

    return dataframe


def validate_galaxy_columns(dataframe: pd.DataFrame) -> None:
    """
    Verifica se o DataFrame possui as colunas finais necessárias.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe deve ser um pandas.DataFrame.")

    missing_columns = [
        column
        for column in FINAL_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "O dataset não possui as colunas obrigatórias: "
            f"{missing_columns}."
        )


def filter_valid_galaxies(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica os filtros definidos para objetos válidos do catálogo.

    Regras vindas da documentação do dataset:
        - type != 9
        - junkflag == 0, quando a coluna existir
        - acs_clean == 1, quando a coluna existir

    A função aceita tanto o catálogo bruto quanto um CSV já pré-filtrado.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe deve ser um pandas.DataFrame.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"O dataset não possui a coluna alvo '{TARGET_COLUMN}'."
        )

    filtered = dataframe[dataframe[TARGET_COLUMN] != 9].copy()

    if "junkflag" in filtered.columns:
        filtered = filtered[filtered["junkflag"] == 0].copy()

    if "acs_clean" in filtered.columns:
        filtered = filtered[filtered["acs_clean"] == 1].copy()

    return filtered.reset_index(drop=True)


def prepare_galaxy_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra, valida e mantém apenas as colunas finais do experimento.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe deve ser um pandas.DataFrame.")

    normalized = dataframe.copy()
    normalized.columns = [
        column.strip().lower()
        for column in normalized.columns
    ]

    filtered = filter_valid_galaxies(normalized)
    validate_galaxy_columns(filtered)

    return filtered[FINAL_COLUMNS].reset_index(drop=True)


def split_features_target(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features morfológicas e rótulo de classificação.

    O identificador ``sequentialid`` é preservado no DataFrame preparado,
    mas não entra em X porque não representa uma característica morfológica.
    """
    validate_galaxy_columns(dataframe)

    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target


def load_prepared_galaxy_dataset(
    path: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Carrega o CSV, aplica filtros e retorna X, y e o DataFrame preparado.
    """
    dataframe = load_galaxy_dataframe(path)
    prepared = prepare_galaxy_dataframe(dataframe)
    features, target = split_features_target(prepared)

    return features, target, prepared