import pandas as pd
import pytest

from src.data.galaxies import (
    FEATURE_COLUMNS,
    FINAL_COLUMNS,
    filter_valid_galaxies,
    load_galaxy_dataframe,
    load_prepared_galaxy_dataset,
    prepare_galaxy_dataframe,
    split_features_target,
)


def create_raw_galaxy_dataframe() -> pd.DataFrame:
    """Cria um dataset mínimo com linhas válidas e inválidas."""
    return pd.DataFrame(
        {
            "SequentialID": [1, 2, 3, 4, 5],
            "GG": [0.60, 0.41, 0.30, 0.72, 0.50],
            "M20": [-1.8, -1.2, -0.6, -1.9, -1.0],
            "CC": [3.8, 2.4, 1.7, 4.1, 2.1],
            "AA": [0.05, 0.20, 0.55, 0.03, 0.25],
            "SERSIC_N_GIM2D": [4.0, 1.2, 0.8, 4.5, 1.0],
            "ELL_GIM2D": [0.20, 0.55, 0.70, 0.10, 0.30],
            "R_GIM2D": [0.90, 1.40, 1.10, 0.80, 1.30],
            "TYPE": [1, 2, 3, 9, 2],
            "JUNKFLAG": [0, 0, 0, 0, 1],
            "ACS_CLEAN": [1, 1, 1, 1, 1],
        }
    )


def test_load_galaxy_dataframe_normalizes_columns(tmp_path) -> None:
    """O carregamento deve normalizar os nomes das colunas."""
    csv_path = tmp_path / "galaxies.csv"
    create_raw_galaxy_dataframe().to_csv(csv_path, index=False)

    dataframe = load_galaxy_dataframe(str(csv_path))

    assert "sequentialid" in dataframe.columns
    assert "gg" in dataframe.columns
    assert "type" in dataframe.columns


def test_load_galaxy_dataframe_rejects_missing_file(tmp_path) -> None:
    """Arquivos inexistentes devem gerar erro claro."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_galaxy_dataframe(str(missing_path))


def test_filter_valid_galaxies_removes_invalid_rows() -> None:
    """O filtro deve remover type 9 e objetos marcados como lixo."""
    dataframe = create_raw_galaxy_dataframe()
    dataframe.columns = [
        column.lower()
        for column in dataframe.columns
    ]

    filtered = filter_valid_galaxies(dataframe)

    assert filtered["sequentialid"].tolist() == [1, 2, 3]
    assert set(filtered["type"]) == {1, 2, 3}


def test_prepare_galaxy_dataframe_keeps_final_columns() -> None:
    """O DataFrame preparado deve conter apenas as colunas finais."""
    prepared = prepare_galaxy_dataframe(create_raw_galaxy_dataframe())

    assert prepared.columns.tolist() == FINAL_COLUMNS
    assert len(prepared) == 3


def test_prepare_galaxy_dataframe_rejects_missing_columns() -> None:
    """A preparação deve falhar quando uma feature obrigatória faltar."""
    dataframe = create_raw_galaxy_dataframe().drop(
        columns=["M20"],
    )

    with pytest.raises(ValueError):
        prepare_galaxy_dataframe(dataframe)


def test_split_features_target_excludes_identifier() -> None:
    """O identificador não deve entrar como feature do modelo."""
    prepared = prepare_galaxy_dataframe(create_raw_galaxy_dataframe())

    features, target = split_features_target(prepared)

    assert features.columns.tolist() == FEATURE_COLUMNS
    assert "sequentialid" not in features.columns
    assert target.tolist() == [1, 2, 3]


def test_load_prepared_galaxy_dataset_returns_features_target_and_data(
    tmp_path,
) -> None:
    """A função de alto nível deve retornar X, y e DataFrame preparado."""
    csv_path = tmp_path / "galaxies.csv"
    create_raw_galaxy_dataframe().to_csv(csv_path, index=False)

    features, target, prepared = load_prepared_galaxy_dataset(
        str(csv_path),
    )

    assert features.shape == (3, len(FEATURE_COLUMNS))
    assert target.tolist() == [1, 2, 3]
    assert prepared.columns.tolist() == FINAL_COLUMNS