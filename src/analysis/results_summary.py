"""
Resumo tabular dos resultados experimentais.
"""

import json
from pathlib import Path
from typing import List

import pandas as pd


def load_experiment_result(path: str) -> dict:
    """
    Carrega um arquivo JSON de resultado experimental.
    """
    if not isinstance(path, str):
        raise TypeError("path deve ser uma string.")

    result_path = Path(path)

    if not result_path.exists():
        raise FileNotFoundError(f"Resultado não encontrado: {path}")

    with result_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def extract_result_row(result: dict) -> dict:
    """
    Extrai uma linha tabular de um resultado completo.
    """
    if not isinstance(result, dict):
        raise TypeError("result deve ser um dicionário.")

    metadata = result["metadata"]
    manifest = result["manifest"]
    model = result["model"]
    model_results = model["results"]

    validation = model_results["validation"]
    test = model_results["test"]

    return {
        "generator_name": metadata["generator_name"],
        "seed": metadata["seed"],
        "dataset_size": manifest["dataset_size"],
        "train_size": manifest["split_sizes"]["train"],
        "validation_size": manifest["split_sizes"]["validation"],
        "test_size": manifest["split_sizes"]["test"],
        "model_name": model["name"],
        "model_random_state": model["random_state"],
        "validation_accuracy": validation["accuracy"],
        "validation_precision_macro": validation["precision_macro"],
        "validation_recall_macro": validation["recall_macro"],
        "validation_f1_macro": validation["f1_macro"],
        "validation_f1_weighted": validation["f1_weighted"],
        "test_accuracy": test["accuracy"],
        "test_precision_macro": test["precision_macro"],
        "test_recall_macro": test["recall_macro"],
        "test_f1_macro": test["f1_macro"],
        "test_f1_weighted": test["f1_weighted"],
    }


def find_result_files(results_directory: str) -> List[Path]:
    """
    Encontra arquivos JSON de resultados Random Forest.
    """
    if not isinstance(results_directory, str):
        raise TypeError("results_directory deve ser uma string.")

    directory = Path(results_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Diretório de resultados não encontrado: {results_directory}"
        )

    return sorted(directory.rglob("*_random_forest.json"))


def create_results_dataframe(results_directory: str) -> pd.DataFrame:
    """
    Cria um DataFrame com uma linha por execução experimental.
    """
    result_files = find_result_files(results_directory)

    rows = [
        extract_result_row(load_experiment_result(str(path)))
        for path in result_files
    ]

    return pd.DataFrame(rows)


def summarize_by_generator(results_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Resume média e desvio padrão das métricas por gerador.
    """
    if not isinstance(results_dataframe, pd.DataFrame):
        raise TypeError(
            "results_dataframe deve ser um pandas.DataFrame."
        )

    metric_columns = [
        "validation_accuracy",
        "validation_f1_macro",
        "validation_f1_weighted",
        "test_accuracy",
        "test_f1_macro",
        "test_f1_weighted",
    ]

    summary = (
        results_dataframe
        .groupby("generator_name")[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(column).strip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    return summary


def save_results_tables(
    results_dataframe: pd.DataFrame,
    generator_summary: pd.DataFrame,
    output_directory: str,
) -> None:
    """
    Salva tabelas CSV com execuções individuais e resumo por gerador.
    """
    if not isinstance(results_dataframe, pd.DataFrame):
        raise TypeError(
            "results_dataframe deve ser um pandas.DataFrame."
        )

    if not isinstance(generator_summary, pd.DataFrame):
        raise TypeError(
            "generator_summary deve ser um pandas.DataFrame."
        )

    if not isinstance(output_directory, str):
        raise TypeError("output_directory deve ser uma string.")

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        output_path / "random_forest_runs.csv",
        index=False,
    )

    generator_summary.to_csv(
        output_path / "random_forest_summary_by_generator.csv",
        index=False,
    )