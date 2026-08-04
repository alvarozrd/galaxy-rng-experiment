"""
Script para executar o experimento Random Forest com dataset de galáxias.
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.experiments.rf_runner import (
    DEFAULT_MODEL_RANDOM_STATE,
    run_batch_random_forest_experiments,
    run_random_forest_experiment,
)


def parse_arguments() -> argparse.Namespace:
    """Lê os argumentos enviados pelo terminal."""
    parser = argparse.ArgumentParser(
        description=(
            "Executa experimentos Random Forest usando splits "
            "reprodutíveis gerados por PRNGs."
        ),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Caminho para o CSV de galáxias.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Diretório onde os resultados serão salvos.",
    )

    parser.add_argument(
        "--generator",
        default=None,
        help=(
            "Nome de um gerador específico. Se omitido, executa "
            "todos os geradores padrão."
        ),
    )

    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help=(
            "Seed específica. Se omitida, executa todas as seeds padrão."
        ),
    )

    parser.add_argument(
        "--model-random-state",
        default=DEFAULT_MODEL_RANDOM_STATE,
        type=int,
        help=(
            "random_state interno do Random Forest. Por padrão, fica "
            "fixo em 42 para isolar o efeito dos splits."
        ),
    )

    return parser.parse_args()


def save_summary(
    result: dict,
    output_directory: str,
) -> None:
    """Salva um resumo compacto quando apenas uma execução é rodada."""
    summary = {
        "generator_name": result["metadata"]["generator_name"],
        "seed": result["metadata"]["seed"],
        "dataset_size": result["manifest"]["dataset_size"],
        "split_sizes": result["manifest"]["split_sizes"],
        "class_distribution": result["manifest"]["class_distribution"],
        "model_random_state": result["model"]["random_state"],
        "validation_metrics": {
            "accuracy": result["model"]["results"]["validation"]["accuracy"],
            "f1_macro": result["model"]["results"]["validation"]["f1_macro"],
            "f1_weighted": result["model"]["results"]["validation"][
                "f1_weighted"
            ],
        },
        "test_metrics": {
            "accuracy": result["model"]["results"]["test"]["accuracy"],
            "f1_macro": result["model"]["results"]["test"]["f1_macro"],
            "f1_weighted": result["model"]["results"]["test"][
                "f1_weighted"
            ],
        },
    }

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = output_path / "latest_single_run_summary.json"

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )


def main() -> None:
    """Executa o experimento conforme os argumentos informados."""
    args = parse_arguments()

    if args.generator is not None and args.seed is not None:
        result = run_random_forest_experiment(
            dataset_path=args.dataset,
            generator_name=args.generator,
            seed=args.seed,
            output_directory=args.output,
            model_random_state=args.model_random_state,
        )

        save_summary(
            result=result,
            output_directory=args.output,
        )

        print("Execução única concluída.")
        print(f"Gerador: {args.generator}")
        print(f"Seed: {args.seed}")
        print(f"Saída: {args.output}")
        return

    generators = None
    seeds = None

    if args.generator is not None:
        generators = [args.generator]

    if args.seed is not None:
        seeds = [args.seed]

    run_batch_random_forest_experiments(
        dataset_path=args.dataset,
        output_directory=args.output,
        generators=generators,
        seeds=seeds,
        model_random_state=args.model_random_state,
    )

    print("Execução em lote concluída.")
    print(f"Saída: {args.output}")


if __name__ == "__main__":
    main()