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
        "--seed-start",
        default=None,
        type=int,
        help=(
            "Primeira seed de uma sequência determinística. Deve ser usada "
            "junto com --seed-count."
        ),
    )

    parser.add_argument(
        "--seed-count",
        default=None,
        type=int,
        help=(
            "Quantidade de seeds consecutivas a partir de --seed-start."
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


def resolve_seeds(
    seed: int = None,
    seed_start: int = None,
    seed_count: int = None,
):
    """
    Define quais seeds serão usadas na execução.

    Prioridade:
        - seed única;
        - sequência seed_start + seed_count;
        - None, para usar as seeds padrão do runner.
    """
    if seed is not None and (
        seed_start is not None or seed_count is not None
    ):
        raise ValueError(
            "Use --seed sozinho ou use --seed-start junto com "
            "--seed-count, mas não misture as duas formas."
        )

    if seed_start is None and seed_count is None:
        if seed is None:
            return None

        return [seed]

    if seed_start is None or seed_count is None:
        raise ValueError(
            "--seed-start e --seed-count devem ser usados juntos."
        )

    if seed_count <= 0:
        raise ValueError("--seed-count deve ser maior que zero.")

    return [
        seed_start + offset
        for offset in range(seed_count)
    ]


def main() -> None:
    """Executa o experimento conforme os argumentos informados."""
    args = parse_arguments()

    resolved_seeds = resolve_seeds(
        seed=args.seed,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
    )

    if (
        args.generator is not None
        and resolved_seeds is not None
        and len(resolved_seeds) == 1
    ):
        selected_seed = resolved_seeds[0]

        result = run_random_forest_experiment(
            dataset_path=args.dataset,
            generator_name=args.generator,
            seed=selected_seed,
            output_directory=args.output,
            model_random_state=args.model_random_state,
        )

        save_summary(
            result=result,
            output_directory=args.output,
        )

        print("Execução única concluída.")
        print(f"Gerador: {args.generator}")
        print(f"Seed: {selected_seed}")
        print(f"Saída: {args.output}")
        return

    generators = None

    if args.generator is not None:
        generators = [args.generator]

    seeds = resolved_seeds

    run_batch_random_forest_experiments(
        dataset_path=args.dataset,
        output_directory=args.output,
        generators=generators,
        seeds=seeds,
        model_random_state=args.model_random_state,
    )

    print("Execução em lote concluída.")
    print(f"Saída: {args.output}")

    if seeds is not None:
        print(f"Seeds executadas: {seeds[0]} até {seeds[-1]}")
        print(f"Quantidade de seeds: {len(seeds)}")


if __name__ == "__main__":
    main()