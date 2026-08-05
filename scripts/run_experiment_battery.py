"""
Executa a bateria principal de experimentos Random Forest.

Os resultados brutos ficam em results/experiment_runs/.
As tabelas pequenas ficam em results/experiment_summaries/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.analysis.results_summary import (  # noqa: E402
    create_results_dataframe,
    save_results_tables,
    summarize_by_generator,
)
from src.data.galaxies import (  # noqa: E402
    TARGET_COLUMN,
    load_galaxy_dataframe,
    prepare_galaxy_dataframe,
)
from src.data.splitting import split_stratified_indices  # noqa: E402
from src.experiments.rf_runner import run_random_forest_experiment  # noqa: E402
from src.experiments.runner import DEFAULT_GENERATORS  # noqa: E402


DEFAULT_DATASET = "data/raw/galaxies.csv"
DEFAULT_SEED_START = 1000
DEFAULT_SEED_COUNT = 30
DEFAULT_SPLIT_GENERATOR = "numpy_pcg64"
DEFAULT_SPLIT_SEED = 123456789
DEFAULT_MODEL_RANDOM_STATE = 42

EXPERIMENTS = {
    "01_split_seed": (
        "Varia gerador e seed do split; Random Forest fixo em 42."
    ),
    "02_model_state": (
        "Mantem o split fixo; varia o random_state interno do Random Forest."
    ),
    "03_coupled_seed": (
        "Usa a mesma seed para split e Random Forest."
    ),
    "04_dataset_size": (
        "Repete o experimento com fracoes diferentes do dataset."
    ),
    "05_split_timing": (
        "Mede o tempo de geracao dos splits por gerador e seed."
    ),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa experimentos controlados de PRNG + Random Forest.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="Lista os experimentos disponiveis e encerra.",
    )
    parser.add_argument(
        "--experiment",
        choices=[*EXPERIMENTS.keys(), "all"],
        default="01_split_seed",
        help="Experimento a executar.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Caminho para o CSV de galaxias.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help="Primeira seed da sequencia.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=DEFAULT_SEED_COUNT,
        help="Quantidade de seeds consecutivas.",
    )
    parser.add_argument(
        "--output-root",
        default="results/experiment_runs",
        help="Diretorio raiz para resultados brutos locais.",
    )
    parser.add_argument(
        "--summary-root",
        default="results/experiment_summaries",
        help="Diretorio raiz para tabelas-resumo versionaveis.",
    )
    return parser.parse_args()


def resolve_seeds(seed_start: int, seed_count: int) -> list[int]:
    if seed_count <= 0:
        raise ValueError("seed_count deve ser maior que zero.")

    return [
        seed_start + offset
        for offset in range(seed_count)
    ]


def write_experiment_note(
    experiment_name: str,
    summary_directory: Path,
    details: dict,
) -> None:
    summary_directory.mkdir(parents=True, exist_ok=True)
    note_path = summary_directory / "experiment_metadata.md"
    details_json = json.dumps(details, indent=2, ensure_ascii=False)

    note_path.write_text(
        f"# {experiment_name}\n\n"
        f"{EXPERIMENTS[experiment_name]}\n\n"
        "```json\n"
        f"{details_json}\n"
        "```\n",
        encoding="utf-8",
    )


def summarize_results(output_directory: Path, summary_directory: Path) -> None:
    dataframe = create_results_dataframe(str(output_directory))
    summary = summarize_by_generator(dataframe)

    save_results_tables(
        results_dataframe=dataframe,
        generator_summary=summary,
        output_directory=str(summary_directory),
    )


def run_split_seed_experiment(
    dataset_path: str,
    seeds: list[int],
    output_root: Path,
    summary_root: Path,
) -> None:
    experiment_name = "01_split_seed"
    output_directory = output_root / experiment_name
    summary_directory = summary_root / experiment_name

    for generator_name in DEFAULT_GENERATORS:
        for seed in seeds:
            run_random_forest_experiment(
                dataset_path=dataset_path,
                generator_name=generator_name,
                seed=seed,
                output_directory=str(output_directory),
                model_random_state=DEFAULT_MODEL_RANDOM_STATE,
            )

    summarize_results(output_directory, summary_directory)
    write_experiment_note(
        experiment_name,
        summary_directory,
        {
            "generators": DEFAULT_GENERATORS,
            "split_seeds": seeds,
            "model_random_state": DEFAULT_MODEL_RANDOM_STATE,
            "dataset": dataset_path,
        },
    )


def run_model_state_experiment(
    dataset_path: str,
    model_states: list[int],
    output_root: Path,
    summary_root: Path,
) -> None:
    experiment_name = "02_model_state"
    output_directory = output_root / experiment_name
    summary_directory = summary_root / experiment_name

    for model_random_state in model_states:
        state_output = output_directory / f"model_state_{model_random_state}"
        run_random_forest_experiment(
            dataset_path=dataset_path,
            generator_name=DEFAULT_SPLIT_GENERATOR,
            seed=DEFAULT_SPLIT_SEED,
            output_directory=str(state_output),
            model_random_state=model_random_state,
        )

    summarize_results(output_directory, summary_directory)
    write_experiment_note(
        experiment_name,
        summary_directory,
        {
            "split_generator": DEFAULT_SPLIT_GENERATOR,
            "split_seed": DEFAULT_SPLIT_SEED,
            "model_random_states": model_states,
            "dataset": dataset_path,
        },
    )


def run_coupled_seed_experiment(
    dataset_path: str,
    seeds: list[int],
    output_root: Path,
    summary_root: Path,
) -> None:
    experiment_name = "03_coupled_seed"
    output_directory = output_root / experiment_name
    summary_directory = summary_root / experiment_name

    for generator_name in DEFAULT_GENERATORS:
        for seed in seeds:
            run_random_forest_experiment(
                dataset_path=dataset_path,
                generator_name=generator_name,
                seed=seed,
                output_directory=str(output_directory),
                model_random_state=seed,
            )

    summarize_results(output_directory, summary_directory)
    write_experiment_note(
        experiment_name,
        summary_directory,
        {
            "generators": DEFAULT_GENERATORS,
            "split_seeds": seeds,
            "model_random_states": "same as split seed",
            "dataset": dataset_path,
        },
    )


def create_fraction_dataset(
    dataset_path: str,
    fraction: float,
    output_path: Path,
) -> None:
    dataframe = load_galaxy_dataframe(dataset_path)
    prepared = prepare_galaxy_dataframe(dataframe)

    if fraction >= 1.0:
        sampled = prepared
    else:
        sampled = (
            prepared
            .groupby(TARGET_COLUMN, group_keys=False)
            .sample(frac=fraction, random_state=DEFAULT_MODEL_RANDOM_STATE)
            .reset_index(drop=True)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)


def run_dataset_size_experiment(
    dataset_path: str,
    seeds: list[int],
    output_root: Path,
    summary_root: Path,
) -> None:
    experiment_name = "04_dataset_size"
    output_directory = output_root / experiment_name
    summary_directory = summary_root / experiment_name
    fractions = [0.25, 0.50, 0.75, 1.00]

    for fraction in fractions:
        fraction_label = f"{int(fraction * 100):03d}"
        sampled_dataset = (
            output_directory
            / "_datasets"
            / f"galaxies_fraction_{fraction_label}.csv"
        )
        create_fraction_dataset(
            dataset_path=dataset_path,
            fraction=fraction,
            output_path=sampled_dataset,
        )

        fraction_output = output_directory / f"fraction_{fraction_label}"
        for generator_name in DEFAULT_GENERATORS:
            for seed in seeds:
                run_random_forest_experiment(
                    dataset_path=str(sampled_dataset),
                    generator_name=generator_name,
                    seed=seed,
                    output_directory=str(fraction_output),
                    model_random_state=DEFAULT_MODEL_RANDOM_STATE,
                )

    summarize_results(output_directory, summary_directory)
    write_experiment_note(
        experiment_name,
        summary_directory,
        {
            "generators": DEFAULT_GENERATORS,
            "split_seeds": seeds,
            "dataset_fractions": fractions,
            "model_random_state": DEFAULT_MODEL_RANDOM_STATE,
            "source_dataset": dataset_path,
        },
    )


def run_split_timing_experiment(
    dataset_path: str,
    seeds: list[int],
    output_root: Path,
    summary_root: Path,
) -> None:
    experiment_name = "05_split_timing"
    output_directory = output_root / experiment_name
    summary_directory = summary_root / experiment_name
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_directory.mkdir(parents=True, exist_ok=True)

    dataframe = load_galaxy_dataframe(dataset_path)
    prepared = prepare_galaxy_dataframe(dataframe)
    labels = prepared[TARGET_COLUMN].tolist()

    rows = []
    for generator_name in DEFAULT_GENERATORS:
        for seed in seeds:
            start = time.perf_counter()
            split = split_stratified_indices(
                labels=labels,
                generator_name=generator_name,
                seed=seed,
            )
            elapsed_seconds = time.perf_counter() - start
            rows.append(
                {
                    "generator_name": generator_name,
                    "seed": seed,
                    "dataset_size": len(labels),
                    "train_size": len(split["train"]),
                    "validation_size": len(split["validation"]),
                    "test_size": len(split["test"]),
                    "elapsed_seconds": elapsed_seconds,
                }
            )

    timing_dataframe = pd.DataFrame(rows)
    timing_dataframe.to_csv(
        summary_directory / "split_timing_runs.csv",
        index=False,
    )
    (
        timing_dataframe
        .groupby("generator_name")["elapsed_seconds"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .to_csv(summary_directory / "split_timing_summary_by_generator.csv", index=False)
    )
    write_experiment_note(
        experiment_name,
        summary_directory,
        {
            "generators": DEFAULT_GENERATORS,
            "split_seeds": seeds,
            "dataset": dataset_path,
        },
    )


def main() -> None:
    args = parse_arguments()

    if args.list_experiments:
        for experiment_name, description in EXPERIMENTS.items():
            print(f"{experiment_name}: {description}")
        return

    seeds = resolve_seeds(args.seed_start, args.seed_count)
    output_root = Path(args.output_root)
    summary_root = Path(args.summary_root)

    runners = {
        "01_split_seed": run_split_seed_experiment,
        "02_model_state": run_model_state_experiment,
        "03_coupled_seed": run_coupled_seed_experiment,
        "04_dataset_size": run_dataset_size_experiment,
        "05_split_timing": run_split_timing_experiment,
    }

    selected_experiments = (
        list(runners.keys())
        if args.experiment == "all"
        else [args.experiment]
    )

    for experiment_name in selected_experiments:
        print(f"Iniciando {experiment_name}: {EXPERIMENTS[experiment_name]}")
        runners[experiment_name](
            args.dataset,
            seeds,
            output_root,
            summary_root,
        )
        print(f"Concluido {experiment_name}")


if __name__ == "__main__":
    main()
