"""
Script para resumir resultados Random Forest em tabelas CSV.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.analysis.results_summary import (
    create_results_dataframe,
    save_results_tables,
    summarize_by_generator,
)


def parse_arguments() -> argparse.Namespace:
    """Lê os argumentos enviados pelo terminal."""
    parser = argparse.ArgumentParser(
        description=(
            "Gera tabelas CSV a partir dos resultados JSON "
            "do experimento Random Forest."
        ),
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Diretório contendo os JSONs de resultados.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Diretório onde as tabelas CSV serão salvas.",
    )

    return parser.parse_args()


def main() -> None:
    """Gera e salva as tabelas de análise."""
    args = parse_arguments()

    results_dataframe = create_results_dataframe(args.results)
    generator_summary = summarize_by_generator(results_dataframe)

    save_results_tables(
        results_dataframe=results_dataframe,
        generator_summary=generator_summary,
        output_directory=args.output,
    )

    print("Resumo dos resultados concluído.")
    print(f"Execuções resumidas: {len(results_dataframe)}")
    print(f"Saída: {args.output}")


if __name__ == "__main__":
    main()