"""
Coleta de metadados para reprodutibilidade dos experimentos.
"""

import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np


def get_git_commit() -> Optional[str]:
    """
    Retorna o hash do commit atual do Git.

    Caso o comando não possa ser executado, retorna None.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def create_execution_metadata(
    generator_name: str,
    seed: int,
) -> dict:
    """
    Cria um registro com informações da execução experimental.
    """
    if not isinstance(generator_name, str):
        raise TypeError("generator_name deve ser uma string.")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um número inteiro.")

    return {
        "generator_name": generator_name,
        "seed": seed,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "operating_system": platform.system(),
        "operating_system_version": platform.release(),
        "machine": platform.machine(),
        "git_commit": get_git_commit(),
    }