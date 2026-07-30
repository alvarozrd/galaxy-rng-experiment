"""
Interface de alto nível para o PCG64-CP A Lite 1.

Este módulo adapta o núcleo canônico do gerador para operações que serão
utilizadas posteriormente em pipelines de aprendizado de máquina.
"""

from .core import PCG64CPALite1


class PCG64CPALite1RNG:
    """Interface reutilizável para o núcleo PCG64-CP A Lite 1."""

    def __init__(self, seed: int) -> None:
        """Inicializa a interface com uma semente inteira."""
        self._core = PCG64CPALite1(seed=seed)

    def random(self) -> float:
        """Retorna um número pseudoaleatório no intervalo [0, 1)."""
        return self._core.random_float()