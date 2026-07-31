"""
Adaptadores para geradores pseudoaleatórios externos.

Os adaptadores oferecem uma interface semelhante à utilizada pelo
PCG64-CP A Lite 1, permitindo substituir o gerador sem alterar o
restante do pipeline experimental.
"""

from typing import Optional

import numpy as np


class NumPyPCG64RNG:
    """Interface de alto nível para o PCG64 disponibilizado pelo NumPy."""

    def __init__(self, seed: int) -> None:
        """Inicializa o gerador PCG64 do NumPy com uma semente inteira."""
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed deve ser um número inteiro.")

        self._generator = np.random.Generator(
            np.random.PCG64(seed)
        )

    def random(self) -> float:
        """Retorna um número pseudoaleatório no intervalo [0, 1)."""
        return float(self._generator.random())

    def integers(
        self,
        low: int,
        high: Optional[int] = None,
    ) -> int:
        """
        Retorna um inteiro pseudoaleatório no intervalo [low, high).

        Quando high não é informado, low é interpretado como o limite
        superior e o limite inferior assume o valor zero.
        """
        if isinstance(low, bool) or not isinstance(low, int):
            raise TypeError("low deve ser um número inteiro.")

        if high is not None and (
            isinstance(high, bool) or not isinstance(high, int)
        ):
            raise TypeError("high deve ser um número inteiro ou None.")

        if high is None:
            high = low
            low = 0

        if high <= low:
            raise ValueError("high deve ser maior que low.")

        return int(self._generator.integers(low, high))

    def shuffle(self, values: list) -> None:
        """Embaralha uma lista diretamente."""
        if not isinstance(values, list):
            raise TypeError("values deve ser uma lista.")

        self._generator.shuffle(values)

    def permutation(self, n: int) -> list:
        """Retorna uma permutação dos índices de 0 até n - 1."""
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n deve ser um número inteiro.")

        if n < 0:
            raise ValueError("n não pode ser negativo.")

        return self._generator.permutation(n).tolist()

    def choice(
        self,
        values: list,
        size: Optional[int] = None,
        replace: bool = True,
    ):
        """Seleciona um ou mais elementos de uma lista."""
        if not isinstance(values, list):
            raise TypeError("values deve ser uma lista.")

        if len(values) == 0:
            raise ValueError("values não pode ser uma lista vazia.")

        if not isinstance(replace, bool):
            raise TypeError("replace deve ser um valor booleano.")

        if size is not None and (
            isinstance(size, bool) or not isinstance(size, int)
        ):
            raise TypeError("size deve ser um número inteiro ou None.")

        if size is not None and size < 0:
            raise ValueError("size não pode ser negativo.")

        if not replace and size is not None and size > len(values):
            raise ValueError(
                "Não é possível selecionar mais elementos que o disponível "
                "quando replace=False."
            )

        result = self._generator.choice(
            values,
            size=size,
            replace=replace,
        )

        if size is None:
            return result.item() if hasattr(result, "item") else result

        return result.tolist()
    
class NumPyMT19937RNG(NumPyPCG64RNG):
    """Interface de alto nível para o MT19937 disponibilizado pelo NumPy."""

    def __init__(self, seed: int) -> None:
        """Inicializa o gerador MT19937 do NumPy com uma semente inteira."""
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed deve ser um número inteiro.")

        self._generator = np.random.Generator(
            np.random.MT19937(seed)
        )