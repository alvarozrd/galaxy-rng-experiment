"""
Interface de alto nível para o PCG64-CP A Lite 1.

Este módulo adapta o núcleo canônico do gerador para operações que serão
utilizadas posteriormente em pipelines de aprendizado de máquina.
"""

from typing import Optional
from .core import PCG64CPALite1


UINT64_RANGE = 2**64


class PCG64CPALite1RNG:
    """Interface reutilizável para o núcleo PCG64-CP A Lite 1."""

    def __init__(self, seed: int) -> None:
        """Inicializa a interface com uma semente inteira."""
        self._core = PCG64CPALite1(seed=seed)

    def random(self) -> float:
        """Retorna um número pseudoaleatório no intervalo [0, 1)."""
        return self._core.random_float()

    def integers(self, low: int, high: Optional[int] = None) -> int:
        """
        Retorna um inteiro pseudoaleatório no intervalo [low, high).

        Quando apenas um argumento é informado, ele é interpretado como
        o limite superior e o limite inferior assume o valor zero.

        Exemplos:
            integers(10) retorna um valor entre 0 e 9.
            integers(5, 10) retorna um valor entre 5 e 9.
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

        interval_size = high - low

        if interval_size <= 0:
            raise ValueError("high deve ser maior que low.")

        if interval_size > UINT64_RANGE:
            raise ValueError(
                "O intervalo não pode possuir mais de 2**64 valores."
            )

        rejection_limit = UINT64_RANGE - (UINT64_RANGE % interval_size)

        while True:
            value = self._core.next_u64()

            if value < rejection_limit:
                return low + (value % interval_size)

    def shuffle(self, values: list) -> None:

        """

        Embaralha uma lista diretamente, utilizando o algoritmo Fisher-Yates.

        A lista original é modificada e o método não retorna uma nova lista.

        """

        if not isinstance(values, list):

            raise TypeError("values deve ser uma lista.")

        for current_index in range(len(values) - 1, 0, -1):

            random_index = self.integers(0, current_index + 1)

            values[current_index], values[random_index] = (

                values[random_index],

                values[current_index],

            )

    def permutation(self, n: int) -> list:
        """
        Retorna uma permutação pseudoaleatória dos inteiros de 0 até n - 1.

        A lista retornada contém todos os índices exatamente uma vez.

        Exemplo:
            permutation(5) pode retornar [3, 0, 4, 1, 2].
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n deve ser um número inteiro.")

        if n < 0:
            raise ValueError("n não pode ser negativo.")

        values = list(range(n))
        self.shuffle(values)

        return values