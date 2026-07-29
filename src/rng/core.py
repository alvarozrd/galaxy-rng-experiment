#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementação canônica do PCG64-CP A Lite 1.

Este módulo contém apenas o núcleo do gerador pseudoaleatório.

Funções de integração, amostragem e adaptação para pipelines de

aprendizado de máquina devem permanecer em módulos separados.
"""

import numpy as np

MASK64 = (1 << 64) - 1
UINT64_MAX_NP = np.uint64((1 << 64) - 1)


def popcount64(x: int) -> int:
    """Conta quantos bits 1 existem em um inteiro de 64 bits."""
    return bin(int(x) & MASK64).count("1")


def rotl64(x: int, r: int) -> int:
    """Rotaciona bits para a esquerda em uma palavra de 64 bits."""
    x &= MASK64
    r &= 63
    if r == 0:
        return x
    return ((x << r) | (x >> (64 - r))) & MASK64


def rotr64(x: int, r: int) -> int:
    """Rotaciona bits para a direita em uma palavra de 64 bits."""
    x &= MASK64
    r &= 63
    if r == 0:
        return x
    return ((x >> r) | (x << (64 - r))) & MASK64


def splitmix64(x: int) -> int:
    """
    Função de mistura de 64 bits usada como pós-processamento.
    """
    x = (int(x) + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return z & MASK64


def logistic_step(x: float, r: float = 3.999999) -> float:
    """
    Executa uma etapa do mapa logístico:

        x[n+1] = r * x[n] * (1 - x[n])

    O valor r próximo de 4 coloca o sistema em regime caótico.
    """
    x = r * x * (1.0 - x)
    if x <= 0.0:
        x = 0.123456789
    elif x >= 1.0:
        x = 0.987654321
    return x


def logistic_to_u64(x: float) -> int:
    """Converte o estado contínuo do mapa logístico para inteiro de 64 bits."""
    return int(abs(x) * float(1 << 64)) & MASK64


def u64_to_unit_interval(x: int) -> float:
    """Converte um inteiro de 64 bits para o intervalo aberto aproximado (0, 1)."""
    v = ((int(x) & MASK64) / float(1 << 64))
    if v <= 0.0:
        return 0.5
    if v >= 1.0:
        return 0.999999999999
    return v


class PCG64CPALite1:
    """
    PCG64-CP A Lite 1.

    Estrutura do gerador:
    1. PCG64 oficial do NumPy gera a base.
    2. Um mapa logístico gera uma perturbação caótica leve.
    3. Um contador interno adiciona variação determinística por saída.
    4. A saída passa por rotação de bits e SplitMix64.
    """

    nome = "PCG64-CP A Lite 1"

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = np.random.Generator(np.random.PCG64(self.seed))

        init = int(self.rng.integers(1, UINT64_MAX_NP, dtype=np.uint64))
        self.chaos = u64_to_unit_interval(splitmix64(init ^ self.seed))
        self.counter = 0

    def next_u64(self) -> int:
        """Gera um inteiro pseudoaleatório de 64 bits."""
        base = int(self.rng.integers(0, UINT64_MAX_NP, dtype=np.uint64))

        self.chaos = logistic_step(self.chaos)
        chaos_word = logistic_to_u64(self.chaos)

        c = splitmix64(self.seed + self.counter)

        x = base ^ chaos_word ^ c
        x = rotl64(x, (self.counter + popcount64(chaos_word)) & 63)
        x = splitmix64(x ^ rotr64(base, 17))

        self.counter += 1
        return x & MASK64

    def random_float(self) -> float:
        """Gera um número pseudoaleatório no intervalo [0, 1)."""
        return self.next_u64() / float(1 << 64)

    def gerar_u64(self, n: int) -> list[int]:
        """Gera uma lista com n inteiros pseudoaleatórios de 64 bits."""
        return [self.next_u64() for _ in range(n)]

    def gerar_float(self, n: int) -> list[float]:
        """Gera uma lista com n números pseudoaleatórios no intervalo [0, 1)."""
        return [self.random_float() for _ in range(n)]


if __name__ == "__main__":
    gerador = PCG64CPALite1(seed=123456789)

    print("Cinco inteiros de 64 bits:")
    for _ in range(5):
        print(gerador.next_u64())

    print("\nCinco valores em [0, 1):")
    for _ in range(5):
        print(gerador.random_float())
