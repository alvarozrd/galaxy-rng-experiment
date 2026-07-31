"""
Geradores pseudoaleatórios utilizados no Galaxy RNG Experiment.
"""

from .core import PCG64CPALite1
from .rng import PCG64CPALite1RNG

__all__ = [
    "PCG64CPALite1",
    "PCG64CPALite1RNG",
]