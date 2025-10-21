
# utils.py
# -*- coding: utf-8 -*-
"""
Hjelpefunksjoner for populasjonsgenetikk-simulatoren.
Refaktorert med type hints, validering og valgfri RNG/seed for reproduksjon.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Tuple, Union

Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]
Array3D = NDArray[np.float64]

@dataclass(frozen=True)
class Fitness:
    wAA: float
    wAa: float
    waa: float

def _rng_from(seed: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    """Returner en np.random.Generator fra heltalls-seed eller eksisterende generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)  # seed kan være None eller int

def _validate_prob(x: float, name: str) -> None:
    """Sjekk at en sannsynlighet ligger i [0,1]."""
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} må være i [0,1]. Fikk {x}.")

def _validate_fitness(f: Fitness, label: str) -> None:
    """Sjekk at alle fitnessverdier ligger i [0,1]."""
    for val, nm in ((f.wAA, "wAA"), (f.wAa, "wAa"), (f.waa, "waa")):
        _validate_prob(val, f"{label}.{nm}")



def simulate_one_pop(
    p0: float,
    N: Optional[int],
    fitness: Fitness,
    mu: float,
    nu: float,
    generations: int,
    bottleneck_start: Optional[int] = None,
    bottleneck_duration: Optional[int] = None,
    bottleneck_size: Optional[int] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[Array2D, Array3D]:
    """
    Simuler én populasjon med seleksjon, mutasjon, drift og evt. flaskehals.

    Returnerer:
        freqs: (T, 1)
        genotypes: (T, 1, 3)
    """
    # Validering
    _validate_prob(p0, "p0")
    _validate_fitness(fitness, "fitness")
    _validate_prob(mu, "mu")
    _validate_prob(nu, "nu")
    if generations < 1:
        raise ValueError("generations må være >= 1.")

    rng = _rng_from(seed)
    p = float(p0)
    freqs = [p]
    genotypes = []

    for gen in range(generations):
        p2 = p * p
        pq = 2 * p * (1 - p)
        q2 = (1 - p) * (1 - p)
        genotypes.append([p2, pq, q2])

        # Seleksjon
        w_bar = p2 * fitness.wAA + pq * fitness.wAa + q2 * fitness.waa
        p_prime = p if w_bar == 0 else (p2 * fitness.wAA + 0.5 * pq * fitness.wAa) / w_bar

        # Mutasjon
        p_prime = p_prime * (1 - mu) + (1 - p_prime) * nu

        # Drift + evt. flaskehals
        if N is not None:
            if (
                bottleneck_start is not None
                and bottleneck_duration is not None
                and bottleneck_size is not None
                and bottleneck_start <= gen < bottleneck_start + bottleneck_duration
            ):
                N_eff = int(bottleneck_size)
            else:
                N_eff = int(N)
            p = rng.binomial(2 * N_eff, p_prime) / (2 * N_eff)
        else:
            p = p_prime

        freqs.append(float(np.clip(p, 0.0, 1.0)))

    freqs_arr = np.array(freqs, dtype=float).reshape(-1, 1)
    genos_arr = np.array(genotypes, dtype=float).reshape(-1, 1, 3)
    return freqs_arr, genos_arr

def simulate_two_pops(
    p0_1: float,
    p0_2: float,
    N: Optional[int],
    fitness1: Fitness,
    fitness2: Fitness,
    mu: float,
    nu: float,
    generations: int,
    migrate: bool = False,
    m12: float = 0.0,
    m21: float = 0.0,
    seed: Optional[Union[int, np.random.Generator]] = None,
    
) -> Tuple[Array2D, Array3D]:
    """
    Simuler to populasjoner med seleksjon, mutasjon, drift og evt. migrasjon.

    Returnerer:
        freqs: (T, 2)
        genotypes: (T, 2, 3)
    """
    # Validering
    for p0, nm in ((p0_1, "p0_1"), (p0_2, "p0_2")):
        _validate_prob(p0, nm)
    _validate_fitness(fitness1, "fitness1")
    _validate_fitness(fitness2, "fitness2")
    for r, nm in ((mu, "mu"), (nu, "nu"), (m12, "m12"), (m21, "m21")):
        _validate_prob(r, nm)
    if generations < 1:
        raise ValueError("generations må være >= 1.")

    rng = _rng_from(seed)
    p = np.array([p0_1, p0_2], dtype=float)
    freqs = [p.copy()]
    genotypes = [[[p[0]**2, 2*p[0]*(1-p[0]), (1-p[0])**2],
                  [p[1]**2, 2*p[1]*(1-p[1]), (1-p[1])**2]]]

    for _ in range(generations):
        new_p = []
        gen = []
        for i, pi in enumerate(p):
            f = fitness1 if i == 0 else fitness2

            p2 = pi * pi
            pq = 2 * pi * (1 - pi)
            q2 = (1 - pi) * (1 - pi)
            gen.append([p2, pq, q2])

            w_bar = p2 * f.wAA + pq * f.wAa + q2 * f.waa
            p_prime = pi if w_bar == 0 else (p2 * f.wAA + 0.5 * pq * f.wAa) / w_bar

            # Mutasjon
            p_prime = p_prime * (1 - mu) + (1 - p_prime) * nu

            # Drift
            if N is not None:
                pi_next = rng.binomial(2 * int(N), p_prime) / (2 * int(N))
            else:
                pi_next = p_prime

            new_p.append(pi_next)

        p = np.array(new_p, dtype=float)

        # Migrasjon
        if migrate:
            p = np.array([
                (1 - m21) * p[0] + m21 * p[1],
                (1 - m12) * p[1] + m12 * p[0],
            ], dtype=float)

        p = np.clip(p, 0.0, 1.0)
        freqs.append(p.copy())
        genotypes.append(gen)

    freqs_arr = np.vstack(freqs).astype(float)   # (T, 2)
    genos_arr = np.array(genotypes, dtype=float) # (T, 2, 3)
    return freqs_arr, genos_arr
