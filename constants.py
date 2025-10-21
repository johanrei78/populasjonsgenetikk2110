
# constants.py
# -*- coding: utf-8 -*-

DEFAULTS = {
    # Tidsparametere
    "generations": 100,
    "num_pops": 1,

    # Startfrekvenser
    "p0": 0.5,
    "p0_1": 0.5,
    "p0_2": 0.5,

    # Fitness-verdier (relativ fitness = 1 = ingen seleksjon)
    "wAA_1": 1.0, "wAa_1": 1.0, "waa_1": 1.0,
    "wAA_2": 1.0, "wAa_2": 1.0, "waa_2": 1.0,

    # Mutasjonsrater
    "mu": 0.0,
    "nu": 0.0,

    # Populasjonsstørrelse (None = uendelig)
    "N": None,

    # Migrasjonsrater
    "m12": 0.0,
    "m21": 0.0,

    # Av/på-flagg
    "use_drift": False,
    "use_bottleneck": False,
    "migrate": False,
}

UI = {
    "GEN_MIN": 10,
    "GEN_MAX": 500,
    "P_STEP": 0.01,
    "FIT_STEP": 0.01,
    "MUT_STEP": 0.000001,
    "MUT_MAX": 0.01,
    "MIG_STEP": 0.01,
    "N_MIN": 10,
    "N_MAX": 10000,
}
