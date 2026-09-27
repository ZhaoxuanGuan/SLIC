from __future__ import annotations

import math
import numpy as np
from numba import njit


def optimal_failure_point(friction_mu: float) -> tuple[float, float]:

    scale = math.sqrt(1.0 + float(friction_mu) ** 2)
    return (-float(friction_mu) / scale, 1.0 / scale)


def failure_distance(sigma_n, tau, friction_mu: float):
    sigma_c, tau_c = optimal_failure_point(friction_mu)
    return np.sqrt((np.asarray(sigma_n) - sigma_c) ** 2 + (np.asarray(tau) - tau_c) ** 2)


def failure_criticality(sigma_n, tau, friction_mu: float):

    sigma_c, tau_c = optimal_failure_point(friction_mu)
    distance_max = math.sqrt((1.0 - sigma_c) ** 2 + tau_c ** 2)
    value = 1.0 - failure_distance(sigma_n, tau, friction_mu) / distance_max
    return np.clip(value, 0.0, 1.0)


@njit(cache=True, fastmath=True, inline='always')
def failure_criticality_numba(sigma_n, tau, friction_mu):
    scale = math.sqrt(1.0 + friction_mu * friction_mu)
    sigma_c = -friction_mu / scale
    tau_c = 1.0 / scale
    distance = math.sqrt((sigma_n - sigma_c) * (sigma_n - sigma_c) + (tau - tau_c) * (tau - tau_c))
    distance_max = math.sqrt((1.0 - sigma_c) * (1.0 - sigma_c) + tau_c * tau_c)
    return min(1.0, max(0.0, 1.0 - distance / distance_max))
