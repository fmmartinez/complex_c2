from __future__ import annotations

import math
import random
from typing import List, Tuple

from .model import HBAR_MAPPING


def sample_focused_mapping_variables(
    n_states: int,
    occupied_state: int,
    rng: random.Random,
) -> Tuple[List[float], List[float]]:
    if not (0 <= occupied_state < n_states):
        raise ValueError(f"occupied_state must be in [0, {n_states - 1}], got {occupied_state}.")

    std = math.sqrt(0.5)
    map_r = []
    map_p = []
    for _ in range(n_states):
        map_r.append(rng.gauss(0.0, std))
        map_p.append(rng.gauss(0.0, std))

    for i in range(n_states):
        target = 3.0 if i == occupied_state else 1.0
        radius = math.sqrt(map_r[i] * map_r[i] + map_p[i] * map_p[i])
        if radius < 1e-14:
            map_r[i] = math.sqrt(target)
            map_p[i] = 0.0
            continue
        scale = math.sqrt(target) / radius
        map_r[i] *= scale
        map_p[i] *= scale

    return map_r, map_p


def compute_mapping_derivatives(h_eff: List[List[float]], map_r: List[float], map_p: List[float]) -> Tuple[List[float], List[float]]:
    dr_dt = [0.0, 0.0, 0.0]
    dp_dt = [0.0, 0.0, 0.0]
    for i in range(3):
        val_r = 0.0
        val_p = 0.0
        for j in range(3):
            val_r += h_eff[i][j] * map_p[j]
            val_p += h_eff[i][j] * map_r[j]
        dr_dt[i] = val_r / HBAR_MAPPING
        dp_dt[i] = -val_p / HBAR_MAPPING
    return dr_dt, dp_dt
