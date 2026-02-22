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
    n_states = len(map_r)
    dr_dt = [0.0 for _ in range(n_states)]
    dp_dt = [0.0 for _ in range(n_states)]
    for i in range(n_states):
        val_r = 0.0
        val_p = 0.0
        for j in range(n_states):
            val_r += h_eff[i][j] * map_p[j]
            val_p += h_eff[i][j] * map_r[j]
        dr_dt[i] = val_r / HBAR_MAPPING
        dp_dt[i] = -val_p / HBAR_MAPPING
    return dr_dt, dp_dt


def _symmetrize_matrix(h: List[List[float]]) -> List[List[float]]:
    n = len(h)
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        out[i][i] = h[i][i]
        for j in range(i + 1, n):
            val = 0.5 * (h[i][j] + h[j][i])
            out[i][j] = val
            out[j][i] = val
    return out


def _jacobi_eigh(h: List[List[float]], max_iter: int = 200, tol: float = 1e-14) -> Tuple[List[float], List[List[float]]]:
    n = len(h)
    a = [row[:] for row in h]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_iter):
        p = 0
        q = 1 if n > 1 else 0
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                aij = abs(a[i][j])
                if aij > off:
                    off = aij
                    p = i
                    q = j

        if off < tol:
            break

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c = math.cos(phi)
        s = math.sin(phi)

        for k in range(n):
            if k == p or k == q:
                continue
            akp = a[k][p]
            akq = a[k][q]
            a[k][p] = c * akp - s * akq
            a[p][k] = a[k][p]
            a[k][q] = s * akp + c * akq
            a[q][k] = a[k][q]

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(n):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq

    evals = [a[i][i] for i in range(n)]
    return evals, v


def propagate_mapping_exact_half_step(map_r: List[float], map_p: List[float], h_eff: List[List[float]], dt_half_fs: float) -> None:
    n_states = len(map_r)
    hsym = _symmetrize_matrix(h_eff)
    evals, evecs = _jacobi_eigh(hsym)

    r_mode = [sum(evecs[row][col] * map_r[row] for row in range(n_states)) for col in range(n_states)]
    p_mode = [sum(evecs[row][col] * map_p[row] for row in range(n_states)) for col in range(n_states)]

    r_rot = [0.0 for _ in range(n_states)]
    p_rot = [0.0 for _ in range(n_states)]
    for i in range(n_states):
        theta = evals[i] * dt_half_fs / HBAR_MAPPING
        c = math.cos(theta)
        s = math.sin(theta)
        r_rot[i] = c * r_mode[i] + s * p_mode[i]
        p_rot[i] = c * p_mode[i] - s * r_mode[i]

    for row in range(n_states):
        map_r[row] = sum(evecs[row][col] * r_rot[col] for col in range(n_states))
        map_p[row] = sum(evecs[row][col] * p_rot[col] for col in range(n_states))
