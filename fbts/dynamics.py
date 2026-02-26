from __future__ import annotations

import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised when numpy is not installed.
    np = None  # type: ignore[assignment]

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when numba is not installed.
    NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator

from .diabatic import diabatic_r_range, interpolate_eigenstates, interpolate_h_diabatic, load_diabatic_tables
from .io_utils import append_xyz_frame
from .electronic import (
    FBTSElectronicState,
    initialize_coherent_fbts_state,
    initialize_focused_fbts_state,
    propagate_fbts_state_exact_step,
)
from .model import (
    CHARGE,
    COULOMB_KCAL_MOL_ANG_E2,
    PLANCK_REDUCED_KCAL_MOL_FS,
    KCAL_MOL_ANG_TO_AMU_ANG_FS2,
    LABEL,
    MASS,
    POLARIZATION,
    PROTON_GRID_STEP,
    Site,
    coulomb_energy_dudr,
    dist,
    dot,
    get_lj_params,
    instantaneous_temperature,
    kinetic_energy_kcal_mol,
    lj_energy_dudr,
    norm,
    polarization_switch,
    random_point_in_sphere,
    random_unit_vector,
    remove_net_linear_momentum,
    sample_velocity,
)

SITE_TYPE_TO_CODE = {"C1": 0, "C2": 1, "A": 2, "B": 3, "H": 4}
Q_A_COV = POLARIZATION["Q_A_cov"]
Q_H_COV = POLARIZATION["Q_H_cov"]
Q_B_COV = POLARIZATION["Q_B_cov"]
Q_A_ION = POLARIZATION["Q_A_ion"]
Q_H_ION = POLARIZATION["Q_H_ion"]
Q_B_ION = POLARIZATION["Q_B_ion"]
R0_POL = POLARIZATION["r0"]
L_POL = POLARIZATION["l"]
AMU_ANG2_FS2_TO_KCAL_MOL = 2390.05736055072


@dataclass
class ForceEvalResult:
    forces: List[List[float]]
    terms: Dict[str, object]


def _validate_fbts_kernel_backend(kernel_backend: str) -> None:
    if kernel_backend != "python":
        raise ValueError("FBTS path currently supports kernel_backend='python' only.")


def _fbts_estimator_matrix(state: FBTSElectronicState) -> List[List[float]]:
    n_states = len(state.r_f)
    s_fb = [[0.0 for _ in range(n_states)] for _ in range(n_states)]
    for i in range(n_states):
        for j in range(n_states):
            s_fb[i][j] = 0.5 * (
                state.r_f[i] * state.r_b[j]
                + state.p_f[i] * state.p_b[j]
                + state.r_b[i] * state.r_f[j]
                + state.p_b[i] * state.p_f[j]
            )
    return s_fb


def _compute_bath_and_coupling_matrices_python(
    sites: List[Site],
    n_solvent_molecules: int,
    grid: List[float],
    eigenstates: List[List[float]],
    estimator_matrix: List[List[float]],
    n_states: int,
) -> Tuple[List[List[float]], float, List[List[float]], float, float, int, int, List[float], List[float], List[float]]:
    forces = [[0.0, 0.0, 0.0] for _ in sites]
    idx_a = idx_h = idx_b = None
    for idx, site in enumerate(sites):
        if site.site_type == "A":
            idx_a = idx
        elif site.site_type == "H":
            idx_h = idx
        elif site.site_type == "B":
            idx_b = idx
    if idx_a is None or idx_h is None or idx_b is None:
        raise RuntimeError("Complex A/H/B sites not found.")

    ra = sites[idx_a].position_angstrom
    rb = sites[idx_b].position_angstrom
    dab = [rb[k] - ra[k] for k in range(3)]
    r_ab = norm(dab)
    uab = [dab[k] / max(r_ab, 1e-12) for k in range(3)]

    v_ss = 0.0
    for i in range(len(sites)):
        si = sites[i]
        xi, yi, zi = si.position_angstrom
        for j in range(i + 1, len(sites)):
            sj = sites[j]
            if si.molecule_id == sj.molecule_id and si.molecule_id < n_solvent_molecules:
                continue
            pair = {si.site_type, sj.site_type}
            dx = xi - sj.position_angstrom[0]
            dy = yi - sj.position_angstrom[1]
            dz = zi - sj.position_angstrom[2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)

            if pair.issubset({"C1", "C2"}) or pair in ({"A", "C1"}, {"A", "C2"}, {"B", "C1"}, {"B", "C2"}):
                lj_params = get_lj_params(si.site_type, sj.site_type)
                if lj_params is not None:
                    sigma, epsilon = lj_params
                    e_lj, dUdr = lj_energy_dudr(r, sigma, epsilon)
                    v_ss += e_lj
                    scale = -dUdr / max(r, 1e-12)
                    fx, fy, fz = scale * dx, scale * dy, scale * dz
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[i][2] += fz
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                    forces[j][2] -= fz
                if pair.issubset({"C1", "C2"}):
                    qi = CHARGE[si.site_type]
                    qj = CHARGE[sj.site_type]
                    e_c, dUdr = coulomb_energy_dudr(r, qi, qj)
                    v_ss += e_c
                    scale = -dUdr / max(r, 1e-12)
                    fx, fy, fz = scale * dx, scale * dy, scale * dz
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[i][2] += fz
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                    forces[j][2] -= fz

    ngrid = len(grid)
    weights = [PROTON_GRID_STEP] * ngrid
    weights[0] *= 0.5
    weights[-1] *= 0.5

    omega = [0.0 for _ in range(ngrid)]
    for g in range(ngrid):
        tmp = 0.0
        for i in range(n_states):
            for j in range(n_states):
                tmp += estimator_matrix[i][j] * eigenstates[i][g] * eigenstates[j][g]
        omega[g] = tmp / (2.0 * PLANCK_REDUCED_KCAL_MOL_FS)

    v_cs = [[0.0] * n_states for _ in range(n_states)]
    for g, r_ah in enumerate(grid):
        fpol, _ = polarization_switch(r_ah)
        q_a = (1.0 - fpol) * POLARIZATION["Q_A_cov"] + fpol * POLARIZATION["Q_A_ion"]
        q_h = (1.0 - fpol) * POLARIZATION["Q_H_cov"] + fpol * POLARIZATION["Q_H_ion"]
        q_b = (1.0 - fpol) * POLARIZATION["Q_B_cov"] + fpol * POLARIZATION["Q_B_ion"]

        rh = [ra[k] + uab[k] * r_ah for k in range(3)]
        projector = [[eigenstates[i][g] * eigenstates[j][g] for j in range(n_states)] for i in range(n_states)]

        for s_idx in range(2 * n_solvent_molecules):
            ss = sites[s_idx]
            rs = ss.position_angstrom
            qs = CHARGE[ss.site_type]

            dx = rh[0] - rs[0]
            dy = rh[1] - rs[1]
            dz = rh[2] - rs[2]
            r_hs = math.sqrt(dx * dx + dy * dy + dz * dz)
            e_hs = 0.0
            dUdr_hs = 0.0
            if abs(q_h * qs) > 0.0:
                e_c, d_c = coulomb_energy_dudr(r_hs, q_h, qs)
                e_hs += e_c
                dUdr_hs += d_c

            dxa = ra[0] - rs[0]
            dya = ra[1] - rs[1]
            dza = ra[2] - rs[2]
            r_as = math.sqrt(dxa * dxa + dya * dya + dza * dza)
            e_as, dUdr_as = coulomb_energy_dudr(r_as, q_a, qs)

            dxb = rb[0] - rs[0]
            dyb = rb[1] - rs[1]
            dzb = rb[2] - rs[2]
            r_bs = math.sqrt(dxb * dxb + dyb * dyb + dzb * dzb)
            e_bs, dUdr_bs = coulomb_energy_dudr(r_bs, q_b, qs)

            v_gs = e_hs + e_as + e_bs
            wg = weights[g] * omega[g]
            for i in range(n_states):
                for j in range(n_states):
                    v_cs[i][j] += weights[g] * projector[i][j] * v_gs

            if abs(e_hs) > 0.0:
                scale_h = -dUdr_hs / max(r_hs, 1e-12)
                f_h = [scale_h * dx, scale_h * dy, scale_h * dz]
                forces[s_idx][0] -= wg * f_h[0]
                forces[s_idx][1] -= wg * f_h[1]
                forces[s_idx][2] -= wg * f_h[2]

                proj = [[(1.0 if a == b else 0.0) - (r_ah / max(r_ab, 1e-12)) * ((1.0 if a == b else 0.0) - uab[a] * uab[b]) for b in range(3)] for a in range(3)]
                for a in range(3):
                    forces[idx_a][a] += wg * sum(proj[b][a] * f_h[b] for b in range(3))
                proj_b = [[(r_ah / max(r_ab, 1e-12)) * ((1.0 if a == b else 0.0) - uab[a] * uab[b]) for b in range(3)] for a in range(3)]
                for a in range(3):
                    forces[idx_b][a] += wg * sum(proj_b[b][a] * f_h[b] for b in range(3))

            scale_a = -dUdr_as / max(r_as, 1e-12)
            f_a = [scale_a * dxa, scale_a * dya, scale_a * dza]
            forces[idx_a][0] += wg * f_a[0]
            forces[idx_a][1] += wg * f_a[1]
            forces[idx_a][2] += wg * f_a[2]
            forces[s_idx][0] -= wg * f_a[0]
            forces[s_idx][1] -= wg * f_a[1]
            forces[s_idx][2] -= wg * f_a[2]

            scale_b = -dUdr_bs / max(r_bs, 1e-12)
            f_b = [scale_b * dxb, scale_b * dyb, scale_b * dzb]
            forces[idx_b][0] += wg * f_b[0]
            forces[idx_b][1] += wg * f_b[1]
            forces[idx_b][2] += wg * f_b[2]
            forces[s_idx][0] -= wg * f_b[0]
            forces[s_idx][1] -= wg * f_b[1]
            forces[s_idx][2] -= wg * f_b[2]

    return forces, v_ss, v_cs, r_ab, idx_a, idx_b, omega, uab, ra, rb



def _edge_taper_weight(r_ab: float, r_min: float, r_max: float, taper_width: float) -> float:
    if taper_width <= 0.0:
        return 1.0
    span = r_max - r_min
    if span <= 0.0:
        return 1.0
    delta = min(taper_width, 0.49 * span)
    d_left = r_ab - r_min
    d_right = r_max - r_ab
    d = d_left if d_left < d_right else d_right
    if d <= 0.0:
        return 0.0
    if d >= delta:
        return 1.0
    x = d / delta
    return 0.5 * (1.0 - math.cos(math.pi * x))


@njit(cache=True)
def _lj_energy_dudr_numba(r: float, sigma: float, epsilon: float) -> Tuple[float, float]:
    inv_r = 1.0 / max(r, 1e-12)
    sr = sigma * inv_r
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    energy = 4.0 * epsilon * (sr12 - sr6)
    dudr = 4.0 * epsilon * (-12.0 * sr12 * inv_r + 6.0 * sr6 * inv_r)
    return energy, dudr


@njit(cache=True)
def _coulomb_energy_dudr_numba(r: float, qi: float, qj: float) -> Tuple[float, float]:
    inv_r = 1.0 / max(r, 1e-12)
    energy = COULOMB_KCAL_MOL_ANG_E2 * qi * qj * inv_r
    dudr = -COULOMB_KCAL_MOL_ANG_E2 * qi * qj * inv_r * inv_r
    return energy, dudr


@njit(cache=True)
def _polarization_switch_numba(r_ah: float) -> Tuple[float, float]:
    dr = r_ah - R0_POL
    l = L_POL
    denom = math.sqrt(dr * dr + l * l)
    f = 0.5 * (1.0 + dr / denom)
    df_dr = 0.5 * (l * l) / (denom ** 3)
    return f, df_dr


@njit(cache=True)
def _get_lj_sigma_epsilon_numba(type_i: int, type_j: int) -> Tuple[float, float, float]:
    a = type_i if type_i <= type_j else type_j
    b = type_j if type_i <= type_j else type_i

    if a <= 1 and b <= 1:
        sigma_a = 3.774 if a == 0 else 3.481
        sigma_b = 3.774 if b == 0 else 3.481
        epsilon_a = 0.238 if a == 0 else 0.415
        epsilon_b = 0.238 if b == 0 else 0.415
        return 1.0, 0.5 * (sigma_a + sigma_b), math.sqrt(epsilon_a * epsilon_b)

    if (a == 0 or a == 1) and (b == 2 or b == 3):
        return 1.0, 3.5, 0.3974

    return 0.0, 0.0, 0.0


@njit(cache=True)
def _compute_forces_numba_core(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    molecule_ids: np.ndarray,
    type_codes: np.ndarray,
    n_solvent_molecules: int,
    grid: np.ndarray,
    eigenstates: np.ndarray,
    h_diab: np.ndarray,
    dh_diab: np.ndarray,
    map_r: np.ndarray,
    map_p: np.ndarray,
) -> Tuple[np.ndarray, float, float, float, float, float, np.ndarray]:
    forces = np.zeros((positions.shape[0], 3), dtype=np.float64)
    idx_a = -1
    idx_h = -1
    idx_b = -1
    for idx in range(type_codes.shape[0]):
        t = type_codes[idx]
        if t == 2:
            idx_a = idx
        elif t == 4:
            idx_h = idx
        elif t == 3:
            idx_b = idx
    if idx_a < 0 or idx_h < 0 or idx_b < 0:
        raise RuntimeError("Complex A/H/B sites not found.")

    ra = positions[idx_a]
    rb = positions[idx_b]
    dab = np.empty(3, dtype=np.float64)
    for k in range(3):
        dab[k] = rb[k] - ra[k]
    r_ab = math.sqrt(dab[0] * dab[0] + dab[1] * dab[1] + dab[2] * dab[2])
    uab = np.empty(3, dtype=np.float64)
    for k in range(3):
        uab[k] = dab[k] / max(r_ab, 1e-12)

    v_ss = 0.0
    for i in range(positions.shape[0]):
        xi, yi, zi = positions[i][0], positions[i][1], positions[i][2]
        type_i = type_codes[i]
        for j in range(i + 1, positions.shape[0]):
            if molecule_ids[i] == molecule_ids[j] and molecule_ids[i] < n_solvent_molecules:
                continue
            type_j = type_codes[j]
            dx = xi - positions[j][0]
            dy = yi - positions[j][1]
            dz = zi - positions[j][2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)

            is_pair_solvent = type_i <= 1 and type_j <= 1
            is_pair_ab_solvent = ((type_i == 2 or type_i == 3) and (type_j <= 1)) or ((type_j == 2 or type_j == 3) and (type_i <= 1))
            if is_pair_solvent or is_pair_ab_solvent:
                has_lj, sigma, epsilon = _get_lj_sigma_epsilon_numba(type_i, type_j)
                if has_lj > 0.5:
                    e_lj, dudr = _lj_energy_dudr_numba(r, sigma, epsilon)
                    v_ss += e_lj
                    scale = -dudr / max(r, 1e-12)
                    fx, fy, fz = scale * dx, scale * dy, scale * dz
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[i][2] += fz
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                    forces[j][2] -= fz
                if is_pair_solvent:
                    qi = 0.25 if type_i == 0 else -0.25
                    qj = 0.25 if type_j == 0 else -0.25
                    e_c, dudr = _coulomb_energy_dudr_numba(r, qi, qj)
                    v_ss += e_c
                    scale = -dudr / max(r, 1e-12)
                    fx, fy, fz = scale * dx, scale * dy, scale * dz
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[i][2] += fz
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                    forces[j][2] -= fz

    weights = np.full(grid.shape[0], PROTON_GRID_STEP, dtype=np.float64)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    n_states = map_r.shape[0]

    s_map = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        for j in range(n_states):
            s_map[i, j] = map_r[i] * map_r[j] + map_p[i] * map_p[j] - (PLANCK_REDUCED_KCAL_MOL_FS if i == j else 0.0)

    omega = np.zeros(grid.shape[0], dtype=np.float64)
    for g in range(grid.shape[0]):
        tmp = 0.0
        for i in range(n_states):
            for j in range(n_states):
                tmp += s_map[i, j] * eigenstates[i, g] * eigenstates[j, g]
        omega[g] = tmp / (2.0 * PLANCK_REDUCED_KCAL_MOL_FS)

    v_cs = np.zeros((n_states, n_states), dtype=np.float64)
    for g in range(grid.shape[0]):
        r_ah = grid[g]
        fpol, _ = _polarization_switch_numba(r_ah)
        q_a = (1.0 - fpol) * Q_A_COV + fpol * Q_A_ION
        q_h = (1.0 - fpol) * Q_H_COV + fpol * Q_H_ION
        q_b = (1.0 - fpol) * Q_B_COV + fpol * Q_B_ION

        rh = np.empty(3, dtype=np.float64)
        for k in range(3):
            rh[k] = ra[k] + uab[k] * r_ah

        for s_idx in range(2 * n_solvent_molecules):
            rs = positions[s_idx]
            qs = 0.25 if type_codes[s_idx] == 0 else -0.25

            dx = rh[0] - rs[0]
            dy = rh[1] - rs[1]
            dz = rh[2] - rs[2]
            r_hs = math.sqrt(dx * dx + dy * dy + dz * dz)
            e_hs = 0.0
            dUdr_hs = 0.0
            if abs(q_h * qs) > 0.0:
                e_c, d_c = _coulomb_energy_dudr_numba(r_hs, q_h, qs)
                e_hs += e_c
                dUdr_hs += d_c

            dxa = ra[0] - rs[0]
            dya = ra[1] - rs[1]
            dza = ra[2] - rs[2]
            r_as = math.sqrt(dxa * dxa + dya * dya + dza * dza)
            e_as, dUdr_as = _coulomb_energy_dudr_numba(r_as, q_a, qs)

            dxb = rb[0] - rs[0]
            dyb = rb[1] - rs[1]
            dzb = rb[2] - rs[2]
            r_bs = math.sqrt(dxb * dxb + dyb * dyb + dzb * dzb)
            e_bs, dUdr_bs = _coulomb_energy_dudr_numba(r_bs, q_b, qs)

            v_gs = e_hs + e_as + e_bs
            wg = weights[g] * omega[g]
            for i in range(n_states):
                for j in range(n_states):
                    v_cs[i, j] += weights[g] * eigenstates[i, g] * eigenstates[j, g] * v_gs

            if abs(e_hs) > 0.0:
                scale_h = -dUdr_hs / max(r_hs, 1e-12)
                f_h = np.empty(3, dtype=np.float64)
                f_h[0] = scale_h * dx
                f_h[1] = scale_h * dy
                f_h[2] = scale_h * dz
                forces[s_idx, 0] -= wg * f_h[0]
                forces[s_idx, 1] -= wg * f_h[1]
                forces[s_idx, 2] -= wg * f_h[2]

                ratio = r_ah / max(r_ab, 1e-12)
                for a in range(3):
                    sum_a = 0.0
                    sum_b = 0.0
                    for b in range(3):
                        delta = 1.0 if a == b else 0.0
                        proj = delta - ratio * (delta - uab[a] * uab[b])
                        proj_b = ratio * (delta - uab[a] * uab[b])
                        sum_a += proj * f_h[b]
                        sum_b += proj_b * f_h[b]
                    forces[idx_a, a] += wg * sum_a
                    forces[idx_b, a] += wg * sum_b

            scale_a = -dUdr_as / max(r_as, 1e-12)
            f_a0 = scale_a * dxa
            f_a1 = scale_a * dya
            f_a2 = scale_a * dza
            forces[idx_a, 0] += wg * f_a0
            forces[idx_a, 1] += wg * f_a1
            forces[idx_a, 2] += wg * f_a2
            forces[s_idx, 0] -= wg * f_a0
            forces[s_idx, 1] -= wg * f_a1
            forces[s_idx, 2] -= wg * f_a2

            scale_b = -dUdr_bs / max(r_bs, 1e-12)
            f_b0 = scale_b * dxb
            f_b1 = scale_b * dyb
            f_b2 = scale_b * dzb
            forces[idx_b, 0] += wg * f_b0
            forces[idx_b, 1] += wg * f_b1
            forces[idx_b, 2] += wg * f_b2
            forces[s_idx, 0] -= wg * f_b0
            forces[s_idx, 1] -= wg * f_b1
            forces[s_idx, 2] -= wg * f_b2

    k_total = 0.0
    for i in range(positions.shape[0]):
        if type_codes[i] == 4:
            continue
        vx, vy, vz = velocities[i][0], velocities[i][1], velocities[i][2]
        k_total += 0.5 * masses[i] * (vx * vx + vy * vy + vz * vz)
    k_total *= AMU_ANG2_FS2_TO_KCAL_MOL

    h_eff = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        for j in range(n_states):
            h_eff[i, j] = h_diab[i, j] + v_cs[i, j]
    e_h = 0.0
    dE_dR = 0.0
    for i in range(n_states):
        for j in range(n_states):
            e_h += h_eff[i, j] * s_map[i, j]
            dE_dR += dh_diab[i, j] * s_map[i, j]
    e_h /= 2.0 * PLANCK_REDUCED_KCAL_MOL_FS
    dE_dR /= 2.0 * PLANCK_REDUCED_KCAL_MOL_FS

    for k in range(3):
        f_b = -dE_dR * uab[k]
        forces[idx_b, k] += f_b
        forces[idx_a, k] -= f_b

    h_map = k_total + v_ss + e_h
    return forces, k_total, v_ss, e_h, h_map, dE_dR, h_eff


def compute_legacy_mapping_forces_and_hamiltonian(
    sites: List[Site],
    n_solvent_molecules: int,
    diabatic_table: Dict[str, object],
    map_r: List[float],
    map_p: List[float],
    kernel_backend: str = "python",
    edge_taper_width_angstrom: float = 0.02,
) -> Tuple[List[List[float]], Dict[str, object]]:
    if kernel_backend not in {"python", "numba"}:
        raise ValueError("kernel_backend must be 'python' or 'numba'.")

    idx_a = idx_h = idx_b = None
    for idx, site in enumerate(sites):
        if site.site_type == "A":
            idx_a = idx
        elif site.site_type == "H":
            idx_h = idx
        elif site.site_type == "B":
            idx_b = idx
    if idx_a is None or idx_h is None or idx_b is None:
        raise RuntimeError("Complex A/H/B sites not found.")

    ra = sites[idx_a].position_angstrom
    rb = sites[idx_b].position_angstrom
    dab = [rb[k] - ra[k] for k in range(3)]
    r_ab = norm(dab)
    r_min, r_max = diabatic_r_range(diabatic_table)
    if r_ab < r_min or r_ab > r_max:
        raise RuntimeError(f"R_AB={r_ab:.6f} outside diabatic table range [{r_min:.6f}, {r_max:.6f}].")

    h_diab, dh_diab = interpolate_h_diabatic(diabatic_table, r_ab)
    eigenstates = interpolate_eigenstates(diabatic_table, r_ab)
    grid = diabatic_table["grid"]
    n_states = int(diabatic_table.get("n_states", len(h_diab)))
    w_taper = _edge_taper_weight(r_ab, r_min, r_max, edge_taper_width_angstrom)
    dh_diab_eff = [[w_taper * dh_diab[i][j] for j in range(n_states)] for i in range(n_states)]
    if len(map_r) != n_states or len(map_p) != n_states:
        raise RuntimeError(
            f"Mapping variable dimension mismatch: len(map_r)={len(map_r)}, len(map_p)={len(map_p)}, n_states={n_states}."
        )

    if kernel_backend == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba backend requested but numba is not installed.")
        if np is None:
            raise RuntimeError("Numba backend requested but numpy is not installed.")

        positions = np.asarray([s.position_angstrom for s in sites], dtype=np.float64)
        velocities = np.asarray([s.velocity_ang_fs for s in sites], dtype=np.float64)
        masses = np.asarray([s.mass_amu for s in sites], dtype=np.float64)
        molecule_ids = np.asarray([s.molecule_id for s in sites], dtype=np.int64)
        type_codes = np.asarray([SITE_TYPE_TO_CODE[s.site_type] for s in sites], dtype=np.int64)
        forces, k_total, v_ss, e_h, h_map, dE_dR, h_eff = _compute_forces_numba_core(
            positions=positions,
            velocities=velocities,
            masses=masses,
            molecule_ids=molecule_ids,
            type_codes=type_codes,
            n_solvent_molecules=n_solvent_molecules,
            grid=np.asarray(grid, dtype=np.float64),
            eigenstates=np.asarray(eigenstates, dtype=np.float64),
            h_diab=np.asarray(h_diab, dtype=np.float64),
            dh_diab=np.asarray(dh_diab_eff, dtype=np.float64),
            map_r=np.asarray(map_r, dtype=np.float64),
            map_p=np.asarray(map_p, dtype=np.float64),
        )
        return forces.tolist(), {
            "K": k_total,
            "V_SS": v_ss,
            "E_map_coupling": e_h,
            "H_map": h_map,
            "R_AB": r_ab,
            "w_RAB_taper": w_taper,
            "dE_dRAB": dE_dR,
            "h_eff": h_eff.tolist(),
        }

    forces = [[0.0, 0.0, 0.0] for _ in sites]

    idx_a = idx_h = idx_b = None
    for idx, site in enumerate(sites):
        if site.site_type == "A":
            idx_a = idx
        elif site.site_type == "H":
            idx_h = idx
        elif site.site_type == "B":
            idx_b = idx
    if idx_a is None or idx_h is None or idx_b is None:
        raise RuntimeError("Complex A/H/B sites not found.")

    ra = sites[idx_a].position_angstrom
    rb = sites[idx_b].position_angstrom
    dab = [rb[k] - ra[k] for k in range(3)]
    r_ab = norm(dab)
    r_min, r_max = diabatic_r_range(diabatic_table)
    if r_ab < r_min or r_ab > r_max:
        raise RuntimeError(f"R_AB={r_ab:.6f} outside diabatic table range [{r_min:.6f}, {r_max:.6f}].")
    uab = [dab[k] / max(r_ab, 1e-12) for k in range(3)]

    h_diab, dh_diab = interpolate_h_diabatic(diabatic_table, r_ab)
    eigenstates = interpolate_eigenstates(diabatic_table, r_ab)
    grid = diabatic_table["grid"]

    v_ss = 0.0
    for i in range(len(sites)):
        si = sites[i]
        xi, yi, zi = si.position_angstrom
        for j in range(i + 1, len(sites)):
            sj = sites[j]
            if si.molecule_id == sj.molecule_id and si.molecule_id < n_solvent_molecules:
                continue
            pair = {si.site_type, sj.site_type}
            dx = xi - sj.position_angstrom[0]
            dy = yi - sj.position_angstrom[1]
            dz = zi - sj.position_angstrom[2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)

            if pair.issubset({"C1", "C2"}) or pair in ({"A", "C1"}, {"A", "C2"}, {"B", "C1"}, {"B", "C2"}):
                lj_params = get_lj_params(si.site_type, sj.site_type)
                if lj_params is not None:
                    sigma, epsilon = lj_params
                    e_lj, dUdr = lj_energy_dudr(r, sigma, epsilon)
                    v_ss += e_lj
                    scale = -dUdr / max(r, 1e-12)
                    fx, fy, fz = scale * dx, scale * dy, scale * dz
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[i][2] += fz
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                    forces[j][2] -= fz
                if pair.issubset({"C1", "C2"}):
                    qi = CHARGE[si.site_type]
                    qj = CHARGE[sj.site_type]
                    e_c, dUdr = coulomb_energy_dudr(r, qi, qj)
                    v_ss += e_c
                    scale = -dUdr / max(r, 1e-12)
                    fx, fy, fz = scale * dx, scale * dy, scale * dz
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[i][2] += fz
                    forces[j][0] -= fx
                    forces[j][1] -= fy
                    forces[j][2] -= fz

    ngrid = len(grid)
    weights = [PROTON_GRID_STEP] * ngrid
    weights[0] *= 0.5
    weights[-1] *= 0.5

    s_map = [[map_r[i] * map_r[j] + map_p[i] * map_p[j] - (PLANCK_REDUCED_KCAL_MOL_FS if i == j else 0.0) for j in range(n_states)] for i in range(n_states)]

    omega = [0.0 for _ in range(ngrid)]
    for g in range(ngrid):
        tmp = 0.0
        for i in range(n_states):
            for j in range(n_states):
                tmp += s_map[i][j] * eigenstates[i][g] * eigenstates[j][g]
        omega[g] = tmp / (2.0 * PLANCK_REDUCED_KCAL_MOL_FS)

    v_cs = [[0.0] * n_states for _ in range(n_states)]
    for g, r_ah in enumerate(grid):
        fpol, _ = polarization_switch(r_ah)
        q_a = (1.0 - fpol) * POLARIZATION["Q_A_cov"] + fpol * POLARIZATION["Q_A_ion"]
        q_h = (1.0 - fpol) * POLARIZATION["Q_H_cov"] + fpol * POLARIZATION["Q_H_ion"]
        q_b = (1.0 - fpol) * POLARIZATION["Q_B_cov"] + fpol * POLARIZATION["Q_B_ion"]

        rh = [ra[k] + uab[k] * r_ah for k in range(3)]
        projector = [[eigenstates[i][g] * eigenstates[j][g] for j in range(n_states)] for i in range(n_states)]

        for s_idx in range(2 * n_solvent_molecules):
            ss = sites[s_idx]
            rs = ss.position_angstrom
            qs = CHARGE[ss.site_type]

            dx = rh[0] - rs[0]
            dy = rh[1] - rs[1]
            dz = rh[2] - rs[2]
            r_hs = math.sqrt(dx * dx + dy * dy + dz * dz)
            e_hs = 0.0
            dUdr_hs = 0.0
            if abs(q_h * qs) > 0.0:
                e_c, d_c = coulomb_energy_dudr(r_hs, q_h, qs)
                e_hs += e_c
                dUdr_hs += d_c

            dxa = ra[0] - rs[0]
            dya = ra[1] - rs[1]
            dza = ra[2] - rs[2]
            r_as = math.sqrt(dxa * dxa + dya * dya + dza * dza)
            e_as, dUdr_as = coulomb_energy_dudr(r_as, q_a, qs)

            dxb = rb[0] - rs[0]
            dyb = rb[1] - rs[1]
            dzb = rb[2] - rs[2]
            r_bs = math.sqrt(dxb * dxb + dyb * dyb + dzb * dzb)
            e_bs, dUdr_bs = coulomb_energy_dudr(r_bs, q_b, qs)

            v_gs = e_hs + e_as + e_bs
            wg = weights[g] * omega[g]
            for i in range(n_states):
                for j in range(n_states):
                    v_cs[i][j] += weights[g] * projector[i][j] * v_gs

            if abs(e_hs) > 0.0:
                scale_h = -dUdr_hs / max(r_hs, 1e-12)
                f_h = [scale_h * dx, scale_h * dy, scale_h * dz]
                forces[s_idx][0] -= wg * f_h[0]
                forces[s_idx][1] -= wg * f_h[1]
                forces[s_idx][2] -= wg * f_h[2]

                proj = [[(1.0 if a == b else 0.0) - (r_ah / max(r_ab, 1e-12)) * ((1.0 if a == b else 0.0) - uab[a] * uab[b]) for b in range(3)] for a in range(3)]
                for a in range(3):
                    forces[idx_a][a] += wg * sum(proj[b][a] * f_h[b] for b in range(3))
                proj_b = [[(r_ah / max(r_ab, 1e-12)) * ((1.0 if a == b else 0.0) - uab[a] * uab[b]) for b in range(3)] for a in range(3)]
                for a in range(3):
                    forces[idx_b][a] += wg * sum(proj_b[b][a] * f_h[b] for b in range(3))

            scale_a = -dUdr_as / max(r_as, 1e-12)
            f_a = [scale_a * dxa, scale_a * dya, scale_a * dza]
            forces[idx_a][0] += wg * f_a[0]
            forces[idx_a][1] += wg * f_a[1]
            forces[idx_a][2] += wg * f_a[2]
            forces[s_idx][0] -= wg * f_a[0]
            forces[s_idx][1] -= wg * f_a[1]
            forces[s_idx][2] -= wg * f_a[2]

            scale_b = -dUdr_bs / max(r_bs, 1e-12)
            f_b = [scale_b * dxb, scale_b * dyb, scale_b * dzb]
            forces[idx_b][0] += wg * f_b[0]
            forces[idx_b][1] += wg * f_b[1]
            forces[idx_b][2] += wg * f_b[2]
            forces[s_idx][0] -= wg * f_b[0]
            forces[s_idx][1] -= wg * f_b[1]
            forces[s_idx][2] -= wg * f_b[2]

    k_total = kinetic_energy_kcal_mol(sites, exclude_h=True)

    h_eff = [[h_diab[i][j] + v_cs[i][j] for j in range(n_states)] for i in range(n_states)]

    e_h = 0.0
    dE_dR = 0.0
    for i in range(n_states):
        for j in range(n_states):
            e_h += h_eff[i][j] * s_map[i][j]
            dE_dR += dh_diab_eff[i][j] * s_map[i][j]
    e_h /= (2.0 * PLANCK_REDUCED_KCAL_MOL_FS)
    dE_dR /= (2.0 * PLANCK_REDUCED_KCAL_MOL_FS)

    for k in range(3):
        f_b = -dE_dR * uab[k]
        forces[idx_b][k] += f_b
        forces[idx_a][k] -= f_b

    h_map = k_total + v_ss + e_h

    return forces, {
        "K": k_total,
        "V_SS": v_ss,
        "E_map_coupling": e_h,
        "H_map": h_map,
        "R_AB": r_ab,
        "w_RAB_taper": w_taper,
        "dE_dRAB": dE_dR,
        "h_eff": h_eff,
    }



def compute_fbts_forces_and_hamiltonian(
    sites: List[Site],
    n_solvent_molecules: int,
    diabatic_table: Dict[str, object],
    fbts_state: FBTSElectronicState,
    kernel_backend: str = "python",
    edge_taper_width_angstrom: float = 0.02,
) -> ForceEvalResult:
    _validate_fbts_kernel_backend(kernel_backend)

    ra = rb = None
    for site in sites:
        if site.site_type == "A":
            ra = site.position_angstrom
        elif site.site_type == "B":
            rb = site.position_angstrom
    if ra is None or rb is None:
        raise RuntimeError("Complex A/B sites not found.")

    r_ab = norm([rb[k] - ra[k] for k in range(3)])
    r_min, r_max = diabatic_r_range(diabatic_table)
    if r_ab < r_min or r_ab > r_max:
        raise RuntimeError(f"R_AB={r_ab:.6f} outside diabatic table range [{r_min:.6f}, {r_max:.6f}].")

    h_diab, dh_diab = interpolate_h_diabatic(diabatic_table, r_ab)
    eigenstates = interpolate_eigenstates(diabatic_table, r_ab)
    grid = diabatic_table["grid"]
    n_states = int(diabatic_table.get("n_states", len(h_diab)))

    if not (
        len(fbts_state.r_f) == len(fbts_state.p_f) == len(fbts_state.r_b) == len(fbts_state.p_b) == n_states
    ):
        raise RuntimeError("FBTS electronic state dimension mismatch with diabatic model.")

    w_taper = _edge_taper_weight(r_ab, r_min, r_max, edge_taper_width_angstrom)
    dh_diab_eff = [[w_taper * dh_diab[i][j] for j in range(n_states)] for i in range(n_states)]

    estimator_matrix = _fbts_estimator_matrix(fbts_state)
    forces, v_ss, v_cs, r_ab, idx_a, idx_b, omega, uab, _ra, _rb = _compute_bath_and_coupling_matrices_python(
        sites,
        n_solvent_molecules,
        grid,
        eigenstates,
        estimator_matrix,
        n_states,
    )

    k_total = kinetic_energy_kcal_mol(sites, exclude_h=True)
    h_eff = [[h_diab[i][j] + v_cs[i][j] for j in range(n_states)] for i in range(n_states)]

    e_h = 0.0
    dE_dR = 0.0
    for i in range(n_states):
        for j in range(n_states):
            e_h += h_eff[i][j] * estimator_matrix[i][j]
            dE_dR += dh_diab_eff[i][j] * estimator_matrix[i][j]
    e_h /= (2.0 * PLANCK_REDUCED_KCAL_MOL_FS)
    dE_dR /= (2.0 * PLANCK_REDUCED_KCAL_MOL_FS)

    for k in range(3):
        f_b = -dE_dR * uab[k]
        forces[idx_b][k] += f_b
        forces[idx_a][k] -= f_b

    h_map = k_total + v_ss + e_h
    terms = {
        "K": k_total,
        "V_SS": v_ss,
        "E_fbts_coupling": e_h,
        "E_map_coupling": e_h,
        "H_fbts": h_map,
        "H_map": h_map,
        "R_AB": r_ab,
        "w_RAB_taper": w_taper,
        "dE_dRAB": dE_dR,
        "h_eff": h_eff,
        "v_cs": v_cs,
        "estimator_matrix": estimator_matrix,
        "omega": omega,
        "kernel_backend": kernel_backend,
    }
    return ForceEvalResult(forces=forces, terms=terms)



def compute_legacy_mapping_forces_and_hamiltonian(
    sites: List[Site],
    n_solvent_molecules: int,
    diabatic_table: Dict[str, object],
    map_r: List[float],
    map_p: List[float],
    kernel_backend: str = "python",
    edge_taper_width_angstrom: float = 0.02,
) -> Tuple[List[List[float]], Dict[str, object]]:
    """Legacy compatibility shim: routes old mapping-style callsites through FBTS evaluator."""
    fbts_state = FBTSElectronicState(
        r_f=list(map_r),
        p_f=list(map_p),
        r_b=list(map_r),
        p_b=list(map_p),
    )
    result = compute_fbts_forces_and_hamiltonian(
        sites=sites,
        n_solvent_molecules=n_solvent_molecules,
        diabatic_table=diabatic_table,
        fbts_state=fbts_state,
        kernel_backend=kernel_backend,
        edge_taper_width_angstrom=edge_taper_width_angstrom,
    )
    return result.forces, result.terms


def clone_sites(sites: List[Site]) -> List[Site]:
    return [
        Site(
            molecule_id=site.molecule_id,
            site_type=site.site_type,
            label=site.label,
            mass_amu=site.mass_amu,
            position_angstrom=list(site.position_angstrom),
            velocity_ang_fs=list(site.velocity_ang_fs),
        )
        for site in sites
    ]


def finite_difference_force_spot_checks(
    sites: List[Site],
    n_solvent_molecules: int,
    diabatic_table: Dict[str, object],
    map_r: List[float],
    map_p: List[float],
    delta: float,
    kernel_backend: str = "python",
    edge_taper_width_angstrom: float = 0.02,
) -> Dict[str, float]:
    forces, terms = compute_legacy_mapping_forces_and_hamiltonian(
        sites,
        n_solvent_molecules,
        diabatic_table,
        map_r,
        map_p,
        kernel_backend=kernel_backend,
        edge_taper_width_angstrom=edge_taper_width_angstrom,
    )

    idx_a = idx_b = idx_h = None
    for idx, site in enumerate(sites):
        if site.site_type == "A":
            idx_a = idx
        elif site.site_type == "B":
            idx_b = idx
        elif site.site_type == "H":
            idx_h = idx

    check_sites = [0, 1]
    if idx_a is not None:
        check_sites.append(idx_a)
    if idx_b is not None:
        check_sites.append(idx_b)
    if idx_h is not None:
        check_sites.append(idx_h)
    check_sites = sorted({i for i in check_sites if 0 <= i < len(sites)})

    max_abs = 0.0
    max_rel = 0.0
    sum_abs = 0.0
    count = 0

    for site_idx in check_sites:
        for axis in range(3):
            plus_sites = clone_sites(sites)
            minus_sites = clone_sites(sites)
            plus_sites[site_idx].position_angstrom[axis] += delta
            minus_sites[site_idx].position_angstrom[axis] -= delta

            _, terms_plus = compute_legacy_mapping_forces_and_hamiltonian(
                plus_sites, n_solvent_molecules, diabatic_table, map_r, map_p, kernel_backend=kernel_backend,
                edge_taper_width_angstrom=edge_taper_width_angstrom
            )
            _, terms_minus = compute_legacy_mapping_forces_and_hamiltonian(
                minus_sites, n_solvent_molecules, diabatic_table, map_r, map_p, kernel_backend=kernel_backend,
                edge_taper_width_angstrom=edge_taper_width_angstrom
            )

            fd_force = -(terms_plus["H_map"] - terms_minus["H_map"]) / (2.0 * delta)
            an_force = forces[site_idx][axis]
            abs_err = abs(an_force - fd_force)
            rel_err = abs_err / max(abs(fd_force), 1e-8)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            sum_abs += abs_err
            count += 1

    mean_abs = sum_abs / max(count, 1)
    return {
        "fd_count": float(count),
        "fd_delta": delta,
        "fd_max_abs_err": max_abs,
        "fd_max_rel_err": max_rel,
        "fd_mean_abs_err": mean_abs,
        "H_map": terms["H_map"],
    }


def enforce_solvent_bond_constraints(sites: List[Site], n_solvent_molecules: int, bond_distance: float) -> None:
    for m in range(n_solvent_molecules):
        i = 2 * m
        j = i + 1
        si = sites[i]
        sj = sites[j]

        rij = [
            sj.position_angstrom[0] - si.position_angstrom[0],
            sj.position_angstrom[1] - si.position_angstrom[1],
            sj.position_angstrom[2] - si.position_angstrom[2],
        ]
        r = norm(rij)
        if r < 1e-12:
            continue

        unit = [rij[0] / r, rij[1] / r, rij[2] / r]
        delta = r - bond_distance
        mi = si.mass_amu
        mj = sj.mass_amu
        mt = mi + mj

        corr_i = (mj / mt) * delta
        corr_j = -(mi / mt) * delta

        for k in range(3):
            si.position_angstrom[k] += corr_i * unit[k]
            sj.position_angstrom[k] += corr_j * unit[k]


def enforce_solvent_velocity_constraints(sites: List[Site], n_solvent_molecules: int) -> None:
    for m in range(n_solvent_molecules):
        i = 2 * m
        j = i + 1
        si = sites[i]
        sj = sites[j]

        rij = [
            sj.position_angstrom[0] - si.position_angstrom[0],
            sj.position_angstrom[1] - si.position_angstrom[1],
            sj.position_angstrom[2] - si.position_angstrom[2],
        ]
        r = norm(rij)
        if r < 1e-12:
            continue
        unit = [rij[0] / r, rij[1] / r, rij[2] / r]

        vrel = [
            sj.velocity_ang_fs[0] - si.velocity_ang_fs[0],
            sj.velocity_ang_fs[1] - si.velocity_ang_fs[1],
            sj.velocity_ang_fs[2] - si.velocity_ang_fs[2],
        ]
        radial_rel = dot(vrel, unit)

        mi = si.mass_amu
        mj = sj.mass_amu
        mt = mi + mj

        for k in range(3):
            si.velocity_ang_fs[k] += (mj / mt) * radial_rel * unit[k]
            sj.velocity_ang_fs[k] -= (mi / mt) * radial_rel * unit[k]


def add_ahb_complex(
    sites: List[Site],
    n_solvent_molecules: int,
    temperature_k: float,
    radius_angstrom: float,
    min_inter_site_distance_angstrom: float,
    seed_rng: random.Random,
) -> None:
    complex_mol_id = n_solvent_molecules

    for _attempt in range(50000):
        a_pos = random_point_in_sphere(seed_rng, radius_angstrom)
        axis = random_unit_vector(seed_rng)
        h_pos = (
            a_pos[0] + 1.0 * axis[0],
            a_pos[1] + 1.0 * axis[1],
            a_pos[2] + 1.0 * axis[2],
        )
        b_pos = (
            a_pos[0] + 2.7 * axis[0],
            a_pos[1] + 2.7 * axis[1],
            a_pos[2] + 2.7 * axis[2],
        )

        if (
            norm([a_pos[0], a_pos[1], a_pos[2]]) > radius_angstrom
            or norm([h_pos[0], h_pos[1], h_pos[2]]) > radius_angstrom
            or norm([b_pos[0], b_pos[1], b_pos[2]]) > radius_angstrom
        ):
            continue

        if all(
            dist(a_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
            and dist(h_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
            and dist(b_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
            for existing in sites
        ):
            sites.append(
                Site(
                    molecule_id=complex_mol_id,
                    site_type="A",
                    label=LABEL["A"],
                    mass_amu=MASS["A"],
                    position_angstrom=[a_pos[0], a_pos[1], a_pos[2]],
                    velocity_ang_fs=sample_velocity(seed_rng, temperature_k, MASS["A"]),
                )
            )
            sites.append(
                Site(
                    molecule_id=complex_mol_id,
                    site_type="H",
                    label=LABEL["H"],
                    mass_amu=MASS["H"],
                    position_angstrom=[h_pos[0], h_pos[1], h_pos[2]],
                    velocity_ang_fs=sample_velocity(seed_rng, temperature_k, MASS["H"]),
                )
            )
            sites.append(
                Site(
                    molecule_id=complex_mol_id,
                    site_type="B",
                    label=LABEL["B"],
                    mass_amu=MASS["B"],
                    position_angstrom=[b_pos[0], b_pos[1], b_pos[2]],
                    velocity_ang_fs=sample_velocity(seed_rng, temperature_k, MASS["B"]),
                )
            )
            return

    raise RuntimeError("Could not place A-H-B complex inside placement radius without overlaps.")


def generate_configuration(
    n_molecules: int,
    temperature_k: float,
    radius_angstrom: float,
    c1_c2_distance_angstrom: float,
    min_inter_site_distance_angstrom: float,
    seed: int,
) -> List[Site]:
    rng = random.Random(seed)
    sites: List[Site] = []
    half_bond = 0.5 * c1_c2_distance_angstrom

    for mol_idx in range(n_molecules):
        placed = False
        for _attempt in range(20000):
            center = random_point_in_sphere(rng, radius_angstrom)
            direction = random_unit_vector(rng)

            c1_pos = (
                center[0] - half_bond * direction[0],
                center[1] - half_bond * direction[1],
                center[2] - half_bond * direction[2],
            )
            c2_pos = (
                center[0] + half_bond * direction[0],
                center[1] + half_bond * direction[1],
                center[2] + half_bond * direction[2],
            )

            if all(
                dist(c1_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
                and dist(c2_pos, existing.position_angstrom) >= min_inter_site_distance_angstrom
                for existing in sites
            ):
                sites.append(
                    Site(
                        molecule_id=mol_idx,
                        site_type="C1",
                        label=LABEL["C1"],
                        mass_amu=MASS["C1"],
                        position_angstrom=[c1_pos[0], c1_pos[1], c1_pos[2]],
                        velocity_ang_fs=sample_velocity(rng, temperature_k, MASS["C1"]),
                    )
                )
                sites.append(
                    Site(
                        molecule_id=mol_idx,
                        site_type="C2",
                        label=LABEL["C2"],
                        mass_amu=MASS["C2"],
                        position_angstrom=[c2_pos[0], c2_pos[1], c2_pos[2]],
                        velocity_ang_fs=sample_velocity(rng, temperature_k, MASS["C2"]),
                    )
                )
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place solvent molecule {mol_idx + 1} without overlaps after many attempts."
            )

    add_ahb_complex(
        sites=sites,
        n_solvent_molecules=n_molecules,
        temperature_k=temperature_k,
        radius_angstrom=radius_angstrom,
        min_inter_site_distance_angstrom=min_inter_site_distance_angstrom,
        seed_rng=rng,
    )

    remove_net_linear_momentum(sites)
    enforce_solvent_bond_constraints(sites, n_molecules, c1_c2_distance_angstrom)
    enforce_solvent_velocity_constraints(sites, n_molecules)
    return sites



def apply_spherical_reflecting_wall(sites: List[Site], wall_radius_angstrom: float) -> None:
    if wall_radius_angstrom <= 0.0:
        return

    for site in sites:
        if site.site_type == "H":
            continue

        x, y, z = site.position_angstrom
        r = math.sqrt(x * x + y * y + z * z)
        if r <= wall_radius_angstrom:
            continue

        ux = x / max(r, 1e-12)
        uy = y / max(r, 1e-12)
        uz = z / max(r, 1e-12)

        overshoot = r - wall_radius_angstrom
        reflected_r = max(wall_radius_angstrom - overshoot, 0.0)
        site.position_angstrom[0] = reflected_r * ux
        site.position_angstrom[1] = reflected_r * uy
        site.position_angstrom[2] = reflected_r * uz

        vn = (
            site.velocity_ang_fs[0] * ux
            + site.velocity_ang_fs[1] * uy
            + site.velocity_ang_fs[2] * uz
        )
        if vn > 0.0:
            site.velocity_ang_fs[0] -= 2.0 * vn * ux
            site.velocity_ang_fs[1] -= 2.0 * vn * uy
            site.velocity_ang_fs[2] -= 2.0 * vn * uz



def compute_observables(sites: List[Site], n_solvent_molecules: int) -> Tuple[float, float]:
    idx_a = idx_b = None
    for idx, site in enumerate(sites):
        if site.site_type == "A":
            idx_a = idx
        elif site.site_type == "B":
            idx_b = idx
    if idx_a is None or idx_b is None:
        return 0.0, 0.0

    ra = sites[idx_a].position_angstrom
    rb = sites[idx_b].position_angstrom
    vab = [rb[k] - ra[k] for k in range(3)]
    rab = norm(vab)
    uab = [v / max(rab, 1e-12) for v in vab]

    s1 = [ra[k] + 1.0 * uab[k] for k in range(3)]
    s2 = [ra[k] + 1.6 * uab[k] for k in range(3)]

    dE = 0.0
    for i in range(2 * n_solvent_molecules):
        si = sites[i]
        qi = CHARGE[si.site_type]
        dx1 = si.position_angstrom[0] - s1[0]
        dy1 = si.position_angstrom[1] - s1[1]
        dz1 = si.position_angstrom[2] - s1[2]
        r1 = math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1)

        dx2 = si.position_angstrom[0] - s2[0]
        dy2 = si.position_angstrom[1] - s2[1]
        dz2 = si.position_angstrom[2] - s2[2]
        r2 = math.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2)

        dE += qi * ((1.0 / max(r1, 1e-12)) - (1.0 / max(r2, 1e-12)))

    m_a = MASS["A"]
    m_b = MASS["B"]
    m_ab = m_a + m_b
    com_complex = [
        (m_a * ra[0] + m_b * rb[0]) / m_ab,
        (m_a * ra[1] + m_b * rb[1]) / m_ab,
        (m_a * ra[2] + m_b * rb[2]) / m_ab,
    ]

    m_sol = 0.0
    com_sol = [0.0, 0.0, 0.0]
    for i in range(2 * n_solvent_molecules):
        si = sites[i]
        m = si.mass_amu
        m_sol += m
        com_sol[0] += m * si.position_angstrom[0]
        com_sol[1] += m * si.position_angstrom[1]
        com_sol[2] += m * si.position_angstrom[2]
    if m_sol > 0.0:
        com_sol = [c / m_sol for c in com_sol]

    dx = com_complex[0] - com_sol[0]
    dy = com_complex[1] - com_sol[1]
    dz = com_complex[2] - com_sol[2]
    r_com = math.sqrt(dx * dx + dy * dy + dz * dz)

    return dE, r_com


def run_nve_md(
    sites: List[Site],
    n_solvent_molecules: int,
    steps: int,
    dt_fs: float,
    write_frequency: int,
    solvent_bond_distance: float,
    trajectory_path: Path,
    energy_log_path: Path,
    diabatic_path: Path,
    validate_forces: bool,
    fd_delta: float,
    occupied_state: int,
    mapping_seed: int,
    occupied_state_choices: Optional[List[int]],
    mapping_init_mode: str,
    mapping_substeps: int,
    h_matrix_log_path: Path,
    mapping_log_path: Path,
    observables_log_path: Path,
    target_temperature_k: float,
    equilibration_ps: float = 1.0,
    edge_taper_width_angstrom: float = 0.02,
    kernel_backend: str = "python",
    wall_radius_angstrom: float = 9.0,
    n_fbts_pairs: int = 8,
    fbts_seed_policy: str = "independent",
    estimator_averaging_cadence: int = 1,
 ) -> None:
    _validate_fbts_kernel_backend(kernel_backend)
    if n_fbts_pairs < 1:
        raise ValueError(f"n_fbts_pairs must be >= 1, got {n_fbts_pairs}.")
    if fbts_seed_policy not in {"independent", "shared"}:
        raise ValueError("fbts_seed_policy must be 'independent' or 'shared'.")
    if estimator_averaging_cadence < 1:
        raise ValueError("estimator_averaging_cadence must be >= 1.")

    diabatic_table = load_diabatic_tables(diabatic_path)
    r_min, r_max = diabatic_r_range(diabatic_table)
    n_states = int(diabatic_table.get("n_states", 3))
    if mapping_init_mode not in {"focused", "global-norm"}:
        raise ValueError("mapping_init_mode must be 'focused' or 'global-norm'.")
    if mapping_substeps < 1:
        raise ValueError(f"mapping_substeps must be >= 1, got {mapping_substeps}.")

    print(f"Diabatic model active range from JSON: R_AB in [{r_min:.6f}, {r_max:.6f}] Angstrom")

    base_rng = random.Random(mapping_seed)
    if mapping_init_mode == "focused":
        if occupied_state_choices is None:
            occupied_state_choices = [occupied_state]
        if len(occupied_state_choices) == 0:
            raise ValueError("occupied_state_choices must contain at least one state index for focused initialization.")
        invalid = [idx for idx in occupied_state_choices if not (0 <= idx < n_states)]
        if invalid:
            raise ValueError(f"occupied_state_choices must be within [0, {n_states - 1}], got {invalid}.")

    fbts_states: List[FBTSElectronicState] = []
    for pair_idx in range(n_fbts_pairs):
        seed_i = mapping_seed if fbts_seed_policy == "shared" else (mapping_seed + pair_idx)
        rng_i = random.Random(seed_i)
        if mapping_init_mode == "focused":
            occ = base_rng.choice(occupied_state_choices)
            sample = initialize_focused_fbts_state(n_states, occ, rng_i)
        else:
            sample = initialize_coherent_fbts_state(n_states, rng_i)
        fbts_states.append(sample.state)

    def _rho_from_state(st: FBTSElectronicState) -> List[List[complex]]:
        rho = [[0j for _ in range(n_states)] for _ in range(n_states)]
        for i in range(n_states):
            cf = complex(st.r_f[i], st.p_f[i])
            for j in range(n_states):
                cb = complex(st.r_b[j], -st.p_b[j])
                rho[i][j] = (cf * cb) / (2.0 * PLANCK_REDUCED_KCAL_MOL_FS)
        return rho

    def _evaluate_fbts_ensemble(step_idx: int) -> Tuple[List[List[float]], Dict[str, object], List[List[complex]]]:
        used_states = fbts_states if (step_idx % estimator_averaging_cadence == 0) else [fbts_states[0]]
        acc_forces = [[0.0, 0.0, 0.0] for _ in sites]
        acc_terms: Optional[Dict[str, object]] = None
        acc_rho = [[0j for _ in range(n_states)] for _ in range(n_states)]

        for st in used_states:
            eval_result = compute_fbts_forces_and_hamiltonian(
                sites=sites,
                n_solvent_molecules=n_solvent_molecules,
                diabatic_table=diabatic_table,
                fbts_state=st,
                kernel_backend=kernel_backend,
                edge_taper_width_angstrom=edge_taper_width_angstrom,
            )
            f = eval_result.forces
            t = eval_result.terms
            for i in range(len(sites)):
                acc_forces[i][0] += f[i][0]
                acc_forces[i][1] += f[i][1]
                acc_forces[i][2] += f[i][2]

            rho = _rho_from_state(st)
            for i in range(n_states):
                for j in range(n_states):
                    acc_rho[i][j] += rho[i][j]

            if acc_terms is None:
                acc_terms = dict(t)
            else:
                for key in ("K", "V_SS", "E_fbts_coupling", "E_map_coupling", "H_fbts", "H_map", "R_AB", "w_RAB_taper", "dE_dRAB"):
                    acc_terms[key] = float(acc_terms[key]) + float(t[key])
                for mat_key in ("h_eff", "v_cs", "estimator_matrix"):
                    for i in range(n_states):
                        for j in range(n_states):
                            acc_terms[mat_key][i][j] += t[mat_key][i][j]
                for g in range(len(acc_terms["omega"])):
                    acc_terms["omega"][g] += t["omega"][g]

        n_used = float(len(used_states))
        for i in range(len(sites)):
            acc_forces[i][0] /= n_used
            acc_forces[i][1] /= n_used
            acc_forces[i][2] /= n_used
        assert acc_terms is not None
        for key in ("K", "V_SS", "E_fbts_coupling", "E_map_coupling", "H_fbts", "H_map", "R_AB", "w_RAB_taper", "dE_dRAB"):
            acc_terms[key] = float(acc_terms[key]) / n_used
        for mat_key in ("h_eff", "v_cs", "estimator_matrix"):
            for i in range(n_states):
                for j in range(n_states):
                    acc_terms[mat_key][i][j] /= n_used
        for g in range(len(acc_terms["omega"])):
            acc_terms["omega"][g] /= n_used

        for i in range(n_states):
            for j in range(n_states):
                acc_rho[i][j] /= n_used

        return acc_forces, acc_terms, acc_rho

    trajectory_path.write_text("", encoding="utf-8")
    energy_log_path.write_text(
        "step time_fs R_AB K_kcal_mol V_SS_kcal_mol E_fbts_coupling_kcal_mol H_fbts_kcal_mol dE_dRAB_kcal_mol_A "
        "fd_count fd_delta_A fd_max_abs_err fd_max_rel_err fd_mean_abs_err w_RAB_taper\n",
        encoding="utf-8",
    )
    h_labels = [f"h{i + 1}{j + 1}" for i in range(n_states) for j in range(n_states)]
    h_matrix_log_path.write_text("step time_fs R_AB " + " ".join(h_labels) + "\n", encoding="utf-8")

    rho_cols = [f"rho{i + 1}{j + 1}_re" for i in range(n_states) for j in range(n_states)] + [f"rho{i + 1}{j + 1}_im" for i in range(n_states) for j in range(n_states)]
    pop_cols = [f"pop{i + 1}" for i in range(n_states)]
    coh_cols = [f"coh{i + 1}{j + 1}_abs" for i in range(n_states) for j in range(i + 1, n_states)]
    mapping_log_path.write_text(
        "step time_fs pair_count avg_weight avg_phase " + " ".join(pop_cols + coh_cols + rho_cols) + "\n",
        encoding="utf-8",
    )
    observables_log_path.write_text("step time_fs dE_pol COM_distance\n", encoding="utf-8")

    def _rescale_nuclear_velocities_to_temperature() -> None:
        current_t = instantaneous_temperature([s for s in sites if s.site_type != "H"], remove_momentum_dof=False)
        if current_t <= 1e-12:
            return
        scale = math.sqrt(target_temperature_k / current_t)
        for site in sites:
            if site.site_type == "H":
                continue
            site.velocity_ang_fs[0] *= scale
            site.velocity_ang_fs[1] *= scale
            site.velocity_ang_fs[2] *= scale

    def _append_logs(step: int, terms: Dict[str, object], rho_avg: List[List[complex]], fd_summary: Dict[str, float], forces: List[List[float]]) -> None:
        with energy_log_path.open("a", encoding="utf-8") as flog:
            flog.write(
                f"{step} {step * dt_fs:.6f} {float(terms['R_AB']):.8f} {float(terms['K']):.10f} {float(terms['V_SS']):.10f} "
                f"{float(terms['E_fbts_coupling']):.10f} {float(terms['H_fbts']):.10f} {float(terms['dE_dRAB']):.10f} "
                f"{fd_summary['fd_count']:.0f} {fd_summary['fd_delta']:.6f} {fd_summary['fd_max_abs_err']:.10e} "
                f"{fd_summary['fd_max_rel_err']:.10e} {fd_summary['fd_mean_abs_err']:.10e} {float(terms['w_RAB_taper']):.6f}\n"
            )

        with h_matrix_log_path.open("a", encoding="utf-8") as fh:
            h_eff = terms["h_eff"]
            h_flat = " ".join(f"{h_eff[i][j]:.10f}" for i in range(n_states) for j in range(n_states))
            fh.write(f"{step} {step * dt_fs:.6f} {float(terms['R_AB']):.8f} {h_flat}\n")

        pops = [rho_avg[i][i].real for i in range(n_states)]
        cohs = [abs(rho_avg[i][j]) for i in range(n_states) for j in range(i + 1, n_states)]
        rho_flat_re = [rho_avg[i][j].real for i in range(n_states) for j in range(n_states)]
        rho_flat_im = [rho_avg[i][j].imag for i in range(n_states) for j in range(n_states)]
        avg_weight = sum(st.weight for st in fbts_states) / len(fbts_states)
        avg_phase = sum(st.phase for st in fbts_states) / len(fbts_states)
        with mapping_log_path.open("a", encoding="utf-8") as fm:
            fields = [
                str(step),
                f"{step * dt_fs:.6f}",
                str(len(fbts_states)),
                f"{avg_weight:.10f}",
                f"{avg_phase:.10f}",
            ]
            fields.extend(f"{v:.10f}" for v in pops)
            fields.extend(f"{v:.10f}" for v in cohs)
            fields.extend(f"{v:.10f}" for v in rho_flat_re)
            fields.extend(f"{v:.10f}" for v in rho_flat_im)
            fm.write(" ".join(fields) + "\n")

        dE_pol, r_com = compute_observables(sites, n_solvent_molecules)
        with observables_log_path.open("a", encoding="utf-8") as fo:
            fo.write(f"{step} {step * dt_fs:.6f} {dE_pol:.10f} {r_com:.10f}\n")

        temperature = instantaneous_temperature([s for s in sites if s.site_type != "H"], remove_momentum_dof=False)
        max_force = max(math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) for f in forces)
        append_xyz_frame(
            trajectory_path,
            sites,
            (
                f"step={step} time_fs={step * dt_fs:.3f} R_AB={float(terms['R_AB']):.6f} "
                f"H_fbts={float(terms['H_fbts']):.6f} K={float(terms['K']):.6f} V_SS={float(terms['V_SS']):.6f} "
                f"E_fbts={float(terms['E_fbts_coupling']):.6f} pair_count={len(fbts_states)} T_noH={temperature:.3f} max|F|={max_force:.6f}"
            ),
        )

    def _advance_one_step(curr_forces: List[List[float]], curr_terms: Dict[str, object], step_idx: int, log_failures: bool) -> Tuple[bool, List[List[float]], Dict[str, object], List[List[complex]]]:
        for idx, site in enumerate(sites):
            if site.site_type == "H":
                continue
            inv_mass = 1.0 / site.mass_amu
            ax = curr_forces[idx][0] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            ay = curr_forces[idx][1] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            az = curr_forces[idx][2] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            site.velocity_ang_fs[0] += 0.5 * dt_fs * ax
            site.velocity_ang_fs[1] += 0.5 * dt_fs * ay
            site.velocity_ang_fs[2] += 0.5 * dt_fs * az

        dt_map_half = 0.5 * dt_fs / mapping_substeps
        for _ in range(mapping_substeps):
            for st in fbts_states:
                propagate_fbts_state_exact_step(st, curr_terms["h_eff"], dt_map_half)

        for site in sites:
            if site.site_type == "H":
                continue
            site.position_angstrom[0] += dt_fs * site.velocity_ang_fs[0]
            site.position_angstrom[1] += dt_fs * site.velocity_ang_fs[1]
            site.position_angstrom[2] += dt_fs * site.velocity_ang_fs[2]

        enforce_solvent_bond_constraints(sites, n_solvent_molecules, solvent_bond_distance)
        apply_spherical_reflecting_wall(sites, wall_radius_angstrom)

        try:
            new_forces, new_terms, rho_new = _evaluate_fbts_ensemble(step_idx + 1)
        except (RuntimeError, ValueError) as exc:
            if log_failures:
                with energy_log_path.open("a", encoding="utf-8") as flog:
                    flog.write(f"# terminated at step {step_idx + 1}: {exc}\n")
            return False, curr_forces, curr_terms, [[0j for _ in range(n_states)] for _ in range(n_states)]

        for _ in range(mapping_substeps):
            for st in fbts_states:
                propagate_fbts_state_exact_step(st, new_terms["h_eff"], dt_map_half)

        for idx, site in enumerate(sites):
            if site.site_type == "H":
                continue
            inv_mass = 1.0 / site.mass_amu
            ax = new_forces[idx][0] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            ay = new_forces[idx][1] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            az = new_forces[idx][2] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            site.velocity_ang_fs[0] += 0.5 * dt_fs * ax
            site.velocity_ang_fs[1] += 0.5 * dt_fs * ay
            site.velocity_ang_fs[2] += 0.5 * dt_fs * az

        enforce_solvent_velocity_constraints(sites, n_solvent_molecules)
        if (step_idx + 1) % 5000 == 0:
            remove_net_linear_momentum(sites)

        return True, new_forces, new_terms, rho_new

    try:
        forces, terms, rho_avg = _evaluate_fbts_ensemble(0)
    except (RuntimeError, ValueError) as exc:
        with energy_log_path.open("a", encoding="utf-8") as flog:
            flog.write(f"# terminated at step 0: {exc}\n")
        return

    equil_steps = max(0, int(round((equilibration_ps * 1000.0) / dt_fs)))
    rescale_interval_steps = max(1, int(round(100.0 / dt_fs)))
    if equil_steps > 0:
        print(
            f"Equilibration phase: {equilibration_ps:.3f} ps ({equil_steps} steps) with velocity rescaling every 100 fs ({rescale_interval_steps} steps)."
        )
        for eq_step in range(equil_steps):
            ok, forces, terms, rho_avg = _advance_one_step(forces, terms, eq_step, log_failures=False)
            if not ok:
                print(f"Warning: equilibration terminated early at step {eq_step + 1} due to force evaluation failure.")
                break
            if (eq_step + 1) % rescale_interval_steps == 0:
                _rescale_nuclear_velocities_to_temperature()

    fd_summary = {"fd_count": 0.0, "fd_delta": fd_delta, "fd_max_abs_err": 0.0, "fd_max_rel_err": 0.0, "fd_mean_abs_err": 0.0}
    if validate_forces:
        with energy_log_path.open("a", encoding="utf-8") as flog:
            flog.write("# force finite-difference validation is currently disabled for FBTS ensemble mode\n")

    for step in range(steps + 1):
        if step % write_frequency == 0:
            _append_logs(step, terms, rho_avg, fd_summary, forces)

        if step == steps:
            break

        ok, forces, terms, rho_avg = _advance_one_step(forces, terms, step, log_failures=True)
        if not ok:
            break

        fd_summary = {"fd_count": 0.0, "fd_delta": fd_delta, "fd_max_abs_err": 0.0, "fd_max_rel_err": 0.0, "fd_mean_abs_err": 0.0}
