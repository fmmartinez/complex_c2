from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
from .mapping import propagate_mapping_exact_half_step, sample_focused_mapping_variables
from .model import (
    CHARGE,
    COULOMB_KCAL_MOL_ANG_E2,
    HBAR_MAPPING,
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

    s_map = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            s_map[i, j] = map_r[i] * map_r[j] + map_p[i] * map_p[j] - (HBAR_MAPPING if i == j else 0.0)

    omega = np.zeros(grid.shape[0], dtype=np.float64)
    for g in range(grid.shape[0]):
        tmp = 0.0
        for i in range(3):
            for j in range(3):
                tmp += s_map[i, j] * eigenstates[i, g] * eigenstates[j, g]
        omega[g] = tmp / (2.0 * HBAR_MAPPING)

    v_cs = np.zeros((3, 3), dtype=np.float64)
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
            for i in range(3):
                for j in range(3):
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

    h_eff = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            h_eff[i, j] = h_diab[i, j] + v_cs[i, j]
    e_h = 0.0
    dE_dR = 0.0
    for i in range(3):
        for j in range(3):
            e_h += h_eff[i, j] * s_map[i, j]
            dE_dR += dh_diab[i, j] * s_map[i, j]
    e_h /= 2.0 * HBAR_MAPPING
    dE_dR /= 2.0 * HBAR_MAPPING

    for k in range(3):
        f_b = -dE_dR * uab[k]
        forces[idx_b, k] += f_b
        forces[idx_a, k] -= f_b

    h_map = k_total + v_ss + e_h
    return forces, k_total, v_ss, e_h, h_map, dE_dR, h_eff


def compute_pbme_forces_and_hamiltonian(
    sites: List[Site],
    n_solvent_molecules: int,
    diabatic_table: Dict[str, object],
    map_r: List[float],
    map_p: List[float],
    kernel_backend: str = "python",
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
            dh_diab=np.asarray(dh_diab, dtype=np.float64),
            map_r=np.asarray(map_r, dtype=np.float64),
            map_p=np.asarray(map_p, dtype=np.float64),
        )
        return forces.tolist(), {
            "K": k_total,
            "V_SS": v_ss,
            "E_map_coupling": e_h,
            "H_map": h_map,
            "R_AB": r_ab,
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

    s_map = [[map_r[i] * map_r[j] + map_p[i] * map_p[j] - (HBAR_MAPPING if i == j else 0.0) for j in range(3)] for i in range(3)]

    omega = [0.0 for _ in range(ngrid)]
    for g in range(ngrid):
        tmp = 0.0
        for i in range(3):
            for j in range(3):
                tmp += s_map[i][j] * eigenstates[i][g] * eigenstates[j][g]
        omega[g] = tmp / (2.0 * HBAR_MAPPING)

    v_cs = [[0.0] * 3 for _ in range(3)]
    for g, r_ah in enumerate(grid):
        fpol, _ = polarization_switch(r_ah)
        q_a = (1.0 - fpol) * POLARIZATION["Q_A_cov"] + fpol * POLARIZATION["Q_A_ion"]
        q_h = (1.0 - fpol) * POLARIZATION["Q_H_cov"] + fpol * POLARIZATION["Q_H_ion"]
        q_b = (1.0 - fpol) * POLARIZATION["Q_B_cov"] + fpol * POLARIZATION["Q_B_ion"]

        rh = [ra[k] + uab[k] * r_ah for k in range(3)]
        projector = [[eigenstates[i][g] * eigenstates[j][g] for j in range(3)] for i in range(3)]

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
            for i in range(3):
                for j in range(3):
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

    h_eff = [[h_diab[i][j] + v_cs[i][j] for j in range(3)] for i in range(3)]

    e_h = 0.0
    dE_dR = 0.0
    for i in range(3):
        for j in range(3):
            e_h += h_eff[i][j] * s_map[i][j]
            dE_dR += dh_diab[i][j] * s_map[i][j]
    e_h /= (2.0 * HBAR_MAPPING)
    dE_dR /= (2.0 * HBAR_MAPPING)

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
        "dE_dRAB": dE_dR,
        "h_eff": h_eff,
    }


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
) -> Dict[str, float]:
    forces, terms = compute_pbme_forces_and_hamiltonian(sites, n_solvent_molecules, diabatic_table, map_r, map_p, kernel_backend=kernel_backend)

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

            _, terms_plus = compute_pbme_forces_and_hamiltonian(
                plus_sites, n_solvent_molecules, diabatic_table, map_r, map_p, kernel_backend=kernel_backend
            )
            _, terms_minus = compute_pbme_forces_and_hamiltonian(
                minus_sites, n_solvent_molecules, diabatic_table, map_r, map_p, kernel_backend=kernel_backend
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
    h_matrix_log_path: Path,
    mapping_log_path: Path,
    observables_log_path: Path,
    kernel_backend: str = "python",
    wall_radius_angstrom: float = 9.0,
 ) -> None:
    if kernel_backend not in {"python", "numba"}:
        raise ValueError("kernel_backend must be 'python' or 'numba'.")
    if kernel_backend == "numba" and not NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not installed.")
    if kernel_backend == "numba" and np is None:
        raise RuntimeError("Numba backend requested but numpy is not installed.")
    diabatic_table = load_diabatic_tables(diabatic_path)
    r_min, r_max = diabatic_r_range(diabatic_table)
    print(f"Diabatic model active range from JSON: R_AB in [{r_min:.6f}, {r_max:.6f}] Angstrom")
    mapping_rng = random.Random(mapping_seed)
    map_r, map_p = sample_focused_mapping_variables(3, occupied_state, mapping_rng)

    trajectory_path.write_text("", encoding="utf-8")
    energy_log_path.write_text(
        "step time_fs R_AB K_kcal_mol V_SS_kcal_mol E_map_coupling_kcal_mol H_map_kcal_mol dE_dRAB_kcal_mol_A "
        "fd_count fd_delta_A fd_max_abs_err fd_max_rel_err fd_mean_abs_err\n",
        encoding="utf-8",
    )
    h_matrix_log_path.write_text(
        "step time_fs R_AB h11 h12 h13 h21 h22 h23 h31 h32 h33\n",
        encoding="utf-8",
    )
    mapping_log_path.write_text(
        "step time_fs M1 M2 M3 R1 P1 R2 P2 R3 P3\n",
        encoding="utf-8",
    )
    observables_log_path.write_text(
        "step time_fs dE_pol COM_distance\n",
        encoding="utf-8",
    )

    def append_logs(step: int, terms: Dict[str, object], fd_summary: Dict[str, float], forces: List[List[float]]) -> None:
        m_norms = [map_r[i] * map_r[i] + map_p[i] * map_p[i] for i in range(3)]
        with energy_log_path.open("a", encoding="utf-8") as flog:
            flog.write(
                f"{step} {step * dt_fs:.6f} {float(terms['R_AB']):.8f} {float(terms['K']):.10f} {float(terms['V_SS']):.10f} "
                f"{float(terms['E_map_coupling']):.10f} {float(terms['H_map']):.10f} {float(terms['dE_dRAB']):.10f} "
                f"{fd_summary['fd_count']:.0f} {fd_summary['fd_delta']:.6f} {fd_summary['fd_max_abs_err']:.10e} "
                f"{fd_summary['fd_max_rel_err']:.10e} {fd_summary['fd_mean_abs_err']:.10e}\n"
            )

        with mapping_log_path.open("a", encoding="utf-8") as fm:
            fm.write(
                f"{step} {step * dt_fs:.6f} {m_norms[0]:.10f} {m_norms[1]:.10f} {m_norms[2]:.10f} "
                f"{map_r[0]:.10f} {map_p[0]:.10f} {map_r[1]:.10f} {map_p[1]:.10f} {map_r[2]:.10f} {map_p[2]:.10f}\n"
            )

        with h_matrix_log_path.open("a", encoding="utf-8") as fh:
            h_eff = terms["h_eff"]
            fh.write(
                f"{step} {step * dt_fs:.6f} {float(terms['R_AB']):.8f} "
                f"{h_eff[0][0]:.10f} {h_eff[0][1]:.10f} {h_eff[0][2]:.10f} "
                f"{h_eff[1][0]:.10f} {h_eff[1][1]:.10f} {h_eff[1][2]:.10f} "
                f"{h_eff[2][0]:.10f} {h_eff[2][1]:.10f} {h_eff[2][2]:.10f}\n"
            )

        dE_pol, r_com = compute_observables(sites, n_solvent_molecules)
        with observables_log_path.open("a", encoding="utf-8") as fo:
            fo.write(f"{step} {step * dt_fs:.6f} {dE_pol:.10f} {r_com:.10f}\n")

        temperature = instantaneous_temperature([s for s in sites if s.site_type != "H"], remove_momentum_dof=False)
        max_force = max(math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) for f in forces)
        append_xyz_frame(
            trajectory_path,
            sites,
            (
                f"step={step} time_fs={step * dt_fs:.3f} "
                f"R_AB={float(terms['R_AB']):.6f} H_map={float(terms['H_map']):.6f} "
                f"K={float(terms['K']):.6f} V_SS={float(terms['V_SS']):.6f} E_map={float(terms['E_map_coupling']):.6f} "
                f"M=({m_norms[0]:.3f},{m_norms[1]:.3f},{m_norms[2]:.3f}) T_noH={temperature:.3f} max|F|={max_force:.6f} "
                f"occ={occupied_state + 1} FDmaxAbs={fd_summary['fd_max_abs_err']:.3e}"
            ),
        )

    try:
        forces, terms = compute_pbme_forces_and_hamiltonian(
            sites, n_solvent_molecules, diabatic_table, map_r, map_p, kernel_backend=kernel_backend
        )
    except (RuntimeError, ValueError) as exc:
        with energy_log_path.open("a", encoding="utf-8") as flog:
            flog.write(f"# terminated at step 0: {exc}\n")
        return

    fd_summary = {"fd_count": 0.0, "fd_delta": fd_delta, "fd_max_abs_err": 0.0, "fd_max_rel_err": 0.0, "fd_mean_abs_err": 0.0}
    if validate_forces:
        fd_summary = finite_difference_force_spot_checks(
            sites, n_solvent_molecules, diabatic_table, map_r, map_p, fd_delta, kernel_backend=kernel_backend
        )

    for step in range(steps + 1):
        if step % write_frequency == 0:
            append_logs(step, terms, fd_summary, forces)

        if step == steps:
            break

        for idx, site in enumerate(sites):
            if site.site_type == "H":
                continue
            inv_mass = 1.0 / site.mass_amu
            ax = forces[idx][0] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            ay = forces[idx][1] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            az = forces[idx][2] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            site.velocity_ang_fs[0] += 0.5 * dt_fs * ax
            site.velocity_ang_fs[1] += 0.5 * dt_fs * ay
            site.velocity_ang_fs[2] += 0.5 * dt_fs * az

        propagate_mapping_exact_half_step(map_r, map_p, terms["h_eff"], 0.5 * dt_fs)

        for site in sites:
            if site.site_type == "H":
                continue
            site.position_angstrom[0] += dt_fs * site.velocity_ang_fs[0]
            site.position_angstrom[1] += dt_fs * site.velocity_ang_fs[1]
            site.position_angstrom[2] += dt_fs * site.velocity_ang_fs[2]

        enforce_solvent_bond_constraints(sites, n_solvent_molecules, solvent_bond_distance)
        apply_spherical_reflecting_wall(sites, wall_radius_angstrom)

        try:
            new_forces, new_terms = compute_pbme_forces_and_hamiltonian(
                sites, n_solvent_molecules, diabatic_table, map_r, map_p, kernel_backend=kernel_backend
            )
        except (RuntimeError, ValueError) as exc:
            with energy_log_path.open("a", encoding="utf-8") as flog:
                flog.write(f"# terminated at step {step + 1}: {exc}\n")
            break

        propagate_mapping_exact_half_step(map_r, map_p, new_terms["h_eff"], 0.5 * dt_fs)

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

        if (step + 1) % 5000 == 0:
            remove_net_linear_momentum(sites)

        terms = new_terms
        forces = new_forces
        fd_summary = {"fd_count": 0.0, "fd_delta": fd_delta, "fd_max_abs_err": 0.0, "fd_max_rel_err": 0.0, "fd_mean_abs_err": 0.0}
