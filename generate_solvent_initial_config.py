#!/usr/bin/env python3
"""Generate and run NVE MD for coarse-grained chloromethane + AHB complex in vacuum."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

AMU_TO_KG = 1.66053906660e-27
KB = 1.380649e-23
COULOMB_KCAL_MOL_ANG_E2 = 332.063713299

# Conversion factors for internal units (angstrom, fs, amu, kcal/mol)
M_S_TO_ANG_FS = 1e-5
AMU_ANG2_FS2_TO_KCAL_MOL = 2390.05736055072
KCAL_MOL_ANG_TO_AMU_ANG_FS2 = 1.0 / AMU_ANG2_FS2_TO_KCAL_MOL

MASS = {
    "C1": 15.0,
    "C2": 35.5,
    "A": 93.0,
    "B": 59.0,
    "H": 1.0,
}
LABEL = {"C1": "C", "C2": "Cl", "A": "O", "B": "N", "H": "H"}
CHARGE = {"C1": +0.25, "C2": -0.25, "A": -0.5, "B": 0.0, "H": +0.5}

POLARIZATION = {
    "Q_A_cov": -0.5,
    "Q_H_cov": +0.5,
    "Q_B_cov": 0.0,
    "Q_A_ion": -1.0,
    "Q_H_ion": +0.5,
    "Q_B_ion": +0.5,
    "r0": 1.43,
    "l": 0.125,
}

# Solvent LJ parameters
SIGMA_SOLVENT = {"C1": 3.774, "C2": 3.481}
EPSILON_SOLVENT = {"C1": 0.238, "C2": 0.415}

# Explicit LJ pairs for complex interactions (kcal/mol, angstrom)
LJ_PARAMS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("A", "C1"): (3.5, 0.3974),
    ("A", "C2"): (3.5, 0.3974),
    ("B", "C1"): (3.5, 0.3974),
    ("B", "C2"): (3.5, 0.3974),
    ("C1", "H"): (3.5, 0.3974),
    ("C2", "H"): (3.5, 0.3974),
}

AHB_PARAMS = {
    "a": 11.2,
    "b": 7.1e13,
    "c": 0.776,
    "d_A": 0.95,
    "d_B": 0.97,
    "D_A": 110.0,
    "n_A": 9.26,
    "n_B": 11.42,
}

HBAR_MAPPING = 1.0
RAB_MIN = 2.65
RAB_MAX = 2.75
PROTON_GRID_MIN = 0.3
PROTON_GRID_MAX = 2.4
PROTON_GRID_STEP = 0.02


@dataclass
class Site:
    molecule_id: int
    site_type: str
    label: str
    mass_amu: float
    position_angstrom: List[float]
    velocity_ang_fs: List[float]


def dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: List[float]) -> float:
    return math.sqrt(dot(a, a))


def random_unit_vector(rng: random.Random) -> Tuple[float, float, float]:
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        mag = math.sqrt(x * x + y * y + z * z)
        if mag > 1e-12:
            return (x / mag, y / mag, z / mag)


def random_point_in_sphere(rng: random.Random, radius: float) -> Tuple[float, float, float]:
    ux, uy, uz = random_unit_vector(rng)
    r = radius * (rng.random() ** (1.0 / 3.0))
    return (r * ux, r * uy, r * uz)


def dist(a: Tuple[float, float, float], b: List[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def sample_velocity(rng: random.Random, temperature_k: float, mass_amu: float) -> List[float]:
    sigma_m_s = math.sqrt(KB * temperature_k / (mass_amu * AMU_TO_KG))
    return [
        rng.gauss(0.0, sigma_m_s) * M_S_TO_ANG_FS,
        rng.gauss(0.0, sigma_m_s) * M_S_TO_ANG_FS,
        rng.gauss(0.0, sigma_m_s) * M_S_TO_ANG_FS,
    ]


def remove_net_linear_momentum(sites: List[Site]) -> None:
    total_mass = 0.0
    px = py = pz = 0.0
    for site in sites:
        vx, vy, vz = site.velocity_ang_fs
        total_mass += site.mass_amu
        px += site.mass_amu * vx
        py += site.mass_amu * vy
        pz += site.mass_amu * vz

    vcm = [px / total_mass, py / total_mass, pz / total_mass]
    for site in sites:
        site.velocity_ang_fs[0] -= vcm[0]
        site.velocity_ang_fs[1] -= vcm[1]
        site.velocity_ang_fs[2] -= vcm[2]


def kinetic_energy_kcal_mol(sites: List[Site]) -> float:
    kinetic_internal = 0.0
    for site in sites:
        vx, vy, vz = site.velocity_ang_fs
        kinetic_internal += 0.5 * site.mass_amu * (vx * vx + vy * vy + vz * vz)
    return kinetic_internal * AMU_ANG2_FS2_TO_KCAL_MOL


def instantaneous_temperature(sites: List[Site], remove_momentum_dof: bool = True) -> float:
    kinetic_energy_joule = 0.0
    for site in sites:
        vx, vy, vz = site.velocity_ang_fs
        v2_m_s2 = (vx / M_S_TO_ANG_FS) ** 2 + (vy / M_S_TO_ANG_FS) ** 2 + (vz / M_S_TO_ANG_FS) ** 2
        kinetic_energy_joule += 0.5 * (site.mass_amu * AMU_TO_KG) * v2_m_s2
    dof = 3 * len(sites)
    if remove_momentum_dof:
        dof -= 3
    return (2.0 * kinetic_energy_joule) / (dof * KB)


def get_lj_params(site_type_i: str, site_type_j: str) -> Optional[Tuple[float, float]]:
    a, b = sorted((site_type_i, site_type_j))
    if a in {"C1", "C2"} and b in {"C1", "C2"}:
        sigma_ij = 0.5 * (SIGMA_SOLVENT[a] + SIGMA_SOLVENT[b])
        epsilon_ij = math.sqrt(EPSILON_SOLVENT[a] * EPSILON_SOLVENT[b])
        return sigma_ij, epsilon_ij
    return LJ_PARAMS.get((a, b))


def coulomb_allowed(site_type_i: str, site_type_j: str) -> bool:
    pair = {site_type_i, site_type_j}
    if pair == {"A", "H"} or pair == {"B", "H"} or pair == {"A", "B"}:
        return False
    if pair == {"H"}:
        return False
    return True


def polarization_switch(r_ah: float) -> Tuple[float, float]:
    dr = r_ah - POLARIZATION["r0"]
    l = POLARIZATION["l"]
    denom = math.sqrt(dr * dr + l * l)
    f = 0.5 * (1.0 + dr / denom)
    # derivative of f wrt r_ah
    df_dr = 0.5 * (l * l) / (denom ** 3)
    return f, df_dr


def compute_complex_polarized_charges(sites: List[Site]) -> Tuple[float, float, float, float, float]:
    idx_a = idx_h = None
    for idx, site in enumerate(sites):
        if site.site_type == "A":
            idx_a = idx
        elif site.site_type == "H":
            idx_h = idx

    if idx_a is None or idx_h is None:
        return CHARGE["A"], CHARGE["H"], CHARGE["B"], 0.0, 0.0

    ra = sites[idx_a].position_angstrom
    rh = sites[idx_h].position_angstrom
    r_ah = math.sqrt((rh[0]-ra[0])**2 + (rh[1]-ra[1])**2 + (rh[2]-ra[2])**2)

    f, _ = polarization_switch(r_ah)
    q_a = (1.0 - f) * POLARIZATION["Q_A_cov"] + f * POLARIZATION["Q_A_ion"]
    q_h = (1.0 - f) * POLARIZATION["Q_H_cov"] + f * POLARIZATION["Q_H_ion"]
    q_b = (1.0 - f) * POLARIZATION["Q_B_cov"] + f * POLARIZATION["Q_B_ion"]
    return q_a, q_h, q_b, f, r_ah


def compute_ahb_potential_and_derivatives(r_ah: float, r_ab: float) -> Tuple[float, float, float]:
    eps = 1e-12
    r = max(r_ah, eps)
    rab = max(r_ab, eps)
    s = max(rab - r, eps)

    a = AHB_PARAMS["a"]
    b = AHB_PARAMS["b"]
    c = AHB_PARAMS["c"]
    d_a = AHB_PARAMS["d_A"]
    d_b = AHB_PARAMS["d_B"]
    dcap = AHB_PARAMS["D_A"]
    n_a = AHB_PARAMS["n_A"]
    n_b = AHB_PARAMS["n_B"]

    term1 = b * math.exp(-a * rab)

    x = -n_a * ((r - d_a) ** 2) / (2.0 * r)
    term2 = dcap * (1.0 - math.exp(x))

    y = -n_b * ((s - d_b) ** 2) / (2.0 * s)
    term3 = c * dcap * (1.0 - math.exp(y))

    potential = term1 + term2 + term3

    dterm1_drab = -a * b * math.exp(-a * rab)

    dx_dr = -n_a * (r * r - d_a * d_a) / (2.0 * r * r)
    dterm2_dr = -dcap * math.exp(x) * dx_dr

    dy_ds = -n_b * (s * s - d_b * d_b) / (2.0 * s * s)
    dterm3_ds = -c * dcap * math.exp(y) * dy_ds

    dterm3_drab = dterm3_ds
    dterm3_dr = -dterm3_ds

    dV_drab = dterm1_drab + dterm3_drab
    dV_dr = dterm2_dr + dterm3_dr

    return potential, dV_dr, dV_drab


def lj_energy_dudr(r: float, sigma: float, epsilon: float) -> Tuple[float, float]:
    inv_r = 1.0 / max(r, 1e-12)
    sr = sigma * inv_r
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    energy = 4.0 * epsilon * (sr12 - sr6)
    dUdr = 4.0 * epsilon * (-12.0 * sr12 * inv_r + 6.0 * sr6 * inv_r)
    return energy, dUdr


def coulomb_energy_dudr(r: float, qi: float, qj: float) -> Tuple[float, float]:
    inv_r = 1.0 / max(r, 1e-12)
    energy = COULOMB_KCAL_MOL_ANG_E2 * qi * qj * inv_r
    dUdr = -COULOMB_KCAL_MOL_ANG_E2 * qi * qj * inv_r * inv_r
    return energy, dUdr


class PchipInterpolator:
    def __init__(self, x: List[float], y: List[float]):
        if len(x) < 2 or len(x) != len(y):
            raise ValueError("Invalid interpolation table.")
        self.x = x
        self.y = y
        self.h = [x[i + 1] - x[i] for i in range(len(x) - 1)]
        self.delta = [(y[i + 1] - y[i]) / self.h[i] for i in range(len(self.h))]
        self.m = self._slopes()

    def _edge_slope(self, h0: float, h1: float, d0: float, d1: float) -> float:
        m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if m * d0 <= 0.0:
            return 0.0
        if d0 * d1 < 0.0 and abs(m) > abs(3.0 * d0):
            return 3.0 * d0
        return m

    def _slopes(self) -> List[float]:
        n = len(self.x)
        if n == 2:
            return [self.delta[0], self.delta[0]]
        m = [0.0] * n
        m[0] = self._edge_slope(self.h[0], self.h[1], self.delta[0], self.delta[1])
        m[-1] = self._edge_slope(self.h[-1], self.h[-2], self.delta[-1], self.delta[-2])
        for i in range(1, n - 1):
            if self.delta[i - 1] * self.delta[i] <= 0.0:
                m[i] = 0.0
            else:
                w1 = 2.0 * self.h[i] + self.h[i - 1]
                w2 = self.h[i] + 2.0 * self.h[i - 1]
                m[i] = (w1 + w2) / (w1 / self.delta[i - 1] + w2 / self.delta[i])
        return m

    def _interval(self, xq: float) -> int:
        if xq < self.x[0] or xq > self.x[-1]:
            raise ValueError(f"R_AB={xq:.6f} outside supported [{self.x[0]:.6f}, {self.x[-1]:.6f}].")
        for i in range(len(self.x) - 1):
            if self.x[i] <= xq <= self.x[i + 1]:
                return i
        return len(self.x) - 2

    def eval(self, xq: float) -> float:
        i = self._interval(xq)
        h = self.h[i]
        t = (xq - self.x[i]) / h
        y0 = self.y[i]
        y1 = self.y[i + 1]
        m0 = self.m[i]
        m1 = self.m[i + 1]
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1
        h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2
        h11 = t ** 3 - t ** 2
        return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

    def deriv(self, xq: float) -> float:
        i = self._interval(xq)
        h = self.h[i]
        t = (xq - self.x[i]) / h
        y0 = self.y[i]
        y1 = self.y[i + 1]
        m0 = self.m[i]
        m1 = self.m[i + 1]
        dh00 = (6 * t ** 2 - 6 * t) / h
        dh10 = 3 * t ** 2 - 4 * t + 1
        dh01 = (-6 * t ** 2 + 6 * t) / h
        dh11 = 3 * t ** 2 - 2 * t
        return dh00 * y0 + dh10 * m0 + dh01 * y1 + dh11 * m1


def load_diabatic_tables(path: Path) -> Dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    results = obj["results"]
    r_values = [float(item["R"]) for item in results]

    h_splines: List[List[PchipInterpolator]] = [[None for _ in range(3)] for _ in range(3)]  # type: ignore
    for i in range(3):
        for j in range(3):
            y = [float(item["hamiltonian_reduced_diabatic"][i][j]) for item in results]
            h_splines[i][j] = PchipInterpolator(r_values, y)

    eigenstates = [[[float(v) for v in state] for state in item["r_diagonalized_eigenstates_grid"]] for item in results]

    grid = [PROTON_GRID_MIN + PROTON_GRID_STEP * k for k in range(106)]
    return {"r_values": r_values, "h_splines": h_splines, "eigenstates": eigenstates, "grid": grid}


def interpolate_h_diabatic(di_table: Dict[str, object], r_ab: float) -> Tuple[List[List[float]], List[List[float]]]:
    h_splines = di_table["h_splines"]
    h = [[0.0] * 3 for _ in range(3)]
    dh = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s_ij = h_splines[i][j]
            h[i][j] = s_ij.eval(r_ab)
            dh[i][j] = s_ij.deriv(r_ab)
    for i in range(3):
        for j in range(i + 1, 3):
            sym = 0.5 * (h[i][j] + h[j][i])
            dsym = 0.5 * (dh[i][j] + dh[j][i])
            h[i][j] = h[j][i] = sym
            dh[i][j] = dh[j][i] = dsym
    return h, dh


def interpolate_eigenstates(di_table: Dict[str, object], r_ab: float) -> List[List[float]]:
    r_values = di_table["r_values"]
    states_all = di_table["eigenstates"]
    if r_ab < r_values[0] or r_ab > r_values[-1]:
        raise ValueError(f"R_AB={r_ab:.6f} outside supported [{r_values[0]:.6f}, {r_values[-1]:.6f}].")

    if abs(r_ab - r_values[-1]) < 1e-12:
        return [row[:] for row in states_all[-1]]

    for i in range(len(r_values) - 1):
        if r_values[i] <= r_ab <= r_values[i + 1]:
            t = (r_ab - r_values[i]) / (r_values[i + 1] - r_values[i])
            out = []
            for state_idx in range(3):
                st = []
                for g in range(len(states_all[i][state_idx])):
                    st.append((1.0 - t) * states_all[i][state_idx][g] + t * states_all[i + 1][state_idx][g])
                out.append(st)
            return out
    return [row[:] for row in states_all[0]]


def compute_pbme_forces_and_hamiltonian(
    sites: List[Site], n_solvent_molecules: int, diabatic_table: Dict[str, object], map_r: List[float], map_p: List[float]
) -> Tuple[List[List[float]], Dict[str, float]]:
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
    if r_ab < RAB_MIN or r_ab > RAB_MAX:
        raise RuntimeError(f"R_AB={r_ab:.6f} outside model validity window [{RAB_MIN:.2f}, {RAB_MAX:.2f}].")
    uab = [dab[k] / max(r_ab, 1e-12) for k in range(3)]

    h_diab, dh_diab = interpolate_h_diabatic(diabatic_table, r_ab)
    eigenstates = interpolate_eigenstates(diabatic_table, r_ab)
    grid = diabatic_table["grid"]

    # V_SS: solvent-solvent + classical LJ between A/B and solvent only
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

            # Only classical interactions not coupled via bra-ket
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
                # solvent-solvent Coulomb only (A/B solvent Coulomb treated in bra-ket)
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

            # H-solvent LJ/Coulomb
            dx = rh[0] - rs[0]
            dy = rh[1] - rs[1]
            dz = rh[2] - rs[2]
            r_hs = math.sqrt(dx * dx + dy * dy + dz * dz)
            e_hs = 0.0
            dUdr_hs = 0.0
            lj_hs = get_lj_params("H", ss.site_type)
            if lj_hs is not None:
                e_lj, d_lj = lj_energy_dudr(r_hs, lj_hs[0], lj_hs[1])
                e_hs += e_lj
                dUdr_hs += d_lj
            if abs(q_h * qs) > 0.0:
                e_c, d_c = coulomb_energy_dudr(r_hs, q_h, qs)
                e_hs += e_c
                dUdr_hs += d_c

            # A-solvent Coulomb (charge depends on r)
            dxa = ra[0] - rs[0]
            dya = ra[1] - rs[1]
            dza = ra[2] - rs[2]
            r_as = math.sqrt(dxa * dxa + dya * dya + dza * dza)
            e_as, dUdr_as = coulomb_energy_dudr(r_as, q_a, qs)

            # B-solvent Coulomb (charge depends on r)
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

            # force from H-solvent interaction with embedded proton
            if abs(e_hs) > 0.0:
                scale_h = -dUdr_hs / max(r_hs, 1e-12)
                f_h = [scale_h * dx, scale_h * dy, scale_h * dz]
                # solvent receives opposite pair force
                forces[s_idx][0] -= wg * f_h[0]
                forces[s_idx][1] -= wg * f_h[1]
                forces[s_idx][2] -= wg * f_h[2]

                proj = [[(1.0 if a == b else 0.0) - (r_ah / max(r_ab, 1e-12)) * ((1.0 if a == b else 0.0) - uab[a] * uab[b]) for b in range(3)] for a in range(3)]
                for a in range(3):
                    forces[idx_a][a] += wg * sum(proj[b][a] * f_h[b] for b in range(3))
                proj_b = [[(r_ah / max(r_ab, 1e-12)) * ((1.0 if a == b else 0.0) - uab[a] * uab[b]) for b in range(3)] for a in range(3)]
                for a in range(3):
                    forces[idx_b][a] += wg * sum(proj_b[b][a] * f_h[b] for b in range(3))

            # direct A/B Coulomb forces
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

    k_total = 0.0
    for site in sites:
        if site.site_type == "H":
            continue
        vx, vy, vz = site.velocity_ang_fs
        k_total += 0.5 * site.mass_amu * (vx * vx + vy * vy + vz * vz)
    k_total *= AMU_ANG2_FS2_TO_KCAL_MOL

    e_h = 0.0
    dE_dR = 0.0
    for i in range(3):
        for j in range(3):
            hij_tot = h_diab[i][j] + v_cs[i][j]
            e_h += hij_tot * s_map[i][j]
            dE_dR += dh_diab[i][j] * s_map[i][j]
    e_h /= (2.0 * HBAR_MAPPING)
    dE_dR /= (2.0 * HBAR_MAPPING)

    # Contribution from dH_diab/dR onto A/B
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
    }


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
) -> Dict[str, float]:
    forces, terms = compute_pbme_forces_and_hamiltonian(sites, n_solvent_molecules, diabatic_table, map_r, map_p)

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
                plus_sites, n_solvent_molecules, diabatic_table, map_r, map_p
            )
            _, terms_minus = compute_pbme_forces_and_hamiltonian(
                minus_sites, n_solvent_molecules, diabatic_table, map_r, map_p
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


def append_xyz_frame(path: Path, sites: List[Site], comment: str) -> None:
    lines = [str(len(sites)), comment]
    for site in sites:
        x, y, z = site.position_angstrom
        lines.append(f"{site.label:2s} {x: .8f} {y: .8f} {z: .8f}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_initial_xyz(path: Path, sites: List[Site], comment: str) -> None:
    lines = [str(len(sites)), comment]
    for site in sites:
        x, y, z = site.position_angstrom
        lines.append(f"{site.label:2s} {x: .8f} {y: .8f} {z: .8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
) -> None:
    del steps, dt_fs, write_frequency, solvent_bond_distance
    diabatic_table = load_diabatic_tables(diabatic_path)
    mapping_rng = random.Random(mapping_seed)
    map_r, map_p = sample_focused_mapping_variables(
        n_states=3,
        occupied_state=occupied_state,
        rng=mapping_rng,
    )

    trajectory_path.write_text("", encoding="utf-8")
    energy_log_path.write_text(
        "step time_fs R_AB K_kcal_mol V_SS_kcal_mol E_map_coupling_kcal_mol H_map_kcal_mol dE_dRAB_kcal_mol_A "
        "R1 P1 R2 P2 R3 P3 fd_count fd_delta_A fd_max_abs_err fd_max_rel_err fd_mean_abs_err\n",
        encoding="utf-8",
    )

    forces, terms = compute_pbme_forces_and_hamiltonian(sites, n_solvent_molecules, diabatic_table, map_r, map_p)
    temperature = instantaneous_temperature([s for s in sites if s.site_type != "H"], remove_momentum_dof=False)

    fd_summary = {
        "fd_count": 0.0,
        "fd_delta": fd_delta,
        "fd_max_abs_err": 0.0,
        "fd_max_rel_err": 0.0,
        "fd_mean_abs_err": 0.0,
    }
    if validate_forces:
        fd_summary = finite_difference_force_spot_checks(
            sites=sites,
            n_solvent_molecules=n_solvent_molecules,
            diabatic_table=diabatic_table,
            map_r=map_r,
            map_p=map_p,
            delta=fd_delta,
        )

    with energy_log_path.open("a", encoding="utf-8") as flog:
        flog.write(
            f"0 0.000000 {terms['R_AB']:.8f} {terms['K']:.10f} {terms['V_SS']:.10f} "
            f"{terms['E_map_coupling']:.10f} {terms['H_map']:.10f} {terms['dE_dRAB']:.10f} "
            f"{map_r[0]:.10f} {map_p[0]:.10f} {map_r[1]:.10f} {map_p[1]:.10f} {map_r[2]:.10f} {map_p[2]:.10f} "
            f"{fd_summary['fd_count']:.0f} {fd_summary['fd_delta']:.6f} {fd_summary['fd_max_abs_err']:.10e} "
            f"{fd_summary['fd_max_rel_err']:.10e} {fd_summary['fd_mean_abs_err']:.10e}\n"
        )

    max_force = max(math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) for f in forces)
    append_xyz_frame(
        trajectory_path,
        sites,
        (
            "step=0 time_fs=0.000 "
            f"R_AB={terms['R_AB']:.6f} H_map={terms['H_map']:.6f} "
            f"K={terms['K']:.6f} V_SS={terms['V_SS']:.6f} E_map={terms['E_map_coupling']:.6f} "
            f"T_noH={temperature:.3f} max|F|={max_force:.6f} "
            f"occ={occupied_state + 1} FDmaxAbs={fd_summary['fd_max_abs_err']:.3e}"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run NVE MD for coarse-grained chloromethane + AHB complex."
    )
    parser.add_argument("--n-molecules", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=50.0, help="Kelvin")
    parser.add_argument("--radius", type=float, default=7.0, help="Angstrom")
    parser.add_argument("--bond-distance", type=float, default=1.7, help="Angstrom (solvent C1-C2)")
    parser.add_argument("--min-distance", type=float, default=3.0, help="Angstrom")
    parser.add_argument("--seed", type=int, default=20260218)

    parser.add_argument("--steps", type=int, default=0, help="Dynamics disabled; kept for interface compatibility")
    parser.add_argument("--dt-fs", type=float, default=1.0)
    parser.add_argument("--write-frequency", type=int, default=1)

    parser.add_argument("--initial-output", type=Path, default=Path("solvent_initial.xyz"))
    parser.add_argument("--trajectory", type=Path, default=Path("solvent_nve.xyz"))
    parser.add_argument("--energy-log", type=Path, default=Path("solvent_energy.log"))
    parser.add_argument("--diabatic-json", type=Path, default=Path("diabatic_matrices.json"))
    parser.add_argument("--validate-forces", action="store_true", help="Run finite-difference force spot checks")
    parser.add_argument("--fd-delta", type=float, default=1e-4, help="Finite-difference displacement in Angstrom")
    parser.add_argument(
        "--occupied-state",
        type=int,
        default=1,
        help="Initially occupied mapping state index (1-based)",
    )
    parser.add_argument(
        "--mapping-seed",
        type=int,
        default=None,
        help="Seed for mapping-variable sampling (defaults to --seed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sites = generate_configuration(
        n_molecules=args.n_molecules,
        temperature_k=args.temperature,
        radius_angstrom=args.radius,
        c1_c2_distance_angstrom=args.bond_distance,
        min_inter_site_distance_angstrom=args.min_distance,
        seed=args.seed,
    )

    initial_temp = instantaneous_temperature(sites)

    write_initial_xyz(
        args.initial_output,
        sites,
        (
            "initial frame chloromethane+AHB PBME-mapping-check "
            f"molecules={args.n_molecules} T_target={args.temperature:.2f}K "
            f"T_inst={initial_temp:.2f}K seed={args.seed}"
        ),
    )

    if not 1 <= args.occupied_state <= 3:
        raise ValueError("--occupied-state must be 1, 2, or 3.")
    mapping_seed = args.seed if args.mapping_seed is None else args.mapping_seed

    run_nve_md(
        sites=sites,
        n_solvent_molecules=args.n_molecules,
        steps=args.steps,
        dt_fs=args.dt_fs,
        write_frequency=args.write_frequency,
        solvent_bond_distance=args.bond_distance,
        trajectory_path=args.trajectory,
        energy_log_path=args.energy_log,
        diabatic_path=args.diabatic_json,
        validate_forces=args.validate_forces,
        fd_delta=args.fd_delta,
        occupied_state=args.occupied_state - 1,
        mapping_seed=mapping_seed,
    )

    print(f"Initial temperature: {initial_temp:.3f} K")
    print("Dynamics is disabled in this PBME verification mode (step=0 only).")
    if args.validate_forces:
        print(f"Finite-difference force checks enabled (delta={args.fd_delta:.2e} A).")
    print(
        f"Mapping variables sampled from N(0,1/2) with focused rescaling; "
        f"occupied state={args.occupied_state}, mapping_seed={mapping_seed}."
    )
    print(f"Initial frame written to: {args.initial_output}")
    print(f"Trajectory written to: {args.trajectory}")
    print(f"Energy log written to: {args.energy_log}")


if __name__ == "__main__":
    main()
