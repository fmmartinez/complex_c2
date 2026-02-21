from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Tuple

AMU_TO_KG = 1.66053906660e-27
KB = 1.380649e-23
COULOMB_KCAL_MOL_ANG_E2 = 332.063713299

M_S_TO_ANG_FS = 1e-5
AMU_ANG2_FS2_TO_KCAL_MOL = 2390.05736055072
KCAL_MOL_ANG_TO_AMU_ANG_FS2 = 1.0 / AMU_ANG2_FS2_TO_KCAL_MOL

MASS = {"C1": 15.0, "C2": 35.5, "A": 93.0, "B": 59.0, "H": 1.0}
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

SIGMA_SOLVENT = {"C1": 3.774, "C2": 3.481}
EPSILON_SOLVENT = {"C1": 0.238, "C2": 0.415}

LJ_PARAMS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("A", "C1"): (3.5, 0.3974),
    ("A", "C2"): (3.5, 0.3974),
    ("B", "C1"): (3.5, 0.3974),
    ("B", "C2"): (3.5, 0.3974),
    ("C1", "H"): (3.5, 0.3974),
    ("C2", "H"): (3.5, 0.3974),
}

HBAR_MAPPING = 0.01594  # (kcal/mol)*fs
PROTON_GRID_MIN = 0.3
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


def kinetic_energy_kcal_mol(sites: List[Site], exclude_h: bool = False) -> float:
    kinetic_internal = 0.0
    for site in sites:
        if exclude_h and site.site_type == "H":
            continue
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


def polarization_switch(r_ah: float) -> Tuple[float, float]:
    dr = r_ah - POLARIZATION["r0"]
    l = POLARIZATION["l"]
    denom = math.sqrt(dr * dr + l * l)
    f = 0.5 * (1.0 + dr / denom)
    df_dr = 0.5 * (l * l) / (denom ** 3)
    return f, df_dr


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
