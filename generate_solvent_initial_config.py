#!/usr/bin/env python3
"""Generate an initial coarse-grained chloromethane solvent configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import List, Tuple

ANGSTROM_TO_M = 1e-10
AMU_TO_KG = 1.66053906660e-27
KB = 1.380649e-23
AVOGADRO = 6.02214076e23
J_TO_KCAL = 1.0 / 4184.0
COULOMB_KCAL_MOL_ANG_E2 = 332.063713299

SIGMA = {"C1": 3.774, "C2": 3.481}
EPSILON = {"C1": 0.238, "C2": 0.415}
CHARGE = {"C1": +0.25, "C2": -0.25}


@dataclass
class Site:
    molecule_id: int
    site_type: str
    label: str
    mass_amu: float
    position_angstrom: Tuple[float, float, float]
    velocity_m_s: Tuple[float, float, float]


def random_unit_vector(rng: random.Random) -> Tuple[float, float, float]:
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        norm = math.sqrt(x * x + y * y + z * z)
        if norm > 1e-12:
            return (x / norm, y / norm, z / norm)


def random_point_in_sphere(rng: random.Random, radius: float) -> Tuple[float, float, float]:
    direction = random_unit_vector(rng)
    u = rng.random()
    r = radius * (u ** (1.0 / 3.0))
    return (r * direction[0], r * direction[1], r * direction[2])


def dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def sample_velocity(rng: random.Random, temperature_k: float, mass_amu: float) -> Tuple[float, float, float]:
    sigma = math.sqrt(KB * temperature_k / (mass_amu * AMU_TO_KG))
    return (
        rng.gauss(0.0, sigma),
        rng.gauss(0.0, sigma),
        rng.gauss(0.0, sigma),
    )


def remove_net_linear_momentum(sites: List[Site]) -> None:
    total_mass = 0.0
    p_x = p_y = p_z = 0.0
    for site in sites:
        mass_kg = site.mass_amu * AMU_TO_KG
        vx, vy, vz = site.velocity_m_s
        total_mass += mass_kg
        p_x += mass_kg * vx
        p_y += mass_kg * vy
        p_z += mass_kg * vz

    vcm = (p_x / total_mass, p_y / total_mass, p_z / total_mass)

    for site in sites:
        vx, vy, vz = site.velocity_m_s
        site.velocity_m_s = (vx - vcm[0], vy - vcm[1], vz - vcm[2])


def instantaneous_temperature(sites: List[Site], remove_momentum_dof: bool = True) -> float:
    kinetic_energy = 0.0
    for site in sites:
        mass_kg = site.mass_amu * AMU_TO_KG
        vx, vy, vz = site.velocity_m_s
        kinetic_energy += 0.5 * mass_kg * (vx * vx + vy * vy + vz * vz)

    dof = 3 * len(sites)
    if remove_momentum_dof:
        dof -= 3
    return (2.0 * kinetic_energy) / (dof * KB)


def kinetic_energy_kcal_mol(sites: List[Site]) -> float:
    kinetic_energy_joule = 0.0
    for site in sites:
        mass_kg = site.mass_amu * AMU_TO_KG
        vx, vy, vz = site.velocity_m_s
        kinetic_energy_joule += 0.5 * mass_kg * (vx * vx + vy * vy + vz * vz)
    return kinetic_energy_joule * AVOGADRO * J_TO_KCAL


def potential_energy_kcal_mol(sites: List[Site]) -> float:
    potential = 0.0
    for i in range(len(sites)):
        site_i = sites[i]
        xi, yi, zi = site_i.position_angstrom
        for j in range(i + 1, len(sites)):
            site_j = sites[j]

            # No intramolecular interactions (same solvent molecule)
            if site_i.molecule_id == site_j.molecule_id:
                continue

            xj, yj, zj = site_j.position_angstrom
            rij = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)

            sigma_ij = 0.5 * (SIGMA[site_i.site_type] + SIGMA[site_j.site_type])
            # Implemented exactly as requested in prompt.
            epsilon_ij = math.sqrt(SIGMA[site_i.site_type] * SIGMA[site_j.site_type])

            sr = sigma_ij / rij
            sr6 = sr**6
            sr12 = sr6**2
            lj = 4.0 * epsilon_ij * (sr12 - sr6)

            qi = CHARGE[site_i.site_type]
            qj = CHARGE[site_j.site_type]
            coulomb = COULOMB_KCAL_MOL_ANG_E2 * (qi * qj) / rij

            potential += lj + coulomb

    return potential


def total_energy_kcal_mol(sites: List[Site]) -> Tuple[float, float, float]:
    kinetic = kinetic_energy_kcal_mol(sites)
    potential = potential_energy_kcal_mol(sites)
    return kinetic, potential, kinetic + potential


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

    c1_mass = 15.0
    c2_mass = 35.5

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
                        label="C",
                        mass_amu=c1_mass,
                        position_angstrom=c1_pos,
                        velocity_m_s=sample_velocity(rng, temperature_k, c1_mass),
                    )
                )
                sites.append(
                    Site(
                        molecule_id=mol_idx,
                        site_type="C2",
                        label="Cl",
                        mass_amu=c2_mass,
                        position_angstrom=c2_pos,
                        velocity_m_s=sample_velocity(rng, temperature_k, c2_mass),
                    )
                )
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place molecule {mol_idx + 1} without overlaps after many attempts."
            )

    remove_net_linear_momentum(sites)
    return sites


def write_xyz(path: Path, sites: List[Site], comment: str) -> None:
    lines = [str(len(sites)), comment]
    for site in sites:
        x, y, z = site.position_angstrom
        lines.append(f"{site.label:2s} {x: .8f} {y: .8f} {z: .8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate initial coarse-grained chloromethane solvent configuration."
    )
    parser.add_argument("--n-molecules", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=50.0, help="Kelvin")
    parser.add_argument("--radius", type=float, default=7.0, help="Angstrom")
    parser.add_argument("--bond-distance", type=float, default=1.7, help="Angstrom")
    parser.add_argument("--min-distance", type=float, default=3.0, help="Angstrom")
    parser.add_argument("--seed", type=int, default=20260218)
    parser.add_argument("--output", type=Path, default=Path("solvent_initial.xyz"))
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

    temp = instantaneous_temperature(sites)
    kinetic, potential, total = total_energy_kcal_mol(sites)
    comment = (
        f"chloromethane coarse-grained initial frame | molecules={args.n_molecules} "
        f"T_target={args.temperature:.2f}K T_inst={temp:.2f}K seed={args.seed} "
        f"E_tot={total:.3f}kcal/mol"
    )

    write_xyz(args.output, sites, comment)

    print(f"Wrote {len(sites)} sites ({args.n_molecules} molecules) to {args.output}")
    print(f"Instantaneous temperature after momentum removal: {temp:.3f} K")
    print(f"Kinetic energy: {kinetic:.6f} kcal/mol")
    print(f"Potential energy: {potential:.6f} kcal/mol")
    print(f"Total energy: {total:.6f} kcal/mol")


if __name__ == "__main__":
    main()
