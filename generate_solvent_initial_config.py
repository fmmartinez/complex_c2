#!/usr/bin/env python3
"""Generate and run NVE MD for coarse-grained chloromethane solvent clusters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import List, Tuple

AMU_TO_KG = 1.66053906660e-27
KB = 1.380649e-23
AVOGADRO = 6.02214076e23
J_TO_KCAL = 1.0 / 4184.0
COULOMB_KCAL_MOL_ANG_E2 = 332.063713299

# Conversion factors for internal units (angstrom, fs, amu, kcal/mol)
M_S_TO_ANG_FS = 1e-5
AMU_ANG2_FS2_TO_KCAL_MOL = 2390.05736055072
KCAL_MOL_ANG_TO_AMU_ANG_FS2 = 1.0 / AMU_ANG2_FS2_TO_KCAL_MOL

SIGMA = {"C1": 3.774, "C2": 3.481}
EPSILON = {"C1": 0.238, "C2": 0.415}
CHARGE = {"C1": +0.25, "C2": -0.25}
MASS = {"C1": 15.0, "C2": 35.5}
LABEL = {"C1": "C", "C2": "Cl"}


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


def compute_forces_and_potential(sites: List[Site]) -> Tuple[List[List[float]], float]:
    forces = [[0.0, 0.0, 0.0] for _ in sites]
    potential = 0.0

    for i in range(len(sites)):
        si = sites[i]
        xi, yi, zi = si.position_angstrom
        for j in range(i + 1, len(sites)):
            sj = sites[j]
            if si.molecule_id == sj.molecule_id:
                continue

            dx = xi - sj.position_angstrom[0]
            dy = yi - sj.position_angstrom[1]
            dz = zi - sj.position_angstrom[2]
            r2 = dx * dx + dy * dy + dz * dz
            r = math.sqrt(r2)

            sigma_ij = 0.5 * (SIGMA[si.site_type] + SIGMA[sj.site_type])
            epsilon_ij = math.sqrt(EPSILON[si.site_type] * EPSILON[sj.site_type])
            qi = CHARGE[si.site_type]
            qj = CHARGE[sj.site_type]

            inv_r = 1.0 / r
            sr = sigma_ij * inv_r
            sr2 = sr * sr
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6

            lj = 4.0 * epsilon_ij * (sr12 - sr6)
            coul = COULOMB_KCAL_MOL_ANG_E2 * qi * qj * inv_r
            potential += lj + coul

            dUdr_lj = 4.0 * epsilon_ij * (-12.0 * sr12 * inv_r + 6.0 * sr6 * inv_r)
            dUdr_coul = -COULOMB_KCAL_MOL_ANG_E2 * qi * qj * inv_r * inv_r
            dUdr = dUdr_lj + dUdr_coul

            # Force on i: F_i = -dU/dr * r_hat, r_hat = (r_i-r_j)/r
            scale = -dUdr * inv_r
            fx = scale * dx
            fy = scale * dy
            fz = scale * dz

            forces[i][0] += fx
            forces[i][1] += fy
            forces[i][2] += fz
            forces[j][0] -= fx
            forces[j][1] -= fy
            forces[j][2] -= fz

    return forces, potential


def enforce_bond_constraints(sites: List[Site], bond_distance: float) -> None:
    n_mol = len(sites) // 2
    for m in range(n_mol):
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


def enforce_velocity_constraints(sites: List[Site]) -> None:
    n_mol = len(sites) // 2
    for m in range(n_mol):
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
                f"Could not place molecule {mol_idx + 1} without overlaps after many attempts."
            )

    remove_net_linear_momentum(sites)
    enforce_bond_constraints(sites, c1_c2_distance_angstrom)
    enforce_velocity_constraints(sites)
    return sites


def append_xyz_frame(path: Path, sites: List[Site], comment: str) -> None:
    lines = [str(len(sites)), comment]
    for site in sites:
        x, y, z = site.position_angstrom
        lines.append(f"{site.label:2s} {x: .8f} {y: .8f} {z: .8f}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_nve_md(
    sites: List[Site],
    steps: int,
    dt_fs: float,
    write_frequency: int,
    bond_distance: float,
    trajectory_path: Path,
    energy_log_path: Path,
) -> None:
    trajectory_path.write_text("", encoding="utf-8")
    energy_log_path.write_text("step time_fs KE_kcal_mol PE_kcal_mol TE_kcal_mol T_K\n", encoding="utf-8")

    forces, potential = compute_forces_and_potential(sites)

    for step in range(steps + 1):
        kinetic = kinetic_energy_kcal_mol(sites)
        temperature = instantaneous_temperature(sites)
        total = kinetic + potential

        with energy_log_path.open("a", encoding="utf-8") as flog:
            flog.write(
                f"{step} {step * dt_fs:.6f} {kinetic:.10f} {potential:.10f} {total:.10f} {temperature:.6f}\n"
            )

        if step % write_frequency == 0:
            append_xyz_frame(
                trajectory_path,
                sites,
                (
                    f"step={step} time_fs={step * dt_fs:.3f} "
                    f"KE={kinetic:.6f} PE={potential:.6f} TE={total:.6f} T={temperature:.3f}"
                ),
            )

        if step == steps:
            break

        # velocity-Verlet: half-kick
        for idx, site in enumerate(sites):
            inv_mass = 1.0 / site.mass_amu
            ax = forces[idx][0] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            ay = forces[idx][1] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            az = forces[idx][2] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            site.velocity_ang_fs[0] += 0.5 * dt_fs * ax
            site.velocity_ang_fs[1] += 0.5 * dt_fs * ay
            site.velocity_ang_fs[2] += 0.5 * dt_fs * az

        # drift
        for site in sites:
            site.position_angstrom[0] += dt_fs * site.velocity_ang_fs[0]
            site.position_angstrom[1] += dt_fs * site.velocity_ang_fs[1]
            site.position_angstrom[2] += dt_fs * site.velocity_ang_fs[2]

        # constraints after drift
        enforce_bond_constraints(sites, bond_distance)

        # new forces
        new_forces, potential = compute_forces_and_potential(sites)

        # second half-kick
        for idx, site in enumerate(sites):
            inv_mass = 1.0 / site.mass_amu
            ax = new_forces[idx][0] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            ay = new_forces[idx][1] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            az = new_forces[idx][2] * KCAL_MOL_ANG_TO_AMU_ANG_FS2 * inv_mass
            site.velocity_ang_fs[0] += 0.5 * dt_fs * ax
            site.velocity_ang_fs[1] += 0.5 * dt_fs * ay
            site.velocity_ang_fs[2] += 0.5 * dt_fs * az

        # velocity constraints after second half-kick
        enforce_velocity_constraints(sites)

        forces = new_forces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run NVE MD for coarse-grained chloromethane solvent."
    )
    parser.add_argument("--n-molecules", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=50.0, help="Kelvin")
    parser.add_argument("--radius", type=float, default=7.0, help="Angstrom")
    parser.add_argument("--bond-distance", type=float, default=1.7, help="Angstrom")
    parser.add_argument("--min-distance", type=float, default=3.0, help="Angstrom")
    parser.add_argument("--seed", type=int, default=20260218)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt-fs", type=float, default=1.0)
    parser.add_argument("--write-frequency", type=int, default=100)

    parser.add_argument("--trajectory", type=Path, default=Path("solvent_nve.xyz"))
    parser.add_argument("--energy-log", type=Path, default=Path("solvent_energy.log"))
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

    initial_ke = kinetic_energy_kcal_mol(sites)
    _, initial_pe = compute_forces_and_potential(sites)
    initial_temp = instantaneous_temperature(sites)

    run_nve_md(
        sites=sites,
        steps=args.steps,
        dt_fs=args.dt_fs,
        write_frequency=args.write_frequency,
        bond_distance=args.bond_distance,
        trajectory_path=args.trajectory,
        energy_log_path=args.energy_log,
    )

    final_ke = kinetic_energy_kcal_mol(sites)
    _, final_pe = compute_forces_and_potential(sites)
    final_temp = instantaneous_temperature(sites)

    print(f"Initial KE: {initial_ke:.6f} kcal/mol")
    print(f"Initial PE: {initial_pe:.6f} kcal/mol")
    print(f"Initial TE: {initial_ke + initial_pe:.6f} kcal/mol")
    print(f"Initial temperature: {initial_temp:.3f} K")
    print(f"Final KE: {final_ke:.6f} kcal/mol")
    print(f"Final PE: {final_pe:.6f} kcal/mol")
    print(f"Final TE: {final_ke + final_pe:.6f} kcal/mol")
    print(f"Final temperature: {final_temp:.3f} K")
    print(f"Trajectory written to: {args.trajectory}")
    print(f"Energy log written to: {args.energy_log}")


if __name__ == "__main__":
    main()
