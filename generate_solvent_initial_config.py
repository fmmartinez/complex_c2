#!/usr/bin/env python3
"""Generate an initial AHB+solvent configuration and run FBTS NVE dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from fbts.diabatic import load_diabatic_tables
from fbts.dynamics import generate_configuration, run_nve_md
from fbts.io_utils import write_initial_xyz
from fbts.model import instantaneous_temperature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FBTS NVE dynamics for coarse-grained chloromethane + AHB complex."
    )
    parser.add_argument("--n-molecules", type=int, default=7, help="Number of solvent molecules.")
    parser.add_argument("--temperature", type=float, default=50.0, help="Target temperature (K).")
    parser.add_argument("--radius", type=float, default=7.0, help="Initial placement sphere radius (Angstrom).")
    parser.add_argument(
        "--wall-offset",
        type=float,
        default=2.0,
        help="Spherical wall offset added to --radius (Angstrom).",
    )
    parser.add_argument("--bond-distance", type=float, default=1.7, help="Solvent C1-C2 distance (Angstrom).")
    parser.add_argument("--min-distance", type=float, default=3.0, help="Minimum inter-site distance (Angstrom).")
    parser.add_argument("--seed", type=int, default=20260218, help="Nuclear configuration RNG seed.")

    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument(
        "--equilibration-ps",
        type=float,
        default=1.0,
        help="Equilibration duration in ps before production NVE.",
    )
    parser.add_argument(
        "--edge-taper-width-angstrom",
        type=float,
        default=0.02,
        help="Edge taper width (Angstrom) for diabatic derivative smoothing near R_AB bounds.",
    )
    parser.add_argument("--write-frequency", type=int, default=10)

    parser.add_argument("--initial-output", type=Path, default=Path("fbts_initial.xyz"))
    parser.add_argument("--trajectory", type=Path, default=Path("fbts_trajectory.xyz"))
    parser.add_argument("--energy-log", type=Path, default=Path("fbts_energy.log"))
    parser.add_argument("--h-matrix-log", type=Path, default=Path("fbts_effective_hamiltonian.log"))
    parser.add_argument("--electronic-log", type=Path, default=Path("fbts_electronic.log"))
    parser.add_argument("--observables-log", type=Path, default=Path("fbts_observables.log"))
    parser.add_argument("--diabatic-json", type=Path, default=Path("diabatic_matrices.json"))

    parser.add_argument("--validate-forces", action="store_true", help="Run finite-difference force spot checks (limited in ensemble mode).")
    parser.add_argument("--fd-delta", type=float, default=1e-4, help="Finite-difference displacement in Angstrom.")

    parser.add_argument(
        "--electronic-preparation",
        choices=("focused", "coherent"),
        default="focused",
        help="FBTS electronic initialization mode.",
    )
    parser.add_argument(
        "--occupied-state",
        type=int,
        default=1,
        help="Initially occupied state index (1-based) for focused preparation.",
    )
    parser.add_argument(
        "--occupied-state-set",
        type=str,
        default=None,
        help="Comma-separated 1-based focused states to sample from per trajectory pair (e.g., '1,3').",
    )
    parser.add_argument(
        "--electronic-seed",
        type=int,
        default=None,
        help="Electronic RNG seed (defaults to --seed).",
    )
    parser.add_argument(
        "--electronic-substeps",
        type=int,
        default=1,
        help="Number of frozen-h_eff electronic substeps per half-kick.",
    )

    parser.add_argument(
        "--n-fbts-pairs",
        type=int,
        default=8,
        help="Number of forward-backward trajectory pairs used for FBTS force/observable averaging.",
    )
    parser.add_argument(
        "--fbts-seed-policy",
        choices=("independent", "shared"),
        default="independent",
        help="Electronic RNG policy across FBTS pairs.",
    )
    parser.add_argument(
        "--estimator-averaging-cadence",
        type=int,
        default=1,
        help="Use all FBTS pairs every N steps (other steps evaluate estimator with the first pair).",
    )
    parser.add_argument(
        "--kernel-backend",
        choices=("python", "numba"),
        default="python",
        help="Force/Hamiltonian backend (FBTS currently supports python backend).",
    )
    return parser.parse_args()


def _parse_occupied_state_set(spec: str) -> List[int]:
    values: List[int] = []
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("--occupied-state-set must contain at least one 1-based state index.")
    return values


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
            "initial frame chloromethane+AHB FBTS "
            f"molecules={args.n_molecules} T_target={args.temperature:.2f}K "
            f"T_inst={initial_temp:.2f}K seed={args.seed}"
        ),
    )

    diabatic_table = load_diabatic_tables(args.diabatic_json)
    n_states = int(diabatic_table.get("n_states", 3))

    occupied_state_set_zero_based = None
    if args.electronic_preparation == "focused":
        if args.occupied_state_set is None:
            if not 1 <= args.occupied_state <= n_states:
                raise ValueError(f"--occupied-state must be in [1, {n_states}] for focused preparation.")
            occupied_state_set_zero_based = [args.occupied_state - 1]
        else:
            selected_1based = _parse_occupied_state_set(args.occupied_state_set)
            invalid = [v for v in selected_1based if not 1 <= v <= n_states]
            if invalid:
                raise ValueError(f"--occupied-state-set indices must be in [1, {n_states}], got {invalid}.")
            occupied_state_set_zero_based = [v - 1 for v in selected_1based]

    electronic_seed = args.seed if args.electronic_seed is None else args.electronic_seed
    wall_radius = args.radius + args.wall_offset

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
        occupied_state_choices=occupied_state_set_zero_based,
        mapping_seed=electronic_seed,
        mapping_init_mode="focused" if args.electronic_preparation == "focused" else "global-norm",
        mapping_substeps=args.electronic_substeps,
        h_matrix_log_path=args.h_matrix_log,
        mapping_log_path=args.electronic_log,
        observables_log_path=args.observables_log,
        target_temperature_k=args.temperature,
        equilibration_ps=args.equilibration_ps,
        edge_taper_width_angstrom=args.edge_taper_width_angstrom,
        kernel_backend=args.kernel_backend,
        wall_radius_angstrom=wall_radius,
        n_fbts_pairs=args.n_fbts_pairs,
        fbts_seed_policy=args.fbts_seed_policy,
        estimator_averaging_cadence=args.estimator_averaging_cadence,
    )

    print(f"Initial temperature: {initial_temp:.3f} K")
    print(f"FBTS dynamics run complete (steps={args.steps}, dt_fs={args.dt_fs}).")
    print(f"Equilibration stage: {args.equilibration_ps:.3f} ps (velocity rescaling every 100 fs), then production NVE.")
    print(f"R_AB edge taper width: {args.edge_taper_width_angstrom:.4f} Angstrom")
    print(f"Electronic substeps per half-kick: {args.electronic_substeps}")
    if args.validate_forces:
        print(f"Finite-difference force checks requested (delta={args.fd_delta:.2e} A).")

    if args.electronic_preparation == "focused":
        if args.occupied_state_set is None:
            print(
                f"Electronic preparation: focused state {args.occupied_state} (electronic_seed={electronic_seed})."
            )
        else:
            print(
                "Electronic preparation: focused random choice from "
                f"{{{args.occupied_state_set}}} (electronic_seed={electronic_seed})."
            )
    else:
        print(f"Electronic preparation: coherent ensemble sampling (electronic_seed={electronic_seed}).")

    print(f"Initial frame written to: {args.initial_output}")
    print(f"Trajectory written to: {args.trajectory}")
    print(f"Energy log written to: {args.energy_log}")
    print(f"Effective Hamiltonian log written to: {args.h_matrix_log}")
    print(f"Electronic log written to: {args.electronic_log}")
    print(f"Observables log written to: {args.observables_log}")
    print(f"Kernel backend: {args.kernel_backend}")
    print(f"Spherical wall radius: {wall_radius:.3f} A")
    print("COM momentum removal frequency: 5000")
    print(f"FBTS pairs per run: {args.n_fbts_pairs}")
    print(f"FBTS seed policy: {args.fbts_seed_policy}")
    print(f"Estimator averaging cadence: every {args.estimator_averaging_cadence} step(s)")


if __name__ == "__main__":
    main()
