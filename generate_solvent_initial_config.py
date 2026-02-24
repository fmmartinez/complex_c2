#!/usr/bin/env python3
"""Generate and run PBME MD for coarse-grained chloromethane + AHB complex in vacuum."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from pbme.diabatic import load_diabatic_tables
from pbme.dynamics import generate_configuration, run_nve_md
from pbme.io_utils import write_initial_xyz
from pbme.model import instantaneous_temperature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run NVE MD for coarse-grained chloromethane + AHB complex."
    )
    parser.add_argument("--n-molecules", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=50.0, help="Kelvin")
    parser.add_argument("--radius", type=float, default=7.0, help="Angstrom")
    parser.add_argument(
        "--wall-offset",
        type=float,
        default=2.0,
        help="Spherical wall offset added to --radius (Angstrom)",
    )
    parser.add_argument("--bond-distance", type=float, default=1.7, help="Angstrom (solvent C1-C2)")
    parser.add_argument("--min-distance", type=float, default=3.0, help="Angstrom")
    parser.add_argument("--seed", type=int, default=20260218)

    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--equilibration-ps", type=float, default=1.0, help="Equilibration duration in ps before production NVE.")
    parser.add_argument("--edge-taper-width-angstrom", type=float, default=0.02, help="Edge taper width (Angstrom) for diabatic derivative smoothing near R_AB bounds.")
    parser.add_argument("--write-frequency", type=int, default=10)

    parser.add_argument("--initial-output", type=Path, default=Path("pbme_initial.xyz"))
    parser.add_argument("--trajectory", type=Path, default=Path("pbme_trajectory.xyz"))
    parser.add_argument("--energy-log", type=Path, default=Path("pbme_energy.log"))
    parser.add_argument("--h-matrix-log", type=Path, default=Path("pbme_effective_hamiltonian.log"))
    parser.add_argument("--mapping-log", type=Path, default=Path("pbme_mapping.log"))
    parser.add_argument("--observables-log", type=Path, default=Path("pbme_observables.log"))
    parser.add_argument("--diabatic-json", type=Path, default=Path("diabatic_matrices.json"))
    parser.add_argument("--validate-forces", action="store_true", help="Run finite-difference force spot checks")
    parser.add_argument("--fd-delta", type=float, default=1e-4, help="Finite-difference displacement in Angstrom")
    parser.add_argument(
        "--occupied-state",
        type=int,
        default=1,
        help="Initially occupied mapping state index (1-based). Used when --occupied-state-set is not provided.",
    )
    parser.add_argument(
        "--occupied-state-set",
        type=str,
        default=None,
        help="Comma-separated 1-based focused states to sample from randomly per trajectory (e.g., '1,3').",
    )
    parser.add_argument(
        "--mapping-seed",
        type=int,
        default=None,
        help="Seed for mapping-variable sampling (defaults to --seed)",
    )
    parser.add_argument(
        "--mapping-init-mode",
        choices=("focused", "global-norm"),
        default="focused",
        help="Mapping initialization mode: focused per-state radii or global-norm Gaussian rescaling.",
    )
    parser.add_argument(
        "--kernel-backend",
        choices=("python", "numba"),
        default="python",
        help="Force/Hamiltonian backend (numba requires numba installed)",
    )
    return parser.parse_args()




def _parse_occupied_state_set(spec: str) -> List[int]:
    values: List[int] = []
    for chunk in spec.split(','):
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
            "initial frame chloromethane+AHB PBME-mapping-check "
            f"molecules={args.n_molecules} T_target={args.temperature:.2f}K "
            f"T_inst={initial_temp:.2f}K seed={args.seed}"
        ),
    )

    diabatic_table = load_diabatic_tables(args.diabatic_json)
    n_states = int(diabatic_table.get("n_states", 3))
    occupied_state_set_zero_based = None
    if args.mapping_init_mode == "focused":
        if args.occupied_state_set is None:
            if not 1 <= args.occupied_state <= n_states:
                raise ValueError(f"--occupied-state must be in [1, {n_states}] for focused initialization.")
            occupied_state_set_zero_based = [args.occupied_state - 1]
        else:
            selected_1based = _parse_occupied_state_set(args.occupied_state_set)
            invalid = [v for v in selected_1based if not 1 <= v <= n_states]
            if invalid:
                raise ValueError(f"--occupied-state-set indices must be in [1, {n_states}], got {invalid}.")
            occupied_state_set_zero_based = [v - 1 for v in selected_1based]
    mapping_seed = args.seed if args.mapping_seed is None else args.mapping_seed
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
        mapping_seed=mapping_seed,
        mapping_init_mode=args.mapping_init_mode,
        h_matrix_log_path=args.h_matrix_log,
        mapping_log_path=args.mapping_log,
        observables_log_path=args.observables_log,
        target_temperature_k=args.temperature,
        equilibration_ps=args.equilibration_ps,
        edge_taper_width_angstrom=args.edge_taper_width_angstrom,
        kernel_backend=args.kernel_backend,
        wall_radius_angstrom=wall_radius,
    )

    print(f"Initial temperature: {initial_temp:.3f} K")
    print(f"PBME dynamics run complete (steps={args.steps}, dt_fs={args.dt_fs}).")
    print(f"Equilibration stage: {args.equilibration_ps:.3f} ps (velocity rescaling every 100 fs), then production NVE.")
    print(f"R_AB edge taper width: {args.edge_taper_width_angstrom:.4f} Angstrom")
    if args.validate_forces:
        print(f"Finite-difference force checks enabled (delta={args.fd_delta:.2e} A).")
    if args.mapping_init_mode == "focused":
        if args.occupied_state_set is None:
            print(
                f"Mapping variables sampled from N(0,1/2) with focused rescaling; "
                f"occupied state={args.occupied_state}, mapping_seed={mapping_seed}."
            )
        else:
            print(
                f"Mapping variables sampled from N(0,1/2) with focused rescaling; "
                f"random occupied state from {{{args.occupied_state_set}}}, mapping_seed={mapping_seed}."
            )
    else:
        print(
            "Mapping variables sampled from N(0,1/2) with global-norm rescaling; "
            f"target sum=(N+2)*hbar, mapping_seed={mapping_seed}."
        )
    print(f"Initial frame written to: {args.initial_output}")
    print(f"Trajectory written to: {args.trajectory}")
    print(f"Energy log written to: {args.energy_log}")
    print(f"Effective Hamiltonian log written to: {args.h_matrix_log}")
    print(f"Mapping variables log written to: {args.mapping_log}")
    print(f"Observables log written to: {args.observables_log}")
    print(f"Kernel backend: {args.kernel_backend}")
    print(f"Spherical wall radius: {wall_radius:.3f} A")
    print("COM momentum removal frequency: 5000")


if __name__ == "__main__":
    main()
