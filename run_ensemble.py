#!/usr/bin/env python3
"""Run multiple FBTS trajectories in parallel via subprocess CLI invocations."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


def detect_available_cpus() -> int:
    """Detect scheduler-assigned CPU count with fallbacks."""
    env_candidates = [
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_CPUS_PER_NODE",
        "PBS_NP",
        "NSLOTS",
        "LSB_DJOB_NUMPROC",
    ]
    for name in env_candidates:
        raw = os.environ.get(name)
        if not raw:
            continue
        # Handle values like "32(x2)" or "4,4"
        token = raw.split("(")[0].split(",")[0].strip()
        if token.isdigit() and int(token) > 0:
            return int(token)

    cpu_count = os.cpu_count()
    return cpu_count if cpu_count and cpu_count > 0 else 1


@dataclass
class TrajectoryResult:
    trajectory_id: int
    seed: int
    electronic_seed: int
    output_dir: str
    status: str
    returncode: int
    runtime_s: float
    stderr_tail: str


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ensemble of independent FBTS trajectories in parallel. "
            "Unknown args are forwarded to the FBTS driver script."
        )
    )
    parser.add_argument("--n-trajectories", type=int, required=True, help="Total number of trajectories to run")
    parser.add_argument("--output-root", type=Path, default=Path("ensemble_runs"), help="Root directory for per-trajectory outputs")
    parser.add_argument("--max-workers", type=int, default=None, help="Max concurrent trajectories (defaults to detected CPUs)")
    parser.add_argument("--base-seed", type=int, default=20260218, help="Base random seed; each trajectory uses base_seed + traj_id")
    parser.add_argument(
        "--base-electronic-seed",
        type=int,
        default=None,
        help="Base electronic seed; each trajectory uses base_electronic_seed + traj_id (default: base_seed)",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to launch each trajectory subprocess",
    )
    parser.add_argument(
        "--driver-script",
        type=Path,
        default=Path("run_fbts_simulation.py"),
        help="Path to single-trajectory driver script",
    )

    args, passthrough = parser.parse_known_args(argv)

    if args.n_trajectories < 1:
        raise ValueError("--n-trajectories must be >= 1")
    if args.max_workers is not None and args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    return args, passthrough


def _sanitize_passthrough_args(passthrough_args: Sequence[str]) -> List[str]:
    """Drop deprecated legacy passthrough args that conflict with FBTS driver flags."""
    deprecated_with_value = {"--mapping-seed", "--mapping-log", "--mapping-init-mode", "--mapping-substeps"}
    sanitized: List[str] = []
    skip_next = False
    for token in passthrough_args:
        if skip_next:
            skip_next = False
            continue
        if token in deprecated_with_value:
            skip_next = True
            continue
        if token.startswith("--mapping-"):
            continue
        sanitized.append(token)
    return sanitized




def run_single_trajectory(
    trajectory_id: int,
    seed: int,
    electronic_seed: int,
    output_dir: Path,
    python_executable: str,
    driver_script: Path,
    passthrough_args: Sequence[str],
) -> TrajectoryResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable,
        str(driver_script),
        *passthrough_args,
        "--seed",
        str(seed),
        "--electronic-seed",
        str(electronic_seed),
        "--initial-output",
        str(output_dir / "fbts_initial.xyz"),
        "--trajectory",
        str(output_dir / "fbts_trajectory.xyz"),
        "--energy-log",
        str(output_dir / "fbts_energy.log"),
        "--h-matrix-log",
        str(output_dir / "fbts_effective_hamiltonian.log"),
        "--electronic-log",
        str(output_dir / "fbts_electronic.log"),
        "--observables-log",
        str(output_dir / "fbts_observables.log"),
    ]

    start = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    runtime_s = time.perf_counter() - start

    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

    stderr_tail = " | ".join(line.strip() for line in completed.stderr.strip().splitlines()[-3:])
    status = "success" if completed.returncode == 0 else "failed"

    return TrajectoryResult(
        trajectory_id=trajectory_id,
        seed=seed,
        electronic_seed=electronic_seed,
        output_dir=str(output_dir),
        status=status,
        returncode=completed.returncode,
        runtime_s=runtime_s,
        stderr_tail=stderr_tail,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args, passthrough = parse_args(argv if argv is not None else sys.argv[1:])
    passthrough = _sanitize_passthrough_args(passthrough)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    detected_cpus = detect_available_cpus()
    max_workers = args.max_workers if args.max_workers is not None else detected_cpus
    max_workers = max(1, min(max_workers, args.n_trajectories))

    base_electronic_seed = args.base_seed if args.base_electronic_seed is None else args.base_electronic_seed

    print(
        f"Starting ensemble: n_trajectories={args.n_trajectories}, "
        f"max_workers={max_workers}, detected_cpus={detected_cpus}, output_root={output_root}"
    )

    results: List[TrajectoryResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for traj_id in range(args.n_trajectories):
            seed = args.base_seed + traj_id
            electronic_seed = base_electronic_seed + traj_id
            traj_dir = output_root / f"traj_{traj_id:06d}"
            future = executor.submit(
                run_single_trajectory,
                traj_id,
                seed,
                electronic_seed,
                traj_dir,
                args.python_executable,
                args.driver_script,
                passthrough,
            )
            future_map[future] = traj_id

        for future in as_completed(future_map):
            traj_id = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                result = TrajectoryResult(
                    trajectory_id=traj_id,
                    seed=args.base_seed + traj_id,
                    electronic_seed=base_electronic_seed + traj_id,
                    output_dir=str(output_root / f"traj_{traj_id:06d}"),
                    status="failed",
                    returncode=-1,
                    runtime_s=0.0,
                    stderr_tail=f"executor exception: {exc}",
                )

            results.append(result)
            print(
                f"[traj {result.trajectory_id:06d}] {result.status} "
                f"rc={result.returncode} runtime={result.runtime_s:.2f}s"
            )

    results.sort(key=lambda r: r.trajectory_id)
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trajectory_id",
                "seed",
                "electronic_seed",
                "output_dir",
                "status",
                "returncode",
                "runtime_s",
                "stderr_tail",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "trajectory_id": r.trajectory_id,
                    "seed": r.seed,
                    "electronic_seed": r.electronic_seed,
                    "output_dir": r.output_dir,
                    "status": r.status,
                    "returncode": r.returncode,
                    "runtime_s": f"{r.runtime_s:.6f}",
                    "stderr_tail": r.stderr_tail,
                }
            )

    n_success = sum(1 for r in results if r.status == "success")
    n_failed = len(results) - n_success
    print(f"Finished ensemble. success={n_success}, failed={n_failed}, manifest={manifest_path}")

    # Do not hard-fail whole campaign if some trajectories fail; return nonzero only if all failed.
    return 0 if n_success > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
