#!/usr/bin/env python3
"""Aggregate PBME trajectory logs and plot smooth histograms/KDEs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot smooth aggregated histograms from PBME logs.")
    parser.add_argument("--ensemble-root", type=Path, default=Path("ensemble_runs"))
    parser.add_argument("--pattern", type=str, default="traj_*/pbme_observables.log")
    parser.add_argument("--output-dir", type=Path, default=Path("ensemble_runs/analysis"))
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument(
        "--n-states",
        type=int,
        default=2,
        help="Subsystem state count used for mapping/effective-H parsing (default: 2).",
    )
    parser.add_argument(
        "--hbar-mapping",
        type=float,
        default=0.01594,
        help="HBAR_MAPPING used for mapping population conversion (default: 0.01594).",
    )
    return parser.parse_args()


def _parse_observables_file(path: Path) -> Tuple[List[float], List[float]]:
    pol: List[float] = []
    com: List[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4 or parts[0] == "step":
            continue
        try:
            p = float(parts[2])
            c = float(parts[3])
        except ValueError:
            continue
        if np.isfinite(p) and np.isfinite(c):
            pol.append(p)
            com.append(c)
    return pol, com


def _parse_effective_h_gap(path: Path) -> List[float]:
    gaps: List[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 7 or parts[0] == "step":
            continue
        try:
            h11 = float(parts[3])
            h22 = float(parts[6])
        except ValueError:
            continue
        if np.isfinite(h11) and np.isfinite(h22):
            gaps.append(h22 - h11)
    return gaps


def _parse_mapping_populations(path: Path, n_states: int, hbar_mapping: float) -> Tuple[List[List[float]], int]:
    pops: List[List[float]] = [[] for _ in range(n_states)]
    available_states = 0

    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts and parts[0] == "step":
            for token in parts[2:]:
                if token.startswith("M") and token[1:].isdigit():
                    available_states += 1
                else:
                    break
            break

    if available_states == 0:
        return pops, 0

    use_states = min(n_states, available_states)
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts or parts[0] == "step":
            continue
        needed = 2 + use_states
        if len(parts) < needed:
            continue
        ok = True
        vals = []
        for i in range(use_states):
            try:
                m_i = float(parts[2 + i])
            except ValueError:
                ok = False
                break
            if not np.isfinite(m_i):
                ok = False
                break
            vals.append((m_i - hbar_mapping) / (2.0 * hbar_mapping))
        if ok:
            for i, v in enumerate(vals):
                pops[i].append(v)

    return pops, available_states


def _scott_bandwidth_1d(values: np.ndarray, scale: float) -> float:
    n = max(values.size, 2)
    std = float(np.std(values, ddof=1))
    if std <= 0.0:
        std = 1.0
    return scale * std * (n ** (-1.0 / 5.0))


def _kde_1d(values: np.ndarray, grid_size: int, bw_scale: float, chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.05 * (hi - lo)
    x = np.linspace(lo - pad, hi + pad, grid_size)
    h = max(_scott_bandwidth_1d(values, bw_scale), 1e-12)

    accum = np.zeros_like(x)
    n = values.size
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        vals = values[start:stop]
        z = (x[:, None] - vals[None, :]) / h
        accum += np.exp(-0.5 * z * z).sum(axis=1)
    density = accum / (n * h * np.sqrt(2.0 * np.pi))
    return x, density


def _scott_bandwidth_2d(x: np.ndarray, y: np.ndarray, scale: float) -> Tuple[float, float]:
    n = max(x.size, 2)
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx <= 0.0:
        sx = 1.0
    if sy <= 0.0:
        sy = 1.0
    factor = scale * (n ** (-1.0 / 6.0))
    return max(factor * sx, 1e-12), max(factor * sy, 1e-12)


def _kde_2d(x: np.ndarray, y: np.ndarray, grid_size: int, bw_scale: float, chunk_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    xpad = 0.05 * (xmax - xmin)
    ypad = 0.05 * (ymax - ymin)
    gx = np.linspace(xmin - xpad, xmax + xpad, grid_size)
    gy = np.linspace(ymin - ypad, ymax + ypad, grid_size)

    hx, hy = _scott_bandwidth_2d(x, y, bw_scale)
    density = np.zeros((gy.size, gx.size), dtype=np.float64)

    n = x.size
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        xc = x[start:stop]
        yc = y[start:stop]
        dx = (gx[:, None] - xc[None, :]) / hx
        dy = (gy[:, None] - yc[None, :]) / hy
        kx = np.exp(-0.5 * dx * dx)
        ky = np.exp(-0.5 * dy * dy)
        density += ky @ kx.T

    density /= (n * 2.0 * np.pi * hx * hy)
    return gx, gy, density


def _plot_1d_kde(values: np.ndarray, title: str, xlabel: str, out_path: Path, args: argparse.Namespace, color: str = "tab:blue") -> None:
    x, d = _kde_1d(values, args.grid_size, args.bandwidth_scale, args.chunk_size)
    plt.figure(figsize=(7, 4.8))
    plt.plot(x, d, lw=2.2, color=color)
    plt.fill_between(x, d, alpha=0.25, color=color)
    plt.xlabel(xlabel)
    plt.ylabel("Probability density")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()


def main() -> None:
    args = parse_args()

    obs_files = sorted(args.ensemble_root.glob(args.pattern))
    if not obs_files:
        raise FileNotFoundError(f"No observables logs found with pattern '{args.pattern}' under '{args.ensemble_root}'.")

    pol_all: List[float] = []
    com_all: List[float] = []
    gaps_all: List[float] = []
    pops_all: List[List[float]] = [[] for _ in range(args.n_states)]

    for obs_path in obs_files:
        traj_dir = obs_path.parent

        pol, com = _parse_observables_file(obs_path)
        pol_all.extend(pol)
        com_all.extend(com)

        map_path = traj_dir / "pbme_mapping.log"
        if map_path.exists():
            pops, available_states = _parse_mapping_populations(map_path, args.n_states, args.hbar_mapping)
            if available_states < args.n_states:
                print(
                    f"Warning: {map_path} provides only {available_states} mapping norms (M_i), "
                    f"requested n_states={args.n_states}; missing states will be skipped."
                )
            for i in range(args.n_states):
                pops_all[i].extend(pops[i])

        if args.n_states == 2:
            h_path = traj_dir / "pbme_effective_hamiltonian.log"
            if h_path.exists():
                gaps_all.extend(_parse_effective_h_gap(h_path))

    if not pol_all:
        raise RuntimeError("No valid observable samples were parsed from discovered log files.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pol_arr = np.asarray(pol_all, dtype=np.float64)
    com_arr = np.asarray(com_all, dtype=np.float64)

    out_pol = args.output_dir / "solvent_polarization_kde.png"
    _plot_1d_kde(pol_arr, f"Aggregated solvent polarization KDE\nN={pol_arr.size} samples from {len(obs_files)} trajectories", "Solvent polarization (dE_pol)", out_pol, args)

    out_com = args.output_dir / "com_distance_kde.png"
    _plot_1d_kde(com_arr, f"Aggregated COM distance KDE\nN={com_arr.size} samples from {len(obs_files)} trajectories", "COM distance", out_com, args, color="tab:orange")

    gx, gy, dens = _kde_2d(com_arr, pol_arr, args.grid_size, args.bandwidth_scale, args.chunk_size)
    plt.figure(figsize=(7.2, 5.6))
    im = plt.pcolormesh(gx, gy, dens, shading="auto", cmap="viridis")
    plt.xlabel("COM distance")
    plt.ylabel("Solvent polarization (dE_pol)")
    plt.title(f"Joint KDE: COM distance vs solvent polarization\nN={pol_arr.size} samples from {len(obs_files)} trajectories")
    cbar = plt.colorbar(im)
    cbar.set_label("Probability density")
    plt.tight_layout()
    out_joint = args.output_dir / "com_vs_polarization_joint_kde.png"
    plt.savefig(out_joint, dpi=args.dpi)
    plt.close()

    plt.figure(figsize=(7.2, 5.6))
    levels = np.linspace(float(dens.min()), float(dens.max()), 12)
    cf = plt.contourf(gx, gy, dens, levels=levels, cmap="viridis")
    cl = plt.contour(gx, gy, dens, levels=levels, colors="white", linewidths=0.6, alpha=0.8)
    plt.clabel(cl, inline=True, fmt="%.2e", fontsize=7)
    plt.xlabel("COM distance")
    plt.ylabel("Solvent polarization (dE_pol)")
    plt.title(f"Joint KDE contours: COM distance vs solvent polarization\nN={pol_arr.size} samples from {len(obs_files)} trajectories")
    cbar = plt.colorbar(cf)
    cbar.set_label("Probability density")
    plt.tight_layout()
    out_joint_contour = args.output_dir / "com_vs_polarization_joint_kde_contours.png"
    plt.savefig(out_joint_contour, dpi=args.dpi)
    plt.close()

    out_gap = None
    if args.n_states == 2:
        if gaps_all:
            gap_arr = np.asarray(gaps_all, dtype=np.float64)
            out_gap = args.output_dir / "effective_hamiltonian_gap_h22_minus_h11_kde.png"
            _plot_1d_kde(
                gap_arr,
                f"Effective Hamiltonian gap KDE (H22 - H11)\nN={gap_arr.size} samples from {len(obs_files)} trajectories",
                "H22 - H11 (kcal/mol)",
                out_gap,
                args,
                color="tab:green",
            )
        else:
            print("Warning: n_states=2 but no valid pbme_effective_hamiltonian.log samples found; skipping gap histogram.")
    else:
        print(f"Warning: n_states={args.n_states}; gap histogram (H22-H11) is only generated for n_states=2.")

    pop_outs: List[Path] = []
    for i in range(args.n_states):
        if not pops_all[i]:
            print(f"Warning: no valid mapping population samples found for state {i + 1}; skipping.")
            continue
        pop_arr = np.asarray(pops_all[i], dtype=np.float64)
        out = args.output_dir / f"mapping_population_state_{i + 1}_kde.png"
        _plot_1d_kde(
            pop_arr,
            f"Mapping population KDE, state {i + 1}\nN={pop_arr.size} samples",
            f"Population state {i + 1}",
            out,
            args,
            color="tab:red",
        )
        pop_outs.append(out)

    print(f"Discovered {len(obs_files)} trajectory observable log files.")
    print(f"Aggregated {pol_arr.size} observables samples.")
    print(f"KDE chunk size: {args.chunk_size}")
    print(f"Assumed n_states: {args.n_states}; hbar_mapping: {args.hbar_mapping}")
    print(f"Wrote: {out_pol}")
    print(f"Wrote: {out_com}")
    print(f"Wrote: {out_joint}")
    print(f"Wrote: {out_joint_contour}")
    if out_gap is not None:
        print(f"Wrote: {out_gap}")
    for out in pop_outs:
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
