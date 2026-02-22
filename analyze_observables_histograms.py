#!/usr/bin/env python3
"""Aggregate PBME observables across ensemble trajectories and plot smooth histograms.

Expected inputs are files matching:
  ensemble_runs/traj_*/pbme_observables.log

Column convention in pbme_observables.log:
  col1 step
  col2 time_fs
  col3 solvent polarization (dE_pol)
  col4 COM distance
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot smooth aggregated histograms from PBME observables logs.")
    parser.add_argument(
        "--ensemble-root",
        type=Path,
        default=Path("ensemble_runs"),
        help="Root folder that contains traj_* subfolders (default: ensemble_runs).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="traj_*/pbme_observables.log",
        help="Glob pattern under --ensemble-root used to discover observable logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ensemble_runs/analysis"),
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=200,
        help="Grid size for KDE evaluation (default: 200).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--bandwidth-scale",
        type=float,
        default=1.0,
        help="Multiplier on Scott bandwidth (e.g., >1 smoother, <1 sharper).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4000,
        help="Chunk size for memory-efficient KDE accumulation (default: 4000).",
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
        if len(parts) < 4:
            continue
        if parts[0] == "step":
            continue
        try:
            pol_val = float(parts[2])
            com_val = float(parts[3])
        except ValueError:
            continue
        if np.isfinite(pol_val) and np.isfinite(com_val):
            pol.append(pol_val)
            com.append(com_val)

    return pol, com


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


def _kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    grid_size: int,
    bw_scale: float,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        kx = np.exp(-0.5 * dx * dx)  # (nx, chunk)
        ky = np.exp(-0.5 * dy * dy)  # (ny, chunk)
        density += ky @ kx.T

    density /= (n * 2.0 * np.pi * hx * hy)
    return gx, gy, density


def main() -> None:
    args = parse_args()

    files = sorted(args.ensemble_root.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No observables logs found with pattern '{args.pattern}' under '{args.ensemble_root}'."
        )

    pol_all: List[float] = []
    com_all: List[float] = []
    for path in files:
        pol, com = _parse_observables_file(path)
        pol_all.extend(pol)
        com_all.extend(com)

    if not pol_all:
        raise RuntimeError("No valid observable samples were parsed from discovered log files.")

    pol_arr = np.asarray(pol_all, dtype=np.float64)
    com_arr = np.asarray(com_all, dtype=np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1D KDE: solvent polarization
    x_pol, d_pol = _kde_1d(pol_arr, args.grid_size, args.bandwidth_scale, args.chunk_size)
    plt.figure(figsize=(7, 4.8))
    plt.plot(x_pol, d_pol, lw=2.2, label="KDE")
    plt.fill_between(x_pol, d_pol, alpha=0.25)
    plt.xlabel("Solvent polarization (dE_pol)")
    plt.ylabel("Probability density")
    plt.title(f"Aggregated solvent polarization KDE\nN={pol_arr.size} samples from {len(files)} trajectories")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_pol = args.output_dir / "solvent_polarization_kde.png"
    plt.savefig(out_pol, dpi=args.dpi)
    plt.close()

    # 1D KDE: COM distance
    x_com, d_com = _kde_1d(com_arr, args.grid_size, args.bandwidth_scale, args.chunk_size)
    plt.figure(figsize=(7, 4.8))
    plt.plot(x_com, d_com, lw=2.2, color="tab:orange", label="KDE")
    plt.fill_between(x_com, d_com, alpha=0.25, color="tab:orange")
    plt.xlabel("COM distance")
    plt.ylabel("Probability density")
    plt.title(f"Aggregated COM distance KDE\nN={com_arr.size} samples from {len(files)} trajectories")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_com = args.output_dir / "com_distance_kde.png"
    plt.savefig(out_com, dpi=args.dpi)
    plt.close()

    # 2D joint KDE
    gx, gy, dens = _kde_2d(com_arr, pol_arr, args.grid_size, args.bandwidth_scale, args.chunk_size)
    plt.figure(figsize=(7.2, 5.6))
    im = plt.pcolormesh(gx, gy, dens, shading="auto", cmap="viridis")
    plt.xlabel("COM distance")
    plt.ylabel("Solvent polarization (dE_pol)")
    plt.title(f"Joint KDE: COM distance vs solvent polarization\nN={pol_arr.size} samples from {len(files)} trajectories")
    cbar = plt.colorbar(im)
    cbar.set_label("Probability density")
    plt.tight_layout()
    out_joint = args.output_dir / "com_vs_polarization_joint_kde.png"
    plt.savefig(out_joint, dpi=args.dpi)
    plt.close()

    # 2D joint KDE (contours)
    plt.figure(figsize=(7.2, 5.6))
    levels = np.linspace(float(dens.min()), float(dens.max()), 12)
    cf = plt.contourf(gx, gy, dens, levels=levels, cmap="viridis")
    cl = plt.contour(gx, gy, dens, levels=levels, colors="white", linewidths=0.6, alpha=0.8)
    plt.clabel(cl, inline=True, fmt="%.2e", fontsize=7)
    plt.xlabel("COM distance")
    plt.ylabel("Solvent polarization (dE_pol)")
    plt.title(f"Joint KDE contours: COM distance vs solvent polarization\nN={pol_arr.size} samples from {len(files)} trajectories")
    cbar = plt.colorbar(cf)
    cbar.set_label("Probability density")
    plt.tight_layout()
    out_joint_contour = args.output_dir / "com_vs_polarization_joint_kde_contours.png"
    plt.savefig(out_joint_contour, dpi=args.dpi)
    plt.close()

    print(f"Discovered {len(files)} trajectory log files.")
    print(f"Aggregated {pol_arr.size} samples.")
    print(f"KDE chunk size: {args.chunk_size}")
    print(f"Wrote: {out_pol}")
    print(f"Wrote: {out_com}")
    print(f"Wrote: {out_joint}")
    print(f"Wrote: {out_joint_contour}")


if __name__ == "__main__":
    main()
