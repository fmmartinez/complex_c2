#!/usr/bin/env python3
"""Aggregate FBTS trajectory logs and plot smooth histograms/KDEs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot smooth aggregated histograms from FBTS logs.")
    parser.add_argument("--ensemble-root", type=Path, default=Path("ensemble_runs"))
    parser.add_argument("--pattern", type=str, default="traj_*/fbts_observables.log")
    parser.add_argument("--output-dir", type=Path, default=Path("ensemble_runs/analysis"))
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument(
        "--n-states",
        type=int,
        default=2,
        help="Subsystem state count used for electronic/effective-H parsing (default: 2).",
    )
    parser.add_argument(
        "--diabatic-json",
        type=Path,
        default=Path("diabatic_matrices.json"),
        help="Path to diabatic_matrices.json used to define R_AB bin extent.",
    )
    parser.add_argument(
        "--rab-bins",
        type=int,
        default=10,
        help="Number of R_AB bins for conditional histograms (default: 10).",
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


def _parse_fbts_electronic(path: Path, n_states: int) -> Tuple[List[List[float]], Dict[str, List[float]]]:
    pops: List[List[float]] = [[] for _ in range(n_states)]
    diagnostics: Dict[str, List[float]] = {}

    lines = path.read_text(encoding="utf-8").splitlines()
    header_tokens: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts and parts[0] == "step":
            header_tokens = parts
            break
    if not header_tokens:
        return pops, diagnostics

    idx_map = {tok: i for i, tok in enumerate(header_tokens)}

    pop_indices: List[int] = []
    for i in range(1, n_states + 1):
        tok = f"pop{i}"
        if tok in idx_map:
            pop_indices.append(idx_map[tok])

    rho_diag_re_indices: List[int] = []
    for i in range(1, n_states + 1):
        tok = f"rho{i}{i}_re"
        if tok in idx_map:
            rho_diag_re_indices.append(idx_map[tok])

    diag_keys = [tok for tok in header_tokens if "weight" in tok.lower()]
    for k in diag_keys:
        diagnostics[k] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts or parts[0] == "step":
            continue

        if pop_indices:
            for i, idx in enumerate(pop_indices[:n_states]):
                if idx >= len(parts):
                    continue
                try:
                    v = float(parts[idx])
                except ValueError:
                    continue
                if np.isfinite(v):
                    pops[i].append(v)
        elif rho_diag_re_indices:
            for i, idx in enumerate(rho_diag_re_indices[:n_states]):
                if idx >= len(parts):
                    continue
                try:
                    v = float(parts[idx])
                except ValueError:
                    continue
                if np.isfinite(v):
                    pops[i].append(v)

        for k in diag_keys:
            idx = idx_map[k]
            if idx >= len(parts):
                continue
            try:
                v = float(parts[idx])
            except ValueError:
                continue
            if np.isfinite(v):
                diagnostics[k].append(v)

    return pops, diagnostics


def _load_rab_bounds_from_diabatic(path: Path) -> Tuple[float, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    results = obj.get("results", [])
    if not results:
        raise RuntimeError(f"No 'results' entries found in {path}.")
    r_vals = [float(item["R"]) for item in results]
    return float(min(r_vals)), float(max(r_vals))


def _parse_rab_series(path: Path) -> List[float]:
    rabs: List[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3 or parts[0] == "step":
            continue
        try:
            rab = float(parts[2])
        except ValueError:
            continue
        if np.isfinite(rab):
            rabs.append(rab)
    return rabs


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
    rab_all: List[float] = []
    gaps_all: List[float] = []
    pops_all: List[List[float]] = [[] for _ in range(args.n_states)]
    weight_diag_all: Dict[str, List[float]] = {}

    if args.rab_bins <= 0:
        raise ValueError("--rab-bins must be >= 1.")
    rab_min, rab_max = _load_rab_bounds_from_diabatic(args.diabatic_json)
    rab_edges = np.linspace(rab_min, rab_max, args.rab_bins + 1)
    pol_by_rab: List[List[float]] = [[] for _ in range(args.rab_bins)]
    com_by_rab: List[List[float]] = [[] for _ in range(args.rab_bins)]

    for obs_path in obs_files:
        traj_dir = obs_path.parent

        pol, com = _parse_observables_file(obs_path)
        pol_all.extend(pol)
        com_all.extend(com)

        h_path_for_rab = traj_dir / "fbts_effective_hamiltonian.log"
        if h_path_for_rab.exists():
            rabs = _parse_rab_series(h_path_for_rab)
            rab_all.extend(rabs)
            n_pair = min(len(pol), len(com), len(rabs))
            if n_pair < len(pol) or n_pair < len(rabs):
                print(
                    f"Warning: length mismatch in {traj_dir} (obs={len(pol)}, rab={len(rabs)}); using first {n_pair} samples for R_AB-binned histograms."
                )
            for i in range(n_pair):
                rab = rabs[i]
                if rab < rab_min or rab > rab_max:
                    continue
                bin_idx = int(np.searchsorted(rab_edges, rab, side="right") - 1)
                if bin_idx == args.rab_bins:
                    bin_idx = args.rab_bins - 1
                if 0 <= bin_idx < args.rab_bins:
                    pol_by_rab[bin_idx].append(pol[i])
                    com_by_rab[bin_idx].append(com[i])
        else:
            print(f"Warning: missing {h_path_for_rab}; skipping R_AB-binned observables for this trajectory.")

        elec_path = traj_dir / "fbts_electronic.log"
        if elec_path.exists():
            pops, diagnostics = _parse_fbts_electronic(elec_path, args.n_states)
            for i in range(args.n_states):
                pops_all[i].extend(pops[i])
            for k, vals in diagnostics.items():
                weight_diag_all.setdefault(k, []).extend(vals)

        if args.n_states == 2:
            h_path = traj_dir / "fbts_effective_hamiltonian.log"
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

    out_rab = None
    if rab_all:
        rab_arr = np.asarray(rab_all, dtype=np.float64)
        out_rab = args.output_dir / "rab_kde.png"
        _plot_1d_kde(
            rab_arr,
            f"Aggregated R_AB KDE\nN={rab_arr.size} samples from {len(obs_files)} trajectories",
            "R_AB",
            out_rab,
            args,
            color="tab:purple",
        )
    else:
        print("Warning: no valid R_AB samples found in fbts_effective_hamiltonian.log files; skipping R_AB histogram.")

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
            print("Warning: n_states=2 but no valid fbts_effective_hamiltonian.log samples found; skipping gap histogram.")
    else:
        print(f"Warning: n_states={args.n_states}; gap histogram (H22-H11) is only generated for n_states=2.")

    pop_outs: List[Path] = []
    for i in range(args.n_states):
        if not pops_all[i]:
            print(f"Warning: no valid FBTS population samples found for state {i + 1}; skipping.")
            continue
        pop_arr = np.asarray(pops_all[i], dtype=np.float64)
        out = args.output_dir / f"fbts_population_state_{i + 1}_kde.png"
        _plot_1d_kde(
            pop_arr,
            f"FBTS population KDE, state {i + 1}\nN={pop_arr.size} samples",
            f"Population state {i + 1}",
            out,
            args,
            color="tab:red",
        )
        pop_outs.append(out)

    weight_outs: List[Path] = []
    for key, values in weight_diag_all.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        out = args.output_dir / f"{key}_kde.png"
        _plot_1d_kde(
            arr,
            f"FBTS diagnostic KDE: {key}\nN={arr.size} samples",
            key,
            out,
            args,
            color="tab:brown",
        )
        weight_outs.append(out)

    rab_pol_outs: List[Path] = []
    rab_com_outs: List[Path] = []
    for b in range(args.rab_bins):
        if not pol_by_rab[b] or not com_by_rab[b]:
            print(
                f"Warning: no samples in R_AB bin {b + 1}/{args.rab_bins} [{rab_edges[b]:.6f}, {rab_edges[b + 1]:.6f}); skipping."
            )
            continue
        pol_bin_arr = np.asarray(pol_by_rab[b], dtype=np.float64)
        com_bin_arr = np.asarray(com_by_rab[b], dtype=np.float64)

        out_pol_bin = args.output_dir / f"solvent_polarization_kde_rab_bin_{b + 1:02d}.png"
        _plot_1d_kde(
            pol_bin_arr,
            f"Solvent polarization KDE, R_AB bin {b + 1}/{args.rab_bins}\n"
            f"[{rab_edges[b]:.4f}, {rab_edges[b + 1]:.4f}] (N={pol_bin_arr.size})",
            "Solvent polarization (dE_pol)",
            out_pol_bin,
            args,
        )
        rab_pol_outs.append(out_pol_bin)

        out_com_bin = args.output_dir / f"com_distance_kde_rab_bin_{b + 1:02d}.png"
        _plot_1d_kde(
            com_bin_arr,
            f"COM distance KDE, R_AB bin {b + 1}/{args.rab_bins}\n"
            f"[{rab_edges[b]:.4f}, {rab_edges[b + 1]:.4f}] (N={com_bin_arr.size})",
            "COM distance",
            out_com_bin,
            args,
            color="tab:orange",
        )
        rab_com_outs.append(out_com_bin)

    print(f"Discovered {len(obs_files)} trajectory observable log files.")
    print(f"Aggregated {pol_arr.size} observables samples.")
    print(f"KDE chunk size: {args.chunk_size}")
    print(f"Assumed n_states: {args.n_states}")
    print(f"R_AB bins: {args.rab_bins} across [{rab_min:.6f}, {rab_max:.6f}] from {args.diabatic_json}")
    print(f"Wrote: {out_pol}")
    print(f"Wrote: {out_com}")
    print(f"Wrote: {out_joint}")
    print(f"Wrote: {out_joint_contour}")
    if out_rab is not None:
        print(f"Wrote: {out_rab}")
    if out_gap is not None:
        print(f"Wrote: {out_gap}")
    for out in pop_outs:
        print(f"Wrote: {out}")
    for out in weight_outs:
        print(f"Wrote: {out}")
    for out in rab_pol_outs:
        print(f"Wrote: {out}")
    for out in rab_com_outs:
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
