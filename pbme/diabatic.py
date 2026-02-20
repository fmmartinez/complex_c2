from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .model import PROTON_GRID_MIN, PROTON_GRID_STEP


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


def diabatic_r_range(di_table: Dict[str, object]) -> Tuple[float, float]:
    r_values = di_table["r_values"]
    return float(r_values[0]), float(r_values[-1])
