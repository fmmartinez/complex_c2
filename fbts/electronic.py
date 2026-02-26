from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .model import PLANCK_REDUCED_KCAL_MOL_FS


@dataclass
class FBTSElectronicState:
    """Forward-backward mapping variables used by FBTS trajectory pairs.

    Attributes:
        r_f/p_f: Forward mapping coordinates and conjugate momenta.
        r_b/p_b: Backward mapping coordinates and conjugate momenta.
        weight: Optional trajectory weight (default 1.0).
        phase: Optional phase accumulator in radians (default 0.0).
    """

    r_f: List[float]
    p_f: List[float]
    r_b: List[float]
    p_b: List[float]
    weight: float = 1.0
    phase: float = 0.0

    def n_states(self) -> int:
        return len(self.r_f)


@dataclass
class FBTSEnsembleSample:
    """Container for an initial FBTS state and how it was generated."""

    state: FBTSElectronicState
    mode: str


def _validate_n_states(n_states: int) -> None:
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}.")


def _complex_amp_to_rp(amplitude: float, phase: float) -> Tuple[float, float]:
    radius = math.sqrt(2.0 * PLANCK_REDUCED_KCAL_MOL_FS) * amplitude
    return radius * math.cos(phase), radius * math.sin(phase)


def initialize_focused_fbts_state(
    n_states: int,
    occupied_state: int,
    rng: random.Random,
    *,
    randomize_phase: bool = True,
    backward_equals_forward: bool = True,
) -> FBTSEnsembleSample:
    """Focused FBTS initializer.

    Uses a focused coherent-state representation for an initially pure diabatic
    state: occupied amplitude = 1, all others = 0 in complex-amplitude space.
    This avoids legacy norm-rescaling assumptions and gives a clear state-prepared
    initial condition for forward/backward trajectories.
    """

    _validate_n_states(n_states)
    if not (0 <= occupied_state < n_states):
        raise ValueError(f"occupied_state must be in [0, {n_states - 1}], got {occupied_state}.")

    r_f = [0.0 for _ in range(n_states)]
    p_f = [0.0 for _ in range(n_states)]
    r_b = [0.0 for _ in range(n_states)]
    p_b = [0.0 for _ in range(n_states)]

    phase_f = rng.uniform(0.0, 2.0 * math.pi) if randomize_phase else 0.0
    r_f[occupied_state], p_f[occupied_state] = _complex_amp_to_rp(1.0, phase_f)

    if backward_equals_forward:
        r_b[occupied_state] = r_f[occupied_state]
        p_b[occupied_state] = p_f[occupied_state]
    else:
        phase_b = rng.uniform(0.0, 2.0 * math.pi) if randomize_phase else 0.0
        r_b[occupied_state], p_b[occupied_state] = _complex_amp_to_rp(1.0, phase_b)

    return FBTSEnsembleSample(
        state=FBTSElectronicState(r_f=r_f, p_f=p_f, r_b=r_b, p_b=p_b),
        mode="focused",
    )


def initialize_coherent_fbts_state(
    n_states: int,
    rng: random.Random,
    *,
    sigma: Optional[float] = None,
) -> FBTSEnsembleSample:
    """Unbiased coherent-state initializer for FBTS ensembles.

    Draws forward/backward mapping variables from independent zero-mean Gaussian
    distributions, giving an unbiased ensemble around the origin in mapping
    phase space. No global norm constraints are applied.
    """

    _validate_n_states(n_states)
    std = math.sqrt(PLANCK_REDUCED_KCAL_MOL_FS) if sigma is None else float(sigma)
    if std <= 0.0:
        raise ValueError(f"sigma must be > 0, got {std}.")

    r_f = [rng.gauss(0.0, std) for _ in range(n_states)]
    p_f = [rng.gauss(0.0, std) for _ in range(n_states)]
    r_b = [rng.gauss(0.0, std) for _ in range(n_states)]
    p_b = [rng.gauss(0.0, std) for _ in range(n_states)]

    return FBTSEnsembleSample(
        state=FBTSElectronicState(r_f=r_f, p_f=p_f, r_b=r_b, p_b=p_b),
        mode="coherent-unbiased",
    )


def _symmetrize_matrix(h: List[List[float]]) -> List[List[float]]:
    n = len(h)
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        out[i][i] = h[i][i]
        for j in range(i + 1, n):
            val = 0.5 * (h[i][j] + h[j][i])
            out[i][j] = val
            out[j][i] = val
    return out


def _jacobi_eigh(h: List[List[float]], max_iter: int = 200, tol: float = 1e-14) -> Tuple[List[float], List[List[float]]]:
    n = len(h)
    a = [row[:] for row in h]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_iter):
        p = 0
        q = 1 if n > 1 else 0
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                aij = abs(a[i][j])
                if aij > off:
                    off = aij
                    p = i
                    q = j

        if off < tol:
            break

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c = math.cos(phi)
        s = math.sin(phi)

        for k in range(n):
            if k == p or k == q:
                continue
            akp = a[k][p]
            akq = a[k][q]
            a[k][p] = c * akp - s * akq
            a[p][k] = a[k][p]
            a[k][q] = s * akp + c * akq
            a[q][k] = a[k][q]

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(n):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq

    evals = [a[i][i] for i in range(n)]
    return evals, v


def _propagate_one_set(r: List[float], p: List[float], evals: List[float], evecs: List[List[float]], dt_fs: float) -> None:
    n_states = len(r)
    r_mode = [sum(evecs[row][col] * r[row] for row in range(n_states)) for col in range(n_states)]
    p_mode = [sum(evecs[row][col] * p[row] for row in range(n_states)) for col in range(n_states)]

    r_rot = [0.0 for _ in range(n_states)]
    p_rot = [0.0 for _ in range(n_states)]
    for i in range(n_states):
        theta = evals[i] * dt_fs / PLANCK_REDUCED_KCAL_MOL_FS
        c = math.cos(theta)
        s = math.sin(theta)
        r_rot[i] = c * r_mode[i] + s * p_mode[i]
        p_rot[i] = c * p_mode[i] - s * r_mode[i]

    for row in range(n_states):
        r[row] = sum(evecs[row][col] * r_rot[col] for col in range(n_states))
        p[row] = sum(evecs[row][col] * p_rot[col] for col in range(n_states))


def propagate_fbts_state_exact_step(state: FBTSElectronicState, h_eff: List[List[float]], dt_fs: float) -> None:
    """Advance forward and backward mapping variables under the same Hamiltonian slice."""

    if dt_fs == 0.0:
        return
    hsym = _symmetrize_matrix(h_eff)
    evals, evecs = _jacobi_eigh(hsym)

    _propagate_one_set(state.r_f, state.p_f, evals, evecs, dt_fs)
    _propagate_one_set(state.r_b, state.p_b, evals, evecs, dt_fs)
