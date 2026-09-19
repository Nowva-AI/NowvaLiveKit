"""Synthetic 30 Hz squat trajectory with spike outliers and multi-frame glitches (e.g. 2D limb swap in one view).

Compares causal outlier stages in front of a fixed-lag CV Kalman smoother:
none, VelocityClamp (repo, 2.5 m/s), causal Hampel (window 7, 3 MAD), Kalman innovation gating (chi2).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/.claude/preik-audit/research")
from filter_lag_experiment import (  # noqa: E402
    DT_S, FS_HZ, _kalman_matrices, make_squat,
)

MAX_VEL_M_S = 2.5


def velocity_clamp(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    prev = x[0]
    max_step = MAX_VEL_M_S * DT_S
    for i in range(1, len(x)):
        step = x[i] - prev
        if abs(step) > max_step:
            y[i] = prev + np.sign(step) * max_step
        prev = y[i]
    return y


def causal_hampel(x: np.ndarray, window: int = 7, n_mad: float = 3.0, floor_m: float = 0.03) -> np.ndarray:
    # trailing-window Hampel on RAW history (no look-ahead): median lags real motion by ~window/2 frames
    y = x.copy()
    for i in range(window, len(x)):
        hist = x[i - window:i]
        med = np.median(hist)
        mad = 1.4826 * np.median(np.abs(hist - med))
        if abs(x[i] - med) > max(n_mad * mad, floor_m):
            y[i] = med
    return y


def gated_kalman_fixed_lag(x: np.ndarray, q: float, r: float, lag: int, gate_sigma: float | None,
                           max_rejects: int = 4) -> np.ndarray:
    F, Q = _kalman_matrices("cv", q)
    H = np.array([[1.0, 0.0]])
    s = np.array([x[0], 0.0]); P = np.eye(2)
    xs, Ps, xps, Pps = [], [], [], []
    rejects = 0
    for z in x:
        sp = F @ s; Pp = F @ P @ F.T + Q
        S = float((H @ Pp @ H.T)[0, 0]) + r
        innov = z - sp[0]
        gate_m = None if gate_sigma is None else max(gate_sigma * np.sqrt(S), 0.04)
        if gate_m is not None and abs(innov) > gate_m and rejects < max_rejects:
            s, P = sp, Pp * 2.0  # inflate so a real step is re-acquired quickly
            rejects += 1
        else:
            K = (Pp @ H.T / S)[:, 0]
            s = sp + K * innov
            P = (np.eye(2) - np.outer(K, H)) @ Pp
            rejects = 0
        xs.append(s.copy()); Ps.append(P.copy()); xps.append(sp); Pps.append(Pp)
    xs = np.array(xs)
    out = np.empty(len(x))
    for k in range(len(x)):
        end = min(len(x) - 1, k + lag)
        sm = xs[end].copy()
        for j in range(end - 1, k - 1, -1):
            C = Ps[j] @ F.T @ np.linalg.inv(Pps[j + 1])
            sm = xs[j] + C @ (sm - xps[j + 1])
        out[k] = sm[0]
    return out


def main() -> None:
    noise_m = 0.01
    hip_true, knee_true = make_squat(1.5, 0.3, 1.0, 1.5, reps=5)
    q, r = 30.0, noise_m**2
    rows: dict[str, list[tuple[float, float]]] = {}
    for seed in range(30):
        rng = np.random.default_rng(seed)
        for label, truth in (("hip", hip_true), ("knee", knee_true)):
            noisy = truth + rng.normal(0, noise_m, len(truth))
            # single-frame spikes: 3% of frames, 5-20 cm
            spike_idx = rng.choice(len(truth), int(0.03 * len(truth)), replace=False)
            noisy[spike_idx] += rng.choice([-1, 1], len(spike_idx)) * rng.uniform(0.05, 0.20, len(spike_idx))
            # multi-frame glitch (limb swap in one view): 4 frames offset by 10 cm, twice
            for start in rng.choice(len(truth) - 5, 2, replace=False):
                noisy[start:start + 4] += 0.10
            candidates = {
                "no outlier stage + fixed-lag CV(3)": gated_kalman_fixed_lag(noisy, q, r, 3, None),
                "VelocityClamp 2.5 m/s + fixed-lag CV(3)": gated_kalman_fixed_lag(velocity_clamp(noisy), q, r, 3, None),
                "trailing Hampel(7, 3MAD, 3cm floor) + fixed-lag CV(3)": gated_kalman_fixed_lag(causal_hampel(noisy), q, r, 3, None),
                "innovation gate max(3 sigma, 4 cm), fixed-lag CV(3)": gated_kalman_fixed_lag(noisy, q, r, 3, 3.0),
                "innovation gate max(4 sigma, 4 cm), fixed-lag CV(3)": gated_kalman_fixed_lag(noisy, q, r, 3, 4.0),
                "raw (no filtering)": noisy,
                "VelocityClamp only (current chain)": velocity_clamp(noisy),
            }
            for name, est in candidates.items():
                rmse_cm = float(np.sqrt(np.mean((est - truth) ** 2)) * 100)
                max_err_cm = float(np.max(np.abs(est - truth)) * 100)
                rows.setdefault(f"{label}: {name}", []).append((rmse_cm, max_err_cm))
    print("noise 1 cm + 3% spikes (5-20 cm) + 2x 4-frame 10 cm glitches, 30 seeds")
    print(f"{'signal: pipeline':58s} {'RMSE cm':>8s} {'p95 max err cm':>15s}")
    for name, vals in rows.items():
        arr = np.array(vals)
        print(f"{name:58s} {arr[:,0].mean():8.2f} {np.percentile(arr[:,1], 95):15.2f}")


if __name__ == "__main__":
    main()
