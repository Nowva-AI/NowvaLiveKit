"""Synthetic squat experiment: lag / peak error / noise of causal vs rep-window filters at 30 Hz.

Signals (metres): hip-mid vertical trajectory (45 cm depth) and a knee medial excursion (3 cm valgus
bump lasting ~0.6 s around the bottom). Noise: white Gaussian (sigma configurable) plus optional
spike outliers. Reports RMSE, bottom-depth error, valgus-peak attenuation and lag for each filter.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from biomechanics.utils.filters import OneEuroFilter  # noqa: E402

FS_HZ = 30.0
DT_S = 1.0 / FS_HZ
RNG = np.random.default_rng(7)


def _min_jerk(n_frames: int) -> np.ndarray:
    tau = np.linspace(0.0, 1.0, n_frames)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def make_squat(descent_s: float, bottom_s: float, ascent_s: float, stand_s: float, reps: int,
               depth_m: float = 0.45, valgus_m: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    hip, knee = [np.zeros(int(stand_s * FS_HZ))], [np.zeros(int(stand_s * FS_HZ))]
    for _ in range(reps):
        d = _min_jerk(int(descent_s * FS_HZ)) * depth_m
        b = np.full(int(bottom_s * FS_HZ), depth_m)
        a = depth_m - _min_jerk(int(ascent_s * FS_HZ)) * depth_m
        s = np.zeros(int(stand_s * FS_HZ))
        hip_rep = np.concatenate([d, b, a, s])
        # valgus bump: raised-cosine centred on end of bottom / start of ascent, 0.6 s wide
        t_rep = np.arange(len(hip_rep)) * DT_S
        centre = descent_s + bottom_s + 0.15
        width = 0.6
        bump = np.where(np.abs(t_rep - centre) < width / 2,
                        0.5 * (1 + np.cos(2 * math.pi * (t_rep - centre) / width)), 0.0) * valgus_m
        hip.append(hip_rep)
        knee.append(bump)
    return np.concatenate(hip), np.concatenate(knee)


def f_raw(x: np.ndarray) -> np.ndarray:
    return x.copy()


def f_ema(alpha: float):
    def run(x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y
    return run


def f_one_euro(min_cutoff: float, beta: float, d_cutoff: float = 1.0):
    def run(x: np.ndarray) -> np.ndarray:
        filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        return np.array([filt.filter(float(v), i * DT_S) for i, v in enumerate(x)])
    return run


def f_butter_causal(cutoff_hz: float, order: int = 2):
    sos = signal.butter(order, cutoff_hz / (FS_HZ / 2), output="sos")

    def run(x: np.ndarray) -> np.ndarray:
        zi = signal.sosfilt_zi(sos) * x[0]
        y, _ = signal.sosfilt(sos, x, zi=zi)
        return y
    return run


def f_butter_zero_phase(cutoff_hz: float, order: int = 4):
    sos = signal.butter(order // 2, cutoff_hz / (FS_HZ / 2), output="sos")

    def run(x: np.ndarray) -> np.ndarray:
        return signal.sosfiltfilt(sos, x)
    return run


def _kalman_matrices(model: str, q: float) -> tuple[np.ndarray, np.ndarray]:
    if model == "cv":
        F = np.array([[1, DT_S], [0, 1]])
        G = np.array([[0.5 * DT_S**2], [DT_S]])
    else:
        F = np.array([[1, DT_S, 0.5 * DT_S**2], [0, 1, DT_S], [0, 0, 1]])
        G = np.array([[DT_S**3 / 6], [0.5 * DT_S**2], [DT_S]])
    Q = G @ G.T * q
    return F, Q


def kalman_forward(x: np.ndarray, model: str, q: float, r: float):
    F, Q = _kalman_matrices(model, q)
    n = F.shape[0]
    H = np.zeros((1, n)); H[0, 0] = 1
    s = np.zeros(n); s[0] = x[0]
    P = np.eye(n) * 1.0
    xs, Ps, xps, Pps = [], [], [], []
    for z in x:
        sp = F @ s; Pp = F @ P @ F.T + Q
        S = H @ Pp @ H.T + r
        K = Pp @ H.T / S
        s = sp + (K[:, 0] * (z - sp[0]))
        P = (np.eye(n) - K @ H) @ Pp
        xs.append(s.copy()); Ps.append(P.copy()); xps.append(sp); Pps.append(Pp)
    return np.array(xs), np.array(Ps), np.array(xps), np.array(Pps), F


def f_kalman(model: str, q: float, r: float):
    def run(x: np.ndarray) -> np.ndarray:
        xs, *_ = kalman_forward(x, model, q, r)
        return xs[:, 0]
    return run


def f_fixed_lag(model: str, q: float, r: float, lag: int):
    """Output at frame k is the RTS-smoothed estimate of frame k computed with data up to k+lag,
    i.e. a latency of `lag` frames. Implemented as windowed RTS (exact for the linear model)."""
    def run(x: np.ndarray) -> np.ndarray:
        xs, Ps, xps, Pps, F = kalman_forward(x, model, q, r)
        out = np.empty(len(x))
        for k in range(len(x)):
            end = min(len(x) - 1, k + lag)
            sm = xs[end].copy()
            for j in range(end - 1, k - 1, -1):
                C = Ps[j] @ F.T @ np.linalg.inv(Pps[j + 1])
                sm = xs[j] + C @ (sm - xps[j + 1])
            out[k] = sm[0]
        return out
    return run


def lag_ms(truth: np.ndarray, est: np.ndarray, max_lag_frames: int = 20) -> float:
    # sub-frame lag: shift estimate by fractional frames (linear interp) and minimise squared error
    t = np.arange(len(truth), dtype=float)
    best_err, best_lag = np.inf, 0.0
    for lag in np.arange(0.0, max_lag_frames, 0.1):
        shifted = np.interp(t + lag, t, est)
        valid = slice(int(max_lag_frames) + 1, len(truth) - int(max_lag_frames) - 1)
        err = float(np.mean((shifted[valid] - truth[valid]) ** 2))
        if err < best_err:
            best_err, best_lag = err, lag
    return best_lag * 1000.0 / FS_HZ


def evaluate(name: str, fn, hip_true, knee_true, noise_m, rep_bottom_idx, n_seeds: int = 30) -> dict:
    # clean pass: deterministic lag and peak attenuation
    hip_clean = fn(hip_true); knee_clean = fn(knee_true)
    depth_under_cm = float(np.mean([(hip_true[lo:hi].max() - hip_clean[lo:hi].max()) * 100 for lo, hi in rep_bottom_idx]))
    valgus_clean_pct = float(np.mean([knee_clean[lo:hi].max() / knee_true[lo:hi].max() for lo, hi in rep_bottom_idx]) * 100)
    lag = lag_ms(hip_true, hip_clean)
    # noisy passes
    rmse, jitter, valgus_noisy = [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        hip_n = hip_true + rng.normal(0, noise_m, len(hip_true))
        knee_n = knee_true + rng.normal(0, noise_m, len(knee_true))
        he = fn(hip_n); ke = fn(knee_n)
        rmse.append(np.sqrt(np.mean((he - hip_true) ** 2)) * 100)
        jitter.append(np.std(np.diff(he[5: int(FS_HZ * 1.5)])) * 1000)
        valgus_noisy.append(np.mean([ke[lo:hi].max() / knee_true[lo:hi].max() for lo, hi in rep_bottom_idx]) * 100)
    return dict(name=name, rmse_hip_cm=float(np.mean(rmse)), depth_under_cm=depth_under_cm,
                valgus_clean_pct=valgus_clean_pct, valgus_noisy_pct=float(np.mean(valgus_noisy)),
                lag_ms=lag, stand_jitter_mm=float(np.mean(jitter)))


def main() -> None:
    noise_m = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    tempo = sys.argv[2] if len(sys.argv) > 2 else "normal"
    if tempo == "fast":
        spec = (0.7, 0.0, 0.5, 1.5)
    else:
        spec = (1.5, 0.3, 1.0, 1.5)
    reps = 5
    hip_true, knee_true = make_squat(*spec, reps=reps)
    hip_noisy = hip_true + RNG.normal(0, noise_m, len(hip_true))
    knee_noisy = knee_true + RNG.normal(0, noise_m, len(knee_true))
    rep_len = int(sum(spec) * FS_HZ)
    start = int(1.5 * FS_HZ)
    rep_bottom_idx = [(start + i * rep_len, start + (i + 1) * rep_len) for i in range(reps)]

    q_cv = 30.0   # (m/s^2)^2 white-accel PSD scale, tuned by sweep below
    q_ca = 900.0
    r = noise_m**2
    filters = [
        ("raw", f_raw),
        ("EMA a=0.75 (ConfidenceBlender @conf 0.7)", f_ema(0.75)),
        ("EMA a=0.5 (ConfidenceBlender @conf 0.5)", f_ema(0.5)),
        ("OneEuro pos 0.8/4.0 (KeypointPositionSmoother)", f_one_euro(0.8, 4.0)),
        ("OneEuro MediaPipe-world 0.1/40", f_one_euro(0.1, 40.0)),
        ("OneEuro 1.0/20", f_one_euro(1.0, 20.0)),
        ("Butter causal 2nd 6Hz", f_butter_causal(6.0)),
        ("Butter causal 2nd 4Hz", f_butter_causal(4.0)),
        ("Butter causal 2nd 3Hz", f_butter_causal(3.0)),
        ("Kalman CV causal", f_kalman("cv", q_cv, r)),
        ("Kalman CA causal", f_kalman("ca", q_ca, r)),
        ("Kalman CV fixed-lag 3fr (100ms)", f_fixed_lag("cv", q_cv, r, 3)),
        ("Kalman CV fixed-lag 6fr (200ms)", f_fixed_lag("cv", q_cv, r, 6)),
        ("Kalman CA fixed-lag 6fr (200ms)", f_fixed_lag("ca", q_ca, r, 6)),
        ("filtfilt 4th 6Hz (rep-window, zero-phase)", f_butter_zero_phase(6.0)),
        ("filtfilt 4th 4Hz (rep-window, zero-phase)", f_butter_zero_phase(4.0)),
    ]
    print(f"noise sigma = {noise_m*100:.1f} cm, tempo = {tempo}, fs = {FS_HZ} Hz, 30 noise seeds")
    print("clean-input columns: lag ms (signal delay), depth undershoot cm, valgus peak kept %; noisy columns: RMSE cm, valgus peak % (noisy), standing jitter mm/frame")
    print(f"{'filter':48s} {'lag':>5s} {'depthU':>6s} {'valg%':>6s} | {'RMSE':>5s} {'valg%n':>6s} {'jit':>5s}")
    for name, fn in filters:
        res = evaluate(name, fn, hip_true, knee_true, noise_m, rep_bottom_idx)
        print(f"{res['name']:48s} {res['lag_ms']:5.0f} {res['depth_under_cm']:6.2f} {res['valgus_clean_pct']:6.1f} | "
              f"{res['rmse_hip_cm']:5.2f} {res['valgus_noisy_pct']:6.1f} {res['stand_jitter_mm']:5.2f}")

    print("\nCV Kalman q sweep (causal): q, lag ms, RMSE cm, valgus clean %, jitter")
    for q in (1.0, 5.0, 30.0, 100.0, 400.0):
        res = evaluate("", f_kalman("cv", q, r), hip_true, knee_true, noise_m, rep_bottom_idx, n_seeds=10)
        print(f"  q={q:6.0f}  lag {res['lag_ms']:.0f}  RMSE {res['rmse_hip_cm']:.2f}  valg {res['valgus_clean_pct']:.1f}  jit {res['stand_jitter_mm']:.2f}")

    # group delay of causal Butterworth at 0.5 Hz (squat fundamental)
    print("\nCausal 2nd-order Butterworth group delay at 0.5 Hz / 2 Hz:")
    for fc in (3.0, 4.0, 6.0, 8.0):
        b, a = signal.butter(2, fc / (FS_HZ / 2))
        w, gd = signal.group_delay((b, a), w=[0.5, 2.0], fs=FS_HZ)
        print(f"  fc={fc} Hz: {gd[0]*1000/FS_HZ:.0f} ms @0.5Hz, {gd[1]*1000/FS_HZ:.0f} ms @2Hz")


if __name__ == "__main__":
    main()
