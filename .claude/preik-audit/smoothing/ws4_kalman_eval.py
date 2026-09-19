"""WS4: evaluate biomechanics.utils.keypoint_kalman.FixedLagKeypointSmoother on the smoothing audit's J objective.

Same data, seeds, lag realignment and J as exp6_adaptive.py (whose best single setting, KalmanVec lag 2, q=10,
R = clip(0.005 m/px * frame-median reprojection error, 3 mm, 8 cm), scored J_mean 3.55 / 5.93 / 9.73 @ 2/4/8 px).

Variants for the production class (C3 confidence in, lag 2):
  A  hip-centred input, conf = C3(frame-median reprojection R)  -> identical R to the baseline (implementation check)
  B  hip-centred input, conf = WS3 metric confidence per keypoint (u = max(sigma_hat, 3 px) * sqrt(tr((J^T J)^-1)))
  C  WORLD input with WS3 confidence, recentred at hips after smoothing (contract C10 architecture)
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/.claude/preik-audit/smoothing")

import ws4_harness_shim  # noqa: E402,F401  (placeholders for filters deleted by the pre-IK overhaul)
import harness as H  # noqa: E402
import smoothers as sm  # noqa: E402
import synth  # noqa: E402
from biomechanics.utils.keypoint_kalman import FixedLagKeypointSmoother  # noqa: E402
from fixedlag_fast import ProposedSmoother  # noqa: E402

SEEDS = [0, 1, 2]
SIGMAS = [2.0, 4.0, 8.0]
LAG = 2
SCALE_M = 0.02


def c3_conf(u_m: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + (u_m / SCALE_M) ** 2)


def ws3_noise(kind: str, seed: int, sigma_px: float) -> dict:
    """Replicates synth.triangulated_noise's RNG draw exactly, keeping world points and per-view residuals."""
    world, _, _, _ = H._SESSION_CACHE[kind + "None"]
    rng = np.random.default_rng(seed)
    P = synth.cameras()
    uv = synth.project(P, world)
    uv_noisy = uv + rng.normal(0.0, sigma_px, uv.shape)
    uv_noisy[..., 0] = np.round(uv_noisy[..., 0] / synth.QUANT_X_PX) * synth.QUANT_X_PX
    uv_noisy[..., 1] = np.round(uv_noisy[..., 1] / synth.QUANT_Y_PX) * synth.QUANT_Y_PX
    X = synth.dlt(P, uv_noisy)
    residual = np.linalg.norm(synth.project(P, X) - uv_noisy, axis=-1)  # (T,K,V)
    Xh = np.concatenate([X, np.ones(X.shape[:-1] + (1,))], axis=-1)
    proj = np.einsum("vij,tkj->tkvi", P, Xh)
    w = proj[..., 2]
    u_px, v_px = proj[..., 0] / w, proj[..., 1] / w
    ju = (P[None, None, :, 0, :3] - u_px[..., None] * P[None, None, :, 2, :3]) / w[..., None]
    jv = (P[None, None, :, 1, :3] - v_px[..., None] * P[None, None, :, 2, :3]) / w[..., None]
    jtj = np.einsum("tkvi,tkvj->tkij", ju, ju) + np.einsum("tkvi,tkvj->tkij", jv, jv)
    meters_per_px = np.sqrt(np.trace(np.linalg.inv(jtj), axis1=-2, axis2=-1))
    views = P.shape[0]
    sigma_hat = np.sqrt(np.sum(residual ** 2, axis=-1) / (2 * views - 3))
    u = np.maximum(sigma_hat, 3.0) * meters_per_px
    return dict(world=X, conf=c3_conf(u), u=u, sigma_hat=sigma_hat, meters_per_px=meters_per_px)


def run_production(noisy: np.ndarray, conf: np.ndarray, ts: np.ndarray, world_mode: bool = False):
    smoother = FixedLagKeypointSmoother(lag_frames=LAG)
    out = np.empty_like(noisy)
    rejections = 0
    for i in range(len(noisy)):
        result = smoother.update(noisy[i], conf[i], float(ts[i]))
        out[i] = result.lagged_points
        rejections += int(result.gate_rejected.sum())
    if world_mode:
        out = synth.recenter(out)
    return out, rejections


def main() -> None:
    table: dict[str, list] = {}
    rejections: dict[str, list] = {}
    detail: dict[str, dict] = {}
    for sigma in SIGMAS:
        scores: dict[str, dict[str, list]] = {}
        rej: dict[str, int] = {}
        for kind in ("smooth", "hard"):
            for seed in SEEDS:
                d = H.make_data(seed, sigma, kind=kind)
                ws3 = ws3_noise(kind, seed, sigma)
                assert np.allclose(synth.recenter(ws3["world"]), d["noisy"]), "RNG replication mismatch"
                noisy, ts = d["noisy"], d["ts"]
                runs = {}
                base = sm.KalmanVec(2, 10.0, 0.01, LAG, None)
                base.reproj_gain, base.reproj_mode = 0.005, "frame_median"
                runs["baseline KalmanVec (exp6 best)"] = (sm.run_causal(base, noisy, ts, d["reproj"]), 0)
                prop = ProposedSmoother()
                prop.reset()
                runs["fixedlag_fast.ProposedSmoother"] = (
                    np.stack([prop.step(noisy[i], ts[i], (d["conf"][i], d["reproj"][i])) for i in range(len(noisy))]), 0)
                r_frame = np.clip(0.005 * np.median(d["reproj"], axis=1), 0.003, 0.08)
                conf_a = np.repeat(c3_conf(r_frame)[:, None], noisy.shape[1], axis=1)
                runs["A prod, hip-centred, frame-median R"] = run_production(noisy, conf_a, ts)
                runs["B prod, hip-centred, WS3 conf"] = run_production(noisy, ws3["conf"], ts)
                runs["C prod, WORLD + WS3 conf, recentre after"] = run_production(ws3["world"], ws3["conf"], ts, True)
                for name, (out, n_rej) in runs.items():
                    m = H.fast_eval(out, d, LAG)
                    scores.setdefault(name, {}).setdefault(kind, []).append(m)
                    rej[name] = rej.get(name, 0) + n_rej
        for name, per_kind in scores.items():
            smooth, hard = H.summarize(per_kind["smooth"]), H.summarize(per_kind["hard"])
            table.setdefault(name, []).append(round((smooth["J"] + hard["J"]) / 2, 2))
            rejections.setdefault(name, []).append(rej[name])
            detail.setdefault(name, {})[sigma] = dict(
                knee_rms_moving=round((smooth["knee_rms_moving_deg"] + hard["knee_rms_moving_deg"]) / 2, 2),
                depth_abs=round((smooth["depth_abs_mean_deg"] + hard["depth_abs_mean_deg"]) / 2, 2),
                valgus_abs=round((smooth["valgus_abs_mean_deg"] + hard["valgus_abs_mean_deg"]) / 2, 2),
                hard_valgus_signed=hard["valgus_peak_err_mean_deg"],
                standing_std=round((smooth["knee_std_standing_deg"] + hard["knee_std_standing_deg"]) / 2, 2),
                jitter_mm=round(smooth["standing_jitter_mm_per_frame"], 2))
        print(f"sigma {sigma:g} px done", flush=True)

    print("\nJ_mean @ 2 / 4 / 8 px (lower is better; reference ProposedSmoother setting 3.55 / 5.93 / 9.73)")
    for name, js in table.items():
        print(f"  {name:44s} {' / '.join(f'{j:.2f}' for j in js)}   gate rejections {rejections[name]}")
    print("\nComponents (mean of smooth+hard): knee RMS moving / |depth| / |valgus peak| / standing std deg; "
          "hard valgus signed; standing jitter mm/frame (smooth)")
    for name, per_sigma in detail.items():
        print(f"  {name}")
        for sigma, m in per_sigma.items():
            print(f"    {sigma:g} px: {m}")

    d = H.make_data(0, 4.0)
    smoother = FixedLagKeypointSmoother(lag_frames=LAG)
    t0 = time.perf_counter()
    for i in range(len(d["noisy"])):
        smoother.update(d["noisy"][i], d["conf"][i], float(d["ts"][i]))
    print(f"\ncost N=19 lag 2 on harness data: {(time.perf_counter() - t0) / len(d['noisy']) * 1e6:.1f} us/frame")


if __name__ == "__main__":
    main()
