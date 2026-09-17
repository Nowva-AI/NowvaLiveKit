"""Duplicate frame delivery: get_synced_frames returns the latest primary frame even if already processed."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from chain_sim import *
from biomechanics.faults.hip_position_counter import SignalRepCounter
from biomechanics.config import HipPositionCounterConfig
from biomechanics.utils.geometry import joint_angle_3_points

calib, cams = make_rig()
tempo = "bw_normal"
N_REPS = 4


def true_knee_at(times):
    b = body()
    s = depth_profile(np.asarray(times), tempo, 0.6, N_REPS)
    out = []
    for si in s:
        p = pose_world(float(si), b)
        out.append(180 - joint_angle_3_points(p[CK.LEFT_HIP], p[CK.LEFT_KNEE], p[CK.LEFT_ANKLE]))
    return np.array(out)


def simulate(cam_fps, loop_hz, mode, seed=0, cam_jitter_ms=0.0):
    """mode: 'unique' (process each camera frame once, at camera ts), 'dup_tri' (identical ts), 'dup_single' (new ts = loop time)."""
    rng = np.random.default_rng(seed)
    dur = N_REPS * (0.6 + sum(TEMPOS[tempo])) + 0.6
    cam_t = np.arange(0, dur, 1 / cam_fps) + rng.normal(0, cam_jitter_ms / 1000, size=int(np.ceil(dur * cam_fps)))[: len(np.arange(0, dur, 1 / cam_fps))]
    cam_t = np.sort(cam_t)
    b = body()
    s = depth_profile(cam_t, tempo, 0.6, N_REPS)
    fr = np.stack([pose_world(float(si), b) for si in s])
    sks = triangulate_sequence(cam_t + PERF_BASE, fr, calib, cams, seed=seed + 1, sigma_px=3.0)
    ch = Chain(phase="descending")
    counter = SignalRepCounter(HipPositionCounterConfig())
    rows = []
    if mode == "unique":
        ticks = [(t, i) for i, t in enumerate(cam_t)]
    else:
        loop_t = np.arange(0, dur, 1 / loop_hz)
        idx = np.searchsorted(cam_t, loop_t, side="right") - 1
        ticks = [(lt, int(i)) for lt, i in zip(loop_t, idx) if i >= 0]
    prev_i = None
    states = []
    for lt, i in ticks:
        sk = sks[i]
        if mode == "dup_single":
            sk = Skeleton3D(keypoints=sk.keypoints, timestamp=WALL_BASE + lt, frame_index=0)
        rec = ch.step(sk, WALL_BASE + lt)
        a = rec["pos"]
        sig = ((a[CK.LEFT_HIP, 1] + a[CK.RIGHT_HIP, 1]) / 2 - (a[CK.LEFT_ANKLE, 1] + a[CK.RIGHT_ANKLE, 1]) / 2) * 100
        rep, _ = counter.update(signal_value=sig, timestamp=WALL_BASE + lt, angles=None)
        states.append(counter.state.value)
        rec.update(loop_t=lt, cap_t=cam_t[i], dup=(i == prev_i), rep=rep is not None)
        rows.append(rec)
        prev_i = i
    cap = np.array([r["cap_t"] for r in rows])
    lt = np.array([r["loop_t"] for r in rows])
    true_now = true_knee_at(cap)
    true_future = true_knee_at(lt + 0.2)   # predictive estimator claims to predict 0.2 s ahead of NOW
    dtt = 1e-3
    true_vel = (true_knee_at(cap + dtt) - true_knee_at(cap - dtt)) / (2 * dtt)
    filt = np.array([r["filt_knee"] for r in rows]); vel = np.array([r["knee_vel"] for r in rows])
    acc = np.array([r["knee_acc"] for r in rows]); pred = np.array([r["pred_knee"] for r in rows])
    dup = np.array([r["dup"] for r in rows])
    n_bottom_entries = sum(1 for a0, a1 in zip(states, states[1:]) if a0 != "bottom" and a1 == "bottom")
    return dict(frames=len(rows), dup_frac=round(float(dup.mean()), 3), reps=int(sum(r["rep"] for r in rows)), bottom_entries=n_bottom_entries,
                filt_err_rms=round(float(np.sqrt(np.mean((filt - true_now) ** 2))), 2),
                vel_err_rms=round(float(np.sqrt(np.mean((vel - true_vel) ** 2))), 1),
                vel_err_rms_dupframes=round(float(np.sqrt(np.mean((vel[dup] - true_vel[dup]) ** 2))), 1) if dup.any() else None,
                acc_abs_p99=float(f"{np.percentile(np.abs(acc), 99):.3g}"), acc_abs_max=float(f"{np.abs(acc).max():.3g}"),
                pred_err_rms=round(float(np.sqrt(np.mean((pred - true_future) ** 2))), 2),
                pred_err_max=round(float(np.abs(pred - true_future).max()), 1),
                clamp_frames=int(sum(r["clamped_n"] > 0 for r in rows)))


res = {}
print(f"{'scenario':42s} frames dup%  reps bottomEntries filtRMS velRMS velRMS@dup  |acc|p99   |acc|max  predRMS predMax clampFrames")
for cam_fps, loop_hz, jit in ((30, 30.3, 3.0), (30, 33.0, 3.0), (24, 30, 2.0), (15, 30, 2.0)):
    for mode in ("unique", "dup_tri", "dup_single"):
        key = f"cam{cam_fps}_loop{loop_hz}_jit{jit}/{mode}"
        r = simulate(cam_fps, loop_hz, mode, seed=2, cam_jitter_ms=jit)
        res[key] = r
        print(f"{key:42s} {r['frames']:6d} {r['dup_frac']*100:4.0f} {r['reps']:5d} {r['bottom_entries']:13d} {r['filt_err_rms']:7.2f} {r['vel_err_rms']:6.1f} {str(r['vel_err_rms_dupframes']):>10s} {r['acc_abs_p99']:9.3g} {r['acc_abs_max']:9.3g} {r['pred_err_rms']:8.2f} {r['pred_err_max']:7.1f} {r['clamp_frames']:6d}")

# single isolated duplicate at mid-descent: frame-level trace
t, fr = sequence_world(tempo, fps=30, n_reps=1, stand_s=0.6)
sks = triangulate_sequence(t + PERF_BASE, fr, calib, cams, seed=3, sigma_px=3.0)
s = depth_profile(t, tempo, 0.6, 1)
mid = int(np.argmin(np.abs(s[: len(s) // 2] - 0.5)))
ref, dupc = Chain(), Chain()
print("\nisolated duplicate at mid-descent (frame k processed twice, identical ts):")
print(" step  knee_filt ref/dup    vel ref/dup        acc ref/dup            pred ref/dup   1e_dt(dup)")
trace = []
seq = list(range(mid - 2, mid + 6))
dup_seq = []
for k in seq:
    dup_seq.append(k)
    if k == mid:
        dup_seq.append(k)
ri = 0
ref_recs = {k: ref.step(sks[k], WALL_BASE + t[k]) for k in seq}
for j, k in enumerate(dup_seq):
    d = dupc.step(sks[k], WALL_BASE + t[k] + (0.0333 if j > seq.index(mid) + 0 and dup_seq[:j].count(mid) == 2 else 0))
    r = ref_recs[k]
    tag = "DUP" if (k == mid and dup_seq[:j].count(mid) == 1) else "   "
    print(f" {k:3d}{tag} {r['filt_knee']:7.2f}/{d['filt_knee']:7.2f}  {r['knee_vel']:7.1f}/{d['knee_vel']:7.1f}  {r['knee_acc']:9.3g}/{d['knee_acc']:10.3g}  {r['pred_knee']:6.1f}/{d['pred_knee']:6.1f}   {d['oneeuro_dt']}")
    trace.append(dict(k=k, dup=tag.strip(), ref=dict(filt=r['filt_knee'], vel=r['knee_vel'], acc=r['knee_acc'], pred=r['pred_knee']),
                      dup_run=dict(filt=d['filt_knee'], vel=d['knee_vel'], acc=d['knee_acc'], pred=d['pred_knee'], clamp_dt=d['clamp_raw_dt'])))
res["isolated_dup_trace"] = trace
json.dump(res, open("exp2_duplicates.json", "w"), indent=1, default=float)
