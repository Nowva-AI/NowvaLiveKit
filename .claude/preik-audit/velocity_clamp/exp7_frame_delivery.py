"""Simulate MultiCameraCapture.get_synced_frames + pipeline_process pacing: duplicates, sync failures, skips, view time offsets."""
import json
import numpy as np

BUF = 5
MAX_SYNC = 0.015


def sim(fps=(30.0, 30.0, 30.0), read_lat_ms=(8.0, 8.0, 8.0), read_jit_ms=2.0, proc_ms=15.0, proc_jit_ms=3.0,
        overhead_ms=2.0, target_fps=30.0, dur_s=600.0, seed=0, drop_prob=0.0):
    rng = np.random.default_rng(seed)
    cams = []
    fps = [f * d for f, d in zip(fps, (1.0, 0.9993, 1.0011))]  # crystal tolerance ~0.1% -> phases drift through all offsets
    for f, lat in zip(fps, read_lat_ms):
        n = int(dur_s * f) + 10
        cap = rng.uniform(0, 1 / f) + np.arange(n) / f
        keep = rng.random(n) >= drop_prob
        cap = cap[keep]
        ts = cap + (lat + rng.normal(0, read_jit_ms, len(cap)).clip(-lat + 0.5, None)) / 1000.0  # perf_counter stamped after read()
        order = np.argsort(ts)
        cams.append((cap[order], ts[order]))
    t = 0.2
    last_primary_idx = None
    n_calls = n_none = n_dup = 0
    used_primary = []
    offsets = []
    while t < dur_s - 0.2:
        n_calls += 1
        cap0, ts0 = cams[0]
        i0 = np.searchsorted(ts0, t, side="right") - 1
        ok = True
        ref_ts = ts0[i0]
        view_caps = [cap0[i0]]
        for cap_k, ts_k in cams[1:]:
            ik = np.searchsorted(ts_k, t, side="right") - 1
            lo = max(0, ik - BUF + 1)
            cand = np.arange(lo, ik + 1)
            d = np.abs(ts_k[cand] - ref_ts)
            j = cand[np.argmin(d)]
            if d.min() > MAX_SYNC:
                ok = False
                break
            view_caps.append(cap_k[j])
        if not ok:
            n_none += 1
            latency = 0.0  # frame None -> early return; latency_ms has capture=0,pose ~ tiny
            t += max(1 / target_fps - latency, 0) + overhead_ms / 1000
            continue
        if last_primary_idx == i0:
            n_dup += 1
        used_primary.append(i0)
        offsets.append(max(view_caps) - min(view_caps))
        last_primary_idx = i0
        proc = max(proc_ms + rng.normal(0, proc_jit_ms), 1.0) / 1000
        sleep = max(1 / target_fps - proc, 0.0)
        t += proc + sleep + overhead_ms / 1000
    used = np.array(used_primary)
    if len(used) < 2:
        return dict(calls=n_calls, none_pct=100.0, dup_pct=0.0, skipped_primary_pct=100.0, view_capture_spread_ms_p50=float("nan"), view_capture_spread_ms_p95=float("nan"))
    uniq = np.unique(used)
    span = used.max() - used.min() + 1
    offsets = np.array(offsets) * 1000
    return dict(calls=n_calls, none_pct=round(100 * n_none / n_calls, 1), dup_pct=round(100 * n_dup / max(1, len(used)), 1),
                skipped_primary_pct=round(100 * (1 - len(uniq) / span), 1),
                view_capture_spread_ms_p50=round(float(np.percentile(offsets, 50)), 1),
                view_capture_spread_ms_p95=round(float(np.percentile(offsets, 95)), 1))


res = {}
cases = {
    "headless_overhead2ms": dict(overhead_ms=2.0),
    "display_overhead8ms": dict(overhead_ms=8.0),
    "no_overhead_proc_jitter6": dict(overhead_ms=0.0, proc_jit_ms=6.0),
    "loop_polls_after_proc_0ms_sleep": dict(overhead_ms=0.0, proc_ms=34.0, proc_jit_ms=2.0),
    "cam_lowlight_24fps": dict(fps=(24.0, 24.0, 24.0), overhead_ms=2.0),
    "cam_lowlight_15fps": dict(fps=(15.0, 15.0, 15.0), overhead_ms=2.0),
    "usb_drops_5pct": dict(drop_prob=0.05, overhead_ms=2.0),
    "jetson_proc45ms": dict(proc_ms=45.0, proc_jit_ms=5.0, overhead_ms=2.0),
    "cams_unequal_latency": dict(read_lat_ms=(6.0, 14.0, 22.0), overhead_ms=2.0),
}
print(f"{'case':28s} {'seed':>4s} calls none%  dup% skip%  viewSpread p50/p95 ms")
for name, kw in cases.items():
    for seed in (0, 1, 2):
        r = sim(seed=seed, **kw)
        res[f"{name}/seed{seed}"] = r
        print(f"{name:28s} {seed:4d} {r['calls']:5d} {r['none_pct']:5.1f} {r['dup_pct']:5.1f} {r['skipped_primary_pct']:5.1f}  {r['view_capture_spread_ms_p50']:6.1f}/{r['view_capture_spread_ms_p95']:.1f}")
json.dump(res, open("exp7_frame_delivery.json", "w"), indent=1)
