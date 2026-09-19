"""Exp 7b: production sync policy vs proposed policy, averaged over random camera phase draws.
Proposed: reference = newest primary frame not yet processed whose timestamp <= min over cams of their newest stamp
(+ small margin), nearest match per cam, tolerance 20 ms, keep any cam within tolerance (>=2 views triangulate)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from exp7_sync import get_synced

OUT = Path(__file__).parent
RNG = np.random.default_rng(99)


def get_synced_v2(stamps, now, last_ref, tol_s=0.020):
    newest = []
    for st in stamps:
        n = int(np.searchsorted(st, now, side="right"))
        if n == 0:
            return None
        newest.append(st[n - 1])
    horizon = min(newest) + 0.005
    p = stamps[0]
    n0 = int(np.searchsorted(p, min(now, horizon), side="right"))
    if n0 == 0:
        return None
    ref = float(p[n0 - 1])
    if last_ref is not None and ref <= last_ref:
        return "dup"
    used = 1
    deltas = []
    for st in stamps[1:]:
        n = int(np.searchsorted(st, now, side="right"))
        w = st[max(0, n - 5):n]
        d = float(np.min(np.abs(w - ref)))
        if d <= tol_s:
            used += 1
            deltas.append(d * 1000)
    return (ref, used, deltas) if used >= 2 else None


def run(policy, fps, proc_ms, draws=10, dur=120.0):
    agg = {"fail": [], "skip": [], "dup": [], "two_view_frac": [], "p95_offset_ms": []}
    for _ in range(draws):
        stamps = []
        for f in fps:
            n = int(dur * f)
            s = RNG.uniform(0, 1 / 30) + np.arange(n) / f + (25 + RNG.normal(0, 3, n)) / 1000
            stamps.append(np.sort(s))
        t, last_ref, calls, fail, dup, two, offs, ids = 0.3, None, 0, 0, 0, 0, [], []
        while t < dur - 0.3:
            calls += 1
            if policy == "prod":
                r = get_synced(stamps, t, 0.015)
                if r is None:
                    fail += 1
                else:
                    if last_ref is not None and r[0] == last_ref:
                        dup += 1
                    else:
                        ids.append(r[0])
                    last_ref = r[0]
                    offs += [abs(d) * 1000 for d in r[2]]
            else:
                r = get_synced_v2(stamps, t, last_ref)
                if r is None:
                    fail += 1
                elif r == "dup":
                    dup += 1
                else:
                    ids.append(r[0]); last_ref = r[0]; offs += r[2]
                    two += r[1] == 2
            t += max(1 / 30, max(proc_ms + RNG.normal(0, 2), 1) / 1000)
        ids = np.array(ids)
        n_primary = int(np.sum((stamps[0] > ids[0]) & (stamps[0] <= ids[-1])))
        agg["fail"].append(fail / calls); agg["dup"].append(dup / calls)
        agg["skip"].append(1 - len(ids) / max(n_primary, 1)); agg["two_view_frac"].append(two / max(len(ids), 1))
        agg["p95_offset_ms"].append(float(np.percentile(offs, 95)))
    return {k: [round(float(np.mean(v)), 3), round(float(np.min(v)), 3), round(float(np.max(v)), 3)] for k, v in agg.items()}


res = {}
for name, fps, proc in (("mac_14ms", [30.0, 29.97, 30.03], 14), ("jetson_45ms", [30.0, 29.97, 30.03], 45)):
    res[f"{name}_prod"] = run("prod", fps, proc)
    res[f"{name}_proposed"] = run("v2", fps, proc)
res["_legend"] = "each value = [mean, min, max] over 10 random camera-phase draws; fail = call returned no skeleton; skip = primary frames never processed"
(OUT / "exp7b_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
