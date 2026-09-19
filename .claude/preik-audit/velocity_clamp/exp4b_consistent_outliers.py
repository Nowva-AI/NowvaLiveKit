"""Outliers that the triangulator does NOT flag: 2-view swap, 2-view-only keypoint with one bad view."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from exp4_outliers import run_variant, metrics, replace, VARIANTS, calib, cams, KNEE
from synth_vc import *

legs = [(CK.LEFT_KNEE, CK.RIGHT_KNEE), (CK.LEFT_ANKLE, CK.RIGHT_ANKLE), (CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX)]
def swap(uv, c):
    for l, r in legs:
        uv[[l, r]] = uv[[r, l]]
    return uv, c
def occlude_knee(uv, c):
    c[KNEE] = 0.0; return uv, c
res = {}
for tempo in ("bw_normal",):
    t, fr = sequence_world(tempo, fps=30, stand_s=0.6)
    base = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0)
    s = depth_profile(t, tempo, 0.6, 1)
    mid = int(np.argmin(np.abs(s[: len(s) // 2] - 0.5)))
    for where, k0 in (("standing", 8), ("mid_descent", mid)):
        lo, hi = k0 - 1, k0 + 25
        for n in (1, 10):
            sks = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at={k: {"1": swap, "2": swap} for k in range(k0, k0 + n)})
            for v in VARIANTS:
                m = metrics(run_variant(sks, v), base, KNEE, lo, hi)
                m["tri_conf"] = round(float(confs(sks[k0])[KNEE]), 3)
                m["raw3d_jump_cm"] = round(float(np.linalg.norm(sks[k0].to_numpy()[KNEE] - base[k0].to_numpy()[KNEE]) * 100), 1)
                res[f"{tempo}/{where}/S3b_legswap_views1+2_{n}f/{v}"] = m
        # knee seen by only 2 views (view0 occluded, all frames), then 40 px outlier in view1 for one frame
        for px in (20, 40, 80):
            def bad(uv, c, px=px):
                uv[KNEE] += np.array([px, 0.0]); return uv, c
            corrupt = {k: {"0": occlude_knee} for k in range(len(t))}
            corrupt[k0] = {"0": occlude_knee, "1": bad}
            base2 = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at={k: {"0": occlude_knee} for k in range(len(t))})
            sks = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at=corrupt)
            for v in VARIANTS:
                m = metrics(run_variant(sks, v), base2, KNEE, lo, hi)
                m["tri_conf"] = round(float(confs(sks[k0])[KNEE]), 3)
                m["raw3d_jump_cm"] = round(float(np.linalg.norm(sks[k0].to_numpy()[KNEE] - base2[k0].to_numpy()[KNEE]) * 100), 1)
                res[f"{tempo}/{where}/S2b_2viewonly_view1_{px}px/{v}"] = m
json.dump(res, open("exp4b.json", "w"), indent=1)
last = None
for key, m in res.items():
    scen = key.rsplit("/", 1)[0]
    if scen != last:
        print(f"\n{scen} tri_conf={m['tri_conf']} raw3d_jump={m['raw3d_jump_cm']}cm"); last = scen
    print(f"   {key.rsplit('/',1)[1]:12s} kneeErr max {m['max_err_cm']:5.1f} cm, >5cm {m['frames_err_gt5cm']:2d}f | geomKnee max {m['max_knee_deg']:5.1f}, >5deg {m['frames_knee_gt5deg']:2d}f | IKknee {m['ik_knee_max_deg']:5.1f} | trace {m['err_trace'][:15]}")
