"""5th dropout-hold frame has confidence 0 -> IK returns 0.0 for every angle -> filtered trunk/knee/hip collapse."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from chain_sim import *

calib, cams = make_rig()
t, fr = sequence_world("bw_normal", fps=30, n_reps=1, stand_s=1.0)
tri = triangulate_sequence(t + PERF_BASE, fr, calib, cams, seed=3, sigma_px=3.0)
res = {}
for phase, k0 in (("idle", 15), ("descending", 40)):
    for label, n in (("hold5", 5), ("hold6_exhausted", 6)):
        sk = list(tri)
        for k in range(k0, k0 + n):
            sk[k] = None
        ch = Chain(phase=phase)
        rows = []
        for k, (s_, tt) in enumerate(zip(sk, t)):
            angles_holder = {}
            r = ch.step(s_, WALL_BASE + tt, hold_clock="perf")   # consistent clock: isolate the zero-confidence effect
            if r is None:
                rows.append(None); continue
            rows.append(r)
        # re-run collecting trunk angle via a patched chain
        ch2 = Chain(phase=phase); trunk = []; conf_min = []
        for k, (s_, tt) in enumerate(zip(sk, t)):
            if s_ is None:
                if ch2.last_valid is None or ch2.hold >= MAX_HOLD:
                    trunk.append(None); conf_min.append(None); continue
                ch2.hold += 1
                decay = 1.0 - ch2.hold / MAX_HOLD
                s_ = Skeleton3D.from_numpy(ch2.last_valid.to_numpy(), confidences=[kp.confidence * decay for kp in ch2.last_valid.keypoints], timestamp=PERF_BASE + tt)
            else:
                ch2.hold = 0; ch2.last_valid = s_
            x = ch2.clamp.clamp(ch2.blender.blend(s_))
            ang = ch2.ik.solve(x)
            fa = ch2.angle_filter.filter_angles(ang)
            trunk.append((round(ang.trunk_flexion, 1), round(fa.trunk_flexion, 1)))
            conf_min.append(round(float(min(kp.confidence for kp in x.keypoints)), 3))
        sl = slice(k0 - 1, k0 + n + 4)
        out = dict(conf_min=conf_min[sl], raw_filtered_trunk=trunk[sl],
                   filt_knee=[None if r is None else round(r["filt_knee"], 1) for r in rows[sl]],
                   pred_knee=[None if r is None else round(r["pred_knee"], 1) for r in rows[sl]],
                   knee_vel=[None if r is None else round(r["knee_vel"], 1) for r in rows[sl]])
        res[f"{phase}/{label}"] = out
        print(f"\n{phase}/{label} (frames {k0}..{k0+n-1} dropped)")
        for key, v in out.items():
            print(f"  {key:18s} {v}")
json.dump(res, open("exp1b_hold_zero_conf.json", "w"), indent=1)
