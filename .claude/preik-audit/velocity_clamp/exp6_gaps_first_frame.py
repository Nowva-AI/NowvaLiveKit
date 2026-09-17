"""First frame after reset (conf-0 kpts at hip-centre origin) and skipped frames (sync failures) through blend+clamp."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from synth_vc import *
from biomechanics.utils.confidence_blend import ConfidenceBlender
from biomechanics.utils.velocity_clamp import VelocityClamp

calib, cams = make_rig()
res = {}
# (a) at the frame the readiness gate latches, toe + wrist untriangulated (conf 0 -> triangulator leaves them at origin)
t, fr = sequence_world("bw_normal", fps=30, stand_s=1.5)
def occl(uv, c):
    c[[CK.LEFT_FOOT_INDEX, CK.LEFT_WRIST]] = 0.0; return uv, c
k0 = 5
sks = triangulate_sequence(t, fr, calib, cams, seed=4, sigma_px=3.0, corrupt_at={k0: {"0": occl, "1": occl, "2": occl}})
truth = triangulate_sequence(t, fr, calib, cams, seed=4, sigma_px=3.0)
print("seed frame conf toe/wrist:", confs(sks[k0])[[CK.LEFT_FOOT_INDEX, CK.LEFT_WRIST]], "pos toe:", sks[k0].to_numpy()[CK.LEFT_FOOT_INDEX])
for variant in ("blend", "blend+clamp"):
    bl, cl = ConfidenceBlender(0.1, 0.9), VelocityClamp(2.5, 30)
    errs = {"toe": [], "wrist": []}
    for k in range(k0, k0 + 20):          # filters reset -> first processed frame is k0
        x = bl.blend(sks[k])
        if variant == "blend+clamp":
            x = cl.clamp(x)
        a, b = x.to_numpy(), truth[k].to_numpy()
        errs["toe"].append(round(float(np.linalg.norm(a[CK.LEFT_FOOT_INDEX] - b[CK.LEFT_FOOT_INDEX]) * 100), 1))
        errs["wrist"].append(round(float(np.linalg.norm(a[CK.LEFT_WRIST] - b[CK.LEFT_WRIST]) * 100), 1))
    n_toe = next((i for i, e in enumerate(errs["toe"]) if i > 0 and e < 3.0), None)
    res[f"first_frame_conf0/{variant}"] = dict(err_toe_cm=errs["toe"], err_wrist_cm=errs["wrist"], frames_until_toe_lt3cm=n_toe)
    print(f"{variant:12s} toe err cm {errs['toe'][:14]} -> <3 cm after {n_toe} frames")
    print(f"{'':12s} wrist err  {errs['wrist'][:14]}")

# (b) sync failures: drop 1-of-2 / random 35% frames (early return, no hold) during bw_fast; dt-aware clamp vs frame-count clamp
for drop_pct in (0.35, 0.5):
    t, fr = sequence_world("bw_fast", fps=30, n_reps=3, stand_s=0.6)
    sks = triangulate_sequence(t, fr, calib, cams, seed=9, sigma_px=3.0)
    rng = np.random.default_rng(1)
    keep = [sk for sk in sks if rng.random() > drop_pct]
    for name, ts_mode in (("dt_aware(actual)", "real"), ("if_dt_were_fallback", "zero")):
        cl = VelocityClamp(2.5, 30)
        n = 0
        for sk in keep:
            x = sk if ts_mode == "real" else Skeleton3D(keypoints=sk.keypoints, timestamp=0.0, frame_index=0)
            y = cl.clamp(x)
            n += int(np.any(np.linalg.norm(y.to_numpy() - x.to_numpy(), axis=1) > 1e-9))
        res[f"sync_drop{drop_pct}/{name}"] = dict(frames=len(keep), clamped_frames=n)
        print(f"sync drops {drop_pct:.0%} bw_fast {name:22s}: clamped {n}/{len(keep)} frames")
json.dump(res, open("exp6_gaps_first_frame.json", "w"), indent=1)
