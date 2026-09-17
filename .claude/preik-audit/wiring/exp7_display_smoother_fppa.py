"""E7: single-camera path — 'display-only' Skeleton2DSmoother output feeds SingleCameraValgusEstimator (pipeline.py:521-522, 639)."""
import math, sys
import numpy as np
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from biomechanics.config import load_pipeline_config
from biomechanics.kinematics.valgus import SingleCameraValgusEstimator
from biomechanics.utils.position_filter import Skeleton2DSmoother
from biomechanics.utils.types import Skeleton2D, CocoKeypoints as CK

cfg = load_pipeline_config().display_filter
smoother = Skeleton2DSmoother(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
est = SingleCameraValgusEstimator()
FPS = 30.0
def pose(depth, cave_px):
    pts = np.zeros((19, 3)); pts[:, 2] = 0.9
    for sign, (h, k, a) in ((1, (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE)), (-1, (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE))):
        x = 640 + sign * 60
        pts[a, :2] = [640 + sign * 80, 650]
        pts[h, :2] = [x, 330 + 170 * depth]
        pts[k, :2] = [640 + sign * (75 - cave_px * depth), 490 + 40 * depth]
    for i in (CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER):
        pts[i, :2] = [640 + (60 if i == CK.LEFT_SHOULDER else -60), 120 + 150 * depth]
    return pts
profile = [0.0]*15 + [0.5-0.5*math.cos(math.pi*i/24) for i in range(24)] + [1.0]*9 + [0.5+0.5*math.cos(math.pi*i/24) for i in range(24)] + [0.0]*15
raw_v, sm_v = [], []
for i, d in enumerate(profile):
    s = Skeleton2D.from_numpy(pose(d, 45.0), timestamp=i / FPS, frame_index=i)
    raw_v.append(est.estimate(s).valgus_l)
    sm_v.append(est.estimate(smoother.smooth(s)).valgus_l)
raw_v, sm_v = np.array(raw_v), np.array(sm_v)
print(f"display_filter cfg min_cutoff={cfg.min_cutoff} beta={cfg.beta} enabled={load_pipeline_config().display_filter.enabled}")
print(f"peak FPPA valgus raw={raw_v.max():.2f}  display-smoothed={sm_v.max():.2f}  (diff {raw_v.max()-sm_v.max():+.2f})")
print(f"max |raw - smoothed| over rep = {np.abs(raw_v-sm_v).max():.2f} deg; peak frame raw={raw_v.argmax()} smoothed={sm_v.argmax()}")
