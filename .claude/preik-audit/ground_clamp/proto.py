"""Prototype replacement for GroundClamp: world-frame foot contact model.

Runs on WORLD coordinates (before hip re-centring). Per foot keypoint (ankle, big toe, heel if present):
detects contact from windowed stationarity, holds a zero-velocity anchor while contact holds (cumulative
mean -> noise / sqrt(N)), keeps a session-fixed rest reference per foot placement, and exposes floor-referenced
signals (ankle/heel rise, stance width, hip height above ankle rest level). Never couples left and right,
never moves knees, never forces symmetry. Cheap numpy, causal, O(1) per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim import CK

WINDOW_FRAMES = 6            # 0.2 s at 30 fps
ENTER_MOTION_M = 0.02        # |mean(window) - mean(previous window)| below this -> stationary
ENTER_CHECKS = 2             # consecutive stationary checks before contact
BREAK_RISE_M = 0.015         # window-mean rise above anchor that breaks contact (heel rise / lift)
BREAK_SLIDE_M = 0.025        # horizontal window-mean displacement that breaks contact
BREAK_DROP_M = 0.03          # window-mean drop below anchor (outlier / mis-track) that breaks contact
UPDATE_GATE_M = 0.01         # only average observations into the anchor while the window residual is small
MERGE_M = 0.03               # re-contact within this distance of the previous floor anchor reuses it
NEW_PLACEMENT_M = 0.05       # horizontal move beyond this = the foot was re-placed (new floor anchor)
FOOT_KPTS_SIDES = (
    (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL),
    (CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL),
)


@dataclass
class _KeypointContact:
    history: list = field(default_factory=list)
    in_contact: bool = False
    floor_contact: bool = False          # current hold is on the floor (not a raised hold)
    stationary_checks: int = 0
    anchor: np.ndarray | None = None
    anchor_count: int = 0
    floor: np.ndarray | None = None      # floor rest anchor of the current placement, kept after breaks
    floor_count: int = 0

    def update(self, obs: np.ndarray) -> np.ndarray:
        self.history.append(obs)
        if len(self.history) > 2 * WINDOW_FRAMES:
            self.history.pop(0)
        recent = np.mean(self.history[-WINDOW_FRAMES:], axis=0)

        if self.in_contact:
            residual = recent - self.anchor
            rise = -residual[1]  # Y-down: up is -y
            slide = float(np.hypot(residual[0], residual[2]))
            if rise > BREAK_RISE_M or slide > BREAK_SLIDE_M or -rise > BREAK_DROP_M:
                self.in_contact = False
                self.stationary_checks = 0
                if self.floor_contact:
                    self.floor, self.floor_count = self.anchor, self.anchor_count
            else:
                if float(np.linalg.norm(residual)) < UPDATE_GATE_M:
                    self.anchor_count += 1
                    self.anchor = self.anchor + (obs - self.anchor) / self.anchor_count
                return self.anchor

        if len(self.history) >= 2 * WINDOW_FRAMES:
            previous = np.mean(self.history[:WINDOW_FRAMES], axis=0)
            if float(np.linalg.norm(recent - previous)) < ENTER_MOTION_M:
                self.stationary_checks += 1
            else:
                self.stationary_checks = 0
            if self.stationary_checks >= ENTER_CHECKS:
                window_mean = np.mean(self.history, axis=0)
                self.in_contact = True
                same_placement = self.floor is not None and float(
                    np.hypot(*(window_mean - self.floor)[[0, 2]])) < NEW_PLACEMENT_M
                raised = same_placement and (self.floor[1] - window_mean[1]) > BREAK_RISE_M
                self.floor_contact = not raised
                if same_placement and not raised and float(np.linalg.norm(window_mean - self.floor)) < MERGE_M:
                    self.anchor, self.anchor_count = self.floor, self.floor_count
                else:
                    self.anchor, self.anchor_count = window_mean, len(self.history)
                return self.anchor
        return obs

    def floor_reference(self) -> np.ndarray | None:
        return self.anchor if (self.in_contact and self.floor_contact) else self.floor


class FootContactModel:
    def __init__(self, num_keypoints: int) -> None:
        self._num_keypoints = num_keypoints
        self._contacts: dict[int, _KeypointContact] = {}
        for side in FOOT_KPTS_SIDES:
            for idx in side:
                if idx < num_keypoints:
                    self._contacts[idx] = _KeypointContact()

    def process(self, world: np.ndarray) -> tuple[np.ndarray, dict]:
        out = world.copy()
        for idx, contact in self._contacts.items():
            out[idx] = contact.update(world[idx])

        state: dict = {}
        for side_name, (ankle, toe, heel) in zip(("l", "r"), FOOT_KPTS_SIDES):
            ankle_c = self._contacts[ankle]
            rest = ankle_c.floor_reference()
            recent = np.mean(ankle_c.history[-4:], axis=0)
            state[f"ankle_rise_{side_name}"] = float(rest[1] - recent[1]) if rest is not None else 0.0
            state[f"ankle_contact_{side_name}"] = ankle_c.in_contact
            state[f"toe_contact_{side_name}"] = self._contacts[toe].in_contact
            if heel in self._contacts:
                heel_c = self._contacts[heel]
                heel_rest = heel_c.floor_reference()
                heel_recent = np.mean(heel_c.history[-4:], axis=0)
                state[f"heel_rise_{side_name}"] = (
                    float(heel_rest[1] - heel_recent[1]) if heel_rest is not None else 0.0
                )
        rest_l = self._contacts[CK.LEFT_ANKLE].floor_reference()
        rest_r = self._contacts[CK.RIGHT_ANKLE].floor_reference()
        if rest_l is not None and rest_r is not None:
            hip_center = (world[CK.LEFT_HIP] + world[CK.RIGHT_HIP]) / 2.0
            state["hip_above_ankle_rest_m"] = float((rest_l[1] + rest_r[1]) / 2.0 - hip_center[1])
        return out, state


OUTLIER_MIN_M = 0.08         # per-frame residual beyond max(this, 6 sigma) = pose-model gross failure, ignored
MOVE_MIN_M = 0.03            # window-mean displacement that counts as the foot moving (floor for the noise-scaled threshold)
RISE_MIN_M = 0.02            # window-mean ankle rise that counts as heel rise (floor for the noise-scaled threshold)
SIGMA_INIT_M = 0.015
SIGMA_EMA = 0.02
LOST_FRAMES = 30             # all foot keypoints outliers this long -> re-acquire


class _RobustKeypoint:
    def __init__(self) -> None:
        self.anchor: np.ndarray | None = None
        self.count = 0
        self.floor: np.ndarray | None = None
        self.sigma = SIGMA_INIT_M
        self.residuals: list = []      # last WINDOW_FRAMES residuals, None for outliers
        self.raw: list = []            # last 2*WINDOW_FRAMES raw observations (released-state stationarity)
        self.last_inlier: np.ndarray | None = None
        self.last_out: np.ndarray | None = None
        self.rejected = 0

    def outlier_gate(self) -> float:
        return max(OUTLIER_MIN_M, 6.0 * self.sigma)

    def window_mean(self) -> np.ndarray | None:
        inliers = [r for r in self.residuals if r is not None]
        if len(inliers) < WINDOW_FRAMES // 2:
            return None
        return np.mean(inliers, axis=0)


class FootContactModelRobust:
    """Foot-level contact hold that ignores gross pose-model outliers and never couples the two feet."""

    def __init__(self, num_keypoints: int) -> None:
        self._sides = []
        for ankle, toe, heel in FOOT_KPTS_SIDES:
            idx = {"ankle": ankle, "toe": toe}
            if heel < num_keypoints:
                idx["heel"] = heel
            self._sides.append({"idx": idx, "kp": {name: _RobustKeypoint() for name in idx},
                                "planted": False, "heel_mode": False, "lost": 0, "still": 0})

    def _thresholds(self, kp: _RobustKeypoint) -> tuple[float, float]:
        root_k = math.sqrt(WINDOW_FRAMES)
        return max(MOVE_MIN_M, 4.0 * kp.sigma / root_k), max(RISE_MIN_M, 3.0 * kp.sigma / root_k)

    def process(self, world: np.ndarray) -> tuple[np.ndarray, dict]:
        out = world.copy()
        state: dict = {}
        for side_name, side in zip(("l", "r"), self._sides):
            kps = side["kp"]
            for name, k in side["idx"].items():
                kp = kps[name]
                kp.raw.append(world[k])
                if len(kp.raw) > 2 * WINDOW_FRAMES:
                    kp.raw.pop(0)
            if side["planted"]:
                self._planted_step(side, world, out)
            else:
                self._released_step(side, world, out)
            ankle = kps["ankle"]
            mean_res = ankle.window_mean()
            if ankle.floor is not None and ankle.last_inlier is not None:
                recent = ankle.anchor + mean_res if (side["planted"] and mean_res is not None) else ankle.last_inlier
                state[f"ankle_rise_{side_name}"] = float(ankle.floor[1] - recent[1])
            else:
                state[f"ankle_rise_{side_name}"] = 0.0
            state[f"ankle_contact_{side_name}"] = side["planted"] and not side["heel_mode"]
            state[f"toe_contact_{side_name}"] = side["planted"]
            state[f"sigma_{side_name}"] = ankle.sigma
        return out, state

    def _planted_step(self, side: dict, world: np.ndarray, out: np.ndarray) -> None:
        kps = side["kp"]
        all_outliers = True
        for name, k in side["idx"].items():
            kp = kps[name]
            residual = world[k] - kp.anchor
            dist = float(np.linalg.norm(residual))
            inlier = dist < kp.outlier_gate()
            kp.residuals.append(residual if inlier else None)
            if len(kp.residuals) > WINDOW_FRAMES:
                kp.residuals.pop(0)
            if inlier:
                all_outliers = False
                kp.last_inlier = world[k]
                if dist < 4.0 * kp.sigma:
                    kp.sigma = math.sqrt((1 - SIGMA_EMA) * kp.sigma ** 2 + SIGMA_EMA * dist ** 2 / 3.0)
        side["lost"] = side["lost"] + 1 if all_outliers else 0

        toe = kps["toe"]
        ankle = kps["ankle"]
        move_thr, rise_thr = self._thresholds(ankle)
        toe_mean = toe.window_mean()
        ankle_mean = ankle.window_mean()
        toe_moved = toe_mean is not None and float(np.linalg.norm(toe_mean)) > move_thr
        ankle_rise = ankle_mean is not None and -ankle_mean[1] > rise_thr
        ankle_slide = ankle_mean is not None and float(np.hypot(ankle_mean[0], ankle_mean[2])) > move_thr

        if (toe_moved and (ankle_rise or ankle_slide)) or side["lost"] > LOST_FRAMES:
            side["planted"] = False
            side["heel_mode"] = False
            side["still"] = 0
            for kp in kps.values():
                if side.get("floor_contact", True):
                    kp.floor = kp.anchor
                kp.residuals.clear()
            for name, k in side["idx"].items():
                kp = kps[name]
                out[k] = kp.last_inlier if kp.last_inlier is not None else kp.anchor
                kp.last_out = out[k]
            return

        side["heel_mode"] = bool(ankle_rise and not toe_moved)
        for name, k in side["idx"].items():
            kp = kps[name]
            mean_res = kp.window_mean()
            follows_heel = side["heel_mode"] and name in ("ankle", "heel")
            if follows_heel:
                out[k] = kp.last_inlier if kp.last_inlier is not None else kp.anchor
                continue
            if mean_res is not None and float(np.linalg.norm(mean_res)) < 1.5 * kp.sigma and kp.residuals[-1] is not None:
                kp.count += 1
                kp.anchor = kp.anchor + (world[k] - kp.anchor) / kp.count
            out[k] = kp.anchor
        for name, k in side["idx"].items():
            kps[name].last_out = out[k]
        if side.get("floor_contact", True):
            for name, kp in kps.items():
                if not (side["heel_mode"] and name in ("ankle", "heel")):
                    kp.floor = kp.anchor

    def _released_step(self, side: dict, world: np.ndarray, out: np.ndarray) -> None:
        kps = side["kp"]
        still = True
        for name, k in side["idx"].items():
            kp = kps[name]
            # A released foot still cannot teleport: gate per-frame jumps against the last output.
            if kp.last_out is None or float(np.linalg.norm(world[k] - kp.last_out)) < kp.outlier_gate() \
                    or kp.rejected > LOST_FRAMES:
                out[k] = world[k]
                kp.last_inlier = world[k]
                kp.rejected = 0
            else:
                out[k] = kp.last_out
                kp.rejected += 1
            kp.last_out = out[k]
            if len(kp.raw) < 2 * WINDOW_FRAMES:
                still = False
                continue
            recent = np.median(kp.raw[WINDOW_FRAMES:], axis=0)
            previous = np.median(kp.raw[:WINDOW_FRAMES], axis=0)
            move_thr, _ = self._thresholds(kp)
            if float(np.linalg.norm(recent - previous)) > move_thr:
                still = False
        side["still"] = side["still"] + 1 if still else 0
        if side["still"] < ENTER_CHECKS:
            return
        side["planted"] = True
        side["heel_mode"] = False
        side["lost"] = 0
        ankle = kps["ankle"]
        new_ankle = np.median(ankle.raw, axis=0)
        same_place = ankle.floor is not None and float(np.hypot(*(new_ankle - ankle.floor)[[0, 2]])) < NEW_PLACEMENT_M
        raised = same_place and (ankle.floor[1] - new_ankle[1]) > RISE_MIN_M * 2
        side["floor_contact"] = not raised
        for name, k in side["idx"].items():
            kp = kps[name]
            window = np.median(kp.raw, axis=0)
            if same_place and not raised and kp.floor is not None and float(np.linalg.norm(window - kp.floor)) < MERGE_M:
                kp.anchor, kp.count = kp.floor, max(kp.count, 2 * WINDOW_FRAMES)
            else:
                kp.anchor, kp.count = window, 2 * WINDOW_FRAMES
                if not raised:
                    kp.floor = kp.anchor
            kp.residuals.clear()
            out[k] = kp.anchor
