"""Proposed pre-IK chain assembled from the REAL wave-1 components (contract C10, triangulated mode):

    world Skeleton3D -> FixedLagKeypointSmoother (C4, lag 2, R from C3 confidences)
                     -> FootContactModel.update on the LAGGED world stream (C5)
                     -> recentre_at_hips (C3) -> hip-centred analysis Skeleton3D

The returned skeleton's frame_index is the tick index of the lagged measurement so the runner aligns it with the
truth at that instant; its timestamp is the lagged timestamp. Frames with no triangulated skeleton (or a missing
hip) go through predict_missing. `proposed_nofoot` skips the foot model to isolate its effect.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

import numpy as np

from biomechanics.triangulation.triangulator import recentre_at_hips
from biomechanics.utils.foot_contact import FootContactModel, FootState
from biomechanics.utils.keypoint_kalman import FixedLagKeypointSmoother
from biomechanics.utils.types import Skeleton3D

from .runner import FrameContext

TIMESTAMP_HISTORY = 32
DEFAULT_LAG_FRAMES = 2


class ProposedChain:
    input_frame = "world"
    handles_missing = True
    lagged_output = True
    tail = "new"

    def __init__(self, foot_contact: bool = True, lag_frames: int = DEFAULT_LAG_FRAMES, **kalman_kwargs) -> None:
        self._smoother = FixedLagKeypointSmoother(lag_frames=lag_frames, **kalman_kwargs)
        self._foot = FootContactModel() if foot_contact else None
        self.last_foot_state: FootState | None = None
        self.last_stage_us: dict[str, float] = {}
        self._tick_of_timestamp: dict[float, int] = {}
        self._timestamps: deque[float] = deque()

    def reset(self) -> None:
        self._smoother.reset()
        if self._foot is not None:
            self._foot.reset()
        self.last_foot_state = None
        self.last_stage_us = {}
        self._tick_of_timestamp.clear()
        self._timestamps.clear()

    def calibrated(self) -> bool | None:
        if self._foot is None:
            return None
        return self.last_foot_state is not None and self.last_foot_state.valid

    def process(self, skeleton: Skeleton3D, context: FrameContext) -> Skeleton3D | None:
        points = skeleton.to_numpy()
        confidences = np.array([kp.confidence for kp in skeleton.keypoints], dtype=np.float64)
        self._remember(skeleton.timestamp, context.tick_index)
        t_start = time.perf_counter()
        output = self._smoother.update(points, confidences, skeleton.timestamp)
        return self._finish(output, t_start, context.tick_index)

    def process_missing(self, context: FrameContext) -> Skeleton3D | None:
        timestamp = context.multi_view.timestamp
        self._remember(timestamp, context.tick_index)
        t_start = time.perf_counter()
        output = self._smoother.predict_missing(timestamp)
        return self._finish(output, t_start, context.tick_index)

    def _remember(self, timestamp: float, tick_index: int) -> None:
        self._tick_of_timestamp[timestamp] = tick_index
        self._timestamps.append(timestamp)
        while len(self._timestamps) > TIMESTAMP_HISTORY:
            self._tick_of_timestamp.pop(self._timestamps.popleft(), None)

    def _finish(self, output, t_start: float, tick_index: int) -> Skeleton3D | None:
        t_kalman = time.perf_counter()
        if output is None:
            self.last_stage_us = {"kalman": (t_kalman - t_start) * 1e6}
            return None
        lagged_points = output.lagged_points
        lagged_confidences = output.lagged_confidences
        if self._foot is not None:
            lagged_points, self.last_foot_state = self._foot.update(lagged_points, lagged_confidences,
                                                                    output.lagged_timestamp)
        t_foot = time.perf_counter()
        world = Skeleton3D.from_numpy(lagged_points, confidences=lagged_confidences,
                                      timestamp=output.lagged_timestamp,
                                      frame_index=self._tick_of_timestamp.get(output.lagged_timestamp, tick_index))
        analysis = recentre_at_hips(world)
        t_recentre = time.perf_counter()
        self.last_stage_us = {"kalman": (t_kalman - t_start) * 1e6, "foot": (t_foot - t_kalman) * 1e6,
                              "recentre": (t_recentre - t_foot) * 1e6}
        return analysis


def proposed_chain(foot_contact: bool = True, **kwargs) -> Callable[[], ProposedChain]:
    def factory() -> ProposedChain:
        return ProposedChain(foot_contact=foot_contact, **kwargs)
    factory.__name__ = "proposed" if foot_contact else "proposed_nofoot"
    return factory
