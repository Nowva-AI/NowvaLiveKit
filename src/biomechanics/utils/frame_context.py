"""
Per-frame context handed to exercise profiles and rep analysers, and the
RepAnalyzer protocol a non-squat exercise implements (SHARED-PLUMBING.md §2-§3).

Frame: Y-down metres, the same world frame as the skeleton. `up` points
physically up (opposite gravity) and is used as is, never negated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pydantic import BaseModel

from biomechanics.utils.geometry import WORLD_UP

if TYPE_CHECKING:
    from biomechanics.utils.foot_contact import FootState
    from biomechanics.utils.segment_lengths import BodyProportions
    from biomechanics.utils.types import BarbellDetection, JointAngles, Skeleton3D

# Where FrameContext.up comes from.
GRAVITY_MEASURED = "measured"          # floor ChArUco, ~/.nowva/gravity_<camera>.json
GRAVITY_BODY = "body"                  # no measurement: WORLD_UP (median standing body axis)
GRAVITY_CAMERA_PITCH = "camera_pitch"  # one camera, pitch-corrected: unvalidated, never cued on

MOUNT_RACK = "rack"
MOUNT_INCLINOMETER = "inclinometer"
MOUNT_NONE = "none"

# Optional keys of an analyser's rep features. Timing: RepData.descent_time is
# the eccentric phase and ascent_time the concentric one, whatever the lift's
# shape. highlight_score: higher is the better rep (finite, or left out).
ECCENTRIC_TIME_FEATURE = "eccentric_time_s"
CONCENTRIC_TIME_FEATURE = "concentric_time_s"
HIGHLIGHT_SCORE_FEATURE = "highlight_score"


@dataclass
class FrameContext:
    """Everything an exercise may read about one frame.

    Readiness checks run before pre-IK and IK: there `angles` is None and
    `skeleton_3d` is the raw, hip-centred skeleton. observe / observe_frame
    get the analysed (lagged, smoothed) skeleton and its joint angles.
    """
    skeleton_3d: Skeleton3D
    angles: JointAngles | None
    timestamp: float
    phase: str
    in_rep: bool
    capture_mode: str
    up: np.ndarray = field(default_factory=lambda: WORLD_UP.copy())
    gravity_source: str = GRAVITY_BODY
    # Single camera: the standing body's tilt from the camera vertical at the
    # last latch (posture included, unvalidated). NaN when not measured.
    camera_pitch_deg: float = math.nan
    # The single camera's physical mount, and whether its pitch is certified
    # (measured mount, not revoked by a delta tripwire this session).
    pitch_certified: bool = False
    mount_source: str = MOUNT_NONE
    mount_pitch_deg: float = math.nan
    mount_height_m: float = math.nan
    # Per-keypoint confidence of the raw capture this frame describes.
    raw_confidences: np.ndarray | None = None
    body_proportions: BodyProportions | None = None
    bar_detection: BarbellDetection | None = None
    foot_state: FootState | None = None
    # Rig only: the analysed skeleton in the world frame (not hip-centred).
    skeleton_world: Skeleton3D | None = None


class RepAnalyzer(Protocol):
    """Owns every per-frame and per-rep state of a non-squat exercise.

    rep_signal (NaN when unknown) drives the rep counter; it is updated by
    observe. Optional methods, checked with hasattr: is_ready(ctx) -> bool
    (ANDed into the readiness gate) and counter_baseline() -> float | None
    (seeds the counter at the latch and after every counter reset).

    Single-camera pitch, also optional: bind_pitch_certification(certification,
    profile_name) hands the analyser the session's tripwire for its profile (it
    outlives exercise switches),
    pitch_tripwire_delta() -> float | None reports a delta trip once (it revokes
    certification for the session), and camera_pitch_untrusted: bool is the
    gross tripwire on an uncertified camera (signalled once a session).
    """

    rep_signal: float

    def observe(self, ctx: FrameContext) -> None: ...

    def on_rep_start(self, piston: bool) -> None: ...

    def finish_rep(self, rep_number: int) -> BaseModel: ...

    def rejection_reason(self, features: BaseModel) -> str | None: ...

    def discard_rep(self) -> None: ...

    def reset_set(self) -> None: ...
