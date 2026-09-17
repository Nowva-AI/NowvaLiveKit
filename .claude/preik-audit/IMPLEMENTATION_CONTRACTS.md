# Pre-IK Overhaul — Implementation Contracts (branch `pre_ik_overhaul`)

Plan: https://claude.ai/artifact/K3RnX2RyUQ87AP6nfHh1A2 · Evidence: `.claude/preik-audit/<topic>/REPORT.md`.

The user approved: all phases A–D; scope beyond pre-IK (remove post-IK angle filter + predictor, fix rep counter + symmetry); 66 ms (2-frame) analysis lag; keep single-camera MediaPipe mode working for testing (user won't always have 3 cameras).

## Ground rules for every workstream
- Work ONLY in the files your workstream owns (listed below). If you need a change in someone else's file, do not make it: say so in your final message under "Needs from other workstreams".
- Do NOT commit, do NOT switch branches, do NOT run `git stash`/`checkout`/`reset`. Several engineers edit this working tree concurrently.
- Python: `/Users/naiahoard/NowvaLiveKit/venv/bin/python`. Run only your own tests: `venv/bin/python -m pytest tests/test_biomechanics/<your files> -x -q`. Other workstreams' in-progress edits may break unrelated tests; ignore those.
- Follow `.claude/rules/code-style.md`, `biomechanics.md`, `testing.md`, `behavior.md`: module docstring, `from __future__ import annotations`, full type hints with builtin generics, `UPPER_SNAKE_CASE` constants (no magic numbers), explicit names, pytest classes, `pytest.approx` for floats. Bug fixes: write the failing test first. Arrays in, arrays out inside hot paths; no per-scalar Python objects per frame.
- Coordinate frame: Y-DOWN (`geometry.WORLD_UP = [0,-1,0]`), meters. In triangulated world frame X = subject's left and +Z = subject's BACK (front camera sits at −Z) — forward is −Z.
- Final message: files changed, public API added/changed, tests added and their pass count, contract deviations, "Needs from other workstreams". Subagents cannot write report .md files — put it in the message.

## C1. Keypoint layout (21 keypoints)
COCO-17 (0–16) + `LEFT_FOOT_INDEX`=17 (halpe26 L big toe 20) + `RIGHT_FOOT_INDEX`=18 (halpe26 R big toe 21) + `LEFT_HEEL`=19 (halpe26 24) + `RIGHT_HEEL`=20 (halpe26 25). `CocoKeypoints` already defines these. RTMPose halpe26 Skeleton2D and the triangulated Skeleton3D both have 21 entries. MediaPipe already produces 21.

## C2. Clock and frame identity
- Every capture timestamp is wall-aligned seconds: `perf_counter() + offset`, where `offset = time.time() - time.perf_counter()` computed ONCE at capture start. Comparable with `time.time()` everywhere.
- Multi-camera: `MultiViewPose.frame_index` and the triangulated `Skeleton3D.frame_index` = the primary camera's capture sequence number (strictly increasing). The provider never returns the same primary sequence twice.
- Filters: `dt <= 0` → no state update, re-emit last output. `dt > 0.5 s` → re-initialise.

## C3. Triangulator output (`biomechanics.triangulation.triangulator`)
- `DLTTriangulator.triangulate(multi_view) -> Skeleton3D | None` returns **WORLD coordinates** (no hip re-centering).
- Per-keypoint confidence has metric meaning: `conf = 1 / (1 + (u / UNCERTAINTY_SCALE_M)**2)` with `UNCERTAINTY_SCALE_M = 0.02`, where `u` is the estimated 3D position std in meters. Inverse: `u = 0.02 * sqrt(1/conf - 1)`. So conf 0.5 ≈ 2 cm, 0.1 ≈ 6 cm.
- Rejected/untriangulated keypoints: `confidence == 0.0` exactly, position `(0, 0, 0)`. Never `conf × 0.1` and never NaN/inf in positions.
- Module-level `recentre_at_hips(skeleton: Skeleton3D) -> Skeleton3D | None`: subtracts the hip midpoint from every keypoint with conf > 0, sets conf-0 keypoints to (0,0,0); returns `None` if either hip has conf 0. Pipeline uses it for the hip-centred views (MediaPipe skeletons are already hip-centred and skip it).

## C4. Kalman smoother (`biomechanics.utils.keypoint_kalman`, NEW)
```python
class KeypointKalmanOutput(NamedTuple):  # or frozen dataclass
    lagged_points: np.ndarray       # (N,3) fixed-lag smoothed, for analysis
    lagged_confidences: np.ndarray  # (N,)
    lagged_timestamp: float
    current_points: np.ndarray      # (N,3) filtered, undelayed, for display
    current_confidences: np.ndarray
    velocities: np.ndarray          # (N,3) m/s, at the lagged time
class FixedLagKeypointSmoother:
    def __init__(self, lag_frames: int = 2, process_noise: float = 10.0,
                 uncertainty_scale_m: float = 0.02, measurement_std_floor_m: float = 0.003,
                 measurement_std_ceiling_m: float = 0.08, gate_sigma: float = 4.0,
                 gate_min_radius_m: float = 0.08, max_predicted_frames: int = 5,
                 min_output_confidence: float = 0.15) -> None: ...
    def update(self, points: np.ndarray, confidences: np.ndarray, timestamp: float) -> KeypointKalmanOutput | None: ...
    def predict_missing(self, timestamp: float) -> KeypointKalmanOutput | None: ...  # no measurement this tick
    def reset(self) -> None: ...
```
- Constant velocity per keypoint per axis; measurement std from conf via C3 inverse, clipped to [floor, ceiling].
- conf 0 → predict only. Predicted (or gated-out) keypoints are output with confidence ≥ `min_output_confidence` until `max_predicted_frames` consecutive misses; then confidence 0 and that keypoint re-initialises on its next valid measurement (no crawl).
- Innovation gate: reject if innovation > max(gate_sigma·σ_innovation, gate_min_radius_m); re-acquire (snap) after 2 consecutive rejected measurements that agree within 10 cm; snap after 0.2 s of rejections.
- Frame-agnostic (world or hip-centred). Returns None until the first valid measurement. Before `lag_frames` history exists, the lagged output uses the shortest available lag.

## C5. Foot contact (`biomechanics.utils.foot_contact`, NEW) — world frame only
```python
class FootState(BaseModel):
    valid: bool
    planted_l: bool; planted_r: bool
    heel_rise_l_cm: float; heel_rise_r_cm: float
    stance_width_cm: float
    toe_out_l_deg: float; toe_out_r_deg: float
    hip_height_above_floor_cm: float
    lateral_hip_offset_cm: float
    floor_roll_deg: float
class FootContactModel:
    def update(self, points: np.ndarray, confidences: np.ndarray, timestamp: float) -> tuple[np.ndarray, FootState]: ...
    def reset(self) -> None: ...
```
Returns stabilized foot keypoints (ankles 15/16, toes 17/18, heels 19/20; never moves knees or hips; never couples L/R) plus FootState. Session-scoped (no reset per set).

## C6. Heel-rise rule + rule context
- `FaultType.HEEL_RISE = "heel_rise"` added in `faults/fault_types.py`.
- `FaultRule.set_frame_context(bar_detection=None, derivatives=None, phase=None, foot_state=None)` stores `self._foot_state`. `RuleEngine.evaluate(..., foot_state: FootState | None = None)` forwards it.
- `faults/rules/heel_rise.py`: `HeelRiseRule(mild_cm=1.5, moderate_cm=3.0, severe_cm=5.0)`, reads `self._foot_state`, no-op when None/invalid (single-camera mode).

## C7. Body measurements (`biomechanics.utils.segment_lengths`, NEW — replaces `bone_constraints.py`)
```python
class BodyProportions(BaseModel):   # moved here; valgus_scale and pelvis_tilt_coupling REMOVED
    hip_width: float; femur_length_avg: float; tibia_length_avg: float; torso_length_avg: float
    shoulder_width: float; foot_length_avg: float
    hip_to_femur_ratio: float; tibia_to_reference_ratio: float
    forward_lean_scale: float
class SegmentLengthEstimator:
    def record(self, points: np.ndarray, confidences: np.ndarray, rep_count: int) -> None: ...
    @property
    def is_complete(self) -> bool: ...
    @property
    def progress(self) -> tuple[int, int]: ...
    @property
    def body_proportions(self) -> BodyProportions | None: ...
    def to_athlete_params(self) -> dict | None: ...   # same keys as pipeline_process._extract_athlete_params today
    @classmethod
    def from_athlete_params(cls, params: dict) -> "SegmentLengthEstimator": ...  # complete immediately
```
Session-scoped: never reset per set. Pipeline exposes it as `pipeline.body_calibration`.
Pipeline (wave 2) calls `body_calibration.record(analysis_points, analysis_confidences, rep_count)` inside `process_frame` on every ready analysis frame until complete, applies proportion scaling ONCE when it completes, and provides `pipeline.apply_athlete_params(params: dict) -> None` (replaces the estimator via `from_athlete_params` and applies scaling once) for returning users. `pipeline_process.py` must no longer wait for body calibration before the assessment reps: it waits for `pipeline.is_ready` only; lengths accumulate during the assessment reps.
Already done by lead: `FaultType.HEEL_RISE`, `FaultRule.set_frame_context(..., foot_state=None)` storing `self._foot_state`, and `RuleEngine.evaluate(..., foot_state=None)` forwarding (C6 plumbing).

## C8. Proportion scaling
`RuleEngine.apply_body_proportion_scaling(proportions)` is idempotent: each rule rescales from base thresholds stored at construction. `KneeValgusRule` no longer scales. `ForwardLeanRule` scales in lean space: `threshold = 180 - (180 - base) * forward_lean_scale`. `AnalyticalIKSolver.set_body_proportions` is removed.

## C9. Missing angles
`AnalyticalIKSolver` returns `float("nan")` (never 0.0) for any angle whose keypoints are missing (conf below its minimum). Rules skip NaN inputs. Anything serialized to JSON/IPC converts NaN → `None`.

## C10. Pre-IK chain + pipeline (integration, wave 2)
`build_preik_chain(config, multi_camera: bool)` → chain with `run(skeleton) -> PreIKResult(analysis: Skeleton3D hip-centred lagged, display: Skeleton3D hip-centred current, foot_state: FootState | None, velocities)`, `predict_missing(timestamp)`, `reset()` (temporal state only), `stage_names`, inspector tap. Triangulated: Kalman (world) → FootContact (world) → recentre. MediaPipe: Kalman (hip-centred, larger measurement floor) → no foot contact. Removed from analysis path: ConfidenceBlender, VelocityClamp, BoneLengthConstraints enforcement, GroundClamp, KeypointPositionSmoother, JointAngleFilter, PredictiveStateEstimator, DerivativeTracker (squat).

## Workstream file ownership (wave 1)
| WS | Owner of |
|---|---|
| WS1 capture | `triangulation/multi_capture.py`, `pose/multi_camera.py`, new `tests/test_biomechanics/test_multi_capture.py`, `test_multi_camera_provider.py` |
| WS2 detector | `pose/rtmpose.py`, `triangulation/calibration.py`, tests for them |
| WS3 triangulation | `triangulation/triangulator.py`, tests for it |
| WS4 kalman | new `utils/keypoint_kalman.py` + test |
| WS5 feet | new `utils/foot_contact.py`, new `faults/rules/heel_rise.py`, `faults/rules/__init__.py`, `profiles/squat.py`, tests |
| WS6 body | new `utils/segment_lengths.py`, `pipeline_process.py`, `coaching/` + `db/` persistence files for athlete_params, tests |
| WS7 kinematics+rules | `kinematics/analytical_ik.py`, `kinematics/valgus.py`, `faults/fault_types.py`, `faults/rule_engine.py`, `faults/rules/*.py` except heel_rise.py and `__init__.py`, rep counter (`hip_position_counter.py` and friends), IPC/bridge NaN sanitization, tests |
| Wave 2 integration | `pipeline.py`, `utils/preik_chain.py`, `config.py`, `config/biomechanics.yaml`, `viz/pipeline_inspector.py`, scripts/tools/benchmarks/visualizers using old filters, deleting old filter modules + their tests |
