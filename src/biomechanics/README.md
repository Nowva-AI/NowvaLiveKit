# Biomechanics Diagnosis Engine

The core IP of Nowva. This module analyzes squat form in real time: it captures
video, estimates pose, computes joint angles, detects form faults, counts reps,
and produces structured diagnoses that the voice agent relays as coaching cues.

Everything here must eventually run on a Jetson-class edge device (~40 TOPS).
No cloud API calls in the inference path.

## Data Flow

```
Camera frame
  |
  v
Pose Estimation (MediaPipe or RTMPose) --> Skeleton2D + Skeleton3D
  |
  v
Standing Gate (validates user is in frame, upright)
  |
  v
Pre-IK Filters (confidence blend, velocity clamp, bone constraints, position smoothing)
  |
  v
Inverse Kinematics (AnalyticalIKSolver) --> JointAngles
  |
  v
Fault Detection (RuleEngine + per-exercise FaultRules) --> list[FaultEvent]
  |
  v
Rep Counting (HipPositionRepCounter or BiLSTM) --> RepData
  |
  v
Diagnosis Engine (HypothesisEngine) --> DiagnosisResult (set-level, after reps complete)
  |
  v
Coaching Layer (IPCBridge + SessionTracker) --> IPC messages to voice agent
```

The pipeline runs frame-by-frame. Each call to `BiomechanicsPipeline.process_frame()`
executes one full iteration and returns a `PipelineFrame` with all results and
per-layer latency measurements.

## Directory Map

### `analysis/`
Post-hoc set analysis. `RepSegmenter` finds precise rep boundaries from smoothed
hip position time series using scipy peak detection. `SetDataCollector` in
`set_finalizer.py` accumulates per-frame data during a set and generates plots
and JSON reports when the set ends.

### `barbell_tracking/`
Real-time barbell detection and tracking. `BarbellDetector` wraps a YOLO11n-pose
model that outputs two keypoints (left/right bar endpoints) plus a bounding box.
`BarPathTracker` smooths those detections through per-endpoint constant-velocity
Kalman filters, computes bar tilt and velocity, and maintains a rolling center-path
history. Feeds the `BarTiltAsymmetryRule` fault detector.

### `coaching/`
Connects the diagnosis pipeline to the live voice agent over IPC.

- `IPCBridge` -- translates pipeline events (faults, reps, frames) into throttled,
  deduplicated JSON messages. Handles cue caching and fault cooldown.
- `SessionTracker` -- detects set boundaries from rep timing gaps, accumulates
  per-set statistics, runs the diagnosis engine at set end, and sends set summaries.
- `CueCache` -- pre-caches exercise-specific audio cue identifiers so the voice
  agent can pre-generate TTS. Rate-limits cue delivery to avoid overwhelming the lifter.

### `diagnosis/`
Set-level causal diagnosis engine. This is the layer that goes beyond "you had
knee valgus" to explain *why* -- e.g., limited ankle dorsiflexion forces
compensatory knee cave.

- `engine.py` -- `HypothesisEngine.diagnose(SetFeatures) -> DiagnosisResult`.
  Detects symptoms from aggregated rep kinematics, maps them to candidate causes
  via a knowledge graph (`graph/symptoms.yaml`, `graph/causes.yaml`), scores each
  cause using evidence tests, and returns tiered hypotheses (immediate / session /
  long-term / contextual).
- `rep_scoring.py` -- scores each rep on 5 dimensions (depth, trunk control,
  knee tracking, symmetry, ankle utilization) to produce a 0-1 composite score.
- `bridge.py` -- maps raw pipeline data (bottom-of-rep keypoints + angles) into
  the `RepKinematicSummary` and `SetFeatures` types the engine expects.
- `keypoint_corrector.py` -- applies geometric corrections to keypoints based on
  diagnosed faults (delta-FK approach matching the visualizer's deformLowerBody).
- `demo_builder.py` -- `build_demo_data()` constructs choreographed coaching-demo
  data (corrected pose stack + cue metadata) from a diagnosis result. Used after
  a failed assessment to generate animated correction sequences.
- `types.py` -- Pydantic models: `RepKinematicSummary`, `SetFeatures`,
  `DiagnosisResult`, `HypothesizedCause`, `DetectedSymptom`, `RepScore`.

### `faults/`
Frame-level fault detection.

- `fault_types.py` -- `FaultRule` ABC that all rules inherit from, plus `FaultType`
  enum and default threshold tables.
- `rule_engine.py` -- `RuleEngine` orchestrates all active rules per frame. Maintains
  a rolling 90-frame angle history, deduplicates consecutive same-fault detections,
  and handles per-rep calibration.
- `rules/` -- one file per fault rule: `depth`, `forward_lean`, `knee_valgus`,
  `symmetry`, `back_rounding`, `tempo`, `bar_path`,
  `bar_tilt_asymmetry`, `lockout`, `elbow_flare`, `shoulder_stability`,
  `trunk_stability`, `range_of_motion`.
- `hip_position_counter.py` -- 4-state FSM rep counter driven by hip vertical
  position (STANDING -> DESCENDING -> BOTTOM -> ASCENDING).

### `kinematics/`
Joint angle computation from 3D keypoints.

- `base.py` -- `IKSolver` abstract interface.
- `analytical_ik.py` -- `AnalyticalIKSolver` computes all joint angles
  (knee flexion, hip flexion/adduction, trunk flexion, ankle dorsiflexion,
  bilateral asymmetry) using vector geometry on MediaPipe world coordinates.
  No external musculoskeletal model needed.

### `ml/`
Machine learning models for rep counting.

- `bilstm_model.py` -- `BiLSTMRepModel`, a 2-layer bidirectional LSTM that outputs
  per-frame squat depth class logits (5 classes: standing, quarter, half, parallel,
  deep).
- `inference.py` -- `BiLSTMInference` wraps the model for real-time use with a
  sliding sequence buffer.
- `feature_extractor.py` / `feature_extractor_base.py` -- extract normalized
  features from `Skeleton3D` for model input.
- `sequence_buffer.py` -- fixed-length sliding window buffer for streaming inference.

### `pose/`
2D and 3D pose estimation backends.

- `base.py` -- `PoseEstimator` ABC defining `estimate()`, `estimate_3d()`, and
  `estimate_both()`. Uses COCO 17-keypoint format (extended to 19 with foot indices).
- `mediapipe_fallback.py` -- `MediaPipePoseEstimator`, the default backend.
  Returns both 2D pixel coordinates and 3D world coordinates from a single frame.
- `rtmpose.py` -- `RTMPoseEstimator`, an ONNX-based alternative for edge deployment.

### `profiles/`
Exercise profile system. Each profile bundles the fault rules, rep counting signal,
coaching cues, and depth categorization for one exercise type.

- `base.py` -- `ExerciseProfile` base class with squat defaults.
- `squat.py` -- `SquatProfile` registered under aliases like `squat`,
  `barbell_back_squat`, `barbell_front_squat`. Creates the 5 core squat fault rules
  and uses hip Y-position as the rep counting signal.
- `registry.py` -- `@register_profile` decorator and `get_profile()` lookup.
- Other profiles (`deadlift`, `lunge`, `overhead_press`, etc.) exist as stubs
  for future exercises. Squats are the only fully implemented exercise.

### `triangulation/`
Multi-camera 3D reconstruction.

- `triangulator.py` -- `DLTTriangulator` performs Direct Linear Transform
  triangulation from 2+ calibrated camera views into 3D world coordinates.
  For 2 views uses `cv2.triangulatePoints`; for 3+ builds the DLT system and
  solves via SVD.
- `calibration.py` -- camera calibration data structures and loading.
- `multi_capture.py` -- synchronized frame capture from multiple cameras.

### `utils/`
Shared types, filters, and geometry helpers used across all layers.

- `types.py` -- core Pydantic data types: `Skeleton2D`, `Skeleton3D`,
  `JointAngles`, `FaultEvent`, `FaultSeverity`, `RepData`, `PipelineFrame`,
  `CocoKeypoints`, `BarbellDetection`, `BarTrackState`.
- `geometry.py` -- vector math: `angle_between_vectors`, `joint_angle_3_points`,
  `flexion_angle`, `project_to_plane`, `normalize_vector`.
- `standing_gate.py` -- `StandingPoseGate` validates the user is standing in frame
  before calibration begins (checks keypoint visibility, knee extension, torso
  uprightness over N consecutive frames).
- `filters.py` -- `OneEuroFilter`, `LowPassFilter`, `ExponentialMovingAverage`,
  `RunningMedian` (display smoothing and the robust per-rep depth statistic).
- `derivatives.py` -- `AngleDerivatives` container (velocity/acceleration) read by
  the tempo rules; the squat path passes none.
- `preik_chain.py` -- `build_preik_chain` / `PreIKChain`: the single pre-IK stage
  order (Kalman → foot contact → hip re-centring) with an inspector tap.
- `keypoint_kalman.py` -- `FixedLagKeypointSmoother`: the only temporal filter
  before IK; lagged analysis stream + undelayed display stream.
- `foot_contact.py` -- `FootContactModel` / `FootState`: world-frame planted-foot
  anchors, heel rise, stance metrics (multi-camera).
- `segment_lengths.py` -- `SegmentLengthEstimator` / `BodyProportions`:
  session-scoped body measurement feeding fault threshold scaling.
- `position_filter.py` -- `Skeleton2DSmoother`, One Euro on the 2D overlay
  (display only).
- `json_safe.py` -- `nan_to_none` for every JSON/IPC boundary (missing angles are NaN).

### `viz/`
Visualization and debugging.

- `overlay_2d.py` -- `draw_skeleton()`, `draw_fps()`, `FPSCounter` for OpenCV
  overlays.
- `dashboard.py` -- `DebugDashboard` for real-time angle/fault display.
- `set_plots.py` -- matplotlib plots for hip position and velocity traces.
- `html_dashboard.py` -- generates self-contained HTML session dashboards with
  per-set charts and segmented rep data.
- `window_anim.py` -- window pre-creation and native macOS fullscreen animation
  helpers. Pre-creates the OpenCV window before the pipeline starts so the
  transition to fullscreen is instant (uses PyObjC `NSApplication` on macOS).
- `demo_ws_bridge.py` -- serves the Three.js choreography viewer over HTTP and
  bridges demo events (start/cue/end) plus live skeleton frames to it over
  WebSocket. Owns the demo timing constants and per-fault joint highlight map
  sent in the init payload; relays the viewer's started/done acks back.
- `choreographer.mjs` -- the choreographer state machine (morph-in, per-cue
  yoyo loops with a "before" ghost pose, settle, final hold, morph-out to the
  live skeleton). Pure logic with injected render callbacks so it runs both in
  the browser and under `node --test tests/js/*.test.mjs`.
- `demo_viewer.html` -- Three.js viewer page. Renders the main and ghost
  skeletons, per-fault camera angles, and captions, driven by the
  choreographer, synchronized with voice narration cues from the agent.

## Key Entry Points

**`pipeline.py`** -- `BiomechanicsPipeline`. The main frame-by-frame pipeline class.
Instantiates all layers (capture, pose, pre-IK filters, IK, faults, rep counting,
optional BiLSTM and barbell tracking). Call `process_frame()` in a loop; returns
a `PipelineFrame` with all outputs and per-layer timing.

**`pipeline_process.py`** -- `run_biomechanics_pipeline()`. Subprocess entry point
launched by `main.py` when a workout starts. Runs the assessment phase (2 bodyweight
reps with diagnosis), calibration phase (5 reps to personalize thresholds), then
the main workout loop. Communicates with the voice agent via `IPCBridge`.

**`calibration.py`** -- `CalibrationTracker` collects peak angle values during
calibration reps. `build_calibration_profile()` converts peaks into personalized
fault thresholds. `apply_calibration_to_rule_engine()` writes those thresholds
directly into the rule engine's rules.

**`config.py`** -- `BiomechanicsConfig` (Pydantic BaseModel) aggregating all
sub-configurations. `load_pipeline_config()` reads from YAML and returns a typed
config instance.

## Configuration

All settings live in `config/biomechanics.yaml` at the project root. The file maps
directly to the `BiomechanicsConfig` Pydantic model in `config.py`, which nests
sub-configs for each subsystem: `PoseConfig`, `FaultsConfig`, `BarbellTrackingConfig`,
`HipPositionCounterConfig`, `StandingGateConfig`, `BiLSTMConfig`, etc.

To load config:
```python
from biomechanics.config import load_pipeline_config
config = load_pipeline_config()  # uses default path
config = load_pipeline_config("path/to/custom.yaml")
```

Defaults are baked into the Pydantic models so the system runs without a YAML file.

## Edge Constraint

The target deployment hardware is a Jetson Orin Nano Super (~40 TOPS). Design
constraints that follow from this:

- Pose estimation must use lightweight models (RTMPose ONNX or MediaPipe).
- The analytical IK solver uses pure NumPy vector math, no OpenSim dependency.
- The BiLSTM rep counter is a small 2-layer model (~500K parameters).
- The YOLO barbell detector uses the nano variant (YOLO11n-pose).
- The diagnosis engine is pure Python with no ML inference -- just rule evaluation
  and graph traversal.
- All processing fits within a single `process_frame()` call targeting 30 FPS.
