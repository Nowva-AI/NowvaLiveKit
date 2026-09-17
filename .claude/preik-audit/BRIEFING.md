# Pre-IK Filter Audit — Shared Briefing

Repo: /Users/naiahoard/NowvaLiveKit  (Python: ./venv/bin/python — NEVER the base conda python)
Scratch root for this audit: /private/tmp/claude-501/-Users-naiahoard-NowvaLiveKit/0f77ba9d-b99a-4ee5-8811-1911932fe16e/scratchpad/preik_audit/
Tests: `./venv/bin/python -m pytest tests/test_biomechanics/<file> -x -q` (run from repo root; tests import `biomechanics.*` via src on path — check tests/conftest.py if imports fail).
Unrelated pre-existing failures exist elsewhere in tests/ (coaching orchestrator, demo narration, v6, program generator) — ignore them.

## Goal (from the CEO/user)
Deep audit of the pre-inverse-kinematics (pre-IK) skeleton filters: are they accurate, do they work, are
they useful or useless, fix errors, and identify missing / wrong filters. We are rebuilding them to be
powerful, and the TRIANGULATED 3-camera pipeline is now the primary mode. The user has noticed errors
in many of them. Earlier (2026-07/08) the user disabled 4 of the 6 stages because empirically they made
the pose WORSE.

## Product guardrails (must respect)
- Everything must eventually run on a Jetson Orin Nano class edge device: cheap numpy, causal/real-time,
  no heavy optimizers per frame unless cheap and bounded.
- Diagnosis accuracy is the core IP: correct first, then fast. Squat only.
- Filters must never erase real movement faults we diagnose: knee valgus, heel rise, lateral hip shift,
  pelvic list, asymmetric depth, forward lean, depth (knee/hip flexion at bottom), stance width.

## The chain (src/biomechanics/utils/preik_chain.py)
Production order: ConfidenceBlender.blend -> VelocityClamp.clamp -> BoneLengthConstraints.enforce ->
GroundClamp.clamp -> KeypointPositionSmoother.smooth (One Euro per axis) -> BoneLengthConstraints.enforce.
CURRENTLY ONLY blend + velocity clamp are active; the other four lines are commented out (also mirrored
in pipeline.py `_apply_preik_filters_inspected`). `scripts/tools/capture_audit.py` replicates the chain
for an HTML audit tool (MediaPipe single-cam).
Files: src/biomechanics/utils/{confidence_blend,velocity_clamp,bone_constraints,ground_clamp,
position_filter,filters,standing_gate,preik_chain,geometry,types}.py ; config in
src/biomechanics/config.py + config/biomechanics.yaml. Tests in tests/test_biomechanics/test_{confidence_blend,
velocity_clamp,bone_constraints,ground_clamp,preik_chain,standing_gate,phase_aware_smoothing}.py (61 pass).
After IK: JointAngleFilter (One Euro on angles, phase-aware) -> DerivativeTracker -> PredictiveStateEstimator
(extrapolates 0.2 s) -> RuleEngine. Previous audit (.claude/pipeline-audit/FINDINGS.md F17) measured ~±9°
phase-lag error from stacked smoothing on synthetic data; F18 lists triangulation robustness gaps.

## Pipeline wiring facts already verified (src/biomechanics/pipeline.py process_frame)
- Coordinate frame is Y-DOWN (geometry.WORLD_UP = [0,-1,0]). Hip midpoint at origin. Larger y = lower.
  (.claude/rules/biomechanics.md text about "Larger Y = higher" is stale; trust geometry.py.)
- Order per frame: pose -> dropout hold (up to 5 frames of last skeleton with decaying confidence,
  timestamp=time.time()) -> BiLSTM on RAW skeleton -> standing_gate.check(RAW) -> readiness_gate.check(RAW),
  early return if not ready (filters never see those frames) -> pre-IK chain -> body-proportion scaling
  once bone calibration completes -> IK -> valgus estimator -> JointAngleFilter -> ... faults.
- reset_readiness_gate() (called at set boundaries) resets ALL filters including bone calibration and
  sets _proportions_applied=False. Rule `scale_for_proportions` does `threshold *= scale` (compounds if
  applied more than once).
- pipeline_process.py assessment wait loop drives `pipeline._bone_constraints.enforce(result.skeleton_3d)`
  record-only (on the already blended+clamped skeleton) because the chain no longer calls it.
- The loop is paced to target_fps=30 by sleeping (sum of latency_ms).

## Triangulated mode facts (NOWVA_MULTI_CAMERA=true)
- src/biomechanics/pose/multi_camera.py: 3 cams, RTMPose-m halpe26 batched. get_pose() returns
  (primary frame, primary Skeleton2D, triangulated Skeleton3D).
- src/biomechanics/triangulation/multi_capture.py get_synced_frames(): returns the LATEST primary frame
  every call whether or not it was already processed (possible duplicate frames with identical
  timestamps); timestamps are time.perf_counter() (different clock from time.time() used by dropout hold
  and MediaPipe).
- src/biomechanics/pose/rtmpose.py: full 1280x720 frame squashed to 192x256 (no person crop), plain argmax
  SimCC decode (no sub-pixel), confidence = sigmoid(max logit) (floors ~0.5), keypoints below
  confidence_threshold 0.3 set to (0,0,conf 0). Only big toes mapped to 17/18; halpe26 heels (24/25) and
  small toes are DROPPED, so the triangulated skeleton has 19 keypoints and no heels.
- src/biomechanics/triangulation/triangulator.py DLTTriangulator: views with conf >= 0.3; >=2 views;
  unweighted DLT (cv2 for 2 views, SVD for 3); confidence = min view conf * (1 - reproj_err/15px), or
  *0.1 if reproj >= 15 px (point still used); re-centered each frame at hip midpoint (only points with
  conf>0 shifted), so global translation is removed. frame_index hardcoded 0.
- src/biomechanics/triangulation/calibration.py: T-pose + user height -> solvePnP per camera against an
  anthropometric canonical T-pose model; intrinsics guessed (f = 0.8*width, principal point = center,
  zero distortion). World frame: Y-down, X to subject's left, Z forward, origin at hip midpoint of the
  T-pose. So world Y is approximately gravity-aligned only if the user stood straight.

## Rules for Phase-1 agents
- DO NOT modify any file inside the repo. Write all scripts/outputs under your own subfolder of the
  scratch root (e.g. .../preik_audit/<your_topic>/). Import repo code by adding
  /Users/naiahoard/NowvaLiveKit/src to sys.path.
- Back every claim with evidence: file:line references and, wherever possible, a small numeric experiment
  whose numbers you report. Distinguish CONFIRMED (reproduced) from SUSPECTED.
- For every finding give: severity (critical/high/medium/low), file:line, concrete failure scenario
  (inputs -> wrong output), measured impact, proposed fix (specific, minimal, edge-friendly).
- End with a verdict per filter you own: KEEP AS IS / FIX / REBUILD (describe design) / REMOVE, specifically
  for triangulated 3-camera input, and note if the single-camera MediaPipe path needs different behaviour.
- Final message: a compact structured report (it is consumed by the lead engineer, not the user). Also save
  it as REPORT.md in your subfolder.
