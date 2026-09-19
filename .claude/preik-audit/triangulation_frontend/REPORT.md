# Stage-0 Audit — What the Pre-IK Filters Receive in 3-Camera Mode

(Saved by lead from the agent's final message; subagent sandbox blocked writing REPORT.md. Numbers also in PROGRESS.md; every `expN_*.py` reruns with `venv/bin/python`.)

**Data/tags.** Real: `data/squats.mov` (portrait, frontal, 361 frames) placed in a 1280x720 canvas so the full-frame squash matches production; person ~470 px tall. Synthetic: 1.885 m squat, 3 cameras yaw −40/0/+40 (also ±70), 3.5 m, focal 0.8w; DLT matches `DLTTriangulator` exactly (max diff 0.0). CONFIRMED = experiment or exact arithmetic on the real code path; SUSPECTED = not reproduced end to end.

## Bottom line
Three pre-filter errors dominate the 3D skeleton and no filter can fix them: (1) **no person crop** — 720p 2D keypoints off median 11 px / p95 61 px → ~60 mm per body keypoint in 3D, 6° mean knee-flexion error, identical in every view so reprojection can't see it; (2) **sigmoid confidence** — every keypoint ≥ 0.53 even on a black/empty image → hallucinated skeletons triangulate; (3) **frame sync** — ~45 % of frames lost in simulation. Tuning the pre-IK chain before fixing these tunes against artefacts.

## FINDINGS.md F13 / F18 status
| Item | Status |
|---|---|
| F13 model path (`scripts/tools/download_models.py:49`, still `parent.parent`) | open |
| F13 no crop (`rtmpose.py:181`) | open, critical |
| F13 argmax decode (`rtmpose.py:232-233`) | open |
| F13 sigmoid confidence (`rtmpose.py:238-239`) | open, critical |
| F18 no view selection | open |
| F18 unweighted/unnormalized DLT | open, low value for this rig |
| F18 per-keypoint Python loop | **fixed** (batched SVD, `triangulator.py:92-121`, 0.11 ms/frame) |
| F18 `frame_index=0` (`multi_camera.py:201`) | open |

## Findings

**S0-1 CRITICAL, CONFIRMED — full-frame squash, no crop** (`rtmpose.py:181`). Full frame vs tracked crop: body keypoints differ median 11.1 px, p95 60.8 px; overlay `overlay_ff_red_crop_green.jpg` shows face keypoints on the chest; gray padding gives the same (9.0 vs 9.4 px) → not the canvas. Hip width 37 % too wide, knee separation 10 % too narrow. Accuracy checks: mirrored-image disagreement 7.9 px full frame vs 1.15 px crop; known 0–8 px image shift followed with slope 0.56 (x) / 0.11 (y) full frame vs 0.96–1.05 crop; motion noise 1.91 vs 0.44 px. 3D (Exp5, measured 2D errors through 3 views): body keypoints 60–65 mm vs 6.7 mm with crop-like noise; knee flexion mean 5.5–6.0° (p95 12–14°) vs 1.2°; hip width +118 mm; 0 % flagged by reprojection error. SUSPECTED: T-pose calibration uses the same squashed detections → camera poses inherit the bias. **Fix:** crop tracked from previous keypoints (3:4 aspect, 1.25 padding), bootstrapped by one full-frame pass; 0.31 ms/view vs 0.23 ms current resize; no detector, no inference change.

**S0-2 CRITICAL, CONFIRMED — sigmoid on keypoint score** (`rtmpose.py:238`). mmpose uses raw `min(max_x, max_y)`; rtmlib raw mean. No-person images (black, noise, background): raw max 0.16–0.27, sigmoid 0.53–0.57 → with sigmoid all 26 keypoints pass 0.3, with raw none do. Knock-on: `_kpts_to_skeleton2d` never returns None; every view always counts toward `min_views`; calibration filter `avg_confs > 0.5` always accepts all. Ranges: raw full frame 0.37–0.96 squeezed by sigmoid to 0.59–0.72; crop raw median 0.97, up to 1.19 → clip. Occlusion (70 px box over knee): crop score only drops to 0.79–0.95 while the knee moves 6–15 px → confidence reflects occlusion weakly. **Fix:** raw score clipped to [0,1], threshold 0.3–0.4 (will expose S0-6).

**S0-3 HIGH — frame sync drops ~half the frames** (CONFIRMED simulation; not measured on hardware) (`multi_capture.py:111-145`). Reference always newest primary frame; tolerance 15 ms < 16.7 ms half-period; one secondary miss → None for all views; no duplicate tracking. Simulation (30.00/29.97/30.03 fps cameras, 25±3 ms USB latency, loop paced 30 fps, 10 phase draws): Mac-speed loop 44.5 % calls return nothing (22–69 % by phase), 45.5 % frames never processed; Jetson-like 45 ms processing 36 % fail, 53 % skipped. **Proposed policy:** reference = newest primary frame every camera has caught up to; nearest frame per camera within 20 ms; triangulate with any in-tolerance cameras if ≥ 2; skip already-processed reference → 0 % failures, 11 % skipped (recoverable by waiting on a new-frame event instead of sleeping), p95 offset 16.6 ms. A camera 15 ms late → ≤ 7.4 mm, ≤ 1.0° knee error.

**S0-4 HIGH, CONFIRMED — one bad view drags the 3D point** (`triangulator.py:107-131`). Synthetic, 1.5 px noise, ±40°:
| Scenario | Production DLT | Best subset | + L/R swap check |
|---|---|---|---|
| Clean | 6.6 mm, knee 1.1° | 6.6 mm | 6.6 mm |
| Knee outlier 40/80/150 px in one view | 71/145/293 mm, knee 9/17/33° | 19/15/24 mm, knee 3.2/2.3/2.6° | same |
| Full L/R swap in one view | 240 mm all keypoints, knee separation off 187 mm | 166 mm (worse) | 6.7 mm |
| Legs-only swap in one view | 240 mm legs | 170 mm | 6.6 mm |
Best subset: keep all 3 views unless one view's residual > 10 px, then lowest-residual pair. Swap check: 7 hypotheses (none + full swap + legs-only swap per view). Heavy tail: best subset leaves 1.7–4 % frames > 50 mm; tie-break by closeness to previous frame's estimate → 9 mm mean, 0 % > 50 mm. Confidence-weighted DLT helps only if the outlier's confidence drops (Exp1: usually doesn't); Hartley normalization ≤ 6 %; depth reweighting 0. **IK interaction** (code path CONFIRMED, end-to-end SUSPECTED): flagged point keeps conf×0.1 ≤ 0.1, below IK minimum (`analytical_ik.py:50`) → `_compute_knee_flexion` returns 0.0° → outlier knee reads as a straight leg mid-squat. Cost (Mac numpy, 21 kpts × 3 views): plain DLT 0.11 ms, best subset 0.52 ms, swap + best subset 1.41 ms (~0.3 ms batched).

**S0-5 HIGH, CONFIRMED — triangulator confidence has no metric meaning** (`triangulator.py:125-131`). Ranks errors (Spearman 0.75) only by flagging gross outliers; 23 % of points get conf < 0.1 with median error 167 mm and stay in the skeleton; 15 px cutoff ignores person size/distance; errors common to all views invisible. **Proposal:** uncertainty `u = max(σ̂_px, 3 px, or 6 px with only 2 views) · sqrt(trace((JᵀJ)⁻¹))`; `conf = 1/(1 + (u/0.02 m)²)` (0.5 ≈ 2 cm, 0.1 ≈ 6 cm); Kalman R = 0.02²·(1/conf − 1); no consistent subset → conf 0. Calibration vs true error, 3 views: error/u median 0.66, p95 1.69; two-view points poorly calibrated (p95 5.2×) → cap their confidence.

**S0-6 HIGH, currently hidden, CONFIRMED — two clocks in 3-camera mode.** Frames stamped `perf_counter` (`multi_capture.py:94`); dropout hold (`pipeline.py:540-557`) uses `time.time()`. Real `DerivativeTracker`: true 60°/s knee reads 600,033°/s on the first real frame after 2 hold frames, decaying 420k, 294k → predictive estimator saturated ~0.8 s. `ipc_bridge.send_fault` (`:145-147`): `now − last_send` = −1.79e9 s after a fault on a held frame → that fault's cue muted for the rest of the process. Hidden only because sigmoid confidence means holds never fire; fixing S0-2 exposes it.

**S0-7 MEDIUM, CONFIRMED — hip re-centering inside the triangulator** (`triangulator.py:134-137`). Untriangulated points at (0,0,0) = hip centre; if either hip missing, centering skipped → skeleton jumps 0.23 m; centering raises per-keypoint noise 4.0 → 5.0 mm and correlates L/R ankle noise 0.00 → 0.37; 80 px hip outlier in one view moves the feet 126 mm; still feet appear to move 0.85 m/s; 0.50 m hip travel lost. **No consumer needs hip-centered input** — `get_rep_signal`, BiLSTM features, IK, 3D valgus, trajectory `hip_y`/`knee_y` (used only as L−R and hip−knee differences), `_ground_and_center` (re-grounds every frame) all use relative quantities. `assessment_logger.py:170` `hip_y_avg` ≈ 0 in any hip-centered frame. GroundClamp's ankle equalization only makes sense in world coordinates (in centered frame it erases pelvic list). **Fix:** output world coordinates, run the pre-IK chain in world coordinates in 3-cam mode, center once right before IK; MediaPipe stays hip-centered → chain needs a frame flag.

**S0-8 MEDIUM, CONFIRMED — heels and small toes dropped** (`rtmpose.py:268-276`, `triangulator.py:27`). `standing_gate` heel-rise returns `[]` for ≤ 20 kpts (`:70`) → flat-foot gate silently off in 3-cam mode; no 3D heel-rise input. **Change:** map halpe 24/25 → 19/20, `NUM_KEYPOINTS = 21`. Nothing breaks: mediapipe_fallback already 21, create_empty 21, pose_validation & keypoint_corrector accept 19/21, bone_constraints & ground_clamp guard by count, bridge slices `15:`, demo_ws_bridge & test visualizer slice `[:19]`, BiLSTM uses indices ≤ 16, capture_audit has only a 2D placeholder. Crop scores: heels 0.77/0.82, small toes 0.82/0.91.

**S0-9 MEDIUM, CONFIRMED — argmax staircase.** Full-frame steps 3.3 px x / 1.4 px y; 38 % of frame-to-frame joint displacements exactly zero. On crop logits parabola fit, local soft-argmax, mmpose DARK equivalent: staircase 0.54 → 0.36 px, motion noise 0.44 → 0.35 px; parabola 0.08 ms for 3 views, DARK 1.7 ms. Crop matters far more than decoder.

**S0-10 MEDIUM, CONFIRMED synthetic (quantified only) — calibration model bias.** True focal 0.55–1.0w vs 0.8w guess, proportions ±5 %, face in front of body plane, imperfect T-pose: wrong focal absorbed as wrong camera distance (0.55w → 5.06 m estimated vs 3.5 m); hip-centered keypoints off 17–29 mm systematically; knee flexion mean 0.9–2.2° (p95 up to 5.3°); trunk-lean bias −1.1 to −1.8°; dorsiflexion bias down to −2.2°; bone lengths off 1–4 % mean (p95 up to 9 %). Similar size to random noise, not filterable, far smaller than S0-1/2/4 → keep calibration for now; don't treat calibrated bone lengths as ground truth. Lens distortion near frame edges (4–9 px at feet) SUSPECTED.

**S0-11 LOW, CONFIRMED — documented axes wrong.** solvePnP returns a proper rotation → X-left + Y-down force Z away from the front camera (Exp6 recovers front camera at z = −3.49, truth −3.5). Docstrings `calibration.py:9-14`, `analytical_ik.py:46`, `bridge.py:24` say Z toward camera. Actual frame matches real MediaPipe → modes agree. "Forward lean = positive Z" at `analytical_ik.py:417` SUSPECTED inverted in both modes (IK owner).

**S0-12 LOW, CONFIRMED — `frame_index=0`** flows into `FaultEvent.frame_index` (`fault_types.py:180`) and BiLSTM rep start/end frames (see wiring W1 for the cooldown consequence).

## Verdicts (3-camera)
| Component | Verdict | Action |
|---|---|---|
| RTMPose preprocess | REBUILD | Tracked person crop |
| Keypoint decode | FIX | Parabola sub-pixel refinement |
| Keypoint confidence | FIX | Raw score clipped [0,1] |
| Keypoint mapping | FIX | Add heels (21 kpts) |
| DLTTriangulator solve | FIX | Best subset + previous-frame tie-break + swap check; skip normalization |
| Triangulator confidence | REBUILD | Metric uncertainty; conf 0 for rejected |
| Re-centering in triangulator | REMOVE (move) | World-coordinate output; center before IK |
| `get_synced_frames` | REBUILD | Proposed sync policy, one clock |
| TPoseCalibrator | KEEP for now | Later: refine focal + bone lengths across 3 views |
| `multi_camera.get_pose` | FIX | Real frame counter, one clock |
Single-cam MediaPipe unaffected by S0-2/4/9/10; its hip-centered output means floor/foot-contact filters should run only in world mode.

## Priority (accuracy gain per effort) — 1–5 required before pre-IK tuning is meaningful
1. Raw score instead of sigmoid — 1 line, 0 cost; no-person keypoints passing 26/26 → 0/26.
2. Tracked person crop — ~40 lines, +0.3 ms/view; 3D body error ~60 → ~7 mm; knee 6.0 → 1.2°; hip-width bias +118 mm → 0.
3. Sync policy + one clock + frame counter — ~50 lines; frame loss 45 % → 0–11 %; removes velocity spikes and permanent cue muting.
4. Best subset + previous-frame tie-break — ~40 lines, +0.4 ms; outliers 145–293 → ≤ 24 mm (9 mm with tie-break); knee 17–33° → 2–3°.
5. Metric confidence, conf 0 for rejected — ~30 lines; real error estimate for Kalman/bone calibration; stops 0° knee drop.
6. Parabola decode — 10 lines, −20 % motion noise.
7. Swap check — ~0.3 ms batched; 240 → 7 mm for one-view swaps; log firing frequency (unknown at ±40°).
8. World-coordinate output — decide before redesigning ground clamp / velocity clamp / smoothing.
9. Heels — 5 lines; re-enables heel-rise gate.
10. Calibration refinement — later.
Skip normalized/reweighted/confidence-weighted DLT for now.

With 1–5 the chain gets ~7 mm near-random noise per keypoint at a steady 30 Hz with meaningful confidence. Today: ~60 mm systematic error, 12–29 cm outliers and swaps, 45 % frame loss, useless confidence.
