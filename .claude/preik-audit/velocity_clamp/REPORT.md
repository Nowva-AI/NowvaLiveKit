# VelocityClamp + Timestamp / Frame-Delivery Audit

(Saved by lead from the agent's final message; subagent sandbox blocked writing REPORT.md. Scripts, outputs, PROGRESS.md in this folder. Real repo classes on synthetic data unless SUSPECTED. No repo files changed.)

**Setup:** hip height min-jerk curve, ~0.6 m travel, 1.885 m user; 3 cameras at 3.2 m, 0° and ±50°, perfect calibration; 2D Gaussian noise 2/3/5 px through real `DLTTriangulator`. Tempos (peak hip speed): heavy 0.8 m/s, bodyweight normal 1.3, loaded fast 1.7, bodyweight fast 2.1, explosive 2.9 (upper bound).

## 0. Verdicts
| Component | Verdict (3-cam triangulated) |
|---|---|
| VelocityClamp | **REBUILD; disable meanwhile.** Does nothing on realistic clean data; moves toward outliers instead of rejecting; state = clamped output → every hold/reset/re-acquire becomes a crawl; distorts fast movement |
| Replacement | **InnovationGate** (§5): reject/predict/re-acquire in world coordinates before hip re-centring; 0 false rejections clean; removes 20–60 cm and hip outliers; 0.04 ms/frame |
| Capture clock (perf_counter vs time.time) | **CRITICAL.** Triangulated + BiLSTM on → every rep ends the set |
| `get_synced_frames` | **HIGH.** None on 26–41 % of calls → only 14–18 Hz of 30 Hz processed; can return same frame twice |
| Dropout hold | **MEDIUM.** 5th held frame conf 0 → IK 0°; different clock. Latent triangulated today, live MediaPipe |
| Filters dt ≤ 0 → 1e-6 | **LOW.** Acceleration spikes 1e6–1e8 deg/s² |

**Single-camera MediaPipe:** clocks consistent (T1, T5 n/a); T4 live; duplicates arrive with same pose but new timestamp → clamp fires falsely next frame (16 frames with a 15 fps camera). Use the same gate with base radius 12–15 cm and visibility as confidence; no clamp there either.

## 1. Findings
**T1 CRITICAL, CONFIRMED.** `ml/inference.py:72` feeds skeleton perf_counter timestamp via `bilstm_counter.py:163` into `RepData.end_time` → `session_tracker.py:111,169` stores as `last_rep_time` → `pipeline_process.py:1287` calls `check_set_timeout(time.time())` → 1.79e9 − 8e4 always > 30 s → **set ends and readiness gate resets after every rep.** YAML default has BiLSTM on. exp9 (real counter + tracker): 3 reps → 3 set_complete; with time.time skeleton clock → 0. Fix F2.

**T2 HIGH, CONFIRMED.** `multi_capture.py:113-141` always picks newest primary frame whose partners often aren't read yet → None; then `pipeline_process.py:1362-1365` sleeps a full frame period. exp7b (real class, fake 30 fps cameras, 8±2 ms read latency): None 25.6–40.8 % of calls (sim model 34 %, analytic 36 %); only 14.4–18.0 Hz processed; capture-time spread p95 23–25 ms; duplicates 0–0.5 % @30 fps, 7 % @24, 42 % @15 (low light). Fix F1.

**T3 HIGH, CONFIRMED, cross-cutting (IK/triangulation own).** `triangulator.py:126-131` conf × 0.1 when reprojection ≥ 15 px; `confidence_blend.py:85-90` passes raw confidence; `analytical_ik.py:52,160,260-262` returns 0.0 when conf < 0.1 → gross outlier gets conf 0.06–0.09, blender holds position but IK outputs knee flexion 0. **Live today.** Raw IK knee error 92–121° (exp4, exp8c). One zeroed frame mid-descent: filtered knee −13°, predicted −39°, velocity +92 → −54 deg/s (exp1b). **Fix:** IK holds previous angle; any stage substituting a position must output confidence ≥ IK minimum.

**V1 HIGH, CONFIRMED.** `velocity_clamp.py:89-98` stores clamped output as state. 10-frame leg swap mid-descent: blend alone recovers in 1 frame (7.4 cm); blend+clamp 4 frames (26.9/21.2/15.0/7.2 cm), 14 frames knee error > 5° vs 11. First frame after reset with conf-0 toe at origin: 92 → 8.5 cm over 12 frames vs 4 with blend alone (exp6). Persistent wrong solution adds 3 extra bad frames on return.

**V2 MEDIUM, CONFIRMED.** Single-frame outlier moved 8.3 cm toward, not rejected. 60 cm knee outlier standing: blend+clamp 9.2/16.0/9.4 cm over 3 frames, 35–39° knee error (blender keeps 75 %); clamp alone 8.9 cm 1 frame; gate ≤ 2.6 cm, ≤ 5°.

**V3 MEDIUM, CONFIRMED.** 2.5 m/s (`config/biomechanics.yaml:54`) clamps real fast movement. Explosive: ~30 % frames clamped, knee lag 9–12 cm, knee angle distortion 16–20°, femur off ≤ 2.8 cm, knee separation off ≤ 3.8 cm, clean knee RMS 3.9° blend alone vs 7.0° blend+clamp. BW fast arms forward: wrists clamped 56–59 frames, 22 cm lag, forearm off 5.8 cm. Static 5 px clamp alone: 10.7 % frames clamped. Bottom depth never changed.

**V4 MEDIUM, CONFIRMED.** 30 cm outlier on one hip shifts all 19 keypoints 15 cm via re-centring (`triangulator.py:134-137`); blend+clamp 16–18° knee error over 2 frames; gate ≤ 3 cm.

**T4 MEDIUM, CONFIRMED.** Latent triangulated (sigmoid conf never < 0.3), live MediaPipe and once raw confidences are used. `pipeline.py:543` decay = 1 − 5/5 = 0 → 5th held frame all conf 0 → IK all zeros. Filtered knee 40.9 → 27.7°, predicted 55.9 → 16.9° (exp1b); 79 → 44° faster phase (exp1). Trunk defaults 180° → no false forward-lean.

**T5 MEDIUM, CONFIRMED, latent triangulated.** Held frames `time.time()` (`pipeline.py:471,550`) vs live `perf_counter` (`multi_capture.py:94`); see §3.

**T6 LOW, CONFIRMED.** Duplicates + dt ≤ 0 → 1e-6 (`filters.py:114`, `derivatives.py:134`): acceleration p99 600 → 5.6e6–1.9e7 deg/s²; predicted knee RMS +2–2.7° (max +4–11°); one spurious hip-counter bottom entry @24 fps; rep counts unchanged; no squat rule reads acceleration today.

**T7 LOW.** `reset_readiness_gate` doesn't reset JointAngleFilter/DerivativeTracker; after 90 s rest derivative average carries 4.15 deg/s stale velocity; harmless.

**V5 LOW.** Clamp ignores confidence; `VelocityClampConfig.target_fps` unused (pipeline passes `pipeline.target_fps`); 6 tests pass, none cover dt ≤ 0, gaps, recovery, conf-0 seeds. Time-awareness works with consistent clock (dropped frames: 12/124 clamped vs 45 if dt ignored).

**Benign:** readiness-gate early returns (gate latches; clamp + blender reset together); identical-timestamp duplicates harmless for the clamp itself; hip rep counter and PipelineFrame use time.time() so the hold clock mix doesn't reach them.

## 2. Is 2.5 m/s right for hip-centred data?
Peak speeds m/s (exp3), hip-centred / world where two numbers:
| Keypoint | Heavy | BW normal | Loaded fast | BW fast | Explosive |
|---|---|---|---|---|---|
| Ankles/toes (world 0) | 0.86 | 1.44 | 1.85 | 2.31 | 3.24 |
| Knees | 0.86 / 0.48 | 1.43 / 0.80 | 1.84 / 1.03 | 2.30 / 1.29 | 3.22 / 1.80 |
| Wrists holding bar | 0.49 | 0.82 | 1.05 | 1.31 | 1.84 |
| Wrists arms forward | – | 2.45 | – | 3.94 | 5.51 |
- Hip-centring leaks hip-midpoint noise into every point: per-axis noise @3 px 0.87 cm hip-centred vs 0.71 world.
- Static frame-to-frame noise p99 2.8/4.3/7.05 cm @2/3/5 px; max @5 px 9.95 cm > 8.33 cm per-frame limit.
- **No single speed limit works:** 2.5 m/s too loose to catch < ~8 cm, too tight for fast BW squats and arm swing.
- **Prediction error depends on acceleration, not speed:** clean CV-predictor innovation p99.9 6.95 cm @3 px, 9.33 cm @5 px (world); ankles/toes 4.5–7.5 cm even at 15 Hz.

**Clamp vs realistic triangulation failures:** 1-view 80–150 px errors or 1–2-view leg swaps → conf 0.08, blender holds, clamp adds only the release crawl. Knee in only 2 views, one bad by 80 px → 25 cm jump at conf 0.37–0.39; blender → 9.4–11.7 cm; clamp improves ≤ 0.6 cm. Consistent 30 cm error → clamp follows after 4 frames. Single-frame outlier d → clamp alone min(d, 8.3 cm) 1 frame; production chain 2–3 frames, peak 16 cm. Real jump D → ~D / (8.3 cm − per-frame motion) frames. Rigid segments: femurs stretched 0.5 cm (fast, ≤ 3 px), 2.2–2.8 cm (5 px or explosive); forearms 5.8–10.5 cm (arm swing); on outliers attached bones change up to 8.3 cm.

## 3. What each temporal filter sees
**Dropout mid-descent (exp1), production clocks vs consistent-clock counterfactual:**
- VelocityClamp: entering hold dt = +1.79e9 → can't clamp; leaving dt = −1.79e9 → falls back to 33 ms regardless of gap; 7-frame case 16 keypoints clamped falsely (0 consistent); ankle error 20.7/15.6/10.2 vs 10.9/5.8/2.7 cm. Even consistent, holding positions → 15 clamped keypoints on release.
- OneEuro (JointAngleFilter): entering alpha → 1, jumps to raw held angle (+10.7°); leaving dt → 1e-6, output frozen one frame (84.0° vs 89.8° true).
- DerivativeTracker: leaving → acceleration spikes 5.2e6 / 8.2e7 / 6.9e7 deg/s² (3/5/7-frame holds), decaying ×0.7/frame, > 1e5 for ~11 frames; velocity +50–150 deg/s over ~8 frames.
- Predictive: capped 15°; 5-frame case dominated by T4 (28.8° predicted vs 104.8° true).
- BiLSTM rep times can mix clocks (SUSPECTED).
**Duplicates, identical timestamps (exp2):** clamp harmless; blender blends same measurement twice; OneEuro no-op; DerivativeTracker +8.7 deg/s velocity, 2.6e6 acceleration spike per duplicate; predictive +1.8°; hip counter velocity dips; BiLSTM window, min_rep_frames, gate consecutive counters count duplicates as new (SUSPECTED, code reading).

## 4. Minimal fixes
- **F1 capture layer (`multi_capture.py`):** per-camera sequence counter; return newest complete set whose primary sequence hasn't been returned; if none, poll every 2 ms up to 40 ms instead of None; return sequence number and use as `frame_index` (fixes wiring W1); remove sleep-based pacing in multi-camera mode. Prototype (exp7c, real class): None 0.4–2.1 %, 0 % duplicates, 24.4–26.9 Hz processed. Optionally `max_sync_delta_ms` 17.
- **F2 one clock at capture:** in `MultiCameraCapture.start()` compute `offset = time.time() − time.perf_counter()` once; stamp `perf_counter() + offset`. Fixes T1, T5 with no consumer changes. Longer term: one monotonic clock, convert to wall time only at IPC edge.
- **F3 `pipeline.py:537-565`:** held-frame confidence never 0 (decay = 1 − n/(MAX+1)) or skip IK + rules on held frames; hold last output, not raw skeleton; clear `_last_valid_skeleton` on reset.
- **F4 filters:** dt ≤ 0 → return previous output, no state update; dt > 0.5 s → re-initialise; DerivativeTracker average dt-aware or reset at set boundaries.
- **F5 single-camera loop:** frame id; skip unchanged ids.

## 5. Replacement: InnovationGate (prototype `gate_proto2.py`; exp8b/8c/8d)
**Placement:** new `triangulation/temporal_gate.py`, called in `MultiCameraPoseProvider.get_pose` between a world-frame `triangulate_world()` and a separate `recentre()` using gated hips. World coordinates keep feet static and stop hip errors spreading. Hip-centred drop-in works but less robust.

**Per frame, per keypoint:**
1. dt ≤ 0 → re-emit previous output; clip dt to 0.25 s; re-init if dt > 1 s.
2. Valid measurement: ≥ 2 views, reprojection < ~12 px (prototype: conf ≥ 0.15).
3. Predict p = x + v·dt from a fast α-β tracker (α 0.85, β 0.5), prediction only.
4. Gate: accept if |z − p| ≤ r0 + ½·a_max·τ² (τ = time since last accepted). r0 8 cm world (10 hip-centred), a_max 80 m/s² → 12.4 cm @30 Hz; covers clean innovation max 8.8 cm (3 px) / 11.5 cm (5 px) + 72 m/s² synthetic knee peak. Optional a_max per part: feet 15, hips 30, knees 80, head/shoulders 40, arms 60.
5. Accept → output z unchanged (no lag).
6. Reject → output prediction, marked predicted, conf ≥ 0.15 (IK never zeroes); keep velocity 2 frames then damp ×0.7/frame.
7. Re-acquire after 2 consecutive rejected measurements agreeing within 10 cm → snap.
8. Timeout: snap after 0.2 s; no valid measurement for 0.3 s → conf 0.
9. Never-valid keypoints don't seed state.
10. Hip-centred version only: subtract common-mode shift only when one hip moves with the crowd and the other opposite (first prototype diverged without this check).

**Measured vs ground truth (3 px):**
| Scenario | Gate | Production blend+clamp |
|---|---|---|
| Clean: 12 sequences, all tempos, 3/5 px, arms | 0 rejections | n/a |
| Single-frame 20–60 cm knee outlier | ≤ 3 cm, ≤ 5.2° | standing 8.4–16 cm, 19–39° |
| 30 cm hip outlier | ≤ 3 cm, ≤ 5.2° | 16.4° |
| 1-view / 2-view-only 80 px | ≤ 3.5 cm, 0 IK-zeroed | up to 9.6 cm, 17.6°, IK zeroed |
| 10-frame leg swap mid-descent | world 9.8 cm (hip-centred 14.3), instant recovery | 33.8 cm + 3-frame crawl |
| Conf-0 seed | no crawl | 12-frame crawl |
- Unfixable temporally: consistent 30 cm error followed after 2 frames → needs view selection / swap check at triangulation.
- Blind spot: errors < ~8 cm.
- Depends on F1: at 15 Hz radius grows to 26 cm.
- Tuning sensitive: hip-centred r0 8 cm, a_max 40 diverged (knee error up to 114°).
- Cost 0.039 ms/frame numpy Mac, ~0.2 ms Jetson (estimate).
- Tests: dt ≤ 0 and > 1 s; single outlier rejected; re-acquire after 2 frames; timeout 0.2 s; conf 0 never seeds; hip outlier doesn't move other keypoints; 0 rejections on explosive + arm-swing; predicted keypoints never below IK confidence minimum.

## 6. Caveats
Synthetic kinematics (knee accelerations probably upper bound); isotropic Gaussian 2D noise (real full-frame-squash RTMPose errors biased, p95 61 px per triangulation_frontend; no recorded 3-cam data); frame delivery used fake cameras with 8±2 ms read latency (real latency may shift None rate, but the cause — newest primary's partners not read yet — doesn't depend on it).

## Files
Experiments `exp1`, `exp1b`, `exp2`, `exp3`, `exp4`, `exp4b`, `exp5`, `exp6`, `exp7`, `exp7b`, `exp7c`, `exp8`, `exp8a`, `exp8b`, `exp8c`, `exp8d`, `exp9` (`.py` + `.txt`/`.json`); prototypes/helpers `gate_proto.py`, `gate_proto2.py`, `synth_vc.py`, `chain_sim.py`, `tri_world.py`, `check_synth.py`; `PROGRESS.md`.
