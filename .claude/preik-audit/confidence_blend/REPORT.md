# ConfidenceBlender Audit

(Saved by lead from the agent's final message; subagent sandbox blocked writing REPORT.md.)
Folder scripts: `e1_time.py`, `e2_lag.py`, `e3_semantics.py`, `e4_latch.py`, `e5_stack.py`, `e6_gate.py`, `e6b_gate_world.py`, `keypoint_gate_proto.py`; results `e1_results.json` … `e6b_results.json`; `PROGRESS.md`. No repo files changed; the 5 existing unit tests pass.

**Data/method:** E0 (real) = RTMPose production decode on 3 real squat videos. Real single-cam weights worked back from 3 recorded MediaPipe sessions (`user_test_runs/*/pipeline_inspect`) where the blender was live — **those sessions looped at 11.5–11.9 fps, not 30**. Simulation: squat session through the production noise path (3 cams, 2D noise, quantization, DLT, triangulator confidence formula, hip re-centering) with real E0 per-camera confidence traces; generator copied from the smoothing agent (`synth_snapshot.py`). Production classes throughout. No real 3-camera recordings exist → thresholds need checking on real replay data.

## Verdict
- **Triangulated: REMOVE.** Replace its one useful job with a keypoint gate inside the triangulator, before hip re-centering (design below). No smoothing in this stage.
- **Single-camera MediaPipe: REMOVE.** Worked-back weights are 1.0 on 80–96 % of leg frames, ~0.9 on heels — it adds only the risks below.
- **Coupling:** blender lag currently hides part of the predictor's bottom-of-rep overshoot. At 11.6 fps predicted peak knee flexion is +6..+11° over truth without the blender, −0.2..+8° with it; DepthRule reads predicted angles. **Remove the blender in the same change that fixes the predictor (F17)**, or shallow reps can pass depth.

## Headline (triangulated, 3 px noise; confidence p5/50/95 = 0.41/0.49/0.57 → median weight 0.49, max ever 0.73; the 0.9 "fully trust" cutoff is never reached)
| Metric (left knee flexion) | raw | blender | One Euro, same noise reduction |
|---|---|---|---|
| Static leg noise RMS | 1.58 cm | 0.89 cm | 0.91 cm |
| Knee/ankle lag moving > 0.5 m/s at 30 / 15 / 11.6 fps | 0 | 3.0/3.4, 5.3/5.8, 6.7/6.9 cm | 1.4/1.6, 2.2/2.1, 2.8/2.5 cm |
| Knee error moving > 100°/s at 30 / 15 / 11.6 fps | 2.1 / 1.9 / 2.4° | 4.8 / 9.4 / 12.1° | 2.2 / 3.4 / 4.4° |
| Session knee RMSE at 30 / 15 / 11.6 fps | 2.56 / 2.62 / 2.53° | 3.23 / 5.61 / 7.33° | 2.01 / 2.51 / 3.07° |
| Peak knee per rep, noiseless, 30 / 11.6 fps | 0 | −0.6..−1.6 / −2.7..−7.4° | −0.1..−0.3 / −0.3..−1.1° |
| Depth undershoot (hip-to-ankle), 30 / 11.6 fps | 0 | up to 1.5 / 7.6 cm | up to 0.3 / 1.1 cm |

## Findings
1. **CB1 HIGH (critical ≤ 15 fps), CONFIRMED — in triangulated mode it is a fixed α≈0.5 smoother that hides depth.** RTMPose confidence stays 0.53–0.72 and the triangulator scales it down by reprojection error, so weight never reaches 1. Same static noise ratio (0.58) and lag (3.0 cm) as fixed α=0.5. Cuts static noise 42 % yet session knee RMSE worsens at every loop rate. After post-IK filters at 11.6 fps: fast-descent error −23.4° with vs −12.0° without; diagnosis bottom frame −7..−15° with vs −4..−9° without. Refs `confidence_blend.py:66-79`, `triangulator.py:125-131`.
2. **CB2 HIGH, CONFIRMED — not time-aware** (never reads timestamps; smoothing per call). Lag at weight 0.49: 35 ms @30 fps, 69 ms @15, 90 ms @11.6; −3 dB cutoff 3.3 → 1.3 Hz. Duplicates: 25 ms loop → 26 % duplicates, lag varies 1.2–2.7 cm; 16.7 ms loop → 50 % duplicates, noise ratio 0.70 vs 0.57; 86 ms loop → 1.6 frames skipped/iteration, lag 7.2 cm. Gaps: state survives any absence — after 10 s, returning mid-squat gives knee errors −35.7, −17.7, −8.9, −4.5° on the first four frames. Dropout-hold weights 0.37/0.24/0.12/0/0 → ~1 cm during hold, 3–6 frames recovery (mixed clocks don't affect the blender itself).
3. **CB3 HIGH, CONFIRMED — triangulated confidence carries no accuracy information except outlier-ness.** 3 % of 2D detections off ~40 px. 3 cams: every confidence bin 0.3–1.0 has the same median 3D error (~1.1 cm); below 0.1 median 8.7 cm. 95.5 % of > 5 cm errors get weight < 0.2, but good points (< 2 cm) also get median weight 0.49 → just delayed. 2 cams: only 61–72 % of > 5 cm errors caught (errors along the baseline don't show in reprojection); 0.3–0.4 bin p95 error 15–24 cm. The 0.1–0.9 mapping suits MediaPipe visibility, not triangulation. Reprojection threshold (3 cams): at 3 px noise, 8 px catches 96.5 % of bad points, 0.1 % false flags; at 6 px noise 8 px false-flags 12 % → use 10–12 px.
4. **CB4 MEDIUM — one NaN/inf coordinate poisons the set** (failure confirmed; trigger — DLT w==0 — suspected rare). A NaN/inf point at confidence 0.06 makes that keypoint NaN on every later frame; IK knee flexion NaN until set reset. Triangulator doesn't drop it: NaN/inf reprojection falls into the 0.1 branch (`triangulator.py:126-131`).
5. **CB5 MEDIUM, CONFIRMED — output confidence contradicts output position.** Passes the current frame's confidence with mostly the previous position (`confidence_blend.py:85-90`). Its 0.1 floor equals the IK/valgus floor (`analytical_ik.py:52,160`): at confidence 0.10 the position is frozen but IK uses it (jump standing → bottom reads knee 4° vs 120° true; 0.14 → 8°; 0.3 → 28.5°). Keypoints seeded at origin or returning after absence report confidence 0.43–0.55 while 50–100 cm wrong.
6. **CB6 MEDIUM, CONFIRMED — missing keypoints stuck at hip midpoint or stale positions.** Untriangulated points are exactly (0,0,0) and not re-centered (`triangulator.py:89,137`); the first frame after reset is stored regardless of confidence (`confidence_blend.py:59-64`). Only readiness-unchecked keypoints (face, arms, toes — toes feed hip rotation) can arrive missing on that frame. Toe seeded 101 cm off: 52, 23, 14, 8, 4 cm with blender alone; 12 frames (400 ms) with the clamp (the crawl is the clamp's). Below 0.1 confidence the point stays at the hip midpoint and confidence-ignoring code reads it: rep signal (`squat.py:110-117`), trajectory samples, bottom/standing keypoints. Ankle lost 1 s mid-descent: held at its standing hip-relative position while hips move → during loss mean 23 / max 57 cm (raw bad points 17 / 38 cm); after return (blender + clamp) 50, 43, 35, 27, 18, 9 cm.
7. **CB7 MEDIUM, CONFIRMED — velocity clamp doesn't stop the blender dragging an outlier out.** 25 cm one-frame spike: clamp alone 8.3 cm for one frame; blender + clamp 8.3, 4.7, 2.3, 1.2 cm (blender stores the unclamped value). Hip re-centering before filtering spreads a hip outlier to every keypoint: same gate after re-centering → knee RMSE 4.70; before → 3.25.
8. **CB8 MEDIUM, CONFIRMED — stacks with post-IK smoothing (F17).** 30 fps: fast-motion knee error 1.8 → 5.4°, predicted-angle RMSE 3.69 → 4.28°. 11.6 fps: 2.5 → 12.5°, 5.1 → 8.6°. The velocity clamp never fired on realistic noise.
9. **CB9 LOW — bone lengths.** Noiseless femur mean −0.15 / −0.49 / −0.75 cm at 30 / 15 / 11.6 fps; tibia jitter ~±1 cm from per-joint weights; small vs raw noise (±1.6 cm).
10. **CB10 LOW — hygiene.** Style-rule violations (no `from __future__`, `Optional`, Args docstrings); comments say 17 keypoints (triangulated 19, MediaPipe 21); frames without triangulated hips aren't re-centered yet are blended against hip-centered state; 5 tests only check step responses on 17 kpts — nothing on timing, duplicates, (0,0,0) seeding, NaN, or the IK confidence-floor overlap.

## Replacement: `KeypointGate` (prototype `keypoint_gate_proto.py`, numpy only, 33 µs/frame vs blender 38 µs)
1. Triangulator outputs per keypoint: world position, number of views, mean reprojection error (px).
2. Valid iff ≥ 2 views, finite coordinates, reprojection ≤ threshold (start 8 px; set ≈ 3× median reprojection error on real replay data).
3. Valid keypoints pass through unchanged, zero lag.
4. Invalid keypoint seen valid within the last 0.10 s → hold last WORLD-frame position with IK-usable confidence (feet don't move in a squat). Plain hold ≥ velocity extrapolation: 2.96 vs 3.25° @30 fps; 4.3 vs 4.2° @11.6 fps.
5. Otherwise confidence 0 at last valid position; never store (0,0,0) or NaN; a returning keypoint snaps straight back.
6. State updates only from valid samples; gap > 0.5 s or backwards clock clears state; duplicates pass through unchanged.
7. Re-center on hips AFTER the gate; invalid, unholdable hips → return None (dropout path).
8. Everything downstream (incl. rep signal, trajectory samples) treats confidence 0 as missing.
9. Tests: valid frame unchanged; hold ≤ 0.1 s then confidence 0; (0,0,0)/NaN never stored; duplicates + backwards clock; snap back after 10 s; invalid hips → None.

**Measured, gate before re-centering vs blender, 3 % outliers:**
| Case | Knee RMSE gate / blender / raw | Fast-motion knee error gate / blender | Frames > 10° off gate / blender |
|---|---|---|---|
| 30 fps, 3 cams | 3.25° (2.96° plain hold) / 4.05° / 6.86° | 2.6° / 5.8° | 13 / 36 |
| 11.6 fps, 3 cams | 4.2° / 9.0° / 6.9° | 3.5° / 13.9° | 13 / 121 |
| Clean, 30 / 11.6 fps | 2.71 / 2.78° (= raw) vs 3.40 / 7.17° | — | — |
| 2 cams, 30 / 11.6 fps | 4.65 / 5.18° vs blender 3.97 / 7.31° | — | — |
| Ankle lost 1 s | 1.5 cm during, 2.0 cm on return | 23 cm during, 50→9 cm on return | — |
Blender still wins with 2 cameras at 30 fps → fix is better camera selection in the triangulator, not a smoother.

**If position smoothing is wanted:** One Euro at equal noise reduction beats the blender everywhere; gate + One Euro → knee RMSE 2.69° @30 fps, 4.07° @11.6 fps. A confidence-aware causal constant-velocity Kalman at equal noise reduction **overshoots at reversals: peak knee +2.3..+5.5°, depth +2..+4.5 cm — unsafe for depth, not recommended** (causal). If a Kalman is used anyway: R scaled by reprojection error and view count, statistical outlier gating, dt from timestamps, high process noise. (Lead note: research agent's 3-frame fixed-lag Kalman reported 97 % peak retention; the causal vs fixed-lag distinction must be settled on the harness.)
