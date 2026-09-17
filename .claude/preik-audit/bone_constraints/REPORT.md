# BoneLengthConstraints Audit

(Saved by lead from the agent's final message; subagent sandbox blocked writing REPORT.md. Milestone log in PROGRESS.md.)

**Evidence:** synthetic squat skeletons from `synth.py` (19 kpts, Y-down, 1.885 m user from `SEGMENT_RATIOS`) with injected valgus, hip shift, pelvic list, heel rise. Noise: iid 1/2/3 cm; "realistic" = 2 cm, 1.5× on depth axis, frame-to-frame correlation 0.6, 1 % outliers of 10–30 cm lasting 3 frames; "drift" variant adds pose-dependent hip/knee keypoint drift + 4 cm spine shortening at bottom. Metrics via production `AnalyticalIKSolver` and `TriangulatedValgusEstimator`. CONFIRMED = reproduced numerically; E5 ran the real `BiomechanicsPipeline` with a fake 3D provider. No repo source files modified.

## Verdict: REBUILD
For triangulated input keep per-frame position correction OFF (current state). Replace the 30-frame standing calibration with a once-per-session robust estimator; use its segment lengths for athlete body measurements and a per-frame quality signal. Delete `valgus_scale` and `pelvis_tilt_coupling`; fix or delete `forward_lean_scale`. Stop wiping calibration every set. Single-cam MediaPipe path needs different handling (F9).

## Findings

**F1 HIGH, CONFIRMED, LIVE — calibration mode loses athlete params and saves None to DB.** Wait loop (`pipeline_process.py:495-505`) calibrates → reset at `:827` calls `pipeline.py:386`, wiping lengths → phase-2 loop (849-905) never calls `enforce()` → `:909` `_extract_athlete_params` returns None → `calibration_complete` without athlete params (922-923) → `coaching_service.py:908-935` → `calibration_utils.py:87` overwrites stored `athlete_params` with None → next session "No stored athlete params — diagnosis unavailable" (`:475`). Failed assessment round same via reset at `:787`. E5 confirmed on real pipeline (= wiring W2). **Fix:** snapshot body measurements once at calibration completion; don't reset segment lengths in `reset_readiness_gate`; upsert must refuse None over existing params.

**F2 HIGH, CONFIRMED — proportion scaling wrong direction, can exceed 180, compounds per set.** `forward_lean.py:50-53` multiplies 180=upright thresholds; fault fires when trunk angle ≤ mild (`:82`). Scale 1.2 → mild 174, 1.3 → 188.5: both fire on 100 % of in-rep frames. Scale 0.933 (this user, meant stricter) → 135.3, fires 0 %. With bone enforcement re-enabled, `*=` + `_proportions_applied=False` each reset (`pipeline.py:388`, 624-632) compounds (E5 real pipeline):
| After set | Valgus mild | Forward-lean mild |
|---|---|---|
| start | 12 | 145 |
| 1 | 14.07 | 130.3 |
| 2 | 14.72 | 122.8 |
| 3 | 16.12 | 114.7 |
| 4 | 16.89 | 108.7 |
By set 4 allowed lean 35° → 71° (rule effectively off). Today: applied once after wait loop, then `apply_calibration_to_rule_engine` (`calibration.py:107`) erases it (valgus 13.0 → 10.0); returning users never get it → only live during calibration-mode assessment. `test_body_proportions.py:234-258` asserts `135*1.2`, locking the bug in. **Fix:** delete rule scaling (diagnosis engine already models femur/torso in lean space, `parameter_deltas.py:33`) or scale in lean space from stored base thresholds.

**F3 HIGH, CONFIRMED synthetic, LATENT — current projection worsens pose and erases faults.** `BONE_PAIRS` (`bone_constraints.py:57-80`) processes shoulder→hip first → hips moved to fit torso. Single pass can't satisfy the shoulder/hip loop: right shoulder–hip off 4.1 mm mean, 32 mm worst (E8). Propagation (E2, true lengths): 3 cm left-shoulder error standing moves left hip/knee/ankle ~1.5 cm each, total movement 12.4 cm (4.1×), 3.1° pelvic-list error, 1.35 cm false heel rise; same error at bottom → 2.2° valgus error; 3 cm knee error at bottom → ankle moves 1.29 cm. Outliers (E8): 20 cm shoulder outlier moves left hip 10.2, knee 6.7, ankle 2.9 cm; knee outlier moves ankle 8.3 cm. Sequences (E3):
| Metric | Noise | None | Current |
|---|---|---|---|
| MPJPE (cm) | iid 2 cm | 3.09 | 3.09 |
| MPJPE (cm) | realistic | 3.88 | 3.98 |
| Hip-shift RMSE (cm) | iid 2 cm | 1.99 | 2.50 |
| Hip-shift RMSE (cm) | realistic | 2.19 | 3.01 |
| Pelvic-list RMSE (°) | realistic | 5.91 | 8.07 |
| Pelvic-list RMSE at bottom (°) | realistic | 5.03 | 8.58 |
Real 6 cm pelvic list reads 2.1–3.3 cm too small (35–55 % erased) — shoulder→hip isn't rigid (spine bends). With drift + spine flexion, depth bias at bottom +4.5 cm vs +2.3 cm uncorrected.

**F4 MEDIUM, CONFIRMED, LIVE — calibration fragile and feeds diagnosis.** 30-frame median right after standing gate latches, no per-frame validity (`bone_constraints.py:157-173`, 229-241). Accuracy (E4a), error mean (p95):
| Noise | Segment | Current 30-frame | Whole-assessment median |
|---|---|---|---|
| 2 cm, corr 0.6 | femur | 8.6 mm (20) | 2.3 mm (5.6) |
| 2 cm, corr 0.6 | hip width | 11 mm (23) | 5.2 mm (8.6) |
| 2 cm, corr 0.9 | femur | 14.5 mm (36) | 3.7 mm (8.9) |
Noise biases lengths long (hip width +5 mm @2 cm, +11 mm @3 cm). 15 cm knee outlier burst in window (E4b): femur +13.6 mm (10 frames), +78 mm (15), +126 mm (16); whole-assessment median ~+3 mm. Downstream: stance ratio (`bridge.py:173`), femur/torso ratio (`parameter_deltas.py:33`, 120°/unit → 2 cm femur error moves target trunk angle ~4°), `expected_knee_valgus_baseline`. Constraint sensitivity (E4c): +1 cm femur+tibia → +1.2° knee flexion, −0.55 cm heel rise, +1.3° dorsiflexion bias; correction breaks even with none at ~±1.2 cm, which current calibration exceeds at p95. Pose-dependent RTMPose keypoint drift SUSPECTED (drift model: femur −4.3 mm, tibia +7.2 mm) — measure on real recordings.

**F5 MEDIUM, CONFIRMED — `valgus_scale` has no valid basis for the 3D metric and hits its clamp.** `bone_constraints.py:263-265`, reference 0.556. This user with `SEGMENT_RATIOS` → 1.064; keypoint hip widths 0.18/0.20/0.22 m → 0.70 (clamp)/0.78/0.86 → 3D valgus thresholds 8.4/11.9/16.8. At same stance and knee swivel GS valgus depends on hip width, not by a multiplier, and direction flips (E6): neutral knee 0.18 m → −6.8°, 0.30 m → −1.7°; 25° swivel 0.18 m → +18.0°, 0.30 m → +9.8°. Duplicates diagnosis engine's additive baseline (`evidence_tests.py:45-47`). **Fix:** delete; if needed, per-user neutral valgus baseline.

**F6 MEDIUM — non-rigid pairs in `BONE_PAIRS`** (per-side shoulder→hip, shoulder width). Even the best projection (5-iteration PBD over all pairs) erases 17–30 % of pelvic list and doubles depth bias under drift. Rigid segments only → list bias −0.3 cm (E3b).

**F7 LOW.** Dead: `pelvis_tilt_coupling`, `AnalyticalIKSolver.set_body_proportions` (`analytical_ik.py:64-66`, 421); `pelvis_tilt` used only by `filters.py:283`, `predictive_state.py:105`, `pipeline_inspector.py:61`, never by a rule or diagnosis. `tolerance` docstring (104-105) wrong; fallback-direction comment "+Y upward" (201-203) but Y is down; forward-lean reference 0.90 vs keypoint proportions 0.84. Private `_calibrated_lengths` read externally (`pipeline_process.py:223-231` + scripts); 17-kpt `foot_avg_m` silently falls back to 0.26. Tests use Y-up skeleton (`test_bone_constraints.py:20-50`); none for 19 kpts, reset idempotence, fault preservation. 19-kpt (no heels) works correctly (E7).

**F8 INFO — reset today.** `reset()` also resets the shared standing gate (311-312). Nothing calls `enforce()` after the wait loop → no compounding today, but it causes F1.

**F9 — single-camera MediaPipe.** Sphere projection pulls reliable x/y along depth-noisy direction. xy noise 1 cm, z 6 cm (E7): depth RMSE 1.02 → 3.16 cm, heel rise 1.40 → 4.26 cm, hip shift 0.97 → 1.51 cm. Depth-only leg solve (keep x/y, solve depth from length) keeps x/y metrics exact, knee flexion error 13.4 → 7.2°, dorsiflexion 8.3 → 5.3°, but MPJPE 4.98 → 5.45 cm from depth sign flips → needs sign continuity from previous frame.

**Side finding for valgus owner (CONFIRMED synthetic):** `valgus.py:294` magnitude from ML axis vs leg plane, but `:304` sign from hip–ankle line. With 15° toe-out, medial knee swivel 0/10/20° reads −0.1/−4.7/+9.2° → **a 10° knee cave reads as varus**.

## Alternatives measured (realistic noise, all frames, vs no correction; E3/E3b)
| Method | Result | Side effects |
|---|---|---|
| (a) Tree rooted at pelvis | MPJPE ≈ none | Heel-rise bias +0.4 to +1.4 cm |
| (b) Two-bone knee solve (law of cosines) | Knee flexion RMSE 7.2° → 11.2° | Breaks near full extension: 3 cm hip error standing → 9.7° knee error |
| (c) Symmetric confidence-weighted split, all pairs | MPJPE −12 % | Pelvic list 17–30 % erased |
| (d) Soft deadband 2.5 SD | ≈ none | — |
| (e) 5-iteration PBD, all pairs | MPJPE −13 % | List erased; depth bias doubles under drift |
| (e′) 5-iteration PBD, rigid only (hip width, femurs, tibias, feet) | MPJPE 3.88 → 3.64, knee 4.16 → 3.50 cm, ankle 4.33 → 3.52 cm, knee flexion 7.23 → 6.16°, depth 2.15 → 1.65 cm; list preserved (−0.31 cm) | Heel rise +0.66 cm (+1.13 with drift), knee deviation −0.5 cm; needs calibration error < ~1 cm |
| (f) Rigid-residual outlier gate | Catches 17 % (2 cm noise) to 44 % (1.5 cm) of 10–30 cm leg outliers, 0.6 % false flags; outlier-joint error 19.1 → 13.8 cm (11.1 with e′) | Misses sideways outliers; secondary check only |
Cost/frame pure Python: current 0.09 ms, PBD5 0.29 ms, two-bone 0.03 ms.

## Rebuilt design
1. **SegmentLengthEstimator:** rigid segments (hip width, femurs, tibias, ankle→big-toe feet, optionally arms); torso/shoulder pairs reported, never corrected. Record a segment only when both endpoints triangulated from ≥ 2 views, confidence ≥ 0.6, low reprojection error, frame not velocity/outlier-flagged (add knee-flexion limit once drift measured). Streaming median + MAD over all valid assessment frames (1 mm histogram, O(1)). Lock at ≥ 150 samples across ≥ 2 reps, 95 % CI < 5 mm, L/R diff < 2.5 cm. Output frozen public `Anthropometrics` snapshot → athlete params + DB; stored values verified at session start, never re-measured per set.
2. **Per-frame (triangulated):** no position correction. Compute rigid residuals; blame a joint only when every rigid segment touching it is violated; multiply its confidence by 0.2 instead of moving it. Main outlier rejection belongs in the triangulator.
3. **Optional later correction:** soft symmetric PBD, rigid only, 3 iterations, stiffness 0.5, deadband max(2·CI, 1 cm), once after temporal smoothing — only if ≥ 5 real 3-cam sessions show femur/tibia keypoint lengths vary < 1 cm across depth AND synthetic fault biases stay < 0.3 cm / 0.5°.
4. **Reset policy:** `reset_readiness_gate` resets temporal state only; lengths, body measurements, threshold adjustments are session-scoped, computed from stored bases by one owner.
5. **BodyProportions:** delete `valgus_scale`, `pelvis_tilt_coupling`; drop `forward_lean_scale` from rules or scale in lean space.
6. **Single camera:** no sphere projection; if anything, depth-only leg solve with sign continuity.
7. **Tests:** 19-kpt Y-down skeleton; thresholds unchanged after N sets; list/heel-rise/valgus survive; forward-lean direction; outlier burst in calibration window; athlete params survive phase-2 reset.

## Files
`synth.py` (fixed foot rotation/pelvic list; correlated noise, drift, spine shortening), `metrics.py`, `methods.py`, `detect.py`; `e1_validate_synth.py`; `e2_impulse.py/.json`; `e3_sequences.py`, `e3_report.py`, `e3_raw.json`, `e3_summary.json`; `e3b_rigid_outliers.py`/`e3b_raw.json`; `e4_calibration.py`, `e4a_calibration.json`, `e4b_outlier_burst.py`; `e5_reset_compounding.py` (run from repo root); `e6_proportions.py`; `e7_keypoint_counts_monocular.py` (repo root); `e8_residual_and_outlier_spread.py`; `PROGRESS.md`.
