# bone_constraints audit — PROGRESS

Working dir (persistent): /Users/naiahoard/NowvaLiveKit/.claude/preik-audit/bone_constraints/
Python: /Users/naiahoard/NowvaLiveKit/venv/bin/python

## Milestone 0 — code read (DONE)
Read: bone_constraints.py, standing_gate.py, preik_chain.py, pipeline.py (init 196, reset 382-388, chain 437-453,
proportions 622-632), pipeline_process.py (_extract_athlete_params 215-239, wait loop 495-525, assessment reset
787/827, phase-2 calibration 849-925, workout resets 1288/1326/1341), rule_engine.py 76-88, knee_valgus.py 64-70,
forward_lean.py 50-53, analytical_ik.py 64-66/383-421, calibration.py (build/apply profile), config.py 199-202,
biomechanics.yaml 56-58, tests test_bone_constraints (8) + test_body_proportions (17) — 25 pass.
Wiring agent (../wiring/PROGRESS.md) independently lists F-B (forward-lean scaling inverted/compounding) and F-C
(calibration mode loses athlete_params) — I confirm both below with evidence from my side.

Code-reading findings (to confirm numerically):
- Cascade order: (LS,LH) moves LH; (RS,RH) moves RH; (LS,RS) moves RS AFTER RH was placed from RS; (LH,RH) moves RH
  again -> single pass over a cycle, constraints not simultaneously satisfied. Legs cascade from moved hips.
- tolerance docstring wrong (says pre-calibration only; code applies it after calibration). Degenerate dir comment
  says +Y upward, Y is down. Heel pairs: _pairs_present(19) drops them, observation lists stay empty, finalize skips.
- Calibration records 30 frames after standing gate LATCHES (no per-frame pose check during those 30 frames).
- CONFIRMED by reading: pipeline_process.py:827 reset_readiness_gate() -> bones.reset(); nothing calls enforce()
  in phase 2 (lines 849-905); line 907 _extract_athlete_params -> None -> calibration_complete has no athlete_params
  -> coaching_service.py:908-935 save_user_calibration upserts athlete_params=None (calibration_utils.py:87
  overwrites existing) -> returning user: "No stored athlete params — diagnosis unavailable" (pipeline_process 473).
- apply_calibration_to_rule_engine (calibration.py:107) sets absolute thresholds at phase-2 end -> erases the
  proportion scaling applied after the wait loop. Returning users (calibration file) never calibrate bones.
- forward_lean.py:50 multiplies 180-convention thresholds (yaml 145/135/125) by forward_lean_scale -> inverted and
  can exceed 180. pelvis_tilt (IK coupling consumer) is not used by any fault rule/diagnosis (only filters,
  predictive_state, inspector).

## Next
E1 validate synth.py (fix heel-rise foot rotation, pelvic-list hip width). E2 impulse propagation. E3 sequence
MPJPE/angles none vs current vs alternatives a-e. E4 calibration accuracy. E5 reset/compounding. E6 proportions.

## Milestone 1 — E1/E2/E3 DONE (scripts + json in this folder)
synth.py fixed: heel rise rotates foot about toe (ankle-toe length constant), pelvic list keeps hip width, added
add_correlated_noise (AR1, z-anisotropy, burst outliers), apply_semantic_drift, spine_shorten_m. metrics.py uses
production AnalyticalIKSolver + TriangulatedValgusEstimator. methods.py: current class, a_tree, b_twobone, c_symm, d_soft, e_pbd.
E1 (e1_validate_synth.py): bone lengths constant (except torso sides under list: 23 mm — real non-rigidity). SIDE FINDING:
 GS valgus metric folds: toe-out 15 deg, medial swivel 0/10/20 deg reads -0.1/-4.7/+9.2 (magnitude from ML-axis vs
 leg-plane, sign from hip-ankle line) -> use valgus swivel 25 (reads ~+11.6) as fault case. Not my scope; flag to lead.
E2 (e2_impulse.json): 3 cm on L shoulder, standing, current: moves L hip 1.51, R sho 1.49, L knee 1.46, L ankle 1.46 cm;
 total displacement 12.4 cm (4.1x); pelvis_list err 3.09 deg, false heel rise 1.35 cm. Bottom: valgus_l err 2.21 deg
 from a shoulder error. Knee 3 cm at bottom -> ankle moves 1.29 cm, heel rise err 1.13 cm. b_twobone near extension:
 hip/ankle 3 cm -> knee_flex err 9.7 deg (singular). c/e spread the error (sum 3.6-4.3 cm) and halve list error.
E3 (e3_raw.json, e3_summary.json, e3_report.py output): 5 noise configs x 5 fault cases x 3 seeds, 420 frames.
 current_cal vs none (iid 2 cm, all frames): MPJPE 3.09 vs 3.09; hip_shift RMSE 2.50 vs 1.99; pelvis_list 6.37 vs 5.66;
 knee_flex 5.32 vs 5.12; depth 1.59 vs 1.95. real_2cm bottom: pelvis_list 8.58 vs 5.03, hip_shift 2.72 vs 1.91.
 FAULT ERASURE: list fault (6 cm hip asym): current bias -2.1..-2.8 cm (oracle -3.1..-3.3) = 35-55% erased; pelvis_list
 bias +4.2..+6.9 deg. Drift+spine-shortening: depth bias at bottom +4.5 cm vs +2.3 none.
 c_symm/e_pbd5: MPJPE -13% (3.09->2.68), knee_flex 5.12->4.40, valgus 7.0->6.0, but list erased 17-30% and depth bias
 doubled under drift (torso sides are not rigid). a_tree: no gain, heel-rise bias +0.4..+1.4 cm. b_twobone: knee_flex
 RMSE 5.1->10.9 (extension singularity). d_soft: ~= none (tiny outlier gain).
 Prod 30-frame median calib error: iid2 mean 4-10 mm max 16; real2 mean 3-14 max 23 mm.
Next: E3b rigid-subset (legs+pelvis only) c/e + outlier-focused detect/repair; E4 calibration; E5 reset; E6 proportions.

## Milestone 2 — E3b/E4/E5/E6/E7/E8 DONE
E3b (e3b_rigid_outliers.py, e3b_raw.json): rigid subset (hip width, femurs, tibias, feet) PBD5, real_2cm all frames vs none:
 MPJPE 3.64 vs 3.88, knee 3.50 vs 4.16, ankle 3.52 vs 4.33 cm, knee_flex 6.16 vs 7.23, depth 1.65 vs 2.15, pelvis_list 5.23 vs 5.91;
 list fault preserved (hip_asym bias -0.31 cm vs current -2.09); costs: heel-rise bias +0.66 cm (+1.13 drift), knee_dev -0.5 cm.
 Bone-residual outlier gate (3 sd, attribution): detects 17% (real_2cm) / 44% (sigma 1.5, 3% outliers) of 10-30 cm leg outliers;
 false flags 0.6%; outlier-joint error 19.1->13.8 cm (gate) / 11.1 (gate+rigid PBD). Weak: tangential outliers invisible.
E4 (e4_calibration.py, e4a_calibration.json, e4b_outlier_burst.py): standing 30-frame median MAE femur sigma2/rho0.6 8.6 mm (p95 20),
 rho0.9 14.5 (p95 36); whole-assessment median 2.3 (p95 5.6) / 3.7 (p95 8.9). Positive noise bias hip_w +5 mm (s2) +11 (s3).
 Drift: running estimate femur -4.3, tibia +7.2 mm. Burst of 15 cm knee outlier inside window: 10 fr -> femur +13.6 mm,
 15 fr -> +78 mm, 16 fr -> +126 mm; whole-assessment median +3 mm. Sensitivity (E4c): femur+tibia +1 cm -> current knee_flex
 bias +1.2 deg, heel -0.55 cm, dorsi +1.3; +2 cm -> +2.35 deg, heel -1.03 cm; pbd_rigid ~75% of that; break-even vs none ~ +-1.2 cm.
E5 (e5_reset_compounding.py, run from repo root; REAL BiomechanicsPipeline with fake 3D provider): CONFIRMED athlete_params None
 after line 827 reset and at line 909; apply_calibration_to_rule_engine erases scaling (13.0 -> 10.0). With bone enforce
 re-enabled in chain: valgus mild 12 -> 14.07 -> 14.72 -> 16.12 -> 16.89 over 4 sets; fwd_lean mild 145 -> 130.3 -> 122.8 ->
 114.7 -> 108.7 (compounding; per-set scale also jitters 1.05-1.17 from noise).
E6 (e6_proportions.py): 1.885 m SEGMENT_RATIOS: valgus_scale 1.064, fwd_lean_scale 0.933, coupling 0.499. Keypoint hip width
 0.18 -> clamp 0.70 (8.4/11.9/16.8 thresholds), 0.20 -> 0.78, 0.22 -> 0.86. GS valgus vs hip width at same knee swivel: neutral
 0.18 -> -6.8, 0.30 -> -1.7; swivel 25: 0.18 -> +18.0, 0.30 -> +9.8 => not multiplicative, sign flips. Forward lean scale 1.2/1.3
 -> mild 174/188.5 -> fires on 100% of in-rep frames; 0.933 -> 135.3 -> 0% (intended stricter).
E7: 17 kpts -> foot_avg_m silently 0.26 default; 19 kpts heel pairs skipped OK; 21 OK. Monocular-like noise (xy 1, z 6 cm):
 sphere projection depth RMSE 1.02 -> 3.16 cm, heel 1.40 -> 4.26 cm, hip_shift 0.97 -> 1.51; depth-only solve keeps x/y
 metrics exact, knee_flex 13.4 -> 7.2 deg, dorsi 8.3 -> 5.3, MPJPE 4.98 -> 5.45 (z sign flips).
E8: after one enforce pass R sho-R hip residual mean 4.1 mm max 32 mm (cycle). Single 20 cm outlier at bottom, current:
 LSho outlier -> LHip 10.2, RSho 9.4, LKnee 6.7, LAnk 2.9 cm; LKnee outlier -> LAnk 8.3, LToe 3.1 cm.
Next: write REPORT.md.

## Milestone 3 — audit complete (REPORT.md write was blocked by harness; full report returned as final message)
Verdict: REBUILD. Triangulated: no per-frame position correction by default; SegmentLengthEstimator (robust running median
over whole assessment, validity-gated, lock on CI<5 mm & L/R check, persisted per user, never reset per set) feeding
anthropometrics + rigid-segment residuals as confidence down-weight. Optional later: soft symmetric PBD on rigid subset
(hip width, femur, tibia, foot) after temporal smoothing, only if real 3-cam data shows keypoint lengths stable over depth.
Delete valgus_scale + pelvis_tilt_coupling; fix/delete forward_lean_scale (lean-space, idempotent). Fix F1 athlete_params
wipe (snapshot before reset; upsert must not write None). Single-camera: no sphere projection; depth-only leg solve if any.
Side finding for valgus owner: GS abduction magnitude/sign fold (valgus.py:294/304).
