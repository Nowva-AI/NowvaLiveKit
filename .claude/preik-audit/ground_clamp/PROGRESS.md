# GroundClamp audit — PROGRESS (checkpoint 2)

Working dir: /Users/naiahoard/NowvaLiveKit/.claude/preik-audit/ground_clamp/
Scope: src/biomechanics/utils/ground_clamp.py + calibration, tests, config, downstream consumers.

## Status
- [x] Read code: ground_clamp.py, standing_gate.py, bone_constraints.py, preik_chain.py, pipeline.py wiring,
      valgus.py, analytical_ik.py (dorsiflexion, pelvis list), bridge.py (grounding, stance ratio),
      ipc_bridge.py (live stance ratio), evidence_tests.py (weight_shift uses hip_y_l-hip_y_r), rtmpose.py
      (heels dropped), triangulator.py (translation-only recentre), calibration.py (T-pose world frame).
- [x] Existing tests pass: test_ground_clamp + test_preik_chain + test_standing_gate = 42 passed, 3 skipped.
- [x] sim.py: synthetic world-frame squat generator (feet planted, exact bone lengths, faults: valgus,
      bilateral/unilateral heel rise, lateral hip shift, pelvic list, stance change, 5deg roll/pitch tilt,
      early descent during calibration, soft-knee calibration) + triangulation noise + hip recentre.
- [x] Variants (run.py, proto.py): raw / GroundClamp only / bone1->GC->bone2 / bone1->bone2 (no GC) / full chain w smoother /
      prototype FootContactModel (world-frame, per-foot rest anchors, contact detection, ZUPT, no symmetry).
- [x] Metrics (results_sigma1.5.json 8 seeds, results_sweep.json sigma .5/1/3 cm; tables table_sigma1.5.txt, table_sweep.txt): ankle/toe RMSE, knee flex err (standing bias, bottom), tibia length err, GS valgus + KASR err,
      heel-rise preservation, stance width preservation, clamp firing rates.
- [ ] REPORT.md

## Key numbers so far (sigma=1.5 cm white + 0.6 cm AR(0.5 s) per axis, 8 seeds; see tables)
- GC fire rates on a CLEAN symmetric squat: floor 30% of post-cal frames (~75% of standing frames, either ankle),
  equalise 52% (63% after bone1), width snap 41%. At sigma 0.5 cm: 28/14/2%. At 3 cm: 32/80/68%.
- GC standing ankle y bias -0.91 cm (raised), tibia -0.81 cm; early descent in calib: -2.80/-2.66 cm; soft-knee calib
  (hips 4 cm lower): -3.77/-3.65 cm, floor fire 45%.
- bone1->GC->bone2: bone2 pushes the ankle back 0.86 of 0.89 cm but NOT the toe -> ankle-above-toe baseline shrinks
  5.0->4.2 cm at standing -> fake +0.8..1.4 cm "heel rise" signal at bottom on a clean squat (diag_stand.py/diag_bottom.py).
- Unilateral heel rise (3.37 cm true ankle L-R diff): raw 3.62, GC 0.04 (erased 99%), bone_gc_bone 2.24, full 2.36, proto 3.21.
  Proto ankle-rise signal 3.48 cm (other foot -0.02). Bilateral: proto 3.18/3.37.
- 5 deg roll tilt: true ankle ydiff 3.31 cm -> GC 0.04, eq fires 92%; proto 3.68.
- Stance widen after calib (30->42 cm): GC width bias -11.0 cm (fires 97%), KASR +0.46 (1.22->1.68), valgus extra -2.2 deg
  (sigma .5: -3.6 deg, KASR +0.48), standing knee flex +3.4..4.4 deg bias, ankle RMSE 6 cm. Proto width bias +0.17, RMSE 0.89.
- Pelvic list & lateral hip shift: untouched by GC (hip ydiff 2.80 vs 2.86 true; shift 5.99/6.00). NOTE cross-scope:
  BoneLengthConstraints erases 56% of pelvic list (2.86 -> 1.25 cm) — tell bone agent/lead.
- GC 'benefits' on clean symmetric data: ankle RMSE 2.77->2.38, width RMSE 2.28->1.08 (prior acting as averaging) but toe RMSE
  2.79->3.34, and every benefit flips to large error when the symmetric/fixed-stance prior is false.
- Proto (world-frame contact ZUPT): ankle RMSE 2.77->1.26, toe 2.79->1.19, knee flex RMSE 4.23->3.88 (bottom 3.36->2.99),
  width RMSE 1.02, valgus RMSE ~unchanged (hip/knee noise dominates). Fixed thresholds break at sigma 3 cm (contact 29%) ->
  thresholds must scale with measured noise.
- Incidental (not GC): GS valgus has -3.3 deg noise-induced bias at sigma 1.5 on a neutral knee (abs() folding in valgus.py:294).

## Code-reading findings so far (CONFIRMED by reading; numbers pending)
1. Hip-centred frame (triangulator.py:134-137 subtracts hip midpoint, translation only). "ankle_y_max" is
   ankle height below hips, NOT a floor. During a squat hips descend so hip-relative ankle y shrinks -> the
   floor-penetration clamp (ground_clamp.py:89-94) never fires legitimately in a squat; it fires only on
   noise near standing / full lockout higher than calibration. Median as a max => ~50% of standing frames
   clamped (expected bias ~0.4 sigma upward on the ankle, knee not moved -> tibia shortened).
2. Flat-floor equalisation (97-101): pelvic list / lateral hip shift do NOT create ankle-y differences under a
   translation-only recentre (both ankles shift equally). It fires on: noise (|diff|>1 cm, diff sigma = sqrt2*sigma
   -> ~64% of frames at sigma=1.5 cm), world roll tilt (5 deg * 0.38 m stance = 3.3 cm), unilateral heel rise
   (erased by half, pushes other foot through floor), staggered stance under pitch tilt. Contradicts its own
   per-side calibration (ankle_y_max_l != _r under tilt).
3. Stance-width lock (104-117): snaps to calibrated width (not to tolerance edge) when |err|>2 cm; noise on
   width sigma ~2.1 cm -> fires often; destroys stance changes after calibration; live IPC stance_width_ratio
   (ipc_bridge.py:122) and diagnosis stance ratio (bridge.py:133) frozen; moves ankles laterally with knee fixed ->
   changes GS abduction (valgus.py:270) and KASR (valgus.py:337) directly.
4. Ankle moved, knee not moved -> tibia length & knee angle altered; bone pass 2 (bone_constraints.py:178-208)
   projects ankle back onto knee sphere along knee->ankle -> undoes most of the vertical clamp, partially the
   lateral one. Net to be quantified.
5. Calibration timing: GC calibrates on the 30 frames right after the standing gate latches, no per-frame
   standing check (ground_clamp.py:75-83). reset_readiness_gate (pipeline.py:362-389) resets GC + bone
   constraints; bone_constraints.reset() also resets the shared standing gate -> recalibration at each set start,
   typically during walk-out/unrack; if user descends within 1 s, descent frames enter the median.
6. Standing gate flat-foot check needs heels (standing_gate.py:70) -> silently skipped in triangulated mode
   (19 kpts). Heels (halpe26 24/25) dropped in rtmpose.py:43-44,268-273 although CK.LEFT_HEEL=19/RIGHT_HEEL=20 exist.
7. No heel-rise rule exists anywhere in src/biomechanics/faults/rules/ (grep). Dorsiflexion (analytical_ik.py:272)
   = shank vs WORLD_UP, biased by world tilt. bridge.py:34 per-frame min-foot grounding (F19).
8. No ground_clamp section in config/biomechanics.yaml -> pydantic defaults (config.py:211-218).
9. Stage currently disabled (preik_chain.py:45, pipeline.py:446).

## Next
- real-video check (RTMPose halpe26 on recordings/squat_20260530_012118.mp4 etc.): planted ankle/toe/heel 2D drift standing->bottom
  and jitter (validity of ZUPT anchoring; heel availability).
- proto with noise-scaled thresholds at sigma 3 cm; valgus+widen scenario at sigma .5 for clean valgus shift number.
- write REPORT.md.

## How to resume
Write sim.py here (import repo via sys.path /Users/naiahoard/NowvaLiveKit/src, run with
/Users/naiahoard/NowvaLiveKit/venv/bin/python), run experiments, fill numbers into REPORT.md.
Do NOT modify repo source files.

## Checkpoint 3 — real video evidence (real_foot_drift.py -> *_h26.npy cache, real_foot_stats.py, real_foot_reps.py, real_traces.png, *_frames.jpg)
- Production RTMPose-m halpe26 decode on 5 real single-cam squat videos (32 reps). Per-rep bottom vs preceding standing,
  foot keypoints displaced >10 cm (approx, scaled by standing shank px = 0.443 m) in 56-78% of reps; pooled median
  16-36 cm. Cleanest video (0515): ankle 5.6-9, heel 4.8-6.4, toe ~12.5 cm. Visual confirmation: squat_20260512_134408_frames.jpg
  frames 122/160 — one foot's ankle/toe/heel drawn at KNEE height while the foot is planted.
- Confidence does not flag it: bottom gross-failure conf median 0.654 vs good 0.651.
- => planted-foot keypoints are least reliable exactly at the squat bottom. Contact hold must be ROBUST (ignore gross
  outliers) — the naive proto breaks contact on them and passes garbage through.
- Next: FootContactModelRobust (outlier-gated, foot-level release, noise-adaptive thresholds) + synthetic "foot snaps to knee
  at bottom" injection; then REPORT.md.

## Checkpoint 4 — COMPLETE (2026-09-16)
- Robust prototype `proto.py::FootContactModelRobust` (outlier gate, noise-adaptive thresholds, foot-level release,
  heel-rise mode with toe held, released-state teleport gate). Final runs: results_sigma1.5.json (14 scenarios x 8 seeds),
  results_sweep.json (sigma .5/3 cm); tables table_sigma1.5.txt / table_sweep.txt; tilt_estimate.py.
- Headline: robust proto ankle RMSE 2.77->0.49 cm, toe 2.79->0.39, foot-snap knee flex RMSE@bottom 41.3->2.8 deg
  (GC 20.6, full chain 30.9), heel rise R 3.06/3.37 cm (GC erases 99%), valgus+widen: GC -7.2 deg bias (15.2 true
  reads 8.0 -> fault missed), proto -0.19. Roll tilt from anchors 5.21+-0.47 deg (true 5).
- VERDICT: GroundClamp REMOVE (both modes); REBUILD as world-frame per-foot contact model (FootState side channel,
  halpe26 heels mapped, heel_rise rule, session floor for bridge F19).
- REPORT.md write was blocked by the subagent harness (no report files); the full report is in the agent's final
  message to the coordinator.
