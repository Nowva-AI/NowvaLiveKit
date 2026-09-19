# Wiring audit — PROGRESS (FINAL)

Scope: integration/wiring of pre-IK filter layer (pipeline.py, pipeline_process.py, preik_chain.py,
standing_gate as used, pipeline_inspector.py, capture_audit.py, debug_filters.py,
visualize_triangulated.py, config, tests, skeleton consumers).

## Status
- DONE: read all in-scope files (code reading complete). Ran existing tests.
- DONE experiments (scripts + .out in this folder; harness.py = fake triangulated provider + fake clock):
  E1 exp1_frame_index_cooldown: CONFIRMED F-A. 6 valgus+lean reps: frame_index=0 -> faults {forward_lean:1, knee_valgus:1}
     all in rep 1, reps 2-6 zero; incrementing fi -> knee_valgus 6, forward_lean 3.
  E2 exp2_proportion_compounding: CONFIRMED F-B. long femur (fwd_lean_scale 1.22, valgus 0.863): set1 FL (176.9,164.7,152.5),
     upright squat (trunk 160) now emits forward_lean faults (default thresholds: 0); set3 FL (183.5,177.4,171.3) >180.
     short femur (0.8, 1.079): FL 116 -> 92.8 -> 74.2 (0.8^n), valgus 5.1 -> 5.5 -> 5.9 (1.079^n). Scaling interleaves with
     RuleEngine 1-rep auto-calibration (valgus reset to 5/10/15 after first clean rep, profile apply_baseline).
     Production (chain off): no re-application because nothing re-calibrates bones.
  E3 exp3_calibration_flow: CONFIRMED F-C/F-D. wait loop exits after 34 frames, athlete_params present at line 543;
     after line-827 reset + 5 calibration reps, line 909 _extract_athlete_params = None, bones progress (0,30).
     coaching_service.py:908-930 saves athlete_params=None -> db calibration_utils.py:87 OVERWRITES row ->
     main.py:418-423 returning users: "Stored calibration has no athlete_params" -> diagnosis dark every later session.
     ENABLE_PREIK_FILTERS=false: AttributeError 'NoneType'.is_calibrated in wait loop + _extract_athlete_params;
     standing gate never reset (is_ready stays True after reset_readiness_gate).
  E4 exp4_dropout_duplicates: CONFIRMED F-E/F-G. 2-frame dropout mid-descent, triangulated clocks: filtered knee jumps to
     held raw (53.73 vs baseline 45.10) then FROZEN 3 frames (57: 53.73 vs 58.00), knee accel 7.0e6 deg/s^2 (base ~500);
     same dropout with agreeing time.time() clocks: smooth (57: 52.72 vs 58.00, accel <=411). Duplicates every 3rd descent
     frame: max knee_filt deviation 6.26 deg, velocity ratio 0.90-1.26, accel 1.69e7. (accel has no squat-rule consumer;
     velocity feeds PredictiveStateEstimator -> DepthRule per-frame max.)
  E5 exp5_rest_and_conf0_seed: CONFIRMED. (a) after 90 s rest + re-arm, 5 None frames: held pre-rest skeleton advanced readiness
     gate to 3/5 (confs 0.72,0.54,0.36). (b) toe conf 0 at origin for first frames of set -> once visible (conf 0.9) VelocityClamp
     crawls it 8.3 cm/frame: error 96.5 -> 0 cm over 11 frames (0.37 s) labelled conf 0.9; hip_rotation_l peaks 38.1 deg
     (truth 10.6), still 13.5 at frame 39 (angle-filter tail).
  E6 exp6_inspector_and_tools: CONFIRMED F-H: rest frame records previous mid-squat raw kpts (knee y 0.283 vs standing 0.45) and
     stale intermediates/raw_angles on rest, gated and None frames; 4/7 inspector stages always None.
     CONFIRMED F-J: visualize_triangulated.py:470 AttributeError '_calibration_frames' on first uncalibrated triangulated frame.
     debug_filters.py imports OK but uses legacy knee-angle RepCounter (not production hip counter / BiLSTM), no gates.
  E7 exp7_display_smoother_fppa: display smoother -> FPPA effect small on synthetic rep (peak 15.82 both, max diff 0.40 deg). LOW.
  E8 exp8_chain_cost: Mac per-frame blend+clamp 0.053 ms, full 6-stage 0.341 ms, one Skeleton3D round trip 0.031 ms,
     array-only clamp math 0.006 ms -> conversions dominate; fine for Jetson but pass arrays between stages.
  E9 exp9_calibrate_on_filtered: NOT reproduced — bone median over 30 frames absorbs <=11 crawl frames (foot 0.173 = truth).
  Extra (reading): T-pose calibration pipeline_process.py:403-411 runs at camera open, no prompt/validation, height from
     env default 1.885 m, saved to outputs/calibration.json but never reloaded (config has no triangulation.calibration_file).
     PipelineFrame.skeleton_3d is RAW on gated frames (pipeline.py:597) and FILTERED on ready frames -> consumers switch streams.
- COMPLETE: all experiments done. REPORT.md could not be written (harness blocks subagent report files); the full report is in the agent's final message to the coordinator (W1-W12, consumer map, dependency map, coverage, rebuild wiring, verdicts).

## Test counts (./venv/bin/python -m pytest, from repo root)
confidence_blend 5, velocity_clamp 6, bone_constraints 8, ground_clamp 13, preik_chain 8 passed + 3 skipped,
standing_gate 21, phase_aware_smoothing 5, body_proportions 17, predictive_state 6, test_pipeline 9.
Total 98 passed, 3 skipped. No test drives process_frame with a triangulated skeleton, time gaps,
duplicates, dropout hold, reset per set, or inspector path.

## Findings so far (from code reading; CONFIRM = needs experiment)
F-A CRITICAL (CONFIRM) triangulated frame_index is always 0 (triangulator.py:152 via multi_camera.py:201
   frame_index=0). Pre-IK stages, IK (analytical_ik.py:84), JointAngleFilter (filters.py:287) preserve it.
   Frame-index cooldowns: knee_valgus.py:88 (<30), forward_lean.py:76 (<150), symmetry.py:70,
   back_rounding.py:74, rule_engine.py:153-162 dedup (15). With fi constant 0 each rule fires at most ONCE
   per session (after first fire _last_fault_frame=0 -> 0-0<cooldown forever). visualize_triangulated.py
   passes real frame indices (446/529) so the test harness hides it.
F-B HIGH (CONFIRM) proportion scaling: forward_lean.py:50 multiplies 180-convention thresholds (145/135/125)
   by forward_lean_scale (0.8..1.3). scale>1 is meant "more lenient" but makes it STRICTER; 1.3 -> 188.5 > 180
   => fires every in-rep frame. knee_valgus.py:64 multiplies too. reset_readiness_gate (pipeline.py:382-388)
   sets _proportions_applied=False and resets bones -> if chain re-enabled, bones recalibrate each set and
   pipeline.py:624-632 re-applies `*=` => compounding (scale^n). Currently masked: nothing calls enforce
   after the assessment wait loop.
F-C HIGH (CONFIRM) calibration mode loses athlete_params: wait loop drives bones.enforce
   (pipeline_process.py:502-505) -> calibrated; line 827 reset_readiness_gate() resets bones; nothing
   re-drives enforce; line 909 _extract_athlete_params -> None -> calibration_complete msg has no
   athlete_params (922-924) and "diagnosis unavailable" log. Need to check who persists calibration file
   (main.py / agent) -> returning users then get "No stored athlete params" (pipeline_process.py:473-477).
   Also: failed assessment round (787) resets bones too.
F-D HIGH ENABLE_PREIK_FILTERS=false -> pipeline._bone_constraints None -> AttributeError at
   pipeline_process.py:502 and 218 (calibration mode crashes). Flag also silently disables
   phase-aware angle smoothing (pipeline.py:653-654) and predictive estimator (690-693), and the standing
   gate is only ever reset via BoneLengthConstraints.reset() (bone_constraints.py:311) -> never reset.
F-E MEDIUM dropout hold (pipeline.py:537-565): held skeleton is RAW last skeleton, timestamp=time.time()
   while triangulated timestamps are perf_counter (multi_capture.py:94) -> VelocityClamp dt ~1.7e9 s (no clamp),
   next real frame dt negative -> fallback; OneEuro/Derivative dt<=0 -> 1e-6 (filters.py:114, derivatives.py:134).
   _last_valid_skeleton not cleared in reset_readiness_gate/presence_only -> stale pre-rest skeleton can
   count toward readiness gate after rest (CONFIRM). Held frames also go to BiLSTM and gates.
F-F MEDIUM readiness early return (pipeline.py:590-601): filters skip gated frames; state reset at same time,
   so gap mostly harmless; but first ready frame seeds blender/clamp with conf-0 keypoints at (0,0,0)
   (triangulator leaves untriangulated pts at origin, rtmpose.py:268 zeros) -> VelocityClamp (ignores
   confidence) crawls that keypoint from origin at 8.3 cm/frame (CONFIRM numbers).
F-G MEDIUM duplicate frames (multi_capture.py:113 returns latest primary frame even if processed) -> identical
   timestamps -> OneEuro dt=1e-6 freeze, DerivativeTracker velocity dip + accel spike (CONFIRM numbers).
F-H MEDIUM inspector staleness: _inspect_intermediates/_inspect_raw_angles cleared only after readiness
   early return (pipeline.py:604-606); _inspect_raw_kpts not cleared on presence_only/None frames -> gated/rest
   frames record stale stages (CONFIRM). Inspector only records in main loop (pipeline_process.py:1357), not
   assessment/calibration phases where bone calibration happens. Stage list duplicated in 4 places:
   preik_chain.py:43-48, pipeline.py:437-453, pipeline_inspector.py:34-51, capture_audit.py:40-47+122-129.
F-I MEDIUM capture_audit.py runs ALL 6 stages by default (enabled unless --disable) -> does not match production
   (2 active); MediaPipe timestamps are time.time() at offline processing, not capture time (meta timestamps
   ignored) -> temporal filters behave differently than live. No dropout hold.
F-J MEDIUM visualize_triangulated.py runs a DIFFERENT chain (no blend, VelocityClamp 4.0, bone enforce tol 0.15
   ACTIVE, smoother 1.2/3.0, no ground clamp, no readiness gate, per-camera coco17 non-batched RTMPose, RepCounter
   not hip counter/BiLSTM) and line 470 reads bone_constraints._calibration_frames which does not exist
   (attr is calibration_frames) -> AttributeError on first triangulated frame (CONFIRM).
F-K LOW debug_filters.py stale: RepCounter.update(final_angles, derivatives, []) signature, bone calib without gate,
   reads bone_con._frame_count.
F-L LOW single-camera: display smoother (pipeline.py:521-522, "display-only") output feeds
   SingleCameraValgusEstimator (639) -> analysis is smoothed by display filter.
F-M LOW standing gate heel check silently skipped for 19-kpt triangulated skeletons (standing_gate.py:70-71).
F-N LOW config: VelocityClampConfig.target_fps unused; triangulation.enabled unused (env var decides);
   primary_camera must be a device id in device_ids else frame None forever (multi_camera.py:180,205).
F-O LOW returning users (--calibration-file, no calibration mode): bones never calibrate -> no rule/IK proportion
   scaling, IK pelvis_tilt_coupling stays 0.4 default; stored athlete_params only reach tracker/bridge.
F-P apply_calibration_to_rule_engine (calibration.py:107-136) sets absolute thresholds -> wipes any proportion
   scaling applied during assessment; later scaling (if re-applied) would multiply calibrated values.

## Dependency map of filter internals (grep done)
- pipeline_process.py:217-239 (_bone_constraints.is_calibrated, body_proportions, _calibrated_lengths pairs
  shoulder-shoulder, ankle-foot L/R), 502/505 (is_calibrated, enforce), 515-521 (_standing_gate.is_ready/progress/
  last_failure, _bone_constraints.progress), 799/885/1315 (_readiness_gate.progress), 955 (_readiness_gate.max_knee_flexion_deg write).
- scripts/demos/test_choreographer.py:54-65,176,186 (same as pipeline_process).
- calibrated_test_visualizer/visualize_triangulated.py:158-195 (body_proportions, _calibrated_lengths), 470 (_frame_count, _calibration_frames).
- scripts/demos/visualize_video_squats.py:195-199 (_calibrated_lengths), builds own chain 519-579.
- scripts/tools/debug_filters.py:220 (_frame_count); scripts/tools/capture_audit.py builds chain 94-130, bones.progress/is_calibrated/body_proportions.
- src/biomechanics/viz/pipeline_inspector.py:137-146 (_standing_gate.is_ready, _inspect_raw_kpts, _inspect_intermediates, _inspect_raw_angles).
- tests/test_calibrated_workout.py:160,225,287-289,348,367; tests/test_workout.py:96,169,189; tests/test_biomechanics/test_pipeline.py:239,252 (_readiness_gate.progress); test_bone_constraints.py:205-206 (_calibrated_lengths).
- benchmarks/components/bench_filters.py:53-73, bench_gates.py:16-17; scripts/tests/compare_skeletons.py:140-150.
- faults/rule_engine.py:15,76 (BodyProportions type), rules forward_lean.py:50, knee_valgus.py:64, fault_types.py:185; analytical_ik.py:64 (pelvis_tilt_coupling).

## Planned experiments (to write under this folder)
1. exp_frame_index_cooldown.py — full BiomechanicsPipeline with fake multi-camera provider, synthetic valgus squats, fi=0 vs incrementing: count KNEE_VALGUS faults.
2. exp_proportion_compounding.py — thresholds after 3 sets with chain re-enabled (patched apply_preik_filters incl. bones).
3. exp_calibration_flow.py — replicate pipeline_process assessment->calibration flow; show _extract_athlete_params None at line 909; ENABLE_PREIK_FILTERS=false AttributeError.
4. exp_dropout_duplicates.py — clock mixing, stale held skeleton latching gate after rest, duplicate timestamps -> derivative spikes.
5. exp_conf0_seed.py — conf-0 keypoint at origin at set start -> crawl frames.
6. exp_inspector_stale.py — stale intermediates on gated frames.
7. exp_visualizer_attr.py — AttributeError on _calibration_frames.
Then write REPORT.md with prioritized defects, dependency map, rebuild wiring recommendation.

## How to resume
Read this file; write scripts above (import repo via sys.path.insert(0, '/Users/naiahoard/NowvaLiveKit/src'),
run with /Users/naiahoard/NowvaLiveKit/venv/bin/python); still need to grep main.py/agent for who persists
calibration_complete athlete_params (F-C).
