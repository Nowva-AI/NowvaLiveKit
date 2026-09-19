# Wiring & Integration Audit — Pre-IK Filter Layer (triangulated primary)

(Saved by lead from the agent's final message; the agent's sandbox blocked writing REPORT.md.)

Evidence harness: `harness.py` builds a REAL `BiomechanicsPipeline` in multi-camera mode with a fake provider emitting synthetic 19-kpt triangulator-layout skeletons and a fake clock advancing both `time.time` and `perf_counter`. Scripts `expN_*.py` + `.out` in this folder (`./venv/bin/python expN_*.py`). CONFIRMED = reproduced; READ = code reading. No repo files modified.

**Headline:** two CRITICAL defects are live today, even with 4/6 stages commented out: (W1) in triangulated mode every fault rule fires at most once per session; (W2) calibration mode discards athlete_params and the DB then overwrites them with None, so returning users get no diagnosis. W3/W4/W6/W7 detonate once the rebuilt chain re-enables bone/ground calibration.

## 1. Prioritized wiring defects

**W1 — CRITICAL, CONFIRMED (exp1). Faults fire once per session in triangulated mode.** Triangulated `frame_index` is always 0 (`multi_camera.py:201` → `triangulator.py:152`), passed through chain, `analytical_ik.py:84`, `filters.py:287`. Rule cooldowns are frame-index based (`knee_valgus.py:88`, `forward_lean.py:76`, `symmetry.py:70`, `back_rounding.py:74`, dedup `rule_engine.py:153-162`) → after the first fault `0-0 < cooldown` forever. 6 valgus+lean reps: frame_index=0 → 1 forward_lean + 1 knee_valgus total (rep 1 only); incrementing index → 6 knee_valgus + 3 forward_lean. Hidden because `visualize_triangulated.py:446,529` passes real indices. **Fix:** restamp `frame_index` from the pipeline counter right after `get_pose`; cooldowns in seconds on one monotonic clock; pipeline test (frame_index=0 skeletons, valgus must fire in rep 2).

**W2 — CRITICAL, CONFIRMED (exp3). Calibration mode loses athlete_params → returning users have no diagnosis.** Wait loop (`pipeline_process.py:502-505`) calibrates bones; params correct at `:543`. `:827` `reset_readiness_gate()` resets bone calibration; nothing re-drives it. `:909` `_extract_athlete_params` → None → `calibration_complete` without athlete_params → `coaching_service.py:908-930` persists None; `db/calibration_utils.py:87` upsert overwrites the row → next session `main.py:418-423` "Stored calibration has no athlete_params", diagnosis off. Measured: wait loop exits after 34 frames with correct params; after reset + 5 calibration reps params None, bone progress (0,30). A failed assessment round also resets bones (`:787`). **Minimal fix:** reuse the `:543` params at `:909`; DB upsert must not overwrite params with None. **Real fix:** session-scoped body calibration that per-set reset never touches.

**W3 — HIGH, CONFIRMED (exp2). Proportion scaling compounds per set; forward-lean scaling is directionally inverted.** Per-set reset re-arms scaling (`pipeline.py:382-388`, `624-632`); rules do `*=` (`forward_lean.py:50`, `knee_valgus.py:64`). Forward-lean thresholds use the 180°-down convention, so scale>1 ("more lenient") is stricter. Full chain re-enabled: long-femur user (lean scale 1.22): set 1 (176.9, 164.7, 152.5) — upright squat faults (0 at defaults); set 3 (183.5, 177.4, 171.3), all >180 → fires regardless of posture. Short-femur user (lean 0.8, valgus 1.079): lean 116 → 74.2 (0.8³); valgus 5.1 → 5.5 → 5.9. Interleaves with 1-rep auto-calibration and `apply_calibration_to_rule_engine` absolute overwrite (`calibration.py:107-136`). Returning users never get scaling (IK pelvis coupling stays 0.4 instead of measured 0.487). **Fix:** scale from stored base thresholds (idempotent), once per session, defined order; for 180-convention use `180 - (180-base)*scale` (rules owner to confirm intent).

**W4 — HIGH, CONFIRMED (exp3). `ENABLE_PREIK_FILTERS=false` crashes calibration mode and silently disables other things.** `pipeline_process.py:502` and `:218` → AttributeError (`_bone_constraints` None). Same flag silently disables phase-aware angle smoothing (`pipeline.py:653`) and the predictive estimator feeding rules and DepthRule's per-frame max (`:690`). Standing gate only reset inside `BoneLengthConstraints.reset` (`bone_constraints.py:311`) → never resets with flag off. **Fix:** per-stage YAML flags; separate flags for phase smoothing and prediction; body calibration + gates always constructed.

**W5 — HIGH, READ. T-pose calibration unvalidated, hardcoded height, never reused.** `pipeline_process.py:403-411` grabs 30 frames at camera open: no prompt/T-pose check; height from env default 1.885 m (founder's), not the user profile; a camera with <6 keypoints skipped with a warning; <2 survive → 3D None forever, silently. Saved to `outputs/calibration.json` but never reloaded → every session recalibrates. Every length scales with assumed/true height (geometric, not run): a 1.65 m user is inflated 14% → shifts velocity clamp limit, gate torso range, stance tolerances, athlete_params. **Fix:** prompted, validated calibration phase using the user's height; persist per rig; reload.

**W6 — MEDIUM, CONFIRMED (exp4, exp5a). Dropout hold mixes clocks and can resurrect stale skeletons.** Held frames (`pipeline.py:537-565`) are the raw last skeleton stamped `time.time()` vs triangulated `perf_counter`. 2-frame dropout mid-descent: filtered knee jumps to the held raw value (53.7° vs 45.1°) then freezes 3 frames (57: 53.7° vs 58.0°); knee acceleration 7e6 °/s². Consistent clocks → smooth. Hold buffer never cleared on reset/rest: after 90 s rest, 5 no-pose frames let a 90 s-old skeleton push readiness to 3/5. **Fix:** stamp held frames on the source clock, never count them toward gates/BiLSTM, clear on reset/rest, drop whole-skeleton hold in triangulated mode.

**W7 — MEDIUM, CONFIRMED (exp5b). First ready frame seeds filters with origin-parked missing keypoints.** Triangulator leaves missing points at (0,0,0) conf 0; blender stores them as prev; velocity clamp ignores confidence and crawls them toward truth: toe error 96.5 → 0 cm over 11 frames (0.37 s) while labelled conf 0.9; `hip_rotation_l` peak 38.1° (true 10.6°), still 13.5° at frame 39. Bone calibration unaffected (30-frame median absorbed it, exp9). **Fix:** per-keypoint valid mask through the chain; re-seed (don't crawl) when a keypoint becomes valid.

**W8 — MEDIUM. Duplicate captures processed as new frames.** `multi_capture.py:113` always returns the newest primary frame. CONFIRMED: duplicating every 3rd descent frame → up to 6.26° knee deviation, velocity factor 0.90–1.26, acceleration 1.7e7. Frequency on hardware not measured. **Fix:** skip frames whose source timestamp hasn't advanced.

**W9 — MEDIUM, READ + exp6. `PipelineFrame.skeleton_3d` switches raw/filtered at gate boundaries.** Raw on gated/rest frames (`:513-519`, `593-601`), filtered on ready frames. Live demo pose, stance metrics (`ipc_bridge.py:117`), set collector, debug recorders see the switch. **Fix:** explicit `skeleton_3d` and `skeleton_3d_raw` fields.

**W10 — MEDIUM, CONFIRMED (exp6). Drifted chain copies.**

| Copy | Difference from production |
|---|---|
| `preik_chain.py` | reference (blend + clamp only) |
| `pipeline._apply_preik_filters_inspected` | hand-mirrored copy |
| `pipeline_inspector.py:34-51` | 7 hardcoded stage names, 4 always None, labelled "Raw MediaPipe" |
| `capture_audit.py` | all 6 stages on by default, offline timestamps |
| `visualize_triangulated.py` | different chain (clamp 4.0, bone tolerance 0.15 active, smoother 1.2/3.0, no blend/ground); crashes at `:470` on nonexistent `_calibration_frames` |
| `visualize_video_squats.py` | no ground clamp |
| `debug_filters.py` | legacy knee-angle rep counter, no gate |
| `bench_filters.py`, `compare_skeletons.py` | no ground clamp, no second bone pass |

Inspector buffers cleared only after the readiness early return → rest/gated/pose-missing frames record the previous frame's data (standing rest frame recorded mid-squat knee y 0.283 vs 0.45). Inspector records only in the main loop, not assessment/calibration.

**W11 — MEDIUM design, READ. Gates judge raw; calibrators record filtered and read a latched gate.** Bone/ground calibration record any frame once the standing gate has latched (`bone_constraints.py:164`, `ground_clamp.py:76`), including squat frames. Heel check silently skipped for 19-kpt skeletons (`standing_gate.py:70`). Readiness vs standing gate configs inconsistent; HUD shows standing-gate hints while readiness blocks. **Fix:** calibrators get a per-frame standing verdict evaluated on the skeleton they record.

**W12 — LOW.** Display-only 2D smoother feeds the single-cam valgus estimator (exp7: ≤0.4°, but retuning display would silently change diagnosis). Angle filter and derivative tracker not reset per set. `VelocityClampConfig.target_fps` and `triangulation.enabled` unused. `primary_camera` must be in `device_ids` or every frame is None. Throttle ignores display/IPC time. Cost (exp8, Mac): blend+clamp 0.053 ms; full 6 stages 0.341 ms; one Skeleton3D↔numpy round trip 0.031 ms; array-only clamp math 0.006 ms → conversions dominate; pass arrays.

## 2. Raw vs filtered consumers
- **Raw:** BiLSTM (`pipeline.py:579`, incl. held/gated frames), standing + readiness gates (`:587-590`), hold buffer, inspector raw panel.
- **Filtered (blend+clamp):** bone calibration in wait loop (`pipeline_process.py:505`), IK, rep signal, 3D valgus, standing/bottom kpt buffers → session tracker, diagnosis bridge, IPC rep_complete → assessment_logger `rep_N_bottom.json` / `rep_N_standing.json`; rep trajectory → `score_set`.
- **Display-smoothed 2D:** single-cam FPPA valgus.
- **Switches (W9):** `PipelineFrame.skeleton_3d` (set collector, send_frame_data, send_live_pose, debug recorders).
- **Rules:** predicted angles only when flag on.

## 3. External dependencies on filter internals
- Private attrs: `pipeline_process.py:217-239` (`_bone_constraints.is_calibrated/.body_proportions/._calibrated_lengths` shoulder-shoulder & ankle-foot pairs), `:502/505` (is_calibrated, enforce), `:515-521` (`_standing_gate.is_ready/progress/last_failure`, `_bone_constraints.progress`), `:799/885/1315` (`_readiness_gate.progress`), `:955` WRITES `_readiness_gate.max_knee_flexion_deg`; `scripts/demos/test_choreographer.py:54-65,176,186` (same); `visualize_triangulated.py:158-195` (`_calibrated_lengths`, `body_proportions`), `:470` (`_frame_count`, `_calibration_frames`); `visualize_video_squats.py:195-199`; `debug_filters.py:220`; `pipeline_inspector.py:137-146` (`_standing_gate.is_ready`, `_inspect_raw_kpts`, `_inspect_intermediates` keyed by stage names, `_inspect_raw_angles`); tests `test_calibrated_workout.py:160,225,287-289,348,367`, `test_workout.py:96,169,189`, `test_pipeline.py:239,252`, `test_bone_constraints.py:205-206`.
- Public: BodyProportions fields (`rule_engine.py:76`, `forward_lean.py:50`, `knee_valgus.py:64`, `analytical_ik.py:64`, `capture_audit.py:465`); `ENABLE_PREIK_FILTERS` (`pipeline.py:79`, two workout tests, README); filter constructor signatures (capture_audit, both visualizers, debug_filters, compare_skeletons, bench_filters, bench_gates, tests); `enable_inspect`, `reset_readiness_gate`, `is_ready`; `apply_preik_filters` kwargs; string check `test_preik_chain.py:227`.
- A rebuild should provide: `pipeline.body_calibration` (is_complete, progress, lengths(pair), proportions, to/from_athlete_params), `pipeline.gate_status()`, `set_readiness_knee_limit()`, `chain.stage_names` + tap hook, `build_preik_chain(config)`, one-release shim for `_bone_constraints`, `_standing_gate`, `_readiness_gate`.

## 4. Test coverage
confidence_blend 5, velocity_clamp 6, bone_constraints 8, ground_clamp 13, preik_chain 8 (+3 skipped bone/ground calibration), standing_gate 21, phase_aware_smoothing 5, body_proportions 17, predictive_state 6, test_pipeline 9 → **98 passed, 3 skipped**.
Untested: process_frame with triangulated skeletons (frame_index=0, 19 kpts, perf_counter); calibration across per-set reset / returning users / DB None; idempotent scaling across sets; flag off in pipeline_process; held frames, duplicates, time gaps; origin-parked missing kpts after reset; raw/filtered switch; inspector staleness; tools using the production chain. `test_preik_chain` uses one idealized clock, conf 1.0, no noise/motion/resets.

## 5. Recommended wiring for the rebuilt layer
1. **One stage list:** `STAGES` specs (name, factory, reset scope, YAML enabled) + `PreIKChain.run(points, conf, valid, t_mono)`; Skeleton3D↔arrays converted once at entry/exit; `build_preik_chain(config)` used by pipeline, capture_audit, visualizer, benchmarks, tests.
2. **Inspector hook:** chain calls `tap(name, points, conf)` after each stage; inspector takes stage names from the chain; clear inspection state at top of process_frame; record from every loop.
3. **Reset policy:** per session — body measurement, proportion scaling, IK coupling, camera extrinsics. Per set — temporal filter state, angle filter/derivatives, readiness gate, hold buffer, rep buffers. Auto re-seed on source gap > ~0.25 s, duplicate, or keypoint becoming valid.
4. **Raw vs filtered:** gates evaluate the frame calibrators record and pass a per-frame verdict; BiLSTM stays raw but never sees held/duplicate frames (check training distribution vs triangulated input); everything else filtered; PipelineFrame carries both explicitly.
5. **Flags:** YAML per-stage flags replacing the env flag; separate phase-smoothing and prediction flags; multi-camera mode resolved once from config.
6. **Clock/index:** one monotonic clock; pipeline stamps frame index + time; duplicates rejected; cooldowns in seconds.
7. **Persistence:** camera extrinsics stored/reloaded; athlete body calibration never overwritten with None; returning-user stored calibration injected into the pipeline, not just tracker/bridge.

## 6. Verdicts (triangulated primary)
| Component | Verdict |
|---|---|
| `apply_preik_filters` | REBUILD per §5 |
| `_apply_preik_filters_inspected` | REMOVE (tap hook) |
| `reset_readiness_gate` filter reset | FIX (per-set scope only) |
| Proportion application | FIX (idempotent, direction) |
| Dropout hold | FIX (single-cam) / REMOVE (triangulated) |
| Gate wiring | FIX |
| `ENABLE_PREIK_FILTERS` | SPLIT / REMOVE |
| Wait loop + `_extract_athlete_params` | FIX (public body-calibration API + persistence) |
| T-pose call placement | FIX |
| Triangulated `frame_index` | FIX (critical) |
| Inspector | FIX |
| `capture_audit` | FIX (shared chain, recorded timestamps, triangulated recordings) |
| `visualize_triangulated` | FIX or retire (prefer production pipeline `--inspect`) |
| `debug_filters`, `compare_skeletons` | REMOVE |
| `bench_filters` | FIX (shared chain, 19-kpt input) |
| Display smoother → single-cam FPPA | FIX (low) |
