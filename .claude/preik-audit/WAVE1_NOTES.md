# Wave 1 Hand-off Notes (for integration)

## WS1 capture — DONE (22/22 tests)
Files: `triangulation/multi_capture.py`, `pose/multi_camera.py`, tests `test_multi_capture.py`, `test_multi_camera_provider.py`.
- `SyncedFrames(NamedTuple)`: `frames: dict[str, np.ndarray]` (primary always present), `sequence: int`, `timestamp: float`. `get_synced_frames()` → `SyncedFrames | None` (**old 2-tuple unpacking breaks**). Newest set whose primary frame hasn't been returned; reference = newest such primary frame every live secondary has caught up to (its newest frame ≥ reference − 5 ms); secondaries contribute nearest frame within tolerance or are omitted; ≥ `min_views`; polls every 2 ms up to 40 ms, else None.
- `MultiCameraCapture.__init__` adds `min_views=2`, `primary_device_id=None` (defaults to `device_ids[0]`, ValueError if not in `device_ids`), injectable `clock_fn`/`sleep_fn`. Default `max_sync_delta_ms` 15 → 20. Constants `DEFAULT_MAX_SYNC_DELTA_MS`, `POLL_INTERVAL_S`, `MAX_WAIT_S`, `CATCH_UP_MARGIN_S`, `STALE_CAMERA_S`. Stalled secondary (> 150 ms behind) stops holding sets back.
- One clock: `start()` computes `offset = time.time() - perf_counter()` once; frames stamped `perf_counter() + offset`.
- `MultiCameraPoseProvider`: `start()` passes `min_views`, `primary_device_id=primary_camera`; `get_pose()` builds `MultiViewPose(frame_index=synced.sequence, timestamp=synced.timestamp)`; missing cameras absent; no recentring; `calibrate()` uses new API; default `max_sync_delta_ms` 20.
- Simulation (10 seeds): 29.4–30.0 Hz processed, 99.8–100 % full sets, ≤ 1.2 % None, 0 duplicates; p95 inter-camera gap 15–17 ms.
- Known limit: timestamps at arrival, not exposure (per-camera USB latency uncalibrated).

**Needs:**
1. config.py/YAML: `TriangulationConfig.max_sync_delta_ms` = 20.0.
2. `calibrated_test_visualizer/visualize_triangulated.py` lines 338–342, 432–436, 507–511: `synced_frames, _ = result` → `result.frames` / `result.timestamp` (`result.sequence` can replace `rec_frame_idx`).
3. pipeline.py: dropout hold (≈537–557) stamps `frame_index=self._frame_index` (different counter) — held frames must reuse last real sequence or not reach rules (predict_missing replaces). Don't overwrite `skeleton_3d.frame_index` after `get_pose`.
4. pipeline_process.py: `get_pose` now waits (counted in `latency_ms["pose"]`); removing sleep pacing in multi-cam mode optional.
5. Run `graphify update .` after merge.

## WS2 detector — DONE (49/49 tests, real model executed)
Files: `pose/rtmpose.py` (rewritten), `triangulation/calibration.py`, tests `test_rtmpose.py` (35), `test_tpose_calibration.py` (14).
- `estimate(frame, camera_id=0)`: crop tracked per camera_id. `estimate_batch(frames, camera_ids=None)`: per-id or positional tracking; ValueError on mismatched/duplicate ids. `reset_tracking()`; `release()` calls it. `NUM_KEYPOINTS = 21` (halpe26 heels 24/25 → 19/20, toes 20/21 → 17/18; coco17 pads 17–20 conf 0). Confidence = raw min(max_x, max_y) clipped [0,1]; below threshold → zeroed; None when nothing passes. Module `decode_simcc(x_logits, y_logits, boxes) -> (B,K,3)` with parabola sub-bin refinement. Constants `NUM_SKELETON_KEYPOINTS`, `HALPE26_LEFT_HEEL/RIGHT_HEEL`, `CROP_PADDING = 1.25`, `MIN_TRACKED_KEYPOINTS = 8`, `INPUT_ASPECT_RATIO`. Private `_preprocess(frame)->(blob,sx,sy)` and `_decode_simcc` method removed (only audit scripts used them).
- Crop: bbox of confident keypoints, height ×1.25, expanded to 192:256, aspect-preserving warpAffine zero border; bootstrap on full frame when no box or < 8/26 pass, then re-run on the new crop in the same call. `_run_padded` unchanged (constant batch shape). No box smoothing (measured no benefit).
- calibration.py: `calibrate()` raises ValueError when < 50 % of a camera's frames pass the 2D T-pose check; module `tpose_frame_mask`, `average_detected_keypoints`; gate `MIN_MEAN_KEYPOINT_SCORE = 0.5` on mean raw score with misses counted as 0; positions averaged over detected frames only (old code averaged (0,0) misses → 45 px error). Docstring: +Z = subject's BACK.
- Measured on squats.mov in 1280×720 canvas: mirror disagreement median 7.98 → 1.19 px; still jitter body 1.70 → 0.89 px; argmax zero-motion 37.8 % → 0 %; hip width 86.7 → 63.5 px; no-person images now return None; 3-view batch 17.67 → 17.04 ms (noise). Per-view extra CPU ≈ 0.14 ms steady state.

**Needs:**
1. WS1 `pose/multi_camera.py`: `estimate_batch(frames, camera_ids=cam_ids)` so crops follow cameras when a view drops; optionally `reset_tracking()` between calibrate and get_pose.
2. WS3 triangulator `NUM_KEYPOINTS` 19 → 21 (done in WS3's rewrite).
3. pipeline.py single-cam rtmpose backend: 21 keypoints, raw scores, real None skeletons now → dropout-hold/clock bugs will fire; `reset_tracking()` on set start.
4. `pose/base.py:82` `NUM_KEYPOINTS = 19` → 21.
5. `calibrated_test_visualizer/visualize_triangulated.py`: pass `camera_id=cam_id` to `estimate()`; `StandingPoseGate(min_confidence=0.5)` tuned for sigmoid confs; `[:19]` slices at 69/207 drop heels.
6. `scripts/tools/capture_audit.py:393` `np.zeros((19,3))` → 21; `scripts/demos/visualize_video_squats.py`, `scripts/tools/generate_opensim_data.py:132`, `scripts/tests/test_demo_renderer_live.py:32` hard-code 19 (cosmetic).
7. `scripts/tools/download_models.py:49` MODEL_DIR path bug (F13) still open.

## WS3 triangulation — DONE (38/38 tests, -W error)
Files: `triangulation/triangulator.py` (rewritten), `tests/test_biomechanics/test_triangulator.py`.
- `NUM_KEYPOINTS = 21`; 19-kpt views work (heels conf 0). Constructor params unchanged + `swap_margin_ratio`, `two_view_confidence_cap`; `min_views < 2` raises. `triangulate()` → WORLD coordinates, rejected = conf 0.0 at (0,0,0), None if nothing accepted, never NaN/inf. New `reset()` (clears tie-break state), `swap_count` property, module `recentre_at_hips(skeleton) -> Skeleton3D | None` per C3. Constants `UNCERTAINTY_SCALE_M = 0.02`, `TWO_VIEW_CONFIDENCE_CAP = 0.3`, `SWAP_MARGIN_RATIO = 0.5`, `PAIR_TIE_MARGIN_PX = 4.0`, `PAIR_AMBIGUITY_DISTANCE_M = 0.05`, `PREVIOUS_POINT_MAX_AGE_S = 0.5`.
- Algorithm: batched eigh on stacked DLT normal matrices; all-view solve; swap test (none / full / legs-only per view) only when a bilateral keypoint with ≥ 3 views fails, applied only if cost < 0.5× no-swap; failing keypoints → best passing pair, ties (< 4 px) broken by previous point (< 0.5 s); ambiguous tied pairs (> 5 cm apart, no history) → rejected. Confidence from analytic Jacobian, two-view cap 0.3 (u ≥ 3.05 cm).
- Measured: clean 6.3 mm; single-view 80/150 px outlier 9.1 mm mean / 21 mm max (old 145/293 mm); one-view L/R swap 6.6 mm, 0 false swaps in 2000 frames; 0.20 ms/frame clean, 0.47 ms with outliers.

**Needs:**
1. WS1 provider: pass real sequence as frame_index (done); call `triangulator.reset()` on set start / long dropout; log `swap_count` periodically. [Lead: `camera_ids=cam_ids` now passed to estimate_batch.]
2. Wave 2 pipeline/preik_chain: call `recentre_at_hips` after Kalman/foot contact; None = hip-dropout path; conf 0 = missing everywhere; IK's 0.1 floor now means u ≈ 6 cm — revisit.
3. `visualize_triangulated.py`: will now see world coords + 21 kpts (regrounds per frame, should work).
4. config (optional): expose `swap_margin_ratio`, `two_view_confidence_cap`.

## WS6 body measurements + calibration flow — DONE (45 tests)
Files: NEW `utils/segment_lengths.py`; `pipeline_process.py`; `db/calibration_utils.py` (upsert keeps stored athlete_params/baseline when new is None); `agent/services/coaching_service.py` (warns when absent); `main.py` (`--user-height-cm` from `state["user.height_cm"]`); tests `test_segment_lengths.py` (25), `test_pipeline_process.py` (12), `test_calibration_utils.py` (4), `test_coaching_service.py` (+1).
- `BodyProportions`: hip_width, femur_length_avg, tibia_length_avg, torso_length_avg, shoulder_width, foot_length_avg, hip_to_femur_ratio, tibia_to_reference_ratio, forward_lean_scale = clip((femur/torso)/0.84, 0.8, 1.3). No valgus_scale / pelvis_tilt_coupling.
- `SegmentLengthEstimator`: `record(points, confidences, rep_count)`, `is_complete`, `progress -> (min rigid samples, 150)`, `body_proportions`, `to_athlete_params()` (keys shoulder_width_m, femur_avg_m, torso_avg_m, hip_width_m, tibia_avg_m, foot_avg_m), `from_athlete_params()`. 1 mm histogram per segment, endpoints conf ≥ 0.6, anthropometric bounds, median+MAD with noise-lengthening correction; complete at ≥150 samples/rigid segment, ≥30/reported, ≥2 rep counts, CI < 5 mm, L/R diff < 2.5 cm; fallback ≥600 frames over ≥3 reps. Synthetic: femur/tibia p95 err ~3 mm; completes ~frame 230 (~2 reps); 30 cm × 30-frame burst moves < 5 mm.
- pipeline_process: wait loop waits for `pipeline.is_ready` only (HUD uses `_readiness_gate`); assessment reps poll `body_calibration`, HUD `MEASURING BODY n/150`; reps finishing before completion are held (`pending_reps`) and flushed once params exist (5 s timeout → flush without kinematics). `athlete_params`/`athlete_baseline` are session-scoped; `calibration_complete` omits keys rather than sending None. Returning users: `pipeline.apply_athlete_params(stored)`. T-pose: config `calibration_file` → `~/.nowva/rig_calibration_cams_<ids>.json` → calibrate + save. Height: `user.height_cm` via CLI `--user-height-cm`, else `NOWVA_USER_HEIGHT_M` with warning. Constants `ASSESSMENT_MEASUREMENT_WAIT_S = 5.0`, `RIG_CALIBRATION_DIR`, `FALLBACK_USER_HEIGHT_M`.

**Needs (integration):**
1. pipeline.py: `pipeline.body_calibration: SegmentLengthEstimator` (always constructed, never reset per set); `record(analysis_points, analysis_confidences, self.rep_count)` on every ready analysis frame until complete; apply proportion scaling once on completion; `pipeline.apply_athlete_params(params)`.
2. rule_engine.py: import BodyProportions from `segment_lengths` (WS7).
3. Remaining `_bone_constraints`/`BoneLengthConstraints`/`_calibrated_lengths` refs: pipeline.py (10), preik_chain.py (2), rule_engine.py (1), scripts/demos/test_choreographer.py (7), scripts/demos/visualize_video_squats.py (6), calibrated_test_visualizer/visualize_triangulated.py (6), scripts/tools/capture_audit.py (3), scripts/tools/debug_filters.py (2), scripts/tests/compare_skeletons.py (2), benchmarks/components/bench_filters.py (3), tests test_preik_chain.py (3), test_bone_constraints.py + test_body_proportions.py (delete with bone_constraints.py).
4. Config: `bone_constraints:` YAML section becomes dead; optional `triangulation.rig_calibration_dir`.

## WS4 fixed-lag Kalman — DONE (26 tests)
Files: NEW `utils/keypoint_kalman.py`, `tests/test_biomechanics/test_keypoint_kalman.py`; eval scripts in `.claude/preik-audit/smoothing/ws4_*.py`.
- `KeypointKalmanOutput` per C4 + `current_timestamp: float`, `gate_rejected: np.ndarray (N,) bool`. `FixedLagKeypointSmoother(...)` exact C4 signatures; constants `MAX_GAP_S=0.5`, `MAX_REJECTION_S=0.2`, `REACQUIRE_AGREEMENT_M=0.10`.
- Cost N=21: lag 2 = 35.6 µs/frame. Harness J_mean 2/4/8 px = 3.55/5.93/9.73 (identical to ProposedSmoother); with per-frame pooled σ̂_px 3.40/5.84/9.60 (beats reference).
- Gate on RMS 3D innovation (4σ ≈ 7–14 cm normal, 8 cm floor, widens during predict-only); re-acquire keeps pre-jump velocity with P11 reset; predicted conf = max(0.15, C3(P00)); never-tracked → (0,0,0) conf 0; NaN/inf inputs = missing; no smoothing propagated across a snap.
- Display (undelayed) stream can deviate ≤ 5 cm on predicted frames during dropouts (q=10) — analysis stream ≤ 1.3 cm.

**Needs:**
1. WS3 triangulator: pool σ̂_px per frame (median over accepted keypoints; keep per-keypoint only where > 3× median) before `max(σ̂, floor)·m_per_px` — costs 0.4–1.5 J otherwise. [Lead: applied — see below.]
2. Wave 2: call `predict_missing(ts)` on None frames; raise `measurement_std_floor_m` for MediaPipe; optionally hold display on `gate_rejected`/missing keypoints instead of extrapolating.

## WS5 feet — DONE (36 tests)
Files: NEW `utils/foot_contact.py`, NEW `faults/rules/heel_rise.py`, `faults/rules/__init__.py`, `profiles/squat.py` (HeelRiseRule appended → 6 rules), tests `test_foot_contact.py` (20), `test_heel_rise_rule.py` (16).
- `FootState` per C5 (NaN when not measurable; heel rise 0.0 while planted flat). `FootContactModel.update(points, confidences, timestamp) -> (points, FootState)`, `reset()`; world frame, 21 kpts, conf 0 = missing; dt ≤ 0 re-emits; gaps > 0.5 s release both feet (anchors kept). Heel rise measured on the ankle (works with missing heels); anchor absorbs an observation only after 0.4 s confirmation; heel mode judged only against anchors with ≥ 30 confirmed samples; update gate floor 4 cm. Knees/hips byte-identical pass-through.
- `HeelRiseRule(mild_cm=1.5, moderate_cm=3.0, severe_cm=5.0, cooldown_s=2.0, min_rise_duration_s=0.23)`, fires in-rep on max(rise_l, rise_r), time-based cooldown, `details["affected_side"]`, messages local to the file.
- Parity vs prototype (σ 1.5 cm + drift): ankle RMSE 0.51 cm (raw 2.77); one-sided 3.37 cm rise reads 3.20 (other foot 0.00); foot-snap + heel rise 1.46 cm ankle (prototype 2.30); floor roll 5.13 ± 0.19° (true 5); 74.5 µs/frame. Behind WS4 Kalman (lag 2): 0.50 cm, heel rise 3.31 cm. 0 false events per clean rep; 1.8 cm rises caught on ~40–55 % of reps (effective floor ≈ 2 cm sustained 0.23 s).

**Needs:**
1. WS7/test_body_proportions.py: `test_forward_lean_rule_in_engine` asserts `rule_count == 5` → 6.
2. Integration: `FootContactModel.update(lagged_world_points, lagged_confidences, lagged_timestamp)` after Kalman in triangulated mode; pass FootState to `RuleEngine.evaluate(..., foot_state=)` (None in MediaPipe); move thresholds/cooldown/persistence into `config.faults`; map `"heel_rise"` in `cue_cache.FAULT_TO_CUE_MAP` to cached `"heels_down"` cue + `FAULT_CUE_PRIORITY`; `bridge.py` F19 re-grounding can use `hip_height_above_floor_cm`.
3. Audit script `.claude/preik-audit/ground_clamp/run.py::_valgus` breaks if WS7 removes `TriangulatedValgusEstimator._abduction` (audit only).

## WS7 kinematics + rules — DONE (233 owned tests; all tests/test_biomechanics 839 pass after lead fixed rule count 5→6)
Files: `kinematics/analytical_ik.py` (rewritten), `kinematics/valgus.py`, `faults/rule_engine.py`, `faults/fault_types.py` (docstring), rules `knee_valgus, forward_lean, symmetry, back_rounding, depth, elbow_flare, shoulder_stability, tempo`, `faults/hip_position_counter.py` (rebuilt), NEW `utils/json_safe.py` (`nan_to_none`), `coaching/ipc_bridge.py` (all sends via `_send()` sanitizer); tests test_kinematics, test_valgus, test_faults, test_body_proportions, NEW test_hip_position_counter, NEW test_json_safe.
- IK: missing keypoints → NaN; `set_body_proportions` removed (`PELVIS_TILT_COUPLING = 0.4`); axis docstring +Z = BACK; sign fixes: `pelvis_tilt` (forward lean positive — was inverted in BOTH modes), `wrist_x` (in front positive), `trunk_rotation`/`pelvis_rotation` (left positive).
- Valgus (triangulated): femur tilt out of the plane spanned by hip–ankle line and the foot's forward direction, medial-positive, `asin(dev/femur)`; needs toe keypoint (`foot_confidence` includes toe); hip_rotation signed (internal positive); missing → NaN. Reads ~7.5° for a 10° knee swivel at 100° flexion vs ~12° GS → **3D thresholds (12/17/24) need re-tuning on real data**. Single-cam FPPA/KASR → NaN when missing.
- Rules: time cooldowns `KNEE_VALGUS_COOLDOWN_S=1.0`, `FORWARD_LEAN_COOLDOWN_S=5.0`, `BACK_ROUNDING_COOLDOWN_S=3.0`, `ELBOW_FLARE_COOLDOWN_S=3.0`, `SHOULDER_STABILITY_COOLDOWN_S=2.0`, `ECCENTRIC_TEMPO_COOLDOWN_S=1.0`, engine `DEDUP_INTERVAL_S=0.5` on `angles.timestamp`; NaN skipped. `ForwardLeanRule` scales `180 − (180 − base)×scale` idempotently; `KneeValgusRule` no scaling. `SymmetryRule` one fault per rep (median L−R within `BOTTOM_WINDOW_DEG=10` of deepest, ≥ 5 samples), emitted the frame after `in_rep` drops. `DepthRule.evaluate()` never emits; only `evaluate_max_depth`/`evaluate_depth_class`.
- `SignalRepCounter`: LS velocity over 5 samples; entry needs vel > entry_vel AND displacement > `entry_gate_cm` (0.1×min_depth); BOTTOM sign-aware once descent peaked ≥ 2×bottom_vel; false starts → IDLE; NaN/non-advancing timestamps ignored; gap > 0.5 s re-inits; new `rejected_rep_max_depth_angle`. 6 no-pause reps → 6/6 clean and noisy. No longer uses `HipPositionCounterConfig.position_min_cutoff/position_beta/velocity_ema_alpha`.

**Needs (integration):**
1. pipeline.py: remove `self._ik_solver.set_body_proportions(...)` (line ~631, now AttributeError); `apply_body_proportion_scaling` takes `segment_lengths.BodyProportions`; non-BiLSTM shallow reps: on `feedback == "go_deeper"` call `rule_engine.evaluate_rep_complete(rep_counter.rejected_rep_max_depth_angle, angles, rep_counter.rep_count + 1)`; never feed NaN into JointAngleFilter/PredictiveStateEstimator/DerivativeTracker (removing anyway); consider attaching post-rep symmetry faults to RepData via a rep-end hook.
2. Scripts: `scripts/tools/capture_audit.py:450`, `scripts/demos/visualize_video_squats.py:579` call `set_body_proportions`.
3. NaN-aware reductions in `_build_trajectory_sample`, `RepKinematicSummary`, session_tracker, `diagnosis/bridge.py`; wrap non-IPC JSON writers (`set_finalizer.py` ×4, `viz/pipeline_inspector.py`, `viz/valgus_debug.py`, `assessment_logger.py`) and direct `ipc_client.send_message` calls in pipeline_process.py with `nan_to_none`.
4. config.py/YAML: drop the three unused counter smoothing fields.

## Verification on harness (wave 2, before integration) — see harness/proposed_summary.md
Perfect calibration, mean over 8 scenarios × 3 seeds: raw(old tri) MPJPE 33.8 mm, knee MAE moving 14.3°, depth final −9.5°, knee NaN∨0 25 % → raw(new tri) 25.0 mm / 4.7° / −3.6° / 4.0 % → **proposed (Kalman+foot) 20.1 mm / 2.1° / +0.8° / 0.1 %**, still jitter 34.9 → 3.1 mm, stance MAE 26.7 → 6.7 mm. Tpose: 43.9 → 35.9 → 31.2 mm; depth −15.2° → +1.6°. Harsh: 75 mm/61 % zeroed → 32 mm/0.65 %. Preservation (proposed): valgus 1.10/1.02×, hip shift 0.90/0.92×, pelvic list 0.98/0.90× (was 0.43×), stance change 0.97/0.90× (was 0.02×); 0 false heel rise on clean reps. `current` (blender) gets WORSE on the new triangulator in tpose (knee@bottom −13.4°) — never ship new triangulator with the old blender.
Remaining error sources: T-pose calibration (17.9 mm floor), slow per-view detector bias (~17 mm; crop fix not simulated), transient excursions from B1/B2.
**Bugs to fix:** B1 Kalman re-acquire snaps onto 2-frame consistent outlier bursts (`repro_kalman_reacquire.py`) → require ≥ 3–4 agreeing rejections, never snap onto two-view-capped conf ≤ 0.3. B2 foot model holds a stepped foot at its old anchor 1–2 s (`repro_foot_step.py`) → re-seed gate reference on release, `REACQUIRE_REJECTED_FRAMES` ≤ 5, coherent ankle+toe jump = step. B3 `segment_lengths` conf ≥ 0.6 gate too strict for C3 confidences (31 %/4 % of frames pass) → ≥ 0.4. B4 IK MIN_CONFIDENCE 0.1 = u > 6 cm (harmless behind Kalman). B5 heel-rise floor ≈ 2 cm. B6 `Skeleton3D.from_numpy` + `recentre_at_hips` 53 µs/frame → integration should recentre in numpy. Depth per rep = single-frame max → use a robust bottom-window statistic (integration).
