# VelocityClamp + timestamp/frame-delivery audit — PROGRESS

Working dir (persistent): /Users/naiahoard/NowvaLiveKit/.claude/preik-audit/velocity_clamp/
Python: /Users/naiahoard/NowvaLiveKit/venv/bin/python (sys.path.insert(0, repo/src)). Do NOT modify repo files.

## Milestone 0 — code reading (DONE)
Read: velocity_clamp.py, test_velocity_clamp.py (6 tests), config (VelocityClampConfig 2.5 m/s; target_fps passed
from pipeline), multi_capture.py, multi_camera.py, triangulator.py, pipeline.py (init, reset_readiness_gate,
process_frame), filters.py (OneEuro, JointAngleFilter), derivatives.py, predictive_state.py, confidence_blend.py,
preik_chain.py, standing_gate.py (latches), hip_position_counter.py (uses `now`=time.time()), ml/inference.py
(BiLSTM counter uses skeleton.timestamp), mediapipe_fallback.py + rtmpose.py (time.time() after inference),
pipeline_process.py pacing (sleep target - sum(latency_ms); display/IPC excluded -> loop slower than 30 Hz).

Code facts:
- VelocityClamp: dt = ts - prev_ts if > 1e-6 else 1/target_fps (velocity_clamp.py:68-72); per-kpt clamp along
  displacement dir to v*dt (74-95); state = CLAMPED output (97); confidence ignored; timestamp default 0.0 never None.
- Triangulated ts = perf_counter at primary cam read (multi_capture.py:94,116; multi_camera.py:200). frame_index=0.
- Dropout hold ts = time.time() (pipeline.py:471,550). Only when frame!=None and skeleton_3d None (<2 views
  multi_camera.py:195 or triangulator returns None). Sync failure -> frame None -> early return, no hold.
- IK copies skeleton ts into JointAngles (analytical_ik.py:83) -> JointAngleFilter (filters.py:251 uses ts or
  time.time() if 0) -> DerivativeTracker (derivatives.py:123, dt<=0 -> 1e-6) -> Predictive (vel*0.2, cap 15 deg).
- Hip rep counter + PipelineFrame use `now`=time.time() (pipeline.py:471,736) -> independent of skeleton clock.
- Readiness gate latches; early returns only between reset_readiness_gate() and latch; clamp/blender reset at the
  same time (pipeline.py:383-384), JointAngleFilter/DerivativeTracker NOT reset.
- Blender runs before clamp; conf-0 triangulated kpts are at hip-centre origin (triangulator.py:89,137).

## Next
exp1 clock mix (real classes), exp2 duplicates, exp3 speeds (world vs hip-centred), exp4 outliers via real
DLTTriangulator (L/R swap, bad view, single-frame), exp5 bone-length distortion, exp6 gaps/first frame,
exp7 frame delivery sim, exp8 prototype gate design. Then REPORT.md.

## Milestone 1 — speeds + clock mix (DONE)
Scripts: synth_vc.py (min-jerk hip-height squat, 19 kpts, 3-cam rig r=3.2 m az 0/±50, f=1024, real DLTTriangulator),
exp3_speeds.py -> exp3_speeds.json, chain_sim.py (real Blender/Clamp/IK/JointAngleFilter/DerivativeTracker/Predictive),
exp1_clock_mix.py -> exp1_clock_mix.json.
exp3: hip-centred peak speeds (m/s): ankles/toes = hip world speed: heavy 0.86, BW normal 1.44, loaded light-fast 1.85,
  BW fast 2.31, explosive 3.24. Knees ~same as ankles. Wrists bar 0.5-1.8; BW arms-forward 2.45 (normal) 3.94 (fast).
  World frame: ankles/toes 0, knees 0.5-1.3 (1.8 explosive), shoulders 1.0-2.6, nose up to 3.1 (fast).
  Noise (static pose, per-axis std): sigma 3 px -> world 0.71 cm, hip-centred 0.87 cm (hip-mid noise leaks in, hips 0.50).
  frame-diff p99 hip-centred: 2.8 (2px) / 4.3 (3px) / 7.05 cm (5px), max 9.95 cm at 5px -> noise alone can hit 8.33 cm.
exp1 (bw_normal, dropout mid-descent): entering hold dt=+1.79e9 (clamp can't clamp; OneEuro alpha=1 snaps to raw held
  angle: +10.7 deg vs consistent clock); leaving dt=-1.79e9 -> clamp fallback 33 ms (16 kpts falsely clamped in 7-frame
  case vs 0 with consistent clock; ankle err 20.7 vs 10.9 cm), OneEuro/Derivative dt=1e-6 -> filter frozen one frame,
  knee accel spike 5e6..8e7 deg/s^2 decaying x0.7/frame (>1e5 for ~11 frames); knee vel +50..150 deg/s vs consistent.
  NEW BUG: 5th held frame has confidence*0 = 0 for every kpt (pipeline.py:543 decay=1-5/5) -> IK min_confidence 0.1 ->
  every angle 0.0 -> filtered knee 79->44 deg, vel -291 deg/s, predicted 28.8 vs 104.8 true (both clocks).
  Context: triangulation_frontend found sigmoid conf never <0.3 => RTMPose never returns None => in triangulated mode
  the hold path is currently ~dead code (latent; goes live once raw confidences are used). MediaPipe path: live.
  Gross DLT outliers (reproj>=15px) get conf*0.1 -> blender weight 0 -> HELD (not clamped).

## Milestone 2 — outliers, false clamping, set-timeout clock bug (DONE)
exp4_outliers.py -> exp4_outliers.txt/.json ; exp4b_consistent_outliers.py -> exp4b.txt/json (NB: importing exp4 re-runs it;
S3b/S2b rows are at the end of exp4b.txt) ; exp5_false_clamp.py -> exp5_false_clamp.txt/json ; exp9_set_timeout_clock.py.
exp4 (knee, error vs uncorrupted triangulation; variants none/clamp/blend/blend+clamp):
- single-frame 3D outlier, conf unchanged: clamp-only 60 cm -> 8.9 cm 1 frame (geom knee 39 -> 2.8 deg); production
  blend+clamp 60 cm -> 10.2/9.6 cm over 2 frames (standing: 9.2,16.0,9.4 over 3 frames, 35 deg) because blender state
  keeps 75% of the outlier and the clamp then lets it through next frame.
- DLT gross outliers (1-view 80/150 px, 1- or 2-view leg swap): reproj>=15 px -> conf 0.08 -> blender weight 0 -> HELD;
  clamp adds nothing. BUT IK min_confidence 0.1 -> knee_flexion returned 0.0 (IK knee error 92-121 deg) in every
  variant because blender passes raw confidences. Cross-cutting CRITICAL.
- 2-view-only keypoint, 80 px bad view: jump 25 cm, conf 0.37-0.39 -> blend 10.1-11.7 cm, blend+clamp 9.6-11.7 (no gain).
- persistent wrong solution 30 cm x10 frames: clamp follows it after 4 frames, then crawls back: +3 frames >5 cm.
- leg swap 10 frames during descent: blender holds (error 7.6 -> 32.9 cm), release: blend alone 7.4,3.0; blend+clamp
  26.9,21.2,15.0,7.2 -> clamp turns a 1-frame recovery into 4 frames (>5 deg knee for 14 vs 11 frames).
- hip outlier 30 cm single frame: all 19 kpts move >5 cm (re-centring); clamp 30 -> 18-19 deg knee but 2 frames.
exp5 (3 reps, no outliers): bw_normal/loaded at sigma<=3 px: clamp fires 0-1.4% frames (no effect). bw_fast sigma 3:
  4.8-8.6% frames, knee angle distortion <=4.4 deg; sigma 5: 13-26%, <=9 deg. bw_explosive: ~30% frames, lag 9-12 cm,
  knee angle distortion 16-20 deg, femur length change up to 2.8 cm, knee separation change up to 3.8 cm. BW arms-forward
  fast: wrists clamped ~57/..frames, lag 22 cm, forearm length change 5.8 cm. Bottom depth never changed (v=0 at bottom).
exp9 CRITICAL CONFIRMED: triangulated + BiLSTM: RepData.end_time = skeleton perf_counter ts (ml/inference.py:72 ->
  bilstm_counter.py:163) -> SessionTracker.last_rep_time (session_tracker.py:111,169); pipeline_process.py:1287
  check_set_timeout(time.time()) -> 1.79e9 - 8e4 > 30 s -> set ends after EVERY rep (3 reps -> 3 set_complete),
  reset_readiness_gate() each rep. MediaPipe clock: 0 set ends.

## Milestone 3 — duplicates, frame delivery, gaps (DONE)
exp2_duplicates.py -> .txt/.json: duplicates (identical ts) in triangulated mode: clamp unaffected (dt fallback, zero
  displacement; next unique frame gets true dt); OneEuro dt=1e-6 ~no-op; DerivativeTracker accel |p99| 600 -> 5.6e6..1.9e7
  deg/s^2 (no squat consumer today); predicted knee RMS +2..2.7 deg, max +4..11 deg; hip counter +1 spurious BOTTOM entry
  at 24 fps cam; rep counts unchanged. Single-cam style dup (new ts) -> clamp false-fires on next frame (16 frames @15 fps).
exp7_frame_delivery.py (model) + exp7b_real_capture.py (REAL MultiCameraCapture, fake 30 fps cams, 8+-2 ms read latency,
  pipeline pacing): get_synced_frames returns None on 26-41% of calls (model 34%; analytic 1-(1-0.20)^2=36%) because the
  newest primary frame's partners often have not been read yet -> only 14-18 Hz processed of 30 Hz; view capture-time
  spread p50 9-12 ms, p95 23-25 ms. Duplicates rare at 30 fps (0-0.5%), 7% at 24 fps, 42% at 15 fps (low light).
exp7c_fixed_sync.py: prototype "newest complete never-returned set + wait up to 100 ms, no pacing sleep": None 0.4-2%,
  dup 0%, processed 24-27 Hz.
exp6_gaps_first_frame.py: conf-0 toe/wrist at origin on first frame after reset: blend alone <3 cm after 4 frames;
  blend+clamp crawls 12 frames (toe err 92,85,76,...,8.5 cm). Skipped frames: dt-aware clamp 12/124 frames clamped at
  bw_fast vs 45/124 if dt were not time-aware (so time-awareness works when clocks are consistent).
NEXT: exp8 prototype innovation gate vs blend+clamp; then REPORT.md.

## Milestone 4 — gate prototypes (IN PROGRESS)
gate_proto.py (v1, alpha-beta 0.6/0.2, r0 6-8 cm, common-mode) -> exp8_gate_eval.txt: great on outliers (60 cm -> 1.7 cm,
  hip 30 cm -> 0.5 cm, first-frame 0 crawl) but DIVERGES on fast clean motion (alpha-beta lag -> rejections -> runaway;
  explosive rms 30-124 cm). Lesson: gate must be tuned for accel bias; common-mode rule must require hip antisymmetry.
exp8a_accel_innov.py: hip-centred peak accel knee 15/38/74 m/s^2 (normal/fast/explosive; synthetic upper bound),
  wrist arms-forward 9/24/48; WORLD: ankles/toes 0, knee 14/37/72, hip 5/13/25. Raw CV innovation p99.9: 7-10 cm @3px,
  11-16 cm @5px.
gate_proto2.py (v2: alpha 0.85 beta 0.5, r = r0 + 0.5 a_max tau^2, reject->predict, 2-frame candidate reacquire, 200 ms
  timeout, conf<0.15 = rejected, dt<=0 re-emit) + tri_world.py (world-frame DLT replica, recentre after gate).
exp8b_gate2_eval.py -> exp8b_clean.txt/json: CLEAN: gateW(r0 8 cm, a_max 80) and gateHC(r0 10 cm, a_max 80): 0 rejections
  in every tempo incl. explosive + arms at 3 and 5 px => output == raw (zero lag). Smaller r0 in hip-centred frame
  diverges (knee max 114 deg @ r0 8/a40 explosive 5px). blend+clamp on clean fast data: explosive knee rms 3.91->6.95 deg
  vs blend alone, wrist max 10.7->41.3 cm (arms explosive).
NEXT: exp8c outliers for gateW_0.08_80 / gateHC_0.10_80, then REPORT.md.
exp8c_gate2_outliers.py -> exp8c_outliers.txt/json (vs GROUND TRUTH): gateW(8cm,a80)/gateHC(10cm,a80): single 20-60 cm
  -> <=3 cm, angle <=5.2 deg (blend+clamp 8-16 cm, 19-39 deg standing); hip 30 cm -> <=3 cm/<=5 deg (b+c 16-18 deg);
  1-view/2-view 80 px -> <=3.5 cm, IK-zeroed 0 frames (b+c 1 frame); legswap 10f mid-descent gateW 9.8 cm / gateHC 14.3 cm
  vs b+c 33.8 cm + 3 crawl frames; persistent consistent 30 cm: all methods follow it (gate after 2 frames, b+c +3 frames).
exp8d_innov_v2.py: v2 predictor clean innovation p99.9/max: world 6.95/8.8 cm @3px, 9.33/11.5 @5px; hip-centred
  7.66/11.2, 11.8/14.9; at 15 Hz ~17-21 cm (world ankles/toes 4.5-7.5). Gate cost 0.039 ms/frame numpy.
exp1b_hold_zero_conf.py: 5th held frame conf 0 -> raw knee 0, trunk 180 (default); filtered knee 40.9->27.7, predicted
  55.9->16.9, vel 92->-54 deg/s mid-descent; idle phase negligible. No false forward lean (trunk defaults upright).
Other checks: test_velocity_clamp.py 6 pass; VelocityClampConfig.target_fps unused (pipeline passes pipeline.target_fps);
  90 s rest gap without filter reset: OneEuro snaps (ok), DerivativeTracker EMA keeps 4.15 deg/s stale velocity (low).
## Milestone 5 — REPORT.md (writing)
## Milestone 5 — DONE. REPORT.md write was blocked by the harness (subagents must return reports as text);
the full structured report was returned as the agent's final message to the lead.
