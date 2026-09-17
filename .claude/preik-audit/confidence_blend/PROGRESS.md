# confidence_blend audit — PROGRESS

## Milestone 0 (done, prior run): E0 real RTMPose-m halpe26 confidences
- Scripts: e0_rtmpose_conf.py (production decode on 3 real 1280x720 squat mp4s, raw pre-threshold), e0b_conf_details.py
- Data: *_rtm.npy (T,19,3) [x_px,y_px,conf], e0_summary.json
- Key numbers: sigmoid conf overall min 0.531, p1 ~0.57, p50 ~0.64, p99 ~0.70, max 0.723 (all 3 videos).
  Never below 0.3 threshold. Leg conf standing ~0.60-0.64, bottom ~0.65. Frame-to-frame |dconf| p50 0.005-0.008, p95 0.02-0.03.
  Low conf does predict larger 2D residual: conf 0.5-0.6 bin resid RMS 21-28 px vs 0.6-0.7 bin 5-11 px.
  Argmax decode: 32-47% of static frames have dx==0 exactly (quantised at ~3.33 px/1280 cell).

## Milestone 1 (in progress, run 2): read source + wiring
- DONE reading: confidence_blend.py, test, config (yaml 68-70 min .1 max .9), preik_chain.py, pipeline.py wiring
  (init 187-191, reset_readiness_gate 382-388, dropout hold 537-565 timestamp=time.time(), readiness early return 590-601,
  chain 608-622), triangulator.py (conf formula 125-131, recentre only conf>0 134-137), rtmpose.py decode 235-239 +
  threshold 262-266, mediapipe_fallback.py (visibility, <0.3 -> (0,0,0) conf 0; 21 kpts), velocity_clamp.py, IK
  _get_point conf>=0.1 (analytical_ik.py:158-162), valgus _MIN_CONFIDENCE 0.1, JointAngleFilter, StandingPoseGate latch.
- Snapshot copies of smoothing agent's synthetic squat + DLT noise generator: synth_snapshot.py, kin_snapshot.py
  (copied 2026-09-16 from ../smoothing/, unmodified) so results are reproducible here.
- REAL live loop rate (single-cam MediaPipe user_test_runs/2026-07-29_*/pipeline_inspect/data.json): measured 11.5-11.9 fps,
  dt p50 86 ms, p5 51 ms, max gaps 3.5-4.4 s. Data has raw + confidence_blend stages (21x3 flattened, 4 dp) -> can back out
  real per-keypoint blend weights w = d(blend)/(raw - prev blend).

## Planned experiments (run 2)
e1_time.py (dt dependence, dup/skip frames, dropout hold), e2_lag.py (squat lag/noise at loop rates + alternatives at equal
noise reduction + bone length), e3_semantics.py (conf vs 3D err calibration, weight range, real MediaPipe weights),
e4_latch.py ((0,0,0) seed, return after absence, NaN poisoning, IK band), e5_stack.py (velocity clamp outlier smear,
post-IK stack contribution). Then REPORT.md.

## Milestone 2 (done): E1 time-awareness — e1_time.py -> e1_results.json
- Not time-aware: alpha=w per call, dt/timestamps never read. Ramp lag = (1-w)/w frames (verified numerically with production class:
  w=0.49 -> 2.78 cm @30fps, 7.18 cm @11.6fps for 0.8 m/s). Lag ms: 35 ms @30, 69 @15, 90 @11.6 (real measured loop rate).
  -3dB cutoff w=0.49: 3.3 Hz @30fps, 1.3 Hz @11.6fps. Static noise std ratio sqrt(w/(2-w)) = 0.57 (rate independent).
- Loop vs 30fps camera (duplicates share ref_ts): 25 ms loop -> 26% dups, lag 2.06 cm p5 1.22 p95 2.67 (smoothing varies
  frame to frame); 16.7 ms loop -> 50% dups, lag 1.6 cm, noise ratio 0.70; 86 ms loop -> 1.6 skipped frames/iter, lag 7.2 cm, 105 ms total.
- Dropout hold (pipeline decays conf 0.8..0 over 5 frames): weights 0.37,0.24,0.12,0,0; blender adds ~1 cm during hold and ~3-6 frames
  recovery after (ankle err 9.5,6.4,4.8,4.0,3.5 cm vs 0 raw). Clock mix irrelevant to blender (ignores timestamps).

## Milestone 3 (done): E2 squat lag/noise/bones — e2_lag.py -> e2_results.json (filters_alt.py = EMA / alpha-beta / One Euro refs)
Sim: synth_snapshot session (6 reps, 0.7-1.5 s descents, 2 valgus reps), 3 cams, 3 px 2D noise + SimCC quantisation, DLT,
real E0 conf traces as view confs -> triangulated conf p5/50/95 0.41/0.49/0.57, reproj 1.0/2.6/4.6 px, blend weight median 0.49.
(bug fixed during run: synth.session(reps=[]) silently uses DEFAULT_REPS; static tuning now uses explicit standing world.)
- Blender == fixed EMA alpha 0.5 in effect (static noise ratio 0.58 vs 0.58; lag 3.0 vs 3.0 cm). Confidence modulation adds nothing measurable.
- Static leg noise RMS 1.58 -> 0.89 cm. But knee-flexion RMSE over session WORSE: raw 2.56 / blender 3.23 deg @30fps; 2.62/5.61 @15; 2.53/7.33 @11.6.
- Lag along velocity (speed>0.5 m/s) knee/ankle: 3.0/3.4 cm @30, 5.3/5.8 @15, 6.7/6.9 @11.6. Knee flex err when |dknee|>100 deg/s:
  raw 2.1 -> 4.8 deg @30, 9.4 @15, 12.1 @11.6. xcorr lag 27/54/73 ms.
- Peak knee flex (noiseless, per rep): -0.6..-1.6 deg @30, -1.7..-5.5 @15, -2.7..-7.4 @11.6 (paused rep ~0). Depth (hip-ankle y) undershoot up to 1.5/4.6/7.6 cm.
- Bones (noiseless input, motion frames): femur mean -0.15/-0.49/-0.75 cm, p5 -0.5/-1.4/-2.3; tibia p5..p95 -0.6..+0.9 (30) -1.6..+1.35 (11.6)
  vs fixed EMA tibia ~0 => per-joint weight differences add +-1 cm tibia jitter. Valgus peak attenuation small (<1 deg).
- Equal-noise One Euro (min_cutoff 4/1.5/1.0 Hz, beta 8 /m/s): lag 1.5/2.2/2.5 cm, fast knee err 2.2/3.4/4.4 deg, knee RMSE 2.0/2.5/3.1, peak knee -0.1..-1.1.
- Equal-noise alpha-beta (CV Kalman steady-state): lag 1.5-1.7 cm @30 but OVERSHOOTS reversals: peak knee +2.3..+5.5 deg, depth +2..+4.5 cm -> unsafe for depth.

## Milestone 4 (done): E3 semantics (e3_semantics.py -> e3_results.json), E4 latching (e4_latch.py -> e4_results.json)
E3 sim (3 px noise, 3% 2D outliers 40 px):
- 3 views: conf bins 0.3-0.4/0.4-0.5/0.5-0.6/0.6+ all have 3D err p50 ~1.1-1.2 cm (NO information above 0.3); conf<0.1 bin p50 8.7 cm.
  95.5% of >5 cm errors get w<0.2 (outlier detection works, binary); but good points (<2 cm) get median w 0.49 -> pure lag.
  Max weight ever 0.68; conf>=0.9 never (passthrough unreachable). Spearman(conf,err) -0.23.
- 2 views: only 72% (opposite cams) / 61% (adjacent cams) of >5 cm errors get w<0.2; conf 0.3-0.4 bin p95 err 15-24 cm.
- REAL single-cam MediaPipe (backed-out production weights from 3 user_test_runs, axis-consistency p95 0.002): legs w p50 1.0,
  frac w>=0.99: lknee .96 rknee .80 lank .89 rank .96 toes .75, heels p50 0.90; nose/shoulders 1.0. => blender ~no-op single-cam,
  strong EMA triangulated. [0.1,0.9] mapping was built for MediaPipe visibility.
- triangulation_frontend agent: sigmoid squashes raw 0.37-0.96 -> 0.59-0.72; no-person images give 0.53-0.57 (all pass 0.3).
E4 (production classes):
- (a) toe untriangulated on first post-reset frame -> stored (0,0,0) (hip mid, 101 cm from true): blend only 52,23,14,8,4 cm (6 frames
  to <3 cm); blend+clamp 93,84,...,9 cm, 12 frames (400 ms) — crawl is the clamp's; output conf 0.43-0.55 throughout (consumers trust it).
- (b) ankle conf 0.06 for 1 s during descent: blender latches standing ankle -> err mean 23 / max 57 cm vs raw outlier 17 / 38 cm;
  rep signal err 26 vs 18 cm. After return: blend 31,16,9,5 cm; blend+clamp 50,43,36,27,19,10 cm (6 frames); conf reported 0.49.
- (c) blender floor 0.1 == IK floor 0.1: conf 0.10 -> w 0 (frozen) but IK uses it: step stand->bottom gives knee 4 deg vs 120 true;
  conf 0.14 -> 8 deg; 0.2 -> 15; 0.3 -> 28.5 (frame 1). conf 0.10-0.14 == reproj 11.5-12.5 px at base 0.6.
- (d) NaN or inf keypoint (conf 0.06) at t=1 -> blender output NaN FOREVER (0*inf=nan, nan*0=nan), conf back to 0.5 -> IK knee_flex NaN
  every frame until reset. Triangulator maps reproj NaN/inf to the 0.1 branch (point kept). Trigger likelihood low (DLT w==0), impact total.
- (e) 10 s gap without reset (None frames return early), return mid-squat: knee flex err -35.7, -17.7, -8.9, -4.5 deg frames 0-3.

## Milestone 5 (done): E5 stack — e5_stack.py -> e5_results.json
- (a) Single-frame knee spike partly flagged (reproj<15): 25 cm @conf .4 (w .375): raw 25 (1 frame), clamp-only 8.3 (1 frame, 8.3 cm*frames),
  blend-only 9.4,4.7,2.3,1.2 (18.7 cm*frames, 4 frames>1cm), blend+clamp 8.3,4.7,2.3,1.2 (17.7) — clamp cuts frame 1 only; blender state
  keeps the unclamped 9.4 and leaks it. 15 cm @ .45: blend 6.6,3.3,1.6 (13.1) vs clamp 8.3 (8.3).
- (b) Post-IK (IK -> JointAngleFilter phase-aware -> DerivativeTracker -> Predictive 0.2 s), knee_flex_l vs truth, 3 px noise:
  30 fps: IK-out fast|e| 1.8 (no filter) vs 5.4 (blend+clamp); filtered fast desc signed -11.9 vs -16.7; filtered peaks -2.7..-4.6 vs -3.3..-6.3;
          predicted RMSE 3.69 vs 4.28, predicted peaks +8..+11.5 vs +7.4..+10.2 (blender partially masks predictor overshoot).
  11.6 fps: IK-out fast 2.5 vs 12.5; filtered desc -12.0 vs -23.4; filtered peaks -3.7..-8.8 vs -7.1..-15.2; predicted RMSE 5.1 vs 8.6, fast 4.2 vs 14.7.
  VelocityClamp never fired on realistic noise (clamp_only == no filter).

## Next: E6 prototype replacement gate (reproj threshold table, hold/extrapolate <=100 ms, dup/NaN/seed tests) then REPORT.md

## Milestone 6 (done): E6 gate prototype — keypoint_gate_proto.py, e6_gate.py -> e6_results.json
- Reproj threshold table (LEG, 3% 40 px 2D outliers): 3view s=3px thr 8 px catches 96.5% of >5 cm / 99.8% of >10 cm, false-flags 0.1% of
  <2 cm; s=6px thr 8 flags 12% good points -> threshold must scale with noise (10-12 px). 2view: thr 8 catches only 73% (epipolar blind).
- Session w/ outliers, gate applied AFTER hip recentre (like blender): 30fps 3view knee RMSE raw 6.86 / blender 4.05 / gate 4.70, fast err
  3.5/5.8/2.7; 11.6fps: 6.86 / 8.99 / 4.65, fast 4.0/13.9/3.3, frames >10deg 46/121/22. Blender wins RMSE only at 30 fps on outliers that pass
  (hip outliers leak into ALL points via recentre) -> next: gate BEFORE recentre (world frame, feet static) + optional One Euro.
- Gate edge cases OK (no (0,0,0) state, NaN rejected, duplicates identity, 10 s gap snap). Cost 33 us/frame numpy vs blender 38 us incl pydantic.

## Milestone 7 (done): E6b world-frame gate — e6b_gate_world.py -> e6b_results.json (+ ZOH vs CV check inline)
- Gate BEFORE hip recentre (world frame), thr 8 px, hold <=0.1 s: 30fps 3view outliers knee RMSE 3.25 (ZOH 2.96) vs blender 4.05 vs raw 6.86;
  fast err 2.6 vs 5.8; frames >10 deg 13 (ZOH 8) vs 36; leg p99 4.8 vs 7.7 cm. 11.6fps: 4.2 vs 9.0, fast 3.5 vs 13.9, >10deg 13 vs 121.
  Clean data: gate == raw exactly (2.71/2.78) vs blender 3.40/7.17. 2-view outliers: gate 4.65 vs blender 3.97 @30 (blender better: gate is
  epipolar-blind), 5.18 vs 7.31 @11.6. Gate + equal-noise One Euro: 2.69/2.27(clean) @30, 4.07/2.85 @11.6.
- Ankle lost 1 s: world-frame gate (ZOH, hold 0.1 s then conf 0 at last valid pos) mean err 1.5 cm vs blender+clamp 22.7; after return 2.0 cm vs
  50,43,35,27,18,9 cm. CV extrapolation with 2-sample velocity over 1 s hold drifts (28 cm) -> keep hold short / ZOH.
- ZOH >= CV at 30 fps (2.96 vs 3.25), slightly worse at 11.6 (4.3 vs 4.2) -> recommend ZOH (simpler).
- No in-place Point3D mutation found in active code (first-frame aliasing harmless today). Existing 5 unit tests pass.

## Next: write REPORT.md (all experiments done)

## Milestone 8 (done): final report delivered as the agent's final message (the harness refused writing REPORT.md from a subagent).
Verdict: triangulated REMOVE (replace with world-frame KeypointGate before hip recentre: reproj<=~8px & finite & n_views>=2 pass-through,
ZOH <=0.1 s, then conf 0 at last valid pos; never store (0,0,0)/NaN). Single-cam MediaPipe REMOVE (backed-out weights ~1.0 => no-op).
Coupling: remove together with PredictiveStateEstimator fix (blender masks predictor overshoot at bottom).
