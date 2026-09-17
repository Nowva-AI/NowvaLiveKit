# Smoothing audit — PROGRESS (checkpoint 1)

Working dir: /Users/naiahoard/NowvaLiveKit/.claude/preik-audit/smoothing/
Python: /Users/naiahoard/NowvaLiveKit/venv/bin/python (add src/ to sys.path)

## Status
Phase: code reading DONE. Numeric experiments NOT yet started (no scripts yet).

## Code facts gathered (file:line)
- OneEuroFilter `src/biomechanics/utils/filters.py:38-144`: alpha=1/(1+tau/dt) (matches paper);
  derivative from RAW previous value (`last_value = x`, :135) = Casiez reference-code variant;
  first sample returns x, dx_hat=0 (:102-110); dt<=0 -> 1e-6 (:114-115); reset clears (:139).
- JointAngleFilter `filters.py:181-294`: phase params idle(0.3,0.003) desc(1.0,0.007) bottom(0.8,0.005)
  asc(1.0,0.007); d_cutoff always 1.0 (not passed, :231). Beta units deg/s. Timestamp = angles.timestamp
  (skeleton ts; perf_counter in triangulated mode) else time.time() (:250-251).
- KeypointPositionSmoother `position_filter.py:19-88`: 3*N python OneEuroFilter objects, per-scalar loop,
  confidence ignored; DISABLED in preik_chain.py:46. Skeleton2DSmoother (:91-137) display only, active.
- PositionFilterConfig config.py:221 (0.8, 4.0, 1.0); DisplayFilterConfig :228 (1.5, 0.5, 1.0).
- DerivativeTracker `derivatives.py:67-194`: finite diff of FILTERED angles, dt<=0 -> 1e-6 (:134-135),
  EMA alpha 0.3 not dt-aware; accel = diff of smoothed vel / dt.
- PredictiveStateEstimator `predictive_state.py:17-110`: knee/hip flexion += vel*0.2 s, clamp 15 deg.
- pipeline.py:654-691: update_phase(rep_counter.phase) (one frame late) -> filter_angles -> derivative ->
  eval_angles = predict(...) used for RuleEngine.evaluate (ALL per-frame rules incl. DepthRule.evaluate
  max-depth tracking depth.py:98-104 and SymmetryRule knee L/R diff symmetry.py:74-76).
- Bottom frame for diagnosis = max avg_knee_flexion of FILTERED (not predicted) angles, pipeline.py:~668.
- reset_readiness_gate (pipeline.py:~362-391) resets position smoother but NOT _angle_filter nor
  _derivative_tracker.
- Dropout hold uses timestamp=time.time() (pipeline.py ~535) while triangulated skeletons use
  perf_counter -> clock mix feeds OneEuro/DerivativeTracker dt (huge +dt then huge -dt -> 1e-6).
  multi_camera.get_pose returns skeleton_3d None when triangulation fails (multi_camera.py:196).
- Squat rules using derivatives: none (tempo not in squat profile). Predictive is the only derivative consumer.
- Hip counter (hip_position_counter.py:66-72) has its own OneEuro(1.5, 0.01 cm units)+EMA 0.3 on rep signal;
  its phase gates KneeValgusRule (bottom only) and JointAngleFilter params.
- RTMPose ONNX outputs simcc_x 384 bins / simcc_y 512 bins -> quantization 0.5 input px ->
  3.33 px (x) / 1.41 px (y) in 1280x720 frame (~1.0 cm / 0.4 cm at 3 m, f=1024).
- No real triangulated recordings found in repo (user_test_runs pipeline_inspect data are single-cam MediaPipe).

## Plan / what's left
1. synth.py: pelvis-driven squat (2-link leg IK, fixed ankles, toe-out, valgus swivel on some reps, trunk lean),
   19 kpts, 30 Hz, varied tempos + standing; noise = 3 synthetic cams (f=1024, 3 m arc) + 2D gaussian +
   SimCC quantization + DLT + hip recentering; also plain per-axis gaussian sensitivity.
2. exp_oneeuro_check.py: repo OneEuro vs reference variants; duplicate-frame / clock-mix / reset tests.
3. exp_lag.py: effective cutoff, group delay, position lag, depth undershoot (cm, deg), valgus attenuation,
   bone chord shortening.
4. exp_stack.py: raw / each layer / full stack / proposals -> angle error, predictive overshoot, false faults.
5. exp_grid.py: One Euro grid search; CV/CA Kalman (vectorized), fixed-lag 2-3 frames, offline RTS per rep.
6. exp_cost.py: per-frame cost of repo smoother vs vectorized.
7. REPORT.md with verdicts.

## Checkpoint 2 (scripts written: synth.py, kin.py, smoothers.py, harness.py, check_synth.py, exp1_oneeuro_correctness.py, dbg_counter.py)
- synth verified: vectorised kin == repo AnalyticalIKSolver/TriangulatedValgusEstimator (max diff 0.0).
  sigma2D 2/4/8 px -> 3D leg err std (x,y,z) cm [0.48,0.43,0.81]/[0.92,0.87,1.50]/[1.79,1.66,2.98];
  quantisation alone [0.14,0.08,0.32]; knee flexion std standing 1.9/2.8/4.7 deg. Real RTMPose per-view jitter
  (confidence_blend/e0_summary.json) knee/ankle 3-4 px on clean 30fps video -> 4 px is baseline.
- E1 (exp1_output.txt): repo KeypointPositionSmoother == OneEuroVec(raw) exactly. Casiez reference (github main,
  saved casiez_reference_OneEuroFilter.py) derives dx from PREVIOUS FILTERED value -> repo (raw prev) differs by up
  to 2.8 cm on knee y; reference ignores non-increasing timestamps (keeps last freq) whereas repo uses dt=1e-6.
  Cutoff with raw variant moving p50 1.36 Hz vs filtered-variant 2.23 Hz (feedback on lag) -> repo variant lags more.
  Duplicate frames / clock-mix: benign for angles (alpha~0), but DerivativeTracker acceleration explodes to
  3.6e6-5.3e6 deg/s^2 (unused by squat rules). No-reset across rest: vel -21 deg/s carry-over, pred 0.9 vs 5 (low).
- PredictiveStateEstimator: at normal squat knee speeds (175 deg/s) 0.2 s*v = 35 deg -> CLAMPED at 15 deg for 40 of
  60 frames of a 1 s descent/ascent -> prediction is basically +/-15 deg square wave.
- NEW HIGH finding (adjacent): SignalRepCounter (hip_position_counter.py:236-239) DESCENDING->BOTTOM requires
  |vel|<5 cm/s; at 30 Hz a no-pause reversal jumps from +v to -v, band skipped -> on CLEAN synthetic data reps 0+1 and
  2+3 merged, BOTTOM phase entered during STANDING (frame 129 vs true bottom 95). Phase drives JointAngleFilter params
  and KneeValgusRule bottom gating. (dbg_counter.py). Fix: sign-aware test (vel < +threshold) or peak detection.
- Next: add oracle-phase option to harness, then exp2 (lag/depth/valgus/bones), exp3 stack table, exp4 grid+Kalman, exp5 cost.

## Checkpoint 3 — E2 + E4 first pass (exp2_output.txt, exp2_results.json, exp4_output_sigma{2,4,8}.txt)
- harness.py: phase_source="oracle" (near-bottom = s>=95% depth) vs "counter"; summarize(); run_stack replicates pipeline.
- E2 current pos One Euro (0.8,4,1): effective cutoff knee idle ~0.93-1.08 Hz (147-171 ms group delay);
  peak descent 1.2/2.2/3.3 Hz (x/y/z) = 131/72/48 ms. Noise-free: knee-angle lag 46 ms, mid-descent -7.9 deg,
  depth undershoot mean -0.8 deg (worst -1.4 fast rep), rep-signal -0.9 cm (worst -1.35), thigh chord min -1.15 cm.
  Paper-derivative variant: lag 29 ms, mid -4.7, depth -0.4. Bone-dir param kills chord shortening (0.0) but beta
  not retuned for unit vectors -> more noise.
  sigma 4px: raw depth peak BIAS +2.0 deg, raw valgus peak bias +4.5 deg (8px: +4.2 / +10.1) => max-of-noise bias,
  smoothing needed for peak metrics. blend_only lag grows with noise (conf drops): 26 ms@2px, 36@4px, 75@8px.
- E4 sigma4 first pass (J = knee rms moving + |depth| + |valgus peak| + standing std, deg): raw 11.3; current OE 9.18
  (lag 46 ms); best OE raw-deriv (0.3,8,2) 6.63; best OE paper-deriv (0.3,8,2) 6.50 lag 16 ms; causal CV Kalman lag0
  best 8.72; CV fixed-lag 1 frame 5.65, 2 frames 4.38, 3 frames 4.31 (q=1,r=0.02); CA lag2 q=100 4.49;
  offline filtfilt Butterworth 3 Hz 4.07. Grid edges hit -> extend; synthetic cosine reps favour heavy smoothing ->
  add HARD session (1.2 g bounce reversal + 300 ms transient valgus) before recommending.

## Checkpoint 4 — E3 provisional (exp3_output_sigma4_oracle.txt; proposals P1-P5 params provisional, grid still running)
- smooth, 4px, oracle phase: B production (blend+vclamp -> JAF -> pred): knee RMS moving 14.5 deg, lag 115 ms,
  mid-descent -18.9, live depth -4.5 mean/-7.8 worst, predicted depth +6.3; bottom frame 4.6 frames late, true knee at
  selected frame -5.1 deg. C designed full (+pos OE): lag 160 ms, depth -5.0/-8.3, bottom frame true knee -12.1 deg.
  F JAF only: lag 77 ms, depth -3.3/-5.6 => JAF is the main lag source. D blend only: lag 36 ms at 4 px (w~0.47).
  G predictive only: pred depth +15 deg, 13.6 false symmetry events/session; A raw: 10.4 false symmetry events.
- hard (bounce + 300 ms transient valgus 17.7 deg): valgus peak attenuation B -12.6 (71%), C -14.9, JAF -9.6, pos OE
  current -7.3, blend -5.2, OE tuned(0.3,8,1 paper) -3.4, KF CV q=1 lag2 -7.3 (over-smoothed!). Live valgus rule can
  never fire on transient mid-ascent cave (bottom-phase gate) -> diagnosis must use trajectory.
- Rep counter: reps counted A raw 6.0, B 5.4, C 4.8, KF lag2 4.4, lag3 4.0 (truth 6): smoother input => MORE merged reps
  (band-skip bug). Depth fault TP/FP dominated by counter (DepthRule per-frame keyed on rep_count+1).

## Checkpoint 5 — E4 extended (smooth+hard sessions, J_mean) + E5 cost + fixedlag_fast.py
- J_mean (avg smooth/hard) @2/4/8 px: raw 5.71/10.03/19.88; CURRENT pos OE (0.8,4,1 raw) 11.86/12.87/14.99 (hard valgus
  -7.4); best OE paper-deriv (1.2,16,2) 4.32/7.09/13.79 (lag 9 ms, hard valgus -1.6); CV KF causal lag0 best 5.39/8.67/
  ~15 (no better than OE); CV KF fixed-lag 1 (q=3,r=.01) 3.68/6.02/10.46; lag 2 3.77/6.00/10.07; CA lag2 q=1000
  3.64/5.86/10.21; offline RTS 3.48/6.04/10.9; offline Butterworth filtfilt 6 Hz 3.40/5.99/10.86 (3 Hz over-smooths
  transient valgus -7.3). => fixed-lag 1-2 frames reaches offline quality; separate between-rep offline smoothing adds ~0.
- Grid details @4px: fixed-lag2 q=3 r=.01 knee rms moving 1.25, |depth| 0.54, |valgus| 1.80, hard valgus -2.7,
  standing std 1.91, jitter 6.3 mm/frame. Best noise-level-specific q/r shifts (q/r^2 higher at 2px, lower at 8px).
- E5 cost (M2, per frame): Skeleton3D round trip 31.8 us; repo ConfidenceBlender 38.2; repo KeypointPositionSmoother
  108.2 (57 objects); OneEuroVec 7.7; KalmanVec lag0 19.6 / lag2 84 (np.linalg.inv); fixedlag_fast.FixedLagCV closed-form
  lag0 11.7 / lag1 22 / lag2 32.5 / lag3 42 us (== reference to 3e-16 m); alpha-beta steady state 10.1 us.
- Running: exp6_adaptive.py (R from reprojection error; One Euro with shared skeleton speed).

## Checkpoint 6 — E6/E7/E8 + final E3 (all outputs saved). NEXT: write REPORT.md (only step left)
- E6: KF CV lag2 with R = clip(0.005 m/px * frame-median reprojection error) and q=10: J_mean 3.55/5.93/9.73 @2/4/8px =
  best single setting across noise (fixed R best-per-sigma 3.42/6.00/9.77). Shared-speed One Euro no gain (7.36@4px).
- E7a: IK missing sentinel 0.0 -> JAF output 78 -> 24.8 deg for 2 dropped frames, 5+ frames recovery (HIGH, symmetry FP).
- E7b: KeypointPositionSmoother with conf-0 (0,0,0) ankle -> 64-76 cm error decaying over 5+ frames; ProposedSmoother
  (R=inf for conf 0) 0.1 cm.
- E7c: Skeleton2DSmoother (1.5, beta 0.5 px/s) standing jitter 8.1 -> 5.6 px/frame only; px-retuned (1.0, 0.02, paper)
  1.6 px/frame and lower moving error 3.97 vs 4.81 px.
- E7d: JAF phase-aware worsens depth (-3.3 vs -2.4 mean, worst -5.6 vs -4.2) vs fixed; with counter phase standing std
  2.5 vs 1.2 and lag 84 ms.
- E7e: lag-only chord shortening: current OE thigh min -1.15 cm; tuned OE / KF lag2 -0.27 cm -> reparameterisation unneeded.
- E8 pure lag (noise-free, 4px conf): production B lag 115 ms, mid-descent -19.2, depth -4.6 mean/-8.1 worst, pred +6.0,
  hard transient valgus -11.9 of 17.7 (67%); C full -5.0/-8.5, valgus -14.4; JAF only -3.6/-5.9, valgus -9.3;
  blend only -0.75/-1.8 (lag 36 ms), valgus -4.6; pos OE current -0.8/-1.4, valgus -6.9; OE tuned -0.09/-0.18, valgus
  -1.05; proposed lag2 -0.13/-0.30, valgus -2.3.
- E3 final (4px oracle, smooth): proposed lag2 knee RMS moving 1.27 (prod 14.46), depth +0.50/-0.39 worst (prod
  -4.46/-7.79), stored bottom-skeleton knee err +0.50 (prod -3.21), symmetry FP 1.2/session (raw 10.4, prod 0.2), +pred
  8.6; +JAF lag 79 ms. 8px: proposed depth +0.73, symmetry FP 6.0 (prod 10.0, raw 13.2).
- E3 counter phase: bottom-frame selection broken by SignalRepCounter: stored bottom knee err raw -38.7, prod -9.7,
  proposed -11.7 deg (vs oracle +0.5) -> counter fix is prerequisite.

## Checkpoint 7 — DONE
All experiments complete. Writing REPORT.md was refused by the subagent harness ("return findings as text"), so the
full report is in the agent's final response to the coordinator. All scripts/outputs/JSON are in this folder.
