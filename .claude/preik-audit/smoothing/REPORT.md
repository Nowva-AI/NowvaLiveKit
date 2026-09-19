# Smoothing Audit — Pre-IK and Post-IK Temporal Filters (triangulated 3-camera)

(Saved by lead from the agent's final message; subagent sandbox blocked writing REPORT.md. Checkpoint log with all numbers in PROGRESS.md. Repo untouched.)

**Summary:** today's smoothing is the wrong kind, but no smoothing is also wrong. Most damage comes from the post-IK angle filter and prediction step, not the pre-IK chain.

## Method
- **Replay harness** (`harness.py`) replicates `pipeline.process_frame` step for step: pre-IK chain → IK → JointAngleFilter → DerivativeTracker → bottom-frame buffering → prediction → real squat RuleEngine → SignalRepCounter.
- **Synthetic data** (`synth.py`): Y-down, hip-centred, 19 kpts, 30 Hz. "Smooth" set: 6 reps, varied tempos, 120/95/75° depth, ~15° knee cave on some reps. "Hard" set: 5 bounce reps with a 300 ms knee cave mid-ascent (peak 17.7°).
- **Noise path:** 3 cameras, per-view 2D jitter 2/4/8 px, RTMPose 3.33 × 1.41 px argmax grid (read from the ONNX model), DLT, triangulator confidence formula, hip recentring → 3D error ~0.5–3 cm per axis. 4 px baseline (real RTMPose knee/ankle jitter on clean video 3–4 px).
- **Checks:** vectorised angle formulas = repo IK + valgus estimator (0.0° diff); vectorised One Euro = `KeypointPositionSmoother` exactly.
- **Caveats:** white noise per view, perfect calibration, no L/R swaps. Lag/attenuation numbers and bugs don't depend on these. Kalman q and R gain must be re-fit on real triangulated recordings (none exist in repo).

## Headline
1. **Production today (blend → JointAngleFilter → prediction) loses depth and knee-cave peaks.** Noise-free (E8): lag 115 ms, knee error at mid-descent −19.2°, depth 4.6° shallow mean / 8.1° worst, 67 % of the 300 ms knee-cave peak lost (−11.9 of 17.7°). The designed full chain (adds position One Euro) is worse: 160 ms lag, −8.5° worst depth, −14.4° valgus.
2. **Raw is also wrong for peak metrics.** 4 px: per-rep valgus peak +4.5° high, depth +2.0° high (max of noisy signal biased upward); 8 px: +10.1°, +4.2°. Raw also yields 10–13 false L/R asymmetry faults per 6-rep set.
3. **Proposed:** constant-velocity Kalman per keypoint before IK, **2-frame (66 ms) fixed lag**, measurement noise from the frame's reprojection error; no post-IK angle filter, no prediction. 4 px, 5 seeds, true phases (E3):
| Metric | Proposed | Production |
|---|---|---|
| Knee RMS while moving | 1.27° | 14.46° |
| Depth error mean / worst | +0.50° / −0.39° | −4.46° / −7.79° |
| 300 ms knee-cave peak | −2.2° | −12.6° |
| Knee angle of stored bottom frame | +0.5° | −3.2° |
| Cost/frame (M2) | 32 µs | 108 µs (old position smoother alone) |
Lag 2 already matches offline zero-phase on the combined objective (6.00 vs offline Kalman smoothing 6.04, Butterworth 5.99) → between-rep diagnosis doesn't need a separate offline smoother.

## Findings (reproduced unless marked SUSPECTED)
- **S1 CRITICAL — JointAngleFilter** (`filters.py:181-294`, called `pipeline.py:654-657`): phase-aware One Euro on every angle, stacked on the blender; main lag source. Noise-free alone: 79 ms lag, depth −3.6° mean / −5.9° worst, knee-cave peak −9.3°. **Fix:** remove from analysis; smooth positions once before IK.
- **S2 HIGH — bottom frame chosen from lagged angles** (`pipeline.py:674-678`): picked by max of lagged filtered knee angle, skeleton stored from less-lagged pre-IK stream. 4 px production true phases: frame 4.6 frames late, true knee at that frame −5.1° from peak, stored skeleton −3.2° (designed full chain −12.1°). Real counter phases: −9.7° production, −38.7° raw. **Fix:** bottom frame, rep signal, trajectory from one smoothed stream (+ S4).
- **S3 HIGH — PredictiveStateEstimator** (`predictive_state.py:56-63`, `pipeline.py:689-691`): 175°/s × 0.2 s = 35° → sits at ±15° clamp for 40 of 60 frames of a 1 s descent/ascent; feeds DepthRule per-frame max (`depth.py:98-107`) and SymmetryRule (`symmetry.py:74-76`); never extrapolates valgus. Predicted depth overshoot +6.3° production, +13.0° behind a good smoother, +15° raw; false symmetry events/set 1.2 → 8.6 (4 px), 0 → 4.2 (2 px). **Fix:** remove (cues fire between reps).
- **S4 HIGH (adjacent) — SignalRepCounter bottom detection** (`hip_position_counter.py:245`): BOTTOM only when |velocity| < 5 cm/s; at 30 Hz a no-pause rep jumps sign and skips the band. Clean data (`dbg_counter.py`): reps 0+1 and 2+3 merge; BOTTOM entered at frame 129 while standing (true 95); DESCENDING 8 frames (267 ms) late. **Worse with better smoothing:** reps counted of 6 — 6.0 raw, 5.4 production, 4.0 with 3-frame-lag Kalman. Counter phase drives JointAngleFilter params, KneeValgusRule bottom-only gate (`knee_valgus.py:84`), bottom-frame buffering, in-rep flag for every rule. **Fix:** sign-aware test (`velocity < +threshold`) or velocity zero-crossing using Kalman velocity — land before or with any smoothing change.
- **S5 HIGH — missing keypoints become 0.0 and are filtered as data** (`analytical_ik.py:262`, 180–287): triggers — keypoint in < 2 views gets conf 0 (`triangulator.py:89-90`); dropout hold decays conf to 0 by 5th frame (`pipeline.py:541-549`). E7a: 2 dropped frames drag knee 78° → 24.8°, still 17° low 5 frames later; one side dips → large false asymmetry. **Fix:** NaN, skip filter update, rules skip NaN.
- **S6 MEDIUM — One Euro derivative source** (`filters.py:118,135`): uses raw previous sample; paper and Casiez's current reference use previous filtered value (copy saved in folder). 46 ms vs 29 ms lag with identical params, up to 2.8 cm on knee height. **Fix:** `dx = (x - self.x_filter.y) / dt`.
- **S7 MEDIUM — non-positive dt** (`filters.py:114-115`, `derivatives.py:133-135`, `pipeline.py:549`): dt ≤ 0 → 1e-6; triggers duplicate frames and `time.time()` hold stamps among `perf_counter`. One Euro barely affected; DerivativeTracker acceleration 3.6e6–5.3e6 °/s², velocity dips 30–40 % for 4 frames. **Fix:** skip update when dt ≤ 0, one monotonic clock, drop duplicates upstream.
- **S8 MEDIUM — position One Euro params** (0.8, 4.0, 1.0; `config.py:221-225`): beta 4 per m/s far too low for hip-centred joint speeds 0.65–1.8 m/s; derivative noise raises standing cutoff to 0.93–1.08 Hz anyway. Peak descent knee cutoff 1.2/2.2/3.3 Hz (x/y/z) = 131/72/48 ms delay; lag 46 ms; knee-cave peak −41 %; J_mean 12.87 vs no filter 10.03. **Fix if kept:** paper derivative with (1.2, 16, 2) → J_mean 4.32 / 7.09 / 13.79 at 2/4/8 px.
- **S9 MEDIUM — KeypointPositionSmoother ignores confidence** (`position_filter.py:69-77`): triangulator emits (0,0,0) at conf 0 and skips recentring when a hip is missing (`triangulator.py:134-136`) → whole skeleton jumps ~1 m. E7b: 2-frame ankle dropout → 64–76 cm error decaying 5+ frames; proposed filter (conf 0 = predict only) within 0.1 cm.
- **S10 MEDIUM — ConfidenceBlender as lagging smoother** (confidence agent owns): triangulated confidence median 0.48 at 4 px → weight ~0.47; lag 26/36/76 ms at 2/4/8 px (lags most when noise worst); knee-cave peak −4.6°. **Fix:** remove; use confidence/reprojection as Kalman R.
- **S11 MEDIUM — JointAngleFilter phase params** (`filters.py:197-202, 231`): "bottom" lowers cutoff exactly at reversal; "idle" (0.3 Hz, ~530 ms) still applies at start of descent because counter enters late; `d_cutoff` not configurable. vs fixed params: depth −3.3 vs −2.4° mean, −5.6 vs −4.2° worst; counter phases double standing jitter. `test_phase_aware_smoothing.py` only checks parameter storage.
- **S12 LOW-MEDIUM — "display-only" 2D smoother feeds single-cam valgus** (`pipeline.py:218, 522 → 639`): beta 0.5 in px/s → near pass-through; standing jitter 8.1 → 5.6 px/frame. Retuned (1.0, 0.02, 1.0) + paper derivative → 1.6 px/frame with lower moving error.
- **S13 LOW — bone shortening from per-axis smoothing (F17):** thigh length −1.15 cm max current One Euro, −0.27 cm low-lag filter → bone-direction smoothing unnecessary; lag is the problem.
- **S14 LOW — no reset between sets** (`pipeline.py:362-391`): JointAngleFilter and DerivativeTracker not reset → −21°/s velocity carries over after rest.
- **S15 LOW — cost/frame M2:** repo position smoother 108 µs; vectorised One Euro 7.7; proposed fixed-lag Kalman lag 2 32; Skeleton3D round trip alone 32; ConfidenceBlender 38.
- **S16 MEDIUM (adjacent) — SymmetryRule per-frame** (`symmetry.py:74-76`): instantaneous L/R comparison → noise fires it. False events/set on symmetric reps: 10.4 raw @4 px, 13.2 raw @8 px; best smoother still 6.0 @8 px. **Fix:** per-rep aggregate (e.g. median over bottom window).
- **S17 LOW (adjacent) — two depth-fault paths:** per-frame on predicted angles (skipped when counter merges reps) + rep-completion on filtered; clean run produced a duplicate "depth mild" event on one rep.

## End-to-end (4 px, 5 seeds, true phases; Sym FP = false asymmetry events per 6-rep set)
| Configuration | Knee RMS moving | Lag (ms) | Depth mean / worst | Valgus peak smooth | Valgus peak hard | Predicted depth | Sym FP |
|---|---|---|---|---|---|---|---|
| Raw | 2.35 | 0 | +1.99 / +0.67 | +4.53 | +0.78 | – | 10.4 |
| Production | 14.46 | 115 | −4.46 / −7.79 | −0.90 | −12.56 | +6.33 | 0.2 |
| Designed full chain | 20.12 | 160 | −4.96 / −8.34 | −1.38 | −14.87 | +5.00 | 0.0 |
| Prediction only | 2.35 | 0 | +1.99 | +4.53 | +0.78 | +15.08 | 13.6 |
| One Euro tuned | 2.16 | 9 | +0.81 / −0.14 | +1.93 | −1.43 | – | 5.0 |
| Kalman no lag | 1.89 | 0 | +1.62 / +0.68 | +3.45 | +1.55 | – | 6.4 |
| **Proposed lag 2** | **1.27** | +66 latency | **+0.50 / −0.39** | +1.71 | **−2.19** | – | 1.2 |
| Proposed + prediction | same | same | same | same | same | +12.99 | 8.6 |
| Proposed + JointAngleFilter | 10.08 | 79 | −3.86 / −6.43 | −0.83 | −10.48 | – | 0.0 |
The valgus rule never fires on the mid-ascent knee cave in any configuration (bottom-phase only) → short knee caves must be diagnosed from the per-rep trajectory.

## Grid search (J_mean at 2 / 4 / 8 px; J = knee RMS moving + mean |peak-depth err| + mean |peak-valgus err| + standing knee spread, averaged over smooth + hard sets)
| Filter | J_mean |
|---|---|
| Raw | 5.71 / 10.03 / 19.88 |
| Current position One Euro | 11.86 / 12.87 / 14.99 |
| Best One Euro (1.2, 16, 2, paper derivative) | 4.32 / 7.09 / 13.79 |
| Kalman no lag | 5.39 / 8.67 / ~15 |
| Kalman lag 1 | 3.68 / 6.02 / 10.46 |
| **Kalman lag 2, R from reprojection (q = 10, 0.005 m/px)** | **3.55 / 5.93 / 9.73** — one setting for all noise levels |
| Constant-acceleration Kalman lag 2 | 3.64 / 5.86 / 10.21 |
| Offline Kalman smoothing | 3.48 / 6.04 / 10.9 |
| Offline Butterworth 6 Hz | 3.40 / 5.99 / 10.86 (3 Hz over-smooths knee cave −7.3°) |

## Delay tolerance per consumer
| Consumer | Tolerable delay | Why |
|---|---|---|
| Between-rep diagnosis (bottom frame, trajectory) | up to rep end | runs after rep completes; lag 2 = offline quality |
| Rep phase / counting | 100–150 ms | drives between-rep logic; entry already 267 ms late today |
| Live cues | ≤ 150 ms | cached audio fires between reps; rule cooldowns 30–150 frames |
| Avatar display | ~0 | Kalman current estimate, free |

## Recommended architecture — smooth once, before IK, on arrays
`KeypointKalmanSmoother` = only temporal filter in the analysis path:
- Constant velocity per keypoint per axis; dt from one monotonic capture clock.
- Process noise q = 10 m²/s³.
- Measurement noise r = clip(0.005 m/px × median reprojection error over conf > 0 keypoints, 3 mm, 8 cm).
- Conf 0 or un-recentred frame → predict only; after ~5 predicted frames output NaN and reset that keypoint.
- Innovation > 6σ → treated as missing (takes over velocity clamp's job).
- dt ≤ 0 → no update, return last output.
- Outputs: 2-frame-lagged estimate for analysis, current estimate for display, velocity for rep counter.
Then IK → valgus estimator → rules; bottom frame, trajectory, rep signal from this one stream.

**Reference code** `fixedlag_fast.py`: `FixedLagCV` closed-form 2×2 on arrays, no linalg calls, matches full-matrix Kalman smoother to 3e-16 m, 11.7 / 22.7 / 32.5 µs/frame at lag 0/1/2 (~0.1–0.2 ms Orin Nano CPU). `ProposedSmoother` adds reprojection noise, conf-0 handling, outlier gate.

**Delete/replace:** ConfidenceBlender as smoother; KeypointPositionSmoother; JointAngleFilter + `update_phase` + `test_phase_aware_smoothing.py`; PredictiveStateEstimator + config; DerivativeTracker in squat path (use Kalman velocity); second bone-length pass; `PositionFilterConfig` → `KalmanSmootherConfig` (q, lag, reprojection gain, R floor/ceiling, gate, max predicted frames).
**Alongside:** S4 counter fix; NaN for missing (S5); per-rep symmetry (S16); one clock + duplicate skipping (S7).
**Tests replacing parameter tests** (synthetic squat → smoother + IK): depth error < 0.5° (lag-aligned); knee-cave peak loss < 25 %; standing jitter −70 %+; conf-0 keypoint moves estimate < 1 cm; duplicate timestamp leaves output unchanged; > 6σ outlier rejected.
**If 66 ms is ever unacceptable:** Kalman current estimate or tuned One Euro (both read peaks ~1.6–1.9° high at 4 px).
**Single-camera MediaPipe:** no reprojection → fixed or visibility-based R; loop ~12 fps → 1-frame lag (~83 ms) enough; MediaPipe internal smoothing SUSPECTED, not measured — re-fit on `user_test_runs/*/pipeline_inspect`, don't stack another low-pass; raw 2D to FPPA valgus.

## Verdicts
| Component | Verdict |
|---|---|
| `OneEuroFilter` | FIX (S6, S7); utility/display only (smoothing formula, first-sample, reset correct) |
| `LowPassFilter`, `ExponentialMovingAverage` | KEEP |
| `KeypointPositionSmoother` + `PositionFilterConfig` | REBUILD as fixed-lag Kalman |
| `Skeleton2DSmoother` + `DisplayFilterConfig` | FIX: px-unit params, vectorise, display only, stop feeding single-cam valgus |
| `JointAngleFilter` | REMOVE from analysis |
| `DerivativeTracker` | REMOVE for squat (fix only if tempo rules return) |
| `PredictiveStateEstimator` | REMOVE |
| `test_phase_aware_smoothing.py` | REPLACE with behavioural tests |
| F17 (smooth bone directions) | SUPERSEDED — lag is the issue |

## Files (this folder)
`PROGRESS.md`; `fixedlag_fast.py` (edge reference + `ProposedSmoother`); `harness.py` (pipeline replay + metrics); `synth.py` (synthetic squats + camera noise path); `exp3_output_sigma4_oracle.txt` (end-to-end table); `exp8_output.txt` (noise-free lag); `dbg_counter.py` (S4 repro); `exp1`–`exp8` scripts + `.txt` + JSON.
