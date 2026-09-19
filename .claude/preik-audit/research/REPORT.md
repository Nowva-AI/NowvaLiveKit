# Pre-IK Filtering — Literature Digest + Recommended Architecture

(Saved by lead from the agent's final message; subagent sandbox blocked writing REPORT.md.)
Folder: experiment scripts + results (`filter_lag_*`, `outlier_*`, `cost_microbench.*`), cached sources in `src/`.
Labels: [SRC] from paper/code · [EXP] reproduced in our synthetic experiment · [JUDG] engineering judgement, unvalidated on real captures.

## 0. TL;DR
1. **Every reviewed system smooths with zero-lag filters/smoothers**, not causal ones — Pose2Sim, OpenCap, Theia3D, anipose, Needham 2021, Stenum 2021 [SRC]. Our cues play between reps = *terminal* feedback, which beats concurrent feedback for learning a complex task (Sigrist 2013) [SRC]. Diagnosis need not be causal → **split outputs**: causal `live` path (HUD, phase detection) and `diag` path (100 ms fixed-lag smoother per frame + zero-phase filter over the rep buffer at rep end).
2. **Our One Euro position smoother erases faults:** `KeypointPositionSmoother` (0.8/4.0) keeps **72.6 % of a 3 cm valgus excursion**, adds **57 ms** lag, RMSE worse than raw (1.92 vs 1.00 cm) [EXP]. Pose2Sim's config: One Euro "tends to blunt RoM" [SRC].
3. **ConfidenceBlender = pure lag:** sigmoid confidence floor ~0.5 makes it a fixed EMA: 33 ms lag, RMSE 1.24 vs 1.00 [EXP]. No reviewed system blends temporally by confidence; confidence weights views in triangulation and rejects points [SRC].
4. **Outliers must be removed inside triangulation, where views disagree:** Pose2Sim drops cameras until error passes; EasyMocap drops outlier views/joints; OpenCap searches camera subsets; Lightning Pose 3D down-weights inconsistent views [SRC]. In our test no 3D-output-only stage removed multi-frame glitches (p95 per-sequence max error 10–19 cm) [EXP].
5. **Best per-frame diagnosis estimator:** constant-velocity Kalman per keypoint + outlier gate + **3-frame (100 ms) fixed-lag smoother**: RMSE 1.00 → 0.47 cm, keeps 97 % of the valgus peak, standing jitter 13.9 → 2.2 mm/frame; Kalman step 18 µs/frame on Mac [EXP].
6. **Rep peaks from zero-phase data:** peak-picking raw noisy data overestimates (3 cm valgus bump reads 137 % at 1 cm noise) [EXP]; 6 Hz zero-phase Butterworth over the rep buffer keeps 99.8 %, no lag, ~0.2 ms/rep [EXP].
7. **Biggest missing pieces:** aspect-preserving person crop for RTMPose; confidence-weighted triangulation with view rejection + L/R swap test; heel/small-toe keypoints + floor plane + per-foot contact (heel rise impossible today — triangulated skeleton has no heels); Kalman + fixed-lag + rep-window chain.
8. **Remove rigid bone snapping and GroundClamp's ankle equalizing + stance lock.** anipose estimates limb lengths and applies them softly; Pose2Sim measures segments only in non-crouched frames with trimmed means; Needham 2021 found pose-dependent systematic keypoint offsets (knees low, hips lateral); equalizing ankle heights erases heel rise/asymmetry by design.

## 1. Literature digest

**Pose2Sim** (Pagnon et al., [Part 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512754/), [Part 2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9002957/), [repo](https://github.com/perfanalytics/pose2sim)); default RTMPose halpe26.
- Triangulation: DLT rows × 2D confidence (common.py:631); views < 0.3 excluded; if reprojection error > threshold (**10 px** paper, **15 px** current config) try all subsets removing 1 then 2 cameras, keep lowest-error; min 3 cams (paper) / 2 (config); failure → NaN (not low-confidence). Gaps < 20 frames interpolated, longer hold last value.
- L/R swaps: `handle_LR_swap` swaps L/R in camera subsets, for few cameras/side views; current config marks it "not implemented yet".
- Filtering (3D): Hampel first (window 7, ~2σ). Default **4th-order zero-lag Butterworth 6 Hz** ("3–6 Hz walking/slow movements"). Kalman: constant acceleration per coordinate, trust_ratio 500, RTS smoothing "unless you need real-time". One Euro: 4 Hz, beta 1.5, forward-backward, "tends to blunt RoM". Acceleration-minimizing "blunts the peaks" (IK output only). Also GCV spline, LOESS 5, Gaussian σ1, median 3, Butterworth-on-speed 10 Hz ("can introduce an offset").
- Segment lengths: frames with hip/knee angle < 90° only, 50 % trimmed mean, L/R symmetric.
- Person crop: detector every 4 frames + box tracking.
- Accuracy: reprojection ~3.5 px (~1.6 cm); joint-angle error 3.0° / 4.1° / 4.0° walking/running/cycling; ankle least robust.

**OpenCap** (Uhlrich 2023, [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1011462), [core](https://github.com/stanfordnmbl/opencap-core)).
- 60 Hz; 20 keypoints incl. heels and toes. 2D: occluded foot/arm side removed when L/R confidence differs > 0.2; confidence < 0.4 → weight 0; 2D Butterworth 12 Hz for gait only (squats effectively unfiltered).
- Confidence-weighted DLT; RANSAC option commented "Not clear that this is helpful".
- LSTM marker augmenter 20 → 43 markers, hip-midpoint-relative, height-normalized, trained with 18 mm injected noise; 8.0 mm test error, ~32 mm vs mocap.
- **Squat accuracy: mean rotational error 4.1° (1.8–7.2)**; pelvis translation 12.3 mm.

**anipose** (Karashchuk 2021, [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8498918/)).
- 2D per view: median (13 frames, 25 px offset), Viterbi (n_back 5), autoencoder confidence recalibration.
- Triangulation optimizer: robust reprojection loss + 3rd-derivative smoothness (scale 2) + soft limb lengths **estimated inside the optimization** (scale 2, weak 1); reprojection threshold 5 px.
- Humans: spatiotemporal constraints helped most (position t = −18.7); median filter helped (t = −14.8) but "may mistakenly identify fast movements as an error"; **RANSAC did not help**. Offline only.

**Other systems.**
- EasyMocap: confidence-weighted batched DLT; drops a 2D detection far from the previous frame's reprojected 3D; drops a whole view when > 40 % of its joints are outliers; optional previous-3D prior row in DLT.
- Lightning Pose 3D ([PMC13131684](https://pmc.ncbi.nlm.nih.gov/articles/PMC13131684/)): measurement noise from ensemble variance; view with Mahalanobis > 5 from others gets variance × 10 (down-weight, not reject).
- 4D Association (CVPR 2020): 30 fps with 5 cameras.
- Needham 2021 ([Sci Rep](https://pmc.ncbi.nlm.nih.gov/articles/PMC8526586/)): RANSAC ray intersection + forward-backward Kalman; OpenPose mean error hip 29–36 mm, knee 30–41 mm, ankle 14–23 mm; **all models place hip centres more lateral and lower, knees ~20–30 mm low**.
- Theia3D: GCVSPL, default 20 Hz, 6–12 Hz recommended for slow movement.
- FreeMoCap [#849](https://github.com/freemocap/freemocap/issues/849): Butterworth designed for hardcoded 30 fps regardless of real rate.
- Captury: proprietary, not reviewed.

**Real-time filters.**
- MediaPipe Pose: image landmarks One Euro min_cutoff 0.05, beta 80; **world landmarks min_cutoff 0.1, beta 40**; crop box min_cutoff 0.01, beta 10 from previous landmarks. With beta 40 the cutoff reaches ~12 Hz at 0.3 m/s; ours only ~2 Hz.
- One Euro tuning ([Casiez](https://gery.casiez.net/1euro/)): beta 0, min_cutoff 1 Hz; lower min_cutoff for slow jitter; raise beta ×10 steps to remove lag.
- SmoothNet (ECCV 2022): 32-frame non-causal window (~1 s look-ahead at 30 Hz), 0.33 M params, 1.3k FPS CPU; authors rate One Euro inferior.
- De Groote 2008: Kalman *smoothing* cut joint-position error > 50 % vs Kalman filtering and standard IK.
- Constrained Kalman (Simon 2010): nonlinear constraints (bone length) no longer optimal; methods disagree.

**Where to filter.**
| Job | Best stage | Evidence |
|---|---|---|
| Outliers, wrong limb, occlusion | 2D, inside triangulation | Pose2Sim/EasyMocap/OpenCap exclude views [SRC]; 3D-only stages left 10–19 cm glitches [EXP] |
| L/R swaps | 2D per view | Pose2Sim swap test; Stenum 2021 ~5 % frames swapped in side views |
| Confidence recalibration | 2D | anipose autoencoder; rtmlib RTMPose score is raw max response, no sigmoid |
| Noise smoothing | 3D after robust triangulation | Pose2Sim, Theia3D, Needham, anipose; OpenCap 2D only for gait |
| Skeleton priors | 3D, soft | anipose |

**Cutoffs, lag, feedback timing.** Zero-lag cutoffs: 5 Hz (Stenum), 6 Hz (Pose2Sim), 6–12 Hz (Theia3D slow). Winter residual analysis picks cutoffs below optimum (Yu 1999) — validate by peak preservation. Causal 2nd-order Butterworth delay ≈ 0.225/fc s (33 ms at 6 Hz, 54 ms at 4 Hz) [EXP]. Terminal > concurrent feedback (Sigrist 2013) → post-rep verbal cues have a budget of seconds; only HUD needs < ~100 ms.

**Feet/ground.** Sports2D/Pose2Sim estimate floor from near-still toe (or ankle) positions and correct angles for floor tilt. OpenCap and Pose2Sim require heel + toe keypoints; OpenCap handles occlusion per side. No reviewed system forces both ankles to the same height.

## 2. Experiments (synthetic 30 Hz squat, 30 seeds; 45 cm hip descent + 3 cm 0.6 s valgus bump near bottom; 1 cm noise, normal tempo)
| Filter | Lag (ms) | Valgus peak kept | RMSE (cm) | Peak read from noisy data | Standing jitter (mm/frame) |
|---|---|---|---|---|---|
| raw | 0 | 100 % | 1.00 | 137 % | 13.9 |
| ConfidenceBlender @ conf 0.5 | 33 | 95 % | 1.24 | 109 % | 5.7 |
| **Current One Euro 0.8/4.0** | **57** | **72.6 %** | **1.92** | 80 % | 2.0 |
| One Euro MediaPipe world 0.1/40 | 10 | 92 % | 0.78 | 113 % | 4.6 |
| Causal Butterworth 2nd-order 6 Hz | 33 | 100 % | 1.25 | 115 % | 5.0 |
| Causal CV Kalman | 0 | 105 % (overshoot) | 0.77 | 126 % | 7.7 |
| **Kalman 3-frame fixed lag** | 0 (100 ms latency) | 97.4 % | **0.47** | 102 % | 2.2 |
| **Rep-window zero-phase 6 Hz** | 0 | **99.8 %** | 0.58 | 111 % | 3.7 |

- 2 cm noise: fixed-lag Kalman RMSE 0.80 vs 1.99 raw, 93 % peak kept.
- Fast tempo: constant-acceleration model overshoots depth 0.85 cm; constant-velocity 0.12 cm → use CV.

**Outliers** (1 cm noise, 3 % spikes 5–20 cm, two 4-frame 10 cm glitches). RMSE / p95 per-sequence max error, cm:
| Pipeline | Hip | Knee |
|---|---|---|
| raw | 2.67 / 22.6 | 2.72 / 23.5 |
| current VelocityClamp only | 1.98 / 18.7 | 1.97 / 17.3 |
| fixed-lag Kalman | 1.44 / 13.3 | 1.46 / 12.1 |
| trailing Hampel + fixed-lag | 1.25 / 14.8 | 0.81 / 10.5 |
| outlier gate (3σ & 4 cm) + fixed-lag | 1.10 / 19.5 | 1.07 / 13.6 |
Trailing median lags the moving hip; no 3D-only stage fixes multi-frame glitches.

**Compute (Mac numpy; Orin Nano ~3–5× slower [JUDG]):** weighted DLT all 4 subsets + reprojection 500 µs; Kalman + gate 19 kpts 18 µs; rep-window zero-phase 212 µs/rep; floor-plane fit 19 µs. Whole chain < 1 ms/frame.

## 3. Recommended pre-IK architecture (3-camera)
Principle: reject bad data in triangulation, estimate in 3D world coordinates, constrain softly, measure peaks from zero-phase data. Outputs `live` (causal) and `diag` (100 ms fixed lag + rep-window refinement).

- **Stage 0 (prereqs):** one clock, drop duplicate frames, real frame index.
- **Stage 1 — 2D per view:** 1a person crop from previous keypoints' bbox, pad 1.25, aspect-preserving (rtmlib), crop box One Euro 0.01/10, re-detect on full frame when < 8/26 keypoints pass or every 4 frames (likely largest 2D gain [JUDG]); 1b map halpe26 heels (24/25) + small toes (22/23); 1c raw RTMPose score, per-keypoint threshold, no sigmoid, never a blend weight; 1d no 2D low-pass.
- **Stage 2 — triangulation:** 2a score-weighted, Hartley-normalized DLT; 2b solve 3 views, if any view reprojection > T solve the 3 pairs, keep best passing pair, else keypoint **missing** (conf 0) — T = 10 px, loosened to 15 px given guessed intrinsics; 2c drop a view for the frame if > 40 % of its joints exceed T; 2d L/R swap test when a view's left AND right leg joints both fail; 2e per-keypoint variance σ² ≈ (σ_px·depth/focal)²/n_views, σ_px = max(reproj RMS, 2 px), floor (1 cm)², borderline views variance × 10; 2f no hip re-centering here — world coordinates through Stage 5, re-center only at IK boundary [JUDG].
- **Stage 3 — 3D estimation (replaces ConfidenceBlender, VelocityClamp, One Euro):** 3a CV Kalman per keypoint axis, real dt, q = 30 (m/s²)², R from 2e; 3b gate: measurement missing if > ~4σ AND > 4 cm from prediction, predict through ≤ 4 rejections then re-init on measurement; 3c `live` = filter, `diag` = 3-frame fixed-lag smoother (6 frames no extra benefit); 3d after > 4 missing frames confidence 0, no fabricated pose.
- **Stage 4 — soft skeleton consistency (diag only, optional):** 4a lengths from frames with hip & knee flexion < 30°, 50 % trimmed mean, L/R averaged, slow update; 4b correct only beyond ±8 %, pull to band edge, move both endpoints by variance [JUDG]. Never constrain hip width, foot height, stance width.
- **Stage 5 — ground & feet (detector, not clamp):** 5a floor plane from heels/toes in standing frames with foot speed < 5 cm/s; normal = gravity vertical (also corrects T-pose tilt); 5b per-foot contact when heel/toe speed < 10 cm/s and height < standing + 1.5 cm; heel rise = heel > 2 cm above standing height for ≥ 5 frames with that foot's toe down [JUDG]; 5c optional zero-velocity pseudo-measurement in contact (σ_v 5 cm/s).
- **Stage 6 — rep-window refinement (diag):** 6a buffer gated measurements rep start − 0.5 s to rep end + 0.5 s, 4th-order zero-phase Butterworth 6 Hz (try 5), pad ≥ 12 frames; 6b rep metrics (depth, peak valgus, heel rise, hip shift, forward lean, asymmetry) from 6a after IK; per-frame live faults advisory.

**Remove/replace:** ConfidenceBlender → remove; VelocityClamp → 3b; One Euro position smoother → 3a/3c; rigid bone snapping → 4a/4b; GroundClamp → remove equalizing/stance lock/floor clamp, rebuild as 5a/5b. Post-IK (linked): JointAngleFilter + 0.2 s predictive extrapolation adds lag then overshoots; causal CV already overshoots valgus peak 5–13 % at reversals [EXP].

**Single-camera MediaPipe:** no Stage 2; MediaPipe already One Euro-filters world landmarks (0.1/40), another One Euro only stacks lag; use Stages 3, 5, 6 with R from visibility; folded legs need a pose-validity gate, not smoothing.

## 4. Missing pieces ranked by expected gain for squat diagnosis
1. Aspect-preserving person crop (1a) — everything, esp. knee/ankle horizontal → valgus, stance, hip shift.
2. Weighted triangulation + view rejection + swap test (2a–2d) — valgus, hip shift, asymmetry, pelvic list (single-view errors become lateral 3D offsets = frontal-plane faults; only this stage can fix).
3. Heel/small-toe keypoints + floor plane + per-foot contact (1b, 5a, 5b) — heel rise has no input today; lean and depth need a true vertical.
4. 3D Kalman + outlier gate + reprojection-based noise (2e, 3a, 3b) — ~23 % lower RMSE causally.
5. Fixed-lag smoother + rep-window zero-phase (3c, 6a) — RMSE halved, peaks 97–99.8 %, removes peak-picking overestimate.
6. Gravity alignment from floor normal (5a) — lean, pelvic list, vertical depth.
7. Soft L/R-symmetric bone lengths from straight-leg frames (4a, 4b) — second order; rigid version harmful.
8. Image-space temporal gate (EasyMocap reprojected-previous-3D check) — catches two-of-three-views-wrong glitches.
9. Learned refiners (SmoothNet-style, OpenCap LSTM augmenter) — need our own data; later.

## 5. Reported pitfalls matching our filters
1. One Euro blunts RoM (our beta keeps cutoff ~2 Hz mid-squat; MediaPipe uses beta 40–80).
2. Stacked causal smoothing lags, extrapolation overshoots (F17 ±9°; 5–13 % overshoot at reversals).
3. Confidence isn't a calibrated probability (anipose autoencoder, Lightning Pose ensemble variance); sigmoid floor makes ConfidenceBlender a fixed EMA.
4. Median/velocity filters remove real fast movement (anipose); VelocityClamp stores clamped state → real change or swap becomes slow drift [JUDG].
5. RANSAC not worth it with 3 cameras; exhaustive leave-one-out is 4 deterministic solves.
6. Keeping failed triangulations at low confidence (we keep > 15 px at conf × 0.1; Pose2Sim/EasyMocap NaN/zero + interpolate).
7. Rigid bone lengths don't match keypoints (knees 20–30 mm low, hips lateral; Pose2Sim excludes crouched frames); proximal-to-distal snapping pushes knee error into ankle/foot.
8. Symmetric-feet assumption removes heel rise, pelvic list, weight shift.
9. Calibrating on a bad pose (use straight-leg frames + trimmed means; our 30-frame medians were poisoned by folded legs).
10. Filters assuming nominal frame rate (FreeMoCap #849); our duplicates + mixed clocks break dt-based filters (One Euro dt = 1e-6).
11. Residual analysis picks cutoffs too low (Yu 1999) — validate by peak preservation.
12. Filtering after hip re-centering spreads hip noise into every joint [JUDG].

## 6. Suggested validation
1. Record 3-camera sessions with deliberate faults (valgus, heel rise, lateral hip shift, forward lean), logging raw per-view 2D + calibration (depends on F21 replay harness).
2. Offline "gold": all-subset triangulation over the full sequence + 6 Hz zero-phase (Pose2Sim recipe).
3. Score causal/fixed-lag variants against gold on per-rep peaks; accept when peak error < 10 % of gold excursion and ≥ 95 % of deliberate faults still exceed thresholds.
4. Tune q, gate, cutoff only against this harness.
