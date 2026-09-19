# triangulation_frontend audit — PROGRESS

Working dir: /Users/naiahoard/NowvaLiveKit/.claude/preik-audit/triangulation_frontend/
Python: /Users/naiahoard/NowvaLiveKit/venv/bin/python

## Milestone 0 — code read + reference sources (DONE)
Read: rtmpose.py, multi_camera.py, triangulator.py, calibration.py, multi_capture.py, FINDINGS F13/F18.
Downloaded reference decode code (mmpose simcc_label.py, post_processing.py, refinement.py, rtcc head; rtmlib).

Verified facts so far (code review, not yet experiments):
- F13 model path bug: model now present at src/biomechanics/pose/models/rtmpose-m-halpe26-256x192.onnx (manually placed).
- F13 no crop: STILL OPEN (rtmpose.py:181 cv2.resize full frame -> 192x256).
- F13 argmax decode: STILL OPEN (rtmpose.py:232-233).
- F13 sigmoid confidence: STILL OPEN (rtmpose.py:239). mmpose get_simcc_maximum uses raw min(max_x, max_y),
  NO sigmoid (mmpose_post.py:90-97); rtmlib main uses raw mean. mmpose optional use_dark = gaussian blur + log
  + 2nd-order Taylor (refine_simcc_dark). decode_visibility uses softmax(beta=150*sigma).
- F18 frame_index=0: STILL OPEN (multi_camera.py:201). Batched/vectorized DLT: DONE (triangulator.py:92-121).
  Unweighted/unnormalized DLT: OPEN. No view selection: OPEN.
- Heels/small toes dropped: rtmpose.py:268-276 only maps big toes (20,21)->17,18. triangulator NUM_KEYPOINTS=19.
- ONNX IO: input [B,3,256,192]; simcc_x [B,26,384], simcc_y [B,26,512].
- Video data/squats.mov: portrait 1080x1920 after cv2 auto-rotate, 361 frames ~29 fps, frontal view, continuous squats.
  Exp1 embeds it into a 1280x720 landscape canvas (405x720 + edge replicate) to mimic production squash.

## Next steps
1. Run exp1_run_detector.py (saves exp1_logits.npz), then exp1_analyze.py (decoders, jitter, confidences,
   sub-pixel shift linearity test, crop vs full-frame).
2. exp2 synthetic triangulation robustness (3 cams, outliers, L/R swap, weighted/normalized/LOO).
3. exp3 confidence vs 3D error.
4. consumer mapping for hip-centering and heels.
5. exp6 calibration focal/proportion bias.
6. exp7 sync/duplicates.
7. REPORT.md

## Milestone 1 — Exp1 2D detector quality (DONE) — scripts exp1_*.py, results exp1_results.json, exp1_canvas_check.txt,
## overlay_ff_red_crop_green.jpg (red = production full-frame squash, green = tracked person crop)
Setup: squats.mov (portrait, frontal) embedded in 1280x720 canvas (person ~470 px tall; crop bbox h median 590 px).
- Full-frame squash is GROSSLY wrong on 720p landscape: FF vs crop body keypoints median 11 px, p95 61 px (canvas px,
  person 470 px tall => ~2% / ~13% body height). Overlay: face keypoints on chest, shoulders ~50 px low at standing.
  Same with gray padding (median 9.0 vs 9.4) => caused by squash+small person, not canvas. Native portrait squash (1.33x)
  2.3 px; undistorted letterbox 1.5 px.
- Hip width FF 86.7 px vs crop 63.5 px (+37%); knee separation FF 151.6 vs crop 168.9 (-10%) => valgus ratio biased.
- Flip-consistency (TTA disagreement, accuracy proxy): FF body median 7.9 px vs crop 1.15 px.
- Sub-pixel shift test (true image shift 0..8 px): FF slope x 0.56 / y 0.11 (standing frame) — predictions do NOT follow
  motion; crop slope 0.96-1.05. Staircase RMS crop argmax 0.54 px -> parabola/DARK 0.36 px.
- Motion HF noise (SG residual) body: FF argmax 1.91 px, crop argmax 0.44, crop parabola 0.35, crop DARK 0.35.
  FF argmax has 38% of frame-to-frame joint displacements exactly 0 (staircase). Still jitter body FF 1.70 -> crop dark 0.91.
- Decoders parabola ~= local soft-argmax(beta10,+-8) ~= mmpose DARK; parabola cost 0.08 ms/3 views; DARK python 1.7 ms.
- Confidence: raw min(max_x,max_y) FF 0.37-0.96 (median .72); sigmoid squashes to 0.59-0.72. Crop raw median 0.97.
  NO-PERSON images (black/noise/background): raw max 0.16-0.27, sigmoid 0.53-0.57 => ALL 26 kpts pass 0.3 with sigmoid,
  0 pass with raw. _kpts_to_skeleton2d never returns None => hallucinated skeletons get triangulated. CRITICAL.
- Occlusion (70 px gray box on knee): crop raw conf drop 0.99->0.95 / 0.98->0.79 / 1.0->0.87 / 0.98->0.93, knee moves
  6-15 px (plausible hallucination); FF conf barely changes. Confidence is only weakly occlusion-informative.
- Cost: crop warpAffine 0.31 ms/view vs resize 0.23 ms.

## Milestone 2 — Exp2 triangulation robustness (DONE) — synth.py, exp2_triangulation_robustness.py, exp2_results.{json,txt}
Synthetic 1.885 m squat (21 kpts), 3 cams yaw -40/0/+40 (also -70/0/70), r=3.5 m, f=0.8w, person ~540 px.
my synth.dlt == DLTTriangulator (max diff 0.0). Rig -40/0/40, sigma 1.5 px unless noted:
- clean 1.5 px: 6.6 mm/kpt, knee flex err 1.1 deg; clean 5 px (FF-like): 22 mm, 3.9 deg (p95 9.4).
- single-view knee outlier 40/80/150 px: prod 71/145/293 mm, knee flex 9/17/33 deg; prod conf -> 0.09-0.11 (kept!)
  best-subset(tau 10 px): 19/15/24 mm, 3.2/2.3/2.6 deg. heavy tail 1.7-4% frames >50 mm; temporal tie-break
  (closest to previous estimate among pairs with rms<=4 px): 9 mm, 0% >50 mm.
- conf-weighted DLT: no help unless outlier conf drops (0.5 -> 144->80 mm). Row-normalized/Hartley: <=6% gain;
  depth-IRLS: 0 gain (cameras at similar depth). => normalization is NOT a priority for this rig.
- one-view full L/R swap: prod 240 mm ALL kpts, knee-separation err 187 mm (valgus destroyed); best-subset alone WORSE;
  swap-check (7 hypotheses: none, full swap per view, legs-only swap per view) + best-subset: 6.7 mm. -70/0/70 rig: 82 mm -> 6 mm.
- cost (Mac, numpy, 21 kpts x 3 views): DLT 0.11 ms, best-subset 0.52 ms, swap-check+best-subset 1.41 ms (unbatched hyps).
- IK interaction: analytical_ik._get_point drops conf < 0.1 and _compute_knee_flexion returns 0.0 -> prod conf*0.1 (<=0.1)
  turns an outlier knee into knee_flexion = 0 deg (with sigmoid view conf 0.6-0.7 => 0.06-0.07).

## Milestone 3 — Exp3 confidence semantics (DONE) — exp3_confidence_semantics.py, exp3b_uncertainty_floor.py, exp3*_results.json
Heteroscedastic 0.5-8 px noise + 10% outliers 10-150 px:
- prod conf IS rank-monotone with prod error (Spearman 0.75) but only because it flags gross outliers: 23% of points
  get conf<0.1 with median error 167 mm and stay in the skeleton; no metric meaning; 15 px threshold not scale-aware.
- best-subset cuts mean error 59.5 -> 35.4 mm, p95 258 -> 84 mm.
- proposed u = max(sigma_hat_px, 3 px floor [x2 for 2-view]) * sqrt(trace((J^T J)^-1)): Spearman 0.47 on best-subset
  error; calibration err/u 3-view median 0.66 p95 1.69; 2-view median 0.58 p95 5.2 (2-view residual only sees epipolar
  component -> flag 2-view points). confidence = 1/(1+(u/0.02 m)^2) keeps [0,1] and is invertible for Kalman R.

## Milestone 4 — Exp4 re-centering (DONE) — exp4_recentering.py, exp4_results.json (real DLTTriangulator)
- CONFIRMED untriangulated kpt -> (0,0,0) conf 0 == hip-midpoint location in the centered frame.
- CONFIRMED if either hip untriangulated, centering is skipped -> whole skeleton jumps by 0.23 m (bottom frame).
- centering raises per-kpt noise 4.0 -> 5.0 mm (+24%) and makes L/R ankle noise correlated (r 0.00 -> 0.37).
- 80 px single-view hip outlier moves the feet 126 mm after centering. Static feet move 0.85 m/s in hip frame; 0.50 m
  hip travel is removed.
Consumers checked (all relative, none needs hip-centering): squat.get_rep_signal (hip-ankle y), BiLSTM
LandmarkFeatureExtractor (angles, bone lengths, y-diffs rel. hip; idx<=16), _build_trajectory_sample hip_y/knee_y (used
as L-R and hip-knee differences in diagnosis engine/rep_scoring/evidence_tests), bridge._ground_and_center (regrounds per
frame), demo_ws_bridge (grounds per frame). IK = relative vectors.
Heels: standing_gate._heel_rise_ratios returns [] when len<=20 -> flat-foot check silently OFF in 3-cam mode;
ground_clamp/bone_constraints guard by n_kpts; keypoint_corrector/pose_validation accept 19 or 21; demo_ws_bridge slices
[:19]; calibrated_test_visualizer slices [:19]; capture_audit zeros((19,3)) 2D placeholder only. BiLSTM unaffected.

## Milestone 5 — Exp5/6/7 (DONE)
Exp5 (exp5_ff_error_in_3d.py): measured FF-minus-crop 2D errors (rescaled) pushed through 3-view DLT:
  crop-like iid 1.5 px: body kpts 6.7 mm, knee flex 1.2 deg (p95 3.0), hip width +0.5 mm.
  FF error same frame all views: body 64.5 mm, knee flex 6.0 deg (p95 12.3), hip width +118 mm, 0% flagged by reproj.
  FF independent per view: body 59.6 mm, knee flex 5.5 deg (p95 14.0), hip width +119 mm, 39% body kpts get conf x0.1.
  => systematic (view-consistent) 2D error is invisible to reprojection confidence.
Exp6 (exp6_calibration_bias.py, exp6_results.txt): T-pose PnP with K=0.8w vs truth f in {0.55,0.65,0.8,1.0}w, ratios +-5%,
  face depth offsets, imperfect T-pose. Focal error absorbed as camera distance (f .55 -> est dist 5.06 m vs 3.5).
  kpt err (hip-centred) 17-29 mm systematic; knee flex mean|err| 0.9-2.2 deg (p95 1.9-5.3); trunk lean bias -1.1..-1.8 deg;
  dorsiflexion bias -0.3..-2.2 deg; bone lengths mean|err| 1-4% (p95 up to 9%, tibia +3.9% bias with imperfect T-pose);
  knee-sep/hip-width ratio err ~0.02-0.05. Induced reprojection 1.4-3.3 px. Second-tier vs FF squash/outliers/swaps;
  comparable to noise but NOT filterable.
  FRAME: recovered front camera centre z = -3.49 (truth -3.5), toe z < heel z => triangulated frame is X-left, Y-down,
  Z-BACKWARD (away from front camera), right-handed (solvePnP cannot produce the documented left-handed Z-forward frame).
  Docstrings calibration.py:12, analytical_ik.py:46, bridge.py:24 say Z toward camera (wrong; matches real MediaPipe though).
Exp7 (exp7_sync.py, exp7b_sync_policy.py): production get_synced_frames logic simulated (30/29.97/30.03 fps, 25+-3 ms
  USB latency, sleep-paced 30 fps loop): 44.5% of calls return None (range 22-69% by camera phase), 45.5% of primary
  frames never processed; Jetson 45 ms: 36% fail, 53% skipped. Proposed (ref = newest primary frame all cams have caught up
  to, nearest match, 20 ms tol, keep >=2 views, dedupe by last ref): 0% fail, 11% skip (sleep pacing; event-driven wait
  would recover), p95 offset 16.6 ms.
  Residual offset in one view: 15 ms -> max kpt err 7.4 mm at 0.85 m/s hip speed, knee flex <=1.0 deg (minor).
  Clock mixing (real DerivativeTracker): true 60 deg/s; after 2 dropout-hold frames (time.time) back to perf_counter
  timestamps -> 600,033 deg/s, decays 420k/294k/206k (EMA 0.3) -> predictive estimator saturates for ~0.8 s.
  ipc_bridge.send_fault cooldown: now - last_send = -1.79e9 s after a fault on a held frame -> that fault type is muted
  for the rest of the process. NOTE: today dropout hold never triggers in 3-cam mode because sigmoid conf >= 0.5 makes
  every view "detected"; fixing the confidence (raw score) will EXPOSE these clock bugs.
  frame_index=0 -> FaultEvent.frame_index (fault_types.py:180) and BiLSTM RepData start/end_frame all 0.
Other: scripts/tools/download_models.py:49 MODEL_DIR still parent.parent -> scripts/src/... (F13 item STILL OPEN).

## Remaining: write REPORT.md (verdict/prioritized list).

## Milestone 6 — DONE. REPORT.md write was blocked by the harness (subagents must return reports as text); the full report is in the agent final message.
