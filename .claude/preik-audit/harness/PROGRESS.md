# Pre-IK evaluation harness — PROGRESS

Folder: /Users/naiahoard/NowvaLiveKit/.claude/preik-audit/harness/  (persistent; repo source untouched)
Python: /Users/naiahoard/NowvaLiveKit/venv/bin/python

## Milestone 0 — code reading (DONE, run 2)
Read: preik_chain, confidence_blend, velocity_clamp, bone_constraints, ground_clamp, position_filter, filters
(JointAngleFilter/OneEuro), standing_gate, triangulator, calibration (TPoseCalibrator), multi_capture
(get_synced_frames), multi_camera, rtmpose (SimCC decode), pipeline.process_frame, pipeline_process loop pacing,
analytical_ik, valgus (TriangulatedValgusEstimator), squat profile rep signal + SignalRepCounter, config.
Colleague data reused: confidence_blend/e0_summary.json + *_rtm.npy (real RTMPose on 3 real 720p videos).

## Plan / layout
- noise_calib/  : run_rtmpose.py (real model on data/squats.mov) + measure_noise.py -> noise_params.json
- preik_harness/: body.py (GT motion), cameras.py (rig + calibration), detector.py (2D noise), delivery.py
  (camera clocks + loop pacing + get_synced_frames), runner.py (process_frame replica), metrics.py, chains.py,
  api.py (evaluate)
- run_baseline.py -> baseline_results.json, baseline_summary.md ; REPORT.md

## Status
- [x] noise calibration
- [x] harness modules
- [x] baseline run (47 s, 6 workers; baseline_results.json + baseline_summary.md)
- [x] final report delivered as the agent's final message (REPORT.md file writing was blocked by the tool policy)

## Milestone 1 — noise calibration (DONE)
- noise_calib/run_rtmpose.py ran real RTMPose-m halpe26 (production decode) on data/squats.mov (361 frames, 720p canvas)
  -> squats_mov_rtm.npz; measure_noise.py pools it with colleague's 3 real 720p recordings -> noise_params.json.
- Measured: still-standing white jitter (fitted through SimCC quantizer) hip 1.9/1.8 px, knee 1.75/1.5, ankle 0.9/0.85,
  toe 2.0/1.0 (canvas clip; 720p 15 fps clip ~2-3.5 px); slow in-view component 1-2 px; conf p50 0.62-0.66, still
  sd 0.015-0.04, lag-1 ac 0.58; outliers >30 px: canvas 0%, 720p ankles 0.4-11%, toes 1-18%, knees 0-0.8%,
  run length p50 1 / mean 1.6 frames, conf during outlier 0.589 vs 0.633; frontal L/R swap frames 0.26%.

## Milestone 2 — harness modules (DONE, run 2)
preik_harness/{body,cameras,detector,delivery,runner,chains,metrics,api}.py + run_baseline.py.
Verified: GT bone lengths exactly rigid (0.000 mm range), knee flexion hits targets, valgus scenario GS valgus
+13-14 deg at bottom vs -2..-4 clean; noise 'none' + perfect calib -> MPJPE 1.0 mm (pipeline plumbing correct).
Key discoveries while building (see REPORT): planar T-pose PnP forces world Z = subject BACK; get_synced_frames fails
whenever a secondary's newer frame has not arrived (5-30% ticks); triangulator conf*0.1 (reproj>=~12.6 px ->
conf<0.1) makes IK return knee flexion 0.0 -> ~25% of frames have a zero knee angle in realistic profile.
Trial 1-seed baseline runs in ~10 s with 6 workers.

## How to resume
cd harness; /Users/naiahoard/NowvaLiveKit/venv/bin/python run_baseline.py  (writes baseline_results.json/summary.md)
Then write REPORT.md from baseline_summary.md.

## Milestone 3 — full baseline (DONE)
run_baseline.py: 8 chains x 8 scenarios x 3 seeds x {perfect, tpose} + noise-none floor + mild/harsh sensitivity +
DiagnosticConfidenceFloor + 24 fps cameras. Metrics: bottom-window ('tb', +-150 ms of true bottom, median across reps)
vs pipeline-selected bottom frame ('sel'). Headline: raw 33.8/43.9 mm MPJPE (perfect/tpose); current adds +50/+72 ms
lag and -2.9/-7.1 deg knee at true bottom; full_original +108/+134 ms, -6/-12 deg; IK zero-knee frames 25-28 % for all
chains (conf floor diagnostic: knee MAE moving 14.3 -> 4.3 deg). Verified API example (custom class chain) runs.

## Milestone 4 — new stack vs old chains on the NEW triangulator (DONE 2026-09-17)
- runner.py now takes WORLD-frame skeletons from the rewritten DLTTriangulator; old chains get `recentre_at_hips` +
  old tail (IK NaN->0.0 + legacy JointAngleFilter); chains with `input_frame="world"`, `handles_missing`,
  `lagged_output`, `tail="new"` (see runner docstring) get the new tail. legacy/ holds byte-copies of the removed
  filter modules; chains.py carries their old config defaults (wave 2 removed the config sections).
- proposed.py: FixedLagKeypointSmoother(lag 2) [-> FootContactModel] -> recentre_at_hips, outputs aligned by tick.
- run_proposed.py -> proposed_results.json + proposed_summary.md (tables + proposed_findings.md). 23 s, 6 workers.
- Diagnostics: trace_foot.py, repro_kalman_reacquire.py (B1), repro_foot_step.py (B2).
