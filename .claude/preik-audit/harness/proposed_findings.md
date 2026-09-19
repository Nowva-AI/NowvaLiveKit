## Findings (verification engineer, 2026-09-17; appended to the generated tables above)

Files: `run_proposed.py` (driver), `preik_harness/proposed.py` (chain factory over the real components),
`preik_harness/runner.py` (world-frame input, `recentre_at_hips`, lagged-index alignment, old/new tails),
`preik_harness/legacy/` (byte-copies of the removed filter modules + their old config defaults in `chains.py`),
`trace_foot.py`, `repro_kalman_reacquire.py`, `repro_foot_step.py`.

### 1. Sanity checks against the audit's pipeline-level claims (perfect / tpose)

| claim | old stack (baseline) | old chains on NEW triangulator | proposed (Kalman + foot + NaN IK, no angle filter) |
|---|---|---|---|
| zeroed/NaN knee angle frames | 25 % / 28 % (raw; every chain) | 4.0 % / 7.1 % — all from accepted keypoints with conf < 0.1 (u > 6 cm, two-view fallbacks); none from the dropout hold | 0.1 % / 0.2 % |
| depth bias, IK / final | raw +0.9 / −9.5°, −5.1 / −15.2°; current −1.8 / −11.2°, −11.4 / −19.8° | raw +2.2 / −3.6°, +3.1 / −4.2°; current −0.1 / −5.0°, −7.2 / −12.9° | +0.8 / +0.8°, +1.6 / +1.6° (final == IK); worst per-rep undershoot −1.4 / −2.2°; mean abs 1.8 / 3.0° |
| stance widening preserved | 0.03 / 0.08× (full chain, GroundClamp) | raw 1.00 / 0.96×; full_original −0.00 / 0.02× | 0.97 / 0.90× (Kalman-only 0.97 / 0.94×) |
| pelvic list preserved @true bottom | 0.43 / 0.36× (bone_only), 0.68 / 0.47× (full) | raw 1.06 / 0.97×; full_original 0.77 / 0.83× | 0.98 / 0.90× (1.18 / 1.04× at the pipeline bottom frame) |

The residual +0.8 / +1.6° depth is the max-of-noise bias of taking a per-rep maximum over unfiltered IK samples
(see error source 3a below). The old `current` chain gets WORSE on the new triangulator in tpose (knee lag 92 ms,
knee at true bottom −13.4°): the ConfidenceBlender weights are now metric confidences (p50 0.38–0.40), so do not run
the new triangulator with the old blender even as an interim step.

### 2. Front-end gain alone (`raw` old triangulator -> `raw` new triangulator, identical inputs otherwise)
MPJPE 33.8 -> 25.0 mm (perfect), 43.9 -> 35.9 (tpose); p95 98.7 -> 48.7, 114 -> 73; knee MAE moving 14.3 -> 4.7°,
16.6 -> 6.5°; lower-body conf < 0.1 fraction 8–9 % -> 1.1–1.9 %; readiness gate opens at 0.36 s / 0.59 s instead of
0.58 / 1.89 s (tpose walkout seeds that never opened now open). Harsh noise: 75.0 -> 43.6 mm, p95 325 -> 81, knee MAE
moving 37 -> 11°, depth final −33.6 -> −11.0°, zeroed knees 61 % -> 13 %. Detector-noise model unchanged, so the WS2
crop fix (measured 1.70 -> 0.89 px still jitter) is NOT in these numbers.

### 3. Top 3 remaining error sources in the proposed stack
1. **T-pose calibration error** (tpose vs perfect): MPJPE 20.1 -> 31.2 mm, stance MAE 6.7 -> 20.4 mm, valgus at true
   bottom +0.0 -> −2.1°, walkout reprojection 12 px / lower-body conf p50 0.27. The harness no-noise floor in tpose is
   17.9 mm, so calibration is now the largest single term and nothing pre-IK can remove it.
2. **Slow per-view detector bias** (the assumed 2.5–4 px AR(1) component): still-standing MPJPE is 19.9 mm while still
   jitter is 3.1 mm, i.e. ~17 mm of low-frequency bias survives the Kalman (knees 22, ankles 21, toes 23–25, hips 14 mm
   per keypoint). Moving error (20.2 mm) ≈ still error, so lag is no longer a factor. Needs real multi-view recordings
   to confirm the magnitude (the crop fix may shrink it).
3. **Transient excursions that survive the chain** — 2.4 % / 4.6 % of frames still have a lower-body keypoint > 10 cm
   off (raw 6.5 / 12.2 %, current 10.5 / 30.0 %); knee error > 10° on 1.0 / 2.2 % of frames (p99 9.6 / 12.3°). Two
   mechanisms: (a) the Kalman re-acquire snap on 2-frame consistent bursts (bug B1) — e.g. stance_change seed 1 rep 3:
   both side cameras swapped legs, the triangulated knees/ankles sat 60 cm off at conf 0.2 for 2 frames, the gate
   rejected both, then snapped, and the per-rep max knee flexion read +25° (depth error +7.5 / +9.9° for that seed
   because the per-rep max is a single-frame statistic once the angle filter is gone); (b) the foot model holding feet at
   a stale anchor after a step (bug B2; walkout lower-body 26 vs 19 mm Kalman-only).
   Smaller: heel-rise attenuation at the 2 cm floor (B6), stance widening −3 to −6 % (anchor settles before the min-jerk
   tail finishes), analysis latency 82 ms instead of the nominal 66 ms because the harness still uses the old
   `get_synced_frames` (sync failures 1.5–22 %, mean ~13 %, stretch two frames of lag in wall time; WS1's new sync
   should bring it to ~70 ms).

### 4. Bugs / issues in the new components (no crashes in 240 + 48 runs; these are behavioural)
- **B1 (WS4 `keypoint_kalman.py`) re-acquire snaps onto 2-frame consistent outlier bursts.** C4's "snap after 2
  consecutive rejected measurements that agree within 10 cm" cannot distinguish a real jump from a burst, and measured
  bursts ARE consistent (RTMPose outlier runs 1.6 frames mean / p90 3 with constant offset; L/R swaps 2–6 frames; a
  swap seen by 2 of 3 cameras beats the triangulator's swap test). Repro `repro_kalman_reacquire.py`: 1-frame burst ->
  1.3 cm; 2-frame burst 60 cm away at conf 0.20 -> lagged error 47 cm for 2 frames, display 60 cm; 3 frames -> 60 cm;
  same at conf 0.55 and at lag 0. Suggest requiring ≥ 3–4 agreeing rejections (or the 0.2 s timeout only) and refusing
  to snap onto measurements whose confidence is at the two-view cap (≤ 0.3).
- **B2 (WS5 `foot_contact.py`) a normal step leaves the foot at its old anchor for 1–2 s.** Repro `repro_foot_step.py`
  (+ instrumented sweep in the session log): 25 cm step in 0.5 s at 30 fps -> released 0.23 s after step start but
  re-planted only 1.13 s after the step ends, because after release the toe's gate reference is `_last_out` (= the old
  anchor), every observation > 8 cm away is rejected and only `REACQUIRE_REJECTED_FRAMES` (30 frames) re-acquires it,
  after which 12 accepted frames + 2 still checks are needed. 25 cm in 0.3 s -> not recognised as movement at all
  (window never gets 3 inliers within the 8 cm gate), foot held 1.6 s by the `DISPLACED_RELEASE_FRAMES` timeout with a
  25 cm output-ankle error; with every 2nd tick dropped (15 Hz effective) -> released after 3.2 s, never re-planted.
  In the harness: walkout tpose seed 0 (28 % sync failures) kept the left foot planted through the whole walk
  (lower-body error 94 mm during, 95 mm / stance 186 mm for ~1 s after; recovered before rep 0). During that time the
  analysis skeleton's ankle/toe are wrong by the step distance, so IK knee/ankle/valgus are wrong too. Suggest: on
  release, re-seed the gate reference from the current observation (or accept unconditionally for a few frames), drop
  `REACQUIRE_REJECTED_FRAMES` to ≤ 5, and treat a coherent ankle+toe(+heel) displacement as a step rather than a pose
  failure.
- **B3 (WS6 `segment_lengths.py` vs C3 confidences) the conf ≥ 0.6 endpoint gate is rarely met on triangulated input.**
  0.6 means u ≤ 1.6 cm. Kalman-lagged confidences reach it on both femur endpoints in 31 % of frames (perfect) and
  4 % (tpose); raw triangulated: 8 % / 1 %; walkout tpose 1 % / 0 %. 150 samples per rigid segment therefore take
  ~17 s (perfect) to ~2 min (tpose); the 600-frame/3-rep fallback would fire first in tpose. A gate of 0.4
  (u ≤ 2.4 cm) passes 94 % / 64 % of frames. Check against real logs before changing; the harness σ̂_px (4.8–6.7 px
  reprojection) is partly assumed.
- **B4 (`analytical_ik.py` MIN_CONFIDENCE = 0.1 on metric confidence)** drops accepted keypoints with u > 6 cm: 4.0 /
  7.1 % knee-NaN frames when IK sees raw triangulated skeletons (old chains). Harmless behind the Kalman (0.1 / 0.2 %),
  but any tool/gate that feeds raw triangulated output to IK inherits it.
- **B5 (heel-rise sensitivity, not a bug)** the harness heel_rise scenario's true ankle rise is 2.22 cm (the "3 cm" is a
  toe-lever assumption), right at the model's ~2 cm floor: FootState peak per rep 1.68 ± 0.64 / 1.82 ± 0.74 cm
  (reads 0 on ~1/3 of feet-reps, 2.0–4.3 cm when caught); bottom-window median only 0.43× because detection lands at
  the bottom. Zero false heel rise on clean reps (peak 0.00 cm every rep). A 3 cm rise (WS5's test) is fine; mild
  threshold 1.5 cm is below what the model can resolve.
- **B6 (compute)** `Skeleton3D.from_numpy` + `recentre_at_hips` cost 53 µs/frame single-process (70–73 µs under load),
  as much as the Kalman itself (64 µs single-process / 85 under load; WS4 bench 36 µs); foot model 86 / 116–120 µs
  (WS5 bench 75). Integration should recentre in numpy and build one Skeleton3D. Whole proposed chain 250–350 µs incl.
  ~45 µs harness overhead; frame incl. IK + valgus 530–730 µs.
- **B7 (old chains)** `current`/`full_original` numbers here use the vendored legacy modules with the old config defaults
  (blend 0.1/0.9, clamp 2.5 m/s, bone 30/0.0, ground 30/0.02/0.01/0.75, One Euro 0.8/4.0/1.0); wave 2 removed those
  config sections mid-run.

### 5. What this run does NOT cover
WS2 crop fix (detector noise as measured before it); WS1's new `get_synced_frames` (delivery still fails sync on
1.5–22 % of ticks); heels (never detected -> foot model runs on ankle + toe); `predict_missing` was exercised on
0.00–0.02 % of frames (triangulation never returned None on realistic/harsh noise) so the dropout path is untested;
gates run on the raw recentred skeleton for every chain (identical timing); the proposed chain is fed from the first
triangulated skeleton (pre-readiness) because Kalman/foot model are session-scoped; timings under 6-process load.
