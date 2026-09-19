# Triangulated Ground-Truth Harness + Baseline

(Saved by lead from the agent's final message; subagent sandbox blocked writing REPORT.md. Raw numbers in `baseline_summary.md` / `baseline_results.json` via `run_baseline.py`; PROGRESS.md current.)

**Headline:** in triangulated mode every pre-IK stage except the velocity clamp makes squat accuracy worse, and the biggest error in the pipeline isn't a filter — it's IK zeroing joint angles when triangulated confidence drops below 0.1.

Full baseline 47 s with 6 workers. No repo source files changed. Deterministic per seed (timings aside). Every chain gets identical input per scenario/seed/calibration → chain-vs-chain differences are paired.

## 1. Layout (this folder)
| Path | Role |
|---|---|
| `noise_calib/run_rtmpose.py`, `measure_noise.py` → `noise_params.json` | Real `RTMPoseEstimator` (production decode) on `data/squats.mov`, pooled with 3 real 720p recordings |
| `preik_harness/body.py` | Ground-truth squat motion |
| `cameras.py` | Rig; `perfect` calibration or `tpose` (real `TPoseCalibrator`) |
| `detector.py` | 2D detector sim; noise profiles `realistic` (default), `mild`, `harsh`, `none` |
| `delivery.py` | Camera clocks, loop pacing, `get_synced_frames()` logic |
| `runner.py` | Inputs through real `DLTTriangulator`; replays `process_frame` |
| `chains.py` | Real production filter classes; `DiagnosticConfidenceFloor` |
| `metrics.py`, `api.py` | Metrics; `evaluate` / `evaluate_chains` / `compare` / `get_prepared` |
| `run_baseline.py` | Writes `baseline_results.json` (3.2 MB), `baseline_summary.md` |

**API (verified):**
```python
import sys; sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/.claude/preik-audit/harness")
from preik_harness import SCENARIOS, FrameContext, DeliveryConfig, evaluate, evaluate_chains, production_chain, compare
class MyChain:                      # process() + reset(); optional calibrated()
    def reset(self): ...
    def process(self, skeleton, context: FrameContext):  # context: tick_index, loop_time_s, is_held,
        return skeleton             # standing_gate, readiness_gate, multi_view (per-view 2D), calibration
one = evaluate(MyChain, scenarios=["clean","valgus"], calibration="tpose", seed=0, n_seeds=3, workers=4)
many = evaluate_chains({"raw": production_chain([]), "current": production_chain(["blend","vclamp"]), "mine": MyChain},
                       scenarios=SCENARIOS, calibration=["perfect","tpose"], workers=6)
print(compare(many, ["mpjpe_mm","lag_knee_flex_ik_ms","knee_flex_tb_err_deg"], calibration="tpose"))
```
Options: `noise=` profile name or `NoiseProfile`; `delivery_config=DeliveryConfig(camera_fps=24, ...)`; `rig_config=RigConfig(...)`. Stage names: `blend`, `vclamp`, `bone`, `ground`, `smooth` (repeated `bone` reuses the instance, as production).

**Real repo code used:** `DLTTriangulator`, `TPoseCalibrator`, all 5 pre-IK filters, both standing gates (`load_pipeline_config()`), `AnalyticalIKSolver`, `build_valgus_estimator(True)`, `JointAngleFilter` (phase from real squat `SignalRepCounter`).
**Order replicated:** sync failure → no frame (production returns before dropout hold) → dropout hold → standing gate on raw → readiness gate early return → chain → IK → valgus → angle filter.
**Not replicated:** BiLSTM, predictive extrapolator, rules engine, body-proportion scaling, multi-set resets.

**Ground truth:** 1.885 m body with proportions deliberately different from calibration model (tibia 0.246H vs 0.235H, torso 0.300H vs 0.290H, nose 9 cm forward, …). Legs solved from moving pelvis to planted feet → bone lengths exactly constant (0.000 mm range). World-coordinate poses (pelvis drops ~0.5 m, moves ~0.2 m back), then hip-centred. 5 reps per scenario; knee flexion hits 100–120° within 0.3°. Scenarios: `valgus` +13–14° at bottom (clean −2 to −4°); `heel_rise` +3 cm; `hip_shift` 4 cm + 5° list; `asymmetric_depth` −8° L−R; `fast_reps` 0.8–1.0 s; `stance_change` each foot 8 cm out; `walkout` 0.5 m forward + 20° turn. Cameras: true focal 0.72w; `tpose` guesses 0.8w → camera-centre errors 0.3–0.9 m, rotation errors 1.8–13°. Loop 28.4 Hz, skips 5.7 % primary frames, < 0.5 % duplicates at 30 fps.

## 2. Measured real RTMPose-m noise
- Still jitter (through 3.33 × 1.41 px decode quantizer), x/y px: hip 1.95/1.8, knee 1.75/1.5, ankle 0.9/0.85, toe 2.0/1.0; 720p/15 fps recording 2–3.5 px legs. Slow in-view drift 1.2–2.7 px.
- Confidence p5/p50/p95: hip 0.60/0.63/0.68, knee 0.60/0.64/0.69, ankle 0.58/0.64/0.70, toe 0.57/0.62/0.67; min 0.531, never < 0.3; lag-1 autocorrelation 0.58.
- Outliers > 30 px: ankles 0.4–11 %, toes 1–18 %, knees/hips 0–0.8 %; runs 1.64 frames mean (p90 3), 35–230 px; confidence 0.589 during vs 0.633 normal.
- L/R leg swaps frontal: 0.26 % frames.
- **Assumed (needs real multi-view data):** per-view slow bias 2.5–4 px (→ ~5 px perfect / ~6.5 px tpose reprojection); far-side occlusion in side cameras growing with depth; leg swaps 15 % of reps per side camera, 2 % front.

## 3. Baseline (`realistic`, 8 scenarios × 3 seeds, averaged)
Columns: knee err at true bottom = median IK knee error ±150 ms of true bottom (valid frames); depth err IK / final = per-rep max knee flexion − true max before/after angle filter (negative = undershoot); bottom-frame delay = lateness of pipeline's chosen bottom frame.
| chain | cal | MPJPE mm | p95 mm | still jitter p50 mm | knee lag (IK) ms | knee err at true bottom ° | depth err IK / final ° | bottom-frame delay ms | chain µs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | perfect | 33.8 | 99 | 34.9 | 4 | +0.1 | +0.9 / −9.5 | 71 | 45 |
| current | perfect | 35.8 | 106 | 9.5 | 54 | −2.9 | −1.8 / −11.2 | 146 | 126 |
| full_original | perfect | 53.5 | 161 | 1.5 | 112 | −6.0 | −2.8 / −11.1 | 214 | 512 |
| blend_only | perfect | 34.7 | 99 | 9.5 | 53 | −2.8 | −1.7 / −11.1 | 146 | 94 |
| vclamp_only | perfect | 31.8 | 92 | 33.8 | 7 | −0.1 | +0.8 / −9.6 | 77 | 93 |
| bone_only | perfect | 34.5 | 98 | 34.9 | 4 | −0.2 | +0.9 / −9.6 | 77 | 138 |
| ground_only | perfect | 35.4 | 103 | 35.1 | 4 | +0.5 | +1.3 / −9.1 | 71 | 94 |
| smooth_only | perfect | 39.7 | 101 | 4.6 | 53 | −1.8 | −0.4 / −9.9 | 148 | 191 |
| raw | tpose | 43.9 | 114 | 35.0 | 6 | −0.1 | −5.1 / −15.2 | 98 | 50 |
| current | tpose | 72.2 | 276 | 7.8 | 82 | −7.5 | −11.4 / −19.8 | 223 | 131 |
| full_original | tpose | 88.3 | 304 | 1.5 | 145 | −12.2 | −13.2 / −20.8 | 288 | 524 |
| vclamp_only | tpose | 41.7 | 107 | 33.8 | 9 | −0.2 | −5.1 / −15.3 | 107 | 104 |
| smooth_only | tpose | 48.2 | 119 | 4.6 | 58 | −1.9 | −5.9 / −15.3 | 168 | 182 |
No-noise floor (raw): perfect 1.4 mm MPJPE, −2.7° final depth (angle filter alone); tpose 17.9 mm. **Zeroed knee angles: 25 % (perfect) / 28 % (tpose) of frames for every chain, 29–38 % near rep bottoms.**

## 4. Per stage (paired vs raw, perfect / tpose)
- **ConfidenceBlender — remove as position filter or rebuild.** +49 / +72 ms lag; knee at true bottom −2.9 / −7.1°, fast reps −6.3 / −12.8°; moving error 36 → 46 mm (perfect), 51 → 110 mm (tpose); tpose walkout 37 → 97 mm. Does cut still jitter 35 → 9.5 mm, bone spread −18 / −10 mm, stance error −17 / −15 mm. Why: weight = triangulated confidence (median 0.45 / 0.40; 0.24 tpose walkout) → heavy per-call moving average ignoring timing; 8 % of lower-body keypoints (25 % tpose walkout) get zero weight and freeze; per-keypoint lag differences exaggerate heel rise 1.7–2.0× (raw 0.67×). Worse as calibration degrades. `current` = `blend_only` on every metric.
- **VelocityClamp — keep (in current form, small gain).** MPJPE −2.0 / −2.2 mm, lower body −5.2 / −5.5, p95 −7, bone spread −7.9, stance −6; lag +2.6 / +3.0 ms; depth & valgus ±0.2°. Can't catch dominant errors (outliers 1–3 frames; 2.5 m/s allows 8 cm/frame).
- **BoneLengthConstraints — rebuild or remove.** Bone spread −4.4 mm only; angles ≈ unchanged; pelvic list amplitude 1.05 → 0.43× (perfect), 0.92 → 0.36× (tpose); noiseless 0.999×, mild noise 0.64×. Hips are the distal end of shoulder→hip bones → shoulder noise projected into hip heights.
- **GroundClamp — remove.** `stance_change` widening erased (ratio −0.01 / 0.02 vs raw 0.99 / 0.92); stance error 133 / 108 vs 28 / 37 mm; overall stance +12 / +8.5 mm, lower body +4 mm. Ankle-over-toe heel rise untouched by design (needs unilateral scenario to show flat-floor harm).
- **Position smoother (One Euro) — rebuild.** Best jitter (35 → 4.6 mm) but +49 / +52 ms lag, MPJPE +5.9 / +4.3 mm; knee at true bottom −1.9 / −1.8°, fast −3.9 / −3.1°; hip-shift amplitude 1.10 → 0.66× (perfect); 344 µs p95.
- **full_original — worst everywhere.** MPJPE +19.7 / +43 mm; lag +108 / +134 ms; knee at true bottom −6.1 / −11.9°, fast −11.8 / −18.8°; pelvic list 0.68 / 0.47×; stance change 0.03 / 0.08×; bottom-frame delay 214 / 288 ms; ~1 ms/frame p95. Confirms the user's decision to disable those stages.
- **Valgus** at true bottom survives every chain (0.86–1.04×, ±2°); at the pipeline's chosen bottom frame ~0.78× for all chains → loss is after pre-IK.

## 5. Bigger problems exposed
1. **CONFIRMED, largest error source — IK zeroing.** Triangulator confidence × (1 − reproj/15) or × 0.1 above 15 px → anything > ~12.6 px ends < 0.1 → IK knee flexion exactly 0.0 (valgus estimator likewise). Diagnostic chain flooring confidence at 0.11 (positions raw): moving knee error 14.3 → 4.3° (perfect), 16.6 → 4.6° (tpose); final depth −9.5 → −3.1°, −15.2 → −5.6°. No current stage touches confidence.
2. **Angle filter drives final depth error.** Raw input: depth +0.9° before filter, −9.5° after; bottom frame 71 ms late; hip flexion at that frame −7.9° (perfect) / −16° (tpose).
3. **T-pose calibration breaks away from the calibration spot.** After walkout reprojection 9–15 px; readiness gate never opened in 1/3 tpose seeds; another seed 77 % zeroed knees.
4. **`get_synced_frames` drops 1.5–22 % of loop iterations** (side camera's matching frame not yet arrived → only candidate ~33 ms old > 15 ms tolerance).
5. **Axis convention:** flat T-pose model with X = subject's left forces solvePnP cameras to −Z → +Z is the subject's BACK in triangulated mode (docstring says forward); direction-dependent IK outputs (pelvis tilt sign, `wrist_x`, hip rotation) mirrored.
6. **Low-light 24 fps:** ~6 % duplicates; blender lag slightly worse (−3.6° vs −2.9° at true bottom).

## 6. Biasing simplifications
- **Noise realism is the biggest uncertainty:** per-view slow bias, occlusion model, swap rates assumed. Zeroed-knee rate very sensitive: 10 % mild, 25 % realistic, 61 % harsh. Check vs real triangulated logs (harness assumes ~5 px mean reprojection, 8–9 % lower-body keypoints < 0.1 confidence). Per-stage ranking held under mild and harsh.
- Keypoints = exact joint centres (no pose-dependent landmark drift correlated across views) → absolute accuracy optimistic; such bias invisible to filters anyway.
- Outliers random-direction (real ones often snap to another body part); swaps only knees/ankles/toes; no lens distortion, motion blur, rolling shutter; ideal T-pose; rigid trunk with pelvis.
- Bone/ground calibration inside the chain (production: bone calibration from assessment wait loop); first rep ~2 s in, as 30-frame calibrations finish (1.7–3.5 s).
- Small-signal metrics noisy (heel-rise ratio ±0.3–1.0, asymmetry ±4° across seeds) → more seeds before acting on small differences; trust paired changes.
- Compute times Python on Mac incl. ~40 µs harness overhead; not Jetson.
