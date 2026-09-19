# Person-based extrinsic calibration (K2) — triangulated 3-camera harness

Generated 2026-09-17 23:48:58; wall 75 s; seeds [0, 1, 2]; noise `realistic`; scenarios: clean, valgus, heel_rise, hip_shift, asymmetric_depth, fast_reps, stance_change, walkout; workers 6.

- `person_ba*`: the production `PersonCalibrator` on the harness's own noisy detections of ONE capture per seed (the `walkout` generator with a separate seed: stand, walk 0.5 m with a 20 deg turn, stand, 2 reps; same detector noise, occlusion, L/R swaps and camera-delivery model as every run; bar ends simulated with 3 px white + 3 px slow noise, ordered by image x like the YOLO detector). Every scenario of that seed is then triangulated with that calibration, as `perfect`/`tpose` are.
- Gauge: the calibrator anchors its world frame on the lifter (heading + origin from the standing frames); the harness truth is anchored on the T-pose spot. Heading (yaw about the true vertical) and origin are removed with the capture's true joints; **tilt of the solved vertical, relative camera poses and scale are not corrected** and are inside every number below. Camera centre / rotation errors are after that yaw + origin alignment.
- `floor mm`: calibration-only error — noise-free true projections triangulated with the calibration, hip-centred MPJPE vs truth (0 for `perfect`).
- Chains: `proposed` (FixedLagKeypointSmoother + FootContactModel, new tail) and `raw`.

## Chain `proposed` — mean over 8 scenarios x 3 seeds

| calibration | MPJPE mm | lower mm | p95 mm | stance MAE mm | bone len MAE mm | knee@true-bottom err deg | knee MAE moving deg | depth err deg | valgus@true-bottom err deg | knee NaN frac | floor mm | reproj px | setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| perfect | 19.6 | 18.9 | 37.5 | 6.3 | 10.2 | -0.07 | 2.06 | 0.67 | -0.04 | 0.001 | 0.0 | 4.87 | true projection matrices |
| tpose | 32.7 | 30.1 | 65.2 | 21.2 | 16.8 | -0.10 | 2.46 | 0.89 | 0.29 | 0.004 | 22.0 | 7.15 | production T-pose PnP, guessed f = 0.8 w |
| person_ba | 19.8 | 19.1 | 38.0 | 6.5 | 10.2 | -0.21 | 2.02 | 0.58 | -0.11 | 0.001 | 2.7 | 4.88 | true K, bar scale, essential-matrix start |
| person_ba_from_tpose | 19.8 | 19.1 | 38.1 | 6.5 | 10.2 | -0.21 | 2.02 | 0.58 | -0.11 | 0.001 | 2.7 | 4.88 | true K, bar scale, refine() from the T-pose calibration |
| person_ba_height | 22.6 | 21.7 | 41.0 | 8.3 | 11.3 | -0.03 | 2.02 | 0.75 | -0.05 | 0.001 | 10.9 | 4.88 | true K, height-prior scale |
| person_ba_k+10 | 25.0 | 23.8 | 45.8 | 8.7 | 12.5 | -1.09 | 2.16 | -0.33 | 0.13 | 0.000 | 14.6 | 4.89 | focal +10 %, bar scale |
| person_ba_k-10 | 28.0 | 27.8 | 50.5 | 10.9 | 15.3 | 0.87 | 2.13 | 1.65 | -0.24 | 0.003 | 19.0 | 4.95 | focal -10 %, bar scale |
| person_ba_k+10_height | 26.7 | 25.5 | 47.3 | 9.9 | 13.6 | -0.83 | 2.14 | -0.02 | 0.05 | 0.000 | 17.6 | 4.91 | focal +10 %, height-prior scale |
| person_ba_k-10_height | 21.4 | 20.5 | 40.7 | 7.0 | 11.0 | 1.03 | 2.16 | 1.77 | -0.15 | 0.002 | 7.7 | 4.90 | focal -10 %, height-prior scale |

## Chain `raw` — mean over 8 scenarios x 3 seeds

| calibration | MPJPE mm | lower mm | p95 mm | stance MAE mm | bone len MAE mm | knee@true-bottom err deg | knee MAE moving deg | depth err deg | valgus@true-bottom err deg | knee NaN frac | floor mm | reproj px | setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| perfect | 25.0 | 27.3 | 48.7 | 13.3 | 14.5 | -0.02 | 4.67 | -3.53 | -0.17 | 0.040 | 0.0 | 4.87 | true projection matrices |
| tpose | 35.9 | 37.1 | 73.0 | 23.0 | 20.6 | 0.14 | 6.49 | -4.13 | 0.30 | 0.071 | 22.0 | 7.15 | production T-pose PnP, guessed f = 0.8 w |
| person_ba | 25.3 | 27.5 | 49.3 | 13.5 | 14.5 | -0.09 | 4.76 | -3.45 | -0.14 | 0.042 | 2.7 | 4.88 | true K, bar scale, essential-matrix start |
| person_ba_from_tpose | 25.3 | 27.5 | 49.3 | 13.5 | 14.5 | -0.09 | 4.76 | -3.45 | -0.14 | 0.042 | 2.7 | 4.88 | true K, bar scale, refine() from the T-pose calibration |
| person_ba_height | 27.4 | 29.2 | 51.8 | 13.6 | 15.2 | 0.08 | 4.39 | -3.03 | -0.21 | 0.035 | 10.9 | 4.88 | true K, height-prior scale |
| person_ba_k+10 | 29.2 | 30.6 | 55.0 | 13.9 | 16.0 | -0.90 | 3.97 | -3.88 | 0.18 | 0.025 | 14.6 | 4.89 | focal +10 %, bar scale |
| person_ba_k-10 | 32.6 | 34.7 | 61.4 | 17.0 | 19.0 | 1.04 | 6.53 | -3.21 | -0.33 | 0.074 | 19.0 | 4.95 | focal -10 %, bar scale |
| person_ba_k+10_height | 30.7 | 31.7 | 56.5 | 14.1 | 16.9 | -0.70 | 3.91 | -3.62 | -0.03 | 0.024 | 17.6 | 4.91 | focal +10 %, height-prior scale |
| person_ba_k-10_height | 26.8 | 28.7 | 52.4 | 13.2 | 15.1 | 1.02 | 5.58 | -2.66 | -0.25 | 0.059 | 7.7 | 4.90 | focal -10 %, height-prior scale |

## Targets (chain `proposed`): within 3 mm of `perfect`

| calibration | MPJPE - perfect mm | stance MAE - perfect mm | met |
|---|---:|---:|---|
| tpose | +13.2 | +14.9 | no |
| person_ba | +0.3 | +0.2 | yes |
| person_ba_from_tpose | +0.3 | +0.2 | yes |
| person_ba_height | +3.1 | +2.0 | no |
| person_ba_k+10 | +5.4 | +2.4 | no |
| person_ba_k-10 | +8.4 | +4.6 | no |
| person_ba_k+10_height | +7.2 | +3.6 | no |
| person_ba_k-10_height | +1.8 | +0.7 | yes |

## Calibration quality (mean over seeds; per-seed values in person_ba_results.json)

| calibration | cam centre err mm | cam rotation err deg | vertical tilt deg | working-volume scale err % | BA reproj RMS px (start -> end) | frames used / captured | standing frames | calibrate() s (inside the parallel pool) |
|---|---:|---:|---:|---:|---|---|---:|---:|
| tpose | 531.4 | 5.69 | - | - | - -> - | - / - | - | - |
| person_ba | 17.9 | 0.33 | 0.25 | 0.03 | 8.08 -> 4.09 | 150 / 255 | 100 | 2.84 |
| person_ba_from_tpose | 17.9 | 0.33 | 0.25 | 0.03 | 15.24 -> 4.09 | 150 / 255 | 100 | 3.13 |
| person_ba_height | 60.5 | 0.51 | 0.35 | -1.66 | 8.08 -> 4.05 | 150 / 255 | 100 | 2.41 |
| person_ba_k+10 | 276.7 | 0.81 | 0.46 | -1.06 | 8.27 -> 4.11 | 150 / 255 | 100 | 2.98 |
| person_ba_k-10 | 274.7 | 1.04 | 0.69 | 1.46 | 10.18 -> 4.15 | 150 / 255 | 100 | 3.33 |
| person_ba_k+10_height | 254.2 | 0.76 | 0.37 | -1.52 | 8.27 -> 4.06 | 150 / 255 | 100 | 2.70 |
| person_ba_k-10_height | 360.5 | 1.00 | 0.34 | -1.82 | 10.18 -> 4.09 | 150 / 255 | 100 | 2.33 |

`tpose` rows: centre / rotation errors as reported by the T-pose harness mode (no alignment needed: the T-pose defines the truth frame).

## Per scenario, chain `proposed`: MPJPE mm / stance MAE mm (mean over seeds)

| scenario | perfect | tpose | person_ba | person_ba_from_tpose | person_ba_height | person_ba_k+10 | person_ba_k-10 | person_ba_k+10_height | person_ba_k-10_height |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 20.4 / 3.9 | 36.0 / 21.0 | 20.6 / 4.7 | 20.6 / 4.7 | 23.0 / 8.4 | 26.4 / 9.1 | 29.2 / 7.8 | 28.1 / 10.9 | 21.3 / 6.0 |
| valgus | 18.5 / 4.4 | 29.4 / 16.7 | 18.7 / 4.1 | 18.7 / 4.1 | 21.5 / 6.2 | 24.1 / 6.9 | 28.5 / 9.5 | 26.0 / 7.5 | 20.3 / 5.1 |
| heel_rise | 20.6 / 6.2 | 34.4 / 16.8 | 20.7 / 6.8 | 20.7 / 6.8 | 23.7 / 6.0 | 27.1 / 7.2 | 28.9 / 12.8 | 29.0 / 7.6 | 21.5 / 5.0 |
| hip_shift | 19.4 / 7.5 | 31.8 / 20.2 | 19.6 / 7.9 | 19.6 / 7.9 | 22.3 / 8.6 | 25.5 / 9.7 | 28.8 / 11.5 | 27.2 / 10.8 | 20.7 / 7.6 |
| asymmetric_depth | 20.4 / 6.0 | 31.1 / 18.4 | 20.7 / 6.2 | 20.7 / 6.2 | 24.0 / 7.2 | 26.9 / 7.4 | 28.0 / 11.1 | 28.9 / 8.8 | 21.3 / 5.7 |
| fast_reps | 19.3 / 4.9 | 30.0 / 20.3 | 20.0 / 5.2 | 20.0 / 5.2 | 22.9 / 7.8 | 25.5 / 8.6 | 29.1 / 8.4 | 27.5 / 10.0 | 21.3 / 6.1 |
| stance_change | 20.3 / 7.2 | 33.1 / 26.7 | 20.6 / 6.6 | 20.6 / 6.6 | 22.6 / 10.2 | 24.6 / 10.4 | 30.9 / 14.7 | 26.4 / 13.2 | 22.5 / 7.4 |
| walkout | 17.5 / 10.5 | 36.0 / 29.1 | 17.7 / 10.2 | 17.7 / 10.2 | 21.0 / 11.9 | 19.7 / 10.5 | 20.2 / 11.5 | 20.8 / 10.5 | 22.2 / 13.5 |

## `calibrate()` timing on this machine (Apple M2, single process, 150 frames, 3 cameras, 19 keypoints + bar; single-threaded BLAS)

| mode | runs | mean s | min s | max s |
|---|---:|---:|---:|---:|
| person_ba | 3 | 2.16 | 1.29 | 3.08 |
| person_ba_from_tpose | 3 | 2.31 | 2.05 | 2.47 |
| person_ba_height | 3 | 1.80 | 1.42 | 2.06 |

