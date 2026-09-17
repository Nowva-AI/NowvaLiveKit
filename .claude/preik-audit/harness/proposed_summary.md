# New pre-IK stack vs old chains — triangulated 3-camera harness

Generated 2026-09-17 21:29:39; wall 23 s; seeds [0, 1, 2]; noise `realistic`; scenarios: clean, valgus, heel_rise, hip_shift, asymmetric_depth, fast_reps, stance_change, walkout; workers 6.

- Inputs: real rewritten `DLTTriangulator` (world frame, metric confidence, 21 keypoints; heels never detected by the simulated detector -> conf 0). Delivery model unchanged (old `get_synced_frames` semantics: sync failures are skipped ticks). Detector noise model unchanged (WS2 crop fix NOT simulated).
- Old chains (`raw`, `current`, `full_original`): `recentre_at_hips` -> chain -> IK (NaN->0.0 as the old IK did) -> `JointAngleFilter` (legacy copy), production dropout hold. Fresh baseline on the new triangulator.
- `raw (old triangulator)`: numbers copied from `baseline_results.json` (old triangulator, same harness).
- `proposed_nofoot`: `FixedLagKeypointSmoother` (lag 2) -> `recentre_at_hips`; `proposed`: + `FootContactModel.update` on the lagged world stream before recentring. New tail: `AnalyticalIKSolver` (NaN for missing), NO angle filter (final == IK), `predict_missing` on frames without a skeleton. Outputs are aligned to the truth of the tick they belong to (2-frame lag removed) — `analysis latency` and `bottom-frame delay (wall)` add it back.
- Valgus: truth is the SAME `TriangulatedValgusEstimator` on the noiseless GT skeleton, so the WS7 metric change cancels in ratios and errors.

## Overall (mean over 8 scenarios x 3 seeds) — calibration `perfect`

| chain | MPJPE mm | lower mm | p95 mm | still jitter p50 mm | knee lag IK ms | knee@true-bottom err deg | knee MAE moving deg | depth err IK deg | depth err final deg | bottom-frame delay ms (aligned) | bottom-frame delay ms (wall) | analysis latency ms | knee NaN/0 frac | stance MAE mm | chain us | frame us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw (old triangulator) | 33.8 | 42.2 | 98.7 | 34.9 | 4 | 0.1 | 14.3 | 0.9 | -9.5 | 71 | - | - | - | 26.7 | 45 | 394 |
| raw | 25.0 | 27.3 | 48.7 | 30.8 | 0 | -0.0 | 4.7 | 2.2 | -3.6 | 107 | 107 | 0 | 0.040 | 13.3 | 48 | 453 |
| current | 29.2 | 30.0 | 73.5 | 11.9 | 39 | -2.0 | 8.0 | -0.1 | -5.0 | 156 | 156 | 0 | 0.040 | 10.2 | 138 | 546 |
| full_original | 46.2 | 45.4 | 133.7 | 1.6 | 91 | -4.0 | 14.6 | -1.5 | -5.3 | 213 | 213 | 0 | 0.040 | 21.4 | 589 | 1015 |
| proposed_nofoot | 21.1 | 22.6 | 40.7 | 4.4 | 1 | -0.1 | 2.2 | 1.1 | 1.1 | 14 | 93 | 82 | 0.001 | 10.5 | 231 | 617 |
| proposed | 20.1 | 20.2 | 37.9 | 3.1 | 0 | -0.2 | 2.1 | 0.8 | 0.8 | 6 | 89 | 82 | 0.001 | 6.7 | 379 | 796 |

### Paired delta vs `raw` (new triangulator), mean ± std over 8 scenarios x seeds — `perfect`

| chain | mpjpe_mm | mpjpe_lower_mm | p95_err_mm | jitter_still_p50_mm | lag_knee_flex_ik_ms | knee_flex_tb_err_deg | knee_flex_ik_moving_mae_deg | depth_err_ik_deg | depth_err_final_deg | valgus_tb_err_deg | stance_width_mae_mm | ik_knee_invalid_frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 4.1 ± 4.9 | 2.7 ± 3.7 | 24.8 ± 22.3 | -19.0 ± 2.1 | 38.3 ± 8.2 | -1.9 ± 1.7 | 3.4 ± 1.9 | -2.3 ± 1.4 | -1.4 ± 1.3 | -0.2 ± 1.1 | -3.1 ± 1.0 | 0.000 ± 0.000 |
| full_original | 21.2 ± 7.4 | 18.1 ± 6.7 | 85.0 ± 31.8 | -29.2 ± 1.8 | 90.6 ± 11.2 | -4.0 ± 3.0 | 9.9 ± 2.6 | -3.7 ± 2.0 | -1.7 ± 2.2 | -0.1 ± 2.0 | 8.1 ± 38.6 | 0.000 ± 0.000 |
| proposed_nofoot | -3.9 ± 0.5 | -4.7 ± 0.9 | -8.0 ± 1.2 | -26.4 ± 1.8 | 0.2 ± 5.4 | -0.1 ± 0.6 | -2.4 ± 2.1 | -1.1 ± 1.4 | 4.6 ± 2.4 | -0.2 ± 1.3 | -2.8 ± 0.8 | -0.039 ± 0.034 |
| proposed | -4.9 ± 2.2 | -7.0 ± 5.0 | -10.7 ± 2.0 | -27.8 ± 1.8 | -0.1 ± 5.1 | -0.2 ± 1.1 | -2.6 ± 2.1 | -1.4 ± 1.4 | 4.4 ± 2.1 | 0.1 ± 1.6 | -6.6 ± 2.4 | -0.039 ± 0.034 |

### Fault-signal preservation and scenario-specific metrics — `perfect` (mean ± std over seeds)

| scenario / metric | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| valgus: valgus amplitude ratio @true bottom (same estimator on truth; 1 = preserved) | 1.02 ± 0.08 | 0.78 ± 0.12 | 0.73 ± 0.15 | 0.65 ± 0.17 | 0.87 ± 0.23 | 1.10 ± 0.06 |
| valgus: valgus amplitude ratio @pipeline bottom frame | 0.78 ± 0.11 | 0.84 ± 0.04 | 0.76 ± 0.10 | 0.72 ± 0.06 | 0.82 ± 0.29 | 1.10 ± 0.08 |
| valgus: valgus @true bottom err deg | 0.62 ± 0.78 | -2.55 ± 1.08 | -3.12 ± 1.42 | -4.87 ± 0.84 | -2.03 ± 1.19 | 0.02 ± 0.72 |
| valgus: valgus @pipeline bottom frame err deg | -3.30 ± 1.45 | -3.50 ± 0.37 | -4.78 ± 0.67 | -5.78 ± 0.49 | -2.86 ± 3.03 | 0.45 ± 0.91 |
| heel_rise: heel rise (ankle over toe, positions) amplitude ratio @true bottom | 0.67 ± 0.17 | 1.01 ± 0.50 | 0.90 ± 0.52 | 1.12 ± 0.27 | 0.89 ± 0.59 | 0.40 ± 0.20 |
| heel_rise: heel rise amplitude ratio @pipeline bottom frame | 0.42 ± 0.14 | 0.91 ± 0.67 | 0.92 ± 0.42 | 1.27 ± 0.47 | 0.98 ± 0.37 | 0.44 ± 0.24 |
| heel_rise: FootState heel_rise_cm / true ankle rise @true bottom | - | - | - | - | - | 0.43 ± 0.13 |
| heel_rise: heel rise PEAK per rep (positions) / true peak ankle rise | - | 6.07 ± 4.03 | 1.76 ± 0.51 | 1.61 ± 0.30 | 1.57 ± 0.11 | 0.98 ± 0.17 |
| heel_rise: FootState heel rise PEAK per rep / true peak ankle rise | - | - | - | - | - | 0.76 ± 0.29 |
| heel_rise: FootState heel rise PEAK per rep cm | - | - | - | - | - | 1.68 ± 0.64 |
| heel_rise: true peak ankle rise per rep cm (harness GT) | - | 2.22 ± 0.00 | 2.22 ± 0.00 | 2.22 ± 0.00 | 2.22 ± 0.00 | 2.22 ± 0.00 |
| hip_shift: lateral hip shift amplitude ratio @true bottom | 1.10 ± 0.19 | 1.08 ± 0.20 | 1.08 ± 0.21 | 1.06 ± 0.07 | 1.13 ± 0.35 | 0.90 ± 0.07 |
| hip_shift: lateral hip shift amplitude ratio @pipeline bottom frame | 0.96 ± 0.32 | 0.92 ± 0.33 | 0.99 ± 0.24 | 1.00 ± 0.10 | 1.25 ± 0.49 | 0.92 ± 0.07 |
| hip_shift: pelvic list amplitude ratio @true bottom | 1.05 ± 0.29 | 1.06 ± 0.23 | 1.00 ± 0.22 | 0.77 ± 0.17 | 0.98 ± 0.28 | 0.98 ± 0.28 |
| hip_shift: pelvic list amplitude ratio @pipeline bottom frame | 0.88 ± 0.18 | 0.96 ± 0.24 | 1.05 ± 0.39 | 0.58 ± 0.32 | 1.17 ± 0.53 | 1.18 ± 0.42 |
| asymmetric_depth: L-R knee asymmetry err @true bottom deg (truth ~ -8) | 0.50 ± 3.57 | -0.17 ± 3.14 | -0.34 ± 3.54 | 1.43 ± 1.78 | -0.57 ± 2.67 | 0.20 ± 1.81 |
| asymmetric_depth: L-R knee asymmetry err @pipeline bottom frame deg | 3.04 ± 4.15 | 0.76 ± 3.56 | 1.90 ± 2.71 | 1.08 ± 2.20 | -0.91 ± 4.18 | -0.85 ± 2.88 |
| clean: clean: false heel-rise amplitude mm (positions) @true bottom | 23.3 ± 20.6 | -1.8 ± 9.0 | -3.8 ± 7.0 | 11.0 ± 9.6 | -2.4 ± 10.8 | -0.7 ± 2.8 |
| clean: clean: FootState false heel rise cm @true bottom | - | - | - | - | - | 0.00 ± 0.00 |
| clean: clean: FootState false heel rise PEAK per rep cm | - | - | - | - | - | 0.00 ± 0.00 |
| clean: clean: true peak ankle rise per rep cm (should be ~0) | - | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| clean: clean: false hip-shift amplitude mm @true bottom | 4.6 ± 9.8 | 7.6 ± 18.8 | 3.5 ± 12.0 | -5.7 ± 16.5 | 6.1 ± 14.0 | 0.7 ± 3.7 |
| clean: clean: depth err final deg | -8.29 ± 3.35 | -3.46 ± 2.11 | -5.18 ± 2.74 | -4.69 ± 1.87 | 0.29 ± 0.84 | -0.14 ± 0.48 |
| stance_change: stance widening ratio (1 = preserved) | 0.99 ± 0.05 | 1.00 ± 0.03 | 0.98 ± 0.04 | -0.00 ± 0.03 | 0.97 ± 0.04 | 0.97 ± 0.05 |
| stance_change: stance width MAE mm | 28.1 ± 2.1 | 14.3 ± 2.0 | 10.4 ± 1.1 | 124.3 ± 2.3 | 11.5 ± 2.4 | 6.6 ± 0.3 |
| walkout: walkout MPJPE mm | 29.2 ± 1.1 | 21.6 ± 0.9 | 22.9 ± 1.4 | 37.4 ± 1.7 | 18.1 ± 0.6 | 21.2 ± 2.6 |
| walkout: walkout processed frames (readiness opened?) | 448 ± 76 | 464 ± 78 | 464 ± 78 | 464 ± 78 | 462 ± 77 | 462 ± 77 |
| walkout: walkout depth err final deg | -10.92 ± 0.46 | -3.15 ± 0.86 | -3.74 ± 0.92 | -3.86 ± 1.03 | -0.29 ± 0.05 | -0.31 ± 0.24 |
| walkout: walkout FootState valid frac | - | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.95 ± 0.01 |
| fast_reps: fast reps knee @true bottom err deg | -0.21 ± 0.55 | -0.17 ± 0.91 | -5.34 ± 1.47 | -8.87 ± 2.79 | -0.84 ± 0.63 | -0.61 ± 0.47 |
| fast_reps: fast reps depth err IK deg | -0.48 ± 0.86 | 0.77 ± 0.37 | -1.71 ± 0.33 | -3.06 ± 0.17 | 0.35 ± 0.31 | 0.13 ± 0.34 |
| fast_reps: fast reps depth err final deg | -13.78 ± 3.75 | -5.87 ± 1.14 | -8.40 ± 0.91 | -8.44 ± 0.78 | 0.35 ± 0.31 | 0.13 ± 0.34 |
| fast_reps: fast reps knee lag IK ms | 10.1 ± 15.7 | 5.6 ± 8.9 | 45.3 ± 12.0 | 86.9 ± 11.8 | 3.4 ± 3.5 | 2.9 ± 2.6 |

### Per scenario: MPJPE mm — `perfect` (mean ± std over seeds)

| scenario | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| clean | 35.4 ± 2.4 | 25.5 ± 1.5 | 31.0 ± 2.2 | 47.8 ± 2.8 | 21.7 ± 1.0 | 20.6 ± 1.5 |
| valgus | 34.2 ± 1.0 | 24.8 ± 1.1 | 28.3 ± 0.7 | 45.6 ± 0.4 | 20.7 ± 0.9 | 18.5 ± 0.9 |
| heel_rise | 35.5 ± 1.3 | 26.5 ± 0.9 | 28.6 ± 1.7 | 44.0 ± 3.7 | 22.6 ± 1.0 | 20.5 ± 0.6 |
| hip_shift | 33.6 ± 1.4 | 25.2 ± 0.7 | 28.6 ± 2.2 | 43.6 ± 3.3 | 21.2 ± 0.5 | 19.8 ± 0.3 |
| asymmetric_depth | 35.3 ± 2.2 | 26.2 ± 0.5 | 27.0 ± 0.8 | 41.9 ± 1.1 | 21.8 ± 0.2 | 20.4 ± 0.6 |
| fast_reps | 31.8 ± 3.1 | 24.3 ± 2.5 | 40.0 ± 5.3 | 62.9 ± 5.2 | 20.5 ± 2.4 | 19.9 ± 2.3 |
| stance_change | 35.6 ± 3.5 | 26.1 ± 1.6 | 26.9 ± 1.7 | 46.4 ± 1.5 | 22.5 ± 1.6 | 20.3 ± 1.2 |
| walkout | 29.2 ± 1.1 | 21.6 ± 0.9 | 22.9 ± 1.4 | 37.4 ± 1.7 | 18.1 ± 0.6 | 21.2 ± 2.6 |

### Per scenario: knee@true-bottom err deg — `perfect` (mean ± std over seeds)

| scenario | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.5 ± 1.1 | -0.5 ± 1.3 | -2.1 ± 1.9 | -3.7 ± 1.2 | -0.4 ± 1.0 | -0.9 ± 0.7 |
| valgus | 1.2 ± 0.5 | 0.7 ± 1.4 | -0.1 ± 1.2 | -4.6 ± 2.0 | 0.9 ± 0.8 | 1.0 ± 0.3 |
| heel_rise | 0.1 ± 0.6 | 0.3 ± 0.4 | -2.0 ± 0.7 | -3.9 ± 1.3 | 0.6 ± 0.4 | 0.4 ± 0.1 |
| hip_shift | -0.4 ± 0.8 | -0.4 ± 1.1 | -2.3 ± 1.1 | -4.4 ± 1.7 | -0.3 ± 1.1 | -1.3 ± 1.1 |
| asymmetric_depth | -0.3 ± 1.1 | -0.3 ± 0.4 | -1.5 ± 0.6 | -3.9 ± 0.9 | -1.1 ± 0.6 | -0.6 ± 1.6 |
| fast_reps | -0.2 ± 0.6 | -0.2 ± 0.9 | -5.3 ± 1.5 | -8.9 ± 2.8 | -0.8 ± 0.6 | -0.6 ± 0.5 |
| stance_change | 0.7 ± 1.2 | 0.5 ± 0.8 | -0.8 ± 1.0 | -0.1 ± 1.8 | 0.6 ± 1.3 | 0.9 ± 1.0 |
| walkout | -0.6 ± 0.4 | -0.4 ± 0.4 | -1.5 ± 0.0 | -2.7 ± 0.6 | -0.5 ± 0.5 | -0.8 ± 0.6 |

### Per scenario: depth err final deg — `perfect` (mean ± std over seeds)

| scenario | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| clean | -8.3 ± 3.4 | -3.5 ± 2.1 | -5.2 ± 2.7 | -4.7 ± 1.9 | 0.3 ± 0.8 | -0.1 ± 0.5 |
| valgus | -8.4 ± 4.3 | -3.2 ± 1.6 | -4.7 ± 2.0 | -7.2 ± 1.9 | 1.1 ± 1.1 | 1.6 ± 0.4 |
| heel_rise | -9.0 ± 2.1 | -2.5 ± 0.9 | -4.8 ± 3.2 | -5.2 ± 2.9 | 1.4 ± 0.4 | 0.8 ± 0.7 |
| hip_shift | -7.6 ± 3.3 | -4.3 ± 1.5 | -5.3 ± 1.1 | -6.6 ± 2.6 | 0.9 ± 1.4 | 0.0 ± 0.8 |
| asymmetric_depth | -10.4 ± 1.5 | -3.7 ± 0.9 | -4.9 ± 1.8 | -5.5 ± 0.4 | 0.9 ± 0.7 | 1.0 ± 0.9 |
| fast_reps | -13.8 ± 3.7 | -5.9 ± 1.1 | -8.4 ± 0.9 | -8.4 ± 0.8 | 0.3 ± 0.3 | 0.1 ± 0.3 |
| stance_change | -7.4 ± 2.2 | -2.3 ± 0.1 | -2.9 ± 0.2 | -0.9 ± 1.3 | 3.9 ± 4.3 | 3.5 ± 2.9 |
| walkout | -10.9 ± 0.5 | -3.2 ± 0.9 | -3.7 ± 0.9 | -3.9 ± 1.0 | -0.3 ± 0.1 | -0.3 ± 0.2 |

### Per scenario: knee NaN/0 frac — `perfect` (mean ± std over seeds)

| scenario | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| clean | - | 0.038 ± 0.034 | 0.038 ± 0.034 | 0.038 ± 0.034 | 0.001 ± 0.002 | 0.001 ± 0.002 |
| valgus | - | 0.044 ± 0.031 | 0.044 ± 0.031 | 0.044 ± 0.031 | 0.004 ± 0.006 | 0.004 ± 0.006 |
| heel_rise | - | 0.036 ± 0.026 | 0.036 ± 0.026 | 0.036 ± 0.026 | 0.003 ± 0.005 | 0.003 ± 0.005 |
| hip_shift | - | 0.058 ± 0.041 | 0.058 ± 0.041 | 0.058 ± 0.041 | 0.001 ± 0.002 | 0.001 ± 0.002 |
| asymmetric_depth | - | 0.044 ± 0.038 | 0.044 ± 0.038 | 0.044 ± 0.038 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| fast_reps | - | 0.037 ± 0.044 | 0.037 ± 0.044 | 0.037 ± 0.044 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| stance_change | - | 0.048 ± 0.030 | 0.048 ± 0.030 | 0.048 ± 0.030 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| walkout | - | 0.013 ± 0.004 | 0.013 ± 0.004 | 0.013 ± 0.004 | 0.000 ± 0.000 | 0.000 ± 0.000 |

### Gates / foot model / compute (mean over scenarios) — `perfect`

| chain | readiness gate s | FootState valid s | FootState valid frac | predicted frac | chain us | kalman us | foot us | recentre+Skeleton3D us | frame us (chain+IK+valgus[+filter]) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw (old triangulator) | 0.58 | - | - | - | 45 | - | - | - | 394 |
| raw | 0.36 | - | 0.00 | 0.000 | 48 | - | - | - | 453 |
| current | 0.36 | - | 0.00 | 0.000 | 138 | - | - | - | 546 |
| full_original | 0.36 | 1.56 | 0.00 | 0.000 | 589 | - | - | - | 1015 |
| proposed_nofoot | 0.36 | - | 0.00 | 0.000 | 231 | 87 | 0 | 73 | 617 |
| proposed | 0.36 | 1.20 | 0.93 | 0.000 | 379 | 96 | 129 | 79 | 796 |

## Overall (mean over 8 scenarios x 3 seeds) — calibration `tpose`

| chain | MPJPE mm | lower mm | p95 mm | still jitter p50 mm | knee lag IK ms | knee@true-bottom err deg | knee MAE moving deg | depth err IK deg | depth err final deg | bottom-frame delay ms (aligned) | bottom-frame delay ms (wall) | analysis latency ms | knee NaN/0 frac | stance MAE mm | chain us | frame us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw (old triangulator) | 43.9 | 49.1 | 114.0 | 35.0 | 6 | -0.1 | 16.6 | -5.1 | -15.2 | 98 | - | - | - | 32.9 | 50 | 416 |
| raw | 35.9 | 37.1 | 73.0 | 32.0 | -0 | 0.1 | 6.5 | 3.1 | -4.2 | 94 | 94 | 0 | 0.071 | 23.0 | 49 | 459 |
| current | 53.8 | 50.5 | 157.6 | 8.5 | 92 | -13.4 | 16.1 | -7.2 | -12.9 | 233 | 233 | 0 | 0.071 | 20.6 | 139 | 548 |
| full_original | 72.3 | 68.5 | 209.9 | 1.2 | 151 | -16.6 | 22.2 | -8.9 | -13.7 | 311 | 311 | 0 | 0.071 | 35.1 | 594 | 1019 |
| proposed_nofoot | 32.3 | 32.9 | 66.9 | 3.6 | 1 | 0.3 | 2.8 | 2.3 | 2.3 | 8 | 91 | 82 | 0.002 | 21.5 | 234 | 614 |
| proposed | 31.2 | 30.4 | 63.9 | 2.5 | 1 | 0.0 | 2.6 | 1.6 | 1.6 | 4 | 89 | 82 | 0.002 | 20.4 | 331 | 695 |

### Paired delta vs `raw` (new triangulator), mean ± std over 8 scenarios x seeds — `tpose`

| chain | mpjpe_mm | mpjpe_lower_mm | p95_err_mm | jitter_still_p50_mm | lag_knee_flex_ik_ms | knee_flex_tb_err_deg | knee_flex_ik_moving_mae_deg | depth_err_ik_deg | depth_err_final_deg | valgus_tb_err_deg | stance_width_mae_mm | ik_knee_invalid_frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 17.9 ± 8.9 | 13.5 ± 6.9 | 84.6 ± 37.4 | -23.5 ± 3.5 | 91.7 ± 30.7 | -13.5 ± 8.6 | 9.6 ± 3.7 | -10.3 ± 5.7 | -8.8 ± 4.7 | -0.3 ± 1.8 | -2.4 ± 2.7 | 0.000 ± 0.000 |
| full_original | 36.4 ± 11.1 | 31.4 ± 9.5 | 136.9 ± 39.5 | -30.8 ± 2.5 | 151.1 ± 35.6 | -16.8 ± 8.8 | 15.7 ± 3.5 | -12.0 ± 4.5 | -9.5 ± 3.9 | -0.2 ± 2.9 | 12.1 ± 41.2 | 0.000 ± 0.000 |
| proposed_nofoot | -3.6 ± 0.6 | -4.1 ± 1.1 | -6.0 ± 1.0 | -28.4 ± 2.6 | 1.6 ± 6.2 | 0.2 ± 0.8 | -3.7 ± 3.2 | -0.9 ± 1.7 | 6.5 ± 3.3 | -0.1 ± 1.1 | -1.5 ± 2.2 | -0.069 ± 0.073 |
| proposed | -4.7 ± 1.1 | -6.7 ± 2.9 | -9.1 ± 1.9 | -29.5 ± 2.5 | 1.4 ± 6.2 | -0.1 ± 1.0 | -3.9 ± 3.2 | -1.5 ± 2.1 | 5.8 ± 3.6 | 0.0 ± 1.2 | -2.6 ± 5.8 | -0.069 ± 0.073 |

### Fault-signal preservation and scenario-specific metrics — `tpose` (mean ± std over seeds)

| scenario / metric | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| valgus: valgus amplitude ratio @true bottom (same estimator on truth; 1 = preserved) | 1.02 ± 0.05 | 0.89 ± 0.29 | 0.61 ± 0.13 | 0.48 ± 0.16 | 0.87 ± 0.08 | 1.02 ± 0.03 |
| valgus: valgus amplitude ratio @pipeline bottom frame | 0.80 ± 0.14 | 0.78 ± 0.09 | 0.65 ± 0.02 | 0.60 ± 0.08 | 0.77 ± 0.20 | 1.04 ± 0.05 |
| valgus: valgus @true bottom err deg | 0.88 ± 0.72 | -3.50 ± 1.46 | -6.93 ± 0.87 | -9.04 ± 2.41 | -3.76 ± 1.07 | -2.06 ± 0.71 |
| valgus: valgus @pipeline bottom frame err deg | -2.91 ± 1.67 | -4.65 ± 1.75 | -7.62 ± 1.31 | -8.23 ± 2.33 | -4.51 ± 1.67 | -1.14 ± 0.48 |
| heel_rise: heel rise (ankle over toe, positions) amplitude ratio @true bottom | 0.37 ± 0.40 | 1.03 ± 0.48 | 0.65 ± 0.62 | 0.98 ± 0.20 | 0.96 ± 0.55 | 0.38 ± 0.23 |
| heel_rise: heel rise amplitude ratio @pipeline bottom frame | 0.24 ± 0.31 | 1.12 ± 0.42 | 0.71 ± 0.51 | 0.80 ± 0.35 | 0.92 ± 0.35 | 0.29 ± 0.30 |
| heel_rise: FootState heel_rise_cm / true ankle rise @true bottom | - | - | - | - | - | 0.45 ± 0.15 |
| heel_rise: heel rise PEAK per rep (positions) / true peak ankle rise | - | 5.10 ± 4.26 | 1.87 ± 0.73 | 1.42 ± 0.23 | 1.35 ± 0.29 | 0.92 ± 0.30 |
| heel_rise: FootState heel rise PEAK per rep / true peak ankle rise | - | - | - | - | - | 0.82 ± 0.33 |
| heel_rise: FootState heel rise PEAK per rep cm | - | - | - | - | - | 1.82 ± 0.74 |
| heel_rise: true peak ankle rise per rep cm (harness GT) | - | 2.22 ± 0.00 | 2.22 ± 0.00 | 2.22 ± 0.00 | 2.22 ± 0.00 | 2.22 ± 0.00 |
| hip_shift: lateral hip shift amplitude ratio @true bottom | 1.12 ± 0.30 | 1.14 ± 0.32 | 0.99 ± 0.34 | 0.84 ± 0.20 | 1.29 ± 0.60 | 0.92 ± 0.18 |
| hip_shift: lateral hip shift amplitude ratio @pipeline bottom frame | 0.96 ± 0.44 | 0.92 ± 0.36 | 0.98 ± 0.25 | 0.93 ± 0.16 | 1.37 ± 0.70 | 0.94 ± 0.18 |
| hip_shift: pelvic list amplitude ratio @true bottom | 0.92 ± 0.34 | 0.97 ± 0.24 | 0.77 ± 0.19 | 0.83 ± 0.21 | 0.90 ± 0.25 | 0.90 ± 0.25 |
| hip_shift: pelvic list amplitude ratio @pipeline bottom frame | 0.78 ± 0.19 | 0.86 ± 0.34 | 0.82 ± 0.35 | 0.64 ± 0.38 | 0.94 ± 0.32 | 1.04 ± 0.08 |
| asymmetric_depth: L-R knee asymmetry err @true bottom deg (truth ~ -8) | -0.02 ± 4.17 | 0.78 ± 3.74 | 3.80 ± 4.01 | 0.29 ± 2.23 | 4.92 ± 1.37 | -0.43 ± 3.74 |
| asymmetric_depth: L-R knee asymmetry err @pipeline bottom frame deg | 3.30 ± 5.71 | 3.77 ± 2.80 | 3.33 ± 4.72 | 0.61 ± 1.65 | 2.46 ± 3.05 | 0.00 ± 7.00 |
| clean: clean: false heel-rise amplitude mm (positions) @true bottom | 29.8 ± 20.2 | 0.2 ± 9.8 | -11.5 ± 5.3 | -20.3 ± 37.1 | -1.5 ± 7.0 | 0.6 ± 2.1 |
| clean: clean: FootState false heel rise cm @true bottom | - | - | - | - | - | 0.00 ± 0.00 |
| clean: clean: FootState false heel rise PEAK per rep cm | - | - | - | - | - | 0.00 ± 0.00 |
| clean: clean: true peak ankle rise per rep cm (should be ~0) | - | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| clean: clean: false hip-shift amplitude mm @true bottom | 9.3 ± 4.2 | 7.5 ± 15.5 | 2.6 ± 4.2 | -12.7 ± 17.6 | 8.9 ± 14.2 | 1.0 ± 1.8 |
| clean: clean: depth err final deg | -10.42 ± 4.63 | -4.27 ± 2.17 | -13.89 ± 6.69 | -13.02 ± 3.91 | 0.73 ± 2.01 | 0.74 ± 0.88 |
| stance_change: stance widening ratio (1 = preserved) | 0.92 ± 0.01 | 0.96 ± 0.04 | 0.95 ± 0.03 | 0.02 ± 0.04 | 0.94 ± 0.04 | 0.90 ± 0.04 |
| stance_change: stance width MAE mm | 37.0 ± 8.7 | 27.4 ± 9.7 | 26.1 ± 9.2 | 146.8 ± 5.2 | 26.8 ± 10.5 | 24.1 ± 8.3 |
| walkout: walkout MPJPE mm | 37.2 ± 1.0 | 35.9 ± 5.0 | 60.2 ± 6.9 | 79.1 ± 8.3 | 31.6 ± 4.3 | 32.9 ± 3.6 |
| walkout: walkout processed frames (readiness opened?) | 294 ± 224 | 445 ± 80 | 445 ± 80 | 445 ± 80 | 444 ± 78 | 444 ± 78 |
| walkout: walkout depth err final deg | -38.51 ± 24.61 | -6.04 ± 4.84 | -17.85 ± 9.66 | -17.88 ± 9.01 | 1.99 ± 0.05 | 2.12 ± 0.07 |
| walkout: walkout FootState valid frac | - | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.97 ± 0.02 |
| fast_reps: fast reps knee @true bottom err deg | -0.49 ± 0.37 | -0.61 ± 0.32 | -22.76 ± 9.57 | -26.90 ± 7.15 | -1.33 ± 0.40 | -1.36 ± 0.97 |
| fast_reps: fast reps depth err IK deg | -7.48 ± 9.20 | 0.41 ± 0.25 | -14.32 ± 8.31 | -14.68 ± 6.35 | -0.07 ± 0.93 | -0.54 ± 1.07 |
| fast_reps: fast reps depth err final deg | -20.35 ± 7.39 | -8.28 ± 2.93 | -21.52 ± 7.83 | -20.70 ± 5.79 | -0.07 ± 0.93 | -0.54 ± 1.07 |
| fast_reps: fast reps knee lag IK ms | 7.7 ± 10.8 | 5.1 ± 8.9 | 80.9 ± 28.8 | 128.2 ± 33.1 | 4.8 ± 4.1 | 4.6 ± 4.7 |

### Per scenario: MPJPE mm — `tpose` (mean ± std over seeds)

| scenario | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| clean | 43.9 ± 4.8 | 35.6 ± 3.5 | 55.1 ± 8.2 | 72.9 ± 8.9 | 32.3 ± 3.2 | 30.9 ± 3.5 |
| valgus | 44.4 ± 1.3 | 34.7 ± 2.7 | 50.4 ± 5.4 | 68.2 ± 5.9 | 30.8 ± 3.1 | 29.3 ± 3.1 |
| heel_rise | 46.8 ± 3.1 | 36.5 ± 1.9 | 50.5 ± 9.0 | 66.6 ± 10.9 | 32.7 ± 1.4 | 32.4 ± 2.9 |
| hip_shift | 45.3 ± 5.3 | 36.0 ± 3.4 | 50.2 ± 8.7 | 66.0 ± 8.7 | 32.6 ± 3.3 | 31.1 ± 3.0 |
| asymmetric_depth | 45.3 ± 5.4 | 36.3 ± 4.2 | 49.7 ± 10.9 | 67.5 ± 13.9 | 32.7 ± 3.4 | 30.8 ± 3.9 |
| fast_reps | 41.0 ± 4.5 | 34.7 ± 5.2 | 66.3 ± 15.9 | 89.9 ± 16.6 | 31.4 ± 5.3 | 29.9 ± 5.3 |
| stance_change | 47.4 ± 5.9 | 37.4 ± 5.5 | 47.9 ± 8.0 | 67.9 ± 6.9 | 33.9 ± 5.5 | 32.2 ± 5.2 |
| walkout | 37.2 ± 1.0 | 35.9 ± 5.0 | 60.2 ± 6.9 | 79.1 ± 8.3 | 31.6 ± 4.3 | 32.9 ± 3.6 |

### Per scenario: knee@true-bottom err deg — `tpose` (mean ± std over seeds)

| scenario | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.2 ± 1.0 | -0.3 ± 1.2 | -15.8 ± 8.5 | -16.5 ± 3.2 | -0.6 ± 1.9 | -1.1 ± 1.6 |
| valgus | 1.2 ± 0.9 | 1.3 ± 0.6 | -10.0 ± 2.9 | -15.0 ± 4.3 | 1.2 ± 0.5 | 1.3 ± 1.2 |
| heel_rise | -0.3 ± 0.7 | -0.4 ± 1.0 | -13.1 ± 9.9 | -17.1 ± 9.1 | -0.3 ± 1.6 | -0.3 ± 1.8 |
| hip_shift | -0.2 ± 1.8 | -0.3 ± 1.8 | -9.7 ± 4.2 | -15.2 ± 4.5 | -0.2 ± 1.8 | -0.0 ± 1.9 |
| asymmetric_depth | -0.4 ± 0.7 | 0.1 ± 0.5 | -11.3 ± 5.1 | -12.6 ± 5.3 | 1.0 ± 0.9 | 0.6 ± 1.6 |
| fast_reps | -0.5 ± 0.4 | -0.6 ± 0.3 | -22.8 ± 9.6 | -26.9 ± 7.1 | -1.3 ± 0.4 | -1.4 ± 1.0 |
| stance_change | -0.1 ± 0.8 | 0.9 ± 0.8 | -7.3 ± 4.1 | -10.0 ± 4.1 | 1.6 ± 0.9 | 0.3 ± 1.1 |
| walkout | -0.5 ± 0.0 | 0.5 ± 0.5 | -17.2 ± 12.0 | -19.8 ± 15.6 | 1.2 ± 0.6 | 0.8 ± 0.5 |

### Per scenario: depth err final deg — `tpose` (mean ± std over seeds)

| scenario | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| clean | -10.4 ± 4.6 | -4.3 ± 2.2 | -13.9 ± 6.7 | -13.0 ± 3.9 | 0.7 ± 2.0 | 0.7 ± 0.9 |
| valgus | -7.9 ± 3.7 | -2.5 ± 0.9 | -10.5 ± 3.0 | -13.5 ± 4.8 | 1.9 ± 0.6 | 1.8 ± 1.0 |
| heel_rise | -12.9 ± 6.3 | -5.0 ± 5.5 | -13.3 ± 9.9 | -14.7 ± 7.5 | 1.5 ± 3.1 | 0.7 ± 3.0 |
| hip_shift | -14.3 ± 12.5 | -4.1 ± 2.0 | -10.2 ± 4.6 | -12.5 ± 4.4 | 1.9 ± 2.5 | 1.6 ± 2.2 |
| asymmetric_depth | -10.3 ± 0.7 | -2.3 ± 2.8 | -9.2 ± 4.1 | -10.0 ± 2.5 | 4.0 ± 1.4 | 2.1 ± 1.8 |
| fast_reps | -20.3 ± 7.4 | -8.3 ± 2.9 | -21.5 ± 7.8 | -20.7 ± 5.8 | -0.1 ± 0.9 | -0.5 ± 1.1 |
| stance_change | -7.4 ± 2.1 | -1.0 ± 1.1 | -6.9 ± 3.6 | -7.2 ± 4.2 | 6.2 ± 2.2 | 4.3 ± 3.3 |
| walkout | -38.5 ± 24.6 | -6.0 ± 4.8 | -17.8 ± 9.7 | -17.9 ± 9.0 | 2.0 ± 0.1 | 2.1 ± 0.1 |

### Per scenario: knee NaN/0 frac — `tpose` (mean ± std over seeds)

| scenario | raw (old triangulator) | raw | current | full_original | proposed_nofoot | proposed |
|---|---:|---:|---:|---:|---:|---:|
| clean | - | 0.064 ± 0.035 | 0.064 ± 0.035 | 0.064 ± 0.035 | 0.002 ± 0.003 | 0.002 ± 0.003 |
| valgus | - | 0.049 ± 0.037 | 0.049 ± 0.037 | 0.049 ± 0.037 | 0.004 ± 0.006 | 0.004 ± 0.006 |
| heel_rise | - | 0.061 ± 0.043 | 0.061 ± 0.043 | 0.061 ± 0.043 | 0.003 ± 0.003 | 0.003 ± 0.003 |
| hip_shift | - | 0.071 ± 0.061 | 0.071 ± 0.061 | 0.071 ± 0.061 | 0.002 ± 0.001 | 0.002 ± 0.001 |
| asymmetric_depth | - | 0.059 ± 0.067 | 0.059 ± 0.067 | 0.059 ± 0.067 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| fast_reps | - | 0.071 ± 0.074 | 0.071 ± 0.074 | 0.071 ± 0.074 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| stance_change | - | 0.058 ± 0.055 | 0.058 ± 0.055 | 0.058 ± 0.055 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| walkout | - | 0.135 ± 0.129 | 0.135 ± 0.129 | 0.135 ± 0.129 | 0.005 ± 0.004 | 0.005 ± 0.004 |

### Gates / foot model / compute (mean over scenarios) — `tpose`

| chain | readiness gate s | FootState valid s | FootState valid frac | predicted frac | chain us | kalman us | foot us | recentre+Skeleton3D us | frame us (chain+IK+valgus[+filter]) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw (old triangulator) | 1.89 | - | - | - | 50 | - | - | - | 416 |
| raw | 0.59 | - | 0.00 | 0.000 | 49 | - | - | - | 459 |
| current | 0.59 | - | 0.00 | 0.000 | 139 | - | - | - | 548 |
| full_original | 0.59 | 1.81 | 0.00 | 0.000 | 594 | - | - | - | 1019 |
| proposed_nofoot | 0.59 | - | 0.00 | 0.000 | 234 | 87 | 0 | 77 | 614 |
| proposed | 0.59 | 1.32 | 0.94 | 0.000 | 331 | 82 | 114 | 70 | 695 |

## Input stats on the new triangulator (per scenario, mean over seeds)

| calibration | scenario | loop_hz | sync_fail_frac | triangulation_none_frac | mean_reprojection_px | tri_lower_conf_p50 | tri_lower_conf_below_0.1_frac | tri_lower_conf_zero_frac | tri_swap_count |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perfect | clean | 28.42 | 0.1835 | 0.0000 | 4.84 | 0.5142 | 0.0110 | 0.0013 | 2.00 |
| perfect | valgus | 28.43 | 0.2156 | 0.0000 | 4.78 | 0.5303 | 0.0137 | 0.0014 | 0.33 |
| perfect | heel_rise | 28.41 | 0.1292 | 0.0000 | 4.78 | 0.5145 | 0.0095 | 0.0009 | 4.33 |
| perfect | hip_shift | 28.41 | 0.1491 | 0.0000 | 4.86 | 0.5217 | 0.0121 | 0.0010 | 0.33 |
| perfect | asymmetric_depth | 28.41 | 0.0731 | 0.0000 | 4.89 | 0.5273 | 0.0130 | 0.0008 | 5.67 |
| perfect | fast_reps | 28.46 | 0.0148 | 0.0000 | 4.85 | 0.5159 | 0.0125 | 0.0014 | 2.33 |
| perfect | stance_change | 28.39 | 0.1277 | 0.0000 | 4.94 | 0.5149 | 0.0123 | 0.0011 | 4.67 |
| perfect | walkout | 28.39 | 0.0936 | 0.0000 | 5.01 | 0.5797 | 0.0069 | 0.0015 | 1.33 |
| tpose | clean | 28.42 | 0.1835 | 0.0000 | 6.47 | 0.3846 | 0.0154 | 0.0019 | 0.00 |
| tpose | valgus | 28.43 | 0.2156 | 0.0000 | 6.36 | 0.4018 | 0.0153 | 0.0012 | 0.00 |
| tpose | heel_rise | 28.41 | 0.1292 | 0.0000 | 6.48 | 0.3799 | 0.0152 | 0.0018 | 0.00 |
| tpose | hip_shift | 28.41 | 0.1491 | 0.0000 | 6.38 | 0.3953 | 0.0156 | 0.0014 | 0.00 |
| tpose | asymmetric_depth | 28.41 | 0.0731 | 0.0000 | 6.66 | 0.3835 | 0.0168 | 0.0010 | 0.33 |
| tpose | fast_reps | 28.46 | 0.0148 | 0.0000 | 6.26 | 0.4058 | 0.0195 | 0.0020 | 0.00 |
| tpose | stance_change | 28.39 | 0.1277 | 0.0000 | 6.58 | 0.3907 | 0.0188 | 0.0019 | 0.00 |
| tpose | walkout | 28.39 | 0.0936 | 0.0000 | 12.00 | 0.2714 | 0.0319 | 0.0062 | 0.00 |

## Noise profile `harsh` (calibration `perfect`, mean over 8 scenarios x seeds)

| chain | MPJPE mm | p95 mm | still jitter p50 mm | knee@true-bottom err deg | knee MAE moving deg | depth err IK deg | depth err final deg | valgus@true-bottom err deg | knee NaN/0 frac | predicted frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw (old triangulator) | 75.0 | 325.2 | 62.4 | 0.3 | 37.0 | -12.1 | -33.6 | 0.1 | - | - |
| raw | 43.6 | 81.2 | 50.0 | -0.4 | 11.0 | 1.8 | -11.0 | 0.6 | 0.132 | 0.000 |
| proposed | 32.2 | 60.2 | 3.3 | 0.8 | 4.0 | 0.6 | 0.6 | 3.2 | 0.006 | 0.000 |

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
