# Pre-IK baseline — triangulated 3-camera harness

Generated 2026-09-16 19:42:40; 384 chain runs; wall 47 s; seeds [0, 1, 2]; noise profile `realistic`; scenarios: clean, valgus, heel_rise, hip_shift, asymmetric_depth, fast_reps, stance_change, walkout.
All chains see identical inputs per (scenario, seed, calibration): comparisons are paired.
'IK' = AnalyticalIKSolver on the chain output; 'final' = after JointAngleFilter (what rules/bottom buffer consume). Depth err = measured max knee flexion per rep - true max (negative = undershoot).

## Overall (mean over 8 scenarios) — calibration `perfect`

| chain | MPJPE mm | lower mm | p95 mm | bone std mm | still jitter p50 mm | lag knee-y ms | lag knee IK ms | lag knee final ms | knee@true-bottom err IK deg | depth err IK deg | depth err final deg | bottom-frame delay ms | knee MAE moving IK deg | valgus@true-bottom err IK deg | stance MAE mm | IK knee=0 frac | chain us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 33.8 | 42.2 | 98.7 | 36.1 | 34.9 | 0.1 | 3.9 | 67.5 | 0.1 | 0.9 | -9.5 | 71.1 | 14.3 | 0.1 | 26.7 | 0.25 | 45.1 |
| current | 35.8 | 36.4 | 106.2 | 18.6 | 9.5 | 51.5 | 54.3 | 123.8 | -2.9 | -1.8 | -11.2 | 146.2 | 17.8 | -0.8 | 9.9 | 0.25 | 127.7 |
| full_original | 53.5 | 52.7 | 160.9 | 5.5 | 1.5 | 104.2 | 111.9 | 179.6 | -6.0 | -2.8 | -11.1 | 213.7 | 23.0 | -0.5 | 20.7 | 0.25 | 512.4 |
| blend_only | 34.7 | 36.0 | 99.2 | 17.8 | 9.5 | 50.0 | 52.8 | 122.0 | -2.8 | -1.7 | -11.1 | 146.2 | 17.6 | -0.7 | 9.9 | 0.25 | 98.0 |
| vclamp_only | 31.8 | 37.0 | 91.7 | 28.3 | 33.8 | 3.8 | 6.5 | 71.2 | -0.1 | 0.8 | -9.6 | 77.2 | 14.6 | 0.1 | 21.0 | 0.25 | 92.2 |
| bone_only | 34.5 | 43.4 | 98.4 | 31.7 | 34.9 | 0.0 | 4.0 | 68.9 | -0.2 | 0.9 | -9.6 | 77.3 | 14.1 | -0.2 | 26.6 | 0.25 | 139.6 |
| ground_only | 35.4 | 45.9 | 102.9 | 36.4 | 35.1 | 0.1 | 3.8 | 68.1 | 0.5 | 1.3 | -9.1 | 70.9 | 14.6 | -0.5 | 38.7 | 0.25 | 91.0 |
| smooth_only | 39.7 | 45.6 | 101.3 | 29.9 | 4.6 | 48.9 | 53.0 | 116.1 | -1.8 | -0.4 | -9.9 | 148.1 | 17.9 | -0.4 | 24.0 | 0.25 | 185.0 |

### Paired delta vs raw (mean ± std over 8 scenarios x seeds) — `perfect`

| chain | mpjpe_mm | mpjpe_lower_mm | p95_err_mm | bone_len_std_mm | jitter_still_p50_mm | lag_knee_flex_ik_ms | knee_flex_tb_err_deg | depth_err_ik_deg | depth_err_final_deg | knee_flex_ik_moving_mae_deg | valgus_tb_err_deg | stance_width_mae_mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 2.0 ± 5.7 | -5.8 ± 4.9 | 7.5 ± 34.3 | -17.5 ± 4.7 | -25.5 ± 2.4 | 50.4 ± 8.5 | -3.0 ± 2.1 | -2.7 ± 1.1 | -1.7 ± 2.0 | 3.5 ± 1.6 | -0.9 ± 1.0 | -16.8 ± 3.8 |
| full_original | 19.7 ± 7.8 | 10.5 ± 6.8 | 62.3 ± 41.4 | -30.6 ± 4.3 | -33.4 ± 2.6 | 108.0 ± 14.9 | -6.1 ± 3.4 | -3.8 ± 1.7 | -1.7 ± 3.2 | 8.7 ± 2.5 | -0.6 ± 2.8 | -6.0 ± 32.6 |
| blend_only | 0.9 ± 4.5 | -6.2 ± 4.5 | 0.6 ± 26.8 | -18.4 ± 4.2 | -25.5 ± 2.4 | 48.8 ± 8.1 | -2.9 ± 2.0 | -2.7 ± 1.1 | -1.7 ± 2.0 | 3.4 ± 1.3 | -0.8 ± 1.0 | -16.8 ± 3.8 |
| vclamp_only | -2.0 ± 2.4 | -5.2 ± 1.7 | -7.0 ± 22.2 | -7.9 ± 2.4 | -1.2 ± 0.8 | 2.6 ± 3.7 | -0.2 ± 0.7 | -0.1 ± 0.6 | -0.2 ± 0.6 | 0.3 ± 0.4 | -0.0 ± 0.6 | -5.8 ± 2.6 |
| bone_only | 0.7 ± 0.4 | 1.2 ± 0.9 | -0.2 ± 2.0 | -4.4 ± 0.6 | -0.1 ± 0.8 | 0.1 ± 1.5 | -0.3 ± 1.4 | 0.0 ± 1.3 | -0.1 ± 1.2 | -0.1 ± 0.2 | -0.3 ± 2.6 | -0.1 ± 0.3 |
| ground_only | 1.6 ± 3.5 | 3.8 ± 8.3 | 4.3 ± 13.0 | 0.3 ± 1.4 | 0.2 ± 1.6 | -0.1 ± 0.6 | 0.4 ± 1.1 | 0.4 ± 1.1 | 0.4 ± 1.4 | 0.3 ± 0.7 | -0.6 ± 1.4 | 11.9 ± 35.5 |
| smooth_only | 5.9 ± 3.2 | 3.4 ± 2.2 | 2.7 ± 19.6 | -6.2 ± 1.0 | -30.3 ± 2.3 | 49.1 ± 5.8 | -1.9 ± 1.7 | -1.4 ± 1.5 | -0.4 ± 1.8 | 3.6 ± 1.2 | -0.5 ± 1.4 | -2.8 ± 1.7 |

### Fault-signal preservation and scenario-specific metrics — `perfect` (mean ± std over seeds)

| scenario / metric | raw | current | full_original | blend_only | vclamp_only | bone_only | ground_only | smooth_only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valgus: valgus amplitude ratio @true bottom, IK (1 = preserved) | 1.02 ± 0.08 | 0.99 ± 0.06 | 0.90 ± 0.06 | 0.99 ± 0.06 | 1.04 ± 0.09 | 1.01 ± 0.05 | 1.00 ± 0.06 | 0.96 ± 0.07 |
| valgus: valgus amplitude ratio @pipeline bottom frame, final | 0.78 ± 0.11 | 0.78 ± 0.10 | 0.76 ± 0.10 | 0.78 ± 0.10 | 0.82 ± 0.11 | 0.80 ± 0.07 | 0.76 ± 0.10 | 0.75 ± 0.10 |
| valgus: valgus @true bottom err deg (IK) | 0.62 ± 0.78 | -0.26 ± 0.71 | -1.44 ± 0.76 | -0.26 ± 0.71 | 0.77 ± 1.34 | 0.34 ± 0.29 | 0.08 ± 0.16 | -0.23 ± 1.25 |
| heel_rise: heel rise (ankle over toe) amplitude ratio @true bottom | 0.67 ± 0.17 | 1.74 ± 0.52 | 3.95 ± 2.96 | 1.79 ± 0.46 | 0.14 ± 0.67 | 0.45 ± 0.29 | 0.67 ± 0.17 | 0.94 ± 0.77 |
| heel_rise: heel rise amplitude ratio @pipeline bottom frame | 0.42 ± 0.14 | 0.90 ± 0.53 | 3.15 ± 2.60 | 0.90 ± 0.53 | 0.49 ± 0.27 | 0.68 ± 0.52 | 0.42 ± 0.14 | 1.32 ± 1.58 |
| hip_shift: lateral hip shift amplitude ratio @true bottom | 1.10 ± 0.19 | 1.01 ± 0.19 | 1.01 ± 0.07 | 1.02 ± 0.20 | 1.11 ± 0.23 | 1.11 ± 0.14 | 1.10 ± 0.19 | 0.66 ± 0.36 |
| hip_shift: lateral hip shift amplitude ratio @pipeline bottom frame | 0.96 ± 0.32 | 1.01 ± 0.24 | 1.02 ± 0.08 | 1.01 ± 0.24 | 0.94 ± 0.29 | 0.97 ± 0.16 | 0.96 ± 0.32 | 0.70 ± 0.18 |
| hip_shift: pelvic list amplitude ratio @true bottom, IK | 1.05 ± 0.29 | 1.04 ± 0.23 | 0.68 ± 0.15 | 1.04 ± 0.23 | 1.05 ± 0.29 | 0.43 ± 0.33 | 1.05 ± 0.29 | 0.98 ± 0.21 |
| hip_shift: pelvic list amplitude ratio @pipeline bottom frame, final | 0.88 ± 0.18 | 0.83 ± 0.17 | 0.52 ± 0.23 | 0.83 ± 0.17 | 0.86 ± 0.16 | 0.48 ± 0.26 | 0.88 ± 0.17 | 0.75 ± 0.21 |
| asymmetric_depth: L-R knee asymmetry err @true bottom, IK deg (truth ~ -8) | 0.50 ± 3.57 | 0.29 ± 4.15 | 1.12 ± 2.05 | 0.29 ± 4.15 | 2.55 ± 4.73 | 1.31 ± 4.43 | 0.76 ± 3.67 | 1.94 ± 5.26 |
| asymmetric_depth: L-R knee asymmetry err @pipeline bottom frame deg | 3.04 ± 4.15 | -1.84 ± 3.40 | 1.53 ± 3.15 | -1.84 ± 3.41 | 4.69 ± 3.98 | 1.44 ± 6.47 | 3.03 ± 4.65 | 0.96 ± 4.58 |
| clean: clean: false heel-rise amplitude mm @true bottom | 23.29 ± 20.55 | 13.41 ± 6.66 | 29.62 ± 5.27 | 13.41 ± 6.66 | 16.72 ± 11.99 | 36.09 ± 34.31 | 23.29 ± 20.55 | 16.16 ± 5.31 |
| clean: clean: false hip-shift amplitude mm @true bottom | 4.57 ± 9.77 | 2.86 ± 12.20 | -6.57 ± 22.60 | 2.86 ± 12.20 | 10.00 ± 18.78 | -4.58 ± 13.77 | 4.57 ± 9.77 | 17.84 ± 53.03 |
| clean: clean depth err final deg | -8.29 ± 3.35 | -11.97 ± 4.66 | -10.90 ± 4.39 | -11.96 ± 4.66 | -8.46 ± 3.57 | -7.53 ± 3.81 | -8.19 ± 3.33 | -9.95 ± 4.37 |
| stance_change: stance widening ratio (1 = preserved) | 0.99 ± 0.05 | 1.00 ± 0.04 | 0.03 ± 0.04 | 1.00 ± 0.04 | 1.00 ± 0.05 | 0.97 ± 0.04 | -0.01 ± 0.00 | 0.96 ± 0.07 |
| stance_change: stance width MAE mm | 28.11 ± 2.12 | 10.08 ± 0.64 | 106.58 ± 7.68 | 10.08 ± 0.64 | 21.20 ± 1.57 | 27.91 ± 2.33 | 133.06 ± 6.55 | 26.13 ± 1.88 |
| walkout: walkout MPJPE mm | 29.17 ± 1.09 | 30.75 ± 1.64 | 46.52 ± 2.84 | 30.23 ± 1.26 | 26.88 ± 0.94 | 30.11 ± 0.82 | 29.83 ± 1.67 | 35.54 ± 2.06 |
| walkout: walkout processed frames (readiness opened?) | 448.00 ± 76.05 | 448.00 ± 76.05 | 448.00 ± 76.05 | 448.00 ± 76.05 | 448.00 ± 76.05 | 448.00 ± 76.05 | 448.00 ± 76.05 | 448.00 ± 76.05 |
| walkout: walkout depth err final deg | -10.92 ± 0.46 | -11.93 ± 2.37 | -12.28 ± 1.43 | -11.93 ± 2.38 | -11.47 ± 0.86 | -10.71 ± 0.60 | -11.31 ± 0.15 | -12.93 ± 2.58 |
| fast_reps: fast reps knee @true bottom err IK deg | -0.21 ± 0.55 | -6.48 ± 2.05 | -11.78 ± 3.10 | -6.30 ± 1.88 | -0.82 ± 0.66 | -0.23 ± 1.15 | -0.21 ± 0.55 | -3.94 ± 1.18 |
| fast_reps: fast reps depth err IK deg | -0.48 ± 0.86 | -4.61 ± 1.48 | -6.10 ± 0.94 | -4.59 ± 1.49 | -1.13 ± 0.53 | -0.12 ± 0.64 | -0.47 ± 0.93 | -2.18 ± 2.22 |
| fast_reps: fast reps depth err final deg | -13.78 ± 3.75 | -17.39 ± 2.42 | -16.46 ± 1.45 | -16.75 ± 2.76 | -13.99 ± 2.72 | -13.75 ± 3.49 | -14.01 ± 3.57 | -13.07 ± 3.40 |
| fast_reps: fast reps knee lag IK ms | 10.08 ± 15.66 | 57.01 ± 11.43 | 104.86 ± 7.13 | 52.31 ± 13.50 | 18.78 ± 13.62 | 9.36 ± 14.24 | 10.03 ± 15.62 | 51.17 ± 16.14 |

### Gates / calibration timing (s from loop start, mean over scenarios) — `perfect`

| chain | standing gate | readiness gate | chain calibrated | frame us (chain+IK+valgus+angle filter) |
|---|---:|---:|---:|---:|
| raw | 0.33 | 0.58 | - | 394 |
| current | 0.33 | 0.58 | - | 474 |
| full_original | 0.33 | 0.58 | 2.00 | 864 |
| blend_only | 0.33 | 0.58 | - | 446 |
| vclamp_only | 0.33 | 0.58 | - | 429 |
| bone_only | 0.33 | 0.58 | 1.74 | 479 |
| ground_only | 0.33 | 0.58 | 2.00 | 431 |
| smooth_only | 0.33 | 0.58 | - | 536 |

## Overall (mean over 8 scenarios) — calibration `tpose`

| chain | MPJPE mm | lower mm | p95 mm | bone std mm | still jitter p50 mm | lag knee-y ms | lag knee IK ms | lag knee final ms | knee@true-bottom err IK deg | depth err IK deg | depth err final deg | bottom-frame delay ms | knee MAE moving IK deg | valgus@true-bottom err IK deg | stance MAE mm | IK knee=0 frac | chain us |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 43.9 | 49.1 | 114.0 | 37.9 | 35.0 | 0.1 | 5.9 | 72.3 | -0.1 | -5.1 | -15.2 | 97.9 | 16.6 | 0.1 | 32.9 | 0.28 | 50.3 |
| current | 72.2 | 51.7 | 276.1 | 28.2 | 7.8 | 77.7 | 82.3 | 160.0 | -7.5 | -11.4 | -19.8 | 222.5 | 22.0 | -0.7 | 18.5 | 0.28 | 131.1 |
| full_original | 88.3 | 68.8 | 303.6 | 14.7 | 1.5 | 134.0 | 144.5 | 216.9 | -12.2 | -13.2 | -20.8 | 287.9 | 27.0 | -0.7 | 26.4 | 0.28 | 523.7 |
| blend_only | 72.0 | 51.5 | 273.4 | 28.2 | 7.8 | 76.9 | 81.4 | 158.9 | -7.5 | -11.4 | -19.8 | 222.2 | 21.9 | -0.8 | 18.6 | 0.28 | 103.0 |
| vclamp_only | 41.7 | 43.7 | 107.4 | 30.0 | 33.8 | 3.5 | 8.8 | 75.3 | -0.2 | -5.1 | -15.3 | 106.8 | 16.9 | -0.0 | 26.9 | 0.28 | 103.5 |
| bone_only | 44.6 | 51.3 | 112.8 | 33.7 | 35.2 | 0.3 | 5.7 | 72.8 | -2.2 | -6.5 | -16.9 | 107.2 | 16.8 | -0.4 | 33.2 | 0.28 | 142.0 |
| ground_only | 45.6 | 53.1 | 117.4 | 38.2 | 34.8 | 0.1 | 5.9 | 73.0 | 0.1 | -4.9 | -15.1 | 96.3 | 16.8 | -0.1 | 40.7 | 0.28 | 93.7 |
| smooth_only | 48.2 | 52.0 | 119.0 | 32.1 | 4.6 | 51.2 | 57.6 | 122.7 | -1.9 | -5.9 | -15.3 | 167.9 | 20.0 | -0.1 | 30.0 | 0.28 | 181.8 |

### Paired delta vs raw (mean ± std over 8 scenarios x seeds) — `tpose`

| chain | mpjpe_mm | mpjpe_lower_mm | p95_err_mm | bone_len_std_mm | jitter_still_p50_mm | lag_knee_flex_ik_ms | knee_flex_tb_err_deg | depth_err_ik_deg | depth_err_final_deg | knee_flex_ik_moving_mae_deg | valgus_tb_err_deg | stance_width_mae_mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 26.9 ± 14.8 | 1.8 ± 9.2 | 154.1 ± 91.6 | -9.9 ± 4.4 | -27.2 ± 3.4 | 72.6 ± 31.7 | -7.1 ± 4.2 | -5.8 ± 4.3 | -4.2 ± 3.4 | 5.3 ± 2.3 | -0.8 ± 1.2 | -14.5 ± 4.4 |
| full_original | 43.2 ± 14.8 | 19.0 ± 10.2 | 182.1 ± 82.6 | -23.6 ± 5.3 | -33.7 ± 3.5 | 134.3 ± 37.3 | -11.9 ± 5.2 | -7.8 ± 3.0 | -5.4 ± 3.3 | 10.4 ± 3.0 | -0.8 ± 2.0 | -6.1 ± 28.9 |
| blend_only | 26.7 ± 15.0 | 1.6 ± 9.2 | 151.3 ± 93.4 | -9.9 ± 4.4 | -27.2 ± 3.4 | 71.6 ± 32.0 | -7.1 ± 4.2 | -5.8 ± 4.3 | -4.3 ± 3.4 | 5.2 ± 2.2 | -0.8 ± 1.1 | -14.5 ± 4.4 |
| vclamp_only | -2.2 ± 2.4 | -5.5 ± 1.8 | -6.6 ± 19.8 | -7.9 ± 2.7 | -1.2 ± 1.1 | 3.0 ± 4.6 | -0.1 ± 0.7 | -0.1 ± 0.8 | -0.1 ± 0.9 | 0.4 ± 0.4 | -0.2 ± 0.6 | -6.1 ± 2.6 |
| bone_only | 0.7 ± 1.3 | 2.2 ± 2.2 | -1.2 ± 2.5 | -4.3 ± 1.2 | 0.3 ± 1.3 | -0.1 ± 1.7 | -1.9 ± 2.3 | -1.5 ± 2.0 | -1.7 ± 1.9 | 0.2 ± 0.3 | -0.6 ± 3.0 | 0.3 ± 0.5 |
| ground_only | 1.7 ± 3.3 | 4.0 ± 7.9 | 3.6 ± 12.4 | 0.3 ± 1.3 | -0.2 ± 1.3 | -0.0 ± 0.2 | 0.3 ± 0.8 | 0.2 ± 0.8 | 0.1 ± 0.9 | 0.2 ± 0.6 | -0.3 ± 0.7 | 8.5 ± 30.4 |
| smooth_only | 4.3 ± 3.1 | 2.9 ± 2.3 | 5.0 ± 16.5 | -5.8 ± 1.1 | -30.5 ± 3.2 | 51.1 ± 9.0 | -1.7 ± 1.9 | -0.9 ± 2.0 | -0.1 ± 2.0 | 3.5 ± 1.6 | -0.2 ± 1.7 | -3.0 ± 2.0 |

### Fault-signal preservation and scenario-specific metrics — `tpose` (mean ± std over seeds)

| scenario / metric | raw | current | full_original | blend_only | vclamp_only | bone_only | ground_only | smooth_only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valgus: valgus amplitude ratio @true bottom, IK (1 = preserved) | 1.02 ± 0.05 | 0.98 ± 0.05 | 0.86 ± 0.06 | 0.98 ± 0.05 | 1.03 ± 0.05 | 1.02 ± 0.03 | 1.03 ± 0.07 | 0.95 ± 0.04 |
| valgus: valgus amplitude ratio @pipeline bottom frame, final | 0.80 ± 0.14 | 0.80 ± 0.10 | 0.75 ± 0.11 | 0.80 ± 0.10 | 0.83 ± 0.14 | 0.84 ± 0.12 | 0.83 ± 0.17 | 0.74 ± 0.13 |
| valgus: valgus @true bottom err deg (IK) | 0.88 ± 0.72 | 0.11 ± 0.86 | -2.17 ± 0.75 | 0.11 ± 0.86 | 0.86 ± 1.39 | 0.99 ± 0.60 | 0.97 ± 0.71 | 0.32 ± 1.62 |
| heel_rise: heel rise (ankle over toe) amplitude ratio @true bottom | 0.37 ± 0.40 | 1.96 ± 1.12 | 2.03 ± 0.77 | 1.96 ± 1.12 | -0.18 ± 0.62 | -0.22 ± 0.33 | 0.37 ± 0.40 | 0.66 ± 0.91 |
| heel_rise: heel rise amplitude ratio @pipeline bottom frame | 0.24 ± 0.31 | 0.96 ± 0.91 | 0.64 ± 0.79 | 0.96 ± 0.91 | 0.22 ± 0.43 | -0.52 ± 0.70 | 0.25 ± 0.32 | 0.73 ± 1.62 |
| hip_shift: lateral hip shift amplitude ratio @true bottom | 1.12 ± 0.30 | 0.99 ± 0.28 | 0.96 ± 0.07 | 1.00 ± 0.29 | 1.12 ± 0.33 | 0.97 ± 0.16 | 1.12 ± 0.30 | 0.84 ± 0.42 |
| hip_shift: lateral hip shift amplitude ratio @pipeline bottom frame | 0.96 ± 0.44 | 1.01 ± 0.31 | 0.98 ± 0.03 | 1.01 ± 0.31 | 0.93 ± 0.41 | 0.95 ± 0.34 | 0.96 ± 0.44 | 0.76 ± 0.25 |
| hip_shift: pelvic list amplitude ratio @true bottom, IK | 0.92 ± 0.34 | 0.93 ± 0.26 | 0.47 ± 0.14 | 0.93 ± 0.26 | 0.92 ± 0.34 | 0.36 ± 0.38 | 0.92 ± 0.34 | 0.95 ± 0.26 |
| hip_shift: pelvic list amplitude ratio @pipeline bottom frame, final | 0.78 ± 0.19 | 0.71 ± 0.17 | 0.24 ± 0.23 | 0.71 ± 0.17 | 0.78 ± 0.19 | 0.27 ± 0.23 | 0.77 ± 0.18 | 0.68 ± 0.28 |
| asymmetric_depth: L-R knee asymmetry err @true bottom, IK deg (truth ~ -8) | -0.02 ± 4.17 | 2.36 ± 6.33 | 0.46 ± 2.01 | 2.36 ± 6.33 | 1.61 ± 5.16 | 0.84 ± 2.90 | 0.35 ± 4.23 | 1.92 ± 6.07 |
| asymmetric_depth: L-R knee asymmetry err @pipeline bottom frame deg | 3.30 ± 5.71 | 2.31 ± 6.87 | 1.24 ± 3.65 | 2.31 ± 6.88 | 5.39 ± 5.51 | 2.43 ± 6.84 | 2.96 ± 6.37 | 0.12 ± 5.27 |
| clean: clean: false heel-rise amplitude mm @true bottom | 29.81 ± 20.25 | 26.41 ± 16.98 | 10.24 ± 25.77 | 26.41 ± 16.98 | 20.80 ± 7.13 | 19.25 ± 26.49 | 29.81 ± 20.25 | 19.29 ± 16.98 |
| clean: clean: false hip-shift amplitude mm @true bottom | 9.26 ± 4.19 | 8.25 ± 1.80 | -17.65 ± 17.10 | 8.25 ± 1.80 | 11.78 ± 9.63 | -2.19 ± 5.32 | 9.26 ± 4.19 | 18.34 ± 37.47 |
| clean: clean depth err final deg | -10.42 ± 4.63 | -16.19 ± 5.81 | -16.60 ± 7.92 | -16.19 ± 5.81 | -10.64 ± 4.71 | -11.15 ± 6.19 | -10.37 ± 4.60 | -11.95 ± 5.65 |
| stance_change: stance widening ratio (1 = preserved) | 0.92 ± 0.01 | 0.92 ± 0.01 | 0.08 ± 0.08 | 0.92 ± 0.01 | 0.92 ± 0.01 | 0.91 ± 0.01 | 0.02 ± 0.01 | 0.88 ± 0.05 |
| stance_change: stance width MAE mm | 37.04 ± 8.73 | 22.79 ± 9.66 | 90.35 ± 53.25 | 22.79 ± 9.67 | 30.47 ± 8.16 | 37.84 ± 8.28 | 107.81 ± 56.20 | 34.05 ± 7.10 |
| walkout: walkout MPJPE mm | 37.19 ± 1.04 | 97.37 ± 18.62 | 109.01 ± 16.13 | 97.37 ± 18.62 | 34.85 ± 0.70 | 37.95 ± 1.37 | 38.33 ± 0.53 | 41.11 ± 1.14 |
| walkout: walkout processed frames (readiness opened?) | 294.00 ± 224.26 | 294.00 ± 224.26 | 294.00 ± 224.26 | 294.00 ± 224.26 | 294.00 ± 224.26 | 294.00 ± 224.26 | 294.00 ± 224.26 | 294.00 ± 224.26 |
| walkout: walkout depth err final deg | -38.51 ± 24.61 | -50.97 ± 27.23 | -48.55 ± 24.54 | -50.97 ± 27.23 | -37.81 ± 25.03 | -39.19 ± 23.81 | -39.20 ± 24.02 | -37.51 ± 22.09 |
| fast_reps: fast reps knee @true bottom err IK deg | -0.49 ± 0.37 | -12.87 ± 4.43 | -18.78 ± 3.68 | -12.82 ± 4.37 | -0.99 ± 0.46 | -1.26 ± 1.24 | -0.49 ± 0.37 | -3.10 ± 1.33 |
| fast_reps: fast reps depth err IK deg | -7.48 ± 9.20 | -14.99 ± 10.15 | -16.54 ± 9.20 | -14.99 ± 10.15 | -8.34 ± 9.99 | -8.07 ± 8.29 | -7.46 ± 9.22 | -8.65 ± 6.58 |
| fast_reps: fast reps depth err final deg | -20.35 ± 7.39 | -25.18 ± 7.83 | -25.66 ± 6.36 | -24.86 ± 8.20 | -20.30 ± 8.10 | -21.40 ± 6.19 | -20.38 ± 7.38 | -19.08 ± 5.72 |
| fast_reps: fast reps knee lag IK ms | 7.67 ± 10.78 | 64.67 ± 8.64 | 114.51 ± 5.03 | 61.77 ± 10.32 | 15.85 ± 9.26 | 7.34 ± 9.77 | 7.64 ± 10.78 | 50.04 ± 12.82 |

### Gates / calibration timing (s from loop start, mean over scenarios) — `tpose`

| chain | standing gate | readiness gate | chain calibrated | frame us (chain+IK+valgus+angle filter) |
|---|---:|---:|---:|---:|
| raw | 0.81 | 1.89 | - | 416 |
| current | 0.81 | 1.89 | - | 468 |
| full_original | 0.81 | 1.89 | 3.51 | 877 |
| blend_only | 0.81 | 1.89 | - | 455 |
| vclamp_only | 0.81 | 1.89 | - | 451 |
| bone_only | 0.81 | 1.89 | 3.06 | 476 |
| ground_only | 0.81 | 1.89 | 3.51 | 424 |
| smooth_only | 0.81 | 1.89 | - | 512 |

## Input realism / delivery stats (per scenario, mean over seeds, calibration `tpose`)

| scenario | loop Hz | sync fail | skipped primary | duplicates | reproj px | tri lower conf<0.1 |
|---|---:|---:|---:|---:|---:|---:|
| clean | 28.4 | 0.184 | 0.057 | 0.000 | 6.47 | 0.079 |
| valgus | 28.4 | 0.216 | 0.058 | 0.000 | 6.36 | 0.090 |
| heel_rise | 28.4 | 0.129 | 0.056 | 0.000 | 6.48 | 0.091 |
| hip_shift | 28.4 | 0.149 | 0.055 | 0.000 | 6.38 | 0.078 |
| asymmetric_depth | 28.4 | 0.073 | 0.059 | 0.003 | 6.66 | 0.093 |
| fast_reps | 28.5 | 0.015 | 0.059 | 0.004 | 6.26 | 0.086 |
| stance_change | 28.4 | 0.128 | 0.059 | 0.001 | 6.58 | 0.089 |
| walkout | 28.4 | 0.094 | 0.059 | 0.003 | 12.00 | 0.251 |

T-pose calibration error (clean scenario, per seed): {"0": {"pnp_reprojection_px": 6.68, "centre_error_m": 0.603, "rotation_error_deg": 9.25}, "1": {"pnp_reprojection_px": 8.87, "centre_error_m": 0.319, "rotation_error_deg": 3.6}, "2": {"pnp_reprojection_px": 6.38, "centre_error_m": 0.496, "rotation_error_deg": 1.8}}; {"0": {"pnp_reprojection_px": 6.99, "centre_error_m": 0.867, "rotation_error_deg": 13.09}, "1": {"pnp_reprojection_px": 7.18, "centre_error_m": 0.652, "rotation_error_deg": 5.59}, "2": {"pnp_reprojection_px": 7.19, "centre_error_m": 0.495, "rotation_error_deg": 4.31}}; {"0": {"pnp_reprojection_px": 6.2, "centre_error_m": 0.548, "rotation_error_deg": 6.22}, "1": {"pnp_reprojection_px": 7.27, "centre_error_m": 0.368, "rotation_error_deg": 4.11}, "2": {"pnp_reprojection_px": 6.65, "centre_error_m": 0.435, "rotation_error_deg": 3.27}}

## Calibration floor: raw chain with noise profile `none` (no 2D noise/quantization)

| calibration | MPJPE mm | lower mm | bone std mm | depth err IK deg | depth err final deg | lag knee final ms | valgus@true-bottom err deg | trunk MAE deg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perfect | 1.4 | 1.0 | 0.2 | -0.0 | -2.7 | 75.5 | 0.0 | 3.6 |
| tpose | 17.9 | 18.3 | 4.5 | 0.7 | -2.0 | 76.0 | 0.1 | 3.6 |

## Noise sensitivity (calibration `perfect`, overall mean over scenarios)

| noise | chain | MPJPE mm | p95 mm | depth err IK | depth err final | knee MAE moving IK | valgus@true-bottom err | IK knee=0 frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mild | raw | 22.2 | 41.1 | 1.1 | -5.5 | 7.2 | -0.4 | 0.100 |
| mild | current | 25.6 | 77.5 | -0.6 | -6.9 | 10.6 | -0.4 | 0.100 |
| mild | full_original | 43.9 | 138.9 | -1.7 | -6.8 | 17.1 | -0.6 | 0.100 |
| harsh | raw | 75.0 | 325.2 | -12.1 | -33.6 | 37.0 | 0.1 | 0.605 |
| harsh | current | 60.6 | 186.3 | -17.2 | -39.2 | 39.4 | -1.4 | 0.605 |
| harsh | full_original | 77.1 | 224.6 | -19.8 | -39.9 | 42.5 | -2.1 | 0.605 |

## Diagnostic: how much error is the IK dropping joints with confidence < 0.1?
`diag_conf_floor` keeps raw positions but floors confidences at 0.11 (NOT a proposed filter).

| calibration | chain | MPJPE mm | IK knee=0 frac | knee MAE moving IK | depth err IK | depth err final | bottom delay ms | valgus@pipeline-bottom err |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| perfect | raw | 33.8 | 0.25 | 14.3 | 0.9 | -9.5 | 71.1 | -0.5 |
| perfect | diag_conf_floor | 33.8 | 0.00 | 4.3 | 3.4 | -3.1 | 111.7 | -0.8 |
| tpose | raw | 43.9 | 0.28 | 16.6 | -5.1 | -15.2 | 97.9 | -0.3 |
| tpose | diag_conf_floor | 43.9 | 0.00 | 4.6 | 0.9 | -5.6 | 135.5 | -0.5 |

## Delivery sensitivity: cameras at 24 fps (low light) vs loop ~28 Hz -> duplicated frames (calibration `perfect`)

| chain | dup frac | MPJPE mm | lag knee IK ms | knee@true-bottom err IK | depth err final | still jitter p50 mm |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.058 | 33.4 | 3.5 | -0.1 | -9.2 | 32.1 |
| current | 0.058 | 35.4 | 54.3 | -3.6 | -10.9 | 9.1 |
| full_original | 0.058 | 53.1 | 110.9 | -5.5 | -9.8 | 1.9 |
| smooth_only | 0.058 | 39.1 | 51.4 | -2.1 | -9.0 | 5.2 |
