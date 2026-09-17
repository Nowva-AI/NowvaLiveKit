# Research agent PROGRESS (pre-IK filter literature review)

Task: literature digest + recommended pre-IK filter architecture for 3-cam RTMPose-m halpe26 / DLT / 30 Hz / causal / Jetson.
Deliverable: REPORT.md in this folder. Raw sources cached in ./src/ (Pose2Sim Config.toml, OpenCap paper text + core code).

## Repo context already read (for tailoring)
- preik_chain.py active: ConfidenceBlender (EMA toward prev, w=(c-0.1)/0.8) -> VelocityClamp (2.5 m/s, clamps along dir; state = clamped pos -> can lock onto stale pos after a real jump). Disabled: BoneLengthConstraints (rigid, tol 0, median of 30 standing frames, projects distal onto sphere, proximal->distal cascade), GroundClamp (ankle y <= standing y, both ankles forced SAME y if diff>1cm => erases heel rise/pelvic list asym; stance width locked +-2cm), One Euro positions (min_cutoff .8, beta 4, d_cutoff 1).
- triangulator.py: unweighted DLT, conf>=0.3, >=2 views, no view rejection; reproj>=15px -> conf*0.1 but point kept; mean reproj over views; recenter at hip mid.
- Post-IK: JointAngleFilter One Euro phase-aware (idle 0.3/0.003 ...), PredictiveStateEstimator +0.2 s extrapolation.

## Findings so far (verified from sources)
### Pose2Sim (Pagnon 2021 Sensors 21:6530, PMC8512754; repo perfanalytics/pose2sim Config.toml)
- Paper: confidence-WEIGHTED DLT (rows multiplied by c); 2D likelihood thr 0.3; reproj thr 10 px (paper) / 15 px (current config); iteratively drop cameras (all combos of k off, pick min error) until < thr or < min_cameras (paper 3 of 8 cams; config 2); dropped points interpolated (cubic, now linear, gap<20 frames), large gaps last_value.
- Filter: zero-lag 4th-order Butterworth 6 Hz (filtfilt, order/2 designed). Config comment: 3-6 Hz walking/slow, 6-15 running. reject_outliers=true: Hampel window 7, n_sigma 2 (MAD*... 95% CI) BEFORE filtering.
- Other filters: kalman constant-ACCELERATION per coordinate, measurement_noise 20, process_noise=20*trust_ratio(500), RTS smoother (smooth=true "unless real-time"); one_euro cutoff 4 Hz beta 1.5 dcut 1.0 ("tends to blunt RoM", run zero-phase fwd-bwd); gcv_spline auto; acc_minimizing (Whittaker, lambda=(fs/(2*pi*fc))^4, "blunts peaks", recommended on IK not triangulated); loess 5; gaussian sigma 1; median k 3; butterworth_on_speed 10 Hz order 4 (offset drift).
- Accuracy (paper, 1 subject, 8 cams, 30 Hz?): reproj ~3.5 px (~1.6 cm) walk/run, 6 px (~3 cm) cycling; no L/R swaps with Body_25b; ankle least robust.
- handle_LR_swap: config says "Not implemented yet" in [pose]; older triangulation.py tries swapping L/R per camera subset when error > thr.
- Kinematics scaling: large_hip_knee_angles=90 (frames beyond excluded), trimmed_extrema_percent=50, right_left_symmetry=true.
- det_frequency=4 (person detection every 4 frames, bbox tracking in between) -> person crop for RTMPose.
- markerAugmentation feet_on_floor option.
### OpenCap (Uhlrich 2023 PLOS Comp Biol e1011462; stanfordnmbl/opencap-core)
- 60 Hz, 720x1280, OpenPose/HRNet, 20 kpts incl heels + small/big toes.
- 2D stage (utilsSync.py): removeOccludedSide (if mean L vs R foot/arm conf differ >0.2 and side conf<thr => that side NaN'd, splined); clean2Dkeypoints conf<0.4 -> conf 0 (>2 cams) or NaN+interp (2 cams); 2D Butterworth order 4 zero-lag, cutoff 12 Hz gait / 'default' 500 (=fs/2, i.e. effectively none) non-gait.
- Triangulation: confidence-weighted DLT; optional min-reproj camera-subset selection and RANSAC (errorUB 20 px, "not clear this is helpful"); spline3dZeros (gaps <= fs/5 frames).
- LSTM augmenter: 43 markers from 20 kpts (body 35 from 15, arm 8 from 7) + height/weight; root(hip mid)-relative, height-normalized, 0.5 s windows at 60 Hz, trained with 18 mm Gaussian noise on inputs; test RMSE 8.0 mm body / 15.2 mm arm; ~32 mm vs mocap in validation.
- Accuracy Table 1: rotations MAE walk 4.1 (2.3-6.6), SQUAT 4.1 deg (1.8-7.2), sit-to-stand 4.7, drop jump 5.1, mean 4.5; pelvis translations 12.3 mm (squat 12.3, 5.8-18.4).
- Dynamics filtering: 4th-order zero-lag Butterworth 12 Hz gait / 30 Hz non-gait on joint pos/vel/acc.
### anipose (Karashchuk 2021 Cell Reports, PMC8498918; lambdaloop/anipose defaults)
- 2D filters: median (medfilt 13, offset_threshold 25 px, spline interp), viterbi (n_back 5, top-n candidates HMM), autoencoder (confidence recalibration). score_threshold 0.05 (2D).
- Triangulation defaults: ransac False, optim False, scale_smooth 2, scale_length 2, scale_length_weak 1, reproj_error_threshold 5, score_threshold 0.8, n_deriv_smooth 3; robust soft-L1/Huber reprojection loss; limb lengths ESTIMATED (not fixed), soft.
- Results: spatiotemporal constraints biggest gain on human (position t=-18.7, angle t=-6.1); RANSAC did not help; median filter helped humans (t=-14.8) but "may remove fast movements"; viterbi humans t=-10.9. Offline batch only.

## Still to do
- EasyMocap / 4D association / FreeMoCap / Theia / Captury; skeleton Kalman (bone-length constrained); MediaPipe One Euro params; SmoothNet cost.
- RTMPose person crop / SimCC sub-pixel; 2D vs 3D filtering evidence.
- Latency for feedback; Winter residual cutoffs for squat; causal Butterworth phase lag numbers (compute locally).
- Foot: contact/zero-velocity, floor plane, heels.
- Write REPORT.md (digest, architecture, missing filters ranked, pitfalls).

## How to resume
Read this file + ./src cache; continue "Still to do" with WebSearch/WebFetch (curl works for raw GitHub); write REPORT.md here.

## Milestone 2 (added)
- MediaPipe pose_landmark_filtering.pbtxt: norm landmarks One Euro min_cutoff 0.05, beta 80, dcut 1.0 (value-scaled by object size); WORLD landmarks min_cutoff 0.1, beta 40, dcut 1.0, disable_value_scaling; visibility low-pass alpha 0.1; ROI (crop) One Euro min_cutoff 0.01, beta 10. Crop comes from previous-frame landmarks.
- SmoothNet (Zeng ECCV 2022, arXiv 2112.13715): window T=32 (non-causal, ~0.5 s lookahead at 60 fps /1 s at 30), 0.33M params, 1.3k FPS CPU; H36M Accel 19.17->1.03, MPJPE 54.55->52.72; says One Euro inferior; SG/Gaussian over-smooth at matched Accel.
- Theia3D: GCVSPL gap fill + smoothing, default 20 Hz cutoff; slow 6-12 Hz, fast 10-20 Hz; saves filtered and unfiltered.
- EasyMocap (zju3dv, src/easymocap_*.py): conf-weighted batch DLT; reconstruction.py batch_triangulate adds previous-frame 3D prior rows (lamb=1e3*conf_pre) = causal temporal regularization inside DLT; iterative_triangulate: min_conf 0.1, min_view 3, dist_max 0.05 (normalized img units), temporal gate dist_vel 0.05 vs reprojection of previous 3D, drop whole VIEW if >40% joints outliers, drop joint if >40% views outliers, robust_triangulate_point tries C(n,3) subsets.
- Needham 2021 Sci Rep 11:20673 (PMC8526586): 9 cams 200 Hz, RANSAC ray intersection, bi-directional Kalman; OpenPose mean diffs hip 29-36 mm, knee 30-41, ankle 14-23; all methods place HJC more LATERAL+INFERIOR; knee ~20-30 mm below markers (dataset labelling bias).
- rtmlib (src/rtmlib_*.py): top-down affine crop with bbox padding 1.25 preserving aspect ratio; SimCC argmax loc/2 (split ratio 2 => 0.5 px of 192x256 input); score = mean of raw max x/y responses (no sigmoid), locs invalid if score<=0; Pose2Sim/Sports2D det_frequency 4 (detector every 4 frames, tracking between).
- Pose2Sim weighted_triangulation: A rows multiplied by likelihood (common.py:631). compute_height/segment lengths: only frames with mean hip/knee angle < large_hip_knee_angles (90), fallback 50 smallest; trimmed_extrema_percent 50; feet from heel-ankle.
- Sports2D compute_floor_line: fit line through toe (or ankle) positions where speed < toe_speed_below px/frame and score>0.3 -> floor angle + origin; correct_segment_angles_with_floor_angle.
- OpenCap 2D Butterworth: 'default' 500 Hz => wn capped 0.99 => effectively NO 2D smoothing for non-gait (squat); gait 12 Hz. (An earlier WebFetch summary claiming "4 Hz squat" was a hallucination - do NOT cite.)

## Milestone 3: numeric experiment DONE (filter_lag_experiment.py -> filter_lag_results.txt)
Synthetic 30 Hz squat (45 cm hip depth, 3 cm/0.6 s valgus bump), white noise, 30 seeds. Key rows (1 cm noise, normal tempo):
- raw: RMSE 1.00 cm, jitter 13.9 mm/frame, noisy valgus peak read 137% (peak-picking on raw data overestimates).
- current One Euro pos 0.8/4.0: lag 57 ms, keeps only 72.6% of valgus peak (clean), RMSE 1.92 (WORSE than raw).
- ConfidenceBlender EMA @conf .5: lag 33 ms, RMSE 1.24 (worse than raw); @conf .7: 10 ms, RMSE .86.
- causal Butterworth 2nd 6 Hz: 33 ms lag, RMSE 1.25 (worse than raw); 4 Hz 53 ms RMSE 1.83. Group delay 0.225/fc s.
- causal Kalman CV/CA: RMSE .77/.76, 5-9% overshoot at reversals.
- fixed-lag Kalman CV 3 frames (100 ms latency): RMSE .47, valgus 97%, jitter 2.2 mm; CV 6 fr: .46; CA 6fr: .40 but overshoots depth at fast tempo.
- rep-window filtfilt 4 Hz: RMSE .48, valgus 99%; 6 Hz: .58.
At 2 cm noise: fixed-lag CV 3fr RMSE .80 vs raw 1.99; One Euro current 2.00. Fast tempo: causal Butterworth 4 Hz RMSE 3.21 (lag dominates).
Next: outlier/spike test, remaining lit (latency, foot contact, L/R swap), then REPORT.md.

## Milestone 4: outlier experiment DONE (outlier_experiment.py -> outlier_results.txt); filter_lag_results.txt regenerated OK
1 cm noise + 3% single-frame spikes (5-20 cm) + two 4-frame 10 cm glitches, 30 seeds:
- raw RMSE 2.67 cm (p95 max err 22.6); VelocityClamp only (current chain) 1.98 (18.7).
- fixed-lag CV(3) alone 1.44 (13.3); + VelocityClamp 1.23; + trailing Hampel(7) hip 1.25 / knee 0.81 (Hampel hurts moving joints: median lags);
  innovation gate max(3sigma,4cm) 1.10 but p95 max err 19.5 (gate lockout on real motion).
- Conclusion: 3D-only temporal outlier rejection cannot remove multi-frame (view-level) glitches (p95 max err stays 10-19 cm) -> must be solved at triangulation (per-view reprojection / leave-one-view-out) where the other 2 views carry the information.
Next: latency/feedback lit, foot contact/heel, L/R swap lit, then REPORT.md.

## Milestone 5: remaining lit + cost bench DONE -> writing REPORT.md now
- Stenum 2021 PLOS CB e1008935: OpenPose L/R limb switch in ~5% of frames (sagittal views), fixed manually; zero-lag 4th Butterworth 5 Hz; gaps <=0.12 s linear; MAE hip 4.0, knee 5.6, ankle 7.4 deg.
- Pose2Sim Part 2 (Sensors 22:2712): mean joint angle error 3.0/4.1/4.0 deg walk/run/cycle; hip 15 deg offset running; ankle occluded cycling CMC .75.
- De Groote 2008 J Biomech: Kalman SMOOTHING (IK in joint space) >50% lower joint position error than GOM/Kalman filtering/LME.
- Lightning Pose 3D (PMC13131684): mvEKS, R = ensemble variance; per-view Mahalanobis vs other views >5 => variance x10 (downweight), iterate; random-walk latent, offline; matches full anipose.
- 4D association (Zhang CVPR 2020): 30 fps, 5 cams, 5 people, joint parsing/matching/tracking graph.
- Sigrist 2013 Psychon Bull Rev review + Sigrist 2013 (PubMed 24006910): terminal feedback outperforms concurrent for complex rowing task; guidance hypothesis.
- FreeMoCap issue #849: Butterworth designed for hardcoded 30 fps regardless of recording rate (pitfall: filter must use real dt/fs).
- 1euro tuning (gery.casiez.net/1euro): beta=0, mincutoff ~1 Hz, lower mincutoff for jitter at slow speed; raise beta for lag at speed (try decades).
- cost_microbench.py (Mac): weighted DLT all 4 subsets+reproj 500 us; CV Kalman step+gate 19x3 18 us; rep filtfilt 120fr 212 us; floor plane fit 19 us.

## STATUS: COMPLETE (2026-09-16)
Final report composed and returned as the agent's final message to the coordinator. Writing REPORT.md was
REFUSED by the harness ("Subagents should return findings as text, not write report files"), so the
coordinator must save the final message as REPORT.md if a file is needed.
Artifacts here: filter_lag_experiment.py + filter_lag_results.txt, outlier_experiment.py + outlier_results.txt,
cost_microbench.py + cost_microbench.txt, src/ (Pose2Sim, OpenCap, EasyMocap, rtmlib, Sports2D, MediaPipe sources).
