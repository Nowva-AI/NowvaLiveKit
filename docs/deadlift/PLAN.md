# Conventional deadlift — plan v10 (built on the multi-exercise platform)

Status: proposal for Ambaka. Version 10. Replaces v1–v9, which assumed a separate deadlift subprocess. Since PR #20 (`squat-v1-integration`), every exercise runs in one pipeline process as a swappable profile. This plan plugs the deadlift into that platform.

## 0. Decisions

| Decision | Consequence |
|---|---|
| The deadlift plugs into the multi-exercise platform (Ambaka, 2026-10-03) | One pipeline process, profile swap through `set_exercise`. No separate subprocess and no copy of the camera calibration |
| Squat behaviour must not change | Every change to shared code is a dispatch whose default is today's squat path. Golden-master tests prove the squat is unchanged, both fresh and after switching to a deadlift and back (§5) |
| Conventional deadlift only | Sumo, trap-bar, deficit, snatch-grip and single-leg variants resolve to an untracked profile in the registry (§3.2) |
| `coaching_ready = False` until the demo gate (§1) passes | This is the feature flag. The menu never offers the deadlift before then; developers turn it on with an env override (§3.1) |
| Back rounding: indirect proxies only | Never claimed, never cued as if measured (FINDINGS honesty principle) |
| The bar must be detected on the floor | 3D bar tracking from all three cameras (§6) |
| LLM learning is out of scope | Sessions, reps and cue events are already tagged per exercise (one DB session per exercise) |
| Thresholds are absolute and are never moved by observed reps | Same two channels as the squat (FINDINGS D2): an absolute standard, plus drift against the athlete's best rep this session |

## 1. Goal and gates

Demo loop:

> "Let's deadlift" → Nova guides the feet in a closed loop ("bar over midfoot — a bit closer… good") → the athlete sets up and pulls → the rep is judged as a whole → **one** cue at the floor before the next rep → the next rep improves → set recap with the numbers.

The demo rests on four "hero" faults: **D1 bar position, D2 hips shoot, D3 bar drift, D4 setup hip height**. Every other v1 fault ships as "provisional": it is only cued from the moderate tier up until there is enough data to trust it (§8).

| Metric (real data, lifters never seen in training or tuning) | Demo gate (J6) | Launch gate (later) |
|---|---|---|
| Rep counting | ≥ 99 %, 0 phantom reps on clean sets | same |
| Event timing (liftoff, knee pass, top, floor) | median error ≤ 100 ms | same |
| Hero-fault precision | ≥ 0.85; 95 % lower bound (lifter-clustered bootstrap) ≥ 0.70; ≥ 0.80 on natural sets | lower bound ≥ 0.75 |
| Hero-fault recall | ≥ 0.70; lower bound ≥ 0.55 | lower bound ≥ 0.60 |
| False corrections on clean reps | ≤ 1 per 10 reps | same |
| Measurement error (P95 vs ground truth) | ≤ 1/3 of the threshold of the lowest tier that is cued (§2.9) | same |
| Squat | golden masters identical, fresh and after a deadlift switch (§5) | same |
| Edge | ≤ 33 ms/frame on a Jetson Orin Nano Super, or the defined degraded mode (§9) | same |

## 2. What we measure and how

### 2.1 Frame and axes

- **Platform conventions apply** (`.claude/rules/biomechanics.md`):
  - 3D arrays are Y-down; up is `geometry.WORLD_UP`; heights go through `height_above` / `is_above`.
  - The foot frame comes from `rep_features._foot_frame` (`rep_features.py:141-177`): lateral is the horizontal hip line; forward is lateral × down, oriented heel → toe.
  - Midfoot = ankle + 0.35 × (ankle→toe), averaged over both feet (`:173-176`).
- **Gravity.** The world Y axis is the median standing hip→ankle direction (`person_calibration.py:734-752`), not gravity, and it can be 2–4° off. Over 55 cm of bar travel, 3° fakes ≈ 2.9 cm of drift, which is the D3 threshold. So the deadlift analyser uses a measured gravity vector:
  - Source: the ChArUco board laid flat on the floor (`charuco.py:476-481`), solved per camera by a new `scripts/tools/measure_gravity.py`.
  - Stored in `~/.nowva/gravity_<camera_key>.json`, in each camera's own frame.
  - At run time it is mapped into the current world frame through the calibration rotations. A camera that disagrees by more than 1° is dropped, because it has moved.
  - Fallback: `WORLD_UP`, flagged `gravity_source="body"`. D2, D3 and D5 are then cued only from moderate up.
  - Squat code keeps `WORLD_UP`; changing it is out of scope (observation O1).
- **Deadlift sagittal frame** (`src/biomechanics/deadlift/frame.py`):
  - up = gravity; lateral = the bar axis at rest (or the hip line) projected horizontal; forward = heel → toe.
  - It gives signed sagittal angles and forward/lateral offsets. The squat's trunk pitch is unsigned and includes sideways lean (`analytical_ik.py:254`); the deadlift needs a **signed** sagittal trunk angle for D5 (lean-back).
- **All deadlift metrics are differences**: bar vs its resting height, bar vs midfoot, wrists vs their standing height. None of them needs an absolute floor.

### 2.2 Inputs per frame

| Input | Source | Today |
|---|---|---|
| World-frame skeleton (21 kpts) | `PreIKResult.analysis_world` (`preik_chain.py:136-141`) | computed but dropped before profiles and rules (§3.3 exposes it) |
| 3D bar state | new `deadlift/bar_tracker_3d.py` (§6) | does not exist |
| Foot state (planted anchors, heel rise) | `FootState` (`foot_contact.py:165-185`) | already in the rule frame context |
| Joint angles | `AnalyticalIK` | already available |
| Gravity | §2.1 | new |

### 2.3 Phases and rep counting (driven by the bar)

**Phases:** `APPROACH` (standing) → `STANCE` (standing over the bar, feet planted) → `SETUP` (hinged, hands on the bar, still) → `PULL` (liftoff → knee pass → top) → `TOP` → `LOWER` → `FLOOR` → then back to `SETUP`, straight into `PULL` (touch-and-go or quick re-pull), or to `APPROACH`.

**Rep signal:** the bar centre's height above its resting height. Fallback when the bar is not tracked: mid-wrist height above its height during `SETUP`.

**Counter:** a custom `DeadliftRepCounter`, created through `create_rep_counter`.
- It implements the pipeline counter API: `in_rep`, `phase`, `rep_count`, `rep_started`, `update`, `reject_last_rep`, `clear_current_faults`, `snapshot_rep_metrics`, `rejected_rep_max_depth_angle`, `reset`, `set_assessment_mode`.
- It never uses the squat's IDLE baseline EMA (`hip_position_counter.py:283-291`), which would follow the setup position.

| Transition | Condition (initial values, `config/deadlift.yaml`) |
|---|---|
| APPROACH → STANCE | Standing, midfoot within 15 cm (horizontal) of the bar, feet still ≥ 0.5 s |
| STANCE → SETUP | Hands within 10 cm above the bar, still ≥ 0.3 s |
| SETUP → STANCE/APPROACH | The lifter stands back up without lifting |
| SETUP → PULL | Bar > rest + 3 cm and vertical velocity > 0.10 m/s |
| PULL → TOP | Bar ≥ expected top height − 8 cm, \|v\| < 0.05 m/s for ≥ 3 frames, trunk within 35° of vertical |
| PULL → FLOOR without reaching TOP | **Failed rep**: the bar rose ≥ 10 cm but never reached top − 8 cm. Logged as an event, not counted |
| TOP → LOWER | v < −0.10 m/s |
| LOWER → FLOOR | **Dead stop**: bar ≤ rest + 2 cm and \|v\| < 0.02 m/s for ≥ 3 frames, or bar within rest + 2 cm for ≥ 0.3 s → **rep counted and judged** |
| LOWER → PULL | **Touch-and-go**: low point within 5 cm of rest, then a rise of more than 3 cm above that low point, with upward velocity sustained ≥ 100 ms and hands on the bar → **the rep that is ending is counted and judged at that moment**. The low point becomes the next rep's liftoff, and `rep_started` is set so the pipeline's existing touch-and-go path applies (`pipeline.py:1045-1048`) |
| FLOOR → SETUP | Dead stop, hands still on the bar ≥ 0.3 s → a new setup is judged |
| FLOOR → PULL | Quick re-pull: the setup is judged on the last 0.2 s before liftoff, or marked "not measured" |
| FLOOR → STANCE/APPROACH | The lifter stands up or steps back (the set may end) |

- **Liftoff is back-dated to the start of motion.** On a slow pull the trigger alone lags by ≈ 150 ms, so the liftoff time is taken as the last frame in a 0.5 s buffer where the bar was within 0.5 cm of rest and moving under 0.02 m/s.
- **Expected top height** = standing mid-wrist height (arms hanging, measured during `APPROACH`) − the learned wrist→bar offset. The 8 cm margin means soft and over-extended lockouts are still counted; D6 and D5 then judge them.
- A bumper bounce (≤ 3 cm, no pull) is not a touch-and-go. Every rep is counted exactly once.
- A bar dropped from the top counts as a rep, with no fault.
- Set boundaries are the platform's existing ones (`set_timeout_seconds`, `rest_start`).

### 2.4 References taken during the approach

While the athlete stands still in `APPROACH` (≥ 1 s), we record:
- signed sagittal hip, knee, trunk and elbow angles;
- standing mid-wrist height;
- the foot anchors and the midfoot, locked before the pull so that plate occlusion during the pull does not matter.

D5 (lean-back), D6 (lockout) and D9 (bent arms) are measured against these references, which removes each person's keypoint bias. Deadlift reps never touch the platform's squat setup snapshot or `_standing_reference_hip_cm` (§3.3).

### 2.5 Per-rep features (`DeadliftRepFeatures`)

There is one feature vector per rep, built from measured frames only (never from Kalman-extrapolated ones). NaN means "not measured", never 0. The robust statistics are the same as in `rep_features.py:24-58`.

| Feature | Definition | Window |
|---|---|---|
| `bar_midfoot_stance_cm` | Median forward offset of the bar centre vs midfoot | Last 0.3 s of STANCE |
| `bar_midfoot_setup_cm` | Same | Last 0.3 s of SETUP |
| `setup_hip_height_cm` | Hip height above the ankle midpoint | SETUP |
| `setup_hip_band_cm` | Expected band from the setup model (§2.7) | — |
| `shoulder_vs_bar_cm` | Forward offset of the shoulder joint vs the bar centre | SETUP |
| `trunk_change_liftoff_knee_deg` | Signed trunk-angle change from the back-dated liftoff to the bar reaching knee height, minus the change the model predicts | Early pull |
| `hip_shoulder_rise_ratio` | Hip vertical rise ÷ shoulder vertical rise | Same window |
| `bar_drift_cm` | p90 forward deviation of the bar centre from its liftoff position | Pull |
| `hip_extension_deficit_deg` | Signed hip extension at the top vs the standing reference | TOP |
| `knee_extension_deficit_deg` | Signed knee flexion at the top vs the standing reference | TOP |
| `lean_back_deg` | Signed backward trunk angle at the top vs the standing reference | TOP |
| `hip_shift_ratio` | (hip_mid − midfoot) projected on the ankle axis ÷ ankle separation; the sustained 80th-percentile deviation vs the start median, same method as the squat (`rep_features.py:371-390`) | Pull |
| `bar_tilt_cm` | p90 height difference between the two ends of the 3D bar | Pull |
| `elbow_flexion_deg` | p90 elbow flexion vs the standing reference | Pull |
| `concentric_velocity_ms` | Bar rise ÷ time from liftoff to top. This uses the true bar, not the shoulder proxy the squat uses | Pull |
| `pull_time_s`, `lower_time_s` | Phase durations | — |
| `bar_source` | `bar` or `wrist_proxy` | — |

### 2.6 Faults (v1)

Every fault gets a new `FaultType` value prefixed `deadlift_`, so it never shares the squat's global priority, observability or cue entries. Each one is a `RepFaultRule`, judged once per rep through `RuleEngine.finish_rep` (`rule_engine.py:226-253`). `_rep_fault` (`fault_types.py:297-328`) stamps the standard details: `side`, `phase`, `observability`, `is_drift`, `value`, `unit`.

| # | fault_type | Meaning | Measured from | Thresholds (mild / moderate / severe) | Observability | Cue key (≤ 4 words) | Priority |
|---|---|---|---|---|---|---|---|
| D1 ★ | `deadlift_bar_position` | Bar not over midfoot at setup | `bar_midfoot_setup_cm` (the stance value drives the closed-loop guidance) | 3 / 5 / 8 cm | `bar_3d` (new class: not observable on one camera) | `deadlift_bar_midfoot` ("Bar over midfoot") | 20 |
| D4 ★ | `deadlift_setup_hips` | Hips too low or too high at setup | `setup_hip_height_cm` vs the band | Outside the band by 4 / 7 / 10 cm | `side_view` | `deadlift_hips_up` / `deadlift_hips_down` | 21 |
| D7 | `deadlift_shoulders_behind` | Shoulders behind the bar at setup | `shoulder_vs_bar_cm` | 2 / 4 / 6 cm behind | `side_view` | `deadlift_shoulders_over` ("Shoulders over bar") | 22 |
| D2 ★ | `deadlift_hips_shoot` | Hips rise before the chest off the floor | `trunk_change_liftoff_knee_deg`; cross-check: ratio > 1.4 | 10 / 15 / 20° | `side_view` (≥ 30 fps) | `deadlift_chest_with_hips` ("Chest and hips together") | 23 |
| D3 ★ | `deadlift_bar_drift` | Bar drifts away from the legs | `bar_drift_cm` | 3 / 5 / 8 cm | `bar_3d` | `deadlift_bar_close` ("Bar close") | 24 |
| D6 | `deadlift_lockout` | Incomplete lockout at the top | Hip or knee extension deficit | 8 / 12 / 20° | `side_view` | `deadlift_lockout` ("Stand tall") | 25 |
| D5 | `deadlift_lean_back` | Over-extension at the top | `lean_back_deg` | 8 / 12 / 18° | `side_view` | `deadlift_finish_neutral` ("Hips through, tall") | 26 |
| D8 | `deadlift_hip_shift` | Hips shift sideways during the pull (side = direction of travel) | `hip_shift_ratio` | 0.10 / 0.15 / 0.22 (the squat's values, above the noise floor) | `lateral_travel` | `deadlift_even_feet` + `_left`/`_right` | 27 |
| D8b | `deadlift_bar_tilt` | Bar tilts | `bar_tilt_cm` | 3 / 5 / 7 cm | `bar_3d` | `deadlift_level_bar` ("Level the bar") | 28 |
| D9 | `deadlift_bent_arms` | Arms bend during the pull | `elbow_flexion_deg` | 15 / 25 / 35° | `side_view` | `deadlift_long_arms` ("Long arms") | 29 |
| D10 | `deadlift_velocity_loss` | Bar speed dropped vs the set's two fastest reps | `concentric_velocity_ms` | 20 / 30 / 40 % (Sánchez-Medina 2011, as for the squat) | `bar_3d` | `deadlift_drive` ("Drive the floor") | 30 |

★ = hero fault.
- The cue wording above is a draft; the final text comes from the cue pipeline (§4.2).
- Priorities 20–30 are appended to `FAULT_CUE_PRIORITY`. Squat ranks 0–11 and `DEFAULT_FAULT_CUE_PRIORITY = 12` do not move.
- D2, D3 and D8 also report `is_drift` against the session best (§2.8).
- Trunk rotation is not part of D8, because a mixed grip rotates the trunk. The grip type is recorded instead (§4.6).
- **No `deadlift_flat_back` cue.** The current placeholder maps `back_rounding → deadlift_flat_back` (`profiles/deadlift.py:92-96`). That mapping is removed, because the spine is not observed (§2.10).

**v1.1:** jerking the bar off the floor (bar acceleration spike at liftoff), uncontrolled lowering, knees moving forward before the bar passes them on the way down, hitching, head position, grip and stance width, heel rise, and foot placement (the squat's stagger/flare measurement, `rep_features.py:295-317`).

### 2.7 Setup model (core IP)

This is the deadlift counterpart of the squat lean model (`diagnosis/lean_model.py`): the right starting position for **this** body.

**Constraints**, expressed as bands:
- bar over midfoot;
- shoulder joint 0–6 cm in front of the bar;
- arms near vertical (≤ 5° back);
- shins touching the bar.

**Inputs:**
- Tibia and femur lengths **projected into the sagittal plane**, measured on SETUP frames, because knees-out and toe-out shorten them.
- Torso and foot lengths from `SegmentLengthEstimator`, which is kept across exercise switches.
- Arm length (shoulder → elbow → wrist) from a deadlift-specific estimator; `SegmentLengthEstimator` does not measure arms (`segment_lengths.py:53-64`).
- Wrist→grip offset, 6–9 cm, learned per athlete.
- Distance from the bar axis to the shin at contact, initially 5 cm.
- Measured bar and ankle positions.
- Everything is relative to the ankle; the model needs no floor.

**Solve 1 (setup):**
- Place the shoulder in its band and fix the ankle. This leaves one degree of freedom.
- Shin contact gives two solutions; keep the one with the knee in front of the ankle and knee flexion in [40°, 120°].
- If there is no solution, disable D4 and D7 for this athlete and log it.

**Solve 2 (knee pass):** bar at knee height, shins near vertical. This gives the predicted trunk angle at the knee pass. The **predicted change used by D2** is solve 2 − solve 1.

**Validation:**
- The reference is **measured** hip height (greater-trochanter marker, sagittal camera, ≤ 1 cm error) on setups the coach labels good.
- If the model's P95 error is ≤ 2.3 cm, D4 is cued from moderate.
- If it is ≤ 3.3 cm, D4 is cued at severe only.
- Otherwise D4 is disabled and the demo switches to plan B (§10 J8).

**Coupling:** D4's band assumes the shoulder sits at the centre of its band. If D4 and D7 fire together, D7 is cued first.

### 2.8 Drift channel and session reference

The profile supplies a deadlift `SessionReference`; the squat one (`session_reference.py:21-27`) is untouched.
- Session bests (lower is better): `bar_drift_cm`, `trunk_change_liftoff_knee_deg`, `hip_shift_abs`.
- Velocity reference: per set, the mean of the two fastest reps, the same rule as the squat (`session_reference.py:15-18`).
- `is_drift = True` when the athlete's best rep this session was under the mild threshold, i.e. the rep is worse than their own best. Drift faults become "same as your best rep" cues, never absolute corrections.

### 2.9 Error budget

A tier is cued only if the P95 measurement error against ground truth (§8.4) is at most 1/3 of that tier's threshold. Otherwise cueing starts one tier higher.

| Fault | Mild threshold | Max error for mild | Expected error (to verify) | Tier cued at the start |
|---|---|---|---|---|
| D1 | 3 cm | 1 cm | ≤ 1 cm (static, ≥ 15 frames) | mild |
| D3 | 3 cm | 1 cm | 1–2 cm | moderate until proven |
| D8 / D8b | 0.10 / 3 cm | 0.033 / 1 cm | 1–2 cm | moderate |
| D2 | 10° | 3.3° | 2–4° plus model error | moderate until proven |
| D5 / D6 | 8° | 2.7° | 2–3° (relative) | moderate unless ground truth error ≤ 1.5° |
| D9 | 15° | 5° | 3–5° | mild |
| D4 | 4 cm | 1.3 cm | §2.7 | moderate or severe, per §2.7 |
| D7 | 2 cm | 0.7 cm | 1–1.5 cm | moderate |
| D10 | 20 % | ~7 % | bar velocity noise below 15–20 % (FINDINGS) | mild if proven, else moderate |

### 2.10 Back rounding: honest proxies

- None of the 21 keypoints sits on the lumbar spine. halpe26 also outputs head, neck and pelvis-centre points, but `rtmpose.py:47-51` drops them.
- "Loss of position" = D2 + D3 + D7. The cues are behavioural; we make no medical claim and never say "flat back".
- **Experimental signals**, measured but not cued in v1:
  - shoulder↔hip chord shortening vs the standing length, averaged over the pull (per-frame noise ≈ 1.6–2.4 cm, `segment_lengths.py:19-21`);
  - the halpe26 neck and head points, read through a deadlift-side estimator mapping that leaves `rtmpose.py` untouched.
- A signal is only switched on if its AUC is ≥ 0.8 on lifters it has never seen.
- The existing `BackRoundingRule` is not used: it flags the normal trunk-angle change of a deadlift.

## 3. Platform integration — pipeline side

### 3.1 Profile

`src/biomechanics/profiles/deadlift.py` is rewritten as `DeadliftProfile`:

| Attribute | Value | Note |
|---|---|---|
| names | `deadlift`, `conventional_deadlift`, `barbell_deadlift` | `sumo_deadlift` is removed from this class (§3.2) |
| `movement_pattern` | `"deadlift"` | |
| `uses_diagnosis_engine` | `False` | The squat assessment, calibration and `HypothesisEngine` stay squat-only (`pipeline_process.py:303`, `:1014`; `calibration.py:19-28`). The deadlift gets its own set diagnosis (§7) through a new hook |
| `uses_bilstm_counter` | `False` | |
| `coaching_ready` | `False` until J8 | Dev override: `NOWVA_DEV_COACHING_READY=deadlift` adds it to `coaching_ready_profiles()` |
| `display_name` | `"conventional deadlift"` | |
| `create_fault_rules` | D1–D10 rules, **no `DepthRule`** | A `DepthRule` would switch on depth-gated counting (`pipeline.py:391-399`) |
| `create_rep_counter` | `DeadliftRepCounter` (§2.3) | |
| `get_rep_signal` | Bar or wrist height from the frame context; NaN when missing | |
| `get_fault_to_cue_map` | `deadlift_*` keys, plus `_left`/`_right` variants in `get_cue_dict` | |

### 3.2 Registry

Name matching is whole-word, with the longest key winning (`registry.py:40-55`). So "sumo deadlift" or "trap bar deadlift" would match `deadlift` today.
- A new `UntrackedVariantProfile(UntrackedProfile)` is registered for `sumo_deadlift`, `trap_bar_deadlift`, `hex_bar_deadlift`, `deficit_deadlift`, `snatch_grip_deadlift`, `single_leg_deadlift` and `rack_pull`. These count no reps and fire no faults, and Nova says they aren't coached yet.
- `romanian_deadlift`, `rdl` and `stiff_leg_deadlift` stay on the existing RDL profile, which is not coaching-ready.
- Tests:
  - every squat alias still resolves to `SquatProfile`;
  - none of these variants resolves to `DeadliftProfile`;
  - "deadlift", "deadlifts", "Barbell Deadlift" and "conventional deadlift" do resolve to it.

### 3.3 New hooks: additive, and the default is the current squat behaviour

| Hook (on `ExerciseProfile`) | Default | Deadlift | Call site |
|---|---|---|---|
| `needs_bar_3d: bool` | `False` | `True` | The provider and pipeline run the 3D bar tracker only when this is true |
| `observe_frame(ctx)` | no-op | Stores the world skeleton, bar state, foot state and gravity for `get_rep_signal` | `pipeline.py`, just before `get_rep_signal` (:974) |
| `create_rep_analyzer()` | `None` | `DeadliftRepAnalyzer` | `pipeline.py` (see below) |
| `create_session_reference()` | `None` (the engine builds `SessionReference`) | Deadlift reference (§2.8) | `RuleEngine(rules, reference=...)` |
| `create_set_diagnosis()` | `None` | Deadlift set diagnosis (§7) | `session_tracker.on_exercise_changed` |
| `allows_camera_refine: bool` | `True` | `False` | The drift refine between sets in `pipeline_process.py` |

How these behave in practice:
- **`create_rep_analyzer`.** When the profile returns an analyser, it receives each measured frame and produces the rep features. The squat-shaped blocks are then skipped: frame sample, bottom frame and trajectory (:976-1019), setup tracking (`_track_setup`/`_start_rep_setup`, :623-641) and `compute_rep_features` (:1075). When it returns `None`, those blocks run exactly as today.
- **`allows_camera_refine`.** A refine fitted on hinged poses, with plates hiding the feet, would degrade the shared rig calibration that the squat loads next. During deadlift sets the drift monitor still measures and warns, but no refine runs.
- **Plumbing.** `analysis_world` is threaded into the frame context. `PipelineFrame` gains an optional `bar_state_3d` field, which is `None` for the squat.
- **Squat standing reference.** `_standing_reference_hip_cm` is never raised by deadlift frames, because the analyser path skips `_start_rep_setup`. Test: squat → deadlift → squat leaves the squat reference unchanged.

### 3.4 Provider and bar tracking

`MultiCameraPoseProvider.get_pose()` returns only the primary frame (`multi_camera.py:307-357`).

**Provider change:** a new attribute `last_synced_frames` exposes the three synced frames of the last call. It is set in `get_pose()` after `_record_views`, so neither the return value nor any squat path changes.

**`deadlift/bar_tracker_3d.py`** reads those frames when `profile.needs_bar_3d` is true:
1. Detect on the three views in one batch and keep **all** candidates. This is a new detector class; `BarbellDetector` is untouched.
2. Undistort the points.
3. Associate candidates across views by minimum reprojection error, consistent with the hands and feet. This rejects a racked bar.
4. Triangulate both ends and smooth them with a 3D Kalman filter.
5. Output `BarState3D`: centre, height vs rest, forward offset vs midfoot, tilt, vertical velocity, source and confidence.

### 3.5 Global maps (additive entries only)

- `FAULT_CUE_PRIORITY` (`cue_cache.py:106-119`): append the `deadlift_*` ranks 20–30.
- `FAULT_MEASUREMENT` (`observability.py:41-54`): append the `deadlift_*` entries, plus a new measurement class `bar_3d` (not observable on one camera, observable when triangulated).
- `config/biomechanics.yaml`: a new top-level `deadlift:` section, parsed by the generic loader (`config.py:451`). The squat sections are unchanged.

### 3.6 Session tracker and diagnosis

`on_exercise_changed(diagnosis_enabled, set_diagnosis=None)`:
- When `set_diagnosis` is provided (deadlift), it runs at the end of each set, plus a rolling update after each rep.
- It sends `diagnosis_complete` in the same shape as the squat contract (`CONTRACT.md`).
- The squat path (`diagnosis_enabled` + `HypothesisEngine`) is unchanged.

## 4. Platform integration — delivery side

### 4.1 Contract

`.claude/deadlift/CONTRACT.md` mirrors `.claude/squat-audit/CONTRACT.md`: the same ownership split, the per-fault table (§2.6), the `details` keys and the IPC fields. It reuses the existing message types (`fault`, `rep_complete` with `features`/`highlights`/`faults_detailed`, `diagnosis_complete`, `cache_cues`). Nothing is renamed. The only new forwarded data is the bar fields inside `frame_data`.

### 4.2 Cues, text, audio

- **Text.** Every deadlift key needs text in `src/assets/cue_text/cues.json`, `CUE_TEXT_MAP` and `CUE_DISPLAY_LABELS`, plus `_fixed` praise in `FIXED_CUE_TEXT`.
- **Drafting.** `scripts/tools/draft_cue_text.py` is squat-only today (:52, :64-139) and would KeyError on a deadlift key (:222-223). It gains an `--exercise deadlift` mode with its own system prompt and `CUE_SCENARIOS`.
- **Audio.** Clips are generated with `generate_cue_audio.py` (ElevenLabs, flat folder, keys distinguished by prefix).
- **Every clip must exist on disk.** If one is missing, the `cache_cues` fallback synthesises it with cloud OpenAI TTS (`audio_cue_service.py:41-48`), and the fix-praise cue stays silent.
- The placeholder `deadlift_flat_back` and `deadlift_even` keys are removed.

### 4.3 Orchestrator and service: squat-only assumptions to dispatch

Each item below gets a dispatch whose default is the squat.

1. **Cue fallback.** When `cue_key` is empty, the orchestrator falls back to the squat `FAULT_TO_CUE_MAP` (`coaching_orchestrator.py:948-951`), as does the rest-complete focus line (`coaching_service.py:1210, :1230`), so a deadlift fault would get no cue. Fix: `cache_cues` carries the profile's `fault_to_cue` map and the orchestrator uses it. For the squat this map equals `FAULT_TO_CUE_MAP`, so nothing changes.
2. **`is_squat`** (`coaching_orchestrator.py:489-492`) gates both the diagnosis wait and the depth lines.
   - Add a `waits_for_diagnosis` flag, true for squat and deadlift.
   - Keep `is_squat` for the depth lines only.
   - Add a deadlift recap branch: bar speed, main fault, best rep.
3. **Adjustment monitor.**
   - `_latest_diagnosis` is never cleared when the exercise switches (:334, :600), so a squat stance diagnosis could arm the monitor during deadlifts. Gate the squat stance monitor on `is_squat`.
   - Add a deadlift **setup-guidance** parameter `bar_midfoot_cm`, published in `frame_data` during STANCE, with target 0 and tolerance 2 cm. Cues: `deadlift_closer` / `deadlift_back` → `adjust_good`.
   - It uses the same machinery and budget (`MAX_ADJUSTMENT_UTTERANCES`) as the squat stance monitor (:381-520).
4. **Squat-defined constants** (`coaching_orchestrator.py:79-104`): `SAFETY_FAULT_TYPE`, `SIDE_VIEW_FAULTS` and `SYMPTOM_FAULT_TYPES` get deadlift entries in separate sets, selected by exercise.
5. **Progress queries** pass `exercise=self._profile_name()`. No caller passes one today (`coaching_service.py:208-210`; `main_menu_agent.py:680-681`), so a deadlift session would get squat baselines. For the squat the profile name is `"squat"`, which is the current default.
6. **`progress_context.py` wording**: "last squat session" (:62, :168-171) becomes exercise-aware.

### 4.4 Menu, first session, knowledge

- **Menu.** It is already gated on `coaching_ready` (`main_menu_agent.py:207-220`; `main_menu_prompt.py:6-13`), so nothing changes until J8. Observation O4: the scheduled `start_workout` does not check `coaching_ready` (:89-166).
- **First deadlift session.** There is no squat assessment, because `uses_diagnosis_engine` is False. Instead, the `WorkoutAgent` gets a deadlift branch with:
  - a 5-step setup briefing: feet hip-width with the bar over midfoot; grip just outside the legs; shins to the bar; shoulders over the bar; take the slack out and push the floor;
  - the grip and plate questions (§4.6);
  - an empty-bar warm-up set, which feeds the setup model and the arm-length estimator.

  "First time" means there is no earlier deadlift session (`get_last_completed_session(exercise="deadlift")`).
- **Knowledge card.** A new `deadlift_card.py` plus an `explain_deadlift` tool, mirroring `squat_card.py`.
  - The tool is exposed only in sessions whose plan contains a deadlift, so squat-only sessions keep an identical tool list (pinned by a test).
  - `workout_prompt.py:155` says upper-back rounding is invisible on every setup. That stays true, and the deadlift card says the same.

### 4.5 Display and DB

- **Display** (`display.html`): `FOCUS_TEXT` gets `deadlift_*` entries, the "Avg depth" tile is hidden when `avg_depth` is null, and `SCORE_DIMS` is chosen per exercise.
- **Set summary**: `main.py` sends `avg_depth = null` for the deadlift.
- **DB.** There is already one session per exercise.
  - `deadlift_*` faults are judged at rep end with their own rep number, so they are **not** added to `END_OF_REP_FAULT_TYPES`.
  - `rep_kinematic_summary` and `features` carry `dl_schema: 1`.

### 4.6 Session metadata

Grip type (double/mixed/hook), plate diameter (default 45 cm), belt, shoes.
- Asked in `CollectExerciseInfoTask` (quick path) or in the deadlift `WorkoutAgent` greeting (scheduled path).
- Sent to the pipeline as an extra `meta` field on `set_exercise`, which is already relayed (`main.py:617-639`; `pipeline_process.py:1743-1748`).

## 5. Squat protection

| Layer | What | Status |
|---|---|---|
| Existing pins | `test_fault_priority.py:73-125`<br>`test_coaching.py:111-124` (`cues == SQUAT_CUES`)<br>`test_pipeline.py` `TestExerciseProfiles` :920-982, incl. the switch-back test<br>`test_pipeline_process.py` `TestSwitchExercise`<br>`test_coaching_orchestrator.py` recap snapshot :1199-1223<br>`test_coaching_service.py`:916-1000<br>`test_main_forwarding.py` | Keep green |
| New: squat golden master, fresh | Replay fixed squat scenarios through `BiomechanicsPipeline` (harness `test_pipeline.py:329-392`) + `SessionTracker`/`IPCBridge`. Snapshot frames, faults, `rep_complete.features` and IPC messages | J0 |
| New: squat golden master **after a deadlift switch** | Same scenarios, run after `set_exercise("deadlift")` → deadlift reps → `set_exercise("squat")`. Expected output = the fresh golden, minus the documented carry-overs: the depth target is carried, and `SessionReference` resets on every switch today | J0 |
| New: delivery golden master | Replay the squat IPC stream through `CoachingService` + `CoachingOrchestrator` (fake TTS/LLM/DB). Snapshot the cues, recap text and recorder operations | J0 |
| New: invariants | Squat rule order; the full `FAULT_TO_CUE_MAP` dict; squat `CUE_TEXT_MAP` strings; squat tool list | J0 |
| Intentional changes, J8 only | `test_profile_registry.py:58-59` (`coaching_ready == ["squat"]`) and `test_agent_prompts.py:116` (`"deadlift" not in` the menu prompt) | Changed in the J8 PR, reviewed by Ambaka |

## 6. Bar on the floor

**Today (§3.4):** a 2D YOLO11n-pose detector runs on the primary camera only. It is off by default (`config.py:221`), keeps only the best detection, and its weights are not in the repo.

**Keypoints:**
- loaded bar: the centre of the outer plate's outer face, on each side;
- empty bar: the sleeve ends, as a separate class;
- to be confirmed on the first real frames (J1).

**Data and training:**
- 2,000–3,000 frames from the three rack cameras, labelled in self-hosted CVAT.
- Coverage: bar on the floor, mid-pull and at the top; the racked bar as a negative; bumper and iron plates; different colours and lighting.
- Training with Ultralytics, then export to ONNX → TensorRT FP16.

**Targets:**
- recall ≥ 95 % with the bar on the floor;
- ≤ 1 % false positives on the racked bar;
- 3D error ≤ 1 cm static and ≤ 1.5 cm dynamic, measured against ArUco hub markers.

**Licence.** Ultralytics is **AGPL-3.0**. Shipping it in a sold device needs an Enterprise licence or a permissive alternative, such as RTMDet/RTMPose from MMPose (Apache-2.0). This must be decided **before J4**, and it also applies to the existing bar detector.

**Wrist proxy without a bar:** the mid-wrist minus the learned offset.
- D1, D3 and D8b then run with `bar_source = wrist_proxy`, wider thresholds, and a moderate minimum tier.
- D10 is not emitted.

## 7. Set diagnosis (static graph)

The method is the squat's: symptoms → causes → hand-set prior × evidence → leak → noisy-OR. The machinery follows the platform patterns: median rep, tail-mean fatigue, measurement confidence and observability blanking (FINDINGS D1, `engine.py`).

**Decision for Ambaka (§11 Q1):**
- **(a) recommended:** parameterise `HypothesisEngine` with its graph, defaulting to the squat graph. It is pinned by `test_engine.py` (730 lines) and the golden masters.
- **(b)** a separate `deadlift/diagnosis/engine.py` (~150 duplicated lines).

| Symptom | From | Causes (tier) |
|---|---|---|
| Bar not over midfoot | D1 | feet position (1, "move X cm closer"); habit (1) |
| Setup hip height | D4 | hips too low/high (1, "hips X cm up/down"); hip/hamstring mobility (3); anthropometry context (0) |
| Shoulders behind bar | D7 | shoulder position (1) |
| Hips shoot | D2 | hips too low at setup (1); slack not pulled (1); weak off the floor (3); load too heavy (2) |
| Bar drift | D3 | bar too far at setup (1); lats not engaged (1, "push the bar into your legs") |
| Incomplete lockout / lean-back | D6 / D5 | glutes not finishing (1); over-extension habit (1) |
| Hip shift / bar tilt | D8 / D8b | uneven stance or grip (1, "shift grip X cm"); unilateral weakness (3) |
| Velocity loss | D10 | fatigue → load (2, "about 10 % lighter") |

- Outputs stay small: `cause_id`, tier, score, **one** numeric delta, one sentence.
- The recap formats the delta itself, because `summarize_cue_magnitude` is squat-only (`demo_builder.py:77-79`).
- Rep score weights: setup 25 %, coordination 25 %, bar path 20 %, lockout 15 %, symmetry 15 %. As in `rep_scoring.py:324-342`, dimensions that weren't measured are dropped from the score.

## 8. Data

### 8.1 Static knowledge

`docs/deadlift/KNOWLEDGE.md`: the phases; each metric's frame, sign and window; thresholds with their rationale; cue text; and the setup model. Reviewed by Ambaka and, ideally, by an external strength coach.

### 8.2 Raw recorder and replay (a J1 prerequisite)

- `scripts/tools/record_rig.py` writes the three synced videos, timestamps, a copy of the calibration and gravity files, and the session metadata. It writes nothing to `~/.nowva`.
- A replay provider feeds recordings through the real pipeline: the deadlift offline, and squat regression replays.

### 8.3 Real capture and test design

**Round 1 (team, weeks 1–2):**
- 2–3 people, from an empty bar up to light loads.
- Clean reps and scripted, safe faults. Rounding is only shown with an empty bar or a PVC pipe.
- Also measures ρ, the within-lifter correlation.

**Round 2 (pilot):**
- ≥ 10 lifters, 1.55–1.95 m tall, with varied femur/torso/arm ratios, at least 4 of them novices.
- Natural sets in **observation mode** (cues off), plus scripted sets.

**Test set (clustered design):**
- Design effect = 1 + (m − 1)ρ, starting from ρ = 0.1 and recomputed after round 1.
- At least 10 test lifters × at least 15 positives per hero fault (≥ 150 per hero fault). Effective lower bounds: ≈ 0.74 for precision, ≈ 0.58 for recall.
- Provisional faults: at least 5 positives per lifter.
- At least 30 clean reps per lifter.
- About 1,150 reps in total, over 2 sessions per lifter.

**Statistics at the J6 gate:**
- lifter-clustered bootstrap and bootstrap-t;
- leave-one-lifter-out;
- a per-lifter floor of 0.70, applied only to lifters with ≥ 15 cues;
- scripted and natural sets reported separately, with ≥ 0.80 required on ≥ 30 natural cases.

Plan B is accepted in advance: if D2 doesn't get enough natural cases, the demo uses D1 + D3.

**Labelling** (CVAT, three views):
- faults and severity per rep;
- event timestamps on a subset;
- 20 % double-labelled; target κ ≥ 0.6 per fault; Ambaka adjudicates disagreements; any fault below κ 0.6 is redefined.

**Training and tuning:** at least 300 reps and 50 positives per fault, on top of the test set, plus 2,000–3,000 bar frames.

**Consent and storage:** written consent, anonymised, stored locally.

### 8.4 Ground truth

| Quantity | Method | Target |
|---|---|---|
| Static bar position / D1 | Floor tape + a foot jig (0 / 3 / 6 / 10 cm) | ≤ 3 mm |
| Bar in motion (D3, D10) | ArUco markers on the plate hubs, triangulated; optionally a linear position transducer | ≤ 5 mm |
| Trunk, hip and knee angles (D2, D5, D6) | A calibrated planar sagittal camera at 120 fps, with markers on the greater trochanter and acromion (the same line the vision metric uses). IMUs are used only for timing and cross-checks: a back IMU measures the thoracic segment, and a thigh IMU moves with the skin | ≈ 1–1.5° |
| Setup hip height (D4) | Greater-trochanter marker on the sagittal camera | ≤ 1 cm |
| Gravity | Flat board + spirit level | ≤ 0.3° |

### 8.5 Synthetic data

- We extend the ground-truth simulator in `.claude/preik-audit/harness/preik_harness/`, after fixing `__init__.py:13` and `runner.py:404`.
- **Deadlift motion generator:** starts on the floor with the hands coupled to the bar. `barbell.py` currently puts the bar on the back; the deadlift needs it on the floor. It also models plate occlusion, a tilted world and a camera that has moved.
- **Scenarios:**
  - D1–D10;
  - soft and over-extended lockouts, which must be counted;
  - a stalled rep at the knees;
  - dead stops, quick re-pulls, touch-and-go (sets of 5 and 10), bumper bounce, dropped bar, re-setup;
  - several body builds;
  - a noisy tracker.
- **Assertion:** every rep is counted exactly once.
- Synthetic data is never used to set the final thresholds.

## 9. Edge

- **J1 measurement** on a Jetson Orin Nano Super: RTMPose on 3 views + the bar detector on 3 views (TensorRT FP16) + the full chain.
- The new geometry, state machine and diagnosis add < 1 ms.
- **Degraded mode:** pose at 30 Hz and the bar at 15 Hz with the Kalman filter; if that isn't enough, everything at 15 Hz.
- The deadlift adds no cloud dependency. The conversational stack is cloud today (observation O3).

## 10. Milestones

Estimates are for one engineer. With two, J2/J4 and J3/J4 can run in parallel.

| | Content | Acceptance | Est. |
|---|---|---|---|
| **J0 Squat net** | Squat golden masters (fresh, after a switch, delivery); invariants; `requirements.lock` environment; CI proposal; licence decision | All green on `main`; changing one squat threshold or one cue text fails the right test | 4–6 d |
| **J1 Foundations** | Recorder + replay; gravity tool, deadlift frame and sign/tilt tests; `KNOWLEDGE.md`; setup model (2 solves); round-1 capture; Jetson numbers; existing bar weights evaluated | Review signed; tests green; ≥ 1 h of raw capture | 2 wk |
| **J2 Simulator** | Deadlift generator and scenarios | Each scenario carries ground truth | 1 wk |
| **J3 Deadlift profile on the platform** | Profile + registry variants; hooks (§3.3); analyser; rep counter; D1–D10 with the wrist proxy; deadlift `SessionReference`; global-map entries; deadlift golden master. **Thin end-to-end slice:** coaching-ready via the dev override, voice counts reps on the rack | Simulator: every rep counted exactly once (incl. touch-and-go and bumper bounce), events ≤ 100 ms, each injected fault detected, clean scenario fault-free; squat goldens unchanged | 2.5 wk |
| **J4 3D bar** | Annotation, training, export, tracker, provider frames, ArUco ground truth | Recall ≥ 95 %; 3D error ≤ 1 / 1.5 cm; Jetson budget met | 2–3 wk |
| **J5 Delivery integration** | Contract; cue text and audio; orchestrator dispatches (§4.3); setup guidance; recap; display; DB; metadata; first-session flow; knowledge card | Full rack session (first time and returning); no cue clip synthesised at run time; squat goldens (pipeline + delivery) unchanged | 2 wk |
| **J6 Real validation** | Round 1 → thresholds, error budget, margins, ρ; round 2 → test set | Demo gate (§1) on ≥ 10 unseen lifters; κ ≥ 0.6; status decided for each fault | 3–4 wk |
| **J7 Set diagnosis** | Graph (option a or b), scoring, recap | Top cause matches the coach's label on ≥ 70 % of sets | 1–1.5 wk |
| **J8 Demo + coaching_ready** | Flip `coaching_ready`; update the two intentional tests; hero scenario: closed-loop D1, then D4/D2/D3 fixed on the next rep; plan B: D1 + D3 | 10 demos in a row without error; Jetson budget met | 1 wk |

About 16–18 weeks for one person, or 9–10 for two. Real capture starts at J1, and a voice end-to-end slice exists from J3.

## 11. Decisions for Ambaka

1. Set diagnosis: parameterise `HypothesisEngine` (recommended), or a separate module?
2. Camera refine disabled during deadlift sets (`allows_camera_refine=False`): OK?
3. Bar detector licence: Ultralytics Enterprise, or an Apache-2.0 alternative?
4. Who labels the reps?
5. Touch-and-go counted in v1? Proposed: yes, with no cue between those reps.
6. Where are the current bar detector weights, and what were they trained on?
7. An accelerometer in the device, for a permanent gravity reading?
8. What plates are in the demo gym?

## 12. Risks

| Risk | Mitigation |
|---|---|
| Squat changed by shared-code edits | Hooks that default to the squat path; goldens fresh and after a switch; delivery golden; invariants |
| Rig calibration degraded by deadlift frames | `allows_camera_refine=False` |
| Vertical axis is not gravity | Measured gravity, per-camera consistency check, tilt tests, wider thresholds in the fallback |
| Back rounding can't be measured | Proxies only; behavioural wording; experimental signals gated on data |
| Pose quality in a deep hinge; occlusion | Measured in round 1; midfoot locked before the pull |
| Bar detector doesn't generalise; licence | Varied data, split by lifter, negatives; licence decided before J4 |
| Thresholds are invented | Initial values only; error budget; real data before the demo |
| Validation is too optimistic | Clustered design, natural sets reported separately, per-lifter floor |
| A cue is dropped while Nova is speaking | Adjustment-monitor budget and retries, as for the squat |
| A variant name resolves to the conventional deadlift | Untracked variant aliases + tests |

## 13. Observations on the current code (outside this plan, not changed)

- **O1** The world vertical is the hip→ankle axis, not gravity, and the squat's trunk pitch inherits that bias (`person_calibration.py:734-752`).
- **O2a** `rep_data.features` is dumped (`pipeline.py:1076`) before `finish_rep` fills in `velocity_loss_pct` (`rule_engine.py:238-239`). On the non-BiLSTM path, the IPC/diagnosis copy therefore keeps NaN.
- **O2b** `HeelRiseRule` emits `affected_side`, not `side` (`heel_rise.py:107-112`), so `heels_down_left/right` is never chosen (`ipc_bridge.py:183`).
- **O3** The conversational stack is cloud (Deepgram, OpenAI `gpt-5.4-mini`, ElevenLabs; `pipeline_factory.py`), contrary to guardrail #1. Missing cue clips fall back to cloud OpenAI TTS at cache time (`audio_cue_service.py:41-48`).
- **O4** The scheduled `start_workout` does not check `coaching_ready` (`main_menu_agent.py:89-166`).
- **O5** The cue-audio script's default ElevenLabs voice (`generate_cue_audio.py:55`) differs from the live agent's (`pipeline_factory.py:43`) unless `ELEVENLABS_VOICE_ID` is set.
- **O6** `_latest_diagnosis` survives exercise switches (`coaching_orchestrator.py:334, :600`).
- **O7** The `set_exercise` display event sends `weight_lbs` without converting from kg (`coaching_service.py:1330`).
