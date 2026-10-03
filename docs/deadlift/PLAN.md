# Conventional deadlift — plan v12 (built on the multi-exercise platform)

Status: proposal for Ambaka. Version 12. Replaces v1–v9, which assumed a separate deadlift subprocess, plus v10 and v11. Since PR #20 (`squat-v1-integration`), every exercise runs in one pipeline process as a swappable profile, and this plan plugs the deadlift into that platform.

## 0. Decisions

| Decision | Consequence |
|---|---|
| The deadlift plugs into the multi-exercise platform (Ambaka, 2026-10-03) | One pipeline process; the profile is activated at startup or swapped via `set_exercise` |
| Squat behaviour must not change | Every shared-code change is a dispatch whose default is today's squat path. Golden-master tests prove the squat unchanged, fresh and after switching to a deadlift and back. Shared state a deadlift could leave behind is listed and pinned (§5.2) |
| Conventional deadlift only | Sumo, trap-bar, deficit, snatch-grip, single-leg and rack pulls resolve to an untracked profile (§3.2) |
| The deadlift is **unreachable** until the demo gate passes | Profile resolution maps deadlift names to `UntrackedProfile` unless the deadlift is coaching-ready or the dev override is set. Scheduled programs ("Barbell Conventional Deadlift", `exercise_library.py:85`) therefore cannot run unvalidated rules (§3.1) |
| Back rounding: indirect proxies only | Never claimed, never cued as if measured (FINDINGS honesty principle) |
| The bar must be detected on the floor | 3D bar tracking from all three cameras (§6) |
| LLM learning out of scope | Sessions, reps and cue events are already tagged per exercise (one DB session per exercise) |
| Thresholds are absolute and never moved by observed reps | Same two channels as the squat (FINDINGS D2): an absolute standard, plus drift against the athlete's best rep this session |

## 1. Goal and gates

Demo loop:

> "Let's deadlift" → while standing at the bar, Nova guides the feet in closed loop ("bar over midfoot — a bit closer… good") → the athlete sets up and pulls → the rep is judged as a whole → **one** cue at the floor before the next rep → the next rep improves → set recap with numbers.

**Demo α** comes first: D1 bar position (closed loop) and D3 bar drift, plus rep counting. It needs three things: the 3D bar (J4), the profile (J3, accepted on the J2 simulator), and a delivery slice J5α (closed-loop mode, D1/D3 cue audio, contract fields). With two engineers (J4 in parallel with J2→J3) it lands around week 8; with one, around week 10. The **full hero set** adds D2 hips shoot and D4 setup hip height.

Every other v1 fault is "provisional": it is cued only at its minimum tier (§2.9) until it has enough data.

| Metric (real data, lifters never seen in training/tuning) | Demo gate (J6) | Launch gate (later) |
|---|---|---|
| Rep counting | ≥ 99 %, 0 phantom reps on clean sets | same |
| Event timing (liftoff, knee pass, top, floor) | median error ≤ 100 ms | same |
| Hero-fault precision | ≥ 0.85; 95 % lower bound (lifter-clustered bootstrap) ≥ 0.70; ≥ 0.80 on natural sets | lower bound ≥ 0.75 |
| Hero-fault recall | ≥ 0.70; lower bound ≥ 0.55 | lower bound ≥ 0.60 |
| False corrections on clean reps | ≤ 1 per 10 reps | same |
| Measurement error (P95 vs ground truth) | ≤ 1/3 of the threshold of the lowest cued tier (§2.9) | same |
| Squat | golden masters identical, fresh and after a deadlift switch (§5) | same |
| Edge | ≤ 33 ms/frame on Jetson Orin Nano Super, or the defined degraded mode (§9) | same |

The statistical analysis plan (metrics, bootstrap scheme, exclusions) is pre-registered in `docs/deadlift/VALIDATION.md` before round 2 is labelled.

## 2. What we measure and how

### 2.1 Frame and axes

**Platform conventions** (`.claude/rules/biomechanics.md`):
- 3D arrays are Y-down. Up is `geometry.WORLD_UP` = (0, −1, 0); use `height_above` / `is_above`.
- Foot frame (`rep_features._foot_frame`, `rep_features.py:141-177`):
  - lateral = horizontal hip line;
  - forward = lateral × `WORLD_UP` (`:169`), flipped to point heel → toe;
  - midfoot = ankle + 0.35 × (ankle→toe), averaged over both feet (`:173-176`).

**Gravity.**
- World Y is the median standing hip→ankle direction (`person_calibration.py:734-752`), not gravity. It can be off by 2–4°, and 3° over 55 cm of bar travel fakes ≈ 2.9 cm of drift, which is the D3 threshold.
- The deadlift analyser therefore uses a measured gravity vector:
  - Source: the ChArUco board laid flat. It is solved per camera, reusing charuco's existing `flat_placement_index` support (`charuco.py:465`, `:565-571`) and `solve_board_pose` (`:334`), wrapped in `scripts/tools/measure_gravity.py`.
  - Stored in `~/.nowva/gravity_<camera_key>.json`, in each camera's own frame. The tool writes nothing else.
  - At run time it is mapped into the current world frame through the calibration rotations. A camera that disagrees by more than 1° is dropped (it moved).
- Fallback: `WORLD_UP` with `gravity_source="body"`. D2, D3 and D5 are then cued only from moderate.
- Squat code keeps `WORLD_UP` (observation O1).

**Deadlift sagittal frame** (`src/biomechanics/deadlift/frame.py`):
- up = gravity; lateral = bar axis at rest (or the hip line) projected horizontal; forward = heel → toe.
- It provides signed sagittal angles and forward/lateral offsets. The deadlift needs a **signed** sagittal trunk angle (for D5); the squat's trunk pitch is unsigned and includes sideways lean (`analytical_ik.py:254`).

**All deadlift metrics are differences** (bar vs its resting height, bar vs midfoot, wrists vs their standing height). None depends on an absolute floor.

### 2.2 Inputs per frame, and time alignment

| Input | Source | Today |
|---|---|---|
| World-frame skeleton (21 kpts) | `PreIKResult.analysis_world` (`preik_chain.py:136-141`) | computed, then dropped before profiles and rules (§3.3 threads it through) |
| 3D bar state | new `deadlift/bar_tracker_3d.py` (§6) | does not exist |
| Foot state | `FootState` (`foot_contact.py:165-185`) | already in the rule frame context |
| Joint angles | `AnalyticalIK` | available |
| Gravity | §2.1 | new |

**Time alignment.** The analysis skeleton lags capture by `kalman.lag_frames = 2` (`config.py:289`; `pipeline.py:938-947`). The bar tracker works on capture-time frames, so its states go into a ring buffer keyed by capture timestamp. The analyser reads the bar state matched to `analysis.timestamp` (nearest within half a frame; NaN otherwise). Without this, D2, liftoff timing and shoulder-vs-bar would mix samples ≈ 67 ms apart. A unit test checks the alignment.

### 2.3 Phases and rep counting (driven by the bar)

**Phases:** `APPROACH` (standing, away from the bar) → `STANCE` (standing over the bar, feet planted) → `SETUP` (hinged, hands on the bar, still) → `PULL` (liftoff → knee pass → top) → `TOP` → `LOWER` → `FLOOR`. From `FLOOR` the athlete goes back to `SETUP`, straight to `PULL` (touch-and-go or quick re-pull), or back to `APPROACH`.

**Rep signal:** bar-centre height above its resting height. Fallback without bar tracking: mid-wrist height above its height in `SETUP`.

**`DeadliftRepCounter`** comes from `create_rep_counter`. It implements the counter API the pipeline uses: `in_rep`, `phase`, `rep_count`, `rep_started`, `update`, `reject_last_rep`, `clear_current_faults`, `snapshot_rep_metrics`, `rejected_rep_max_depth_angle`, `reset`, `set_assessment_mode`. It never uses the squat's IDLE baseline EMA (`hip_position_counter.py:283-291`).

| Transition | Condition (initial values in the `deadlift:` config section, §3.5) |
|---|---|
| APPROACH → STANCE | Standing still ≥ 0.5 s, facing the bar (foot forward axis toward it), with the bar ≤ 40 cm ahead of the midfoot horizontally. Within that range the closed loop (§4.3) guides the feet in: a coarse "step closer" beyond 15 cm, then fine cues |
| STANCE → SETUP | Hands within 10 cm above the bar, still ≥ 0.3 s |
| SETUP → STANCE / APPROACH | Stands back up without lifting |
| SETUP → PULL | Bar > rest + 3 cm and vertical velocity > 0.10 m/s |
| PULL → TOP | Bar ≥ expected top height − 8 cm, \|v\| < 0.05 m/s for ≥ 3 frames, trunk within 35° of vertical |
| PULL → FLOOR without TOP | **Failed rep** (bar rose ≥ 10 cm but never reached top − 8 cm): an event, not counted |
| TOP → LOWER | v < −0.10 m/s |
| LOWER → FLOOR | **Dead stop**: bar ≤ rest + 2 cm and \|v\| < 0.02 m/s for ≥ 3 frames, or bar within rest + 2 cm for ≥ 0.3 s → **rep counted and judged** |
| LOWER → PULL | **Touch-and-go**: low point within 5 cm of rest, then a rise of > 3 cm above that low point, with upward velocity sustained ≥ 100 ms and hands on the bar → **the finishing rep is counted and judged now**. The low point is the next rep's liftoff, and `rep_started` is set (§3.3 explains how the pipeline's touch-and-go block, `pipeline.py:1115-1121`, routes to the analyser) |
| FLOOR → SETUP | Dead stop with hands still on the bar ≥ 0.3 s → a new setup is judged |
| FLOOR → PULL | Quick re-pull: the setup is judged on the last 0.2 s before liftoff, or marked "not measured" |
| FLOOR → STANCE / APPROACH | Stands up or steps back (the set may end) |

- **Liftoff is back-dated** to motion onset: the last frame in a 0.5 s buffer where the bar was within 0.5 cm of rest and moving < 0.02 m/s.
- **Expected top height** = standing mid-wrist height (arms hanging, measured in `APPROACH`) minus the learned wrist → bar offset.
  - *Fallback* when the athlete never stands still with arms hanging: the median top height of the set's counted reps (the first rep uses top − 8 cm against its own peak).
  - The 8 cm margin counts soft and over-extended lockouts; D6 and D5 then judge them.
- A bumper bounce (≤ 3 cm, no pull) is not a touch-and-go. Each rep is counted exactly once. A bar dropped from the top counts, with no fault.
- **Set boundaries.**
  - Pipeline: the existing `set_timeout_seconds` and `rest_start`.
  - Voice: the orchestrator ends a set after `SET_IDLE_TIMEOUT_S = 15 s` with no rep (`coaching_orchestrator.py:108`), which a deadlift re-setup or foot guidance can exceed. The deadlift gets 30 s through a per-profile `set_idle_timeout_s`, carried in `cache_cues` (§4.3).

### 2.4 Standing references and the midfoot lock

- **Standing references.** While standing still in `APPROACH` (≥ 1 s) the analyser records the signed sagittal hip, knee, trunk and elbow angles, and the standing mid-wrist height. D5 (lean-back), D6 (lockout) and D9 (bent arms) are measured against these, which removes per-person keypoint bias.
- **Midfoot.**
  - It is tracked **live during `STANCE`**; the closed-loop guidance (§4.3) uses this live value.
  - It is **locked when `SETUP` starts**, when the feet are final, by averaging the last ≥ 15 still frames of `STANCE`.
  - The pull is then judged against the locked midfoot, so plate occlusion during the pull does not matter.
- **Squat state stays untouched.** The squat setup snapshot and `_standing_reference_hip_cm` are never touched by deadlift reps (§3.3).

### 2.5 Per-rep features (`DeadliftRepFeatures`)

There is one feature vector per rep, built from measured frames only. NaN means "not measured", never 0. The robust statistics follow `rep_features.py:24-58`.

| Feature | Definition | Window |
|---|---|---|
| `bar_midfoot_stance_cm` | forward offset of bar centre vs live midfoot, median | last 0.3 s of STANCE |
| `bar_midfoot_setup_cm` | forward offset of bar centre vs the locked midfoot, median | last 0.3 s of SETUP |
| `setup_hip_height_cm` | hip height above ankle midpoint | SETUP |
| `setup_hip_band_cm` | expected band from the setup model (§2.7) | — |
| `shoulder_vs_bar_cm` | forward offset of the shoulder joint vs bar centre | SETUP |
| `trunk_change_liftoff_knee_deg` | signed trunk-angle change from back-dated liftoff to the bar at knee height, minus the model-predicted change | early pull |
| `hip_shoulder_rise_ratio` | hip vertical rise ÷ shoulder vertical rise | same window |
| `bar_drift_cm` | p90 forward deviation of the bar centre from its liftoff position | pull |
| `hip_extension_deficit_deg`, `knee_extension_deficit_deg` | signed deficits at the top vs the standing reference | TOP |
| `lean_back_deg` | signed backward trunk angle at the top vs the standing reference | TOP |
| `hip_shift_ratio` | (hip_mid − locked midfoot) on the ankle axis ÷ ankle separation; sustained 80th-percentile deviation vs start median (squat method, `rep_features.py:371-390`) | pull |
| `bar_tilt_cm` | p90 height difference between the two 3D bar ends | pull |
| `elbow_flexion_deg` | p90 elbow flexion vs the standing reference | pull |
| `concentric_velocity_ms` | bar rise ÷ time from liftoff to top (the true bar, not the squat's shoulder proxy) | pull |
| `pull_time_s`, `lower_time_s` | phase durations | — |
| `bar_source`, `gravity_source` | `bar` / `wrist_proxy`; `measured` / `body` | — |
| `dl_schema` | `1` | — |

### 2.6 Faults (v1)

**Fault types.** All are new `FaultType` values prefixed `deadlift_`, so none shares the squat's global priority, observability or cue entries.

**Judging and details.**
- Each fault is a `RepFaultRule`, judged once per rep through `RuleEngine.finish_rep` (`rule_engine.py:226-253`).
- `_rep_fault` (`fault_types.py:297-328`) stamps `side`, `phase`, `is_drift`, `value` and `unit`.
- `observability` is stamped by `RuleEngine._tag_observability` (`rule_engine.py:176-182`) from `FAULT_MEASUREMENT`.

**Minimum cue tier.** "Min tier" below is enforced as described in §4.3.

| # | fault_type | Meaning | Measured from | Thresholds (mild / moderate / severe) | Min tier at start | Observability | Cue key | Prio |
|---|---|---|---|---|---|---|---|---|
| D1 ★α | `deadlift_bar_position` | Bar not over midfoot at setup | `bar_midfoot_setup_cm` (the stance value drives the closed loop) | 3 / 5 / 8 cm | mild | `bar_3d` (new class) | `deadlift_bar_midfoot` | 20 |
| D4 ★ | `deadlift_setup_hips` | Hips too low / too high at setup | `setup_hip_height_cm` vs band | outside band by 4 / 7 / 10 cm | §2.7 | `side_view` | `deadlift_hips_up` / `deadlift_hips_down` | 21 |
| D7 | `deadlift_shoulders_behind` | Shoulders behind the bar at setup | `shoulder_vs_bar_cm` | 2 / 4 / 6 cm | moderate | `side_view` | `deadlift_shoulders_over` | 22 |
| D2 ★ | `deadlift_hips_shoot` | Hips rise before the chest off the floor | `trunk_change_liftoff_knee_deg`; cross-check ratio > 1.4 | 10 / 15 / 20° | moderate | `side_view` (≥ 30 fps) | `deadlift_chest_with_hips` | 23 |
| D3 ★α | `deadlift_bar_drift` | Bar drifts away from the legs | `bar_drift_cm` | 3 / 5 / 8 cm | moderate | `bar_3d` | `deadlift_bar_close` | 24 |
| D6 | `deadlift_lockout` | Incomplete lockout | hip or knee extension deficit | 8 / 12 / 20° | moderate | `side_view` | `deadlift_lockout` | 25 |
| D5 | `deadlift_lean_back` | Over-extension at the top | `lean_back_deg` | 8 / 12 / 18° | moderate | `side_view` | `deadlift_finish_neutral` | 26 |
| D8 | `deadlift_hip_shift` | Hips shift sideways (side = direction) | `hip_shift_ratio` | 0.10 / 0.15 / 0.22 | moderate | `lateral_travel` | `deadlift_even_feet` (+ `_left` / `_right`) | 27 |
| D8b | `deadlift_bar_tilt` | Bar tilts | `bar_tilt_cm` | 3 / 5 / 7 cm | moderate | `bar_3d` | `deadlift_level_bar` | 28 |
| D9 | `deadlift_bent_arms` | Arms bend during the pull | `elbow_flexion_deg` | 15 / 25 / 35° | mild | `side_view` | `deadlift_long_arms` | 29 |
| D10 | `deadlift_velocity_loss` | Bar speed dropped vs the set's two fastest reps | `concentric_velocity_ms` | 20 / 30 / 40 % (Sánchez-Medina 2011) | moderate | `bar_3d` | `deadlift_drive` | 30 |

★ = hero fault; α = in Demo α.

- **Cue text.** Cue keys get ≤ 4-word external-focus text from the cue pipeline (§4.2). Drafts: "Bar over midfoot", "Bar close", "Chest and hips together", "Stand tall", "Long arms".
- **Priority.** Ranks 20–30 are appended to `FAULT_CUE_PRIORITY`. Squat ranks 0–11 and `DEFAULT_FAULT_CUE_PRIORITY = 12` stay unchanged.
- **Drift.** D2, D3 and D8 also report `is_drift` against the session best (§2.8).
- **Mixed grip.** Trunk rotation is not part of D8, because a mixed grip rotates the trunk. The grip type is recorded (§4.6).
- **No `deadlift_flat_back` cue.** The placeholder mapping `back_rounding → deadlift_flat_back` (`profiles/deadlift.py:92-96`) is removed (§2.10).
- **v1.1 faults.**
  - jerking the bar off the floor (bar acceleration spike);
  - uncontrolled lowering;
  - knees forward too early on the way down;
  - hitching;
  - head position;
  - grip and stance width;
  - heel rise;
  - foot placement (the squat's stagger/flare method, `rep_features.py:295-317`).

### 2.7 Setup model (core IP)

The deadlift counterpart of the squat lean model (`diagnosis/lean_model.py`): the right start position for **this** body.

**Constraints (bands):**
- bar over midfoot;
- shoulder joint 0–6 cm in front of the bar;
- arms near vertical (≤ 5° back);
- shins touching the bar.

**Inputs:**
- tibia and femur lengths **projected into the sagittal plane**, measured on SETUP frames;
- torso and foot lengths (`SegmentLengthEstimator`, kept across switches);
- arm length (shoulder → elbow → wrist), from a deadlift estimator, since `SegmentLengthEstimator` has no arms (`segment_lengths.py:53-64`);
- wrist → grip offset, 6–9 cm, learned per athlete;
- bar-axis-to-shin distance at contact, initially 5 cm;
- measured bar and ankle positions.

All relative to the ankle; no floor needed.

**Solve 1 (setup).**
- Place the shoulder in its band and fix the ankle, which leaves one degree of freedom.
- Shin contact then gives two solutions. Keep the one with the knee in front of the ankle and flexion in [40°, 120°].
- If there is no solution, D4 and D7 are disabled for this athlete and the case is logged.

**Solve 2 (knee pass).** Bar at knee height, shins near vertical. This gives the predicted trunk angle, and **D2's predicted change** is solve 2 − solve 1.

**Validation.**
- The reference is the **measured** hip joint height, from a greater-trochanter marker on the sagittal camera (≤ 1 cm), on setups the coach labels good.
- The error budget combines the model error **and** the vision hip-keypoint bias in deep flexion: the keypoint sits above or behind the trochanter when the hips are flexed, and the round-1 data measures that offset.
- Cueing depends on the total P95 error:

| Total P95 error | D4 cueing |
|---|---|
| ≤ 2.3 cm | from moderate |
| ≤ 3.3 cm | severe only |
| larger | disabled, and the demo uses Demo α (D1 + D3) |

**Coupling.** D4's band assumes the shoulder at the centre of its band. If D4 and D7 co-fire, D7 is cued first.

**Timing.** D4 and D7 are judged at liftoff and cued at the floor, so they apply to the next rep. Cueing during the setup hold is deferred to v1.1: the hold is often under 1.5 s, and a mid-setup cue risks a reset.

### 2.8 Drift channel and session reference

`DeadliftSessionReference` **subclasses `SessionReference`**, so the depth-target API stays intact: `depth_target_ratio`, `set_depth_target` and `reaches_depth_target`. These are read on every switch and at calibration (`pipeline.py:586`, `:1077`; `calibration.py:67`, `:80`; `pipeline_process.py:1065`, `:1748`). The subclass only overrides which metrics it tracks:
- Session bests (lower is better): `bar_drift_cm`, `trunk_change_liftoff_knee_deg`, `hip_shift_abs`.
- Velocity reference: per set, the mean of the two fastest reps (`session_reference.py:15-18`).

`is_drift = True` when the best rep this session was below mild. Drift faults become "same as your best rep" cues.

`test_pipeline.py:965-983` (the depth target survives a switch) is extended to a squat → deadlift → squat round trip: the squat's calibrated target must come back unchanged.

### 2.9 Error budget and minimum cue tier

A tier is cued only if the P95 measurement error against ground truth (§8.4) is ≤ 1/3 of that tier's threshold. Otherwise cueing starts one tier higher. That is the "min tier" column in §2.6, re-set from data at J6.

| Fault | Mild | Max error for mild | Expected error (to verify) | Min tier at start |
|---|---|---|---|---|
| D1 | 3 cm | 1 cm | ≤ 1 cm (static, ≥ 15 frames) | mild |
| D3 | 3 cm | 1 cm | 1–2 cm | moderate |
| D8 / D8b | 0.10 / 3 cm | 0.033 / 1 cm | 1–2 cm | moderate |
| D2 | 10° | 3.3° | 2–4° + model error | moderate |
| D5 / D6 | 8° | 2.7° | 2–3° (relative) | moderate unless ground truth ≤ 1.5° |
| D9 | 15° | 5° | 3–5° | mild |
| D4 | 4 cm | 1.3 cm | model + hip-keypoint bias (§2.7) | per §2.7 |
| D7 | 2 cm | 0.7 cm | 1–1.5 cm | moderate |
| D10 | 20 % | ~7 % | bar-velocity noise below 15–20 % (FINDINGS) | moderate until proven |

### 2.10 Back rounding: honest proxies

- None of the 21 keypoints is on the lumbar spine. halpe26 also outputs head, neck and pelvis-centre, but `rtmpose.py:47-51` drops them.
- "Loss of position" is D2 + D3 + D7. Cues are behavioural only: nothing medical, and no "flat back" claim.
- Experimental signals are measured but not cued in v1:
  - shoulder ↔ hip chord shortening vs the standing length, averaged over the pull (per-frame noise ≈ 1.6–2.4 cm, `segment_lengths.py:19-21`);
  - halpe26 neck and head points, via a deadlift-side estimator mapping that leaves `rtmpose.py` untouched.
- A signal is enabled only if its AUC is ≥ 0.8 on unseen lifters.
- `BackRoundingRule` is not used: it flags the normal trunk-angle change of a deadlift.

## 3. Platform integration — pipeline side

### 3.1 Profile and gating

Rewrite `src/biomechanics/profiles/deadlift.py` as `DeadliftProfile`. It is **stateless**, as the base class requires (`base.py:26-27`): all per-frame state lives in the analyser and the counter.

| Attribute | Value | Note |
|---|---|---|
| names | `deadlift`, `deadlifts`, `conventional_deadlift`, `barbell_deadlift`, `barbell_conventional_deadlift` | `sumo_deadlift` removed from this class (§3.2) |
| `movement_pattern` | `"deadlift"` | |
| `uses_diagnosis_engine` | `False` | Squat assessment, calibration and `HypothesisEngine` stay squat-only (`pipeline_process.py:303`, `:1014`; `calibration.py:19-28`) |
| `uses_bilstm_counter` | `False` | |
| `coaching_ready` | computed at import: `"deadlift" in NOWVA_DEV_COACHING_READY.split(",")` (False by default) until J8, then `True` | Set as the class attribute itself, so every reader sees the same value: `main_menu_agent.py:215` reads the attribute directly, and `coaching_ready_profiles()` (`registry.py:76-82`) does too. The env var must be set for the agent process; the pipeline subprocess inherits it. Tests patch the class attribute (`monkeypatch.setattr(DeadliftProfile, "coaching_ready", True)`), not the env |
| `gate_until_ready` | `True` (new attribute, default `False`) | `get_profile` returns `UntrackedProfile` for such a profile while it is not coaching-ready. This closes the scheduled-workout leak (O4) without changing other placeholder profiles |
| `display_name` | `"conventional deadlift"` | |
| `create_fault_rules` | D1–D10 rules, **no `DepthRule`** | A `DepthRule` would turn on depth-gated counting (`pipeline.py:391-399`) |
| `create_rep_counter` | `DeadliftRepCounter` (§2.3) | |
| `get_rep_signal` | Overridden to return NaN (the base raises `NotImplementedError`). Never called for the deadlift: the analyser supplies the signal (§3.3) | |
| `get_fault_to_cue_map` / `get_cue_dict` | `deadlift_*` keys, with `_left` / `_right` variants | |

### 3.2 Registry

Matching is whole-word, longest key wins (`registry.py:40-55`).
- A new `UntrackedVariantProfile(UntrackedProfile)` is registered for `sumo_deadlift`, `trap_bar_deadlift`, `hex_bar_deadlift`, `deficit_deadlift`, `snatch_grip_deadlift`, `single_leg_deadlift` and `rack_pull`.
- `romanian_deadlift`, `rdl` and `stiff_leg_deadlift` stay on the RDL profile.
- **Existing pins changed on purpose**, in the J3 PR with Ambaka's review:
  - `test_profile_registry.py:30` (`"Barbell Sumo Deadlift" → "deadlift"`) becomes `→ untracked` permanently;
  - `:29` (`"Barbell Conventional Deadlift" → "deadlift"`) becomes `→ untracked` while gated, and resolves to `DeadliftProfile` again at J8 (or when a test patches the override).
- New tests:
  - every squat alias → `SquatProfile`;
  - the variants never resolve to `DeadliftProfile`;
  - "deadlift", "deadlifts", "Barbell Deadlift", "conventional deadlift" and "Barbell Conventional Deadlift" → `DeadliftProfile` when ready, and `UntrackedProfile` when not.

### 3.3 Hooks and the `process_frame` fork

Hooks are added to `ExerciseProfile`. Each default reproduces today's squat path.

| Hook | Default | Deadlift |
|---|---|---|
| `needs_bar_3d: bool` | `False` | `True` |
| `create_rep_analyzer()` | `None` | `DeadliftRepAnalyzer` (owns all per-frame state) |
| `create_session_reference()` | `None` (the engine builds `SessionReference`) | `DeadliftSessionReference` (§2.8) |
| `create_set_diagnosis()` | `None` | deadlift set diagnosis (§7) |
| `allows_camera_refine: bool` | `True` | `False` |
| `feeds_body_calibration: bool` | `True` | `False` (§5.2) |
| `set_idle_timeout_s`, `min_cue_tiers`, `tracking_keypoints` | squat values | deadlift values (§4.3) |

**The `process_frame` fork** in `pipeline.py`, with `analyzer = self._rep_analyzer`. Every branch below is `if analyzer is not None: <deadlift> else: <existing lines, unchanged>`:

| Lines today | Squat path (unchanged) | Deadlift path |
|---|---|---|
| :974 | `rep_signal = profile.get_rep_signal(analysis, angles)` | `analyzer.observe(ctx)` with ctx = analysis, `analysis_world`, angles, foot state, time-aligned bar state (§2.2), gravity; then `rep_signal = analyzer.rep_signal` |
| :976-986 | `build_frame_sample(...)` | skipped (the analyser builds its own samples) |
| :990-1019 | standing frame, `_track_setup`, `_start_rep_setup`, bottom frame, trajectory | skipped; `_standing_kpts` is set from the analyser's standing frame for replay |
| :1072 | `max_depth_angle = _rep_max_knee_flex` | `max_depth_angle = NaN` |
| :1075-1077 | `compute_rep_features`, dump, `depth_target_met` | `features = analyzer.finish_rep()`; `depth_target_met = True` (the field is `bool = True`, `types.py:488`; with no depth target the requirement is trivially met, so no depth messaging triggers, and depth lines are also gated on `is_squat`); features dumped **after** `finish_rep` (avoids O2a) |
| :1082-1086 | `evaluate_rep_complete` (DEPTH rules only) | same line, no-op without a `DepthRule` |
| :1115-1121 (touch-and-go) | `_track_setup`, `_start_rep_setup`, trajectory append | `analyzer.start_rep(liftoff=low_point)` |

- **Layers.** `_build_exercise_layers` (`:359-399`) also builds the analyser and passes `reference=profile.create_session_reference()` to `RuleEngine`. `RuleEngine` gets an optional `reference` argument; the default is `None`, which means `SessionReference()`.
- **Frame output.** `PipelineFrame` gains an optional `bar_state_3d` (`None` for the squat).
- **Pinning.** The squat golden masters (§5) pin the `else` branches.

### 3.4 One activation path (fixes the demo entry)

Today, "Let's deadlift" launches the pipeline with the exercise as a CLI argument (`main.py:470`). Startup only sets `session_tracker.diagnosis_enabled` and calls `prepare_exercise` (`pipeline_process.py:1013-1017`). `_switch_exercise` (`:290-305`) is used only for switches.

- A single `_activate_profile(pipeline, session_tracker, bridge, exercise_name, meta)` runs both at startup and on a switch. It:
  - sets `diagnosis_enabled`;
  - installs `create_set_diagnosis()`;
  - applies the session metadata (§4.6);
  - calls `prepare_exercise`.
- For the squat it sets exactly what the two current call sites set, which `TestSwitchExercise` and the goldens pin.
- **Metadata at startup** comes in on `start_capture`. The pose subprocess is always launched preloaded (`main.py:1129-1133`), so `start_capture` is always sent after `greeting_done` (`main.py:1139-1155`). `main.py` adds an `exercise_meta` field read from state, and `_wait_for_start_capture` (`pipeline_process.py:906-920`) returns the message instead of a bool.
- **Metadata on a switch** comes in on `set_exercise.meta`, which is already relayed (`main.py:617-645`; `pipeline_process.py:1743-1748`).

### 3.5 Config

Two additions to the existing config, both additive:
- **Fault thresholds** go under `faults:` as new `deadlift_*` fields of `FaultsConfig`. That model is validated whole, so a fault block can't be silently dropped (`config.py:448-451`).
- **Counter and analyser parameters** (§2.3) go in a new `DeadliftConfig`. This needs a `deadlift` field on `BiomechanicsConfig` and an explicit `if "deadlift" in raw_config` parse branch in `load_pipeline_config` (`config.py:427-483`), because sections are parsed one by one.

Squat sections are unchanged.

### 3.6 Provider, bar and camera calibration

**Provider.** `get_pose()` returns only the primary frame (`multi_camera.py:307-357`). Add a `last_synced_frames` attribute, set after `_record_views`. The return value does not change.

**Bar tracker.** `deadlift/bar_tracker_3d.py` reads `last_synced_frames` when `needs_bar_3d` is set. It:
1. detects on all three views in one batch, keeping every candidate (a new detector class; `BarbellDetector` is untouched);
2. undistorts the detections;
3. associates them across views by minimum reprojection error, consistent with the hands and feet (this rejects a racked bar);
4. triangulates both ends;
5. smooths with a 3D Kalman filter;
6. writes the result to the timestamped ring buffer (§2.2).

**Camera calibration.** Shared rig state stays squat-safe.
- `allows_camera_refine=False` gates **every** refine started at a set boundary (`pipeline_process.py:610-651`): the drift refine, **and** the world-anchor refine (`keep_world_frame=False`, `:619-621`). The vertical is therefore never re-anchored on deadlift frames. The drift *check* is skipped too during deadlift sets, because it would read stale views, and resumes at the next squat set. The deadlift's own health signal is the bar/skeleton reprojection residual, logged per set.
- While the deadlift is active **and no capture window is open**, the provider stops appending views to `_view_buffer` (`multi_camera.py:108`). Deadlift frames can therefore never feed a later squat refine. The bootstrap capture window is exempt (`_capture_window_open`, `multi_camera.py:393-396`; `pipeline_process.py:681-693`): a deadlift-first session on an uncalibrated rig still calibrates, and the person-calibration solve asks for its two slow bodyweight squats, as today.

## 4. Platform integration — delivery side

### 4.1 Contract

`.claude/deadlift/CONTRACT.md` mirrors `.claude/squat-audit/CONTRACT.md`:
- the same ownership split;
- the per-fault table (§2.6), including the min tier;
- the `details` keys and IPC fields.

It reuses the existing message types (`fault`, `rep_complete` with `features` / `highlights` / `faults_detailed`, `diagnosis_complete`, `cache_cues`, `set_exercise`, `start_capture`). Nothing is renamed. The new fields are:
- `cache_cues`: `fault_to_cue`, `min_cue_tiers`, `set_idle_timeout_s`, `waits_for_diagnosis`;
- `start_capture` / `set_exercise`: `exercise_meta`;
- `frame_data`: `deadlift_phase`, `bar_midfoot_live_cm`, `bar_source`.

### 4.2 Cues, text, audio — the FINDINGS touchpoint checklist (`FINDINGS.md:662-697`)

| Touchpoint | Change |
|---|---|
| `src/assets/cue_text/cues.json` | deadlift keys, side variants, `_fixed` praise |
| `CUE_TEXT_MAP`, `CUE_DISPLAY_LABELS`, `FIXED_CUE_TEXT` (`coaching_constants.py`) | deadlift entries |
| `scripts/tools/draft_cue_text.py` | `--exercise deadlift` mode: deadlift system prompt and `CUE_SCENARIOS`. It is squat-only today (:52, :64-139, KeyError at :222-223) |
| `scripts/tools/generate_cue_audio.py` | generate every clip. A missing clip falls back to cloud OpenAI TTS at cache time (`audio_cue_service.py:41-48`, O3) and fix-praise stays silent |
| `FAULT_MESSAGES` (`fault_types.py`), `FAULT_LABELS` (`progress_context.py:21`), vocabulary terms, `CUE_PROMPTS` | deadlift entries |
| display `FOCUS_TEXT` | deadlift entries |
| placeholder `deadlift_flat_back`, `deadlift_even` | removed |

### 4.3 Orchestrator and service

Every item is a dispatch with squat as the default, or an addition.

1. **Cue fallback.** When `cue_key` is empty, the orchestrator falls back to the squat `FAULT_TO_CUE_MAP` (`coaching_orchestrator.py:948-951`), and so does the rest-complete focus line (`coaching_service.py:1210`, `:1230`).
   - Both now use `cache_cues.fault_to_cue`, which equals `FAULT_TO_CUE_MAP` for the squat.
2. **Minimum cue tier.**
   - The orchestrator cues mild faults that repeat on 2 of the last 3 reps (`_passes_bandwidth`, `:1139-1143`). It now also drops any cue whose severity is below `min_cue_tiers[fault_type]`, received via `cache_cues`.
   - A set focus taken from the diagnosis (`carry_focus_from`, `:613-617`) is adopted only if that fault reached its min tier in the set.
   - The fault is still recorded and still reaches the recap and the DB.
   - The squat's map is empty, so nothing changes for it.
3. **`is_squat`** (`:489-492`).
   - Add `waits_for_diagnosis` (true for squat and deadlift). `is_squat` keeps the depth lines.
   - Add a deadlift recap branch: bar speed, main fault and best rep. The numeric delta is formatted by `deadlift/diagnosis`, because `summarize_cue_magnitude` is squat-only (`demo_builder.py:77-79`).
4. **Set idle timeout.** Use `cache_cues.set_idle_timeout_s`, default 15 s (`:108`).
5. **Closed-loop D1 guidance.** This is its own small mode, not the squat stance monitor.
   - The squat monitor speaks only when `rep_phase == "idle"` (`:857`) and only arms after a fault cue plus a stance diagnosis (`:1709-1715`).
   - Deadlift guidance arms whenever `frame_data.deadlift_phase == "stance"` and `|bar_midfoot_live_cm| >` 2 cm.
   - It speaks `deadlift_closer` / `deadlift_back` through the existing cached-cue path. It reuses the squat monitor's speaking flag and utterance budget (`MAX_ADJUSTMENT_UTTERANCES`; monitor code at `:125-131`, `:348-357`, `:763-905`), says `adjust_good` once the bar is in tolerance, and disarms at SETUP or PULL.
   - The squat stance monitor is additionally gated on `is_squat`. It already arms only on bodyweight sets with a stance cause (`:1709-1715`, `:1725-1727`); this gate stops a squat diagnosis left over in `_latest_diagnosis` (O6) from arming it during deadlifts.
6. **Tracking-lost gate.** `ipc_bridge.py:391-428` mutes cues when tracking is lost, and plates can hide the shins.
   - The deadlift gate uses `tracking_keypoints` = hips, shoulders and wrists, plus the bar state.
   - Shins and ankles are not required during PULL, since the midfoot is locked.
7. **Squat-defined constants** (`:79-104`): `SAFETY_FAULT_TYPE`, `SIDE_VIEW_FAULTS` and `SYMPTOM_FAULT_TYPES` get deadlift sets, selected by exercise.
8. **Progress queries.** Pass `exercise=self._profile_name()`; no caller does today (`coaching_service.py:208`; `main_menu_agent.py:680`). For the squat this is `"squat"`, the current default.
   - The wording "last squat session" in `progress_context.py` (:62, :168-171) becomes exercise-aware.

### 4.4 Menu, first session, tools

- **Menu.** It is gated on `coaching_ready` (`main_menu_agent.py:207-220`; `main_menu_prompt.py:6-13`). The scheduled path is closed by `gate_until_ready` (§3.1).
- **First deadlift session.** There is no squat assessment (`uses_diagnosis_engine=False`). The `WorkoutAgent` deadlift branch gives:
  - a 5-step setup briefing: feet hip-width with the bar over midfoot; grip just outside the legs; shins to the bar; shoulders over the bar; slack out, push the floor;
  - an empty-bar warm-up set, which feeds the setup model and the arm-length estimator.
  - "First time" means `get_last_completed_session(exercise="deadlift")` returns nothing.
- **Knowledge card.** `deadlift_card.py` plus an `explain_deadlift` tool, mirroring `squat_card.py`.
  - The tool is exposed only in sessions whose plan contains a deadlift, so squat-only sessions keep an identical tool list (pinned).
  - `workout_prompt.py:155` (upper-back rounding is invisible) stays true for both exercises.
- **Form-check tool.** It is squat-shaped (`workout_agent.py:1119`). It gets a deadlift branch that answers from the latest deadlift rep features, or tells the athlete to ask after a rep.

### 4.5 Display and DB

- **Display** (`display.html`):
  - `FOCUS_TEXT` gets deadlift entries;
  - the "Avg depth" tile is hidden when `avg_depth` is null;
  - `SCORE_DIMS` is chosen per exercise;
  - `main.py` sends `avg_depth=null` for the deadlift.
- **DB.**
  - There is already one session per exercise.
  - `deadlift_*` faults are judged at rep end with their own rep number, so they are **not** added to `END_OF_REP_FAULT_TYPES`.
  - Features carry `dl_schema`.

### 4.6 Session metadata

Grip type (double / mixed / hook), plate diameter (default 45 cm), belt, shoes.
- Asked in `CollectExerciseInfoTask` (quick path) or in the deadlift `WorkoutAgent` greeting before `greeting_done` (scheduled path).
- Carried on `start_capture` at startup, or on `set_exercise` for a switch (§3.4).

## 5. Squat protection

### 5.1 Tests

| Layer | What | When |
|---|---|---|
| Existing pins | `test_fault_priority.py:73-125`; `test_coaching.py:111-124`; `test_pipeline.py` `TestExerciseProfiles` :920-983; `test_pipeline_process.py` `TestSwitchExercise`; `test_coaching_orchestrator.py` :1199-1223; `test_coaching_service.py`:916-1000; `test_main_forwarding.py` | keep green |
| Squat golden master, fresh | Replay fixed squat scenarios through `BiomechanicsPipeline` (harness `test_pipeline.py:329-392`) + `SessionTracker` / `IPCBridge`. Snapshot frames, faults, `rep_complete.features` and IPC messages | J0 |
| Squat golden master **after a switch** | Same scenarios after `set_exercise("deadlift")` → simulated deadlift reps → `set_exercise("squat")`. Expected output = the fresh golden, apart from the carry-overs listed in §5.2. Both goldens run in two variants, so the body-measurement path is compared like for like: with the same stored athlete params applied up front, and with none (first-time user) | J0 (deadlift part at J3) |
| Delivery golden master | Squat IPC stream through `CoachingService` + `CoachingOrchestrator` with fake TTS / LLM / DB. Snapshot cues, recap text and recorder ops | J0 |
| Invariants | Squat rule order; the full `FAULT_TO_CUE_MAP`; squat `CUE_TEXT_MAP` strings; squat tool list; squat `cache_cues` payload | J0 |
| Intentional changes | J3: `test_profile_registry.py:30` (sumo → untracked) and `:29` (conventional → untracked while gated). J8: `:29` back to deadlift, `:58-59` (`coaching_ready == ["squat"]`), `test_agent_prompts.py:116` (`"deadlift" not in` the menu prompt) | reviewed by Ambaka |

### 5.2 Shared state a deadlift could leave behind (each pinned by the after-switch golden)

| State | Behaviour today | Plan |
|---|---|---|
| Depth target | carried across switches (`pipeline.py:586`) | must survive squat → deadlift → squat unchanged (§2.8) |
| `SessionReference` bests | reset on every switch (`rule_engine.py:86`) | unchanged: documented carry-over |
| `_standing_reference_hip_cm` | raised only by `_start_rep_setup` | the deadlift skips that path, so it is unchanged |
| Rig calibration (`_refined`) | refined at set boundaries | no refine of any kind during deadlift sets (§3.6) |
| Provider `_view_buffer` | holds recent views for refines | not appended during deadlift sets (§3.6) |
| Foot-contact anchors and floor | session-scoped (`preik_chain.py:78-90`) | reset when switching to or from the deadlift. The squat then re-plants exactly as at the start of a session; the golden compares against a fresh session |
| Body measurements (`pipeline.body_calibration`) | session-scoped; fed every frame by `_record_body_measurements` (`pipeline.py:749-752`, `:950`) | Stored squat params load only when a calibration profile exists (`main.py:1100-1101`). Without a fix, a first-time user whose program puts deadlifts before squats would have the squat use body lengths measured on deadlift frames. New profile flag `feeds_body_calibration` (default `True`, deadlift `False`): deadlift frames never feed `pipeline.body_calibration`. The deadlift keeps its own `SegmentLengthEstimator` for the setup model, seeded read-only from `body_calibration` when that is already complete. The squat after a switch therefore measures exactly as in a fresh session |
| `_latest_diagnosis` | not cleared on a switch (O6) | squat monitor gated on `is_squat` |

## 6. Bar on the floor

**Today.**
- A 2D YOLO11n-pose detector runs on the primary camera only. It is off by default (`config.py:221`).
- It keeps only the best detection, and its weights are not in the repo.

**Keypoints.**
- Loaded bar: centre of the outer face of the outer plate, on each side.
- Empty bar: sleeve ends, as a separate class.
- Confirmed on the first real frames at J1.

**Visibility model (J1).**
- Map which cameras see each plate hub and each ankle through each phase:
  - head-on camera: both hubs;
  - each 45° upright camera: mainly its own side's hub, since the far hub can be occluded by the body and the plates.
- Requirement: each hub triangulated from at least 2 views.
- Check exposure and motion blur at ~1 m/s of bar speed, which sets a shutter limit.
- If a hub is seen by one camera only, use the other end plus a constant bar length (2.2 m) and the bar axis.

**Data and training.**
- 2,000–3,000 frames, labelled in self-hosted CVAT:
  - bar on the floor, mid-pull and at the top;
  - racked bar as negatives;
  - bumper and iron plates, varied colours and lighting.
- Ultralytics training, then ONNX → TensorRT FP16.

**Targets.**
- Recall ≥ 95 % with the bar on the floor.
- ≤ 1 % false positives on the racked bar.
- 3D error ≤ 1 cm static and ≤ 1.5 cm dynamic, measured against ArUco hub markers.

**Licence.** Ultralytics is **AGPL-3.0**. Shipping it needs an Enterprise licence or an Apache-2.0 alternative (for example RTMDet / RTMPose from MMPose). Decide **before J4**; this also applies to the existing bar detector.

**Wrist proxy without a bar.** Mid-wrist minus the learned offset.
- D1, D3 and D8b run with `bar_source = wrist_proxy`, wider thresholds and a moderate minimum.
- D10 is not emitted.

## 7. Set diagnosis (static graph)

The method is the squat's: symptoms → causes → hand-set prior × evidence → leak → noisy-OR. It follows the platform patterns: median rep, tail-mean fatigue, measurement confidence, observability blanking (FINDINGS D1, `engine.py`).

It is installed by `_activate_profile` (§3.4), so it also runs when the deadlift is the startup exercise. It produces a rolling update after each rep and a final `diagnosis_complete` at set end, in the contract shape.

**Decision for Ambaka (Q1).**
- **(a) Recommended:** parameterise `HypothesisEngine` with its graph, defaulting to the squat graph. It is pinned by `test_engine.py` (730 lines) and the goldens.
- **(b)** A separate module (~150 duplicated lines).

| Symptom | From | Causes (tier) |
|---|---|---|
| Bar not over midfoot | D1 | feet position (1, "move X cm closer"); habit (1) |
| Setup hip height | D4 | hips too low / too high (1, "hips X cm up / down"); hip / hamstring mobility (3); anthropometry (0) |
| Shoulders behind bar | D7 | shoulder position (1) |
| Hips shoot | D2 | hips too low at setup (1); slack not pulled (1); weak off the floor (3); load (2) |
| Bar drift | D3 | bar too far at setup (1); lats not engaged (1, "push the bar into your legs") |
| Incomplete lockout / lean-back | D6 / D5 | glutes not finishing (1); over-extension habit (1) |
| Hip shift / bar tilt | D8 / D8b | uneven stance or grip (1, "shift grip X cm"); unilateral weakness (3) |
| Velocity loss | D10 | fatigue → load (2, "about 10 % lighter") |

- **Outputs:** `cause_id`, tier, score, **one** numeric delta and one sentence.
- **Rep score weights:** setup 25 %, coordination 25 %, bar path 20 %, lockout 15 %, symmetry 15 %. Unmeasured dimensions are dropped (`rep_scoring.py:324-342`).

## 8. Data

### 8.1 Static knowledge

`docs/deadlift/KNOWLEDGE.md` covers:
- phases;
- each metric's frame, sign and window;
- thresholds and their rationale;
- cue text;
- the setup model.

It is reviewed by Ambaka and, ideally, by an external strength coach.

### 8.2 Raw recorder and replay (J1 prerequisite)

- `scripts/tools/record_rig.py` saves three synchronised videos, the timestamps, a copy of the calibration and gravity files, and the metadata. It writes nothing to `~/.nowva`.
- A replay provider runs recordings through the real pipeline. It is used for the deadlift offline and for squat regression replays.

### 8.3 Real capture and test design

**Round 1 (team, weeks 1–2).**
- 2–3 people, from an empty bar up to light loads.
- Clean reps, plus scripted safe faults. Rounding is shown with an empty bar or a PVC pipe only.
- Measures ρ, the within-lifter correlation.
- Measures the hip-keypoint bias against the trochanter marker (§2.7).
- Produces the camera visibility map (§6).

**Round 2 (pilot).**
- ≥ 10 lifters, 1.55–1.95 m, with varied femur, torso and arm ratios; ≥ 4 novices.
- Natural sets in **observation mode** (cues off), plus scripted sets.

**Test set, with a clustered design.**
- Design effect = 1 + (m − 1)ρ, starting from ρ = 0.1 and recomputed after round 1.
- **Hero faults:** ≥ 10 test lifters × ≥ 15 positives each, so ≥ 150 per hero fault. Effective lower bounds are ≈ 0.74 for precision and ≈ 0.58 for recall.
- **Provisional faults:** ≥ 5 positives per lifter.
- **Clean reps:** ≥ 30 per lifter.
- About 1,150 reps in total, over 2 sessions per lifter.
- Scripted positives will dominate per-lifter counts. That is why natural sets are reported separately, and why a hero fault must reach ≥ 0.80 on ≥ 30 natural cases.

**Gate statistics.**
- Lifter-clustered bootstrap and bootstrap-t, plus leave-one-lifter-out.
- Per-lifter floor of 0.70, applied only when a lifter has ≥ 15 cues.
- With 10 clusters, the bootstrap can be fragile. The analysis plan is therefore pre-registered (§1), and borderline results go to round 3 rather than being re-analysed.
- **Plan B, accepted in advance:** if D2 or D4 misses, the demo uses Demo α (D1 + D3).

**Labelling (CVAT, three views).**
- Faults and severity per rep, plus event timestamps on a subset.
- 20 % of reps double-labelled; target κ ≥ 0.6 per fault; Ambaka adjudicates; any fault below κ 0.6 is redefined.

**Training and tuning.** ≥ 300 reps and ≥ 50 positives per fault, on top of the test set. Plus 2,000–3,000 bar frames.

**Consent and storage.** Written consent, anonymised data, stored locally.

### 8.4 Ground truth

| Quantity | Method | Target |
|---|---|---|
| Static bar / D1 | floor tape + foot jig (0 / 3 / 6 / 10 cm) | ≤ 3 mm |
| Bar in motion (D3, D10) | ArUco markers on plate hubs, triangulated; optional linear position transducer | ≤ 5 mm |
| Trunk / hip / knee angles (D2, D5, D6) | calibrated planar sagittal camera, 120 fps; markers on greater trochanter and acromion (same line as the vision metric). IMUs only for timing and cross-checks: a back IMU measures the thoracic segment, and a thigh IMU suffers skin motion | ≈ 1–1.5° |
| Setup hip height (D4) | greater-trochanter marker, sagittal camera | ≤ 1 cm |
| Gravity | flat board + spirit level | ≤ 0.3° |

### 8.5 Synthetic data

Extend `.claude/preik-audit/harness/preik_harness/`, after fixing `__init__.py:13` and `runner.py:404`.

**Deadlift generator:**
- starts from the floor, with the hands coupled to the bar;
- the bar is on the floor (`barbell.py` currently puts it on the back);
- adds plate occlusion, a tilted world, a moved camera and **capture-vs-analysis lag**.

**Scenarios:**
- D1–D10;
- soft and over-extended lockouts that must be counted;
- a rep that stalls at the knees;
- dead stop, quick re-pull, touch-and-go (sets of 5 and 10), bumper bounce, dropped bar, re-setup;
- several builds;
- a noisy tracker.

**Assertion:** every rep is counted exactly once. Synthetic data never sets final thresholds.

## 9. Edge

- **J1 measurement** on a Jetson Orin Nano Super: RTMPose on 3 views + the bar detector on 3 views (TensorRT FP16) + the full chain.
- The new geometry, state machine and diagnosis add < 1 ms.
- **Degraded mode**, in order of preference:
  1. pose at 30 Hz and bar at 15 Hz. On frames without a detection, the bar Kalman filter still writes a *predicted* state (flagged `predicted`) into the ring buffer, so the time alignment (§2.2) finds a state on every frame;
  2. otherwise, everything at 15 Hz.
- The deadlift adds no cloud dependency. The conversational stack is cloud today (O3).

## 10. Milestones

J4 starts in parallel with J1. With two engineers, J2 and J3 also run alongside J4.

| | Content | Acceptance | Est. |
|---|---|---|---|
| **J0 Squat net** | Squat goldens (fresh, delivery, after-switch scaffold); invariants; `requirements.lock` env; CI proposal; licence decision; `VALIDATION.md` skeleton | All green on `main`; changing one squat threshold or one cue text fails the right test | 4–6 d |
| **J1 Foundations** | Recorder + replay; gravity tool, deadlift frame, sign/tilt/lag tests; `KNOWLEDGE.md`; setup model (2 solves); round-1 capture (ρ, keypoint bias, visibility map); Jetson numbers; existing bar weights evaluated | Review signed; tests green; ≥ 1 h raw capture | 2 wk |
| **J4 3D bar** (from J1) | Annotation, training, export, tracker, provider frames, time-aligned buffer, ArUco ground truth | Recall ≥ 95 %; 3D error ≤ 1 / 1.5 cm; Jetson budget | 2–3 wk |
| **J2 Simulator** | Deadlift generator and scenarios | Each scenario carries ground truth | 1 wk |
| **J3 Deadlift profile on the platform** | Profile, gating, registry variants; `_activate_profile`; hooks and `process_frame` fork; analyser, counter, D1–D10; deadlift reference; config; global-map entries; camera-refine and view-buffer gates; after-switch golden. **Thin slice:** dev override, voice counts reps on the rack | Simulator: exact-once counting (incl. touch-and-go, bumper bounce), events ≤ 100 ms, each injected fault detected, clean scenario fault-free; squat goldens unchanged | 2.5 wk |
| **J5α delivery slice** | Closed-loop mode (§4.3 item 5); D1/D3 cue text and audio; contract fields in `cache_cues` and `frame_data` | Slice runs on the rack under the dev override | 1 wk |
| **Demo α** (≈ week 8 with two engineers, ≈ week 10 with one) | D1 closed loop + D3 at the floor, rep counting | 10 runs in a row; team round-1 lifters | — |
| **J5 Delivery integration** | Contract; FINDINGS checklist touchpoints; orchestrator items in §4.3; recap; display; DB; metadata; first session; card and form-check | Full rack session (first time and returning, quick and scheduled under the override); no clip synthesised at run time; squat goldens (pipeline + delivery) unchanged | 2 wk |
| **J6 Real validation** | Round 1 → thresholds, min tiers, margins, ρ; round 2 → test set | Demo gate (§1) on ≥ 10 unseen lifters, per the pre-registered plan; κ ≥ 0.6; status per fault | 3–4 wk |
| **J7 Set diagnosis** | Graph (a or b), scoring, recap | Top cause matches the coach on ≥ 70 % of sets | 1–1.5 wk |
| **J8 Ship** | `coaching_ready=True`; the two intentional test changes; hero demo | 10 consecutive demos; Jetson budget | 1 wk |

Total: about 16–18 weeks for one person, or about 10 weeks for two. Demo α lands at about week 8 with two engineers, or about week 10 with one.

## 11. Decisions for Ambaka

1. Set diagnosis: parameterise `HypothesisEngine` (recommended) or a separate module?
2. No camera refine (drift or world-anchor) during deadlift sets: OK?
3. `gate_until_ready` on the deadlift only, or for every placeholder profile (a wider change)?
4. Bar detector licence: Ultralytics Enterprise or an Apache-2.0 alternative?
5. Who labels the data?
6. Count touch-and-go reps in v1? Proposed: yes, with no cue between those reps.
7. Where are the current bar weights, and what were they trained on?
8. An accelerometer in the device?
9. Which plates are in the demo gym?

## 12. Risks

| Risk | Mitigation |
|---|---|
| Squat changed by shared-code edits | Squat-default dispatch; fresh, after-switch and delivery goldens; invariants; carry-over table (§5.2) |
| Rig calibration degraded by deadlift frames | No refine of any kind; view buffer frozen |
| Unvalidated deadlift reachable via programs | `gate_until_ready` |
| Bar / skeleton time misalignment | Timestamped ring buffer + test |
| Vertical axis is not gravity | Measured gravity, per-camera consistency, tilt tests, wider fallback |
| Back rounding unmeasurable | Proxies; behavioural wording; experiments gated on data |
| Pose quality in a deep hinge; occlusion | Measured in round 1; live midfoot locked at SETUP; deadlift tracking-lost gate |
| Far plate hub hidden | Visibility map; bar-length constraint |
| Bar detector generalisation; licence | Varied data split by lifter, negatives; licence decided before J4 |
| Thresholds invented | Initial values only; min tiers; error budget; real data |
| Optimistic validation | Clustered design, pre-registration, natural sets reported separately |
| Long timeline for YC | Demo α at about week 8 with two engineers (about week 10 with one) |

## 13. Observations on the current code (outside this plan, not changed)

- **O1:** world vertical = hip→ankle axis, not gravity. The squat's trunk pitch inherits the bias (`person_calibration.py:734-752`).
- **O2a:** `rep_data.features` is dumped (`pipeline.py:1076`) before `finish_rep` annotates `velocity_loss_pct` (`rule_engine.py:238-239`). The non-BiLSTM IPC/diagnosis copy therefore keeps NaN.
- **O2b:** `HeelRiseRule` emits `affected_side`, not `side` (`heel_rise.py:107-112`), so `heels_down_left/right` are never chosen (`ipc_bridge.py:183`).
- **O3:** the conversational stack is cloud (Deepgram, OpenAI `gpt-5.4-mini`, ElevenLabs; `pipeline_factory.py`), against guardrail #1. Missing cue clips fall back to cloud OpenAI TTS at cache time (`audio_cue_service.py:41-48`).
- **O4:** scheduled `start_workout` does not check `coaching_ready` (`main_menu_agent.py:89-166`). Placeholder profiles run in programs; this plan closes it for the deadlift only (Q3).
- **O5:** the cue-audio script's default ElevenLabs voice (`generate_cue_audio.py:55`) differs from the live agent's (`pipeline_factory.py:43`) unless `ELEVENLABS_VOICE_ID` is set.
- **O6:** `_latest_diagnosis` survives exercise switches (`coaching_orchestrator.py:334`, `:600`). Low impact: the stance monitor arms only on bodyweight sets with a stance cause.
- **O7:** the `set_exercise` display event sends `weight_lbs` without converting from kg (`coaching_service.py:1330`).
