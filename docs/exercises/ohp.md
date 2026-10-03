# Standing barbell overhead press (from the rack): implementation plan

Code of record: `~/NowvaLiveKit-exercises` @ `265ef97` (squat V1 + multi-exercise platform, PR #20).
Every `file:line` below was read on that commit.

The plan mirrors `.claude/squat-audit/FINDINGS.md` / `CONTRACT.md`:
- one feature set per rep, and one `judge_rep` per fault;
- absolute thresholds, with drift only from the session reference;
- observability gating;
- exercise-prefixed cue keys.

Shared platform changes follow **`SHARED-PLUMBING.md`** (the one spec for OHP/RDL/deadlift). §7.0 lists
only the press-specific additions to it.

---

## 0. Facts verified on main (the plan builds on these)

| # | Fact | Where |
|---|---|---|
| V1 | The production frame is **Y-down**: `WORLD_UP = (0,-1,0)`, X = subject's left, +Z = subject's back, forward = −Z. `.claude/rules/biomechanics.md` ("larger Y = higher") is stale; this plan follows the code. | `utils/geometry.py:17`, `kinematics/analytical_ik.py:1-7` |
| V2 | Arm outputs: <br>• `elbow_flexion_l/r` = 180 − ∠(S,E,W), 3D, **unsigned** <br>• `shoulder_abduction_l/r`, XY-projected, ~170° overhead <br>• `wrist_y_l/r`, cm above the shoulder midpoint <br>• `wrist_x_l/r`, cm forward along the **depth axis** <br>• `trunk_rotation`, computed from shoulder **z** only, so monocular-depth based | `analytical_ik.py:287-310, 393-471` |
| V3 | 21 keypoints (COCO-17 + toes + heels). **No hand landmarks.** | `utils/types.py:78-99` |
| V4 | The scaffold is wrong: <br>• `ElbowFlareRule` fires on every press (abduction > 75°) <br>• `BarPathRule` uses `wrist_x` (depth axis) with an 8 cm threshold <br>• `LockoutRule` has a single 10° cut and no SEVERE tier <br>• `get_depth_metric` returns `avg_elbow_flexion` <br>• the `shoulder_press` alias routes seated/dumbbell presses here | `profiles/overhead_press.py:42-95`, `rules/elbow_flare.py:62-80`, `rules/bar_path.py:66-86`, `rules/lockout.py:86-122` |
| V5 | `get_readiness_check()` is never called. The readiness gate latches, and the pipeline returns before pre-IK until it does. | `profiles/base.py:121`, `pipeline.py:246-256, 898-913` |
| V6 | `pipeline.py:1072` overwrites `max_depth_angle` with knee flexion. `:1075-1077` build squat features and `depth_target_met` for every profile. | `pipeline.py` |
| V7 | `SessionReference.update()` reads squat fields only. | `faults/session_reference.py` |
| V8 | Observability and cue priority are keyed by the fault_type string in global tables. | `faults/observability.py:42-55`, `coaching/cue_cache.py:106-120` |
| V9 | Counter internals: <br>• counts excursions away from its baseline <br>• `_s` duration fields <br>• velocity = least-squares slope over `VELOCITY_WINDOW_FRAMES = 5` <br>• `LOCKOUT_ZONE_FRACTION = 0.25`, `MAX_REP_DURATION_S = 12`, `MAX_GAP_S = 0.5` <br>• baseline = first sample after `reset()` | `config.py:249-271`, `faults/hip_position_counter.py:37-52` |
| V10 | `uses_bilstm_counter=False` ⇒ `_bilstm is None` (tested). Assessment and calibration are skipped. | `pipeline.py:371`, `calibration.py:26`, `test_pipeline.py:926-929` |
| V11 | Rest end calls `pipeline.rep_counter.reset()`. | `pipeline_process.py:1229, 1467, 1604, 2008` |
| V12 | Cue fallback: <br>• the bridge drops a repeat fault for 3 s <br>• a fault without a cue key falls back to the **squat** `FAULT_TO_CUE_MAP` and is skipped, so press cues go silent from the 2nd occurrence (fixed by SHARED §4) | `ipc_bridge.py:169-203`, `coaching_orchestrator.py:948-951, 1014-1019, 1170-1174` |
| V13 | Agent squat assumptions: <br>• effort reads `ascent_time_s` <br>• the set summary emits depth fields and counts every fault per rep <br>• `send_set_complete` averages `max_depth_angle` <br>• "deepest clean rep" wording <br>• `SAFETY_FAULT_TYPE = "knee_valgus"` <br>• knee-angle depth class fallback | `coaching_orchestrator.py:79, 695-707, 1433-1490, 2116`; `ipc_bridge.py:359-382, 465-470` |
| V14 | Existing recordings: `set_finalizer` stores no arm keypoints. `--inspect` drops confidences and records nothing before the latch. Replay needs SHARED §8. | `set_finalizer.py:359-371`, `pipeline_inspector.py:57-64`, `pipeline.py:681, 902` |
| V15 | Rig person-calibration bootstrap waits for `min_squat_excursions: 2` without a calibration file. | `config/biomechanics.yaml` |
| V16 | The fixed-lag Kalman (`lag_frames: 2`) delays analysis 2 frames. Pre-IK starts on the latch frame, so post-latch frames are delayed, not dropped (pinned by test). | `utils/preik_chain.py:92-98`, `pipeline.py:898-913` |
| V17 | Midfoot = ankle + 0.35·(toe − ankle). | `diagnosis/lean_model.py:20`, `rep_features.py:107-140` |
| V18 | **The rig's world "up" is the median standing hip→ankle direction, not gravity** (2–4° off per the deadlift plan §2.1). Measured gravity comes from `measure_gravity.py`; the spec exposes it as `FrameContext.up` (physically up, never negated) plus `gravity_source` (SHARED §3). | `triangulation/person_calibration.py:734-752` |

---

## 1. Summary and demo value

Add the strict standing press, bar taken from J-hooks, as Nova's second coached lift, built the
squat-V1 way. V1 coaches what the cameras can actually see.

**Single frontal camera (~12 fps), the setup of every real run so far:**
- rep counting;
- **leg drive**;
- **uneven press**;
- **grip width**, delivered between sets;
- **lockout**, measured but recap-only until validated on one camera.

**Start tiers on mono at launch (§5.0).** These are the conservative choices wherever the budget is
tight. They depend on the camera mount (SHARED §3):

| Mono camera mount | Leg drive | Uneven press |
|---|---|---|
| **Certified and level**: rack mount or inclinometer-entered pitch, \|θ\| ≤ 3° | moderate (provisional; replay decides) | **severe**; promoted to moderate only if the replayed P95 at this pitch is ≤ 2.0° |
| Certified, 3° < \|θ\| ≤ 10° (levelled) | severe once the tilted-band gate passes (§8.4); recap-only until then | severe once the tilted-band gate passes; recap-only until then |
| Uncertified, or \|θ\| > 10° | recap-only (approximate, hedged) | recap-only (approximate, hedged) |

- Why tilting costs so much: levelling a tilted camera mixes MediaPipe's regressed depth into every
  height at sin θ, and an unlevelled tilt leaks forward and back motion. §5.0 has the budget.
- **Demo and setup.** Use the rack mount, or a webcam **kept level within 3°**. A phone level app
  entered as `NOWVA_CAMERA_PITCH_DEG` counts as an inclinometer.
  - Frame the lifter by stepping back or turning the camera to portrait, never by tilting it.
  - On an uncertified webcam, mono coaching is rep counting, the grip note and the recap only.
- **Impact on the demo's headline cue.**
  - On a level mount, "Keep your knees locked" is the dependable moderate cue.
  - "Drive your left hand up" fires on a hand-height lag (≥ 9°) until the validation replay promotes
    uneven to moderate (≈ 5.5 cm lag).
  - The pooled tilt reference (§3.2) puts the level-mount P95 at 1.9–2.1°, right at the 2.0° allowance,
    which is why uneven stays at severe until proven.

**Triangulated rig (≥30 fps):**
- adds **layback** and **bar finishing forward of mid-foot**, the two faults a coach fixes first;
- plus 3D lockout and velocity loss;
- recap-only until rig recordings and measured gravity validate them.

There is no causal diagnosis for the press in V1 (§9.1). Squat behaviour is untouched, pinned by the
squat golden masters, both fresh and after a press switch.

**Demo value.** Same rack, same camera, a second lift, the same YC loop:

> unrack → press (knees dip) → "Keep your knees locked" → two strict reps (the hold window) → "Legs stayed locked, good."

The uneven-press loop ("Drive your left hand up" → "Bar stayed level, nice.") joins the demo once replay
promotes uneven to moderate (§8.3). Until then it fires only on a hand-height lag.

**Demo setup:** the rack mount, or a webcam kept level within 3° with the pitch entered. Run the mono
validation replay before filming, so the promotion decision is made first (§8, Sequencing).

The press reads well from the front: the arms move in the image plane, so asymmetry and knee dips are
obvious on video. It also gives the hardware pitch an honest line: one webcam coaches the frontal
faults, and the rack's cameras are what see layback and bar path.

---

## 2. Coaching model

### 2.1 What a world-class coach watches on a strict press, ranked

Ranked by what a coach corrects first in-set: safety, then bar path, rep standard, strictness,
symmetry, speed. Setup (grip) comes first *chronologically*, at the rack, but it can't change mid-set,
so it never competes for an in-set cue slot.

| Rank | What the coach watches | Why | Single camera | Rig | V1 |
|---|---|---|---|---|---|
| 1 | **Layback**: trunk leaning back past what clearing the chin needs | Spinal load, and it turns the lift into an incline press. The "backbend" press is why the lift left Olympic competition in 1972 (Soriano 2019). | Not observable: sagittal; mono trunk pitch is dominated by camera placement (squat A12) | Observable with measured gravity (trunk line, not lumbar curve) | `press_layback` |
| 2 | **Bar path / finish**: bar ahead of mid-foot, head not "through" | Balance and shoulder moment arm; the most common technical miss (Rippetoe; StrongLifts) | Not observable (depth axis) | Observable with measured gravity | `press_bar_forward` |
| 3 | **Lockout**: elbows straight overhead | Defines a completed rep; overhead stability | Approximate | Observable | `press_lockout` |
| 4 | **Legs**: knees locked, no dip-and-drive | Strictness; hides a load that is too heavy | Observable for real dips; kicks under ~30° are invisible | Observable | `press_leg_drive` |
| 5 | **Symmetry**: bar level, arms finish together | Strength asymmetry | Observable | Observable | `press_uneven` |
| 6 | **Setup**: forearms vertical from the front | Bar path and force direction (Gundersen 2025) | Observable | Observable | `press_grip` (between sets) |
| 7 | **Bar speed / grind** | Proximity to failure | Not trusted at 12 fps | Observable | `press_velocity_loss` |
| — | **Rep standard**: shoulders to full lockout | Counting | Observable | Observable | ROM acceptance + `press_missed` |

### 2.2 Deliberately not judged in V1

| Not judged | Why |
|---|---|
| Wrist extension / bar in the fingers | No hand keypoints (V3). |
| Lumbar hyperextension as such | No spine landmarks. The rig measures the hip→shoulder line, cued as "glutes tight, ribs down". It is never claimed as "your lower back is arching". |
| Head position, directly | The bar occludes the face, and ears are unreliable overhead. Covered indirectly by bar-over-mid-foot. |
| Shrug at lockout | Debated cue; ear–shoulder distance is unreliable overhead. |
| Elbows behind the bar in the rack | Sagittal, so rig-only. V1.1. |
| Elbow flare | The scaffold metric reads ~170° on every press (V4). Flare is a symptom of grip and bar path. |
| Mid-press bar loop, lateral trunk lean | Recorded only. Lateral lean co-occurs with `press_uneven`, and coaches give one asymmetry cue per set. |
| Eccentric control, bracing, breathing, stance | Low value or not visible. |
| Bar tilt from the bar detector | Off by default and primary-camera only. V1.1. |

---

## 3. Setup and rep definition

### 3.1 Unracking and the front-rack start

- The lifter faces the rack and its cameras, as in the squat. J-hooks sit at mid-chest to collarbone
  height.
- Grip just outside the shoulders. Step under the bar with it on the front delts, stand up, take 1–2
  steps back, brace, then press.
- Before set 1, Nova plays a pre-generated `press_setup` line: "Unrack, step back, hold it on your
  shoulders".
- The rep signal is shoulder-relative, so the walk-out and re-racking leave it ≈ constant.

### 3.2 Readiness check (wired for the first time)

**Mechanism.** The readiness gate keeps every `StandingPoseGate` check (visibility, knees, torso). The
press adds a rack-position predicate as the optional analyser method `PressRepAnalyzer.is_ready(ctx)`
(SHARED §2).
- It is **ANDed** with the gate's own checks and must hold over the gate's consecutive-frame window
  (`required_consecutive_frames = 5`). It never replaces them.
- The analyser owns the rack reference. The profile holds no analyser state, and its
  `get_readiness_check()` stays `None`.

Before the latch, `ctx.skeleton_3d` is the raw frame, `ctx.raw_confidences` holds its confidences, and
`ctx.angles` is None. The predicate uses only vertical and image-plane quantities.
`h(p, q) = q_y − p_y` (`geometry.height_above`).

**Per-frame checks (all must pass):**
1. Arm length estimate: `torso_m = |S_mid − H_mid|`, `arm_est = 1.15·torso_m`. Basis: Drillis & Contini
   (upper arm 0.186H + forearm 0.146H = 0.332H, against shoulder–hip ≈ 0.288H).
2. Wrists and elbows have confidence ≥ 0.3.
3. Bar height: −0.25 ≤ h(W_mid, S_mid)/arm_est ≤ +0.30.
4. Forearms up: h(W, E) ≥ 0.10 m on both sides.
5. Hands level: |h(W_L, W_R)| ≤ 0.08·arm_est.
6. Hands on a bar: |W_L.x − W_R.x| ≥ 0.9·|S_L.x − S_R.x|. This uses image X on mono and `lat_R` on the rig.

**Rack reference.**
- `is_ready` keeps the last 5 frames that passed the rack predicate, within a 2 s time bound, and clears
  them when the predicate fails.
- The gate evaluates `is_ready` on every pre-latch frame (no short-circuit) and ANDs the result with its
  own checks.
- A gate-only failure (knees, say) doesn't invalidate buffered frames. Each one is still a valid rack
  pose, and the latch itself needs 5 consecutive AND-passes.
- At the latch, the analyser holds the per-set rack reference, the medians of those frames:
`rack_bar_cm`, `rack_hip_cm`, `rack_shoulder_above_hip_cm`, `rack_tilt_deg`, `rack_grip_tilt_deg`,
`rack_shoulder_width_x`.

**`rack_tilt_deg` is pooled.**
- It is the median over every in-band, near-still rack/setup frame **in the set so far**: the latch frames
  plus each rep's setup ring buffer, with `facing_ok`.
- That is ≈ 15 frames by rep 1 and 25+ later.
- Pooling is valid because camera roll and the lifter's wrist offset at the rack are constant within a
  set, and grip only changes on a re-rack.
- It resets in `reset_set()`. A mid-set re-rack (signal NaN for > 2 s) restarts the pool.

**Baseline seeding (SHARED §2).**
- `PressRepAnalyzer.counter_baseline()` returns `rack_bar_cm`. The pipeline calls
  `rep_counter.seed_baseline(...)` at the latch and again after every counter reset, through
  `pipeline.reset_rep_counter()`.
- `reset_set()`, called from `reset_readiness_gate()`, clears the rack reference. `counter_baseline()`
  then returns `None` until the next latch.
- The squat has no analyser, so it is unchanged.

### 3.3 Rep signal, frame and phases

**Signal (cm).** `bar_cm = 100·h(W_mid, S_mid)`, which is `JointAngles.avg_wrist_y`.

It is NaN when:
- a wrist or shoulder has confidence < 0.1; or
- it is below the **rack floor**, `bar_cm < −0.35·100·torso_m`. With the hands at the sides the signal
  is −55 to −65 cm, so the baseline never drifts to "hands down".

**Phases:**

| Counter state (wire string kept) | Press phase |
|---|---|
| `idle` | `setup` |
| `descending` | `press` (concentric) |
| `bottom` | `lockout` |
| `ascending` | `lower` (eccentric) |

The agent only reads `"idle"`. The analyser derives its own phases (§4.3).

### 3.4 Counter config (`faults.press.counter: HipPositionCounterConfig`, SHARED §5)

**Velocity noise.** The least-squares slope over n samples spaced Δt has σ_v = σ_pos / (Δt·√(n(n²−1)/12)).

With today's fixed 5 samples:
- 3.8 cm/s per cm of σ_pos at 12 fps;
- **9.5 cm/s per cm at 30 fps**, which would put a 12 cm/s threshold at ~1.3σ on the rig.

The press therefore adds `velocity_window_s` (default None = 5 frames, so the squat is unchanged) and
sets it to 0.40 s:
- n = 5 at 12 fps: 3.8 cm/s per cm;
- n = 12 at 30 fps: 2.5 cm/s per cm.

Expected noise: mono σ_pos ≈ 1.0 cm (σ_v ≈ 3.8 cm/s); rig σ_pos ≈ 0.5 cm (≈ 1.3 cm/s). Both are measured
in replay before the thresholds freeze. The ~0.2 s lag doesn't matter, because features find the press
start from the trajectory.

| Field | Press | Squat yaml | Rationale (σ = worst case, 1 cm σ_pos) |
|---|---|---|---|
| `velocity_window_s` | 0.40 | (5 frames) | Above |
| `entry_vel_threshold` | 12.0 cm/s | 3.0 | ≥ 3.2σ in both modes. The slowest press averages 0.19 m/s at 1RM (Balsalobre-Fernández 2018). |
| `bottom_vel_threshold` | 6.0 cm/s | 5.0 | Lockout = bar stopped. Arming needs a 12 cm/s peak. |
| `ascending_vel_threshold` | 10.0 cm/s | 3.0 | ≥ 2.6σ on mono, ≥ 4σ at 30 fps |
| `min_depth_cm` | 20.0 | 10.0 | Minimum press attempt. Filters re-grips, shrugs and J-hook lip clearance (3–5 cm). Counting is §3.5. |
| `standing_return_cm` | 6.0 | 3.0 | Bar back on the shoulders |
| `min_descending_s` (press) | 0.25 | 0.1 | Fastest strict concentric ≈ 0.4 s |
| `min_bottom_s` (lockout) | 0.08 | 0.067 | Touch-and-go lockouts last 1–2 frames at 12 fps |
| `min_ascending_s` (lower) | 0.25 | 0.1 | |
| `min_rep_duration_s` | 0.8 | 0.5 | Fastest strict rep ≈ 1.0 s |

The rack zone is `LOCKOUT_ZONE_FRACTION = 0.25` (unchanged): ≈ 14 cm above the rack, about chin height.
A turnaround at or below the chin ends the rep, and a touch-and-go starts the next one.

### 3.5 ROM acceptance (when a rep counts)

A counter rep is counted only when `analyzer.rejection_reason(features)` is None. That needs all three
criteria:

1. **Started from the rack.** Any one of:
   - ≥ 3 setup samples within 0.5 s before the press start, inside the rack band;
   - the **first rep after the latch** (the latch frames are the rack hold);
   - a touch-and-go continuation of a counted rep, signalled by `on_rep_start(piston=True)`.
     - Call order (SHARED §2): the ended rep's `finish_rep`, rejection and verdicts run first.
     - On a piston rep the setup window is the turnaround frames, used for timing only.
     - The setup-dependent grip verdict is not judged on a piston rep.
     - Within-rep faults are still judged (SHARED §2): `press_lockout`, `press_uneven`, `press_layback`,
       `press_bar_forward`, `press_velocity_loss`.
     - Leg drive and tilt use the set-level references (`rack_hip_cm`, pooled `rack_tilt_deg`) instead
       of a per-rep hold.

   This rejects bare-hand reaches from the sides.
2. **Reached overhead** (hip-anchored, so shoulder-keypoint migration can't move it).
   - `top_height_ratio = (p95 h(W_mid, H_mid) − rack_shoulder_above_hip) / arm_ref ≥ 0.80`.
   - Margin: the crown sits ≈ 0.60 arm lengths up. A straight lockout reads ≈ 0.95–1.0 and a 35°-bent one
     ≈ 0.92, so the margin is ≥ 0.12. Layback costs ≤ 0.01, and rig gravity error costs < 0.003
     (cos 4°).
   - Validation re-checks the margin against measured migration (§8.2).
   - A soft lockout counts, with `press_lockout`.
3. **Clipped wrists.** If `wrists_clipped` and the last height measured on the way up was ≥ 0.60,
   criterion 2 is waived: the rep counts and lockout is NaN.

**On rejection** (SHARED §2): `reject_last_rep`, `discard_rep` (engine and analyser), then
`rep_data = None`. This happens before any verdict and before any `SessionReference.update`.

Then `profile.on_rep_rejected(reason, features, angles)` runs:
- `"no_lockout"` (criterion 2 failed after ≥ 20 cm of travel) returns `[press_missed]`. The **pipeline**
  tags it through `rule_engine.tag_observability` (`FAULT_MEASUREMENT`, SHARED §2), and drops it if it is
  not observable. The event is never cued, is recorded for the recap, and the rep is not counted.
- `"no_rack_start"` returns `[]`.

### 3.6 Set boundaries

- The same mechanisms as the squat: `rest_start`, the 30 s no-rep timeout, and the idle timer.
- `reset_rep_counter()` re-seeds the baseline.
- The gate re-arms before rest ends and must latch in the rack again.
- A mid-set re-rack sends the signal to NaN while the hands are off the bar.

### 3.7 Edge cases (each is a scripted validation clip, §8.1)

| Case | Behaviour |
|---|---|
| Walk-out or walk-in with the bar | Shoulder-relative signal ≈ constant, so no rep |
| Approaching the bar, hands at the sides | NaN below the rack floor |
| **First-rep timing** | A **hold** is consecutive frames passing the rack predicate (bar inside the +0.30·arm band), whether the bar is still or already rising. Rep 1 counts when ≥ 5 in-band frames pass before the bar leaves the band. <br>• That is 0.42 s at 12 fps or 0.17 s at 30 fps. <br>• The Kalman lag delays and drops nothing (V16), and criterion 1 is met by the latch frames. <br>• The first 0.15–0.2 s of the rise is still in-band (≈ 2 frames at 12 fps), so a ≈ 0.30 s stationary hold plus the rise reaches 5 frames. <br>• Pinned in frames (§7.2): 5 in-band frames counted, 4 missed, and a 0.30 s stationary hold plus the rise counted. <br>• If latch frames include rising frames, the median baseline stays at rack height (3 of 5 are still). The analyser marks `press_start_observed = False` when the rise began before the latch, and that rep's velocity is NaN. |
| Touch-and-go at the chin | Split by the rack zone |
| Turnaround above the chin | Merged (not a rep by the strict standard); `partial_bottom` recorded |
| Missed rep | `"no_lockout"` → `press_missed` |
| Wrists leave the frame top | Counted via criterion 3; lockout NaN. Tracked keypoints include the wrists, so `tracking_lost` plays once per set (P1). |
| Bare-hand stretch mid-set | `"no_rack_start"`. A bar-less hold at rack height can't be told apart (known limitation). |
| Lifter rotated > 20° (mono) | Frontal features NaN (§4.1) |
| Re-rack at set end | Under 20 cm |
| Exercise switch | Fresh analyser, session reference and predicate (SHARED §2) |
| First set before body measurement completes | In-set normalizers |

---

## 4. Per-rep features (`analysis/press_features.py`, owned by `PressRepAnalyzer`)

### 4.1 Frame, axes, normalizers, facing check

**Axes**
- `h(p, q) = q_y − p_y`.
- **Lateral `lat_R`** (points to the subject's right):
  - mono: `(−1, 0, 0)`, image-plane X; frontal metrics never touch Z;
  - rig: `unit(horiz(R_hip − L_hip))`.
- **Vertical for sagittal rig metrics: `up = ctx.up`**, used directly and never negated (SHARED §3,
  V18).
  - It points physically up in the Y-down world frame.
  - `"body"` source: `up = WORLD_UP = (0,−1,0)`.
  - `"measured"` source: the floor-ChArUco vector.
  - `horiz_g(v) = v − (v·up)·up`.
  - `fwd = unit(cross(up, lat_R⊥))`, where lat_R⊥ is lat_R made perpendicular to `up`.
  - Check: up = (0,−1,0) and lat_R = (−1,0,0) give fwd = (0,0,−1), forward ✓.
  - Check: a trunk T along `up` gives T·fwd = 0, so layback = 0° ✓.
  - Vertical and difference metrics (bar height, hip drop, tilt vs rack) keep `WORLD_UP`. A 2–4° error
    changes them by < 0.3% or cancels in the difference.
- **Midfoot (rig):** the median over the setup window of the both-feet mean of ankle + 0.35·(toe −
  ankle) (V17). It is locked per rep; the feet don't move and plates don't occlude them in the press.
- **Mono levelling (SHARED §3, pitch leak).** On one camera the world frame is camera-aligned, so with
  pitch θ a "vertical" difference reads Δy·cos θ + Δz·sin θ.
  - When `ctx.pitch_certified`, `is_ready` and `observe` first apply
    `geometry.level_camera_frame(points, ctx.mount_pitch_deg)`. Every vertical and tilt metric is then
    computed in the levelled frame: `bar_cm`, `bar_above_hip_cm`, `hip_cm`, `tilt_deg`, the rack checks
    and grip.
  - **Levelling is not free (SHARED §3).**
    - A levelled height is y·cos θ + z·sin θ, so it brings MediaPipe's regressed depth in at sin θ: depth
      noise (σ_z ≈ 3σ_y) and depth bias.
    - This is why the certified tiers apply only in a **level band, |θ| ≤ 3°**. The band is derived from
      the §5.0 budget.
  - Without certification, the raw camera frame is used, and the unknown pitch leaks forward and back
    motion into heights.
  - **Pitch band** in the features, from `ctx.pitch_certified` and `ctx.mount_pitch_deg`:
    - `"level"`: certified, |θ| ≤ 3°;
    - `"tilted"`: certified, 3° < |θ| ≤ 10°;
    - `"unknown"`: uncertified, or |θ| > 10°.
  - **Press certification requirements:**
    - `mount_source` ∈ {`"rack"`, `"inclinometer"`};
    - the delta tripwire: if the standing-pitch estimate in the rack posture moves more than **3°** from
      its certification reading, certification is revoked for the session (`ctx.pitch_certified`
      becomes False).
  - **Start tiers are session-dependent, so the profile overrides
    `cue_start_tier(fault_type, ctx)`** (SHARED §4).
    - For mono `press_leg_drive` and `press_uneven`, it returns the tier for the band:
      - `level`: leg drive moderate; uneven severe, or moderate when `faults.press.uneven_promoted` (§8.3);
      - `tilted`: severe if `faults.press.tilted_band_validated` (default False, set by the §8.4 tilted-band
        gate); otherwise the observability cap applies (recap-only);
      - `unknown`: severe.
    - In the `unknown` band, and in the `tilted` band until it is validated, the two rules also cap
      `observability` at `approximate`, which makes them recap-only and hedged.
    - `min_cue_severity` stays the static default (the level-band tiers plus the rig tiers) and is never
      mutated.
    - A mid-session revocation takes effect on the next verdict through `ctx`.

**Normalizers**
- `arm_ref_m = 1.15 × torso`, with torso from `ctx.body_proportions.torso_length_avg`, else the in-set
  median |S_mid − H_mid| over rack frames.
- `leg_ref_m` = femur + tibia, else the in-set median.

**Facing check (mono), image-plane only.** `trunk_rotation` uses monocular z (V2), so it isn't used.
- `width_ratio = |S_L.x − S_R.x| / W_ref`.
- **W_ref = the max over this session's sets of the per-set median `shoulder_width_x` over rack and setup
  frames.** A median, not a p95, so it isn't biased high. Taken from frames where the arms are down.
- Judged on **setup and press-start frames only** (bar < rack + 0.3·ROM). Overhead frames, where the
  shoulder keypoints narrow, are excluded.
- `facing_ok = (median width_ratio over those frames ≥ cos 20° = 0.94)`.
- Noise: σ_w ≈ 1–1.4 cm on 38 cm is 2.6–3.7% per frame; the per-rep median over ≥ 4 frames is ≤ 1.9%.
  A frontal lifter therefore sits ≥ 3σ above the cutoff (synthetic test ≥ 95% pass; §8.3 gate).
- **First set** (W_ref from the same set): a lifter rotated throughout set 1 can't be caught by the
  ratio. A weak anthropometric guard flags `W_ref < 0.87 × 0.70·torso` (> ~30°). Rotation of 20–30° held
  for all of set 1 is undetectable on mono and is reported by validation.

### 4.2 Per-frame `PressFrameSample` (`observe(ctx)`, measured frames only)

| Field | Formula |
|---|---|
| `bar_cm` | 100·h(W_mid, S_mid) |
| `bar_above_hip_cm` | 100·h(W_mid, H_mid) |
| `tilt_deg` | deg(atan2(h(W_L, W_R), \|(W_L − W_R)·lat_R\|)); + = left hand higher |
| `elbow_l/r_deg` / `elbow2d_l/r_deg` | 3D `elbow_flexion`; frontal projection 180 − ∠(S′,E′,W′) with ′ = (x, y, 0) |
| `hip_cm` | 100·h(H_mid, A_mid) |
| `knee_flex_deg` | `JointAngles.avg_knee_flexion` |
| `layback_deg` (rig) | deg(atan2(−(T·fwd), T·up)), with T = S_mid − H_mid. Checks: T = (0, −0.5, +0.1) with up = WORLD_UP → +11.3° ✓; upright T ∥ up → 0° ✓ |
| `bar_midfoot_m` (rig) | horiz_g(W_mid − midfoot)·fwd; + = bar ahead of mid-foot |
| `bar_fwd_shoulder_m` (rig) | horiz_g(W_mid − S_mid)·fwd |
| `grip_tilt_l/r_deg` | deg(atan2((W − E)·out_side, (W − E)·WORLD_UP)); + = wrist outside the elbow |
| `shoulder_width_x` | \|S_L.x − S_R.x\| |
| `trunk_lateral_deg` | `JointAngles.trunk_lateral_flexion` (recorded) |

Missing keypoints give NaN, never 0. Extrapolated frames are skipped.

### 4.3 Rep windows (centred 3-sample median of `bar_cm`)

- `top` = the p95 of the bar height.
- `rack_h` = the median of setup samples, else `rack_bar_cm`.
- `ROM = top − rack_h`.
- `i_start` = the last sample before the top that is ≤ rack + 0.05·ROM.
- `i_lock` = the first sample ≥ top − 0.05·ROM.
- **Lockout window**: the contiguous samples ≥ top − 0.05·ROM. If fewer than 2, use the 2 highest.
- **Press window**: [i_start, i_lock).
- **Lower window**: after lockout.
- **Setup window**: the analyser's 1.0 s idle ring buffer, restricted to in-band frames that are nearly
  still (|Δbar| < 2 cm/frame). For rep 1, use the rack reference.
- **Dip window**: [t_start − 0.5 s, t_start + 0.5·press_time].
- Minimum sample counts (otherwise NaN): press ≥ 4, lockout ≥ 2, setup ≥ 4.

### 4.4 Feature table

O = observable, A = approximate, N = not observable. The mono/rig column gives single camera first, then
the rig.

| Feature | Formula | Normalization | Window | Mono / rig | Noise handling at 12 fps |
|---|---|---|---|---|---|
| `top_height_ratio` | p95 `bar_above_hip_cm` − rack shoulder-above-hip | ÷ 100·arm_ref | rep | O / O | Hip-anchored |
| `rack_height_ratio` | median `bar_cm` | ÷ 100·arm_ref | setup | O / O | median of ≥ 3 |
| `lockout_elbow_l/r_deg` | median `elbow2d` (mono) / `elbow` (rig) | angle | lockout | A / O | median of ≥ 2 |
| `leg_drive_ratio` | hip_ref − min(med3 `hip_cm`) | ÷ 100·leg_ref | dip | O / O | med3 before the min |
| `knee_flexion_change_deg` | max − setup median | angle | dip | N / O | recorded |
| `layback_deg` | p90 of med3 `layback_deg` | from `ctx.up` | press only | N / O | p90 of ≥ 4 |
| `layback_change_deg` | p90 press − setup median | angle | press vs setup | N / O | gravity-bias-free; supporting |
| `bar_over_midfoot_ratio` | median `bar_midfoot_m` | ÷ arm_ref | lockout | N / O | needs measured gravity (§5.0) |
| `bar_forward_shoulder_ratio` | median `bar_fwd_shoulder_m` | ÷ arm_ref | lockout | N / O | fallback metric when gravity is `"body"` |
| `bar_tilt_deg` | p90 \|tilt − rack_tilt\|, sign of median | angle; `rack_tilt` is pooled over the set (§3.2) | press + lockout, bar ≥ rack + 0.2·ROM | O / O | med3 + p90; pooled reference (≈ 15–25+ frames); facing check |
| `grip_tilt_deg` | mean of L/R setup medians | angle | setup | O / O | median of ≥ 4; facing check |
| `press_time_s`, `lower_time_s` | timestamps | — | phases | O / O | ±1 frame |
| `concentric_velocity_mps` / `velocity_loss_pct` | Δbar / Δt; via `PressSessionReference.annotate` | — | press | N / O | rig only; NaN if `press_start_observed` is False |
| `highlight_score` | `concentric_velocity_mps` on clean reps ("fastest") | — | press | absent / O | rig only (SHARED §4) |
| `facing_ok`, `wrists_clipped`, `partial_bottom`, `press_start_observed`, `gravity_source` | flags | | | | |
| `exercise = "overhead_press"`, `rep_number`, `max_depth_angle` (= 180 − lockout elbow), `arm_ref_m`, `leg_ref_m` | required keys (SHARED §2) + provenance | | | | |

**Monocular honesty.**
- Every mono-observable feature is a vertical height, an image-X distance or a frontal-projected angle.
- On a levelled (certified, tilted) mount, those **heights and tilts include the regressed depth at
  sin θ**. That is budgeted in §5.0, and it is why the certified tiers stop at |θ| ≤ 3°.
- Pure depth-axis values (layback, bar offsets, 3D elbow, velocity) are computed on mono for research
  only. Their faults are never emitted there.

---

## 5. Fault catalog

Common to all faults:
- One verdict per rep, via `RepFaultRule.judge_rep(features, reference, angles)` → `_rep_fault(...)`
  (`fault_types.py:276-330`).
- `details` carry `side`, `phase`, `observability`, `is_drift`, `value`, `unit` (+ `gravity_source` for
  rig sagittal faults).
- `is_drift = reference.is_drift(metric, mild)`. NaN means no verdict. Fault types are `press_`-prefixed.

### 5.0 Error budget, start tiers and recorded-only verdicts

**Rule (deadlift plan §2.9).** A tier is cued only when the P95 measurement error against ground truth
(§8.2) is ≤ 1/3 of that tier's threshold. Otherwise cueing starts one tier higher.

**Mechanism (SHARED §4).** `OverheadPressProfile.min_cue_severity: dict[fault_type, FaultSeverity]` holds
the start tiers below. It is built for the active capture mode. A fault that is missing starts at mild.

The pipeline stamps `details["recorded_only"] = True` on any verdict below its fault's start tier and
sets its cue to None. Per SHARED §4, these verdicts are filtered explicitly on every path:
- the bridge flags `recorded_only: true` on the `fault` message and keeps them out of `faults_in_rep`;
- `coaching_service` passes the flag through;
- the orchestrator's `_add_fault_candidate` drops them on both the `fault` path and the `faults_detailed`
  loop.

That last step matters because a None cue alone would be refilled from `fault_to_cue`. A recorded-only
verdict:
- does **not** make the rep unclean (`RepData.is_clean`, `utils/types.py:404-406`), so the clean
  highlight is kept;
- does **not** count as "persists" in `_resolve_cue_outcome` (`coaching_orchestrator.py:1219`). The
  SHARED §4 praise rule applies. It is decided over the existing hold window (`FIX_HOLD_REPS` = 2, timing
  unchanged).
  - A lifter cued at a 7° tilt whose hold window contains any recorded-only uneven verdict (say 5°) hears
    `press_uneven_better`, or nothing if the clip is missing, but never `_fixed`.
  - That resolves the outcome, so no later `_fixed` plays for the same cue.
  - `_fixed` plays only when no hold rep carries any verdict of the fault;
- is **excluded** from recap fault counts. It appears in the recap only hedged, and only when its type
  is in `faults.press.hedged_recap_faults` (empty by default; filled from the §8.3 gates).
  - The hedge **never states a size** (a moderate verdict can be recorded-only under a severe start tier).
    Example: "possibly, hard to judge from this camera".
  - Otherwise it is never mentioned.

The tiers below are pre-validation estimates. §8.3 replaces each one with the measured P95.

**Vertical-reference term (rig).** The bar at lockout is ≈ 2.15 m above mid-foot, so 1° of `up` error
gives **3.75 cm** of fake forward offset. Layback takes the error 1:1.
- `"body"` (hip→ankle, 2–4° off): bar-over-mid-foot is biased by 7.5–15 cm, more than the moderate
  threshold.
- `"measured"` (floor ChArUco, per-camera agreement ≤ 1°, residual target ≤ 0.5°): bias ≤ 1.9 cm.
- **Pitch-leak term (mono, SHARED §3).** A vertical difference reads Δy·cos θ + Δz·sin θ, and the
  sin θ term is the leak:
  - **Leg drive.** The hips shift 3–6 cm forward as the lifter leans to clear the chin, inside the dip
    window. That leaks 0.4–0.8 cm at 8° and **0.8–1.6 cm at 15°**. Added to the 1.5–1.9 cm noise P95,
    it exceeds the 1.9 cm moderate allowance.
  - **Uneven.** Bar yaw (one hand 2–5 cm further forward overhead than at the rack) leaks 0.3–0.7 cm at
    8° and **0.5–1.3 cm at 15°**, which is 0.5–1.4° of tilt on top of the ≈ 2.0° pooled P95. It is over
    the allowance.
  - **Acceptance** (`top_height_ratio`). The bar sits 0–10 cm in front of the hips, so it leaks up to
    2.6 cm (0.04 arm lengths) at 15°. That is within the 0.12 margin.
  - **Grip, facing.** Lateral (image-X) or ratio metrics, so the leak is negligible.
  - **Certified (levelled) mount.** The motion leak is gone (the ±1° inclinometer residual leaves
    ≈ 0.1 cm / 0.09°), but levelling adds two depth terms at sin θ (SHARED §3).
    - **Depth noise.** The per-frame vertical σ becomes σ_y·f(θ), with f(θ) = √(cos²θ + 9 sin²θ)
      (assuming σ_z = 3σ_y). That is f = 1.011 at 3°, 1.03 at 5°, 1.075 at 8°, 1.114 at 10°, 1.24 at
      15° and 1.39 at 20°. The noise part of each P95 scales with f.
    - **Depth bias.** Assumption, to be measured on the ≈ 15° sessions: the within-rep change in depth
      error has a P95 of ≈ 3 cm, both for the hips over the dip window and for the left-right wrist
      depth between rack and overhead. It enters as 3·sin θ cm (0.16 cm at 3°, 0.52 cm at 10°,
      0.78 cm at 15°). For tilt that is atan(3·sin θ / 55) (0.16° at 3°, 0.54° at 10°, 0.81° at 15°).
    - Budget = noise P95 · f(θ) + depth bias + inclinometer residual, added linearly to stay
      conservative:

      | θ | Leg drive P95 (central 1.5 / upper 1.9 cm noise) | vs moderate 1.9 / severe 3.1 cm | Uneven P95 (central 2.0° noise) | vs moderate 2.0 / severe 3.0° |
      |---|---|---|---|---|
      | 0° | 1.60 / 2.00 | moderate: central passes, upper doesn't | 2.09 | moderate: no; severe: yes |
      | 3° | 1.77 / 2.18 | moderate: central passes | 2.28 | severe: yes |
      | 5° | 1.91 / 2.32 | moderate: no | 2.42 | severe: yes |
      | 8° | 2.13 / 2.56 | severe: yes | 2.68 | severe: yes |
      | 10° | 2.29 / 2.74 | severe: yes | 2.86 | severe: yes |
      | 15° | 2.73 / 3.23 | severe: upper fails | 3.38 | severe: no |

    - **Derived limits (conservative).**
      - Leg drive keeps its moderate start only at |θ| ≤ 3°, provisionally (the upper-noise case fails
        even when level, so replay decides).
      - Uneven exceeds the moderate allowance at every pitch, so it starts at severe on mono and is
        promoted only by replay.
      - Both stay within the severe allowances up to 10°.
      - Above 10° they are recap-only.
  - **Uncertified mount** (pitch unknown; the unlevelled leak reaches 1.6 cm / 1.4° at 15°): no tier
    can be guaranteed, so mono leg drive and uneven are capped at `approximate`, recap-only and hedged.
  - The raw `camera_pitch` estimate is never used as a correction (unvalidated, SHARED §3). It serves
    only as the certified-mount delta tripwire.

**Two worked P95 estimates (mono).**
- **Uneven press.** Per-frame tilt noise is atan(√2·σ_wrist/spacing) ≈ 1.5° (σ_wrist ≈ 1 cm, 0.55 m
  spacing).
  - After med3 (σ of a median of 3 ≈ 0.67σ): ≈ 1.0° per frame.
  - The rack-tilt reference is a constant offset b within the rep.
    - A 5-frame reference (round 4) gives σ_b ≈ 0.54σ ≈ 0.8°, P95 1.3°, and a total of ≈ 2.6°. That is
      the avoidable half of the error.
    - **Pooled over the set's in-band rack frames** (§3.2): σ_b ≈ 1.25·1.5°/√N = 0.48° at N ≈ 15 (rep 1)
      and 0.38° at N ≈ 25. That gives P95_b ≈ 0.8° and ≈ 0.6°.
  - On a clean rep, the feature is the p90 over ~10 frames of |b + e|. Total ≈ P95_b + 1.28·1.0 ≈
    **2.1° at rep 1, 1.9° later**.
  - That sits at the moderate allowance (6°/3 = 2.0°). The inclinometer residual alone (+0.09°) tips it
    over, and frame correlation within a hold shrinks the effective N. So, conservatively, **mono
    starts at severe** (allowance 3.0°).
    - It is promoted to moderate when the replayed clean-rep P95 at the deployed pitch is ≤ 2.0°.
  - Why promotion is safe once replay confirms it:
    - On a truly level bar, a false moderate verdict (≥ 6°) needs ≈ 4σ of clean-rep noise.
    - On a slightly tilted bar it still names the correct lagging hand.
  - The rig (σ_wrist ≈ 0.5 cm) gives ≈ 1.0°, so the rig starts at moderate.
- **Leg drive.** Assume σ_hip ≈ 0.8–1.0 cm per frame.
  - After med3: 0.54–0.67 cm. hip_ref, the setup median: σ ≈ 0.45–0.55 cm.
  - The min over ~6 effective samples is biased by ≈ −1.27σ_med3, with spread ≈ 0.43σ_med3.
  - That gives a clean-rep P95 of ≈ **1.5–1.9 cm**, which fits the moderate allowance (1.9 cm). Mono
    therefore starts at **moderate**, provisionally.
  - Pre-registered fallback: if the replayed clean-rep P95 exceeds 1.9 cm, it starts at severe.
  - The rig (σ ≈ 0.5 cm) gives ≈ 0.95 cm, which allows mild if confirmed.

| Fault | Thresholds mild / mod / sev | Allowed error (1/3 of each) | Expected P95 error (to verify) | `min_cue_severity`, mono | `min_cue_severity`, rig |
|---|---|---|---|---|---|
| `press_layback` | 12 / 18 / 25° | 4 / 6 / 8.3° | measured: gravity ≤ 0.5° + trunk line ≈ 2° + partial migration ≈ 1° → ≈ 2.5° | — (not observable) | measured: mild. `"body"`: approximate cap, recap-only |
| `press_bar_forward` | 6.2 / 10.5 / 15.5 cm | 2.1 / 3.5 / 5.2 cm | measured: 1.1–1.9 (gravity) ⊕ 1 (wrist) ⊕ 0.5 (locked midfoot) → 1.6–2.2 cm. `"body"` shoulder fallback: ≈ 4–5 cm | — | measured: moderate (mild once Kinovea P95 ≤ 2.1 cm). `"body"`: approximate cap |
| `press_lockout` | 15 / 25 / 35° | 5 / 8.3 / 11.7° | rig 3D ≈ 3–5° ⊕ migration ≤ 6° → ≈ 6–7°. Mono 2D unknown | approximate until promoted, then by measured P95 | moderate |
| `press_leg_drive` | 3.3 / 5.6 / 9.3 cm | 1.1 / 1.9 / 3.1 cm | mono level ≈ 1.6–2.0 cm, plus depth terms when levelled at a tilt (budget table); rig ≈ 0.95 cm | `cue_start_tier`: level band moderate, provisional (severe if replay P95 > 1.9 cm); tilted band severe once validated (recap-only before); unknown band recap-only | mild if P95 ≤ 1.1 cm, else moderate |
| `press_uneven` | 4 / 6 / 9° | 1.3 / 2.0 / 3.0° | mono level ≈ 2.0–2.1° with the pooled reference (≈ 2.6° without), plus depth terms at a tilt; rig ≈ 1.0° | `cue_start_tier`: level band **severe**, promoted to moderate only if replay P95 ≤ 2.0°; tilted band severe once validated (recap-only before); unknown band recap-only | moderate |
| `press_grip` | 10 / 15 / 22° | 3.3 / 5 / 7.3° | L/R mean of setup medians ≈ 2° (fits mild) | moderate: **a deliberate coaching choice**, not error-driven (below) | moderate |
| `press_velocity_loss` | 20 / 30 / 40% | 6.7 / 10 / 13.3% | 30 fps boundary ±3% ⊕ velocity noise | — | moderate (mild if P95 ≤ 6.7%) |

**Why grip starts at moderate even though its error would allow mild.**
- Grip is a once-per-set, between-set note that competes with the next set's focus line.
- A 10–15° forearm tilt (≈ 4.5–7 cm per side) is within the "roughly vertical" range coaches accept
  without comment.
- At moderate (≥ 15°, ≈ 7 cm per side), a coach says "a thumb wider/narrower".
- Mild grip verdicts are recorded and can be hedged only through `hedged_recap_faults`.

### 5.1 `press_layback` (priority 50, safety, rig only)

- **Definition:** the trunk leans back beyond what clearing the chin requires.
- **Metric:** `layback_deg` from measured vertical, p90 over the press window.
- **Thresholds:**
  - 12° mild: clearing the chin needs ~7–9° if the head doesn't retract.
  - 18° moderate: a steep incline press; shoulder-flexion limits are compensated by lordosis (2024
    study).
  - 25° severe: the continental backbend (Soriano 2019).
- **Vertical reference:** `ctx.up` from measured gravity. With `"body"` the fault is capped at
  approximate. `layback_change_deg` is recorded as the gravity-bias-free companion.
- **Observability:** mono N. Rig A until promotion (§8.3: rig gates **and** gravity residual ≤ 0.5°),
  then O when measured.
- **Drift:** `press_layback_deg`. **Side:** none. **Phase:** `press`.
- **Safety:** severe bypasses the focus rule once observable.
- **Cue `press_layback`:** "Glutes tight, ribs down".
  - Preemptive: "glutes tight, ribs down this set".
  - Positive: "Stayed stacked, good."
- **Labels:** "Ribs down" / "leaning back too far under the bar".

### 5.2 `press_bar_forward` (priority 51, rig only)

- **Definition:** the bar finishes ahead of mid-foot.
- **Metric:**
  - `"measured"` gravity: `bar_over_midfoot_ratio`, median over lockout, one-sided. Ground-anchored, so
    shoulder migration can't bias it.
  - `"body"` gravity: `bar_forward_shoulder_ratio` (short lever), capped at approximate.
- **Thresholds** (÷ arm_ref; for 188.5 cm, ankle→toe ≈ 17 cm and midfoot ≈ 6 cm ahead of the ankle):
  - **0.10 mild** (≈ 6 cm, over the ball of the foot). The standard is bar over mid-foot and the
    shoulder joint (Rippetoe; StrongLifts).
  - **0.17 moderate** (≈ 10.5 cm, over the toe tips; the lifter must lean back).
  - **0.25 severe** (≈ 15.5 cm, beyond the toes; shoulder moment ≈ load × 0.15 m).
- **Error budget:** §5.0. 1° of vertical error = 3.75 cm, so `"body"` gravity is unusable for this
  metric.
- **Observability:** mono N. Rig A until promotion (rig gates, plus gravity residual ≤ 0.5° **or**
  Kinovea bar-over-mid-foot P95 error ≤ 2.1 cm), then O with measured gravity.
- **Drift:** `press_bar_forward`. **Side:** none. **Phase:** `lockout`.
- **Cue `press_bar_forward`:** "Head through at the top".
  - Preemptive: "get your head through at the top".
  - Positive: "Bar right over you, good."
- **Labels:** "Head through" / "finishing with the bar out in front".

### 5.3 `press_lockout` (priority 52)

- **Metric:** per side, the median over the lockout window (rig 3D, mono frontal). Judged on the worse
  side.
- **Thresholds:**
  - 15° mild: judges see a bend at ~10°, plus ≈ 5° of error. Hyperextension of ≤ 10° (Beighton) reads
    below 15°.
  - 25° moderate.
  - 35° severe; the rep still counts.
- **Observability:** mono **A** and rig A. Promotion per mode (§8.3). The start tier is in §5.0.
- **Drift:** `press_lockout_deg`.
- **Side:** the worse side if |L − R| ≥ 8°, else `both`. **Phase:** `lockout`.
- **Cues:**
  - `press_lockout`: "Reach the bar to the ceiling".
  - `_left` / `_right`: "Straighten your left arm up top" / "…right arm…".
  - Preemptive: "finish every rep reaching tall". Positive: "Locked out tall, that's it."
- **Labels:** "Lock out" / "not locking the elbows out overhead".

### 5.4 `press_leg_drive` (priority 53)

- **Metric:** `leg_drive_ratio`. Hip drop d = L(1 − cos θ/2). A jerk or push-press dip ≈ 58 ± 9° of knee
  flexion (Bishop et al., SCJ 2017).
- **Thresholds:**
  - 0.035 mild (≈ 3.3 cm, θ ≈ 30°).
  - 0.06 moderate (≈ 5.6 cm, θ ≈ 40°).
  - 0.10 severe (≈ 9.3 cm, θ ≈ 51°, a push-press dip).
  - Start tier (`cue_start_tier(fault_type, ctx)`; `min_cue_severity` holds only the static default):
    - mono, level band (certified, |θ| ≤ 3°): moderate, provisional (severe if the replayed P95
      exceeds 1.9 cm);
    - mono, tilted band (certified, 3–10°): severe once the tilted-band gate passes, recap-only before;
    - mono, unknown band (uncertified or > 10°): recap-only (approximate);
    - rig: mild or moderate (§5.0).
  - Verdicts below the start tier are recorded-only.
- **Gap:** knee kicks under ~30° are invisible on mono. The rig logs `knee_flexion_change_deg`
  (V1.1 thresholds 10/20/35°).
- **Observability:** O / O. **Drift:** `press_leg_drive`.
- **Phase:** `setup` if the minimum falls before `t_start`, else `press`.
- **Cue:** "Keep your knees locked".
  - Preemptive: "knees stay locked the whole set". Positive: "Legs stayed locked, good."
- **Labels:** "Knees locked" / "bending the knees to drive the bar up".

### 5.5 `press_uneven` (priority 54)

- **Metric:** `bar_tilt_deg`.
- **Thresholds:**
  - 4° mild (≈ 3.5–4 cm at 0.50–0.60 m spacing).
  - 6° moderate.
  - 9° severe (one hand-height of lag).
  - The squat's bar-detector thresholds of 2/4/7° use endpoints 1.3–2 m apart, so the press is
    deliberately coarser.
  - Start tier (`cue_start_tier(fault_type, ctx)`):
    - mono, level band: **severe**, promoted to moderate only if the replayed P95 is ≤ 2.0°. The pooled
      reference gives 1.9–2.1° plus the 0.09° inclinometer residual, against the 2.0° allowance.
    - mono, tilted band: severe once the tilted-band gate passes, recap-only before;
    - mono, unknown band: recap-only (approximate).
    - rig: moderate (§5.0).
  - Verdicts below the start tier are recorded-only.
- **Observability:** O / O, behind the facing check. **Drift:** `press_tilt_abs`.
- **Side:** the lagging hand. **Phase:** the peak phase.
- **Cues:**
  - `press_uneven`: "Push both hands up evenly".
  - `_left` / `_right`: "Drive your left hand up" / "…right…".
  - Preemptive: "keep the bar level all the way up". Positive: "Bar stayed level, nice."
- **Labels:** "Press evenly" / "one hand lagging, bar tilting".

### 5.6 `press_grip` (setup, between sets only)

- **Metric:** signed `grip_tilt_deg` over the setup window. **Judged once per set, on the first rep with
  `facing_ok` and a finite grip reading.** The rule's flag resets in `reset()`.
- **Thresholds:** 10° / 15° / 22° (≈ 4.5 / 7 / 10 cm per side on a 26 cm forearm).
  - Basis: vertical forearms (Rippetoe; StrongLifts).
  - Gundersen 2025: grip width changes the horizontal bar forces.
- **Observability:** O / O.
- **Direction:** `direction = "narrower"` if the value is > 0, else `"wider"`.
- **Delivery:**
  - no mid-set cue;
  - excluded from per-rep counts;
  - appears as the recap `setup_note`;
  - the next-set prompt carries one action, e.g. "Next set, hands a thumb wider".
- **Label:** direction-aware `fault_label("press_grip", details)`: "grip too wide — forearms leaning in"
  or "grip too narrow — elbows flaring out". Report label: "Grip width".

### 5.7 `press_velocity_loss` (priority 55, rig only)

- **Metric:** `velocity_loss_pct`. Thresholds 20/30/40% (Sánchez-Medina & González-Badillo 2011).
- **Why rig only:** press velocities are low (MVT ≈ 0.19 m/s), so the frame error is relatively larger.
- **Observability:** mono N, rig A → O. Always `is_drift`.
- **Implementation:** `VelocityLossRule(fault_type, phase)`.
- **Cue:** "Fast off your shoulders". Positive: "That one moved, good."
- **Labels:** "Drive" / "reps slowing down".

### 5.8 `press_missed` (event, never cued)

- Emitted via `on_rep_rejected` (§3.5).
- **Observability:** O / O.
- Recap: the `missed_reps` count. LLM label: "a press that stalled before lockout (not counted)".
- Listed in `END_OF_REP_FAULT_TYPES`, and excluded from per-rep counts.

### 5.9 Observability matrix (new classes in `faults/observability.py`)

| Fault | Class | single_camera | triangulated | Promotion |
|---|---|---|---|---|
| `press_leg_drive`, `press_missed` | `press_vertical` | observable | observable | — |
| `press_uneven`, `press_grip` | `press_frontal` | observable | observable | — |
| `press_lockout` | `press_elbow` | **approximate** | approximate | per mode, §8.3 |
| `press_layback`, `press_bar_forward`, `press_velocity_loss` | `press_side_view` | not_observable | approximate | §8.3 rig gates + gravity residual ≤ 0.5° (or Kinovea for bar forward) |

- **Runtime cap.** With `gravity_source == "body"`, the rules set `observability = "approximate"` for
  `press_layback` and `press_bar_forward` regardless of the table, so they stay recap-only.
- **Formula transfer.** A metric validated on mono keeps its formula on the rig. Depth-axis metrics need
  their own rig validation.
- **Promotion record.** Each promotion is a one-line table edit plus a test, citing the report.

---

## 6. Delivery

**Priorities (SHARED §6).** These are additive `FAULT_CUE_PRIORITY` entries, ranks 50–59:

| Fault | Rank |
|---|---|
| `press_layback` | 50 |
| `press_bar_forward` | 51 |
| `press_lockout` | 52 |
| `press_leg_drive` | 53 |
| `press_uneven` | 54 |
| `press_velocity_loss` | 55 |

`press_grip` and `press_missed` have no cue key. Mono start tiers come from `cue_start_tier` by pitch band (§4.1, §5.0):
- level band: leg drive from moderate (provisional), uneven from severe until promoted;
- tilted band: both severe once the tilted-band gate passes (recap-only before);
- unknown band: both recap-only.

`press_lockout` joins if it is promoted.

**Cue path (SHARED §4).**
- `cache_cues` carries `fault_to_cue`.
- `orchestrator.set_exercise(name, is_squat, fault_to_cue, reports_depth)` stores it.
- `_add_fault_candidate` and the rest-complete line use it.
- Cues therefore survive the 3 s fault cooldown (V12).

**Orchestrator rules, unchanged:**
- one focus fault per set;
- ≤ 2 cues per fault per set;
- moderate+ is cued;
- mild on 2 of 3 reps is cued, but a verdict below the fault's `min_cue_severity` is `recorded_only`: never cued, not unclean, not "persists", not counted in the recap (§5.0, SHARED §4);
- approximate faults are never cued;
- **Praise rule (SHARED §4)**, decided over the hold window after a cue (`FIX_HOLD_REPS` = 2; timing
  unchanged):
  - any hold rep carries a recorded-only verdict of the cued fault → `<cue>_better`, or nothing when the
    clip is missing. It never falls back to `_fixed`;
  - no hold rep carries any verdict of the cued fault → `_fixed`.
  - Either praise **resolves** the outcome: after `_better`, no later `_fixed` for the same cue. A new
    cue starts a new outcome.
- **Every agent-side reader** of `faults_detailed` / `RepData.faults` goes through
  `fault_filters.reportable_faults(faults, spoken)` (SHARED §4). That covers the orchestrator loop,
  `last_rep_verdict` → `check_my_form`, the `show_me("last_rep")` replay narration, recap fault counts
  and progress summaries.
  - Recorded-only verdicts are always dropped.
  - When spoken, approximate verdicts (e.g. mono `press_lockout`) are dropped unless hedged.

**`_better` keys** are drafted for every press fault whose start tier is above mild. Each is ≤ 6 words:

| Key | Line |
|---|---|
| `press_leg_drive_better` | "Better, knees staying straighter." |
| `press_uneven_better` | "Better, bar's leveling out." |
| `press_lockout_better` | "Better, almost locked out." |
| `press_bar_forward_better` | "Better, bar closer over you." |
| `press_velocity_loss_better` | "Better, that one moved more." |

There is no `_better` key for `press_layback` (start tier mild when gravity is measured; recap-only under
`"body"`) or `press_grip` (no in-set cue).

**Press constant set (SHARED §4):**
- `SAFETY_FAULT_TYPES = {press_layback}`;
- `SIDE_VIEW_FAULTS = {press_layback, press_bar_forward, press_velocity_loss}`.

Start tiers come from the profile's `min_cue_severity`, not from a constant set.

**Timing and positives.**
- The cue lands between reps (the rep ends at the rack or chin turnaround).
- Positives: `_fixed` lines, `clean` (recorded-only verdicts don't break it), and `best_rep_so_far` on the
  rig only.
- The `depth_target_met` highlight is only added when `diagnosis_enabled` is set.

**Next-set line.** The rest-complete prompt names the focus fault through the profile map. A moderate+
grip adds one setup action.

**Set summary and recap depth (`reports_depth = False`, SHARED §4).**
- `cache_cues` carries `reports_depth`, and `orchestrator.set_exercise(..., reports_depth)` stores it.
  `coaching_service` doesn't coerce a null depth to 0.0.
- The producer (`ipc_bridge.send_set_complete`, per-rep fields) sends `avg_depth`, `depth_consistency`,
  per-rep `max_depth_angle` and `depth_category` as null. The display hides null tiles.
- The orchestrator's `_build_set_summary` and `_queue_exercise_recap` (non-squat branch):
  - drop the `shallow_*` fields and per-rep `depth` / `depth_angle`;
  - guard `max_depth_angle` = None;
  - add `missed_reps` and `setup_note: {fault_id: "press_grip", direction}`;
  - exclude `press_grip` and `press_missed` from `fault_summary` and per-rep `faults`.
- Recap prompt:
  - a missed-reps line ("information about the load, not failure");
  - a setup line;
  - hedging for approximate faults;
  - **best-rep line via `highlight_score`** (SHARED §4):
    - Rig: `PressRepFeatures.highlight_score = concentric_velocity_mps` on clean reps, meaning
      "fastest". The line reads "Best rep: rep N, their fastest clean press".
    - Mono: the field is absent, so the line is omitted. Velocity isn't trusted at 12 fps, and ranking
      clean reps by tilt would rank noise.
    - No line ever uses depth terms;
  - no `last_depth` in the motivation context.

**Recap payload** (flat; the small local model relays, never interprets):

```
{exercise: "overhead_press", set, reps_counted, missed_reps, clean_reps,
 faults: [{fault_id, label, reps:[n..], worst_severity, side, phase,
           is_drift, observability, cued, fixed_after_cue}],   # recorded_only verdicts excluded
 possible: [{fault_id, label}] | [], # recorded-only, hedged without a size; only types in faults.press.hedged_recap_faults
 setup_note: {fault_id, direction} | null,
 effort: fresh|working|near_limit|null,   # concentric-time ratio (P4)
 best_rep: n|null}                 # by highlight_score; rig only
```

No raw angles.

---

## 7. Implementation plan (in order)

### 7.0 Shared plumbing

**The press conforms to `SHARED-PLUMBING.md` §1–§8, with no redefinitions.** How the press uses it:

**Profile and analyser.**
- `create_rep_analyzer(config) → PressRepAnalyzer`. The analyser reads `config.faults.press` and
  implements §2 exactly: `observe(ctx)`, `on_rep_start(piston)`, `finish_rep(rep_number)`,
  `rejection_reason(features)`, `discard_rep()` and `reset_set()`.
  - **Piston order:** the ended rep's `finish_rep` runs first; grip is not judged on piston reps (§3.5).
  - Required feature keys: `exercise`, `rep_number`, `max_depth_angle`. Features are dumped after
    `rule_engine.finish_rep`.
  - Optional methods used: `is_ready(ctx)` (the rack predicate, ANDed with the standing gate) and
    `counter_baseline()` (rack height, re-seeded after every counter reset).
- `create_session_reference() → PressSessionReference(SessionReference)`. It overrides `update()` with
  the press drift metrics and keeps `annotate`.
- `create_set_diagnosis() → None`, so `waits_for_diagnosis` is False.
- `get_readiness_check() → None`; the profile holds no analyser state.
- `on_rep_rejected` returns `[press_missed]`, tagged by the pipeline via `tag_observability`.
  `shallow_rep_cue` keeps its default (`None`), because no press fault sets `shallow_rep`.
- `reports_depth = False`.
- `min_cue_severity` for the capture mode (§5.0).
- `observe_frame`, `allows_camera_refine` and `needs_bar_3d` keep their defaults.

**Context.**
- `FrameContext.up` (physically up, used directly) and `gravity_source` feed every rig sagittal metric.
- One camera uses `pitch_certified`, `mount_pitch_deg`, `geometry.level_camera_frame` and the delta
  tripwire (tolerance 3°) per §4.1, with the level band at 3° and the tilted band at 10°.
- `cue_start_tier(fault_type, ctx)` is overridden for mono leg drive and uneven (§4.1);
  `min_cue_severity` stays static.
- The raw `camera_pitch` estimate is never used as a correction.

**Config, maps, registry.**
- Config lives in `faults.press`: thresholds, the §3.5 constants, `arm_to_torso_ratio`, `counter`,
  `hedged_recap_faults` (empty by default), `tilted_band_validated` (False) and `uneven_promoted` (False;
  set from the §8.3 promotion row and read by `cue_start_tier`).
- Ranks 50–55; `press_*` measurement rows.
- The spec's press variant list, plus the longest-match additions in Step 4.

**Delivery.**
- The `fault_to_cue` and `reports_depth` fields of `cache_cues`.
- Recorded-only verdicts.
- `highlight_score`.
- Per-exercise constant sets.
- Per-exercise progress queries.
- `draft_cue_text.py --exercise overhead_press`.

**Tooling.**
- Golden masters, including squat → press → squat.
- Raw recorder and replay.
- Dev override.

**Press-only additions** (same style; the defaults keep the squat unchanged):

| # | Addition | Why |
|---|---|---|
| P1 | `ExerciseProfile.tracked_keypoints`. Default `LEG_KEYPOINTS`, moved first from `pipeline.py:87` to `utils/types.py` to avoid a circular import. | Clipped wrists must trigger tracking-lost |
| P2 | `HipPositionCounterConfig.velocity_window_s` (default None = 5 frames) | 30 fps noise (§3.4) |
| P3 | `VelocityLossRule(fault_type, phase)` parameters | Reuse without collision |
| P4 | Analyser profiles set `RepData.ascent_time` = concentric and `descent_time` = eccentric, from the features | The orchestrator's effort reads the concentric time |
| P5 | `fault_label(fault_type, details=None)`, direction-aware | Grip wording |
| P6 | Non-squat set-summary additions: `missed_reps`, `setup_note`, `possible`, and per-set events excluded from per-rep counts | §6 |
| P7 | `press_setup` pre-set clip | Readiness hold |

**No spec gaps and no conflicts remain.** Round 3's gaps are now in the spec: the `"measured"` gravity
source, `is_ready`, and re-seeding `counter_baseline`.

### 7.1 Steps

**Step 0 — baselines and contract**
- Build the squat golden masters (SHARED §8).
- Add a press section to `.claude/squat-audit/CONTRACT.md`:
  - fault types and ranks;
  - phases `setup` / `press` / `lockout` / `lower`;
  - press feature keys;
  - cue keys;
  - `gravity_source` in rig details;
  - set-summary fields.
- Fix CONTRACT §5 Cartesia → ElevenLabs.

**Step 1 — SHARED-PLUMBING items and P1–P3.** Whichever branch lands first provides the shared items.
After each change, the golden masters must stay byte-identical.

**Step 1j — `pipeline.py` hooks (beyond SHARED §1–§2)**
- **H1 (SHARED §2).** `StandingPoseGate` ANDs `analyzer.is_ready(ctx)` with its own checks over the
  consecutive-frame window. The readiness `FrameContext` uses the raw, hip-centred skeleton, with
  `angles = None`. The failure reason is `"setup_position"`.
- **H2 (SHARED §2).** At the latch, and inside `pipeline.reset_rep_counter()` (which replaces the
  rest-end resets in V11), `counter_baseline()` seeds the counter.
- **H3 (P1).** `missing_keypoints`, `_legs_present` and the extrapolation check use
  `profile.tracked_keypoints`.
- **H4, rep end on the analyser path (SHARED §2 order):**
  1. `features = analyzer.finish_rep(n)`.
  2. On a rejection reason: run the §3.5 rejection sequence.
  3. Otherwise: `rep_faults = rule_engine.finish_rep(angles, n, features)` (annotates
     `velocity_loss_pct`), **then** `rep_data.features = features.model_dump()`, then P4 timing.
- **H5.** `get_depth_metric` → 180 − mean elbow flexion (Step 4). `RepData.max_depth_angle` stays the
  counter's metric and is nulled on the wire (`reports_depth`).

**Step 1k — other pipeline-side files**

| File | Change |
|---|---|
| `coaching/session_tracker.py` | `depth_target_met` highlight only when `diagnosis_enabled`; press `best_rep_so_far` on the rig |
| `coaching/ipc_bridge.py` | `cache_cues.fault_to_cue`; depth nulls and `_rep_depth_category = ""` when `not profile.reports_depth` (the bridge stores the flag at `prepare_exercise`); `send_set_complete`'s `fault_summary` skips `details.recorded_only` verdicts (SHARED §4, "Persistence and summaries") |

**Step 2 — `analysis/press_features.py` (new)**
- `PressFrameSample`, `PressRepFeatures` (with `drift_metrics()`).
- `PressRepAnalyzer`: `is_ready`, mono levelling via `level_camera_frame` when certified, `pitch_band`, rack reference,
  ring buffer, windows, `ctx.up`-based axes, piston handling, `rejection_reason`, `counter_baseline`,
  `highlight_score`.
- `PressSessionReference`, `missed_rep_event()`.
- Pure NumPy, with the frame stated in the module docstring.

**Step 3 — rules (one `RepFaultRule` file each)**
- `press_layback.py`, `press_bar_forward.py` (metric chosen by `gravity_source`), `press_lockout.py`,
  `press_leg_drive.py`, `press_uneven.py`, `press_grip.py` (once per set, first valid rep).
- Velocity via P3.
- Export them in `rules/__init__.py`.

**Step 4 — `profiles/overhead_press.py` (rewrite)**
- **Delete:**
  - the scaffold rules from this profile (the rule files stay; deadlift, RDL and curl use them);
  - the scaffold counter config and the `"lockout"→press_lockout` map;
  - `get_depth_metric` → `avg_elbow_flexion`, replaced by `180 − avg_elbow_flexion`;
  - the aliases `shoulder_press` and `barbell_press`.
- **Names:** `overhead_press`, `ohp`, `military_press`, `standing_press`, `strict_press`,
  `barbell_overhead_press`.
- **`UntrackedVariantProfile` registrations:**
  - The SHARED §7 press list.
  - **Longest-match additions**, needed because `overhead_press` / `military_press` would otherwise
    match: `seated_overhead_press`, `seated_barbell_overhead_press`, `seated_military_press`,
    `dumbbell_overhead_press`, `kettlebell_overhead_press`, `single_arm_overhead_press`,
    `smith_machine_overhead_press`.
  - Other variants (Arnold, Z, push jerk) match no press name and fall to untracked by default.
- `display_name = "the standing barbell overhead press, from the rack"`; `coaching_ready = False`;
  `reports_depth = False`.
- Hooks per §7.0. Rules in priority order.
- `get_rep_signal` per §3.3; counter config from `faults.press.counter`.
- `get_fault_to_cue_map()`, plus `get_cue_dict()` with side variants and `press_setup`.

**Step 5 — delivery (agent side)**

| File | Change |
|---|---|
| `coaching/cue_cache.py` | Ranks 50–55 |
| `agent/services/coaching_constants.py` | Delete `press_elbows`, `press_bar_path`, `press_even`; add the §5 keys, side variants, `press_setup`, press `_fixed`, and the §6 `_better` lines |
| `agent/services/progress_context.py` | Press `FAULT_LABELS`; P5. The "most common fault" lines come from the persistence aggregates below, which skip recorded-only verdicts. |
| `agent/services/coaching_orchestrator.py` | Press constant set; `_tempo_effort` matches `*velocity_loss`; §6 set-summary and recap changes (P6); `highlight_score` best-rep line; `press_setup` once before set 1. Recorded-only handling, the `reportable_faults` filter in the `faults_detailed` loop and recap counts, and the `_better` praise rule all come from SHARED §4. |
| `agent/services/coaching_service.py` | Rest-complete grip `setup_note`; `last_rep_verdict()` (`:394`, feeds `check_my_form`) returns `reportable_faults(faults, spoken=True)`, with approximate verdicts carried only as hedged |
| `agent/agents/workout_agent.py` | `check_my_form` (`:1095`) and the `show_me("last_rep")` replay narration (`:1141`) read faults only through `reportable_faults(..., spoken=True)` |
| `agent/services/fault_filters.py` | Shared `reportable_faults` (SHARED §4). The press adds nothing to it. |
| `agent/agents/shared/helpers.py` | Aliases "strict press", "standing press", "standing overhead press", "barbell overhead press", plus plurals. "push press" / "shoulder press" / "seated press" / "dumbbell press" stay unmapped, so Nova declines and offers the standing barbell press. |
| `src/visual/display.html` | `FOCUS_TEXT` press entries |
| `src/db/biomechanics_persistence.py` | `END_OF_REP_FAULT_TYPES += {"press_missed"}`. Recorded-only verdicts are stored (flagged) in `faults_detailed` but skipped by `top_faults` (`:181-215`), `get_fault_progress` (`:853`) and `get_multi_session_fault_trends` (`:929`), per SHARED §4. |

**Step 6 — cue text and audio**
1. Draft with `draft_cue_text.py --exercise overhead_press`, word limit 6. Drop "never mention the bar"
   for the press.
2. The user reviews `cues.json`.
3. Generate with `generate_cue_audio.py --cue <press keys>`.
4. Flip-gate test: every press `get_cue_dict()` key, every press `_fixed` key and every press `_better`
   key (§6) has a clip on disk. The drafting scenarios for `_better` say "improved but not proven gone";
   the lines are reviewed with the rest.
   Clips stay gitignored, so the device image must ship them.

**Step 7 — recorder and replay (SHARED §8).** Every processed frame is recorded, pre-latch included:
- raw 3D with confidences (21×4) and 2D (21×3);
- capture time, frame size, capture mode, `lost_cameras`;
- control events: `reset_readiness_gate`, `presence_only`, `reset_rep_counter`, `set_exercise`,
  `rest_start`, `apply_athlete_params`, `set_depth_target`;
- live outputs: latch times, reps, faults, cues;
- config hash and git SHA;
- mono mp4 with PTS = capture time.

Replay:
- `ReplayPoseSource` is passed as the pipeline's `pose_source`.
- `replay_press.py` replays the control events at their timestamps and runs gate → pre-IK → IK →
  analyser → counter → rules → `SessionTracker`.
- Outputs: features CSV, confusion matrices with confidence intervals, and a report.
- Fixture (npz): a mono session with walk-in, unrack, walk-out, 2 sets, a missed rep and a re-rack.

**Step 8 — validation, tuning, promotions (§8). Step 9 — flip `coaching_ready`, then run `graphify
update .` once.**

### 7.2 Tests

**Unit**
- `test_press_features.py`:
  - production-frame signs;
  - mono lateral = image X (a Z perturbation leaves frontal features unchanged);
  - windows and min-sample NaNs;
  - hip-anchored acceptance is unchanged by a +5 cm shoulder lift;
  - normalizer fallbacks.
- **Facing:** a frontal lifter with σ = 1 cm (and 1.4 cm) keypoint noise has `facing_ok` on ≥ 95% of
  reps over 200 synthetic reps. Overhead frames are excluded from the facing window. 30° yaw → NaN.
- **Gravity sign and tilt** (shared test, SHARED §3):
  - Under both `"body"` (`up = WORLD_UP`) and a 3°-tilted `"measured"` `up`:
    - an upright trunk (T ∥ `up`) reads `layback_deg` = 0 ± 0.5°;
    - a trunk leaning back 10° reads +10 ± 0.5°;
    - a bar 8 cm in front of mid-foot reads positive `bar_over_midfoot_ratio` (+0.13 ± 0.01).
  - With the world tilted 3° from true gravity and the true `up` supplied as `"measured"`: both metrics
    are unbiased (within 0.3 cm / 0.2°).
  - With `"body"` in the same scene: `press_bar_forward` switches to the shoulder metric, and both faults
    carry `observability="approximate"`.
  - A sign-trap test: passing −`up` (the round-3 bug) must fail the upright-trunk assertion.
  - Vertical metrics (`top_height_ratio`, `leg_drive_ratio`, `bar_tilt_deg`) change by < 1%.
- **Mono pitch leak and levelling** (`test_press_synthetic.py`). Anisotropic noise at the budget's
  **central** values:
  - hips: σ_xy = 0.8 cm, **σ_z = 2.4 cm**;
  - wrists: σ_xy = 1.0 cm, **σ_z = 3.0 cm**.

  Seeded, so the test isn't flaky.
  - **Geometry, noise-free:** a 15° pitch, a 5 cm hips-forward shift in the dip window, and 4 cm of bar
    yaw overhead (none at the rack).
    - Levelled with `mount_pitch_deg=15`: `leg_drive_ratio` < 0.003 and `bar_tilt_deg` < 0.3°.
    - Uncertified: the leak is visible (≈ 0.014 and ≈ 1.1°).
  - **Noise, 200 clean synthetic reps per pitch** at 0°, 3°, 10° and 15°, levelled:
    - the noise-only clean-rep P95 matches the §5.0 noise term (central noise P95 · f(θ)) within ±10% at
      every pitch;
    - adding the stated bias and residual terms, leg drive is ≤ 1.9 cm at 0° and 3° (central 1.60 /
      1.77);
    - both metrics are within the severe allowances (3.1 cm / 3.0°) at 10°;
    - the test asserts nothing about moderate above 3°;
    - at 15°, the test only asserts that the band is `unknown` and the faults are recap-only.
  - **Tripwire mid-set** (`test_press_start_tiers.py`):
    - On a level certified mount with uneven promoted to moderate, a moderate (7°) `press_uneven`
      verdict is cueable.
    - Then a tripwire delta of 4° sets `ctx.pitch_certified=False`.
    - The **next** moderate uneven verdict has `recorded_only=True` and `observability="approximate"`,
      and is not cued. The same holds for a moderate leg-drive verdict.
    - `min_cue_severity` is unchanged; only `cue_start_tier(…, ctx)` changed.
- **Start tiers and recorded-only** (`test_press_start_tiers.py`):
  - On mono, a 5° (mild) `press_uneven` verdict carries `recorded_only=True` and no cue, and the rep
    stays clean.
  - With uneven at a moderate start (rig, or mono level band after promotion): after a moderate uneven
    cue (7°), a hold window of [4.5° recorded-only, clean] plays
    `press_uneven_better` at the end of the window and resolves the outcome. A later clean rep plays
    **no** `_fixed` for that cue.
  - So does [clean, 4.5° recorded-only].
  - A separate cue followed by a hold window with no uneven verdict on either rep plays
    `press_uneven_fixed`.
  - A recorded-only verdict is absent from recap fault counts. It appears under `possible`, with no size,
    only when `"press_uneven" in faults.press.hedged_recap_faults`.
  - With the default mono level-band tier (severe): a 7° verdict is recorded-only, and a 10° verdict is
    cued.
- **Recorded-only verdicts are never cued** (`test_coaching_orchestrator.py`, the press instance of the
  SHARED §4 test). A mono `press_uneven` recorded-only verdict (5° with uneven promoted to moderate;
  7° under the default severe tier):
  - (a) arrives as a `fault` message with `cue=None, recorded_only=true`;
  - (b) on the next rep, arrives only inside `faults_detailed` (flagged).

  In both cases there is no cue, no candidate in `_add_fault_candidate`, and no "persists" in
  `_resolve_cue_outcome`. A pending outcome for an earlier uneven cue resolves to `press_uneven_better`
  (never `_fixed`). It plays nothing if that clip is missing.
- **Recorded-only verdicts never reach aggregates** (`test_biomechanics_persistence.py`,
  `test_progress_context.py`, and a new `tests/test_biomechanics/test_ipc_bridge_set_summary.py`; the
  press instance of the SHARED §4 test). A mono
  session in which `press_uneven` was only ever recorded-only produces:
  - no "most common fault: one hand lagging" line;
  - no `get_fault_progress` or `get_multi_session_fault_trends` entry for `press_uneven`;
  - no `press_uneven` count in the set-summary `fault_summary` the display shows.

  The verdicts are still present, flagged, in the stored `faults_detailed`.
- **Voice tools never state untrusted verdicts** (`tests/test_fault_filters.py`, `test_coaching_service.py`,
  `test_workout_agent.py`; the press instance of the SHARED §4 test).
  - Setup: a mono rep whose only `press_uneven` verdict is recorded-only (5°), plus an approximate mono
    `press_lockout` (20°).
  - `check_my_form` (via `last_rep_verdict`) and `show_me("last_rep")` must either:
    - say there is nothing to correct in what this camera can see; or
    - hedge the lockout ("possibly not fully locked out, hard to judge from this camera").
  - They never say "one hand lagging" and never state the lockout as fact.
- **Praise rule:**
  - A cued `press_uneven` (moderate) whose 2-rep hold window contains any recorded-only uneven verdict
    plays `press_uneven_better`.
  - With that clip removed, it plays nothing (never `_fixed`).
  - Either way the outcome is resolved: no later `_fixed` for that cue.
  - `_fixed` plays only when no rep in the hold window carries any uneven verdict.
- `test_press_rules.py`: tiers, side, phase, `is_drift`, NaN → None. Grip is judged once per set on the
  first valid rep.
- `test_press_analyzer.py`:
  - every `rejection_reason`;
  - the first rep after the latch is accepted with no ring samples;
  - a rejected rep gets no verdicts and no reference update;
  - `velocity_loss_pct` is present in `RepData.features` (dump after `finish_rep`).
- `test_hip_position_counter.py`:
  - `seed_baseline` and `velocity_window_s`;
  - the press config at 12 and 30 fps with injected noise (σ 1.0 and 0.5 cm, plus 1.0 cm at 30 fps);
  - zero phantom reps over 60 s of rack hold, and correct counts on 10-rep sequences.
- Other updates:
  - `test_standing_gate.py`: `is_ready` is ANDed with the gate's checks. A rack pose with bent knees
    fails; hands at the sides fail; the rack pose passes. The readiness `FrameContext` has
    `angles = None`.
  - Piston order: the ended rep's verdicts are emitted before the new rep's `on_rep_start(piston=True)`,
    and grip is never judged on a piston rep.
  - `test_profile_registry.py` (names, spec and longest-match variants → untracked);
  - `test_pipeline.py::TestExerciseProfiles` (press rules, `_bilstm is None`, analyser path skips the
    squat blocks, `reset_rep_counter` re-seeds);
  - the Kalman-lag test (delayed, not dropped);
  - `test_coaching.py` (scaffold press tests replaced).
- **Agent:**
  - `test_press_cue_survives_suppressed_fault_message`:
    - Rig mode (uneven start tier moderate).
    - Rep N sends `press_uneven` **moderate** (left) via `fault` and is cued.
    - Rep N+1, 2 s later, has a moderate uneven verdict only in `faults_detailed` (the bridge cooldown
      dropped the message).
    - The candidate resolves to the cue key `press_uneven_left` through the profile's `fault_to_cue`,
      not None. It is cued again once the cue gap allows (≤ 2 per set).
    - Repeat on the mono level band with severe verdicts (10°).
    - Mild verdicts are recorded-only in both modes and never become candidates.
  - Set summary: no depth fields, `missed_reps` and `setup_note` present, grip and missed excluded.
  - Effort uses concentric time; the press safety set applies; `press_setup` plays once.
  - `highlight_score`: the rig names the fastest clean rep; mono omits the line; no depth words appear
    (`reports_depth=False` via `cache_cues`).
  - Rest-complete grip note; direction-aware label; persistence; cue-clip presence.

**Synthetic poses in the production frame** (`test_press_synthetic.py`). Builder:
`world_press_points(bar_height_m, elbow_flex_deg, layback_deg, hip_drop_m, tilt_deg, grip_tilt_deg,
bar_forward_m, yaw_deg, shoulder_lift_m, gravity_tilt_deg)`.
- Proportions: Y-down, toes at −Z, 188.5 cm (torso 0.543, femur 0.462, tibia 0.464, upper arm 0.35,
  forearm 0.275, shoulder width 0.38).
- Sequences at 12 and 30 fps:
  - rack → press → lockout → lower;
  - touch-and-go;
  - missed rep;
  - re-rack with the hands down;
  - bare-hand reach;
  - clipped wrists.
- **First rep, pinned in frames:**
  - 5 in-band frames before the bar leaves the band → counted;
  - 4 → missed;
  - a 0.30 s stationary hold plus the in-band rise (≥ 5 in-band frames at 12 fps) → counted, with
    `press_start_observed = False` when the rise started before the latch.
- **Assertions:**
  - clean sequences produce zero faults;
  - each injected fault fires its tier only;
  - a Y-up-flipped skeleton fails the rack predicate.

**Replay** (`test_press_replay.py`): the mono fixture's raw stream reproduces the live run's rep count,
rep timestamps and fault list exactly, unrack and latch included. The squat golden masters run in the
same suite.

---

## 8. Validation protocol

### 8.1 Recordings

**Who.** ≥ 8 lifters, 155–195 cm, including the 188.5 cm user, both sexes. Tape-measure biacromial width,
arm length and leg length.

**How it's recorded.**
- **Mono:** `--record-raw` at ~12 fps, framed feet to wrists overhead by **stepping back or turning
  the camera to portrait, not tilting it**.
  - Most sessions use a level, inclinometer-certified mount.
  - **At least 2 sessions per lifter (≈ 25% of mono reps) at ≈ 15° pitch**, recorded both with the
    pitch entered (to measure σ_z and the depth bias after levelling) and treated as uncertified (to
    measure the raw leak).
  - Plus 1 session at ≈ 3° (the level-band edge).
  - Plus **1 session per lifter at ≈ 8° (tilted band)**, containing the obvious leg-drive and uneven sets,
    which gives ≈ 43 severe positives each. These feed the tilted-band gate.
- **Rig:** ≥ 30 fps with a loaded calibration (V15) **and a ChArUco floor gravity measurement**
  (`measure_gravity.py`) at every session. The per-camera residual is logged.
- **Reference video:** synced side-view phone (60–240 fps) plus a frontal phone (Kinovea).
- **Markers:** acromion and wrist markers on 4 lifters, for migration (§8.2).

**Acted sets.** Light load, empty bar to ~50% of 5RM. Each set is 6 reps, faulty on reps 2, 4 and 6.

| Fault | Acted sets per lifter | Expected moderate+ positives per lifter | Total (× 8) |
|---|---|---|---|
| `press_leg_drive` | 2 obvious (full push-press dip, hip drop ≥ 10 cm, which is ≥ severe) + 1 moderate (half dip, ~5–7 cm) + 1 subtle (~25° knee bend) | severe ≈ 0.9·6 ≈ 5.4; moderate+ ≈ 5.4 + 0.8·3 ≈ 7.8 | severe ≈ 43; moderate+ ≈ 62 |
| `press_uneven` | 1 obvious per side, lag **≥ 10°** (≈ 9.5 cm, a hand-height, so ≥ severe) + 1 moderate per side (6–8°) + 1 subtle | severe ≈ 5.4; moderate+ ≈ 5.4 + 0.8·6 ≈ 10 | severe ≈ 43 (≈ 21 per side); moderate+ ≈ 82 |
| `press_lockout` | 2 obvious (~25–30° short) + 1 subtle (~10–15°) | ≈ 6.3 | ≈ 50 |
| `press_layback` (rig) | 2 obvious + 1 subtle | ≈ 6.3 | ≈ 50 |
| `press_bar_forward` (rig) | 2 obvious + 1 subtle | ≈ 6.3 | ≈ 50 |
| `press_velocity_loss` (rig) | 2 sets to near failure at ~75–80% 1RM, with an LPT if available | ≈ 5 | ≈ 40 |
| `press_missed` | 3 acted stalls + natural | 3+ | ≥ 24 |
| `press_grip` | Per-setup protocol: 3 wide, 3 narrow, 3 correct mini-sets of 2 reps, each with a re-rack and regrip | 3 per direction | 24 wide, 24 narrow, 24 correct |

**Yield margin.** The ≥ 40 target leaves ~20% margin. If round-1 yield falls below 5 per lifter, add a
third obvious set or two lifters.

**Other sets.**
- Natural sets: 3 × 5, reported separately.
- Negatives: ≥ 100 per fault per mode.
- The §3.7 edge-case script: ≥ 30 phantom opportunities per mode, including 4- and 5-frame first-rep
  holds and a 25° rotated stance.
- **Facing set:** ≥ 3 sets per lifter recorded frontally with no rotation, to measure the facing-NaN
  rate.

**Sequencing.**
- Run the **mono replay early**: on the first 3–4 lifters' level-band sessions, before the demo is
  filmed. That way the uneven promotion and the leg-drive confirmation (§8.3) are decided first.
- The demo is then filmed on the rack mount or a webcam level within 3°, with the pitch entered.

**Volume.** ≈ 115 mono reps and ≈ 150 rig reps per lifter, over 2 sessions each.

### 8.2 Labelling and measurements

**Coach labels.** A certified coach (CSCS or SSC) labels blind, from frontal and side video:
- rep status;
- per fault: absent / mild / moderate / severe + side;
- grip per setup.

A second labeller covers 20% of reps. Any fault with κ < 0.6 is redefined and relabelled.

**Kinovea measurements:**
- elbow angle at lockout;
- trunk angle at mid-press;
- bar offset from mid-foot at lockout (the side camera is levelled with a spirit level, giving
  gravity-true ground truth);
- hip drop;
- forearm frontal angle.

**Shoulder-keypoint migration.** Rack→lockout displacement of the keypoint against the acromion marker.
The acceptance margin (0.12) must exceed the p95 vertical migration ÷ arm_ref. Forward migration bounds
the bias in the elbow angle and the shoulder fallback.

**Gravity.**
- `measure_gravity.py` residual per camera, against a spirit-level board (target ≤ 0.5°).
- The angle between body-up and measured gravity is reported per session.
- `bar_over_midfoot_ratio` error is computed against Kinovea under both gravity sources.

**Format:** a `labels.json` per recording.

### 8.3 Acceptance metrics (per capture mode)

**Statistics.** Lifter-clustered bootstrap 95% confidence intervals (2,000 resamples), with design effect
1 + (m − 1)ρ and ρ = 0.1 to start. With ≈ 50 positives (m ≈ 6), the effective n is ≈ 33.

| Metric | Gate |
|---|---|
| Rep counting | precision ≥ 0.98 (lower bound ≥ 0.96), recall ≥ 0.97; zero phantom reps; ≤ 1 set in 20 off; first-rep miss rate reported |
| Cued faults, **at the active start tier** | Per fault, mode and band: system cue-eligible verdicts (at or above the active `cue_start_tier`, observable, not recorded-only) against coach labels at or above **that same tier**. Precision ≥ 0.80 (lower bound ≥ 0.65); recall ≥ 0.65 (lower bound ≥ 0.50); severity within one tier ≥ 0.90; natural-set precision ≥ 0.75 when there are ≥ 15 natural positives. |
| Moderate-tier stats (reported, not gating) | For faults whose active tier is above moderate (mono uneven; leg drive in the tilted band): system verdicts ≥ moderate, including recorded-only, against coach ≥ moderate, plus the clean-rep P95. These are the inputs to the "Uneven promotion" row. |
| Error budget → `min_cue_severity` (§5.0) | Per fault and mode, the lowest tier whose threshold/3 ≥ the measured P95 error (vs Kinovea, lifter-clustered) becomes the start tier. Mono uneven starts at severe and is promoted per the "Uneven promotion" row; mono leg drive keeps its provisional moderate start (level band) only at a replayed P95 ≤ 1.9 cm, otherwise severe. |
| **Hedged recap mentions** (`faults.press.hedged_recap_faults`) | A fault type is added only when, for that fault and mode: mild precision ≥ 0.70 (lower bound ≥ 0.55) **and** mild threshold ≥ 2× the clean-rep p95 of its metric (noise floor). Otherwise recorded-only verdicts are never mentioned. |
| Grip (per setup) | precision ≥ 0.80, recall ≥ 0.70 per direction, with confidence intervals |
| **Facing** | facing-NaN rate on frontal sets ≤ 5% per lifter (no lifter above 10%); otherwise retune W_ref or the cutoff |
| Clean reps | ≤ 10% carry any moderate+ press fault |
| Approximate faults | in the recap only at precision ≥ 0.70 |
| Promotions | mono `press_lockout`: precision ≥ 0.80, recall ≥ 0.60. Rig side-view faults: cued gates **and** measured-gravity residual ≤ 0.5° (bar forward: or Kinovea P95 ≤ 2.1 cm); valid only when `gravity_source == "measured"`. |
| Pitch bands | On the ≈ 15° sessions with the pitch entered, measure σ_z / σ_y and the within-rep depth-bias change. Recompute the §5.0 budget. Widen the level band (3°) or the tilted band (10°) only if the measured budget allows; narrow them if it doesn't. The uncertified band stays recap-only. |
| Uneven promotion (mono) | Clean-rep P95 ≤ 2.0° at the deployed pitch (level band), with a lifter-clustered upper bound reported → `cue_start_tier` returns moderate |
| Counter noise | measured σ_pos gives entry ≥ 3σ_v in both modes |
| Shoulder migration | acceptance margin rule (§8.2) |
| Arm normalizer | \|arm ÷ (1.15·torso) − 1\| ≤ 0.10 for ≥ 7 of 8 lifters, else add arm segments |
| Latency | p95 rep end → cue ≤ 1.0 s |
| Tuning | thresholds move only toward coach agreement; changes are logged |

### 8.4 Gate for `coaching_ready = True`

1. All tests are green, and the squat golden masters are identical (fresh and after a press switch).
2. Mono §8.3 passes for counting and facing, plus the following tiers in the **level band**:

   | Fault | Tier validated |
   |---|---|
   | `press_leg_drive` | moderate (or severe, if its replayed P95 forced the fallback) |
   | `press_uneven` | severe, plus moderate if promoted |
   | `press_grip` | moderate, per setup |

   - `press_lockout` is either promoted or stays recap-only.
   - The **tilted band is not part of the flip**. Its severe tiers turn on later, via
     `faults.press.tilted_band_validated`, when the ≈ 8° sessions meet the same active-tier gates on
     ≥ 30 severe positives per fault.
   - The **unknown band** is recap-only; only the approximate-precision row applies.
3. Every press cue key, including `_fixed` and `_better`, has a reviewed clip on disk.
4. Three live sessions via `NOWVA_DEV_COACHING_READY`: no phantom reps, at least one cue → fix →
   `_fixed` loop, and a correct recap.
5. The user signs off the press lines in `cues.json`.

Rig faults never block the flip. They stay approximate on the rig until rig and gravity validation, and
are absent on mono.

---

## 9. Architecture fit, risks, open questions, scope

### 9.1 Architecture decisions

- **No causal diagnosis for the press in V1.** The press's causal links are either rig-only symptoms or
  already expressed by `is_drift`. Guardrail 2 keeps the recap flat. V2 candidates: load, overhead
  mobility, lockout strength.
- **Reuse:** squat V1 patterns, plus `SHARED-PLUMBING.md`.
- **Edge:** per frame, 21 keypoints; per rep, ≤ 600 samples. No new model and no cloud in the coaching
  path.
- **Squat untouched:** pinned by the golden masters.

### 9.2 Risks (ranked)

1. **Framing and occlusion.**
   - Overhead reach is ~2.4 m for the 188.5 cm user, so a landscape webcam clips the wrists.
   - Mitigated by the criterion-3 waiver and tracking-lost.
   - Guidance is "step back or turn the camera to portrait; don't tilt it". Tilting leaks pitch into the
     vertical metrics (§5.0).
   - Plates can occlude a pure side camera on the rig.
2. **Thin mono coaching.**
   - The two top faults are rig-only, and the rig has no real-camera recordings or gravity measurement
     yet.
   - **Level, certified mount** (|θ| ≤ 3°): mono V1 cues leg drive from moderate (provisional) and uneven
     press from severe until replay promotes it, plus a grip note.
   - **Uncertified webcam:** counting, the grip note and the recap only.
   - **Demo impact:** the demo needs the rack mount or a webcam kept level within 3° (pitch entered).
     - "Keep your knees locked" is the dependable cue.
     - "Drive your left hand up" fires only on a hand-height lag (≥ 9°) until promotion. Script an
       obvious lag if it is shown.
   - **Hardware requirement for the rack:** the frontal camera is mounted level (0°). Overhead framing
     comes from its height, lens field of view or portrait orientation, never from tilt (open question
     for the hardware workstream).
   - V1.1 bar-detector tilt (~3× the angular resolution) would remove the dependence.
3. **Rig vertical reference.**
   - Without a floor gravity measurement (`gravity_source == "body"`), bar-over-mid-foot is unusable
     (3.75 cm/°) and layback is biased 2–4°.
   - Both stay recap-only until measured gravity is available.
   - A device accelerometer (deadlift open question 7) would remove the dependence.
4. **Rep counting.** The interplay of predicate, seeding, NaN floor, velocity window and acceptance at
   12 fps. Mitigated by noise-injected tests, exact replay, and the zero-phantom gate.
5. **Shoulder-keypoint migration.**
   - Acceptance and bar-over-mid-foot are anchored to hips and feet.
   - The elbow angle (≈ 6°) and the shoulder fallback remain biased. Measured in §8.2.
6. **Facing check on the first set:** 20–30° of rotation held all set is undetectable on mono.
7. **Unsigned elbow angle:** hyperextension > 15° reads as soft.
8. **Leg drive under ~30°** is invisible on mono.
9. **Validation labour, and cue-clip shipping** (clips are gitignored).

### 9.3 Open questions for the user

1. Should a soft lockout of moderate or worse still count? (The plan says yes, with a cue.)
2. Ship mono lockout as recap-only if it fails promotion?
3. Approve the `press_setup` line and the grip note?
4. When will the rig be mounted, and can the ChArUco floor gravity measurement be part of every rig
   install?
5. Is a certified coach available for ≈ 900 mono and ≈ 1,200 rig reps?

### 9.4 Out of scope (V1)

- Push press and jerk.
- Seated, dumbbell, kettlebell, machine, Smith, landmine, Z, behind-the-neck and Arnold presses
  (untracked).
- Bench press.
- Bar-detector tilt and path.
- Rack sagittal checks.
- Mid-press bar loop.
- Signed elbow angle.
- Press diagnosis graph.
- Press visualizer choreography.

---

## 10. References

- Kroell & Mike (2017), *SCJ* 39(6):70–75.
- Gundersen et al. (2025), *Sports Biomechanics*, grip width in the shoulder press.
- Soriano et al. (2019), review of weightlifting overhead pressing derivatives.
- Bishop et al. (2017), *SCJ*, push press (jerk dip ≈ 58 ± 9°).
- Sánchez-Medina & González-Badillo (2011).
- Balsalobre-Fernández et al. (2018), military press MVT.
- Shoulder flexion vs lordosis (2024), PMC11617340.
- Drillis & Contini (1966), via Winter.
- Dill et al. (2023), MediaPipe 3D accuracy.
- Rippetoe, *Starting Strength*, "The Press"; the StrongLifts OHP guide.
- Wulf (2013).
- AAOS shoulder flexion norm; Beighton score.
- Deadlift plan v10 §2.1 (gravity) and §2.9 (error budget).

---

## Changelog

- **Round 1 (2026-10-03):** first draft.
- **Round 2 (2026-10-03):**
  - cue-map fallback;
  - raw recorder and exact replay;
  - per-fault recording arithmetic and confidence intervals;
  - shoulder migration;
  - 30 fps counter window;
  - non-squat set summary;
  - H5 order, `get_depth_metric`, `LEG_KEYPOINTS` move and first-rep timing;
  - §7.0 shared plumbing with the deadlift.
- **Round 3 (2026-10-03):** revised after the 8.8 review and the shared spec.
  1. **Rig vertical (new).**
     - V18: the rig's up is hip→ankle, not gravity.
     - Every rig sagittal metric uses `up_g = −ctx.gravity` (SHARED §3, from `measure_gravity.py`).
     - With `gravity_source == "body"`, `press_layback` and `press_bar_forward` are capped at
       approximate, and bar forward falls back to the short-lever shoulder metric.
     - Added the §5.0 error budget: 3.75 cm per degree at a 2.15 m lever, plus the 1/3-of-threshold
       start-tier rule from deadlift §2.9.
     - Rig promotion is gated on a gravity residual ≤ 0.5° (or Kinovea agreement for bar forward).
     - Gravity measurement, a gravity test, and risk 3 added.
  2. **Facing check (new).**
     - W_ref is now the max of per-set **medians** over rack/setup frames.
     - Facing is judged only on setup and press-start frames (bar < rack + 0.3·ROM).
     - Grip is judged on the first rep with valid facing and a finite reading.
     - Facing-NaN gate in §8.3, and a synthetic ≥ 95% pass test.
     - A first-set rotation guard plus a stated limit.
  3. **First rep.**
     - A "hold" is now defined as in-band frames.
     - Tests pinned in frames: 5 counted, 4 missed, 0.30 s stationary + rise counted.
     - `press_start_observed` flag added.
  4. **Convergence.**
     - Adopted `SHARED-PLUMBING.md` exactly: the analyser protocol without args or ctx; the analyser owns
       the rack reference via its bound `check_rack`; `on_rep_rejected` emits `press_missed`;
       `reports_depth` replaces the `features.exercise` / S14a decision; `exercise` is a required key.
     - The counter moved under `faults.press.counter`; features are dumped after `finish_rep`, so
       `velocity_loss_pct` reaches RepData.
     - The spec's untracked list is used, plus the longest-match additions; ranks stay 50–55.
     - §7.0 now points at the spec and lists only press additions (P1–P9) and spec gaps. No conflicts
       remain.
- **Round 4 (2026-10-03):** revised after the 8.6 review and the updated `SHARED-PLUMBING.md`.
  1. **Gravity sign.**
     - Every rig sagittal formula now uses `ctx.up` directly (§4.1, §4.2), which is physically up and
       never negated. Round 3's `up_g = −ctx.gravity` pointed down under `"body"`: an upright trunk read
       ≈ 180° of layback and `fwd` flipped.
     - The gravity test now checks: an upright trunk reads 0 ± 0.5° and a bar in front reads positive,
       under both `"body"` and a tilted `"measured"`, plus a sign-trap test.
  2. **Untrusted mild verdicts.**
     - Adopted the spec's recorded-only mechanism. Verdicts below `min_cue_severity` no longer make a rep
       unclean, no longer block `_fixed` via `_resolve_cue_outcome`, and are excluded from recap counts.
     - Restored the §8.3 mild gates (mild precision ≥ 0.70, mild threshold ≥ 2× the clean-rep p95) for
       hedged `slight` recap mentions. Recap payload gained `slight`.
  3. **Start tiers.**
     - Replaced `MILD_REPEAT_CUE_FAULTS` / P8 with the spec's `min_cue_severity` (it supports severe).
     - §5.0 now shows the worked P95 estimates. Uneven on mono ≈ 2.6°, above the 2.0° moderate
       allowance, so **mono uneven starts at severe**. Leg drive ≈ 1.5–1.9 cm, so it starts at moderate,
       with a pre-registered severe fallback.
     - The demo impact is stated in §1 and §9.2.
  4. **Readiness and stale notes.**
     - The rack predicate is now `PressRepAnalyzer.is_ready(ctx)`, ANDed with the standing gate. The
       profile returns no readiness check and holds no analyser state.
     - Re-seeding comes from the spec, so the old P1 is gone (P numbering updated) and the stale spec
       gaps are dropped.
     - `press_missed` is tagged by the pipeline via `tag_observability`.
     - Adopted `create_rep_analyzer(config)` and `on_rep_start(piston)`, with the piston order (grip not
       judged on piston reps).
     - Adopted agent-side `reports_depth` via `cache_cues`. Defined the press `highlight_score` as
       concentric velocity, rig only; on mono the best-rep line is omitted.
- **Round 5 (2026-10-03):** revised after the 8.9 review and the updated `SHARED-PLUMBING.md` §4.
  1. **Recorded-only flow.**
     - Adopted the spec's explicit filtering: the bridge flag, the service pass-through, and
       `_add_fault_candidate` dropping verdicts on both the `fault` and `faults_detailed` paths.
     - Added the press never-cued test: a recorded-only `press_uneven` arriving as a `fault` with
       cue=None, and again only in `faults_detailed`.
     - The hedge never states a size; the recap field is renamed from `slight` to `possible`.
  2. **Uneven press.**
     - The tilt reference is pooled over every in-band rack/setup frame in the set (≈ 15 → 25+ frames).
     - Recomputed P95: ≈ 2.1° at rep 1 and 1.9° later. Mono uneven now starts at **moderate,
       provisionally**, with a pre-registered severe fallback.
     - The false-cue margin (≈ 4σ) is stated. §1, §6 and §9.2 demo text updated, and §3.5 piston rule
       updated (within-rep faults still judged).
  3. **Consistency.**
     - Grip's moderate start is now explained as a deliberate coaching choice for a between-set note.
     - The flag is named `faults.press.hedged_recap_faults` (spec name) and used in the tests and §8.3.
- **Round 6 (2026-10-03):** adopted the SHARED §4 "Persistence and summaries" skip rule. Recorded-only
  verdicts are now excluded from `top_faults`, `get_fault_progress`, `get_multi_session_fault_trends` and
  the `send_set_complete` `fault_summary` (Step 1k, Step 5). Added the press instance of the aggregate
  test (§7.2).
- **Round 7 (2026-10-03):**
  - Adopted SHARED §4 `reportable_faults` for every agent-side fault reader the press touches: the
    orchestrator loop, `last_rep_verdict` / `check_my_form`, the `show_me` replay, recap counts and
    summaries.
  - Added the press voice-tool test (recorded-only uneven plus approximate mono lockout → "nothing to
    correct in what this camera can see", or a hedged lockout).
  - Renamed the bridge test file to `tests/test_biomechanics/test_ipc_bridge_set_summary.py`.
  - Adopted the `_better` praise rule. Drafted `_better` lines for leg drive, uneven, lockout, bar forward
    and velocity loss, and added them to the cue-text step, the clip gate and the tests.
  - Camera mount: stated that no press metric depends on camera pitch, so `pitch_certified` and the
    tripwire are unused.
- **Round 8 (2026-10-03):**
  1. **Pitch leak (mono).**
     - Withdrew the claim that no press metric depends on camera pitch.
     - Added the sin θ term to the §5.0 budget. Leg drive leaks 0.8–1.6 cm and uneven 0.5–1.4° at 15°.
     - Certified mounts are levelled with `geometry.level_camera_frame(mount_pitch_deg)` before every
       vertical and tilt metric (§4.1). Press certification needs `rack` or `inclinometer`,
       |pitch| ≤ 20°, and a 3° delta tripwire.
     - Uncertified mounts start mono leg drive and uneven at severe.
     - Framing guidance is now "step back or portrait, don't tilt".
     - Added a synthetic test (15° pitch, 5 cm hips-forward shift, 4 cm yaw → no verdict after levelling)
       and ≈ 15° validation sessions with a pitch gate.
     - §1 and §9.2 now say the demo needs the rack mount or an entered pitch.
  2. **Praise consistency.** The §5.0 example and the two §7.2 tests now follow the SHARED §4 rule: a
     recorded-only verdict plays `_better` and resolves the outcome (no later `_fixed`). `_fixed` plays
     only when the reps carry no verdict at all.
  3. **Hold-window wording.** Following the reworded SHARED §4, the praise rule is decided over the
     `FIX_HOLD_REPS` = 2 hold window, not "the next rep":
     - any hold rep with a recorded-only verdict → `_better`;
     - `_fixed` only when no hold rep has any verdict.

     §5.0, §6 and both §7.2 praise tests use this wording; the start-tier test covers both window
     orders.
- **Round 9 (2026-10-03):**
  1. **Levelling is not free.**
     - The certified budget now includes depth noise (σ_z = 3σ_y, giving f(θ) = √(cos²θ + 9 sin²θ)) and
       a depth-bias term (3·sin θ cm, assumed and then measured), added linearly (§5.0 table).
     - **Derived bands:**
       - level |θ| ≤ 3°: leg drive moderate, provisional;
       - tilted 3–10°: severe;
       - unknown (uncertified or > 10°): recap-only.
     - **Conservative changes:** mono uneven now starts at **severe** in every band and is promoted to
       moderate only by replay. Uncertified mono leg drive and uneven are now recap-only rather than
       severe. The 20° certification limit is dropped.
     - The levelling test uses anisotropic noise and asserts moderate only up to 3°.
     - "Monocular honesty" now says levelled heights include depth at sin θ.
     - Demo and setup text: "keep the camera level within 3°" (§1, §9.2), plus a level rack-camera
       hardware requirement.
  2. **Session-dependent tiers.**
     - The profile overrides `cue_start_tier(fault_type, ctx)` from `ctx.pitch_certified`, the band and
       the capture mode; `min_cue_severity` stays static.
     - Added a test: a tripwire revocation mid-set turns the next moderate uneven (and leg-drive)
       verdict recorded-only.
- **Round 10 (2026-10-03):**
  1. **Flip gate vs launch configuration.**
     - Cued-fault precision and recall are now measured per fault at its **active start tier**: system
       cues vs coach labels at or above that tier, per band.
     - Moderate-tier stats are reported separately as the uneven-promotion input.
     - The obvious acted uneven lag must be ≥ 10° and the obvious leg drive a ≥ 10 cm push-press dip, so
       severe has ≈ 43 positives. Moderate acted sets were added.
     - §8.4.2 now states the tier validated per fault (level band).
     - The tilted band is now behind its own gate and ≈ 8° sessions (`tilted_band_validated`), and is
       recap-only until it passes (conservative).
  2. **Consistency.**
     - The levelling test runs at the central σ (hips 0.8 cm, wrists 1.0 cm, σ_z = 3σ_xy), seeded, and
       asserts the computed budget terms.
     - The §1 demo loop now leads with "Keep your knees locked" → "Legs stayed locked, good."
     - §5.4/§5.5 band tiers are labelled `cue_start_tier`.
  3. **Scorer follow-ups recorded:** run the mono replay before filming (§8.1 Sequencing, §1); the demo
     uses the rack mount or a webcam level within 3°.
