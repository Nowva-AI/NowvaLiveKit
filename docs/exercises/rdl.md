# Barbell Romanian Deadlift (rack) — coaching plan V1

**Code of record.** `~/NowvaLiveKit-exercises`, `main` @265ef97 (squat V1 + multi-exercise platform,
PR #20). Paths are relative to that checkout.

**Production frame.**
- Verified: `utils/geometry.py:17` `WORLD_UP = (0, -1, 0)`, so **Y is down**.
- `analytical_ik.py`: X = subject's left, +Z = subject's back, so forward = −Z.
- `.claude/rules/biomechanics.md` ("larger Y = higher") is stale.
- Every height below is `h(p) = 100 · dot(p − ankle_mid, up)`: cm above the ankle midpoint of the same
  frame.
  - `up` is the RDL's vertical (§4.1): `FrameContext.up`, the physical-up unit vector of SHARED-PLUMBING
    §3, with fallback `WORLD_UP = (0, −1, 0)`. Nothing negates it.
  - With `WORLD_UP` this equals the existing `rep_features._height_cm`, `(ankle_mid.y − p.y) · 100`.
- `horizontal(v) = v − dot(v, up) · up`.

**Shared plumbing.** Platform hooks, the `RepAnalyzer` protocol, `FrameContext`, the delivery dispatch,
the registry variants, the dev override, the goldens and the raw recorder all follow
`SHARED-PLUMBING.md`, which is aligned with the deadlift plan. This plan does not redefine them. §10
lists only the RDL-specific additions and gaps.

---

## 1. Summary and demo value

**What ships.** The RDL becomes the second coached lift. Everything judged on one frontal camera is a
**vertical height ratio** or a **timing**:
- how far the hips sink;
- how far the hands travel toward the knees;
- how far the torso shortens in the image;
- how long the lowering takes;
- whether one hand drops.

These need no monocular depth and are invariant to the camera's yaw around the athlete. They are not
bias-free. Three effects leak horizontal travel into the vertical readings (budget in §4.3):
- camera **pitch**;
- **perspective**: points that move toward or away from the lens appear to shift vertically in
  proportion to their height offset from the lens;
- on the rig, the world vertical being 2–4° off gravity (deadlift plan §2.1 / O1).

Every absolute threshold is set to survive that budget. Range is only ever judged relative to the
athlete's own range. Faults along the camera axis (bar drift, finish position) are rig-only. Spinal
rounding is not judged, and Nova says so.

**V1 faults.** Each is judged once per rep (`RepFaultRule.judge_rep`). Priorities are the RDL block
40–49 of `FAULT_CUE_PRIORITY` (SHARED-PLUMBING §6); lower wins.

| P | fault_type | One camera | Rig | Cue |
|---|---|---|---|---|
| 40 | `rdl_hip_sink` (squatting it) | observable | observable | "Hips back to the wall" |
| 41 | `rdl_bar_drift` | not observable | approximate until G6, then observable | "Drag the bar down your legs" |
| 42 | `rdl_lockout` | not observable | approximate until G6, then observable | "Finish tall, hips under shoulders" |
| 43 | `rdl_bar_tilt` | observable | observable | "Keep the bar level" |
| 44 | `rdl_range_drift` | observable | observable | "Same depth every rep" |
| 45 | `rdl_tempo` | observable | observable | "Two seconds on the way down" |
| 46 | `rdl_partial` (uncounted attempt; shallow-rep path) | observable | observable | "Take the bar lower" |

**Demo loop on one camera.**
1. The athlete unracks and "squats" the RDL: knees forward, hips dropping.
2. The system measures hips sinking 15% of standing hip height, which is moderate. It cues "Hips back
   to the wall".
3. On the next rep the hips go back instead of down. After two clean reps Nova says "That's it, hips
   went back."

On one camera, `rdl_hip_sink` cues from ~41–57° of knee flexion (mild), depending on shin angle, on a
**certified mount**. That is the demo and product setup: rack mount, or an inclinometer ≤ 2° at setup
(§3.2, §5.1). An uncertified installation cues from moderate (~46–64°) if the lens checks pass, and from
severe otherwise. It deliberately does not catch
subtler knee travel; that limit keeps correct hinges from ever being cued.

**Why the RDL helps the YC demo.**
- It shows the engine generalizes beyond squats with no new model: O(21) arithmetic per frame on the
  Jetson.
- The honesty line ("I can't see your lower back, so stop before it rounds") shows judgement a fitness
  app does not have.

**Guardrail #5 (scope beyond squats).** Squat V1 merged on 2026-10-03 and has not been live-tested. This
plan rides on the deadlift branch's hooks, so it adds little new shared code. Recommendation:
- start after the squat V1 live test, or with a separate owner;
- squat stays byte-identical (G9);
- `coaching_ready` stays False until §8 passes.

---

## 2. Coaching model

### 2.1 What a world-class coach watches, ranked

**Reference.** NSCA RDL technique (Kinetic Select "Romanian Deadlift"; *Exercise Technique Manual*):
- knees slightly flexed (~15°), held through the rep;
- bar as close to the body as possible;
- lower to just below the knees, or until a neutral spine can't be kept.

The NSCA errors are rounded lower back, thoracic kyphosis, excessive lumbar extension at the finish,
not keeping the knee angle, and extending the knees before the hips on the ascent. The in-repo library
entry (`exercise_library.py`, `barbell_romanian_deadlift`) lists the same mistakes.

| Rank | Coach item | One frontal camera, ~12 fps | Rig, ≥30 fps | V1 decision |
|---|---|---|---|---|
| 1 | Spine rounding | not observable | not observable | **Not judged** (§2.2) |
| 2 | Squatting it: hips go down instead of back, knees bend | **Hip sink** (vertical), observable. It measures combined thigh + shin inclination, **not knee travel** (§5.1) | observable | `rdl_hip_sink`, P40 |
| 3 | Bar drifts off the legs; weight to the toes | not observable (camera axis) | observable after G6: wrist midpoint vs midfoot | `rdl_bar_drift`, P41 |
| 4 | Not pushing the hips back | Only when it shows up as hip sink. A straight-legged bend with hips stacked over the heels is **not judged on one camera** (§2.2) | It moves the bar off midfoot, so `rdl_bar_drift` catches it | Covered by 2 (both modes) and 3 (rig) |
| 5 | Range: too short, or too deep for these hamstrings | Count gate uses hand travel; drift uses torso foreshortening relative to own range | observable | Count gate + `rdl_partial` + `rdl_range_drift`, P44. Never "go deeper" past own range |
| 6 | Finish: hips not through, or leaning back | not observable: the cosine is flat near vertical, and the sign is ambiguous | observable after G6: signed trunk pitch | `rdl_lockout`, P42 |
| 7 | Uneven hinge: one side drops or rotates | observable: wrist height difference in the image plane | observable | `rdl_bar_tilt`, P43 |
| 8 | Eccentric control | observable (timing) | observable | `rdl_tempo`, P45 |
| 9 | Hip shift | approximate (`lateral_travel`) | observable | Deferred; bar tilt covers the coachable RDL asymmetry |

Priority among the judged faults follows coach rank. Lockout sits above tilt because its leaning-back
half is a lumbar-load error (NSCA). Range drift sits low because it is a fatigue signal.

### 2.2 Deliberately not judged, and why

**Lumbar and thoracic rounding.** COCO-21 has only shoulder and hip points on the trunk, so hip flexion
and spinal flexion move the shoulder point the same way.
- `BackRoundingRule` reads every correct hinge as SEVERE (§5.8).
- Rejected proxy: shoulder–hip chord shortening (within noise; scapular protraction confounds it).
- Deferred to V2 on the rig: "shoulders keep dropping after the hips stop travelling back". That
  measures the end of hip travel, not the spine.
- Mitigations:
  - honesty clip;
  - the range reference is the athlete's own, and Nova never cues "lower" beyond it;
  - best-rep praise rewards hips-back, never depth;
  - recap ban.

**Not pushing the hips back, on one camera, when the knees stay straight.** A lifter who bends forward
with locked knees and the hips stacked over the heels produces no hip sink. The forward load shift that
follows is along the camera axis. Mitigation:
- the setup clip says "push your hips back";
- on the rig, `rdl_bar_drift` catches it (the load moves ahead of midfoot).

**Knees extending before the hips on the ascent** (NSCA). Hip vertical travel in a correct RDL is 1–4 cm
(§4.4), below 12 fps keypoint noise. V2 on the rig.

**Knee travel (shin inclination) at moderate knee bend, on one camera.** At a fixed knee angle,
inclining the shin *reduces* hip sink (§4.4). The knee keypoint's own height drop is ~1.5 cm between good
and squatty reps, and the hands occlude the knee at the bottom. V1.1 on the rig: 3D knee flexion and shin
angle.

**Locked or hyperextended knees.** The stiff-leg deadlift is a legitimate lift. Separating 0° from 15° of
knee flexion needs 3D knee angles. V2 on the rig.

**Head and neck, shrug, scapular set, grip, bracing, breathing.** Not observable, or depth-based on one
camera, with low value.

**Lateral hip shift.** Approximate on one camera. Rig V1.1.

**Velocity loss.** The RDL is an accessory lift (3×8–10 @ RPE 7 in the program spec). Concentric
velocity is recorded only.

**Foot placement.** Hip-width stance, low value; toe direction is depth-dependent.

---

## 3. Setup and rep definition

### 3.1 Start position, unracking, camera

**Setup.**
- J-hooks at about mid-thigh. The bar on the hooks sits ~6 cm below where the hands hang.
- Overhand or mixed grip just outside the thighs; hip-width stance.
- Unrack by standing tall, step back one or two steps, start standing with soft knees.

**Setup clips.** At the first RDL set of a session Nova plays two pre-generated, agent-side clips (§6):
- `rdl_setup`: "Hooks at mid-thigh. Unrack, one step back, soft knees, then push your hips back."
- `rdl_back_note`: "I can't see your lower back, so stop before it rounds."

**Mono camera placement** (setup screen and validation protocol): level, lens at **hip height
(~0.9–1.0 m)**, 3–4 m away, facing the lifter.
- A level webcam at 3.6 m still frames the whole body: ±25° vertical half-FOV covers −0.7 to 2.6 m.
- This is the same placement as the squat runs; confirm in validation.

**Why hip height** (geometry: D = lens distance, h_hip ≈ 0.93 m):
- The view goes end-on when the lens sits on the extended torso axis, i.e. at hinge
  `θ* = arctan(D / (h_lens − h_hip))`. There the hips hide behind the upper back and head.

  | Lens height | End-on at hinge | Degrades lifters reaching |
  |---|---|---|
  | 0.95 m (hip) | ≈ 90° | ≥ ~85° (horizontal torso) |
  | 1.1 m | 87° | ~82–90° |
  | 1.3 m (chest) | 84° | 80–88° |
  | 1.6 m | 80° | 75–85° |
  | 1.95 m | 74° | 70–80° |

- Below `θ*` the lens sees the chest and abdomen from below, with the hips visible. Above it, the lens
  sees the back from above.
- A hip-height lens also gives the largest shoulder–hip image separation for typical 60–80° hinges: the
  line of sight is horizontal, so the separation is ∝ S·cos θ.
- Perspective is ~0 for the hips (they sit at lens height) and ~1.3 cm for the shoulders at the bottom.
  A chest-height lens adds ~2 cm on the hips and ~4 cm on the shoulders (§4.3).
- A lens below hip height never goes end-on, but shoulder perspective grows (~7 cm at knee height).

**Flexible lifters (≥ 85° hinge) with a hip-height lens.** Bottom frames lose hip visibility. Those reps
fall back to "counting only" (§4.2): no hip-sink or range verdict that rep.

### 3.2 Readiness check

The RDL has an analyser, so its check is the optional `RdlRepAnalyzer.is_ready(ctx)` (SHARED-PLUMBING
§2).
- It is **ANDed** with `StandingPoseGate`'s own checks (visibility, knees, torso, flat feet) and must
  hold over the gate's 5-frame window. It never replaces the gate.
- It is stateless: no analyser setup state is read.
- **Readiness context** (spec §3): readiness runs before pre-IK and IK, so `ctx.angles` is None and
  `ctx.skeleton_3d` is the raw, hip-centred skeleton. The check uses only those raw points and
  `ctx.raw_confidences`.
- It passes when both wrists have raw confidence ≥ `readiness_gate.min_confidence` (0.3) and sit below
  the hip midpoint: `dot(wrist − hip_mid, up) < −0.05 m`, i.e. `wrist.y > hip_mid.y + 0.05` with
  `WORLD_UP`.
- This is a **visibility check**: it guarantees the wrist-based features exist. It cannot confirm the
  bar is held, because arms hanging at the sides also pass.
- On False, the gate's `last_failure = "profile_check"`, and the HUD shows the profile's
  `readiness_hint`, "HANDS IN VIEW" (RDL addition, §10).
- `counter_baseline()` is not implemented. The signal counter keeps its own standing baseline, which the
  spec's `pipeline.reset_rep_counter()` resets as today.

**Camera pitch: certify the installation, not the estimator.**

The mount fields are [SP §3], shared `CaptureConfig`: `mount_source` (`"rack"` / `"inclinometer"` /
`"none"`), `mount_pitch_deg`, `mount_height_m`, env `NOWVA_CAMERA_PITCH_DEG`, plus
`FrameContext.pitch_certified`. The spec lets each exercise state what certification requires and
unlocks. For the RDL:

- **Requirements.** The hip-sink mild tier is unlocked only when **all** of these hold.
  1. **Measured pitch.** `mount_source = "rack"` (factory-fixed angle), or `"inclinometer"` with
     `|mount_pitch_deg| ≤ 2°`.
     - Inclinometer mounts are **re-measured at every session start**: a phone level on the camera body,
       about 10 s; the setup screen asks for the reading. That is the expected procedure. The rack mount
       is a fixed bracket.
     - Setup guidance (spec §3): "step back, or turn the camera to portrait. Don't tilt it."
  2. **Lens height (installation):** `0.8 m ≤ mount_height_m ≤ 1.1 m`.
     - Perspective over-reads hip sink only when the lens is **below** the hips: a level lens at 0.5 m
       adds ~2.3% (§4.3), so the worst correct rep would read ~12.2%, at mild.
     - A lens above the hips under-reads, which is the safe direction. 1.1 m caps that loss and keeps
       ≥ 85° hingers visible (§3.1).
  3. **Lens height (per athlete):** `mount_height_m ≥ hip_floor_m − 0.10`.
     - `hip_floor_m = ctx.body_proportions.femur_length_avg + tibia_length_avg + 0.075` m of ankle height.
       It is read from `ctx` only, so `cue_start_tier` needs no analyser state.
     - Until body proportions exist, only requirement 2 applies.
     - At ≤ 10 cm below the hips, perspective is ≤ 0.7%. A tall lifter (195 cm, hips ≈ 1.03 m) on a
       0.8 m lens fails and runs at moderate.
     - **Why not relax to ~30 cm.** 30 cm is the zero-margin limit at 2° and leaves no room for a knock.
       At 20 cm, perspective ≈ 1.4% and the knock tolerance shrinks from 3.1° to 1.6°. The 10 cm limit
       buys knock tolerance.
  4. **No in-session knock** (delta tripwire below).

  Requirements 1 and 4 make up `ctx.pitch_certified` [SP §3]. Requirements 2 and 3 are checked in
  `cue_start_tier` from the ctx mount values and `ctx.body_proportions`.
  - If only the pitch requirements (1, 4) fail, hip sink runs at start tier **moderate**.
  - If the lens checks (2–3) fail, or `mount_height_m` is unknown, it runs at **severe** (§5.1, spec §3).
- **Leveling on certified mounts** [SP §3]. The analyser applies
  `geometry.level_camera_frame(points, ctx.mount_pitch_deg)` to the skeleton before **every** vertical or
  tilt metric: heights, hip sink, hand travel, hinge, bar tilt, signed pitch.
  - The residual pitch is the inclinometer error (±1°), giving a hip-sink leak ≤ 0.5%.
  - **Leveling is not free** (spec §3): each leveled height is y·cos θ + z·sin θ, so monocular depth enters
    at sin θ. Assumptions, checked by G2(b):
    - **σ_z = 3 cm per frame** (≈ 3 σ_y);
    - **hip depth bias |b_z| ≤ 10 cm** relative to the ankles.
  - Cost at θ ≤ 2° (sin = 0.035):
    - noise 3 cm × 0.035 = 0.1 cm/frame, ≈ 0.05% after the B′ median;
    - bias 10 cm × 0.035 = 0.35 cm, ≈ **0.4%** of hip height.
  - **Largest certified pitch for the mild tier**, from the same budget: 12 − (8.0 + 0.5 + 0.7 + 1.0) =
    1.8% for depth terms plus knock.
    - With no knock allowance, b_z sin θ ≤ 1.67 cm gives θ ≤ ~9.6°.
    - Keeping a ≥ 3° knock tolerance needs θ ≤ 2°: at 10° the bias term alone is ~1.9%.
    - So the RDL's certified limit stays **2°** (the spec's "don't tilt it").
- **Why not the estimator.** `pipeline.camera_pitch_deg` is `_standing_pitch_deg` (`pipeline.py:113-121`),
  taken over the latch window (`:712-724`). It is a depth-regressed ankle→shoulder tilt, so it equals:
  - camera pitch,
  - plus posture (holding a bar leans the body back ~3–5°),
  - plus MediaPipe depth error.

  The latch can happen with or without the bar, so an absolute correction is uncertain by about ±2.5°.
  No tier depends on its absolute value.
- **Knock tolerance** (certified, leveled). Mild holds while the worst correct rep stays < 12%:
  - 12 − (8.0 envelope + 0.7 perspective + 1.0 noise + 0.4 depth bias at 2°) = 1.9% leak budget;
  - 25 cm·sin φ ≤ 1.76 cm gives φ ≤ 4.1°;
  - minus the 1° inclinometer error, that tolerates a **3.1° knock**.
- **In-session delta tripwire** (spec §3).
  - **Axis.** Computed on the **ankle→hip line**, which the standing gate requires straight at latch. It
    is less posture-sensitive than ankle→shoulder: bar-holding lean shows mostly as trunk extension,
    though whole-body sway still enters.
  - **Per-frame noise is large.** σ_z = 3 cm on both hip and ankle gives Δz σ ≈ 4.2 cm over a 0.78–0.92 m
    ankle→hip line, i.e. **≈ 2.6–3.1° per frame**. A 5-frame latch average would still be ~1.5–2°, too
    noisy.
  - **Averaging.** The tilt is averaged over the set's **idle setup frames**: the analyser-observed,
    gate-passing standing frames from the latch to rep 1, plus the idle tops between reps.
    - It needs **≥ 36 frames** (≈ 3 s at 12 fps); with fewer, the check waits for the next set.
    - Assuming ~3-frame temporal correlation, N_eff ≥ 12, giving depth noise σ ≈ 2.6–3.1°/√12 ≈
      0.75–0.9°.
    - Add bar-in-hand posture repeatability ≈ 0.5° for a combined **σ ≈ 0.9–1.0°**. G2(e) measures it.
  - **Reference.** The same average pooled over the session's **first 3 RDL sets** after the session-start
    measurement (2 if the session has ≤ 4 sets). Its σ is σ_set/√3 ≈ 0.55°.
    - Checks run from the next set on. The reference sets themselves are covered only by the
      session-start re-measurement.
  - **Check.** At each later set: `Δ = |set mean tilt − reference|`. Two averages are compared, so
    σ_Δ = √(σ_set² + σ_ref²) ≈ √(0.95² + 0.55²) ≈ **1.1°** (≈ 1.15° at σ_set = 1.0°).
  - **Threshold T = 2.5°.** Predicted rates:
    - **false trips ≈ 2–3% per set**. A false trip is the safe direction: hip sink drops to moderate and
      `level_camera` plays once;
    - **detection ≈ 90–91% at a 4° knock**;
    - **≥ 90% reliable from ≈ 3.9°** (T + 1.28 σ_Δ).
  - **Residual band.** Knocks of **3.1–3.9°** (between the knock tolerance and reliable detection) may go
    undetected.
    - In that band the worst correct rep over-reads by ≤ 0.4% (≈ 12.4%), so it can reach mild only at the
      extreme of the envelope.
    - The exposure is bounded to the rest of that session by the session-start re-measurement.
  - **A trip revokes `pitch_certified` for the session.** Hip sink drops to moderate (`cue_start_tier`),
    `send_pipeline_status("camera_pitch", {"delta_deg": …})`, and Nova plays `level_camera` once.
- **`level_camera` text** is neutral, so it fits both the delta trip and the uncertified gross tripwire
  (where the camera may never have been level): "Check the camera — it looks tilted. Keep it straight for
  better feedback."

- **Gross tripwire, uncertified installations.** If `|camera_pitch_deg − 4°| > 8°` (the existing
  `MAX_CAMERA_PITCH_DEG`): `level_camera`, and `rdl_hip_sink` stamped `observability="approximate"` for
  the session. The rule pre-sets it, and `tag_observability` uses `setdefault`.
- **Implementation of the per-session start tier** ([SP §4] `cue_start_tier`).
  - `RomanianDeadliftProfile.cue_start_tier(fault_type, ctx)` returns, for `rdl_hip_sink`:
    - **mild** while `ctx.pitch_certified` and the lens checks (requirements 2–3) pass;
    - **moderate** when the lens checks pass but pitch is not certified, including right after the
      delta tripwire revokes certification mid-session;
    - **severe** when the lens checks fail or `mount_height_m` is unknown, whatever the pitch.
  - For `rdl_bar_drift` / `rdl_lockout` it returns `rig_min_cue_severity`. Everything else falls back
    to the static `min_cue_severity` lookup (mild).
  - The pipeline calls it as each verdict is formed and is the **only** place that stamps
    `recorded_only`. The `rdl_hip_sink` rule emits plain verdicts.

### 3.3 Rep signal

`RomanianDeadliftProfile.get_rep_signal` returns `(shoulder_mid.y − ankle_mid.y) · 100`, which is
−shoulder height in cm.
- It rises during the descent, as the counter expects: ≈ −147 standing, ≈ −105 at the bottom for
  188.5 cm.
- It returns NaN when any of the 4 shoulder/ankle points has confidence ≤ 0 (squat pattern).

Why shoulders: an image-plane travel of 28–47 cm (§4.4), on the most stable keypoints. It replaces the
scaffold's `−trunk_flexion`, a 3D angle that the squat audit A12 found camera-placement dominated on one
camera (6.6–56.6° per-run medians). Alternatives rejected:
- wrist height: noisier, and the bar and plates are near it;
- hip flexion: depth-based.

The signal drives only the counter's coarse state machine. Every bias-sensitive decision uses the
features in §4.

### 3.4 Phases

`SignalRepCounter` states (`idle → descending → bottom → ascending`) are unchanged. Feature windows key
off the lowest shoulder sample (§4.2).

### 3.5 Counter config

New `rdl.counter: HipPositionCounterConfig` in YAML. The duration fields are confirmed on main.

| Field | Value | Rationale |
|---|---|---|
| `entry_vel_threshold` | 4.0 cm/s | Above standing sway (~1–3 cm/s); below the start of a 3-s eccentric (mean 13 cm/s) |
| `bottom_vel_threshold` | 5.0 cm/s | With `BOTTOM_ARM_VELOCITY_RATIO` 2, the descent must peak ≥ 10 cm/s; a 4-s eccentric peaks > 15 |
| `ascending_vel_threshold` | 4.0 cm/s | Symmetric with entry |
| `min_depth_cm` | 8.0 | Excursion floor only, so every deliberate dip gets features. The count gate is §3.6. Walking bob (±2–3 cm) never reaches it |
| `standing_return_cm` | 4.0 | A finish 20° short of vertical drops the shoulders 3.3 cm; further off ends through the 25% lockout zone |
| `min_descending_s` | 0.25 | 3 frames at 12 fps |
| `min_bottom_s` | 0.08 | 1 frame at 12 fps; RDLs turn around without a pause |
| `min_ascending_s` | 0.25 | 3 frames at 12 fps |
| `min_rep_duration_s` | 0.8 | Rejects fidgets. The fastest real RDL still counts and gets `rdl_tempo` |

### 3.6 ROM acceptance: when a rep counts

`RdlRepAnalyzer.rejection_reason(features)` (SHARED-PLUMBING §2) runs on every completed excursion. It
returns None for a counted rep, `"partial"`, or `"ignore"`.
- On a reason, the pipeline runs the spec's rejection sequence: `reject_last_rep`, `discard_rep` on the
  engine and the analyser, `rep_data = None`, then `profile.on_rep_rejected(reason, features, angles)`.
- A rejected excursion therefore never reaches `SessionReference.update`.

Counting uses an absolute floor on **hand travel**, because hand travel is robust to the bias budget.
Coaching uses the athlete's own range.

- **Hand travel** `bar_travel_ratio = (wrist_ref − median_B wrist_mid) / (wrist_ref − knee_ref)`.
  - 0 means the hands are at hang height; 1 means the hands are at knee height.
  - It is pitch- and perspective-robust: the hands move only ~5–8 cm horizontally, so the bias is
    ≤ ±0.05 (§4.3).
  - Arm length varies the hinge at a given value by ±0.045, about ±3°.
- **rep**: `bar_travel_ratio ≥ 0.45`, which is ≈ 45° of hinge for Winter proportions (§4.4).
  - Any healthy adult reaches this with soft knees.
  - A grip or re-rack dip to mid-thigh hooks reads 0.14 (≤ 0.19 with bias).
  - If the wrists are unseen in the bottom window, fall back to `hinge_deg ≥ 50°`. The 5° margin covers
    hinge bias.
- **partial**: `0.25 ≤ bar_travel_ratio < 0.45` and `duration_s ≥ 1.0`. Not counted.
  - `on_rep_rejected("partial", …)` returns **one** fault for **every** partial, with
    `details["shallow_rep"]=True` and `counted=False`. So every partial produces a `shallow_rep` message,
    and the recap count is complete.
  - The fault is `rdl_partial`, or `rdl_hip_sink` when `hip_sink_ratio ≥ features.hip_sink_bands[1]`
    (moderate).
  - `on_rep_rejected` receives no config, so the analyser stamps `hip_sink_bands`,
    `count_floor_travel` and `partial_bands` into the features. Faults go through the public
    `tag_observability` (spec §2).
  - **Cue.** `shallow_rep_cue(fault, streak, cued_this_set)`. The pipeline owns the streak (spec §2): it
    counts consecutive "partial" rejections in the set; a counted rep resets it; "ignore" leaves it
    unchanged; `reset_set` clears it and `cued_this_set`.
    - The RDL returns the emitted fault's mapped cue (`rdl_lower` for `rdl_partial`, `rdl_hips_back` for
      `rdl_hip_sink`) only when `streak ≥ 2` and `cued_this_set < 2`.
    - Otherwise it returns None. Cue-less messages are counted by `on_shallow_rep` and play nothing
      (`coaching_orchestrator.py:971-987`).
    - The analyser keeps no streak state.

    A lone unrack or re-rack dip can therefore never trigger a cue.
- **ignore**: anything else (NaN, travel < 0.25, or short). Unrack, re-rack, re-grip, peeking.
  `on_rep_rejected("ignore", …)` returns `[]`: not counted, nothing said.
- **Coaching reference** (`rdl_range_drift`):
  - `HingeSessionReference.hinge_reference()` = median of the session's 3 deepest counted `hinge_deg`,
    capped at 95°.
  - Active once 3 counted reps exist.
  - `hinge_deg` biases are constant within a session, so they cancel here.
- **`depth_target_met`.** The pipeline does not set it on the analyser path (spec §1), so `RepData`
  keeps its default `True` (`utils/types.py:488`). `session_tracker._rep_highlights` therefore tags
  every counted RDL rep `"depth_target_met"`.
  - That is harmless: nothing in `src/agent`, `src/db` or `main.py` reads the field or that highlight
    (verified by grep).
  - The RDL's "reached own range" notion lives only in `highlight_score` (§4.2).
- **Considered for V1.1:** a guided "find your range" rep that seeds the reference.

### 3.7 Set boundaries and edge cases

- **Set start.** Rest end resets the counter (`pipeline_process.py`, rest expiry). The readiness gate
  pre-arms in the last seconds of rest.
- **Unrack / step-back.** The grip dip is "ignore" (travel 0.14). Standing up and stepping back bob
  ±2–3 cm, below `min_depth_cm`.
- **Re-rack.** Same dip, so "ignore". The set ends by target reps, the orchestrator idle timeout (15 s),
  voice, or the pipeline set timeout (30 s).
- **Hooks set low** (knee height): the grip dip reads ~0.75 and counts as rep 1.
  - Mitigation: the setup clip, and G1 (0 false counts in ≥ 40 sequences).
  - Contingency if G1 fails: a step detector in the RDL analyser. A foot lift is
    `|ankle_l.y − ankle_r.y| > 4 cm` for ≥ 2 frames. An excursion starting < 1.0 s after a step is
    "ignore".
- **Quick first rep.** The counter's baseline updates only while still. Walking frames sit at the same
  shoulder height.
- **Piston (touch-and-go, constant-tension) reps.** The counter ends the rep in its lockout zone and
  starts the next one on the same frame.
  - The spec's call order applies: the ended rep's `finish_rep` + verdicts first, then
    `on_rep_start(piston=True)`.
  - The new rep's setup and pre-roll come from the turnaround frames, and its `rdl_lockout` is NaN
    (§4.1).
  - References come from `_setups`, which turnarounds never enter, so a bent turnaround cannot bias them.
- **Yaw.** Vertical features are invariant to it. Tilt is skipped when the wrist separation falls below
  0.6 × its start value.
- **Failed or held rep.** `MAX_REP_DURATION_S` (12 s) abort.
- **Exercise switch.** `set_exercise` rebuilds the rules, the analyser and the reference. The squat depth
  target carries over (existing behaviour).
- **BiLSTM.** `uses_bilstm_counter=False`. A test with `bilstm.enabled=True` asserts
  `pipeline._bilstm is None`.
- **Names.** "Barbell Romanian Deadlift", "Romanian Deadlift" and "RDL" resolve here (verified).
  - Variants go to `UntrackedVariantProfile` with the SHARED-PLUMBING §7 RDL list:
    - single-leg, snatch-grip, dumbbell, kettlebell, deficit and B-stance, in both `_rdl` and
      `_romanian_deadlift` forms;
    - `stiff_leg_deadlift`.
  - Longest whole-word match then wins: "Dumbbell Single Leg Romanian Deadlift" →
    `single_leg_romanian_deadlift` → untracked.
  - Update `test_profile_registry.py:28` ("Single-Leg RDL" → untracked).

---

## 4. Per-rep features

### 4.1 Architecture: the RDL analyser

The RDL implements `create_rep_analyzer()`, and `RdlRepAnalyzer` follows the `RepAnalyzer` protocol of
SHARED-PLUMBING §2. The pipeline then skips its squat-shaped blocks (spec §1), so the squat
`_standing_reference_hip_cm` is never raised by RDL frames. `RdlRepAnalyzer` lives in
`analysis/hinge_features.py`. It owns its frame samples, setup history and pre-roll. The rejection streak is pipeline-owned (§3.6).

**RDL behaviour per protocol method.**
- `observe(ctx)` is called on processed, non-extrapolated frames after the latch, idle and in-rep. It
  appends a `HingeFrameSample` and maintains the pre-roll and setup tracking.
- `on_rep_start(piston)` opens the rep buffer.
  - **`piston=False`:** close the idle setup and push it onto `_setups`; freeze the last 0.5 s of idle
    samples as the pre-roll; `S_top` = median of S over the pre-roll.
  - **`piston=True`** (spec call order: the previous rep's `finish_rep` and verdicts already ran):
    - The **setup is the turnaround frames**, meaning the previous rep's samples after its `i_low` whose
      shoulder height is within 1 cm of their maximum. They are kept only for this rep and never pushed
      onto `_setups`.
    - The **pre-roll** is the previous rep's samples after its `i_low`, which `finish_rep` retains for
      exactly this case.
    - `S_top` = their **smoothed maximum** (max of a 3-sample running median).
    - The rep is flagged `started_at_turnaround=True`, so its `lockout_pitch_deg` is NaN and
      `rdl_lockout` is not judged (spec §2: no setup-dependent faults).
    - The references (`hip_ref` …) still come from `_setups`, i.e. real standing setups.
- `finish_rep(rep_number) -> HingeRepFeatures` returns the §4.2 model.
  - It carries `exercise="romanian_deadlift"`, `rep_number` and `max_depth_angle = hinge_deg`
    (required by the spec). The depth fields are nulled downstream because `reports_depth=False`.
  - It also carries the stamped thresholds `on_rep_rejected` needs (§3.6).
  - It retains the samples after `i_low` until the next `on_rep_start` or idle frame.
- `rejection_reason(features)` implements §3.6: None, `"partial"` or `"ignore"`. It is pure: the
  pipeline owns the streak.
- `discard_rep()` drops the rep buffer; setups survive.
- `reset_set()` clears the pre-roll, the rep buffer, the retained post-`i_low` samples and the idle-peak
  tracking. **`_setups` survives**; only a fresh analyser (`set_exercise`) starts it empty.

**`FrameContext` fields used** (spec §3, no additions): `skeleton_3d`, `angles`, `timestamp`, `in_rep`,
`capture_mode`, `camera_pitch_deg`, `raw_confidences`, `up` and `gravity_source`.
- `raw_confidences` are the measured capture's per-keypoint confidences: MediaPipe visibility on one
  camera, triangulation metric confidence on the rig.
- **Vertical axis `up`.**
  - The analyser uses `ctx.up` directly and never negates it.
  - Exception: on one camera V1 does not trust `gravity_source == "camera_pitch"`, because it is
    unvalidated (spec §3; it includes posture). In that case it uses `WORLD_UP`; the V1.1 correction uses
    the measured mount angle instead (§4.3).
  - With `gravity_source == "body"` on the rig, `rdl_bar_drift` and `rdl_lockout` start at moderate
    (§5.2–5.3).
- Shared test (spec §3): with both `"body"` and a tilted `"measured"` `up`, an upright trunk reads
  `trunk_pitch_signed = 0 ± 0.5°`, and a bar held in front of the feet reads a positive
  `bar_offset_ratio`.

**Per-frame `HingeFrameSample`** (analyser-internal; squat `RepTrajectorySample` untouched). Heights are
`h(·)` cm.
- shoulder, hip, knee (mean) and wrist L/R heights;
- `wrist_lateral_sep_cm = |horizontal(R_WRIST − L_WRIST) · lat| · 100`, with
  `lat = unit(horizontal(R_HIP − L_HIP))`;
- `bar_tilt_signed_deg = degrees(atan2(wrist_h_l − wrist_h_r, wrist_lateral_sep_cm))`. Positive means the
  right hand is lower;
- `bar_offset_ratio = horizontal(wrist_mid − midfoot) · forward / foot_len`. `forward`, `midfoot` and
  `foot_len` come from `rep_features._foot_frame`. NaN when there is no foot frame or `|·| > 1.5`;
- `trunk_pitch_signed = degrees(atan2(trunk·forward, trunk·up))`, with `up` as above,
  `trunk = shoulder_mid − hip_mid`, and `forward` ⟂ `up` (foot frame built on `horizontal()`). Positive
  means forward. Use the deadlift frame helper if it has landed;
- `hip_conf = min(raw_confidences[L_HIP], raw_confidences[R_HIP])`;
- `knee_flexion_mean`, `trunk_pitch` (180 − `trunk_flexion`), for validation only.

**Setups and references.**
- **Setup.** Idle frames since the last rep whose hip height is within 1 cm of that interval's maximum.
  Each setup stores the medians of: hip, knee, wrist-mid and shoulder heights; torso vertical
  (shoulder − hip height); `trunk_pitch_signed`.
- **Lifecycle of `_setups: deque(maxlen=5)`:**
  - created empty with the analyser, i.e. on every `set_exercise`;
  - kept across sets within the exercise, because stance barely changes and the 5-deep median adapts if
    it does;
  - **not** cleared by `reset_set()`.
- **References for each rep.** `hip_ref`, `knee_ref`, `wrist_ref`, `torso_ref` = medians over `_setups`,
  including this rep's setup.
  - A setup 15° short of vertical biases `hinge_deg` by ≤ 2.1° at 45° and ≤ 0.5° at 75°.
- **Pre-roll.** The last 0.5 s of idle samples (at least 6 frames at 12 fps), or, for a piston rep, the
  previous rep's post-`i_low` samples. It is prepended at `on_rep_start` and serves tempo (§4.2).

### 4.2 Windows and features (`HingeRepFeatures`)

**Windows.** `S_i` is the shoulder height.
- `i_low = argmin S`.
- `S_top` = median of S over the idle pre-roll, or the smoothed maximum of the turnaround pre-roll for a
  piston rep (§4.1).
- `A = S_top − S_low` (the rep's shoulder drop).
- **Bottom window** `B = {i : S_i ≤ S_low + 3.0 cm}`: ≈ 3° of hinge at depth, ~6–8 frames at 12 fps for
  a 2-s eccentric.
- **Lower half** `L = {i : S_top − S_i ≥ 0.5 A}`.
- **`B'` (hips seen)** depends on capture mode:
  - one camera: `{i ∈ B : hip_conf ≥ 0.5}` (raw MediaPipe visibility);
  - rig: every `i ∈ B`. On the rig, any analysed frame already has hips triangulated from ≥ 2 views (raw
    confidence ≥ `MIN_KEYPOINT_CONFIDENCE` 0.1; frames with carried legs are excluded). That sits below
    `TWO_VIEW_CONFIDENCE_CAP` 0.3 (`triangulator.py:40`) and the 2-camera gate 0.25 (`pipeline.py:68`),
    so two-view hips are never dropped.
- If `|B'| < 2`, the rep is `hinge_hip_source="setup"`: counting only, no hip-sink or range verdict.

| Feature | Formula | Normalization | Window | One camera / rig | 12 fps noise handling |
|---|---|---|---|---|---|
| `hip_sink_ratio` | `(hip_ref − median_{B'} H_i) / hip_ref` | standing hip height ≈ leg length | B′ | observable / observable (bias §4.3) | Median over B′ (≥ 2 frames, typically 4–8): ~0.5–1% |
| `bar_travel_ratio` | `(wrist_ref − median_B wrist_mid) / (wrist_ref − knee_ref)` | hang-to-knee span | B | observable / observable | Median; bias ≤ ±0.05 |
| `hinge_deg` | `median_{B'} degrees(arccos(clip((S_i − H_i)/torso_ref, −1, 1)))`; uses `H_i := hip_ref` when `"setup"` | `torso_ref` | B′ | observable / observable. **Relative use only** (range drift, highlight) plus the 50° fallback floor | Median; 1.1°/cm at 75°, 1.5°/cm at 45° |
| `bar_height_ratio`, `range_label` | `median_B wrist_mid / knee_ref`. Bins: > 1.35 "above the knee"; 1.05–1.35 "at the knee"; 0.8–1.05 "just below the knee"; 0.55–0.8 "mid-shin"; ≤ 0.55 "low on the shin" (bins absorb the bar ≈ 0.2·knee below the wrist) | standing knee height | B | observable / observable | Recap descriptor only |
| `bar_tilt_deg`, `bar_tilt_side` | `Δ = median_B(tilt_signed) − median(first 3 rep samples' tilt_signed)`; value `|Δ|`; side `"right"` if Δ > 0, else `"left"` (the side that dropped). NaN if the separation in B is < 0.6 × the start separation or < 20 cm | Degrees; the start subtraction cancels camera roll and arm/grip asymmetry | B | observable / observable | Medians; σ ≈ 2° |
| `bar_offset_ratio_p80`, peak phase | p80 of `bar_offset_ratio` over L | ankle→toe length | L | not observable / observable after G6 | p80; ≥ 4 frames at 30 fps |
| `lockout_pitch_deg` | this rep's idle setup `trunk_pitch_signed` (median over the setup frames = the finish of the previous rep). **NaN when the rep started at a turnaround** (piston) | Degrees, signed | setup | not observable / observable after G6 | Median of setup frames |
| `descent_time_s` | **5%→95% shoulder-drop time** on `S̃` (3-sample running median of S). Let `i_top = argmax S̃` over the pre-roll (the standing frames, or for a piston rep the previous ascent up to its turnaround). Both crossings are **downward** crossings searched only in `(i_top, i_low]`: `t_start` = first `i` with `S̃[i−1] ≥ S_top − 0.05A > S̃[i]`, `t_end` = first later `i` with `S̃[i−1] > S_top − 0.95A ≥ S̃[i]`, each linearly interpolated. `descent = t_end − t_start`. On a piston rep the previous ascent's upward crossings lie before `i_top` and are never read | — | `(i_top, i_low]` | observable / observable | Noise: ~0.6 cm (after the median) ÷ crossing speed, per crossing, two crossings. For the threshold-relevant fast descents (full ≤ 1 s, crossing speed ≥ 27 cm/s) ≈ ±0.03 s (1σ). For a 2-s descent (≈ 14 cm/s) ≈ ±0.06 s, far from any threshold. The raw-sample (no median) figure is ±0.04–0.07 s per crossing |
| `ascent_time_s`, `duration_s` | `t_last − t(i_low)`; `t_last − t(5% crossing)` | — | — | — | — |
| `concentric_velocity_mps` | shoulder rise, same method as `rep_features._concentric_velocity_mps` (made public) | m/s | ascent | recorded | — |
| `knee_flexion_bottom_deg`, `trunk_pitch_bottom_3d` | medians over B | — | B | approximate / observable | Validation and recap only |
| `highlight_score`, `highlight_meaning` | `−hip_sink_ratio` if `hinge_deg ≥ reference − 8°` (or no reference yet), else NaN; meaning = `"their most hips-back clean rep"` | — | — | — | Higher is better (spec §4). Best rep = most hips-back, never deepest |
| `max_depth_angle` | `hinge_deg` (field required by the spec) | — | — | — | Not used downstream (`reports_depth=False` nulls the depth fields). The recap reads `hinge_deg` from features (§6) |
| `exercise`, `started_at_turnaround`, `gravity_source` | `"romanian_deadlift"`; piston flag; copied from ctx | — | — | — | Spec-required / used by rules |
| `hip_sink_bands`, `count_floor_travel`, `partial_bands` | Stamped from `config.faults.rdl` | — | — | — | Lets `on_rep_rejected` build faults without config |

**Drift metrics** (`HingeSessionReference.update`; all lower is better): `rdl_hip_sink`,
`rdl_bar_offset`, `rdl_tilt`, `rdl_lockout_abs`, and `rdl_descent_deficit = max(0, mild − descent)`.

### 4.3 Bias budget

Effects of camera pitch (5° and 8°), perspective, and the rig's world vertical (up to 4° off gravity).
Pitch leak is `horizontal travel · sin(pitch)`. Perspective error is `|y − h_lens| · δ / (D − δ)`, where δ is the
travel toward the lens and D = 3.6 m. The lens is at hip height unless stated.

| Feature | Horizontal travel | Pitch 5° / 8° | Perspective (hip lens / chest lens) | Rig vertical 4° | Use |
|---|---|---|---|---|---|
| `hip_sink_ratio` | hips go 16–25 cm back (vertical shin, k 20–33°) | ≤ 2.4% / ≤ 3.8% | 0.3% / 2.1% | ≤ 1.9% | absolute; covered by the §5.1 start tier |
| `bar_travel_ratio` | hands ±5–8 cm | ≤ 0.02 / ≤ 0.03 | 0.03 / 0.05 | ≤ 0.02 | absolute count floor (±0.05 at 5°, ±0.06 at 8°) |
| `hinge_deg` | shoulders 25–46 cm toward the lens | ≈ 1° per ° of pitch | ≈ 1.5° / ≈ 4° | ≈ 4° | relative only; constant offsets cancel |
| `bar_tilt_deg` | — | 0 (vertical difference of two hands at equal depth) | ~0 | 0 | start-relative |
| `descent_time_s` | — | 0 (amplitude-normalized timing) | 0 | 0 | absolute |

**Pitch actually assured.**
- **Certified mount** (§3.2: measured pitch ≤ 2°, lens 0.8–1.1 m and ≤ 10 cm below this lifter's hips,
  no delta trip), **leveled** with `geometry.level_camera_frame` by `mount_pitch_deg`.
  - The residual is the inclinometer error (±1°), giving a hip-sink leak ≤ 0.5%. Perspective is ≤ 0.7%.
  - The leveling depth terms add ≤ 0.4% bias plus ≈ 0.05% noise (σ_z = 3 cm, |b_z| ≤ 10 cm at sin 2°).
  - Knock tolerance is 3.1° (§3.2).
- **Uncertified:** only the setup instruction plus the tripwire, so the budget is **8°**, giving a leak
  ≤ 3.8%.

**Rotate by what?**
- On certified mounts, rotate by the **measured mount angle** [SP §3] (V1). Never by
  `camera_pitch_deg`, which is depth-regressed, includes posture, and has unknown error.
- **Mild stays 0.12.** For 0.10: 10 − (8.0 + 0.7 + 1.0 + 0.4 depth bias) leaves **−0.1%**, a negative
  budget before any pitch residual. The budget with depth terms does not support it.
- Using `ctx.up` with `gravity_source == "camera_pitch"` stays off unless a future validation shows the
  estimator within 1.5°.
- On the rig, `ctx.up` is used directly (measured when available, `WORLD_UP` with
  `gravity_source="body"` otherwise; SHARED-PLUMBING §3).

### 4.4 Geometry behind the thresholds (`scratchpad/rdl_geom*.py`)

Winter (2009) proportions, which match the squat audit's 188.5 cm model: tibia 0.246H, femur 0.245H,
shoulder–hip 0.288H, arm 0.332H. Ratios are scale-free.

**Hip sink (%) vs knee flexion k, across shin inclination s.** Standing reference at 3° knee flexion;
the scorer's 0° reference reads ~0.1 higher.

| k | 15° | 20° | 25° | 30° | 33° | 35° | 40° | 45° | 50° | 55° | 60° |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s = 0 (vertical shin) | 1.6 | 2.9 | 4.6 | 6.6 | **8.0** | 9.0 | 11.6 | 14.6 | 17.8 | 21.2 | 24.9 |
| s = k/4 | 1.0 | 1.8 | 2.9 | 4.2 | 5.0 | 5.7 | 7.4 | 9.3 | 11.4 | 13.8 | 16.3 |
| s = k/2 | 0.8 | 1.5 | 2.3 | 3.3 | 4.1 | 4.6 | 6.0 | 7.5 | 9.3 | 11.2 | 13.3 |

- Hip height = T·cos s + F·cos(k − s).
- The metric therefore measures **hip sink, i.e. combined shin + thigh inclination**. At fixed k it is
  largest with a vertical shin, and *falls* as the knees travel forward.

**Hand travel vs hinge** (k 20°, shin 5°): 25° → 0.15, 35° → 0.28, 45° → 0.44, 50° → 0.53, 60° → 0.74,
70° → 0.96, 80° → 1.21.
- A grip dip to mid-thigh hooks reads 0.14.
- Arm length ±0.02H moves the 45° value by ±0.045.

**Shoulder drop.** k20 + lean 70° → 37 cm; lean 80° → 47 cm; k15 + lean 60° → 28 cm.

### 4.5 Alternatives considered

- **Hip-sink metric.**
  - 3D knee flexion: depth regression on one camera. Recorded; the rig V1.1 candidate.
  - Hinge ratio (hip drop ÷ shoulder drop): conflates range with knee bend (0.10 at 45° vs 0.04 at 80°).
  - Knee-height drop: a ~1.5 cm signal, occluded by the hands.
- **Count gate on `hinge_deg`.** Pitch and perspective bias it by up to ~9° on one camera (a 40° grip dip
  would read 45° at +5° pitch). Replaced by hand travel.
- **ROM coaching on bar height.** Rewards squatting; it stays a descriptor.
- **Fixed-angle ROM target.** Forces tight hamstrings into rounding; replaced by a per-athlete reference.
- **Asymmetry options:**
  - hip-flexion `SymmetryRule`: monocular far-leg noise;
  - lateral hip shift: approximate on one camera;
  - YOLO bar tilt: off by default; usable when enabled;
  - wrist tilt: chosen.
- **Generalising the squat `SessionReference`** (Round 1): replaced by `create_session_reference()`, so
  squat code is not touched.

---

## 5. Fault catalog

Each fault is a `RepFaultRule` in `faults/rules/rdl_*.py` and emits through `_rep_fault`, so it carries
`side`, `phase`, `is_drift`, `value` and `unit`. Observability comes from `FAULT_MEASUREMENT`; a rule may
pre-set it to downgrade. Approximate faults are recorded and hedged, never cued. Thresholds are expert
estimates until G2–G6. Each rationale is stated.

### 5.1 `rdl_hip_sink` — squatting the RDL (P40)

**Definition.** The hips sink instead of travelling back.
- On one camera the metric is **hip sink**, i.e. combined thigh + shin inclination. It is **not** knee
  travel (§4.4).
- Not judged when `hinge_hip_source == "setup"`.

**Metric.** `hip_sink_ratio` over B′, once per rep.

**Band derivation.**
- The acceptable envelope is the NSCA ~15° knee bend widened to 33° for tight-hamstring lifters, at any
  shin angle from 0 to k/2. Its worst case is a **vertical shin at 33°: 8.0%** (8.1% vs a straight-leg
  reference).
- Bands: **mild ≥ 0.12, moderate ≥ 0.15, severe ≥ 0.19.**
- The **start tier** (`min_cue_severity`, SHARED-PLUMBING §4) depends on whether the mount is certified
  and on the lens height (§3.2, §4.3). Worst accepted reading = envelope (8.0%) + pitch leak + perspective
  + window noise (≈ 1%) [+ leveling depth bias when leveled]. Perspective is ≤ 0.7% inside the lens checks
  (0.8–1.1 m, ≤ 10 cm below the hips) and up to ~2.3% for a level lens at 0.5 m:

  | Installation | Worst correct-rep reading | Start tier | Effect |
  |---|---|---|---|
  | **Certified mount, leveled** (§3.2: measured pitch ≤ 2° re-measured each session, lens 0.8–1.1 m and ≤ 10 cm below the lifter's hips, no delta trip; `level_camera_frame` applied): **the demo and product setup** | 8.0 + 0.5 (pitch residual) + 0.7 (perspective) + 1.0 (noise) + 0.4 (leveling depth bias) ≈ **10.6%** | **mild** | A correct rep never reaches mild (12%). The common novice "knees forward" error (45–55°, shins 15–20°) is cued at its pronounced end, about 53–55°, where it reads ≥ 12% |
  | **Uncertified pitch, lens checks pass** (`mount_height_m` entered, 0.8–1.1 m, ≤ 10 cm below the hips) | 8.0 + 3.8 (8° leak) + 0.7 + 1.0 ≈ **13.5%** | **moderate** | Correct reps can't reach moderate (15%). A 12–15% reading is `recorded_only` |
  | **Lens checks fail or height unknown** (any pitch) | 8.0 + 3.8 + 2.3 (lens at 0.5 m) + 1.0 ≈ **15.1%** | **severe** | 15.1% would false-cue at moderate, so this starts at severe (19%) per spec §3. Mild and moderate readings are `recorded_only` |

- **Recorded-only verdicts** (sub-start-tier verdicts on uncertified or out-of-band installations)
  carry no cue. Per spec §4 they are filtered on every
  delivery path, excluded from `is_clean`, cue outcomes and recap counts.
  - They may appear in the recap hedged, never with a size, and only if `rdl_hip_sink` is listed in
    `faults.rdl.hedged_recap_faults`.
  - That set is empty by default and filled from the validation report.
- **Praise at a moderate start tier** uses the [SP §4] rule. A rep that drops from moderate+ into the
  recorded-only band (12–15%) is "better", not "fixed", because 12–15% is still above a correct rep's
  13.5% uncertified worst case.
  - Praise is decided over the existing hold window (`FIX_HOLD_REPS` = 2; timing unchanged).
  - If **any** hold rep carries a recorded-only `rdl_hip_sink`, play `rdl_hips_back_better` ("Better,
    hips went further back.").
  - `rdl_hips_back_fixed` ("That's it, hips went back.") plays only when **no** hold rep carries any
    hip-sink verdict.
  - Either praise resolves the outcome: no later `_fixed` after `_better`. A new cue starts a new outcome.
  - A missing `_better` clip is **skipped**, never replaced by `_fixed`.
  - The rig faults (moderate start) get `_better` clips under the same rule (§5.2–5.3).

**What the bands catch** (no bias):

| Band | Vertical shin | s = k/4 | s = k/2 |
|---|---|---|---|
| mild | k ≈ 41° | k ≈ 51° | k ≈ 57° |
| moderate | k ≈ 46° | k ≈ 58° | k ≈ 64° |
| severe | k ≈ 52° | k ≈ 65° | k ≈ 72° |

Mild covers the common novice "knees forward" error at its more pronounced end, and moderate and severe
cover the squat pattern. With a **certified mount** (the demo and product setup) cueing starts at the
mild row. On an uncertified installation it starts at moderate if the lens checks pass, and at severe
otherwise.
- Subtler forward knee travel is missed on one camera, by design (R2); the rig V1.1 knee-flexion/shin
  metric targets it.
- Mild stays 0.12 even with leveling. With the depth terms, 0.10 has a negative budget (§4.3).

**Observability.** `vertical`: observable on both modes.
- Downgraded to approximate by the pitch guard (§3.2).
- If G2 fails, it becomes approximate on one camera.

**Other fields.**
- Drift: session best `rdl_hip_sink` < mild.
- Side `"both"`. Phase `"bottom"`. Extras: `hinge_deg`, `knee_flexion_bottom_deg`.

**Delivery.**
- **Cue** `rdl_hips_back`: "Hips back to the wall" (5 words). Drafting alternates: "Hips back, not
  down" · "Push your hips behind you".
- **Fixed praise:** "That's it, hips went back." (`_fixed`, hold reps below mild). **Better praise:**
  "Better, hips went further back." (`_better`, next rep recorded-only; [SP §4], §5.1).
- **Pre-set line:** "This set: hips back, not down."
- **Label:** "Hips back".
- **`FAULT_LABELS`:** "letting the hips sink (squatting it) instead of pushing them back".

### 5.2 `rdl_bar_drift` — bar off the legs, load toward the toes (P41, rig)

**Metric.** `bar_offset_ratio_p80` over L.

**Thresholds** (ankle→toe lengths ahead of midfoot): mild ≥ 0.25, moderate ≥ 0.40, severe ≥ 0.60.
- A bar touching a near-vertical mid-shin sits ≈ −0.15 (midfoot = ankle + 0.35·ankle→toe).
- mild ≈ 8–9 cm off the shin; severe = over the toes.
- G6 measures the wrist-to-bar horizontal offset and subtracts it.

**Observability.** New class `rig_unvalidated` = {single_camera: not observable, triangulated:
approximate}. It switches to `side_view` after G6. With the deadlift's 3D bar tracker (`needs_bar_3d`),
V1.1 can swap wrists for the bar.

**Start tier.** `min_cue_severity["rdl_bar_drift"] = MODERATE` in V1, as spec §3 requires for
`gravity_source = "body"` rigs; no rig has measured gravity yet. The config switch
`faults.rdl.rig_min_cue_severity: mild` lowers it once measured `up` is installed and G6 passes. Mild
verdicts until then are `recorded_only`.

**Other fields.**
- Drift: `rdl_bar_offset`.
- Phase of the peak.
- Joins the RDL safety set at severe once observable.

**Delivery.**
- **Cue** `rdl_bar_close`: "Drag the bar down your legs" (6).
- **Fixed praise:** "Good, bar stayed on your legs." **Better praise** (while starting at moderate):
  "Better, bar stayed closer."
- **Pre-set line:** "This set: bar stays on your legs."
- **Label:** "Bar close". **`FAULT_LABELS`:** "the bar drifting away from the legs".

### 5.3 `rdl_lockout` — finish not tall (P42, rig)

**Metric.** `lockout_pitch_deg`; direction `"forward"` or `"back"`.

**Thresholds.**
- Forward (hips not through): mild ≥ 10°, moderate ≥ 15°, severe ≥ 22°.
- Back (leaning back): mild ≥ 8°, moderate ≥ 12°, severe ≥ 18°.
- Rationale: NSCA has the torso vertical at the end. Keypoint pitch varies ±3–4° upright. A loaded bar
  in front adds ~3–5° of natural lean-back.
- Lean-back is the honest whole-trunk proxy for lumbar extension; Nova says "leaning back", never
  "arching". These are the deadlift plan's D5 bands (8/12/18°).

**Observability.** `rig_unvalidated`, then `side_view` after G6.

**Start tier.** `min_cue_severity["rdl_lockout"] = MODERATE` in V1, with the same rule and switch as
§5.2.

**Piston reps.** NaN, so not judged, when the rep started at a turnaround (§4.1). A constant-tension set
is never told "hips not through".

**Rep tagging.** The verdict describes the **previous rep's finish** but carries this rep's number.
- `rdl_lockout` is therefore **added to `END_OF_REP_FAULT_TYPES`** (`db/biomechanics_persistence.py:38`),
  so next-rep cue-outcome evaluation and rep linking skip it. Test it.
- `LAGGED_FAULT_TYPES = {"rdl_lockout"}` shifts the evaluation window by one rep everywhere the
  orchestrator judges a cue's effect: the first rep whose verdict can show the fix is the one after next.
  - fix praise (`_resolve_cue_outcome`): `after_rep + 1`;
  - **`_cue_outcome`'s `reps_after`** (`coaching_orchestrator.py:1497`): reps with
    `rep_number > after_rep + 1`.

**Other fields.** Drift: `rdl_lockout_abs`. Phase `"top"`.

**Delivery.**
- **Cue** `rdl_finish_tall`: "Finish tall, hips under shoulders" (5). It corrects both directions.
- **Fixed praise:** "Good, clean finish up top." **Better praise** (while starting at moderate):
  "Better, finishing taller."
- **Pre-set line:** "This set: finish tall at the top."
- **Label:** "Finish tall". **`FAULT_LABELS`:** "not finishing straight and tall at the top".

### 5.4 `rdl_bar_tilt` — one side drops or rotates (P43)

**Metric.** `bar_tilt_deg`; side = the side that dropped.

**Thresholds.** mild ≥ 4°, moderate ≥ 7°, severe ≥ 10°.
- Squat detector bands are 2/4/7° (`BarbellTrackingConfig.tilt_asym_*`), widened for the wrist proxy
  (σ ≈ 2°).
- 4° over a ~50 cm grip = 3.5 cm of hand-height difference.

**Observability.** `frontal`: observable on both modes.

**Other fields.** Drift: `rdl_tilt`. Phase `"bottom"`. No side clips in V1; the side goes to the recap.

**Delivery.**
- **Cue** `rdl_level_bar`: "Keep the bar level" (4).
- **Fixed praise:** "Good, bar stayed level."
- **Pre-set line:** "This set: keep the bar level."
- **Label:** "Level bar". **`FAULT_LABELS`:** "the bar tilting to one side at the bottom".

### 5.5 `rdl_range_drift` — shorter than own range (P44)

**Metric.** `hinge_reference − hinge_deg`. Requires ≥ 3 counted reps and `hinge_hip_source ==
"measured"`.

**Thresholds.** mild ≥ 8°, moderate ≥ 15°, severe ≥ 25°.
- Expert estimate of trained rep-to-rep variation: ~3–4° SD. 8° ≈ 2 SD ≈ 7 cm of bar height at depth.
- 25° ≈ below the knee → mid-thigh.
- Within-session biases cancel.

**Observability.** `vertical`: observable on both modes.

**Other fields.** Always `is_drift=True`. Phase `"bottom"`. Extras: `reference_deg`, `range_label`.

**Delivery.**
- **Cue** `rdl_same_depth`: "Same depth every rep" (4). It never says "deeper".
- **Fixed praise:** "Right back to your depth."
- **Pre-set line:** "This set: same depth every rep."
- **Label:** "Same depth". **`FAULT_LABELS`:** "reps getting shorter than their own range".

### 5.6 `rdl_tempo` — lowering too fast (P45)

**Metric.** `descent_time_s`, the 5–95% shoulder-drop time (§4.2). Value = `mild_s − descent`.

**Thresholds** (measured 5–95% time): **mild < 0.67 s, moderate < 0.50 s, severe < 0.33 s.**
- The intent is full descents (motion start to bottom) of ~1.0 / 0.75 / 0.5 s.
- For smooth human profiles, 5–95% time = 0.62× (minimum-jerk) to 0.71× (cosine) of the full time.
  The thresholds use 0.67× (±7%).
- Basis:
  - the repo prescribes a 2-s eccentric (`program_generator/docs/v5-program-generator-spec-FINAL.md`:
    "Barbell Romanian Deadlift — 3×8-10 @ RPE 7, rest 120s, tempo 2-1-1-0");
  - Coratella et al. 2022 used 2-s phases;
  - under 1 s is half the prescription; under 0.5 s is a drop (free fall over 40 cm = 0.29 s).
- Round 1's start-at-trajectory/end-at-bottom-window definition read a true 2-s descent as ~1.3 s. The
  pre-roll and the 5–95% crossings remove that bias.
- G5 confirms the 0.67 ratio on real reps and rescales if it is off by > 10%.

**Observability.** `vertical`: observable on both modes.

**Other fields.** Drift: `rdl_descent_deficit`. Phase `"descent"`. `kind: "fast_descent"`. New file:
`DescentControlRule` hard-codes `FaultType.TEMPO`.

**Delivery.**
- **Cue** `rdl_slow_down`: "Two seconds on the way down" (6).
- **Fixed praise:** "Better, nice and controlled."
- **Pre-set line:** "This set: two seconds down."
- **Label:** "Control the descent". **`FAULT_LABELS`:** "lowering the bar too fast".

### 5.7 `rdl_partial` — attempt stopped above the count floor (P46, shallow path)

**Metric.** `0.45 − bar_travel_ratio` for a "partial" excursion (§3.6).

**Thresholds.** mild > 0, moderate ≥ 0.07, severe ≥ 0.14 short of the floor (≈ 4° and 8° of hinge,
at ~0.018 per degree near 45°).
- Returned by `profile.on_rep_rejected("partial", features, angles)` for **every** partial, with
  `details.shallow_rep=True`, `counted=False` and `range_label`. It therefore always yields a
  `shallow_rep` message.
- Its cue is `shallow_rep_cue(fault, streak, cued_this_set)`, using the pipeline-owned streak (§3.6):
  `rdl_lower` only when `streak ≥ 2` and `cued_this_set < 2`. Otherwise it is cue-less.
- When the partial instead emits `rdl_hip_sink` (hip sink ≥ the stamped moderate band), the same rule
  yields `rdl_hips_back`. That verdict is ≥ moderate by construction, so it is never `recorded_only`.

**Observability.** `vertical`.

**Delivery.**
- **Cue** `rdl_lower`: "Take the bar lower" (4). Played on the shallow path, which bypasses the gap.
- No fixed or pre-set line.
- **Label:** "Lower to count". **`FAULT_LABELS`:** "stopping too high for the rep to count".

**`FAULT_MESSAGES`.** Three plain lines per type. Example for `rdl_hip_sink`:
- mild: "Hips sinking a bit — push them back"
- moderate: "Sitting down into it — hips back"
- severe: "That's a squat — hips back, shins still"

### 5.8 Scaffold deletions (verified on main)

- **Hinge `RangeOfMotionRule`.** Its 90° target is past most lifters' range.
- **Knee `RangeOfMotionRule`.** `direction="min"` tracks the *minimum* knee flexion, which is the
  standing value, so it never fires. The two rules also share `range_of_motion`, so they dedup each
  other and both map to `rdl_hips_back`.
- **Hip-flexion `SymmetryRule`.** Monocular far-leg noise; the squat knee version false-fired on 74% of
  reps.
- **`BackRoundingRule`.** A correct hinge (180 → ~105 `trunk_flexion`) reads as SEVERE every rep.
- **Unused `TrunkStabilityRule` import, `get_depth_metric` → `trunk_flexion`.**
- **Cue keys `rdl_flat_back` and `rdl_even`.** "Hinge evenly!" also breaks the drafting jargon ban.
- **The `stiff_leg_deadlift` alias.** It moves to `UntrackedVariantProfile`.
- **The same `BackRoundingRule` bug in `barbell_row.py`.** Flag only; the deadlift plan removes its own.

---

## 6. Delivery

**Priorities.** `FAULT_CUE_PRIORITY` gains the additive RDL block: `rdl_hip_sink` 40, `rdl_bar_drift`
41, `rdl_lockout` 42, `rdl_bar_tilt` 43, `rdl_range_drift` 44, `rdl_tempo` 45, `rdl_partial` 46.
- `FAULT_CUE` (7) is already the lowest queue priority, so the offset changes nothing outside RDL.
- Orchestrator rules are unchanged:
  - one focus fault per set;
  - moderate+, or mild on 2 of the last 3 reps;
  - ≤ 2 cues per fault per set;
  - fix praise after 2 clean reps (lagged for `rdl_lockout`).

**Per-exercise orchestrator sets** (SHARED-PLUMBING §4). RDL safety: `{rdl_bar_drift: severe}`,
active only once G6 passes. RDL side-view: `{rdl_bar_drift, rdl_lockout}`.

**Cue fallback** (spec §4). `cache_cues` carries the profile's `fault_to_cue`, and
`orchestrator.set_exercise(name, is_squat=False, fault_to_cue)` stores it. Neither the orchestrator nor
the rest-complete focus line can fall back to a squat clip.

**Positives.**
- `_fixed` lines.
- "strong" on `best_rep_so_far`. For non-diagnosis profiles, `session_tracker._rep_highlights` picks the
  clean rep with the highest `features["highlight_score"]` (RDL addition; squat path unchanged).

**Setup clips.** `rdl_setup` and `rdl_back_note` play once per session at the first RDL set
(`SETUP_CUE_KEYS_BY_PROFILE`, from the `cache_cues` handler). `level_camera` is played by the pitch
guard. All three are agent-side keys: a missing clip is skipped, never sent to cloud TTS.

**Pre-set line.** The previous set's focus pre-set line goes into the rest-complete prompt as a fixed
string. There is no new audio path, and it is trivial for a small local model.

**Recap gets.** The existing flat set summary: clean/total, `fault_summary` + `FAULT_LABELS`,
approximate flags, focus, cue outcomes, effort. RDL additions:
- **Depth fields** (spec §4, `reports_depth=False`).
  - The producer sends the generic depth fields as null.
  - `cache_cues` carries `"reports_depth": False`, and `orchestrator.set_exercise(..., reports_depth)`
    stores it.
  - Per-rep entries carry no `depth` / `depth_angle`, and `coaching_service` no longer coerces a null
    `max_depth_angle` to 0.0 (`coaching_service.py:586`).
- **Best-rep line.** `_build_rep_highlights` (`coaching_orchestrator.py:2109-2117`) today says "Best rep:
  rep N, their deepest clean rep", ranked by `depth_angle`.
  - Gated on `reports_depth`, it is replaced for the RDL by the spec's `highlight_score` line: "Best rep:
    rep N, their most hips-back clean rep" (`highlight_meaning`).
  - It is omitted when no clean rep has a finite score.
  - No recap line describes the RDL in depth terms.
- **Range.**
  - `coaching_service` forwards `features` from `rep_complete` into `orchestrator.on_rep_complete(...,
    features=None)` (RDL addition, additive parameter). Per-rep entries keep `highlight_score`,
    `hinge_deg` and `range_label`.
  - The RDL recap branch computes the hinge SD over the set's `hinge_deg` values, voiced with the
    existing 4°/8° consistency words, and the mode of `range_label`.
  - Line: "Range was {consistency}; the bar usually reached {mode of range_label}."
- **Uncounted attempts.** "{n} attempts stopped too high to count", from the complete `shallow_reps`
  (§3.6). It is voiced in the RDL recap branch; today the shallow line is gated on `is_squat`.
- **One static context sentence** for the focus fault, once per session (`HINGE_FAULT_CONTEXT`).
  Examples:
  - hip sink: "When the hips sink, the hamstrings are usually running out of stretch — a slightly
    shorter range with soft knees trains them better."
  - Range reference < 60°: "Your range is set by your hamstrings, not a target — it grows with
    practice."
- **RDL clause** on `UNOBSERVABLE_FAULTS_LINE`: "Never say or imply anything about lower-back rounding
  or spine position." On one camera it adds "or where the weight sits on the foot". The squat line is
  untouched.

**No diagnosis engine in V1.** `uses_diagnosis_engine=False`, `create_set_diagnosis() → None`,
`waits_for_diagnosis=False`.
- The graph is squat mechanics.
- RDL causes for the V1 faults are few (hamstring length, an unfamiliar hinge, load or fatigue), and a
  static sentence per fault delivers them.
- Calibration is skipped (`get_movement_pattern` → None). The RDL does not need it, because every feature
  self-normalizes to the athlete's own setups.

---

## 7. File-by-file implementation plan (in order)

Items marked **[SP]** are SHARED-PLUMBING.md changes. Build them once, to that spec, on whichever branch
lands first. Everything else is RDL-only.

### Phase A — biomechanics

1. **`faults/fault_types.py`** — add the 7 `RDL_*` `FaultType` members and their `FAULT_MESSAGES`.
2. **`faults/observability.py`** — [SP §6] add the shared `rig_unvalidated` level. RDL
   `FAULT_MEASUREMENT` entries:
   - `vertical`: hip_sink, range_drift, partial, tempo;
   - `frontal`: bar_tilt;
   - `rig_unvalidated`: bar_drift, lockout.
3. **`profiles/base.py`** — [SP §1] hooks: `create_rep_analyzer(config)`, `observe_frame`,
   `create_session_reference`, `create_set_diagnosis`, `allows_camera_refine`, `needs_bar_3d`,
   `get_readiness_check`, `on_rep_rejected`, `shallow_rep_cue(fault, streak, cued_this_set)`,
   `reports_depth`, and [SP §4] `min_cue_severity`.
   - RDL addition: `readiness_hint: str = ""`.
4. **`profiles/registry.py`** — [SP §7] `UntrackedVariantProfile` with the spec's names (the RDL list is
   complete there) and the `NOWVA_DEV_COACHING_READY` override.
5. **NEW `analysis/hinge_features.py`**
   - `HingeFrameSample`, `HingeRepFeatures`, `RdlRepAnalyzer`, implementing the [SP §2] protocol per
     §4.1–4.2 and §3.6.
     - `on_rep_start(piston)`;
     - `rejection_reason`, pure (no streak);
     - optional `is_ready(ctx)` (§3.2);
     - no `counter_baseline`.
   - Constants: `BOTTOM_WINDOW_CM = 3.0`, `SINGLE_CAMERA_MIN_HIP_CONFIDENCE = 0.5`, `PRE_ROLL_S = 0.5`,
     `DESCENT_START_FRACTION = 0.05`, `DESCENT_END_FRACTION = 0.95`, `CROSSING_MEDIAN_SAMPLES = 3`,
     `SETUP_HIP_BAND_CM = 1.0`, `TURNAROUND_BAND_CM = 1.0`, `MAX_SETUPS = 5`,
     `MIN_WRIST_SEPARATION_CM = 20.0`, `MIN_SEPARATION_FRACTION = 0.6`, `BAR_HOLD_LEAN_DEG = 4.0`,
     `RANGE_LABEL_BINS`.
   - Make `rep_features._median`, `_percentile`, `_concentric_velocity_mps` and `_foot_frame` public in a
     no-op rename. `_foot_frame` gains an optional `up` parameter (default `WORLD_UP`, so the squat is
     unchanged).
6. **NEW `faults/hinge_reference.py`** — `HingeSessionReference`, returned by
   `create_session_reference()`.
   - API the engine needs: `depth_target_ratio=None`, `reaches_depth_target → True`, `annotate`,
     `update`, `start_new_set`, `best`, `is_drift`.
   - Plus `hinge_reference()`.
7. **`faults/rule_engine.py`** — [SP §2] `RuleEngine(rules, reference=None)` and the public
   `tag_observability`.
8. **NEW rules** in `faults/rules/`: `rdl_hip_sink.py` (with the pitch-guard observability downgrade),
   `rdl_bar_drift.py`, `rdl_lockout.py` (NaN when `started_at_turnaround`), `rdl_bar_tilt.py`,
   `rdl_range_drift.py`, `rdl_tempo.py`.
   - New `rdl_partial.py` holds `build_rejected_faults(features, angles) -> list[FaultEvent]`, a pure
     builder for `on_rep_rejected`. It reads the thresholds stamped in the features and is not a rule.
   - Export the rules from `rules/__init__.py`.
9. **`config.py` + `config/biomechanics.yaml`** — [SP §5] `RdlConfig` as `FaultsConfig.rdl`, auto-parsed
   with the other `faults:` sub-models:

   ```yaml
   faults:
     rdl:
       counter:   {entry_vel_threshold: 4.0, bottom_vel_threshold: 5.0, ascending_vel_threshold: 4.0,
                   min_depth_cm: 8.0, standing_return_cm: 4.0, min_descending_s: 0.25,
                   min_bottom_s: 0.08, min_ascending_s: 0.25, min_rep_duration_s: 0.8}
       rom:       {count_floor_travel: 0.45, partial_min_travel: 0.25, fallback_floor_deg: 50.0,
                   partial_min_duration_s: 1.0, partial_cue_streak: 2, max_partial_cues_per_set: 2,
                   highlight_range_tolerance_deg: 8.0, reference_cap_deg: 95.0}
       camera:    {tripwire_deg: 8.0, correct_by_mount_angle: false}   # V1.1
       rig_min_cue_severity: moderate                         # until measured `up` + G6
       hip_sink:  {mild: 0.12, moderate: 0.15, severe: 0.19}     # start tier: mild iff mount certified
       bar_drift: {mild: 0.25, moderate: 0.40, severe: 0.60}     # ankle->toe lengths ahead of midfoot
       lockout:   {forward_mild: 10.0, forward_moderate: 15.0, forward_severe: 22.0,
                   back_mild: 8.0, back_moderate: 12.0, back_severe: 18.0}
       bar_tilt:  {mild: 4.0, moderate: 7.0, severe: 10.0}       # deg, bottom minus start
       range_drift: {mild: 8.0, moderate: 15.0, severe: 25.0}    # deg short of own reference
       partial:   {moderate: 0.07, severe: 0.14}                 # travel short of the floor
       tempo:     {mild_seconds: 0.67, moderate_seconds: 0.50, severe_seconds: 0.33}   # 5-95% descent
   ```

10. **`profiles/romanian_deadlift.py`** — rewrite.
    - `@register_profile("romanian_deadlift", "rdl")`; `stiff_leg_deadlift` lives in the variant profile.
    - `display_name = "Romanian deadlifts (barbell, from the rack)"`.
    - Flags: `coaching_ready = False`; `uses_diagnosis_engine` and `uses_bilstm_counter` stay False;
      `reports_depth = False`; `allows_camera_refine = False` (hinged poses with plates near the shins
      would degrade the shared rig calibration; deadlift rationale).
    - `readiness_hint = "HANDS IN VIEW"`.
    - `min_cue_severity`:
      - `rdl_hip_sink` → absent statically. `cue_start_tier` [SP §4] returns:
        - mild while `ctx.pitch_certified` and the lens checks pass;
        - moderate when the lens checks pass without certified pitch;
        - severe when the lens checks fail or the height is unknown (§3.2, §5.1);
      - `pitch_delta_tolerance_deg` = T from §3.2: `max(2.5°, 2.6 σ)` on the ≥ 36-frame set average, set from G2(e) (the
        profile's stated delta-tripwire tolerance, spec §3), on the ankle→hip axis;
      - on certified mounts the analyser levels the skeleton with `geometry.level_camera_frame` by
        `ctx.mount_pitch_deg` [SP §3];
      - `rdl_bar_drift`, `rdl_lockout` → `rig_min_cue_severity`;
      - the rest → mild (absent).
    - Signal (§3.3); counter config from `config.faults.rdl.counter`; rules in priority order; cue map
      (§5).
    - Hooks:
      - `create_rep_analyzer(config)` → `RdlRepAnalyzer(config.faults.rdl, config.readiness_gate)`;
      - `create_session_reference`;
      - `on_rep_rejected` (→ `build_rejected_faults` for `"partial"`, `[]` for `"ignore"`);
      - `shallow_rep_cue(fault, streak, cued_this_set)`: the mapped cue iff `streak ≥ 2` and
        `cued_this_set < 2`.
11. **`utils/standing_gate.py`** — [SP §2] AND in the analyser's `is_ready` over the consecutive-frame
    window. On False, `last_failure = "profile_check"`.
12. **`pipeline.py`** — [SP §1–§2] the analyser path:
    - `FrameContext` with `up`;
    - `observe`;
    - `on_rep_start(piston)` in the spec's call order;
    - `finish_rep`;
    - `rejection_reason` → the rejection sequence → `on_rep_rejected` → `tag_observability`;
    - the pipeline-owned rejection streak and `cued_this_set`;
    - a `shallow_rep` fault sets `shallow_rep_class`;
    - `rule_engine.finish_rep` before `features.model_dump()`;
    - [SP §4] the `recorded_only` stamping against `min_cue_severity`;
    - `go_deeper` / abort → `analyzer.discard_rep()`;
    - `_recent_captures` keeps raw confidences for `ctx.raw_confidences`;
    - `reset_rep_counter()`.

    RDL-only: the pitch-guard `send_pipeline_status("camera_pitch", …)`. The squat `_depth_gated` /
    BiLSTM branches are untouched.
13. **`coaching/cue_cache.py`** — the RDL priority block, 40–46 ([SP §6]).
14. **`coaching/ipc_bridge.py`**
    - [SP §4] null depth fields when `reports_depth` is False.
    - `send_shallow_rep(..., cue_key="deeper", label=None)`: callers pass `profile.shallow_rep_cue(...)`
      and `details["range_label"]`.
    - [SP §4] `send_set_complete`'s `fault_summary` (`:363-372`, the display counts) skips recorded-only
      verdicts.
15. **`coaching/session_tracker.py`** — `highlight_score` best rep when diagnosis is off; [SP] the
    `set_diagnosis` hook, the `reports_depth` nulls and `recorded_only` excluded from `is_clean`.
16. **`pipeline_process.py`**
    - HUD uses `profile.readiness_hint` for `"profile_check"`.
    - Pass the shallow cue and label.
    - [SP] skip camera refine when `allows_camera_refine` is False.
    - [SP §8] raw skeleton recorder hook.
17. **`db/biomechanics_persistence.py`**
    - Add `rdl_lockout` to `END_OF_REP_FAULT_TYPES`.
    - [SP §4] Every aggregation skips `details.recorded_only` verdicts: the session summary's
      `top_faults` (`:181-215`), `get_fault_progress` (`:853`) and `get_multi_session_fault_trends`
      (`:929`). Recorded-only rows stay stored, flagged, in `faults_detailed`.

### Phase B — delivery

17b. **NEW `agent/services/fault_filters.py`** — [SP §4] `reportable_faults(faults, spoken: bool) -> list`.
     It always drops `details.recorded_only` verdicts. With `spoken=True` it also drops `approximate`
     verdicts unless the caller hedges them. **Every** agent-side reader of `faults_detailed` /
     `RepData.faults` below goes through it; none filters on its own.
18. **`agent/services/coaching_constants.py`**
    - `CUE_TEXT_MAP` and `CUE_DISPLAY_LABELS` for the 7 RDL keys. Replace the scaffold `rdl_hips_back`
      text; delete `rdl_flat_back` and `rdl_even`.
    - `FIXED_CUE_TEXT` `rdl_*_fixed`; `BETTER_CUE_TEXT` `rdl_hips_back_better`, `rdl_bar_close_better`,
      `rdl_finish_tall_better`; `PRESET_LINE_TEXT`.
    - `rdl_setup`, `rdl_back_note`, `level_camera` (neutral text, §3.2).
    - `HINGE_FAULT_CONTEXT`.
19. **`agent/services/coaching_orchestrator.py`**
    - [SP §4]:
      - `set_exercise(name, is_squat, fault_to_cue, reports_depth)`;
      - `waits_for_diagnosis`;
      - per-exercise constant sets;
      - `_build_rep_highlights` gated on `reports_depth`, with the `highlight_score` best-rep line;
      - **the recorded-only path:**
        - `ipc_bridge` sends `"recorded_only": true` on the `fault` message and keeps those verdicts out
          of `faults_in_rep`; they stay flagged in `faults_detailed`;
        - `coaching_service` passes `recorded_only` into `on_fault(...)`;
        - `_add_fault_candidate` drops them on both paths, `on_fault` and the `faults_detailed` loop in
          `on_rep_complete`. The loop goes through `reportable_faults(..., spoken=True)`, so
          `_passes_bandwidth` and `_resolve_cue_outcome` never see them;
        - recap fault counts go through `reportable_faults`;
        - a hedged mention, with no size, only for types in `faults.rdl.hedged_recap_faults`.
    - RDL:
      - `LAGGED_FAULT_TYPES`, applied to fix praise **and** `_cue_outcome`'s `reps_after`;
      - [SP §4] `_better` / `_fixed` praise rule. A missing `_better` clip is skipped, never `_fixed`.
        RDL supplies the three `_better` keys;
      - `on_rep_complete(..., features=None)` and the RDL recap branch (range from `hinge_deg` /
        `range_label`, uncounted attempts);
      - setup clips once per session;
      - `camera_pitch` status → `level_camera`;
      - unobservable clause;
      - pre-set line.
20. **`agent/services/coaching_service.py`**
    - [SP §4]:
      - pass `fault_to_cue` and `reports_depth` from `cache_cues`;
      - no 0.0 coercion of a null depth;
      - rest-complete focus line via the profile map;
      - progress queries pass `exercise=`.
    - [SP §4] pass `recorded_only` into `on_fault`.
    - [SP §4] `last_rep_verdict()` (`:393-413`, feeds `check_my_form`) reads faults through
      `reportable_faults(..., spoken=True)`, and hedges any approximate verdict it keeps.
    - RDL: forward `features` to the orchestrator; relay the `camera_pitch` status.
21. **`agent/services/progress_context.py`** — `FAULT_LABELS` for the 7 types; [SP] exercise-aware
    wording. Progress summaries ("most common fault", trends) read faults through `reportable_faults` and
    the recorded-only-skipping DB aggregations (step 17).
21b. **`agent/agents/workout_agent.py`** — [SP §4] the `show_me("last_rep")` replay narration
     (`:1165-1175`) reads faults through `reportable_faults(..., spoken=True)`.
22. **`src/visual/display.html`** — `FOCUS_TEXT` for the 7 types.
23. **`agent/agents/shared/helpers.py`** — aliases → "Romanian Deadlift": "rdls", "romanian
    deadlifts", "barbell rdl(s)", "barbell romanian deadlift(s)".
24. **`agent/core/pipeline_factory.py`** — add `STT_KEYTERMS` "RDL" and "Barbell Romanian Deadlift".

### Phase C — cue text and audio (nothing runs without the user's go-ahead)

25. **`scripts/tools/draft_cue_text.py`** — [SP] `--exercise romanian_deadlift`. RDL adds:
    - its system prompt (barbell RDL; the bar may be mentioned; the "hinge", "eccentric" etc. bans stay);
    - `CUE_SCENARIOS` for the 7 keys plus `_fixed`, the three **`_better`** keys (`rdl_hips_back_better`,
      `rdl_bar_close_better`, `rdl_finish_tall_better`; "improved but not yet right" scenarios, per spec
      §4) and the setup keys;
    - `MAX_CUE_WORDS = 6` for RDL (squat stays 4);
    - explain keys `rdl_setup`, `rdl_back_note`, `level_camera` (≤ 16 words).
26. **`src/assets/cue_text/cues.json`** — 3 reviewed lines per key, including the `_better` keys.
27. **`scripts/tools/generate_cue_audio.py`** (ElevenLabs on main) — `--cue` for the RDL keys.

### Phase D — tests

- **Unit**
  - `test_hinge_features.py`:
    - every formula;
    - B′ per capture mode: one-camera hips at visibility 0.3 → `"setup"`; **rig hips at the two-view cap
      0.3 → judged**;
    - pre-roll and 5–95% crossings with interpolation at 12/30 fps: a cosine 2.0-s descent reads
      1.43 ± 0.05 s;
    - rotated athlete → no tilt;
    - setup lifecycle: `reset_set` keeps setups; a new analyser starts empty;
    - `on_rep_start(piston=True)`: setup = turnaround frames (not pushed onto `_setups`), pre-roll =
      post-`i_low` samples, `S_top` = smoothed max, `started_at_turnaround=True`;
    - features carry `exercise="romanian_deadlift"` and the stamped thresholds;
    - `up` test (spec §3): an upright trunk reads `trunk_pitch_signed = 0 ± 0.5°` with both `"body"` and a
      tilted `"measured"` `up`, and a bar ahead of the feet reads positive `bar_offset_ratio`.
  - `test_rdl_rules.py`:
    - band edges, side, drift;
    - pitch guard → approximate;
    - `rejection_reason` is pure;
    - `on_rep_rejected` returns one `shallow_rep` fault per partial (`rdl_hip_sink` above the stamped
      moderate band);
    - `shallow_rep_cue(fault, streak, cued_this_set)` matrix: streak 1 → None; streak ≥ 2 with
      `cued_this_set` < 2 → `rdl_lower` / `rdl_hips_back` by fault type; `cued_this_set` 2 → None;
    - `cue_start_tier` truth table: `ctx.pitch_certified` × lens band × per-athlete hip check × height
      known → mild / moderate / severe; an uncertified install with a 0.5 m lens → severe;
    - rig faults mild → `recorded_only`;
    - the `rdl_hip_sink` rule never sets `recorded_only` itself;
    - **mid-session revocation:** a certified session emits a cued mild hip-sink verdict. The delta
      tripwire then revokes `ctx.pitch_certified` once that set's 36-frame average is in, and the very next mild verdict
      is stamped `recorded_only` by the pipeline (tier moderate), with no cue;
  - `test_hinge_reference.py`.
  - `test_rdl_profile.py`:
    - flags;
    - rule order = priority;
    - every key `rdl_`-prefixed and present in `CUE_TEXT_MAP`, `CUE_DISPLAY_LABELS`, `cues.json`,
      `FAULT_LABELS`, `FAULT_CUE_PRIORITY` (40–46) and `FAULT_MEASUREMENT`;
    - ≤ 6 words, no banned words;
    - signal rises on descent and is NaN without shoulders;
    - `is_ready` uses only the raw skeleton (angles None); `readiness_hint`.
  - `test_profile_registry.py`: the RDL variants → untracked; "Single-Leg RDL" updated.
  - `test_standing_gate.py`.
  - `test_shared_helpers.py`.
  - `test_coaching_orchestrator.py`:
    - no squat cue for RDL faults;
    - side-view RDL faults silent on one camera;
    - lagged lockout praise and lagged `_cue_outcome` `reps_after`;
    - best-rep line = "most hips-back clean rep", never "deepest"; no 0.0 depth coercion;
    - `recorded_only` verdicts give no cue and are absent from recap counts. **Three consecutive mild
      recorded-only `rdl_hip_sink` verdicts produce no cue**, on both the `on_fault` and the
      `faults_detailed` path;
    - after a moderate hip-sink cue, over the 2-rep hold window:
      - hold reps (recorded-only mild, none) → `rdl_hips_back_better`;
      - hold reps (none, none) → `rdl_hips_back_fixed`;
      - after `_better`, two further clean reps → no `_fixed` (outcome resolved);
      - `rdl_hips_back_better` removed from disk → nothing plays, never `_fixed`;
    - certification: lens 0.5 m → tier moderate; a 195-cm lifter on a 0.8 m lens → moderate; delta 4° at a
      set's 36-frame idle average → `pitch_certified` revoked for the session, `level_camera` once;
    - setup clips once;
    - `level_camera` once;
    - range recap line;
    - cue-less shallow reps counted.
  - `test_progress_context.py`.
  - `test_biomechanics_persistence.py`:
    - `rdl_lockout` is end-of-rep; the other RDL types are not;
    - `top_faults`, `get_fault_progress` and `get_multi_session_fault_trends` skip recorded-only rows.
  - `test_fault_filters.py` + **RDL shared test** [SP §4]. One rep carries a recorded-only `rdl_hip_sink`
    (mild, uncertified) and an approximate `rdl_bar_drift`:
    - `check_my_form` (`last_rep_verdict`) and `show_me("last_rep")` say there is nothing to correct in
      what this camera can see, or hedge the bar drift. They never state either fault as fact;
    - a session whose only hip-sink verdicts are recorded-only gives no "most common fault: hip sink"
      line in `progress_context`, no trend entry, and no display `fault_summary` count.
- **Synthetic poses in the production frame** (`test_rdl_synthetic.py`). Add `world_rdl_points(hinge_deg,
  knee_flex_deg, shank_deg, tilt_deg, bar_forward_m, height_m)` to `tests/test_biomechanics/conftest.py`:
  (21, 3), Y-down, −Z forward, Winter proportions at 188.5 / 160 cm, optional camera pitch and N(0, 1 cm)
  noise. Run cosine reps at 12/30 fps through `AnalyticalIKSolver` → `RdlRepAnalyzer` →
  `SignalRepCounter` → `RuleEngine` (both modes). Assert:
  - 10 clean reps (k20, s5, lean 75°) → 10 counted, zero faults;
  - **vertical-shin k33, lean 70°: at pitch ±8° no cueable `rdl_hip_sink` (default start tier); at ±5°
    no verdict at any severity**;
  - **certified mount at 2°, leveled**, anisotropic noise (σ_z = 3·σ_xy, spec §3) plus a 10 cm hip depth
    bias: vertical-shin k33 correct reps → no hip-sink verdict at any severity;
  - k50 s12 → `rdl_hip_sink` ≥ mild; k60 s15 → ≥ moderate (cued);
  - unrack dip (bar to mid-thigh hooks) → ignored, with no cue even when repeated after a counted rep;
  - two travel-0.35 attempts → second emits `rdl_partial`; both send `shallow_rep`;
  - full 0.5-s descent → `rdl_tempo` ≥ moderate; full 2.0-s descent → none;
  - 7° drop of one hand → `rdl_bar_tilt` moderate;
  - single camera → no `rdl_bar_drift` / `rdl_lockout` event; triangulated → approximate;
  - `hip_conf` 0.3 on one camera → no hip-sink verdict;
  - two-view rig hips → verdict;
  - **piston set**: 5 constant-tension reps turning around at 25% of range →
    - 5 counted;
    - no `rdl_lockout`;
    - `descent_time_s` of reps 2–5 within ±0.08 s of the generated 5–95% time;
    - references unchanged from the standing setup.
- **Pipeline integration** (`test_pipeline.py`; `_pipeline_for`, the `estimate_both` stub,
  `_FakeProvider`):
  - RDL end-to-end in both modes;
  - `_bilstm is None` with `bilstm.enabled=True`;
  - [SP §8] squat → RDL → squat switch golden.
- **Replay.** [SP §8] The shared raw skeleton recorder and replay run every validation session through
  gate → pre-IK → IK → analyser/counter → rules. Replays run at the native rate and decimated to 12 fps.
  RDL adds only a small report script: a per-rep CSV of `HingeRepFeatures` plus faults, and P/R against
  the label CSV.
- **Squat regression (G9).** [SP §8] golden masters (fresh, after a switch, delivery) and the full suite:
  byte-identical.

### Phase E — flip `coaching_ready = True`

Flip when G1, G2, G7, G8 and G9 pass on one camera.
- G3–G5 faults that miss their gate are demoted to approximate (a `FAULT_MEASUREMENT` change) rather than
  blocking the flip.
- Bar drift and lockout stay `rig_unvalidated` until G6.
- `pitch_certified` is an installation fact, not a gate outcome: the rack mount, or an inclinometer
  reading ≤ 2° entered at setup (§3.2). The demo rig is set up certified. An uncertified installation
  runs hip sink at start tier moderate if the lens checks pass, and at severe otherwise.
- Validation sessions use [SP §7] `NOWVA_DEV_COACHING_READY=romanian_deadlift`.
- Update `test_only_squats_are_offered_for_coaching`.

---

## 8. Validation protocol

**Recordings.**
- ≥ 8 lifters, 155–195 cm, including the 188.5 cm test user, ≥ 3 novices, and **≥ 2 flexible lifters who
  hinge ≥ 80°**.
- Empty bar and a working weight at RPE 6–8.
- Every session captures, simultaneously:
  - frontal mono at 30 fps, hip-height lens: the shared raw skeleton recorder plus video for labelling;
    replayed at 30 and decimated to 12;
  - a side reference camera at hip height, perpendicular, 60–120 fps, with greater-trochanter, acromion,
    lateral-knee and lateral-malleolus markers;
  - the rig when available.
- **Per lifter:**
  1. **5** unrack + re-rack sequences with no reps (8 × 5 = 40).
  2. 2 × 8 clean reps at natural tempo, varying knee bend 15–33° and shin angle as the lifter naturally
     does.
  3. Acted faults.
     - **Squatting it, spanning the detection range.** Three sets of 4 reps, coached by knee angle and
       shin angle checked on the side video:
       - (i) "a bit more knee": ~35–40° knee flexion, shins near vertical (coach mild);
       - (ii) "knees forward": ~45–55°, shins 15–20° forward (coach moderate; reads ~8–12%, cued only at its
         ≥ 12% end on a certified mount — the R2 limit);
       - (iii) "sit down into it": ≥ 65°, shins ~20–25° (coach severe; reads ≥ 17%, which is moderate,
         i.e. cued). At 60° with 20° shins it reads 14.6%, just under moderate.

       That gives ≈ 32 reps per level across 8 lifters.
     - Other faults, one set of 6 at 2–3 graded severities:
     - shortening reps 4–6 by 10–25°;
     - dropping in < 0.7 s;
     - rig only: bar 10+ cm off the legs, unfinished hips, leaning back.
  4. 4 partial attempts to mid-thigh.
  5. **Camera sweep** (3 lifters, including both ≥ 80° hingers):
     - lens at 0.5 / 0.8 / 1.0 (hip) / 1.3 m (chest);
     - 4° camera re-aims between sets, on an inclinometer-certified mount;
     - pitch 0°, ±2°, ±5°, ±8°, ±10° (inclinometer), with the lifter holding the bar, so the posture term in
       `camera_pitch_deg` is measured;
     - athlete yaw 20–30°.
- **Totals.** ≈ 650 reps; ~96 squatting-it reps (≈ 32 per level), ~40 acted reps per other fault;
  40 unrack/re-rack sequences.

**Labelling.**
- A CSCS coach labels each rep per fault (none/mild/moderate/severe) from front + side video, blind.
  - "Squatting it" rubric: none = soft, still knees with hips travelling back; mild = noticeable extra
    knee bend; moderate = clear knees-forward / hips-down; severe = a squat or conventional-deadlift
    pattern.
- 20% double-labelled; Cohen's κ ≥ 0.6 per fault, else the fault is redefined (deadlift plan §8.3).
- Side-video annotation (Kinovea) on every clean rep and 1 in 4 acted reps, measured with **the same
  definitions as the features**:
  - **hip sink** = trochanter marker drop ÷ standing trochanter height;
  - **thigh inclination** (trochanter→knee vs vertical) and **shin inclination**;
  - knee angle;
  - trunk angle (trochanter→acromion) at the bottom and at the finish;
  - bar–shin distance at the bottom;
  - **5–95% shoulder-drop time** and full descent time.

**Acceptance.**
- Tune leave-lifter-out; report held-out lifters with a lifter-clustered bootstrap 95% lower bound
  (deadlift plan §1/§8.3).
- **Error-budget rule** (deadlift plan §2.9): a tier is cued only if the P95 measurement error vs ground
  truth is ≤ 1/3 of that tier's threshold; otherwise cueing starts one tier higher.

| Gate | Metric | Pass |
|---|---|---|
| G1 rep count | Rep precision / recall; exact-count sets; false counts in unrack/re-rack | ≥ 0.99 / ≥ 0.98; ≥ 95%; **0 in 40** |
| G2 hip sink (one camera and rig) | (a) Clean reps (side-video knee ≤ 33°), **stratified by shin < 5°, 5–15°, > 15°**: p95 `hip_sink_ratio` in every stratum, recorded on a certified mount (inclinometer ≤ 2°). (b) P95 error vs side-video hip sink. (c) Spearman vs side-video hip sink, and vs thigh inclination. (d) **Precision** of cued verdicts (≥ mild, certified) vs coach ≥ mild. **Recall** vs **coach-severe**: share flagged at a cued tier. **Recall vs coach-moderate** is reported, not gated, as the known limit (R2). (e) Pitch sweep: hip-sink bias at ±2° / ±5° / ±8°. **Delta tripwire (ankle→hip axis):** σ and p95 |Δ| over repeated latches on an unmoved mount, with and without the bar; trip rate at **4° knocks** (camera re-aimed between sets). Lens-height sweep 0.5 / 0.8 / 1.0 / 1.3 m: hip-sink bias on clean reps. (f) Hip-lens vs chest-lens hip visibility and bias at ≥ 80° | (a) < 0.11. (b) ≤ 0.04. (c) ≥ 0.85 / ≥ 0.80. (d) P ≥ 0.85 (LB ≥ 0.70); severe recall ≥ 0.70 (LB ≥ 0.55). (e) bias ≤ 0.009 / 0.024 / 0.038. Consistent set, on ≥ 36-frame set averages against a 3-set reference: unmoved σ_set ≤ 1.0° and σ_Δ ≤ 1.15°, false trips ≤ 3% per set, **and** 4°-knock trip ≥ 90% at T = 2.5°. Knocks of 3.1–3.9° are the stated residual. If σ_Δ > 1.15°, T = 2.6 σ_Δ (keeps false trips ≤ 1%) and the residual band widens to 3.1° – (T + 1.28 σ_Δ), bounded by the session-start re-measurement (the expected procedure for inclinometer mounts either way). Lens bias at 0.8–1.1 m ≤ 0.007; the 0.5 m result is reported (expected ≈ +0.023, the reason for the band). If (a) fails on certified mounts, start tier moderate everywhere. (f) reported; picks the shipped placement |
| G3 bar tilt | P / R (mild+); P95 tilt error | ≥ 0.80 / ≥ 0.70; ≤ 1.3° |
| G4 range drift | P95 relative `hinge_deg` error (rep minus own reference) vs side video; drift P / R | ≤ 2.7°; ≥ 0.80 / ≥ 0.75 |
| G5 tempo | 5–95% time MAE vs side video (same definition); measured/full ratio | ≤ 0.08 s; 0.67 ± 0.07, else rescale; P ≥ 0.90 / R ≥ 0.85 |
| G6 rig faults | Bar-drift and lockout P / R on triangulated ≥ 30 fps; wrist–bar offset | ≥ 0.80 / ≥ 0.75; measured and subtracted |
| G7 audio | Every pipeline-listed RDL key (missing clips fall back to **cloud TTS**, `audio_cue_service.cache_cues`) and every agent-side key has 3 reviewed clips. That covers `rdl_setup`, `rdl_back_note`, `level_camera`, the 6 `_fixed` keys and the **3 `_better` keys**. Missing `_better` clips would silently drop the praise (spec §4 skip rule) | 100% (test) |
| G8 coach review | Blind review of 30 cue events; false cues on clean sets | ≥ 80% compatible; ≤ 1 per 20 clean reps |
| G9 squat regression | [SP] goldens (fresh, after a switch, delivery) and the full suite | byte-identical / green |

With ~40 acted reps per fault, P/R intervals are about ±0.1. G2 is the only fault gate the flip waits on;
the others demote.

---

## 9. Risks, open questions, out of scope

**Risks (ranked).**

1. **R1: hip-sink measurement on one camera.** Three threats: hip-landmark migration in deep flexion,
   occlusion for ≥ 85° hingers (hip-height lens), and pitch leak. Mitigations:
   - hip-height level lens (§3.1);
   - the raw-visibility gate B′, with "setup" fallback (counting only);
   - certified mount (rack, or inclinometer ≤ 2°) plus a posture-aware tripwire for knocks (§3.2);
   - bands and start tier built on the worst-case envelope plus the installation's pitch budget (2°
     certified, 8° otherwise);
   - G2 stratified by shin angle and camera height.

   Fallback: approximate on one camera; the demo then runs on tilt, range and tempo.
2. **R2: one-camera sensitivity is limited by design.**
   - Hip sink cannot see knee travel at moderate knee bend. A coach-moderate squatted RDL (45–55° knees,
     15–20° shins) reads 8–12%: cued only at its pronounced end (≥ 12%) on a certified mount, and not at
     all on an uncertified one: start tier moderate (≥ ~60° patterns) with the lens checks passing, and
     severe (≥ ~52–72°, depending on shin) outside the lens band.
   - G2 reports recall at coach-moderate as this known limit.
   - The rig knee/shin metric (V1.1) narrows the gap. On one camera, mild 0.10 is not supported by the
     pitch budget (§4.3).
3. **R3: spinal rounding is unjudged**, and users may over-trust Nova. Mitigations: honesty clip; range
   never pushed past own reference; praise for hips-back; recap ban.
4. **Unrack / re-rack** counted or cued. Mitigations: travel floor, streak rule, hook-height clip, G1.
   Contingency: step detector.
5. **Thresholds are estimates.** Rig faults stay approximate until triangulated recordings exist.
6. **Rig occlusion.** Side cameras see plates over the near shin at the bottom.
7. **Dependency on the deadlift branch's hooks.** Whichever lands first builds them. If the deadlift
   slips, RDL builds the [SP] hooks itself to SHARED-PLUMBING.md.
8. **Scope (guardrail #5).** Sequence after the squat V1 live test.

**Open questions.**
- Can the product rack fix J-hook height for RDL presets?
- Should `rdl_partial` stay cued, or be recap-only?
- Range reference: session-wide (chosen) vs per load.
- Does the squat's current mono placement already match the hip-height lens? G2(f) decides.

**Out of scope (V1).**
- conventional or sumo deadlift (the deadlift plan);
- stiff-leg deadlift;
- single-leg, dumbbell, kettlebell, deficit, snatch-grip and B-stance RDL (untracked variants);
- trap bar;
- spine sensing;
- rig hip shift, knee/shin metric, knee lock, knee-before-hip ascent (V1.1 / V2);
- velocity autoregulation;
- hinge diagnosis graph.

---

## 10. Shared plumbing

All shared platform changes follow **`SHARED-PLUMBING.md`**:
- hooks (§1);
- `RepAnalyzer`, including `on_rep_start(piston)`, the pipeline-owned rejection streak, `shallow_rep_cue`,
  `tag_observability` and the optional `is_ready` / `counter_baseline` (§2);
- `FrameContext`, `up` and `gravity_source` (§3);
- cue-map fallback, `min_cue_severity` / `recorded_only`, `waits_for_diagnosis`, `reports_depth` at
  producer and agent, the best-rep line, per-exercise sets, progress queries and `draft_cue_text
  --exercise` (§4);
- `faults.rdl` (§5);
- priorities 40–49 and `rig_unvalidated` (§6);
- `UntrackedVariantProfile` (complete RDL list) and the dev override (§7);
- goldens and the raw recorder/replay (§8).

This plan uses them as specified. Only the RDL-specific items are listed here.

| Item | Kind | Note |
|---|---|---|
| `readiness_hint` profile attribute | RDL addition | The readiness predicate returns a bool, so the HUD needs a hint for `"profile_check"` |
| One-camera vertical axis and leveling | **Uses spec §3** | `geometry.level_camera_frame(points, mount_pitch_deg)` on certified mounts; `ctx.up` on the rig; never the unvalidated `"camera_pitch"` (§3.2, §4.3) |
| Gross tripwire on uncertified installs (`camera_pitch` status, `level_camera`, hip-sink observability downgrade) | RDL-only | Posture-aware reading `camera_pitch_deg − 4°` (§3.2) |
| Camera-mount fields, `pitch_certified`, delta tripwire | **Uses spec §3** | RDL requirements: measured pitch ≤ 2° re-measured each session, lens 0.8–1.1 m and ≤ 10 cm below the lifter's hips, ankle→hip delta with T matched to measured noise (§3.2) |
| `reportable_faults` and the recorded-only skips in persistence, display and agent readers (`check_my_form`, `show_me`, progress) | **Uses spec §4** | RDL adds the shared test with recorded-only hip sink plus approximate bar drift (Phase D) |
| `cue_start_tier` override (hip sink mild iff `ctx.pitch_certified` and lens checks; rig faults `rig_min_cue_severity`) | **Uses spec §4** | The pipeline is the only stamper of `recorded_only` |
| `<cue>_better` / `_fixed` praise and the skip rule | **Uses spec §4** | RDL supplies `rdl_hips_back_better`, `rdl_bar_close_better`, `rdl_finish_tall_better` |
| `orchestrator.on_rep_complete(..., features=None)` + RDL recap branch | RDL-only, additive | Range consistency and `range_label` come from features |
| `session_tracker` best rep via `highlight_score` when diagnosis is off | RDL addition | The recap line follows the spec |
| `END_OF_REP_FAULT_TYPES` += `rdl_lockout`; `LAGGED_FAULT_TYPES` (fix praise and `_cue_outcome`) | RDL-only | The deadlift judges its lockout on the rep's own top |
| `send_shallow_rep(cue_key, label)` parameters | RDL addition | Defaults keep the squat's `"deeper"` |
| `_foot_frame(..., up=WORLD_UP)` parameter | RDL addition | The squat default is unchanged |

No conflicts with the spec remain.

## Changelog

**Round 1 (2026-10-03).** Initial plan from code on main @265ef97 plus geometry modelling (Winter
proportions, 188.5/160 cm). Sources:
- NSCA RDL technique and errors;
- McCurdy & Koldenhoven 2025 (IJSC);
- Coratella et al. 2022;
- in-repo program spec.

**Round 2 (2026-10-03).** Scorer 7.6 → revision.
1. **P0 bands.**
   - Re-tabulated hip drop across shin 0..k/2 (§4.4). It is largest with a vertical shin.
   - Renamed the fault `rdl_knee_bend` → `rdl_hip_sink` and stated it measures hip sink (thigh + shin
     inclination), not knee travel.
   - Re-derived the bands from the vertical-shin 33° envelope (8.0%) plus the worst bias budget:
     0.12 / 0.15 / 0.19. A correct rep cannot reach mild.
   - Stated what one camera misses (R2).
   - G2 now references side-video hip sink and thigh inclination, stratified by shin angle.
2. **B′ gate is capture-mode dependent.**
   - One camera: raw MediaPipe visibility ≥ 0.5. Rig: triangulated from ≥ 2 views, so two-view hips
     capped at 0.3 are kept.
   - Raw confidences are plumbed through the ctx.
   - Added the two-view synthetic test.
3. **Camera geometry redone.**
   - Hip-height level lens; end-on hinge `θ* = arctan(D / (h_lens − h_hip))` tabulated per lens height.
   - The chest lens degrades 80–88° hingers.
   - R1 rewritten; ≥ 80° hingers added to the camera sweep.
4. **Tempo.**
   - Analyser pre-roll; 5–95% shoulder-drop crossings with interpolation.
   - Thresholds restated in that measurand (0.67 / 0.50 / 0.33 s ≈ full 1.0 / 0.75 / 0.5 s, ratio
     0.62–0.71).
   - G5 uses the identical definition.
5. **Bias budget** (§4.3) adds perspective and rig vertical.
   - The count floor and partial window moved from `hinge_deg` to pitch-robust hand travel
     (≥ 0.45; 0.25–0.45); `hinge_deg` is relative-only plus a 50° fallback.
   - Pitch guard on `camera_pitch_deg` > 5°; rotation correction as a G2-gated V1.1 lever.
   - Fixed §1's "only pitch" claim.
6. **`rdl_lockout`** added to `END_OF_REP_FAULT_TYPES` (tested); lagged fix praise.
7. **§2.2** lists straight-knee "hips not back" as not judged on one camera, with the setup-clip
   mitigation.
8. **Details an engineer would have guessed.**
   - The analyser ctx is defined.
   - The `_setups` lifecycle is defined.
   - The partial streak is defined: ignore keeps it; counted rep and `reset()` clear it.
   - `depth_target_met` before 3 reps = True.
   - A cue-less `shallow_rep` is sent for every partial.
   - The readiness check is now described as visibility-only ("HANDS IN VIEW").
9. **Consistency fixes.** 5 unrack/re-rack sequences per lifter (40 total). Snatch-grip and other variants
   moved to `UntrackedVariantProfile`.
10. **Coordination with the deadlift plan.**
    - Adopted `create_rep_analyzer`, `observe_frame`, `create_session_reference`, `create_set_diagnosis`,
      `allows_camera_refine`, `needs_bar_3d`, the `cache_cues` map, `waits_for_diagnosis`, per-exercise
      sets, `draft --exercise`, the variant profile, priorities 30–36, the dev override, goldens and the
      error-budget rule.
    - Added §10 Shared plumbing.
    - One conflict: `stiff_leg_deadlift`.

**Round 2b (2026-10-03, conformance to `SHARED-PLUMBING.md`).**
- Analyser follows spec §2: `classify_excursion` → `rejection_reason`; partial faults via
  `on_rep_rejected`; one `shallow_rep` fault per partial, cued only when eligible.
- `RdlPartialRule` and `judge_rejected_rep` removed.
- Priorities 40–46.
- `FrameContext` per spec (no RDL fields).
- `reports_depth=False`, with recap range from features.
- `faults.rdl` config placement.
- Shared recorder replaces `replay_video.py`.
- `depth_target_met` dropped.
- §10 cut to RDL-only items and spec gaps.

**Round 3 (2026-10-03).** Scorer 8.1, plus spec updates.
- Best-rep recap line now gated on `reports_depth` and replaced by the `highlight_score` "most hips-back"
  line; no 0.0 depth coercion.
- `shallow_rep_cue(fault, streak, cued_this_set)` with the pipeline-owned streak; the analyser streak
  and `partial_cue_eligible` are removed.
- `create_rep_analyzer(config)`, `tag_observability` and stamped thresholds; `exercise` field; spec
  section refs fixed; the variant list is the spec's.
- Piston reps: turnaround setup, post-`i_low` pre-roll, smoothed `S_top`, lockout NaN; piston test
  added.
- G2(d): precision at cued tiers, recall vs coach-severe, coach-moderate recall reported as a limit.
  Acted squatting now spans 3 levels.
- Pitch: the guard is posture-aware; hip sink starts at moderate (`recorded_only` mild) until G2(e)
  certifies pitch, replacing the unstated "widened bands".
- `depth_target_met` default explained; `LAGGED` also shifts `_cue_outcome`; tempo noise restated
  (3-sample median, ±0.03–0.06 s).
- `up` replaces `gravity`; `is_ready` is ANDed with the standing gate; `counter_baseline` is not used.

**Round 4 (2026-10-03).** Scorer 8.8.
- `pitch_certified` comes from the measured mount (rack mount, or inclinometer ≤ 2° entered at setup),
  not the estimator. That makes mild cueable in the demo; uncertified installations keep moderate.
- The estimator is only a posture-aware knock tripwire.
- `_better` praise when hold reps carry recorded-only verdicts.
- §7 step 19 names the spec's recorded-only path; test: 3 recorded-only mild verdicts → no cue.
- `descent_time_s` uses downward crossings in `(argmax S̃, i_low]`, so a piston pre-roll's ascent is
  never read.
- `hedged_recap_faults` gating.
- Cosmetic: priorities are spec §6; the streak is pipeline-owned.

**Round 5 (2026-10-03).** Scorer 8.9.
- Certification now also requires lens height 0.8–1.1 m and ≤ 10 cm below the lifter's hips.
- The spec's delta tripwire (trip at Δ > 3.5° vs the certification reading, bar held) revokes
  certification per session. G2(e) tests 4° knocks and the lens-height sweep.
- The 3 `_better` keys run through drafting, `cues.json` and G7; the spec §4 skip rule is referenced.
- Mount fields and praise rule marked [SP].
- Certified row states the novice error is cued at its pronounced end (~53–55°); worst correct rep
  10.6%.
- Round 5b: the per-session hip-sink tier moves to [SP §4] `cue_start_tier(fault_type, ctx)`. The rule
  emits plain verdicts, the pipeline is the only `recorded_only` stamper, and a mid-session
  revocation test was added.

**Round 6 (2026-10-03).** Scorer 8.9.
- `reportable_faults` [SP §4] adopted: new `fault_filters.py`; orchestrator, `last_rep_verdict` /
  `check_my_form`, `show_me("last_rep")`, progress summaries, persistence aggregations and display
  `fault_summary` routed through it; RDL shared test added.
- Praise follows the spec's hold-window rule.
- Delta tripwire on the ankle→hip axis, with threshold matched to measured noise; G2(e) criteria made
  consistent; session-start re-measurement is the expected procedure for inclinometer mounts.
- `level_camera_frame` by the measured mount angle on certified mounts (V1). Mild stays 0.12, since
  0.10 would need ≤ 0.6° residual.
- Neutral `level_camera` text; requirement 3 from `ctx.body_proportions`; 10 cm lens limit kept and
  justified by knock tolerance.
- Round 6 addendum: leveling depth terms are budgeted (σ_z = 3 cm, |b_z| ≤ 10 cm → +0.4% at 2°).
  Certified worst case is 10.6% and mild holds. The largest certified pitch is derived (2° keeps a ≥ 3°
  knock margin). Knock tolerance drops to 3.1°, so the tripwire becomes T = max(2.5°, 2.6σ) with
  σ ≤ 0.45°. Mild 0.10 is unsupported (negative budget).

**Round 7 (2026-10-03, post-pass).**
- Start-tier perspective figures made consistent (0.7% inside the lens checks; 0.4% leveling bias).
- Uncertified installs need the lens checks for moderate; otherwise severe via `cue_start_tier`, since a
  0.5 m lens at 8° reads 15.1%.
- Tripwire averaged over ≥ 36 idle setup frames (σ ≈ 0.9–1.0°, T = 2.5°, detection ≥ 3.7°); residual
  undetected band 3.1–3.7° stated.
- Round 8 (final cleanup): tripwire reference pooled over the first 3 sets, giving σ_Δ ≈ 1.1° at
  T = 2.5° with ~2–3% false trips per set, ~90–91% detection at 4°, and a residual band of 3.1–3.9°;
  G2(e) aligned. The stale "uncertified = moderate" lines (§1, bands note, R2, Phase E) now include the
  severe fallback. Praise worst case is 13.5%.
