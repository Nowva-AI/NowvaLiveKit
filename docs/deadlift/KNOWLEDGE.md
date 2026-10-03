# Conventional deadlift: static knowledge

PLAN.md §8.1 (J1). This is the reference a coach reviews: the phases, every metric's frame,
sign and window, the thresholds and why they are what they are, the cue text, and the setup
model. It describes the code on `claude/deadlift-v1-impl`. When the two disagree, the code is
right and this file is out of date.

**Review status: not reviewed yet.** It needs sign-off from Ambaka and, ideally, from an
external strength coach. Every threshold below is an initial value; J6 resets them from real
lifts (`docs/deadlift/VALIDATION.md`).

## 1. Frame, signs, units

- **World frame:** Y-down metres (`.claude/rules/biomechanics.md`).
- **Deadlift sagittal frame** (`deadlift/frame.py`), rebuilt on every frame:
  - **up** is measured gravity: the ChArUco board laid flat, mapped into the current world
    frame through the calibration (`deadlift/gravity.py`). Without it, up is the body
    vertical `WORLD_UP`, and D2, D3 and D5 are then cued only from moderate.
  - **lateral** runs from the lifter's left to their right. It is the tracked bar's axis (or
    the resting bar's axis), checked against the hip line.
  - **forward** runs heel to toe.
- **Signs:**
  - Heights are positive up.
  - Forward offsets are positive toward the toes.
  - Segment angles are signed from up, positive when the upper end is ahead of the lower one:
    a trunk leaning forward over the bar is positive, and so is a shin tilted forward.
  - Lateral offsets are positive toward the lifter's right.
- **Midfoot:** ankle + 0.35 × (ankle → toe), averaged over both feet (the squat's
  convention).
- **Bar:**
  - On a tracked bar, the bar is the centre of the two plate hubs. The lifter's left hub is
    decided in the lifter's own frame, not by the world's X, so an unanchored calibration
    cannot swap D8b's side.
  - Without bar tracking, the **wrist proxy** stands in: mid-wrist minus the wrist-to-bar
    offset. The offset is 7.5 cm by default, and is learned per lifter (clipped to 4–12 cm)
    from any set where the bar was tracked.
  - A set never mixes the two sources.

## 2. Phases (`deadlift/analyzer.py`)

The bar drives the rep. The rep signal is the bar's height above its resting height.

| Phase | Means | Leaves when |
|---|---|---|
| APPROACH | Away from the bar, or walking in | **STANCE:** standing (knee flexion and trunk both within 20° of straight), settled, near the bar (≤ 40 cm ahead of the midfoot, ≤ 10 cm behind it, ≤ 30 cm sideways) and facing it (feet within 60° of square to the bar) for 0.5 s. **SETUP:** hands on the bar for 0.3 s. **PULL:** the bar leaves its rest in the lifter's hands |
| STANCE | Standing at the bar, feet planted. The closed-loop foot guidance runs here | **SETUP:** hands on the bar for 0.3 s. **PULL:** liftoff. **APPROACH:** away or turned away for 0.5 s |
| SETUP | Hinged, hands on the bar | **PULL:** liftoff. **STANCE/APPROACH:** stands up without the bar for 0.3 s |
| PULL | Liftoff to the top | **TOP:** see below. **LOWER:** the bar falls from a peak that reached the top. **FLOOR:** a failed rep |
| TOP | Holding the top | **LOWER:** the bar falls faster than 0.10 m/s |
| LOWER | Lowering | **FLOOR:** dead stop, and the rep is counted. **PULL:** touch-and-go, and the rep is counted |
| FLOOR | Bar back on the floor | **SETUP:** hands on the still bar for 0.3 s. **PULL:** quick re-pull. **STANCE/APPROACH:** stands up |

**What each condition means:**
- **Settled:** the hips and shoulders moved less than 0.20 m/s. Speed is the displacement
  between the median positions of the first and last thirds of a 0.5 s window. Keypoint noise
  of 2 cm, even when correlated by the pipeline's Kalman, reads well under that, while walking
  reads 0.5–1.5 m/s.
- **Hands on the bar:**
  - With a tracked bar: both wrists between 6 cm below and 16 cm above the bar centre, and
    within 15 cm of it front to back. The wrist joint sits ~7.5 cm above the bar centre, which
    leaves room for 2 cm of noise either way.
  - On the wrist proxy: a hinge of at least 30° with both wrists below the knees.
- **Holds:** a hold survives a lapse of up to 0.15 s, such as one noisy frame or one dropped
  keypoint.
- **Liftoff:** the bar more than 3 cm above its rest and rising faster than 0.10 m/s, with the
  hands on it. This is checked from every phase outside a rep, so a grip-and-rip start with no
  setup pause still counts.
- **Rest:**
  - The median of the last 30 still bar heights (1 s), taken outside a rep.
  - On the tracked bar: from APPROACH, STANCE, SETUP or FLOOR, but never while hands hold the
    bar off the floor outside a setup.
  - On the wrist proxy: while hinged with hands on the bar.
  - "Still" means a slope under 0.02 m/s, or under 2.5 standard errors of the bar's own noise.
    The proxy's wrists jitter far more than a tracked bar.
- **Top:** the bar rose at least 10 cm, reached the expected top height minus 8 cm, and held
  still (under 0.05 m/s for 3 frames) with the trunk within 35° of vertical.
  - The expected top height is the standing mid-wrist height minus the wrist-to-bar offset.
    Without a standing reference, it is the median top of the set's earlier reps.
  - The 8 cm margin lets soft and over-extended lockouts count; D6 and D5 then judge them.
- **Dead stop:** within 2 cm of the rest, and either still for 3 frames or there for 0.3 s.
- **Touch-and-go:**
  - The low point is within 5 cm of the rest.
  - The bar then rises more than 3 cm above that low point, rising the whole way, for at least
    0.1 s, with the hands on the bar.
  - The rep ending at the low point is counted, and the low point is the next rep's liftoff.
  - A bumper bounce (≤ 2–3 cm, hands off) is not a touch-and-go.
- **Failed rep:** the bar rose at least 10 cm and came back to the floor without a top. It is
  an event, not a rep.
- **Gaps:**
  - A bar missing for up to 0.2 s is a tracking gap, not a lost bar. The last bar keeps its
    geometry (hands on the bar, facing) but gives no height.
  - Feet the plates hide keep their last measured position once the lifter is at the bar,
    since they do not move during a set.

**Events** (`deadlift/features.py`):
- Liftoff, top arrival and touchdown are each fitted as the change point of a parabola: the
  bar leaving (or reaching) a level from rest is c·(t − t₀)² on one side and flat on the
  other. The fit uses the frames within 4 cm of the level, and needs at least 3 frames clear
  of the noise band (3 × the rest's noise, never under 0.5 cm). A fast bar crosses that band
  in one or two frames, and then the detecting frame is the event.
- Knee pass is the first frame with the bar centre at or above the knee midpoint's height.

## 3. Metrics (`DeadliftRepFeatures`)

NaN means "not measured", never 0.

**Setup windows:**
- 0.3 s before liftoff after a visible setup.
- 0.2 s for a quick re-pull or a grip-and-rip.
- None for a touch-and-go.
- Only frames with the legs measured count, and setup bar values come only from bar states
  measured on that frame (not carried, not predicted).

| Feature | Frame and sign | Window | Rule |
|---|---|---|---|
| `bar_midfoot_stance_cm` | bar centre ahead of the live midfoot, cm (> 0 = bar too far out) | last 0.3 s of STANCE | context |
| `bar_midfoot_setup_cm` | bar centre ahead of the locked midfoot, cm | setup window | D1 |
| `shoulder_vs_bar_cm` | shoulder midpoint ahead of the bar centre, cm (< 0 = behind) | setup window | D7 |
| `setup_hip_height_cm` | hip midpoint above the ankle midpoint, cm | setup window | D4 |
| `setup_hip_band_low/high_cm` | the setup model's band (§5) | setup window | D4 |
| `setup_trunk_deg` | trunk angle, deg forward of vertical | setup window | context |
| `trunk_change_liftoff_knee_deg` | (trunk at knee pass − trunk at liftoff) − the model's predicted change; the raw change when there is no model | liftoff → knee pass | D2 |
| `hip_shoulder_rise_ratio` | hip rise ÷ shoulder rise (> 1 = the hips out-rose the chest) | liftoff → knee pass | D2 cross-check |
| `bar_drift_cm` | p90 of the bar centre's forward travel from its liftoff position, cm (away from the legs only) | pull | D3 |
| `hip_extension_deficit_deg`, `knee_extension_deficit_deg` | median flexion at the top minus standing flexion, deg (> 0 = short of standing) | top frames | D6 |
| `lean_back_deg` | standing trunk minus median trunk at the top, deg (> 0 = behind standing) | top frames | D5 |
| `hip_shift_ratio` | hip midpoint's sideways position vs the locked midfoot ÷ ankle separation, as the median over the second half of the pull minus the median over the first fifth (> 0 = toward the right foot) | pull | D8 |
| `bar_tilt_cm` | p90 of the left hub's height above the right hub, on a 5-frame running median, cm. On the proxy, the hands' height difference is carried out to the 1.70 m hub span | pull | D8b |
| `elbow_flexion_deg` | p90 of the sagittal elbow bend minus standing, deg | pull | D9 |
| `concentric_velocity_mps` | bar rise ÷ (top − liftoff), m/s | pull | D10 |
| `velocity_loss_pct` | loss against the mean of the set's two fastest reps, % | set | D10 |

**References:**
- **Standing reference:** the medians over 1 s of settled standing in APPROACH or STANCE.
  It gives the trunk, hip, knee and elbow angles, the arm length and the wrist height.
  Lockout, lean-back and bent arms are measured against it, which removes each lifter's
  keypoint bias.
- **Midfoot lock:** the mean of up to 15 settled STANCE frames whose feet were measured,
  taken at setup or liftoff. A lifter who never stood still uses the current frame.

## 4. Thresholds and why

All of these are initial values (`config.py` `FaultsConfig.deadlift_*`). A tier is cued only
at or above its minimum tier (`details.min_tier`).

| # | Fault | Mild / moderate / severe | Min tier | Rationale |
|---|---|---|---|---|
| D1 | Bar not over midfoot | 3 / 5 / 8 cm | mild | The coaching standard is bar over midfoot. 3 cm is about a third of the midfoot zone, and the static measurement should be ≤ 1 cm (§2.9) |
| D7 | Shoulders behind the bar | 2 / 4 / 6 cm | moderate | The shoulders belong slightly ahead (0–6 cm). Behind the bar, the lats cannot keep it close |
| D4 | Hips outside the model band | 4 / 7 / 10 cm | severe | The band is the model's (§5). Severe-only until the model's error, plus the hip-keypoint bias in deep flexion, is measured |
| D2 | Hips shoot | 5 / 8 / 11° | moderate when the rise ratio is > 1, else severe | See §5 |
| D3 | Bar drift | 3 / 5 / 8 cm | moderate | 3 cm forward is where coaches see the bar leave the legs. The moving bar's error is 1–2 cm |
| D6 | Incomplete lockout | 8 / 12 / 20° | moderate | Against the lifter's own standing angles, so it is not anatomy |
| D5 | Lean-back | 8 / 12 / 18° | moderate | The same reference. Gravity-sensitive: moderate on the body vertical |
| D8 | Hip shift | 0.10 / 0.15 / 0.22 of ankle separation | moderate | 0.10 is ~3 cm at a hip-width stance |
| D8b | Bar tilt | 3 / 5 / 7 cm hub to hub | moderate | ×1.5 on the wrist proxy |
| D9 | Bent arms | 15 / 25 / 35° | mild | Elbows should stay long. Measured against standing |
| D10 | Velocity loss | 20 / 30 / 40 % | recap | Sánchez-Medina 2011. Fatigue is load advice, never a mid-set cue. Not emitted on the wrist proxy |

**Raised minimum tiers:**
- From the wrist proxy, D1, D3 and D8b widen ×1.5 and are cued only from moderate.
- From the body vertical, D2, D3 and D5 are cued only from moderate.

## 5. The setup model (`deadlift/setup_model.py`, PLAN.md §2.7)

A planar linkage with the ankle at the origin. Its inputs are the lifter's own projected
tibia, femur and torso lengths and arm length (from their setup frames), the measured bar
position, the learned wrist-to-bar offset, and the bar-to-shin contact distance (measured per
lifter, clipped to 2–10 cm).

- **Solve 1 (setup):**
  - Shins touch the bar, arms are straight from the grip, and the shoulder joint sits
    `shoulder_ahead` in front of the bar.
  - Of the two shin-contact solutions it keeps the one with the knee forward and knee flexion
    between 40° and 120°.
  - The D4 band is the hip heights for shoulder offsets of 0 and 6 cm.
- **Solve 2 (knee pass):** the bar at knee height and the shins near vertical.
- **D2's expected change** is solve 2's trunk minus solve 1's. For typical bodies the trunk
  comes 8–20° more upright between the floor and the knees.
- **No solution:** when no setup fits the measured lengths, D4 and D2 are not model-judged
  for that lifter, and this is logged.

**What is and is not validated:**
- **Checked (simulator):**
  - The model's solves meet their own constraints when rebuilt with independent forward
    kinematics: the shins sit at the contact distance and knee flexion is in range.
  - On poses scripted by trunk angle rather than built by the model, the analyser measures
    the trunk change within 2°, the rise ratio within 0.03 and the setup hip height within
    1 cm, on four body types.
- **Not checked:** that real good lifters move the way the model says. That needs J6 data
  (§8.4 instruments).
- **Consequences:**
  - D2's and D4's sizes, and therefore their thresholds, are model-derived.
  - A lifter who holds their back angle from the floor to the knees reads 8–10° against the
    model. For average proportions this needs the knees nearly locked by the knee pass, which
    is a stiff-legged first pull. For a long torso it is reachable with bent knees.
  - So, until J6, D2 is cued below severe only when the rep's own model-free evidence agrees:
    the hips rose faster than the shoulders (rise ratio > 1).

## 6. Cue text (`src/assets/cue_text/cues.json`, `coaching_constants.py`)

External focus, ≤ 4 words, nothing medical, no "flat back" claim.

| Key | Spoken | After a fix |
|---|---|---|
| `deadlift_bar_midfoot` | "Bar over midfoot!" | "Good, bar over midfoot." |
| `deadlift_hips_up` / `_down` | "Hips a bit higher!" / "Hips a bit lower!" | "Good, hips set higher." / "…lower." |
| `deadlift_shoulders_over` | "Shoulders over the bar!" | "Good, shoulders over the bar." |
| `deadlift_chest_with_hips` | "Chest and hips together!" | "Better, chest and hips together." |
| `deadlift_bar_close` | "Keep the bar close!" | "Good, bar stayed close." |
| `deadlift_lockout` | "Stand tall!" | "Good, all the way up." |
| `deadlift_finish_neutral` | "Stand tall, no lean!" | "Good, tall with no lean." |
| `deadlift_even_feet` (+ `_left` / `_right`) | "Push evenly!" ("Drifting left, push evenly!") | "Good, staying centered." |
| `deadlift_level_bar` | "Keep the bar level!" | "Good, bar stayed level." |
| `deadlift_long_arms` | "Long arms!" | "Good, arms stayed long." |
| `deadlift_step_closer` / `deadlift_closer` / `deadlift_back` | "Step closer." / "A bit closer." / "Back a little." | `adjust_good`: "Right there — hold that." |

**When cues are spoken:**
- Only after a dead stop, or while standing at the bar. Never between touch-and-go reps.
- The closed-loop foot guidance runs in STANCE on the tracked bar only. The wrist proxy says
  nothing about where the bar sits on the floor.

## 7. Known limits (measured on the simulator)

The tested envelope is in `docs/deadlift/IMPLEMENTATION.md`, "Measured on the simulator".
Outside it:
- **Grip-and-rip on the wrist proxy:** the first rep is lost. With no tracked bar there is no
  rest height until the wrists hold still at the bottom, so the lifter needs a short pause
  (≈ 0.2 s) there.
- **Rising from a hinge on the proxy:** someone standing up out of a hinge with the hands
  below the knees (loading plates) looks like a pull to the wrist proxy. The tracked bar does
  not move, so it does not.
- **Back rounding:** proxies only (PLAN.md §2.10).
