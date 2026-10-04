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
  - Without a bar axis, the lifter's left-right is locked from the frames at the bar (hips,
    ankles and wrists together; §3, "The lifter's left-right"). Rebuilt from the noisy hip
    line every frame, it turned forward travel into sideways hip shift.

## 2. Phases (`deadlift/analyzer.py`)

The bar drives the rep. The rep signal is the bar's height above its resting height.

| Phase | Means | Leaves when |
|---|---|---|
| APPROACH | Away from the bar, or walking in | **STANCE:** standing (knee flexion and trunk both within 20° of straight), settled, near the bar (≤ 40 cm ahead of the midfoot, ≤ 10 cm behind it, ≤ 30 cm sideways) and facing it (feet within 60° of square to the bar) for 0.5 s. **SETUP:** hands on the bar for 0.3 s. **PULL:** the bar leaves its rest in the lifter's hands |
| STANCE | Standing at the bar, feet planted. The closed-loop foot guidance runs here | **SETUP:** hands on the bar for 0.3 s. **PULL:** liftoff. **APPROACH:** away or turned away for 0.5 s |
| SETUP | Hinged, hands on the bar | **PULL:** liftoff. **STANCE/APPROACH:** stands up without the bar for 0.3 s |
| PULL | Liftoff to the top | **TOP:** see below. **LOWER:** the bar falls out of a peak that reached the top (as from TOP). **FLOOR:** a failed rep |
| TOP | Holding the top | **LOWER:** the bar falls faster than 0.10 m/s and its median over the last 5 frames is below the hold by the hold band; the lowering is dated back to the bar's last frame at the top. **PULL:** the bar rises clear of the top it held (a stall, not a top) |
| LOWER | Lowering | **FLOOR:** dead stop, and the rep is counted. **PULL:** touch-and-go, and the rep is counted; or the bar rises clear of the top it held (a stall, not a top) |
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
  - The 8 cm margin lets soft lockouts count; D6 then judges them. A trunk 10° or more
    behind vertical with straight legs is an over-extended lockout and counts at any bar
    height (D5 judges it). Straight: the knees bent no more than 40°, or, the knees hidden,
    the hips no more than 6 cm closer to the ankles than standing (leaning back pushes the
    hips forward and down but leaves the legs' length). Leaning back from mid-thigh with the
    knees bent ~60° (the leg ~12 cm shorter) is a failed pull. A lockout short by 30° or more
    is a failed rep.
  - **The hold:** frames in TOP with the bar within the hold band of its top height, above
    as below: 2 cm, or two noise bands on a noisier bar (the wrist proxy's are 3–11 cm). A
    shrug lifting the bar off the lockout is not the hold.
  - **A stall is not a top.** A bar that stalls short of lockout (a grind), or anywhere with
    no standing reference to expect the top at (a hitch), can read as a top. The pull resumes
    when the hips or knees held at least 8° short of standing (D6's mild threshold) or,
    without a standing reference, the hips and the knees have both extended that much since
    the hold (over the last 5 frames), and the bar rises above the top it held by the hold
    band. A shrug or a lockout settling upward rises on straight legs, so it does not resume
    the pull; with the knees hidden nothing tells the two apart, and the pull never resumes
    (a stall there stays the top). One frame, against the whole band: a median clearing a
    smaller band resumed soft lockouts on the noise.
  - A grind closer to the lockout than the hold band stays in TOP, its stall inside the hold.
    The features time its top and judge its lockout from the bar's last climb:
    - **Judging the lockout:** D6 and D5 are read on the hold's last 0.5 s before the
      lowering (a top that never held: the 0.1 s either side of its peak, on the wrist proxy
      the vertex of a parabola through the bar's height within 0.3 s of its highest frame,
      which is only the highest of the wrists' noise), chosen by time and never by the angles
      judged (a window chosen as the most
      extended selects the noise: a 14° soft lockout read ~4° straighter at 2 cm noise).
      After a grind the hold ends locked out.
    - **The final climb** begins at the end of the last stall: the bar's 5-frame running
      median flat (within the event band) for 0.3 s, below the lockout's level by more than
      the band, with the hips and knees together at least 12° more bent than at the lockout.
      A noise dip is not flat for that long, a slow approach rises, and a shrug lifts the bar
      off a lockout as straight as the rest of it.
    - **The lockout's level** is the window's lowest plateau: the median of its frames whose
      5-frame running median sits within the event band of the lowest. A shrug inside the
      hold band only adds frames above it, which a plain median followed up (a 1.5 cm shrug
      dated tops up to 0.6 s late).
    - **The top** is the arrival at that level on that climb, fitted (§2, Events) over the
      climb's upper half and the hold's first 0.2 s, from below: a climb out of a stall starts
      flat, a frame above the level has arrived, and more of the hold would outweigh the few
      frames of a short last climb. A dropped bar frame is no height; a bar lost as it
      arrives may have arrived in the gap, so the fit may date it from the gap's first frame,
      and with no fit the top is the gap's middle.
    - **On the wrist proxy** the wrists' noise hides a stall a few cm short, and a slow
      pull's last centimetres, inside the hold. The top there is the hips and knees reaching
      the lockout: their summed flexion (5-frame running median) within 5° of the lockout's,
      from the bar's arrival at its level. It is searched no later than 0.5 s after the bar's
      own top, or after the joints' last frame still 20° more bent than the lockout before
      the lockout window, so the angles' slow wander over a long hold does not move it.
  - Tested (`scripts/tools/deadlift_envelope.py tops` and `shrug`; `IMPLEMENTATION.md` has
    the rows): tracked grinds 4–6 % short are timed within one frame, one finishing its last
    2.5 cm over 0.3 s within 0.1 s, over 1 s with a median of 0.07 s (its last 0.17 s move the
    bar under 2 mm, inside its noise). Wrist-proxy tops: §7. A stall about 1 cm short, with
    the knees inside D6's mild threshold of standing, is the top on the tracked bar and is
    dated at the stall, with no lockout cue. With a standing reference, a 1.5–4 cm shrug moves
    no top more than 0.18 s, the knees seen or hidden, and reads no velocity loss; without
    one, see §7.
- **Dead stop:** within 2 cm of the rest, and either still for 3 frames or there for 0.3 s.
- **Touch-and-go:**
  - The low point is within 5 cm of the rest.
  - The bar then rises more than 3 cm plus the noise band above that low point, for at least
    0.1 s, with the hands on the bar, and is rising: its slope over the frames since the low
    point is over 0.10 m/s.
  - Not "rising at every step since the low point": one noisy frame would defeat that, and
    with no dead stop in between every rep after it would be lost. And not a fixed window:
    on the proxy the noise widens the velocity window to 0.4 s, which on a fast pull reached
    back into the descent and never read rising.
  - On the wrist proxy the hands count at the low point: "wrists below the knees" stops being
    true within about 0.2 s of a fast pull.
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
  in one or two frames, and then the detecting frame is the event. On the wrist proxy the
  top is the hips and knees reaching the lockout instead (§2, Top).
- Knee pass is the first frame with the bar centre at or above the knee midpoint's height.

## 3. Metrics (`DeadliftRepFeatures`)

NaN means "not measured", never 0.

**Setup windows:**
- 0.3 s before liftoff after a visible setup.
- 0.2 s for a quick re-pull or a grip-and-rip.
- None for a touch-and-go.
- Hip height, trunk and the setup model need frames with the legs measured. Bar over midfoot,
  shoulders vs bar and the grip offset need only the bar (measured on that frame, not carried,
  not predicted), the shoulders and the midfoot locked at the stance, so feet hidden by the
  plates at setup still leave D1 and D7 judged.

| Feature | Frame and sign | Window | Rule |
|---|---|---|---|
| `bar_midfoot_stance_cm` | bar centre ahead of the live midfoot, cm (> 0 = bar too far out) | last 1 s of settled STANCE | context |
| `bar_midfoot_setup_cm` | bar centre ahead of the locked midfoot, cm | setup window | D1 |
| `shoulder_vs_bar_cm` | shoulder midpoint ahead of the bar centre, cm (< 0 = behind) | setup window | D7 |
| `setup_hip_height_cm` | hip midpoint above the ankle midpoint, cm | setup window | D4 |
| `setup_hip_band_low/high_cm` | the setup model's band (§5) | setup window | D4 |
| `setup_trunk_deg` | trunk angle, deg forward of vertical | setup window | context |
| `trunk_change_liftoff_knee_deg` | (trunk at knee pass − trunk at liftoff) − the model's predicted change; the raw change when there is no model. Liftoff: median over 0.2 s before it; knee pass: a parabola fitted through the 0.2 s leading to it (a line with 4–5 frames). A line read a 0.6 s pull's accelerating rise up to 0.12 low in rise ratio; the parabola reads within 0.04 at 0.6–3 s pulls, with no more noise | liftoff → knee pass | D2 size |
| `set_bar_tilt_cm` | median left-above-right tilt over the set's last 3 reps, this one included, signed, cm; NaN until two | set | D8b gate (wrist proxy) |
| `hip_shoulder_rise_ratio` | hip rise ÷ shoulder rise over the same window, same estimates (> 1 = the hips out-rose the chest) | liftoff → knee pass | D2 gate |
| `set_rise_ratio` | median `hip_shoulder_rise_ratio` over the set's last 3 reps, this one included; NaN until two | set | D2 gate |
| `bar_drift_cm` | p90 of the bar centre's forward travel from its liftoff position, cm (away from the legs only) | pull | D3 |
| `hip_extension_deficit_deg`, `knee_extension_deficit_deg` | median flexion at the lockout minus standing flexion, deg (> 0 = short of standing) | the hold's last 0.5 s (§2, Top) | D6 |
| `lean_back_deg` | standing trunk minus median trunk at the lockout, deg (> 0 = behind standing) | the hold's last 0.5 s | D5 |
| `hip_shift_ratio` | hip midpoint's sideways position vs the locked midfoot ÷ ankle separation, both along the lifter's own left-right (below), as the median over the second half of the pull minus the median over the first fifth (> 0 = toward the right foot) | pull | D8 |
| `bar_tilt_cm` | p90 of the left hub's height above the right hub, on a 5-frame running median, cm. On the proxy, the hands' height difference carried out to the 1.70 m hub span, as the median from mid-pull through the top hold | pull | D8b |
| `elbow_flexion_deg` | p90 of the sagittal elbow bend minus standing, deg | pull | D9 |
| `concentric_velocity_mps` | bar rise ÷ (top − liftoff), m/s; tracked bar only (NaN on the wrist proxy, so no speed in the recap, the spoken summary or the diagnosis) | pull | D10 |
| `velocity_loss_pct` | loss against the mean of the set's two fastest reps, % | set | D10 |

**References:**
- **Standing reference:** the medians over 1 s of settled standing in APPROACH or STANCE.
  It gives the trunk, hip, knee and elbow angles, the arm length and the wrist height.
  Lockout, lean-back and bent arms are measured against it, which removes each lifter's
  keypoint bias.
- **Midfoot lock:** the mean of up to 15 settled STANCE frames whose feet were measured,
  taken at setup or liftoff. A lifter who never stood still uses the current frame.
  - **Re-setup at the floor:** at the next liftoff (after FLOOR → SETUP, or a quick re-pull)
    the midfoot is re-locked from the last ≤ 15 frames since the dead stop whose feet were
    measured. A lifter cued "bar over midfoot" who shuffles the feet without standing up is
    then judged where they now stand (6 → 1 cm measured 6 → 1 cm). Feet hidden by the plates
    throughout keep the old lock.
- **D8's two axes.** The hips travel ~45 cm forward in a pull, and an axis a degree off
  square to that travel reads ~0.03 of shift. No single line is square to it for every
  lifter: the bar's axis is not on a stance 5–8° off square to the bar (13 of 18 reps
  falsely cued at 8°), the ankle line is not on a staggered stance (a foot 6 cm ahead turns
  it 13°: 10 of 24 cued), the hip line is not on a pelvis turned against the legs or a hip
  keypoint biased front-to-back (4° read −0.09). So the shift is read on two independent
  axes and D8 keeps the smaller reading, none when they disagree on the side: a real shift
  reads on both, a misaligned axis leaks into one.
  - **The hip line before the rep**, summed over the frames at the bar from the last 15 s
    (settled STANCE, SETUP and FLOOR with the hands on the bar): a pelvis twisting during the
    pull cannot turn it. Older frames count only while the heading holds: going back 1.5 s
    at a time, a block whose hip and ankle lines turn more than 6° from the newer frames'
    ends them (a turn in place leaves the hips and shoulders still, so a lifter who stood
    turned toward the device and then squared up reads settled throughout).
  - **The bar's line**: the tracked bar's resting axis; on the wrist proxy, the hands on the
    bar over the rep (a twisting pelvis does not turn the hands), or, with one wrist never
    seen, the feet's line before the rep (not the hip line a second time).
  - Both axes can leak the same way: a stance off square to the bar *and* a hip line turned
    against the legs in the same direction. That case is cued more often than a clean set
    (§7).
  - Without a tracked bar, the same at-bar frames' hip, ankle and wrist lines give the
    proxy's sagittal frame, locked at each liftoff after a setup.

## 4. Thresholds and why

All of these are initial values (`config.py` `FaultsConfig.deadlift_*`). A tier is cued only
at or above its minimum tier (`details.min_tier`).

| # | Fault | Mild / moderate / severe | Min tier | Rationale |
|---|---|---|---|---|
| D1 | Bar not over midfoot | 3 / 5 / 8 cm | mild | The coaching standard is bar over midfoot. 3 cm is about a third of the midfoot zone, and the static measurement should be ≤ 1 cm (§2.9) |
| D7 | Shoulders behind the bar | 2 / 4 / 6 cm | moderate | The shoulders belong slightly ahead (0–6 cm). Behind the bar, the lats cannot keep it close |
| D4 | Hips outside the model band | 4 / 7 / 10 cm | severe | The band is the model's (§5). Severe-only until the model's error, plus the hip-keypoint bias in deep flexion, is measured |
| D2 | Hips shoot | 5 / 8 / 11° | moderate; emitted only when the hips decisively led (rise ratio > 1.05, and the set's > 1.15 or this rep's > 1.30) | See §5 |
| D3 | Bar drift | 3 / 5 / 8 cm | moderate | 3 cm forward is where coaches see the bar leave the legs. The moving bar's error is 1–2 cm |
| D6 | Incomplete lockout | 8 / 12 / 20° | moderate | Against the lifter's own standing angles, so it is not anatomy |
| D5 | Lean-back | 8 / 12 / 18° | moderate | The same reference. Gravity-sensitive: moderate on the body vertical |
| D8 | Hip shift | 0.10 / 0.15 / 0.22 of ankle separation | moderate | 0.10 is ~3 cm at a hip-width stance |
| D8b | Bar tilt | 3 / 5 / 7 cm hub to hub | moderate (severe on the wrist proxy) | ×1.5 on the wrist proxy, where the hands' noise is carried ~3× out to the hubs. There it reads the smaller of this rep's tilt and the median of the set's last 3 reps (`set_bar_tilt_cm`, signed), to the same side, and nothing on a set's first rep: the noise tilts each rep its own way, an uneven pull repeats |
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
    the trunk change within 2°, the rise ratio within 0.05 and the setup hip height within
    1 cm, on four body types.
- **Not checked:** that real good lifters move the way the model says. That needs J6 data
  (§8.4 instruments).
- **Consequences:**
  - D2's and D4's sizes, and therefore their thresholds, are model-derived.
  - A lifter who holds their back angle from the floor to the knees reads 6–12° against the
    model. Hips and chest then rise together (rise ratio ~1.0), which coaches teach, so it
    must never be cued "Chest and hips together".
  - So the model only sizes D2; whether there is a D2 at all is the rep's model-free
    evidence: the hips must have out-risen the shoulders, decisively. One rep's ratio scatters
    by ~0.1 at 2 cm of Kalman-correlated keypoint noise, so the rep must read > 1.05 and the
    set's last reps > 1.15 (or the rep alone > 1.30). A held back angle is then cued on 0 of
    90 reps at 2 cm i.i.d. noise and 4 of 90 at 2 cm correlated noise (simulator); a
    hips-first pull (ratio ~1.24) from the set's second rep on, on 60–87 % of reps.

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
- The closed-loop foot guidance runs in STANCE on the tracked bar only, while the lifter
  stands settled at the bar before the set's first rep, from a 1 s median of the offset
  (once 10 settled frames are in). It arms past 3 cm (D1's mild threshold) and then guides
  down to 2 cm, so keypoint noise on a bar placed right does not start it. The wrist proxy
  says nothing about where the bar sits on the floor.

## 7. Known limits (measured on the simulator)

The tested envelope is in `docs/deadlift/IMPLEMENTATION.md`, "Measured on the simulator".
Outside it:
- **Grip-and-rip on the wrist proxy:** the first rep is lost. With no tracked bar there is no
  rest height until the wrists hold still at the bottom, so the lifter needs a short pause
  (≈ 0.2 s) there.
- **Standing up off the bar on the proxy counts as a rep.** To the wrists, standing up from
  the setup without lifting the bar (a re-setup), or out of a hinge with the hands below the
  knees (loading plates), is a pull: 5 reps counted for 3 with two re-setups. The tracked bar
  does not move, so it counts 3.
- **A lifter who turns in place by less than ~6°** between reps is not re-locked: the
  heading scan cannot tell so small a turn from noise. Its leak is at most ~0.18 of shift
  per 6°; turning 10° at the floor cued D8 on 0–3 of 90 reps at 1.5 cm AR(0.8).

The numbers below are printed by `scripts/tools/deadlift_envelope.py <sweep>` (noise drawn
independently per body and seed; five bodies × seeds 0–5 unless stated), at AR(0.8) keypoint
noise.

- **D8's recall** [hip_shift]: the smaller of two readings is biased low under noise. At
  2 cm, a shift of 0.15 of ankle separation is cued on 33 of 90 tracked reps (median reading
  0.13) and 20 of 90 proxy reps (0.09); 0.20 on 58 and 35 (0.17 and 0.13); 0.25 on 75 and 57
  (0.22 and 0.18). A pelvis turning ±10° with a 0.20 shift: 58 / 58 tracked, 42 / 39 proxy.
  Clean reps: 1 tracked and 2 proxy of 90 at 2 cm (3 and 0 at 2.5 cm), and 0–2 with a stance
  off square to the bar, staggered, turned and squared up, or a pelvis turned or twisting.
- **D8 when both axes leak the same way** [hip_shift]: a stance 6° off square to the bar
  whose hip keypoints' line stays square to the bar puts both axes 6° off the hips' travel.
  It reads 0.13 of shift noise-free (mild, not cued) and is cued on 25 of 90 tracked reps and
  34 of 90 proxy reps at 2 cm. No consensus of two lines can tell it from a shift; the legs'
  own line would, but it is the one a staggered stance turns.
- **Event timing on the wrist proxy** [tops] is not held to the 100 ms gate. Its top is the
  hips and knees reaching the lockout. At 1.5 / 2 cm: plain reps median 67 ms (beyond 0.3 s:
  1 / 2 of 90, max 0.5 s); grinds 3 cm short median 33 ms (0 / 3 of 90 beyond 0.3 s, max
  1.0 s early at 2 cm); a 12° soft lockout held 2 s median 67 ms (4 / 6 of 90 beyond 0.3 s,
  0 / 3 beyond 0.5 s, max 1.5 s late at 2 cm, where the angles' slow wander reads the hold
  ~20° more bent than its end); slow 5 s pulls are dated early, median 0.23 s (27–28 of 90
  beyond 0.3 s, max 0.73 s), the joints creeping their last few degrees. Noise-free, proxy
  events read up to ~0.1 s early. With the knees hidden at the top, the bar's own fit dates
  it (a review measured −0.2 to +0.57 s at 2 cm).
- **A proxy rep with no pause at the top** [no_pause] (five bodies × seeds 0–7) is judged on
  the 0.1 s either side of its peak, a few frames of noisy angles. At 2.4–2.5 cm, the top of
  the platform's documented noise, fast touch-and-go reps (0.9 s pulls) with 0 s tops draw a
  false cue on 28–30 of 200 (mostly D6 and D8), over `VALIDATION.md`'s 1 in 10; dead stops
  with 0–0.1 s holds on 7–13 of 120. At 2 cm: 0–10 per row.
- **A stall with the knees hidden is the top.** Nothing then tells it from a shrug (both hold
  the hips), so the pull does not resume; its top is dated between the stall and the
  lockout (a review measured −0.7 to +0.3 s from the lockout on 8–10 % stalls), with no
  false cue.
- **A shrug before the top registers** [shrug] (in its first frames, before the bar has read
  still for 3 frames) becomes the top when there is no standing reference: 3–8 of 48 reps
  +0.83–0.87 s late with every rep shrugged, and the recap then reads a speed loss (D10) in
  3–7 of 16 sets. With a standing reference: none.
- **The bar lost at a touch-and-go low point:** 0.3 s of contiguous loss there loses reps
  (35 of 45 counted), 0.5 s most of them (review, round 7); random dropouts do not.
- **Back rounding:** proxies only (PLAN.md §2.10).
