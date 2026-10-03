# Squat Pipeline V1: Faults, Rules and Diagnosis Graph Audit

Audit date: 2026-10-01 (Opus 5.5). Scope: everything that decides what the athlete is told
about their squat. That covers the six squat fault rules, the squat profile and its calibration,
the diagnosis knowledge graph (`symptoms.yaml`, `causes.yaml`, evidence tests, parameter deltas,
engine), rep scoring, and the path from a fault to the athlete's ears.

The question each item answers: "would a world-class strength coach agree with what we detect,
what we say caused it, and what we tell the athlete to do?"

How it was verified:
- **Read-through** of every file named here.
- **Synthetic poses** in the production Y-down frame (188.5 cm proportions: torso 0.543 m,
  femur 0.462 m, tibia 0.464 m), pushed through the real IK and valgus code.
- **Replay over recorded user test runs** (see "Real-data check").
- **End-to-end trace** of the fault → cue → LLM-recap path.

"Verified" below means numerically reproduced. Everything else is from reading the code.

## Bottom line

1. **Several current faults are wrong at the measurement level.** This is a fact about the
   code, not a judgement call. Examples:
   - The live rep scorer reads Y-down data as Y-up.
   - Calibration turns each athlete's faults into their personal "normal".
   - "Parallel" is defined three ways, ~20 cm apart.
   - The expected trunk lean is ~12° for real anthropometry, so every textbook squat reads as
     "excessive lean".

   Part A must land before any new fault is worth adding. New faults built on these
   measurements inherit the errors.
2. **The detectors look at the wrong moments.** What a coach watches most is what happens
   coming *out of the hole*: hips shooting up, knees caving on the way up, hips shifting, the
   bar slowing down. Today valgus is only judged in the `bottom` phase, lean is judged as an
   absolute angle, and nothing looks at the ascent.
3. **The knowledge graph is the right shape but too thin, and contains coaching errors a
   professional would catch.**
   - It has 4 symptoms and 13 causes.
   - Copenhagen planks are prescribed for *abductor* weakness, but they train the adductors.
   - Hip-flexor work is prescribed for limited hip *flexion*.
   - "Foot arch collapse" is inferred from toe-out angle.
   - It misses the best-evidenced link in the literature: limited ankle dorsiflexion → knee
     valgus.
4. **Fault rules and the diagnosis graph are two disconnected detectors.** They measure the
   same faults with different metrics and thresholds and never feed each other. The intra-set
   cue and the post-set recap can contradict each other.
5. **Highest-value additions for the V1 demo loop** (squat → diagnose → cue → improve):
   - hip shoot (good-morning squat)
   - ascent knee valgus
   - balance (bar over midfoot), unified with heel rise
   - hip shift
   - velocity loss, which also gives an honest "you had ~2 reps left" signal and replaces the
     current fatigue proxy

   All are cheap geometry on keypoints the pipeline already has, so they are edge-safe.
6. **Every recorded user run is single-camera, and single-camera side-view angles are mostly
   noise from camera placement.** Per-run trunk-pitch medians range from 7° to 57° (A12).
   - Frontal-plane faults (hip shift, valgus, foot setup) can ship on one camera.
   - Side-view faults (lean, hip shoot, balance, velocity) need the triangulated rig at ≥30 fps.
     Not one triangulated user recording exists yet.

Severity tags: **[blocker]** gives wrong output today. **[coach]** is a coaching-logic or
content problem. **[gap]** is missing coverage. **[hygiene]** is dead code or a smaller issue.

---

## Part A — Measurement correctness (fix first)

### [x] A1. Live rep scorer reads Y-down heights as Y-up  [blocker] (verified)

**Where:**
- `pipeline.py:473-493` `_build_trajectory_sample` stores `hip_y`/`knee_y` as raw
  `kpts[idx][1] * 100`. That is Y-down and hip-centred.
- `rep_scoring.py:91-96` treats `hip − knee` as "hip above knee", which only holds for Y-up.
- The `test_rep_scoring` fixtures are Y-up, so the suite passes.

**What happens:**
- The ratio is −1.0 standing and about −0.26 at the bottom.
- So the "loaded window" (`rep_scoring.py:110-123`) selects the **standing** frames. Trunk,
  knee and symmetry are therefore scored while the athlete stands upright.
- The depth score is always 1.0. A 42° quarter squat scores depth 1.0 with the trajectory and
  0.13 without it.

**Who is affected:** the live per-rep form scores (`session_tracker.py:189`, `:239-240`,
`:299-301`). These feed FORM SCORE / TREND in the recap. `engine.py:44` scores without
trajectories, on a correctly Y-up summary, so the two scoring paths disagree.

**Fix:**
- Negate the heights at the sample boundary, or measure height against the frame's own floor.
- Regenerate the fixtures in the production frame.
- Add one test that pushes a real recorded rep through `pipeline → session_tracker → score`.

### [x] A2. Calibration turns each athlete's faults into their "normal"  [blocker] (verified)

**What a coach would say:** a coach never adopts rep 1's fault as the standard. A personal
baseline is the right tool for spotting *drift* within a set (fatigue). It is the wrong tool
for the absolute standard. Three code paths do it today.

**1. Production calibration — `calibration.py:44-70`, applied by `:105-135`**
- **Valgus:** thresholds are set to `|hip_adduction| peak + 5/10/15`.
  - Hip adduction is measured against world X, so it mostly reflects toe-out (see A6). A
    lifter with 20° toe-out gets thresholds of about 24/29/34.
  - A real 12° knee cave reads 9.5 on the 3D metric. **Valgus can never fire after
    calibration.**
- **Depth:** thresholds are set to the athlete's own average knee flexion minus 10/25/55. A
  lifter who quarter-squats during calibration (~70°) is graded "below parallel" from then on.
- **Forward lean:** thresholds are set to their own peak minus 10/15/20.
- **Symmetry:** thresholds are set to a single-frame max + 5. One noisy frame inflates them for
  the whole session (see L in the convention audit).

**2. One-rep auto-baseline — `squat.py:191-244`**
- Same pattern as above.
- Plus a bug at `squat.py:225-228`: when the toe metric is unavailable, *or* a clean rep reads
  valgus ≤ 0 (normal for 2D with toe-out), it writes hip-adduction values into the **primary**
  thresholds. That takes them from 6/10/16 to about 24/29/34. The fallback thresholds are never
  touched.
- `abs()` at `:160` counts knees-*out* as adduction.
- The per-rep peaks are never reset, so they build up across non-clean reps.

**3. Returning users**
- `pipeline_process.py:1045` applies the stored calibration.
- Then `:1053` `apply_athlete_params` → `forward_lean.py:69-90` `_rebuild_thresholds` resets
  the forward-lean thresholds to the YAML values.
- The same reset can hit new users if body measurement finishes after calibration
  (`pipeline.py:529-530`). This one is inferred, not verified.

**Fix:** split every fault into two channels.
- **Absolute standard:** fixed thresholds, anthropometry-aware where justified (the balance
  model in A4). Never moved by observed reps.
- **Drift:** this rep against the athlete's own best rep this session. This becomes a fatigue
  signal ("your knees started caving from rep 5"), which is exactly how a coach uses a
  personal baseline.

Delete threshold-writing from observed reps.

### [x] A3. "Parallel" has three definitions, ~20 cm apart  [blocker] (verified)

**The geometry:** with a thigh parallel to the floor and a 30° shin, knee flexion is **120°**,
not 90°. 90° of knee flexion leaves the hip **19.5 cm above** the knee.

**Where it goes wrong:**
- `DepthRule` (`depth.py:47-51`, `config.py:80-81`, `config/biomechanics.yaml:37-38`) calls
  90° "parallel" and 100° "below parallel". Reps ~25° above parallel pass as good depth.
- `bridge.classify_depth` (`bridge.py:117-132`) calls 105° class 4 ("below parallel") while the
  hip is 12 cm above the knee. `evidence_tests.test_depth_unfamiliarity` reads this class.
- `engine._extract_feature("depth_deficit")` and `rep_scoring.score_depth` use the correct
  geometric definition: hip joint centre against knee joint centre. On the same rep,
  `depth_limit` fires at severity 1.0 while `DepthRule` says depth is fine.
- **BiLSTM labels:** the BiLSTM gates rep counting on class ≥ 3 ("Parallel"). Its training-label
  code is not in the repo. Check whether those labels came from knee-angle thresholds. If they
  did, the counter inherits the same ~25° error.

**Why this matters:** knee angle cannot define depth. It depends on shin angle, so ankle
mobility and stance change it at the same hip height.

**Fix:**
- Use one depth definition everywhere: hip-joint-centre height minus knee-joint-centre height,
  normalized by femur length. This is the existing `_hip_above_knee_ratio`, once fixed per A1.
- Add a small offset for the hip crease sitting lower than the joint centre (start around
  0.05 femur lengths). Calibrate it on side-view reps labelled by a coach.

### [x] A4. Expected trunk lean is mis-scaled; the real model already exists in the codebase  [blocker] (verified)

**The bug:**
- `parameter_deltas.expected_trunk_lean_geometric` = `30 + (femur_torso_ratio − 1) · 120`
  assumes the ratio sits around 1.0.
- With "torso" = shoulder→hip keypoint distance (`segment_lengths.py:61-75`), real ratios are
  about 0.84 (the repo's own `REFERENCE_FEMUR_TO_TORSO_RATIO`). Expected lean comes out at
  about **11–12°**. Nobody back-squats to parallel at 12°.

**The consequences:**
- A textbook 45° high-bar rep fires `excessive_trunk_lean` at severity 1.0.
- The rep's trunk score is 0.
- `bracing_failure` gets evidence.
- The tier-0 "your femurs are long, this is structural" note (`test_femur_torso_ratio`, which
  needs a ratio above 1.0) almost never fires. It should be one of the most common and most
  reassuring things the coach says.

**What a coach would say:** trunk angle is not a free choice. It is whatever keeps the bar
(or, unloaded, the centre of mass) over midfoot, given the athlete's femur, tibia and torso
lengths and how far their knees travel forward (ankle dorsiflexion). For this athlete:

| Ankle DF (shin tilt) | Bar-over-midfoot lean at parallel | Current formula |
|---|---|---|
| 20° | 42° | 12° |
| 25° | 37° | 12° |
| 30° | 32° | 12° |
| 35° | 28° | 12° |

That range matches the published high-bar figures (35–45°). Low-bar runs 45–60°
(Glassbrook et al. 2017).

**Fix:**
- `keypoint_corrector.py` already has a segment-mass COM model (`SEGMENT_MASS_FRAC`,
  `compute_com_ground`), a midfoot target (`compute_balance_target_ground`) and a balance-lean
  Newton solver (`solve_balance_lean`). Today only the choreographer uses them.
- Use the same model for *detection*: expected lean = the lean that balances this athlete's own
  measured bottom position.
- Lean beyond that is a genuine balance or control fault.
- Lean within it is anatomy plus ankle mobility, and should be explained, not corrected.
- This makes "adapts to their anatomy" literally true, and it gives a pitch-ready story.
- Needs one extra input: bar position (high/low/front/none). Ask once at setup.
- Under heavy load the bar's mass dominates, so balance the **bar** (≈ shoulder midpoint)
  rather than the body COM once load > ~0.5× bodyweight.

### [x] A5. Single-camera FPPA reads normal toe-out as knees-out; valgus can't fire  [blocker] (verified)

**The bias:** `valgus.py:161-188` (2D FPPA) has no correction for foot direction. With knees
tracking over the toes, the "neutral" reading is:

| Toe-out | Neutral FPPA |
|---|---|
| 0° | −4° |
| 10° | −18° |
| 20° | −35° |
| 30° | −52° |

**The consequences:**
- A real 12° cave at 20° toe-out still reads −19.7°, so the 6/10/16 thresholds can never fire.
- `rep_scoring.py:174`, `:191-196` take `abs()`, so a textbook 2D rep scores knee tracking
  **0.0**.

**Fix:**
- Measure deviation from the knees-over-toes line using the 2D foot direction. MediaPipe's
  `foot_index` is available. This is the same construct as the 3D metric, which is correct.
- Make the knee score one-sided: penalize medial deviation, and only penalize lateral deviation
  past a generous tolerance.

### [x] A6. The hip-adduction "valgus" fallback measures foot angle  [blocker] (verified)

**The problem:** `analytical_ik.py:169-203` measures the thigh against world X, not against the
pelvis or the foot.
- 20° of toe-*in* with perfect tracking reads +19.3, which `knee_valgus.py:101-109` would
  report as MODERATE valgus.
- A real cave with toe-out stays negative, so it is missed.
- In 2D mode, `foot_confidence` is built from hip/knee/ankle confidence × a facing score and
  uses **no foot landmarks**. The "toe metric available" gate therefore means nothing there.

**Fix:**
- Drop the fallback. When the feet aren't visible, say nothing about valgus. A coach who
  can't see the knees doesn't guess.
- Alternatively, rebuild it in the pelvis frame relative to foot direction.
- Rename or fix `foot_confidence`.

### [x] A7. Foot direction angle is unsigned  [blocker] (verified)

**The problem:**
- `bridge.compute_foot_direction_angle` (`bridge.py:98-114`) uses `arccos`, so 20° toe-in
  reads exactly like 20° toe-out.
- It feeds `narrow_foot_angle`, `knee_track_cue`, `weak_hip_abductors` and
  `foot_collapse_arch`.
- It returns 0.0 instead of NaN when degenerate.

**Fix:** a signed angle measured from the athlete's sagittal axis (perpendicular to the pelvis
lateral vector), mirrored left/right.

### [x] A8. ROM inputs don't measure what their names say  [blocker]

- **`rom["avg_depth"]` is knee flexion, not hip flexion.** It is the mean of the calibration
  reps' peak knee flexion (`calibration.py:225-228`). It is consumed as **hip** flexion
  capacity (`evidence_tests.py:172-173`, `:218-225`).
- **`rom["peak_dorsiflexion"]` is noisy.** It is a single-frame max over every frame, standing
  included (`calibration.py:238-242`).
- **The `limited_ankle_df` template is circular.** It fills `expected_df` with the athlete's own
  peak, producing sentences like "Ankle dorsiflexion is 25° (need >25°)" (`engine.py:356`).

**Fix:**
- Measure hip-flexion capacity from `hip_flexion_l/r`.
- Take DF as the p95 over the bottom window of clean reps.
- Fill the template from `ANKLE_DF_UNRESTRICTED_DEG`.

### [x] A9. Live YAML thresholds make "forward lean" fire on textbook reps  [blocker] (verified)

**The numbers:**
- `config/biomechanics.yaml:43-46` (145/135/125) overrides `config.py:93-95` (135/125/115).
- In lean-from-vertical terms that is mild 35°, moderate 45°, severe 55°.
- A 45° high-bar rep is therefore MODERATE.

**Why it compounds:** `forward_lean` is cue priority 0 (`cue_cache.py:103-109`), and the cue gap
is shared by every fault. So the athlete hears "chest up" on good reps, and that cue crowds out
valgus. The comment justifying rank 0 ("root-cause fault") rests on mis-scaled lean
measurements (A4, A9).

### [x] A10. NaN handling and presence gates  [hygiene] (verified)

- `classify_depth(NaN)` returns 4, the *best* class.
- `bridge.py:65-68` `max(nan, x)` is order-dependent.
- `pipeline.py:496-499` counts a leg as present at confidence > 0, but IK needs ≥ 0.1. NaN
  angles can therefore reach the bottom frame.

### [x] A11. Stance ratio denominator  [hygiene]

**The problem:** `shoulder_width_m` is the distance between shoulder joint centres
(0.33–0.38 m), not biacromial width (~0.41 m). Ratios therefore read 10–20% wider than
coaching-language "shoulder width".

**Fix:** athletes don't think in ratios anyway. Speak in absolute terms ("move each foot out
about 3 cm") and keep the ratio internal.

---

## Part B — Coaching logic in the existing faults and graph

### [x] B1. Fault rules and diagnosis are two disconnected detectors  [coach]

**How the two paths split:**
- **Fault rules** produce `FaultEvent`s → pre-recorded cue.
- **The diagnosis engine** builds its own `RepKinematicSummary` from a bottom frame and a top
  frame (`bridge.py:156`). It never reads a `FaultEvent`.

**The consequences:**
- Valgus, asymmetry, depth and lean are each measured twice, with different metrics and
  thresholds.
- The athlete can hear "knees out" mid-set and then get a recap that never mentions knees, or
  the reverse.
- A new FaultType does nothing to diagnosis, and a new symptom does nothing to cues.

**Fix:** build one per-rep feature vector from the full rep trajectory: phase-tagged peaks,
bottom values, ascent values, timing and velocities. Both the intra-set verdicts and the
set-level symptoms should read from it. See D1.

### [x] B2. Knee valgus is only judged in the `bottom` phase  [coach]

**Where:** `knee_valgus.py:86` returns unless `phase == "bottom"`.

**What a coach watches:** knees caving **on the way up, out of the hole**, and especially
caving that *grows* as the set gets harder. Valgus that is present from rep 1 is a different
problem: a setup or habit issue that a stance cue fixes. Valgus that shows up at rep 6 is a
load or fatigue signal. A small transient inward wobble that recovers is common in strong
lifters and is debated. Persistent or increasing collapse is the coachable pattern.

**Fix:**
- Judge valgus once per rep over the bottom + ascent window, using a p90 statistic. This is
  the pattern `SymmetryRule` already uses.
- Report the phase of the peak (`bottom` / `ascent`), the side, and whether it is new relative
  to the athlete's best rep this session.

### [x] B3. Forward lean is judged as an absolute angle  [coach]

**What a coach would say:**
- Absolute lean mostly reflects bar position and anatomy (A4). It is rarely cued on its own.
- "Chest up" is the cue for the chest **dropping** while the hips rise out of the hole: the
  hip shoot, or "good-morning squat".
- The generated `chest_up` clips were prompted with exactly that description:
  `generate_cue_audio.py` says "chest is dropping forward and his upper back is starting to
  round as he comes out of the hole".
- So the audio describes a fault the detector doesn't measure. It also describes thoracic
  rounding, which the pipeline cannot see at all (see C-gaps).

**Fix:**
- Replace the intra-set forward-lean rule with hip shoot (C1).
- Keep absolute lean only as a set-level symptom, measured against the balance-model
  expectation (A4).

### [x] B4. The symmetry rule measures something coaches don't watch  [coach]

**The problem:**
- `SymmetryRule` compares left vs right *knee-flexion angle* near the bottom.
- In monocular mode the far leg is the noisiest keypoint chain.
- "Heavier side = more knee flexion" is not an established relationship.

**What a coach watches:**
- **Hip shift:** the pelvis translating sideways, usually on the ascent.
- **Bar tilt:** already implemented in `BarTiltAsymmetryRule`, and good.

**Fix:**
- Add hip shift (C4) as the primary asymmetry fault.
- Demote knee-angle asymmetry to a supporting feature.
- Bug: the bar-tilt fault is emitted on the frame *after* the rep ends, with the next rep's
  number, so it is effectively never in `RepData.faults`.

### [x] B5. Depth is "parallel for everyone"  [coach]

**What a coach would say:** a coach sets depth per athlete. Go as deep as you can with heels
down, no hip shift and a controlled pelvis, then build from there. Forcing depth past someone's
capacity is how you buy lumbar flexion at the bottom.

**What the code does:**
- The BiLSTM counter refuses reps below class 3, and the agent says "deeper".
- A lifter with legitimately limited hips can therefore get **no reps counted**, and "deeper"
  every rep.
- `depth_limit` aggregates with `last`, so it is judged on the final rep only.

**Fix:**
- Set a per-athlete depth target at assessment: the deepest position before heel rise or hip
  shift appears.
- Count reps against that target, and keep "parallel" as a goal label.
- Judge depth on the median rep.
- Add depth drift (C6) separately.

### [x] B6. Knowledge-graph content errors a coach would catch immediately  [coach]

**Prescription errors:**
- `weak_hip_abductors` prescribes "Copenhagen planks". Those train the **adductors** (Harøy et
  al. 2019). Replace with side-lying hip abduction, banded lateral walks, and single-leg work
  (step-downs, split squats).
- `limited_hip_flexion` prescribes "90-90 stretches and hip flexor work". Tight hip flexors
  limit hip **extension**, not flexion. Depth limited by the hip is posterior-hip, adductor,
  capsular or bony (acetabular depth, FAI). Use rockbacks, adductor rockbacks and goblet
  "prying" holds. A wider stance with more toe-out clears the femoral neck. Some hips simply
  stop where they stop, so add a tier-0 "hip anatomy" note.
- `limited_ankle_df` is tier 3 only. The fix a coach applies in session one is to **elevate the
  heels** (plates or lifting shoes). Add that as a tier-1 cause or delta, and keep mobility as
  tier 3.

**Unobservable or unsupported causes:**
- `foot_collapse_arch` takes its evidence from toe-out ≥ 20°, which says nothing about the arch.
  The arch isn't observable from ankle, heel and toe points. Remove it, or turn it into a
  "check your feet" prompt with low prior until foot keypoints support it.
- `narrow_stance` as a *valgus* cause: what drives knees inside the foot line is a stance width
  vs toe angle **mismatch** (wide stance, toes forward). Narrow stance itself is not the cause.
  Replace with a `stance_toe_mismatch` cause.
- `expected_knee_valgus_baseline` assumes more hip width means more expected valgus
  (`min(3, (hip_width − 0.25) · 30)`). There is no basis for this. With the 3D knees-over-toes
  metric, neutral is 0 by construction.

**Missing evidence-backed links:**
- Limited ankle DF → knee valgus (Macrum et al. 2012; Dill et al. 2014). This is the strongest
  link in the literature, and it isn't in the graph.
- Unilateral ankle DF → asymmetric depth or hip shift. DF is already measured per side
  (`ankle_df_l_max` / `_r_max`).
- Foot placement asymmetry → asymmetry. This is tier 1, instantly correctable and visually
  verifiable (C8).
- Load or fatigue → depth limit. People cut depth when the weight is heavy.
- Heel rise appears nowhere in the graph, yet it is a coach's #1 visible sign of ankle
  restriction. Today it is a cue only.

**Fatigue proxy:**
- `weight_too_heavy` takes its evidence from the composite-score trend slope, which A1 and A3
  corrupt. Replace with velocity loss (C5).

### [x] B7. Evidence tests read only the single worst rep  [coach]

**The mismatch:**
- Symptoms are detected over all reps (`engine.py:83-125`).
- Every evidence test runs on `_pick_representative_rep`, the single worst composite rep.
- A cause can therefore be scored against a rep that doesn't show its symptom.

**Fix:** score evidence on aggregated features, e.g. the median over the reps that contribute to
the symptom.

### [x] B8. Rep score shape  [coach]

**Where:** `rep_scoring.py`.
- **Trunk score is two-sided.** It penalizes being *more upright* than expected, which is not a
  fault.
- **Knee score uses `abs()`.** It penalizes knees-out, which is the correction we prescribe.
- **Depth is 42% of the composite.** A controlled rep 2 cm above parallel scores below a deep
  rep with knee cave.

**Fix:** a coach ranks position control above depth. Make both penalties one-sided, and
rebalance the weights once the C faults exist.

### [x] B9. "Confidence" is severity, not certainty  [coach]

**The problem:**
- `engine._compute_confidence` is mean symptom severity.
- Nothing models *measurement* confidence: capture mode, keypoint quality, number of reps, or
  whether the camera geometry can see the fault at all.

**Why it matters:** a coach who can't see something says nothing.

**Fix:** add a per-fault observability matrix (D3) and suppress faults the current setup can't
see.

### [x] B10. Delivery: what reaches the athlete's ears  [coach]

**What gets spoken:**
- **Severity-blind and side-blind.** The spoken cue ignores severity and side.
  `FAULT_MESSAGES` per severity is never voiced. `affected_side` / `heavier_side` are dropped at
  `ipc_bridge.send_fault`.
- **Clips may name the wrong side.** The clips are 10 gpt-4o-realtime improvisations per key,
  with no stored transcripts. The `even_it_out` prompt says "favoring his right leg", so some
  clips may name the **right** side regardless of which side actually shifted. Listen to them,
  then regenerate side-specific variants with transcripts stored.
- **Tier-3 causes are never spoken.** These are the mobility and strength homework items, the
  "long-term program" value. They are not serialized at `ipc_bridge.py:321-329`.

**Possible timing bug:** the recap is queued at `on_rep_complete` and reads the diagnosis
immediately. The diagnosis only exists after `rest_start` reaches the pipeline. The set-N recap
likely reads set N−1's diagnosis, or none. Confirm against the logs.

**Guardrail #1:** the cue clips are gitignored (`.gitignore:178`). A fresh device therefore
falls back to OpenAI TTS for every cue. `PREEMPTIVE_TEXT` lines are cloud TTS too. Ship the
clips with the device image, or render them with a local TTS.

### [x] B11. Cue language  [coach]

**The problem:** the strings use jargon ("Significant knee valgus — knees out!") and
internal-focus phrasing ("Grinding — push through!").

**What works better:** external-focus cues reliably beat internal ones for performance and
learning (Wulf 2013 review). Build a small cue library keyed by (fault, phase), with one
primary cue and two alternates:

| Fault | External cues |
|---|---|
| Ascent valgus | "Spread the floor" · "knees toward your pinky toes" |
| Hip shoot | "Drive your upper back into the bar" · "chest and hips rise together" |
| Balance / heel | "Whole foot — feel your heel and big toe" |
| Hip shift | "Push the floor away evenly with both feet" |
| Depth | "Sit between your heels" |

**One focus per set.** Coaches give one thing at a time. The orchestrator's 8 s gap is a timing
rule, not a focus rule.

### [ ] B12. Dead or unwired code relevant to new faults  [hygiene]

- `pipeline.py:744` passes `derivatives=None`, so `EccentricTempoRule` / `StallingRule` can
  never fire even if registered.
- `utils/types.py:409-417` has a duplicate, unused `FaultType` with `KNEE_VARUS` / `HIP_SHIFT`.
  Delete it, or merge into `fault_types.FaultType` when hip shift lands.
- `progress_context.FAULT_LABELS` has dead keys (`butt_wink`, `shallow_depth`,
  `asymmetric_loading`) and is missing `bilateral_asymmetry` and `heel_rise`.
- `combined_perturbation` is computed, sent and stored, but never read.
- Cue clips `slow_down` and `brace` exist with no producer. `hips_through`, `flat_back` and
  `lockout` are mapped nowhere.
- The recap dimension list asks for "ankle", which never exists, and omits tempo
  (`coaching_orchestrator.py:1405-1410`).

---

## Part C — What a world-class coach watches that we don't

Ranked by coaching value × observability with today's keypoints. Every metric is O(keypoints)
geometry per frame, so all are edge-safe. All are squat-only.

| # | Fault | What the coach sees | Metric (current keypoints) | Mono / multi | Primary cue | Graph causes |
|---|---|---|---|---|---|---|
| **C1** | **Hip shoot / good-morning squat** | Hips rise faster than the chest out of the hole; the bar drifts forward | Trunk pitch at ⅓ of hip recovery on the ascent − pitch at bottom (°). Alternative: hip-rise / shoulder-rise ratio over the first 30% of the ascent | Both (sagittal; mono relies on depth regression) | "Drive your back into the bar" | Quads limiting vs posterior chain (T3: pause/front squats); load or fatigue (T2); brace lost at bottom (T1); ankle restriction (T3) |
| **C2** | **Ascent knee valgus** (extends B2) | Knees cave on the way up, worse on hard reps | 3D knees-over-toes deviation, p90 over bottom + ascent; phase of peak; Δ against the athlete's best rep | Multi best; 2D after A5 fix | "Spread the floor" | Ankle DF (T3 + T1 heel lift); stance/toe mismatch (T1); load/fatigue (T2); hip abductor strength (T3) |
| **C3** | **Balance: bar/COM over midfoot**, unified with heel rise | Weight drifts to the toes or heels; heels lift | Sagittal offset of bar (≈ shoulder midpoint) or COM from midfoot, ÷ foot length. Reuses `compute_com_ground` / `compute_balance_target_ground` | Multi best; mono approximate | "Whole foot" / "heels down" | Ankle DF; knees-first initiation (T1); bar position (T0); load (T2) |
| **C4** | **Hip shift** | Pelvis slides sideways, usually out of the hole | Lateral offset of pelvis midpoint from ankle midpoint ÷ stance width; peak and phase | **Both**: lateral is in the image plane for a frontal camera, so mono 2D is good | "Push evenly through both feet" | Unilateral ankle DF (measured per side); unilateral hip ROM; foot placement asymmetry (T1); strength asymmetry or old injury (ask, don't guess) |
| **C5** | **Velocity loss / grind** | The bar slows; reps get harder; sticking point | Mean concentric vertical velocity of the shoulder midpoint (bar proxy), m/s. Set velocity loss % = (best − current) / best | Both | Not a form cue: "that's a hard one — last rep" / set-end call | Replaces `weight_too_heavy` evidence. VL > 20–30% marks a set taken near failure (Sánchez-Medina & González-Badillo 2011; Pareja-Blanco 2017) |
| **C6** | **Depth drift** | Later reps get shallower | Per-rep geometric depth (A3) against the athlete's best rep this set | Both | "Same depth as rep 1" | Fatigue or load (T2) |
| **C7** | **Incomplete lockout** | Not standing tall between reps ("piston" reps) | Top-of-rep hip height below standing baseline, or knee flexion > ~15° at the top | Both | "Stand tall, squeeze at the top" | Fatigue (T2) |
| **C8** | **Foot placement asymmetry** (setup, pre-rep) | One foot staggered forward; one foot flared more | Before rep 1: front-to-back ankle offset ÷ foot length; signed L/R toe-out difference (needs A7) | Multi best | "Bring your left foot back an inch" | T1 only. Instantly correctable, visually verifiable, ideal for the demo |
| **C9** | **Eccentric control** | Dive-bombing, or crashing into the bottom | Descent time plus deceleration at the bottom (needs `derivatives` wired, B12) | Both | "Control the way down" | Novice or skill (T1). Only cue when position is also lost at the bottom; fast-but-controlled is fine for trained lifters |
| C10 | Descent initiation | Knees-first (heels lift) or hips-first (folds over) | Hip vs knee flexion velocity ratio over the first 20% of the descent | Both | "Break at hips and knees together" | Links C3 / C1 |
| C11 | Head / neck | Cranking the head up, or looking at the floor | Nose/ear angle relative to the shoulder→hip line | Both | "Eyes on a spot on the floor" | Low priority |

**Deliberately not adding:**
- **"Knees past toes."** Restricting forward knee travel raises hip torque and trunk lean (Fry et
  al. 2003). A good coach does not cue it.
- **Absolute trunk lean as an intra-set cue.** See B3.

**Honest observability gaps (V2 sensing; don't claim these):**
- **Lumbo-pelvic flexion at the bottom ("butt wink").** This is the most-asked-about squat fault.
  It is not observable from COCO-17 or MediaPipe-33, which have no lumbar or pelvic-orientation
  landmarks.
- **Thoracic rounding.**
- **Bracing.**
- **Foot arch collapse.**

`FAULT_LABELS` already contains `butt_wink`, and the `chest_up` clips mention upper-back
rounding. Make sure the agent never asserts these.

Options for V2:
- A keypoint model with spine landmarks.
- Depth-camera back contour.
- Self-report ("do you feel your lower back tuck at the bottom?").

---

## Part D — Architecture of the diagnosis layer

### [x] D1. One per-rep feature vector from the full trajectory

**Today:** only a bottom frame and a top frame reach diagnosis (`RepKinematicSummary`). Most C
faults live on the ascent.

**Proposal:** a `RepFeatures` record per rep.
- Phase timings and velocities.
- Bottom values.
- Ascent peaks with phase tags.
- Hip shoot Δ, balance offset, hip shift, valgus (bottom / ascent), depth ratio, lockout.

Intra-set verdicts and set-level symptoms both read it. This is the fix for B1.

### [x] D2. Two-channel thresholds (the fix for A2)

- **Absolute:** anthropometry-aware standards, never moved by observed reps.
- **Drift:** each rep against the athlete's best rep this session, which gives the fatigue
  signal.

The second channel is what makes the coach sound attentive: "your knees started caving from
rep 5".

### [x] D3. Observability matrix

A static table of fault × capture mode (single frontal camera / triangulated) × required
keypoints → `observable` / `approximate` / `not observable`.

| Fault | Single frontal camera | Triangulated |
|---|---|---|
| Hip shift, frontal valgus | Good | Good |
| Sagittal metrics (hip shoot, balance, lean) | Approximate (monocular depth regression) | Good |
| Heel rise | Not observable | Good |

Faults marked not observable are never detected or spoken. Faults marked approximate carry a
hedge flag to the LLM.

### [x] D4. Output contract to the small model (guardrail #2)

Each surfaced item should be flat:

```
{fault_id, side, phase, reps, severity_tier, is_drift, cue_id,
 magnitude_phrase, cause_id, cause_tier, observability}
```

No raw angles. The LLM picks wording; it never interprets kinematics. The current
`explanation_template` strings, once B6 is fixed, can remain the canonical sentences.

### [x] D5. Positive events

Coaches reinforce more than they correct. Emit:
- "best rep of the set" (already computed as `best_rep_number`, never voiced by design)
- "fixed after cue" (partially exists via `PREEMPTIVE_TEXT`)
- "depth matched target"

### [ ] D6. Ground-truth labels

Every threshold here is an opinion until it has been checked against a coach's eye.

**Proposal:** about 200 reps from 10+ lifters (the 8-user validation plan fits), each labelled
per fault and phase by a certified S&C coach. Use them to report per-fault precision and
recall, and to set thresholds.

This also gives the YC application a defensible claim: "agrees with a CSCS on X% of faults".

---

## Part E — Recommended sequence to finalize Squat V1

1. **Measurement (blocking).**
   - A1, A2, A3, A4, A5, A6, A7, A9.
   - Each fix gets a regression test in the production frame.
   - Plus one real-recording end-to-end test. Verify: replay a recorded set and get sane depth,
     lean and valgus numbers.
   - A12: gate every sagittal fault behind triangulated capture (D3) so the single-camera demo
     never says it.
2. **Make the existing faults coach-grade.**
   - B2: ascent valgus, per-rep verdict.
   - B3 → C1: hip shoot replaces intra-set lean.
   - B6: graph content.
   - B10: side-aware, transcribed, locally-shipped cues; recap timing.
   - B7.
   - Re-rank cue priority: ascent valgus, hip shoot, balance/heel, hip shift, depth.
3. **Add the demo-critical new faults.**
   - **Frontal plane first** (robust on one camera): C4 hip shift, C8 foot setup asymmetry.
   - **Then sagittal faults**, once triangulated ≥30 fps capture is the default (A12):
     - C1 hip shoot
     - C3 balance, which reuses the existing COM code
     - C5 velocity loss, which feeds the `weight_too_heavy` → "load" recommendation
   - **Then** C6, C7.
   - Each new fault follows the touchpoint checklist below.
4. **Unify the architecture.** D1/D4: one feature vector, one output contract.
5. **Calibrate.** D6: labels, thresholds, precision/recall. Collect them on the **triangulated**
   rig, since no triangulated user data exists today.

### Touchpoint checklist for one new squat fault (e.g. `hip_shoot`)

1. `fault_types.py`: enum value plus messages.
2. `faults/rules/<fault>.py` + `rules/__init__.py`. Use `finish_rep` for a once-per-rep verdict.
3. `config.py`: config model, a `FaultsConfig` field, **and the explicit parse at `:381-390`**
   (otherwise YAML values are silently ignored). Plus `config/biomechanics.yaml`.
4. `profiles/squat.py`: register the rule. List order breaks same-frame ties.
5. `cue_cache.py`: `SQUAT_CUES`, `FAULT_TO_CUE_MAP`, `FAULT_CUE_PRIORITY`. Unmapped faults get
   rank 4 and are starved by every gap.
6. `coaching_constants.py`: `CUE_TEXT_MAP`, `CUE_DISPLAY_LABELS`, and optionally
   `PREEMPTIVE_TEXT`.
7. `scripts/tools/generate_cue_audio.py`: `CUE_PROMPTS`, then regenerate the clips and store
   transcripts.
8. `progress_context.FAULT_LABELS`: so the LLM gets plain words.
9. `src/visual/display.html` `FOCUS_TEXT`; optionally `src/vocab/terms.py`.
10. `biomechanics_persistence.END_OF_REP_FAULT_TYPES`: decide membership.
11. Tests:
    - a rule test (mirror `test_heel_rise_rule.py`)
    - `test_faults.py`
    - `test_fault_priority.py`
    - `test_audio_cue_service.py`
    - `test_coaching_orchestrator.py`
    - `test_progress_context.py`
    - `test_biomechanics_persistence.py`

For a new symptom and cause:
- **Graph entries:** `symptoms.yaml` + the feature (a `RepKinematicSummary` field filled in
  `bridge.py`, or `engine._extract_feature`), plus `causes.yaml` + the evidence test + an
  optional delta and `magnitude_*`.
- **Template placeholders:** every one must be filled by `engine._fill_template`, or the raw
  `{...}` gets spoken.
- **Tier 3 needs extra plumbing:** serialization in `ipc_bridge.py:321-329` and
  `pipeline_process.py:810-833`.
- **Visuals:** `CORRECTOR_CAUSE_ORDER` plus a corrector branch.

---

## Real-data check

### Method

**Replay:** frame by frame through the current `AnalyticalIKSolver`, the real `SignalRepCounter`,
`RuleEngine` + `SquatProfile`, then `bridge` → `HypothesisEngine` / `score_set`.

**Scripts:** the session scratchpad holds the scripts (`audit_build.py`, `audit_report.py`,
`probe_*.py`). They are not committed.

**Primary cohort:** 07-28 → 08-11, after the ROMClamp/FPPA fixes. **190 reps, 33 sets, 17 runs.**

**Exclusions:** 13 of 633 replayed reps were dropped as corrupted. All 6 reps of the 07-22 run
have folded legs.

**Caveats:**
- **Every user run is monocular MediaPipe.** There is not a single triangulated user recording.
- The data is about 12 fps.
- The sets are bodyweight.

### Fire rates (current era)

| Signal | Fires on | Notes |
|---|---|---|
| `excessive_trunk_lean` symptom | **79% of sets**, severity 1.0 | 93–100% in older cohorts. Re-pivoting the formula to the code's own 0.84 ratio drops it to 33% (A4) |
| `anthropometric_femur_torso_ratio` cause | **0% in every cohort** | Keypoint ratios are 0.82 [0.80, 0.92]; the test needs > 1.0 |
| `SymmetryRule` (bilateral asymmetry) | **74% of reps** | L–R knee flexion difference is 12.7° at the bottom vs 4° standing, which is monocular depth error. This also blocks auto-calibration (it completed in 10 of 33 sets) |
| `forward_lean` rule | 28% of reps | |
| `knee_valgus` rule / `knee_not_tracking_toes` | 1% of reps / 6% of sets | 2D FPPA at the bottom reads −34° [−42, −21] (A5) |
| Knee-tracking sub-score | **0.00 at p10 / p50 / p90** | `abs()` on −34° FPPA. Caps every composite at ~0.84 (B8) |
| `narrow_stance` cause | 45% of sets (71–100% older) | Stance base 1.1× keypoint shoulder width (0.318 m) marks normal stances narrow (A11) |
| `depth_limit` | 27% of sets | Knee-angle "parallel" and geometric depth disagree on 31% of reps (81% in older data). Always in the same direction: the knee angle says parallel while the hip is above the knee (A3) |
| `heel_rise` | 0% | Expected: there is no FootState in monocular mode |

**Older code fired on the wrong direction.** The live-recorded events from older code include 81
`knee_valgus` faults that fired on knees-**out** readings (`abs()`; `valgus_debug` 07-28 shows
L/R −42/−36 reported as 42). The current signed code fixed that event path, but `rep_scoring`
still uses `abs()`.

**A1 is confirmed on real data.** The scorer's "loaded" frames average 50° of knee flexion
against a rep max of 114°. The live composite reads 0.78 where the correct value is 0.67.

### [x] A12. Monocular sagittal-plane measurement is dominated by camera placement  [blocker for sagittal faults]

**The evidence:**
- Trunk pitch at the bottom has a median of 20°, but per-run medians range from **6.6° to
  56.6°**. The athlete's anatomy does not change that much between runs, so camera placement is
  driving the number.
- Monocular depth error also creates the knee-flexion "asymmetry" above.

**What this means:**
- Until capture is triangulated, every sagittal-plane fault is unreliable on the current
  recordings: lean, hip shoot, balance and depth from knee angle.
- Frontal-plane faults survive monocular capture well, because they live in the image plane:
  hip shift, valgus after the A5 fix, and foot setup.
- This promotes F12 from the July audit (enable triangulation) to a **prerequisite** for the
  sagittal faults in Part C.
- No triangulated user data exists, so the D6 labels must be collected on the triangulated rig.

### Candidate new metrics on this data

| Metric | Result on recorded data | Verdict |
|---|---|---|
| **Hip shoot** (C1): pitch at ⅓ ascent recovery − pitch at bottom | −3.3° [−16.5, +4.3]; the trunk normally gets *more* upright | Only ~5 frames separate bottom from ⅓ at 12 fps, and frame jitter is 1.3° (p90 3.3°). **Noise-dominated here.** Needs ≥30 fps and triangulation before thresholds can be set |
| **Velocity loss** (C5): shoulder height vs ankle midpoint | Mean concentric velocity 0.53 m/s [0.38, 0.64]; within-set CV 12%; set VL 2.6% [0, 24.9] | One frame of boundary error is ±10% of a 0.84 s ascent. **Noise below ~15–20% VL at 12 fps.** Usable for coarse "set got much slower" calls now; precise RIR needs ≥30 fps |

**Revised priority for a single-camera demo:** C4 hip shift, C2 valgus (after A5) and C8 foot
setup are the robust frontal-plane faults. C1, C3 and C5 go live once triangulated ≥30 fps capture
is the default. The product's rack cameras are that path.

---

## Status — 2026-10-01, branch `squat-v1-coach-grade`

Implemented on the branch. The interface between the pipeline and the voice agent is frozen in
`CONTRACT.md`.

**Tests:**
- Biomechanics: 1306 passing.
- Rest of the suite: 540 passing. The 11 failures are the same 11 that fail on `main`
  (`test_rep_sound`, `test_v6_verification`).

### Done

**Part A — measurement**
- **A1** Heights are Y-up at the source. An end-to-end test drives pipeline output through the
  session tracker into a diagnosis.
- **A2** Calibration measures capacities and sets only the depth target. The rep-1 threshold
  calibration is deleted, and drift is tracked by the session reference.
- **A3** Depth has one geometric definition. The BiLSTM only segments movement; the athlete's
  target decides what counts.
- **A4** Trunk lean is judged against the shoulders-over-midfoot balance model.
- **A5** Single-camera knee tracking is corrected for toe-out and put on the femur-tilt scale.
- **A6** No foot-angle fallback. Unseen feet give no knee reading.
- **A7** Foot angle is signed.
- **A8** Capacities are the median rep's p95.
- **A9** Forward lean is no longer a squat fault.
- **A10** Missing values become NaN, never a perfect or deepest class.
- **A11** Stance is measured against biacromial width, and stance advice is spoken in cm per side.
- **A12** Observability gating: approximate faults get no cue mid-set and are hedged in the recap.

**Part B — coaching logic**
- **B1 / D1** One per-rep feature set drives both the cues and the diagnosis.
- **B2–B9**:
  - knee tracking over the bottom and the ascent, with phase and drift
  - hip shoot
  - hip shift and bar tilt
  - per-athlete depth target
  - graph content: 12 symptoms, 25 causes, prescriptions fixed
  - evidence judged on the median rep
  - one-sided scoring that excludes unmeasured dimensions
  - measurement confidence including coverage
- **B10 / B11 / D4 / D5** (agent side):
  - side-aware cues
  - plain external-focus cue text
  - long-term and contextual causes voiced
  - recap race fixed
  - highlights
  - `force_end_current_set` set count fixed

**Part C — new faults**
- **C1–C9** are rules.
- **C10** (descent initiation) and **C11** (head position) feed the graph.

### Real-data replay (190 reps, single camera, ~12 fps) — before → after

| Measure | Before | After |
|---|---|---|
| `excessive_trunk_lean` sets | 79% | 13–21% |
| `narrow_stance` sets | 45% | 14–36% |
| `weight_too_heavy` sets | 39% | 6% (2 of 33) |
| Knee score | 0.00 for every rep | not scored (no 2D or toes recorded) |
| Composite median | 0.67 | 0.90–0.92 |
| Production diagnosis crash | — | found by replay, fixed, covered by test |

**Hip shift** is approximate on a single camera. Backward hip travel leaks into the sideways
measurement through monocular depth error. The rate was ~50% of reps at the noise-floor
thresholds before they were raised.

**Depth gate:**
- 13% of descents are not counted at the uncalibrated target (0.2).
- About 16% are not counted at a simulated calibrated target. 9 of 17 runs could reach parallel,
  so they are held to it.

### Still open (needs a person or a product decision)

- **Cue audio:**
  - Draft text with `scripts/tools/draft_cue_text.py` (`gpt-6-astra`), then the user reviews
    `src/assets/cue_text/cues.json`.
  - Then generate with `scripts/tools/generate_cue_audio.py` (Cartesia sonic-3, Nova's voice,
    3 per cue, full regeneration).
- **Shipping and cloud TTS:** cue clips are still gitignored, and `PREEMPTIVE_TEXT` still goes
  through cloud TTS (guardrail #1).
- **D6:** coach-labelled reps on the triangulated rig. Every threshold here is an expert estimate
  or a noise floor until then.
- **Triangulated capture as the default** for side-view faults (A12). Not one triangulated user
  recording exists yet.
- **BiLSTM training-label provenance** is unknown; the training code is not in the repo.
- **Project rules file:** `.claude/rules/biomechanics.md` says "larger Y = higher", but the
  production frame is Y-down.
- **`EccentricTempoRule` / `StallingRule`** are still unwired (`derivatives=None`). Squat tempo now
  comes from rep features.
- **`combined_perturbation`** is still computed and never read.
- **Monocular hip shift:** a candidate fix is to measure it in the image plane with a facing
  gate.
