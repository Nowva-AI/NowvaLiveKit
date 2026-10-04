# Conventional deadlift — implementation status

Branch `claude/deadlift-v1-impl`. It implements the software of `docs/deadlift/PLAN.md` (v18).
The interface between the pipeline and the voice agent is frozen in
`.claude/deadlift/CONTRACT.md`. What the analyser does, metric by metric, is in
`docs/deadlift/KNOWLEDGE.md`.

**The deadlift is gated.** Normal users cannot reach it until it is validated (J8). To try it,
set `NOWVA_DEV_COACHING_READY=deadlift` for the agent process; the pipeline subprocess inherits
it. Without that variable:
- every deadlift name resolves to an untracked stand-in that counts nothing and judges nothing;
- the menu does not offer the deadlift;
- a scheduled deadlift runs no briefing and no rules.

## What is built, by milestone

| Plan | Status | Where |
|---|---|---|
| **J0** squat net | Done. Every golden passes: squat output is byte-identical fresh, after RDL / untracked / **conventional deadlift** sessions, and through the voice-agent delivery path. The `VALIDATION.md` skeleton and a CI proposal are written | `tests/test_biomechanics/test_squat_golden.py` (+ fixtures), `test_squat_invariants.py`, `tests/test_squat_delivery_golden.py`, `tests/test_squat_agent_invariants.py`, `tests/test_squat_session_pins.py`, `docs/deadlift/VALIDATION.md`, `docs/deadlift/CI.md` |
| **J1** software parts | Done. Deadlift sagittal frame; measured gravity (tool, runtime mapping, re-resolved whenever a calibration is installed); setup model (two solves, shin-to-bar contact measured per lifter); raw rig recorder and replay through the real pipeline (`NOWVA_REPLAY_DIR`); `KNOWLEDGE.md` (not yet reviewed) | `deadlift/frame.py`, `deadlift/gravity.py`, `scripts/tools/measure_gravity.py`, `deadlift/setup_model.py`, `triangulation/rig_recording.py`, `scripts/tools/record_rig.py`, `docs/deadlift/KNOWLEDGE.md` |
| **J2** simulator | Done, as a standalone generator rather than the §8.5 extension of `.claude/preik-audit/harness` (that harness's import bugs, §8.5, are still open). Synthetic sets with ground truth: dead stop, touch-and-go, quick re-pull, grip-and-rip, re-setup (standing up off the bar between reps), a sticking point mid-pull, failed rep, dropped bar with bumper bounce, every v1 fault, plate occlusion, keypoint and bar noise, wrist proxy (hands ride the tilted bar), tilted world. Each rep is lowered into the next rep's setup, so a set can change from rep to rep ("the next rep improves") without the lifter jumping. **Model-free poses:** the setup and knee-pass trunk angles can be scripted instead of taken from the setup model, and the truth (trunk angles, setup hip height, rise ratio, knee flexion) is read off the emitted poses. Capture lag is exercised through the real pipeline (`test_deadlift_pipeline.py`); a moved camera is not simulated | `deadlift/simulator.py`, `tests/test_biomechanics/test_deadlift_analyzer.py` |
| **J3** profile on the platform | Done: profile and gate; registry variants; squat-default hooks; `process_frame` fork; analyser (measurement / features / state machine); counter; D1–D10; deadlift session reference; config; `_activate_profile`; refine and view-buffer gates; foot-contact reset; body-measurement flag; after-switch golden | `profiles/deadlift.py`, `deadlift/measure.py`, `deadlift/features.py`, `deadlift/analyzer.py`, `deadlift/rep_counter.py`, `deadlift/rule_base.py`, `faults/rules/deadlift_*.py`, `pipeline.py`, `pipeline_process.py`, `config.py` |
| **J4** 3D bar, software | Done: batched multi-view detector wrapper; cross-view association with racked-bar rejection (heights along measured gravity when known); single-view hub fallback; 3D Kalman with predicted states; capture-time buffer; per-set bar health log | `deadlift/bar_detector_multi.py`, `deadlift/bar_tracker_3d.py`, `deadlift/bar_buffer.py` |
| **J5α** delivery slice | Done. Closed-loop bar-over-midfoot guidance (tracked bar only); D1/D3 (and all other) cue text; the contract fields on `cache_cues` / `frame_data`; min tier; no cue between touch-and-go reps | `src/agent/services/coaching_orchestrator.py`, `coaching_service.py`, `coaching_constants.py`, `src/assets/cue_text/cues.json`, `coaching/cue_cache.py`, `coaching/ipc_bridge.py` |
| **J5** delivery integration | Done. Deadlift recap; session metadata (grip, plates, belt, shoes) on `start_capture` / `set_exercise`; first-session briefing and empty-bar warm-up; knowledge card and `explain_deadlift` tool; form-check branch; display tiles and score dimensions; per-exercise fault sets in the orchestrator; DB tagging | `src/agent/agents/**`, `src/main.py`, `src/visual/display.html`, `scripts/tools/draft_cue_text.py` |
| **J7** set diagnosis | Done. `HypothesisEngine(graph)` defaults to the squat graph; the deadlift graph has 10 symptoms and 16 causes; scoring is weighted setup 25 / coordination 25 / bar path 20 / lockout 15 / symmetry 15; a rolling update after each rep and `diagnosis_complete` at set end | `diagnosis/engine.py`, `diagnosis/graph/deadlift_*`, `deadlift/diagnosis.py` |

## Measured on the simulator (J2/J3 acceptance), with the tested envelope

Every number below comes from the simulator, not from real lifts. It shows that the code does
what it claims under the conditions listed. It says nothing about how real bodies move; that is
J6 (`VALIDATION.md`).

**Noise models.** Keypoint noise is added at the analyser's input, after where the pipeline's
Kalman sits:
- "i.i.d." is independent per frame.
- "AR(0.8)" is the slow, correlated wander a Kalman-smoothed triangulation leaves. Of the two,
  it is the closer to the platform's documented 1.6–2.4 cm.

"Cued" means at or above the fault's min tier, as the agent would speak it. Every set has 3 mm
of bar noise when the bar is tracked.

**Tracked bar, dead-stop sets.** Each row is 12 sets of 3 clean reps: pulls of 0.6, 1.2 and
3 s, 4 seeds each.

| Keypoint noise | Rep counting | Event error (liftoff, knee, top, floor) | Cued faults on clean reps |
|---|---|---|---|
| none | 12/12 sets exact | median 14 ms, p95 61 ms | 0 / 36 |
| 2.0 cm i.i.d. | 12/12 | median 16 ms, p95 67 ms | 0 / 36 |
| 2.0 cm AR(0.8) | 12/12 | median 18 ms, p95 66 ms | 0 / 36 |
| 2.5 cm i.i.d. | 12/12 | median 16 ms, p95 67 ms | 0 / 36 |
| 2.5 cm AR(0.8) | 12/12 | median 19 ms, p95 66 ms | 1 / 36 |

**Tracked bar, touch-and-go.** Sets of 4 touch-and-go reps plus one dead stop (12 per row):

| Keypoint noise | Rep counting | Event error | Cued on clean reps |
|---|---|---|---|
| none | 12/12 | median 15 ms, p95 67 ms | 0 / 60 |
| 2.0 cm AR(0.8) | 12/12 | median 17 ms, p95 67 ms | 1 / 60 |

Also counted exactly on the tracked bar:
- 5-rep touch-and-go sets at 2–5 mm bar noise (25 random tempos per noise level);
- 10-rep sets at 3 mm bar noise plus 2 cm AR keypoint noise.

At 5 mm of bar noise a low point can read as a dead stop: the next rep is then counted as a
quick re-pull instead of a touch-and-go, but no rep is lost.

**Wrist proxy** (no bar tracking), 12 sets per row:

| Keypoint noise | Dead-stop sets: counting / events / cued | Touch-and-go sets: counting / events / cued |
|---|---|---|
| none | 12/12, median 2 ms (p95 33), 0 / 36 | 12/12, median 3 ms (p95 33), 0 / 60 |
| 1.0 cm i.i.d. | 12/12, median 67 ms (p95 200), 0 / 36 | 12/12, median 62 ms (p95 433), 0 / 60 |
| 1.0 cm AR(0.8) | 12/12, median 67 ms (p95 333), 0 / 36 | — |
| 2.0 cm i.i.d. | 12/12, median 133 ms (p95 567), 0 / 36 | — |
| 2.0 cm AR(0.8) | 12/12, median 100 ms (p95 467), 0 / 36 | 12/12, median 133 ms (p95 533), 2 / 60 |

The cues on clean reps at 2–2.5 cm AR(0.8) are D9 (bent arms) on the tracked bar, and D8b and
D9 on the proxy's touch-and-go sets. They stay within the `VALIDATION.md` gate of 1 false
correction per 10 reps. A second sweep of clean dead-stop sets, five bodies (default, short,
tall, narrow hips with a wide stance, long femurs) × three tempos × four seeds, cues 0 of 180
tracked reps at 2 cm AR(0.8) and 1 of 180 at 2.5 cm (D8; round 5 cued 7 and 15, all D8).

**Fast touch-and-go on the wrist proxy.** Six-rep sets (5 touch-and-go, 0.4 s lowerings, 0.2 s
tops), 10 seeds per row: every rep counted at 0.5, 0.6 and 0.8 s pulls, at both 2 cm AR(0.8)
and 1.5 cm i.i.d. keypoint noise (0 of 360 lost per condition pair). Before round 4, 0.5–0.6 s
pulls lost whole sets (1 rep counted of 6).

The proxy's events are bound by the wrists' noise: event timing past the 100 ms gate is a
tracked-bar result. Two proxy limits are documented in `KNOWLEDGE.md` §7:
- a grip-and-rip with no pause at the bottom loses the first rep;
- standing up off the bar without lifting it counts as a rep (5 counted for 3 with two
  re-setups; the tracked bar counts 3).

**Through the real pipeline and its Kalman** (tracked bar and proxy):
- 3/3 reps at 1, 1.5 and 2 cm triangulation noise, with the closed-loop STANCE reached;
- touch-and-go 5/5 at 0–2 cm on both bar sources;
- no cued fault on clean proxy reps at 2 cm i.i.d. or 1.5 cm correlated noise;
- standing references captured at 1–2 cm.

**Other cases**, each counting every rep:
- **Setups:** setup holds of 0, 0.1, 0.2 and 0.3 s with 0.5 s and 1 s hinges (grip-and-rip).
  Standing up off the bar between reps re-judges each setup.
- **Tempo:** 0.45 s to 5 s pulls with 3 mm bar noise, all events within 100 ms. A
  0.8 s sticking point mid-pull is ground through and counted. A hitch 90 % of the way up
  is not taken for the top, with or without a standing reference (top within 100 ms; it
  was ~1 s early). A grind 2.3–5.8 cm short of lockout for 0.4–0.8 s is timed at the real
  top within one frame, with no lockout fault (it was 0.75–0.93 s early with a false
  moderate D6). On the wrist proxy, grinds 3–5 cm short at 1–2 cm keypoint noise draw no
  false lockout cue (12–17 of 18 before round 6) and are timed within 0.3 s on 14–18 of 18
  reps. A stall about 1 cm short, with the knees inside D6's mild threshold of standing, is
  the top and is dated at the stall, with no lockout cue. A 2.5 cm shrug at the top moves
  the top about 0.1 s and reads no velocity loss (it read D10 severe).
- **Re-setups at the floor:** feet shuffled 5 cm toward the bar while hinged are judged where
  they now stand (D1 6 → 1 cm measured 6 → 1 cm, at 0 and 2 cm noise); feet the plates hide
  keep the stance's lock.
- **Stance, pelvis and turns (D8):** 15 sets of 3 reps over five bodies at 2 cm AR(0.8), on
  each bar source, false D8 cues: lower body 8° off the bar 1 of 45 (round 3: 32 of 48), left
  foot 6 cm ahead 1 (round 4: 10 of 24), hip line turned 4° against the legs 1–2 (round 5:
  10 of 36), a 15° pelvis twist with no sideways travel 1, against 1 for a clean stance.
  Standing turned 20° then squaring up, or turning 10° at the floor, at 1.5 cm: 0–1 of 45.
  A real shift with the pelvis turning ±10° with it reads the same as with a square pelvis
  (noise-free 0.185 either way; round 5 read 0.10 and 0.27).
- **Occlusion:**
  - Plates hiding the feet whenever the bar is off the floor, also through the pipeline.
  - Feet hidden from the setup on: the counting holds and D1 is still judged.
- **Bar dropouts:** 20 % and 40 % random dropouts, and a bar at 15 Hz. The set never switches
  to the wrists.
- **Drops and failures:**
  - Dropped bars, with the lifter standing over the bar until the next setup.
  - Failed reps, including a stall at the knees, are events, not reps.
- **Lockouts:**
  - Over-extended lockouts (25–40° behind vertical) are counted and judged by D5.
  - A lockout 30° short is a failed rep, and so is a pull that leans back 40° from mid-thigh
    with the knees still bent ~60°, whether the knees are seen or hidden. Over-extended
    lockouts with the knees hidden are still counted (leg length against standing).

**Faults:**
- **Injected faults:** with noise-free keypoints, every injected fault (D1–D10) fires at
  moderate or worse on every rep. D2's is a hips-first pull scripted without the setup model.
- **Clean sets:** clean, noisy and tilted-world sets are fault-free.
- **D2 against a held back angle**, which the setup model reads as 6–12° of excess:
  - cued on 0 of 90 reps at 2 cm i.i.d. noise and 0 of 90 at 2 cm AR(0.8);
  - before round 3 it was cued on 31 and 35 of 90.
- **D2 on a scripted hips-first pull** (ratio ~1.24): cued from the set's second rep on, on
  64–65 of 90 reps under the same noise. A first rep is cued alone only beyond a 1.30 ratio.
- **D2's rise ratio** reads within 0.04 of the poses' kinematic truth at 0.6, 1.2 and 3 s
  pulls (four pull shapes, two bodies). A line through the frames before the knee pass read
  fast pulls up to 0.12 high.
- **The next rep improves:** two hips-first reps then three with the chest rising are cued
  on the first two only, and every rep's ratio matches its own truth.
- **Near-threshold faults:** at 2 cm i.i.d. keypoint noise, a fault injected just above
  moderate reads moderate on 3–4 of 8 reps for D5 and D6 (+1–2° over the threshold), and on
  6–8 of 8 for the others. D9 at +5° reads mild.

**Gravity:** measured gravity keeps a 3° tilted world from faking drift. Against the body
vertical, the same set fakes more than 2 cm of drift.

**Compute (x86):**
- Analyser, at 1.5 cm keypoint noise: 0.3 ms median and 0.5 ms p95 per frame, and about
  3 ms on the frames that complete a rep (event fits, features, D8's two axes). The first frames
  of a process pay numpy's warm-up once (~12 ms).
- Bar association and tracking: about 0.6–1.1 ms.
- The Jetson numbers are still to be measured (J1).

## Findings that changed the plan

0. **The shin-to-bar contact distance must be measured, not assumed.** A fixed 5 cm moved the
   setup model's hip band ~7 cm, and its trunk prediction ~7°, per 2 cm of real difference.
   That faked D4 and D2 on clean setups. It is now measured on each lifter's setup frames (§2.7
   already said "learned"), and clean sets stay fault-free across four body types × three shin
   contacts.

1. **D2 thresholds 10/15/20° are mostly unreachable.**
   - With the bar at the knees, straight legs cap the hips-shoot excess at about 8–12° for
     typical bodies.
   - The initial thresholds are now 5/8/11°.
   - The rise-ratio cross-check is 1.0 instead of 1.4: a normal pull reads 0.5–0.75, a full
     hips shoot about 0.8–1.1 or more.

2. **D7 ranks before D4** (priorities 21 and 22). The §2.6 table had them the other way round,
   which contradicted §2.7 ("if D4 and D7 co-fire, D7 is cued first").

3. **Every deadlift set would have crashed the set summary.** Both
   `IPCBridge.send_set_complete` and `SessionTracker._compute_set_summary` ran
   `statistics.stdev` on NaN depths. A set with no depth now reports null depth; sets with depth
   take the unchanged squat path.

4. **Today's after-switch carry-overs for other exercises** (squat golden allow-list, not
   changed here):
   - RDL reps raise the squat's standing reference.
   - RDL and untracked frames feed a first-time athlete's body measurement.
   - `set_depth_target(None)` does not survive a switch.

   The conventional deadlift has **none** of these (§5.2).

5. **Pre-existing squat bug, not fixed here.** On the voice-agent side, `shallow_rep` messages
   are dropped before dispatch: `CoachingService._handle_message` hits a `category` keyword clash
   in the profiler call, outside its `try`. The delivery golden pins today's behaviour; fixing it
   is a separate change.

6. **The first analyser was built for clean keypoints** (independent review, round 1).
   - Body stillness under 0.05 m/s, holds reset by one false frame, and a 2.5 cm hands band
     made rep counting collapse at the platform's own documented keypoint noise (1.6–2.4 cm):
     0/3 reps at 1.5 cm through the Kalman.
   - A pull without a ≥ 0.4 s setup hold lost rep 1.
   - Feet hidden by the plates lost every rep.
   - The rebuilt gates are in PLAN.md v15 §2.3, and the envelope above is tested.

7. **Closed-loop foot guidance must not run on the wrist proxy.** The hanging wrists read
   −4.9 cm for a bar exactly over the midfoot, which would have said "Back a little". The live
   offset is now null on the proxy, and the agent ignores it there (CONTRACT.md §3, §5.5).

8. **D2 and D4 are model-relative, so the simulator cannot validate them.**
   - The default simulator builds its setups from the analyser's own setup model, so "clean
     sets are clean" partly holds by construction.
   - With poses scripted by trunk angle instead, the analyser's *measurements* match the
     kinematic truth: trunk change within 2°, rise ratio within 0.04, hip height within 1 cm,
     on four body types.
   - The model's *premise* is untested. It says the chest comes up 8–20° off the floor, so a
     lifter who holds the back angle reads 6–12° against it, though hips and chest rise
     together (the coaching standard).
   - D2 therefore fires only when the rep's model-free evidence says the hips led (finding 11);
     the model only sizes it. D4 stays severe-only. Both thresholds are derived from the model
     and are reset at J6 (`VALIDATION.md` §7).

9. **D10 runs on the tracked bar only** (PLAN.md §6). The wrists' speed is not the bar's.

10. **Touch-and-go needed a noise-proof rise** (review, round 2).
    - It required the bar to rise at *every* frame-to-frame step after the low point, within
      a fixed 3 mm.
    - One noisy frame defeated that, and since there is no dead stop between touch-and-go
      reps, the analyser swallowed every rep after it: 4 of 10 counted.
    - It now requires a rise clear of the noise band and a still-rising slope over the last
      0.1 s.

11. **D2's gate moved from the trunk to the rise ratio.**
    - The round-2 tier floor sat exactly on a held back angle's geometry (ratio 1.0), so noise
      decided it.
    - D2's poses are now taken from several frames: a median before liftoff, and a fit
      through the 0.2 s up to the knee pass (a curve since finding 19). One past the knee pass
      biased the change by −2°.
    - D2 is emitted only when the hips decisively out-rose the shoulders, rep and set.
    - The set diagnosis applies the same test, so a recap never names a hips shoot the rule
      would not.

12. **The wrist proxy's left-right must be locked per set.**
    - Rebuilt every frame from the noisy hip line, ~6.7° of rotation leaked the hips' forward
      travel into "sideways hip shift": 18 false D8 cues in 36 clean proxy reps.
    - Locking it from the stance and setup frames removed all of them.
    - D8b on the proxy is severe-only: the hands' noise is carried ~3× out to the hubs.

13. **Foot guidance must not arm on one frame, nor after the set.** The live offset became a
    median (1 s since finding 20), reported only while the lifter stands settled at the bar
    before the set's first rep. Previously it was reported while hinging down to the bar and while walking off
    after the set.

14. **A squat-path change, in a case that crashed.** The set summary's depth statistics now
    report null for a set of two or more reps with no depth at all. A deadlift set has no
    depth. On `main`, such a set crashed `statistics.stdev`. Every set with any depth, and a
    one-rep set, takes the unchanged path, and the squat goldens pin it.

15. **The midfoot must be re-locked at a re-setup on the floor** (review, round 3). A lifter
    cued "bar over midfoot" who moved the feet without standing up was judged against the
    stance's midfoot and cued again at the old offset, which breaks the demo loop's "the next
    rep improves" for Demo α's headline fault. The midfoot is now re-locked at the next
    liftoff from the feet measured since the dead stop (PLAN.md v16 §2.4).

16. **D8 is measured along the lifter's left-right, not the bar's.** PLAN.md §2.5 said "the
    ankle axis", and the code used the bar's axis on a tracked bar. The hips travel ~45 cm
    forward in a pull, so a stance 8° off square read as hip shift on 13 of 18 reps. The axis
    is now the hip and ankle lines over up to 8 s of frames at the bar, for both bar sources.
    Each degree of axis error reads ~0.03 of shift, which is why the lock needs seconds of
    frames (a lock from a 1 s floor dwell alone read false shifts).

17. **Touch-and-go's slope must start at the low point.** On the wrist proxy the noise widens
    the velocity window to 0.4 s; on a 0.5–0.6 s pull that window reached back into the
    descent and never read rising, while "wrists below the knees" stopped being true 0.2 s
    after the low point. Whole sets collapsed into one rep. The slope is now taken over the
    frames since the low point, and on the proxy the hands are checked at the low point.

18. **A hitch is not a top, and a lean is not a lockout.** Without a standing reference, a
    stall upright enough read as the top (the top event ~1–2 s early); the pull now resumes
    when the bar rises 5 cm past the top it held. An over-extended lockout now also needs the
    knees bent no more than 40°: leaning back from mid-thigh with bent knees counted as a rep.

19. **D2's knee-pass pose is a curve, not a line.** A fast pull accelerates through the 0.2 s
    before the knee pass; a line evaluated at its end read the rise ratio up to 0.12 high on
    0.6 s pulls. A parabola reads within 0.04 at every tempo, with no more noise (held back
    angle at 2 cm AR(0.8): 0 of 90 cued).

20. **Foot guidance needed hysteresis.** A 0.5 s median of a perfectly placed bar still read
    over 2 cm on 3–11 of 40 sets at 2–2.5 cm AR(0.8). The live offset is now a 1 s median,
    reported once 10 settled frames are in, per set (`reset_set` re-opens it), and the agent
    arms only past 3 cm (D1's mild threshold) and then guides down to 2 cm: 0 of 40 such sets
    arm at 2.5 cm, while a 3 cm offset arms on 36 of 40.

21. **The registry needs an allow-list for the conventional deadlift.** A list of variants
    kept missing names ("Band Deadlift" in the exercise library itself, "Suitcase",
    "Elevated", "with Chains", "Paused Deadlifts"). A name with any word beside "deadlift"
    other than barbell / conventional / touch and go is now an untracked variant. Squat names
    resolve as before (the profile hook defaults to "any name").

22. **The simulator's `hips_shoot_deg` could not inject D2**, and consecutive scripts with
    different setups moved the lifter 25 cm in one frame. `hips_shoot_deg` is removed (D2 is
    injected with the model-free trunk angles), and each rep is now lowered into the next
    rep's setup.

23. **D8's axis is the hip line, not the hip and ankle lines** (review, round 4). A staggered
    stance turns the ankle line (13° for a foot 6 cm ahead) while the hips stay square: 10 of
    24 reps falsely cued, and −0.15 noise-free through the real pipeline. The 8 s window also
    kept the stance frames from before a lifter squared up (8 of 12 cued after standing 20°
    off). The axis is now the hip line over the frames at the bar since the heading last
    turned (a 1.5 s block turning more than 6° ends them), plus the rep's own frames. Those
    extra frames pay for most of the shorter line: on clean reps at 2 cm AR(0.8), 2 of 72
    tracked and 1 of 72 proxy reps cued, as with the ankles; at 2.5 cm, 4 of 72 against 2. (Superseded by finding 27: the hip line before the rep is one of two axes.)

24. **A grind short of lockout is a stall, not the top.** The 5 cm rise that resumed a hitch
    left grinds 2–5 cm short of lockout taken as the top, with a false moderate D6 on lifters
    who then locked out. The pull now resumes once the bar clears the 2 cm hold band, and the
    top is timed on the climb from the bar's last still frame at the stall.

25. **Hidden knees are not straight knees.** The over-extended lockout's knee gate passed a
    lean-back failed pull when the plates hid the knees. Unmeasured knees now fail it.

26. **Considered and not done.** Telling a stand-up off the bar from a pull on the wrist proxy
    by the hands' separation (on the bar vs hanging): about 8 cm apart against 2.8 cm of
    per-frame noise, and person-dependent, so it needs real data first. Analyser windows
    counted in frames (rest, midfoot lock, live offset) are sized for the pipeline's fixed
    30 fps analysis rate (`pipeline.target_fps`); the left-right lock is in seconds.

27. **D8 needs two axes that agree** (review, round 5). The hip line alone read any yaw of
    the hip keypoints against the legs as shift (a habitual pelvic rotation, or a 1.5 cm
    front-back bias between the two hip keypoints: 4° read −0.09), and summing the rep's own
    frames let a pelvis turning with a real shift absorb or inflate it (0.10 or 0.27 for
    0.185). Every single line has a stance that defeats it (findings 16, 23). D8 now reads
    the shift on the hip line taken before the rep and on the bar's line (tracked axis, or
    the hands on the bar), and keeps the smaller reading, none when they disagree on the side.
    Recall pays a little: the smaller of two noisy readings is biased low (KNOWLEDGE §7).

28. **The lockout is judged on its most extended stretch.** A stall short of lockout that
    stayed inside the hold set D6 on the wrist proxy (false "Stand tall!" on 12–17 of 18
    reps that locked out). D6 and D5 now read the hold's 0.3 s with the least hip and knee
    flexion.

29. **A stall resumes only from a deficit, on medians.** The resume band was two noise bands
    on the proxy (3.6–6.5 cm), so grinds there stayed at the stall; and a shrug past 2 cm
    moved a locked-out top, reading as a slower rep (D10). The pull now resumes only when
    the hips or knees held ≥ 8° short of standing, on the bar's 0.2 s median against the
    hold's, by 0.5 cm or its noise.

30. **Hidden knees are judged by the leg's length.** Treating unmeasured knees as bent made
    real over-extended lockouts with the knees hidden into failed reps. Hip height would not
    do (leaning back pushes the hips forward and ~9 cm down on straight legs); the hip–ankle
    distance against standing does: a 60° bend shortens it ~12 cm, a lean leaves it.

## Deferred: needs data, hardware or keys (not code)

| Item | Why it can't be done here |
|---|---|
| Bar detector training (J4) | The weights `models/barbell_keypoints.pt` are not in the repo (Q7), and floor-bar frames need labelling. Until then the pipeline runs on the **wrist proxy**: wider D1/D3/D8b thresholds, min tier moderate, no closed-loop foot guidance, no D10 |
| Visibility map, round-1 capture, ρ, hip-keypoint bias (J1) | Needs the rig and lifters |
| Gravity on the rig | Run `scripts/tools/measure_gravity.py` with the board flat. Without the files, the deadlift runs on the body vertical, with D2/D3/D5 raised to moderate |
| `KNOWLEDGE.md` review (J1) | Ambaka, ideally an external strength coach |
| Real validation (J6), thresholds and min tiers from data; freezing `VALIDATION.md` | Needs round-1 data for the **[R1]** values, then the freeze before round-2 labelling |
| CI | `docs/deadlift/CI.md` is a proposal: turning CI on is a team decision |
| Cue audio clips | Needs the TTS key: `generate_cue_audio.py` covers every new key. Until clips exist, the cache falls back to cloud TTS (O3) |
| Jetson timing (§9) | Needs the device |
| J8 ship | `coaching_ready=True` plus the two intentional pin changes listed in PLAN §5.1 |
| Failed reps to the voice agent | A failed rep is an event the analyser counts and logs; the contract has no message for it yet, so Nova does not mention it. Adding one is a contract change for after the demo |

## Tests

- **Biomechanics:** `PYTHONPATH=src python3 -m pytest tests/test_biomechanics -q`. Everything
  passes (3 skipped).
- **Agent side:** `DATABASE_URL=<any postgres URL> PYTHONPATH=src python3 -m pytest tests -q
  --ignore=tests/test_biomechanics`, excluding the modules this container cannot import
  (PortAudio, program-generator data; the exact command is in `docs/deadlift/CI.md`).
  Everything passes except 12 failures that already fail on `main` in this environment:
  1 listener reconnect, 4 rep sound, 7 v6 program verification.
