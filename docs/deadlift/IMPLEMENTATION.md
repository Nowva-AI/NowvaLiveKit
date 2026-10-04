# Conventional deadlift — implementation status

Branch `claude/deadlift-v1-impl`. It implements the software of `docs/deadlift/PLAN.md` (v28).
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

**Reproducing them.** Each number marked [sweep] is printed by
`PYTHONPATH=src python scripts/tools/deadlift_envelope.py <sweep>`, which also prints each row's
bodies and seeds. The keypoint noise is drawn independently for every sweep, row (its noise level
included), body and seed: a row of n sets is n noise draws. `--salt N` draws every row afresh;
a rate quoted as a range ("9–14 of 200") spans salts 0, 1 and 2, and a single number is salt 0,
the default. Numbers marked (test) are pinned by the named test in `tests/test_biomechanics/`;
`tests/test_deadlift_envelope_script.py` checks the script itself.

**Noise models.** Keypoint noise is added at the analyser's input, after where the pipeline's
Kalman sits:
- "i.i.d." is independent per frame.
- "AR(0.8)" is the slow, correlated wander a Kalman-smoothed triangulation leaves. Of the two,
  it is the closer to the platform's documented 1.6–2.4 cm.

"Cued" means at or above the fault's min tier, as the agent would speak it. Every set has 3 mm
of bar noise when the bar is tracked. The five bodies are default, short, tall, narrow hips on a
wide stance, and long femurs.

**Counting, events and clean sets** [events], default body, 12 sets per row (pulls of 0.6, 1.2
and 3 s × seeds 0–3). Event error is over liftoff, knee pass, top and floor.

| Keypoint noise | Tracked, dead stop (36 reps) | Tracked, touch-and-go (4 + 1 dead stop, 60 reps) |
|---|---|---|
| none | 12/12 exact, median 15 ms, p95 67 ms, 0 cued | 12/12, median 18 ms, p95 67 ms, 0 cued |
| 2.0 cm i.i.d. | 12/12, median 19 ms, p95 67 ms, 0 cued | — |
| 2.0 cm AR(0.8) | 12/12, median 19 ms, p95 67 ms, 0 cued | 12/12, median 20 ms, p95 67 ms, 0 cued |
| 2.5 cm i.i.d. | 12/12, median 19 ms, p95 67 ms, 0 cued | — |
| 2.5 cm AR(0.8) | 12/12, median 20 ms, p95 72 ms, 0 cued | — |

| Keypoint noise | Wrist proxy, dead stop (36 reps) | Wrist proxy, touch-and-go (60 reps) |
|---|---|---|
| none | 12/12, median 8 ms, p95 100 ms, 0 cued | 12/12, median 11 ms, p95 100 ms, 0 cued |
| 1.0 cm i.i.d. | 12/12, median 63 ms, p95 167 ms, 0 cued | 12/12, median 44 ms, p95 202 ms, 0 cued |
| 1.0 cm AR(0.8) | 12/12, median 67 ms, p95 228 ms, 0 cued | — |
| 2.0 cm i.i.d. | 12/12, median 107 ms, p95 462 ms, 0 cued | — |
| 2.0 cm AR(0.8) | 12/12, median 67 ms, p95 467 ms, 0 cued | 12/12, median 67 ms, p95 267 ms, 0 cued |

The proxy's events are bound by the wrists' noise: event timing past the 100 ms gate is a
tracked-bar result. On the proxy the top is the hips and knees reaching the lockout (finding
34): noise-free it reads a frame or three early on slow approaches, under noise it is the better
date. A touch-and-go floor event (the low point) can read up to 0.37 s off on the tracked bar.

**Rep counting** [counting]: every set exact for 5-rep touch-and-go sets at 2, 3, 4 and 5 mm of
bar noise (25 random tempos each, pulls 0.6–2 s), 10-rep touch-and-go sets at 3 mm plus 2 cm
AR(0.8) (10 seeds), and fast touch-and-go on the wrist proxy (5 touch-and-go reps, 0.4 s
lowerings, 0.2 s tops; 0.5, 0.6 and 0.8 s pulls; 10 seeds; 2 cm AR(0.8) and 1.5 cm i.i.d.): 0 of
360 reps lost. Before round 4, 0.5–0.6 s proxy pulls lost whole sets. At 5 mm of bar noise a low
point can read as a dead stop: the next rep is then counted as a quick re-pull, not a
touch-and-go, but no rep is lost.

**False cues on clean reps over five bodies** [clean], [no_pause], AR(0.8) keypoint noise,
as ranges over three draws (salts 0–2). `VALIDATION.md`'s gate is 1 false correction per 10
reps.

| Set | 2 cm | 2.5 cm |
|---|---|---|
| Tracked, dead stop, 3 tempos × seeds 0–3 (180 reps) | 0 | 5–7 (D9 and D8) |
| Proxy, 0.6 s holds, seeds 0–5 (90 reps; 0 at 1.5 cm) | 0–1 | 4–7 (mostly D8) |

No pause at the top, seeds 0–7 (a top that never held is judged on the few frames either
side of its peak):

| Set (pull / lowering) | Proxy, 2 cm | Proxy, 2.4 cm | Tracked, 2 cm | Tracked, 2.4 cm |
|---|---|---|---|---|
| Dead stop, 0 s hold, 1.2 / 1.0 s (120 reps) | 1–8 | 6–13 | 0–2 | 5–7 |
| Dead stop, 0.1 s hold, 1.2 / 1.0 s (120 reps) | 1–3 | 4–6 | 1–2 | 2–6 |
| Touch-and-go, 0 s hold, 1.2 / 1.0 s (200 reps) | 7–9 | 18–22 | 1–3 | 8–14 |
| Touch-and-go, 0 s hold, 0.9 / 0.8 s (200 reps) | 6–12 | 17–21 | 1–3 | 7–13 |
| Touch-and-go, 0 s hold, 1.2 / 0.5 s (200 reps) | 4–7 | 17–22 | 3–11 | 10–17 |
| Touch-and-go, 0 s hold, 0.6 / 0.5 s (200 reps) | 7–14 | 24–29 | 4 | 8–17 |
| Touch-and-go, 0.1 s hold, 0.9 / 0.8 s (200 reps) | 3–5 | 7–15 | 1–6 | 4–9 |

Within the gate at 2 cm on every row and draw. At 2.4 cm, the top of the platform's
documented noise, the proxy's touch-and-go reps with no pause at the top reach or pass it
(17–29 of 200, mostly D6 and D8, then D8b and D9), and one draw of its no-pause dead stops does (13 of
120); the tracked bar stays within it (at most 17 of 200). Findings 49 and 52 took the fast
lowering and the 0.6 s pull back inside the gate at 2 cm (the review's draws: proxy 9–26 of
200 at 1.2 / 0.5 s, tracked 35 at 0.6 / 0.5 s). At 2.4 cm, 2 of the proxy's 600 touch-and-go
sets (three draws) lost a rep; the previous commit lost the same two.

**Through the real pipeline and its Kalman** (test, `test_deadlift_pipeline.py`; tracked bar
and proxy): 3/3 reps at 1, 1.5 and 2 cm triangulation noise with the closed-loop STANCE
reached; touch-and-go 5/5 at 0–2 cm on both bar sources; no cued fault on clean proxy reps at
2 cm i.i.d. or 1.5 cm correlated noise; standing references captured at 1–2 cm.

**Other cases** (test, unless marked), each counting every rep:
- **Setups:** setup holds of 0, 0.1, 0.2 and 0.3 s with 0.5 s and 1 s hinges (grip-and-rip).
  Standing up off the bar between reps re-judges each setup.
- **Tempo:** 0.45 s to 5 s pulls with 3 mm bar noise, all events within 100 ms. A 0.8 s
  sticking point mid-pull is ground through and counted. A hitch 90 % of the way up is not
  taken for the top, with or without a standing reference (top within 100 ms; it was ~1 s
  early). [tops]: grinds 4–6 % short for 0.4–0.8 s with their default quick finish are timed
  within one frame (one body), with no lockout fault (they were 0.75–0.93 s early with a false
  moderate D6); a grind finishing its last 2.5 cm over 0.3 s within 0.1 s (24 reps), over 1 s
  with a median of 0.07 s (12 reps; max 0.37 s, its last 0.17 s moving the bar under 2 mm).
  Stalled 3–4 % short with a 1 s finish, the top is early (finding 66): 0.4 s on a median at
  96 % (five bodies, 20 reps; max 0.5 s) and 0.42 s at 97 % (max 0.67 s). 5 s pulls at 2 cm
  AR(0.8) over five bodies within 0.1 s on 86 of 90 (max 0.41 s early; 88 of 90 without
  finding 50's settle rule, on the same draw). On the wrist proxy see `KNOWLEDGE.md` §7.
- **Shrugs at the top** [shrug] (1.2 s holds, every rep shrugged, 4 bodies × seeds 0–3): with a
  standing reference, 1.5–4 cm shrugs move no top more than 0.18 s, the knees seen or hidden,
  except one of 48 at 2.5 cm AR(0.8) (+0.37 s), and read no velocity loss. Without one, a shrug
  that starts before the bar has read still is taken for the top: with the knees seen, on 0
  of 48 reps noise-free and 0–4 of 48 at 2.5 cm (+0.83–0.87 s, D10 in 0–3 of 16 sets); with
  the knees hidden, on 3–4 of 48 (D10 in 3–4 of 16 sets).
- **A lockout settling upward** [shrug] (the bar and upper body 0.8–1.5 cm up from 0.2–0.3 s
  after the top until the lowering, 4 bodies × seeds 0–3), tops more than 0.18 s late and sets
  with a velocity loss:

  | Settle | One rep, noise-free | One rep, 2 cm AR(0.8) | Every rep, noise-free | Every rep, 2 cm AR(0.8) |
  |---|---|---|---|---|
  | 0.8 cm from +0.2 s | 1 of 16, 0 sets | 6 of 16, 4 sets | 14 of 64, 4 sets | 24 of 64, 7 sets |
  | 1.0 cm from +0.3 s | 0 of 16, 0 sets | 4 of 16, 4 sets | 0 of 64, 0 sets | 10 of 64, 7 sets |
  | 1.5 cm from +0.2 s | 0 of 16, 0 sets | 3 of 16, 3 sets | 0 of 64, 0 sets | 9 of 64, 4 sets |

  On 36e6a4d the same rows dated 13–16 of 16 and 54–64 of 64 tops late, with a velocity loss
  in up to 16 of 16 sets. Finding 50 has what is left and why. A lockout that sags 1.2–2 cm
  (the shoulders relaxing) and is tightened again before the lowering, every rep: every top
  within 0.07 s and no velocity loss, noise-free and at 2 cm AR(0.8) (finding 59: on
  36e6a4d, 2–9 of 64 tops ~0.7 s late at 2 cm, a velocity loss in up to 6 of 16 sets).
- **Re-setups at the floor:** feet shuffled 5 cm toward the bar while hinged are judged where
  they now stand (D1 6 → 1 cm measured 6 → 1 cm, at 0 and 2 cm noise); feet the plates hide
  keep the stance's lock.
- **Stance, pelvis and turns (D8)** [hip_shift], 90 reps per row (five bodies × seeds 0–5),
  2 cm AR(0.8), tracked / proxy, false D8 cues: clean stance 1 / 2; lower body 8° off the bar
  0 / 2 (round 3: 32 of 48); left foot 6 cm ahead 1 / 1 (round 4: 10 of 24); hip line turned 4°
  against the legs 0 / 0 (round 5: 10 of 36); a 15° pelvis twist with no sideways travel 0 / 2.
  Turned 20° then squaring up, or turning 10° at the floor, at 1.5 cm: 0–3. Both axes 6° off
  the hips' travel (the stance off the bar, the hip line square to it): 25 / 34, the one case
  two lines cannot tell from a shift (`KNOWLEDGE.md` §7). A real 0.20 shift reads the same with
  the pelvis turning ±10° with it (0.17 / 0.13–0.14 median; round 5 read 0.10 and 0.27).
- **Occlusion:** plates hiding the feet whenever the bar is off the floor, also through the
  pipeline; feet hidden from the setup on: the counting holds and D1 is still judged.
- **Bar dropouts and gaps:** 20 % and 40 % random dropouts, and a bar at 15 Hz; the set never
  switches to the wrists. [gaps], 1.5 cm AR(0.8), the bar lost as the tracker reports it (its
  last velocity carried on for 0.15 s as predicted states, then nothing): at 10 % and 20 %
  dropouts on 2.5 s pulls (3 bodies × seeds 0–3) every top is within 0.11 s. A bar lost for
  0.2–1 s on its way to the top (the loss starting between 0.9 s before it and 0.2 s after;
  12 rows of 27 tops, 3 bodies × seeds 0–2):
  - 1.5 s plain pulls: within 0.13 s on 11 rows. Lost from 0.7 s before the top to 0.3 s
    after, +0.10 to +0.31 s: the last speed, seen mid-pull, finishes slower than the pull.
  - 93 % grinds: within 0.21 s on 9 rows. Lost through the whole finish (from 0.7 or 0.3 s
    before the top to 0.3 s after), up to +0.33 s on 2–3 of 27: the joints date it, through
    the keypoint noise; lost for 0.6 s from 0.1 s before the top, one at +0.49 s.
  - Lost on the way into the stall (0.4 and 0.6 s stalls, from 0.1–0.2 s before them to
    0.1 s after the top): within 0.21 s, 1 of 27 beyond 0.1 s.
  - Touch-and-go tops with no pause, lost 0.2 s either side of reps 2–4's (1.2 / 1.0 s and
    0.9 / 0.8 s, five bodies × seeds 0–3): every set counted, tops within 0.1 s, D6 cued on
    1 of 120 (a 0.9 / 0.8 s rep, finding 82).
  - A 0.6 s hold never seen, the bar lost until 0.8 s after the top (five bodies × seeds 0–2,
    45 tops a row): 1.2, 1.5, 2 and 2.5 s pulls lost from 0.2, 0.3 or 0.4 s before the top,
    medians 0.03–0.10 s, 1 of 540 beyond 0.3 s (−0.41 s: a 2.5 s pull's crawl held still
    before the gap, finding 82), the rest within 0.27 s. The 1.2 s pull lost from 0.4 s
    before reads −0.07 to +0.17 s. 0.6 s pulls lost from 0.2–0.4 s before: medians 0–0.03 s,
    2 of 135 beyond 0.3 s (max +0.37 s). From 0.1 s before the top, within 0.3 s; just after
    it arrived, within 0.15 s. 5 s pulls: medians 0.24–0.43 s, up to 30 of 45 beyond 0.3 s,
    D6 on 4–5 of 45 lost from 0.3–0.4 s before (finding 82). No velocity loss on any (on
    536ad8a: up to +0.57 s, D10 in up to 15 of 15 sets), no D6 below 2.5 s.
  - A grind's finish and hold lost together (stall 0.6 s, finish 0.4 s, lost from 0.2 s before
    the top to 0.8 s after; 3 bodies × seeds 0–2): 93 % and 95 %, 1 of 27 beyond 0.3 s each
    (max +0.37 s), no D6. At 97 %, 10 of 27 never resume (the hips finish such a stall by
    8.5–9°, little more than the resume deficit) and are dated at the stall, a second early,
    with D6 on 4 of 27 (findings 79, 82).
  - Faulted lockouts on 1.2 s pulls held 0.6 s, lost until the lowering (five bodies × seeds
    0–2): a 15° lean-back lost from 0.2 s before the top, or seen arriving for 0.05–0.1 s,
    within 0.2–0.34 s (1 of 135 beyond 0.3 s), D5 cued on 44 of 45 in each row; a 15° soft
    lockout, within 0.15–0.28 s, D6 cued on 42–44 of 45. (On da89661, seen arriving and lost,
    noise-free: up to +0.6 s.)
  - A 1.5 s hold seen for 0.15 s, then lost until 0.2 s into the lowering (clean, a 15° soft
    lockout, no standing reference; 1.5 and 2 cm AR(0.8), five bodies × seeds 0–2): every top
    within 0.08 s, none resumed (on da89661, at 2 cm: up to 9 of 45 resumed, up to +0.96 s).
  - No standing reference, every top of 1.5 s pulls lost from 0.3 s before it: median 0.03 s,
    1 of 45 beyond 0.3 s (+0.4 s).
  - The knees hidden with the bar from 0.2–0.3 s before the top until the lowering (1.2 and
    2 s pulls, five bodies × seeds 0–2): every rep counted, tops within 0.23 s, no cue on 180
    clean reps (on acf0e31: dated at the last frame seen, D6 on every rep; failed reps on a
    tall lifter).
  - The bar lost for 0.1 s every 0.25 s through a 1.5 s hold, no standing reference, 2.5 cm:
    every top within 0.05 s (on acf0e31 a review read holds resumed, up to +1.37 s).
  - A 15 Hz detector, every other frame the tracker's prediction (five bodies × seeds 0–2):
    0.6, 1.2 and 5 s pulls and no-pause touch-and-go 1.2 / 1.0 s, every top within 0.1 s, no
    cue.
  - The wrist proxy with the wrists hidden (five bodies × seeds 0–2): from 0.1 s before the
    top to 0.3 s after, within 0.3 s; through a 0.6 s hold until the lowering, 3 of 45 beyond
    0.3 s (max 0.33 s); 0.2 s either side of a no-pause top, within 0.1 s; no cue on any. A 15°
    soft lockout hidden through its hold is cued on 38 of 45 reps, 45 of 45 seen (finding 65).
  Noise-free (test): losses from 0.5–0.9 s before the top of 1.2 s pulls, into and through a
  grind's stall, across tops that never held and coasted over by the tracker, holds lost
  until the lowering on 0.6–2.5 s pulls (from 0.2–0.4 s before the top to just after it),
  leaned-back and soft lockouts lost before the top or seen arriving, every top lost without
  a standing reference, a bar lost into a stall until the lowering, and a 93–97 % grind's
  finish lost with its hold, within 0.1–0.15 s, with no false velocity loss or D6. At 1.5 cm
  AR(0.8) (test), holds lost from 0.3–0.4 s before the top of 1.2–2.5 s pulls: at most 1 in
  20 beyond 0.3 s, no velocity loss; 97 % grinds lost through their finish, at most half
  beyond 0.3 s. At 2 cm (test), long holds seen then lost (soft, no standing reference) do
  not resume. Through the real tracker at 15 Hz
  (test), every event within 0.1 s.
- **Drops and failures:** dropped bars, with the lifter standing over the bar until the next
  setup; failed reps, including a stall at the knees, are events, not reps.
- **Lockouts:**
  - Over-extended lockouts (25–40° behind vertical) are counted and judged by D5. With no
    standing reference and no earlier top to expect them at they are counted too; D5 needs
    the standing reference to judge them.
  - [lockout], 2 cm AR(0.8), five bodies: a 14° soft lockout (15.5–16.3° noise-free) is cued
    on 70 of 90 reps; a 12° one reads 12.1° (median) against 12.9° noise-free. A 10° one
    (mild, never cued noise-free) is cued on 8 of 60 tracked reps and 15 of 60 proxy reps.
    [tops]: held 2 s, its top is within 0.14 s on a 5 mm tracked bar (90 reps).
  - A lockout 30° short is a failed rep, and so is a pull that leans back 40° from mid-thigh
    with the knees still bent ~60°, whether the knees are seen or hidden. Over-extended
    lockouts with the knees hidden are still counted (leg length against standing).

**Faults:**
- **Injected faults:** with noise-free keypoints, every injected fault (D1–D10) fires at
  moderate or worse on every rep. D2's is a hips-first pull scripted without the setup model.
- **Clean sets:** clean, noisy and tilted-world sets are fault-free.
- **D2 against a held back angle** [d2], which the setup model reads as 6–12° of excess (60°
  to 60°, seeds 0–29): cued on 0 of 90 reps at 2 cm i.i.d. noise and 0 of 90 at 2 cm AR(0.8);
  before round 3 it was cued on 31 and 35 of 90.
- **D2 on a scripted hips-first pull** [d2] (45° to 55°, ratio ~1.24): cued from the set's
  second rep on, on 60 of 90 reps under either noise. A first rep is cued alone only beyond a
  1.30 ratio.
- **D2's rise ratio** (test) reads within 0.04 of the poses' kinematic truth at 0.6, 1.2 and
  3 s pulls (four pull shapes, two bodies). A line through the frames before the knee pass
  read fast pulls up to 0.12 high.
- **The next rep improves** (test): two hips-first reps then three with the chest rising are
  cued on the first two only, and every rep's ratio matches its own truth.

**Gravity:** measured gravity keeps a 3° tilted world from faking drift. Against the body
vertical, the same set fakes more than 2 cm of drift.

**Compute (x86)** [compute], 1.5 cm keypoint noise:
- Analyser: 0.30–0.33 ms median and 0.44–0.53 ms p95 per frame; frames that complete a rep
  (event fits, the final climb, features, D8's two axes) 2.9–3.1 ms median on default pulls
  and 4.6 ms (tracked) / 5.1 ms (proxy) on 5 s pulls with 2 s holds. The first frames of
  a process pay numpy's warm-up once (~12–18 ms).
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

31. **The lockout window is chosen by time, never by the angles it judges** (review, round 6;
    replaces finding 28's window). The hold's most extended 0.3 s is the minimum over windows
    of the very angles D6 judges: under keypoint noise it selected the noise, reading a 14°
    soft lockout ~4° straighter (cued on 8 of 18 reps at 2 cm AR(0.8)) and biasing D5. D6 and
    D5 now read the hold's last 0.5 s before the lowering: 18 of 18 cued on the default body
    (median 15.9°; 70 of 90 over five bodies with independent noise, the `lockout` sweep).

32. **A noisy median must not resume a held lockout** (review, round 6; replaces finding 29's
    rule). The resume gate opens on every soft lockout (a D6 fault), and a 0.2 s median then
    cleared the hold's median by one noise band at random: soft lockouts on the wrist proxy
    resumed and were re-dated to the end of the hold (+1.1 s median), and on a 5 mm tracked bar
    by up to +1.4 s, which the recap would read as velocity loss. The resume is back to one
    frame clearing the whole hold band, behind the deficit gate. A grind inside the band stays
    in TOP, and the features time it: the final climb starts at the end of the last flat stall
    short of the lockout (the running median within the band for 0.3 s, the hips and knees
    ≥ 12° more bent than at the lockout; a velocity test read a slow 2.5 cm/s climb as still,
    and 8° found false stalls in 5 s pulls), and the top is fitted over its upper half (a climb
    out of a stall starts flat: fitting all of it made slow finishes ~0.13 s late). After a
    resume the top had been the band arrival with no fit, 0.27–0.47 s early on a grind that
    finishes slowly and up to 0.47 s early on 5 s pulls under 2 cm AR(0.8); now the median is
    0.145 s on the slow finish (its last 0.17 s move the bar under 2 mm, inside its noise),
    and every 5 s pull is within 92 ms.

33. **The lowering takes the bar out of the hold.** A 2.5 cm shrug coming back down began the
    lowering at the shrug, so the lockout window (finding 31) sat on the shrug and its level
    dated the top 0.4–0.9 s late; and on the wrist proxy one noisy frame below the peak sent
    PULL straight to LOWER, leaving a hold of 7 frames around a noise peak whose level dated
    the top 0.4 s late. Both transitions now
    need the bar's 5-frame median below the hold by the hold band, and date the lowering back
    to the bar's last frame at the top.

34. **On the wrist proxy the joints date the top.** The wrists' event band (3 × their rest
    noise, 1.3–5.6 cm at 1.5 cm AR(0.8)) is wider than the event fit's 4 cm span, so the fit
    failed on 41 of 60 proxy tops and fell back to the first frame inside the band: slow 5 s
    pulls 0.3–0.65 s early (median, v17–v18), grinds 3 cm short dated at the stall once the
    median resume was gone (to −1.3 s, 5 of 18 beyond 0.3 s), and a hold whose level wandered
    up dated late. In the simulator, as in a
    lift, the top is where the hips and knees finish extending; on the proxy it is now dated
    there (§2.3, KNOWLEDGE §2). At 1.5 cm AR(0.8), 30 reps each: plain 67 ms median (max
    0.17 s), grinds 33 ms (0 of 30 beyond 0.3 s), soft lockouts 67 ms (max 0.30 s), slow pulls
    0.23 s early. Noise-free, proxy events read 8 ms median (p95 100 ms; it was 2 ms): the 5°
    margin dates a slow approach a frame or three early. The tracked bar keeps the bar fit.

35. **An over-extended lockout with no top to expect is a lockout.** With no standing
    reference and no earlier top, only "trunk within 35° of vertical" made a top, so a 40°
    lean-back with straight, visible knees was a failed rep. The straight-legs lean-back now
    counts there too.

36. **Without a standing reference, a shrug is not a stall.** The deficit gate had nothing to
    measure against, so the bar alone decided: a 2.5–4 cm shrug cleared the 2 cm hold band,
    resumed the pull and moved the top 0.8 s later (D10 severe on a later rep). The gate now
    asks, without a standing reference, whether the hips or knees extended ≥ 8° since the
    hold as the bar rose: a stall finishes that way, a shrug rises on straight legs (top
    within 0.02 s; hitches 90 % of the way up still resume).

37. **One wrist missing on the proxy falls back to the feet's line.** With no hands' line, the
    bar's axis fell back to the hip line, so D8's consensus read one line twice and a hip line
    turned 4° against the legs read as shift. The ankle line, independent of the hips, now
    stands in.

38. **A dropped bar frame is no height** (review, round 7). A dropped frame stays in the rep
    with a NaN height; the running median turned every window touching it into NaN, and a
    window starting with NaN passed the stall's flatness test (`max`/`min` of a list starting
    with NaN are NaN). So a "stall" was found mid-pull, and the climb fitted from there: at
    10 % dropouts 11 of 36 tracked tops were beyond 100 ms. The running median now skips NaN
    and the flatness test reads finite heights: 0 of 36 at 10 and 20 %.

39. **The top is fitted from below, on the climb and 0.2 s of the hold** (review, round 7).
    After a stall a few cm short, the climb's upper half is three or four frames, against a
    whole hold whose distance from the level (noise above it counted too) a late-ending
    parabola explains better: grinds with a 0.3 s finish read up to 0.47 s late. Frames
    above the level now count as arrived, and the fit keeps the hold's first 0.2 s: within
    0.2 s on 24 of 24, the slow finish's median 0.07 s (was 0.145 s), 5 s pulls within 76 ms
    on the default body (within 0.1 s on 88 of 90 over five bodies, the `tops` sweep).

40. **On the wrist proxy the lowering is dated within the noise, not an event band**
    (review, round 7). An event band (3 × the wrists' rest noise, 3–5 cm) reached 0.13–0.17 s
    into the lowering, so the lockout window judged knees and hips already flexing: a 10°
    soft lockout (mild, never cued) was cued as moderate on 21 of 60 proxy reps against 5 of
    60 tracked. The proxy's lowering now begins at the last frame whose 5-frame median sat
    within one noise sigma of the hold's last 0.3 s; the peak path's frames are cut there too.

41. **A stall resumes only on evidence from the joints** (review, round 7). With the knees
    hidden the deficit is NaN, and `NaN < 8` is False, so the gate stood open: a shrug at the
    top resumed the pull and moved the top 0.9 s (D10 severe). Without a standing reference,
    either joint's 5-frame median against the hold opened it on keypoint noise (4–12 of 48
    shrugs at 2–2.5 cm). Unknown now closes the gate, and without a standing reference the
    hips *and* the knees must both have extended. Hitches 90 % of the way up still resume.

42. **The proxy's bar tilt needs the set to agree.** The hands' height difference carried out
    to the hubs reaches D8b's severe threshold on clean reps (3–18 per 100 with no pause at
    the top, at 2–2.5 cm), each rep tilted its own way. A lifter's uneven pull repeats, so on
    the proxy D8b reads the smaller of this rep's tilt and the set's last 3 reps' median to
    the same side, and nothing on a set's first rep.

43. **No bar speed on the wrist proxy.** D10 was already off there, but the proxy's speed still
    reached the recap ("fastest on rep …", "… percent slower"), the spoken summary ("the bar
    came up at about …") and the diagnosis's load cause. Its top is now the joints' and its
    rise and liftoff the wrists': `concentric_velocity_mps` is NaN on the proxy, which every
    consumer already reads as unmeasured.

44. **The running median is one vectorised call.** Per-window `np.median` cost ~27 µs a frame,
    twice per rep: rep-completing frames on a 5 s pull with a 2 s hold went from 8–10 ms to
    4–5 ms.

45. **A top that never held is judged around its peak, not the highest noisy frame**
    (review, round 8). On the wrist proxy the highest frame is the highest of the wrists'
    noise, up to ~0.15 s before the top with the knees still 19–32° bent: touch-and-go reps
    at 2 cm AR(0.8) drew a false D6 on 14 of 200 (2 severe) against 1 of 200 tracked. The
    peak is now the vertex of a parabola through the bar's height within 0.3 s of the
    highest frame.

46. **The lockout's level is its window's lowest plateau.** A shrug inside the hold band
    (1.5 cm) stays in the hold, and the last 0.5 s caught its way down: their median sat up
    to ~1 cm above the lockout, so the bar "arrived" during the shrug (13 of 48 tops up to
    0.63 s late, with a standing reference). The level is now the median of the frames whose
    running median is within the event band of the window's lowest.

47. **A bar lost as it arrives is dated in the gap.** The fit could not date the top before
    the first frame the bar was seen again: a 0.4 s loss from 0.1 s before the top read
    +0.33 s, with a false velocity loss in the recap (a review measured up to +0.70 s). The
    fit now starts at the gap's first frame. With no fit the top was the gap's middle, which
    finding 51 replaced.

48. **The envelope is a committed script with independent noise.** The tables were drawn
    with one noise sequence per seed, shared by every body in a row, and several quoted rates
    did not hold on other draws (review, round 8). Every number marked [sweep] in this file
    and `KNOWLEDGE.md` §7 now comes from `scripts/tools/deadlift_envelope.py`, which draws the
    noise per sweep, row, body and seed and prints each row's bodies and seed range. Numbers
    marked (test) come from the named tests, and those credited to a review from its
    measurements.

49. **A proxy peak that never held is fitted with each side's own curvature** (review,
    round 9). A touch-and-go is usually lowered faster than it is pulled, so the wrists'
    height around the peak is lopsided, and finding 45's symmetric parabola put the vertex
    0.07–0.10 s early, the knees still bending. Noise-free, the proxy read the lockout 4.7°
    short at a 1.2 s pull and 0.5 s lowering (2.2–2.7° at 1.5 / 0.5 s and 2.0 / 0.6 s; the
    tracked bar reads 0 to −1.5°), and at 2 cm AR(0.8) those reps drew D6 on 18 of 200 against
    0 tracked (the test's draw). The vertex is now refined within 0.1 s by a parabola with its
    own curvature on each side (the steeper at most 8× the other), through the bar's 5-frame
    running median: fitted to the raw heights, the two-sided fit followed the noise (on round
    8's layout at 2.4 cm, 1 severe D6). Noise-free the proxy now reads the tracked bar's
    lockout at all three tempos, and the test's draw cues 3 of 200 (test).

50. **A lockout that settles upward keeps its first plateau's level** (review, round 9). The
    shoulders drawn back on straight legs lift the bar 0.8–1.5 cm, inside the hold band, and
    keep it there until the lowering. The lockout window (the hold's last 0.5 s) then sat on
    the raised plateau, and the top was dated where the bar reached it, 0.35–0.9 s late, with
    a velocity loss in the recap in 7–16 of 16 sets (review). Below the window's level by more
    than the band, a stretch flat for 0.2 s with the hips and knees within 5° of the lockout's
    is now that lockout already reached (a stall is ≥ 12° more bent), and the level is the
    first such plateau after the final climb began. Noise-free, 1 and 1.5 cm settles on one
    rep or every rep date every top within 0.05 s with no velocity loss [shrug] (test: 1 cm,
    five bodies, within 0.1 s). What it does not reach:
    - A settle smaller than the event band (3 × the rest's noise, 0.5–0.9 cm on a 3 mm bar)
      stays inside the hold: 0.8 cm on every rep still dates 14 of 64 tops up to 0.41 s late,
      with a velocity loss in 4 of 16 sets.
    - At 2 cm AR(0.8) the joints' 5° test misses some settles: 3–6 of 16 tops late with one
      rep settling, 9–24 of 64 with every rep, and a velocity loss in 3–7 of 16 sets.
    - It costs slow pulls a little. A 5 s pull's last centimetres can sit flat for 0.2 s,
      and through 2 cm of keypoint noise its joints can pass the 5° test: on the [tops] draw
      2 more of 90 tracked tops land beyond 0.1 s (up to 0.41 s early).
    A settle depth of half the band caught the 0.8 cm settle but put 10 of 90 slow pulls
    beyond 0.1 s; a tighter joints test (2–4°) changed neither measurably.

51. **A bar lost on its way to the top is dated by its last speed, or by the joints**
    (review, round 9). Finding 47's gap middle is far before the arrival when the bar is lost
    low on the climb: lost from 0.9 s before the top of a 1.2 s pull to 0.1 s after, the top
    read 0.40 s early and the recap read a velocity loss in 9 of 12 sets. With no fit:
    - A bar slowing evenly from its last seen speed into the top covers the rest in twice
      the time that speed would; the arrival is there, from the gap's first frame on.
    - When that does not land inside the gap (the bar lost still in a grind's stall, or
      speeding up out of it), the hips and knees, still seen, reaching the lockout date it.
    - After a gap long enough to hide a stall (over 0.3 s), a bar seen again at its level
      had arrived by then, which bounds the fit too: a 1 s loss from before a grind's stall
      had let the fit run over the whole climb and date the top up to 0.5 s late. A shorter
      gap keeps the fit's usual reach (bounded at every dropped frame, a slow approach's
      last centimetres inside the band read 0.17 s early).
    Noise-free, losses of 0.5–0.9 s before the top on 1.2 s pulls, and losses through a
    grind's finish, date the top within 0.15 s with no false velocity loss (test). Tried on
    the [gaps] rows and not kept: the joints alone, worse than the bar's speed on plain
    pulls.

52. **A fast top that never held is judged on a narrower window** (review, round 9). On a
    0.6 s pull the knees are still bending 0.1 s either side of the top: noise-free, a
    touch-and-go with no pause read the lockout 7.5° short on the tracked bar, and at 2 cm
    AR(0.8) cued D6 on 24 of 200 tracked reps (a review's draw: 35). The window is now 0.1 s
    either side of the peak, or 0.11 of the pull when that is shorter (pulls under ~0.9 s):
    noise-free the same reps read −1° on both bar sources (test).

53. **The envelope's draws carry their noise level and can be redrawn** (review, round 9).
    Rows of one sweep at different noise levels shared a draw, and one draw's rate could
    differ ~2× from another's. The keys now include the noise level, `--salt N` draws every
    row afresh (the false-cue rates above are ranges over three draws), and
    `tests/test_deadlift_envelope_script.py` checks the draws' independence and runs a sweep
    end to end.

54. **The features' median is a sort.** Finding 50's joints test runs on every flat stretch,
    and the wrists' noise makes a slow proxy pull flat almost throughout: rep-completing
    frames on proxy 5 s pulls with 2 s holds went from 6.4 to 9.8 ms, nearly all of it
    `np.median`'s call on windows of a few frames. `statistics.median` returns the same values
    (every feature identical on 30 noisy sets, both bar sources) in 5.5 ms.

55. **A gap is no hold** (review, round 10). With fewer than two seen frames in its window
    the bar's velocity read 0, so a bar unseen across a top counted as still and the rep
    entered TOP inside the gap. On a top that never held, the top was then dated at the last
    frame seen and the lockout judged on the climb: noise-free, a 0.4 s loss across 1.2 s
    touch-and-go tops with no pause cued D6 on 30 of 30 reps. An unseen frame now neither
    starts nor breaks a hold.

56. **A top lost from view on its way up is dated by the joints** (review, round 10). The
    peak of a top that never held is the highest frame seen; lost on the way up and seen
    again coming down, that frame is on the climb. On the tracked bar the peak is now where
    the hips and knees were most extended among the frames the bar went unseen around its
    highest one: the lockout is judged there and the top dated there. Seen again higher, the
    bar arrived in or after the gap, which the features date (finding 51; the joints' most
    extended frame is anywhere in a hold). If the highest frame seen sits short of the
    expected top, the lifter seen standing while the bar was unseen makes it a top: 0.9 s
    pulls had merged three reps into one in 10 of 10 sets. Noise-free (test), for a bar
    dropped or coasted over: within 0.1 s, D6 on at most 1 in 10, every rep counted.

57. **A coasting prediction is no height** (review, round 10). The tracker carries a lost bar
    on at its last velocity for 0.15 s and the pipeline passes those predicted states on;
    near the top they overshoot a bar that stops, and the overshoot became the rep's peak
    (noise-free: the top 0.13–0.17 s early, a false D6 on 4 of 12). A prediction now keeps
    its height only on the frame between two detections of a 15 Hz detector (0.05 s after
    the last measurement), and is geometry only after that. Giving no prediction a height
    would halve a 15 Hz detector's heights: event medians stayed within 100 ms, but a 5 s
    pull's worst top went from 0.20 to 0.30 s (a probe, every other frame dropped). The
    top's level, final climb, arrival, fit and gap rules read measured heights only: one
    coasted frame inside the band had become the arrival, or moved the level, and grinds
    read 0.3–0.6 s late. The gap tests and the [gaps] rows now emulate the tracker's
    coasting, and read the same as with the bar dropped outright (test: within 0.1 s, no
    D6). Round 12 dropped the one-frame exception (finding 62).

58. **A bar lost on its way into a stall is dated by the joints** (review, round 10).
    Finding 51's speed estimate assumes the bar slows evenly into the top; lost while still
    moving into a grind's stall, it put the arrival in the stall, 0.3–0.8 s early (worse
    than the gap's middle). Where the hips and knees over the 0.15 s after the estimate are
    still clearly bent past the lockout (20°, the bound the wrist proxy already uses; a
    stall 3–7 % short reads 25–45°), the estimate gives way to the joints reaching the
    lockout. The stall's own 12° bound vetoed good estimates: through 1.5 cm of correlated
    keypoint noise the joints drift up to ~15° in 0.3 s, and held tops read 0.2–0.33 s
    late. Compared with the joints where the bar was seen again instead of the lockout
    window, a bar seen again just short of the top, still rising, dated it 0.2 s early.
    Noise-free (test): within 0.15 s.

59. **A sag is neither a settle nor a stall** (review, round 10). Finding 50 read any
    straight-legged plateau below the hold as the lockout, so a lockout that sagged 1.2–2 cm
    (the shoulders relaxing) and was tightened again was dated by the sag, 0.13–0.17 s early
    (18–48 of 64 tops beyond 0.1 s, noise-free). Only a plateau that began before the bar
    first reached the window's level is a settle now. Under 2 cm AR(0.8) the same sag had
    another, older failure (on 36e6a4d too): the joints on its plateau could read a stall's
    12°, and the tightening back up became the last climb, 9 of 64 tops ~0.7 s late with a
    velocity loss in 6 of 16 sets. A stall now ends before the bar first reaches the level
    (tests: noise-free within 0.1 s; at 2 cm, at most 1 of 64 beyond 0.15 s, no D10).

60. **The analyser's median is a sort too**, as finding 54's (every feature identical on 30
    noisy sets).

61. **A hold lost from view is dated by the bar's last rise** (review, round 11). Finding 55
    made a gap no hold, so a top whose whole hold went unseen took finding 56's path, and the
    joints' most extended frame dated it. Through a hold the joints are flat and that frame
    can be anywhere in it: 0.6 s holds lost until the lowering read +0.23 to +0.57 s late,
    with a velocity loss (D10) in up to 15 of 15 sets, and 0.6 s pulls with every other bar
    frame dropped the same. Such a top was then dated where the bar's last measured rise
    arrives: the mean of a parabola's vertex through the last 0.3 s of the rise and the last
    speed slowing evenly into the expected top height, kept within the frames it went unseen
    on. Where the hips and knees over the 0.15 s after it are still 20° bent past their most
    extended (lost on the way into a stall), or the bar was not slowing, the top is where
    they first came within 5° of their most extended. A top counts as lost from view after
    two unseen frames: one is a 15 Hz detector's skipped detection. That estimator held only
    on the 1.2 s pulls it was tried on, and the numbers given here for its two terms did not
    reproduce (a review measured the vertex alone at +0.08 to +0.42 s, the speed alone at
    −0.11 to +0.02 s on that row): finding 69 replaced it.

62. **No predicted bar state has a height** (review, round 11). Finding 57 kept a
    prediction's height on the frame between two detections of a 15 Hz detector. Through the
    real `BarTracker3D` at 15 Hz that frame overshoots a bar that stops by ~1 cm, became the
    peak of tops that never held, and dated them 1–4 frames early (at 2 cm AR(0.8), 70 of 200
    beyond 0.1 s). Every prediction is now geometry only. Finding 57's case for the exception
    (with half the heights a 5 s pull's worst top went from 0.20 to 0.30 s) did not hold
    through the tracker's coasting: [gaps], a 15 Hz detector at 1.5 cm AR(0.8), five bodies,
    0.6, 1.2 and 5 s pulls and no-pause touch-and-go, every top within 0.1 s, no cue. With
    half the heights, the lockout window of a top that never held has two to four, and the
    lowest of them is a side of the peak: their "lowest plateau" dated tops early. With fewer
    measured heights than the 5-frame running median takes, the level is now their median
    (the peak, or their median within the band of the highest, sat high on a hold seen for
    a few frames: a 93 % grind's top read 0.27 s late). Tests (`test_deadlift_bar_tracker.py`,
    through the real tracker at 15 Hz): no-pause touch-and-go 1.2 / 1.0 s and held 0.45, 1.2
    and 5 s pulls, every event within 0.1 s.

63. **Seen again just short of its level, the bar may have arrived in the gap** (found
    regenerating round 12's rows). A gap right before the bar's first frame at its level is
    where the bar may have arrived; a first frame back a hair below that band (0.3 mm, on a
    hold seen for its last frames) was not linked to the gap, and its top read 0.57–0.63 s
    late. A bar seen again within a second band of the level now links the gap the same way.

64. **A lockout is judged when the bar was first seen leaving late.** The lockout is judged
    on the top's frames within 0.5 s of where the bar left the top. With the wrists hidden
    through a hold and seen again on the lowering, the proxy dated the lowering after the
    gap, that window held none of the top's frames, and the lockout went unjudged (no
    deficit, so a soft lockout went uncued). When the bar went unseen right after the top's
    frames, they are all judged. (Applied to any top with no frame in the window, it judged
    a clean proxy rep whose wrists' noise had left the hold band, and cued it.) Narrowed again
    by finding 73.

65. **The wrists lost beside their peak** (review, round 11). On the proxy with the wrists
    hidden at 1.5 cm AR(0.8), a no-pause top hidden 0.2 s either side drew a false D6 on 21 of
    75 reps and a held top hidden through its hold 9 of 45 (the review's draws, on 536ad8a
    and before). A gap of two frames or more beside the wrists' highest frame, after it or
    before it (seen again coming down higher than last seen going up), now puts the top in
    the gap: the lockout is judged where the hips and knees were most extended while the
    wrists were unseen (in a hold any of its frames is a lockout frame), the lifter seen
    standing then makes it a top, and the joints date the arrival from the gap's start rather
    than where the wrists were seen again (hidden from 0.1 s before the top to 0.3 s after,
    15 of 45 tops had read beyond 0.3 s, up to 0.43 s). The window is chosen by the angles it
    judges, which reads straight: a 15° soft lockout hidden through its hold is cued on 38–40
    of 45 reps (43–45 seen). The review's probe now: no false D6 on 75 and 45 reps; 2 of 45
    tops beyond 0.3 s.

66. **Slow finishes out of a stall 3–4 % short stay a limit** (review, round 11; pre-existing).
    A last climb of 1–2 cm over a second has too few frames clear of the noise band to fit,
    and its top is the bar's first frame at its level: at 96–97 %, 0.4 s early on a median, up
    to 0.67 s ([tops], five bodies). Three fallbacks were tried and dropped: a fit on the
    whole climb (96 % median 0.09 s, but grinds 4–6 % short with a quick finish, timed within a
    frame, read up to 0.11 s late); the hips and knees reaching the lockout (0.17–0.2 s
    noise-free, but at 2 cm of keypoint noise a 95 % grind's top, within a frame, up to 0.3 s
    late); the same for slow climbs only (no speed tells the crawl apart: its stall is read to
    end late, leaving a climb of a centimetre or two). "Within one frame" in this document is
    for grinds 4–6 % short with their default quick finish, one body.

67. **Frame counts assume 30 fps.** The running medians (5 frames), the event fits' 3 frames
    clear of the band, a lost top's 2 unseen frames and the dead stop's 3 count frames, tuned
    on a 30 fps pose stream. The plan's first degraded mode (pose at 30 Hz, the bar detector at
    15 Hz) is tested; its second, everything at 15 Hz, is not (a review read a 93 % grind
    +0.43 s late there): it needs those counts in seconds first.

68. **Housekeeping** (review, round 11): a pull resumed after a top lost from view clears its
    unseen flag; the analyser imports the features' `median` and `running_median` instead of
    keeping copies; long lines wrapped.

69. **A lost top is dated by a line through the square root of the bar's last rise** (review,
    round 12). Finding 61's estimator held only on the 1.2 s pulls it was tried on: noise-free,
    a 2 s pull lost from 0.4 s before its top read up to +0.56 s, and its four-frame speed
    scattered ±20 %. A bar slowing evenly into the expected top height H has √(H − h) falling
    on a straight line to zero at the top; a line through the last 0.3 s of the rise is fitted,
    its zero (no later than the gap's end) is the top, and it is used when
    the bar was still rising into the gap (the line's speed at its last frame at least the
    0.05 m/s that holds a top still). The joints' 20° veto of finding 61 stays (there, the
    hips and knees first within 5° of their most extended date it), read against their
    plateau in the gap, their 20th percentile: through a hold, keypoint noise wanders them
    by ~15°, and against their least (a dip) a slow pull's last extension read as a stall
    and the top as late as +0.47 s (2 cm AR(0.8), 1.5 s pulls; at 1.5 cm it was the [gaps]
    grid's whole tail beyond 0.3 s, 3 of 585 at +0.50–0.53 s; 21 vetoes on a 1,620-top grid at
    0–2 cm became 7, the into-a-stall vetoes unchanged). A top whose bar was not
    rising into the gap (it arrived, then went unseen) takes the seen path. [gaps], 1.2–2.5 s
    pulls lost from 0.2–0.4 s before the top: medians 0.03–0.13 s, none of 540 beyond 0.3 s.
    Tests: noise-free, 1.2–2.5 s pulls lost from 0.3–0.4 s before the top (1.2 s pulls from
    0.3 s) to just after it, within 0.15 s; at 1.5 cm AR(0.8), at most 1 in 20 beyond 0.3 s;
    the wandering draw at 2 cm, within 0.3 s. Round 13's review found the fit held only on
    that grid (soft and leaned-back lockouts, quick pulls, bars seen arriving): finding 77
    makes the hips and knees the date and the fit a refinement of it.

70. **Seen again coming down higher** (review, round 12). A 1.2 s pull lost from 0.3–0.4 s
    before its top and seen again on the lowering higher than last seen on the climb had its
    peak on the lowering: its level came from a lowering frame, the speed estimate was vetoed,
    and the joints dated it −0.25 to +0.5 s, with a velocity loss in 3–7 sets. A gap of two
    frames or more before the peak (finding 65's rule, for the bar) now puts the top in the
    gap, fitted on the climb up to it. On a seen bar, the anchor is the last frame of the rise
    within the event band of the peak, so a frame a noise sigma under it does not hide the gap
    that follows. [gaps] rows from 0.3–0.4 s before, above. (That walk is gone, finding 80.)

71. **A top dated in a gap has no bar speed** (review, round 12). A top inside a gap has no
    measured concentric time: its speed came from an estimate, and a late one read as a
    velocity loss. Such a rep now has no concentric velocity and D10 skips it. Test: the rep
    whose top went unseen reads no speed; the two either side do.

72. **A grind lost through its finish and hold resumes on the joints** (review, round 12;
    pre-existing). A stall seen and held enters TOP; with the bar unseen for the finish and the
    hold, nothing could resume the pull, the top was dated at the stall (0.7–1.04 s early) and
    D6 judged the stall's bent frames (9 of 9). In TOP, while the bar is unseen, the hips and
    knees straightening by the resume deficit (8°) past the held stall now resume the pull,
    with the climb marked unseen: the top is where the hips and knees first come within 5° of
    their plateau (the 20th percentile of the gap's angles, so noise below the plateau does
    not set it). [gaps], stall 0.6 s, finish 0.4 s, lost from 0.2 s before the top to 0.8 s
    after: 93 % and 95 %, at most 2 of 27 beyond 0.3 s (max +0.48 s), no D6. A 97 % stall
    straightens less than 8° to the lockout under noise: 6 of 27 never resume, are dated at
    the stall, and 2 of 27 draw D6 (finding 75). Tests: 93, 95 and 97 % noise-free, within
    0.15 s, no D6. Frame by frame, the keypoint noise of a long unseen hold resumed it too
    (round 13's review): finding 79 decides it once, over the gap.

73. **Withdrawn.** It narrowed finding 64's rule (all of a top's frames judged when the bar
    went unseen right after them) to frames ending in the gap, for a 5 s crawl's bent frames
    (D6 on 8–11 of 45). Those numbers were measured on an intermediate state: on the final
    code, round 13's review found reverting the narrowing changed no row and no test.
    Finding 64's rule is back.

74. **Closed-loop foot guidance stops after the set's first rep** (review, round 12). It was
    gated on the tops' heights recorded, and a top lost from view records none: after such a
    rep guidance ran for 106 frames. It is now gated on a count of the set's reps. Test: no
    live offset after a rep whose top went unseen.

75. **Limits of the lost-top fit** (measured, round 13). (1) A 5 s pull's top lost from view:
    TOP is entered on the crawl before the gap, and the top is the crawl's end, not the fit's:
    medians 0.24–0.43 s, up to 29 of 45 beyond 0.3 s, D6 on 3 of 45 lost from 0.3–0.4 s before
    the top. (2) A 1.2 s pull lost from 0.4 s before the top is extrapolated over a third of
    the pull, +0.04 to +0.20 s. (3) A grind stalled 3 % short straightens too little to resume
    under noise (finding 72). (4) On a slow pull the fit can land 0.1–0.15 s early where the
    hips and knees, wandering with the noise, still read 20° past their plateau: the veto
    then dates it by them, late (none on the [gaps] grid; the round-12 review's draws at
    1.5 cm, 1 of 135 at +0.33 s on 1.5–2.5 s pulls). The fit leans on the expected top
    height, the standing pose's, which carries about 1 cm of the keypoint noise; the grid's
    spread is within that. Each needs real lifts to say how often it occurs; none is cued
    falsely on a faster pull.

76. **Housekeeping** (review, round 12): the analyser's copy of the gap speed constant is gone
    (it uses the fit); the hips' and knees' flexion track is one helper; long lines wrapped. A
    test pins the gap-linking walk of finding 63 (it fails without it).

77. **A lost top is dated by the hips and knees, refined by the bar's fit** (review, round
    13). The fit of finding 69 alone held only on the grid it was tried on. It runs into the
    expected top height, which a 15° leaned-back or soft lockout never reaches (seen arriving
    and then lost, noise-free: 32 and 22 of 45 beyond 0.3 s, up to +0.6 s); and it assumes a
    bar slowing evenly into the top, which a 0.6 s pull is not until its last frames (lost
    from 0.3 s before the top: +0.45 to +0.59 s on 45 of 45). A top lost from view is now
    where the hips and knees first come within 5° of their plateau in the gap (its 20th
    percentile); the fit dates it instead unless the joints are still 20° bent past the
    plateau at the fitted top (lost on the way into a stall: the fit lands in the stall,
    −0.78 s) or reached it more than 0.15 s before (the fit overshoots). The veto reads the
    joints at the fitted top, not over the 0.15 s after it: after a no-pause top they bend
    with the lowering, and the veto dated those tops 0.1 s early, onto the climb. The rule
    was chosen with a development probe (not in the repository) over the [gaps] cases at
    0–2 cm, against the joints alone and the fit unless the two disagree either way: it put
    about half as many tops beyond 0.3 s as either, and none noise-free. The reproducible
    measure is [gaps] and the tests below. A top seen arriving is
    dated there: the joints are at their plateau at the anchor (a check sending such tops to
    the seen path, tried, changed no maximum and one median, 0.05 against 0.12 s, and was
    dropped). Without a standing reference there is no
    expected height to fit: the joints date it. [gaps], 1.2–2.5 s
    pulls lost from 0.2–0.4 s before the top: medians 0.03–0.10 s, 1 of 540 beyond 0.3 s (a
    crawl, finding 82); 0.6 s pulls, 2 of 135 (on da89661, noise-free, 45 of 45 lost from
    0.3 s before read +0.45 to +0.59 s).
    Tests: holds lost until the lowering, 0.6–2.5 s pulls; leaned-back and soft lockouts lost
    before the top or seen arriving; no standing reference; a bar lost into a stall until the
    lowering; all noise-free, within 0.15 s.

78. **A lost top's lockout window ends where the hips and knees bend into the lowering.**
    Unseen across the top, the window keeps the frames after it (a hold). After a no-pause
    top dated a frame late, those frames were the lowering, quicker than the pull: the knees
    read 10.7° short (the test's 6°). The window now ends at the first frame after the top
    where the hips and knees bend 20° past their most extended within it; a hold never does.
    Test: the no-pause top lost either side, within 6°.

79. **A resume with the bar unseen is decided once, over the gap** (review, round 13).
    Finding 72's resume ran frame by frame on 5-frame medians: over a long unseen hold the
    keypoint noise reached the deficit on both joints (1.5 s holds seen 0.15 s then lost, 2 cm
    AR(0.8): 8 of 45 soft lockouts and 9 of 45 holds without a standing reference resumed,
    tops up to +0.96 s late). Now, when the bar is seen again after a gap, the hips and knees'
    medians over the whole gap against the hold's resume the pull if they extended by twice
    the resume deficit together (16°). Requiring the deficit of each, as seen, resumed fewer
    97 % grinds (the hips finish a stall 3 % short by 8.5–9°: the test's draw at 1.5 cm, 19
    of 27 beyond 0.3 s against 10 with the sum). [gaps]: long holds seen then
    lost at 1.5 and 2 cm, none resumed; 97 % grinds, 10 of 27 never resume (6 frame by frame,
    which also resumed the noise). Tests: soft
    lockouts and no standing reference, 2 cm, none beyond 0.3 s (frame by frame, 0.33–1.04 s
    on 11 of 90); 97 % grinds at 1.5 cm, at most half beyond 0.3 s (each joint by the deficit:
    19 of 27); the noise-free grinds of finding 72.

80. **One rule for a top lost from view, on both bar sources** (review, round 13). Finding 70's
    anchor walk over frames within the event band of the peak let a bar seen arriving count as
    lost, and on the wrist proxy, whose band is 3–5 cm, it walked deep into the lowering: D6
    on 45 of 225 clean no-pause proxy reps with the wrists hidden just after the peak (29 on
    92fe516). The walk is gone (it changed 1 of 630 tops on the hold grid): a top is lost
    after two unseen frames right after the highest one, or before it (seen again higher),
    for both sources. The proxy's own rate there is pre-existing and over the 1-in-10 gate
    (finding 82).

81. **Housekeeping** (review, round 13): `_unseen_top_time` takes the anchor and the gap's end
    (the expected height from the analyser); the climb-unseen flag is gone (a resume after
    the gap leaves the highest frame where it is, and the gap rules date it); the 0.05 m/s
    gate on the fit is gone (a line not reaching zero ahead is no fit; the joints check the
    rest); finding 73 withdrawn.

82. **Limits** (measured, round 14). (1) A slow pull's crawl into its top, held still
    before the gap (|v| < 0.05 m/s for 3 frames), enters TOP there, and the top is the
    crawl's end: 5 s pulls lost from 0.2–0.4 s before the top, medians 0.24–0.43 s, D6 on 4–5
    of 45 (the crawl's bent frames); on 2.5 s pulls, 1 of 135 (−0.41 s, D6). The frame-by-frame
    resume of round 13 rescued some of these (3 of 45 D6), along with the noise; the gap's
    hips and knees past the crawl are usually short of twice the resume deficit. (2) A grind
    3 % short lost through its finish: 10 of 27 never resume at 1.5 cm (finding 79). (3) A
    no-pause top lost from view dated a frame late by the fit puts a fast lowering's first
    bent frame in its window: D6 on 1 of 60 at 0.9 / 0.8 s, 1.5 cm (none at 1.2 / 1.0 s).
    (4) Pre-existing, unchanged since 92fe516: on the wrist proxy, clean no-pause reps with
    the wrists hidden from 0.07 s after the top to 0.4 s after draw D6 on 8 of 75 at 1.5 cm
    (round 13's review, on 92fe516: 29 of 225), over the 1-in-10 gate: the lockout is judged where the
    hips and knees were most extended while the wrists were unseen, the lowering's first
    frames. (5) The fit's expected top height is the standing pose's: a lockout that ends
    higher (the shoulders drawn back further than standing) reaches it early, one that ends
    lower (soft, leaned back) never; the joints check bounds both, at the cost of their own
    noise (1.2 s pulls lost from 0.4 s before, −0.07 to +0.17 s). (6) At 2 cm the joints'
    wander can read as a stall at a fitted top that was right, and they date it late (round
    14's review: 1 of 45, +0.43 s).

83. **A lost top with the hips and knees unmeasured is dated by the bar's fit alone**
    (review, round 14). Finding 77 read the joints' plateau in the gap; the pipeline marks a
    frame's legs unmeasured when any hip, knee or ankle is missing from its triangulation
    (the plates hiding the feet do it for the whole pull), and the pose tracker's carried
    frames are unmeasured too. The plateau of the few frames left at the gap's start put the
    top at the last frame seen (0.2–0.37 s early) and judged the climb's bent knees: D6
    severe on every clean rep (the review: 90 of 90 legs carried, 180 of 180 knees hidden,
    18 of 18 through the pipeline). With the joints measured on fewer than half the gap's
    frames the fit alone dates it, as da89661 did. The fit alone carries the expected top
    height's noise, and on a slow pull it can date the top early; a top dated in the gap is
    now judged on no frame from before the gap (the climb seen, knees bent: 2 s pulls, knees
    and bar hidden from 0.2 s before the top, 1.5 cm, D6 on 8 of 45 clean reps; now none).
    [gaps], knees and bar hidden from 0.2–0.3 s before the top to 0.8 s after, 1.2 and 2 s
    pulls (five bodies × seeds 0–2, 1.5 cm): every rep counted, tops within 0.23 s, no cue.
    Tests: legs carried and knees hidden through the gap, 1.2 and 2 s pulls, within 0.15 s,
    no D6; the early fit's draw at 1.5 cm, no D6; through the pipeline, the plates hiding the
    feet and every top lost, within 0.15 s, no D6. (The window rule is narrowed in
    finding 87.)

84. **A resume on the joints needs a gap of 0.3 s** (review, round 14). Finding 79 decides
    once per gap, but a hold the bar drops out of for a few frames again and again is many
    decisions on a few frames' medians: without a standing reference at 2.5 cm, holds
    resumed and read up to 1.4 s late. A gap shorter than 0.3 s (the shortest stall the
    analyser holds) resumes nothing; a grind lost through its finish is lost for longer.
    [gaps]: the bar lost 0.1 s every 0.25 s through 1.5 s holds without a standing
    reference, 2.5 cm: every top within 0.05 s. Test: the bar dropped 0.1 s every 0.25 s through 1.5 s holds, 2.5 cm, none
    beyond 0.3 s.

85. **Seen standing with the knees hidden** (review, round 14; pre-existing). A top lost from
    view short of the expected height is a top when the lifter was seen standing while the
    bar was unseen; with the knees hidden too, no frame read standing (it needs the knee
    angle), and every rep of a tall lifter was a failed rep. Upright on legs as long as
    standing's (the leg-length test the analyser uses for straight legs) now reads standing
    there. Test: knees and bar hidden from 0.3 s before every top, every rep counted.
    (One noisy frame within 6 cm counted failed reps: finding 89 reads it on medians,
    within 3 cm.)

86. **Housekeeping** (review, round 14): the envelope's long-hold rows draw their noise
    keyed by its level too (the 1.5 and 2 cm rows had shared one draw); PLAN's PULL → LOWER
    row no longer names the removed walk. Long lines in the tests and the script stay: the
    repository has no line-length rule (the code this branch builds on has 127 lines over
    120), and wrapping them by the formatter rewrites ~800 lines of the script.

87. **The climb is dropped from a gap-dated top's window only when the fit alone dated it**
    (review, round 15). Finding 83 dropped every frame up to the bar's last frame seen from
    a top dated in the gap. A no-pause top lost a frame or two after it is dated at that
    last frame, and its window held the lowering alone: 0.8 / 0.6 s tops lost from 0.03 s
    after, D6 on 12 of 75 clean reps noise-free and 22 of 75 at 1.5 cm (0 and 1 on acf0e31);
    with the plates hiding the feet, 12 and 23. The climb is now dropped only when no knee
    is measured in the gap's frames, the case of an early fit alone: 0 and 3 of 75; with
    the plates, 0 and 5. Test: no-pause tops lost from 0.03 s after, the feet seen or under
    the plates, noise-free, no D6.

88. **The plates hiding the feet leave the hips and knees to the lost top** (review, round
    15). The pipeline marks a frame's legs unmeasured when the feet are hidden, but the hips
    and knees are seen (the planted feet are kept): finding 83 dated those tops by the fit
    alone, which overshoots a quick pull (0.6 s pulls lost from 0.3 s before the top,
    through the pipeline: +0.45 to +0.59 s on 18 of 18). The joints' track now keeps a
    frame whose only unmeasured part is the feet. Test: 0.6 s pulls under the plates lost
    from 0.3 s before, within 0.15 s.

89. **Standing with the knees hidden, on medians within 3 cm** (review, round 15). Finding 85
    read one frame upright within the over-extended lockout's 6 cm of leg shortening (~40°
    of knee bend; the knees' standing limit, 20°, is ~1.3 cm): failed reps 10–12 cm short,
    the bar and knees hidden around their peak at 1.5 cm, were reps on 16 of 60. Standing is
    now read on 5-frame running medians of the trunk and the legs' length, within 3 cm.
    Test: such failed reps, none counted.

90. **An over-extended lockout seen while the bar was unseen is a top** (review, round 15;
    pre-existing). A lockout leaned back 25–40° hangs the bar well short of the expected
    top; lost from 0.2–0.3 s before it, its highest frame seen was on the climb, and only
    an upright lifter made it a top: 0 of 60 counted, D5 never cued. Leaned back past the
    over-extended top on straight legs, seen while the bar was unseen, is now a top too.
    Test: 25° and 40° lean-backs lost from 0.2 s before, every rep counted and D5 cued.

91. **The stall check reads the joints near the fitted top only** (review, round 15). It took
    the joints' nearest frame to the fitted top however far off: the knees hidden over the
    first 50–62 % of the gap, that frame was the climb's at the gap's start, bent, and the
    top read +0.27 to +0.37 s late on 45 of 45. It now reads only frames within 0.15 s of
    the fitted top: 1.2 s pulls within 0.05 s at every share hidden; on 2 s pulls one rep
    in 45, whose fit fails, is still dated by the joints' first frame seen (+0.37 s at
    60 %). Test: the knees hidden over the gap's first 55 %, within 0.15 s.

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
- **Simulated envelope:** `PYTHONPATH=src python scripts/tools/deadlift_envelope.py all`
  re-measures every [sweep] number above (each sweep 2–30 min on one core; `compute` times
  this machine, so run it alone).
