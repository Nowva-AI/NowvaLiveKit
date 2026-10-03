# Conventional deadlift — implementation status

Branch `claude/deadlift-v1-impl`. It implements the software of `docs/deadlift/PLAN.md` (v15).
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
| **J2** simulator | Done, as a standalone generator rather than the §8.5 extension of `.claude/preik-audit/harness` (that harness's import bugs, §8.5, are still open). Synthetic sets with ground truth: dead stop, touch-and-go, quick re-pull, grip-and-rip, re-setup (standing up off the bar between reps), a sticking point mid-pull, failed rep, dropped bar with bumper bounce, every v1 fault, plate occlusion, keypoint and bar noise, wrist proxy (hands ride the tilted bar), tilted world. **Model-free poses:** the setup and knee-pass trunk angles can be scripted instead of taken from the setup model, and the truth (trunk angles, setup hip height, rise ratio, knee flexion) is read off the emitted poses. Capture lag is exercised through the real pipeline (`test_deadlift_pipeline.py`); a moved camera is not simulated | `deadlift/simulator.py`, `tests/test_biomechanics/test_deadlift_analyzer.py` |
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
| 2.5 cm AR(0.8) | 12/12 | median 19 ms, p95 66 ms | 2 / 36 |

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
| 1.0 cm i.i.d. | 12/12, median 67 ms (p95 228), 0 / 36 | 12/12, median 66 ms (p95 433), 0 / 60 |
| 1.0 cm AR(0.8) | 12/12, median 67 ms (p95 345), 0 / 36 | — |
| 2.0 cm i.i.d. | 12/12, median 137 ms (p95 567), 0 / 36 | — |
| 2.0 cm AR(0.8) | 12/12, median 100 ms (p95 500), 1 / 36 | 12/12, median 133 ms (p95 533), 4 / 60 |

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
  0.8 s sticking point mid-pull is ground through and counted.
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
  - A lockout 30° short is a failed rep.

**Faults:**
- **Injected faults:** with noise-free keypoints, every injected fault (D1–D10) fires at
  moderate or worse on every rep. D2's is a hips-first pull scripted without the setup model.
- **Clean sets:** clean, noisy and tilted-world sets are fault-free.
- **D2 against a held back angle**, which the setup model reads as 6–12° of excess:
  - cued on 0 of 90 reps at 2 cm i.i.d. noise and 4 of 90 at 2 cm AR(0.8);
  - before round 3 it was cued on 31 and 35 of 90.
- **D2 on a scripted hips-first pull** (ratio ~1.24): cued from the set's second rep on, on
  60–87 % of reps under the same noise. A first rep is cued alone only beyond a 1.30 ratio.
- **Near-threshold faults:** at 2 cm i.i.d. keypoint noise, a fault injected just above
  moderate reads moderate on 3–4 of 8 reps for D5 and D6 (+1–2° over the threshold), and on
  6–8 of 8 for the others. D9 at +5° reads mild.

**Gravity:** measured gravity keeps a 3° tilted world from faking drift. Against the body
vertical, the same set fakes more than 2 cm of drift.

**Compute (x86):**
- Analyser, at 1.5 cm keypoint noise: 0.3 ms median and 0.6 ms p95 per frame, and at most
  about 2.3 ms on the frame that completes a rep (event fits and features). The first frames
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
     kinematic truth: trunk change within 2°, rise ratio within 0.05, hip height within 1 cm,
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
    - D2's poses are now taken from several frames: a median before liftoff, and a line
      fitted through the 0.2 s up to the knee pass. One past the knee pass biased the change
      by −2°.
    - D2 is emitted only when the hips decisively out-rose the shoulders, rep and set.
    - The set diagnosis applies the same test, so a recap never names a hips shoot the rule
      would not.

12. **The wrist proxy's left-right must be locked per set.**
    - Rebuilt every frame from the noisy hip line, ~6.7° of rotation leaked the hips' forward
      travel into "sideways hip shift": 18 false D8 cues in 36 clean proxy reps.
    - Locking it from the stance and setup frames removed all of them.
    - D8b on the proxy is severe-only: the hands' noise is carried ~3× out to the hubs.

13. **Foot guidance must not arm on one frame, nor after the set.** The live offset is now a
    0.5 s median, reported only while the lifter stands settled at the bar before the set's
    first rep. Previously it was reported while hinging down to the bar and while walking off
    after the set.

14. **A squat-path change, in a case that crashed.** The set summary's depth statistics now
    report null for a set of two or more reps with no depth at all. A deadlift set has no
    depth. On `main`, such a set crashed `statistics.stdev`. Every set with any depth, and a
    one-rep set, takes the unchanged path, and the squat goldens pin it.

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
