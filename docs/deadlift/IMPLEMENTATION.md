# Conventional deadlift — implementation status

Branch `claude/deadlift-v1-impl`. It implements the software of `docs/deadlift/PLAN.md` (v13). The
interface between the pipeline and the voice agent is frozen in `.claude/deadlift/CONTRACT.md`.

**The deadlift is gated.** Normal users cannot reach it until it is validated (J8). To try it,
set `NOWVA_DEV_COACHING_READY=deadlift` for the agent process; the pipeline subprocess inherits it.
Without that variable:
- every deadlift name resolves to an untracked stand-in that counts nothing and judges nothing;
- the menu does not offer the deadlift;
- a scheduled deadlift runs no briefing and no rules.

## What is built, by milestone

| Plan | Status | Where |
|---|---|---|
| **J0** squat net | Done. Every golden passes on this branch: squat output is byte-identical fresh, after RDL / untracked / **conventional deadlift** sessions, and through the voice-agent delivery path | `tests/test_biomechanics/test_squat_golden.py` (+ fixtures), `test_squat_invariants.py`, `tests/test_squat_delivery_golden.py`, `tests/test_squat_agent_invariants.py`, `tests/test_squat_session_pins.py` |
| **J1** software parts | Done: deadlift sagittal frame, measured gravity (tool + runtime mapping), setup model (two solves) | `deadlift/frame.py`, `deadlift/gravity.py`, `scripts/tools/measure_gravity.py`, `deadlift/setup_model.py` |
| **J2** simulator | Done. Synthetic sets with ground truth: dead stop, touch-and-go, quick re-pull, failed rep, dropped bar with bumper bounce, every v1 fault, keypoint and bar noise, wrist proxy, tilted world | `deadlift/simulator.py`, `tests/test_biomechanics/test_deadlift_analyzer.py` |
| **J3** profile on the platform | Done: profile and gate; registry variants; squat-default hooks; `process_frame` fork; analyser and counter; D1–D10; deadlift session reference; config; `_activate_profile`; refine and view-buffer gates; foot-contact reset; body-measurement flag; after-switch golden | `profiles/deadlift.py`, `deadlift/analyzer.py`, `deadlift/rep_counter.py`, `deadlift/rule_base.py`, `faults/rules/deadlift_*.py`, `pipeline.py`, `pipeline_process.py`, `config.py` |
| **J4** 3D bar, software | Done: batched multi-view detector wrapper, cross-view association with racked-bar rejection, single-view hub fallback, 3D Kalman with predicted states, capture-time buffer | `deadlift/bar_detector_multi.py`, `deadlift/bar_tracker_3d.py`, `deadlift/bar_buffer.py` |
| **J5α** delivery slice | Done. Covers closed-loop bar-over-midfoot guidance; D1/D3 (and all other) cue text; the contract fields on `cache_cues` / `frame_data`; min tier; no cue between touch-and-go reps | `src/agent/services/coaching_orchestrator.py`, `coaching_service.py`, `coaching_constants.py`, `src/assets/cue_text/cues.json`, `coaching/cue_cache.py`, `coaching/ipc_bridge.py` |
| **J5** delivery integration | Done. Covers the deadlift recap; session metadata (grip, plates, belt, shoes) on `start_capture` / `set_exercise`; first-session briefing and empty-bar warm-up; knowledge card and `explain_deadlift` tool; form-check branch; display tiles and score dimensions; per-exercise fault sets in the orchestrator; DB tagging | `src/agent/agents/**`, `src/main.py`, `src/visual/display.html`, `scripts/tools/draft_cue_text.py` |
| **J7** set diagnosis | Done. `HypothesisEngine(graph)` defaults to the squat graph; the deadlift graph has 10 symptoms and 16 causes; scoring is weighted setup 25 / coordination 25 / bar path 20 / lockout 15 / symmetry 15; a rolling update after each rep and `diagnosis_complete` at set end | `diagnosis/engine.py`, `diagnosis/graph/deadlift_*`, `deadlift/diagnosis.py` |

## Measured on the simulator (J2/J3 acceptance)

- **Rep counting:** exact once per rep. This holds for dead stops, touch-and-go, quick re-pulls, dropped bars with a bumper bounce, noisy keypoints (6 mm) and bars (3 mm), the wrist proxy and a 3°-tilted world. A failed rep is an event, not a rep.
- **Events:** liftoff, knee pass, top and floor all land within 77 ms of truth (gate: 100 ms).
- **Faults:** every injected fault, D1–D10, fires at moderate or worse on every rep. Clean, noisy and tilted-world sets are fault-free.
- **Gravity:** measured gravity keeps a 3° tilted world from faking drift. Against the body vertical, the same set fakes more than 2 cm of drift.
- **Compute (x86):**
  - analyser: 0.2 ms median, 0.4 ms p95, 0.9 ms worst steady-state frame (rep completion);
  - bar association and tracking: about 0.6–1.1 ms;
  - the Jetson numbers are still to be measured (J1).

## Findings that changed the plan

1. **D2 thresholds 10/15/20° are mostly unreachable.** With the bar at the knees, straight legs cap the hips-shoot excess at about 8–12° for typical bodies, as the simulator showed across three body types. The initial thresholds are now 5/8/11°, min tier still moderate. The rise-ratio cross-check is now 1.0 instead of 1.4: a normal pull reads 0.5–0.75 and a full hips shoot about 0.8–1.1. Both are re-set at J6.
2. **D7 ranks before D4** (priorities 21 and 22). The §2.6 table had them the other way round, which contradicted §2.7 ("if D4 and D7 co-fire, D7 is cued first").
3. **Every deadlift set would have crashed the set summary.** Both `IPCBridge.send_set_complete` and `SessionTracker._compute_set_summary` ran `statistics.stdev` on NaN depths. A set with no depth now reports null depth; sets with depth take the unchanged squat path.
4. **Today's after-switch carry-overs for other exercises** (squat golden allow-list, not changed here):
   - RDL reps raise the squat's standing reference.
   - RDL and untracked frames feed a first-time athlete's body measurement.
   - `set_depth_target(None)` does not survive a switch.

   The conventional deadlift has **none** of these (§5.2).
5. **Pre-existing squat bug, not fixed here.** On the voice-agent side, `shallow_rep` messages are dropped before dispatch: `CoachingService._handle_message` hits a `category` keyword clash in the profiler call, outside its `try`. The delivery golden pins today's behaviour; fixing it is a separate change.

## Deferred: needs data, hardware or keys (not code)

| Item | Why it can't be done here |
|---|---|
| Bar detector training (J4) | The weights `models/barbell_keypoints.pt` are not in the repo (Q7), and floor-bar frames need labelling. Until then the pipeline runs on the **wrist proxy**: wider D1/D3/D8b thresholds, min tier moderate |
| Visibility map, round-1 capture, ρ, hip-keypoint bias (J1) | Needs the rig and lifters |
| Gravity on the rig | Run `scripts/tools/measure_gravity.py` with the board flat. Without the files, the deadlift runs on the body vertical, with D2/D3/D5 raised to moderate |
| Real validation (J6), thresholds and min tiers from data, `VALIDATION.md` | Needs round-2 data |
| Cue audio clips | Needs the TTS key: `generate_cue_audio.py` covers every new key. Until clips exist, the cache falls back to cloud TTS (O3) |
| Recorder / replay provider (§8.2) | Not built yet; the simulator stands in for offline tests |
| Jetson timing (§9) | Needs the device |
| J8 ship | `coaching_ready=True` plus the two intentional pin changes listed in PLAN §5.1 |

## Tests

- **Biomechanics:** `PYTHONPATH=src python3 -m pytest tests/test_biomechanics -q` → 1893 passed, 3 skipped.
- **Agent side:** `DATABASE_URL=<any postgres URL> PYTHONPATH=src python3 -m pytest tests -q --ignore=tests/test_biomechanics`, excluding the modules this container cannot import (PortAudio, program-generator data). Everything passes except 12 failures that already fail on `main` in this environment: 1 listener reconnect, 4 rep sound, 7 v6 program verification.
