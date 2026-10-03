# Deadlift v1 contract: pipeline → delivery

This file freezes the interface between the biomechanics side (deadlift analyser, rules,
set diagnosis) and the delivery side (cue cache, voice agent, cue text/audio) for the
conventional deadlift. It mirrors `.claude/squat-audit/CONTRACT.md` and implements
`docs/deadlift/PLAN.md` §2.6, §4. Change this file before changing either side.

**Squat rule:** every field below is either deadlift-only or sent only when it differs from
the squat default, so the squat's IPC stream stays byte-identical.

## Ownership

Same split as the squat contract:
- **Biomechanics:** everything under `src/biomechanics/` except `coaching/cue_cache.py`;
  `config/biomechanics.yaml`; `tests/test_biomechanics/`.
- **Delivery:** `src/biomechanics/coaching/cue_cache.py`, `src/agent/**`, `src/db/**`,
  `src/visual/**`, cue text/audio scripts, `src/assets/cue_text/`, tests outside
  `tests/test_biomechanics/`.

## 1. Deadlift fault types (`FaultEvent.fault_type`)

All are judged once per rep (`RepFaultRule.judge_rep` via `RuleEngine.finish_rep`), when the
rep is counted: at the dead stop, or at the next liftoff for a touch-and-go rep.

| # | fault_type | Meaning | value / unit | Cue base key | Variants | Static min tier | Prio |
|---|---|---|---|---|---|---|---|
| D1 | `deadlift_bar_position` | Bar not over midfoot at setup | `abs(bar_midfoot_setup_cm)` cm; `direction`: `"forward"` (bar too far) / `"back"` (bar too close) | `deadlift_bar_midfoot` | — | mild | 20 |
| D7 | `deadlift_shoulders_behind` | Shoulders behind the bar at setup | cm behind | `deadlift_shoulders_over` | — | moderate | 21 |
| D4 | `deadlift_setup_hips` | Hips too low / too high at setup | cm outside the setup-model band; `direction`: `"up"` (raise hips) / `"down"` (lower hips) | `deadlift_hips` | `deadlift_hips_up`, `deadlift_hips_down` (by `direction`) | severe | 22 |
| D2 | `deadlift_hips_shoot` | Hips rise before the chest off the floor; emitted only when the hips decisively out-rose the shoulders (rise ratio > 1.05 and the set's recent reps > 1.15, or this rep > 1.30) | deg of trunk change beyond the model's prediction; `rise_ratio`, `set_rise_ratio` | `deadlift_chest_with_hips` | — | moderate | 23 |
| D3 | `deadlift_bar_drift` | Bar drifts away from the legs | p90 forward drift, cm | `deadlift_bar_close` | — | moderate | 24 |
| D6 | `deadlift_lockout` | Incomplete lockout | worst of hip/knee extension deficit, deg; `joint`: `"hip"`/`"knee"` | `deadlift_lockout` | — | moderate | 25 |
| D5 | `deadlift_lean_back` | Over-extension at the top | deg behind standing | `deadlift_finish_neutral` | — | moderate | 26 |
| D8 | `deadlift_hip_shift` | Hips shift sideways | ratio of ankle separation; `side` = direction the hips moved | `deadlift_even_feet` | `deadlift_even_feet_left`, `deadlift_even_feet_right` | moderate | 27 |
| D8b | `deadlift_bar_tilt` | Bar tilts | cm height difference of the ends; `side` = low end | `deadlift_level_bar` | — | moderate | 28 |
| D9 | `deadlift_bent_arms` | Arms bend during the pull | deg elbow flexion vs standing | `deadlift_long_arms` | — | mild | 29 |
| D10 | `deadlift_velocity_loss` | Bar slower than the set's two fastest reps | pct | — (never cued) | — | `recap` | 30 |

`deadlift_bar_position` and every closed-loop key are the Demo α pair together with
`deadlift_bar_drift`. D7 ranks ahead of D4: the D4 band assumes the shoulders in their band,
so when both fire the shoulders are the correction (PLAN.md §2.7 "Coupling").

## 2. `FaultEvent.details`

Every deadlift fault carries the squat keys (`side`, `phase`, `observability`, `is_drift`,
`value`, `unit`) plus:

| key | type | values |
|---|---|---|
| `min_tier` | `str` | `"mild"`, `"moderate"`, `"severe"` or `"recap"`. The fault's effective minimum cue tier for this rep: the static tier, raised to at least `"moderate"` when the rep was measured from the wrist proxy (D1, D3, D8b; D8b to `"severe"`) or from the body vertical instead of measured gravity (D2, D3, D5) |
| `bar_source` | `str` | `"bar"` or `"wrist_proxy"` |
| `gravity_source` | `str` | `"measured"` or `"body"` |
| `direction` | `str` | D1 and D4 only (see §1) |
| `joint` | `str` | D6 only |

`phase` is `"setup"` (D1, D4, D7), `"pull"` (D2, D3, D8, D8b, D9, D10) or `"top"` (D5, D6).

## 3. IPC messages

### `cache_cues` (from `IPCBridge.prepare_exercise`)
The existing fields, plus — **each sent only when it differs from the squat default**:

| field | type | squat default (omitted) | deadlift |
|---|---|---|---|
| `fault_to_cue` | `dict[str, str]` | the squat `FAULT_TO_CUE_MAP` | deadlift map (§1) |
| `min_cue_tiers` | `dict[str, str]` | `{}` | static tiers (§1) |
| `set_idle_timeout_s` | `float` | `15.0` | `30.0` |
| `waits_for_diagnosis` | `bool` | absent (the agent's `is_squat` path already waits) | `true` |
| `closed_loop` | `str` | absent | `"deadlift_bar_midfoot"` |

`waits_for_diagnosis` is sent only for profiles that run a set diagnosis other than the squat's;
when absent, the agent keeps today's behaviour (wait only on the squat).

### `frame_data`
The existing fields, plus, for the deadlift only:
- `deadlift_phase`: one of `approach`, `stance`, `setup`, `pull`, `top`, `lower`, `floor`
- `bar_midfoot_live_cm`: forward offset of the bar centre vs the live midfoot (cm, > 0 = bar
  ahead of midfoot, i.e. step closer), the median over the last 0.5 s; null outside `stance`,
  while the lifter is moving (hinging down, walking), after the set's first rep, and always null when
  `bar_source` is `"wrist_proxy"` (the hanging wrists say nothing about where the bar sits)
- `bar_source`: `"bar"` or `"wrist_proxy"`

`rep_phase` for the deadlift carries the same deadlift phase string, so the squat stance
monitor (armed only on `rep_phase == "idle"`) never fires.

### `rep_complete`
Same message. For the deadlift:
- `features` is the `DeadliftRepFeatures` dict (`src/biomechanics/deadlift/types.py`), with
  `dl_schema: 1`. It is dumped **after** `finish_rep`, so `velocity_loss_pct` is filled.
- `max_depth_angle` is null and `depth_category` is `"n/a"`; `depth_target_met` is `true`.
- `highlights` uses `"clean"` and `"best_rep_so_far"` (fastest rep with no fault at or above
  its min tier).

### `diagnosis_complete`
Same shape as the squat. Deadlift causes come from the parameterised `HypothesisEngine`
with the deadlift graph. Each immediate cause carries **one** numeric delta in
`parameter_delta` and one plain sentence in `explanation`.

## 4. Cue keys and scenarios (≤ 4 words, external focus, no jargon, nothing medical)

| key | Scenario for the drafting LLM | Draft |
|---|---|---|
| `deadlift_bar_midfoot` | Bar was not over the middle of the foot at setup. | "Bar over midfoot" |
| `deadlift_hips` / `_up` / `_down` | Hips set too low (`_up`) or too high (`_down`) at setup. | "Hips a bit higher" / "Hips a bit lower" |
| `deadlift_shoulders_over` | Shoulders behind the bar at setup. | "Shoulders over the bar" |
| `deadlift_chest_with_hips` | Hips rose before the chest off the floor. | "Chest and hips together" |
| `deadlift_bar_close` | Bar drifted away from the legs. | "Bar close" |
| `deadlift_lockout` | Did not finish standing tall. | "Stand tall" |
| `deadlift_finish_neutral` | Leaned back at the top. | "Stand tall, no lean" |
| `deadlift_even_feet` / `_left` / `_right` | Hips slid to one side; side variants name the side they slid to. | "Push evenly" |
| `deadlift_level_bar` | Bar tilted. | "Keep the bar level" |
| `deadlift_long_arms` | Arms bent during the pull. | "Long arms" |
| `deadlift_step_closer` | Closed loop, bar > 15 cm ahead of midfoot while standing at the bar. | "Step closer" |
| `deadlift_closer` | Closed loop, bar 2–15 cm ahead of midfoot. | "A bit closer" |
| `deadlift_back` | Closed loop, bar behind midfoot (shins too close). | "Back a little" |
| `adjust_good` | Closed loop, bar now over midfoot (existing key). | existing |

Each correction key also gets a `<key>_fixed` praise line, as the squat keys do.

## 5. Delivery behaviour required

1. **Cue fallback** uses `cache_cues.fault_to_cue` (squat map when absent).
2. **Minimum tier.** Drop any cue whose severity is below `max(min_cue_tiers[fault_type],
   fault.details.min_tier)`; `"recap"` is never cued. The fault is still recorded and reaches
   the recap and the DB. A diagnosis focus is adopted only if its fault reached its min tier.
3. **Touch-and-go.** No cue between touch-and-go reps: a deadlift cue is spoken only while
   `deadlift_phase` is `floor`, `stance`, `setup` or `approach`, i.e. after a dead stop; a cue
   pending at a touch-and-go waits for the next dead stop or goes into the recap.
4. **Set idle timeout** from `cache_cues.set_idle_timeout_s` (default 15 s).
5. **Closed-loop D1 guidance** (deadlift only, tracked bar only: never when `bar_source` is
   `"wrist_proxy"`): arms whenever `deadlift_phase == "stance"` and
   `abs(bar_midfoot_live_cm) > 2`; speaks `deadlift_step_closer` (> 15 cm), then
   `deadlift_closer` / `deadlift_back`, then `adjust_good` once within 2 cm; shares the squat
   monitor's speaking flag and utterance budget; disarms at `setup` / `pull`.
6. The squat stance monitor is additionally gated on the squat profile.
7. **Recap** for the deadlift: bar speed, main fault, best rep; no depth lines.
8. Progress queries pass the active profile's exercise.
