# Squat V1 contract: pipeline → delivery

This file freezes the interface between the biomechanics side (pipeline, faults, diagnosis) and
the delivery side (cue cache, voice agent, cue audio). It implements `FINDINGS.md`. Both sides
code against this file. Change it before changing either side.

## Ownership (while both workstreams run in this worktree)

**Biomechanics owner:**
- everything under `src/biomechanics/`, except `src/biomechanics/coaching/cue_cache.py`
- `config/biomechanics.yaml`
- `tests/test_biomechanics/`

**Delivery owner:**
- `src/biomechanics/coaching/cue_cache.py`
- `src/agent/**`
- `src/db/**`
- `src/visual/**`
- `scripts/tools/generate_cue_audio.py` and the new `scripts/tools/draft_cue_text.py`
- `src/assets/cue_text/`
- every test outside `tests/test_biomechanics/`

## 1. Squat fault types (`FaultEvent.fault_type`)

| fault_type | Emitted | Meaning | Cue base key | Side-specific cue variants | Priority (lower wins) |
|---|---|---|---|---|---|
| `knee_valgus` | once per rep (rep end) | Knees cave inside the toe line, over the bottom and ascent | `knees_out` | `knees_out_left`, `knees_out_right` | 0 |
| `hip_shoot` | once per rep | Hips rise faster than the chest out of the hole (good-morning squat) | `chest_up` | none | 1 |
| `heel_rise` | in rep (per frame, gated) | A heel lifts off the floor (triangulated only) | `heels_down` | `heels_down_left`, `heels_down_right` | 2 |
| `balance` | once per rep | Weight drifts to the toes (`direction: forward`) or the heels (`direction: backward`) | `whole_foot` | none | 3 |
| `hip_shift` | once per rep | Pelvis slides sideways. Approximate on a single camera, so no mid-set cue there; cued on the triangulated rig | `even_it_out` | `even_it_out_left`, `even_it_out_right` (side = direction the hips moved) | 4 |
| `bilateral_asymmetry` | once per rep | Bar tilts; only fires when a bar is detected | `level_bar` | none | 5 |
| `depth` | once per rep, and for rejected shallow descents | Rep missed the athlete's own depth target | `deeper` | none | 6 |
| `foot_placement` | once per rep (judged from the setup before the rep) | One foot staggered forward or flared out more | `square_feet` | `square_feet_left`, `square_feet_right` (side = the foot to move) | 7 |
| `lockout` | once per rep (the top before this rep) | Not standing tall between reps | `lockout` | none | 8 |
| `tempo` | once per rep | Dive-bombed descent (`kind: fast_descent`) | `slow_down` | none | 9 |
| `depth_drift` | once per rep | Rep noticeably shallower than the athlete's best this session (target still met) | `same_depth` | none | 10 |
| `velocity_loss` | once per rep | Rep speed dropped ≥20% vs the best rep this set | `drive` | none | 11 |

**Removed for squat:** `forward_lean` (replaced by `hip_shoot` intra-set and by set-level
diagnosis). Pose-based knee-angle `bilateral_asymmetry` (`SymmetryRule`) is no longer registered;
`bilateral_asymmetry` now comes from bar tilt only.

**Kept outside squat:** `forward_lean`, `back_rounding` and the other `FaultType` values still exist
for other profiles.

## 2. `FaultEvent.details` (every squat fault carries these keys)

| key | type | values |
|---|---|---|
| `side` | `str \| None` | `"left"`, `"right"`, `"both"`, or None |
| `phase` | `str \| None` | `"setup"`, `"descent"`, `"bottom"`, `"ascent"`, `"top"`, or None |
| `observability` | `str` | `"observable"` or `"approximate"` (not-observable faults are never emitted) |
| `is_drift` | `bool` | True when the athlete's best rep this session did NOT show this fault, i.e. it appeared with fatigue or load |
| `value` | `float` | Primary measured value |
| `unit` | `str` | `"deg"`, `"cm"`, `"ratio"`, `"pct"` or `"s"` |

Fault-specific extras may follow (for example `direction`, `kind`, `deficit_cm`, `velocity_mps`).

## 3. IPC messages (built in `src/biomechanics/coaching/ipc_bridge.py`)

### `fault`
The existing fields, plus:
- `side`, `phase`, `observability`, `is_drift`, `value`, `unit`
- `details` (the full dict, NaN → None)

`cue` is None when `observability == "approximate"`. Approximate faults are recorded but get no
intra-set audio.

### `rep_complete`
The existing fields, plus:
- `features`: the `RepFeatures` dict (see `src/biomechanics/analysis/rep_features.py`). Keys
  include `depth_ratio`, `depth_cm_above_parallel`, `valgus_l`, `valgus_r`, `hip_shoot_deg`,
  `hip_shift_ratio`, `balance_ratio`, `concentric_velocity_mps`, `velocity_loss_pct`,
  `lockout_deficit_ratio`, `descent_time_s`, `ascent_time_s`.
- `highlights`: `list[str]`, a subset of:
  - `"best_rep_so_far"`
  - `"depth_target_met"`
  - `"clean"` (no fault at all)
- `depth_target_met`: `bool`

### `diagnosis_complete`
The existing fields, plus:
- Each cause entry gains `tier` and `observability`.
- `longterm_causes`: list of `{cause_id, tier, score, explanation, observability}`.
- `contextual_notes`: the same shape (tier 0: anatomy explanations, said once, reassuringly).
- Each `detected_symptoms` entry gains `observability`.
- `diagnosis.confidence` is now **measurement** confidence, 0–1. Below 0.5 the recap must hedge.

## 4. Cue keys and the scenario each cue must express

All cues use external focus, no jargon (never "valgus"), and are short (≤ 4 words spoken).

| key | Scenario for the drafting LLM |
|---|---|
| `knees_out` / `_left` / `_right` | Knees (or the named knee) cave inward on the way up. Cue pushes them out over the little toes ("spread the floor"). Side variants name the side. |
| `chest_up` | Hips rise faster than the chest out of the bottom. Cue: drive the upper back up into the bar, chest and hips rise together. |
| `heels_down` / `_left` / `_right` | A heel lifts. Cue: keep the whole foot planted, weight through heel and big toe. |
| `whole_foot` | Weight drifting onto the toes. Cue: whole foot, feel the heel. |
| `even_it_out` / `_left` / `_right` | Hips sliding to one side. Side variants: "you're drifting left/right — stay centered, push evenly". |
| `level_bar` | Bar tilting. Cue: keep the bar level, push evenly. |
| `deeper` | Not reaching their target depth. Cue: sit a little lower. |
| `square_feet` / `_left` / `_right` | Feet set up uneven. Side variants name the foot to adjust ("bring your left foot even"). |
| `lockout` | Not standing all the way up between reps. Cue: stand tall at the top. |
| `slow_down` | Dropping into the bottom too fast. Cue: control the way down. |
| `same_depth` | Later reps getting shallower than earlier ones. Cue: same depth as your first rep. |
| `drive` | The rep slowed down a lot. Cue: drive hard out of the bottom. |

**Existing keys that stay** (re-voiced in Nova's voice): `rep_1` … `rep_20`, positive cues
(`good_rep`, `great_depth`, `strong`, `clean`, `perfect`), `brace`, `breathe`, and every other key
currently in `SQUAT_CUES`.

`hips_through` and `flat_back` have no producer for squat; drop them.

## 5. Cue audio pipeline (decided with the user)

**Step 1 — draft text.** `scripts/tools/draft_cue_text.py` sends the scenario for each cue key to
an LLM (model **`gpt-6-astra`**, overridable with `--model`) and asks for **3** short variants per
key. It writes:
- `src/assets/cue_text/cues.json`: `{key: [line, line, line]}`, committed
- a review page

The user reviews and edits this text before step 2.

**Step 2 — synthesize audio.** `scripts/tools/generate_cue_audio.py` reads `cues.json` and speaks
each line with **Cartesia Sonic-3** using `CARTESIA_VOICE_ID` (Nova's live voice). That gives:
- exact transcripts (`cues.json` is the transcript)
- 3 clips per key
- the same output format and paths the cue loader expects today

**Nothing in either step runs without the user's go-ahead.**

## 6. Delivery behaviour required

**Cue selection**
- Pick the cue from `fault.cue`, which is side-aware. Fall back to the base key when a side
  variant is missing.
- Approximate faults (`cue is None`) never trigger intra-set audio.

**Set recap**
- Wait for `diagnosis_complete` with this set's `set_number` (bounded wait, e.g. ≤ 2.5 s) before
  composing. This fixes the race where recap N read set N−1's diagnosis.
- Use `confidence < 0.5` and `observability == "approximate"` as hedge flags. Never assert a fault
  the system can't see ("butt wink", "upper back rounding").
- Voice at most one long-term cause per session, in the first recap where it appears.
- Voice contextual notes (tier 0) once, as reassurance.

**Positive reinforcement**
- Use `highlights` occasionally (no more than once every few reps), e.g. on `best_rep_so_far`.

**Plain words for the LLM**
- `FAULT_LABELS` gets a plain phrase for every fault type above.
- Delete the dead keys `butt_wink`, `shallow_depth` and `asymmetric_loading`.

**Stance / toe-out follow-up**
- Today it is armed after a `forward_lean` cue. Re-key it so it is armed when the latest diagnosis's
  top immediate cause is `narrow_stance`, `stance_toe_mismatch` or `narrow_foot_angle`.
