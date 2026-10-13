# Shared plumbing for non-squat exercises (OHP, RDL, conventional deadlift)

This is the one spec for the platform changes that the press, RDL and deadlift plans all need. `ohp.md` and `rdl.md` code against it and don't redefine it.

It is aligned with the deadlift plan, which is being implemented in the cloud on branch `claude/epic-einstein-t65x3c` (reference copy: `deadlift-PLAN-v10-reference.md`, §3.2–3.6 and §4.2–4.3).

Rules for this spec:
- Names below that come from that plan are kept exactly.
- Where that plan is silent, this file decides.
- If the deadlift branch lands first with different method names, rename to match it. The semantics stay.

Every hook defaults to today's squat behaviour. The squat's output must stay byte-identical, as the golden masters below show.

## 1. Profile hooks (`profiles/base.py`)

| Hook | Default | Source |
|---|---|---|
| `create_rep_analyzer(config: BiomechanicsConfig) -> RepAnalyzer \| None` | `None` | deadlift §3.3. It takes the config the way `create_fault_rules(config)` does, so an analyser reads `config.faults.<exercise>` |
| `observe_frame(ctx: FrameContext) -> None` | no-op | deadlift §3.3 |
| `create_session_reference(config: BiomechanicsConfig) -> SessionReference \| None` | `None`, so the engine builds the squat `SessionReference` | deadlift §3.3 |
| `create_set_diagnosis()` | `None` | deadlift §3.3 |
| `allows_camera_refine: bool` | `True` | deadlift §3.3 |
| `needs_bar_3d: bool` | `False` | deadlift §3.3 |
| `get_readiness_check() -> Callable[[FrameContext], bool] \| None` | `None`, which keeps today's standing gate unchanged | already on base; wired here. For profiles **without** an analyser. The predicate is ANDed into `StandingPoseGate` (see `is_ready`) |
| `on_rep_rejected(reason, features, angles, config: BiomechanicsConfig) -> list[FaultEvent]` | `[]` | this file |
| `shallow_rep_cue(fault: FaultEvent, streak: int, cued_this_set: int, config: BiomechanicsConfig) -> str \| None` | `None` (cue-less) | this file. See §2, "Rejection streak" |
| `reports_depth: bool` | `True` for squat, `False` for others | this file |

Profiles hold no config. Every hook that reads thresholds receives the pipeline's `config` as an argument, including `cue_start_tier` (§4). A profile that overrides one of these hooks must take the same arguments: the pipeline always passes `config`.

When a profile returns an analyser, `pipeline.py` skips its squat-shaped blocks:
- the frame sample, bottom frame and trajectory;
- `_track_setup` / `_start_rep_setup`;
- `compute_rep_features`;
- the knee-flexion overwrite of `max_depth_angle`;
- `depth_target_met`.

When the profile returns no analyser, each of those blocks runs exactly as it does today.

## 2. `RepAnalyzer` protocol (this file decides; the deadlift plan names only the hook)

```python
class RepAnalyzer(Protocol):
    def observe(self, ctx: FrameContext) -> None: ...
    def on_rep_start(self, piston: bool) -> None: ...
    def finish_rep(self, rep_number: int) -> BaseModel: ...
    def rejection_reason(self, features: BaseModel) -> str | None: ...
    def discard_rep(self) -> None: ...
    def reset_set(self) -> None: ...
```

- **`observe`** runs on every processed, non-extrapolated frame after the readiness gate latches, both idle and in-rep. The analyser owns its pre-roll, setup history, rack or standing reference, and trajectory.
- **`on_rep_start(piston)`** runs on the frame the rep counter enters a rep. It also runs on `rep_started` turnarounds, where a new rep begins at the previous rep's turnaround with no idle frames in between. On those, `piston=True`.
  - **Call order on a piston frame:** the ended rep's `finish_rep` runs first, together with its rejection handling and verdicts. The new rep's `on_rep_start(piston=True)` runs after that.
  - On a piston start, the analyser has no idle setup frames. It must define its setup and pre-roll from the turnaround frames.
    - It must not judge setup-dependent faults for a rep that started at a turnaround. Examples are the press's grip, the RDL's lockout judged at the previous rep's finish, and the squat's standing lockout.
    - Faults judged within the rep itself are still judged, for example the press's `press_lockout` at the top.
- **`finish_rep`** returns the exercise's pydantic feature model. That model always has `exercise` (the profile name, required), `rep_number` and `max_depth_angle` (a float, NaN allowed), plus its own fields. Downstream code that asks "is this a non-squat rep?" reads `features["exercise"]` or the profile, never an exercise-specific key such as `dl_schema`.
- **Feature dump order.** On a counted rep, the pipeline calls `rule_engine.finish_rep(...)` first and dumps `features.model_dump()` onto `RepData.features` afterwards. That way, values the engine fills in, such as `velocity_loss_pct` from the session reference, reach RepData and IPC. This applies to the analyser path only; the squat path is unchanged.
- **`rejection_reason`** returns None when the rep counts. Otherwise it returns a reason string, for example `"partial"`, `"no_lockout"` or `"ignore"`. On a reason, the pipeline does all of the following before any verdicts are formed, so a rejected rep never feeds `SessionReference.update`:
  1. calls `rep_counter.reject_last_rep()`;
  2. calls `rule_engine.discard_rep()`;
  3. calls `analyzer.discard_rep()`;
  4. sets `rep_data = None`;
  5. calls `profile.on_rep_rejected(reason, features, angles, config)`.
  Each returned fault goes through the same observability gate as rule faults: `RuleEngine._tag_observability`, made public as `tag_observability`, using `FAULT_MEASUREMENT`. A fault that is not observable is dropped, exactly like a rule verdict.
  The remaining faults are emitted. A fault with `details["shallow_rep"] = True` also produces a `shallow_rep` message, whose `cue` is `profile.shallow_rep_cue(fault, streak, cued_this_set, config)`.
- **Rejection streak.** The pipeline owns all streak state; analysers keep none.
  - `streak` counts consecutive rejections with the same reason within the set, starting at 1.
  - A counted rep resets it to 0.
  - A rejection with a different reason starts that reason's streak at 1, and leaves the other reason's count unchanged.
  - The reason `"ignore"` emits nothing and leaves every streak unchanged.
  - `cued_this_set` is the number of `shallow_rep` messages with a non-None cue so far this set.
  - Both counters reset in `reset_set`.
- **`discard_rep`** handles a false start, an abort, `go_deeper`, or a rep that is too short.
- **`reset_set`** is called from `reset_readiness_gate()`. Each analyser documents what survives it.

Optional analyser methods. The pipeline checks with `hasattr` and falls back to today's behaviour when a method is absent:
- **`is_ready(ctx) -> bool`.** An extra readiness predicate for an analyser profile. The analyser owns its setup state, such as the press rack reference.
  - It is **ANDed** with `StandingPoseGate`'s own checks (visibility, knees, torso) and must hold over the gate's consecutive-frame window. It never replaces the gate.
  - Profiles without an analyser get the same AND semantics through `profile.get_readiness_check()`.
  - A profile must not hold analyser state to build its readiness check.
- **`counter_baseline() -> float | None`.** A float seeds the rep counter's baseline through `rep_counter.seed_baseline(value)`, for example with the press's rack height. `None` keeps the counter's own baseline.
  - It is called when the readiness check latches, and again after **every** rep-counter reset. The pipeline wraps the rest-end and set-boundary `rep_counter.reset()` calls (`pipeline_process.py:1229, 1467, 1604, 2008`) in `pipeline.reset_rep_counter()`, which re-seeds.
- **Single-camera pitch** (§3, "Camera mount"):
  - `bind_pitch_certification(certification, profile_name)` hands the analyser the session's `PitchCertification` and its own tripwire, keyed by profile name. The pipeline calls it on every analyser it builds.
  - `pitch_tripwire_delta() -> float | None` reports a delta trip once. The pipeline polls it after `observe` and again right after `finish_rep`, so a trip found in a rep's own setup judges that rep's verdicts uncertified.
  - `camera_pitch_untrusted: bool` is the gross tripwire on an uncertified camera. The pipeline signals its first transition once a session, whatever exercise trips it.
- Each analyser observes a frame timestamp once. The pipeline skips `observe` on a repeated timestamp (the Kalman warm-up after the latch repeats the latched frame).

`set_exercise()` builds a fresh analyser and a fresh session reference. Two things carry over: the depth target, which belongs to the squat, and the single camera's `PitchCertification` (§3), which belongs to the session.

## 3. `FrameContext` (passed to `observe_frame`, `observe` and the readiness checks)

The readiness checks (`get_readiness_check` / `analyzer.is_ready`) run before pre-IK and IK (`pipeline.py:896-919`). At that point `angles` is None and `skeleton_3d` is the raw, hip-centred skeleton. `observe_frame` and `observe` run after IK and get the full context.


`skeleton_3d`, `angles`, `timestamp`, `phase`, `in_rep`, `capture_mode`, `camera_pitch_deg` (NaN when not measured), `raw_confidences`, `body_proportions` (or None), `bar_detection` (or None), `foot_state` (or None), `up` and `gravity_source`.

Profiles may add fields additively.

- **Gravity (deadlift §2.1, shared).** On the rig, world "up" is the median standing hip→ankle direction, which can sit 2–4° off true gravity.
  - **`up`** is the unit vector pointing **physically up**, opposite to gravity, in the same Y-down world frame as the skeleton. Every metric uses `up` directly; nothing negates it.
  - Any sagittal rig metric with a long vertical lever uses the measured `up`. Examples: the press's bar-over-mid-foot and layback, the RDL's bar drift and lockout, the deadlift's D2/D3/D5.
    - It comes from the ChArUco-on-floor `measure_gravity.py` → `~/.nowva/gravity_<camera>.json`. That file stores physical gravity (pointing down); `up` = −gravity after it is mapped into the world frame.
    - The deadlift plan's "up = gravity" means this same axis.
  - `gravity_source` is one of `"measured"` (floor ChArUco), `"body"` (no measurement) or `"camera_pitch"` (one camera, pitch-corrected).
  - With no measurement, `up = WORLD_UP = (0, -1, 0)` and `gravity_source = "body"`. Those metrics are then cued only from moderate up, or stay recap-only, as each plan states.
  - Shared test: with both `"body"` and a tilted `"measured"` `up`, an upright trunk reads 0 ± 0.5° of lean, and a bar held in front of the feet reads a positive forward offset.
  - On one camera, `camera_pitch_deg` is `_standing_pitch_deg` (`pipeline.py:113-121`): a depth-regressed ankle→shoulder tilt that includes the lifter's posture. Holding a bar in front leans the body back 3–5°.
    - `gravity_source = "camera_pitch"` is therefore **unvalidated**. It is not a gravity measurement, and no cue may depend on it until validation shows it is within a stated error.
  - **Camera mount (shared `CaptureConfig`, one camera).** These fields describe the physical installation, not an estimate:
    - `mount_source`: `"rack"` (the product's fixed mount), `"inclinometer"` (entered at setup) or `"none"`;
    - `mount_pitch_deg`;
    - `mount_height_m`;
    - env override `NOWVA_CAMERA_PITCH_DEG`.

    `FrameContext` gets `pitch_certified: bool` and the mount values. The rules:
    - Each exercise states what certification requires for its own metrics, for example a pitch limit and a lens-height band, and what it unlocks.
    - On a certified mount, the pitch estimator is used only as a **delta** tripwire: the change from the reading recorded at certification, taken with the same posture. Posture and depth bias largely cancel in a delta.
    - If the delta exceeds the exercise's stated tolerance, certification is revoked for the session.
    - **State.** `PitchCertification` (`utils/camera_mount.py`) is session-owned and created by the pipeline. It holds the revoked flag and one tripwire per profile name, so a reading survives exercise switches and split sets (press → squat → press, an RDL pool with a press in between). Each exercise only compares readings taken in its own posture; the press catches a knock made during another exercise when it comes back.
    - **Signal.** A revocation or a gross trip sends `pipeline_status` with status `"camera_pitch"` and the reading in `details` (`{"delta_deg": …}` or `{"pitch_deg": …}`). The agent plays `level_camera` once a session.
  - **Pitch leak on one camera.** On one camera the world frame is aligned to the camera, not to gravity. With camera pitch θ, a "vertical" difference reads Δy·cos θ + Δz·sin θ. The sin term leaks forward and back motion into every height and tilt metric: 14% at 8°, 26% at 15°.
    - When `pitch_certified`, an analyser profile rotates the camera-frame skeleton about the camera X axis by `mount_pitch_deg` before computing any vertical or tilt metric. A shared helper does this: `geometry.level_camera_frame(points, pitch_deg)`. An inclinometer accurate to ±1° leaves a pitch-error leak under 2%.
    - **Levelling is not free.** Each levelled height becomes y·cos θ + z·sin θ, so it brings in the monocular regressed depth at sin θ.
      - Per-frame vertical noise becomes √(σ_y²cos²θ + σ_z²sin²θ). With σ_z ≈ 3·σ_y, that is +13% at 10°, +24% at 15° and +39% at 20°. Depth bias also enters at sin θ.
      - Each exercise budgets both terms with a stated σ_z assumption, and derives from that budget the largest certified pitch at which its mono metrics may start at their certified tier. Expect about 5–8° for tight allowances.
      - Above that limit, the mount is still levelled, but the uncertified tiers apply.
      - Synthetic levelling tests use anisotropic noise (σ_z = 3·σ_xy).
    - When the mount is not certified, each exercise states either a start tier for its mono vertical metrics that absorbs the leak (usually severe), or a hard pitch limit.
    - Setup guidance for every exercise is "step back, or turn the camera to portrait. Don't tilt it."
    - The squat is unchanged.
    - The squat keeps `WORLD_UP`.

## 4. Delivery side

- **Cue fallback (deadlift §4.3.1).** `cache_cues` carries `"fault_to_cue": profile.get_fault_to_cue_map()`. `orchestrator.set_exercise(name, is_squat, fault_to_cue)` stores the map, and `_add_fault_candidate` and the rest-complete focus line use it instead of the squat `FAULT_TO_CUE_MAP`. For the squat the map is `FAULT_TO_CUE_MAP`, so nothing changes.
- **Start tier and recorded-only verdicts.**
  - Profile attribute `min_cue_tiers: dict[fault_type, FaultSeverity]`, the static start tiers. A fault type that is missing starts at mild, so the squat is unchanged.
  - Profile method `cue_start_tier(fault_type, ctx: FrameContext, config: BiomechanicsConfig) -> FaultSeverity`. By default it looks the fault up in `min_cue_tiers`. A profile overrides it when a start tier depends on session state. Example: RDL hip sink starts at mild only while `ctx.pitch_certified`, and at moderate otherwise, including after the delta tripwire revokes certification mid-session.
  - The pipeline calls `cue_start_tier` when each verdict is formed, and is the only place that stamps `details["recorded_only"] = True` on a verdict below the start tier. Rules never stamp it themselves.
  - A cue of None is not enough on its own: the orchestrator reads a None cue as "decide here" and fills one in from the fault-to-cue map (`coaching_orchestrator.py:948-951`). Recorded-only verdicts are therefore filtered explicitly on every path:
    - **Bridge.** `ipc_bridge` sends them with `"recorded_only": true` on the `fault` message, and keeps them out of `rep_complete.faults_in_rep`. They stay in `faults_detailed`, flagged, for the DB and the recap.
    - **Service.** `coaching_service` passes `recorded_only` through to `orchestrator.on_fault(...)`.
    - **Orchestrator.** `_add_fault_candidate` drops recorded-only verdicts on both paths, the `fault` path and the `faults_detailed` loop (`:1016-1020`). It drops them the same way it drops approximate ones. `_passes_bandwidth` and `_resolve_cue_outcome` never see them.
    - **Persistence and summaries.** Recorded-only verdicts are stored, flagged, in the per-rep `faults_detailed` rows. Every aggregation skips faults whose `details.recorded_only` is true:
      - the session summary's `top_faults` (`db/biomechanics_persistence.py:181-215`);
      - `get_fault_progress` (`:853`);
      - `get_multi_session_fault_trends` (`:929`);
      - the set-summary `fault_summary` that `ipc_bridge.send_set_complete` produces (`:363-372`) for the display.
      So no "most common fault", trend or on-screen count ever rests on an untrusted verdict. A setup note (`details.setup_note`, for example the press grip, judged once per set) is skipped by the same aggregations, by `RepData.is_clean` and by both set-summary `fault_summary` loops: it rides on a rep but is not a rep fault. `evaluate_cue_outcome` needs no change, because a recorded-only verdict always reads as "decreased".
    - **One filter for every agent-side reader.** `agent/services/fault_filters.py::reportable_faults(faults, spoken: bool) -> list`:
      - It always drops `details.recorded_only` verdicts.
      - When `spoken` is true, it also drops `approximate` verdicts unless the caller hedges them.
      - Every agent-side reader of `faults_detailed` / `RepData.faults` must use it. That covers the orchestrator's `faults_detailed` loop, `coaching_service.last_rep_verdict()` (`:393-413`, which feeds `check_my_form`), the `show_me("last_rep")` replay narration (`workout_agent.py:1165-1175`), recap fault counts and progress summaries. No reader filters on its own.
    - **Shared tests.**
      - A recorded-only verdict that arrives as a `fault` with cue None, and again only in `faults_detailed`, is never cued and never counts as "persists".
      - A session whose only verdicts of some fault type are recorded-only produces no "most common fault" line for that type in `progress_context`, no trend entry, and no display fault count.
      - Take a rep whose only verdict of some fault is recorded-only, plus one approximate verdict of another fault. `check_my_form` and `show_me("last_rep")` must say there is nothing to correct in what this camera can see, or hedge the approximate one. They must never state either fault as fact.
  - A recorded-only verdict:
    - is excluded from `RepData.is_clean` (`utils/types.py`, the `is_clean` property);
    - is excluded from cue-outcome persistence (`_resolve_cue_outcome`). This gives every exercise one praise rule, decided over the existing hold window. Praise plays only after `FIX_HOLD_REPS` (2) hold reps (`coaching_orchestrator.py:1207-1235`), and that timing does not change.
      - If any hold rep carries a recorded-only verdict of the cued fault, the fault improved but is not proven gone. Play `<cue>_better` ("Better, …").
      - If no `_better` clip exists, play nothing. Never fall back to `_fixed`, which would overclaim.
      - Play `_fixed` only when no hold rep carries any verdict of the fault.
      - Each exercise drafts `<cue>_better` for every fault whose start tier is above mild.
      - Either praise resolves the pending outcome. After `_better`, `_fixed` is not played later for the same cue (no double praise). A new cue for the fault starts a new outcome.
      - Each exercise drafts `<cue>_better` for every fault whose start tier is above mild.
    - is excluded from recap fault counts. It may appear in the recap only hedged, and only when its fault type is listed in that exercise's `faults.<exercise>.hedged_recap_faults: set[str]`. That set is empty by default and filled from the validation report.
      - The hedge never states a size, because with a start tier of severe a moderate verdict can be recorded-only. Use wording like "possibly, hard to judge from this camera".
  - This replaces any per-exercise variant of `MILD_REPEAT_CUE_FAULTS` / `CUE_SEVERITIES`.
  - Squat behaviour stays byte-identical, because no squat fault gets a start tier above mild.
- **`waits_for_diagnosis`** is true for the squat, and for any profile with `create_set_diagnosis()`. It gates the recap's diagnosis wait. `is_squat` gates only the depth lines.
- **Set summary and recap depth.** When `profile.reports_depth` is False:
  - **Producer.** `ipc_bridge` / `session_tracker` send `avg_depth`, `depth_consistency` and the per-rep depth fields as null. `main.py` relays them unchanged, and the display hides null tiles. One place decides this, and it isn't `main.py`.
  - **Agent.** `cache_cues` carries `"reports_depth"`, and `orchestrator.set_exercise(..., reports_depth)` stores it. When it is False:
    - per-rep entries carry no `depth`/`depth_angle`;
    - `coaching_service` does not coerce a null depth to 0.0;
    - the recap's "best rep … deepest clean rep" line (`_build_rep_highlights`) is replaced. If the features carry `highlight_score` (higher is better, defined per exercise, with a one-line meaning such as "most hips-back"), the line names the best rep by it and uses that meaning. Otherwise the line is omitted.
  - **No depth claims.** Nothing in the recap may describe a non-depth exercise in depth terms.
- **Per-exercise orchestrator constant sets.** These are `SAFETY_FAULT_TYPE`, `SIDE_VIEW_FAULTS` and `SYMPTOM_FAULT_TYPES`, chosen by profile name. Squat values are unchanged.
- **Progress queries** pass `exercise=<profile name>`.
- **Cue audio.** `draft_cue_text.py --exercise <profile>` drafts each exercise's keys into `cues.json`. Every key needs a clip on disk before `coaching_ready` flips, because a missing clip triggers cloud TTS.

## 5. Config placement

Each exercise's tunables (thresholds and counter values) live in one pydantic model hung off `FaultsConfig`, for example `faults.press`, `faults.rdl` and `faults.deadlift`. `config.py:448-451` already auto-parses `FaultsConfig` sub-models from `config/biomechanics.yaml`.

There is no generic top-level section loader: `config.py:427-478` lists its sections by hand. The deadlift plan's top-level `deadlift:` section would therefore need its own parse line. That is a coordination item for the deadlift branch; the preferred fix is to move it under `faults:`.

## 6. Global maps (additive only)

| Map | Squat | Deadlift | RDL | OHP |
|---|---|---|---|---|
| `FAULT_CUE_PRIORITY` ranks | 0–11 (unchanged) | 20–30 | **40–49** | **50–59** |
| `FAULT_MEASUREMENT` entries | unchanged | `deadlift_*` | `rdl_*` | `press_*` |

New measurement levels, such as `rig_unvalidated`, are added to `faults/observability.py` once and shared.

## 7. Registry

`UntrackedVariantProfile(UntrackedProfile)` counts no reps, fires no faults, and makes Nova say the variant isn't coached yet. It is registered for:
- **Deadlift:** `sumo_deadlift`, `trap_bar_deadlift`, `hex_bar_deadlift`, `deficit_deadlift`, `snatch_grip_deadlift`, `single_leg_deadlift`, `rack_pull`.
- **RDL:** `single_leg_rdl`, `single_leg_romanian_deadlift`, `snatch_grip_rdl`, `snatch_grip_romanian_deadlift`, `stiff_leg_deadlift`, `dumbbell_rdl`, `dumbbell_romanian_deadlift`, `kettlebell_rdl`, `kettlebell_romanian_deadlift`, `deficit_rdl`, `deficit_romanian_deadlift`, `b_stance_rdl`, `b_stance_romanian_deadlift`.
- **Press:** `push_press`, `seated_press`, `seated_shoulder_press`, `dumbbell_shoulder_press`, `dumbbell_press`, `behind_the_neck_press`, `landmine_press`. The `shoulder_press` alias is removed from `OverheadPressProfile`.

Dev override: `NOWVA_DEV_COACHING_READY=<profile>[,<profile>]` adds profiles to `coaching_ready_profiles()`.

## 8. Safety net and tooling (one shared implementation)

- **Squat golden masters**, fresh and after `set_exercise(<other>)` → reps → `set_exercise("squat")`. The expected output is the fresh golden, apart from the documented carry-overs.
- **Raw recorder and replay.** Record the raw skeleton with confidences, the capture timestamp and control events (`set_exercise`, `rest_start`) on every processed frame, before the latch too. Replay runs the real pipeline from gate → pre-IK → IK → analyser/counter → rules.
  - Deadlift §8.2's rig video recorder is a superset; this skeleton recorder serves single-camera replay and is what OHP/RDL need first.

## 9. Reconciliation with deadlift plan v13 (applies at implementation)

The deadlift plan moved to v13 (`deadlift-PLAN-v13-reference.md` §3.1–3.5). Its shared names and hooks win wherever they overlap with this spec. The implementation follows this section where it differs from earlier text.

| Item | Decision |
|---|---|
| Start-tier table | Named `min_cue_tiers` (deadlift name). `cue_start_tier(fault_type, ctx, config)` stays as the session-aware override. |
| Keypoints and idle timeout | Profile attributes `tracking_keypoints` and `set_idle_timeout_s`. The defaults are squat values. |
| `feeds_body_calibration: bool` | New, default `True`. A profile whose poses would corrupt the standing body measurement sets it `False`. Each plan decides (the deadlift says False). |
| `gate_until_ready: bool` | New, default `False`. When True and the profile isn't coaching-ready (the dev override counts as ready), `get_profile` returns an `UntrackedProfile` that keeps the gated profile's `allows_camera_refine` and `feeds_body_calibration` values. This stops scheduled programs ("Barbell Romanian Deadlift", "Barbell Overhead Press") from running unvalidated rules. **OHP and RDL set it True** until their flip gates pass. |
| Dev override | `NOWVA_DEV_COACHING_READY=<profile>[,<profile>]`. Each profile computes `coaching_ready` from it at import, as a class attribute (deadlift §3.1). |
| Analyser rep signal | The analyser exposes `rep_signal: float`, NaN when unknown, updated in `observe`. When an analyser exists, the pipeline reads `analyzer.rep_signal` instead of `profile.get_rep_signal`. |
| `finish_rep` / rep start | This spec's `finish_rep(rep_number)` and `on_rep_start(piston)` are kept. They correspond to the deadlift's `finish_rep()` and `start_rep(liftoff=...)`. Whichever branch lands second renames to the first. |
| Activation | A single `_activate_profile(pipeline, session_tracker, bridge, exercise_name, meta=None)` in `pipeline_process.py` runs at startup and on every `set_exercise`. It sets `diagnosis_enabled`, installs `create_set_diagnosis()`, applies metadata and calls `prepare_exercise`. It replaces the separate startup lines and `_switch_exercise`. |
| `RuleEngine(reference=...)` | Optional argument, default `None`, meaning `SessionReference()`. It is passed from `profile.create_session_reference(config)`. |
| Config | Either form is fine: `faults.<exercise>` sub-models (OHP, RDL), or `<exercise>_*` fields in `FaultsConfig` plus a top-level section with an explicit parse branch (deadlift). |
| Analyser path in `RepData` | `max_depth_angle` comes from `features.max_depth_angle` (NaN allowed). `depth_target_met = True`. Features are dumped after `finish_rep`. |
