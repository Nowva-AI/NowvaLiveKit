# Conventional deadlift: validation analysis plan (skeleton)

PLAN.md §1 and §8.3. This file **pre-registers** how the J6 demo gate is judged, so the analysis
is fixed before anyone sees round-2 results.

**Status: skeleton.** The structure, metrics, statistics and decision rules are set here. The
open values marked **[R1]** are filled from round-1 data. The file is then frozen by a commit,
before round 2 is labelled. After the freeze, any change is a dated amendment at the end, with
its reason, and it never applies to results already seen.

## 1. Questions

1. Does the analyser count deadlift reps on real lifters as well as the gate requires?
2. Are its events on time?
3. When it cues a hero fault (D1, D3, D2, D4), is the fault there (precision)?
4. When the fault is there, is it cued (recall)?
5. On clean reps, how often does it cue a correction that is not needed?
6. Is each cued fault's measurement error within its error budget (PLAN.md §2.9)?

## 2. Data

- **Round 1 (team, tuning only):** 2–3 people, from an empty bar up to light loads. It sets
  the thresholds, the min tiers, ρ and the hip-keypoint bias. **No round-1 rep is ever a test
  rep.**
- **Round 2 (test):**
  - ≥ 10 lifters never seen in tuning, 1.55–1.95 m, with varied femur, torso and arm ratios,
    and ≥ 4 novices.
  - 2 sessions per lifter. Natural sets in observation mode (cues off), plus scripted sets.
- **Sizes:**
  - Hero faults: ≥ 15 positives per lifter (≥ 150 per hero fault).
  - Provisional faults: ≥ 5 positives per lifter.
  - Clean reps: ≥ 30 per lifter.
  - About 1,150 reps in total.
  - The design effect, 1 + (m − 1)ρ, uses ρ = 0.1 until **[R1]** replaces it.
- **Recording:** every set is recorded raw (`scripts/tools/record_rig.py`), and every metric
  is computed by replaying the recording through the real pipeline (`NOWVA_REPLAY_DIR`), at
  the commit frozen in §8.

## 3. Ground truth and labels

- **Measured quantities** come from the PLAN.md §8.4 instruments on instrumented sessions:
  - D1: floor tape and a foot jig.
  - D3 and D10: ArUco hub markers.
  - D2, D5 and D6: a 120 fps sagittal camera with trochanter and acromion markers.
  - D4: the trochanter marker.
  - Gravity: board and spirit level.
- **Scripted sets** carry their label by construction, checked against the instruments.
- **Natural sets** are labelled in CVAT on three views: faults and severity per rep, and event
  timestamps on a subset.
  - The deadlift engineer labels everything. Ambaka double-labels 20 % and adjudicates.
  - Cohen's κ is computed per fault. A fault with κ < 0.6 is redefined before its result is
    reported.
- **Labellers do not see the analyser's output.**

## 4. Endpoints

All are computed per rep, then aggregated as in §5. "Cued" means at or above the fault's
effective min tier (`details.min_tier`), as the voice agent would speak it.

| Endpoint | Definition | Demo gate (J6) |
|---|---|---|
| Rep counting | counted reps that match a labelled rep ÷ labelled reps; phantom reps on clean sets counted separately | ≥ 99 %, 0 phantom reps on clean sets |
| Event timing | \|detected − labelled\| for liftoff, knee pass, top and floor, on the timestamped subset | median ≤ 100 ms, per event |
| Hero-fault precision | cued reps with the fault ÷ cued reps, per hero fault | ≥ 0.85; 95 % lower bound ≥ 0.70; ≥ 0.80 on ≥ 30 natural cases |
| Hero-fault recall | cued reps with the fault ÷ reps with the fault, per hero fault | ≥ 0.70; lower bound ≥ 0.55 |
| False corrections | cued faults on labelled-clean reps ÷ clean reps | ≤ 1 per 10 reps |
| Measurement error | P95 \|measured − instrument\| per measured fault | ≤ 1/3 of the lowest cued tier's threshold |
| Squat | golden masters, fresh and after a deadlift switch | identical |

**Rules for counting:**
- A fault "is there" when the label's severity is at or above the cued tier's threshold.
- A rep labelled one tier below counts as **near miss**: reported, but not counted against
  precision.

## 5. Statistics

- **Unit of resampling:** the lifter (a cluster).
- **Primary interval:** lifter-clustered percentile bootstrap, 10,000 resamples, seed fixed
  in §8. Lower bounds are one-sided 95 %.
- **Secondary:**
  - Bootstrap-t on the same resamples.
  - Leave-one-lifter-out: the gate must still pass with any single lifter removed, or the
    result is flagged as fragile.
- **Per-lifter floor:** precision ≥ 0.70 for each lifter with ≥ 15 cues.
- **Natural sets** are reported separately from scripted sets. A hero fault must reach ≥ 0.80
  precision on ≥ 30 natural cases.
- **No re-analysis:** a result within 0.02 of a gate threshold goes to round 3. It is never
  re-analysed with a different method.

## 6. Exclusions (fixed now)

Exclude a set, and report how many were excluded, only when:
- the recording is incomplete: a camera is lost for > 10 % of the set, or the timestamps are
  corrupt;
- the calibration's reprojection RMS exceeds 2× its installed baseline, or no gravity
  measurement exists for an instrumented session;
- the lifter withdraws consent;
- an instrument failed, for measured quantities only.

**Never** exclude a set because the analyser did badly on it.

## 7. Decisions

- **Gate passes:** the fault keeps its min tier, or moves down one tier if its measurement
  error allows it (PLAN.md §2.9).
- **Hero fault fails:**
  - D2 or D4: Plan B. The demo uses Demo α (D1 + D3), as accepted in advance (PLAN.md §8.3).
  - D1 or D3: the demo waits for a fix and a new round.
- **D2 specifically:**
  - D2's size comes from the setup model, which only this data validates
    (`docs/deadlift/KNOWLEDGE.md` §5). Whether D2 fires at all comes from the model-free rise
    ratio, rep and set.
  - Report the model's predicted trunk change against the instrument-measured change on
    labelled-good reps.
  - Report the rise ratio's distribution on labelled held-back-angle reps and on labelled
    hips-first reps. The 1.05 / 1.15 / 1.30 gates are re-set from these before the D2 result
    is reported.
  - If the model's error makes good reps read ≥ 5° (the mild threshold), D2 is sized on the
    raw trunk change instead.

## 8. Freeze record (filled at the freeze)

| Item | Value |
|---|---|
| Analyser commit | **[freeze]** |
| `config.py` thresholds and min tiers | **[R1]** |
| ρ (round 1) | **[R1]** |
| Hip-keypoint bias correction | **[R1]** |
| Bootstrap seed | **[freeze]** |
| Date, signed by | **[freeze]** |

## Amendments

None yet.
