# GroundClamp Audit

(Saved by lead from the agent's final message; subagent sandbox blocked writing REPORT.md. Checkpoints in PROGRESS.md. No repo files changed. Run scripts from this folder with `/Users/naiahoard/NowvaLiveKit/venv/bin/python`.)

**Verdict: REMOVE GroundClamp in both triangulated and single-camera mode; rebuild the ground constraint as a per-foot contact model in world coordinates.**

## Method
- `sim.py`: 1.885 m subject, exact bone lengths, feet planted on a real floor. Faults: valgus, heel rise (one/both feet), 6 cm hip shift, 6° pelvic list, stance change after calibration, 5° roll/pitch world tilt, early descent during calibration, soft knees during calibration, "foot snaps to knee at bottom" (then confirmed on real video).
- Noise 1.5 cm white + 0.6 cm slow drift per axis, then hip re-centering; sweep 0.5–3 cm, 8 seeds.
- Real classes in production order: StandingPoseGate, BoneLengthConstraints, GroundClamp, One Euro smoother. Variants: raw; GC only; bone → GC → bone; full chain with/without GC; two prototypes in `proto.py` (naive and robust contact models).

## Why GroundClamp can't work here
Triangulator re-centers each frame on the measured hip midpoint (`triangulator.py:134-137`) → no floor; the clamp stores how far the ankle sits below the hips while standing.
- **Floor clamp** (`ground_clamp.py:89-94`): hips drop toward ankles in a squat, so the value only shrinks → never fires legitimately. Fires on noise on 30 % of frames (~75 % of standing frames); raises standing ankle 0.91 cm, shortens tibia 0.81 cm (knee never moved). Calibration catching early descent → 2.80 cm; soft knees → 3.77 cm.
- **Equal ankle heights** (`:97-101`): pelvic list and hip shift don't change hip-relative ankle heights (not the trigger); fires on noise (52–63 % of frames), world tilt, one-sided heel rise, pose failures.
- **Stance width lock** (`:104-117`): snaps back whenever error > 2 cm — 41 % of frames on noise alone, 97 % after a real stance change; moves ankles sideways without knees.
- **Calibration** (`:75-83`): 30 frames right after standing gate latches, no per-frame check, recalibrates every set; `BoneLengthConstraints.reset()` un-latches the shared standing gate → typically calibrates during walk-out. Leg-extension check accepts early-descent / soft-knee contamination. No YAML section → defaults (`config.py:211-218`). Tests assert the erasing behaviours as correct.

## Findings
| # | Sev | What goes wrong | Measured |
|---|---|---|---|
| G1 | critical | Stance lock after walk-out at 30 cm, squat at 42 cm | Width stuck 11 cm narrow; KASR +41 %; true valgus 15.2° reads 8.0° (moderate fault missed, below 12° mild). Live + diagnosis stance ratios freeze → "widen your stance" can never register |
| G2 | critical | Equalization with one-sided heel rise 3.37 cm | Erased to 0.04 cm (99 %); 2.24 cm after 2nd bone pass; pushes other foot 1.7 cm into floor |
| G3 | high | 5° world roll tilt | True 3.31 cm ankle height difference erased; fires 92 % of frames |
| G4 | high | Floor clamp = median of standing noise | See above |
| G5 | high | GC then bone pass 2 | Pass 2 moves ankle back down but not the toe → clean squat shows fake +1.4 cm heel rise; real 3.37 cm bilateral rise reads 5.42 cm |
| G6 | high | Foot keypoints snap to knee at bottom | Knee flexion error at bottom RMSE: raw 41.3°, GC 20.6°, full chain 30.9°; valgus error GC 13.8°, full 16.9°; bridge depth feature off 16.4 cm with GC. Robust prototype: 2.8°, 4.3°, 0.05 cm |
| G7 | medium | Calibration timing | Walk-out/unrack locked in |
| G8 | medium | Heels dropped (`rtmpose.py:41-44, 268-273`) | Standing gate flat-foot check silently skipped in triangulated mode (`standing_gate.py:70`) |
| G9 | medium | No heel-rise rule | Dorsiflexion vs world vertical (`analytical_ik.py:272`) → 5° tilt = 5° bias; `bridge.py:34` re-grounds every frame (F19) |
GroundClamp does NOT distort pelvic list (2.80 vs 2.86 cm true) or lateral hip shift (5.99 vs 6.00 cm).

## Real-video evidence
Production RTMPose-m halpe26 decode on 5 real squat recordings (32 reps, single camera each): at squat bottom ankle/toe/heel keypoints move > 10 cm in 56–78 % of reps (pooled median 16–36 cm; px→cm approximate). Frames 122 and 160 of `squat_20260512_134408_frames.jpg` show a flat foot's keypoints drawn at knee height. Confidence doesn't flag it (0.654 failures vs 0.651 good). Every camera runs the same decode + unweighted triangulation → feet least reliable exactly where depth, valgus, dorsiflexion are read.

## Robust contact prototype vs raw (σ = 1.5 cm)
- Ankle/toe error 2.77 → 0.49 cm, 2.79 → 0.39 cm; knee flexion error at bottom 3.36 → 2.83°.
- Stance width error 2.28 → 0.42 cm; stance changes tracked (bias −0.08 cm).
- Heel rise: 3.06 of 3.37 cm on rising foot, −0.04 other; both feet 2.85 cm; 2.66 cm even with foot failures.
- Tilt: 3.48 cm kept of 3.31 true. Valgus + stance change bias −0.19° (GC −7.2°). Roll tilt from foot anchors 5.21 ± 0.47° vs 5° true.
- Holds contact on 91–93 % of frames at 3 cm noise; naive version 29 % and 13.8 cm false heel rise when a foot snaps → robustness required.
- 89 µs/frame plain Python, causal, no optimizer.
- Not solved: valgus noise from hip/knee keypoints (4.66 → 4.26°) — model only fixes feet.

## Rebuild design
1. **World coordinates:** run before hip re-centering, or triangulator also returns the subtracted hip centre (fixed cameras → fixed world frame). First pre-IK stage for foot keypoints.
2. **Per foot, per keypoint:** ignore per-frame jumps > max(8 cm, 6σ) as pose failures (σ learned online). Thresholds: move max(3 cm, 4σ/√6), rise max(2 cm, 3σ/√6). Planted → output anchor (running mean). Heel rise = ankle/heel rise with toe planted → release ankle+heel, keep toe held. Foot moved = toe and ankle both move → release whole foot (still reject teleports). Re-plant within 5 cm keeps old floor reference; a raised hold never becomes floor. Never couple L/R; never move knees or hips.
3. **`FootState` output:** planted flags, heel rise cm per foot; stance width + toe-out from anchors; hip height above floor (replaces F19 per-frame grounding; noise 1.15 vs 1.76 cm); lateral hip offset from foot midpoint; floor roll angle.
4. **Keypoints:** map halpe26 heels 24/25 → 19/20 (optionally small toes 22/23); triangulator 21+ kpts.
5. **New rule** `faults/rules/heel_rise.py`, three tiers (e.g. 1.5/3/5 cm, to validate); standing gate flat-foot check reads `FootState`.
6. **Lifecycle:** no calibration window, no reset at set boundaries. Pitch tilt can't be seen from one stance → from rack camera geometry at install.
7. **Single camera:** remove GroundClamp; if wanted, contact in 2D image space.
Validate on real 3-cam recordings: ankle keypoint drift with shin angle (2–9 cm in cleanest video), slow triangulation drift.

## Out of scope, for the lead
- BoneLengthConstraints erases ~56 % of pelvic list (2.86 → 1.25 cm) via shoulder→hip pairs (`bone_constraints.py:59-62`) → corrupts weight-shift evidence (`evidence_tests.py:149`).
- 3D valgus noise bias −3.3° at σ 1.5 cm (`valgus.py:294-306`).
- `_hip_internal_rotation` returns an unsigned angle despite its docstring (`valgus.py:334`).
- One Euro smoother drives most of the full chain's error.
- RTMPose foot failures need the upstream person-crop fix.

## Files
`sim.py`, `proto.py`, `run.py`, `table.py`; `results_sigma1.5.json`, `results_sweep.json`, `table_sigma1.5.txt`, `table_sweep.txt`; `diag_bottom.py`, `diag_stand.py` (G5); `tilt_estimate.py`; real video `real_foot_drift.py`, `real_foot_stats.py`, `real_foot_reps.py` (+JSON), `real_traces.png`, `*_frames.jpg`, `*_h26.npy`; `PROGRESS.md`.
