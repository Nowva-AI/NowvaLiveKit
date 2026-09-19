"""Per-phase trace of the proposed chain's foot model on one (scenario, calibration, seed): planted flags, stance
width error, lower-body error and FootState heel rise, split into still / moving / rep windows. Diagnostic only.
Run: venv/bin/python trace_foot.py walkout tpose 0
"""

from __future__ import annotations

import sys

import numpy as np

from preik_harness import get_prepared, production_chain, proposed_chain
from preik_harness.metrics import _stance_width
from preik_harness.runner import run_chain

M_TO_MM = 1000.0
M_TO_CM = 100.0


def main() -> None:
    scenario, calibration, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    prepared = get_prepared(scenario, seed, calibration)
    sc = prepared.scenario
    times = prepared.truth_time_s
    truth = prepared.truth_hc[:, :19]
    width_true = _stance_width(truth)
    records = {name: run_chain(prepared, factory) for name, factory in
               (("raw", production_chain([])), ("nofoot", proposed_chain(False)), ("foot", proposed_chain(True)))}
    still = sc.still_mask(times)
    phases: list[tuple[str, np.ndarray]] = []
    if sc.moving_intervals:
        start, end = sc.moving_intervals[0][0], sc.moving_intervals[-1][1]
        phases.append(("still before move", still & (times < start)))
        phases.append(("moving", (times >= start) & (times <= end)))
        phases.append(("still after move, before reps", still & (times > end) & (times < sc.reps[0].start_s)))
    else:
        phases.append(("still before reps", still & (times < sc.reps[0].start_s)))
    for k, rep in enumerate(sc.reps):
        phases.append((f"rep {k} ({rep.start_s:.1f}-{rep.end_s:.1f}s)", (times >= rep.start_s) & (times <= rep.end_s)))
    phases.append(("still between/after reps", still & (times > sc.reps[0].start_s)))
    ankle_y_true = 0.5 * (prepared.truth_world[:, 15, 1] + prepared.truth_world[:, 16, 1])
    base = np.median(ankle_y_true[still & (times < sc.reps[0].start_s)])
    print(f"{scenario} {calibration} seed {seed}: ticks {len(times)}, moving {sc.moving_intervals}")
    header = f"{'phase':36s} {'n':>4s} | " + " | ".join(f"{name:>28s}" for name in records)
    print(header)
    print("  columns per chain: lower MPJPE mm / stance err mm (median) / planted L,R frac / FootState rise cm (max)")
    for label, mask in phases:
        cells = []
        for name, rec in records.items():
            sel = mask & (rec.status == 2)
            if sel.sum() == 0:
                cells.append(f"{'-':>28s}")
                continue
            err = np.linalg.norm(rec.out_xyz[sel][:, 11:19] - truth[sel][:, 11:19], axis=2)
            width_err = np.abs(_stance_width(np.nan_to_num(rec.out_xyz[sel])) - width_true[sel])
            planted = rec.foot_planted[sel].mean(axis=0)
            rise = rec.foot_heel_rise_cm[sel]
            rise_max = np.nanmax(rise) if np.isfinite(rise).any() else float("nan")
            cells.append(f"{np.nanmean(err) * M_TO_MM:5.1f} / {np.median(width_err) * M_TO_MM:5.1f} / "
                         f"{planted[0]:.2f},{planted[1]:.2f} / {rise_max:5.2f}")
        true_rise = (base - ankle_y_true[mask]).max() * M_TO_CM if mask.any() else float("nan")
        print(f"{label:36s} {int(mask.sum()):4d} | " + " | ".join(cells) + f"   true ankle rise max {true_rise:.2f} cm")


if __name__ == "__main__":
    main()
