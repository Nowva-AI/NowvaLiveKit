"""Can the planted-foot anchors recover world-frame roll tilt (floor normal) for gravity correction? Robust model, 5 deg roll."""
from __future__ import annotations
import math
import numpy as np
from sim import CK, NoiseModel, generate, measure
from run import SCENARIOS
from proto import FootContactModelRobust

gen = generate(SCENARIOS["tilt_roll5"])
for sigma in (0.005, 0.015, 0.03):
    est = []
    for seed in range(8):
        meas = measure(gen, NoiseModel(sigma, 0.4 * sigma), seed)
        model = FootContactModelRobust(19)
        for i in range(59):  # initial standing (2 s)
            model.process(meas["meas_world"][i, :19])
        rows = []
        for ankle_name in ("ankle", "toe"):
            a_l = model._sides[0]["kp"][ankle_name].anchor
            a_r = model._sides[1]["kp"][ankle_name].anchor
            if a_l is None or a_r is None:
                continue
            rows.append(math.degrees(math.atan2(a_l[1] - a_r[1], a_l[0] - a_r[0])))
        if rows:
            est.append(np.mean(rows))
    est = np.array(est)
    print(f"sigma {sigma*100:.1f} cm: roll estimate {est.mean():.2f} +- {est.std():.2f} deg (true 5.00) after 2 s standing (None anchors skipped)")
