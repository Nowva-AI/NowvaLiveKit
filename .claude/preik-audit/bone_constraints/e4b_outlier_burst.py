"""E4b: outlier burst on L knee INSIDE the 30-frame calibration window (after the standing gate latched)."""
from __future__ import annotations
import logging
import numpy as np
from synth import sequence, add_correlated_noise
from methods import lengths_from
from biomechanics.utils.types import CocoKeypoints as CK
from e3_sequences import prod_calibration
logging.disable(logging.WARNING)
N = 600
print("burst of B frames, L knee displaced 15 cm DOWN (along the leg) starting at the first recorded calibration frame; sigma 1.5 cm, rho 0.6")
for burst in (0, 5, 10, 14, 15, 16, 20, 30):
    errs, run_errs = [], []
    for seed in range(20):
        rng = np.random.default_rng(300 + seed)
        truth, s = sequence(N)
        noisy, _ = add_correlated_noise(truth, rng, 0.015, rho=0.6, z_scale=1.5)
        _, used0 = prod_calibration(noisy)
        start = used0[0]
        noisy[start:start + burst, CK.LEFT_KNEE] += np.array([0.0, 0.15, 0.0])
        bc, used = prod_calibration(noisy)
        L_true = lengths_from(truth[0])
        fem = (CK.LEFT_HIP, CK.LEFT_KNEE); tib = (CK.LEFT_KNEE, CK.LEFT_ANKLE)
        errs.append([1000 * (bc._calibrated_lengths[fem] - L_true[fem]), 1000 * (bc._calibrated_lengths[tib] - L_true[tib])])
        Lf = np.linalg.norm(noisy[:, fem[1]] - noisy[:, fem[0]], axis=1)
        run_errs.append(1000 * (np.median(Lf) - L_true[fem]))
    errs = np.array(errs)
    print(f"  burst {burst:2d}: stand30 femur {errs[:,0].mean():+7.1f} mm, tibia {errs[:,1].mean():+7.1f} mm | whole-assessment median femur {np.mean(run_errs):+5.1f} mm")
