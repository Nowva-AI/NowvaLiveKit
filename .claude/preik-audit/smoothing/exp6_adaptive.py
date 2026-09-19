"""E6: noise-adaptive variants — KF with R from triangulation reprojection error; One Euro with shared skeleton speed."""
from __future__ import annotations

import itertools
import json

import numpy as np

import harness as H
import smoothers as sm

SEEDS = [0, 1, 2]


def score(make, sigma, feed="conf", lag=0):
    res = {}
    for kind in ("smooth", "hard"):
        ms = []
        for sd in SEEDS:
            d = H.make_data(sd, sigma, kind=kind)
            f = make(); f.reset()
            side = d["reproj"] if feed == "reproj" else d["conf"]
            out = np.stack([f.step(d["noisy"][i], d["ts"][i], side[i]) for i in range(len(d["noisy"]))])
            ms.append(H.fast_eval(out, d, lag))
        res[kind] = H.summarize(ms)
    return round((res["smooth"]["J"] + res["hard"]["J"]) / 2, 2), res


def kf_reproj(q, r, lag, gain, mode):
    k = sm.KalmanVec(2, q, r, lag, None)
    k.reproj_gain, k.reproj_mode = gain, mode
    return k


rows = []
for sigma in (2.0, 4.0, 8.0):
    for q, r in ((3.0, 0.01), (10.0, 0.02), (30.0, 0.02), (100.0, 0.04)):
        rows.append((sigma, "kf_cv_lag2_fixedR", dict(q=q, r=r), score(lambda: sm.KalmanVec(2, q, r, 2), sigma, lag=2)[0]))
    for q, gain, mode in itertools.product((3.0, 10.0, 30.0), (0.002, 0.003, 0.005), ("per_kpt", "frame_median")):
        rows.append((sigma, "kf_cv_lag2_R_from_reproj", dict(q=q, gain_m_per_px=gain, mode=mode),
                     score(lambda: kf_reproj(q, 0.01, 2, gain, mode), sigma, feed="reproj", lag=2)[0]))
    for mc, beta, dc in itertools.product((0.3, 0.8), (1.0, 2.0, 4.0, 8.0), (1.0, 2.0)):
        rows.append((sigma, "oneeuro_shared_speed_paper", dict(min_cutoff=mc, beta=beta, d_cutoff=dc),
                     score(lambda: sm.OneEuroShared(mc, beta, dc, "filtered"), sigma)[0]))
    print("sigma", sigma, "done", flush=True)

json.dump(rows, open("exp6_results.json", "w"), indent=0)
for sigma in (2.0, 4.0, 8.0):
    print(f"\n## sigma {sigma:g}px  (J_mean; compare exp4 best: OE 4.09/7.09/11.75, KF lag2 3.42/6.00/9.77)")
    for fam in ("kf_cv_lag2_fixedR", "kf_cv_lag2_R_from_reproj", "oneeuro_shared_speed_paper"):
        sub = sorted([r for r in rows if r[0] == sigma and r[1] == fam], key=lambda r: r[3])[:3]
        for r in sub:
            print(f"  {fam:28s} {json.dumps(r[2]):60s} J_mean {r[3]:.2f}")
# robustness: single parameter set across sigmas
print("\n## single setting across sigma 2/4/8")
keys = {(r[1], json.dumps(r[2])) for r in rows}
table = []
for fam, p in keys:
    js = [next(r[3] for r in rows if r[0] == s and r[1] == fam and json.dumps(r[2]) == p) for s in (2.0, 4.0, 8.0)]
    table.append((sum(js), fam, p, js))
for tot, fam, p, js in sorted(table)[:10]:
    print(f"  {fam:28s} {p:60s} {js}")
