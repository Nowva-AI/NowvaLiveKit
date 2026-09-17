"""E1: truth sequences have constant bone lengths; production metrics see the injected faults."""
from __future__ import annotations
import numpy as np
from synth import sequence, body
from metrics import seq_metrics
from methods import PAIRS19, lengths_from

cases = {
    "clean": {},
    "valgus_l10": dict(valgus_deg_l=10.0),
    "hip_shift_6cm": dict(hip_shift_m=0.06),
    "list_3cm": dict(list_m=0.03),
    "heel_rise_4cm": dict(heel_rise_m=0.04),
}
for name, fault in cases.items():
    truth, s = sequence(420, **fault)
    L = np.array([[np.linalg.norm(f[d] - f[p]) for p, d in PAIRS19] for f in truth])
    spread = (L.max(0) - L.min(0)).max() * 1000
    m = seq_metrics(truth)
    bottom = np.argmax(s)
    print(f"{name:14s} max bone-length spread {spread:.3f} mm | stand/bottom: knee_l {m['knee_flex_l'][0]:.1f}/{m['knee_flex_l'][bottom]:.1f} "
          f"valgus_l {m['valgus_l'][0]:.1f}/{m['valgus_l'][bottom]:.1f} valgus_r {m['valgus_r'][bottom]:.1f} trunk {m['trunk_flex'][0]:.1f}/{m['trunk_flex'][bottom]:.1f} "
          f"hip_shift {m['hip_shift_cm'][0]:.1f}/{m['hip_shift_cm'][bottom]:.1f} heel_l {m['heel_rise_cm_l'][0]:.1f}/{m['heel_rise_cm_l'][bottom]:.1f} "
          f"hip_asym {m['hip_asym_cm'][bottom]:.1f} depth {m['depth_cm'][0]:.1f}/{m['depth_cm'][bottom]:.1f} pelvis_list {m['pelvis_list'][bottom]:.1f}")
b = body()
print({k: round(v, 3) for k, v in lengths_from(sequence(1)[0][0]).items()})
