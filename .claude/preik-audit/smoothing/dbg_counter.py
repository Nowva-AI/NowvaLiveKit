import numpy as np, harness as H, kin
from biomechanics.faults.hip_position_counter import SignalRepCounter
d = H.make_data(0, 0.0, noise="none")
c = SignalRepCounter(H.CONFIG.hip_counter)
prev = None
sig = kin.rep_signal_cm(d["truth"])
for i in range(len(sig)):
    rd, fb = c.update(signal_value=float(sig[i]), timestamp=float(d["ts"][i]))
    vel = c._vel_ema.value
    if c.phase != prev or rd is not None or fb:
        print(i, d["phase"][i], "->", c.phase, "sig %.1f base %.1f vel %.1f" % (sig[i], c._standing_baseline or 0, vel or 0), "REP" if rd else "", fb or "")
    prev = c.phase
print(d["windows"])
