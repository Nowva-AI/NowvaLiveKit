"""Plot hip / knee / foot keypoint pixel traces for the real squat videos (visual segmentation aid)."""
from __future__ import annotations
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

files = sorted(glob.glob(str(Path(__file__).parent / "*_h26.npy")))
fig, axes = plt.subplots(len(files), 2, figsize=(16, 3.2 * len(files)))
for row, f in enumerate(files):
    a = np.load(f)
    t = np.arange(len(a))
    ax = axes[row, 0]
    for k, lab in ((11, "hipL"), (13, "kneeL"), (15, "ankL"), (20, "toeL"), (24, "heelL"), (16, "ankR"), (25, "heelR")):
        ax.plot(t, a[:, k, 1], label=lab, lw=0.8)
    ax.invert_yaxis(); ax.set_title(Path(f).stem + " y px"); ax.legend(fontsize=6, ncol=7)
    ax = axes[row, 1]
    for k, lab in ((11, "hipL"), (12, "hipR"), (15, "ankL"), (16, "ankR"), (20, "toeL"), (21, "toeR"), (24, "heelL"), (25, "heelR")):
        ax.plot(t, a[:, k, 0], label=lab, lw=0.8)
    ax.set_title("x px"); ax.legend(fontsize=6, ncol=8)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "real_traces.png", dpi=70)
