"""Triangulated mode: BiLSTM RepData.end_time uses skeleton perf_counter clock; SessionTracker.check_set_timeout uses time.time()."""
import sys, time
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
import numpy as np
from biomechanics.ml.bilstm_counter import BiLSTMRepCounter, BiLSTMCounterConfig
from biomechanics.coaching.session_tracker import SessionTracker


class StubBridge:
    def __init__(self): self.sent = []
    def __getattr__(self, name):
        def _f(*a, **k): self.sent.append(name)
        return _f


for label, clock in (("triangulated (perf_counter skeleton ts)", time.perf_counter), ("mediapipe (time.time skeleton ts)", time.time)):
    bridge = StubBridge()
    tracker = SessionTracker(bridge)
    counter = BiLSTMRepCounter(BiLSTMCounterConfig())
    reps, set_ends = 0, 0
    # 3 reps: 10 frames standing (class 0), 20 frames deep (class 3), repeated
    for rep in range(3):
        for cls, n in ((0, 10), (3, 20), (0, 10)):
            for _ in range(n):
                probs = np.zeros(5); probs[cls] = 1.0
                rep_data, _ = counter.update(probs, timestamp=clock(), frame_index=0)
                if rep_data is not None:
                    reps += 1
                    tracker.on_rep_complete(rep_data)
                if tracker.check_set_timeout(time.time()):   # pipeline_process.py:1287
                    set_ends += 1
    print(f"{label}: reps={reps} set_complete_events={set_ends} total_sets={tracker.total_sets} "
          f"last_rep_time={tracker.last_rep_time:.1f} bridge_calls={sorted(set(bridge.sent))}")
