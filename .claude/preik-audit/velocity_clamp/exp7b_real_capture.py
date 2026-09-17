"""Drive the REAL MultiCameraCapture.get_synced_frames with fake 30 fps cameras (random phase, read jitter) and the
pipeline_process pacing (sleep target - processing)."""
import sys, time, threading, json
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
import numpy as np
import biomechanics.triangulation.multi_capture as mc

T0 = time.perf_counter()


class FakeCap:
    def __init__(self, dev_id):
        rng = np.random.default_rng(dev_id * 7 + int(sys.argv[1]) if len(sys.argv) > 1 else dev_id)
        self.fps = 30.0 * (1.0, 0.9993, 1.0011)[dev_id]
        self.phase = rng.uniform(0, 1 / self.fps)
        self.k = 0
        self.rng = rng
        self.dev = dev_id
    def set(self, *a): return True
    def isOpened(self): return True
    def get(self, prop): return 30.0
    def read(self):
        # frame k captured at T0 + phase + k/fps, delivered after 8 ms +- 2 ms
        while True:
            cap_t = T0 + self.phase + self.k / self.fps
            deliver = cap_t + max(0.001, 0.008 + self.rng.normal(0, 0.002))
            self.k += 1
            now = time.perf_counter()
            if deliver >= now:
                time.sleep(deliver - now)
                frame = np.full((2, 2, 1), 0, dtype=np.uint8)
                frame_id = self.k - 1
                return True, (self.dev, frame_id, cap_t)
    def release(self): pass


mc.cv2.VideoCapture = FakeCap
cap = mc.MultiCameraCapture([0, 1, 2], max_sync_delta_ms=15.0)
cap.start()
time.sleep(0.5)
calls = none = dup = 0
used = []
spreads = []
last = None
end = time.perf_counter() + 25.0
PROC_MS, TARGET_MS, OVERHEAD_MS = 15.0, 1000 / 30, 2.0
while time.perf_counter() < end:
    calls += 1
    t_start = time.perf_counter()
    r = cap.get_synced_frames()
    if r is None:
        none += 1
        time.sleep(TARGET_MS / 1000 + OVERHEAD_MS / 1000)   # latency ~0 -> full sleep (pipeline_process.py:1363)
        continue
    frames, ts = r
    fid = frames["0"][1]
    if fid == last:
        dup += 1
    last = fid
    used.append(fid)
    caps = [frames[c][2] for c in ("0", "1", "2")]
    spreads.append((max(caps) - min(caps)) * 1000)
    time.sleep(PROC_MS / 1000)                                  # pose + triangulation
    time.sleep(max(TARGET_MS - PROC_MS, 0) / 1000 + OVERHEAD_MS / 1000)
cap.release()
used = np.array(used)
span = used.max() - used.min() + 1
res = dict(calls=calls, none_pct=round(100 * none / calls, 1), dup_pct=round(100 * dup / len(used), 1),
           skipped_primary_pct=round(100 * (1 - len(np.unique(used)) / span), 1),
           processed_hz=round(len(np.unique(used)) / 25.0, 1),
           spread_ms_p50=round(float(np.percentile(spreads, 50)), 1), spread_ms_p95=round(float(np.percentile(spreads, 95)), 1))
print(json.dumps(res))
