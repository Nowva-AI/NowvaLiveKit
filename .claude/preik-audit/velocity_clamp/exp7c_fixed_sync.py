"""Prototype fix for get_synced_frames: newest *complete, never-returned* set, short wait instead of None; no sleep pacing."""
import sys, time, json
sys.argv = sys.argv[:2]
import numpy as np
exec(open("exp7b_real_capture.py").read().split("mc.cv2.VideoCapture = FakeCap")[0])
mc.cv2.VideoCapture = FakeCap


def get_next_synced(self, last_ts, timeout_s=0.1):
    primary_id = str(self._device_ids[0])
    deadline = time.perf_counter() + timeout_s
    while True:
        with self._locks[primary_id]:
            prim = list(self._buffers[primary_id])
        for ref_ts, ref_frame in reversed(prim):
            if last_ts is not None and ref_ts <= last_ts:
                break
            result = {primary_id: ref_frame}
            ok = True
            for dev_id in self._device_ids[1:]:
                cam_id = str(dev_id)
                with self._locks[cam_id]:
                    buf = list(self._buffers[cam_id])
                if not buf:
                    ok = False; break
                ts_best, fr_best = min(buf, key=lambda e: abs(e[0] - ref_ts))
                if abs(ts_best - ref_ts) > self._max_sync_delta_s:
                    ok = False; break
                result[cam_id] = fr_best
            if ok:
                return result, ref_ts
        if time.perf_counter() > deadline:
            return None
        time.sleep(0.002)


cap = mc.MultiCameraCapture([0, 1, 2], max_sync_delta_ms=15.0)
cap.start()
time.sleep(0.5)
calls = none = dup = 0
used, spreads = [], []
last_ts = None; last = None
end = time.perf_counter() + 25.0
while time.perf_counter() < end:
    calls += 1
    r = get_next_synced(cap, last_ts)
    if r is None:
        none += 1; continue
    frames, ts = r
    last_ts = ts
    fid = frames["0"][1]
    dup += fid == last
    last = fid
    used.append(fid)
    caps = [frames[c][2] for c in ("0", "1", "2")]
    spreads.append((max(caps) - min(caps)) * 1000)
    time.sleep(0.015)   # pose + triangulation; NO pacing sleep (the blocking call paces the loop)
cap.release()
used = np.array(used)
span = used.max() - used.min() + 1
print(json.dumps(dict(calls=calls, none_pct=round(100 * none / calls, 1), dup_pct=round(100 * dup / len(used), 1),
                      skipped_primary_pct=round(100 * (1 - len(np.unique(used)) / span), 1), processed_hz=round(len(np.unique(used)) / 25.0, 1),
                      spread_ms_p50=round(float(np.percentile(spreads, 50)), 1), spread_ms_p95=round(float(np.percentile(spreads, 95)), 1))))
