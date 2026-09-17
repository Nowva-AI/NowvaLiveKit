import json, time
import harness as H
d0 = H.make_data(0, 0.0, noise="none")
for ps in ("counter", "oracle"):
    r = H.run_stack(d0, jaf=False, predictive=False, phase_source=ps)
    print(ps, "CLEAN no filters", json.dumps(H.fault_summary(r, d0)))
d = H.make_data(0, 4.0)
t=time.time()
res = H.run_stack(d, blend=True, vclamp=True, jaf=True, predictive=True, phase_source="oracle")
print("elapsed", round(time.time()-t,2))
m = H.angle_metrics(res["angles"], d); m.update(H.angle_metrics(res["eval"], d, "pred_")); m.update(H.position_metrics(res["skel"], d))
print(json.dumps(m))
print(json.dumps(H.fault_summary(res, d)))
