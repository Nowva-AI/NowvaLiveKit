"""Print compact comparison tables from results JSON."""
from __future__ import annotations
import json, sys
path = sys.argv[1]
keys = sys.argv[2].split(",")
variants = sys.argv[3].split(",") if len(sys.argv) > 3 else ["raw", "gc", "bone_bone", "bone_gc_bone", "full_nogc", "full", "proto", "proto21"]
r = json.load(open(path))
for scen, v in r.items():
    print(f"\n== {scen}")
    print("%-36s" % "metric" + "".join("%13s" % x for x in variants))
    for key in keys:
        row = "%-36s" % key[:36]
        for x in variants:
            val = v.get(x, {}).get(key)
            row += "%13s" % ("-" if val is None else "%.2f" % val)
        print(row)
