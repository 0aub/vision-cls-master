"""Collect the detection runs into one table."""
import glob, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import common as C                                      # noqa: E402
rows = []
for f in sorted(glob.glob(os.path.join(C.LOG_ROOT, "bench-detect-*", "detect_test.json"))):
    r = json.load(open(f))
    base = os.path.basename(os.path.dirname(f)).replace("bench-detect-", "")
    task, name = base.split("-", 1)
    m, t = r.get("meta", {}), r.get("test", {})
    v = r.get("val", {})
    rows.append({"task": task, "detector": name, "epochs": m.get("epochs"),
                 "params_M": round((m.get("params_total") or 0) / 1e6, 2),
                 "train_min": m.get("train_wallclock_min"),
                 "val_mAP": round(v.get("map", float("nan")), 4),
                 "test_mAP": round(t.get("map", float("nan")), 4),
                 "test_mAP50": round(t.get("map_50", float("nan")), 4),
                 "test_mAP75": round(t.get("map_75", float("nan")), 4),
                 "test_mAR100": round(t.get("mar_100", float("nan")), 4)})
if rows:
    d = pd.DataFrame(rows).sort_values(["task", "test_mAP"], ascending=[True, False])
    d.to_csv(os.path.join(C.LOG_ROOT, "bench-detect.csv"), index=False)
    print(d.to_string(index=False))
else:
    print("no detection runs yet")
