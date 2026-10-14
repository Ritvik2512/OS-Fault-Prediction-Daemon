import numpy as np
from drain3 import TemplateMiner

INPUT = "data/BGL.log"

miner = TemplateMiner()
template_ids = []
is_alert = []

with open(INPUT, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split(None, 9)        # split on whitespace, keep message whole
        if len(parts) < 10:
            continue                       # skip malformed lines
        label, content = parts[0], parts[9]
        is_alert.append(0 if label == "-" else 1)
        template_ids.append(miner.add_log_message(content)["cluster_id"])
        if (i + 1) % 500000 == 0:
            print(f"{i+1} lines parsed...")

template_ids = np.array(template_ids, dtype=np.int32)
is_alert = np.array(is_alert, dtype=np.int8)

np.save("data/template_ids.npy", template_ids)
np.save("data/is_alert.npy", is_alert)

print("TOTAL lines :", len(template_ids))
print("Templates   :", len(np.unique(template_ids)))
print("Alert lines :", int(is_alert.sum()), f"({100 * is_alert.mean():.3f}%)")