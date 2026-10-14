# build_replay.py — pre-build the replay slice once: for each window in the slice,
# attach {gidx, window_text, true_label, raw_lines}. Reconstructs each window's source
# line range by replaying window.py's loop, then pulls those raw lines from BGL.log.
# Self-verifies: reconstructed template-id string must equal the on-disk test_x line.

import json
import numpy as np

W, K = 20, 100
DATA_DIR = "data"
BGL   = r"C:\programs\OS project\data\BGL.log"   # <-- your BGL path
SLICE = (21180, 21315)                            # find_slice.py candidate #0
OUT   = "data/replay_slice.jsonl"

tids  = np.load(f"{DATA_DIR}/template_ids.npy")
alert = np.load(f"{DATA_DIR}/is_alert.npy")
n = len(tids)

starts = np.array([s for s in range(0, n - W - K, W) if alert[s:s + W].sum() == 0])
total = len(starts)

test_y = np.load(f"{DATA_DIR}/test_y.npy")
split = int(0.8 * total)
assert total - split == len(test_y), "test size mismatch — need split.py"
with open(f"{DATA_DIR}/test_x.txt", encoding="utf-8") as f:
    test_x = [ln.rstrip("\n") for ln in f]

s, e = SLICE
line_starts = starts[split + s: split + e]

# verify mapping against the exact strings the model scored
for j, ls in enumerate(line_starts):
    recon = " ".join(str(int(x)) for x in tids[int(ls):int(ls) + W])
    if recon != test_x[s + j]:
        raise SystemExit(f"MAPPING FAILED at test idx {s + j}")
print("mapping verified: reconstructed window text == test_x.txt")

# pull needed raw lines from BGL.log
need = set()
for ls in line_starts:
    need.update(range(int(ls), int(ls) + W))
max_line = max(need)
raw = {}
with open(BGL, encoding="utf-8", errors="replace") as f:
    for idx, line in enumerate(f):
        if idx in need:
            raw[idx] = line.rstrip("\n")
        if idx >= max_line:
            break

with open(OUT, "w", encoding="utf-8") as out:
    for j, ls in enumerate(line_starts):
        t = s + j
        out.write(json.dumps({
            "gidx": t,
            "window_text": test_x[t],
            "true_label": int(test_y[t]),
            "raw_lines": [raw[int(ls) + k] for k in range(W)],
        }) + "\n")

print(f"wrote {OUT}: {len(line_starts)} windows with raw lines")