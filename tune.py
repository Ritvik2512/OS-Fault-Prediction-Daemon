# tune.py — score the slice once, cache confidences, report run structure so we can
# pick tau / debounce N / cooldown. Confidence is saturated, so N (run length) is the lever.

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/fault_detector"
DATA_DIR  = "data"
SLICE     = (21180, 21315)
MAX_LEN   = 64
DEVICE    = "cpu"
TAU       = 0.5
CACHE     = f"{DATA_DIR}/replay_conf.npy"

s, e = SLICE
with open(f"{DATA_DIR}/test_x.txt", encoding="utf-8") as f:
    windows = [ln.rstrip("\n") for ln in f][s:e]
labels = np.load(f"{DATA_DIR}/test_y.npy")[s:e]

# score once, cache (delete replay_conf.npy to force a re-score)
try:
    conf = np.load(CACHE)
    assert len(conf) == len(windows)
    print("loaded cached confidences")
except Exception:
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception:
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    conf = np.zeros(len(windows))
    with torch.no_grad():
        for i, w in enumerate(windows):
            enc = tok(w, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            conf[i] = model(**enc).logits.softmax(-1)[0, 1].item()
    np.save(CACHE, conf)
    print("scored and cached")

# is confidence saturated / bimodal?
print("\nconfidence bands:")
for lo, hi in [(0, .1), (.1, .5), (.5, .9), (.9, 1.01)]:
    print(f"  [{lo:.1f},{hi:.2f}): {int(((conf >= lo) & (conf < hi)).sum())}")

pos = [s + j for j, y in enumerate(labels) if y == 1]
print(f"\ntrue pre-fault windows: {pos[0]}-{pos[-1]} ({len(pos)} labelled)")

# maximal runs of consecutive windows over tau
hot = conf >= TAU
runs, i = [], 0
while i < len(hot):
    if hot[i]:
        j = i
        while j + 1 < len(hot) and hot[j + 1]:
            j += 1
        runs.append((s + i, s + j, j - i + 1))
        i = j + 1
    else:
        i += 1

print(f"\nover-tau (>= {TAU}) runs   [start-end  len  position]:")
for a, b, L in runs:
    if b >= pos[0] and a <= pos[-1]:
        tag = "overlaps run-up"
    elif b < pos[0]:
        tag = f"{pos[0] - b} windows before run-up"
    else:
        tag = "after run-up"
    print(f"  {a}-{b}  len={L:<3} {tag}")