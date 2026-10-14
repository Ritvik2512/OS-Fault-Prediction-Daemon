# daemon_core.py — Day 2 core: replay the held-out slice, run detect() per window,
# apply threshold + debounce + cooldown, print the full confidence trace.
# No LLM yet — first we confirm the detector fires on the real run-up and tune tau.

import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/fault_detector"
DATA_DIR  = "data"
SLICE     = (21180, 21315)   # find_slice.py candidate #0 (cluster 21260-21294)
MAX_LEN   = 64
DEVICE    = "cpu"            # detector on CPU per plan (frees 6GB for Ollama later)

TAU        = 0.5            # fire threshold on P(fault); raise toward 0.7-0.9 for demo
DEBOUNCE_N = 2             # consecutive windows over TAU before an incident fires
COOLDOWN   = 30           # windows to suppress new incidents after one fires
SLEEP      = 0.0          # seconds between windows; raise (e.g. 0.1) for a live feel

# --- model + detect() ---
try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
except Exception:
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")   # identical vocab
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()

@torch.no_grad()
def detect(window_text):
    enc = tok(window_text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    conf = model(**enc).logits.softmax(-1)[0, 1].item()   # P(pre-fault)
    return conf >= TAU, conf

# --- load replay slice (test_x.txt is 1:1 with test_y.npy) ---
s, e = SLICE
with open(f"{DATA_DIR}/test_x.txt", encoding="utf-8") as f:
    windows = [ln.rstrip("\n") for ln in f][s:e]
labels = np.load(f"{DATA_DIR}/test_y.npy")[s:e]

pos_idx = [s + j for j, yy in enumerate(labels) if yy == 1]
print(f"replaying test windows [{s}:{e}]  ({len(windows)} windows, {int(labels.sum())} pre-fault)")
if pos_idx:
    print(f"true pre-fault run-up: windows {pos_idx[0]}-{pos_idx[-1]}")
print(f"TAU={TAU}  DEBOUNCE_N={DEBOUNCE_N}  COOLDOWN={COOLDOWN}\n")

# --- daemon loop ---
over = 0            # consecutive windows over TAU
cool = 0           # cooldown counter
incidents = []
n_over = 0         # raw windows over TAU (pre-debounce)
for i, (w, y) in enumerate(zip(windows, labels)):
    gidx = s + i
    is_fault, conf = detect(w)
    if cool > 0:
        cool -= 1
    over = over + 1 if is_fault else 0
    n_over += is_fault
    fired = ""
    if over >= DEBOUNCE_N and cool == 0:
        incidents.append((gidx, conf, int(y)))
        cool, over, fired = COOLDOWN, 0, "  <-- INCIDENT FIRED"
    star = "*" if y == 1 else " "                 # true pre-fault marker
    hot  = "FAULT" if is_fault else "     "
    print(f"[{gidx}] {star} true={int(y)}  conf={conf:.3f}  {hot}{fired}")
    if SLEEP:
        time.sleep(SLEEP)

# --- summary ---
print(f"\nwindows over TAU: {n_over}/{len(windows)}   incidents fired: {len(incidents)}")
for gidx, conf, y in incidents:
    print(f"  incident @ window {gidx}  conf={conf:.3f}  true_label={y}")