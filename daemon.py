# daemon.py — replay the held-out slice, detect per window, and treat each fault
# EPISODE as one incident. An episode is a run of FAULT windows that tolerates short
# dips (MAX_GAP normal windows) so a one-window flicker doesn't split one fault in two.
# It closes only on a real quiet stretch. One card per episode, over the whole block.

import json
import time
from collections import Counter
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR    = "models/fault_detector"
REPLAY       = "data/replay_slice.jsonl"
MAX_LEN      = 64
DEVICE       = "cpu"

TAU          = 0.5
MIN_RUN      = 7            # min FAULT windows for an episode to count (noise filter)
MAX_GAP      = 3           # normal windows tolerated inside one episode before it closes
SLEEP        = 0.05

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# --- detector ---
try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
except Exception:
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()

@torch.no_grad()
def detect(window_text):
    enc = tok(window_text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    return model(**enc).logits.softmax(-1)[0, 1].item()

# --- collapse repeated log lines so the LLM sees signal, not 200 copies ---
def compress(block_lines, top=15):
    msgs = []
    for ln in block_lines:
        f = ln.split()
        msgs.append(" ".join(f[6:]) if len(f) > 6 else ln)   # drop timestamp/node prefix
    out = []
    for msg, cnt in Counter(msgs).most_common(top):
        out.append(f"[x{cnt}] {msg}" if cnt > 1 else msg)
    return out

# --- local LLM incident writer ---
PROMPT = """You are an OS incident assistant. The following raw Blue Gene/L log lines are \
one continuous fault episode that precedes a predicted failure (repeated lines are shown \
once with an [xN] count). Write a JSON incident report.

Return ONLY a JSON object with exactly these keys:
- "summary": one sentence describing what is happening across these lines
- "component": the subsystem named in the lines (e.g. KERNEL, APP, torus, EDRAM, discovery, mmcs)
- "severity": one of "low", "medium", "high"
- "suggested_action": one short, concrete next step

Example:
{{"summary":"Repeated EDRAM error-correction events indicate memory degradation on the node.","component":"EDRAM","severity":"medium","suggested_action":"Run a memory diagnostic on the affected node and flag it for service."}}

Log lines:
{lines}"""

def llm_incident(block_lines):
    prompt = PROMPT.format(lines="\n".join(compress(block_lines)))
    r = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt,
        "format": "json", "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0.2, "num_predict": 200},
    }, timeout=120)
    r.raise_for_status()
    txt = r.json()["response"]
    try:
        d = json.loads(txt)
    except Exception:
        d = {"summary": txt.strip()[:200], "component": "unknown",
             "severity": "unknown", "suggested_action": "review logs manually"}
    for k in ("summary", "component", "severity", "suggested_action"):
        d.setdefault(k, "unknown")
    return d

def warmup():
    try:
        requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": "ok",
            "stream": False, "keep_alive": "10m", "options": {"num_predict": 1}}, timeout=120)
        print("LLM warmed up")
    except Exception as ex:
        print("warm-up failed:", ex)

def close_episode(ep, incidents):
    while ep and not ep[-1]["hot"]:          # trim trailing tolerated-gap windows
        ep.pop()
    hot = [r for r in ep if r["hot"]]
    if len(hot) < MIN_RUN:
        return
    block = [ln for r in ep for ln in r["raw_lines"]]
    a, b = ep[0]["gidx"], ep[-1]["gidx"]
    n_pos = sum(r["true_label"] for r in ep)
    print(f" INCIDENT: fault episode windows {a}-{b}  ({len(hot)} fault windows, {len(block)} lines, {n_pos} labelled pre-fault)")
    t0 = time.time()
    card = llm_incident(block)
    dt = time.time() - t0
    card = {**card, "window_range": [a, b], "window_lines": block}
    incidents.append((a, b, len(hot), dt))
    print("  --- incident card ---")
    print(f"  summary   : {card['summary']}")
    print(f"  component : {card['component']}")
    print(f"  severity  : {card['severity']}")
    print(f"  action    : {card['suggested_action']}")
    print(f"  windows   : {a}-{b}   labelled pre-fault: {n_pos}   llm={dt:.1f}s")
    print("  ---------------------")

# --- stream ---
with open(REPLAY, encoding="utf-8") as f:
    stream = [json.loads(ln) for ln in f]
warmup()
print(f"streaming {len(stream)} windows  (tau={TAU}, min run={MIN_RUN}, max gap={MAX_GAP})\n")

ep = []            # current episode (fault windows + tolerated internal gaps)
gap = 0            # trailing consecutive normal windows
incidents = []
for rec in stream:
    conf = detect(rec["window_text"])
    rec["hot"] = conf >= TAU
    print(f"[{rec['gidx']}] conf={conf:.3f}" + ("  FAULT" if rec["hot"] else ""))
    if rec["hot"]:
        ep.append(rec)
        gap = 0
    elif ep:
        gap += 1
        ep.append(rec)                       # tolerate the dip for now
        if gap > MAX_GAP:                    # real quiet stretch -> episode over
            close_episode(ep, incidents)
            ep, gap = [], 0
    if SLEEP:
        time.sleep(SLEEP)
if ep:
    close_episode(ep, incidents)

print(f"\ndone. incidents fired: {len(incidents)}")
for a, b, n, dt in incidents:
    print(f"  episode {a}-{b}  ({n} fault windows)  llm {dt:.1f}s")