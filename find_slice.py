# find_slice.py — locate a good replay slice in the held-out test set.
# A good slice = a cluster of pre-fault (label 1) windows with quiet lead-in
# before it, so the daemon runs normal, then hits a real run-up it should flag.
# Only depends on data/test_y.npy (+ a sanity check on data/test_x.txt).

import numpy as np

y = np.load("data/test_y.npy")
N = len(y)
print(f"test windows: {N}, pre-fault (label 1): {int(y.sum())} ({y.mean()*100:.2f}%)\n")

# group label-1 windows into clusters: 1s within GAP of each other = one fault run-up
GAP = 5
pos = np.where(y == 1)[0]
clusters = []
if len(pos):
    start = prev = pos[0]
    for i in pos[1:]:
        if i - prev <= GAP:
            prev = i
        else:
            clusters.append((start, prev))
            start = prev = i
    clusters.append((start, prev))

LEAD = 80   # normal windows we'd like to include before the cluster (cap)
TAIL = 20   # a little tail after
cands = []
for cs, ce in clusters:
    size = int(y[cs:ce + 1].sum())        # positives in the cluster
    span = ce - cs + 1
    lead = 0                              # consecutive normal windows right before it
    j = cs - 1
    while j >= 0 and y[j] == 0:
        lead += 1
        j -= 1
    cands.append((cs, ce, size, span, lead))

# rank: prefer >=3 positives (debounce needs a few), then more lead-in, then bigger cluster
cands.sort(key=lambda c: (c[2] >= 3, min(c[4], LEAD), c[2]), reverse=True)

print("top candidate clusters (test-window indices):")
print(f"{'#':>2}  {'cluster':>15}  {'pos':>4}  {'span':>5}  {'lead':>5}  {'proposed slice':>16}")
for k, (cs, ce, size, span, lead) in enumerate(cands[:8]):
    s = max(0, cs - min(lead, LEAD))
    e = min(N, ce + 1 + TAIL)
    print(f"{k:>2}  {cs:>7}-{ce:<7}  {size:>4}  {span:>5}  {lead:>5}  [{s}:{e}] ({e - s}w)")

# sanity: is test_x.txt 1:1 with test_y.npy? decides if we can replay it directly
with open("data/test_x.txt", encoding="utf-8") as f:
    nlines = sum(1 for _ in f)
print(f"\ntest_x.txt lines: {nlines}  |  test_y.npy: {N}  ->  " + (
    "1:1 aligned — replay test_x.txt directly, encoding parity free"
    if nlines == N else
    "MISMATCH — window text is multi-line; need window.py render to re-slice"))

print("\npick a row above; its [start:end] is the replay slice we stream on Day 2.")