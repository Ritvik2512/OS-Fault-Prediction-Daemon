import numpy as np

W = 20   # window size (lines per window)
K = 100   # look-ahead: alert within next K lines => pre-fault

tids = np.load("data/template_ids.npy")
alert = np.load("data/is_alert.npy")
n = len(tids)

windows, labels = [], []
for start in range(0, n - W - K, W):          # non-overlapping
    win = alert[start : start + W]
    if win.sum() > 0:                          # window contains a fault -> discard
        continue
    lookahead = alert[start + W : start + W + K]
    label = 1 if lookahead.sum() > 0 else 0
    text = " ".join(str(t) for t in tids[start : start + W])
    windows.append(text)
    labels.append(label)

labels = np.array(labels, dtype=np.int8)
np.save("data/labels.npy", labels)
with open("data/windows.txt", "w") as f:
    f.write("\n".join(windows))

print("Windows total :", len(labels))
print("Pre-fault (1) :", int(labels.sum()), f"({100*labels.mean():.2f}%)")
print("Normal (0)    :", int((labels==0).sum()))