import numpy as np

labels = np.load("data/labels.npy")
with open("data/windows.txt") as f:
    windows = f.read().splitlines()

n = len(labels)
cut = int(n * 0.8)                 # first 80% by time = train, last 20% = test

train_x, test_x = windows[:cut], windows[cut:]
train_y, test_y = labels[:cut], labels[cut:]

np.save("data/train_y.npy", train_y)
np.save("data/test_y.npy", test_y)
with open("data/train_x.txt", "w") as f:
    f.write("\n".join(train_x))
with open("data/test_x.txt", "w") as f:
    f.write("\n".join(test_x))

print("TRAIN:", len(train_y), "| pre-fault:", int(train_y.sum()),
      f"({100*train_y.mean():.2f}%)")
print("TEST :", len(test_y), "| pre-fault:", int(test_y.sum()),
      f"({100*test_y.mean():.2f}%)")