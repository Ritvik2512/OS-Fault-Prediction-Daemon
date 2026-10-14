import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from sklearn.metrics import precision_recall_fscore_support

# ---- REAL DATA ----
with open("data/train_x.txt") as f: train_x = f.read().splitlines()
with open("data/test_x.txt")  as f: test_x  = f.read().splitlines()
train_y = np.load("data/train_y.npy")
test_y  = np.load("data/test_y.npy")
# -------------------

MODEL = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL)

class WinDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tok(texts, truncation=True, padding="max_length", max_length=64)
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item

train_ds = WinDataset(train_x, train_y)
test_ds  = WinDataset(test_x,  test_y)

# class weighting: positives are ~2%, so weight class 1 heavily
pos = max(int(train_y.sum()), 1)
neg = len(train_y) - pos
weights = torch.tensor([1.0, min(neg / pos, 12.0)], dtype=torch.float)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            out.logits, labels, weight=weights.to(out.logits.device))
        return (loss, out) if return_outputs else loss

def metrics(pred):
    preds = pred.predictions.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(
        pred.label_ids, preds, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1}

args = TrainingArguments(
    output_dir="models/ckpt",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=test_ds,
    compute_metrics=metrics,
)

trainer.train()
print(trainer.evaluate())

model.save_pretrained("models/fault_detector")
tok.save_pretrained("models/fault_detector")
print("Saved to models/fault_detector")