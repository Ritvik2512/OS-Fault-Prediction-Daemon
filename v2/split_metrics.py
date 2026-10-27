# split_metrics.py — training / validation / test metrics for every model, computed directly from the
# models' predictions and the true labels, at each model's chosen threshold (tau from results/eval_<model>.json).
#
#   python v2/split_metrics.py [--force]
#
# Prints one table per model and writes results/split_metrics.csv. Training-set predictions are produced here
# (and cached to results/preds_<model>_train.npy); validation and test predictions come from training.

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import common as C


def predict_train(cfg, name, info, lines, starts):
    W = cfg["window"]["W"]
    model_dir = C.path(cfg, "models_dir", name)
    if info["type"] == "hgb":
        import joblib
        X = C.features_for_starts(starts, lines, W, info["n_tid"])
        return joblib.load(model_dir / "model.joblib").predict_proba(X)[:, 1].astype(np.float32)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from train_transformer import predict, tokenize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    ids = tokenize(tok, C.render_windows(starts, lines, W, info.get("rle", True)), info["max_length"])
    return predict(model, tok, ids, cfg["train"]["eval_batch"], device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute cached training-set predictions")
    args = ap.parse_args()
    cfg = C.load_config()
    res = C.path(cfg, "results_dir")
    names = list(pd.read_csv(res / "model_comparison.csv").dropna(subset=["type"]).model)
    labels = {sp: C.load_split(cfg, sp) for sp in ("train", "val", "test")}
    lines = None
    rows = []
    for name in names:
        info = C.read_json(res / f"train_{name}.json")
        tau = C.read_json(res / f"eval_{name}.json")["tau"]
        cache = res / f"preds_{name}_train.npy"
        if cache.exists() and not args.force:
            p_train = np.load(cache)
        else:
            print(f"[{name}] scoring {len(labels['train'][1]):,} training windows ...", flush=True)
            lines = lines if lines is not None else C.load_lines(cfg)
            p_train = predict_train(cfg, name, info, lines, labels["train"][0])
            np.save(cache, p_train)
        preds = {"train": p_train,
                 "val": np.load(res / f"preds_{name}_val.npy"),
                 "test": np.load(res / f"preds_{name}_test.npy")}
        print(f"\n=== {name}   threshold tau = {tau:.4f} (chosen on the validation set) ===")
        print(f"{'split':<11}{'windows':>9}{'pre-fault':>10}{'accuracy':>10}{'precision':>11}{'recall':>8}"
              f"{'F1':>8}{'PR-AUC':>8}{'ROC-AUC':>9}{'TP':>7}{'FP':>7}{'FN':>7}{'TN':>8}")
        for sp, label_sp in (("train", "training"), ("val", "validation"), ("test", "test")):
            y, p = labels[sp][1], preds[sp]
            assert len(y) == len(p), f"{name}/{sp}: {len(y)} labels vs {len(p)} predictions"
            m = C.window_metrics(y, p, tau)
            acc = accuracy_score(y, (p >= tau).astype(int))
            rows.append({"model": name, "split": label_sp, "tau": tau, "windows": len(y), "positives": int(y.sum()),
                         "accuracy": acc, **{k: m[k] for k in ("precision", "recall", "f1", "pr_auc", "roc_auc",
                                                               "tp", "fp", "fn", "tn")}})
            print(f"{label_sp:<11}{len(y):>9,}{int(y.sum()):>10,}{acc:>10.3f}{m['precision']:>11.3f}{m['recall']:>8.3f}"
                  f"{m['f1']:>8.3f}{m['pr_auc']:>8.3f}{m['roc_auc']:>9.3f}{m['tp']:>7,}{m['fp']:>7,}{m['fn']:>7,}"
                  f"{m['tn']:>8,}")
    out = res / "split_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

if __name__ == "__main__":
    main()
