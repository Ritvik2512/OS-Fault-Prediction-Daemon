# train_classical.py — section 6: gradient-boosted trees on template-count features (CPU).
#
#   python v2/train_classical.py --name hgb [--variant X] [--force]

import argparse
import itertools
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hgb")
    ap.add_argument("--variant", default="default", choices=list(C.VARIANTS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = C.load_config(args.variant)
    res, out_dir = C.path(cfg, "results_dir"), C.path(cfg, "models_dir", args.name)
    done = res / f"train_{args.name}.json"
    if done.exists() and not args.force:
        print(f"[{args.name}] already trained ({done}), skipping (use --force)")
        return
    C.set_seed(cfg["seed"])
    W = cfg["window"]["W"]
    lines = C.load_lines(cfg)
    n_tid = C.read_json(C.path(cfg, "data_dir", "parse_meta.json"))["templates"] + 1

    tr_s, tr_y = C.load_split(cfg, "train")
    va_s, va_y = C.load_split(cfg, "val")
    Xtr = C.features_for_starts(tr_s, lines, W, n_tid)
    Xva = C.features_for_starts(va_s, lines, W, n_tid)
    print(f"[{args.name}] train {Xtr.shape} ({int(tr_y.sum()):,} pos), val {Xva.shape}")

    grid, best, t0 = [], None, time.time()
    hp = cfg["models"]["hgb"]
    for max_iter, lr in itertools.product(hp["max_iter"], hp["learning_rate"]):
        clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr, class_weight="balanced",
                                             early_stopping=False, random_state=cfg["seed"])
        clf.fit(Xtr, tr_y)
        ap_val = float(average_precision_score(va_y, clf.predict_proba(Xva)[:, 1]))
        grid.append({"max_iter": max_iter, "learning_rate": lr, "val_pr_auc": ap_val})
        print(f"[{args.name}] max_iter={max_iter} lr={lr}: val PR-AUC {ap_val:.4f} ({time.time() - t0:.0f}s)")
        if best is None or ap_val > best[0]:
            best = (ap_val, clf, max_iter, lr)
    train_seconds = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best[1], out_dir / "model.joblib")
    for sp in ("val", "test"):
        p = best[1].predict_proba(C.features_for_starts(C.load_stream(cfg, sp), lines, W, n_tid))[:, 1]
        np.save(res / f"preds_{args.name}_{sp}_stream.npy", p.astype(np.float32))
        np.save(res / f"preds_{args.name}_{sp}.npy", p[C.labelled_positions(cfg, sp)].astype(np.float32))
    C.write_json(done, {"name": args.name, "type": "hgb", "variant": args.variant, "n_tid": n_tid,
                        "n_features": int(Xtr.shape[1]), "grid": grid, "max_iter": best[2],
                        "learning_rate": best[3], "best_val_pr_auc": best[0], "max_length": None,
                        "train_seconds": round(train_seconds, 1), "peak_vram_mb": 0.0,
                        "train_windows": int(len(tr_y))})
    print(f"[{args.name}] done: best val PR-AUC {best[0]:.4f} (max_iter={best[2]}, lr={best[3]}), {train_seconds:.0f}s for the whole grid")


if __name__ == "__main__":
    main()
