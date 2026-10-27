# train_transformer.py — section 6: fine-tune a HF sequence classifier on rendered windows.
#
#   python v2/train_transformer.py --model distilbert-base-uncased --name distilbert
#   python v2/train_transformer.py --model roberta-base --name roberta [--focal] [--variant X] [--force]

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

import common as C
from common import print_epoch


def tokenize(tok, texts, max_len):
    return tok(texts, truncation=True, max_length=max_len)["input_ids"]


def batch_of(tok, ids, idx, device):
    enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def predict(model, tok, ids, bs, device):
    model.eval()
    order = np.argsort([len(x) for x in ids])          # length-sorted: little padding
    out = np.zeros(len(ids), np.float32)
    for i in range(0, len(ids), bs):
        idx = order[i:i + bs]
        x, m = batch_of(tok, ids, idx, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=x, attention_mask=m).logits
        out[idx] = logits.float().softmax(-1)[:, 1].cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="default", choices=list(C.VARIANTS))
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--epochs", type=int, help="override train.epochs (used by the ablation)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = C.load_config(args.variant)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    res, out_dir = C.path(cfg, "results_dir"), C.path(cfg, "models_dir", args.name)
    done = res / f"train_{args.name}.json"
    if done.exists() and not args.force:
        print(f"[{args.name}] already trained ({done}), skipping (use --force)")
        return
    C.set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = cfg["models"].get(args.name.split("_abl_")[0].replace("_focal", ""), {})
    bs, accum, lr = base.get("batch", 16), base.get("accum", 1), float(base.get("lr", 2e-5))
    tc, W = cfg["train"], cfg["window"]["W"]
    meta = C.read_json(C.path(cfg, "data_dir", args.variant, "meta.json"))
    max_len, rle = meta["max_length"], cfg["render"]["rle"]

    lines = C.load_lines(cfg)
    tok = AutoTokenizer.from_pretrained(args.model)
    tr_s, tr_y = C.load_split(cfg, "train")
    va_s, va_y = C.load_split(cfg, "val")
    tr_ids = tokenize(tok, C.render_windows(tr_s, lines, W, rle), max_len)
    stream_ids = {sp: tokenize(tok, C.render_windows(C.load_stream(cfg, sp), lines, W, rle), max_len)
                  for sp in ("val", "test")}
    va_pos = C.labelled_positions(cfg, "val")
    va_ids = [stream_ids["val"][i] for i in va_pos]
    te_y = C.load_split(cfg, "test")[1]
    te_ids = [stream_ids["test"][i] for i in C.labelled_positions(cfg, "test")]
    eval_sets = {"training": (tr_ids, tr_y), "validation": (va_ids, va_y), "test": (te_ids, te_y)}
    print(f"[{args.name}] {args.model} on {device}: train {len(tr_y):,} ({int(tr_y.sum()):,} pos), "
          f"val {len(va_y):,}, max_length {max_len}, batch {bs}x{accum}, lr {lr}, focal={args.focal}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = (len(tr_y) // (bs * accum) + 1) * tc["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(tc["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    weight = torch.tensor([1.0, tc["pos_weight"]], device=device)
    y_t = torch.tensor(tr_y, dtype=torch.long, device=device)
    rng = np.random.default_rng(cfg["seed"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best, bad, t0 = [], -1.0, 0, time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        perm = rng.permutation(len(tr_y))
        run, nb = 0.0, 0
        for b, i in enumerate(range(0, len(perm), bs)):
            idx = perm[i:i + bs]
            x, m = batch_of(tok, tr_ids, idx, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(input_ids=x, attention_mask=m).logits
            ce = F.cross_entropy(logits.float(), y_t[idx], weight=weight, reduction="none")
            if args.focal:
                ce = ce * (1 - torch.exp(-F.cross_entropy(logits.float(), y_t[idx], reduction="none"))) ** tc["focal_gamma"]
            loss = ce.mean() / accum
            scaler.scale(loss).backward()
            if (b + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run, nb = run + loss.item() * accum, nb + 1
            if nb % 500 == 0:
                print(f"[{args.name}] epoch {epoch} batch {nb}/{len(perm) // bs} loss {run / nb:.4f} ({time.time() - t0:.0f}s)")
        # end-of-epoch scores on all three splits; early stopping uses validation PR-AUC only
        preds = {sp: predict(model, tok, ids, tc["eval_batch"], device) for sp, (ids, _) in eval_sets.items()}
        tau, scores = C.split_scores(cfg, preds, {sp: y for sp, (_, y) in eval_sets.items()})
        ap_val = scores["validation"]["pr_auc"]
        history.append({"epoch": epoch, "train_loss": run / nb, "val_pr_auc": ap_val, "tau": tau, "metrics": scores})
        print_epoch(args.name, epoch, run / nb, scores, tau, C.threshold_rule_text(cfg))
        if ap_val > best:
            best, bad = ap_val, 0
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        else:
            bad += 1
            if bad >= tc["patience"]:
                print(f"[{args.name}] early stop (no val PR-AUC gain)")
                break
    train_seconds = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0

    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    for sp in ("val", "test"):
        p = predict(model, tok, stream_ids[sp], tc["eval_batch"], device)
        np.save(res / f"preds_{args.name}_{sp}_stream.npy", p)
        np.save(res / f"preds_{args.name}_{sp}.npy", p[C.labelled_positions(cfg, sp)])
    C.write_json(done, {"name": args.name, "type": "transformer", "hf": args.model, "variant": args.variant,
                        "focal": args.focal, "max_length": max_len, "rle": rle, "batch": bs, "accum": accum,
                        "lr": lr, "pos_weight": tc["pos_weight"], "history": history, "best_val_pr_auc": best,
                        "train_seconds": round(train_seconds, 1), "peak_vram_mb": round(peak, 1),
                        "train_windows": int(len(tr_y))})
    print(f"[{args.name}] done: best val PR-AUC {best:.4f}, {train_seconds / 60:.1f} min, peak VRAM {peak:.0f} MB")


if __name__ == "__main__":
    main()
