# build_dataset.py — section 5: parse BGL, mine + persist Drain3 templates, encode every
# line match-only, build windows/labels and the chronological train/val/test split.
#
#   python v2/build_dataset.py [--variant default|no_rle|no_buffer|no_overlap|no_downsample] [--force]

import argparse
import time

import numpy as np

import common as C

LINE_KEYS = ("tid", "alert", "ts", "level", "key", "offset")


def iter_lines(bgl):
    """Yield (byte_offset, label, label-free line) for every well-formed line."""
    off = 0
    with open(bgl, "rb") as f:
        for raw in f:
            here, off = off, off + len(raw)
            label, rest = C.split_label(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            if C.parse_stream_line(rest) is not None:
                yield here, label, rest


def parse(cfg):
    from drain3 import TemplateMiner
    from drain3.file_persistence import FilePersistence
    d = C.path(cfg, "data_dir")
    d.mkdir(parents=True, exist_ok=True)
    bgl = C.path(cfg, "bgl")

    t0 = time.time()
    n = sum(1 for _ in iter_lines(bgl))
    train_end = int(n * cfg["split"]["train"])
    mine_end = train_end if cfg["drain"]["mine_on"] == "train" else n
    print(f"[parse] {n:,} well-formed lines ({time.time() - t0:.0f}s); mining templates on the first {mine_end:,}")

    miner = TemplateMiner(config=C.drain_config())
    for i, (_, _, rest) in enumerate(iter_lines(bgl)):
        if i >= mine_end:
            break
        miner.add_log_message(C.parse_stream_line(rest)[3])
        if (i + 1) % 1_000_000 == 0:
            print(f"[mine] {i + 1:,} lines, {len(miner.drain.clusters)} templates ({time.time() - t0:.0f}s)")
    miner.persistence_handler = FilePersistence(str(d / "drain3_state.bin"))
    miner.save_state("v2 build")
    print(f"[mine] {len(miner.drain.clusters)} templates saved to drain3_state.bin")

    # second pass: encode every line match-only, through the same code path the daemon uses
    enc = C.LineEncoder(cfg)
    tid = np.zeros(n, np.int32); alert = np.zeros(n, np.int8); ts = np.zeros(n, np.int64)
    level = np.zeros(n, np.int8); key = np.zeros(n, np.int32); offset = np.zeros(n, np.int64)
    key_of, key_text, last_ts = {}, [], 0
    for i, (off, label, rest) in enumerate(iter_lines(bgl)):
        t, tid[i], level[i], text = enc.encode(rest)
        last_ts = t or last_ts
        ts[i], alert[i], offset[i] = last_ts, label != "-", off
        k = key_of.get(text)
        if k is None:
            k = key_of[text] = len(key_text)
            key_text.append(text)
        key[i] = k
        if (i + 1) % 1_000_000 == 0:
            print(f"[encode] {i + 1:,} lines ({time.time() - t0:.0f}s)")
    for name, arr in zip(LINE_KEYS, (tid, alert, ts, level, key, offset)):
        np.save(d / f"line_{name}.npy", arr)
    C.write_json(d / "key_text.json", key_text)
    C.write_json(d / "parse_meta.json", {
        "lines": n, "templates": len(miner.drain.clusters), "mine_on": cfg["drain"]["mine_on"],
        "alert_lines": int(alert.sum()), "distinct_line_texts": len(key_text),
        "parse_seconds": round(time.time() - t0, 1)})


def windows_for(alert_cum, a, b, W, K, stride):
    """Labelled windows in lines [a, b): no alert inside, label = alert in the next K lines."""
    s = np.arange(a, b - W - K + 1, stride)
    clean = (alert_cum[s + W] - alert_cum[s]) == 0
    s = s[clean]
    y = ((alert_cum[s + W + K] - alert_cum[s + W]) > 0).astype(np.int8)
    return s, y


def build(cfg):
    d = C.path(cfg, "data_dir")
    lines = C.load_lines(cfg)
    alert, n = lines["alert"], len(lines["alert"])
    W, K = cfg["window"]["W"], cfg["window"]["K"]
    cum = np.concatenate([[0], np.cumsum(alert, dtype=np.int64)])
    train_end = int(n * cfg["split"]["train"])
    val_end = int(n * (cfg["split"]["train"] + cfg["split"]["val"]))
    bounds = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}
    rng = np.random.default_rng(cfg["seed"])

    # --- train: overlap, ambiguity buffer, negative downsampling (all training-only)
    s, y = windows_for(cum, 0, train_end, W, K, cfg["window"]["train_stride"])
    raw_n, raw_pos = len(y), int(y.sum())
    buf = cfg["window"]["buffer"]
    far = np.minimum(s + W + K + buf, n)
    ambiguous = (y == 0) & ((cum[far] - cum[s + W + K]) > 0)
    s, y = s[~ambiguous], y[~ambiguous]
    ratio = cfg["window"]["neg_ratio"]
    neg = np.flatnonzero(y == 0)
    dropped_neg = 0
    if ratio is not None and len(neg) > ratio * y.sum():
        keep_neg = rng.choice(neg, int(ratio * y.sum()), replace=False)
        dropped_neg = len(neg) - len(keep_neg)
        keep = np.sort(np.concatenate([np.flatnonzero(y == 1), keep_neg]))
        s, y = s[keep], y[keep]
    splits = {"train": (s, y)}

    # --- val / test: true distribution, non-overlapping, no buffer, no downsampling
    for name in ("val", "test"):
        a, b = bounds[name]
        splits[name] = windows_for(cum, a, b, W, K, W)
        np.save(d / f"{name}_stream_starts.npy", np.arange(a, b - W + 1, W))

    # --- no line may be shared between splits (train windows overlap each other, so check lines)
    owner = np.zeros(n, np.int8)
    for code, name in enumerate(("train", "val", "test"), start=1):
        st = splits[name][0]
        used = np.zeros(n, bool)
        used[(st[:, None] + np.arange(W)).ravel()] = True
        assert not (owner[used] != 0).any(), f"line overlap between {name} and an earlier split"
        owner[used] = code
        a, b = bounds[name]
        assert st.min() >= a and st.max() + W + K <= b, f"{name} windows leave their line range"
    print("[split] no line overlap between train / val / test windows: OK")

    (d / cfg["variant"]).mkdir(exist_ok=True)
    for name, (st, yy) in splits.items():
        out = d / cfg["variant"] if name == "train" else d
        np.save(out / f"{name}_starts.npy", st)
        np.save(out / f"{name}_y.npy", yy)

    meta = {"variant": cfg["variant"], "W": W, "K": K, "bounds": bounds,
            "train_stride": cfg["window"]["train_stride"], "buffer": buf, "neg_ratio": ratio,
            "train_before_filters": {"windows": raw_n, "positives": raw_pos},
            "train_ambiguous_dropped": int(ambiguous.sum()), "train_negatives_downsampled": dropped_neg,
            "splits": {k: {"windows": int(len(v[1])), "positives": int(v[1].sum()),
                           "positive_rate": float(v[1].mean())} for k, v in splits.items()}}
    print(f"[train] {raw_n:,} windows ({raw_pos:,} pos) before filters; "
          f"buffer dropped {int(ambiguous.sum()):,} negatives, downsampling dropped {dropped_neg:,}")
    for k, v in meta["splits"].items():
        print(f"[split] {k:5s} {v['windows']:>8,} windows  {v['positives']:>6,} positive  ({100 * v['positive_rate']:.2f}%)")
    for name in ("train", "val", "test"):
        a, b = bounds[name]
        print(f"[lines] {name:5s} unmatched by a saved template: {100 * (lines['tid'][a:b] == 0).mean():.2f}%")
        meta["splits"][name]["unmatched_line_rate"] = float((lines["tid"][a:b] == 0).mean())

    # --- 5.1 token lengths of the rendered windows, per tokenizer
    from transformers import AutoTokenizer
    rle = cfg["render"]["rle"]
    texts = {k: C.render_windows(v[0], lines, W, rle) for k, v in splits.items()}
    print("[render] example:", texts["train"][len(texts["train"]) // 2][:300])
    meta["tokens"], need = {}, 0
    for name, m in cfg["models"].items():
        if "hf" not in m:
            continue
        tok = AutoTokenizer.from_pretrained(m["hf"])
        L = np.concatenate([[len(x) for x in tok(texts[k], truncation=False)["input_ids"]] for k in texts])
        p50, p90, p95, p99 = (float(x) for x in np.percentile(L, [50, 90, 95, 99]))
        need = max(need, p95)
        meta["tokens"][name] = {"p50": p50, "p90": p90, "p95": p95, "p99": p99, "max": int(L.max()), "_L": L}
    max_len = cfg["render"]["max_length"] or next(
        (c for c in (64, 128, 256, 512) if c >= need), cfg["render"]["max_length_cap"])
    meta["max_length"] = int(max_len)
    for name, t in meta["tokens"].items():
        t["pct_truncated"] = float(100 * (t.pop("_L") > max_len).mean())
        print(f"[tokens] {name:10s} p50={t['p50']:.0f} p90={t['p90']:.0f} p95={t['p95']:.0f} "
              f"p99={t['p99']:.0f} max={t['max']}  ->  max_length={max_len}, {t['pct_truncated']:.2f}% truncated")
    C.write_json(d / cfg["variant"] / "meta.json", meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="default", choices=list(C.VARIANTS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config(args.variant)
    d = C.path(cfg, "data_dir")
    if args.force or not all((d / f"line_{k}.npy").exists() for k in LINE_KEYS):
        parse(cfg)
    else:
        print("[parse] line arrays exist, skipping (use --force to rebuild)")
    if args.force or not (d / args.variant / "meta.json").exists():
        build(cfg)
    else:
        print(f"[build] {args.variant} dataset exists, skipping (use --force to rebuild)")
        print(open(d / args.variant / "meta.json").read())
