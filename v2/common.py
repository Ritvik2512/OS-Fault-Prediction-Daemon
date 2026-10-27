# common.py — shared loaders, line encoding, window->text rendering, features, metrics.
# Training, evaluation and the daemon all go through the functions here, so a window
# is encoded identically offline and at runtime.

import json
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
LEVELS = ["INFO", "WARNING", "ERROR", "SEVERE", "FATAL", "FAILURE", "OTHER"]
LEVEL_CODE = {l: i for i, l in enumerate(LEVELS)}

# ablation variants: each switches one section-5 lever off
VARIANTS = {
    "default": {},
    "no_rle": {"render": {"rle": False, "max_length": 64}},
    "no_buffer": {"window": {"buffer": 0}},
    "no_overlap": {"window": {"train_stride": None}},     # None -> stride = W
    "no_downsample": {"window": {"neg_ratio": None}},
}


def load_config(variant="default"):
    with open(ROOT / "v2" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section, kv in VARIANTS[variant].items():
        cfg[section].update(kv)
    if cfg["window"]["train_stride"] is None:
        cfg["window"]["train_stride"] = cfg["window"]["W"]
    cfg["variant"] = variant
    return cfg


def path(cfg, key, *parts):
    p = ROOT / cfg["paths"][key]
    return p.joinpath(*parts) if parts else p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------- line parsing

def split_label(raw):
    """Full BGL line -> (ground-truth label column, rest of the line)."""
    parts = raw.split(None, 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (raw, "")


def parse_stream_line(line):
    """Line WITHOUT the label column -> (epoch, component, level, content), or None if malformed."""
    p = line.split(None, 8)
    if len(p) < 9:
        return None
    return (int(p[0]) if p[0].isdigit() else 0), p[6], p[7], p[8].rstrip()


def _collapse(tokens, max_tokens):
    out = []
    for t in tokens:
        if t == "*" and out and out[-1] == "*":
            continue
        out.append(t)
    return " ".join(out[:max_tokens])


def clean_template(template, max_tokens):
    return _collapse([t.replace("<*>", "*") for t in template.split()], max_tokens)


def mask_content(content, max_tokens):
    """Fallback text for a line no saved template matches: numeric tokens -> *."""
    return _collapse(["*" if any(c.isdigit() for c in t) else t for t in content.split()], max_tokens)


def drain_config():
    from drain3.template_miner_config import TemplateMinerConfig
    return TemplateMinerConfig()          # Drain3 defaults, as in v1 (no drain3.ini)


class LineEncoder:
    """Match-only Drain3 against the saved template set. Never mines at runtime."""

    def __init__(self, cfg):
        from drain3 import TemplateMiner
        from drain3.file_persistence import FilePersistence
        state = path(cfg, "data_dir", "drain3_state.bin")
        self.miner = TemplateMiner(FilePersistence(str(state)), drain_config())
        self.max_tokens = cfg["render"]["max_template_tokens"]
        self.include_meta = cfg["render"]["include_meta"]
        self.templates = {c.cluster_id: clean_template(c.get_template(), self.max_tokens)
                          for c in self.miner.drain.clusters}
        self.n_tid = max(self.templates) + 1          # id 0 = unmatched
        self._cache = {}

    def encode(self, line):
        """Label-free line -> (epoch, template_id, level_code, line_text), or None."""
        parsed = parse_stream_line(line)
        if parsed is None:
            return None
        ts, comp, level, content = parsed
        key = (comp, level, content)
        hit = self._cache.get(key)
        if hit is None:
            cluster = self.miner.match(content, full_search_strategy="never")
            if cluster is None:
                tid, body = 0, mask_content(content, self.max_tokens)
            else:
                tid, body = cluster.cluster_id, self.templates[cluster.cluster_id]
            text = f"{comp} {level} {body}" if self.include_meta else body
            hit = (tid, LEVEL_CODE.get(level, LEVEL_CODE["OTHER"]), text)
            if len(self._cache) > 1_000_000:
                self._cache.clear()
            self._cache[key] = hit
        return (ts,) + hit


# ---------------------------------------------------------------- rendering

def render_window(line_texts, rle=True):
    if not rle:
        return " | ".join(line_texts)
    out, prev, n = [], None, 0
    for t in list(line_texts) + [None]:
        if t == prev:
            n += 1
            continue
        if prev is not None:
            out.append(prev if n == 1 else f"{prev} x{n}")
        prev, n = t, 1
    return " | ".join(out)


def render_windows(starts, lines, W, rle=True):
    key, text = lines["key"], lines["key_text"]
    return [render_window([text[k] for k in key[s:s + W]], rle) for s in starts]


# ---------------------------------------------------------------- classical features

EXTRA_FEATURES = ["n_distinct", "max_run"] + [f"n_{l}" for l in LEVELS] + ["span_seconds"]


def window_features(tid_mat, lvl_mat, span, n_tid):
    """[n, W] template ids / level codes + [n] span seconds -> uint16 feature matrix."""
    n, W = tid_mat.shape
    X = np.zeros((n, n_tid + len(EXTRA_FEATURES)), dtype=np.uint16)
    np.add.at(X, (np.arange(n)[:, None], np.minimum(tid_mat, n_tid - 1)), 1)
    X[:, n_tid] = 1 + (np.diff(np.sort(tid_mat, axis=1), axis=1) != 0).sum(1)
    run = np.ones(n, dtype=np.uint16)
    best = run.copy()
    for j in range(1, W):
        run = np.where(tid_mat[:, j] == tid_mat[:, j - 1], run + 1, 1).astype(np.uint16)
        best = np.maximum(best, run)
    X[:, n_tid + 1] = best
    for c in range(len(LEVELS)):
        X[:, n_tid + 2 + c] = (lvl_mat == c).sum(1)
    X[:, -1] = np.clip(span, 0, 65535)
    return X


def features_for_starts(starts, lines, W, n_tid):
    idx = np.asarray(starts)[:, None] + np.arange(W)
    ts = lines["ts"]
    span = ts[idx[:, -1]] - ts[idx[:, 0]]
    return window_features(lines["tid"][idx], lines["level"][idx], span, n_tid)


# ---------------------------------------------------------------- dataset loading

def load_lines(cfg):
    d = path(cfg, "data_dir")
    lines = {k: np.load(d / f"line_{k}.npy") for k in ("tid", "alert", "ts", "level", "key", "offset")}
    lines["key_text"] = read_json(d / "key_text.json")
    return lines


def load_split(cfg, split):
    """-> (starts, y) for labelled windows. Train depends on the variant; val/test never do."""
    d = path(cfg, "data_dir")
    if split == "train":
        d = d / cfg["variant"]
    return np.load(d / f"{split}_starts.npy"), np.load(d / f"{split}_y.npy")


def load_stream(cfg, split):
    """Every non-overlapping window of a split, including ones that contain alerts."""
    return np.load(path(cfg, "data_dir", f"{split}_stream_starts.npy"))


def labelled_positions(cfg, split):
    """Index of each labelled window inside the split's stream windows."""
    starts, _ = load_split(cfg, split)
    stream = load_stream(cfg, split)
    pos = (starts - stream[0]) // cfg["window"]["W"]
    assert np.array_equal(stream[pos], starts)
    return pos


# ---------------------------------------------------------------- metrics

def tune_threshold(y, p, min_recall):
    """tau* = max precision subject to recall >= min_recall; else max F0.5. -> (tau, constraint_met)."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= min_recall
    if ok.any():
        best = np.flatnonzero(ok)[np.lexsort((rec[ok], prec[ok]))[-1]]
        return float(thr[best]), True
    f05 = 1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-12)
    return float(thr[int(np.argmax(f05))]), False


def tune_threshold_precision(y, p, target):
    """tau* = lowest threshold whose precision >= target (the highest recall at that precision). -> (tau, met)."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.flatnonzero(prec[:-1] >= target)
    if len(ok):
        return float(thr[ok[0]]), True
    return float(thr[-1]), False


def choose_threshold(cfg, y_val, p_val):
    """The configured VAL-only threshold rule. -> (tau, rule_met)."""
    ev = cfg["eval"]
    if ev.get("precision_target") is not None:
        return tune_threshold_precision(y_val, p_val, ev["precision_target"])
    return tune_threshold(y_val, p_val, ev["min_recall"])


def threshold_rule_text(cfg):
    ev = cfg["eval"]
    if ev.get("precision_target") is not None:
        return (f"the lowest threshold whose val precision >= {ev['precision_target']} "
                "(i.e. the highest val recall at that precision)")
    return f"the threshold with the highest val precision subject to val recall >= {ev['min_recall']}"


def window_metrics(y, p, tau):
    from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                                 precision_recall_fscore_support, roc_auc_score)
    pred = (p >= tau).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1),
            "f0.5": float(fbeta_score(y, pred, beta=0.5, zero_division=0)),
            "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def find_episodes(alert, a, b, gap, W):
    """Fault episodes in lines [a, b): alert runs, merged when fewer than `gap` lines apart.
    Episodes starting in the first W lines are dropped: no full window can precede them."""
    idx = np.flatnonzero(alert[a:b]) + a
    if len(idx) == 0:
        return np.array([], int), np.array([], int)
    cut = np.flatnonzero(np.diff(idx) - 1 >= gap)
    starts = np.concatenate([[idx[0]], idx[cut + 1]])
    ends = np.concatenate([idx[cut], [idx[-1]]])
    keep = starts >= a + W
    return starts[keep], ends[keep]


def simulate_gate(probs, tau, N, cooldown):
    """The daemon gate: fire after N consecutive windows >= tau. One incident per run of flagged
    windows (the gate re-arms only after a window below tau), then `cooldown` quiet windows."""
    fired, over, cool, armed = [], 0, 0, True
    for i, hot in enumerate(probs >= tau):
        if cool > 0:
            cool -= 1
        if not hot:
            over, armed = 0, True
            continue
        over += 1
        if armed and over >= N and cool == 0:
            fired.append(i)
            cool, armed = cooldown, False
    return np.array(fired, dtype=int)


def score_incidents(fired, stream_starts, W, K, ep_starts, ep_ends, ts):
    """An incident fires once its window's last line is read (next unread line = t).
    predictive : an episode starts within the next K lines   (counts for incident precision)
    in_episode : not predictive, fired while an episode was already under way (late, not a prediction)
    false      : neither"""
    t = stream_starts[fired] + W
    nxt = np.searchsorted(ep_starts, t, side="left")
    has_next = nxt < len(ep_starts)
    predictive = np.zeros(len(t), bool)
    predictive[has_next] = ep_starts[nxt[has_next]] < t[has_next] + K
    prev = np.searchsorted(ep_starts, t, side="left") - 1
    in_ep = np.zeros(len(t), bool)
    has_prev = prev >= 0
    in_ep[has_prev] = ep_ends[prev[has_prev]] >= stream_starts[fired][has_prev]
    in_ep &= ~predictive

    caught, leads = 0, []
    for e in ep_starts:
        lo, hi = np.searchsorted(t, e - K, side="right"), np.searchsorted(t, e, side="right")
        if hi > lo:
            caught += 1
            leads.append(int(ts[e] - ts[t[lo] - 1]))
    n = len(t)
    return {"incidents": int(n), "predictive": int(predictive.sum()), "in_episode": int(in_ep.sum()),
            "false": int(n - predictive.sum() - in_ep.sum()),
            "incident_precision": float(predictive.sum() / n) if n else 0.0,
            "incident_precision_lenient": float((predictive | in_ep).sum() / n) if n else 0.0,
            "episodes": int(len(ep_starts)), "episodes_caught": int(caught),
            "episode_recall": float(caught / len(ep_starts)) if len(ep_starts) else 0.0,
            "lead_seconds": leads}


def lead_stats(leads):
    if not leads:
        return {"lead_median_s": None, "lead_q1_s": None, "lead_q3_s": None}
    q1, med, q3 = np.percentile(leads, [25, 50, 75])
    return {"lead_median_s": float(med), "lead_q1_s": float(q1), "lead_q3_s": float(q3)}


# ---------------------------------------------------------------- per-epoch / per-setting training printout

def epoch_metrics(y, p, threshold):
    """Window-level metrics (plus accuracy) at a given decision threshold."""
    from sklearn.metrics import accuracy_score
    m = window_metrics(y, p, threshold)
    m["accuracy"] = float(accuracy_score(y, (p >= threshold).astype(int)))
    return m


def split_scores(cfg, preds, labels):
    """Pick the threshold on VALIDATION with the deployment rule (eval.precision_target), then score every split
    at that threshold - the same rule used for the final reported numbers."""
    tau, _ = choose_threshold(cfg, labels["validation"], preds["validation"])
    return tau, {sp: epoch_metrics(labels[sp], preds[sp], tau) for sp in preds}


def print_epoch(name, epoch, loss, scores, tau, rule, header=None):
    print()
    print(header or f"[{name}] ===== epoch {epoch} finished - training loss {loss:.4f} =====")
    print(f"{'split':<11}{'windows':>9}{'accuracy':>10}{'precision':>11}{'recall':>8}{'F1':>8}{'PR-AUC':>8}"
          f"{'ROC-AUC':>9}{'TP':>7}{'FP':>7}{'FN':>7}{'TN':>8}")
    for sp, m in scores.items():
        n = m["tp"] + m["fp"] + m["fn"] + m["tn"]
        print(f"{sp:<11}{n:>9,}{m['accuracy']:>10.3f}{m['precision']:>11.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
              f"{m['pr_auc']:>8.3f}{m['roc_auc']:>9.3f}{m['tp']:>7,}{m['fp']:>7,}{m['fn']:>7,}{m['tn']:>8,}")
    print(f"(threshold tau = {tau:.4f}: {rule}, chosen on the validation predictions. Model selection uses "
          "validation PR-AUC only - test is shown for reporting, never for tuning)")
    print(flush=True)
