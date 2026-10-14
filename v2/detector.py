# detector.py — one interface for whichever model won:
#     detect(lines: list[str]) -> (is_fault: bool, confidence: float)
# `lines` are raw BGL lines WITHOUT the ground-truth label column. They are encoded with the
# same common.py functions used to build the training set (match-only Drain3, same rendering).

import numpy as np

import common as C


def read_lines(cfg, lines, a, b):
    """Raw lines [a, b) from BGL.log -> (label-free lines, is_alert flags)."""
    out, labels = [], []
    with open(C.path(cfg, "bgl"), "rb") as f:
        for off in lines["offset"][a:b]:
            f.seek(int(off))
            label, rest = C.split_label(f.readline().decode("utf-8", errors="replace").rstrip("\r\n"))
            out.append(rest)
            labels.append(label != "-")
    return out, labels


def available_models(cfg):
    """Models that finished training and evaluation (ablation runs excluded)."""
    res = C.path(cfg, "results_dir")
    return sorted(f.stem[5:] for f in res.glob("eval_*.json") if "_abl_" not in f.stem)


def model_spec(cfg, name=None):
    """Operating point for one model: the selected model's selected.json, or the same fields for any
    other evaluated model (its own val-tuned tau / N / cooldown and its own showcase slice)."""
    sel = C.read_json(C.path(cfg, "models_dir", "selected.json"))
    slices_path = C.path(cfg, "models_dir", "demo_slices.json")
    slices = C.read_json(slices_path) if slices_path.exists() else {}
    if name is None or name == sel["name"]:
        return {**sel, "own_slice": slices.get(sel["name"], sel["demo_slice"]), "shared_slice": sel["demo_slice"]}
    res = C.path(cfg, "results_dir")
    info, ev = C.read_json(res / f"train_{name}.json"), C.read_json(res / f"eval_{name}.json")
    return {"name": name, "path": f"models/{name}", "type": info["type"], "tau": float(ev["tau"]),
            "N": int(ev["N"]), "cooldown": int(ev["cooldown"]), "max_length": info.get("max_length"),
            "rle": info.get("rle", True), "n_tid": info.get("n_tid"),
            "own_slice": slices.get(name), "shared_slice": sel["demo_slice"]}


class Detector:
    def __init__(self, cfg, spec=None, device=None):
        """spec: models/selected.json (default) or a results/train_<name>.json record."""
        self.cfg = cfg
        self.spec = spec or C.read_json(C.path(cfg, "models_dir", "selected.json"))
        self.tau = float(self.spec.get("tau", 0.5))
        self.enc = C.LineEncoder(cfg)
        model_dir = C.path(cfg, "models_dir", self.spec["name"])
        if self.spec["type"] == "hgb":
            import joblib
            self.model = joblib.load(model_dir / "model.joblib")
            self.n_tid = self.spec["n_tid"]
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            want = device or cfg["daemon"]["device"]
            self.device = ("cuda" if torch.cuda.is_available() else "cpu") if want == "auto" else want
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
            self.torch = torch

    def confidence(self, lines):
        rows = [r for r in map(self.enc.encode, lines) if r is not None]
        if not rows:
            return 0.0
        if self.spec["type"] == "hgb":
            tid = np.array([[r[1] for r in rows]])
            lvl = np.array([[r[2] for r in rows]])
            ts = [r[0] for r in rows if r[0]] or [0]
            X = C.window_features(tid, lvl, np.array([ts[-1] - ts[0]]), self.n_tid)
            return float(self.model.predict_proba(X)[0, 1])
        text = C.render_window([r[3] for r in rows], self.spec.get("rle", True))
        enc = self.tok(text, truncation=True, max_length=self.spec["max_length"], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return float(self.model(**enc).logits.float().softmax(-1)[0, 1])

    def detect(self, lines):
        conf = self.confidence(lines)
        return conf >= self.tau, conf


_default = None


def detect(lines):
    """Module-level convenience: uses the model in models/selected.json."""
    global _default
    if _default is None:
        _default = Detector(C.load_config())
    return _default.detect(lines)
