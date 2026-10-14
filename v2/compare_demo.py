# compare_demo.py — run the real daemon (with LLM cards) once per model and write a detailed,
# side-by-side comparison of all three models: offline test metrics + what each one did live.
#
#   python v2/compare_demo.py [--force]
#
# Runs use --eval so ground truth can be scored afterwards; their events go to results/demo_runs/,
# never to the dashboard's live_events.jsonl.

import argparse
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
from detector import available_models, model_spec
from evaluate import fmt, md_table

COLORS = {"distilbert": "#4C9BE8", "roberta": "#7C5CFF", "electra": "#3DD68C", "hgb": "#F2A541"}


def run_daemon(cfg, name, mode, force):
    out = C.path(cfg, "results_dir", "demo_runs", f"{name}_{mode}.jsonl")
    out.parent.mkdir(exist_ok=True)
    if out.exists() and not force:
        print(f"[{name}/{mode}] run exists, skipping (use --force)")
    else:
        print(f"[{name}/{mode}] running the daemon...")
        log = out.with_suffix(".log")
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, str(C.ROOT / "v2" / "daemon.py"), "--eval", "--speed", "0",
                            "--model", name, "--slice-mode", mode, "--events", str(out.relative_to(C.ROOT))],
                           stdout=f, stderr=subprocess.STDOUT, check=True, cwd=C.ROOT)
    return [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]


def analyse(cfg, events, lines):
    W, K = cfg["window"]["W"], cfg["window"]["K"]
    start = next(e for e in events if e["type"] == "start")
    a, b = start["slice"]
    wins = [e for e in events if e["type"] == "window"]
    conf = np.array([w["confidence"] for w in wins])
    starts = a + W * np.arange(len(wins))
    fired = np.array([e["window_range"][1] for e in events if e["type"] == "incident_fired"], dtype=int)
    cards = [e for e in events if e["type"] == "incident"]
    cum = np.concatenate([[0], np.cumsum(lines["alert"], dtype=np.int64)])
    has_alert = (cum[starts + W] - cum[starts]) > 0
    ahead = (cum[np.minimum(starts + W + K, len(cum) - 1)] - cum[starts + W]) > 0
    prefault = ~has_alert & ahead
    eps = C.find_episodes(lines["alert"], a, b, cfg["eval"]["episode_gap"], W)
    inc = C.score_incidents(fired, starts, W, K, *eps, lines["ts"])
    hot = conf >= start["tau"]
    leads, kinds = [], {}                        # same episode rule as common.score_incidents
    for i in fired:
        t = int(starts[i] + W)
        k = np.searchsorted(eps[0], t)
        if k < len(eps[0]) and eps[0][k] < t + K:
            e = int(eps[0][k])
            leads.append((e - t, int(lines["ts"][e] - lines["ts"][t - 1])))
            kinds[int(i)] = "before fault"
        elif k > 0 and eps[1][k - 1] >= starts[i]:
            kinds[int(i)] = "late (fault already started)"
        else:
            kinds[int(i)] = "false alarm"
    return {
        "start": start, "conf": conf, "fired": fired, "kinds": kinds, "cards": cards, "has_alert": has_alert, "prefault": prefault,
        "row": {
            "slice": f"{a:,}-{b:,}", "windows": len(wins), "episodes_in_slice": len(eps[0]),
            "windows_over_tau": int(hot.sum()),
            "slice_window_precision": float((hot & prefault).sum() / max(hot[~has_alert].sum(), 1)),
            "slice_window_recall": float((hot & prefault).sum() / max(prefault.sum(), 1)),
            "incidents": inc["incidents"], "predictive": inc["predictive"], "late": inc["in_episode"],
            "false": inc["false"],
            "first_warning_lines": leads[0][0] if leads else None,
            "first_warning_s": leads[0][1] if leads else None,
            "mean_conf_normal": float(conf[~has_alert & ~prefault].mean()) if (~has_alert & ~prefault).any() else None,
            "mean_conf_prefault": float(conf[prefault].mean()) if prefault.any() else None,
            "detect_ms_median": float(np.median([w["detect_ms"] for w in wins])),
            "llm_s_median": float(np.median([c["llm_seconds"] for c in cards])) if cards else None,
            "cards_parsed": sum(c["card"]["summary"] != "LLM parse failed" for c in cards),
        }}


def plot_runs(runs, title, path):
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.3 * len(runs) + 0.6), sharex=False, squeeze=False)
    for ax, (name, r) in zip(axes[:, 0], runs):
        x = np.arange(len(r["conf"]))
        for i in np.flatnonzero(r["has_alert"]):
            ax.axvspan(i - 0.5, i + 0.5, color="#E5484D", alpha=0.18, lw=0)
        for i in np.flatnonzero(r["prefault"]):
            ax.axvspan(i - 0.5, i + 0.5, color="#F2A541", alpha=0.18, lw=0)
        ax.plot(x, r["conf"], color=COLORS.get(name, "C0"), lw=1.8)
        ax.axhline(r["start"]["tau"], color="#E5484D", ls="--", lw=1)
        if len(r["fired"]):
            ax.scatter(r["fired"], r["conf"][r["fired"]], marker="v", s=90, color="#E5484D", zorder=3)
        ax.set_ylim(0, 1.05); ax.set_ylabel("P(pre-fault)")
        ax.set_title(f"{name}  (tau={r['start']['tau']:.3f}, N={r['start']['N']}, cooldown={r['start']['cooldown']})",
                     loc="left", fontsize=10)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Window index in the replayed slice (20 lines each)")
    fig.suptitle(title + "\nred band = window contains alert lines, amber = pre-fault (alert within next 100 lines), "
                 "▼ = incident fired", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    res = C.path(cfg, "results_dir")
    lines = C.load_lines(cfg)
    names = available_models(cfg)
    sel = C.read_json(C.path(cfg, "models_dir", "selected.json"))
    specs = {n: model_spec(cfg, n) for n in names}

    shared, own = [], []
    for n in names:
        shared.append((n, analyse(cfg, run_daemon(cfg, n, "shared", args.force), lines)))
        if specs[n]["own_slice"] and specs[n]["own_slice"] != specs[n]["shared_slice"]:
            own.append((n, analyse(cfg, run_daemon(cfg, n, "own", args.force), lines)))
    plot_runs(shared, "Same demo slice for every model", res / "demo_traces_shared.png")
    if own:
        plot_runs(own, "Each model's own showcase slice", res / "demo_traces_own.png")

    # ---- offline test-set metrics
    df = pd.read_csv(res / "model_comparison.csv")
    df = df[df.model.isin(names)].set_index("model")
    yv, yt = C.load_split(cfg, "val")[1], C.load_split(cfg, "test")[1]
    prof = []
    for n in names:
        info = C.read_json(res / f"train_{n}.json")
        hist = info.get("history") or []
        if info["type"] == "hgb":
            import joblib
            m = joblib.load(C.path(cfg, "models_dir", n, "model.joblib"))
            size = f"{m.n_iter_} trees, {info['n_features']} features"
        else:
            from transformers import AutoConfig, AutoModelForSequenceClassification
            m = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(C.path(cfg, "models_dir", n)))
            size = f"{sum(p.numel() for p in m.parameters()) / 1e6:.0f}M parameters"
        hf = info.get("hf", "HistGradientBoostingClassifier").replace("\\", "/")
        if "pretrained/electra" in hf:
            hf = "google/" + hf.rsplit("/", 1)[-1]
        prof.append({"model": n, "family": hf, "input": (
            "rendered template text (RLE), max_length %s" % info["max_length"] if info["type"] != "hgb"
            else "template-count vector + extras"), "size": size,
            "training": (f"best epoch {max(hist, key=lambda h: h['val_pr_auc'])['epoch']} of {len(hist)} (early stop)"
                         if hist else f"grid of {len(info['grid'])}, best max_iter={info['max_iter']}, lr={info['learning_rate']}"),
            "train_minutes": info["train_seconds"] / 60, "peak_vram_mb": info["peak_vram_mb"] or "CPU only",
            "infer_ms": df.loc[n, "infer_ms_per_window"]})
    prof = pd.DataFrame(prof)

    sweep = []
    for n in names:
        pv, pt = np.load(res / f"preds_{n}_val.npy"), np.load(res / f"preds_{n}_test.npy")
        for target in (0.5, 0.8, 0.9):
            tau, met = C.tune_threshold_precision(yv, pv, target)
            t = C.window_metrics(yt, pt, tau)
            sweep.append({"model": n, "val_target": target, "reached_on_val": met, "tau": tau,
                          "test_precision": t["precision"], "test_recall": t["recall"], "test_tp": t["tp"], "test_fp": t["fp"]})
    sweep = pd.DataFrame(sweep)

    def best(col, low=False):
        s = df[col].astype(float)
        return s.idxmin() if low else s.idxmax()

    winners = [("Val PR-AUC (selection metric)", "val_pr_auc", False), ("Test PR-AUC", "test_pr_auc", False),
               ("Test ROC-AUC", "test_roc_auc", False), ("Test window precision", "test_precision", False),
               ("Test window recall", "test_recall", False), ("Test window F1", "test_f1", False),
               ("Test incident precision", "test_incident_precision", False),
               ("Test episode recall", "test_episode_recall", False), ("Inference speed (ms/window)", "infer_ms_per_window", True),
               ("Training time (min)", "train_minutes", True)]
    sh = pd.DataFrame([{"model": n, **r["row"]} for n, r in shared])
    own_df = pd.DataFrame([{"model": n, **r["row"]} for n, r in own])
    demo_cols = [("model", "Model", 0), ("slice", "Slice (test lines)", 0), ("windows", "Windows", 0),
                 ("windows_over_tau", "Windows >= tau", 0), ("incidents", "Incidents", 0),
                 ("predictive", "Before fault", 0), ("late", "Late", 0), ("false", "False alarm", 0),
                 ("first_warning_lines", "Warning lead (lines)", 0), ("first_warning_s", "Warning lead (s)", 0),
                 ("mean_conf_normal", "Mean conf, normal", 3), ("mean_conf_prefault", "Mean conf, pre-fault", 3),
                 ("detect_ms_median", "Detect ms (median)", 1), ("llm_s_median", "LLM s (median)", 1),
                 ("cards_parsed", "Cards parsed", 0)]

    md = [f"# Detailed model comparison", "",
          "Generated by `v2/compare_demo.py`. Part 1 is offline evaluation over the whole held-out test split "
          "(thresholds and gates tuned on val only). Part 2 replays slices of the test split through the real "
          "`daemon.py` once per model, with LLM incident cards; ground truth is used only afterwards, to score the runs. "
          f"The selected model is **`{sel['name']}`** ({sel['reason'].rstrip('.')}).", "",
          "## 1. The models", "",
          md_table(prof, [("model", "Model", 0), ("family", "Family", 0), ("input", "Input", 0), ("size", "Size", 0),
                          ("training", "Training", 0), ("train_minutes", "Train (min)", 1),
                          ("peak_vram_mb", "Peak VRAM (MB)", 0), ("infer_ms", "Inference (ms/window)", 1)]), "",
          "## 2. Window-level metrics on the test split", "",
          f"At each model's val-chosen threshold ({C.threshold_rule_text(cfg)}).", "",
          md_table(df.reset_index(), [("model", "Model", 0), ("val_pr_auc", "val PR-AUC", 3), ("tau", "tau*", 4),
                                      ("test_precision", "Precision", 3), ("test_recall", "Recall", 3),
                                      ("test_f1", "F1", 3), ("test_pr_auc", "PR-AUC", 3), ("test_roc_auc", "ROC-AUC", 3),
                                      ("test_tp", "TP", 0), ("test_fp", "FP", 0), ("test_fn", "FN", 0)]), "",
          "### Precision / recall trade-off", "",
          "Threshold chosen on val as the lowest one reaching each val precision target, then measured once on test. "
          "Shows how much recall each model keeps when pushed to high precision.", "",
          md_table(sweep, [("model", "Model", 0), ("val_target", "Val precision target", 1),
                           ("reached_on_val", "Reached on val", 0), ("tau", "tau", 4),
                           ("test_precision", "Test precision", 3), ("test_recall", "Test recall", 3),
                           ("test_tp", "TP", 0), ("test_fp", "FP", 0)]), "",
          "## 3. Incident-level metrics on the test split", "",
          "The daemon gate replayed over the whole test stream. *Before fault* = an episode starts within the next "
          f"{cfg['window']['K']} lines (the only kind credited); *late* = fired while a fault was already under way.", "",
          md_table(df.reset_index(), [("model", "Model", 0), ("N", "N", 0), ("cooldown", "Cooldown", 0),
                                      ("test_incidents", "Incidents", 0), ("test_predictive", "Before fault", 0),
                                      ("test_in_episode", "Late", 0), ("test_false", "False alarm", 0),
                                      ("test_incident_precision", "Incident precision", 3),
                                      ("test_episodes_caught", "Faults warned", 0), ("test_episodes", "Faults", 0),
                                      ("test_episode_recall", "Episode recall", 3), ("lead_median_s", "Lead median (s)", 1)]), "",
          "## 4. Best model per metric", "",
          "| Metric | Best | Value |", "|---|---|---|",
          *[f"| {label} | `{best(col, low)}` | {fmt(float(df.loc[best(col, low), col]), 3)} |" for label, col, low in winners],
          "", "## 5. Live daemon runs - same demo slice for every model", "",
          "Every model replays the same slice (the selected model's demo slice) with its own tau / N / cooldown. "
          "Precision/recall in this section cover one short slice and are anecdotal; sections 2-3 are the real measure.", "",
          md_table(sh, demo_cols), "", "![traces](demo_traces_shared.png)", ""]
    if own:
        md += ["## 6. Live daemon runs - each model's own showcase slice", "",
               "A showcase slice is a test fault that this model's gate warned about offline (chosen with ground truth). "
               "Shown so every model can be demoed firing; not a measure of performance.", "",
               md_table(own_df, demo_cols), "", "![traces](demo_traces_own.png)", ""]
    md += ["## 7. Incident cards written during the runs", "",
           "LLM-generated summaries (`" + cfg["llm"]["model"] + "`), not verified diagnoses. Verdict from ground truth.", ""]
    for n, r in shared + own:
        for c in r["cards"]:
            w = c["window_range"][1]
            verdict = r["kinds"].get(w, "-")
            card = c["card"]
            md += [f"- **{n}**, {r['start']['slice_mode']} slice, window {w}, confidence {c['confidence']:.3f}, "
                   f"{verdict}, LLM {c['llm_seconds']:.1f} s - *{card['severity']}* / {card['component']}: "
                   f"{card['summary']} Action: {card['suggested_action']}"]
    (res / "model_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {res / 'model_report.md'}, demo_traces_shared.png" + (", demo_traces_own.png" if own else ""))


if __name__ == "__main__":
    main()
