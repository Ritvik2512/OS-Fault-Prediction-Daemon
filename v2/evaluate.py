# evaluate.py — section 7: threshold tuning (VAL), window + episode metrics (TEST),
# gate tuning (VAL), comparison table, plots, ablation, run report.
#
#   python v2/evaluate.py              # compare all trained models
#   python v2/evaluate.py --ablation   # retrain the selected model with each lever off
#   python v2/evaluate.py --report     # write results/RUN_REPORT.md

import argparse
import itertools
import json
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

import common as C
from detector import Detector, read_lines

V1 = {"model": "v1 distilbert (reported; template-ID input, 80/20 split, tau=0.5)",
      "test_precision": 0.193, "test_recall": 0.649, "test_f1": 0.297}
TARGET_WINDOW_P, TARGET_INCIDENT_P = 0.50, 0.80


def load_preds(res, name):
    return {f"{sp}{sfx}": np.load(res / f"preds_{name}_{sp}{sfx}.npy")
            for sp in ("val", "test") for sfx in ("", "_stream")}


def tune_gate(cfg, probs, tau, stream, eps, ts):
    """Pick (N, cooldown) on VAL: max incident precision s.t. episode recall >= floor; else max F0.5."""
    W, K, ev = cfg["window"]["W"], cfg["window"]["K"], cfg["eval"]
    rows = []
    for N, cd in itertools.product(ev["N_grid"], ev["cooldown_grid"]):
        s = C.score_incidents(C.simulate_gate(probs, tau, N, cd), stream, W, K, *eps, ts)
        rows.append((N, cd, s["incident_precision"], s["episode_recall"]))
    ok = [r for r in rows if r[3] >= ev["min_episode_recall"]]
    if ok:
        return max(ok, key=lambda r: (r[2], r[3], -r[0], -r[1]))[:2], True, rows
    f05 = lambda r: 1.25 * r[2] * r[3] / max(0.25 * r[2] + r[3], 1e-12)
    return max(rows, key=lambda r: (f05(r), -r[0], -r[1]))[:2], False, rows


def write_gate_sweep(cfg, lines=None):
    """Section 7.3 gate sweep on VAL for every evaluated model -> results/gate_sweep.csv.
    Same gate, scoring and choice rule as tune_gate; one row per (model, N, cooldown)."""
    res, W, K, ev = C.path(cfg, "results_dir"), cfg["window"]["W"], cfg["window"]["K"], cfg["eval"]
    lines = lines if lines is not None else C.load_lines(cfg)
    meta = C.read_json(C.path(cfg, "data_dir", "default", "meta.json"))
    stream = C.load_stream(cfg, "val")
    eps = C.find_episodes(lines["alert"], *meta["bounds"]["val"], ev["episode_gap"], W)
    rows = []
    for f in sorted(res.glob("eval_*.json")):
        if "_abl_" in f.stem:
            continue
        e = C.read_json(f)
        probs = np.load(res / f"preds_{e['model']}_val_stream.npy")
        (N, cd), floor_met, _ = tune_gate(cfg, probs, e["tau"], stream, eps, lines["ts"])
        chosen_by = (f"max incident precision s.t. episode recall >= {ev['min_episode_recall']}" if floor_met
                     else "fallback: max F0.5 (episode recall floor unreachable)")
        for n, c in itertools.product(ev["N_grid"], ev["cooldown_grid"]):
            sc = C.score_incidents(C.simulate_gate(probs, e["tau"], n, c), stream, W, K, *eps, lines["ts"])
            rows.append({"model": e["model"], "split": "val", "tau": e["tau"], "N": n, "cooldown_windows": c,
                         "incidents": sc["incidents"], "predictive": sc["predictive"],
                         "late_in_episode": sc["in_episode"], "false_alarms": sc["false"],
                         "incident_precision": sc["incident_precision"],
                         "incident_precision_lenient": sc["incident_precision_lenient"],
                         "episodes_caught": sc["episodes_caught"], "episodes": sc["episodes"],
                         "episode_recall": sc["episode_recall"],
                         "meets_recall_floor": sc["episode_recall"] >= ev["min_episode_recall"],
                         "chosen": (n, c) == (N, cd), "chosen_by": chosen_by})
    out = res / "gate_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


def infer_latency(cfg, info, lines, stream, expected, n=60):
    """Single-window latency through detector.py from raw lines; also checks runtime == offline encoding."""
    det = Detector(cfg, info)
    W = cfg["window"]["W"]
    idx = np.random.default_rng(cfg["seed"]).choice(len(stream), n, replace=False)
    wins = [read_lines(cfg, lines, int(stream[i]), int(stream[i]) + W)[0] for i in idx]
    det.confidence(wins[0])
    t0 = time.perf_counter()
    conf = np.array([det.confidence(w) for w in wins])
    ms = 1000 * (time.perf_counter() - t0) / n
    return ms, float(np.abs(conf - expected[idx]).max())


def evaluate_model(cfg, info, lines):
    res, name = C.path(cfg, "results_dir"), info["name"]
    W, K, ev = cfg["window"]["W"], cfg["window"]["K"], cfg["eval"]
    p = load_preds(res, name)
    y = {sp: C.load_split(cfg, sp)[1] for sp in ("val", "test")}
    stream = {sp: C.load_stream(cfg, sp) for sp in ("val", "test")}
    meta = C.read_json(C.path(cfg, "data_dir", "default", "meta.json"))
    eps = {sp: C.find_episodes(lines["alert"], *meta["bounds"][sp], ev["episode_gap"], W) for sp in ("val", "test")}

    tau, met = C.choose_threshold(cfg, y["val"], p["val"])
    val_m, test_m = C.window_metrics(y["val"], p["val"], tau), C.window_metrics(y["test"], p["test"], tau)
    (N, cd), gate_met, grid = tune_gate(cfg, p["val_stream"], tau, stream["val"], eps["val"], lines["ts"])
    inc = {sp: C.score_incidents(C.simulate_gate(p[f"{sp}_stream"], tau, N, cd), stream[sp], W, K, *eps[sp], lines["ts"])
           for sp in ("val", "test")}
    ms, parity = infer_latency(cfg, info, lines, stream["test"], p["test_stream"])
    assert parity < 0.02, f"{name}: runtime encoding disagrees with offline predictions (max diff {parity:.4f})"

    row = {"model": name, "type": info["type"], "val_pr_auc": val_m["pr_auc"], "tau": tau,
           "threshold_rule_met": met, "val_precision": val_m["precision"], "val_recall": val_m["recall"]}
    row.update({f"test_{k}": v for k, v in test_m.items()})
    row.update({"N": N, "cooldown": cd, "gate_recall_floor_met": gate_met,
                "val_incident_precision": inc["val"]["incident_precision"],
                "val_episode_recall": inc["val"]["episode_recall"]})
    for k in ("incidents", "predictive", "in_episode", "false", "incident_precision",
              "incident_precision_lenient", "episodes", "episodes_caught", "episode_recall"):
        row[f"test_{k}"] = inc["test"][k]
    row.update(C.lead_stats(inc["test"]["lead_seconds"]))
    row.update({"train_minutes": info["train_seconds"] / 60, "peak_vram_mb": info["peak_vram_mb"],
                "infer_ms_per_window": ms, "runtime_parity_max_diff": parity})
    C.write_json(res / f"eval_{name}.json", {**row, "val_gate_grid": grid, "test_lead_seconds": inc["test"]["lead_seconds"]})
    print(f"[{name}] tau*={tau:.4f} (threshold rule met: {met})  TEST window P={test_m['precision']:.3f} "
          f"R={test_m['recall']:.3f} F1={test_m['f1']:.3f} PR-AUC={test_m['pr_auc']:.3f} | gate N={N} cooldown={cd} "
          f"(floor met: {gate_met})  TEST incident P={inc['test']['incident_precision']:.3f} "
          f"episode R={inc['test']['episode_recall']:.3f} | {ms:.1f} ms/window, parity diff {parity:.4f}")
    return row, y, p, inc


def pick_winner(df, margin):
    """Selection rule: highest val PR-AUC; within `margin`: higher val incident precision, then faster inference."""
    top = df["val_pr_auc"].max()
    tied = df[df["val_pr_auc"] >= top - margin]
    best = tied.sort_values(["val_incident_precision", "infer_ms_per_window"], ascending=[False, True]).iloc[0]
    if len(tied) == 1:
        why = f"highest val PR-AUC ({best.val_pr_auc:.4f}); no other model within {margin}"
    else:
        others = ", ".join(f"{r.model} {r.val_pr_auc:.4f}" for r in tied.itertuples() if r.model != best.model)
        why = (f"val PR-AUC {best.val_pr_auc:.4f} is within {margin} of the top ({others}); tiebreak on val incident "
               f"precision ({best.val_incident_precision:.3f}), then inference speed ({best.infer_ms_per_window:.1f} ms)")
    return best["model"], why


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def md_table(df, cols):
    out = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.to_dict("records"):
        out.append("| " + " | ".join(fmt(r.get(c), nd) for c, _, nd in cols) + " |")
    return "\n".join(out)


WINDOW_COLS = [("model", "Model", 0), ("val_pr_auc", "val PR-AUC", 3), ("tau", "tau*", 4),
               ("test_precision", "Precision", 3), ("test_recall", "Recall", 3), ("test_f1", "F1", 3),
               ("test_f0.5", "F0.5", 3), ("test_pr_auc", "PR-AUC", 3), ("test_roc_auc", "ROC-AUC", 3),
               ("test_tp", "TP", 0), ("test_fp", "FP", 0), ("test_fn", "FN", 0), ("test_tn", "TN", 0)]
EPISODE_COLS = [("model", "Model", 0), ("N", "N", 0), ("cooldown", "Cooldown", 0),
                ("test_incidents", "Incidents fired", 0), ("test_predictive", "Predictive", 0),
                ("test_in_episode", "Late (in-episode)", 0), ("test_false", "False", 0),
                ("test_incident_precision", "Incident precision", 3),
                ("test_episodes_caught", "Episodes preceded", 0), ("test_episodes", "Episodes", 0),
                ("test_episode_recall", "Episode recall", 3), ("lead_median_s", "Lead median (s)", 1),
                ("lead_q1_s", "Lead Q1 (s)", 1), ("lead_q3_s", "Lead Q3 (s)", 1)]
COST_COLS = [("model", "Model", 0), ("train_minutes", "Train (min)", 1), ("peak_vram_mb", "Peak VRAM (MB)", 0),
             ("infer_ms_per_window", "Inference (ms/window)", 1)]


def compare(cfg):
    res = C.path(cfg, "results_dir")
    infos = [C.read_json(f) for f in sorted(res.glob("train_*.json")) if "_abl_" not in f.name]
    lines = C.load_lines(cfg)
    rows, curves, incs = [], {}, {}
    for info in infos:
        row, y, p, inc = evaluate_model(cfg, info, lines)
        rows.append(row)
        curves[info["name"]] = (y, p, row["tau"], row)
        incs[info["name"]] = inc
    df = pd.DataFrame(rows)
    winner, why = pick_winner(df, cfg["eval"]["tie_margin"])
    pd.concat([df, pd.DataFrame([V1])]).to_csv(res / "model_comparison.csv", index=False)
    w = df[df.model == winner].iloc[0]

    # --- plots
    plt.figure(figsize=(7, 5))
    for name, (y, p, tau, row) in curves.items():
        prec, rec, _ = precision_recall_curve(y["val"], p["val"])
        line, = plt.plot(rec, prec, label=f"{name} (val PR-AUC {row['val_pr_auc']:.3f})")
        plt.scatter([row["val_recall"]], [row["val_precision"]], color=line.get_color(), zorder=3)
    plt.axhline(float(curves[winner][0]["val"].mean()), ls=":", c="gray", label="positive rate (chance)")
    if cfg["eval"].get("precision_target") is not None:
        plt.axhline(cfg["eval"]["precision_target"], ls="--", c="gray", lw=0.8)
    else:
        plt.axvline(cfg["eval"]["min_recall"], ls="--", c="gray", lw=0.8)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Window-level precision-recall, VALIDATION set (dots = tau*)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(res / "pr_curves.png", dpi=130); plt.close()
    for r in rows:
        cm = np.array([[r["test_tn"], r["test_fp"]], [r["test_fn"], r["test_tp"]]])
        plt.figure(figsize=(4.2, 3.8))
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        plt.xticks([0, 1], ["pred normal", "pred pre-fault"]); plt.yticks([0, 1], ["normal", "pre-fault"])
        plt.title(f"{r['model']}: TEST windows at tau*={r['tau']:.3f}")
        plt.tight_layout(); plt.savefig(res / f"confusion_{r['model']}.png", dpi=130); plt.close()
    leads = incs[winner]["test"]["lead_seconds"]
    plt.figure(figsize=(6, 4))
    if leads:
        plt.hist(leads, bins=30)
    plt.xlabel("Lead time before the episode's first alert line (s)"); plt.ylabel("Episodes")
    plt.title(f"{winner}: lead time, TEST episodes preceded by an incident (n={len(leads)})")
    plt.tight_layout(); plt.savefig(res / "lead_time_hist.png", dpi=130); plt.close()

    # --- markdown
    meta = C.read_json(C.path(cfg, "data_dir", "default", "meta.json"))
    win_ok, inc_ok = w.test_precision >= TARGET_WINDOW_P and w.test_recall >= 0.40, w.test_incident_precision >= TARGET_INCIDENT_P
    md = [
        "# Model comparison (v2)", "",
        "All numbers below were produced by `v2/evaluate.py`. Thresholds and gate parameters were tuned on the "
        "**validation** set only; every metric in the tables is on the held-out **test** set unless marked `val`.", "",
        f"**Selection rule:** highest val PR-AUC; tiebreak (within {cfg['eval']['tie_margin']}): higher val incident "
        "precision, then faster inference.", "",
        f"**Selected model: `{winner}`** - {why}.", "",
        "## Window-level metrics (TEST, at each model's tau*)", "",
        f"A window is positive (*pre-fault*) when it contains no alert line but an alert occurs within the next "
        f"{cfg['window']['K']} lines. tau* = {C.threshold_rule_text(cfg)}. Test positive rate (chance precision): {meta['splits']['test']['positive_rate']:.3f}.", "",
        md_table(pd.concat([df, pd.DataFrame([V1])]), WINDOW_COLS), "",
        "The v1 row is the figure reported for v1 (its own 80/20 split, threshold 0.5); it was not re-measured here.", "",
        "## Episode-level (incident) metrics (TEST)", "",
        f"These are **not** window metrics. The daemon gate (tau*, N consecutive windows, cooldown in windows) is replayed "
        f"over the whole test stream, including windows that contain alerts. Fault episode = run of alert lines, merged "
        f"when fewer than {cfg['eval']['episode_gap']} lines apart. An incident counts as *predictive* only if an episode "
        f"**starts** within the next {cfg['window']['K']} lines; incidents fired once an episode is already under way are "
        "counted as *late* and are **not** credited. Incident precision = predictive / fired. Lead time is capped by "
        f"construction at the time spanned by {cfg['window']['K']} lines.", "",
        md_table(df, EPISODE_COLS), "",
        "N and cooldown were chosen on val to maximise incident precision subject to episode recall >= "
        f"{cfg['eval']['min_episode_recall']}" + ("." if df.gate_recall_floor_met.all() else
        "; **no (N, cooldown) setting reached that recall floor on val for: "
        + ", ".join(df[~df.gate_recall_floor_met].model) + "**, so those fall back to the setting with the best "
        "F0.5 of (incident precision, episode recall)."), "",
        "## Cost", "", md_table(df, COST_COLS), "",
        "Inference is single-window latency through `detector.py` from raw log lines (transformers on the GPU).", "",
        "## Precision targets", "",
        f"- Window-level test precision >= {TARGET_WINDOW_P} at recall >= 0.40: **{'MET' if win_ok else 'MISSED'}** "
        f"(`{winner}`: precision {w.test_precision:.3f}, recall {w.test_recall:.3f}; v1 reported 0.193 / 0.649)"
        + (f". The precision part is met; the recall part is not, because the operating point is precision-first "
           f"({C.threshold_rule_text(cfg)})." if (w.test_precision >= TARGET_WINDOW_P and not win_ok) else "."),
        f"- Incident-level test precision >= {TARGET_INCIDENT_P}: **{'MET' if inc_ok else 'MISSED'}** "
        f"(`{winner}`: {w.test_incident_precision:.3f}, episode recall {w.test_episode_recall:.3f}).", "",
        "What was tried: template-string rendering with run-length encoding instead of v1's bare template IDs "
        f"(max_length {meta['max_length']}), a {meta['buffer']}-line ambiguity buffer on train negatives, overlapping "
        f"train windows (stride {meta['train_stride']}), {meta['neg_ratio']}:1 negative downsampling, a chronological "
        "70/10/20 split with precision-first threshold tuning on val, three model families, and a val-tuned gate. "
        "See `ablation.md` for the measured effect of each data lever.", "",
        "![PR curves](pr_curves.png)", ""]
    (res / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    write_gate_sweep(cfg, lines)
    print(f"\nSELECTED: {winner} - {why}")
    print(f"wrote {res / 'model_comparison.csv'}, model_comparison.md, pr_curves.png, confusion_*.png, lead_time_hist.png")


def ablation(cfg, force):
    res = C.path(cfg, "results_dir")
    sel = C.read_json(C.path(cfg, "models_dir", "selected.json"))
    base, py, here = C.read_json(res / f"train_{sel['name']}.json"), sys.executable, C.ROOT / "v2"
    rows = []

    def row_for(label, name):
        y = {sp: C.load_split(cfg, sp)[1] for sp in ("val", "test")}
        pv, pt = np.load(res / f"preds_{name}_val.npy"), np.load(res / f"preds_{name}_test.npy")
        tau, met = C.choose_threshold(cfg, y["val"], pv)
        m = C.window_metrics(y["test"], pt, tau)
        info = C.read_json(res / f"train_{name}.json")
        return {"variant": label, "train_windows": info["train_windows"], "val_pr_auc": C.window_metrics(y["val"], pv, tau)["pr_auc"],
                "tau": tau, "test_precision": m["precision"], "test_recall": m["recall"], "test_f1": m["f1"],
                "test_pr_auc": m["pr_auc"], "train_minutes": info["train_seconds"] / 60}

    hist = base.get("history") or []
    best_ep = max(hist, key=lambda h: h["val_pr_auc"])["epoch"] if hist else None
    rows.append(row_for("all levers on (selected model" + (f", early-stopped, best epoch {best_ep} of {len(hist)} run)" if hist else ")"), sel["name"]))
    labels = {"default": "all levers on - control trained under the ablation budget",
              "no_rle": "no RLE / truncation fix (no run-length encoding, max_length 64)",
              "no_buffer": "no ambiguity buffer", "no_overlap": "no overlapping train windows (stride = W)",
              "no_downsample": "no negative downsampling"}
    budget = [] if sel["type"] == "hgb" else ["--epochs", str(cfg["train"]["ablation_epochs"])]
    skipped = []
    for variant, label in labels.items():
        if variant == "default" and not budget:
            continue                      # hgb: the selected model itself is the control
        if variant == "no_rle" and sel["type"] == "hgb":
            skipped.append(label)
            continue
        name = f"{sel['name']}_abl_{variant}"
        extra = ["--force"] if force else []
        subprocess.run([py, str(here / "build_dataset.py"), "--variant", variant] + extra, check=True)
        cmd = ([py, str(here / "train_classical.py")] if sel["type"] == "hgb" else
               [py, str(here / "train_transformer.py"), "--model", base["hf"]] + budget + (["--focal"] if base.get("focal") else []))
        subprocess.run(cmd + ["--name", name, "--variant", variant] + extra, check=True)
        rows.append(row_for(label, name))
    df = pd.DataFrame(rows)
    df.to_csv(res / "ablation.csv", index=False)
    cols = [("variant", "Variant", 0), ("train_windows", "Train windows", 0), ("val_pr_auc", "val PR-AUC", 3),
            ("tau", "tau*", 4), ("test_precision", "Test precision", 3), ("test_recall", "Test recall", 3),
            ("test_f1", "Test F1", 3), ("test_pr_auc", "Test PR-AUC", 3), ("train_minutes", "Train (min)", 1)]
    md = [f"# Ablation: `{sel['name']}` with each data lever switched off", "",
          "Each row retrains the selected model with exactly one section-5 lever disabled. tau* is re-tuned on val "
          f"(" + C.threshold_rule_text(cfg) + "); precision/recall are window-level on TEST. "
          "Val and test sets are identical in every row. Single run per row (seed 42), so small differences are within noise." +
          (f" To keep the runs affordable on the RTX 3050, every ablation row (and its all-levers-on control) is trained for "
           f"exactly {cfg['train']['ablation_epochs']} epoch(s); compare ablation rows with that control, not with the "
           "fully trained selected model." if sel["type"] != "hgb" else ""), "",
          md_table(df, cols), ""]
    ctrl = df.iloc[1] if budget else df.iloc[0]
    md += [f"Effect of switching each lever **off**, relative to the control row ({ctrl.variant}). A negative change "
           "means the lever was helping:", ""]
    for r in df.iloc[2 if budget else 1:].itertuples():
        md.append(f"- {r.variant}: val PR-AUC {r.val_pr_auc - ctrl.val_pr_auc:+.3f}, test precision "
                  f"{r.test_precision - ctrl.test_precision:+.3f}, test recall {r.test_recall - ctrl.test_recall:+.3f}, "
                  f"test PR-AUC {r.test_pr_auc - ctrl.test_pr_auc:+.3f}")
    md += ["", "Read val PR-AUC (the metric everything was tuned on) as the primary signal. Test precision at a val-tuned "
           "threshold is noisy here because the val and test distributions differ (see RUN_REPORT.md), so a lever is "
           "only clearly useful when val and test move in the same direction.", ""]
    if skipped:
        md += ["Not applicable to this model type (its input is template counts, not rendered text): " + "; ".join(skipped) + ".", ""]
    (res / "ablation.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def report(cfg):
    res = C.path(cfg, "results_dir")
    sel = C.read_json(C.path(cfg, "models_dir", "selected.json"))
    df = pd.read_csv(res / "model_comparison.csv")
    models, w = df[df.type.notna()], df[df.model == sel["name"]].iloc[0]
    meta = C.read_json(C.path(cfg, "data_dir", "default", "meta.json"))
    # LLM latency from a run of the SELECTED model: the latest live run if it used that model,
    # otherwise its recorded comparison run (so demoing another model doesn't change the report)
    lat, lat_src = [], None
    for src in (res / "live_events.jsonl", res / "demo_runs" / f"{sel['name']}_shared.jsonl"):
        if not src.exists():
            continue
        evs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        start = next((e for e in evs if e.get("type") == "start"), {})
        cards = [e["llm_seconds"] for e in evs if e.get("type") == "incident"]
        if start.get("model", sel["name"]) == sel["name"] and cards:
            lat, lat_src = cards, src.relative_to(C.ROOT).as_posix()
            break
    md = ["# RUN REPORT - OS fault-prediction daemon v2", "",
          "Generated by `v2/evaluate.py --report` from the files in `results/`. The system **predicts** pre-fault "
          "windows on a **simulated live stream from held-out BGL logs** (a supercomputer system log); it does not "
          "monitor a live server, and tailing real `journalctl` output is future work. LLM incident cards are generated "
          "summaries, not verified diagnoses.", "",
          "## Data", "",
          f"- Chronological split by line, 70/10/20. Windows: W={meta['W']}, K={meta['K']}.",
          *[f"- {k}: {v['windows']:,} windows, {v['positives']:,} positive ({100 * v['positive_rate']:.2f}%), "
            f"{100 * v['unmatched_line_rate']:.1f}% of lines unmatched by a train-mined template" for k, v in meta["splits"].items()],
          f"- Rendered-window tokens: " + "; ".join(f"{k} p95={t['p95']:.0f}, {t['pct_truncated']:.2f}% truncated at "
                                                    f"max_length {meta['max_length']}" for k, t in meta["tokens"].items()) + ".", "",
          "## Final metrics (TEST)", "", "Window-level:", "", md_table(df, WINDOW_COLS), "",
          "Episode-level (incident) - a different unit from the table above:", "", md_table(models, EPISODE_COLS), "",
          "## Selected model and operating point", "",
          f"- Model: `{sel['name']}` ({sel['type']}). {sel['reason']}.",
          f"- tau = {sel['tau']:.4f}, N = {sel['N']}, cooldown = {sel['cooldown']} windows (all tuned on val).",
          f"- Window-level test precision {w.test_precision:.3f} / recall {w.test_recall:.3f} (v1 reported 0.193 / 0.649).",
          f"- Incident-level test precision {w.test_incident_precision:.3f}, episode recall {w.test_episode_recall:.3f} "
          f"({int(w.test_episodes_caught)} of {int(w.test_episodes)} episodes preceded by an incident).",
          f"- Median lead time: {fmt(float(w.lead_median_s), 1)} s (IQR {fmt(float(w.lead_q1_s), 1)}-{fmt(float(w.lead_q3_s), 1)} s) "
          "from the first incident to the episode's first alert line, over the preceded test episodes.",
          f"- LLM latency ({cfg['llm']['model']}): " + (f"{len(lat)} incident card(s) in the last daemon run, median "
          f"{np.median(lat):.1f} s (min {min(lat):.1f}, max {max(lat):.1f}); source: `{lat_src}`." if lat
          else f"no daemon run of `{sel['name']}` with incident cards recorded yet."), ""]
    abl = res / "ablation.md"
    if abl.exists():
        md += [abl.read_text(encoding="utf-8").replace("# Ablation", "## Ablation", 1)]
    (res / "RUN_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {res / 'RUN_REPORT.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--gate-sweep", action="store_true", help="only write results/gate_sweep.csv")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    if args.gate_sweep:
        write_gate_sweep(cfg)
    elif args.ablation:
        ablation(cfg, args.force)
    elif args.report:
        report(cfg)
    else:
        compare(cfg)
