# select_model.py — section 7.4: apply the selection rule to results/model_comparison.csv
# and write models/selected.json (the only model the daemon and dashboard use).

import numpy as np
import pandas as pd

import common as C
from evaluate import pick_winner


def demo_slice(cfg, sel, lines):
    """A held-out test slice for the daemon demo: quiet lead-in, then a fault episode whose
    run-up the gate fired on in the offline replay. Chosen with ground truth, offline, once;
    it is a showcase slice and says nothing about average performance (see the test metrics)."""
    W, K, dm = cfg["window"]["W"], cfg["window"]["K"], cfg["daemon"]
    meta = C.read_json(C.path(cfg, "data_dir", "default", "meta.json"))
    a, b = meta["bounds"]["test"]
    stream = C.load_stream(cfg, "test")
    probs = np.load(C.path(cfg, "results_dir", f"preds_{sel['name']}_test_stream.npy"))
    ep_s, ep_e = C.find_episodes(lines["alert"], a, b, cfg["eval"]["episode_gap"], W)
    t = stream[C.simulate_gate(probs, sel["tau"], sel["N"], sel["cooldown"])] + W
    cum = np.concatenate([[0], np.cumsum(lines["alert"], dtype=np.int64)])
    best = None
    for e in ep_s:
        lo = max(a, e - dm["lead_lines"])
        lo += (-(lo - a)) % W                          # keep the slice on the test window grid
        hits = t[(t > e - K) & (t <= e)]
        quiet = cum[e] - cum[lo] == 0                  # no alert lines in the lead-in
        early = ((t > lo) & (t <= e - K)).sum()        # incidents in the lead-in that precede nothing
        if len(hits):
            cand = (not quiet, early, -(e - hits[0]), int(lo), int(min(b, e + dm["tail_lines"])))
            best = min(best, cand) if best else cand
    return None if best is None else [best[3], best[4]]


def main():
    cfg = C.load_config()
    res = C.path(cfg, "results_dir")
    df = pd.read_csv(res / "model_comparison.csv")
    df = df[df.type.notna()]
    name, why = pick_winner(df, cfg["eval"]["tie_margin"])
    w = df[df.model == name].iloc[0]
    info = C.read_json(res / f"train_{name}.json")
    sel = {"name": name, "path": str(C.path(cfg, "models_dir", name).relative_to(C.ROOT)).replace("\\", "/"),
           "type": info["type"], "tau": float(w.tau), "N": int(w.N), "cooldown": int(w.cooldown),
           "max_length": info.get("max_length"), "rle": info.get("rle", True), "n_tid": info.get("n_tid"),
           "reason": why[0].upper() + why[1:]}
    lines = C.load_lines(cfg)
    sel["demo_slice"] = demo_slice(cfg, sel, lines)
    C.write_json(C.path(cfg, "models_dir", "selected.json"), sel)
    # each model's own showcase slice, so the dashboard can demo any of them (its own tau / N / cooldown)
    slices = {}
    for r in df.itertuples():
        op = {"name": r.model, "tau": float(r.tau), "N": int(r.N), "cooldown": int(r.cooldown)}
        slices[r.model] = sel["demo_slice"] if r.model == name else demo_slice(cfg, op, lines)
    C.write_json(C.path(cfg, "models_dir", "demo_slices.json"), slices)

    print(f"Selection rule: highest val PR-AUC; tiebreak (within {cfg['eval']['tie_margin']}): higher val incident "
          "precision, then faster inference.\n")
    print(df[["model", "val_pr_auc", "val_incident_precision", "infer_ms_per_window"]].to_string(index=False))
    print(f"\nWinner: {name} - {why}.")
    print(f"tau={sel['tau']:.4f}  N={sel['N']}  cooldown={sel['cooldown']}  max_length={sel['max_length']}")
    print(f"demo slice (test lines): {sel['demo_slice']}")
    print(f"per-model showcase slices: {slices}")
    print(f"wrote {C.path(cfg, 'models_dir', 'selected.json')}")


if __name__ == "__main__":
    main()
