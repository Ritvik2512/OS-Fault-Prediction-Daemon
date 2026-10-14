# daemon.py — section 8: replay a held-out BGL slice line by line, predict pre-fault windows
# with the selected model, gate them into incidents, and have a local LLM write the card.
# This is a SIMULATED live stream from held-out logs, not a monitored live server.
#
#   python v2/daemon.py --demo                 # default: demo slice, never touches ground truth
#   python v2/daemon.py --eval                 # also logs ground-truth labels (validation only)
#   python v2/daemon.py --full-test --speed 0  # whole test split, as fast as possible

import argparse
import json
import queue
import threading
import time
from collections import Counter

import common as C
from detector import Detector, model_spec, read_lines

PROMPT = """You are an OS incident assistant. The Blue Gene/L system log lines below were flagged by a \
fault-prediction model as the possible run-up to a failure (repeated lines are shown once with an [xN] count). \
Write a JSON incident report.

Return ONLY a JSON object with exactly these keys:
- "summary": one sentence describing what is happening across these lines
- "component": the subsystem named in the lines (e.g. KERNEL, APP, torus, EDRAM, discovery, mmcs)
- "severity": one of "low", "medium", "high", "critical"
- "suggested_action": one short, concrete next step

Example:
{"summary":"Repeated EDRAM error-correction events indicate memory degradation on the node.","component":"EDRAM","severity":"medium","suggested_action":"Run a memory diagnostic on the affected node and flag it for service."}

Log lines:
"""
KEYS = ("summary", "component", "severity", "suggested_action")
SEVERITIES = ("low", "medium", "high", "critical")


def dedupe(lines, cap):
    """Collapse repeats (ignoring timestamp/node prefix) so the LLM sees signal, capped at `cap` lines."""
    msgs = []
    for ln in lines:
        p = C.parse_stream_line(ln)
        msgs.append(f"{p[1]} {p[2]} {p[3]}" if p else ln)
    return [f"[x{n}] {m}" if n > 1 else m for m, n in Counter(msgs).most_common(cap)]


def parse_card(text):
    d = json.loads(text)
    if not isinstance(d, dict) or any(not str(d.get(k, "")).strip() for k in KEYS):
        raise ValueError("missing keys")
    d = {k: str(d[k]).strip() for k in KEYS}
    d["severity"] = d["severity"].lower()
    if d["severity"] not in SEVERITIES:
        raise ValueError("bad severity")
    return d


def llm_card(cfg, lines):
    """Streamed JSON card from the local LLM; one retry on malformed output, then a fallback card."""
    import ollama
    lc = cfg["llm"]
    prompt = PROMPT + "\n".join(dedupe(lines, cfg["daemon"]["max_llm_lines"]))
    err = ""
    for _ in range(2):
        try:
            chunks = ollama.generate(model=lc["model"], prompt=prompt, format="json", stream=True,
                                     keep_alive=lc["keep_alive"],
                                     options={"num_predict": lc["num_predict"], "temperature": lc["temperature"]})
            return parse_card("".join(c["response"] for c in chunks))
        except Exception as ex:          # malformed JSON, missing keys, or Ollama unreachable
            err = f"{type(ex).__name__}: {ex}"
    return {"summary": "LLM parse failed", "component": "unknown", "severity": "medium",
            "suggested_action": "Review the flagged log lines manually.", "error": err[:200]}


def warm_up(cfg):
    import ollama
    try:
        ollama.generate(model=cfg["llm"]["model"], prompt="ok", keep_alive=cfg["llm"]["keep_alive"],
                        options={"num_predict": 1})
        print("LLM warmed up")
    except Exception as ex:
        print(f"LLM warm-up failed ({ex}); cards will fall back until Ollama is reachable")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="default: never print or log ground truth")
    mode.add_argument("--eval", action="store_true", help="log ground-truth labels next to predictions")
    ap.add_argument("--full-test", action="store_true")
    ap.add_argument("--slice", type=int, nargs=2, metavar=("START", "END"), help="line range to replay")
    ap.add_argument("--model", help="any evaluated model (default: the selected one in models/selected.json)")
    ap.add_argument("--slice-mode", choices=["own", "shared"], default="own",
                    help="own = this model's showcase slice; shared = the selected model's demo slice")
    ap.add_argument("--events", help="event file to write (default results/live_events.jsonl)")
    ap.add_argument("--speed", type=float, help="lines per second (0 = no delay)")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    cfg = C.load_config()
    sel = model_spec(cfg, args.model)
    W, K = cfg["window"]["W"], cfg["window"]["K"]
    tau, N, cooldown = sel["tau"], sel["N"], sel["cooldown"]
    speed = cfg["daemon"]["speed"] if args.speed is None else args.speed
    det = Detector(cfg, sel)
    lines_meta = C.load_lines(cfg)
    bounds = C.read_json(C.path(cfg, "data_dir", "default", "meta.json"))["bounds"]["test"]
    slice_ = sel["own_slice"] if args.slice_mode == "own" and sel.get("own_slice") else sel["shared_slice"]
    a, b = bounds if args.full_test else (args.slice or slice_)
    assert bounds[0] <= a < b <= bounds[1], "the replay slice must lie inside the held-out test split"
    stream, truth = read_lines(cfg, lines_meta, a, b)
    if not args.eval:
        truth = None                      # demo mode: ground truth is dropped before the loop starts
    del lines_meta

    out = (C.ROOT / args.events if args.events else
           C.path(cfg, "results_dir", "live_events_eval.jsonl" if args.eval else "live_events.jsonl"))
    out.parent.mkdir(exist_ok=True)
    log = open(out, "w", encoding="utf-8")
    lock = threading.Lock()

    def emit(rec):
        with lock:
            log.write(json.dumps(rec) + "\n")
            log.flush()

    def writer():                         # LLM cards are written off the hot path so the stream keeps flowing
        while True:
            job = jobs.get()
            if job is None:
                return
            t0 = time.time()
            card = llm_card(cfg, job["lines"])
            dt = time.time() - t0
            emit({"type": "incident", "t": time.time(), "window_range": job["window_range"],
                  "confidence": job["confidence"], "card": card, "llm_seconds": round(dt, 2)})
            print(f"\n  --- incident card (generated summary, not a verified diagnosis) ---\n"
                  f"  summary   : {card['summary']}\n  component : {card['component']}\n"
                  f"  severity  : {card['severity']}\n  action    : {card['suggested_action']}\n"
                  f"  windows   : {job['window_range'][0]}-{job['window_range'][1]}   llm={dt:.1f}s\n"
                  f"  --------------------------------------------------------------------\n")

    jobs = queue.Queue()
    worker = threading.Thread(target=writer, daemon=True)
    worker.start()
    if not args.no_llm:
        warm_up(cfg)
    emit({"type": "start", "t": time.time(), "model": sel["name"], "tau": tau, "N": N, "cooldown": cooldown,
          "mode": "eval" if args.eval else "demo", "windows": (b - a) // W, "slice": [a, b],
          "slice_mode": "full-test" if args.full_test else "custom" if args.slice else args.slice_mode,
          "selected": args.model in (None, C.read_json(C.path(cfg, "models_dir", "selected.json"))["name"])})
    print(f"Simulated live stream from held-out BGL logs: {b - a:,} lines at {speed:g} lines/s")
    print(f"model={sel['name']}  tau={tau:.4f}  N={N}  cooldown={cooldown} windows  mode={'eval' if args.eval else 'demo'}\n")

    buf, run, over, cool, fired, widx, armed = [], [], 0, 0, 0, 0, True
    for i, line in enumerate(stream):
        buf.append(line)
        if speed:
            time.sleep(1.0 / speed)
        if len(buf) < W:
            continue
        t_det = time.perf_counter()
        hot, conf = det.detect(buf)
        rec = {"type": "window", "t": time.time(), "window_idx": widx, "confidence": round(conf, 4), "lines": buf,
               "detect_ms": round(1000 * (time.perf_counter() - t_det), 2)}
        note = ""
        if truth is not None:             # --eval only
            win = truth[i + 1 - W:i + 1]
            rec["truth"] = {"alert_in_window": any(win), "alert_within_K": any(truth[i + 1:i + 1 + K])}
            note = f"  [eval: alert_in_window={any(win)} alert_within_K={rec['truth']['alert_within_K']}]"
        emit(rec)
        cool = max(cool - 1, 0)
        if hot:
            over += 1
            run.append((widx, buf))
        else:
            over, run, armed = 0, [], True    # the flagged run (episode) is over: re-arm
        tag = "  PRE-FAULT" if hot else ""
        if armed and over >= N and cool == 0:   # gate trips: one incident per flagged run
            fired += 1
            block = [ln for _, w in run for ln in w]
            job = {"window_range": [run[0][0], widx], "confidence": round(conf, 4), "lines": block}
            emit({"type": "incident_fired", "t": time.time(), "window_range": job["window_range"], "confidence": job["confidence"]})
            tag += f"  <-- INCIDENT PREDICTED (windows {run[0][0]}-{widx})"
            if not args.no_llm:
                jobs.put(job)
            cool, armed = cooldown, False
        print(f"[{widx:>5}] conf={conf:.3f}{tag}{note}")
        buf, widx = [], widx + 1

    jobs.put(None)
    worker.join()
    emit({"type": "end", "t": time.time(), "windows": widx, "incidents": fired})
    log.close()
    print(f"\ndone. {widx} windows, {fired} incident(s) predicted. Events: {out}")


if __name__ == "__main__":
    main()
