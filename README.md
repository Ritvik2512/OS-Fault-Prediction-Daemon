# OS Fault-Prediction Daemon

A daemon that reads a system log as a stream, **predicts the run-up to a fault** before the fault's first alert line appears, and has a local LLM write a plain-English incident card. A Streamlit dashboard shows the stream, the model's confidence, and the incident cards live.

> **Scope, stated plainly.** The stream is a **simulated live stream from held-out [BGL](https://github.com/logpai/loghub) logs** (Blue Gene/L supercomputer system logs), replayed line by line; it is not a monitored live server, and tailing real `journalctl` output is future work. The models **predict** pre-fault windows; they do not detect faults directly. LLM incident cards are generated summaries, not verified diagnoses.

---

## Results

All numbers come from the scripts in `v2/` and are on the **held-out test split** (last 20% of the log, chronologically). Thresholds and gate settings were tuned on the validation split only. Full write-ups are in [`results/RUN_REPORT.md`](results/RUN_REPORT.md) and [`results/model_report.md`](results/model_report.md).

### Window level

Window level asks: is this 20-line window followed by an alert within the next 100 lines?

| Model | Val PR-AUC | Precision | Recall | Test PR-AUC | Train time | Inference |
|---|---|---|---|---|---|---|
| v1 DistilBERT (reported, earlier pipeline) | – | 0.193 | 0.649 | – | – | – |
| DistilBERT | 0.240 | 0.946 | 0.170 | 0.378 | 24 min | 8.5 ms |
| **RoBERTa (selected)** | **0.259** | **0.964** | **0.303** | **0.473** | 48 min | 14 ms |
| ELECTRA | 0.217 | 0.954 | 0.150 | 0.431 | 33 min | 14 ms |
| Gradient-boosted trees (HGB) | 0.213 | 1.000 | 0.012 | 0.363 | 0.9 min (CPU) | 1.4 ms |

### Incident level

Incident level asks: did the daemon warn before a fault started? This is a different unit from the window table above.

| Model | Gate (N, cooldown) | Incident precision | Faults warned in advance |
|---|---|---|---|
| DistilBERT | 2, 30 | 0.577 | 30 / 218 |
| **RoBERTa** | 1, 0 | 0.343 | 35 / 218 |
| ELECTRA | 1, 0 | 0.452 | 51 / 218 |
| HGB | 1, 5 | 0.706 | 12 / 218 |

- **Selection rule (fixed in advance):** highest validation PR-AUC, with ties broken by validation incident precision and then by speed. RoBERTa wins.
- **Operating point:** precision-first. The threshold is the lowest one that reaches **0.8 precision on validation**. This was adopted after the first results, which used a recall ≥ 0.40 rule. It costs recall: RoBERTa's recall is 0.303, below the original 0.40 floor.
- **Targets missed:** incident-level precision ≥ 0.80 is not met by the selected model. The reports list this.
- **Lead time:** median about 12 s for RoBERTa. An incident only counts if the fault starts within 100 lines, so lead time is bounded by how fast the log is being written.

## How it works

```
BGL.log ──► Drain3 templates (mined on train lines only, match-only at runtime)
        ──► 20-line windows rendered as text:  "KERNEL INFO <template> x7 | APP FATAL <template> | ..."
        ──► model ──► P(pre-fault) per window
        ──► gate: fire after N windows ≥ τ, one incident per flagged run, then a cooldown
        ──► local LLM (Ollama, llama3.2:3b) ──► JSON card {summary, component, severity, suggested_action}
        ──► results/live_events.jsonl ──► Streamlit dashboard
```

**Labels.** A window is *pre-fault* if it contains no alert line but an alert occurs within the next K = 100 lines. Windows that contain alerts are not used for training.

**Data handling** (`v2/build_dataset.py`):
- Chronological split by line: 70 / 10 / 20 for train / val / test. The script checks that no line is shared between splits.
- Template text instead of v1's bare template IDs, with consecutive duplicate lines collapsed. max_length is 512 tokens, and 1–2% of windows are truncated.
- Training-only measures:
  - a 200-line ambiguity buffer that drops negatives just outside the 100-line look-ahead;
  - overlapping windows (stride 10);
  - 10:1 negative downsampling.
- Templates are mined on the training lines only, as a deployed system would be. 46% of test lines match no training template; they fall back to digit-masked raw text. That drift is the main reason validation and test numbers differ.

**Models.**
- Three fine-tuned transformers: DistilBERT, RoBERTa-base and ELECTRA-base. They use fp16 on an RTX 3050 6 GB and early-stop on validation PR-AUC.
- One classical baseline: scikit-learn `HistGradientBoostingClassifier` on template-count features.

**Ablation.** Each data measure was switched off in turn for the selected model; see [`results/ablation.md`](results/ablation.md). The template-text/run-length fix is the only one that helped on every metric.

## Quick start (Windows, Command Prompt)

Requirements: Python 3.13, an NVIDIA GPU with CUDA (CPU works but trains slowly), [Ollama](https://ollama.com) with `llama3.2:3b` pulled, and `BGL.log` from Loghub at `data\BGL.log`.

```bat
cd /d "C:\programs\OS project"
python -m venv fault-env
fault-env\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate drain3 scikit-learn pandas matplotlib streamlit ollama pyyaml
ollama pull llama3.2:3b
```

ELECTRA's weights are downloaded into a local folder, because Windows can block the symlinks HuggingFace's cache normally creates:

```bat
set HF_HUB_DISABLE_XET=1
python -c "from huggingface_hub import snapshot_download as d; d('google/electra-base-discriminator', local_dir='models/pretrained/electra-base-discriminator', allow_patterns=['*.json','vocab.txt','*.bin','*.safetensors'])"
```

### Run the whole pipeline

```bat
v2\run_all.bat
```

The script builds the dataset, trains all four models, evaluates them, selects the winner, runs the ablation, writes the reports, and runs every model through the daemon for the comparison. Each step skips work whose outputs already exist; pass `--force` to a step to redo it. A full run from scratch takes about 5 hours on an RTX 3050, most of it the ablation.

### Run the demo

Make sure Ollama is running: `curl http://localhost:11434/api/version` should print a version. If it doesn't, start the Ollama app or run `ollama serve`. Then:

```bat
fault-env\Scripts\streamlit.exe run v2\dashboard.py
```

In the sidebar, choose a model (RoBERTa, DistilBERT, ELECTRA or gradient-boosted trees), a replay slice and a speed, then press **▶ Start demo**.
- **Model's showcase** replays a test fault that model warned about, so it fires at least once. The slice was picked using the labels, so it shows the system working, not how often it works.
- **Shared demo** replays the same slice for every model, for a side-by-side.

The daemon also runs from the command line:

| Command | What it does |
|---|---|
| `fault-env\Scripts\python.exe v2\daemon.py --demo` | selected model on its demo slice |
| `fault-env\Scripts\python.exe v2\daemon.py --demo --model electra` | any model (`distilbert`, `roberta`, `electra`, `hgb`) |
| `fault-env\Scripts\python.exe v2\daemon.py --eval --no-llm` | also logs ground truth, to a separate file |
| `fault-env\Scripts\python.exe v2\daemon.py --full-test --speed 0` | replays the whole test split |

`--demo` mode never prints or logs ground-truth labels.

## Project layout

```
v2/
  config.yaml            all tunables (window sizes, split, models, threshold rule, gate grid, LLM)
  common.py              line parsing, match-only Drain3 encoder, window rendering, features, metrics, gate
  build_dataset.py       parse + mine templates, windows/labels, train/val/test split, token-length stats
  train_transformer.py   fine-tune a HuggingFace model (DistilBERT / RoBERTa / ELECTRA)
  train_classical.py     HistGradientBoostingClassifier on template counts
  evaluate.py            thresholds, window + incident metrics, plots, comparison, ablation, RUN_REPORT
  select_model.py        selection rule -> models/selected.json, demo slices
  detector.py            detect(lines) -> (is_fault, confidence) for any model
  daemon.py              replay + gate + LLM incident cards -> results/live_events.jsonl
  dashboard.py           Streamlit UI with a built-in demo launcher
  compare_demo.py        runs every model through the daemon -> results/model_report.md
  run_all.bat            whole pipeline
results/                 reports (*.md), plots (*.png), tables (*.csv), metrics (*.json), training logs
models/selected.json     the selected model and its operating point (weights are not in git)
*.py (project root)      the original v1 scripts, kept for reference
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'fault-env' is not recognized` in cmd | Use backslashes: `fault-env\Scripts\python.exe ...` |
| `bind: Only one usage of each socket address` from `ollama serve` | Ollama is already running; nothing to do. |
| Cards say "LLM parse failed" | Ollama isn't reachable; start it and run the demo again. |
| HuggingFace download 404 on `xet-read-token` | `set HF_HUB_DISABLE_XET=1` |
| `WinError 1314` (symlink privilege) on download | Download with `local_dir=...` as shown above. |
| CUDA out of memory while training | Stop Ollama first (`ollama stop llama3.2:3b`); it keeps ~2.5 GB in VRAM. |

## Data and credits

- **Dataset:** BGL from [Loghub](https://github.com/logpai/loghub) (Zhu et al., *Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics*). 4,713,493 lines, of which 7.39% are alert lines.
- **Log parsing:** [Drain3](https://github.com/logpai/Drain3).
- **Pre-trained models:** `distilbert-base-uncased`, `FacebookAI/roberta-base` and `google/electra-base-discriminator` from HuggingFace.
- **Incident writer:** `llama3.2:3b` via Ollama.
