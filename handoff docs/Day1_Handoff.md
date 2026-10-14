# Day 1 Handoff — OS Fault-Prediction Daemon

## Project (context for every day's chat)
Building a **product** (not a comparative study — teacher rejected studies): a real-time OS log fault-prediction daemon.

- **Fine-tuned model (HARD REQ):** one DistilBERT, fine-tuned on a **public dataset (HARD REQ)**, detects fault windows in OS logs.
- **OS functionality (HARD REQ):** a daemon that reads a live/replayed log stream and flags predicted faults.
- **Product layer (the "wow"):** when a fault is flagged, a **local LLM** (Ollama, e.g. Llama 3.2 3B) writes a plain-English incident report.
- **Not a study:** train one model, ship one tool, never compare models. No comparison table anywhere.
- **Framing = fault prediction:** label the window *before* a fault (precursor), not the fault line itself.
- Owners: Ritvik Battula (24BAI0156), Shaurya Kumar (24BCE0876). Hardware: RTX 3050 6GB.
- 3-day sprint to a working demo. Each day is its own chat; a handoff doc starts each one.

## Day 1 goal
A working fine-tuned DistilBERT detector that flags fault windows on held-out public log data. Data pipeline done.

## Day 1 tasks
1. **Dataset:** download a Loghub set — **BGL** is easiest (labelled, public, standard). github.com/logpai/loghub
2. **Parse** raw logs into templates with **Drain3** (`pip install drain3`). Collapses millions of lines into a small template vocab.
3. **Windowing + labelling:** slice template sequences into fixed windows; label a window **PRE-FAULT (1)** if a labelled fault occurs within the next Δt / K lines after it, else **normal (0)**. Exclude the window containing the fault itself, so the model only learns the run-up. *This is what makes it prediction, not detection.*
4. **Write the training script against the data shape first** (dummy data ok) so it runs end-to-end, then slot real data in.
5. **Fine-tune one** `distilbert-base-uncased` for binary classification (`num_labels=2`). Save it behind `detect(window) -> (is_fault, confidence)`. Target a *usable* F1, not a great one — a detector that fires correctly on held-out data is the goal. Report **F1, not accuracy** (faults are rare).
6. **Stack:** Python, PyTorch, HuggingFace transformers, drain3. Trains in minutes on the 3050 — iterate.

## Definition of done (Day 1)
- Held-out BGL windows classified fault/normal by a saved fine-tuned DistilBERT, with a sensible F1.
- A `detect(window)` function ready for the daemon to call on Day 2.
- **Note down:** dataset path, window size, Δt, label balance, final F1, and any parsing quirks — the Day 2 chat needs these.

## Watch out
- Faults are rare → class imbalance. Use class weighting / oversampling; report F1.
- Keep the model small and training short; don't over-engineer. One model only.
- Don't start the LLM/daemon today — Day 1 is data + detector only.
- Save everything (parsed data, windows, model weights) to disk so Day 2's fresh chat can load it.

## End of day
Report back what actually happened (F1, window/Δt chosen, label balance, any breakage). The Day 2 handoff will be written only then, reflecting reality.
