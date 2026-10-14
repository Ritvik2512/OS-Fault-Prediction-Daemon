# Ablation: `roberta` with each data lever switched off

Each row retrains the selected model with exactly one section-5 lever disabled. tau* is re-tuned on val (the lowest threshold whose val precision >= 0.8 (i.e. the highest val recall at that precision)); precision/recall are window-level on TEST. Val and test sets are identical in every row. Single run per row (seed 42), so small differences are within noise. To keep the runs affordable on the RTX 3050, every ablation row (and its all-levers-on control) is trained for exactly 1 epoch(s); compare ablation rows with that control, not with the fully trained selected model.

| Variant | Train windows | val PR-AUC | tau* | Test precision | Test recall | Test F1 | Test PR-AUC | Train (min) |
|---|---|---|---|---|---|---|---|---|
| all levers on (selected model, early-stopped, best epoch 1 of 2 run) | 47311 | 0.259 | 0.9302 | 0.964 | 0.303 | 0.461 | 0.473 | 47.5 |
| all levers on - control trained under the ablation budget | 47311 | 0.247 | 0.9816 | 0.958 | 0.162 | 0.278 | 0.396 | 15.1 |
| no RLE / truncation fix (no run-length encoding, max_length 64) | 47311 | 0.196 | 0.9931 | 0.947 | 0.014 | 0.028 | 0.283 | 8.3 |
| no ambiguity buffer | 47311 | 0.217 | 0.9860 | 0.977 | 0.270 | 0.423 | 0.471 | 15.0 |
| no overlapping train windows (stride = W) | 23705 | 0.219 | 0.9472 | 0.957 | 0.230 | 0.370 | 0.419 | 8.1 |
| no negative downsampling | 296804 | 0.262 | 0.9447 | 0.958 | 0.258 | 0.407 | 0.457 | 84.4 |

Effect of switching each lever **off**, relative to the control row (all levers on - control trained under the ablation budget). A negative change means the lever was helping:

- no RLE / truncation fix (no run-length encoding, max_length 64): val PR-AUC -0.050, test precision -0.010, test recall -0.148, test PR-AUC -0.113
- no ambiguity buffer: val PR-AUC -0.030, test precision +0.019, test recall +0.107, test PR-AUC +0.074
- no overlapping train windows (stride = W): val PR-AUC -0.028, test precision -0.001, test recall +0.067, test PR-AUC +0.023
- no negative downsampling: val PR-AUC +0.015, test precision +0.001, test recall +0.096, test PR-AUC +0.060

Read val PR-AUC (the metric everything was tuned on) as the primary signal. Test precision at a val-tuned threshold is noisy here because the val and test distributions differ (see RUN_REPORT.md), so a lever is only clearly useful when val and test move in the same direction.
