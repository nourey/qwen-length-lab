# Continue after the Kaggle run

Provide the downloaded results archive and this experiment folder to the next Codex session. Suggested task:

> Inspect the Qwen length-prediction experiment's real Kaggle run. First verify the manifest, frozen predictions, model revision, actual GPU configuration, full 16/32 split, error coverage and cap-hit rate. Recompute metrics from the saved JSONL and compare with the saved report. Explain whether the predeclared pilot signal was met, with paired baseline comparisons, threshold support, calibration limitations and prediction runtime overhead. Do not treat synthetic demo data as model results or tune against the same held-out test. If the run failed, diagnose the earliest failure and propose the smallest concrete fix. Preserve all original artifacts. A later experiment should have a new run directory, frozen configuration and fresh test prompts.

Useful files: `manifest.json`, `generation_configs.json`, `predictions_frozen.json`, `predictions.jsonl`, `generations.jsonl`, `results.jsonl`, `metrics.json`, `comparison.csv`, `calibration.csv`, `baseline_fit.json` and `nvidia-smi.txt`.

If there is a clear quality signal but high prediction cost, the next bounded experiment is a **smaller predictor with the 8B target fixed**. Its tokenizer must still measure target lengths with the generator's tokenizer, and model IDs/configurations must be logged separately. That comparison is deliberately outside the first experiment.
