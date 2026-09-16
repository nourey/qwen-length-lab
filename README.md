# Qwen output-length pilot on Kaggle

Predict the next response's token length **before generation**, then compare the forecast with the response actually emitted. The primary question is whether a structured Qwen forecast beats cheap baselines on held-out prompts. No model accuracy results are bundled: the GPU experiment still needs to run on Kaggle.

## Start here

1. Create a **Kaggle Notebook**, import `kaggle_qwen_length.ipynb`, select **GPU T4 ×2**, and enable **Internet**.
2. Leave `MODEL_SIZE = "8b"`. Run all cells. Four separate smoke prompts run first; the 48-prompt pilot runs only if the smoke predictions and generations succeed. Set `RUN_PILOT = False` to stop after smoke.
3. Inspect the comparison table and `evaluation.png`. Download `qwen_length_results_8b.zip` from the notebook output, or save a Kaggle version to retain `/kaggle/working` artifacts.

The notebook embeds the scripts, prompts, tests and requirements, so importing it directly also works without cloning. No dataset upload, API key, Jev account, or separate file attachment is needed. Internet is used to install packages and download the public model. All inference runs in the Kaggle session.

## Clone and run from GitHub

Public repository: https://github.com/nourey/qwen-length-lab, branch `main`.
In a fresh Kaggle session select **T4 ×2** and enable **Internet**. Use one option below.
Paste each shell block into a notebook code cell beginning with `%%bash`, or run it directly in the Kaggle terminal. A shell `cd` does not persist into a different notebook cell; each execution block starts with its own `cd`.

### Option A: execute the existing notebook

```bash
set -euo pipefail
cd /kaggle/working
git clone --branch main https://github.com/nourey/qwen-length-lab.git
cd qwen-length-lab
bash scripts/setup_kaggle.sh --notebook
/kaggle/temp/qwen-length-env/bin/python -m nbconvert \
  --to notebook --execute kaggle_qwen_length.ipynb \
  --ExecutePreprocessor.kernel_name=qwen-length-lab \
  --ExecutePreprocessor.timeout=-1 \
  --output kaggle_qwen_length.executed.ipynb \
  --output-dir /kaggle/working/qwen-length-runs
```

This executes all cells of the checked-in notebook, including its smoke gate, pilot, plots and result ZIP. It saves the executed notebook to `/kaggle/working/qwen-length-runs/kaggle_qwen_length.executed.ipynb`. Notebook execution follows the documented [nbconvert interface](https://nbconvert.readthedocs.io/en/latest/execute.html).

For the interactive Kaggle editor instead, download `kaggle_qwen_length.ipynb` from GitHub and use **File → Import Notebook** in a fresh Kaggle notebook, then enable T4 ×2 and Internet and Run All. The notebook itself installs its dependencies; no manual shell install is needed for that route.

### Option B: run the scripts without opening the notebook

```bash
set -euo pipefail
cd /kaggle/working
git clone --branch main https://github.com/nourey/qwen-length-lab.git
cd qwen-length-lab
bash scripts/setup_kaggle.sh
/kaggle/temp/qwen-length-env/bin/python -u run_kaggle.py
```

`run_kaggle.py` runs local tests and the hardware check, then exactly four smoke prompts. It requires four successful generations and zero prediction errors, pins the pilot to that smoke run's model revision, and only then runs all 48 pilot prompts. It uses the original 8B/NF4/non-thinking settings without modifying the inference or scoring code.

Both options automatically create and write:

```text
/kaggle/working/qwen-length-runs/
├── v1-smoke-qwen3-8b/
└── v1-pilot-qwen3-8b/
    ├── predictions.jsonl
    ├── predictions_frozen.json
    ├── generations.jsonl
    ├── results.csv
    ├── results.jsonl
    ├── scores.csv
    ├── metrics.json
    ├── comparison.csv
    ├── calibration.csv
    ├── confusion_*.csv
    ├── evaluation.png
    └── ... manifests, exact token IDs, seeds and runtime records
```

Option A additionally writes `/kaggle/working/qwen_length_results_8b.zip`. Model weights and the Python environment are placed under `/kaggle/temp`, not committed or included in the results ZIP. Save a Kaggle version or download outputs before ending the session.

For an exact repository revision, run `git checkout --detach <commit SHA>` after cloning and before setup. The published handoff supplies that SHA. `git rev-parse HEAD` shows your current source commit. For resume, skip the clone and rerun the same option from the existing clone; do not edit an existing run's files. Choose a fresh session or different run directory for a changed experiment.

### Repository layout

```text
qwen-length-lab/
├── kaggle_qwen_length.ipynb
├── experiment.py          # Existing inference and checkpointing
├── metrics.py             # Existing evaluation and baselines
├── plot_results.py
├── run_kaggle.py          # Smoke-before-pilot packaging entry point
├── build_notebook.py
├── predictor_system.txt
├── prompts.jsonl          # 16 development + 32 test prompts
├── smoke_prompts.jsonl    # 4 disjoint smoke prompts
├── requirements-kaggle.txt
├── requirements-notebook.txt
├── scripts/setup_kaggle.sh
├── tests/
│   ├── test_experiment.py
│   └── test_packaging.py
├── README.md
├── CONTINUE_IN_CODEX.md
└── .gitignore
```

Only source, dependency specifications and authored prompt data are tracked. Generated results, model files, environments, caches and local credentials are ignored. Tests and evaluation work from the repository root without the originating workspace. The `/kaggle/...` paths are runtime destinations, not personal workstation paths.

## Hardware and compatibility, checked September 16, 2026

Kaggle's [P100 retirement announcement](https://www.kaggle.com/discussions/product-announcements/735239) schedules P100 retirement for September 15, 2026 and identifies T4 ×2 with 16 GB per GPU as the replacement option. Its [notebook documentation](https://www.kaggle.com/docs/notebooks) lists T4 ×2 and 29 GB host RAM, although that page still contains older P100 guidance. The actual session checks `nvidia-smi`, CUDA availability, GPU count, free memory and compute capability. Your account's current quota and session allocation cannot be established from public documentation; confirm them in Kaggle.

Default: **Qwen/Qwen3-8B, bitsandbytes NF4 with double quantization, FP16 compute, batch size 1, explicit `device_map={"": 0}`**. This is expected to fit the pilot's short contexts on one T4; it has not been measured here. The second T4 remains available. Two cards do not provide one automatically combined memory pool, and splitting a small quantized model adds complexity without being necessary for this first test. There is no CPU/disk model offload. `--device 1` selects the other GPU if needed.

The [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) requires Transformers ≥4.51.0 and documents the non-thinking template switch and sampling settings. The pinned stack is Transformers **4.57.6**, Accelerate **1.12.0**, bitsandbytes **0.48.2**. These are deliberate fixed versions, not a claim to use the latest releases. The [bitsandbytes installation guide](https://huggingface.co/docs/bitsandbytes/main/en/installation) supports NF4 on compute capability ≥6.0, including T4. Eager attention avoids a FlashAttention dependency; FP16 is chosen for T4 compatibility. The notebook preserves Kaggle's installed CUDA-enabled PyTorch and requires ≥2.3, as specified by [bitsandbytes 0.48.2](https://pypi.org/project/bitsandbytes/0.48.2/).

The 8B preset checks for at least 10 GiB free as a conservative gate, not a fit guarantee. If unavailable, clear stale GPU allocations or choose `MODEL_SIZE = "4b"` and use a **new run directory**. This selects [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) with the same quantization and a 6 GiB free-memory gate. Model switching is never automatic. The conclusion then applies to the 4B target; do not pool the two runs.

## Exactly what is predicted

The primary length `L` is the number of **new token IDs emitted**, including a terminal EOS token if emitted, excluding input IDs and the assistant prefix supplied by the template. It measures decoder work for a capped request. Text is also saved, but it is never decoded and re-tokenized to obtain `L`.

The generator uses the same loaded quantized model as the predictor, in an independent call with only the original conversation. Thinking is explicitly disabled in both calls. Target decoding is sampled with temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1, one beam and `max_new_tokens=1536`. Predictor decoding is greedy, with a 192-token response limit. A fresh generation configuration prevents unrelated model defaults from changing the experiment.

The predictor receives the **full pre-generation conversation**, exact input token count, target model/revision, quantization and target generation settings. It receives no answer, test label, split or task-family metadata. It emits:

```json
{"bucket_probs": [0.10, 0.20, 0.30, 0.25, 0.15]}
```

This is an illustrative format, not a measured prediction. Buckets are `[1,128]`, `[129,256]`, `[257,512]`, `[513,1024]`, `[1025,1536]`. Their tail sums give `P(L>128)`, `P(L>256)`, `P(L>512)` and `P(L>1024)`. Strict inequalities and coherent bucket sums avoid contradictory thresholds. All forecasts are saved and hashed **before any target response in that run is generated**.

`predictor_system.txt` uses a **state → typed question → structured decision** format inspired by System One. This is an ordinary Qwen prompting experiment. It does not use Jev, implement JEPA, depend on the System One Adapter, or provide native/calibrated probability guarantees. These are verbalized probabilities whose calibration is an empirical question.

The derived expected-token estimate uses the probability-weighted bucket midpoints: `64.5, 192.5, 384.5, 768.5, 1280.5`. This approximates a uniform conditional distribution inside each bucket. It is coarse, especially for tiny outputs and cap hits; MAE is secondary to threshold Brier score. The final midpoint changes when `--cap` changes.

## Dataset and fixed test

The pilot has **48 authored prompts: 16 development, 32 test**, with two development and four test prompts in each of eight families. Families cover facts, explanations, code, structured extraction, long-input/short-output, extended writing, open-ended requests and conversation history. English and Turkish prompts are included. Source data and split assignments are in `prompts.jsonl`; no expected lengths or model answers are embedded. Smoke uses a disjoint four-prompt file, `smoke_prompts.jsonl`.

The inputs intentionally include adversarial cases for a prompt-length heuristic. This is a small constructed dataset, not a representative sample of production traffic. Families appear on both sides of the split. Each prompt produces one sampled response; the pilot cannot establish the conditional length distribution of a single prompt or production calibration. A later frozen replication should use fresh prompts and multiple target seeds.

Three baselines are fixed before examining test outputs:

| Method | Prediction |
|---|---|
| `dev_prior` | Development bucket frequencies, with 0.5 pseudocount per bucket |
| `prompt_length` | OLS of `log1p(output_tokens)` on `log1p(input_tokens)`, fitted only on development; development residuals form a predictive distribution, clipped to the cap, with 0.5 bucket pseudocounts |
| `heuristic` | Fixed rules for requested word count, brevity, code and detail; 0.8 point mass on a predicted bucket plus 0.2 uniform smoothing |

All methods use the same midpoint convention for token-error metrics. The heuristic's smoothing is arbitrary and fixed, not calibrated. No baseline coefficients, rules or predictor prompts are tuned using test outcomes. Do not revise them after inspecting the held-out test and call the same test a new validation set.

## Metrics and falsification rule

The primary score is the average of four binary Brier scores: `mean((p_gt_t - 1[L>t])²)`. Lower is better. Reports also include bucket accuracy (argmax probability; ties choose the earlier bucket), midpoint MAE and median absolute error, all four threshold Brier scores, threshold positive/negative counts, five-bin reliability tables and ECE, and a confusion matrix with actual buckets as rows. ECE with this few samples is descriptive only.

The main comparison uses **the exact same successful test rows for every method**. Full-test baseline scores are also reported so failed forecasts cannot silently disappear. Uncapped-only scores are a sensitivity analysis with selection bias; capped-length scores remain primary.

The predeclared exploratory `pilot_signal` requires all of:

- At least 16 successful development and 32 successful test generations; no generation failures; ≥95% valid paired test coverage.
- No more than 10% of test responses hitting the cap, and at least three positive and three negative observations per threshold.
- Mean threshold Brier at least 10% lower than the strongest baseline on the paired test set; bucket accuracy at least as high as `dev_prior`.
- A paired family-cluster bootstrap (2,000 resamples) whose descriptive 95% interval for the Brier difference is entirely below zero.

If validity conditions fail, the verdict is `inconclusive`. If they pass but improvement conditions fail, it is `no_clear_pilot_signal`. The strongest comparator is selected on the test set as a conservative benchmark, not to tune a method; the eight-cluster bootstrap interval is descriptive and does not adjust for this selection. Even `pilot_signal` warrants a fresh larger replication, not a deployment claim. A lower Brier score does not by itself prove calibration.

## Cap hits, failures and cost

Responses that reach 1536 tokens without EOS are marked `capped=true`. An EOS emitted at exactly the cap is a completed response. The primary quantity remains the observed capped length; an uncapped stopping length is unknown for cap hits. Every threshold lies below the cap, so its outcome is observable. If cap hits exceed the validity gate, raise `--cap` in a new run and freeze a new experiment. The code allows caps from 1025 to 4096 and does not silently truncate overlong inputs (target input limit 4096; predictor input limit 6144 tokens).

Invalid JSON, nonfinite values, wrong keys or probabilities summing outside 1±0.02 are saved as prediction failures, with raw text. Within tolerance the vector is normalized and its original sum is saved. No repair prompts or hidden retries are used. Targets are generated even when their forecasts fail. Persisted error rows remain errors on resume; diagnose and start a fresh run if a configuration fix is necessary. GPU OOM stops before recording that row, allowing an unchanged run to resume after freeing memory.

Both prediction and target durations are synchronized GPU wall times, excluding model loading. Input/output IDs and counts allow token-cost analysis. An 8B predictor may cost more than any routing benefit: forecast quality and runtime overhead must both justify a later cheaper predictor. Do not infer speed gains from this quality pilot.

## Files and reproducibility

Each run produces:

- `manifest.json`: configuration, model Hub commit SHA, source/dataset hashes, selected IDs, package versions, GPU and CUDA details.
- `predictions.jsonl`, `predictions_frozen.json`, `generations.jsonl`: append-and-flush checkpoints, raw forecasts and exact generated IDs.
- `results.jsonl`, `results.csv`: combined conversations, predictions, responses, token counts, errors, timings and seeds.
- `scores.csv`: per-method probabilities, derived thresholds, buckets and token estimates.
- `metrics.json`, `comparison.csv`, `calibration.csv`, `confusion_*.csv`, `baseline_fit.json`: evaluation and baseline fit audit.
- `generation_configs.json`, `pip-freeze.txt`, `nvidia-smi.txt`: exact decoding configuration and runtime record.
- `evaluation.png`: optional static evaluation figure.

`main` is resolved to an immutable model SHA before loading; that SHA is reused on resume. To repeat a run, pass `--revision <SHA from manifest>` in a new output directory, retaining the same code, data, settings and runtime stack. The notebook pins the pilot to its smoke run's resolved model SHA. Per-prompt seeds are stable and reset separately for prediction/generation. Fixed seeds are not a guarantee of bit-identical results across different GPUs or numerical libraries.

Resume the identical command and directory after interruption. Changed source, data, model, seed, settings or numerical runtime versions cause refusal rather than mixed results. Model checkpoints are cached outside `/kaggle/working` to avoid consuming the saved-output disk allowance; a fresh session may download them again. The run archive contains artifacts, not model weights or the package environment.

## Script commands on Kaggle

The two supported end-to-end entry points above enforce the smoke gate. For diagnostics or individual phases, use these lower-level commands in the installed environment (calling the pilot command alone does not enforce that gate):

```sh
python experiment.py preflight --model-size 8b
python experiment.py run --mode smoke --out runs/smoke-qwen3-8b
python experiment.py run --mode pilot --out runs/pilot-qwen3-8b
python experiment.py evaluate --out runs/pilot-qwen3-8b
python plot_results.py runs/pilot-qwen3-8b
```

For the fallback use `--model-size 4b` and new paths. For custom prompts, supply `--dataset path.jsonl` with unique IDs, `split` (`dev` or `test`), `family` and text `messages`; smoke expects exactly two dev and two test rows. Keep the target system message in `messages` so the forecaster sees the actual context.

Local checks, without GPU or third-party packages:

```sh
python -m unittest discover -s tests -v
python experiment.py demo --out /tmp/qwen-length-synthetic-check
```

`demo` only checks the result/evaluation pipeline with visibly labeled synthetic fixtures. Its CSV files are not model evidence. No synthetic results are included in the deliverable bundle.

## Validation status

Locally checked: unit/integration tests for boundaries, probability validation, token/EOS/cap accounting, analytic metrics, development-only fitting, prediction/generation ordering, freeze/resume integrity, missing forecasts and generation failures; synthetic end-to-end evaluation; nbformat schema and code-cell validation; rendered and visually inspected synthetic evaluation plots. An independent read-only reviewer also checked mocked OOM recovery and the pinned chat-template API.

Not yet executed: actual model download, NF4 load and generation on Kaggle GPUs. The notebook's four-prompt smoke run is the hardware/runtime validation gate. See `CONTINUE_IN_CODEX.md` for the handoff after that run.
