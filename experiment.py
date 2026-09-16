#!/usr/bin/env python3
"""Kaggle-first output-length pilot; see README.md before interpreting results."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

from metrics import evaluate, parse_prediction, write_csv

ROOT = Path(__file__).resolve().parent
MODELS = {"8b": "Qwen/Qwen3-8B", "4b": "Qwen/Qwen3-4B"}
TARGET_SYSTEM = "You are a helpful assistant. Follow the user's requested language, format and level of detail."


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}. Preserve the file, repair the incomplete line, then resume.") from e
    return rows


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def validate_dataset(rows):
    seen = set()
    for r in rows:
        if not isinstance(r.get("id"), str) or r["id"] in seen:
            raise ValueError("Dataset IDs must be unique strings")
        seen.add(r["id"])
        if r.get("split") not in ("dev", "test") or not isinstance(r.get("family"), str):
            raise ValueError(f"Missing split/family for {r['id']}")
        messages = r.get("messages", [])
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("Each conversation must end with the user request being forecast")
        if any(m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("Only text system/user/assistant messages are supported")
    return rows


def select_rows(rows, mode):
    if mode == "smoke":
        if len(rows) != 4 or sum(r["split"] == "dev" for r in rows) != 2:
            raise ValueError("Smoke dataset must contain four separate prompts: two dev, two test")
    return rows


def runtime_preflight(model_size, device):
    import torch
    from transformers import Qwen3ForCausalLM  # Fail on package incompatibility before weight downloads.
    import bitsandbytes
    del Qwen3ForCausalLM, bitsandbytes
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. In Kaggle select Settings > Accelerator > GPU T4 x2.")
    if device < 0 or device >= torch.cuda.device_count():
        raise ValueError(f"GPU {device} does not exist; found {torch.cuda.device_count()} devices")
    gpus = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        props = torch.cuda.get_device_properties(i)
        gpus.append({"index": i, "name": props.name, "total_gib": total / 2**30,
                     "free_gib": free / 2**30, "compute_capability": list(torch.cuda.get_device_capability(i))})
    # Conservative operational gates, not measured model footprints or fit guarantees.
    needed = 10 if model_size == "8b" else 6
    if gpus[device]["compute_capability"] < [6, 0]:
        raise RuntimeError("NF4 requires CUDA compute capability >= 6.0")
    if gpus[device]["free_gib"] < needed:
        raise RuntimeError(f"Need at least {needed} GiB free for this preset. Clear stale GPU allocations; "
                           "or use --model-size 4b with a NEW output directory. No silent model switching.")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "accelerate", "bitsandbytes", "tokenizers", "huggingface-hub", "safetensors")}
    if tuple(int(x) for x in versions["torch"].split("+")[0].split(".")[:2]) < (2, 3):
        raise RuntimeError("This bitsandbytes preset needs PyTorch >= 2.3")
    info = {"python": sys.version, "platform": platform.platform(), "versions": versions,
            "cuda_runtime": torch.version.cuda, "gpus": gpus,
            "selected_device": device, "compute_dtype": "float16", "device_map": {"": device}}
    print(json.dumps(info, indent=2), flush=True)
    return info


def derive_seed(base, sample_id, phase):
    return (base + int(hashlib.sha256((phase + ":" + sample_id).encode()).hexdigest()[:8], 16)) % (2**31)


def count_output(output_ids, eos_ids, cap):
    # batch_size=1, no post-EOS batch padding. Do not decode/re-tokenize to count.
    if not output_ids or len(output_ids) > cap:
        raise ValueError("Invalid generated token length")
    stopped_eos = output_ids[-1] in eos_ids
    return {"actual_tokens": len(output_ids), "capped": len(output_ids) == cap and not stopped_eos,
            "stop_reason": "eos" if stopped_eos else "max_new_tokens" if len(output_ids) == cap else "other"}


class HFRunner:
    def __init__(self, model_id, revision, device, cap):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=False, dtype=torch.float16,
            quantization_config=quantization, device_map={"": device},
            attn_implementation="eager", low_cpu_mem_usage=True)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        if not self.eos_ids or any(x is None for x in self.eos_ids):
            raise ValueError("Target model has no EOS token")
        common = {"eos_token_id": eos, "pad_token_id": self.tokenizer.pad_token_id or self.eos_ids[0],
                  "bos_token_id": self.tokenizer.bos_token_id, "use_cache": True,
                  "num_beams": 1, "num_return_sequences": 1, "repetition_penalty": 1.0}
        self.target_config = GenerationConfig(**common, max_new_tokens=cap, do_sample=True,
                                             temperature=.7, top_p=.8, top_k=20)
        self.predictor_config = GenerationConfig(**common, max_new_tokens=192, do_sample=False)
        self.model_id, self.revision = model_id, revision
        self.cap = cap

    def encode(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False, return_tensors="pt", return_dict=True)

    def generate(self, messages, phase, seed):
        from transformers import set_seed
        set_seed(seed)
        inputs = self.encode(messages)
        input_tokens = inputs["input_ids"].shape[-1]
        limit = 6144 if phase == "predict" else 4096
        if input_tokens > limit:
            raise ValueError(f"{phase} input is {input_tokens} tokens, above the {limit} pilot limit. No truncation performed.")
        inputs = inputs.to(f"cuda:{self.device}")
        config = self.predictor_config if phase == "predict" else self.target_config
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=config)
        self.torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        ids = output[0, input_tokens:].tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return {"text": text, "output_ids": ids, "input_tokens": input_tokens, "seconds": seconds}

    def prediction_messages(self, row):
        input_tokens = self.encode(row["messages"])["input_ids"].shape[-1]
        if input_tokens > 4096:
            raise ValueError("Target input exceeds 4096 tokens; no silent truncation")
        state = {"messages": row["messages"], "input_tokens": input_tokens,
                 "target_model": self.model_id, "target_revision": self.revision,
                 "quantization": "bitsandbytes NF4 double quant, float16 compute",
                 "enable_thinking": False, "max_new_tokens": self.cap,
                 "generation": {"do_sample": True, "temperature": .7, "top_p": .8, "top_k": 20,
                                "repetition_penalty": 1, "num_beams": 1, "stop": "EOS or max_new_tokens"}}
        return [{"role": "system", "content": (ROOT / "predictor_system.txt").read_text(encoding="utf-8")},
                {"role": "user", "content": "STATE\n" + json.dumps(state, ensure_ascii=False)}], input_tokens


def unique_index(rows, selected):
    result = {}
    allowed = {r["id"] for r in selected}
    for row in rows:
        if row["id"] in result or row["id"] not in allowed:
            raise ValueError("Duplicate or unexpected result ID. Refusing mixed results.")
        result[row["id"]] = row
    return result


def join_records(selected, predictions, generations):
    rows = []
    for row in selected:
        record = dict(row)
        record.update(predictions.get(row["id"], {"prediction_status": "missing"}))
        record.update(generations.get(row["id"], {"generation_status": "missing"}))
        rows.append(record)
    return rows


def write_results(output, records):
    temp = output / "results.jsonl.tmp"
    temp.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    temp.replace(output / "results.jsonl")
    # JSONL is authoritative; CSV keeps nested fields as JSON strings for round-tripping.
    fields = sorted({k for r in records for k in r})
    write_csv(output / "results.csv", [{k: json.dumps(r[k], ensure_ascii=False) if isinstance(r.get(k), (list, dict)) else r.get(k) for k in fields} for r in records])


def run_phases(runner, selected, output, seed):
    predictions = unique_index(read_jsonl(output / "predictions.jsonl"), selected)
    generations = unique_index(read_jsonl(output / "generations.jsonl"), selected)
    freeze_path = output / "predictions_frozen.json"
    if generations and not freeze_path.exists():
        raise ValueError("Generation results exist without a prediction freeze; refusing possible leakage")
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        if len(predictions) != len(selected) or file_hash(output / "predictions.jsonl") != freeze["sha256"]:
            raise ValueError("Frozen prediction file was modified; start a new run directory")
    # Complete and persist every forecast before generating ANY target responses.
    for i, row in enumerate(selected):
        if row["id"] in predictions:
            continue
        rec = {"id": row["id"], "prediction_status": "error",
               "prediction_seed": derive_seed(seed, row["id"], "predict")}
        try:
            messages, input_tokens = runner.prediction_messages(row)
            rec["input_tokens"] = input_tokens
            rec["predictor_messages"] = messages
            generated = runner.generate(messages, "predict", rec["prediction_seed"])
            rec.update({"prediction_raw": generated["text"], "prediction_output_ids": generated["output_ids"],
                        "prediction_seconds": generated["seconds"], "prediction_input_tokens": generated["input_tokens"]})
            p, raw_sum = parse_prediction(generated["text"])
            rec.update({"prediction_status": "ok", "bucket_probs": p, "raw_probability_sum": raw_sum})
        except Exception as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError("GPU OOM: stop, clear allocations or choose 4b in a new run; no fallback was applied.") from e
            rec["prediction_error"] = f"{type(e).__name__}: {e}"
        append_jsonl(output / "predictions.jsonl", rec)
        predictions[row["id"]] = rec
        print(f"Predict {i+1}/{len(selected)} {row['id']}: {rec['prediction_status']}", flush=True)
    if not freeze_path.exists():
        atomic_json(freeze_path, {"sha256": file_hash(output / "predictions.jsonl"), "rows": len(predictions),
                                  "created_unix": time.time()})
    for i, row in enumerate(selected):
        if row["id"] in generations:
            continue
        rec = {"id": row["id"], "generation_status": "error",
               "generation_seed": derive_seed(seed, row["id"], "generate")}
        try:
            generated = runner.generate(row["messages"], "generate", rec["generation_seed"])
            rec.update(count_output(generated["output_ids"], runner.eos_ids, runner.cap))
            rec.update({"generation_status": "ok", "input_tokens": generated["input_tokens"],
                        "response": generated["text"], "generated_ids": generated["output_ids"],
                        "generation_seconds": generated["seconds"]})
        except Exception as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError("GPU OOM: predictions and prior responses are saved. Resume unchanged or start a new model preset run.") from e
            rec["generation_error"] = f"{type(e).__name__}: {e}"
        append_jsonl(output / "generations.jsonl", rec)
        generations[row["id"]] = rec
        print(f"Generate {i+1}/{len(selected)} {row['id']}: {rec['generation_status']} {rec.get('actual_tokens', '')}", flush=True)
    records = join_records(selected, predictions, generations)
    write_results(output, records)
    return records


def build_manifest(args, selected):
    files = {name: file_hash(ROOT / name) for name in ("experiment.py", "metrics.py", "predictor_system.txt", "requirements-kaggle.txt")}
    spec = {"model_id": MODELS[args.model_size], "requested_revision": args.revision,
            "model_size": args.model_size, "mode": args.mode, "seed": args.seed,
            "device": args.device, "cap": args.cap, "enable_thinking": False,
            "quantization": "NF4 double quant", "compute_dtype": "float16",
            "attention": "eager", "device_map": {"": args.device},
            "source_hashes": files, "dataset_sha256": file_hash(args.dataset),
            "selected_sha256": digest(selected), "selected_ids": [r["id"] for r in selected]}
    return {"fingerprint": digest(spec), "spec": spec, "created_unix": time.time()}


def start_run(args):
    selected = select_rows(validate_dataset(read_jsonl(args.dataset)), args.mode)
    if not selected:
        raise ValueError("Empty dataset")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    requested = build_manifest(args, selected)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else requested
    if manifest["fingerprint"] != requested["fingerprint"]:
        raise ValueError("Run configuration, dataset or source changed. Use a NEW --out directory.")
    if not manifest_path.exists() and any(output.iterdir()):
        raise ValueError("Output directory contains files but no manifest. Choose an empty directory.")
    atomic_json(manifest_path, manifest)
    atomic_json(output / "selected_prompts.json", selected)
    info = runtime_preflight(args.model_size, args.device)
    if "environment" in manifest and manifest["environment"] != info:
        # Free VRAM changes naturally; compare the fields affecting numerical reproducibility.
        old = manifest["environment"]
        comparable = lambda x: {k: x[k] for k in ("versions", "cuda_runtime", "compute_dtype", "device_map")}
        old_hardware = [(g["name"], g["compute_capability"]) for g in old["gpus"]]
        new_hardware = [(g["name"], g["compute_capability"]) for g in info["gpus"]]
        if comparable(old) != comparable(info) or old_hardware != new_hardware:
            raise ValueError("Runtime versions/hardware changed. Use a new output directory to avoid mixed results.")
    manifest.setdefault("environment", info)
    if "resolved_revision" not in manifest:
        from huggingface_hub import HfApi
        manifest["resolved_revision"] = HfApi().model_info(MODELS[args.model_size], revision=args.revision).sha
    atomic_json(manifest_path, manifest)
    (output / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        smi = str(e)
    (output / "nvidia-smi.txt").write_text(smi, encoding="utf-8")
    runner = HFRunner(MODELS[args.model_size], manifest["resolved_revision"], args.device, args.cap)
    atomic_json(output / "generation_configs.json", {"target": runner.target_config.to_dict(),
                                                     "predictor": runner.predictor_config.to_dict(),
                                                     "chat_template_sha256": digest(runner.tokenizer.chat_template),
                                                     "eos_ids": runner.eos_ids})
    records = run_phases(runner, selected, output, args.seed)
    report = evaluate(records, args.cap, output, args.mode)
    print(json.dumps({"out": str(output), "verdict": report["verdict"], "counts": report["counts"], "reasons": report["reasons"]}, indent=2))


def demo(args):
    """Synthetic plumbing fixture ONLY; never claims a Qwen experiment was run."""
    selected = validate_dataset(read_jsonl(args.dataset))
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo needs an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = random.Random(args.seed)
    for i, row in enumerate(selected):
        p = [rng.random() for _ in range(5)]
        p = [v / sum(p) for v in p]
        rows.append({**row, "synthetic": True, "input_tokens": 30 + i * 9,
                     "actual_tokens": [24, 150, 360, 700, 1200, args.cap][i % 6],
                     "capped": i % 6 == 5, "prediction_status": "ok", "generation_status": "ok",
                     "bucket_probs": p, "response": "SYNTHETIC FIXTURE: no model response generated."})
    atomic_json(output / "manifest.json", {"synthetic": True, "mode": "demo", "cap": args.cap, "seed": args.seed})
    write_results(output, rows)
    report = evaluate(rows, args.cap, output, "demo")
    print("SYNTHETIC DEMO ONLY: " + json.dumps(report["counts"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "evaluate", "preflight", "demo"])
    parser.add_argument("--mode", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--model-size", choices=MODELS, default="8b")
    parser.add_argument("--revision", default="main", help="Resolved to immutable Hub SHA before loading; pinned in manifest")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cap", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", default="runs/smoke-qwen3-8b")
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = ROOT / ("smoke_prompts.jsonl" if args.mode == "smoke" and args.command == "run" else "prompts.jsonl")
    if not 1024 < args.cap <= 4096:
        parser.error("--cap must be 1025..4096 to keep every threshold observable and bound GPU memory")
    if args.command == "run":
        start_run(args)
    elif args.command == "demo":
        demo(args)
    elif args.command == "preflight":
        runtime_preflight(args.model_size, args.device)
    else:
        output = Path(args.out)
        manifest = json.loads((output / "manifest.json").read_text())
        spec = manifest.get("spec", manifest)
        report = evaluate(read_jsonl(output / "results.jsonl"), spec["cap"], output, spec["mode"])
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
