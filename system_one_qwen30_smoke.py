#!/usr/bin/env python3
"""Four-prompt Qwen30 System One capability smoke; run in the control environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx2 as httpx
import msgspec
import openai
from system_one_adapter import Choice, SystemOneAdapterClient
from system_one_adapter.providers.base import record_request, render_messages, translating
from system_one_adapter.providers.openai import OpenAIProvider, _response_format, _result


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CHECKPOINT = "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
REVISION = "84455dc68ee6574b9f43659e32588d0bf0ca3a21"
ADAPTER_COMMIT = "0bb819b85d67a98c736d7c3004eae95f49f3daa3"
EXPECTED_IDS = ("smoke_fact", "smoke_explain", "smoke_extract", "smoke_extended")
BUCKETS = ("B1", "B2", "B3", "B4", "B5")
MAX_MODEL_LEN = 3072
PREDICTION_CAP = 192
TARGET_CAP = 1536
BASE_SEED = 20260916
VLLM_PYTHON = Path("/kaggle/temp/qwen30-vllm-env/bin/python")
VLLM_COMMAND = Path("/kaggle/temp/qwen30-vllm-env/bin/vllm")

CHOICE_INSTRUCTIONS = (
    "Predict the number of generated completion token IDs that the target model "
    "will produce for its next response under the provided state and generation "
    "configuration. Use only the pre-generation state; do not answer or simulate "
    "the target request. Count all new completion token IDs, including a final "
    "end-of-sequence token if emitted. Exclude input tokens and the chat-template "
    "assistant prefix. Predict length capped at max_new_tokens; account for "
    "sampling uncertainty. A long input does not imply a long output."
)
CHOICE_CRITERIA = {
    "B1": "1 through 128 generated completion token IDs",
    "B2": "129 through 256 generated completion token IDs",
    "B3": "257 through 512 generated completion token IDs",
    "B4": "513 through 1024 generated completion token IDs",
    "B5": "1025 through 1536 generated completion token IDs",
}
CHOICE = {"length_bucket": Choice(instructions=CHOICE_INSTRUCTIONS, criteria=CHOICE_CRITERIA)}
ADAPTER_CONFIG = {
    "structured_outputs": True,
    "llm_answer_mode": "probabilities",
    "normalize_probabilities": True,
    "n_retry_malformed_structure": 0,
    "transient_retries": 0,
    "provider": "OpenAIProvider subclass with fixed prediction sampling",
    "prediction_sampling": {
        "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        "repetition_penalty": 1.0, "max_tokens": PREDICTION_CAP,
        "seed": 0, "enable_thinking": False, "stop_token_ids": [151643],
    },
}
TARGET_SAMPLING = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
    "repetition_penalty": 1.0, "max_tokens": TARGET_CAP, "ignore_eos": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "stop_token_ids": [151643], "return_token_ids": True,
    "n": 1, "stream": False,
}
HISTORICAL_MANUAL = {
    "smoke_fact": [0.95, 0.04, 0.01, 0.00, 0.00],
    "smoke_explain": [0.15, 0.45, 0.30, 0.08, 0.02],
    "smoke_extract": [0.05, 0.15, 0.30, 0.40, 0.10],
    "smoke_extended": [0.05, 0.25, 0.55, 0.12, 0.03],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_json(path: Path, value: Any) -> bytes:
    data = json_bytes(value)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return data


def exclusive_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


class RequestJournal:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("x", encoding="utf-8")
        self.phase = "preflight"
        self.row_id: str | None = None

    def record(self, request: httpx.Request) -> None:
        body = request.content.decode("utf-8")
        entry = {
            "at": now(), "phase": self.phase, "id": self.row_id,
            "method": request.method, "url": str(request.url),
            "raw_body": body, "payload": json.loads(body) if body else None,
        }
        self.handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


class FixedSamplingProvider(OpenAIProvider):
    """Keep the adapter's provider behavior; set and journal prediction sampling."""

    def __init__(self, base_url: str, journal: RequestJournal) -> None:
        super().__init__(MODEL, base_url=base_url, api_key="EMPTY", api="chat_completions")
        self._client.close()
        transport = httpx.Client(event_hooks={"request": [journal.record]})
        self._client = openai.OpenAI(
            base_url=base_url, api_key="EMPTY", max_retries=0,
            timeout=300.0, http_client=transport,
        )

    def request(self, messages, *, schema, structured):
        kwargs = {
            "model": self.model_name,
            "messages": render_messages(messages),
            "response_format": _response_format(schema, structured=structured),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": PREDICTION_CAP,
            "seed": 0,
            "extra_body": {
                "top_k": 0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
                "stop_token_ids": [151643],
            },
        }
        with translating(self.translate_error):
            record_request(kwargs, api=self.api)
            response = self._client.chat.completions.create(**kwargs)
        return _result(response)


def load_prompts() -> tuple[list[dict[str, Any]], str]:
    path = ROOT / "smoke_prompts.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if tuple(row.get("id") for row in rows) != EXPECTED_IDS:
        raise ValueError("smoke_prompts.jsonl must contain exactly the original four smoke IDs in order")
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(item, dict)
            or item.get("role") not in ("system", "user", "assistant")
            or not isinstance(item.get("content"), str)
            for item in messages
        ):
            raise ValueError(f"Invalid messages for {row['id']}")
    return rows, hashlib.sha256(raw).hexdigest()


def validate_environment() -> dict[str, Any]:
    versions = {name: importlib.metadata.version(name) for name in
                ("system-one-adapter", "openai", "httpx2")}
    if versions["system-one-adapter"] != "0.1.4":
        raise RuntimeError(f"Expected System One Adapter 0.1.4; got {versions['system-one-adapter']}")
    if versions["openai"] != "3.14.1" or versions["httpx2"] != "2.13.0":
        raise RuntimeError(f"Unexpected control HTTP package versions: {versions}")
    if not VLLM_PYTHON.is_file() or not VLLM_COMMAND.is_file():
        raise RuntimeError("Missing isolated vLLM environment; run scripts/setup_system_one_kaggle.sh")
    check = subprocess.run(
        [str(VLLM_PYTHON), "-c", (
            "import importlib.metadata as m, json, torch; "
            "print(json.dumps({'vllm': m.version('vllm'), 'torch': torch.__version__, "
            "'cuda': torch.version.cuda, 'device_count': torch.cuda.device_count(), "
            "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
        )], capture_output=True, text=True, timeout=120, check=True,
    )
    runtime = json.loads(check.stdout.strip())
    if not (runtime["vllm"].split("+", 1)[0] == "0.29.0"
            and runtime["torch"].startswith("2.13.0+cu129")
            and runtime["cuda"] == "12.9"
            and runtime["device_count"] == 2
            and all("T4" in name for name in runtime["devices"])):
        raise RuntimeError(f"Runtime differs from the tested Kaggle T4x2 setup: {runtime}")
    return {"control": versions, "server": runtime}


def state_for(row: dict[str, Any], input_tokens: int) -> dict[str, Any]:
    return {
        "messages": row["messages"], "input_tokens": input_tokens,
        "target_model": MODEL, "target_revision": REVISION,
        "quantization": "AWQ 4-bit runtime checkpoint",
        "enable_thinking": False, "max_new_tokens": TARGET_CAP,
        "generation": {
            "do_sample": True, "temperature": 0.7, "top_p": 0.8,
            "top_k": 20, "repetition_penalty": 1,
            "num_beams": 1, "stop": "EOS or max_new_tokens",
            "seed": derive_seed(BASE_SEED, row["id"], "generate"),
        },
    }


def derive_seed(base: int, sample_id: str, phase: str) -> int:
    return (base + int(hashlib.sha256((phase + ":" + sample_id).encode()).hexdigest()[:8], 16)) % (2**31)


def send_json(client: httpx.Client, url: str, payload: dict[str, Any],
              journal: RequestJournal) -> dict[str, Any]:
    request = client.build_request("POST", url, json=payload)
    journal.record(request)
    response = client.send(request)
    response.raise_for_status()
    return response.json()


def tokenize(client: httpx.Client, base_url: str, messages: list[dict[str, str]],
             journal: RequestJournal) -> int:
    payload = {
        "model": MODEL, "messages": messages, "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    reply = send_json(client, base_url + "/tokenize", payload, journal)
    tokens = reply.get("tokens")
    count = reply.get("count")
    if not isinstance(tokens, list) or not isinstance(count, int) or count != len(tokens):
        raise ValueError(f"Invalid /tokenize response: {reply}")
    return count


def wait_for_server(client: httpx.Client, base_url: str, process: subprocess.Popen,
                    log_path: Path, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            raise RuntimeError(f"vLLM exited {process.returncode} during startup:\n{tail}")
        try:
            health = client.get(base_url + "/health", timeout=5)
            if health.status_code == 200:
                models = client.get(base_url + "/v1/models", timeout=10)
                models.raise_for_status()
                listing = models.json()
                if MODEL in [item.get("id") for item in listing.get("data", [])]:
                    return listing
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(3)
    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
    raise TimeoutError(f"vLLM did not become ready within {timeout_seconds}s:\n{tail}")


def preflight(client: httpx.Client, base_url: str, rows: list[dict[str, Any]],
              adapter: SystemOneAdapterClient, journal: RequestJournal) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        journal.row_id = row["id"]
        target_tokens = tokenize(client, base_url, row["messages"], journal)
        if target_tokens + TARGET_CAP > MAX_MODEL_LEN:
            raise ValueError(f"{row['id']} target request exceeds {MAX_MODEL_LEN} tokens")
        state = state_for(row, target_tokens)
        # The pinned adapter prepares these same messages in system_one(). This
        # read-only check catches oversized predictor requests before Phase A.
        run = adapter._prepare_evaluation(state, CHOICE, MODEL)
        predictor_messages = render_messages(run.base_messages)
        predictor_tokens = tokenize(client, base_url, predictor_messages, journal)
        if predictor_tokens + PREDICTION_CAP > MAX_MODEL_LEN:
            raise ValueError(f"{row['id']} adapter request exceeds {MAX_MODEL_LEN} tokens")
        prepared.append({"row": row, "state": state,
                         "target_input_tokens": target_tokens,
                         "prediction_input_tokens": predictor_tokens})
    journal.row_id = None
    return prepared


def predict(prepared: list[dict[str, Any]], adapter: SystemOneAdapterClient,
            journal: RequestJournal) -> list[dict[str, Any]]:
    results = []
    journal.phase = "prediction"
    for item in prepared:
        row = item["row"]
        journal.row_id = row["id"]
        record: dict[str, Any] = {
            "id": row["id"], "state": item["state"],
            "choice": {"question_id": "length_bucket",
                       "instructions": CHOICE_INSTRUCTIONS, "criteria": CHOICE_CRITERIA},
            "adapter_config": ADAPTER_CONFIG,
            "prediction_input_tokens_preflight": item["prediction_input_tokens"],
        }
        started = time.perf_counter()
        try:
            response = adapter.system_one(item["state"], CHOICE)
            built = msgspec.to_builtins(response)
            debug = built["debug"]
            normalized = built["answers"]["length_bucket"]["probabilities"]
            raw = debug.get("original_probabilities", {}).get("length_bucket", normalized)
            if set(normalized) != set(BUCKETS) or any(
                not isinstance(normalized[b], (int, float)) or not math.isfinite(normalized[b])
                for b in BUCKETS
            ) or abs(sum(normalized.values()) - 1.0) > 1e-6:
                raise ValueError(f"Invalid normalized distribution: {normalized}")
            attempts = debug.get("llm_attempts", [])
            if not attempts:
                raise ValueError("Adapter returned no provider attempt trace")
            usage = built["usage"]
            if usage["input_tokens"] != item["prediction_input_tokens"]:
                raise ValueError("Prediction usage disagrees with /tokenize preflight")
            record.update({
                "status": "ok", "raw_probabilities": raw,
                "normalized_probabilities": normalized,
                "argmax": built["answers"]["length_bucket"]["choice"],
                "adapter_response": built,
                "raw_model_response": attempts[-1].get("llm_response"),
                "parsed_model_json": json.loads(attempts[-1]["llm_response"]["choices"][0]["message"]["content"]),
                "threshold_probabilities": {
                    "P(L>128)": sum(normalized[b] for b in BUCKETS[1:]),
                    "P(L>256)": sum(normalized[b] for b in BUCKETS[2:]),
                    "P(L>512)": sum(normalized[b] for b in BUCKETS[3:]),
                    "P(L>1024)": normalized["B5"],
                },
                "prediction_latency_seconds": time.perf_counter() - started,
            })
        except Exception as error:
            record.update({
                "status": "error", "error_type": type(error).__name__,
                "error": str(error), "adapter_debug": getattr(error, "debug", None),
                "prediction_latency_seconds": time.perf_counter() - started,
            })
        results.append(record)
    journal.row_id = None
    return results


def bucket_for(length: int) -> str:
    for label, maximum in zip(BUCKETS, (128, 256, 512, 1024, 1536)):
        if length <= maximum:
            return label
    raise ValueError(f"Completion length above cap: {length}")


def generate(client: httpx.Client, base_url: str, prepared: list[dict[str, Any]],
             journal: RequestJournal) -> list[dict[str, Any]]:
    results = []
    journal.phase = "target_generation"
    for item in prepared:
        row = item["row"]
        journal.row_id = row["id"]
        payload = {"model": MODEL, "messages": row["messages"], **TARGET_SAMPLING,
                   "seed": derive_seed(BASE_SEED, row["id"], "generate")}
        record: dict[str, Any] = {"id": row["id"], "generation_request": payload}
        started = time.perf_counter()
        try:
            response = send_json(client, base_url + "/v1/chat/completions", payload, journal)
            choice = response["choices"][0]
            token_ids = choice.get("token_ids")
            usage = response["usage"]
            if not isinstance(token_ids, list) or not token_ids or not all(isinstance(t, int) for t in token_ids):
                raise ValueError("vLLM did not return nonempty completion token IDs")
            if (len(token_ids) > TARGET_CAP
                    or usage["prompt_tokens"] != item["target_input_tokens"]
                    or usage["completion_tokens"] != len(token_ids)):
                raise ValueError("Token counts disagree with cap or /tokenize preflight")
            record.update({
                "status": "ok", "generated_token_ids": token_ids,
                "generated_text": choice["message"].get("content") or "",
                "actual_tokens": len(token_ids),
                "actual_bucket": bucket_for(len(token_ids)),
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "natural_length_interpretation": "Lout >= 1536"
                    if len(token_ids) == TARGET_CAP and choice.get("finish_reason") == "length"
                    else None,
                "usage": usage, "raw_response": response,
                "generation_latency_seconds": time.perf_counter() - started,
            })
        except Exception as error:
            record.update({"status": "error", "error_type": type(error).__name__,
                           "error": str(error),
                           "generation_latency_seconds": time.perf_counter() - started})
        results.append(record)
    journal.row_id = None
    return results


def print_summary(predictions: list[dict[str, Any]], generations: list[dict[str, Any]]) -> None:
    print("\nid              B1     B2     B3     B4     B5   argmax  actual  bucket  finish  pred_s")
    for prediction, generation in zip(predictions, generations):
        probs = prediction.get("normalized_probabilities", {})
        values = " ".join(f"{probs.get(b, float('nan')):5.2f}" for b in BUCKETS)
        print(f"{prediction['id']:<15} {values}  {prediction.get('argmax', '-'):>3}"
              f" {str(generation.get('actual_tokens', '-')):>7}"
              f" {generation.get('actual_bucket', '-'):>6}"
              f" {str(generation.get('finish_reason', '-')):>7}"
              f" {prediction['prediction_latency_seconds']:7.1f}")
        manual = HISTORICAL_MANUAL[prediction["id"]]
        print("  old manual (non-parity, no saved code): " + " ".join(f"{value:.2f}" for value in manual))
        if prediction["id"] in ("smoke_extract", "smoke_extended") and probs:
            tails = prediction["threshold_probabilities"]
            print("  critical tails: " + ", ".join(f"{key}={value:.3f}" for key, value in tails.items()))
    if all(p["status"] == "ok" and g["status"] == "ok" for p, g in zip(predictions, generations)):
        extract = predictions[2]["threshold_probabilities"]["P(L>512)"]
        extended = predictions[3]["threshold_probabilities"]
        print(f"\nBoundary readout: extraction P(L>512)={extract:.3f}; "
              f"tutorial P(L>512)={extended['P(L>512)']:.3f}, "
              f"P(L>1024)={extended['P(L>1024)']:.3f}.")
        print("Four-prompt falsification only. Review boundary behavior before any larger pilot.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/system-one-qwen30-smoke")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    args = parser.parse_args()
    rows, prompts_hash = load_prompts()
    versions = validate_environment()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    journal = RequestJournal(out / "requests.jsonl")
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(VLLM_COMMAND), "serve", CHECKPOINT,
        "--revision", REVISION, "--tokenizer-revision", REVISION,
        "--served-model-name", MODEL, "--tensor-parallel-size", "2",
        "--dtype", "float16", "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", "1", "--gpu-memory-utilization", "0.92",
        "--enforce-eager", "--host", "127.0.0.1", "--port", str(args.port),
    ]
    metadata = {
        "started_at": now(), "model": MODEL, "checkpoint": CHECKPOINT,
        "checkpoint_revision": REVISION, "adapter_commit": ADAPTER_COMMIT,
        "prompts_sha256": prompts_hash, "versions": versions,
        "server_command": command, "server_environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "adapter_config": ADAPTER_CONFIG, "target_sampling": TARGET_SAMPLING,
        "historical_manual_nonparity": HISTORICAL_MANUAL,
        "ordering": "all four predictions and predictions.sha256 before first target generation",
    }
    atomic_json(out / "metadata.json", metadata)
    process = None
    provider = None
    try:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError(f"Port {args.port} is already in use")
        environment = os.environ.copy()
        environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        log_path = out / "server.log"
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
        with httpx.Client(timeout=900) as http:
            metadata["served_models"] = wait_for_server(http, base_url, process, log_path,
                                                        args.startup_timeout)
            provider = FixedSamplingProvider(base_url + "/v1", journal)
            adapter = SystemOneAdapterClient(
                structured_outputs=True, llm_answer_mode="probabilities",
                normalize_probabilities=True, n_retry_malformed_structure=0,
                model=provider,
            )
            prepared = preflight(http, base_url, rows, adapter, journal)
            metadata["token_preflight"] = [
                {"id": item["row"]["id"],
                 "target_input_tokens": item["target_input_tokens"],
                 "prediction_input_tokens": item["prediction_input_tokens"]}
                for item in prepared
            ]
            atomic_json(out / "metadata.json", metadata)
            print("Preflight passed. Predicting all four prompts with the real adapter...", flush=True)
            predictions = predict(prepared, adapter, journal)
            frozen = atomic_json(out / "predictions.json", predictions)
            digest = hashlib.sha256(frozen).hexdigest()
            exclusive_text(out / "predictions.sha256", f"{digest}  predictions.json\n")
            print(f"Predictions frozen: SHA256 {digest}", flush=True)
            generations = generate(http, base_url, prepared, journal)
            atomic_json(out / "results.json", {
                "prediction_sha256": digest, "predictions": predictions,
                "generations": generations,
            })
            metadata["completed_at"] = now()
            metadata["predictions_sha256"] = digest
            metadata["prediction_errors"] = sum(p["status"] != "ok" for p in predictions)
            metadata["generation_errors"] = sum(g["status"] != "ok" for g in generations)
            atomic_json(out / "metadata.json", metadata)
            print_summary(predictions, generations)
            return 0 if all(p["status"] == "ok" for p in predictions) and all(
                g["status"] == "ok" for g in generations
            ) else 1
    finally:
        try:
            if provider is not None:
                provider.close()
        finally:
            try:
                journal.close()
            finally:
                if process is not None and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Smoke run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
