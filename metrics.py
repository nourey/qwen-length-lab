"""Dependency-free evaluation. No model or GPU is needed to re-score a run."""
import csv
import json
import math
import random
import re
import statistics
from pathlib import Path

THRESHOLDS = [128, 256, 512, 1024]
LABELS = ["1-128", "129-256", "257-512", "513-1024", "1025-cap"]


def bucket(length):
    return sum(length > t for t in THRESHOLDS)


def midpoints(cap):
    return [64.5, 192.5, 384.5, 768.5, (1025 + cap) / 2]


def expected_tokens(probs, cap):
    # Approximation: uniform length conditional on each bucket, including the last.
    return sum(p * m for p, m in zip(probs, midpoints(cap)))


def tails(probs):
    return [sum(probs[i + 1:]) for i in range(4)]


def parse_prediction(raw):
    """Strict JSON: invalid responses remain failures, never silently repaired."""
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result
    value = json.loads(raw.strip(), object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != {"bucket_probs"}:
        raise ValueError("expected exactly one key: bucket_probs")
    p = value["bucket_probs"]
    if not isinstance(p, list) or len(p) != 5:
        raise ValueError("expected five bucket probabilities")
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in p):
        raise ValueError("probabilities must be finite numbers in [0,1]")
    total = sum(p)
    if abs(total - 1) > 0.02:
        raise ValueError("probabilities must sum to 1 within 0.02")
    return [x / total for x in p], total


def mean(xs):
    return statistics.mean(xs) if xs else None


def distribution(lengths, alpha=0.5):
    counts = [alpha] * 5
    for length in lengths:
        counts[bucket(length)] += 1
    return [c / sum(counts) for c in counts]


def heuristic(messages, cap):
    text = next(m["content"].lower() for m in reversed(messages) if m["role"] == "user")
    words = re.search(r"\b(\d+)\s*[- ]?\s*(?:words?|kelime|sözcük)\b", text)
    if words:
        point = int(words.group(1)) * (2 if re.search(r"kelime|sözcük", words.group()) else 1.4)
    elif any(s in text for s in ("one word", "single word", "yes or no", "tek kelime", "only the", "yalnızca")):
        point = 16
    elif any(s in text for s in ("one sentence", "single sentence", "tek cümle")):
        point = 48
    elif any(s in text for s in ("complete implementation", "comprehensive", "detailed", "ayrıntılı", "kapsamlı")):
        point = 1000
    elif any(s in text for s in ("code", "python", "sql", "function", "kod")):
        point = 384
    elif any(s in text for s in ("brief", "short", "kısaca", "kısa")):
        point = 128
    else:
        point = 256
    index = bucket(min(cap, max(1, point)))
    # Fixed smoothing, set before seeing outputs. Not a calibrated probability.
    return [0.04 + (0.8 if i == index else 0) for i in range(5)]


def fit_baselines(dev, cap):
    if len(dev) < 2:
        raise ValueError("need at least two successful development responses")
    x = [math.log1p(r["input_tokens"]) for r in dev]
    y = [math.log1p(r["actual_tokens"]) for r in dev]
    xm, ym = mean(x), mean(y)
    var = sum((a - xm) ** 2 for a in x)
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / var if var > 1e-12 else 0
    intercept = ym - slope * xm
    residuals = [b - (intercept + slope * a) for a, b in zip(x, y)]
    return {"prior": distribution([r["actual_tokens"] for r in dev]),
            "intercept": intercept, "slope": slope, "residuals": residuals,
            "development_ids": [r["id"] for r in dev], "cap": cap}


def baseline_predictions(row, fitted):
    log_point = fitted["intercept"] + fitted["slope"] * math.log1p(row["input_tokens"])
    lengths = [min(fitted["cap"], max(1, math.expm1(min(30, log_point + e))))
               for e in fitted["residuals"]]
    return {"dev_prior": fitted["prior"],
            "prompt_length": distribution(lengths),
            "heuristic": heuristic(row["messages"], fitted["cap"])}


def row_brier(probs, length):
    return mean([(p - int(length > t)) ** 2 for p, t in zip(tails(probs), THRESHOLDS)])


def score(rows, method, cap):
    if not rows:
        return {"n": 0}
    matrix = [[0] * 5 for _ in range(5)]
    errors, threshold_scores = [], []
    for row in rows:
        p = row["methods"][method]
        predicted = max(range(5), key=lambda i: p[i])
        matrix[bucket(row["actual_tokens"])][predicted] += 1
        errors.append(abs(expected_tokens(p, cap) - row["actual_tokens"]))
    for i, t in enumerate(THRESHOLDS):
        probs = [tails(r["methods"][method])[i] for r in rows]
        outcomes = [int(r["actual_tokens"] > t) for r in rows]
        reliability = []
        for j in range(5):
            pairs = [(p, y) for p, y in zip(probs, outcomes) if min(4, int(p * 5)) == j]
            reliability.append({"lower": j / 5, "upper": (j + 1) / 5,
                                "n": len(pairs), "mean_probability": mean([p for p, _ in pairs]),
                                "observed_frequency": mean([y for _, y in pairs])})
        threshold_scores.append({"threshold": t, "positives": sum(outcomes),
                                 "negatives": len(rows) - sum(outcomes),
                                 "brier": mean([(p - y) ** 2 for p, y in zip(probs, outcomes)]),
                                 "ece_5_bins": sum(b["n"] / len(rows) * abs(b["mean_probability"] - b["observed_frequency"])
                                                   for b in reliability if b["n"]),
                                 "reliability": reliability})
    return {"n": len(rows), "bucket_accuracy": sum(matrix[i][i] for i in range(5)) / len(rows),
            "mae_midpoint_tokens": mean(errors), "median_ae_midpoint_tokens": statistics.median(errors),
            "macro_threshold_brier": mean([s["brier"] for s in threshold_scores]),
            "confusion_matrix_actual_rows_predicted_columns": matrix, "thresholds": threshold_scores}


def bootstrap_delta(rows, baseline, seed=20260916, iterations=2000):
    """Paired, family-cluster bootstrap; descriptive uncertainty for this pilot."""
    groups = {}
    for row in rows:
        groups.setdefault(row["family"], []).append(
            row_brier(row["methods"]["llm"], row["actual_tokens"]) -
            row_brier(row["methods"][baseline], row["actual_tokens"]))
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    keys = list(groups)
    samples = []
    for _ in range(iterations):
        values = [v for key in rng.choices(keys, k=len(keys)) for v in groups[key]]
        samples.append(mean(values))
    samples.sort()
    return {"mean_delta_llm_minus_baseline": mean([v for g in groups.values() for v in g]),
            "percentile_95_interval": [samples[int(iterations * .025)], samples[int(iterations * .975)]],
            "clusters": len(groups), "iterations": iterations}


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(records, cap, output, mode):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    successful = [r for r in records if r.get("generation_status") == "ok"]
    dev = [r for r in successful if r["split"] == "dev"]
    test = [r for r in successful if r["split"] == "test"]
    paired = [r for r in test if r.get("prediction_status") == "ok"]
    report = {"mode": mode, "synthetic": mode == "demo", "cap": cap, "bucket_labels": LABELS,
              "counts": {"planned": len(records), "generation_successes": len(successful),
                         "dev": len(dev), "test": len(test), "paired_test": len(paired),
                         "prediction_failures": sum(r.get("prediction_status") != "ok" for r in records),
                         "generation_failures": len(records) - len(successful),
                         "cap_hits_test": sum(r["capped"] for r in test)},
              "verdict": "inconclusive", "reasons": []}
    if len(dev) < 2:
        report["reasons"].append("Fewer than two successful development rows; cannot fit baselines.")
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    fitted = fit_baselines(dev, cap)
    for row in successful:
        row["methods"] = baseline_predictions(row, fitted)
        if row.get("prediction_status") == "ok":
            row["methods"]["llm"] = row["bucket_probs"]
    methods = ["llm", "dev_prior", "prompt_length", "heuristic"]
    report["paired_test"] = {m: score(paired, m, cap) for m in methods}
    report["baseline_all_test"] = {m: score(test, m, cap) for m in methods[1:]}
    report["uncapped_paired_test_sensitivity"] = {m: score([r for r in paired if not r["capped"]], m, cap) for m in methods}
    report["latency_seconds"] = {
        "prediction_median": mean_or_median(records, "prediction_seconds"),
        "generation_median": mean_or_median(successful, "generation_seconds"),
        "prediction_total": sum(r.get("prediction_seconds", 0) for r in records),
        "generation_total": sum(r.get("generation_seconds", 0) for r in successful)}
    # Guard against complete-case success hiding invalid JSON / failed generations.
    planned_test = [r for r in records if r["split"] == "test"]
    report["test_prediction_coverage"] = len(paired) / len(planned_test) if planned_test else 0
    if mode != "pilot":
        report["reasons"].append("Only a complete real pilot can produce a pilot verdict.")
    if len(dev) < 16 or len(test) < 32:
        report["reasons"].append("Need at least 16 dev and 32 test generations.")
    if report["counts"]["generation_failures"]:
        report["reasons"].append("Some target generations failed.")
    if report["test_prediction_coverage"] < .95:
        report["reasons"].append("Valid paired test coverage is below 95%.")
    if test and sum(r["capped"] for r in test) / len(test) > .10:
        report["reasons"].append("More than 10% of test responses hit the cap; increase cap in a new run.")
    if paired:
        best = min(methods[1:], key=lambda m: report["paired_test"][m]["macro_threshold_brier"])
        report["strongest_test_baseline"] = best
        report["paired_brier_delta_bootstrap"] = bootstrap_delta(paired, best)
        if any(min(t["positives"], t["negatives"]) < 3 for t in report["paired_test"]["llm"]["thresholds"]):
            report["reasons"].append("At least one threshold has fewer than three positives or negatives.")
        if not report["reasons"]:
            llm = report["paired_test"]["llm"]
            baseline = report["paired_test"][best]
            improvement = llm["macro_threshold_brier"] <= .9 * baseline["macro_threshold_brier"]
            accuracy = llm["bucket_accuracy"] >= report["paired_test"]["dev_prior"]["bucket_accuracy"]
            ci = report["paired_brier_delta_bootstrap"]
            report["verdict"] = "pilot_signal" if improvement and accuracy and ci and ci["percentile_95_interval"][1] < 0 else "no_clear_pilot_signal"
    else:
        report["reasons"].append("No valid paired test predictions.")
    (output / "baseline_fit.json").write_text(json.dumps(fitted, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    flat = []
    for r in successful:
        for method, p in r["methods"].items():
            flat.append({"id": r["id"], "split": r["split"], "family": r["family"], "method": method,
                         "input_tokens": r["input_tokens"], "actual_tokens": r["actual_tokens"],
                         "capped": r["capped"], "actual_bucket": LABELS[bucket(r["actual_tokens"])],
                         "predicted_bucket": LABELS[max(range(5), key=lambda i: p[i])],
                         "expected_tokens_midpoint": expected_tokens(p, cap),
                         **{f"p_bucket_{i}": v for i, v in enumerate(p)},
                         **{f"p_gt_{t}": v for t, v in zip(THRESHOLDS, tails(p))}})
    write_csv(output / "scores.csv", flat)
    comparison = []
    calibration = []
    for method, s in report["paired_test"].items():
        if not s["n"]:
            continue
        comparison.append({"method": method, **{k: s[k] for k in ("n", "bucket_accuracy", "mae_midpoint_tokens", "median_ae_midpoint_tokens", "macro_threshold_brier")}})
        write_csv(output / f"confusion_{method}.csv", [{"actual_bucket": LABELS[i], **dict(zip(LABELS, row))}
                  for i, row in enumerate(s["confusion_matrix_actual_rows_predicted_columns"])])
        for t in s["thresholds"]:
            calibration.extend({"method": method, "threshold": t["threshold"], **b} for b in t["reliability"])
    write_csv(output / "comparison.csv", comparison)
    write_csv(output / "calibration.csv", calibration)
    return report


def mean_or_median(rows, key):
    vals = [r[key] for r in rows if key in r]
    return statistics.median(vals) if vals else None
