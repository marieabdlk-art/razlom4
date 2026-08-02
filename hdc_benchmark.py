"""Preregistered multi-arm benchmark for the runnable HDC prototype."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Sequence

from hdc_cognitive import (
    BackoffNGramBaseline,
    CharNGramKNNBaseline,
    HDCCognitiveSystem,
    NGramBaseline,
    evaluate_next_token,
    synthetic_language,
    tokenize,
)


DEV_SEEDS = (1, 2, 3, 4, 5)
TEST_SEEDS = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110)


def _corrupt(token: str, variant: int) -> str:
    if len(token) < 3 or not token.isalpha():
        return token
    index = 1 + variant % (len(token) - 1)
    kind = variant % 3
    if kind == 0:
        return token[:index] + token[index + 1 :]
    if kind == 1 and index < len(token) - 1:
        chars = list(token)
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
        return "".join(chars)
    replacement = "я" if token[index] != "я" else "ю"
    return token[:index] + replacement + token[index + 1 :]


def noisy_probes(sequences: Sequence[str], context_length: int = 4) -> list[tuple[list[str], str]]:
    probes: list[tuple[list[str], str]] = []
    ordinal = 0
    for sequence in sequences:
        tokens = tokenize(sequence)
        for index in range(1, len(tokens)):
            context = tokens[max(0, index - context_length) : index]
            candidates = [position for position, token in enumerate(context) if len(token) >= 3 and token.isalpha()]
            if candidates:
                position = candidates[ordinal % len(candidates)]
                context = list(context)
                context[position] = _corrupt(context[position], ordinal)
            probes.append((list(context), tokens[index]))
            ordinal += 1
    return probes


def evaluate_probes(model, probes: Sequence[tuple[Sequence[str], str]]) -> dict[str, float | int]:
    correct = known = 0
    for context, target in probes:
        prediction = model.predict(context)
        known += prediction.token is not None
        correct += prediction.token == target
    total = len(probes)
    return {
        "total": total,
        "correct": correct,
        "known": known,
        "accuracy": correct / max(1, total),
        "coverage": known / max(1, total),
    }


def order_language(
    seed: int, train_size: int = 20, test_size: int = 10
) -> tuple[list[str], list[tuple[list[str], str]]]:
    agents = ["анна", "борис", "вера", "глеб", "дина", "егор"]
    rows = []
    for left in agents:
        for right in agents:
            if left == right:
                continue
            # The target is an observable ordered rule: repeat the first agent.
            rows.append(f"{left} перед {right} значит {left} первый .")
    import random

    rng = random.Random(seed)
    rng.shuffle(rows)
    split = min(train_size, max(1, len(rows) - test_size))
    train = rows[:split] * 4
    probes = []
    for row in rows[split : split + test_size]:
        tokens = tokenize(row)
        marker = tokens.index("значит")
        probes.append((tokens[max(0, marker - 3) : marker + 1], tokens[marker + 1]))
    return train, probes


def retention_streams(
    seed: int,
) -> tuple[list[str], list[str], list[str], list[tuple[list[str], str]]]:
    train_a, test_a = synthetic_language(seed=seed, train_size=180, test_size=50)
    train_b, test_b = order_language(seed=seed + 10_000, train_size=20, test_size=10)
    return train_a, test_a, train_b, test_b


def loop_failure(tokens: Sequence[str]) -> bool:
    if len(tokens) < 4:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(tokens) - 3):
        gram = tuple(tokens[index : index + 4])
        counts[gram] = counts.get(gram, 0) + 1
    return any(count > 2 for count in counts.values())


def build_arms(seed: int, capacity: int = 1024):
    common = dict(
        dimension=1024,
        context_length=4,
        stm_capacity=capacity,
        ltm_capacity=capacity * 2,
        query_threshold=0.20,
    )
    return {
        "exact_ngram": NGramBaseline(context_length=4, max_contexts=capacity),
        "backoff_ngram": BackoffNGramBaseline(context_length=4, max_contexts=capacity),
        "char_knn": CharNGramKNNBaseline(context_length=4, max_contexts=capacity),
        "static_hdc": HDCCognitiveSystem(seed=f"bench-{seed}", dynamic_semantics=False, **common),
        "dynamic_hdc": HDCCognitiveSystem(seed=f"bench-{seed}", dynamic_semantics=True, **common),
    }


def train(model, rows: Sequence[str]) -> None:
    model.train_sequences(rows)


def logical_bytes(model) -> int:
    return int(model.logical_bytes()) if hasattr(model, "logical_bytes") else 0


def run_seed(seed: int) -> dict[str, object]:
    clean_train, clean_test = synthetic_language(seed=seed, train_size=240, test_size=80)
    order_train, order_test = order_language(seed=seed, train_size=20, test_size=10)
    combined_train = clean_train + order_train
    arms = build_arms(seed)
    result: dict[str, object] = {"seed": seed, "arms": {}}

    for name, model in arms.items():
        started = time.perf_counter()
        train(model, combined_train)
        train_seconds = time.perf_counter() - started
        clean = evaluate_next_token(model, clean_test)
        noise = evaluate_probes(model, noisy_probes(clean_test))
        order = evaluate_probes(model, order_test)
        result["arms"][name] = {
            "clean": clean,
            "noise": noise,
            "order": order,
            "logical_bytes": logical_bytes(model),
            "train_seconds": train_seconds,
        }

    # Separate models avoid leaking the clean/order benchmark into retention.
    retention_arms = build_arms(seed + 50_000)
    train_a, test_a, train_b, test_b = retention_streams(seed)
    for name, model in retention_arms.items():
        train(model, train_a)
        before = evaluate_next_token(model, test_a)["accuracy"]
        train(model, train_b)
        after = evaluate_next_token(model, test_a)["accuracy"]
        acquired = evaluate_probes(model, test_b)["accuracy"]
        harmonic = 2 * after * acquired / max(1e-12, after + acquired)
        result["arms"][name]["retention"] = {
            "a_before": before,
            "a_after": after,
            "drop": before - after,
            "b_acquired": acquired,
            "harmonic": harmonic,
        }

    loop_model = HDCCognitiveSystem(
        dimension=1024,
        seed=f"loop-{seed}",
        stm_capacity=512,
        ltm_capacity=1024,
    )
    loop_rows = [
        "красный ключ открывает дверь . за дверью тихий сад .",
        "тихий сад скрывает ключ . красный ключ открывает дверь .",
        "робот идёт в сад . робот идёт в сад .",
    ] * 12
    loop_model.train_sequences(loop_rows)
    prompts = ["красный ключ", "тихий сад", "робот идёт", "за дверью"] * 50
    failures = 0
    lengths = []
    for prompt in prompts:
        generated = loop_model.generate(prompt, max_tokens=32)
        failures += loop_failure(generated)
        lengths.append(len(generated) - len(tokenize(prompt)))
    result["loop"] = {
        "prompts": len(prompts),
        "failures": failures,
        "failure_rate": failures / len(prompts),
        "mean_generated_tokens": statistics.mean(lengths),
    }
    return result


def aggregate(results: Sequence[dict[str, object]]) -> dict[str, object]:
    arm_names = list(results[0]["arms"])
    summary: dict[str, object] = {"seeds": [row["seed"] for row in results], "arms": {}}
    for name in arm_names:
        arm_summary = {}
        for metric_path in (
            ("clean", "accuracy"),
            ("noise", "accuracy"),
            ("order", "accuracy"),
            ("retention", "drop"),
            ("retention", "harmonic"),
        ):
            values = [row["arms"][name][metric_path[0]][metric_path[1]] for row in results]
            arm_summary["_".join(metric_path)] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        arm_summary["logical_bytes_mean"] = statistics.mean(
            row["arms"][name]["logical_bytes"] for row in results
        )
        summary["arms"][name] = arm_summary
    loop_rates = [row["loop"]["failure_rate"] for row in results]
    summary["loop_failure_rate_mean"] = statistics.mean(loop_rates)
    non_hdc = ("exact_ngram", "backoff_ngram", "char_knn")
    strongest_noise = max(summary["arms"][name]["noise_accuracy"]["mean"] for name in non_hdc)
    strongest_clean = max(summary["arms"][name]["clean_accuracy"]["mean"] for name in non_hdc)
    dynamic = summary["arms"]["dynamic_hdc"]
    static = summary["arms"]["static_hdc"]
    noise_advantage = dynamic["noise_accuracy"]["mean"] - strongest_noise
    clean_regression = strongest_clean - dynamic["clean_accuracy"]["mean"]
    retention_advantage = (
        dynamic["retention_harmonic"]["mean"] - static["retention_harmonic"]["mean"]
    )
    hard_gates = {
        "loop_rate_at_most_2pct": summary["loop_failure_rate_mean"] <= 0.02,
        "retention_drop_at_most_5pct": dynamic["retention_drop"]["mean"] <= 0.05,
        "clean_regression_at_most_2pct": clean_regression <= 0.02,
    }
    value_gates = {
        "noise_advantage_at_least_3pct": noise_advantage >= 0.03,
        "retention_harmonic_advantage_at_least_3pct": retention_advantage >= 0.03,
    }
    summary["gates"] = {
        "hard": hard_gates,
        "value": value_gates,
        "noise_advantage": noise_advantage,
        "clean_regression": clean_regression,
        "retention_harmonic_advantage": retention_advantage,
        "verdict": (
            "VALUABLE_PROTOTYPE"
            if all(hard_gates.values()) and any(value_gates.values())
            else "NOT_YET_VALUABLE"
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seeds = DEV_SEEDS if args.split == "dev" else TEST_SEEDS
    rows = []
    for seed in seeds:
        print(f"running seed {seed}...", flush=True)
        rows.append(run_seed(seed))
    artifact = {
        "benchmark_version": "hdc-preregistered-v1",
        "split": args.split,
        "results": rows,
        "aggregate": aggregate(rows),
    }
    Path(args.output).write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
