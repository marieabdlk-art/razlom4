"""Blind, deterministic scorer for the РАЗЛОМ-4 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ARMS = (
    "single_agent",
    "best_of_4",
    "ordinary_debate",
    "razlom4_no_roles",
    "razlom4_full",
)
MECHANISM_FIELDS = (
    "input",
    "trigger",
    "transformation",
    "state_changed",
    "output",
    "observable_prediction",
    "falsifier",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def _write(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_ids(items: Any, path: str, errors: list[str]) -> set[str]:
    if not isinstance(items, list) or not items:
        errors.append(f"{path} must be a non-empty list")
        return set()
    result: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not _is_text(item.get("id")):
            errors.append(f"{path}[{index}].id must be non-empty text")
        else:
            result.append(item["id"])
    if len(set(result)) != len(result):
        errors.append(f"{path} ids must be unique")
    return set(result)


def validate_banks(public: Any, private: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(public, dict) or not isinstance(private, dict):
        return ["both banks must be objects"]
    if public.get("schema") != "razlom4-public-task-bank/1":
        errors.append("public.schema is invalid")
    if private.get("schema") != "razlom4-private-task-bank/1":
        errors.append("private.schema is invalid")
    benchmark_id = public.get("benchmark_id")
    if not _is_text(benchmark_id) or private.get("benchmark_id") != benchmark_id:
        errors.append("benchmark_id must be non-empty and match across banks")
    config = public.get("configuration")
    if not isinstance(config, dict):
        errors.append("public.configuration must be an object")
        config = {}
    budget = config.get("total_token_budget_per_task")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        errors.append("total_token_budget_per_task must be a positive integer")
    call_caps = config.get("call_caps")
    if not isinstance(call_caps, dict) or set(call_caps) != set(ARMS):
        errors.append("call_caps must contain exactly the five benchmark arms")
    elif any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in call_caps.values()):
        errors.append("every call cap must be a positive integer")

    public_tasks = public.get("tasks")
    private_tasks = private.get("tasks")
    if not isinstance(public_tasks, list) or not public_tasks:
        errors.append("public.tasks must be a non-empty list")
        public_tasks = []
    if not isinstance(private_tasks, list) or not private_tasks:
        errors.append("private.tasks must be a non-empty list")
        private_tasks = []
    public_ids: list[str] = []
    public_by_id: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(public_tasks):
        path = f"public.tasks[{index}]"
        if not isinstance(task, dict):
            errors.append(f"{path} must be an object")
            continue
        task_id = task.get("id")
        if not _is_text(task_id):
            errors.append(f"{path}.id must be non-empty text")
            continue
        public_ids.append(task_id)
        public_by_id[task_id] = task
        for field in ("stratum", "question", "baseline"):
            if not _is_text(task.get(field)):
                errors.append(f"{path}.{field} must be non-empty text")
        constraint_ids = _string_ids(task.get("hard_constraints"), f"{path}.hard_constraints", errors)
        operator_ids = _string_ids(task.get("operator_catalog"), f"{path}.operator_catalog", errors)
        assumption_ids = _string_ids(task.get("assumptions"), f"{path}.assumptions", errors)
        probe_ids = _string_ids(task.get("probes"), f"{path}.probes", errors)
        failure_ids = _string_ids(task.get("failure_catalog"), f"{path}.failure_catalog", errors)
        if not constraint_ids or not operator_ids or not assumption_ids or not probe_ids or not failure_ids:
            continue
    if len(set(public_ids)) != len(public_ids):
        errors.append("public task ids must be unique")

    private_ids: list[str] = []
    for index, oracle in enumerate(private_tasks):
        path = f"private.tasks[{index}]"
        if not isinstance(oracle, dict) or not _is_text(oracle.get("id")):
            errors.append(f"{path}.id must be non-empty text")
            continue
        task_id = oracle["id"]
        private_ids.append(task_id)
        public_task = public_by_id.get(task_id)
        if public_task is None:
            errors.append(f"{path}.id has no public task")
            continue
        known_operators = {item["id"] for item in public_task["operator_catalog"]}
        known_assumptions = {item["id"] for item in public_task["assumptions"]}
        known_probes = {item["id"] for item in public_task["probes"]}
        known_failures = {item["id"] for item in public_task["failure_catalog"]}
        no_solution = oracle.get("no_valid_solution")
        solutions = oracle.get("valid_solutions")
        if not isinstance(no_solution, bool):
            errors.append(f"{path}.no_valid_solution must be boolean")
            no_solution = False
        if not isinstance(solutions, list):
            errors.append(f"{path}.valid_solutions must be a list")
            solutions = []
        if no_solution == bool(solutions):
            errors.append(f"{path} must define either no_valid_solution or valid_solutions")
        for solution_index, solution in enumerate(solutions):
            spath = f"{path}.valid_solutions[{solution_index}]"
            if not isinstance(solution, dict):
                errors.append(f"{spath} must be an object")
                continue
            operators = solution.get("operator_ids")
            assumptions = solution.get("rejected_assumption_ids")
            probes = solution.get("probe_outcomes")
            failures = solution.get("required_failure_ids")
            if not isinstance(operators, list) or not operators or set(operators) - known_operators:
                errors.append(f"{spath}.operator_ids are invalid")
            if not isinstance(assumptions, list) or set(assumptions) - known_assumptions:
                errors.append(f"{spath}.rejected_assumption_ids are invalid")
            if not isinstance(probes, dict) or set(probes) != known_probes or any(type(v) is not bool for v in probes.values()):
                errors.append(f"{spath}.probe_outcomes must cover every public probe with booleans")
            if not isinstance(failures, list) or set(failures) - known_failures:
                errors.append(f"{spath}.required_failure_ids are invalid")
    if len(set(private_ids)) != len(private_ids):
        errors.append("private task ids must be unique")
    if set(public_ids) != set(private_ids):
        errors.append("public and private banks must contain exactly the same task ids")
    return errors


def make_manifest(public: dict[str, Any], private: dict[str, Any]) -> dict[str, Any]:
    errors = validate_banks(public, private)
    if errors:
        raise ValueError("invalid task bank:\n- " + "\n- ".join(errors))
    return {
        "schema": "razlom4-benchmark-manifest/1",
        "benchmark_id": public["benchmark_id"],
        "public_bank_sha256": canonical_hash(public),
        "private_bank_sha256": canonical_hash(private),
        "task_ids": [task["id"] for task in public["tasks"]],
        "arms": list(ARMS),
        "total_token_budget_per_task": public["configuration"]["total_token_budget_per_task"],
        "preregistered_seed": public["configuration"]["preregistered_seed"],
    }


def verify_manifest(public: dict[str, Any], private: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = make_manifest(public, private)
    if manifest != expected:
        raise ValueError("manifest does not match the current public/private banks")


def validate_submission(submission: Any, public: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(submission, dict):
        return ["submission must be an object"]
    if submission.get("schema") != "razlom4-benchmark-submission/1":
        errors.append("submission.schema is invalid")
    if submission.get("benchmark_id") != public.get("benchmark_id"):
        errors.append("submission benchmark_id does not match")
    arm = submission.get("arm")
    if arm not in ARMS:
        errors.append("submission.arm is invalid")
    results = submission.get("results")
    if not isinstance(results, list):
        return errors + ["submission.results must be a list"]
    public_by_id = {task["id"]: task for task in public["tasks"]}
    seen: list[str] = []
    budget = public["configuration"]["total_token_budget_per_task"]
    call_cap = public["configuration"]["call_caps"].get(arm, 0)
    for index, result in enumerate(results):
        path = f"results[{index}]"
        if not isinstance(result, dict):
            errors.append(f"{path} must be an object")
            continue
        task_id = result.get("task_id")
        task = public_by_id.get(task_id)
        if task is None:
            errors.append(f"{path}.task_id is unknown")
            continue
        seen.append(task_id)
        status = result.get("status")
        if status not in {"SOLUTION", "NO_VALID_SOLUTION", "ERROR"}:
            errors.append(f"{path}.status is invalid")
        operators = result.get("selected_operator_ids")
        assumptions = result.get("rejected_assumption_ids")
        predictions = result.get("probe_predictions")
        failures = result.get("detected_failure_ids")
        known_operators = {item["id"] for item in task["operator_catalog"]}
        known_assumptions = {item["id"] for item in task["assumptions"]}
        known_probes = {item["id"] for item in task["probes"]}
        known_failures = {item["id"] for item in task["failure_catalog"]}
        if not isinstance(operators, list) or set(operators) - known_operators:
            errors.append(f"{path}.selected_operator_ids are invalid")
        if not isinstance(assumptions, list) or set(assumptions) - known_assumptions:
            errors.append(f"{path}.rejected_assumption_ids are invalid")
        if not isinstance(predictions, dict) or set(predictions) - known_probes or any(type(v) is not bool for v in predictions.values()):
            errors.append(f"{path}.probe_predictions are invalid")
        if not isinstance(failures, list) or set(failures) - known_failures:
            errors.append(f"{path}.detected_failure_ids are invalid")
        mechanism = result.get("mechanism")
        if not isinstance(mechanism, dict) or any(not _is_text(mechanism.get(field)) for field in MECHANISM_FIELDS):
            errors.append(f"{path}.mechanism must contain all canonical text fields")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            errors.append(f"{path}.usage must be an object")
        else:
            values = [usage.get("calls"), usage.get("input_tokens"), usage.get("output_tokens")]
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in values):
                errors.append(f"{path}.usage values must be non-negative integers")
            elif usage["calls"] > call_cap:
                errors.append(f"{path}.usage.calls exceeds arm call cap")
            elif usage["input_tokens"] + usage["output_tokens"] > budget:
                errors.append(f"{path}.usage exceeds total token budget")
    if len(set(seen)) != len(seen):
        errors.append("submission contains duplicate task results")
    if set(seen) != set(public_by_id):
        errors.append("submission must contain exactly one result for every task")
    return errors


def _matches(result: dict[str, Any], solution: dict[str, Any]) -> bool:
    return (
        set(result["selected_operator_ids"]) == set(solution["operator_ids"])
        and set(result["rejected_assumption_ids"]) == set(solution["rejected_assumption_ids"])
        and result["probe_predictions"] == solution["probe_outcomes"]
        and set(solution["required_failure_ids"]).issubset(result["detected_failure_ids"])
    )


def _wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def evaluate_submission(
    public: dict[str, Any], private: dict[str, Any], submission: dict[str, Any]
) -> dict[str, Any]:
    errors = validate_submission(submission, public)
    if errors:
        raise ValueError("invalid submission:\n- " + "\n- ".join(errors))
    oracle_by_id = {task["id"]: task for task in private["tasks"]}
    public_by_id = {task["id"]: task for task in public["tasks"]}
    rows: list[dict[str, Any]] = []
    total_probe_hits = total_probes = total_failure_hits = total_failures = 0
    for result in submission["results"]:
        oracle = oracle_by_id[result["task_id"]]
        task = public_by_id[result["task_id"]]
        declared_solution = result["status"] == "SOLUTION"
        if oracle["no_valid_solution"]:
            exact = result["status"] == "NO_VALID_SOLUTION"
            reference_probes: dict[str, bool] = {}
            reference_failures = set(oracle.get("required_failure_ids", []))
        else:
            exact = declared_solution and any(_matches(result, solution) for solution in oracle["valid_solutions"])
            reference_probes = oracle["valid_solutions"][0]["probe_outcomes"]
            reference_failures = set().union(*(set(s["required_failure_ids"]) for s in oracle["valid_solutions"]))
        probe_hits = sum(result["probe_predictions"].get(key) == value for key, value in reference_probes.items())
        failure_hits = len(reference_failures.intersection(result["detected_failure_ids"]))
        total_probe_hits += probe_hits
        total_probes += len(reference_probes)
        total_failure_hits += failure_hits
        total_failures += len(reference_failures)
        rows.append({
            "task_id": result["task_id"],
            "stratum": task["stratum"],
            "exact_success": exact,
            "false_solution": declared_solution and not exact,
            "declared_solution": declared_solution,
            "probe_hits": probe_hits,
            "probe_total": len(reference_probes),
            "failure_hits": failure_hits,
            "failure_total": len(reference_failures),
            "usage": result["usage"],
        })
    total = len(rows)
    exact_count = sum(row["exact_success"] for row in rows)
    return {
        "schema": "razlom4-benchmark-arm-result/1",
        "benchmark_id": public["benchmark_id"],
        "arm": submission["arm"],
        "task_count": total,
        "metrics": {
            "exact_success": exact_count,
            "exact_success_rate": exact_count / total,
            "exact_success_wilson_95": _wilson(exact_count, total),
            "false_solutions": sum(row["false_solution"] for row in rows),
            "false_solution_rate": sum(row["false_solution"] for row in rows) / total,
            "coverage": sum(row["declared_solution"] for row in rows) / total,
            "probe_accuracy": total_probe_hits / total_probes if total_probes else 1.0,
            "failure_recall": total_failure_hits / total_failures if total_failures else 1.0,
            "calls": sum(row["usage"]["calls"] for row in rows),
            "input_tokens": sum(row["usage"]["input_tokens"] for row in rows),
            "output_tokens": sum(row["usage"]["output_tokens"] for row in rows),
        },
        "tasks": rows,
    }


def _sign_test(full_wins: int, control_wins: int) -> float:
    discordant = full_wins + control_wins
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(full_wins, control_wins) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def compare_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {result["arm"]: result for result in results}
    if set(by_arm) != set(ARMS):
        raise ValueError("comparison requires exactly one result for each benchmark arm")
    task_orders = [[row["task_id"] for row in result["tasks"]] for result in results]
    if any(order != task_orders[0] for order in task_orders[1:]):
        raise ValueError("all arms must contain tasks in identical order")
    full_rows = {row["task_id"]: row for row in by_arm["razlom4_full"]["tasks"]}
    comparisons = []
    for arm in ARMS:
        if arm == "razlom4_full":
            continue
        control_rows = {row["task_id"]: row for row in by_arm[arm]["tasks"]}
        full_wins = sum(full_rows[task]["exact_success"] and not control_rows[task]["exact_success"] for task in full_rows)
        control_wins = sum(control_rows[task]["exact_success"] and not full_rows[task]["exact_success"] for task in full_rows)
        comparisons.append({
            "control": arm,
            "full_only_success": full_wins,
            "control_only_success": control_wins,
            "discordant_pairs": full_wins + control_wins,
            "two_sided_sign_test_p": _sign_test(full_wins, control_wins),
            "exact_success_rate_delta": by_arm["razlom4_full"]["metrics"]["exact_success_rate"] - by_arm[arm]["metrics"]["exact_success_rate"],
        })
    return {
        "schema": "razlom4-benchmark-comparison/1",
        "benchmark_id": results[0]["benchmark_id"],
        "arms": {arm: by_arm[arm]["metrics"] for arm in ARMS},
        "primary_comparisons": comparisons,
        "claim_eligible": len(task_orders[0]) >= 40,
        "interpretation": "Smoke runs validate the harness only; method claims require the preregistered minimum task count and an independently sealed bank.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-bank")
    validate.add_argument("public")
    validate.add_argument("private")
    seal = sub.add_parser("seal")
    seal.add_argument("public")
    seal.add_argument("private")
    seal.add_argument("output")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("public")
    evaluate.add_argument("private")
    evaluate.add_argument("manifest")
    evaluate.add_argument("submission")
    compare = sub.add_parser("compare")
    compare.add_argument("public")
    compare.add_argument("private")
    compare.add_argument("manifest")
    compare.add_argument("submission_directory")
    args = parser.parse_args()
    public = _load(args.public)
    private = _load(args.private)
    if args.command == "validate-bank":
        errors = validate_banks(public, private)
        print(json.dumps({"valid": not errors, "errors": errors}, ensure_ascii=False, indent=2))
        return 0 if not errors else 1
    manifest = make_manifest(public, private)
    if args.command == "seal":
        _write(args.output, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    verify_manifest(public, private, _load(args.manifest))
    if args.command == "evaluate":
        result = evaluate_submission(public, private, _load(args.submission))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    submissions = []
    directory = Path(args.submission_directory)
    for arm in ARMS:
        path = directory / f"{arm}.json"
        if not path.exists():
            raise ValueError(f"missing submission: {path}")
        submissions.append(evaluate_submission(public, private, _load(path)))
    print(json.dumps(compare_results(submissions), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
