import copy
import json
import unittest
from pathlib import Path

from razlom4_benchmark import (
    ARMS,
    compare_results,
    evaluate_submission,
    make_manifest,
    validate_banks,
    validate_submission,
    verify_manifest,
)


ROOT = Path(__file__).parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class Razlom4BlindBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.public = load("benchmark/public-task-bank.json")
        self.private = load("benchmark/private-task-bank.json")
        self.submission = load("benchmark/examples/smoke-submission.json")

    def test_smoke_banks_are_semantically_valid(self):
        self.assertEqual(validate_banks(self.public, self.private), [])

    def test_manifest_is_deterministic_and_binds_both_banks(self):
        first = make_manifest(self.public, self.private)
        second = make_manifest(copy.deepcopy(self.public), copy.deepcopy(self.private))
        self.assertEqual(first, second)
        changed = copy.deepcopy(self.private)
        changed["tasks"][0]["valid_solutions"][0]["probe_outcomes"]["P1"] = True
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            verify_manifest(self.public, changed, first)

    def test_correct_smoke_submission_scores_exactly(self):
        result = evaluate_submission(self.public, self.private, self.submission)
        self.assertEqual(result["metrics"]["exact_success"], 3)
        self.assertEqual(result["metrics"]["false_solutions"], 0)
        self.assertEqual(result["metrics"]["probe_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["failure_recall"], 1.0)

    def test_wrong_solution_is_not_hidden_as_coverage(self):
        submission = copy.deepcopy(self.submission)
        row = submission["results"][0]
        row["selected_operator_ids"] = ["auto_send"]
        row["probe_predictions"] = {"P1": True, "P2": True}
        result = evaluate_submission(self.public, self.private, submission)
        self.assertEqual(result["metrics"]["exact_success"], 2)
        self.assertEqual(result["metrics"]["false_solutions"], 1)
        self.assertEqual(result["metrics"]["coverage"], 2 / 3)

    def test_missing_task_and_budget_overrun_fail_closed(self):
        missing = copy.deepcopy(self.submission)
        missing["results"].pop()
        self.assertTrue(any("exactly one result" in error for error in validate_submission(missing, self.public)))
        over = copy.deepcopy(self.submission)
        over["results"][0]["usage"] = {"calls": 12, "input_tokens": 10000, "output_tokens": 2001}
        self.assertTrue(any("token budget" in error for error in validate_submission(over, self.public)))

    def test_arm_call_caps_are_enforced(self):
        submission = copy.deepcopy(self.submission)
        submission["arm"] = "single_agent"
        self.assertTrue(any("call cap" in error for error in validate_submission(submission, self.public)))

    def test_comparison_is_paired_and_smoke_is_not_claim_eligible(self):
        evaluated = []
        for index, arm in enumerate(ARMS):
            submission = copy.deepcopy(self.submission)
            submission["arm"] = arm
            cap = self.public["configuration"]["call_caps"][arm]
            for row in submission["results"]:
                row["usage"]["calls"] = cap
            if arm != "razlom4_full":
                row = submission["results"][index % 2]
                row["selected_operator_ids"] = [
                    "auto_send" if row["task_id"] == "smoke-support-routing" else "blind_retry"
                ]
            evaluated.append(evaluate_submission(self.public, self.private, submission))
        comparison = compare_results(evaluated)
        self.assertFalse(comparison["claim_eligible"])
        self.assertEqual(len(comparison["primary_comparisons"]), 4)
        self.assertTrue(all(item["full_only_success"] >= 1 for item in comparison["primary_comparisons"]))


if __name__ == "__main__":
    unittest.main()
