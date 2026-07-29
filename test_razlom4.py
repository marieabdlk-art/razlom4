import copy
import json
import unittest
from pathlib import Path

from razlom4 import select_candidate, validate_session


HERE = Path(__file__).parent


def load_example():
    return json.loads((HERE / "example_session.json").read_text(encoding="utf-8"))


class Razlom4Tests(unittest.TestCase):
    def test_example_is_valid_and_selects_uncertainty_gate(self):
        session = load_example()
        self.assertEqual(validate_session(session), [])
        result = select_candidate(session)
        self.assertEqual(result["status"], "VALID_MUTATION")
        self.assertEqual(result["selected_candidate_id"], "D2")
        self.assertEqual(result["novelty_label"], "panel_novel")

    def test_self_review_is_rejected(self):
        session = load_example()
        session["reviews"][0]["reviewer_role"] = "architect"
        errors = validate_session(session)
        self.assertTrue(any("self-review" in error for error in errors))

    def test_relabelled_frozen_primitive_is_rejected(self):
        session = load_example()
        session["candidates"][1]["causal_operator"] = "automatic_send"
        errors = validate_session(session)
        self.assertTrue(any("reuses a frozen proposal primitive" in error for error in errors))

    def test_discarded_assumption_must_be_frozen(self):
        session = load_example()
        session["candidates"][1]["discarded_self_assumption"] = "Новая удобная формулировка"
        errors = validate_session(session)
        self.assertTrue(any("frozen assumption" in error for error in errors))

    def test_maximin_beats_higher_average(self):
        session = load_example()
        d1 = next(item for item in session["candidates"] if item["id"] == "D1")
        d1["experiment_result"] = "pass"
        d1_reviews = [item for item in session["reviews"] if item["candidate_id"] == "D1"]
        for review, score in zip(d1_reviews, (0.99, 0.99, 0.60)):
            review["role_score"] = score
            review["kill_test_result"] = "pass"
        d2_reviews = [item for item in session["reviews"] if item["candidate_id"] == "D2"]
        for review in d2_reviews:
            review["role_score"] = 0.80
        result = select_candidate(session)
        self.assertEqual(result["selected_candidate_id"], "D2")

    def test_confirmed_veto_eliminates_otherwise_strong_candidate(self):
        session = load_example()
        d4_reviews = [item for item in session["reviews"] if item["candidate_id"] == "D4"]
        for review in d4_reviews:
            review["role_score"] = 1.0
        result = select_candidate(session)
        d4_eval = next(
            item for item in result["evaluations"] if item["candidate_id"] == "D4"
        )
        self.assertEqual(d4_eval["status"], "INVALID")
        self.assertIn("confirmed hard-constraint veto", d4_eval["reasons"])

    def test_unrun_experiment_cannot_be_valid_mutation(self):
        session = load_example()
        result = select_candidate(session)
        d1_eval = next(
            item for item in result["evaluations"] if item["candidate_id"] == "D1"
        )
        self.assertEqual(d1_eval["status"], "NOVELTY_UNCONFIRMED")

    def test_all_failed_candidates_return_failure_certificate_status(self):
        session = load_example()
        for candidate in session["candidates"]:
            candidate["experiment_result"] = "fail"
        result = select_candidate(session)
        self.assertEqual(result["status"], "NO_VALID_MUTATION")
        self.assertIsNone(result["selected_candidate_id"])

    def test_selection_is_deterministic(self):
        original = load_example()
        first = select_candidate(copy.deepcopy(original))
        second = select_candidate(copy.deepcopy(original))
        self.assertEqual(first, second)

    def test_malformed_nested_values_return_errors_instead_of_crashing(self):
        session = load_example()
        session["candidates"][0]["source_roles"] = [{"not": "a role"}]
        session["reviews"][0]["hard_failures"] = [{"not": "an id"}]
        errors = validate_session(session)
        self.assertTrue(any("source_roles" in error for error in errors))
        self.assertTrue(any("hard_failures" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
