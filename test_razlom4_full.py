import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from razlom4 import ROLES, validate_session
from razlom4_full import (
    FullPipeline,
    CheckpointProvider,
    OpenRouterProvider,
    PipelineError,
    ReplayProvider,
    _normalize_candidate,
    _normalize_reviews,
    main,
    score_and_select_seeds,
)


HERE = Path(__file__).parent


def load_example():
    return json.loads((HERE / "example_session.json").read_text(encoding="utf-8"))


def seed_from_proposal(proposal, index):
    return {
        "id": f"K{index}",
        "title": f"Divergent seed {index}",
        "thesis": proposal["thesis"],
        "primitives": proposal["primitives"],
        "assumptions": proposal["assumptions"],
        "invariant": proposal["invariant"],
        "mechanism": proposal["mechanism"],
        "kill_test": proposal["kill_test"],
        "source_ops": ["inversion", f"analogy:domain-{index}", "recombination"],
    }


def replay_records(session):
    seeds = [seed_from_proposal(item, i) for i, item in enumerate(session["proposals"], 1)]
    records = [
        {
            "stage": "kuds_divergence",
            "response": {
                "baseline_snapshot": [
                    "Automate every answer",
                    "Route uncertain answers to a human",
                    "Add a confidence threshold",
                ],
                "ideas": seeds,
            },
        }
    ]
    proposals_by_role = {item["author_role"]: item for item in session["proposals"]}
    candidates_by_role = {item["author_role"]: item for item in session["candidates"]}
    for role in ROLES:
        records.append(
            {"stage": f"proposal:{role}", "response": copy.deepcopy(proposals_by_role[role])}
        )
    for role in ROLES:
        records.append(
            {"stage": f"forge:{role}", "response": copy.deepcopy(candidates_by_role[role])}
        )
    for role in ROLES:
        role_reviews = [
            copy.deepcopy(item)
            for item in session["reviews"]
            if item["reviewer_role"] == role
        ]
        records.append(
            {"stage": f"review:{role}", "response": {"reviews": role_reviews}}
        )
    return records


class Razlom4FullTests(unittest.TestCase):
    def test_full_pipeline_builds_valid_session_and_dossier(self):
        session = load_example()
        provider = ReplayProvider(replay_records(session))

        result = FullPipeline(provider, n_candidates=4).run(session["contract"])

        self.assertEqual(result["method"], "kuds_razlom4_full")
        self.assertEqual(result["model_calls"], 13)
        self.assertEqual(validate_session(result["session"]), [])
        self.assertEqual(result["selection"]["selected_candidate_id"], "D2")
        self.assertEqual(result["idea_dossier"]["id"], "D2")
        self.assertEqual(len(result["kuds"]["selected_seed_ids"]), 4)
        self.assertEqual(set(result["kuds"]["seed_assignments"]), set(ROLES))
        self.assertEqual(
            set(result["kuds"]["seed_assignments"].values()),
            set(result["kuds"]["selected_seed_ids"]),
        )
        self.assertEqual(provider.index, 13)

    def test_seed_selection_is_deterministic(self):
        session = load_example()
        ideas = [seed_from_proposal(item, i) for i, item in enumerate(session["proposals"], 1)]
        baseline = ["Ordinary automatic answer with human review"]

        first, first_report = score_and_select_seeds(baseline, ideas, count=3)
        second, second_report = score_and_select_seeds(baseline, ideas, count=3)

        self.assertEqual([item["id"] for item in first], [item["id"] for item in second])
        self.assertEqual(first_report, second_report)
        self.assertEqual(sum(item["selected"] for item in first_report), 3)

    def test_replay_fails_closed_on_wrong_stage(self):
        provider = ReplayProvider([{"stage": "wrong", "response": {}}])
        with self.assertRaisesRegex(PipelineError, "stage mismatch"):
            provider.complete_json("", stage="expected", temperature=0.0)

    def test_openrouter_provider_tracks_usage_and_cost(self):
        payload = {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "completion_tokens_details": {"reasoning_tokens": 200},
                "cost": 0.004,
            },
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        provider = OpenRouterProvider(api_key="test-key")
        with patch("urllib.request.urlopen", return_value=Response()) as mocked:
            response = provider.complete_json(
                "prompt", stage="proposal:architect", temperature=0.5
            )

        self.assertEqual(response, {"ok": True})
        request_body = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(request_body["model"], "z-ai/glm-5.1")
        self.assertEqual(request_body["max_tokens"], 3000)
        self.assertEqual(request_body["reasoning"]["effort"], "none")
        summary = provider.usage_summary()
        self.assertEqual(summary["prompt_tokens"], 1000)
        self.assertEqual(summary["completion_tokens"], 500)
        self.assertEqual(summary["reasoning_tokens"], 200)
        self.assertEqual(summary["reported_cost"], 0.004)

    def test_empty_content_retries_without_reasoning(self):
        payloads = [
            {
                "choices": [
                    {"message": {"content": None}, "finish_reason": "length"}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5000},
            },
            {
                "choices": [
                    {"message": {"content": '{"recovered": true}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        ]

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        responses = [Response(payload) for payload in payloads]
        provider = OpenRouterProvider(api_key="test-key", reasoning_effort="low")
        with patch("urllib.request.urlopen", side_effect=responses) as mocked:
            result = provider.complete_json(
                "prompt", stage="forge:architect", temperature=0.7
            )

        self.assertEqual(result, {"recovered": True})
        self.assertEqual(mocked.call_count, 2)
        retry_body = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(retry_body["reasoning"]["effort"], "none")
        self.assertEqual(retry_body["max_tokens"], 7500)
        self.assertEqual(len(provider.usage_records), 2)

    def test_checkpoint_reuses_successful_stage(self):
        class CountingProvider:
            model = "test-model"

            def __init__(self):
                self.calls = 0

            def complete_json(self, prompt, *, stage, temperature):
                del prompt, temperature
                self.calls += 1
                return {"stage": stage, "call": self.calls}

        with tempfile.TemporaryDirectory() as tmp:
            inner = CountingProvider()
            checkpoint = CheckpointProvider(inner, Path(tmp) / "checkpoint.json")
            first = checkpoint.complete_json("same", stage="stage-1", temperature=0.1)
            second = checkpoint.complete_json("same", stage="stage-1", temperature=0.1)

            self.assertEqual(first, second)
            self.assertEqual(inner.calls, 1)

    def test_glm_wrapped_camel_case_cards_are_normalized(self):
        mechanism = load_example()["candidates"][0]["mechanism"]
        raw_candidate = {
            "deltaCandidate": {
                "id": "wrapped",
                "canonicalMechanism": mechanism,
                "simplicityScore": "80%",
                "reversibilityScore": "0.65",
            }
        }
        candidate = _normalize_candidate(
            raw_candidate, role="architect", fallback_id="D1"
        )
        self.assertEqual(candidate["mechanism"], mechanism)
        self.assertEqual(candidate["simplicity"], 0.8)
        self.assertEqual(candidate["reversibility"], 0.65)
        self.assertEqual(set(candidate["source_roles"]), set(ROLES))

        expected = [{"id": "D2"}, {"id": "D3"}, {"id": "D4"}]
        raw_reviews = [
            {
                "candidateReview": {
                    "candidateId": "duplicate",
                    "hardFailures": "C1 and unknown C9",
                    "roleScore": "75%",
                }
            }
            for _ in expected
        ]
        reviews = _normalize_reviews(
            raw_reviews,
            reviewer_role="architect",
            expected_candidates=expected,
            constraint_ids={"C1", "C2"},
        )
        self.assertEqual([item["candidate_id"] for item in reviews], ["D2", "D3", "D4"])
        self.assertEqual(reviews[0]["hard_failures"], ["C1"])
        self.assertEqual(reviews[0]["role_score"], 0.75)
        self.assertEqual(reviews[0]["equivalence_verdict"], "uncertain")

    def test_cli_can_run_fully_offline_from_replay(self):
        session = load_example()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract_path = root / "contract.json"
            replay_path = root / "replay.json"
            output_path = root / "run.json"
            contract_path.write_text(
                json.dumps(session["contract"], ensure_ascii=False), encoding="utf-8"
            )
            replay_path.write_text(
                json.dumps(replay_records(session), ensure_ascii=False), encoding="utf-8"
            )

            status = main(
                [
                    str(contract_path),
                    "--output",
                    str(output_path),
                    "--candidates",
                    "4",
                    "--replay",
                    str(replay_path),
                ]
            )

            self.assertEqual(status, 0)
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["selection"]["selected_candidate_id"], "D2")


if __name__ == "__main__":
    unittest.main()
