"""Deterministic validation and selection core for the РАЗЛОМ-4 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROLES = ("architect", "skeptic", "simplifier", "failure_hunter")
MECHANISM_FIELDS = (
    "input",
    "trigger",
    "transformation",
    "state_changed",
    "output",
    "observable_prediction",
    "falsifier",
)
MUTATION_OPERATORS = {
    "invert",
    "gate",
    "amputate",
    "transplant",
    "reify",
    "delay_or_expire",
    "price_or_budget",
}


def _norm(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_mechanism(mechanism: Any, path: str, errors: list[str]) -> None:
    if not isinstance(mechanism, dict):
        errors.append(f"{path} must be an object")
        return
    for field in MECHANISM_FIELDS:
        if not _is_text(mechanism.get(field)):
            errors.append(f"{path}.{field} must be non-empty text")


def validate_session(session: Any) -> list[str]:
    """Return all semantic protocol violations; an empty list means valid."""

    errors: list[str] = []
    if not isinstance(session, dict):
        return ["session must be an object"]

    contract = session.get("contract")
    if not isinstance(contract, dict):
        errors.append("contract must be an object")
        contract = {}
    for field in ("task_id", "question", "baseline"):
        if not _is_text(contract.get(field)):
            errors.append(f"contract.{field} must be non-empty text")

    constraints = contract.get("hard_constraints")
    if not isinstance(constraints, list) or not constraints:
        errors.append("contract.hard_constraints must be a non-empty list")
        constraints = []
    constraint_ids = {
        item.get("id")
        for item in constraints
        if isinstance(item, dict) and _is_text(item.get("id"))
    }
    if len(constraint_ids) != len(constraints):
        errors.append("hard constraint ids must be present and unique")

    tests = contract.get("success_tests")
    if not isinstance(tests, list) or not tests:
        errors.append("contract.success_tests must be a non-empty list")
    if contract.get("novelty_scope") not in {"panel", "corpus"}:
        errors.append("contract.novelty_scope must be panel or corpus")
    if contract.get("novelty_scope") == "corpus" and not _is_text(
        contract.get("corpus_description")
    ):
        errors.append("corpus scope requires contract.corpus_description")

    proposals = session.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != 4:
        errors.append("exactly four proposals are required")
        proposals = proposals if isinstance(proposals, list) else []

    proposal_ids: set[str] = set()
    proposal_roles: set[str] = set()
    proposal_by_role: dict[str, dict[str, Any]] = {}
    primitive_union: set[str] = set()
    mechanism_pairs: set[tuple[str, str]] = set()

    for index, proposal in enumerate(proposals):
        path = f"proposals[{index}]"
        if not isinstance(proposal, dict):
            errors.append(f"{path} must be an object")
            continue
        proposal_id = proposal.get("id")
        role = proposal.get("author_role")
        if not _is_text(proposal_id) or proposal_id in proposal_ids:
            errors.append(f"{path}.id must be non-empty and unique")
        else:
            proposal_ids.add(proposal_id)
        if role not in ROLES or role in proposal_roles:
            errors.append(f"{path}.author_role must be a unique protocol role")
        else:
            proposal_roles.add(role)
            proposal_by_role[role] = proposal
        for field in ("thesis", "invariant", "kill_test"):
            if not _is_text(proposal.get(field)):
                errors.append(f"{path}.{field} must be non-empty text")
        primitives = proposal.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            errors.append(f"{path}.primitives must be a non-empty list")
        else:
            primitive_union.update(_norm(item) for item in primitives if _is_text(item))
        assumptions = proposal.get("assumptions")
        if not isinstance(assumptions, list) or not assumptions:
            errors.append(f"{path}.assumptions must be a non-empty list")
        _validate_mechanism(proposal.get("mechanism"), f"{path}.mechanism", errors)
        mechanism = proposal.get("mechanism")
        if isinstance(mechanism, dict):
            mechanism_pairs.add(
                (
                    _norm(mechanism.get("transformation", "")),
                    _norm(mechanism.get("state_changed", "")),
                )
            )

    if proposal_roles != set(ROLES):
        errors.append("proposals must cover each of the four roles exactly once")

    candidates = session.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        errors.append("exactly four candidates are required")
        candidates = candidates if isinstance(candidates, list) else []

    candidate_ids: set[str] = set()
    candidate_roles: set[str] = set()
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        path = f"candidates[{index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{path} must be an object")
            continue
        candidate_id = candidate.get("id")
        role = candidate.get("author_role")
        if not _is_text(candidate_id) or candidate_id in candidate_ids:
            errors.append(f"{path}.id must be non-empty and unique")
        else:
            candidate_ids.add(candidate_id)
            candidate_by_id[candidate_id] = candidate
        if role not in ROLES or role in candidate_roles:
            errors.append(f"{path}.author_role must be a unique protocol role")
        else:
            candidate_roles.add(role)

        source_roles = candidate.get("source_roles")
        valid_source_roles = (
            isinstance(source_roles, list)
            and all(item in ROLES for item in source_roles)
        )
        if (
            not valid_source_roles
            or len(set(source_roles)) < 3
        ):
            errors.append(f"{path}.source_roles must contain at least three unique roles")

        discarded = candidate.get("discarded_self_assumption")
        own_proposal = proposal_by_role.get(role, {})
        own_assumptions = own_proposal.get("assumptions", [])
        if not _is_text(discarded) or _norm(discarded) not in {
            _norm(item) for item in own_assumptions
        }:
            errors.append(
                f"{path}.discarded_self_assumption must match an author's frozen assumption"
            )

        for field in (
            "shared_hidden_assumption",
            "causal_operator",
            "ablation",
            "prediction",
            "experiment",
            "rollback",
        ):
            if not _is_text(candidate.get(field)):
                errors.append(f"{path}.{field} must be non-empty text")

        if candidate.get("mutation_operator") not in MUTATION_OPERATORS:
            errors.append(f"{path}.mutation_operator is not allowed")
        causal_operator = candidate.get("causal_operator")
        if _is_text(causal_operator) and _norm(causal_operator) in primitive_union:
            errors.append(f"{path}.causal_operator reuses a frozen proposal primitive")

        _validate_mechanism(candidate.get("mechanism"), f"{path}.mechanism", errors)
        mechanism = candidate.get("mechanism")
        if isinstance(mechanism, dict):
            pair = (
                _norm(mechanism.get("transformation", "")),
                _norm(mechanism.get("state_changed", "")),
            )
            if pair in mechanism_pairs:
                errors.append(
                    f"{path}.mechanism must change transformation or state_changed"
                )

        if candidate.get("experiment_result") not in {"pass", "fail", "unrun"}:
            errors.append(f"{path}.experiment_result must be pass, fail, or unrun")
        for field in ("simplicity", "reversibility"):
            value = candidate.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
                errors.append(f"{path}.{field} must be a number from 0 to 1")

    if candidate_roles != set(ROLES):
        errors.append("candidates must cover each of the four roles exactly once")

    reviews = session.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 12:
        errors.append("exactly twelve external reviews are required")
        reviews = reviews if isinstance(reviews, list) else []

    review_pairs: set[tuple[str, str]] = set()
    reviewers_by_candidate: dict[str, set[str]] = {item: set() for item in candidate_ids}
    for index, review in enumerate(reviews):
        path = f"reviews[{index}]"
        if not isinstance(review, dict):
            errors.append(f"{path} must be an object")
            continue
        reviewer = review.get("reviewer_role")
        candidate_id = review.get("candidate_id")
        candidate = candidate_by_id.get(candidate_id) if _is_text(candidate_id) else None
        if reviewer not in ROLES:
            errors.append(f"{path}.reviewer_role is invalid")
        if candidate is None:
            errors.append(f"{path}.candidate_id does not exist")
        elif reviewer == candidate.get("author_role"):
            errors.append(f"{path} is a forbidden self-review")
        pair = (str(reviewer), str(candidate_id))
        if pair in review_pairs:
            errors.append(f"{path} duplicates a reviewer/candidate pair")
        review_pairs.add(pair)
        if _is_text(candidate_id) and candidate_id in reviewers_by_candidate and reviewer in ROLES:
            reviewers_by_candidate[candidate_id].add(reviewer)

        hard_failures = review.get("hard_failures")
        if not isinstance(hard_failures, list) or not all(
            _is_text(item) for item in hard_failures
        ):
            errors.append(f"{path}.hard_failures must be a list of constraint ids")
            hard_failures = []
        unknown = set(hard_failures) - constraint_ids
        if unknown:
            errors.append(f"{path}.hard_failures contains unknown ids: {sorted(unknown)}")
        kill_result = review.get("kill_test_result")
        if kill_result not in {"pass", "fail", "unrun"}:
            errors.append(f"{path}.kill_test_result must be pass, fail, or unrun")
        if hard_failures and kill_result != "fail":
            errors.append(f"{path} claims hard failure without observed kill-test failure")
        if not _is_text(review.get("kill_test")):
            errors.append(f"{path}.kill_test must be non-empty text")

        verdict = review.get("equivalence_verdict")
        if verdict not in {"equivalent", "distinct", "uncertain"}:
            errors.append(f"{path}.equivalence_verdict is invalid")
        if not _is_text(review.get("aligned_fields")):
            errors.append(f"{path}.aligned_fields must be non-empty text")
        if verdict == "distinct" and not _is_text(review.get("distinguishing_test")):
            errors.append(f"{path}.distinct verdict requires a distinguishing test")
        score = review.get("role_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
            errors.append(f"{path}.role_score must be a number from 0 to 1")
        if not _is_text(review.get("residual_risk")):
            errors.append(f"{path}.residual_risk must be non-empty text")

    for candidate_id, reviewers in reviewers_by_candidate.items():
        candidate = candidate_by_id[candidate_id]
        expected = set(ROLES) - {candidate.get("author_role")}
        if reviewers != expected:
            errors.append(
                f"candidate {candidate_id} must be reviewed once by all three non-authors"
            )

    return errors


def _candidate_evaluation(
    candidate: dict[str, Any], reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate_reviews = [
        review for review in reviews if review["candidate_id"] == candidate["id"]
    ]
    confirmed_veto = any(
        review["hard_failures"] and review["kill_test_result"] == "fail"
        for review in candidate_reviews
    )
    distinct_votes = sum(
        review["equivalence_verdict"] == "distinct" for review in candidate_reviews
    )
    scores = [float(review["role_score"]) for review in candidate_reviews]
    all_kill_tests_run = all(
        review["kill_test_result"] != "unrun" for review in candidate_reviews
    )
    status = "VALID_MUTATION"
    reasons: list[str] = []
    if confirmed_veto:
        status = "INVALID"
        reasons.append("confirmed hard-constraint veto")
    if candidate["experiment_result"] == "fail":
        status = "INVALID"
        reasons.append("candidate experiment failed")
    if distinct_votes < 2:
        status = "INVALID"
        reasons.append("fewer than two external distinct verdicts")
    if status != "INVALID" and (
        candidate["experiment_result"] == "unrun" or not all_kill_tests_run
    ):
        status = "NOVELTY_UNCONFIRMED"
        reasons.append("one or more required tests are unrun")
    return {
        "candidate_id": candidate["id"],
        "status": status,
        "reasons": reasons,
        "distinct_votes": distinct_votes,
        "min_external_score": min(scores),
        "mean_external_score": sum(scores) / len(scores),
        "simplicity": float(candidate["simplicity"]),
        "reversibility": float(candidate["reversibility"]),
        "canonical_hash": _canonical_hash(candidate),
    }


def select_candidate(session: dict[str, Any]) -> dict[str, Any]:
    """Validate the session, apply gates, and select by maximin."""

    errors = validate_session(session)
    if errors:
        raise ValueError("invalid session:\n- " + "\n- ".join(errors))

    evaluations = [
        _candidate_evaluation(candidate, session["reviews"])
        for candidate in session["candidates"]
    ]
    valid = [item for item in evaluations if item["status"] == "VALID_MUTATION"]
    pool = valid or [
        item for item in evaluations if item["status"] == "NOVELTY_UNCONFIRMED"
    ]
    if not pool:
        return {
            "status": "NO_VALID_MUTATION",
            "selected_candidate_id": None,
            "novelty_label": None,
            "evaluations": evaluations,
        }

    ranked = sorted(
        pool,
        key=lambda item: (
            -item["min_external_score"],
            -item["mean_external_score"],
            -item["reversibility"],
            -item["simplicity"],
            item["canonical_hash"],
        ),
    )
    winner = ranked[0]
    return {
        "status": winner["status"],
        "selected_candidate_id": winner["candidate_id"],
        "novelty_label": "panel_novel",
        "external_novelty": "not_established_by_protocol",
        "selection_rule": "hard gates, then maximin; mean/reversibility/simplicity/hash tie-breaks",
        "evaluations": evaluations,
    }


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "select"))
    parser.add_argument("session", help="path to a РАЗЛОМ-4 session JSON")
    args = parser.parse_args()
    session = _load(args.session)
    if args.command == "validate":
        errors = validate_session(session)
        result = {"valid": not errors, "errors": errors}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not errors else 1
    try:
        result = select_candidate(session)
    except ValueError as exc:
        print(str(exc))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
