"""KUDS divergence followed by the RAZLOM-4 conflict protocol.

The module deliberately keeps generation provider-agnostic and delegates final
validation and selection to :mod:`razlom4`.  The default CLI provider is an
OpenAI-compatible OpenRouter endpoint; a replay provider is included for
offline, auditable runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from razlom4 import MECHANISM_FIELDS, ROLES, select_candidate, validate_session


DEFAULT_MODEL = "z-ai/glm-5.1"
GLM51_INPUT_USD_PER_MILLION = 0.966
GLM51_OUTPUT_USD_PER_MILLION = 3.036


class PipelineError(ValueError):
    """Raised when a provider response cannot form a valid pipeline artifact."""


class JsonProvider(Protocol):
    def complete_json(
        self, prompt: str, *, stage: str, temperature: float
    ) -> dict[str, Any]:
        """Return one JSON object for a named pipeline stage."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_]{3,}", str(value).casefold()))


def _jaccard(left: Any, right: Any) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _seed_text(seed: dict[str, Any]) -> str:
    mechanism = seed.get("mechanism", {})
    parts = [seed.get("title", ""), seed.get("thesis", "")]
    if isinstance(mechanism, dict):
        parts.extend(mechanism.get(field, "") for field in MECHANISM_FIELDS)
    parts.extend(seed.get("primitives", []))
    return " ".join(str(item) for item in parts)


def _validate_contract(contract: Any) -> None:
    if not isinstance(contract, dict):
        raise PipelineError("contract must be an object")
    for field in ("task_id", "question", "baseline"):
        if not isinstance(contract.get(field), str) or not contract[field].strip():
            raise PipelineError(f"contract.{field} must be non-empty text")
    constraints = contract.get("hard_constraints")
    if not isinstance(constraints, list) or not constraints:
        raise PipelineError("contract.hard_constraints must be a non-empty list")
    tests = contract.get("success_tests")
    if not isinstance(tests, list) or not tests:
        raise PipelineError("contract.success_tests must be a non-empty list")
    if contract.get("novelty_scope") not in {"panel", "corpus"}:
        raise PipelineError("contract.novelty_scope must be panel or corpus")


def _validate_seed(seed: Any, index: int) -> None:
    path = f"kuds.ideas[{index}]"
    if not isinstance(seed, dict):
        raise PipelineError(f"{path} must be an object")
    for field in ("id", "title", "thesis", "invariant", "kill_test"):
        if not isinstance(seed.get(field), str) or not seed[field].strip():
            raise PipelineError(f"{path}.{field} must be non-empty text")
    for field in ("primitives", "assumptions", "source_ops"):
        value = seed.get(field)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise PipelineError(f"{path}.{field} must be a non-empty text list")
    mechanism = seed.get("mechanism")
    if not isinstance(mechanism, dict):
        raise PipelineError(f"{path}.mechanism must be an object")
    for field in MECHANISM_FIELDS:
        if not isinstance(mechanism.get(field), str) or not mechanism[field].strip():
            raise PipelineError(f"{path}.mechanism.{field} must be non-empty text")


def score_and_select_seeds(
    baseline_snapshot: list[str], ideas: list[dict[str, Any]], count: int = 4
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a novel and mutually diverse seed set deterministically."""

    if count <= 0 or len(ideas) < count:
        raise PipelineError(f"at least {count} KUDS ideas are required")
    if not baseline_snapshot or not all(
        isinstance(item, str) and item.strip() for item in baseline_snapshot
    ):
        raise PipelineError("kuds.baseline_snapshot must be a non-empty text list")

    scored: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, seed in enumerate(ideas):
        _validate_seed(seed, index)
        if seed["id"] in seen_ids:
            raise PipelineError("KUDS idea ids must be unique")
        seen_ids.add(seed["id"])
        text = _seed_text(seed)
        novelty = 1.0 - max(_jaccard(text, baseline) for baseline in baseline_snapshot)
        transform = seed["mechanism"]["transformation"]
        state = seed["mechanism"]["state_changed"]
        structural = 1.0 if _tokens(transform) and _tokens(state) else 0.0
        source_diversity = min(len(set(seed["source_ops"])) / 3.0, 1.0)
        base_score = 0.55 * novelty + 0.30 * structural + 0.15 * source_diversity
        scored.append(
            {
                "seed": seed,
                "novelty": round(novelty, 6),
                "structural": round(structural, 6),
                "source_diversity": round(source_diversity, 6),
                "base_score": round(base_score, 6),
                "hash": _canonical_hash(seed),
            }
        )

    selected: list[dict[str, Any]] = []
    remaining = scored[:]
    while len(selected) < count:
        def key(item: dict[str, Any]) -> tuple[float, float, str]:
            if not selected:
                diversity = 1.0
            else:
                diversity = min(
                    1.0 - _jaccard(_seed_text(item["seed"]), _seed_text(other))
                    for other in selected
                )
            combined = 0.70 * item["base_score"] + 0.30 * diversity
            return combined, diversity, item["hash"]

        winner = sorted(remaining, key=lambda item: (-key(item)[0], -key(item)[1], key(item)[2]))[0]
        selected.append(winner["seed"])
        remaining.remove(winner)

    report = [
        {
            "id": item["seed"]["id"],
            "novelty": item["novelty"],
            "structural": item["structural"],
            "source_diversity": item["source_diversity"],
            "base_score": item["base_score"],
            "selected": item["seed"] in selected,
        }
        for item in sorted(scored, key=lambda item: (-item["base_score"], item["hash"]))
    ]
    return selected, report


def _json_prompt(title: str, instructions: str, payload: dict[str, Any]) -> str:
    return (
        f"{title}\n\n{instructions}\n\n"
        "Return one valid JSON object only. Do not use Markdown fences.\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _kuds_prompt(contract: dict[str, Any], n_candidates: int) -> str:
    return _json_prompt(
        "KUDS DIVERGENCE STAGE",
        f"""Infer 5 ordinary baseline solutions, then generate exactly {n_candidates}
structurally different ideas. Use inversion, contextual displacement, distant
analogy, conceptual conflict, recombination, and constraint reification. Novelty
must change a causal transformation or the state being changed; decorative
metaphors and renamed baselines are invalid.

Output schema:
{{"baseline_snapshot":["..."],"ideas":[{{"id":"K1","title":"...",
"thesis":"...","primitives":["..."],"assumptions":["..."],
"invariant":"...","mechanism":{{"input":"...","trigger":"...",
"transformation":"...","state_changed":"...","output":"...",
"observable_prediction":"...","falsifier":"..."}},"kill_test":"...",
"source_ops":["inversion","analogy:domain"]}}]}}""",
        {"contract": contract},
    )


def _proposal_prompt(
    contract: dict[str, Any], role: str, seed: dict[str, Any]
) -> str:
    return _json_prompt(
        f"RAZLOM-4 BLIND COMMIT — {role}",
        """Develop the assigned KUDS seed according to your role. Do not assume
other proposals exist. Preserve a concrete causal mechanism and expose the
role-specific cost or weakness. Return a ProposalCard with exactly these fields:
id, author_role, thesis, primitives, assumptions, invariant, mechanism, kill_test.
The mechanism has input, trigger, transformation, state_changed, output,
observable_prediction, and falsifier. author_role must equal the requested role.""",
        {"contract": contract, "role": role, "kuds_seed": seed},
    )


def _forge_prompt(
    contract: dict[str, Any], own: dict[str, Any], anonymous: list[dict[str, Any]]
) -> str:
    return _json_prompt(
        f"RAZLOM-4 CONFLICT AND FORGE — {own['author_role']}",
        """Attack all three foreign proposals. Identify a shared hidden
assumption, preserve at least one foreign invariant, explicitly discard one
assumption copied verbatim from your own frozen proposal, and create one causal
mutation. Use one allowed mutation_operator: invert, gate, amputate, transplant,
reify, delay_or_expire, or price_or_budget. A compromise or relabel is invalid.

Return one DeltaCandidate with: id, author_role, source_roles, title,
discarded_self_assumption, shared_hidden_assumption, mutation_operator,
causal_operator, mechanism, ablation, prediction, experiment,
experiment_result, rollback, simplicity, reversibility. Use experiment_result
"unrun" unless the input contains actual observed evidence. source_roles must
contain at least three protocol roles.""",
        {
            "contract": contract,
            "own_frozen_proposal": own,
            "anonymous_foreign_proposals": anonymous,
        },
    )


def _review_prompt(
    contract: dict[str, Any], reviewer_role: str, candidates: list[dict[str, Any]]
) -> str:
    return _json_prompt(
        f"RAZLOM-4 GUILLOTINE — {reviewer_role}",
        """Review every supplied non-self candidate. Return {"reviews":[...]}
with exactly one review per candidate. Each review has reviewer_role,
candidate_id, hard_failures, kill_test, kill_test_result, equivalence_verdict,
aligned_fields, distinguishing_test, role_score, residual_risk. A hard failure
is valid only when tied to a known hard-constraint id and kill_test_result is
"fail". Use "unrun" when evidence is absent. equivalence_verdict is equivalent,
distinct, or uncertain. role_score is from 0 to 1.""",
        {
            "contract": contract,
            "reviewer_role": reviewer_role,
            "anonymous_non_self_candidates": candidates,
        },
    )


def _anonymize(records: list[dict[str, Any]], *, hide: set[str]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in record.items() if key not in hide}
        for record in records
    ]


def _snake_key(key: str) -> str:
    key = key.replace("-", "_").replace(" ", "_")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", key).casefold()


def _normalize_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {_snake_key(str(key)): _normalize_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_keys(item) for item in value]
    return value


def _unwrap_card(value: dict[str, Any], names: set[str]) -> dict[str, Any]:
    normalized = _normalize_keys(value)
    for name in names:
        nested = normalized.get(name)
        if isinstance(nested, dict):
            return nested
    if len(normalized) == 1:
        nested = next(iter(normalized.values()))
        if isinstance(nested, dict):
            return nested
    return normalized


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            parsed = float(match.group())
            if parsed > 1 and parsed <= 100:
                parsed /= 100
            return parsed
    return None


def _normalize_candidate(
    raw: dict[str, Any], *, role: str, fallback_id: str
) -> dict[str, Any]:
    candidate = _unwrap_card(
        raw,
        {"candidate", "delta_candidate", "deltacandidate", "mutation"},
    )
    aliases = {
        "canonical_mechanism": "mechanism",
        "causal_mechanism": "mechanism",
        "new_mechanism": "mechanism",
        "mechanism_card": "mechanism",
        "simplicity_score": "simplicity",
        "reversibility_score": "reversibility",
    }
    for source, target in aliases.items():
        if target not in candidate and source in candidate:
            candidate[target] = candidate[source]
    scores = candidate.get("scores")
    if isinstance(scores, dict):
        candidate.setdefault("simplicity", scores.get("simplicity"))
        candidate.setdefault("reversibility", scores.get("reversibility"))
    candidate["id"] = candidate.get("id") or fallback_id
    candidate["author_role"] = role
    candidate["source_roles"] = list(ROLES)

    mechanism = candidate.get("mechanism")
    if not isinstance(mechanism, dict):
        transformation = mechanism if isinstance(mechanism, str) and mechanism.strip() else None
        transformation = transformation or candidate.get("causal_operator") or candidate.get("title")
        prediction = candidate.get("prediction")
        ablation = candidate.get("ablation")
        candidate["mechanism"] = {
            "input": candidate.get("input") or "Inputs defined by the task contract",
            "trigger": candidate.get("trigger") or candidate.get("shared_hidden_assumption") or "Activation condition in the proposed experiment",
            "transformation": transformation or "Apply the candidate causal operator",
            "state_changed": candidate.get("state_changed") or candidate.get("causal_operator") or "Decision state governed by the candidate",
            "output": candidate.get("output") or prediction or "Observable candidate outcome",
            "observable_prediction": prediction or "The candidate must outperform the registered baseline",
            "falsifier": ablation or "The registered prediction does not hold under ablation",
        }
    for field in ("simplicity", "reversibility"):
        parsed = _number(candidate.get(field))
        candidate[field] = max(0.0, min(1.0, parsed)) if parsed is not None else 0.0
    return candidate


def _normalize_reviews(
    raw_reviews: list[Any],
    *,
    reviewer_role: str,
    expected_candidates: list[dict[str, Any]],
    constraint_ids: set[str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, (raw, candidate) in enumerate(zip(raw_reviews, expected_candidates)):
        if not isinstance(raw, dict):
            raise PipelineError(f"review:{reviewer_role}[{index}] must be an object")
        review = _unwrap_card(raw, {"review", "candidate_review", "assessment"})
        review["reviewer_role"] = reviewer_role
        # The input and output arrays are positionally bound. This prevents a
        # model typo from producing duplicate or forbidden reviewer pairs.
        review["candidate_id"] = candidate["id"]

        failures = review.get("hard_failures", [])
        if isinstance(failures, str):
            failures = [item for item in constraint_ids if item in failures]
        elif isinstance(failures, list):
            failures = [item for item in failures if isinstance(item, str) and item in constraint_ids]
        else:
            failures = []
        review["hard_failures"] = failures

        result_aliases = {
            "passed": "pass",
            "failed": "fail",
            "not_run": "unrun",
            "not run": "unrun",
            "unknown": "unrun",
        }
        kill_result = review.get("kill_test_result")
        if isinstance(kill_result, str):
            kill_result = result_aliases.get(kill_result.casefold(), kill_result.casefold())
        if kill_result not in {"pass", "fail", "unrun"}:
            kill_result = "unrun"
        review["kill_test_result"] = kill_result
        if failures and kill_result != "fail":
            # An unexecuted or malformed test cannot establish a hard veto.
            review["hard_failures"] = []

        if not isinstance(review.get("kill_test"), str) or not review["kill_test"].strip():
            mechanism = candidate.get("mechanism")
            fallback_test = mechanism.get("falsifier") if isinstance(mechanism, dict) else None
            review["kill_test"] = (
                fallback_test
                or candidate.get("experiment")
                or "Run the candidate's registered falsifier before accepting this review."
            )

        if not isinstance(review.get("aligned_fields"), str) or not review["aligned_fields"].strip():
            review["aligned_fields"] = (
                "Canonical alignment was not supplied; equivalence remains uncertain."
            )
            review["equivalence_verdict"] = "uncertain"
        review.setdefault("distinguishing_test", "")
        score = _number(review.get("role_score"))
        if score is not None:
            review["role_score"] = max(0.0, min(1.0, score))
        normalized.append(review)
    return normalized


def build_idea_dossier(
    session: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any] | None:
    selected_id = selection.get("selected_candidate_id")
    if not selected_id:
        return None
    candidate = next(item for item in session["candidates"] if item["id"] == selected_id)
    reviews = [item for item in session["reviews"] if item["candidate_id"] == selected_id]
    evaluation = next(
        item for item in selection["evaluations"] if item["candidate_id"] == selected_id
    )
    return {
        "id": selected_id,
        "title": candidate.get("title", candidate["causal_operator"]),
        "status": selection["status"],
        "novelty_label": selection.get("novelty_label"),
        "causal_operator": candidate["causal_operator"],
        "mechanism": candidate["mechanism"],
        "discarded_assumption": candidate["discarded_self_assumption"],
        "hidden_assumption_broken": candidate["shared_hidden_assumption"],
        "prediction": candidate["prediction"],
        "ablation": candidate["ablation"],
        "experiment": candidate["experiment"],
        "rollback": candidate["rollback"],
        "external_scores": {
            "minimum": evaluation["min_external_score"],
            "mean": evaluation["mean_external_score"],
        },
        "falsifiers": [review["kill_test"] for review in reviews],
        "residual_risks": [review["residual_risk"] for review in reviews],
    }


@dataclass
class FullPipeline:
    provider: JsonProvider
    n_candidates: int = 12

    def run(self, contract: dict[str, Any]) -> dict[str, Any]:
        _validate_contract(contract)
        if self.n_candidates < 4:
            raise PipelineError("n_candidates must be at least 4")

        kuds = self.provider.complete_json(
            _kuds_prompt(contract, self.n_candidates),
            stage="kuds_divergence",
            temperature=0.9,
        )
        if not isinstance(kuds, dict):
            raise PipelineError("KUDS provider response must be an object")
        ideas = kuds.get("ideas")
        if not isinstance(ideas, list) or len(ideas) != self.n_candidates:
            raise PipelineError(
                f"KUDS must return exactly {self.n_candidates} ideas"
            )
        seeds, seed_scores = score_and_select_seeds(
            kuds.get("baseline_snapshot"), ideas, count=4
        )
        assignment_offset = int(_canonical_hash(contract)[:8], 16) % len(ROLES)
        assigned_seeds = seeds[assignment_offset:] + seeds[:assignment_offset]
        seed_assignments = {
            role: seed["id"] for role, seed in zip(ROLES, assigned_seeds)
        }

        proposals: list[dict[str, Any]] = []
        for role, seed in zip(ROLES, assigned_seeds):
            proposal = self.provider.complete_json(
                _proposal_prompt(contract, role, seed),
                stage=f"proposal:{role}",
                temperature=0.65,
            )
            if not isinstance(proposal, dict):
                raise PipelineError(f"proposal:{role} must return an object")
            proposal = _unwrap_card(proposal, {"proposal", "proposal_card", "proposalcard"})
            proposal["author_role"] = role
            proposals.append(proposal)

        candidates: list[dict[str, Any]] = []
        for role_index, role in enumerate(ROLES, 1):
            own = next(item for item in proposals if item["author_role"] == role)
            foreign = [item for item in proposals if item["author_role"] != role]
            anonymous = _anonymize(foreign, hide={"author_role"})
            candidate = self.provider.complete_json(
                _forge_prompt(contract, own, anonymous),
                stage=f"forge:{role}",
                temperature=0.75,
            )
            if not isinstance(candidate, dict):
                raise PipelineError(f"forge:{role} must return an object")
            candidate = _normalize_candidate(
                candidate, role=role, fallback_id=f"D{role_index}"
            )
            if any(existing["id"] == candidate["id"] for existing in candidates):
                candidate["id"] = f"D{role_index}"
            candidates.append(candidate)

        reviews: list[dict[str, Any]] = []
        constraint_ids = {
            item["id"]
            for item in contract["hard_constraints"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        for role in ROLES:
            non_self = [item for item in candidates if item["author_role"] != role]
            anonymous = _anonymize(non_self, hide={"author_role"})
            response = self.provider.complete_json(
                _review_prompt(contract, role, anonymous),
                stage=f"review:{role}",
                temperature=0.2,
            )
            role_reviews = response.get("reviews") if isinstance(response, dict) else None
            if not isinstance(role_reviews, list) or len(role_reviews) != 3:
                raise PipelineError(f"review:{role} must return exactly three reviews")
            reviews.extend(
                _normalize_reviews(
                    role_reviews,
                    reviewer_role=role,
                    expected_candidates=non_self,
                    constraint_ids=constraint_ids,
                )
            )

        session = {
            "contract": contract,
            "proposals": proposals,
            "candidates": candidates,
            "reviews": reviews,
        }
        errors = validate_session(session)
        if errors:
            raise PipelineError("provider produced an invalid session:\n- " + "\n- ".join(errors))
        selection = select_candidate(session)
        artifact = {
            "method": "kuds_razlom4_full",
            "version": "0.2.3",
            "model_calls": 13,
            "kuds": {
                "baseline_snapshot": kuds["baseline_snapshot"],
                "pool_size": len(ideas),
                "selected_seed_ids": [item["id"] for item in seeds],
                "seed_assignments": seed_assignments,
                "selection_scores": seed_scores,
                "ideas": ideas,
            },
            "session": session,
            "selection": selection,
            "idea_dossier": build_idea_dossier(session, selection),
        }
        usage_summary = getattr(self.provider, "usage_summary", None)
        if callable(usage_summary):
            artifact["usage"] = usage_summary()
        return artifact


def _extract_json(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise PipelineError("provider returned empty message content")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise PipelineError("provider did not return a JSON object") from exc
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as nested:
            raise PipelineError(f"provider returned malformed JSON: {nested}") from nested
    if not isinstance(value, dict):
        raise PipelineError("provider JSON response must be an object")
    return value


@dataclass
class OpenRouterProvider:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: int = 120
    reasoning_effort: str = "none"
    max_retries: int = 1

    def __post_init__(self) -> None:
        self.usage_records: list[dict[str, Any]] = []

    @staticmethod
    def _max_tokens(stage: str) -> int:
        if stage == "kuds_divergence":
            return 10000
        if stage.startswith("proposal:"):
            return 3000
        return 5000

    def complete_json(
        self, prompt: str, *, stage: str, temperature: float
    ) -> dict[str, Any]:
        last_error: PipelineError | None = None
        for attempt in range(self.max_retries + 1):
            retrying = attempt > 0
            effort = "none" if retrying else self.reasoning_effort
            max_tokens = self._max_tokens(stage)
            if retrying:
                max_tokens = int(max_tokens * 1.5)
            body = json.dumps(
                {
                    "model": self.model,
                    "temperature": min(temperature, 0.2) if retrying else temperature,
                    "max_tokens": max_tokens,
                    "reasoning": {"effort": effort, "exclude": True},
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return one complete, strictly valid JSON object. "
                                "Do not spend the output budget on hidden reasoning."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                self.base_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "RAZLOM-4 KUDS Pipeline",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise PipelineError(f"{stage} provider request failed: {exc}") from exc

            try:
                choice = payload["choices"][0]
                content = choice["message"].get("content")
                finish_reason = choice.get("finish_reason")
            except (KeyError, IndexError, TypeError, AttributeError):
                content = None
                finish_reason = None

            usage = payload.get("usage") if isinstance(payload, dict) else None
            if isinstance(usage, dict):
                details = usage.get("completion_tokens_details") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
                reported_cost = usage.get("cost")
                estimated_cost = (
                    prompt_tokens * GLM51_INPUT_USD_PER_MILLION
                    + completion_tokens * GLM51_OUTPUT_USD_PER_MILLION
                ) / 1_000_000
                self.usage_records.append(
                    {
                        "stage": stage,
                        "attempt": attempt + 1,
                        "finish_reason": finish_reason,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
                        "reported_cost": reported_cost,
                        "estimated_cost_usd": round(estimated_cost, 8),
                    }
                )
            try:
                return _extract_json(content)
            except PipelineError as exc:
                last_error = PipelineError(
                    f"{stage} attempt {attempt + 1} returned unusable JSON "
                    f"(finish_reason={finish_reason!r}): {exc}"
                )
        assert last_error is not None
        raise last_error

    def usage_summary(self) -> dict[str, Any]:
        reported = [
            float(item["reported_cost"])
            for item in self.usage_records
            if isinstance(item.get("reported_cost"), (int, float))
        ]
        return {
            "model": self.model,
            "requests": len(self.usage_records),
            "prompt_tokens": sum(item["prompt_tokens"] for item in self.usage_records),
            "completion_tokens": sum(
                item["completion_tokens"] for item in self.usage_records
            ),
            "reasoning_tokens": sum(
                item["reasoning_tokens"] for item in self.usage_records
            ),
            "reported_cost": round(sum(reported), 8) if reported else None,
            "estimated_cost_usd": round(
                sum(item["estimated_cost_usd"] for item in self.usage_records), 8
            ),
            "pricing_assumption_usd_per_million": {
                "input": GLM51_INPUT_USD_PER_MILLION,
                "output_including_reasoning": GLM51_OUTPUT_USD_PER_MILLION,
            },
            "stages": self.usage_records,
        }


class CheckpointProvider:
    """Cache successful stages and their usage so interrupted runs can resume."""

    def __init__(
        self,
        provider: JsonProvider,
        path: str | Path,
        *,
        allow_stale_prefixes: tuple[str, ...] = (),
    ) -> None:
        self.provider = provider
        self.path = Path(path)
        self.allow_stale_prefixes = allow_stale_prefixes
        self.data: dict[str, Any] = {"responses": {}, "usage_records": []}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data.update(loaded)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def complete_json(
        self, prompt: str, *, stage: str, temperature: float
    ) -> dict[str, Any]:
        fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self.data.get("responses", {}).get(stage)
        hash_matches = isinstance(cached, dict) and cached.get("prompt_hash") == fingerprint
        stale_allowed = any(stage.startswith(prefix) for prefix in self.allow_stale_prefixes)
        if isinstance(cached, dict) and (hash_matches or stale_allowed):
            response = cached.get("response")
            if isinstance(response, dict):
                return json.loads(json.dumps(response, ensure_ascii=False))

        records = getattr(self.provider, "usage_records", None)
        before = len(records) if isinstance(records, list) else 0
        try:
            response = self.provider.complete_json(
                prompt, stage=stage, temperature=temperature
            )
        finally:
            records = getattr(self.provider, "usage_records", None)
            if isinstance(records, list) and len(records) > before:
                self.data.setdefault("usage_records", []).extend(records[before:])
                self._save()
        self.data.setdefault("responses", {})[stage] = {
            "prompt_hash": fingerprint,
            "response": response,
        }
        self._save()
        return response

    def usage_summary(self) -> dict[str, Any]:
        records = self.data.get("usage_records", [])
        reported = [
            float(item["reported_cost"])
            for item in records
            if isinstance(item.get("reported_cost"), (int, float))
        ]
        return {
            "model": getattr(self.provider, "model", None),
            "requests": len(records),
            "cached_stages": len(self.data.get("responses", {})),
            "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in records),
            "completion_tokens": sum(
                int(item.get("completion_tokens") or 0) for item in records
            ),
            "reasoning_tokens": sum(
                int(item.get("reasoning_tokens") or 0) for item in records
            ),
            "reported_cost": round(sum(reported), 8) if reported else None,
            "estimated_cost_usd": round(
                sum(float(item.get("estimated_cost_usd") or 0) for item in records), 8
            ),
            "pricing_assumption_usd_per_million": {
                "input": GLM51_INPUT_USD_PER_MILLION,
                "output_including_reasoning": GLM51_OUTPUT_USD_PER_MILLION,
            },
            "stages": records,
        }


class ReplayProvider:
    """Replay ordered provider responses from a JSON artifact."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.index = 0

    @classmethod
    def from_path(cls, path: str | Path) -> "ReplayProvider":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise PipelineError("replay file must contain a JSON list")
        return cls(value)

    def complete_json(
        self, prompt: str, *, stage: str, temperature: float
    ) -> dict[str, Any]:
        del prompt, temperature
        if self.index >= len(self.records):
            raise PipelineError(f"replay exhausted before stage {stage}")
        record = self.records[self.index]
        self.index += 1
        if record.get("stage") != stage:
            raise PipelineError(
                f"replay stage mismatch: expected {stage}, got {record.get('stage')}"
            )
        response = record.get("response")
        if not isinstance(response, dict):
            raise PipelineError(f"replay response for {stage} must be an object")
        return json.loads(json.dumps(response, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate with KUDS, then refine and select with RAZLOM-4"
    )
    parser.add_argument("contract", help="path to a RAZLOM-4 contract JSON")
    parser.add_argument("--output", "-o", required=True, help="full run artifact path")
    parser.add_argument("--candidates", type=int, default=12, help="KUDS pool size")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--replay", help="ordered offline provider responses")
    source.add_argument("--api-key", help="OpenRouter API key (prefer the environment)")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    if args.replay:
        provider: JsonProvider = ReplayProvider.from_path(args.replay)
    else:
        api_key = args.api_key or os.getenv(args.api_key_env)
        if not api_key:
            parser.error(
                f"set {args.api_key_env}, pass --api-key, or use --replay for offline execution"
            )
        provider = OpenRouterProvider(api_key=api_key, model=args.model)

    try:
        artifact = FullPipeline(provider, n_candidates=args.candidates).run(contract)
    except PipelineError as exc:
        print(f"pipeline failed: {exc}")
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": artifact["selection"]["status"],
                "selected_candidate_id": artifact["selection"]["selected_candidate_id"],
                "selected_seed_ids": artifact["kuds"]["selected_seed_ids"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
