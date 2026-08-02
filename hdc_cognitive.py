"""Runnable reference prototype for HDC Cognitive Architecture v3.

This module intentionally implements the falsifiable single-agent core, not a
claim of consciousness or language understanding.  It has deterministic HDC
encoding, a mutable semantic overlay, bounded STM/LTM, simple regulators,
versioned consolidation, rollback, and equal-budget evaluation helpers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


UINT32_MAX = (1 << 32) - 1


class CapacityError(RuntimeError):
    """Raised when a declared memory bound is reached without safe eviction."""


def tokenize(text: str) -> list[str]:
    """Small deterministic Unicode tokenizer used by the reference demo."""

    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def _stable_bytes(*parts: object) -> bytes:
    return "\x1f".join(str(part) for part in parts).encode("utf-8")


class HDCEncoder:
    """Deterministic bipolar HDC encoder with stable identity vectors."""

    def __init__(self, dimension: int = 2048, seed: str = "hdc-ca-v3") -> None:
        if dimension < 256 or dimension % 8:
            raise ValueError("dimension must be >=256 and divisible by 8")
        self.dimension = dimension
        self.seed = seed
        self._atom_cache: dict[tuple[str, str], np.ndarray] = {}
        self._token_cache: dict[str, np.ndarray] = {}

    def atom(self, namespace: str, value: object) -> np.ndarray:
        key = (namespace, str(value))
        cached = self._atom_cache.get(key)
        if cached is not None:
            return cached
        needed = self.dimension // 8
        blocks = []
        counter = 0
        prefix = _stable_bytes(self.seed, namespace, value)
        while len(blocks) * 32 < needed:
            blocks.append(hashlib.sha256(prefix + counter.to_bytes(4, "big")).digest())
            counter += 1
        raw = np.frombuffer(b"".join(blocks)[:needed], dtype=np.uint8)
        bits = np.unpackbits(raw)[: self.dimension]
        vector = (bits.astype(np.int8) * 2 - 1).astype(np.int8)
        vector.setflags(write=False)
        self._atom_cache[key] = vector
        return vector

    def bundle(self, vectors: Sequence[np.ndarray], tie_name: str = "bundle") -> np.ndarray:
        if not vectors:
            return self.atom("empty", tie_name).copy()
        total = np.sum(np.stack(vectors).astype(np.int32), axis=0)
        result = np.sign(total).astype(np.int8)
        ties = total == 0
        if np.any(ties):
            tie = self.atom("tie", tie_name)
            result[ties] = tie[ties]
        return result

    @staticmethod
    def bind(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (left * right).astype(np.int8)

    def permute(self, vector: np.ndarray, position: int) -> np.ndarray:
        return np.roll(vector, (137 * position) % self.dimension).astype(np.int8)

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(left.astype(np.int32), right.astype(np.int32))) / self.dimension

    def token_identity(self, token: str) -> np.ndarray:
        cached = self._token_cache.get(token)
        if cached is not None:
            return cached
        marked = f"⟨{token}⟩"
        grams = sorted(
            {
                marked[index : index + size]
                for size in (3, 4, 5)
                for index in range(max(0, len(marked) - size + 1))
            }
        )[:96]
        if not grams:
            vector = self.atom("token", token).copy()
        else:
            vector = self.bundle([self.atom("ngram", gram) for gram in grams], f"token:{token}")
        vector.setflags(write=False)
        self._token_cache[token] = vector
        return vector


@dataclass
class OutcomeTable:
    max_outcomes: int = 8
    counts: dict[str, int] = field(default_factory=dict)

    def observe(self, token: str) -> None:
        if token in self.counts:
            self.counts[token] += 1
        elif len(self.counts) < self.max_outcomes:
            self.counts[token] = 1
        else:
            # Deterministic Space-Saving: minimum count, then lexical maximum.
            minimum = min(self.counts.values())
            victim = max(key for key, value in self.counts.items() if value == minimum)
            del self.counts[victim]
            self.counts[token] = minimum + 1
        if self.counts and max(self.counts.values()) >= UINT32_MAX:
            self.counts = {key: max(1, value // 2) for key, value in self.counts.items()}

    def top(self) -> tuple[str | None, float]:
        if not self.counts:
            return None, 0.0
        best_count = max(self.counts.values())
        best = min(key for key, count in self.counts.items() if count == best_count)
        return best, best_count / sum(self.counts.values())

    def l1_distance(self, other: "OutcomeTable") -> float:
        keys = set(self.counts) | set(other.counts)
        left_total = max(1, sum(self.counts.values()))
        right_total = max(1, sum(other.counts.values()))
        return sum(
            abs(self.counts.get(key, 0) / left_total - other.counts.get(key, 0) / right_total)
            for key in keys
        )


@dataclass
class Prototype:
    record_id: int
    key_hv: np.ndarray
    semantic_key_hv: np.ndarray | None = None
    outcomes: OutcomeTable = field(default_factory=OutcomeTable)
    support: int = 0
    created_tick: int = 0
    last_used_tick: int = 0
    parents: tuple[int, ...] = ()
    witnesses: list[tuple[tuple[str, ...], str]] = field(default_factory=list)
    store: str = "STM"

    def observe(self, context: Sequence[str], target: str, tick: int) -> None:
        self.outcomes.observe(target)
        self.support += 1
        self.last_used_tick = tick
        witness = (tuple(context), target)
        if witness not in self.witnesses:
            if len(self.witnesses) < 16:
                self.witnesses.append(witness)
            else:
                # Keep the first eight and the latest eight deterministically.
                self.witnesses = self.witnesses[:8] + self.witnesses[-7:] + [witness]


@dataclass
class PatternStats:
    exposures: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    mode: str = "DISCOVERY"

    def update(self, success: bool) -> None:
        self.exposures += 1
        if success:
            self.successful_uses += 1
        else:
            self.failed_uses += 1
        confidence = self.successful_uses / max(1, self.successful_uses + self.failed_uses)
        if self.exposures >= 8 and self.successful_uses >= 3 and confidence >= 0.70:
            self.mode = "PRACTICE"
        elif self.exposures >= 3 and confidence >= 0.45:
            self.mode = "MEMORIZATION"
        else:
            self.mode = "DISCOVERY"
        if self.mode == "PRACTICE" and self.failed_uses >= 2:
            self.mode = "MEMORIZATION"


@dataclass
class Hormones:
    dopamine: float = 0.50
    serotonin: float = 0.50
    cortisol: float = 0.20
    adrenaline: float = 0.10
    two_ag: float = 0.10

    def update(self, *, success: bool, error: float, loop_score: float, urgency: float = 0.0) -> None:
        baseline = {
            "dopamine": 0.50,
            "serotonin": 0.50,
            "cortisol": 0.20,
            "adrenaline": 0.10,
            "two_ag": 0.10,
        }
        release = {
            "dopamine": 0.01 if success else 0.0,
            "serotonin": 0.005 if success and error < 0.25 else 0.0,
            "cortisol": min(0.02, 0.02 * error),
            "adrenaline": min(0.02, 0.02 * urgency),
            "two_ag": min(0.03, 0.03 * loop_score),
        }
        for name in baseline:
            value = getattr(self, name)
            value += release[name] - 0.08 * (value - baseline[name])
            # Anti-windup leaves recovery room in both directions.
            setattr(self, name, min(0.95, max(0.05, value)))

    def memory_gain(self) -> float:
        return max(-0.10, min(0.10, 0.10 * (self.dopamine - self.cortisol)))

    def as_dict(self) -> dict[str, float]:
        return {name: round(float(getattr(self, name)), 8) for name in self.__dataclass_fields__}


@dataclass
class Resources:
    energy: float = 1.0
    sleep_debt: float = 0.0

    def spend(self, amount: float) -> bool:
        if amount > self.energy:
            return False
        self.energy = max(0.0, self.energy - amount)
        self.sleep_debt = min(1.0, self.sleep_debt + amount * 0.5)
        return True

    def recover(self, amount: float = 0.05) -> None:
        self.energy = min(1.0, self.energy + amount)
        self.sleep_debt = max(0.0, self.sleep_debt - amount)


@dataclass
class Prediction:
    token: str | None
    confidence: float
    similarity: float
    store: str | None
    record_id: int | None


class HDCCognitiveSystem:
    """Bounded online associative memory implementing the runnable v3 core."""

    def __init__(
        self,
        *,
        dimension: int = 2048,
        seed: str = "hdc-ca-v3",
        context_length: int = 4,
        stm_capacity: int = 512,
        ltm_capacity: int = 2048,
        max_outcomes: int = 8,
        learn_threshold: float = 0.82,
        query_threshold: float = 0.20,
        promotion_support: int = 3,
        dynamic_semantics: bool = False,
        positional_binding: bool = True,
        auto_consolidate: bool = True,
        semantic_drift_fraction: float = 0.01,
        neighbor_count: int = 5,
        stable_key_weight: float = 0.80,
    ) -> None:
        self.encoder = HDCEncoder(dimension, seed)
        self.seed = seed
        self.context_length = context_length
        self.stm_capacity = stm_capacity
        self.ltm_capacity = ltm_capacity
        self.max_outcomes = max_outcomes
        self.learn_threshold = learn_threshold
        self.query_threshold = query_threshold
        self.promotion_support = promotion_support
        self.dynamic_semantics = dynamic_semantics
        self.positional_binding = positional_binding
        self.auto_consolidate = auto_consolidate
        if not 0.0 < semantic_drift_fraction <= 0.02:
            raise ValueError("semantic_drift_fraction must be in (0, 0.02]")
        self.semantic_drift_fraction = semantic_drift_fraction
        if neighbor_count < 1 or neighbor_count > 32:
            raise ValueError("neighbor_count must be between 1 and 32")
        self.neighbor_count = neighbor_count
        if not 0.5 <= stable_key_weight <= 1.0:
            raise ValueError("stable_key_weight must be in [0.5, 1.0]")
        self.stable_key_weight = stable_key_weight
        self.epoch = 0
        self.tick = 0
        self._next_record_id = 1
        self.semantic_acc: dict[str, np.ndarray] = {}
        self.stm: dict[int, Prototype] = {}
        self.ltm: dict[int, Prototype] = {}
        self.pattern_stats: dict[str, PatternStats] = {}
        self.hormones = Hormones()
        self.resources = Resources()
        self.coherence = 1.0
        self.event_log: list[dict[str, object]] = []
        self._snapshots: list[dict[str, object]] = []
        self._recent_predictions: list[str | None] = []
        self._health_history: list[tuple[float, float, float, float]] = []

    def _semantic_acc(self, token: str) -> np.ndarray:
        if token not in self.semantic_acc:
            self.semantic_acc[token] = (self.encoder.token_identity(token).astype(np.int16) * 8)
        return self.semantic_acc[token]

    def semantic_hv(self, token: str) -> np.ndarray:
        acc = self._semantic_acc(token)
        result = np.sign(acc).astype(np.int8)
        ties = acc == 0
        if np.any(ties):
            identity = self.encoder.token_identity(token)
            result[ties] = identity[ties]
        return result

    def token_hv(self, token: str) -> np.ndarray:
        if not self.dynamic_semantics:
            return self.encoder.token_identity(token).copy()
        return self.semantic_hv(token)

    def _padded_context(self, context: Sequence[str]) -> list[str]:
        trimmed = list(context)[-self.context_length :]
        return ["⟨PAD⟩"] * (self.context_length - len(trimmed)) + trimmed

    def stable_context_hv(self, context: Sequence[str]) -> np.ndarray:
        bound = []
        for position, token in enumerate(self._padded_context(context), 1):
            if self.positional_binding:
                role = self.encoder.atom("role", position)
                positioned = self.encoder.permute(self.encoder.token_identity(token), position)
                bound.append(self.encoder.bind(role, positioned))
            else:
                bound.append(self.encoder.token_identity(token))
        return self.encoder.bundle(bound, "stable-context")

    def semantic_context_hv(self, context: Sequence[str]) -> np.ndarray | None:
        if not self.dynamic_semantics:
            return None
        bound = []
        for position, token in enumerate(self._padded_context(context), 1):
            if self.positional_binding:
                role = self.encoder.atom("semantic-role", position)
                positioned = self.encoder.permute(self.semantic_hv(token), position)
                bound.append(self.encoder.bind(role, positioned))
            else:
                bound.append(self.semantic_hv(token))
        return self.encoder.bundle(bound, "semantic-context")

    def context_hv(self, context: Sequence[str]) -> np.ndarray:
        stable = self.stable_context_hv(context)
        semantic = self.semantic_context_hv(context)
        if semantic is None:
            return stable
        return self.encoder.bundle([stable, semantic], "combined-context")

    def _all_records(self) -> Iterable[Prototype]:
        yield from (self.stm[key] for key in sorted(self.stm))
        yield from (self.ltm[key] for key in sorted(self.ltm))

    def _record_similarity(
        self,
        stable_key_hv: np.ndarray,
        semantic_key_hv: np.ndarray | None,
        record: Prototype,
    ) -> float:
        stable = self.encoder.similarity(stable_key_hv, record.key_hv)
        if stable >= 0.999 or semantic_key_hv is None or record.semantic_key_hv is None:
            return stable
        semantic = self.encoder.similarity(semantic_key_hv, record.semantic_key_hv)
        return self.stable_key_weight * stable + (1.0 - self.stable_key_weight) * semantic

    def _nearest(
        self, stable_key_hv: np.ndarray, semantic_key_hv: np.ndarray | None = None
    ) -> tuple[Prototype | None, float]:
        best: Prototype | None = None
        best_similarity = -1.0
        for record in self._all_records():
            similarity = self._record_similarity(stable_key_hv, semantic_key_hv, record)
            if similarity > best_similarity or (
                math.isclose(similarity, best_similarity) and best is not None and record.record_id < best.record_id
            ):
                best = record
                best_similarity = similarity
        return best, best_similarity

    def _nearest_k(
        self, stable_key_hv: np.ndarray, semantic_key_hv: np.ndarray | None = None
    ) -> list[tuple[Prototype, float]]:
        scored = [
            (record, self._record_similarity(stable_key_hv, semantic_key_hv, record))
            for record in self._all_records()
        ]
        scored.sort(key=lambda item: (-item[1], item[0].record_id))
        return scored[: self.neighbor_count]

    def predict(self, context: Sequence[str]) -> Prediction:
        if not self.stm and not self.ltm:
            return Prediction(None, 0.0, 0.0, None, None)
        stable_key_hv = self.stable_context_hv(context)
        semantic_key_hv = self.semantic_context_hv(context)
        exact_records = [
            record
            for record in self._all_records()
            if self.encoder.similarity(stable_key_hv, record.key_hv) >= 0.999
        ]
        if exact_records:
            record = min(exact_records, key=lambda item: item.record_id)
            token, confidence = record.outcomes.top()
            return Prediction(token, confidence, 1.0, record.store, record.record_id)
        neighbors = self._nearest_k(stable_key_hv, semantic_key_hv)
        if not neighbors or neighbors[0][1] < self.query_threshold:
            similarity = neighbors[0][1] if neighbors else 0.0
            return Prediction(None, 0.0, max(0.0, similarity), None, None)
        if self.neighbor_count == 1:
            record, similarity = neighbors[0]
            token, outcome_confidence = record.outcomes.top()
            confidence = max(0.0, similarity) * outcome_confidence
            return Prediction(token, confidence, similarity, record.store, record.record_id)
        votes: dict[str, float] = {}
        total_weight = 0.0
        for record, similarity in neighbors:
            if similarity < self.query_threshold:
                continue
            weight = (similarity - self.query_threshold + 1e-6) ** 2
            outcome_total = max(1, sum(record.outcomes.counts.values()))
            total_weight += weight
            for token, count in record.outcomes.counts.items():
                votes[token] = votes.get(token, 0.0) + weight * count / outcome_total
        if not votes:
            return Prediction(None, 0.0, max(0.0, neighbors[0][1]), None, None)
        top_score = max(votes.values())
        token = min(key for key, value in votes.items() if math.isclose(value, top_score))
        record, similarity = neighbors[0]
        return Prediction(token, top_score / max(1e-12, total_weight), similarity, record.store, record.record_id)

    def _update_semantic(self, token: str, context_hv: np.ndarray, mode: str) -> None:
        if not self.dynamic_semantics:
            return
        eta = {"DISCOVERY": 1, "MEMORIZATION": 2, "PRACTICE": 4}[mode]
        acc = self._semantic_acc(token)
        candidate = np.clip(
            acc.astype(np.int32)
            + eta * context_hv.astype(np.int32),
            -32768,
            32767,
        ).astype(np.int16)
        old_hv = self.semantic_hv(token)
        new_hv = np.sign(candidate).astype(np.int8)
        ties = candidate == 0
        new_hv[ties] = self.encoder.token_identity(token)[ties]
        drift = 1.0 - self.encoder.similarity(old_hv, new_hv)
        if drift > 0.02:
            differing = np.flatnonzero(old_hv != new_hv)
            max_changes = max(1, int(self.encoder.dimension * self.semantic_drift_fraction))
            selected = differing[:max_changes]
            bounded = acc.copy()
            bounded[selected] = candidate[selected]
            candidate = bounded
        self.semantic_acc[token] = candidate

    def _new_record(
        self, key_hv: np.ndarray, semantic_key_hv: np.ndarray | None = None
    ) -> Prototype:
        if len(self.stm) >= self.stm_capacity:
            raise CapacityError("STM_CAPACITY_FULL")
        record = Prototype(
            record_id=self._next_record_id,
            key_hv=key_hv.copy(),
            semantic_key_hv=semantic_key_hv.copy() if semantic_key_hv is not None else None,
            outcomes=OutcomeTable(self.max_outcomes),
            created_tick=self.tick,
            last_used_tick=self.tick,
        )
        self._next_record_id += 1
        self.stm[record.record_id] = record
        return record

    def learn_pair(self, context: Sequence[str], target: str) -> dict[str, object]:
        if not self.resources.spend(0.0005):
            return {"status": "RESOURCE_EXHAUSTED"}
        before = self.predict(context)
        success = before.token == target
        error = 0.0 if success else 1.0
        pattern_key = hashlib.sha256(_stable_bytes(*context)).hexdigest()[:16]
        stats = self.pattern_stats.setdefault(pattern_key, PatternStats())
        stats.update(success)
        key_hv = self.stable_context_hv(context)
        semantic_key_hv = self.semantic_context_hv(context)
        record, similarity = self._nearest(key_hv, semantic_key_hv)
        if record is None or similarity < self.learn_threshold:
            record = self._new_record(key_hv, semantic_key_hv)
        record.observe(context, target, self.tick)
        learning_context = self.context_hv(context)
        self._update_semantic(target, learning_context, stats.mode)
        repeated = 0.0
        if self._recent_predictions[-3:] and len(set(self._recent_predictions[-3:])) == 1:
            repeated = 1.0
        self._recent_predictions.append(before.token)
        self._recent_predictions = self._recent_predictions[-8:]
        self.hormones.update(success=success, error=error, loop_score=repeated)
        self.tick += 1
        self._update_coherence(error=error, loop_score=repeated)
        return {
            "status": "LEARNED",
            "record_id": record.record_id,
            "mode": stats.mode,
            "predicted_before": before.token,
            "success_before": success,
        }

    def _update_coherence(self, *, error: float, loop_score: float) -> None:
        queue_pressure = 0.0
        resource_risk = max(0.0, 0.20 - self.resources.energy) / 0.20
        regulator_instability = max(0.0, self.hormones.two_ag - 0.80)
        self._health_history.append((error, loop_score, resource_risk, regulator_instability))
        self._health_history = self._health_history[-32:]
        error = sum(item[0] for item in self._health_history) / len(self._health_history)
        loop_score = sum(item[1] for item in self._health_history) / len(self._health_history)
        resource_risk = sum(item[2] for item in self._health_history) / len(self._health_history)
        regulator_instability = sum(item[3] for item in self._health_history) / len(self._health_history)
        self.coherence = max(
            0.0,
            min(
                1.0,
                1.0
                - 0.35 * error
                - 0.25 * loop_score
                - 0.15 * queue_pressure
                - 0.15 * resource_risk
                - 0.10 * regulator_instability,
            ),
        )

    def _checkpoint(self) -> None:
        self._snapshots.append(copy.deepcopy(self._state_payload()))
        self._snapshots = self._snapshots[-3:]

    def train_sequences(self, sequences: Sequence[Sequence[str] | str]) -> dict[str, object]:
        self._checkpoint()
        learned = 0
        rejected = 0
        for sequence in sequences:
            tokens = tokenize(sequence) if isinstance(sequence, str) else list(sequence)
            for index in range(1, len(tokens)):
                context = tokens[max(0, index - self.context_length) : index]
                try:
                    result = self.learn_pair(context, tokens[index])
                except CapacityError:
                    rejected += 1
                    continue
                if result["status"] == "LEARNED":
                    learned += 1
                else:
                    rejected += 1
        self.epoch += 1
        self.event_log.append({"epoch": self.epoch, "event": "TRAIN", "learned": learned, "rejected": rejected})
        auto_result = None
        utilization = len(self.stm) / max(1, self.stm_capacity)
        if self.auto_consolidate and (utilization >= 0.75 or self.resources.sleep_debt >= 0.80):
            auto_result = self.consolidate()
        return {
            "epoch": self.epoch,
            "learned": learned,
            "rejected": rejected,
            "auto_consolidation": auto_result,
            **self.summary(),
        }

    def promote(self) -> int:
        candidates = []
        for record in self.stm.values():
            _, confidence = record.outcomes.top()
            salience = min(
                1.0,
                0.35 * min(1.0, record.support / self.promotion_support)
                + 0.25 * confidence
                + 0.20 * 0.75
                + 0.20 * (0.5 + self.hormones.memory_gain()),
            )
            if record.support >= self.promotion_support and confidence >= 0.60 and salience >= 0.60:
                candidates.append(record.record_id)
        promoted = 0
        for record_id in sorted(candidates):
            if len(self.ltm) >= self.ltm_capacity:
                break
            record = self.stm.pop(record_id)
            record.store = "LTM"
            self.ltm[record_id] = record
            promoted += 1
        return promoted

    def consolidate(self, *, max_merges: int = 32, threshold: float = 0.94) -> dict[str, object]:
        self._checkpoint()
        promoted = self.promote()
        records = [self.ltm[key] for key in sorted(self.ltm)]
        candidates: list[tuple[float, int, int]] = []
        for index, left in enumerate(records):
            left_top, _ = left.outcomes.top()
            for right in records[index + 1 :]:
                right_top, _ = right.outcomes.top()
                if left_top != right_top or left.outcomes.l1_distance(right.outcomes) > 0.10:
                    continue
                similarity = self.encoder.similarity(left.key_hv, right.key_hv)
                if similarity >= threshold:
                    candidates.append((-similarity, left.record_id, right.record_id))
        candidates.sort()
        consumed: set[int] = set()
        merged = 0
        for _, left_id, right_id in candidates:
            if merged >= max_merges or left_id in consumed or right_id in consumed:
                continue
            if left_id not in self.ltm or right_id not in self.ltm:
                continue
            left, right = self.ltm[left_id], self.ltm[right_id]
            combined = self.encoder.bundle([left.key_hv, right.key_hv], f"merge:{left_id}:{right_id}")
            semantic_combined = None
            if left.semantic_key_hv is not None and right.semantic_key_hv is not None:
                semantic_combined = self.encoder.bundle(
                    [left.semantic_key_hv, right.semantic_key_hv],
                    f"semantic-merge:{left_id}:{right_id}",
                )
            outcome = OutcomeTable(self.max_outcomes, dict(left.outcomes.counts))
            for token, count in sorted(right.outcomes.counts.items()):
                for _ in range(count):
                    outcome.observe(token)
            new = Prototype(
                record_id=self._next_record_id,
                key_hv=combined,
                semantic_key_hv=semantic_combined,
                outcomes=outcome,
                support=left.support + right.support,
                created_tick=self.tick,
                last_used_tick=max(left.last_used_tick, right.last_used_tick),
                parents=(left_id, right_id),
                witnesses=(left.witnesses + right.witnesses)[:16],
                store="LTM",
            )
            self._next_record_id += 1
            del self.ltm[left_id]
            del self.ltm[right_id]
            self.ltm[new.record_id] = new
            consumed.update((left_id, right_id))
            merged += 1
        self.resources.recover(0.10)
        self.epoch += 1
        self.event_log.append(
            {"epoch": self.epoch, "event": "CONSOLIDATE", "promoted": promoted, "merged": merged}
        )
        return {"epoch": self.epoch, "promoted": promoted, "merged": merged, **self.summary()}

    def generate_with_trace(self, seed_text: str, max_tokens: int = 16) -> dict[str, object]:
        output = tokenize(seed_text)
        seen: dict[tuple[str, ...], int] = {}
        four_grams: dict[tuple[str, ...], int] = {}
        for index in range(max(0, len(output) - 3)):
            gram = tuple(output[index : index + 4])
            four_grams[gram] = four_grams.get(gram, 0) + 1
        stop_reason = "MAX_TOKENS"
        for _ in range(max_tokens):
            context = tuple(output[-self.context_length :])
            seen[context] = seen.get(context, 0) + 1
            if seen[context] > 2:
                stop_reason = "CONTEXT_LOOP"
                break
            prediction = self.predict(context)
            if prediction.token is None:
                stop_reason = "UNKNOWN"
                break
            if len(output) >= 3:
                gram = tuple(output[-3:] + [prediction.token])
                if four_grams.get(gram, 0) >= 2:
                    stop_reason = "NGRAM_LOOP"
                    self.hormones.update(success=False, error=0.5, loop_score=1.0)
                    break
                four_grams[gram] = four_grams.get(gram, 0) + 1
            output.append(prediction.token)
        return {"tokens": output, "stop_reason": stop_reason, "generated": len(output) - len(tokenize(seed_text))}

    def generate(self, seed_text: str, max_tokens: int = 16) -> list[str]:
        return list(self.generate_with_trace(seed_text, max_tokens)["tokens"])

    def _state_payload(self) -> dict[str, object]:
        def record_payload(record: Prototype) -> dict[str, object]:
            return {
                "record_id": record.record_id,
                "key_hv": record.key_hv.copy(),
                "semantic_key_hv": (
                    record.semantic_key_hv.copy() if record.semantic_key_hv is not None else None
                ),
                "outcomes": dict(record.outcomes.counts),
                "max_outcomes": record.outcomes.max_outcomes,
                "support": record.support,
                "created_tick": record.created_tick,
                "last_used_tick": record.last_used_tick,
                "parents": tuple(record.parents),
                "witnesses": copy.deepcopy(record.witnesses),
                "store": record.store,
            }

        return {
            "epoch": self.epoch,
            "tick": self.tick,
            "next_record_id": self._next_record_id,
            "semantic_acc": {key: value.copy() for key, value in self.semantic_acc.items()},
            "stm": {key: record_payload(value) for key, value in self.stm.items()},
            "ltm": {key: record_payload(value) for key, value in self.ltm.items()},
            "pattern_stats": copy.deepcopy(self.pattern_stats),
            "hormones": copy.deepcopy(self.hormones),
            "resources": copy.deepcopy(self.resources),
            "coherence": self.coherence,
            "event_log": copy.deepcopy(self.event_log),
            "recent_predictions": list(self._recent_predictions),
            "health_history": list(self._health_history),
        }

    @staticmethod
    def _restore_record(payload: dict[str, object]) -> Prototype:
        return Prototype(
            record_id=int(payload["record_id"]),
            key_hv=np.asarray(payload["key_hv"], dtype=np.int8).copy(),
            semantic_key_hv=(
                np.asarray(payload["semantic_key_hv"], dtype=np.int8).copy()
                if payload.get("semantic_key_hv") is not None
                else None
            ),
            outcomes=OutcomeTable(int(payload["max_outcomes"]), dict(payload["outcomes"])),
            support=int(payload["support"]),
            created_tick=int(payload["created_tick"]),
            last_used_tick=int(payload["last_used_tick"]),
            parents=tuple(payload["parents"]),
            witnesses=copy.deepcopy(payload["witnesses"]),
            store=str(payload["store"]),
        )

    def _restore_payload(self, payload: dict[str, object]) -> None:
        self.epoch = int(payload["epoch"])
        self.tick = int(payload["tick"])
        self._next_record_id = int(payload["next_record_id"])
        self.semantic_acc = {key: value.copy() for key, value in payload["semantic_acc"].items()}
        self.stm = {int(key): self._restore_record(value) for key, value in payload["stm"].items()}
        self.ltm = {int(key): self._restore_record(value) for key, value in payload["ltm"].items()}
        self.pattern_stats = copy.deepcopy(payload["pattern_stats"])
        self.hormones = copy.deepcopy(payload["hormones"])
        self.resources = copy.deepcopy(payload["resources"])
        self.coherence = float(payload["coherence"])
        self.event_log = copy.deepcopy(payload["event_log"])
        self._recent_predictions = list(payload["recent_predictions"])
        self._health_history = list(payload.get("health_history", []))

    def rollback(self) -> dict[str, object]:
        if not self._snapshots:
            return {"status": "NO_SNAPSHOT", **self.summary()}
        payload = self._snapshots.pop()
        self._restore_payload(payload)
        return {"status": "ROLLED_BACK", **self.summary()}

    def behavior_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            _stable_bytes(
                self.seed,
                self.encoder.dimension,
                self.context_length,
                self.dynamic_semantics,
                self.positional_binding,
                self.semantic_drift_fraction,
                self.neighbor_count,
                self.stable_key_weight,
            )
        )
        for token in sorted(self.semantic_acc):
            digest.update(_stable_bytes("semantic", token))
            digest.update(self.semantic_acc[token].astype("<i2").tobytes())
        for store_name, store in (("STM", self.stm), ("LTM", self.ltm)):
            for record_id in sorted(store):
                record = store[record_id]
                digest.update(_stable_bytes(store_name, record_id, record.support, record.parents))
                digest.update(record.key_hv.tobytes())
                if record.semantic_key_hv is not None:
                    digest.update(record.semantic_key_hv.tobytes())
                digest.update(json.dumps(record.outcomes.counts, sort_keys=True).encode())
        digest.update(json.dumps(self.hormones.as_dict(), sort_keys=True).encode())
        digest.update(_stable_bytes(round(self.coherence, 8), round(self.resources.energy, 8)))
        return digest.hexdigest()

    def summary(self) -> dict[str, object]:
        return {
            "stm_records": len(self.stm),
            "ltm_records": len(self.ltm),
            "semantic_tokens": len(self.semantic_acc),
            "coherence": round(self.coherence, 4),
            "energy": round(self.resources.energy, 4),
            "hormones": self.hormones.as_dict(),
            "behavior_hash": self.behavior_hash()[:16],
            "logical_bytes": self.logical_bytes(),
        }

    def logical_bytes(self) -> int:
        """Approximate portable state bytes, excluding Python object overhead."""

        vector_bytes = self.encoder.dimension // 8
        prototype_bytes = sum(
            vector_bytes * (2 if record.semantic_key_hv is not None else 1)
            + 32
            + 8 * len(record.outcomes.counts)
            for record in self._all_records()
        )
        semantic_bytes = (
            len(self.semantic_acc) * self.encoder.dimension * 2 if self.dynamic_semantics else 0
        )
        return int(prototype_bytes + semantic_bytes)


class NGramBaseline:
    """Equal-context exact n-gram baseline for honest comparison."""

    def __init__(self, context_length: int = 4, max_contexts: int = 2048) -> None:
        self.context_length = context_length
        self.max_contexts = max_contexts
        self.table: dict[tuple[str, ...], OutcomeTable] = {}

    def train_sequences(self, sequences: Sequence[Sequence[str] | str]) -> None:
        for sequence in sequences:
            tokens = tokenize(sequence) if isinstance(sequence, str) else list(sequence)
            for index in range(1, len(tokens)):
                context = tuple(tokens[max(0, index - self.context_length) : index])
                if context not in self.table and len(self.table) >= self.max_contexts:
                    continue
                self.table.setdefault(context, OutcomeTable()).observe(tokens[index])

    def predict(self, context: Sequence[str]) -> Prediction:
        key = tuple(context[-self.context_length :])
        table = self.table.get(key)
        if table is None:
            return Prediction(None, 0.0, 0.0, None, None)
        token, confidence = table.top()
        return Prediction(token, confidence, 1.0, "NGRAM", None)

    def logical_bytes(self) -> int:
        return sum(
            8 * len(context) + 8 * len(table.counts) + 16
            for context, table in self.table.items()
        )


class BackoffNGramBaseline(NGramBaseline):
    """Bounded longest-context-first n-gram baseline."""

    def train_sequences(self, sequences: Sequence[Sequence[str] | str]) -> None:
        for sequence in sequences:
            tokens = tokenize(sequence) if isinstance(sequence, str) else list(sequence)
            for index in range(1, len(tokens)):
                for size in range(1, min(self.context_length, index) + 1):
                    context = tuple(tokens[index - size : index])
                    if context not in self.table and len(self.table) >= self.max_contexts:
                        continue
                    self.table.setdefault(context, OutcomeTable()).observe(tokens[index])

    def predict(self, context: Sequence[str]) -> Prediction:
        trimmed = list(context)[-self.context_length :]
        for size in range(len(trimmed), 0, -1):
            table = self.table.get(tuple(trimmed[-size:]))
            if table is not None:
                token, confidence = table.top()
                return Prediction(token, confidence, size / self.context_length, "BACKOFF_NGRAM", None)
        return Prediction(None, 0.0, 0.0, None, None)


def _context_features(context: Sequence[str], context_length: int) -> frozenset[str]:
    trimmed = list(context)[-context_length:]
    padded = ["⟨PAD⟩"] * (context_length - len(trimmed)) + trimmed
    features: set[str] = set()
    for position, token in enumerate(padded):
        marked = f"⟨{token}⟩"
        for size in (3, 4, 5):
            for index in range(max(0, len(marked) - size + 1)):
                features.add(f"{position}:{marked[index:index + size]}")
    return frozenset(features)


class CharNGramKNNBaseline:
    """Sparse character-feature nearest-neighbor control without HDC algebra."""

    def __init__(self, context_length: int = 4, max_contexts: int = 2048) -> None:
        self.context_length = context_length
        self.max_contexts = max_contexts
        self.records: dict[tuple[str, ...], tuple[frozenset[str], OutcomeTable]] = {}

    def train_sequences(self, sequences: Sequence[Sequence[str] | str]) -> None:
        for sequence in sequences:
            tokens = tokenize(sequence) if isinstance(sequence, str) else list(sequence)
            for index in range(1, len(tokens)):
                context = tuple(tokens[max(0, index - self.context_length) : index])
                if context not in self.records:
                    if len(self.records) >= self.max_contexts:
                        continue
                    self.records[context] = (_context_features(context, self.context_length), OutcomeTable())
                self.records[context][1].observe(tokens[index])

    def predict(self, context: Sequence[str]) -> Prediction:
        if not self.records:
            return Prediction(None, 0.0, 0.0, None, None)
        query = _context_features(context, self.context_length)
        best_context: tuple[str, ...] | None = None
        best_score = -1.0
        best_table: OutcomeTable | None = None
        for stored_context in sorted(self.records):
            features, table = self.records[stored_context]
            union = len(query | features)
            score = len(query & features) / union if union else 1.0
            if score > best_score:
                best_context, best_score, best_table = stored_context, score, table
        token, outcome_confidence = best_table.top() if best_table else (None, 0.0)
        return Prediction(token, best_score * outcome_confidence, best_score, "CHAR_KNN", None)

    def logical_bytes(self) -> int:
        return sum(
            sum(len(feature.encode("utf-8")) for feature in features)
            + 8 * len(table.counts)
            for features, table in self.records.values()
        )


def evaluate_next_token(
    model: HDCCognitiveSystem | NGramBaseline,
    sequences: Sequence[Sequence[str] | str],
    context_length: int = 4,
) -> dict[str, float | int]:
    correct = 0
    known = 0
    total = 0
    for sequence in sequences:
        tokens = tokenize(sequence) if isinstance(sequence, str) else list(sequence)
        for index in range(1, len(tokens)):
            context = tokens[max(0, index - context_length) : index]
            prediction = model.predict(context)
            total += 1
            if prediction.token is not None:
                known += 1
            if prediction.token == tokens[index]:
                correct += 1
    return {
        "total": total,
        "correct": correct,
        "known": known,
        "accuracy": correct / max(1, total),
        "coverage": known / max(1, total),
    }


def synthetic_language(seed: int = 7, train_size: int = 300, test_size: int = 100) -> tuple[list[str], list[str]]:
    """Generate a small compositional corpus with held-out combinations."""

    rng = np.random.default_rng(seed)
    subjects = ["лиса", "кот", "робот", "птица", "девочка", "мальчик"]
    verbs = ["видит", "ищет", "помнит", "несёт"]
    objects = ["ключ", "мяч", "книгу", "звезду", "цветок"]
    places = ["в саду", "у реки", "в доме", "на мосту"]

    all_rows = [f"{s} {v} {o} {p} ." for s in subjects for v in verbs for o in objects for p in places]
    rng.shuffle(all_rows)
    split = min(train_size, len(all_rows) - test_size)
    return all_rows[:split], all_rows[split : split + test_size]


__all__ = [
    "CapacityError",
    "HDCCognitiveSystem",
    "HDCEncoder",
    "BackoffNGramBaseline",
    "CharNGramKNNBaseline",
    "NGramBaseline",
    "Prediction",
    "evaluate_next_token",
    "synthetic_language",
    "tokenize",
]
