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
            "dopamine": 0.03 if success else 0.0,
            "serotonin": 0.01 if success and error < 0.25 else 0.0,
            "cortisol": min(0.05, 0.05 * error),
            "adrenaline": min(0.05, 0.05 * urgency),
            "two_ag": min(0.05, 0.05 * loop_score),
        }
        for name in baseline:
            value = getattr(self, name)
            value += release[name] - 0.05 * (value - baseline[name])
            setattr(self, name, min(1.0, max(0.0, value)))

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
        return self.encoder.bundle(
            [self.encoder.token_identity(token), self.semantic_hv(token)],
            f"dynamic-token:{token}",
        )

    def context_hv(self, context: Sequence[str]) -> np.ndarray:
        trimmed = list(context)[-self.context_length :]
        padded = ["⟨PAD⟩"] * (self.context_length - len(trimmed)) + trimmed
        bound = []
        for position, token in enumerate(padded, 1):
            role = self.encoder.atom("role", position)
            positioned = self.encoder.permute(self.token_hv(token), position)
            bound.append(self.encoder.bind(role, positioned))
        return self.encoder.bundle(bound, "context")

    def _all_records(self) -> Iterable[Prototype]:
        yield from (self.stm[key] for key in sorted(self.stm))
        yield from (self.ltm[key] for key in sorted(self.ltm))

    def _nearest(self, key_hv: np.ndarray) -> tuple[Prototype | None, float]:
        best: Prototype | None = None
        best_similarity = -1.0
        for record in self._all_records():
            similarity = self.encoder.similarity(key_hv, record.key_hv)
            if similarity > best_similarity or (
                math.isclose(similarity, best_similarity) and best is not None and record.record_id < best.record_id
            ):
                best = record
                best_similarity = similarity
        return best, best_similarity

    def predict(self, context: Sequence[str]) -> Prediction:
        if not self.stm and not self.ltm:
            return Prediction(None, 0.0, 0.0, None, None)
        key_hv = self.context_hv(context)
        record, similarity = self._nearest(key_hv)
        if record is None or similarity < self.query_threshold:
            return Prediction(None, 0.0, max(0.0, similarity), None, None)
        token, outcome_confidence = record.outcomes.top()
        confidence = max(0.0, similarity) * outcome_confidence
        return Prediction(token, confidence, similarity, record.store, record.record_id)

    def _update_semantic(self, token: str, context_hv: np.ndarray, mode: str) -> None:
        eta = {"DISCOVERY": 1, "MEMORIZATION": 2, "PRACTICE": 4}[mode]
        acc = self._semantic_acc(token)
        candidate = np.clip(
            acc.astype(np.int32)
            + eta * context_hv.astype(np.int32)
            + self.encoder.token_identity(token).astype(np.int32),
            -32768,
            32767,
        ).astype(np.int16)
        old_hv = self.semantic_hv(token)
        new_hv = np.sign(candidate).astype(np.int8)
        ties = candidate == 0
        new_hv[ties] = self.encoder.token_identity(token)[ties]
        drift = 1.0 - self.encoder.similarity(old_hv, new_hv)
        if drift <= 0.02:
            self.semantic_acc[token] = candidate

    def _new_record(self, key_hv: np.ndarray) -> Prototype:
        if len(self.stm) >= self.stm_capacity:
            raise CapacityError("STM_CAPACITY_FULL")
        record = Prototype(
            record_id=self._next_record_id,
            key_hv=key_hv.copy(),
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
        key_hv = self.context_hv(context)
        record, similarity = self._nearest(key_hv)
        if record is None or similarity < self.learn_threshold:
            record = self._new_record(key_hv)
        record.observe(context, target, self.tick)
        self._update_semantic(target, key_hv, stats.mode)
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
        return {"epoch": self.epoch, "learned": learned, "rejected": rejected, **self.summary()}

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
            outcome = OutcomeTable(self.max_outcomes, dict(left.outcomes.counts))
            for token, count in sorted(right.outcomes.counts.items()):
                for _ in range(count):
                    outcome.observe(token)
            new = Prototype(
                record_id=self._next_record_id,
                key_hv=combined,
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

    def generate(self, seed_text: str, max_tokens: int = 16) -> list[str]:
        output = tokenize(seed_text)
        seen: dict[tuple[str, ...], int] = {}
        for _ in range(max_tokens):
            context = tuple(output[-self.context_length :])
            seen[context] = seen.get(context, 0) + 1
            if seen[context] > 2:
                break
            prediction = self.predict(context)
            if prediction.token is None:
                break
            output.append(prediction.token)
        return output

    def _state_payload(self) -> dict[str, object]:
        def record_payload(record: Prototype) -> dict[str, object]:
            return {
                "record_id": record.record_id,
                "key_hv": record.key_hv.copy(),
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
        }

    @staticmethod
    def _restore_record(payload: dict[str, object]) -> Prototype:
        return Prototype(
            record_id=int(payload["record_id"]),
            key_hv=np.asarray(payload["key_hv"], dtype=np.int8).copy(),
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

    def rollback(self) -> dict[str, object]:
        if not self._snapshots:
            return {"status": "NO_SNAPSHOT", **self.summary()}
        payload = self._snapshots.pop()
        self._restore_payload(payload)
        return {"status": "ROLLED_BACK", **self.summary()}

    def behavior_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(_stable_bytes(self.seed, self.encoder.dimension, self.context_length))
        for token in sorted(self.semantic_acc):
            digest.update(_stable_bytes("semantic", token))
            digest.update(self.semantic_acc[token].astype("<i2").tobytes())
        for store_name, store in (("STM", self.stm), ("LTM", self.ltm)):
            for record_id in sorted(store):
                record = store[record_id]
                digest.update(_stable_bytes(store_name, record_id, record.support, record.parents))
                digest.update(record.key_hv.tobytes())
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
        }


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
    "NGramBaseline",
    "Prediction",
    "evaluate_next_token",
    "synthetic_language",
    "tokenize",
]
