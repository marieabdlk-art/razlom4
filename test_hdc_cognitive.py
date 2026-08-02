import unittest

import numpy as np

from hdc_cognitive import (
    BackoffNGramBaseline,
    CharNGramKNNBaseline,
    HDCCognitiveSystem,
    HDCEncoder,
    NGramBaseline,
    evaluate_next_token,
    synthetic_language,
)


class HDCEncoderTests(unittest.TestCase):
    def test_atoms_are_deterministic(self):
        left = HDCEncoder(512, "seed")
        right = HDCEncoder(512, "seed")
        np.testing.assert_array_equal(left.atom("word", "кот"), right.atom("word", "кот"))

    def test_bind_unbind_recovers_operand(self):
        encoder = HDCEncoder(512, "seed")
        left = encoder.atom("word", "лево")
        right = encoder.atom("word", "право")
        bound = encoder.bind(left, right)
        recovered = encoder.bind(bound, left)
        np.testing.assert_array_equal(recovered, right)

    def test_permutation_encodes_order(self):
        encoder = HDCEncoder(512, "seed")
        vector = encoder.atom("word", "порядок")
        self.assertLess(encoder.similarity(encoder.permute(vector, 1), encoder.permute(vector, 2)), 0.3)


class HDCCognitiveSystemTests(unittest.TestCase):
    def new_model(self, **kwargs):
        options = {
            "dimension": 512,
            "seed": "test-seed",
            "stm_capacity": 128,
            "ltm_capacity": 256,
        }
        options.update(kwargs)
        return HDCCognitiveSystem(**options)

    def test_online_learning_predicts_seen_sequence(self):
        model = self.new_model()
        model.train_sequences(["у лукоморья дуб зелёный"] * 4)
        prediction = model.predict(["у", "лукоморья", "дуб"])
        self.assertEqual(prediction.token, "зелёный")

    def test_ninth_outcome_is_total_and_deterministic(self):
        rows = [["контекст", f"цель-{index}"] for index in range(9)]
        left = self.new_model(max_outcomes=8)
        right = self.new_model(max_outcomes=8)
        left.train_sequences(rows)
        right.train_sequences(rows)
        self.assertEqual(left.behavior_hash(), right.behavior_hash())
        record = next(iter(left.stm.values()))
        self.assertEqual(len(record.outcomes.counts), 8)

    def test_replay_is_deterministic(self):
        rows = ["кот видит ключ .", "кот ищет ключ .", "лиса видит мяч ."] * 3
        left = self.new_model()
        right = self.new_model()
        left.train_sequences(rows)
        right.train_sequences(rows)
        left.consolidate()
        right.consolidate()
        self.assertEqual(left.behavior_hash(), right.behavior_hash())

    def test_rollback_restores_behavior(self):
        model = self.new_model()
        model.train_sequences(["кот видит ключ ."] * 3)
        expected = model.behavior_hash()
        model.train_sequences(["робот несёт книгу ."] * 3)
        self.assertNotEqual(model.behavior_hash(), expected)
        result = model.rollback()
        self.assertEqual(result["status"], "ROLLED_BACK")
        self.assertEqual(model.behavior_hash(), expected)

    def test_capacity_is_bounded(self):
        model = HDCCognitiveSystem(
            dimension=512,
            seed="capacity",
            stm_capacity=1,
            ltm_capacity=1,
            learn_threshold=1.01,
        )
        result = model.train_sequences([["a", "b", "c"]])
        self.assertGreaterEqual(result["rejected"], 1)
        self.assertLessEqual(len(model.stm), 1)

    def test_consolidation_promotes_supported_records(self):
        model = self.new_model(promotion_support=2)
        model.train_sequences(["кот видит ключ ."] * 3)
        result = model.consolidate()
        self.assertGreater(result["promoted"], 0)
        self.assertGreater(len(model.ltm), 0)

    def test_evaluation_reports_accuracy_and_coverage(self):
        train, test = synthetic_language(seed=3, train_size=100, test_size=20)
        model = self.new_model(stm_capacity=512)
        baseline = NGramBaseline(max_contexts=512)
        model.train_sequences(train)
        baseline.train_sequences(train)
        hdc = evaluate_next_token(model, test)
        ngram = evaluate_next_token(baseline, test)
        self.assertEqual(hdc["total"], ngram["total"])
        self.assertGreaterEqual(hdc["coverage"], 0.0)
        self.assertLessEqual(hdc["coverage"], 1.0)

    def test_backoff_and_char_controls_are_runnable(self):
        rows = ["красный робот видит ключ .", "синий робот ищет ключ ."] * 3
        backoff = BackoffNGramBaseline(max_contexts=128)
        char_knn = CharNGramKNNBaseline(max_contexts=128)
        backoff.train_sequences(rows)
        char_knn.train_sequences(rows)
        self.assertIsNotNone(backoff.predict(["новый", "робот"]).token)
        self.assertIsNotNone(char_knn.predict(["красны", "робот", "видит"]).token)

    def test_static_hdc_does_not_create_semantic_accumulators(self):
        model = self.new_model(dynamic_semantics=False)
        model.train_sequences(["кот видит ключ ."] * 2)
        self.assertEqual(model.semantic_acc, {})

    def test_loop_guard_stops_before_third_repeated_fourgram(self):
        model = self.new_model()
        model.train_sequences(["а б в г а б в г а б в г"] * 4)
        report = model.generate_with_trace("а б в", max_tokens=40)
        tokens = report["tokens"]
        counts = {}
        for index in range(len(tokens) - 3):
            gram = tuple(tokens[index : index + 4])
            counts[gram] = counts.get(gram, 0) + 1
        self.assertLessEqual(max(counts.values(), default=0), 2)
        self.assertIn(report["stop_reason"], {"NGRAM_LOOP", "CONTEXT_LOOP", "UNKNOWN", "MAX_TOKENS"})

    def test_hormones_do_not_saturate_on_repeated_success(self):
        model = self.new_model()
        model.train_sequences(["кот видит ключ ."] * 30)
        self.assertLess(model.hormones.dopamine, 0.95)
        self.assertGreater(model.hormones.dopamine, 0.50)

    def test_coherence_uses_history_not_only_last_step(self):
        model = self.new_model()
        model._update_coherence(error=1.0, loop_score=1.0)
        model._update_coherence(error=0.0, loop_score=0.0)
        self.assertLess(model.coherence, 1.0)

    def test_auto_consolidation_moves_supported_records(self):
        model = HDCCognitiveSystem(
            dimension=512,
            seed="auto",
            stm_capacity=4,
            ltm_capacity=16,
            promotion_support=2,
            auto_consolidate=True,
        )
        result = model.train_sequences(["кот видит ключ ."] * 4)
        self.assertIsNotNone(result["auto_consolidation"])
        self.assertGreater(len(model.ltm), 0)

    def test_stable_key_preserves_exact_memory_after_semantic_learning(self):
        model = self.new_model(
            neighbor_count=5,
            stable_key_weight=0.8,
            dynamic_semantics=True,
        )
        model.train_sequences(["кот видит ключ ."] * 4)
        before = model.predict(["кот", "видит"])
        model.train_sequences(["кот ищет книгу .", "робот видит ключ ."] * 12)
        after = model.predict(["кот", "видит"])
        self.assertEqual(before.token, "ключ")
        self.assertEqual(after.token, before.token)


if __name__ == "__main__":
    unittest.main()
