import unittest

import numpy as np

from hdc_cognitive import (
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


if __name__ == "__main__":
    unittest.main()
