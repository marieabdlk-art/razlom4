import unittest

from hdc_benchmark import aggregate


class HDCBenchmarkTests(unittest.TestCase):
    def test_value_gate_is_computed_from_preregistered_metrics(self):
        def arm(clean, noise, harmonic, drop=0.0, logical_bytes=100):
            return {
                "clean": {"accuracy": clean},
                "noise": {"accuracy": noise},
                "order": {"accuracy": 0.5},
                "retention": {"drop": drop, "harmonic": harmonic},
                "logical_bytes": logical_bytes,
            }

        row = {
            "seed": 1,
            "arms": {
                "exact_ngram": arm(0.40, 0.00, 0.0),
                "backoff_ngram": arm(0.45, 0.35, 0.0),
                "char_knn": arm(0.34, 0.38, 0.40),
                "static_hdc": arm(0.44, 0.44, 0.55),
                "dynamic_hdc": arm(0.45, 0.45, 0.59),
            },
            "loop": {"failure_rate": 0.0},
        }
        result = aggregate([row])
        self.assertEqual(result["gates"]["verdict"], "VALUABLE_PROTOTYPE")
        self.assertTrue(all(result["gates"]["hard"].values()))


if __name__ == "__main__":
    unittest.main()
